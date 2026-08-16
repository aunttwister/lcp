"""Intelligent model routing — prompt classification → capability scoring → best-fit model.

Three routing strategies, from simple to smart:
  1. CapabilityRouter — task classification + benchmark-derived scores (NEW, recommended)
  2. DynamicRouter (legacy) — token/tool count heuristics
  3. Disabled — static chain (current default)

The CapabilityRouter loads per-model scores from the model_capabilities DB table,
classifies each incoming prompt into a task type (agentic, coding, debugging,
reasoning, planning, chat), and scores all available models to pick the best fit.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

from .cost_estimator import count_tokens
from .logging_config import get_logger

logger = get_logger("lcp.router")


# ── Task classification (keyword/heuristic, no ML dependency) ────────────────

TASK_SIGNALS: dict[str, list[str]] = {
    "agentic_multi_step": [
        "you are an ai agent", "you are a coding agent",
        "autonomous", "multi-step", "multi step",
        "tools:", "function call", "tool call",
    ],
    "code_generation": [
        "write a function", "implement", "create a script",
        "write code", "def ", "class ", "import ",
        "write a program", "in python", "in javascript",
        "in rust", "in go", "html", "css", "react",
    ],
    "debugging": [
        "debug", "error", "exception", "traceback",
        "stack trace", "why does this fail", "not working",
        "bug", "fix this", "what's wrong",
    ],
    "research_deep": [
        "explain", "analyze", "compare and contrast",
        "research", "literature review", "in detail",
        "comprehensive", "thorough",
    ],
    "reasoning_chain": [
        "solve", "proof", "prove", "calculate",
        "logic puzzle", "step by step", "mathematical",
        "equation", "theorem",
    ],
    "planning": [
        "design", "architecture", "how should i structure",
        "plan", "roadmap", "strategy", "approach",
        "best practice", "recommend", "suggest",
    ],
}

CASUAL_SIGNALS = [
    "hello", "hi ", "hey", "thanks", "thank you", "how are you",
    "what's up", "good morning", "good night",
]


def classify_task(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    max_tokens: int = 1024,
) -> str:
    """Classify a request into a task type.

    Examines system prompt, tool usage, message content, and metadata.
    First match wins — order matters.
    """
    # Gather all text to classify
    combined = ""
    for msg in messages or []:
        content = msg.get("content", "")
        if isinstance(content, str):
            combined += content.lower() + " "
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    combined += block["text"].lower() + " "

    # System prompt signals (weighted first — system prompt defines the agent)
    system_text = ""
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content", "")
        if isinstance(content, str):
            system_text = content.lower()

    # Check system prompt first
    for task, keywords in TASK_SIGNALS.items():
        for kw in keywords:
            if kw in system_text:
                return task

    # Then check all messages
    for task, keywords in TASK_SIGNALS.items():
        for kw in keywords:
            if kw in combined:
                return task

    # Tool count signal — many tools = agentic
    tool_count = len(tools) if tools else 0
    if tool_count > 5:
        return "agentic_multi_step"

    # Token count signal — very long prompt = long_document
    token_count = count_tokens(messages, tools)
    if token_count > 8000:
        return "research_deep"

    # Expected output signal — long max_tokens suggests reasoning/planning
    if max_tokens > 4096:
        return "planning"

    # Check casual signals
    for kw in CASUAL_SIGNALS:
        if kw in combined:
            return "casual_chat"

    # Default: the most common LCP use case is agentic coding
    return "code_generation"


# ── CapabilityRouter — DB-backed, task-classifying, N-model scorer ────────────

# Default scores for models not in the DB (conservative: assume pro-level)
DEFAULT_CAPABILITY: dict[str, float] = {
    "deepseek-v4-pro": 0.85,
    "deepseek-v4-flash": 0.65,
}

# Cost bias: how much to boost cheaper models (0.0 = pure capability, 0.3 = strong cost bias)
DEFAULT_COST_BIAS = 0.15

# Known model pricing (USD per 1M output tokens) — from gateway.yaml
_MODEL_PRICES: dict[str, float] = {
    "deepseek-v4-pro": 0.87,
    "deepseek-v4-flash": 0.27,
}


# ── Model registry — DB-backed, explicit, no runtime name parsing ────────────
#
# Each provider uses its own model-ID convention, and each benchmark publishes
# its own (often dated) names. Instead of guessing from string patterns, we
# keep ONE explicit registry (persisted in the ``model_registry`` table) that
# maps every alias back to a logical name and pins that logical name to the
# benchmark snapshot it should be scored by.
#
#   logical_name:  the canonical gateway name (also the key in _MODEL_PRICES
#                  and pricing configs — used for pricing and aggregation).
#   benchmark_key: the STABLE, release-independent model key inside the seeded
#                  capability matrix (LiveBench / Arena data). Dated snapshot
#                  names are releases, not keys.
#   aliases:       every provider-side ID that means the same logical model.
#
# The curated defaults live in seed_capabilities.DEFAULT_MODEL_REGISTRY and are
# seeded into the DB on first run. After seeding, the DB is the source of truth
# and is editable via the /models page — no code changes required when a
# provider rolls a new dated snapshot.


_registry_cache: Optional[dict[str, dict]] = None
_registry_db_path: Optional[str] = None


def get_model_registry(db_path: str = "data/costs.db") -> dict[str, dict]:
    """Return the model registry (cached), loading/seeding from DB as needed."""
    global _registry_cache, _registry_db_path
    if _registry_cache is not None and _registry_db_path == db_path:
        return _registry_cache
    from .seed_capabilities import load_model_registry, seed_model_registry
    registry = load_model_registry(db_path)
    if not registry:
        seed_model_registry(db_path)
        registry = load_model_registry(db_path)
    _registry_cache = registry
    _registry_db_path = db_path
    return _registry_cache


def invalidate_registry_cache() -> None:
    """Clear the cached registry so the next lookup re-reads the DB."""
    global _registry_cache, _registry_db_path
    _registry_cache = None
    _registry_db_path = None


def _alias_to_logical(registry: dict[str, dict]) -> dict[str, str]:
    """Build reverse index: alias (lowercased) → logical name."""
    return {
        alias.lower(): logical
        for logical, entry in registry.items()
        for alias in entry.get("aliases", [])
    }


def logical_model_name(model: str, db_path: str = "data/costs.db") -> str:
    """Map any model ID to its logical gateway name via the DB registry.

    Unknown names pass through unchanged (lowercased) so the router still
    works for models not yet listed in the registry.
    """
    if not model:
        return model
    registry = get_model_registry(db_path)
    return _alias_to_logical(registry).get(model.strip().lower(), model.strip().lower())


def benchmark_model_name(logical: str, db_path: str = "data/costs.db") -> str:
    """Return the benchmark snapshot key for a logical model name."""
    registry = get_model_registry(db_path)
    entry = registry.get(logical)
    if entry:
        return entry["benchmark_key"]
    return logical


def provider_model_name(logical: str, provider: str, db_path: str = "data/costs.db") -> str:
    """Return the provider-side model ID for a logical model + provider.

    Uses the registry's explicit ``provider_mappings`` (e.g. Command Code's
    ``deepseek/deepseek-v4-pro`` vs OpenCode's bare ``deepseek-v4-pro``).
    Falls back to the first alias containing the provider, then to the
    logical name unchanged.
    """
    registry = get_model_registry(db_path)
    entry = registry.get(logical)
    if entry:
        mapping = entry.get("provider_mappings") or {}
        if provider in mapping:
            return mapping[provider]
        for alias in entry.get("aliases", []):
            if alias.lower().startswith(provider.lower() + "/"):
                return alias
    return logical


class CapabilityRouter:
    """Routes to the best available model using capability scores + cost awareness."""

    def __init__(
        self,
        enabled: bool = False,
        db_path: str = "data/costs.db",
        cost_bias: float = DEFAULT_COST_BIAS,
    ):
        self.enabled = enabled
        self.db_path = db_path
        self.cost_bias = cost_bias
        self._matrix: Optional[dict[str, dict[str, float]]] = None

    def load_matrix(self) -> dict[str, dict[str, float]]:
        """Load capability matrix from DB. Cached in memory."""
        if self._matrix is not None:
            return self._matrix
        try:
            from .seed_capabilities import load_capability_matrix
            self._matrix = load_capability_matrix(self.db_path)
            logger.info("capability_matrix_loaded", tasks=len(self._matrix))
        except Exception as e:
            logger.warning("capability_matrix_load_failed", error=str(e))
            self._matrix = {}
        return self._matrix

    def get_model_score(self, model: str, task: str) -> float:
        """Get capability score for a model on a task (0.0–1.0).

        Resolves the model ID through the explicit registry: alias → logical
        name → benchmark snapshot, then looks up the score. Unknown models
        fall back to a default.
        """
        matrix = self.load_matrix()
        task_scores = matrix.get(task, {})

        logical = logical_model_name(model, self.db_path)
        benchmark = benchmark_model_name(logical, self.db_path)

        if benchmark != model or logical != model:
            logger.debug(
                "capability_resolved",
                model=model,
                logical=logical,
                benchmark=benchmark,
                task=task,
            )

        return task_scores.get(benchmark, DEFAULT_CAPABILITY.get(logical, 0.5))

    def rank_models(
        self,
        task: str,
        available_models: list[str],
    ) -> list[tuple[str, float]]:
        """Rank available models by (capability * cost_adjustment).

        Returns list of (model, score) sorted best-first.
        """
        if not available_models:
            return []

        priced_models = []
        for model in available_models:
            capability = self.get_model_score(model, task)
            price = _MODEL_PRICES.get(model, 1.0)
            max_price = max(v for v in _MODEL_PRICES.values() if v > 0)
            cost_factor = price / max_price if max_price > 0 else 0.5
            # Score = capability + cost_bias * (1 - cost_factor)
            # A cheaper model gets a boost: flash (0.27/0.87=0.31 → boost 0.15*0.69=0.104)
            score = capability + self.cost_bias * (1.0 - cost_factor)
            priced_models.append((model, score))

        priced_models.sort(key=lambda x: -x[1])
        return priced_models

    def select_model(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        max_tokens: int = 1024,
        available_models: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Select the best model for this request.

        Returns the recommended model name, or None to use the chain's default.
        """
        if not self.enabled:
            return None

        task = classify_task(messages, tools, max_tokens)

        # Build available model list from chain if not provided
        if available_models is None:
            available_models = list(_MODEL_PRICES.keys())

        ranked = self.rank_models(task, available_models)
        if not ranked:
            return None

        best_model, best_score = ranked[0]
        chain_default = available_models[0] if available_models else "unknown"

        # Only override if the selected model is meaningfully better
        # (avoids flapping between models that are nearly tied)
        default_score = self.get_model_score(chain_default, task)
        if best_score > default_score + 0.05:  # 5% threshold
            logger.info(
                "router_override",
                task=task,
                chain_default=chain_default,
                recommended=best_model,
                default_score=round(default_score, 3),
                recommended_score=round(best_score, 3),
            )
            return best_model

        logger.debug(
            "router_keep_default",
            task=task,
            model=chain_default,
            score=round(default_score, 3),
        )
        return None


# ── Global instances ──────────────────────────────────────────────────────────

_dynamic_router = CapabilityRouter(enabled=False)


def get_dynamic_router() -> CapabilityRouter:
    return _dynamic_router


def init_router(db_path: str = "data/costs.db", enabled: bool = False):
    """Initialize the global router. Call once at startup."""
    global _dynamic_router
    _dynamic_router = CapabilityRouter(enabled=enabled, db_path=db_path)
    if enabled:
        _dynamic_router.load_matrix()  # warm the cache


# ── Legacy: DynamicRouter (kept for reference, not used) ─────────────────────

class DynamicRouter:
    """[DEPRECATED] Simple flash-vs-pro heuristic. Use CapabilityRouter instead."""

    SHORT_PROMPT_THRESHOLD = 500
    LONG_PROMPT_THRESHOLD = 2000
    TOOL_COUNT_THRESHOLD = 3
    MODEL_MAP = {"flash": "deepseek-v4-flash", "pro": "deepseek-v4-pro"}

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def should_use_flash(self, messages, tools=None, max_tokens=1024):
        token_count = count_tokens(messages, tools)
        tool_count = len(tools) if tools else 0
        if token_count > self.LONG_PROMPT_THRESHOLD:
            return False
        if tool_count > self.TOOL_COUNT_THRESHOLD:
            return False
        if max_tokens > 2048:
            return False
        return True

    def get_recommended_model(self, messages, tools=None, max_tokens=1024):
        if self.enabled and self.should_use_flash(messages, tools, max_tokens):
            return self.MODEL_MAP["flash"]
        return self.MODEL_MAP["pro"]
