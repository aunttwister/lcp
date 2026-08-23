"""Intelligent model routing — prompt classification → capability scoring → best-fit model.

Three routing strategies, from simple to smart:
  1. CapabilityRouter — task classification + benchmark-derived scores (NEW, recommended)
  2. DynamicRouter (legacy) — token/tool count heuristics
  3. Disabled — static chain (current default)

The CapabilityRouter loads per-model scores from the model_capabilities DB table,
classifies each incoming prompt into a task type (agentic, unit tests, coding,
debugging, reasoning, planning, chat), and scores all available models to pick
the best fit.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

from .cost_estimator import count_tokens
from .logging_config import get_logger

logger = get_logger("lcp.router")


# ── Model-ID normalization (llama.cpp gguf paths, quantization) ──────────────

# Common GGUF quantization tags, matched as the last path segment.
_QUANT_RE = __import__("re").compile(
    r"(q\d+[a-z]*(?:_[a-z0-9]+)*|f16|f32|bf16|fp16|fp32|i?q\d+_\d+|[a-z]+\d+(?:\.\d+)?b)"
)


def normalize_model_id(model: str) -> str:
    """Normalize a raw provider-side model ID into a clean logical name.

    - strips ``/models/`` and leading slashes (llama.cpp file paths)
    - strips the ``.gguf`` extension
    - lowercases and collapses whitespace

    e.g. ``/models/qwen3.6-27b-q4_k_m.gguf`` → ``qwen3.6-27b-q4_k_m``.
    """
    if not model:
        return model
    name = str(model).strip()
    # Strip path prefix like /models/ or leading slash
    if name.startswith("/models/"):
        name = name[len("/models/"):]
    elif name.startswith("/"):
        name = name.lstrip("/")
    # Take the last path segment
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    if name.lower().endswith(".gguf"):
        name = name[:-5]
    return name.strip().lower()


def detect_quantization(model: str) -> Optional[str]:
    """Return a quantization tag like ``Q4_K_M``, or None.

    Looks for GGUF quantization tokens (``q4_k_m``, ``q4_0``, ``f16``, …) in
    the model ID. Returns the canonical uppercase form when found.
    """
    if not model:
        return None
    name = str(model).strip().lower()
    if name.endswith(".gguf"):
        name = name[:-5]
    # Strip any path so we look at the filename stem.
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    m = _QUANT_RE.search(name)
    if not m:
        return None
    return m.group(1).upper()


# ── Task classification (keyword/heuristic, no ML dependency) ────────────────

TASK_SIGNALS: dict[str, list[str]] = {
    "agentic_multi_step": [
        "you are an ai agent", "you are a coding agent",
        "autonomous", "multi-step", "multi step",
        "tools:", "function call", "tool call",
    ],
    "unit_tests": [
        "unit test", "unit tests", "write tests", "write a test",
        "test case", "test cases", "test suite", "add tests",
        "create tests", "pytest", "unittest", "test coverage",
        "mocking", "mock object", "mock the",
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

# Task types that carry concrete user intent (vs. ``agentic_multi_step``, which
# is the generic "I am an agent" preamble most agents send). Classification
# checks these against the system prompt / user messages FIRST so the agentic
# catch-all can't mask e.g. a planning request.
_SPECIFIC_TASKS = (
    "planning", "debugging", "unit_tests", "code_generation",
    "reasoning_chain", "research_deep",
)


def classify_task(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    max_tokens: int = 1024,
) -> str:
    """Classify a request into a task type.

    Examines system prompt, tool usage, message content, and metadata.

    Priority (first match wins):
      1. System prompt — but only for SPECIFIC task signals (planning,
         debugging, unit_tests, code_generation, reasoning_chain,
         research_deep). An agentic system prompt does NOT immediately win:
         it's the generic "I am an agent" preamble most agents send, and it
         must not mask the user's actual intent.
      2. All messages (user + assistant) — specific task signals.
      3. Agentic system prompt (the catch-all) — only now.
      4. Tool-count / token-count / max_tokens heuristics, then casual, then
         the code_generation default.

    This ordering matters: a Hermes agent sends an agentic system prompt
    ("you are an AI agent", "tools:", …) on EVERY request, so if that won
    first, a user message like "design the architecture" would never reach
    the ``planning`` rule. User intent now wins over the agent preamble.
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

    # System prompt signals — but only the SPECIFIC tasks (not agentic).
    system_text = ""
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content", "")
        if isinstance(content, str):
            system_text = content.lower()

    # 1. System prompt: specific tasks only.
    for task in _SPECIFIC_TASKS:
        for kw in TASK_SIGNALS[task]:
            if kw in system_text:
                return task

    # 2. All messages: specific tasks first (this is where user intent lives).
    for task in _SPECIFIC_TASKS:
        for kw in TASK_SIGNALS[task]:
            if kw in combined:
                return task

    # 3. Agentic system prompt — the generic agent preamble, checked AFTER the
    #    user's explicit task so it can't mask planning/debugging/unit_tests etc.
    for kw in TASK_SIGNALS["agentic_multi_step"]:
        if kw in system_text:
            return "agentic_multi_step"

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

# Hysteresis: only override/reorder when the best step beats the default by this much.
_HYSTERESIS = 0.05

# Known model pricing (USD per 1M output tokens) — from gateway.yaml
_MODEL_PRICES: dict[str, float] = {
    "deepseek-v4-pro": 0.87,
    "deepseek-v4-flash": 0.27,
}

# Health bonus per circuit-breaker status (adds to a step's score).
_HEALTH_BONUS: dict[str, float] = {
    "healthy": 0.05,
    "degraded": -0.03,
    "dead": -0.25,
}
# Score penalty when a provider's cached usage suggests it is running low.
_LOW_CREDIT_PENALTY = -0.10


# ── Model registry — DB-backed, explicit, no runtime name parsing ────────────
#
# Each provider uses its own model-ID convention, and each benchmark publishes
# its own (often dated) names. Instead of guessing from string patterns, we
# keep ONE explicit registry (persisted in the ``model_registry`` table) that
# maps every provider-side model ID back to a logical name and pins that
# logical name to the benchmark snapshot it should be scored by.
#
#   logical_name:   the canonical gateway name (also the key in _MODEL_PRICES
#                   and pricing configs — used for pricing and aggregation).
#   benchmark_key:  the STABLE, release-independent model key inside the seeded
#                   capability matrix (LiveBench / Arena data).
#   provider_mappings: {provider: provider-side model ID}. The mapping VALUES
#                   are the provider-side spellings the reverse index is
#                   built from.
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
    """Build reverse index: provider-side model ID / benchmark key → logical name."""
    index: dict[str, str] = {}
    for logical, entry in registry.items():
        index[logical.lower()] = logical
        if entry.get("benchmark_key"):
            index.setdefault(entry["benchmark_key"].lower(), logical)
        for provider_side in (entry.get("provider_mappings") or {}).values():
            if provider_side:
                index.setdefault(provider_side.lower(), logical)
    return index


def logical_model_name(model: str, db_path: str = "data/costs.db") -> str:
    """Map any model ID to its logical gateway name via the DB registry.

    Unknown names are normalized (strips ``/models/`` prefix and ``.gguf``
    extension, lowercased) so a llama.cpp path like
    ``/models/qwen3.6-27b-q4_k_m.gguf`` resolves to ``qwen3.6-27b-q4_k_m``.
    """
    if not model:
        return model
    registry = get_model_registry(db_path)
    key = model.strip().lower()
    mapped = _alias_to_logical(registry).get(key)
    if mapped:
        return mapped
    return normalize_model_id(key)


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
    Falls back to the logical name unchanged when the provider is unmapped.
    """
    registry = get_model_registry(db_path)
    entry = registry.get(logical)
    if entry:
        mapping = entry.get("provider_mappings") or {}
        if provider in mapping:
            return mapping[provider]
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
        # Bounded log of recent routing decisions, surfaced in the UI
        # (/api/routing/status, Providers → Routing tab).
        self._decisions: list[dict] = []

    # ── Matrix ────────────────────────────────────────────────────────────

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

    def invalidate_matrix(self) -> None:
        """Drop the cached matrix so the next call re-reads the DB.

        Called when a benchmark run completes or the registry changes, so
        routing picks up fresh scores without a restart.
        """
        self._matrix = None

    # ── Policy + decisions ────────────────────────────────────────────────

    def _effective_policy(self, config: Optional[object] = None) -> tuple[str, float]:
        """Return (policy, min_score): runtime settings override config.

        Policy ∈ {eager, cost_first, explore}; min_score is a 0–1 floor below
        which a reorder is never recommended.
        """
        policy = "eager"
        min_score = 0.0
        try:
            from .cost_cache import get_settings
            settings = get_settings()
            if settings is not None:
                policy = settings.get_routing_policy(default=policy)
                min_score = settings.get_routing_min_score(default=min_score)
        except Exception:  # noqa: BLE001
            pass
        if config is not None:
            try:
                dr = config.dynamic_routing or {}
                policy = dr.get("policy", policy)
                min_score = float(dr.get("min_score", min_score))
            except Exception:  # noqa: BLE001
                pass
        return policy, min_score

    def is_enabled(self, config: Optional[object] = None) -> bool:
        """Effective enabled state: a runtime toggle (settings table, UI) wins,
        otherwise fall back to the boot-time value (seeded from config).

        ``config`` is accepted for API symmetry (callers pass it through) but
        the boot-time ``self.enabled`` already reflects config at startup.
        """
        try:
            from .cost_cache import get_settings
            settings = get_settings()
            if settings is not None:
                override = settings.get_routing_enabled(default=None)
                if override is not None:
                    return override
        except Exception:  # noqa: BLE001 — toggle must never break routing
            pass
        return self.enabled

    def _record_decision(self, decision: dict) -> None:
        self._decisions.append(decision)
        if len(self._decisions) > 50:
            self._decisions = self._decisions[-50:]

    def recent_decisions(self, limit: int = 25) -> list[dict]:
        return list(self._decisions[-limit:])

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
        if not self.is_enabled():
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

    # ── Provider-aware selection (Phase 2) ────────────────────────────────

    def _health_bonus(self, step: dict, profile: Optional[str] = None,
                      config: Optional[object] = None) -> float:
        """Bonus/penalty from the circuit breaker's per-provider health.

        A 96% model on a degraded provider can lose to a 94% model on a
        healthy one. Returns 0 when the breaker is unavailable (e.g. tests).
        """
        try:
            from .circuit_breaker import get_circuit_breaker
            provider = step["provider"]
            base_url = step.get("base_url") or ""
            if config is not None and not base_url:
                base_url = (config.providers or {}).get(provider, {}).get("api_base", "")
            cb = get_circuit_breaker()
            status = cb.status_of(provider, base_url, profile or "")
            return _HEALTH_BONUS.get(status, 0.0)
        except Exception:  # noqa: BLE001 — tiebreaker must never break routing
            return 0.0

    def _provider_available(self, step: dict, profile: Optional[str] = None,
                            config: Optional[object] = None) -> bool:
        """True when the step's provider is available to the circuit breaker.

        This is the breaker's provider-level GATE (dead/hard-tripped providers
        are excluded). When dynamic routing is enabled, the router decides
        model selection + ordering and the breaker only gates providers — so
        the router never proposes a provider the breaker would skip. Returns
        True when the breaker is unavailable (e.g. tests) to never break
        routing.
        """
        try:
            from .circuit_breaker import get_circuit_breaker
            provider = step["provider"]
            base_url = step.get("base_url") or ""
            if config is not None and not base_url:
                base_url = (config.providers or {}).get(provider, {}).get("api_base", "")
            cb = get_circuit_breaker()
            return cb.is_available(provider, base_url, profile or "")
        except Exception:  # noqa: BLE001 — gate must never break routing
            return True

    def _provider_serves_model(self, provider: str, logical_model: str,
                               config: Optional[object] = None) -> bool:
        """True when *provider* exposes *logical_model* (per gateway.yaml models).

        Providers with no explicit model list are treated as serving the model
        (optimistic), so a prefer never silently drops a provider just because
        its model list is unconfigured. Never raises.
        """
        try:
            pcfg = (config.providers or {}).get(provider, {}) or {}
            models = pcfg.get("models") or []
            if not models:
                return True
            for m in models:
                if logical_model_name(m, self.db_path) == logical_model:
                    return True
        except Exception:  # noqa: BLE001 — never break routing
            return True
        return False

    def _credit_bonus(self, provider: str) -> float:
        """Penalty when a provider's cached usage suggests low credits.

        Reads the DB cost cache (never scrapes): opencode monthly% >= 95,
        commandcode remaining credits <= $5, deepseek balance <= $1.
        """
        try:
            from .cost_cache import get_cost_cache
            cache = get_cost_cache()
            if cache is None:
                return 0.0
            sub = cache.get(provider, "subscription")
            if sub:
                p = sub["payload"] or {}
                if p.get("_error"):
                    return _LOW_CREDIT_PENALTY
                if provider == "opencode":
                    mpct = p.get("monthly_pct")
                    if mpct is not None and mpct >= 95:
                        return _LOW_CREDIT_PENALTY
                elif provider == "commandcode":
                    rem = p.get("monthly_credits_remaining")
                    if rem is not None and rem <= 5.0:
                        return _LOW_CREDIT_PENALTY
            bal = cache.get(provider, "balance")
            if bal:
                p = bal["payload"] or {}
                b = p.get("balance")
                if isinstance(b, dict):
                    avail = b.get("available")
                    if avail is not None and avail <= 1.0:
                        return _LOW_CREDIT_PENALTY
        except Exception:  # noqa: BLE001 — tiebreaker must never break routing
            pass
        return 0.0

    def score_step(self, step: dict, task: str, profile: Optional[str] = None,
                   config: Optional[object] = None, bias: Optional[float] = None) -> float:
        """Score a single ``{provider, model}`` chain step.

        capability + cost-bias boost + health bonus + credit penalty.
        ``bias`` overrides the default cost bias (used by ``cost_first``).
        """
        model = step["model"]
        capability = self.get_model_score(model, task)
        price = _MODEL_PRICES.get(model, 1.0)
        max_price = max(_MODEL_PRICES.values()) if _MODEL_PRICES else 1.0
        cost_factor = price / max_price if max_price > 0 else 0.5
        b = self.cost_bias if bias is None else bias
        score = capability + b * (1.0 - cost_factor)
        score += self._health_bonus(step, profile, config)
        score += self._credit_bonus(step["provider"])
        return score

    # ── Routing rules (UI-defined overrides) ─────────────────────────────

    def _rules(self, config: Optional[object] = None) -> list:
        """Return the effective routing-rules list.

        Runtime settings (settings table, UI-editable) win; ``config``
        ``dynamic_routing.rules`` seeds defaults when no setting exists.
        """
        try:
            from .cost_cache import get_settings
            settings = get_settings()
            if settings is not None:
                stored = settings.get_routing_rules()
                if stored:
                    return stored
        except Exception:  # noqa: BLE001
            pass
        if config is not None:
            try:
                dr = config.dynamic_routing or {}
                return list(dr.get("rules", []) or [])
            except Exception:  # noqa: BLE001
                pass
        return []

    @staticmethod
    def _rule_matches(rule: dict, task: str, profile: str) -> bool:
        """A rule matches when its profile/task equal (or '*' / contains)."""
        if not rule.get("enabled", True):
            return False
        rp = rule.get("profile", "*")
        if rp not in ("*", "") and rp != profile:
            return False
        rt = rule.get("task", "*")
        if rt in ("*", ""):
            return True
        if isinstance(rt, list):
            return task in rt
        return rt == task

    def _rule_target(self, rule: dict, step: dict) -> bool:
        """True if the step matches a prefer/block rule's provider/model.

        ``"*"`` / ``""`` act as wildcards, so a rule can target ONLY a model
        (any provider) or ONLY a provider (any model). At least one concrete
        provider/model must be present.

        Model IDs are normalized through the registry (``logical_model_name``)
        so a rule written with the logical name (e.g. ``deepseek-v4-pro``)
        matches a chain step whose model is a provider-side ID (e.g.
        commandcode's ``deepseek/deepseek-v4-pro``). Provider names are
        compared literally (they are plain config names).
        """
        provider = rule.get("provider")
        model = rule.get("model")
        if provider and provider not in ("*", "") and provider != step["provider"]:
            return False
        if model and model not in ("*", ""):
            rule_model = logical_model_name(model, self.db_path)
            step_model = logical_model_name(step["model"], self.db_path)
            if rule_model != step_model:
                return False
        has_provider = bool(provider and provider not in ("*", ""))
        has_model = bool(model and model not in ("*", ""))
        return has_provider or has_model

    def _apply_rules(self, chain: list, task: str, profile: str,
                     config: Optional[object] = None) -> tuple[list, list]:
        """Apply block/prefer rules to a COPY of the chain.

        Returns (candidates, fired_rule_descriptions). ``block`` removes
        matching steps; ``prefer`` moves the first matching step to the front
        (with an optional ``min_score`` gate). The global ``min_score`` floor
        still applies afterwards.
        """
        rules = self._rules(config)
        if not rules:
            return list(chain), []
        candidates = [dict(step) for step in chain]
        fired: list[dict] = []
        blocked_providers: set[str] = set()

        # Pass 1 — blocks (also collect provider-wide blocks).
        for rule in rules:
            if rule.get("action") != "block" or not self._rule_matches(rule, task, profile):
                continue
            if not rule.get("provider") and not rule.get("model"):
                continue
            before = len(candidates)
            candidates = [s for s in candidates if not self._rule_target(rule, s)]
            if len(candidates) < before:
                if rule.get("provider") and not rule.get("model"):
                    blocked_providers.add(rule["provider"])
                fired.append({
                    "action": "block",
                    "provider": rule.get("provider") or "*",
                    "model": rule.get("model") or "*",
                    "profile": rule.get("profile", "*"),
                    "task": rule.get("task", "*"),
                })

        # Pass 2 — prefers (first-match wins). A prefer rule means "this task
        # must use the preferred model on EVERY provider that serves it, in the
        # chain's provider order, before any other model". We EXPAND the rule to
        # one step per unique provider (deduped, first-seen order) that serves
        # the preferred model — so a degraded provider falls to the NEXT provider
        # of the SAME model, not to a cheaper one. Provider-only prefers (no
        # model) keep the group-to-front behavior. Later prefers are ignored.
        for rule in rules:
            if rule.get("action") != "prefer" or not self._rule_matches(rule, task, profile):
                continue
            if not rule.get("provider") and not rule.get("model"):
                continue
            rp = rule.get("provider") or "*"
            rm = rule.get("model") or "*"
            gate = rule.get("min_score")

            pref_logical = None if rm in ("*", "") else logical_model_name(rm, self.db_path)

            if pref_logical is not None:
                # min_score gate on the preferred model's capability.
                if gate is not None:
                    cap = self.get_model_score(pref_logical, task)
                    if cap < float(gate):
                        fired.append({
                            "action": "prefer_skipped_low_score",
                            "provider": rp, "model": rm,
                            "score": round(cap, 3), "min_score": float(gate),
                        })
                        continue

                # One preferred step per unique provider (chain order) that
                # serves the preferred model.
                preferred: list[dict] = []
                pref_keys: set[tuple] = set()
                seen: set[str] = set()
                for s in candidates:
                    p = s["provider"]
                    if p in seen:
                        continue
                    seen.add(p)
                    if rp not in ("*", "") and p != rp:
                        continue
                    if not self._provider_serves_model(p, pref_logical, config):
                        continue
                    model_id = provider_model_name(pref_logical, p, self.db_path)
                    step = {"provider": p, "model": model_id}
                    if s.get("base_url"):
                        step["base_url"] = s["base_url"]
                    preferred.append(step)
                    pref_keys.add((p, model_id))

                if not preferred:
                    continue
                rest = [s for s in candidates
                        if (s["provider"], s["model"]) not in pref_keys]
                candidates = preferred + rest
                fired.append({
                    "action": "prefer",
                    "provider": rp, "model": rm,
                    "profile": rule.get("profile", "*"),
                    "task": rule.get("task", "*"),
                    "steps": len(preferred),
                })
                break

            # Provider-only prefer: group the provider's steps to the front.
            matches = [i for i, s in enumerate(candidates)
                       if self._rule_target(rule, s)]
            if not matches:
                continue
            matched = [candidates[i] for i in matches]
            match_set = set(matches)
            rest = [s for i, s in enumerate(candidates) if i not in match_set]
            candidates = matched + rest
            fired.append({
                "action": "prefer",
                "provider": rp, "model": rm,
                "profile": rule.get("profile", "*"),
                "task": rule.get("task", "*"),
                "steps": len(matched),
            })
            break

        return candidates, fired

    def select_step(self, messages: list[dict], tools: Optional[list[dict]] = None,
                    max_tokens: int = 1024, chain: Optional[list[dict]] = None,
                    profile: Optional[str] = None, config: Optional[object] = None,
                    ) -> Optional[list[dict]]:
        """Provider-aware selection: reorder a COPY of the chain so the best
        (provider, model) step is tried first.

        Policy (from runtime settings, falling back to config):
          - ``eager``     : deterministic — reorder when the best step beats the
                            current first step by more than the hysteresis.
          - ``cost_first``: same, but with a stronger cost-bias boost.
          - ``explore``   : weighted random pick among the steps within the
                            hysteresis of the best (spread traffic + A/B).

        ``min_score`` is a floor: no reorder unless the best step's score is
        >= it. Returns None to keep the chain's existing order. The caller
        applies it to its own copy — this never mutates the profile config.
        """
        if not self.is_enabled(config) or not chain:
            return None
        policy, min_score = self._effective_policy(config)
        task = classify_task(messages, tools, max_tokens)

        # The circuit breaker only GATES PROVIDERS; the router owns model
        # selection and ordering. Drop steps whose provider is currently
        # unavailable (dead / hard-tripped) so the ordering never proposes a
        # provider the breaker would skip, and so a degraded provider serving
        # the preferred model falls to the NEXT provider of the same model
        # instead of a cheaper model. (When routing is off, try_chain uses the
        # static chain + breaker as before.)
        chain = [s for s in chain if self._provider_available(s, profile, config)]
        if not chain:
            return None

        # Apply UI-defined rules: blocks filter candidates; prefers pin a step;
        # policy rules override the policy for this scope.
        for rule in self._rules(config):
            if rule.get("action") == "policy" and self._rule_matches(rule, task, profile or ""):
                p = rule.get("policy")
                if p in ("eager", "cost_first", "explore"):
                    policy = p

        candidates, fired_rules = self._apply_rules(
            chain, task, profile or "", config)
        if not candidates:
            return None
        chain = candidates
        fired_desc = [f["action"] for f in fired_rules]

        # Audit: why didn't a rule fire? Surfaces silent rule misses in the
        # decision log (e.g. "rules exist for planning but task=agentic…").
        note = None
        rules = self._rules(config)
        if rules:
            scope_matched = [r for r in rules
                             if self._rule_matches(r, task, profile or "")]
            if scope_matched and not fired_desc:
                note = (
                    f"{len(scope_matched)} rule(s) matched scope for task "
                    f"{task!r} but none fired"
                )
            elif scope_matched:
                note = (
                    f"{len(scope_matched)} rule(s) matched scope; fired: "
                    f"{', '.join(fired_desc) or 'none'}"
                )

        # A fired `prefer` rule is MANDATORY: the preferred step is pinned
        # first and the router must NOT reorder away from it. Eager/cost_first/
        # explore scoring and the min_score floor are bypassed for this request.
        # Chain fallback still applies at call time (try_chain skips the pinned
        # step if its provider is dead or vision-incompatible).
        if "prefer" in fired_desc:
            pinned = chain[0]
            self._record_decision({
                "ts": datetime.now(timezone.utc).isoformat(),
                "profile": profile or "", "task": task, "policy": policy,
                "action": "prefer", "model": pinned["model"],
                "provider": pinned["provider"],
                "score": round(self.score_step(pinned, task, profile, config), 3),
                "rules": fired_desc, "note": note,
            })
            return chain

        # cost_first: boost the cost component of the score.
        bias = None
        if policy == "cost_first":
            bias = min(self.cost_bias + 0.15, 0.5)

        scored = sorted(
            ((self.score_step(step, task, profile, config, bias=bias), i, step)
             for i, step in enumerate(chain)),
            key=lambda x: -x[0],
        )
        best_score, best_idx, best = scored[0]
        default_score = next((s for s, i, _ in scored if i == 0), best_score)
        if best_score < min_score:
            self._record_decision({
                "ts": datetime.now(timezone.utc).isoformat(),
                "profile": profile or "", "task": task, "policy": policy,
                "action": "below_min_score", "model": best["model"],
                "provider": best["provider"], "score": round(best_score, 3),
                "rules": fired_desc, "note": note,
            })
            return None

        # explore: weighted random among steps within hysteresis of the best.
        if policy == "explore" and len(scored) > 1:
            import random
            cutoff = best_score - _HYSTERESIS
            pool = [s for s in scored if s[0] >= cutoff]
            if len(pool) > 1:
                weights = [max(s[0], 0.0) + 0.05 for s in pool]
                picked_score, pick_idx, pick = random.choices(pool, weights=weights, k=1)[0]
                if pick_idx != 0:
                    rest = [st for i, st in enumerate(chain) if i != pick_idx]
                    self._record_decision({
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "profile": profile or "", "task": task, "policy": policy,
                        "action": "explore", "model": pick["model"],
                        "provider": pick["provider"], "score": round(picked_score, 3),
                        "rules": fired_desc, "note": note,
                    })
                    return [pick, *rest]
                self._record_decision({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "profile": profile or "", "task": task, "policy": policy,
                    "action": "keep_default", "model": chain[0]["model"],
                    "provider": chain[0]["provider"], "score": round(best_score, 3),
                    "rules": fired_desc, "note": note,
                })
                return None

        if best_idx == 0 or best_score <= default_score + _HYSTERESIS:
            self._record_decision({
                "ts": datetime.now(timezone.utc).isoformat(),
                "profile": profile or "", "task": task, "policy": policy,
                "action": "keep_default", "model": chain[0]["model"],
                "provider": chain[0]["provider"], "score": round(default_score, 3),
                "rules": fired_desc, "note": note,
            })
            return None

        rest = [step for i, step in enumerate(chain) if i != best_idx]
        logger.info(
            "router_step_override",
            task=task, profile=profile or "",
            from_provider=chain[0]["provider"], from_model=chain[0]["model"],
            to_provider=best["provider"], to_model=best["model"],
            default_score=round(default_score, 3),
            recommended_score=round(best_score, 3),
        )
        self._record_decision({
            "ts": datetime.now(timezone.utc).isoformat(),
            "profile": profile or "", "task": task, "policy": policy,
            "action": "reorder", "model": best["model"],
            "provider": best["provider"], "score": round(best_score, 3),
            "from_model": chain[0]["model"], "from_provider": chain[0]["provider"],
            "rules": fired_desc, "note": note,
        })
        return [best, *rest]


# ── Global instances ──────────────────────────────────────────────────────────

_dynamic_router = CapabilityRouter(enabled=False)


def get_dynamic_router() -> CapabilityRouter:
    return _dynamic_router


def init_router(db_path: str = "data/costs.db", enabled: bool = False,
                cost_bias: float = DEFAULT_COST_BIAS):
    """Initialize the global router. Call once at startup."""
    global _dynamic_router
    _dynamic_router = CapabilityRouter(enabled=enabled, db_path=db_path,
                                       cost_bias=cost_bias)
    if enabled:
        _dynamic_router.load_matrix()  # warm the cache


def sync_router_enabled_from_settings() -> bool:
    """Re-apply the persisted ``routing_enabled`` toggle to the global router.

    Boot seeds ``enabled`` from ``gateway.yaml`` (which may be the ``false``
    baseline). Once the settings store is available (after ``init_settings``),
    this re-syncs so the effective state — and the boot log — reflects the UI
    toggle. Returns the effective enabled state.
    """
    global _dynamic_router
    try:
        from .cost_cache import get_settings
        settings = get_settings()
        if settings is not None:
            override = settings.get_routing_enabled(default=None)
            if override is not None:
                _dynamic_router.enabled = override
    except Exception:  # noqa: BLE001 — never fail boot
        pass
    return _dynamic_router.enabled


def invalidate_router_matrix() -> None:
    """Drop the global router's cached capability matrix.

    Called when a benchmark run completes or the registry changes so routing
    picks up fresh scores without a restart.
    """
    _dynamic_router.invalidate_matrix()


def routing_status(config: Optional[object] = None) -> dict:
    """Return a snapshot for the UI (Providers → Routing tab).

    Includes enabled state, effective policy/min_score, the top recommended
    model per task (from the capability matrix, restricted to models that are
    actually selected — i.e. referenced by a profile chain), the active rules,
    and recent decisions.
    """
    router = get_dynamic_router()

    # Models the router can actually pick = those named in any profile chain.
    selected: set[str] = set()
    try:
        profiles = (config or {}).profiles or {}
        for pcfg in profiles.values():
            for step in (pcfg.get("chain") or []):
                m = step.get("model")
                if not m:
                    continue
                selected.add(m)
                selected.add(normalize_model_id(m))
                try:
                    selected.add(benchmark_model_name(
                        logical_model_name(m, router.db_path), router.db_path))
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        selected = set()

    policy, min_score = router._effective_policy(config)
    profiles: list = []
    providers: list = []
    try:
        if config is not None:
            profiles = sorted((config.profiles or {}).keys())
            providers = sorted((config.providers or {}).keys())
    except Exception:  # noqa: BLE001
        pass
    per_task: dict[str, dict] = {}
    try:
        matrix = router.load_matrix()
        for task, scores in matrix.items():
            if not scores:
                continue
            # Only recommend models that are selected; fall back to the overall
            # top when no selection info is available (e.g. tests/no config).
            candidates = {k: v for k, v in scores.items()
                          if not selected or k in selected}
            if not candidates:
                continue
            top = max(candidates.items(), key=lambda kv: kv[1])
            per_task[task] = {"model": top[0], "score": round(top[1], 3)}
    except Exception:  # noqa: BLE001
        pass
    return {
        "enabled": router.is_enabled(config),
        "policy": policy,
        "min_score": min_score,
        "rules": router._rules(config),
        "per_task": per_task,
        "recent_decisions": router.recent_decisions(25),
        # For rule-dropdowns in the UI: available profiles and providers.
        "profiles": profiles,
        "providers": providers,
    }


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
