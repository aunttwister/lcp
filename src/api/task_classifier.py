"""Semantic task classification for the dynamic router.

The router's ``classify_task`` is keyword/heuristic. This module adds an
optional **embedding-based** classifier that uses its OWN embedder
(bge-small-en-v1.5, 384-dim) so a prompt is classified by its *meaning* rather
than exact keywords.

This is the SEMANTIC ROUTING module — it is independent of the memory plugin.
Its embedder comes from the ``plugins.router`` config block (installed as the
``router`` module, grouped with LiveBench), not ``plugins.memory``.

Design
------
* Each task type has a small set of **exemplar prompts** (canonical examples).
  When the embedder is available, the exemplars are embedded once (cached) and
  averaged into a per-task centroid. A prompt is embedded and classified to the
  nearest centroid by cosine similarity (threshold-gated).
* When no embedder is available (sentence-transformers not installed), the
  module is a no-op and the caller falls back to the existing keyword path.
* It returns the SAME task strings as the keyword classifier, so the rest of
  the router is unchanged: ``agentic_multi_step``, ``unit_tests``,
  ``code_generation``, ``debugging``, ``research_deep``, ``reasoning_chain``,
  ``planning``, ``casual_chat``.
"""

from __future__ import annotations

import math
from typing import Optional

from .memory.embeddings import DEFAULT_MODEL

# Task type -> canonical exemplar prompts (short, representative).
TASK_EXEMPLARS: dict[str, list[str]] = {
    "agentic_multi_step": [
        "You are an AI agent. Use the available tools to complete this multi-step task.",
        "Use your tools to browse the web and gather information, then act on it.",
        "Call the functions provided to fulfill this request step by step.",
        "You have access to tools — delegate and execute the plan autonomously.",
    ],
    "unit_tests": [
        "Write a pytest suite covering this function's edge cases.",
        "Add unit tests for the new module, including mocking the database.",
        "Create test cases for the API endpoint and assert the responses.",
        "Write a test for this bug to prevent regression.",
        "Run the test suite and report which tests fail.",
    ],
    "code_generation": [
        "Write a Python function that parses this CSV file.",
        "Implement a REST endpoint in FastAPI that returns JSON.",
        "Create a bash script to deploy this service.",
        "Write a React component that renders a list of items.",
        "Review this existing file and make the changes we discussed.",
        "Refactor this function to be cleaner and update the callers.",
        "Look at this code and fix the issue in this file.",
        # "algorithm" in isolation reads like reasoning; pin the implement-a-
        # function/algorithm-in-code sense to code_generation, not reasoning.
        "Implement a sorting algorithm in Python.",
        "Write a function that implements this algorithm.",
        "Implement the binary search algorithm in code.",
    ],
    "debugging": [
        "Why does this code throw a KeyError? Here is the traceback.",
        "Debug this failing test — it works locally but not in CI.",
        "This endpoint returns 500. Help me find the bug.",
        "The build is failing with an import error. What's wrong?",
    ],
    "research_deep": [
        "Explain how attention mechanisms work in transformers, in detail.",
        "Analyze the trade-offs between SQL and NoSQL for this workload.",
        "Compare and contrast microservices and monoliths thoroughly.",
        "Research the latest approaches to vector databases and summarize.",
    ],
    "reasoning_chain": [
        "Solve this logic puzzle step by step.",
        "Prove that this algorithm has O(n log n) complexity.",
        "Calculate the time complexity of this recurrence relation.",
        "Work through this mathematical problem carefully.",
    ],
    "planning": [
        "Design the architecture for a multi-tenant SaaS product.",
        "How should I structure this microservice codebase?",
        "Create a roadmap and data model for this new feature.",
        "Plan the tech stack and schema for this application.",
        # "make a plan to implement X" / "plan for X" is planning, not codegen.
        "Make a plan for implementing this feature.",
        "Plan out the steps to build this component.",
        "Let's plan how to implement this change.",
    ],
    "casual_chat": [
        "hello",
        "hi, how are you?",
        "thanks!",
        "good morning",
        "what's up?",
    ],
}


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors (0..1)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


class SemanticClassifier:
    """Embedding-based task classifier with a keyword-free fallback.

    Lazily embeds the task exemplars on first use. When no embedder is
    available, ``classify()`` returns None and the caller should use the
    keyword path.
    """

    # Cosine-similarity floor for a semantic match to be trusted. A low value
    # lets a generic coding message land on a broad centroid (e.g. unit_tests)
    # even when the user's intent is clearly something else. High-confidence
    # semantic matches only.
    DEFAULT_MIN_SCORE = 0.35

    def __init__(self, embed=None, min_score: float = DEFAULT_MIN_SCORE):
        self._embed = embed  # callable(texts: list[str]) -> list[list[float]]
        self._min_score = min_score
        self._centroids: Optional[dict[str, list[float]]] = None
        self._task_order: list[str] = list(TASK_EXEMPLARS.keys())

    @property
    def available(self) -> bool:
        return self._embed is not None

    @property
    def min_score(self) -> float:
        """The effective cosine gate applied by ``classify()`` (read-only)."""
        return self._min_score

    def _build_centroids(self) -> None:
        if self._centroids is not None or not self.available:
            return
        centroids: dict[str, list[float]] = {}
        for task, exemplars in TASK_EXEMPLARS.items():
            if not exemplars:
                continue
            try:
                vectors = self._embed(exemplars)  # type: ignore[misc]
            except Exception:
                continue
            if not vectors:
                continue
            dim = len(vectors[0])
            if dim == 0:
                continue
            centroid = [0.0] * dim
            for v in vectors:
                for i in range(dim):
                    centroid[i] += v[i]
            centroid = [x / len(vectors) for x in centroid]
            # Normalize the centroid.
            norm = math.sqrt(sum(x * x for x in centroid)) or 1.0
            centroids[task] = [x / norm for x in centroid]
        self._centroids = centroids if centroids else {}

    def top_scores(self, text: str, k: int = 5) -> list[tuple[str, float]]:
        """Return the top-k ``(task, cosine)`` pairs for *text*, sorted desc.

        Returns ``[]`` when the embedder is unavailable or embedding fails.
        Does NOT apply the ``min_score`` gate — the caller decides, so the
        router can surface near-threshold scores for observability.
        """
        if not self.available or not text:
            return []
        self._build_centroids()
        if not self._centroids:
            return []
        try:
            vec = self._embed([text])[0]  # type: ignore[misc]
        except Exception:
            return []
        scored = sorted(
            ((task, _cosine(vec, centroid))
             for task, centroid in self._centroids.items()),
            key=lambda x: -x[1],
        )
        return scored[:k]

    def classify(self, text: str) -> Optional[str]:
        """Return the best task for *text*, or None when unavailable/uncertain.

        Uses cosine similarity to the nearest task centroid, gated by
        ``min_score`` so low-confidence prompts fall through to the keyword
        path.
        """
        if not self.available or not text:
            return None
        self._build_centroids()
        if not self._centroids:
            return None
        scores = self.top_scores(text, 1)
        if not scores:
            return None
        best_task, best_score = scores[0]
        if best_score < self._min_score:
            return None
        return best_task


# Module-level cached classifier (built lazily from the memory embedder).
_classifier: Optional[SemanticClassifier] = None


def _probe_embed(embed) -> bool:
    """Return True when *embed* actually produces real (non-zero) vectors.

    Guards against the memory plugin's ``_noop_embed`` fallback (all-zero
    vectors) and against sentence-transformers being absent (raises) — both of
    which would produce garbage classification.
    """
    if embed is None:
        return False
    try:
        vec = embed(["semantic probe"])[0]
    except Exception:
        return False
    if not vec:
        return False
    return any(abs(x) > 1e-9 for x in vec)


def get_semantic_classifier() -> Optional[SemanticClassifier]:
    """Return the shared classifier if a real embedder is available, else None.

    Builds an ``EmbeddingModel`` from ``plugins.router`` (the SEMANTIC ROUTING
    module — independent of the memory plugin) and probes it. When
    sentence-transformers isn't installed (or the embedder is the noop
    fallback), returns None so callers use the keyword path.

    This intentionally does NOT depend on ``plugins.memory``: semantic routing
    and the memory bank are separate installable modules. Both share the same
    embedder type, but the router module is what powers task classification.
    """
    global _classifier
    if _classifier is not None:
        return _classifier
    try:
        from ..api.config import get_config
        from .memory.embeddings import EmbeddingModel

        config = get_config()
        router_cfg = (getattr(config, "plugins", None) or {}).get("router") or {}
        if not router_cfg.get("enabled", True):
            return None
        # The router embedder lives in the router module's own deps dir so it
        # is independent from the memory plugin's install.
        from .memory import memory_models as _router_models  # reuse models dir pattern
        models_dir = router_cfg.get("models_dir") or _router_models()
        model = EmbeddingModel(
            model_name=router_cfg.get("embedding", {}).get("model", DEFAULT_MODEL),
            device=router_cfg.get("embedding", {}).get("device", "cpu"),
            cache_dir=models_dir,
        )
        if not _probe_embed(model.embed):
            return None
        # Gate the semantic path on a HIGH-confidence match so the deterministic
        # keyword signals still win for clear intent; only genuinely ambiguous
        # prompts fall through to meaning-based classification. Configurable via
        # plugins.router.min_score.
        min_score = float(
            router_cfg.get("min_score", SemanticClassifier.DEFAULT_MIN_SCORE)
            or SemanticClassifier.DEFAULT_MIN_SCORE
        )
        _classifier = SemanticClassifier(embed=model.embed, min_score=min_score)
    except Exception:
        _classifier = None
    return _classifier if (_classifier is not None and _classifier.available) else None


def invalidate_semantic_classifier() -> None:
    """Drop the cached classifier (e.g. after router module install/uninstall)."""
    global _classifier
    _classifier = None
