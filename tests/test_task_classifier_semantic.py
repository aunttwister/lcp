"""Tests for the semantic task classifier (src/api/task_classifier.py)."""

import pytest

from src.api.task_classifier import (
    SemanticClassifier,
    TASK_EXEMPLARS,
    get_semantic_classifier,
    invalidate_semantic_classifier,
    _cosine,
)


# ── Fake embedder (keyword-hash based, deterministic, non-zero vectors) ────
VOCAB = {
    "test": 0, "pytest": 0, "unit": 0, "bug": 1, "debug": 1, "error": 1,
    "write": 2, "implement": 2, "function": 2, "code": 2,
    "design": 3, "architecture": 3, "plan": 3, "roadmap": 3,
    "explain": 4, "analyze": 4, "research": 4, "prove": 5, "solve": 5,
    "calculate": 5, "hello": 6, "hi": 6, "thanks": 6, "agent": 7, "tools": 7,
}


def fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * 8
        for w in t.lower().split():
            idx = VOCAB.get(w.strip(".,:;()"))
            if idx is not None:
                v[idx % 8] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / norm for x in v])
    return out


@pytest.fixture
def classifier():
    return SemanticClassifier(embed=fake_embed, min_score=0.05)


class TestCosine:
    def test_identical(self):
        assert _cosine([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        assert _cosine([], [1, 2]) == 0.0
        assert _cosine([1], []) == 0.0


class TestSemanticClassifier:
    def test_classify_unit_tests(self, classifier):
        assert classifier.classify("write a unit test with pytest") == "unit_tests"

    def test_classify_debugging(self, classifier):
        assert classifier.classify("help me debug this error traceback") == "debugging"

    def test_classify_code_generation(self, classifier):
        assert classifier.classify("implement a function to write code") == "code_generation"

    def test_classify_planning(self, classifier):
        assert classifier.classify("design the architecture and roadmap") == "planning"

    def test_classify_casual(self, classifier):
        assert classifier.classify("hello there") == "casual_chat"

    def test_low_confidence_returns_none(self, classifier):
        clf = SemanticClassifier(embed=fake_embed, min_score=0.99)
        assert clf.classify("zzz qqq unknown") is None

    def test_available_false_without_embed(self):
        assert SemanticClassifier(embed=None).available is False

    def test_classify_returns_none_when_unavailable(self):
        clf = SemanticClassifier(embed=None)
        assert clf.classify("hello") is None

    def test_build_centroids_cached(self, classifier):
        classifier._build_centroids()
        first = classifier._centroids
        classifier._build_centroids()
        assert classifier._centroids is first  # cached


class TestExemplars:
    def test_all_tasks_have_exemplars(self):
        for task in ("agentic_multi_step", "unit_tests", "code_generation",
                     "debugging", "research_deep", "reasoning_chain",
                     "planning", "casual_chat"):
            assert TASK_EXEMPLARS[task], f"{task} has no exemplars"


class TestGetClassifier:
    def test_returns_none_when_router_disabled(self, monkeypatch):
        # No router config / no embedder -> None (keyword fallback).
        invalidate_semantic_classifier()
        _Cfg = type("C", (), {"plugins": {"router": {"enabled": False}}})
        monkeypatch.setattr("src.api.config.get_config", lambda: _Cfg())
        assert get_semantic_classifier() is None
        invalidate_semantic_classifier()

    def test_returns_none_when_router_absent(self, monkeypatch):
        # No plugins config at all -> enabled by default; whether a real
        # classifier is returned depends on the environment (the embedder may
        # be installed or not). Assert it doesn't crash and, when the embedder
        # is the noop fallback, returns None.
        invalidate_semantic_classifier()
        _M = type("M", (), {"embed": lambda texts: [[0.0] * 384 for _ in texts]})
        monkeypatch.setattr(
            "src.api.memory.embeddings.EmbeddingModel", lambda **k: _M()
        )
        _Cfg = type("C", (), {"plugins": {}})
        monkeypatch.setattr("src.api.config.get_config", lambda: _Cfg())
        # All-zero (noop) embedder -> None (keyword fallback).
        assert get_semantic_classifier() is None
        invalidate_semantic_classifier()


class TestTopScores:
    def test_sorted_desc_and_capped(self, classifier):
        scores = classifier.top_scores("write a unit test with pytest", 3)
        assert scores[0][0] == "unit_tests"
        assert scores == sorted(scores, key=lambda x: -x[1])
        assert len(scores) <= 3

    def test_min_score_not_applied(self, classifier):
        # A high gate makes classify() return None, but top_scores still
        # surfaces the scores (near-threshold visibility for observability).
        clf = SemanticClassifier(embed=fake_embed, min_score=0.99)
        assert clf.classify("zzz qqq unknown") is None
        assert clf.top_scores("zzz qqq unknown", 2)

    def test_empty_when_unavailable(self):
        assert SemanticClassifier(embed=None).top_scores("hello") == []

    def test_classify_delegates_to_top_scores(self, classifier):
        assert classifier.classify("write a unit test with pytest") == \
            classifier.top_scores("write a unit test with pytest", 1)[0][0]

    def test_exposes_effective_min_score(self, classifier):
        assert classifier.min_score == 0.05
        clf = SemanticClassifier(embed=fake_embed, min_score=0.5)
        assert clf.min_score == 0.5

    def test_probe_rejects_noop_embed(self, monkeypatch):
        invalidate_semantic_classifier()
        _M = type("M", (), {"embed": lambda texts: [[0.0] * 384 for _ in texts]})
        monkeypatch.setattr(
            "src.api.memory.embeddings.EmbeddingModel", lambda **k: _M()
        )
        _Cfg = type("C", (), {"plugins": {"router": {"enabled": True}}})
        monkeypatch.setattr("src.api.config.get_config", lambda: _Cfg())
        # All-zero vectors -> probed as not real -> None.
        assert get_semantic_classifier() is None
        invalidate_semantic_classifier()