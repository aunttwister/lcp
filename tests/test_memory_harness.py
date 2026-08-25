"""Tests for the memory harness (src/api/memory/harness.py)."""

import pytest

from src.api.memory.harness import (
    _build_context_block,
    _latest_user_text,
    config_for,
    inject_memory_context,
    recall_for_request,
)


class _FakeBackend:
    """A scripted backend that returns preset memories on recall."""

    def __init__(self, results):
        self._results = results
        self.calls = []

    def recall(self, query, top_k=10, tag_filter=None, profile="default"):
        self.calls.append({"query": query, "top_k": top_k,
                           "tag_filter": tag_filter, "profile": profile})
        return list(self._results)


MEMORIES = [
    {"id": "1", "content": "node01 has an RTX 3090 GPU", "metadata": {"host": "node01"},
     "tags": ["gpu"], "score": 0.91},
    {"id": "2", "content": "wifi password is hunter2", "metadata": {},
     "tags": [], "score": 0.6},
]


class TestLatestUserText:
    def test_returns_last_user_message(self):
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "second"}]
        assert _latest_user_text(msgs) == "second"

    def test_handles_list_content(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hello there"}]}]
        assert _latest_user_text(msgs) == "hello there"

    def test_empty_when_no_user(self):
        assert _latest_user_text([{"role": "system", "content": "x"}]) == ""


class TestBuildContextBlock:
    def test_renders_memories(self):
        block = _build_context_block(MEMORIES)
        assert "node01" in block and "RTX 3090" in block
        assert "wifi password is hunter2" in block

    def test_empty_input(self):
        assert _build_context_block([]) == ""

    def test_skips_blank(self):
        assert _build_context_block([{"content": "  "}]) == ""


class TestRecallForRequest:
    def test_recalls_matching_profile(self, monkeypatch):
        backend = _FakeBackend(MEMORIES)
        monkeypatch.setattr("src.api.memory.get_memory", lambda: backend)
        results = recall_for_request(
            [{"role": "user", "content": "which gpu in node01"}],
            profile="l2", top_k=3,
        )
        assert results == MEMORIES
        assert backend.calls[0]["profile"] == "l2"
        assert backend.calls[0]["top_k"] == 3

    def test_noop_when_memory_inactive(self, monkeypatch):
        monkeypatch.setattr("src.api.memory.get_memory", lambda: None)
        assert recall_for_request([{"role": "user", "content": "x"}]) == []

    def test_empty_query_returns_empty(self, monkeypatch):
        backend = _FakeBackend(MEMORIES)
        monkeypatch.setattr("src.api.memory.get_memory", lambda: backend)
        assert recall_for_request([{"role": "system", "content": "x"}]) == []

    def test_min_score_filter(self, monkeypatch):
        backend = _FakeBackend(MEMORIES)
        monkeypatch.setattr("src.api.memory.get_memory", lambda: backend)
        results = recall_for_request(
            [{"role": "user", "content": "x"}], min_score=0.8,
        )
        assert [r["id"] for r in results] == ["1"]  # only score 0.91

    def test_recall_failure_returns_empty(self, monkeypatch):
        class _Bad:
            def recall(self, *a, **k):
                raise RuntimeError("boom")
        monkeypatch.setattr("src.api.memory.get_memory", lambda: _Bad())
        assert recall_for_request([{"role": "user", "content": "x"}]) == []


class TestInjectMemoryContext:
    def test_prepends_context_when_enabled(self, monkeypatch):
        backend = _FakeBackend(MEMORIES)
        monkeypatch.setattr("src.api.memory.get_memory", lambda: backend)
        msgs = [{"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "which gpu in node01"}]
        out = inject_memory_context(msgs, profile="l2", enabled=True, top_k=3)
        # Merged into the existing system prompt, so still 2 messages.
        assert len(out) == 2
        assert out[0]["role"] == "system"
        assert "RTX 3090" in out[0]["content"]
        assert "You are a coding agent." in out[0]["content"]
        assert out[1] == msgs[1]

    def test_disabled_returns_unchanged(self, monkeypatch):
        backend = _FakeBackend(MEMORIES)
        monkeypatch.setattr("src.api.memory.get_memory", lambda: backend)
        msgs = [{"role": "user", "content": "hi"}]
        assert inject_memory_context(msgs, enabled=False) is msgs

    def test_no_memories_returns_unchanged(self, monkeypatch):
        backend = _FakeBackend([])
        monkeypatch.setattr("src.api.memory.get_memory", lambda: backend)
        msgs = [{"role": "user", "content": "hi"}]
        assert inject_memory_context(msgs, enabled=True) is msgs

    def test_no_system_prompt_prepends_new(self, monkeypatch):
        backend = _FakeBackend(MEMORIES)
        monkeypatch.setattr("src.api.memory.get_memory", lambda: backend)
        msgs = [{"role": "user", "content": "which gpu"}]
        out = inject_memory_context(msgs, enabled=True)
        assert out[0]["role"] == "system"
        assert "RTX 3090" in out[0]["content"]
        assert out[1] == msgs[0]


class TestConfigFor:
    def test_defaults_when_no_memory_cfg(self):
        class _C:
            plugins = {}
        cfg = config_for(_C())
        assert cfg["enabled"] is False
        assert cfg["top_k"] == 3
        assert cfg["min_score"] == 0.0

    def test_reads_memory_cfg(self):
        class _C:
            plugins = {"memory": {"auto_recall": True, "top_k": 5, "min_score": 0.4}}
        cfg = config_for(_C())
        assert cfg["enabled"] is True
        assert cfg["top_k"] == 5
        assert cfg["min_score"] == 0.4

    def test_none_config(self):
        cfg = config_for(None)
        assert cfg["enabled"] is False