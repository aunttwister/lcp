"""Tests for the reasoning-content store and capture helpers."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.api.reasoning_store import (
    ReasoningStore,
    get_reasoning_store,
    reset_reasoning_store,
)
from src.api.request_pipeline import (
    capture_reasoning_from_response,
    capture_reasoning_from_sse,
    ensure_thinking_reasoning_content,
)


@pytest.fixture(autouse=True)
def _reset_store():
    reset_reasoning_store()
    yield
    reset_reasoning_store()


class TestReasoningStore:
    def test_capture_and_get(self):
        store = ReasoningStore()
        store.capture(["call_1", "call_2"], "Let me think about this")
        assert store.get_for_tool_call_id("call_1") == "Let me think about this"
        assert store.get_for_tool_call_id("call_2") == "Let me think about this"

    def test_capture_accepts_single_string(self):
        store = ReasoningStore()
        store.capture("call_1", "thinking")
        assert store.get_for_tool_call_id("call_1") == "thinking"

    def test_missing_id_returns_none(self):
        store = ReasoningStore()
        assert store.get_for_tool_call_id("never_seen") is None

    def test_empty_ids_noop(self):
        store = ReasoningStore()
        store.capture([], "nothing")
        store.capture(None, "nothing")
        assert len(store) == 0

    def test_rehydrate_attaches_stored_content(self):
        store = ReasoningStore()
        store.capture("call_9", "I should check the weather first")
        messages = [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "Let me look",
             "tool_calls": [{"id": "call_9", "type": "function",
                             "function": {"name": "get_weather", "arguments": "{}"}}]},
        ]
        store.rehydrate(messages)
        assert messages[1]["reasoning_content"] == "I should check the weather first"

    def test_rehydrate_skips_messages_with_existing_content(self):
        store = ReasoningStore()
        store.capture("call_1", "stored reasoning")
        messages = [
            {"role": "assistant", "content": "x",
             "reasoning_content": "already here",
             "tool_calls": [{"id": "call_1"}]},
        ]
        store.rehydrate(messages)
        assert messages[0]["reasoning_content"] == "already here"

    def test_rehydrate_skips_non_tool_call_messages(self):
        store = ReasoningStore()
        store.capture("call_1", "stored")
        messages = [{"role": "assistant", "content": "plain turn"}]
        store.rehydrate(messages)
        assert "reasoning_content" not in messages[0]

    def test_ttl_expiry_drops_entry(self):
        import time as _time
        store = ReasoningStore(ttl_seconds=1)
        store.capture("call_1", "stored")
        # Advance time past TTL — get should return None and drop the entry
        with patch("time.time", return_value=_time.time() + 60):
            assert store.get_for_tool_call_id("call_1") is None
        assert len(store) == 0

    def test_max_entries_prunes_oldest(self):
        store = ReasoningStore(max_entries=2)
        store.capture(["a"], "1")
        store.capture(["b"], "2")
        store.capture(["c"], "3")  # exceeds cap → drops oldest ('a')
        assert store.get_for_tool_call_id("a") is None
        assert store.get_for_tool_call_id("b") == "2"
        assert store.get_for_tool_call_id("c") == "3"


class TestCaptureReasoning:
    def test_capture_from_response_body(self):
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Here you go",
                    "reasoning_content": "User needs a function",
                    "tool_calls": [{"id": "call_abc", "type": "function",
                                    "function": {"name": "write_file", "arguments": "{}"}}],
                }
            }]
        }
        capture_reasoning_from_response(response)
        store = get_reasoning_store()
        assert store.get_for_tool_call_id("call_abc") == "User needs a function"

    def test_capture_from_response_no_tool_calls(self):
        response = {"choices": [{"message": {"content": "plain", "reasoning_content": "think"}}]}
        capture_reasoning_from_response(response)
        assert len(get_reasoning_store()) == 0

    def test_capture_from_sse(self):
        sse = (
            b'data: {"choices":[{"delta":{"role":"assistant","reasoning_content":"First "}}]}\n\n'
            b'data: {"choices":[{"delta":{"reasoning_content":"thought"}}]}\n\n'
            b'data: {"choices":[{"delta":{"tool_calls":[{"id":"call_sse1","type":"function"}]}}]}\n\n'
            b'data: [DONE]\n\n'
        )
        capture_reasoning_from_sse(sse)
        store = get_reasoning_store()
        assert store.get_for_tool_call_id("call_sse1") == "First thought"

    def test_capture_from_sse_no_tool_ids(self):
        sse = b'data: {"choices":[{"delta":{"reasoning_content":"just thinking"}}]}\n\n'
        capture_reasoning_from_sse(sse)
        assert len(get_reasoning_store()) == 0

    def test_capture_handles_garbage(self):
        capture_reasoning_from_response({})
        capture_reasoning_from_response(None)
        capture_reasoning_from_sse(b"not sse at all")
        assert len(get_reasoning_store()) == 0


class TestEnsureWithStore:
    def _thinking_config(self):
        cfg = MagicMock()
        cfg.get_model_limits.return_value = {"supports_thinking": True}
        return cfg

    def test_rehydrates_real_content_before_empty_fallback(self):
        store = get_reasoning_store()
        store.capture("call_42", "Real chain of thought text")
        messages = [
            {"role": "assistant", "content": "x",
             "tool_calls": [{"id": "call_42"}]},
            {"role": "assistant", "content": "y",
             "tool_calls": [{"id": "call_unknown"}]},
        ]
        result = ensure_thinking_reasoning_content(
            messages, "deepseek-v4-pro", self._thinking_config())
        # Known id → real content
        assert result[0]["reasoning_content"] == "Real chain of thought text"
        # Unknown id → empty presence fallback
        assert result[1]["reasoning_content"] == ""
