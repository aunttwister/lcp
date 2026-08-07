"""Tests for main.py utility functions and HTTP endpoints."""
import json
import os
import tempfile
import threading
import time
import urllib.request
from http.server import HTTPServer
import pytest
import sys
from unittest.mock import patch, MagicMock

from src.api.request_pipeline import (
    strip_forbidden_tools,
    calculate_cost,
    normalize_messages_for_cache,
    has_image_content,
)
from src.api.circuit_breaker import get_circuit_breaker
from src.server import LCPHandler


# ═══════════════════════════════════════════════════════════════════════
# strip_forbidden_tools
# ═══════════════════════════════════════════════════════════════════════

class TestStripForbiddenTools:
    def test_none_blocks_all(self):
        """None means ALL tools are forbidden, strip everything."""
        body = {"tools": [{"function": {"name": "bash"}}]}
        result, blocked = strip_forbidden_tools(body, None)
        assert result["tools"] == []
        assert blocked == ["bash"]

    def test_empty_forbidden(self):
        body = {"tools": [{"function": {"name": "bash"}}]}
        result, blocked = strip_forbidden_tools(body, [])
        assert result == body
        assert blocked == []

    def test_blocks_tool(self):
        body = {"tools": [{"function": {"name": "execute_bash"}}]}
        result, blocked = strip_forbidden_tools(body, ["execute_bash"])
        assert result["tools"] == []
        assert blocked == ["execute_bash"]

    def test_blocks_multiple(self):
        body = {"tools": [
            {"function": {"name": "bash"}},
            {"function": {"name": "terminal"}},
            {"function": {"name": "read_file"}},
        ]}
        result, blocked = strip_forbidden_tools(body, ["bash", "read_file"])
        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == "terminal"
        assert sorted(blocked) == ["bash", "read_file"]

    def test_no_tools_in_body(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        result, blocked = strip_forbidden_tools(body, ["bash"])
        assert result == body
        assert blocked == []

    def test_no_blocked(self):
        body = {"tools": [{"function": {"name": "safe_tool"}}]}
        result, blocked = strip_forbidden_tools(body, ["dangerous"])
        assert result == body
        assert blocked == []

    def test_type_field_variant(self):
        body = {"tools": [{"type": "function", "function": {"name": "block_me"}}]}
        result, blocked = strip_forbidden_tools(body, ["block_me"])
        assert result["tools"] == []
        assert blocked == ["block_me"]


# ═══════════════════════════════════════════════════════════════════════
# calculate_cost
# ═══════════════════════════════════════════════════════════════════════

class TestCalculateCost:
    @pytest.fixture
    def mock_config(self):
        cfg = MagicMock()
        cfg.get_pricing.return_value = {
            "cache_hit": 0.02,
            "cache_miss": 0.55,
            "output": 1.10,
        }
        return cfg

    def test_success_with_usage(self, mock_config):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        response = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}
        result = calculate_cost("deepseek", "deepseek-v4-flash", body, response, mock_config)
        assert "cost" in result
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50

    def test_no_response(self, mock_config):
        result = calculate_cost("deepseek", "deepseek-v4-flash", {}, None, mock_config)
        assert result["prompt_tokens"] == 0
        assert result["completion_tokens"] == 0

    def test_cache_hit_tokens(self, mock_config):
        response = {"usage": {
            "prompt_tokens": 0,
            "completion_tokens": 25,
            "prompt_cache_hit_tokens": 500,
            "prompt_cache_miss_tokens": 0,
        }}
        result = calculate_cost("deepseek", "deepseek-v4-flash", {}, response, mock_config)
        assert result["cache_hit_tokens"] == 500
        assert result["cache_miss_tokens"] == 0


# ═══════════════════════════════════════════════════════════════════════
# normalize_messages_for_cache
# ═══════════════════════════════════════════════════════════════════════

class TestNormalizeMessagesForCache:
    def test_preserves_reasoning_content(self):
        """Assistant message with reasoning_content keeps the field."""
        messages = [
            {"role": "user", "content": "Write a function"},
            {"role": "assistant", "content": "Here it is:", "reasoning_content": "Let me think about this..."},
        ]
        result = normalize_messages_for_cache(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Write a function"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Here it is:"
        assert result[1]["reasoning_content"] == "Let me think about this..."

    def test_preserves_reasoning_with_tool_calls(self):
        """Assistant with both tool_calls AND reasoning_content preserves both."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me write that file.",
                "reasoning_content": "The user wants a file written...",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "write_file", "arguments": "{}"}}
                ],
            },
        ]
        result = normalize_messages_for_cache(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["tool_calls"] == messages[0]["tool_calls"]
        assert result[0]["reasoning_content"] == "The user wants a file written..."

    def test_strips_reasoning_from_non_assistant(self):
        """System/user messages don't get reasoning_content preserved."""
        messages = [
            {"role": "system", "content": "You are helpful", "reasoning_content": "nope"},
            {"role": "user", "content": "Hello", "reasoning_content": "nope"},
        ]
        result = normalize_messages_for_cache(messages)
        assert len(result) == 2
        assert "reasoning_content" not in result[0]
        assert "reasoning_content" not in result[1]

    def test_empty_reasoning_content_is_preserved(self):
        """Empty reasoning_content must be passed through — providers (especially
        OpenCode's proxy) validate field presence, not value, in thinking-mode."""
        messages = [
            {"role": "assistant", "content": "OK", "reasoning_content": ""},
        ]
        result = normalize_messages_for_cache(messages)
        assert "reasoning_content" in result[0]
        assert result[0]["reasoning_content"] == ""

    def test_full_conversation_roundtrip(self):
        """Mixed conversation: user → assistant(with reasoning) → assistant(tool_calls + reasoning) → tool."""
        messages = [
            {"role": "user", "content": "Write a function"},
            {"role": "assistant", "content": "Sure", "reasoning_content": "I need to write a Python function..."},
            {
                "role": "assistant",
                "content": "Here is the function:",
                "reasoning_content": "The function will be simple...",
                "tool_calls": [{"id": "call_1", "function": {"name": "write_file", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Wrote file successfully."},
        ]
        result = normalize_messages_for_cache(messages)
        assert len(result) == 4
        # User
        assert result[0]["role"] == "user"
        assert "reasoning_content" not in result[0]
        # Assistant with reasoning, no tool_calls
        assert result[1]["role"] == "assistant"
        assert result[1]["reasoning_content"] == "I need to write a Python function..."
        # Assistant with tool_calls + reasoning
        assert result[2]["role"] == "assistant"
        assert result[2]["reasoning_content"] == "The function will be simple..."
        assert result[2]["tool_calls"] == messages[2]["tool_calls"]
        # Tool
        assert result[3]["role"] == "tool"
        assert result[3]["tool_call_id"] == "call_1"


# ═══════════════════════════════════════════════════════════════════════
# has_image_content
# ═══════════════════════════════════════════════════════════════════════

class TestHasImageContent:
    def test_no_images(self):
        assert not has_image_content([
            {"role": "user", "content": "Hello"},
        ])

    def test_simple_text_content(self):
        assert not has_image_content([
            {"role": "user", "content": "What is in this picture?"},
        ])

    def test_has_image_url(self):
        assert has_image_content([
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ]},
        ])

    def test_image_only(self):
        assert has_image_content([
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ]},
        ])

    def test_empty_messages(self):
        assert not has_image_content([])

    def test_multiple_messages_with_image(self):
        assert has_image_content([
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": [
                {"type": "text", "text": "And this?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpg;base64,xyz"}},
            ]},
        ])


# ═══════════════════════════════════════════════════════════════════════
# HTTP endpoint tests (in-process server)
# ═══════════════════════════════════════════════════════════════════════
# Skip these — server requires full plumbing and Docker. Test utility
# functions instead. The dockerized dev instance is tested separately.
