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
sys.path.insert(0, "/opt/lcp")
from unittest.mock import patch, MagicMock

from src.main import (
    strip_forbidden_tools,
    calculate_cost,
    _health_key,
    record_provider_success,
    record_provider_failure,
    is_provider_available,
    LCPHandler,
)


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
# _health_key
# ═══════════════════════════════════════════════════════════════════════

class TestHealthKey:
    def test_returns_tuple(self):
        key = _health_key("deepseek", "https://api.deepseek.com", "l2")
        assert isinstance(key, tuple)
        assert len(key) == 3

    def test_same_args_same_key(self):
        k1 = _health_key("deepseek", "url1", "l2")
        k2 = _health_key("deepseek", "url1", "l2")
        assert k1 == k2

    def test_different_args_different_key(self):
        k1 = _health_key("deepseek", "url1", "l2")
        k2 = _health_key("openai", "url1", "l2")
        assert k1 != k2


# ═══════════════════════════════════════════════════════════════════════
# HTTP endpoint tests (in-process server)
# ═══════════════════════════════════════════════════════════════════════
# Skip these — server requires full plumbing and Docker. Test utility
# functions instead. The dockerized dev instance is tested separately.
