"""Tests for src/server/sse_helpers.py — SSE parsing and cost-from-tokens."""
import json
from unittest.mock import MagicMock, patch

import pytest

from src.server.sse_helpers import extract_last_sse_chunk, estimate_cost_from_tokens


# ═══════════════════════════════════════════════════════════════════════
# extract_last_sse_chunk
# ═══════════════════════════════════════════════════════════════════════

class TestExtractLastSseChunk:
    """Tests for extract_last_sse_chunk()."""

    # ── happy path ──────────────────────────────────────────────────────

    def test_single_data_chunk(self):
        """Extracts the last data: chunk from an SSE stream."""
        raw = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n'
        result = extract_last_sse_chunk(raw)
        assert result is not None
        assert result["choices"][0]["delta"]["content"] == "hello"

    def test_multiple_data_chunks_returns_last(self):
        """When multiple data: lines exist, returns the last non-DONE one."""
        raw = (
            b'data: {"chunk":1}\n\n'
            b'data: {"chunk":2}\n\n'
            b'data: {"chunk":3}\n\n'
            b'data: [DONE]\n'
        )
        result = extract_last_sse_chunk(raw)
        assert result == {"chunk": 3}

    def test_ignores_done(self):
        """data: [DONE] is skipped."""
        raw = b'data: [DONE]\n'
        result = extract_last_sse_chunk(raw)
        assert result is None

    def test_ignores_empty_data(self):
        """data: with only whitespace is skipped."""
        raw = b'data:   \n\ndata: [DONE]\n'
        result = extract_last_sse_chunk(raw)
        assert result is None

    # ── edge cases ──────────────────────────────────────────────────────

    def test_no_data_lines(self):
        """SSE with no data: lines returns None."""
        raw = b'event: ping\n\nevent: close\n'
        result = extract_last_sse_chunk(raw)
        assert result is None

    def test_empty_bytes(self):
        """Empty buffer returns None."""
        assert extract_last_sse_chunk(b'') is None

    def test_invalid_json_returns_none(self):
        """Malformed JSON in data: line returns None gracefully."""
        raw = b'data: {not valid json}\n'
        result = extract_last_sse_chunk(raw)
        assert result is None

    def test_usage_chunk(self):
        """Extracts usage info from a typical SSE usage chunk."""
        raw = b'data: {"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":50}}\n\ndata: [DONE]\n'
        result = extract_last_sse_chunk(raw)
        assert result is not None
        assert result["usage"]["prompt_tokens"] == 100
        assert result["usage"]["completion_tokens"] == 50

    def test_non_utf8_bytes(self):
        """Non-UTF8 bytes are handled with errors=replace."""
        raw = b'data: {"ok":true}\n\ndata: \xff\xfe\n'
        result = extract_last_sse_chunk(raw)
        # The invalid bytes line won't parse as JSON, returns None gracefully
        assert result is None  # last non-DONE data line was garbage

    def test_multiline_data(self):
        """SSE data: lines that arrive in multiple reads — only actual data: prefix counts."""
        # In practice chunks arrive like:
        # b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n'
        raw = b'data: {"a":1}\n\ndata: {"a":2}\n\ndata: [DONE]\n'
        result = extract_last_sse_chunk(raw)
        assert result == {"a": 2}


# ═══════════════════════════════════════════════════════════════════════
# estimate_cost_from_tokens
# ═══════════════════════════════════════════════════════════════════════

class TestEstimateCostFromTokens:
    """Tests for estimate_cost_from_tokens()."""

    @pytest.fixture
    def mock_config(self):
        cfg = MagicMock()
        cfg.get_pricing.return_value = {
            "cache_hit": 0.14,
            "cache_miss": 0.55,
            "output": 1.10,
        }
        return cfg

    @pytest.fixture
    def mock_registry(self):
        """Patch get_registry at its source module since it's imported inside the function."""
        with patch("src.api.cost_plugins.get_registry") as mock:
            yield mock

    # ── config-based pricing fallback ───────────────────────────────────

    def test_config_pricing(self, mock_config, mock_registry):
        """When no plugin matches, uses config-based pricing."""
        cost_info = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "cache_hit_tokens": 200,
            "cache_miss_tokens": 800,
        }
        mock_registry.return_value.calculate_cost.return_value = None
        result = estimate_cost_from_tokens(
            "unknown", "unknown-model", cost_info, mock_config
        )
        # (200/1M)*0.14 + (800/1M)*0.55 + (500/1M)*1.10
        expected = round((200 / 1_000_000) * 0.14 + (800 / 1_000_000) * 0.55 + (500 / 1_000_000) * 1.10, 8)
        assert result == expected
        assert result > 0

    def test_config_pricing_zero_tokens(self, mock_config, mock_registry):
        """Zero tokens produces zero cost."""
        cost_info = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
        }
        mock_registry.return_value.calculate_cost.return_value = None
        result = estimate_cost_from_tokens(
            "p", "m", cost_info, mock_config
        )
        assert result == 0.0

    # ── plugin delegation ───────────────────────────────────────────────

    def test_plugin_registry_takes_priority(self, mock_config, mock_registry):
        """Plugin cost is used when the registry returns a value."""
        cost_info = {
            "prompt_tokens": 500,
            "completion_tokens": 200,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 500,
        }
        mock_registry.return_value.calculate_cost.return_value = 0.0042
        result = estimate_cost_from_tokens(
            "opencode", "deepseek-v4-pro", cost_info, mock_config
        )
        assert result == 0.0042

    # ── token merging ───────────────────────────────────────────────────

    def test_merges_prompt_and_cache_miss_for_plugin(self, mock_config, mock_registry):
        """prompt_tokens + cache_miss_tokens are summed for plugin usage dict."""
        cost_info = {
            "prompt_tokens": 300,
            "completion_tokens": 100,
            "cache_hit_tokens": 50,
            "cache_miss_tokens": 700,
        }
        mock_registry.return_value.calculate_cost.return_value = 0.001
        estimate_cost_from_tokens(
            "p", "m", cost_info, mock_config
        )
        call_kwargs = mock_registry.return_value.calculate_cost.call_args
        provider, model, usage = call_kwargs[0]
        assert usage["prompt_tokens"] == 1000  # 300 + 700
        assert usage["completion_tokens"] == 100
        assert usage["prompt_cache_hit_tokens"] == 50
        assert usage["prompt_cache_miss_tokens"] == 700
