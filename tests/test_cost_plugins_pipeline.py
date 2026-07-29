"""Tests for the cost-plugin integration in request_pipeline.py.

Verifies that calculate_cost and record_cost delegate to the plugin
registry correctly, and fall back to config-based pricing when no
plugin handles a provider.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.api.request_pipeline import calculate_cost, record_cost
from src.api.models import get_engine, Base


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_db():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()
    for ext in ["", "-wal", "-shm"]:
        try:
            os.unlink(db_path + ext)
        except FileNotFoundError:
            pass


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.get_pricing.return_value = {
        "cache_hit": 0.02,
        "cache_miss": 0.55,
        "output": 1.10,
    }
    return cfg


# ═══════════════════════════════════════════════════════════════════════
# calculate_cost plugin delegation
# ═══════════════════════════════════════════════════════════════════════

class TestCalculateCostPluginDelegation:
    """calculate_cost should try the plugin registry first, fall back to config."""

    def test_plugin_returns_cost(self, mock_config):
        """When a plugin provides a cost, use it (skip config pricing)."""
        mock_plugin = MagicMock()
        mock_plugin.calculate_cost.return_value = 0.12345678

        mock_registry = MagicMock()
        mock_registry.calculate_cost.return_value = 0.12345678

        body = {"messages": [{"role": "user", "content": "hi"}]}
        response = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}

        with patch("src.api.request_pipeline.get_registry", return_value=mock_registry):
            result = calculate_cost("deepseek", "deepseek-v4-pro", body, response, mock_config)

        assert result["cost"] == 0.12345678
        # Verify config pricing was NOT consulted
        mock_config.get_pricing.assert_not_called()

    def test_plugin_returns_none_falls_back_to_config(self, mock_config):
        """When no plugin handles a provider, use config pricing."""
        mock_registry = MagicMock()
        mock_registry.calculate_cost.return_value = None

        body = {"messages": [{"role": "user", "content": "hi"}]}
        response = {"usage": {"prompt_tokens": 1000, "completion_tokens": 500}}

        with patch("src.api.request_pipeline.get_registry", return_value=mock_registry):
            result = calculate_cost("deepseek", "deepseek-v4-flash", body, response, mock_config)

        # Config pricing is used: cache_hit=0, cache_miss=1000, output=500
        expected_cost = (0 / 1_000_000) * 0.02 + (1000 / 1_000_000) * 0.55 + (500 / 1_000_000) * 1.10
        assert result["cost"] == pytest.approx(expected_cost)
        mock_config.get_pricing.assert_called_once()

    def test_plugin_receives_usage_dict(self, mock_config):
        """The plugin should receive token breakdown in the usage dict."""
        mock_plugin = MagicMock()
        mock_plugin.calculate_cost.return_value = 0.5

        mock_registry = MagicMock()
        mock_registry.calculate_cost.side_effect = (
            lambda provider, model, usage: mock_plugin.calculate_cost(provider, model, usage)
        )

        body = {"messages": [{"role": "user", "content": "hi"}]}
        response = {"usage": {"prompt_tokens": 100, "completion_tokens": 50,
                              "prompt_cache_hit_tokens": 30, "prompt_cache_miss_tokens": 20}}

        with patch("src.api.request_pipeline.get_registry", return_value=mock_registry):
            calculate_cost("testco", "test-model", body, response, mock_config)

        mock_registry.calculate_cost.assert_called_once()
        _provider, _model, usage = mock_registry.calculate_cost.call_args[0]
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["prompt_cache_hit_tokens"] == 30
        assert usage["prompt_cache_miss_tokens"] == 20


# ═══════════════════════════════════════════════════════════════════════
# record_cost plugin hook
# ═══════════════════════════════════════════════════════════════════════

class TestRecordCostPluginHook:
    """record_cost should call plugin.record_tokens on success."""

    def test_calls_record_tokens_on_success(self, temp_db):
        """Successful requests should trigger plugin.record_tokens."""
        mock_plugin = MagicMock()
        mock_plugin.record_tokens = MagicMock()

        mock_registry = MagicMock()
        mock_registry.for_provider.return_value = mock_plugin

        cost_info = {
            "prompt_tokens": 100, "completion_tokens": 50,
            "cache_hit_tokens": 20, "cache_miss_tokens": 80,
            "cost": 0.001, "latency_ms": 500,
        }

        with patch("src.api.request_pipeline.get_registry", return_value=mock_registry):
            record_cost(temp_db, "l2", "deepseek-v4-flash", "llamacpp",
                        cost_info, True, None, [])

        mock_plugin.record_tokens.assert_called_once_with(
            "deepseek-v4-flash",
            prompt_tokens=180,   # prompt_tokens + cache_miss_tokens = 100 + 80
            completion_tokens=50,
            cache_hit_tokens=20,
        )

    def test_does_not_call_record_tokens_on_failure(self, temp_db):
        """Failed requests should NOT trigger plugin.record_tokens."""
        mock_plugin = MagicMock()
        mock_plugin.record_tokens = MagicMock()

        mock_registry = MagicMock()
        mock_registry.for_provider.return_value = mock_plugin

        cost_info = {"prompt_tokens": 0, "completion_tokens": 0,
                     "cache_hit_tokens": 0, "cache_miss_tokens": 0,
                     "cost": 0, "latency_ms": 0}

        with patch("src.api.request_pipeline.get_registry", return_value=mock_registry):
            record_cost(temp_db, "l2", "deepseek-v4-flash", "llamacpp",
                        cost_info, False, "timeout", [])

        mock_plugin.record_tokens.assert_not_called()

    def test_skips_plugin_without_record_tokens(self, temp_db):
        """Plugins without record_tokens attr should be skipped gracefully."""
        mock_plugin = MagicMock(spec=[])  # no record_tokens

        mock_registry = MagicMock()
        mock_registry.for_provider.return_value = mock_plugin

        cost_info = {"prompt_tokens": 10, "completion_tokens": 5,
                     "cache_hit_tokens": 0, "cache_miss_tokens": 10,
                     "cost": 0.0, "latency_ms": 100}

        with patch("src.api.request_pipeline.get_registry", return_value=mock_registry):
            # Should not raise
            record_cost(temp_db, "l2", "test-model", "test-provider",
                        cost_info, True, None, [])

    def test_handles_plugin_exception_gracefully(self, temp_db):
        """Exceptions in record_tokens should be caught and logged, not raised."""
        mock_plugin = MagicMock()
        mock_plugin.record_tokens.side_effect = RuntimeError("oops")

        mock_registry = MagicMock()
        mock_registry.for_provider.return_value = mock_plugin

        cost_info = {"prompt_tokens": 10, "completion_tokens": 5,
                     "cache_hit_tokens": 0, "cache_miss_tokens": 10,
                     "cost": 0.0, "latency_ms": 100}

        with patch("src.api.request_pipeline.get_registry", return_value=mock_registry):
            # Should not raise
            record_cost(temp_db, "l2", "test-model", "test-provider",
                        cost_info, True, None, [])

    def test_no_plugin_for_provider(self, temp_db):
        """When no plugin exists for a provider, record_cost should work normally."""
        mock_registry = MagicMock()
        mock_registry.for_provider.return_value = None

        cost_info = {"prompt_tokens": 10, "completion_tokens": 5,
                     "cache_hit_tokens": 0, "cache_miss_tokens": 10,
                     "cost": 0.0, "latency_ms": 100}

        with patch("src.api.request_pipeline.get_registry", return_value=mock_registry):
            record_cost(temp_db, "l2", "test-model", "unknown-provider",
                        cost_info, True, None, [])

        # Verify the record was still written to DB
        from sqlalchemy import text
        with temp_db.connect() as conn:
            row = conn.execute(text("SELECT * FROM requests")).fetchone()
            assert row is not None
            assert row[11] == 1
