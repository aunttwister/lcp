"""Tests for the Command Code cost tracking plugin.

Uses a temporary SQLite database with the gateway ``requests`` table
(single source of truth for cost history).  Verifies the plugin reads and
aggregates correctly via SQLAlchemy engine, and that the subscription fetch
wires the credential-store cookie into the billing API.
"""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.api.cost_plugins.commandcode import CommandCodeCostPlugin, _COMMANDCODE_PRICING


# ── Helpers ─────────────────────────────────────────────────────────────────

def _create_engine_and_table():
    """Create a SQLAlchemy engine + ``requests`` table in a temp in-memory DB."""
    from src.api.models import Base, get_engine
    engine = get_engine(":memory:")
    Base.metadata.create_all(engine, tables=[Base.metadata.tables["requests"]])
    return engine


def _insert_request(engine, **kwargs):
    """Insert a row into the ``requests`` table."""
    from src.api.models import Request, get_session
    defaults = {
        "timestamp": "2026-08-08T12:00:00",
        "profile": "default",
        "model": "deepseek-v4-pro",
        "provider": "commandcode",
        "prompt_tokens": 500,
        "completion_tokens": 200,
        "cache_hit_tokens": 0,
        "cache_miss_tokens": 0,
        "cost": 0.0,
        "latency_ms": 100,
        "success": 1,
        "error_type": None,
        "tools_blocked": None,
    }
    defaults.update(kwargs)
    with get_session(engine) as session:
        req = Request(**defaults)
        session.add(req)
        session.commit()


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    """Create an in-memory SQLAlchemy engine with the requests table."""
    return _create_engine_and_table()


@pytest.fixture
def plugin(engine):
    """Return a CommandCode plugin bound to the test engine."""
    return CommandCodeCostPlugin(engine=engine)


# ═══════════════════════════════════════════════════════════════════════
# Identity & pricing
# ═══════════════════════════════════════════════════════════════════════

class TestCommandCodeIdentity:
    def test_provider_name(self, plugin):
        assert plugin.provider_name == "commandcode"

    def test_supported_models(self, plugin):
        models = plugin.get_supported_models()
        assert "deepseek-v4-pro" in models
        assert "deepseek-v4-flash" in models
        assert "claude-sonnet-5" in models
        assert "gpt-5.6-luna" in models
        assert len(models) == len(_COMMANDCODE_PRICING)

    def test_preset(self, plugin):
        preset = plugin.preset
        assert preset["api_base"] == "https://api.commandcode.ai/v1"
        assert "deepseek-v4-pro" in preset["models"]

    def test_get_pricing_known(self, plugin):
        pricing = plugin.get_pricing("deepseek-v4-pro")
        assert pricing == _COMMANDCODE_PRICING["deepseek-v4-pro"]
        assert pricing["output"] > 0

    def test_get_pricing_unknown(self, plugin):
        assert plugin.get_pricing("nonexistent-model") is None

    def test_calculate_cost(self, plugin):
        usage = {"prompt_tokens": 1000, "completion_tokens": 500,
                 "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 1000}
        cost = plugin.calculate_cost("deepseek-v4-flash", usage)
        expected = (1000 / 1_000_000) * 0.14 + (500 / 1_000_000) * 0.28
        assert cost == pytest.approx(expected)

    def test_calculate_cost_zero_tokens(self, plugin):
        cost = plugin.calculate_cost("deepseek-v4-flash", {"prompt_tokens": 0, "completion_tokens": 0})
        assert cost == 0.0

    def test_calculate_cost_unknown_model(self, plugin):
        assert plugin.calculate_cost("nonexistent", {"prompt_tokens": 10}) is None


# ═══════════════════════════════════════════════════════════════════════
# Usage + summary (gateway DB)
# ═══════════════════════════════════════════════════════════════════════

class TestFetchUsage:
    def test_empty_db(self, plugin):
        assert plugin.fetch_usage() == []

    def test_aggregates_commandcode_rows(self, plugin):
        _insert_request(engine=plugin._engine, prompt_tokens=500, completion_tokens=200,
                        cost=0.01, timestamp="2026-08-08T10:00:00")
        _insert_request(engine=plugin._engine, prompt_tokens=300, completion_tokens=100,
                        cost=0.005, timestamp="2026-08-08T12:00:00")
        rows = plugin.fetch_usage()
        assert len(rows) == 1
        r = rows[0]
        assert r["provider"] == "commandcode"
        assert r["prompt_tokens"] == 800
        assert r["completion_tokens"] == 300
        assert r["request_count"] == 2
        assert r["cost"] == pytest.approx(0.015)

    def test_excludes_other_providers(self, plugin):
        _insert_request(engine=plugin._engine, provider="deepseek")
        _insert_request(engine=plugin._engine, provider="commandcode")
        rows = plugin.fetch_usage()
        assert len(rows) == 1
        assert rows[0]["provider"] == "commandcode"

    def test_filters_by_date_range(self, plugin):
        _insert_request(engine=plugin._engine, timestamp="2026-08-01T10:00:00")
        _insert_request(engine=plugin._engine, timestamp="2026-08-08T10:00:00")
        _insert_request(engine=plugin._engine, timestamp="2026-08-15T10:00:00")
        rows = plugin.fetch_usage(start_date="2026-08-01", end_date="2026-08-08")
        assert len(rows) == 2

    def test_excludes_failed_requests(self, plugin):
        _insert_request(engine=plugin._engine, success=0)
        _insert_request(engine=plugin._engine, success=1)
        rows = plugin.fetch_usage()
        assert len(rows) == 1


class TestFetchSummary:
    def test_empty_db(self, plugin):
        summary = plugin.fetch_summary()
        assert summary["daily"] == {"tokens": 0, "cost": 0.0, "requests": 0}
        assert summary["weekly"] == {"tokens": 0, "cost": 0.0, "requests": 0}
        assert summary["monthly"] == {"tokens": 0, "cost": 0.0, "requests": 0}

    def test_periods(self, plugin):
        now = datetime.now(timezone.utc)
        _insert_request(engine=plugin._engine,
                        timestamp=(now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S"),
                        prompt_tokens=1000, completion_tokens=200, cost=0.01)
        _insert_request(engine=plugin._engine,
                        timestamp=(now - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%S"),
                        prompt_tokens=500, completion_tokens=100, cost=0.005)
        _insert_request(engine=plugin._engine,
                        timestamp=(now - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%S"),
                        prompt_tokens=100, completion_tokens=50, cost=0.001)

        # Compute expected counts from the same boundary logic as the plugin,
        # so the test is deterministic regardless of the current date.
        daily_cutoff = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        weekly_cutoff = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        monthly_cutoff = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%S")

        daily_expected = 1 if (now - timedelta(hours=2)).strftime("%Y-%m-%d") >= daily_cutoff else 0
        weekly_expected = sum(
            1 for d in [2 / 24, 5, 3]
            if (now - timedelta(days=d)).strftime("%Y-%m-%d") >= weekly_cutoff
        )
        monthly_expected = sum(
            1 for d in [2 / 24, 5, 3]
            if (now - timedelta(days=d)).strftime("%Y-%m-%dT%H:%M:%S") >= monthly_cutoff
        )

        summary = plugin.fetch_summary()
        assert summary["daily"]["requests"] == daily_expected
        assert summary["weekly"]["requests"] == weekly_expected
        assert summary["monthly"]["requests"] == monthly_expected

    def test_no_engine_returns_none(self):
        p = CommandCodeCostPlugin(engine=None)
        assert p.fetch_summary() is None
        assert p.fetch_usage() == []


# ═══════════════════════════════════════════════════════════════════════
# Balance + subscription
# ═══════════════════════════════════════════════════════════════════════

class TestBalanceAndSubscription:
    def test_fetch_balance_returns_none(self, plugin):
        assert plugin.fetch_balance() is None

    def test_subscription_mock_data(self, plugin):
        with patch.dict(os.environ, {"LCP_MOCK_PLUGIN_DATA": "1"}, clear=False):
            result = plugin.fetch_subscription()
        assert result["monthly_credits_remaining"] == 8.78
        assert result["five_hour_pct"] == 32.0
        assert result["weekly_pct"] == 41.0
        assert result["plan_id"] == "go"

    def test_subscription_no_cookie(self, plugin):
        # No cookie in the store and no mock data
        with patch.dict(os.environ, {}, clear=False):
            with patch("src.api.credential_store.get_credential_store", return_value=MagicMock(get_cookie=lambda _: "")):
                result = plugin.fetch_subscription()
        assert result["_error"] == "auth_failed"
        assert "cookie" in result["detail"].lower()

    def test_subscription_cookie_fetches_api(self, plugin):
        """With a cookie set, the plugin delegates to the billing API."""
        fake_store = MagicMock()
        fake_store.get_cookie.return_value = "session=valid"
        fake_snapshot = {
            "monthly_credits_remaining": 5.0,
            "purchased_credits": 1.0,
            "premium_monthly_credits": 0.0,
            "opensource_monthly_credits": 5.0,
            "five_hour_pct": 50.0,
            "weekly_pct": 60.0,
            "five_hour_reset_sec": 100,
            "weekly_reset_sec": 200,
            "five_hour_reset_at": "",
            "weekly_reset_at": "",
            "plan_id": "pro",
            "plan_status": "active",
            "billing_period_end": None,
        }
        with patch.dict(os.environ, {}, clear=False):
            with patch("src.api.credential_store.get_credential_store", return_value=fake_store):
                with patch("src.api.cost_plugins.commandcode_api.fetch_subscription_snapshot_dict", return_value=fake_snapshot):
                    result = plugin.fetch_subscription()
        assert result["monthly_credits_remaining"] == 5.0
        assert result["plan_id"] == "pro"
        fake_store.get_cookie.assert_called_with("commandcode")

    def test_subscription_api_error(self, plugin):
        fake_store = MagicMock()
        fake_store.get_cookie.return_value = "session=valid"
        with patch.dict(os.environ, {}, clear=False):
            with patch("src.api.credential_store.get_credential_store", return_value=fake_store):
                with patch("src.api.cost_plugins.commandcode_api.fetch_subscription_snapshot_dict", return_value=None):
                    result = plugin.fetch_subscription()
        assert result["_error"] == "auth_failed"


# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

class TestRegistration:
    """These use a FRESH PluginRegistry to avoid depending on the global
    singleton, which other test files (test_cost_plugins_base.py) reset."""

    def _fresh_registry_with_commandcode(self):
        from src.api.cost_plugins.base import PluginRegistry
        reg = PluginRegistry()
        reg.register(CommandCodeCostPlugin())
        return reg

    def test_plugin_registered(self):
        registry = self._fresh_registry_with_commandcode()
        plugin = registry.for_provider("commandcode")
        assert plugin is not None
        assert plugin.provider_name == "commandcode"

    def test_preset_in_registry(self):
        registry = self._fresh_registry_with_commandcode()
        presets = registry.presets
        assert "commandcode" in presets
        assert presets["commandcode"]["api_base"] == "https://api.commandcode.ai/v1"
