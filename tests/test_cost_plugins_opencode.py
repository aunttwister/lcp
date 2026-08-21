"""Tests for the OpenCode cost tracking plugin.

Uses a temporary SQLite database with the gateway ``requests`` table
(single source of truth).  Verifies the plugin reads and aggregates
correctly via SQLAlchemy engine.
"""

from unittest.mock import MagicMock, patch

import pytest
from src.api.cost_plugins.opencode import OpenCodeCostPlugin, _OPENCODE_PRICING, _FREE_MODELS


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
        "timestamp": "2025-10-10T12:00:00",
        "profile": "default",
        "model": "deepseek-v4-pro",
        "provider": "opencode",
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


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def engine():
    """Create an in-memory SQLAlchemy engine with the requests table."""
    return _create_engine_and_table()


@pytest.fixture
def plugin(engine):
    """Return an OpenCode plugin bound to the test engine."""
    return OpenCodeCostPlugin(engine=engine)


# ═══════════════════════════════════════════════════════════════════════
# Plugin identity & pricing
# ═══════════════════════════════════════════════════════════════════════

class TestOpenCodeIdentity:
    def test_provider_name(self, plugin):
        assert plugin.provider_name == "opencode"

    def test_supported_models_includes_all(self, plugin):
        models = plugin.get_supported_models()
        assert "deepseek-v4-pro" in models
        assert "deepseek-v4-flash" in models
        for free in _FREE_MODELS:
            assert free in models

    def test_get_pricing_pro(self, plugin):
        p = plugin.get_pricing("deepseek-v4-pro")
        assert p == _OPENCODE_PRICING["deepseek-v4-pro"]

    def test_get_pricing_flash(self, plugin):
        p = plugin.get_pricing("deepseek-v4-flash")
        assert p == _OPENCODE_PRICING["deepseek-v4-flash"]

    def test_get_pricing_free_models(self, plugin):
        for free in _FREE_MODELS:
            p = plugin.get_pricing(free)
            assert p == {"cache_hit": 0.0, "cache_miss": 0.0, "output": 0.0}

    def test_get_pricing_unknown(self, plugin):
        assert plugin.get_pricing("nonexistent") is None


# ═══════════════════════════════════════════════════════════════════════
# calculate_cost
# ═══════════════════════════════════════════════════════════════════════

class TestOpenCodeCalculateCost:
    def test_v4_pro_cost(self, plugin):
        cost = plugin.calculate_cost("deepseek-v4-pro", {
            "prompt_cache_hit_tokens": 500_000,
            "prompt_cache_miss_tokens": 1_000_000,
            "completion_tokens": 200_000,
        })
        expected = (
            (500_000 / 1_000_000) * 0.003625
            + (1_000_000 / 1_000_000) * 0.435
            + (200_000 / 1_000_000) * 0.87
        )
        assert cost == pytest.approx(expected)

    def test_free_model_zero_cost(self, plugin):
        cost = plugin.calculate_cost("qwen3-coder", {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 500_000,
        })
        assert cost == 0.0

    def test_unknown_model_returns_none(self, plugin):
        cost = plugin.calculate_cost("unknown", {"prompt_tokens": 100})
        assert cost is None

    def test_cache_miss_fallback(self, plugin):
        """When cache_hit and cache_miss are both 0/absent, fall back to prompt_tokens."""
        cost = plugin.calculate_cost("deepseek-v4-pro", {
            "completion_tokens": 200_000,
        })
        assert cost == pytest.approx(0.174)


# ═══════════════════════════════════════════════════════════════════════
# fetch_usage (reading from gateway requests table)
# ═══════════════════════════════════════════════════════════════════════

class TestOpenCodeFetchUsage:
    def test_empty_db_returns_empty(self, plugin):
        assert plugin.fetch_usage() == []

    def test_single_request(self, plugin, engine):
        _insert_request(engine,
            timestamp="2025-10-10T12:00:00",
            model="deepseek-v4-pro",
            prompt_tokens=1000, completion_tokens=500,
        )
        result = plugin.fetch_usage()
        assert len(result) == 1
        row = result[0]
        assert row["date"] == "2025-10-10"
        assert row["model"] == "deepseek-v4-pro"
        assert row["provider"] == "opencode"
        assert row["prompt_tokens"] == 1000
        assert row["completion_tokens"] == 500
        assert row["request_count"] == 1

    def test_multiple_requests_same_day(self, plugin, engine):
        _insert_request(engine,
            timestamp="2025-10-10T12:00:00",
            prompt_tokens=1000, completion_tokens=200,
        )
        _insert_request(engine,
            timestamp="2025-10-10T12:01:00",
            prompt_tokens=500, completion_tokens=100,
        )
        result = plugin.fetch_usage()
        assert len(result) == 1
        row = result[0]
        assert row["prompt_tokens"] == 1500  # 1000 + 500
        assert row["completion_tokens"] == 300  # 200 + 100
        assert row["request_count"] == 2

    def test_only_opencode_provider(self, plugin, engine):
        """Non-opencode requests should be excluded."""
        _insert_request(engine,
            timestamp="2025-10-10T12:00:00",
            provider="openai", prompt_tokens=999,
        )
        result = plugin.fetch_usage()
        assert result == []

    def test_only_successful_requests(self, plugin, engine):
        """Failed requests should be excluded."""
        _insert_request(engine,
            timestamp="2025-10-10T12:00:00",
            success=0, error_type="timeout",
            prompt_tokens=100, completion_tokens=50,
        )
        result = plugin.fetch_usage()
        assert result == []

    def test_date_filtering(self, plugin, engine):
        _insert_request(engine, timestamp="2025-10-09T12:00:00",
                        prompt_tokens=100, completion_tokens=10)
        _insert_request(engine, timestamp="2025-10-10T12:00:00",
                        prompt_tokens=200, completion_tokens=20)
        result = plugin.fetch_usage(start_date="2025-10-10")
        assert len(result) == 1
        assert result[0]["date"] == "2025-10-10"

        result2 = plugin.fetch_usage(end_date="2025-10-09")
        assert len(result2) == 1
        assert result2[0]["date"] == "2025-10-09"

    def test_no_engine_returns_empty(self):
        p = OpenCodeCostPlugin(engine=None)
        assert p.fetch_usage() == []

    def test_db_error_returns_empty(self, engine):
        """Session error should return empty list gracefully."""
        with patch("src.api.models.get_session",
                   side_effect=RuntimeError("boom")):
            p = OpenCodeCostPlugin(engine=engine)
            result = p.fetch_usage()
            assert result == []


# ═══════════════════════════════════════════════════════════════════════
# fetch_balance
# ═══════════════════════════════════════════════════════════════════════

class TestOpenCodeFetchBalance:
    def test_no_cookie_returns_none(self, plugin):
        """No configured cookie → plugin stays quiet (returns None)."""
        store = MagicMock()
        store.get_cookie.return_value = ""
        with patch("src.api.credential_store.get_credential_store", return_value=store):
            assert plugin.fetch_balance() is None

    def test_returns_error_when_api_returns_none(self, plugin):
        """Cookie present but no credit data → error dict (not quiet)."""
        store = MagicMock()
        store.get_cookie.return_value = "auth=test-cookie"
        store.get_workspace_id.return_value = "wrk_1"
        with patch("src.api.credential_store.get_credential_store", return_value=store):
            with patch("src.api.cost_plugins.opencode_api.fetch_billing_dict",
                       return_value=None):
                result = plugin.fetch_balance()
        assert result is not None
        assert result["_error"] == "api_error"

    def test_returns_balance_data(self, plugin):
        """Happy path: returns available credits from the billing page."""
        mock_data = {"available_credits": 12.34, "balance": 12.34,
                     "currency": "USD", "plan": "pro"}
        store = MagicMock()
        store.get_cookie.return_value = "auth=test-cookie"
        store.get_workspace_id.return_value = "wrk_1"
        with patch("src.api.credential_store.get_credential_store", return_value=store):
            with patch("src.api.cost_plugins.opencode_api.fetch_billing_dict",
                       return_value=mock_data):
                result = plugin.fetch_balance()
        assert result == mock_data

    def test_uses_credential_store_cookie_and_workspace(self, plugin, tmp_path):
        """The UI-managed cookie + workspace ID are passed to the billing fetch."""
        from src.api.credential_store import CredentialStore
        import src.api.credential_store as cs_module
        from src.api.models import get_engine, Base
        import os as _os

        engine = get_engine(":memory:")
        Base.metadata.create_all(engine)
        cs_module._credential_store = CredentialStore(engine, data_dir=str(tmp_path))
        with patch.dict(_os.environ, {"LCP_SECRET_KEY": "test-master"}, clear=False):
            cs_module._credential_store.set_cookie("opencode", "auth=store-cookie")
            cs_module._credential_store.set_workspace_id("opencode", "wrk_store")

            mock_data = {"available_credits": 5.0, "balance": 5.0}
            with patch("src.api.cost_plugins.opencode_api.fetch_billing_dict",
                       return_value=mock_data) as m:
                result = plugin.fetch_balance()
        assert result == mock_data
        assert m.call_args[0][0] == "auth=store-cookie"
        assert m.call_args[1].get("workspace_id") == "wrk_store"


# ═══════════════════════════════════════════════════════════════════════
# fetch_summary
# ═══════════════════════════════════════════════════════════════════════

class TestOpenCodeFetchSummary:
    def test_summary_empty_db(self, plugin):
        """Empty DB returns zeros for all periods."""
        result = plugin.fetch_summary()
        assert result is not None
        for period in ("daily", "weekly", "monthly"):
            assert result[period]["tokens"] == 0
            assert result[period]["cost"] == 0.0
            assert result[period]["requests"] == 0

    def test_summary_with_data(self, engine):
        """Recent requests should appear in daily/weekly/monthly aggregates."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        recent = now.strftime("%Y-%m-%dT%H:%M:%S")

        _insert_request(engine, timestamp=recent,
                        prompt_tokens=1000, completion_tokens=500,
                        cost=0.00087)
        _insert_request(engine, timestamp=recent,
                        prompt_tokens=2000, completion_tokens=1000,
                        cost=0.005)
        plugin = OpenCodeCostPlugin(engine=engine)
        result = plugin.fetch_summary()
        assert result is not None

        assert result["daily"]["tokens"] == 4500  # 1000+500+2000+1000
        assert result["daily"]["cost"] == pytest.approx(0.00587, rel=1e-5)
        assert result["daily"]["requests"] == 2

        # Weekly/monthly should be same (both within current window)
        assert result["weekly"]["tokens"] >= 4500
        assert result["weekly"]["requests"] >= 2
        assert result["monthly"]["tokens"] >= 4500

    def test_summary_none_when_no_engine(self):
        """Should return None when engine is None."""
        plugin = OpenCodeCostPlugin(engine=None)
        assert plugin.fetch_summary() is None

    def test_summary_none_on_db_error(self, engine):
        """DB error should return None gracefully."""
        with patch("src.api.models.get_session",
                   side_effect=RuntimeError("boom")):
            plugin = OpenCodeCostPlugin(engine=engine)
            result = plugin.fetch_summary()
            assert result is None


# ═══════════════════════════════════════════════════════════════════════
# fetch_subscription
# ═══════════════════════════════════════════════════════════════════════

class TestOpenCodeFetchSubscription:
    def test_returns_error_when_no_cookie(self, plugin):
        """When no cookie is stored, returns error dict."""
        store = MagicMock()
        store.get_cookie.return_value = ""
        with patch("src.api.credential_store.get_credential_store", return_value=store):
            result = plugin.fetch_subscription()
        assert result is not None
        assert result["_error"] == "auth_failed"

    def test_returns_error_when_api_returns_none(self, plugin):
        """When the API call returns None (invalid cookie, etc.), returns error dict."""
        store = MagicMock()
        store.get_cookie.return_value = "auth=test-cookie"
        with patch("src.api.credential_store.get_credential_store", return_value=store):
            with patch("src.api.cost_plugins.opencode_api.fetch_subscription_dict",
                       return_value=None):
                result = plugin.fetch_subscription()
        assert result is not None
        assert result["_error"] == "auth_failed"

    def test_returns_error_when_api_raises(self, plugin):
        """When the API call raises, returns error dict."""
        store = MagicMock()
        store.get_cookie.return_value = "auth=test-cookie"
        with patch("src.api.credential_store.get_credential_store", return_value=store):
            with patch("src.api.cost_plugins.opencode_api.fetch_subscription_dict",
                       side_effect=RuntimeError("network down")):
                result = plugin.fetch_subscription()
        assert result is not None
        assert result["_error"] == "api_error"

    def test_returns_subscription_data(self, plugin):
        """Happy path: returns subscription snapshot."""
        mock_data = {
            "rolling_pct": 17.0, "weekly_pct": 75.0,
            "rolling_reset_sec": 5944, "weekly_reset_sec": 278201,
        }
        store = MagicMock()
        store.get_cookie.return_value = "auth=test-cookie"
        with patch("src.api.credential_store.get_credential_store", return_value=store):
            with patch("src.api.cost_plugins.opencode_api.fetch_subscription_dict",
                       return_value=mock_data):
                result = plugin.fetch_subscription()
        assert result == mock_data

    def test_uses_credential_store_cookie(self, plugin, tmp_path):
        """A UI-managed cookie (credential store) is used for the subscription call."""
        from src.api.credential_store import CredentialStore
        import src.api.credential_store as cs_module
        from src.api.models import get_engine, Base
        import os as _os

        engine = get_engine(":memory:")
        Base.metadata.create_all(engine)
        cs_module._credential_store = CredentialStore(engine, data_dir=str(tmp_path))
        with patch.dict(_os.environ, {"LCP_SECRET_KEY": "test-master"}, clear=False):
            cs_module._credential_store.set_cookie("opencode", "auth=store-cookie")

            mock_data = {"rolling_pct": 10.0}
            with patch("src.api.cost_plugins.opencode_api.fetch_subscription_dict",
                       return_value=mock_data) as m:
                result = plugin.fetch_subscription()
        assert result == mock_data
        # Verify the store cookie was passed through
        assert m.call_args[0][0] == "auth=store-cookie"
