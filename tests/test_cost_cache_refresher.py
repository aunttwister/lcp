"""Tests for the CacheRefresher background scraper (TTL, retry, throttle)."""

from datetime import datetime, timedelta, timezone

import pytest

from src.api.models import Base, CostPluginCacheEntry, get_engine, get_session
from src.api.cost_cache import (
    CacheRefresher,
    CostPluginCache,
    SettingsStore,
    plugin_supports,
)
from src.api.cost_plugins.base import CostPlugin


class FakeSubPlugin(CostPlugin):
    """Supports subscriptions only; configurable result/failure."""

    provider_name = "opencode"

    def __init__(self, sub=None, fail=False, return_error=None, calls=None):
        super().__init__()
        self.sub = sub
        self.fail = fail
        self.return_error = return_error
        self.calls = calls if calls is not None else {"sub": 0}

    def fetch_subscription(self):
        self.calls["sub"] += 1
        if self.fail:
            raise RuntimeError("network down")
        if self.return_error:
            return {"_error": self.return_error, "detail": "something happened"}
        return self.sub


class FakeNoSubPlugin(CostPlugin):
    """Supports balance only (fetch_subscription is inherited = unsupported)."""

    provider_name = "deepseek"

    def __init__(self, bal=None, calls=None):
        super().__init__()
        self.bal = bal
        self.calls = calls if calls is not None else {"bal": 0}

    def fetch_balance(self):
        self.calls["bal"] += 1
        return self.bal


class FakeRegistry:
    def __init__(self, plugins):
        self._plugins = plugins

    @property
    def providers(self):
        return list(self._plugins.keys())

    def for_provider(self, name):
        return self._plugins.get(name)


@pytest.fixture
def engine(tmp_path):
    db_path = str(tmp_path / "test.db")
    e = get_engine(db_path)
    Base.metadata.create_all(e)
    return e


def _make_refresher(engine, registry, ttl=30, **kw):
    cache = CostPluginCache(engine)
    settings = SettingsStore(engine)
    settings.set_ttl_minutes(ttl)
    defaults = dict(tick_seconds=1000, throttle_seconds=0, backoff_base=60, backoff_cap=1800)
    defaults.update(kw)
    return CacheRefresher(cache, settings, registry_getter=lambda: registry, **defaults)


def _age_row(engine, provider, kind, minutes):
    """Backdate a cache row's fetched_at so it is stale."""
    old = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
    with get_session(engine) as s:
        row = s.query(CostPluginCacheEntry).filter_by(provider=provider, kind=kind).first()
        row.fetched_at = old
        s.commit()


class TestPluginSupports:
    def test_subclass_override_detected(self):
        assert plugin_supports(FakeSubPlugin(), "subscription") is True
        assert plugin_supports(FakeSubPlugin(), "balance") is False
        assert plugin_supports(FakeNoSubPlugin(), "subscription") is False
        assert plugin_supports(FakeNoSubPlugin(), "balance") is True
        assert plugin_supports(None, "subscription") is False


class TestRefreshBasics:
    def test_missing_entry_is_scraped(self, engine):
        reg = FakeRegistry({"opencode": FakeSubPlugin(sub={"monthly_pct": 42.0})})
        r = _make_refresher(engine, reg)
        r._pass()
        ent = r._cache.get("opencode", "subscription")
        assert ent["payload"] == {"monthly_pct": 42.0}
        assert reg._plugins["opencode"].calls["sub"] == 1

    def test_fresh_entry_not_rescraped(self, engine):
        plugin = FakeSubPlugin(sub={"monthly_pct": 42.0})
        reg = FakeRegistry({"opencode": plugin})
        r = _make_refresher(engine, reg)
        r._cache.set("opencode", "subscription", {"monthly_pct": 1.0})
        r._pass()
        assert plugin.calls["sub"] == 0
        assert r._cache.get("opencode", "subscription")["payload"] == {"monthly_pct": 1.0}

    def test_stale_entry_is_rescraped(self, engine):
        plugin = FakeSubPlugin(sub={"monthly_pct": 99.0})
        reg = FakeRegistry({"opencode": plugin})
        r = _make_refresher(engine, reg, ttl=30)
        r._cache.set("opencode", "subscription", {"monthly_pct": 1.0})
        _age_row(engine, "opencode", "subscription", 60)
        r._pass()
        assert plugin.calls["sub"] == 1
        assert r._cache.get("opencode", "subscription")["payload"] == {"monthly_pct": 99.0}

    def test_unsupported_kind_not_scraped(self, engine):
        plugin = FakeNoSubPlugin(bal={"balance": 10.0})
        reg = FakeRegistry({"deepseek": plugin})
        r = _make_refresher(engine, reg)
        r._pass()
        # balance scraped, subscription never attempted
        assert plugin.calls["bal"] == 1
        assert r._cache.get("deepseek", "balance")["payload"] == {"balance": 10.0}
        assert r._cache.get("deepseek", "subscription") is None

    def test_none_result_quieted_not_cached(self, engine):
        plugin = FakeSubPlugin(sub=None)  # returns None (no data)
        reg = FakeRegistry({"opencode": plugin})
        r = _make_refresher(engine, reg)
        r._pass()
        assert r._cache.get("opencode", "subscription") is None
        assert r.diagnostics() == {}  # no failure recorded
        # Quieted → not re-attempted on subsequent passes.
        r._pass()
        assert plugin.calls["sub"] == 1
        # A credential change re-checks it.
        r.request_refresh(provider="opencode")
        r._pass()
        assert plugin.calls["sub"] == 2


class TestRetryAndBackoff:
    def test_transient_failure_keeps_stale_and_backs_off(self, engine):
        plugin = FakeSubPlugin(sub={"monthly_pct": 50.0}, fail=True)
        reg = FakeRegistry({"opencode": plugin})
        r = _make_refresher(engine, reg)
        r._cache.set("opencode", "subscription", {"monthly_pct": 50.0})
        _age_row(engine, "opencode", "subscription", 60)  # make it stale so we attempt

        r._pass()
        ent = r._cache.get("opencode", "subscription")
        assert ent["payload"] == {"monthly_pct": 50.0}  # old payload kept
        assert ent["stale_error"] == "network down"
        diag = r.diagnostics()
        assert diag["opencode/subscription"]["consecutive_failures"] == 1
        assert diag["opencode/subscription"]["next_attempt_at"] > 0

        # A second pass within the backoff window must not re-scrape.
        r._pass()
        assert plugin.calls["sub"] == 1

    def test_auth_failure_no_early_retry(self, engine):
        plugin = FakeSubPlugin(return_error="auth_failed")
        reg = FakeRegistry({"opencode": plugin})
        r = _make_refresher(engine, reg, ttl=30)
        r._pass()
        ent = r._cache.get("opencode", "subscription")
        assert ent["payload"] == {"_error": "auth_failed", "detail": "something happened"}
        assert ent["stale_error"] == "something happened"
        # auth failure → next attempt scheduled at TTL (30 min), not backoff
        diag = r.diagnostics()
        next_at = diag["opencode/subscription"]["next_attempt_at"]
        import time as _time
        assert next_at > _time.time() + (29 * 60)

    def test_success_resets_failures(self, engine):
        plugin = FakeSubPlugin(sub={"monthly_pct": 50.0}, fail=True)
        reg = FakeRegistry({"opencode": plugin})
        r = _make_refresher(engine, reg)
        r._cache.set("opencode", "subscription", {"monthly_pct": 50.0})
        _age_row(engine, "opencode", "subscription", 60)  # stale so we attempt
        r._pass()  # fail
        assert r.diagnostics()["opencode/subscription"]["consecutive_failures"] == 1

        plugin.fail = False
        r.request_refresh(provider="opencode")
        r._pass()  # forced refresh → success
        assert r.diagnostics() == {}  # retry state cleared
        assert r._cache.get("opencode", "subscription")["payload"] == {"monthly_pct": 50.0}
        assert r._cache.get("opencode", "subscription")["stale_error"] is None

    def test_request_refresh_clears_backoff(self, engine):
        plugin = FakeSubPlugin(sub={"monthly_pct": 50.0}, fail=True)
        reg = FakeRegistry({"opencode": plugin})
        r = _make_refresher(engine, reg)
        r._cache.set("opencode", "subscription", {"monthly_pct": 50.0})
        _age_row(engine, "opencode", "subscription", 60)  # stale so we attempt
        r._pass()  # fail → backoff scheduled
        # Even though backoff is active, request_refresh forces an immediate retry.
        r.request_refresh(provider="opencode")
        assert "opencode/subscription" not in r.diagnostics()  # backoff cleared
        plugin.fail = False
        r._pass()
        assert plugin.calls["sub"] == 2


class TestThrottle:
    def test_throttle_prevents_burst(self, engine):
        plugin = FakeSubPlugin(sub={"monthly_pct": 42.0})
        reg = FakeRegistry({"opencode": plugin})
        r = _make_refresher(engine, reg, throttle_seconds=1000)
        r._pass()
        r._pass()
        # Second pass is throttled → still only one scrape.
        assert plugin.calls["sub"] == 1


class TestIsolation:
    def test_one_failing_provider_does_not_block_others(self, engine):
        good = FakeNoSubPlugin(bal={"balance": 5.0})
        bad = FakeSubPlugin(sub={"monthly_pct": 1.0}, fail=True)
        reg = FakeRegistry({"deepseek": good, "opencode": bad})
        r = _make_refresher(engine, reg)
        r._pass()
        # good provider succeeded, bad provider recorded a failure
        assert r._cache.get("deepseek", "balance")["payload"] == {"balance": 5.0}
        assert r.diagnostics()["opencode/subscription"]["consecutive_failures"] == 1
