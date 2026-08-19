"""Tests for src.api.cost_cache: SettingsStore + CostPluginCache."""

import pytest

from src.api.models import Base, get_engine
from src.api.cost_cache import CostPluginCache, SettingsStore


@pytest.fixture
def engine(tmp_path):
    db_path = str(tmp_path / "test.db")
    e = get_engine(db_path)
    Base.metadata.create_all(e)
    return e


class TestSettingsStore:
    def test_default_ttl(self, engine):
        s = SettingsStore(engine)
        assert s.get_ttl_minutes() == 30

    def test_set_and_persist_ttl(self, engine):
        s = SettingsStore(engine)
        s.set_ttl_minutes(5)
        # A fresh store must read the persisted value from the DB.
        s2 = SettingsStore(engine)
        assert s2.get_ttl_minutes() == 5

    def test_ttl_min_floor(self, engine):
        s = SettingsStore(engine)
        s.set_ttl_minutes(0)
        assert s.get_ttl_minutes() == 1

    def test_get_default_when_absent(self, engine):
        s = SettingsStore(engine)
        assert s.get("missing", "fallback") == "fallback"

    def test_invalid_ttl_falls_back(self, engine):
        s = SettingsStore(engine)
        s.set("cost_cache_ttl_minutes", "not-a-number")
        assert s.get_ttl_minutes() == 30

    def test_set_overwrites(self, engine):
        s = SettingsStore(engine)
        s.set("k", "1")
        s.set("k", "2")
        assert s.get("k") == "2"

    def test_per_provider_ttl_override(self, engine):
        s = SettingsStore(engine)
        s.set_ttl_minutes(5, provider="deepseek")
        assert s.get_ttl_minutes(provider="deepseek") == 5
        # Other providers fall back to the default.
        assert s.get_ttl_minutes(provider="opencode") == 30
        assert s.ttl_overrides() == {"deepseek": 5}

    def test_per_provider_ttl_global_default_applies(self, engine):
        s = SettingsStore(engine)
        s.set_ttl_minutes(10)  # global default
        assert s.get_ttl_minutes(provider="opencode") == 10
        assert s.get_ttl_minutes() == 10

    def test_clear_ttl_override(self, engine):
        s = SettingsStore(engine)
        s.set_ttl_minutes(7, provider="opencode")
        s.clear_ttl_minutes("opencode")
        assert s.get_ttl_minutes(provider="opencode") == 30
        assert s.ttl_overrides() == {}

    def test_per_provider_ttl_persisted(self, engine):
        s = SettingsStore(engine)
        s.set_ttl_minutes(3, provider="commandcode")
        s2 = SettingsStore(engine)
        assert s2.get_ttl_minutes(provider="commandcode") == 3

    def test_routing_policy_default(self, engine):
        s = SettingsStore(engine)
        assert s.get_routing_policy() == "eager"

    def test_routing_policy_set_and_persist(self, engine):
        s = SettingsStore(engine)
        s.set_routing_policy("explore")
        s2 = SettingsStore(engine)
        assert s2.get_routing_policy() == "explore"

    def test_routing_policy_invalid(self, engine):
        s = SettingsStore(engine)
        with pytest.raises(ValueError):
            s.set_routing_policy("bogus")

    def test_routing_min_score(self, engine):
        s = SettingsStore(engine)
        assert s.get_routing_min_score() == 0.0
        s.set_routing_min_score(0.6)
        s2 = SettingsStore(engine)
        assert s2.get_routing_min_score() == 0.6

    def test_routing_rules_default_empty(self, engine):
        s = SettingsStore(engine)
        assert s.get_routing_rules() == []

    def test_routing_rules_set_and_persist(self, engine):
        s = SettingsStore(engine)
        rules = [{"task": "debugging", "action": "prefer",
                  "provider": "deepseek", "model": "deepseek-v4-pro"}]
        s.set_routing_rules(rules)
        s2 = SettingsStore(engine)
        assert s2.get_routing_rules() == rules

    def test_routing_rules_clears_to_empty(self, engine):
        s = SettingsStore(engine)
        s.set_routing_rules([{"task": "x", "action": "prefer", "model": "m"}])
        s.set_routing_rules([])
        assert s.get_routing_rules() == []

    def test_routing_rules_bad_json_returns_default(self, engine):
        s = SettingsStore(engine)
        s.set("routing_rules", "{not json")
        assert s.get_routing_rules() == []


class TestCostPluginCache:
    def test_get_missing_is_none(self, engine):
        c = CostPluginCache(engine)
        assert c.get("opencode", "subscription") is None

    def test_set_get(self, engine):
        c = CostPluginCache(engine)
        c.set("opencode", "subscription", {"monthly_pct": 42.0})
        ent = c.get("opencode", "subscription")
        assert ent["payload"] == {"monthly_pct": 42.0}
        assert ent["stale_error"] is None
        assert ent["fetched_at"]

    def test_update_in_place_single_row(self, engine):
        c = CostPluginCache(engine)
        c.set("opencode", "subscription", {"a": 1})
        c.set("opencode", "subscription", {"b": 2})
        assert len(c.entries()) == 1
        assert c.get("opencode", "subscription")["payload"] == {"b": 2}

    def test_mark_stale_keeps_payload(self, engine):
        c = CostPluginCache(engine)
        c.set("opencode", "subscription", {"a": 1})
        c.mark_stale("opencode", "subscription", "boom")
        ent = c.get("opencode", "subscription")
        assert ent["payload"] == {"a": 1}
        assert ent["stale_error"] == "boom"

    def test_invalidate_by_provider(self, engine):
        c = CostPluginCache(engine)
        c.set("opencode", "subscription", {})
        c.set("commandcode", "subscription", {})
        c.invalidate(provider="opencode")
        assert c.get("opencode", "subscription") is None
        assert c.get("commandcode", "subscription") is not None

    def test_invalidate_all(self, engine):
        c = CostPluginCache(engine)
        c.set("opencode", "subscription", {})
        c.set("deepseek", "balance", {})
        c.clear()
        assert c.entries() == []

    def test_is_stale_boundary(self, engine):
        c = CostPluginCache(engine)
        c.set("opencode", "subscription", {})
        # Just written → fresh for any positive TTL.
        assert c.is_stale("opencode", "subscription", 1800) is False
        # TTL of zero → immediately stale.
        assert c.is_stale("opencode", "subscription", 0) is True
        # Missing entry → stale.
        assert c.is_stale("missing", "subscription", 1800) is True

    def test_entries_reports_age_and_stale(self, engine):
        c = CostPluginCache(engine)
        c.set("opencode", "subscription", {})
        c.set("deepseek", "balance", {}, stale_error="boom")
        entries = {e["provider"]: e for e in c.entries()}
        assert entries["opencode"]["stale_error"] is None
        assert entries["opencode"]["age_seconds"] >= 0
        assert entries["deepseek"]["stale_error"] == "boom"
