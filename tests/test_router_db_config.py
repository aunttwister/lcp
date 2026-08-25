"""Tests for dynamic routing reading the DB-backed config.

The gateway config is now DB-backed (``gateway_config:<section>`` JSON blobs in
the settings table). These tests verify the router consumes the DB-backed
``dynamic_routing`` section through the real ``Config`` object, and that the
routing policy endpoint's ``_sync_dynamic_routing_enabled`` helper keeps the
section in sync with the global toggle.
"""

import pytest

from src.api.config import Config
from src.api.cost_cache import SettingsStore
from src.api.models import Base, get_engine
from src.api.router import CapabilityRouter, init_router, routing_status


@pytest.fixture
def registry_db():
    """A fresh DB seeded with the default model registry."""
    import os
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = get_engine(path)
    Base.metadata.create_all(engine)
    engine.dispose()
    from src.api.seed_capabilities import seed_model_registry
    seed_model_registry(path)
    yield path
    for ext in ("", "-wal", "-shm"):
        try:
            os.unlink(path + ext)
        except FileNotFoundError:
            pass


@pytest.fixture
def db_config(tmp_path):
    """A real DB-backed Config with a seeded dynamic_routing section.

    Mirrors the real seed: enabled + cost_bias only (no policy/min_score), so
    the Routing-tab settings keys (routing_policy/routing_min_score) win.
    """
    db_path = str(tmp_path / "test.db")
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    store = SettingsStore(engine)
    store.set_config_section("dynamic_routing", {
        "enabled": True, "cost_bias": 0.15,
    })
    return Config(store=store), store


@pytest.fixture
def db_config_with_policy(tmp_path):
    """A DB-backed Config whose dynamic_routing section carries policy/min_score."""
    db_path = str(tmp_path / "test.db")
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    store = SettingsStore(engine)
    store.set_config_section("dynamic_routing", {
        "enabled": True, "policy": "cost_first", "min_score": 0.4,
    })
    return Config(store=store), store


class TestEffectivePolicyFromDbConfig:
    def test_effective_policy_reads_db_section(self, registry_db, db_config_with_policy, monkeypatch):
        cfg, store = db_config_with_policy
        monkeypatch.setattr("src.api.cost_cache.get_settings", lambda: store)
        router = CapabilityRouter(enabled=True, db_path=registry_db)
        policy, min_score = router._effective_policy(cfg)
        assert policy == "cost_first"
        assert min_score == 0.4

    def test_effective_policy_settings_override_wins(self, registry_db, db_config, monkeypatch):
        cfg, store = db_config
        store.set_routing_policy("explore")
        store.set_routing_min_score(0.7)
        monkeypatch.setattr("src.api.cost_cache.get_settings", lambda: store)
        router = CapabilityRouter(enabled=True, db_path=registry_db)
        policy, min_score = router._effective_policy(cfg)
        # The seeded section has no policy/min_score, so the settings win.
        assert policy == "explore"
        assert min_score == 0.7


class TestRoutingStatusFromDbConfig:
    def test_routing_status_reflects_db_policy(self, registry_db, db_config_with_policy, monkeypatch):
        cfg, store = db_config_with_policy
        monkeypatch.setattr("src.api.cost_cache.get_settings", lambda: store)
        init_router(registry_db, enabled=True)
        try:
            st = routing_status(cfg)
            assert st["policy"] == "cost_first"
            assert st["min_score"] == 0.4
        finally:
            init_router(enabled=False)

    def test_routing_status_per_profile_from_db_config(self, registry_db, db_config_with_policy, monkeypatch):
        cfg, store = db_config_with_policy
        monkeypatch.setattr("src.api.cost_cache.get_settings", lambda: store)
        init_router(registry_db, enabled=True)
        try:
            st = routing_status(cfg)
            assert "per_profile" in st
            for block in st["per_profile"].values():
                assert block["policy"] == "cost_first"
        finally:
            init_router(enabled=False)


class TestIsEnabledPrecedence:
    def test_settings_override_wins_over_boot(self, registry_db, db_config, monkeypatch):
        cfg, store = db_config
        store.set_routing_enabled(False)  # runtime toggle off
        monkeypatch.setattr("src.api.cost_cache.get_settings", lambda: store)
        router = CapabilityRouter(enabled=True, db_path=registry_db)  # boot on
        assert router.is_enabled(cfg) is False

    def test_boot_value_used_when_no_override(self, registry_db, db_config, monkeypatch):
        cfg, store = db_config
        monkeypatch.setattr("src.api.cost_cache.get_settings", lambda: store)
        router = CapabilityRouter(enabled=True, db_path=registry_db)
        assert router.is_enabled(cfg) is True


class TestSyncDynamicRoutingEnabled:
    def test_sync_updates_db_section(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)
        store = SettingsStore(engine)
        store.set_config_section("dynamic_routing", {"enabled": False, "cost_bias": 0.15})
        from src.server.endpoints import _sync_dynamic_routing_enabled
        _sync_dynamic_routing_enabled(store, True)
        section = store.get_config_section("dynamic_routing")
        assert section["enabled"] is True
        assert section["cost_bias"] == 0.15  # other keys preserved

    def test_sync_creates_section_when_absent(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)
        store = SettingsStore(engine)
        from src.server.endpoints import _sync_dynamic_routing_enabled
        _sync_dynamic_routing_enabled(store, False)
        assert store.get_config_section("dynamic_routing")["enabled"] is False