"""Tests for scripts/migrate_gateway_yaml_to_db.py — legacy YAML → DB config."""

import textwrap

import pytest
import yaml

from src.api.cost_cache import SettingsStore
from src.api.exceptions import ConfigError
from src.api.models import Base, get_engine


def _write_yaml(tmp_path, data) -> str:
    path = tmp_path / "gateway.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return str(path)


def _db_store(tmp_path) -> SettingsStore:
    db_path = str(tmp_path / "test.db")
    e = get_engine(db_path)
    Base.metadata.create_all(e)
    return SettingsStore(e), db_path


def _sample_yaml() -> dict:
    return {
        "server": {"port": 8734, "default_profile": "l2"},
        "profiles": {
            "l2": {
                "forbidden_tools": ["write_file"],
                "chain": [{"provider": "test_prov", "model": "test-model",
                           "base_url": "https://test.api/v1"}],
                "auth_required": False,
            },
        },
        "providers": {
            "test_prov": {
                "api_key_env": "TEST_API_KEY",
                "api_base": "https://test.api/v1",
                "models": ["test-model"],
            },
        },
        "pricing": [
            {"provider": "test_prov", "model": "test-model",
             "cache_hit": 0.01, "cache_miss": 0.5, "output": 1.0},
        ],
        "circuit_breaker": {"failures_degraded": 3, "failures_dead": 6,
                            "degraded_cooldown_seconds": 30,
                            "dead_cooldown_seconds": 120},
        "database": {"path": "/app/data/costs.db", "wal_mode": True},
        "dynamic_routing": {"enabled": True, "cost_bias": 0.4},
    }


from scripts.migrate_gateway_yaml_to_db import migrate_gateway_yaml  # noqa: E402


class TestMigrate:
    def test_writes_present_sections_by_default(self, tmp_path):
        store, db_path = _db_store(tmp_path)
        yaml_path = _write_yaml(tmp_path, _sample_yaml())
        summary = migrate_gateway_yaml(yaml_path, db_path, overwrite=True)
        # Present sections are written.
        assert summary["server"] == "written"
        assert summary["profiles"] == "written"
        assert summary["providers"] == "written"
        assert summary["pricing"] == "written"
        assert summary["circuit_breaker"] == "written"
        assert summary["database"] == "written"
        assert summary["dynamic_routing"] == "written"
        # Absent sections are skipped.
        assert summary["retry"] == "absent"
        assert summary["model_limits"] == "absent"
        assert summary["plugins"] == "absent"
        # DB now holds the YAML values.
        assert store.get_config_section("profiles") == _sample_yaml()["profiles"]
        assert store.get_config_section("pricing") == _sample_yaml()["pricing"]
        assert store.get_config_section("dynamic_routing") == {"enabled": True, "cost_bias": 0.4}

    def test_if_absent_skips_existing(self, tmp_path):
        store, db_path = _db_store(tmp_path)
        store.set_config_section("profiles", {"custom": {"chain": []}})
        yaml_path = _write_yaml(tmp_path, _sample_yaml())
        summary = migrate_gateway_yaml(yaml_path, db_path, overwrite=False)
        assert summary["profiles"] == "skipped_exists"
        # DB value preserved.
        assert store.get_config_section("profiles") == {"custom": {"chain": []}}
        # Not-yet-present sections are still written.
        assert summary["providers"] == "written"
        assert summary["pricing"] == "written"

    def test_overwrite_replaces_existing(self, tmp_path):
        store, db_path = _db_store(tmp_path)
        store.set_config_section("dynamic_routing", {"enabled": False, "cost_bias": 0.1})
        yaml_path = _write_yaml(tmp_path, _sample_yaml())
        summary = migrate_gateway_yaml(yaml_path, db_path, overwrite=True)
        assert summary["dynamic_routing"] == "written"
        # Verify via a fresh store (the original store's in-memory cache is stale).
        fresh, _ = _db_store(tmp_path)
        assert fresh.get_config_section("dynamic_routing") == {"enabled": True, "cost_bias": 0.4}

    def test_dry_run_writes_nothing(self, tmp_path):
        store, db_path = _db_store(tmp_path)
        yaml_path = _write_yaml(tmp_path, _sample_yaml())
        summary = migrate_gateway_yaml(yaml_path, db_path, overwrite=True, dry_run=True)
        assert summary["profiles"] == "would_write"
        assert store.config_sections() == []  # nothing written

    def test_missing_required_section_raises(self, tmp_path):
        store, db_path = _db_store(tmp_path)
        data = _sample_yaml()
        del data["profiles"]
        yaml_path = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="profiles"):
            migrate_gateway_yaml(yaml_path, db_path)

    def test_missing_file_raises(self, tmp_path):
        store, db_path = _db_store(tmp_path)
        with pytest.raises(ConfigError, match="not found"):
            migrate_gateway_yaml(str(tmp_path / "nope.yaml"), db_path)
