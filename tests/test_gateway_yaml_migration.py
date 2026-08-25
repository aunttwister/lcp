"""Tests for scripts/migrate_gateway_yaml_to_db.py — legacy YAML → DB config.

The script is self-contained (stdlib + yaml, no app imports), so these tests
exercise it directly against a temp SQLite DB.
"""

import json
import sqlite3

import pytest
import yaml

from scripts.migrate_gateway_yaml_to_db import (
    ConfigError,
    migrate_gateway_yaml,
)


def _write_yaml(tmp_path, data) -> str:
    path = tmp_path / "gateway.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return str(path)


def _db_path(tmp_path) -> str:
    return str(tmp_path / "test.db")


def _read_section(db_path, section):
    con = sqlite3.connect(db_path)
    try:
        row = con.execute(
            "SELECT value FROM settings WHERE key = ?",
            (f"gateway_config:{section}",),
        ).fetchone()
        return json.loads(row[0]) if row else None
    finally:
        con.close()


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


class TestMigrate:
    def test_writes_present_sections_by_default(self, tmp_path):
        db_path = _db_path(tmp_path)
        yaml_path = _write_yaml(tmp_path, _sample_yaml())
        summary = migrate_gateway_yaml(yaml_path, db_path, overwrite=True)
        assert summary["server"] == "written"
        assert summary["profiles"] == "written"
        assert summary["providers"] == "written"
        assert summary["pricing"] == "written"
        assert summary["circuit_breaker"] == "written"
        assert summary["database"] == "written"
        assert summary["dynamic_routing"] == "written"
        assert summary["retry"] == "absent"
        assert summary["model_limits"] == "absent"
        assert summary["plugins"] == "absent"
        assert _read_section(db_path, "profiles") == _sample_yaml()["profiles"]
        assert _read_section(db_path, "pricing") == _sample_yaml()["pricing"]
        assert _read_section(db_path, "dynamic_routing") == {"enabled": True, "cost_bias": 0.4}

    def test_if_absent_skips_existing(self, tmp_path):
        db_path = _db_path(tmp_path)
        # Pre-seed a profiles section directly.
        con = sqlite3.connect(db_path)
        con.execute(
            "CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "key TEXT UNIQUE NOT NULL, value TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        con.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("gateway_config:profiles", json.dumps({"custom": {"chain": []}}), "now"),
        )
        con.commit()
        con.close()
        yaml_path = _write_yaml(tmp_path, _sample_yaml())
        summary = migrate_gateway_yaml(yaml_path, db_path, overwrite=False)
        assert summary["profiles"] == "skipped_exists"
        assert _read_section(db_path, "profiles") == {"custom": {"chain": []}}
        assert summary["providers"] == "written"
        assert summary["pricing"] == "written"

    def test_overwrite_replaces_existing(self, tmp_path):
        db_path = _db_path(tmp_path)
        con = sqlite3.connect(db_path)
        con.execute(
            "CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "key TEXT UNIQUE NOT NULL, value TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        con.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("gateway_config:dynamic_routing",
             json.dumps({"enabled": False, "cost_bias": 0.1}), "now"),
        )
        con.commit()
        con.close()
        yaml_path = _write_yaml(tmp_path, _sample_yaml())
        summary = migrate_gateway_yaml(yaml_path, db_path, overwrite=True)
        assert summary["dynamic_routing"] == "written"
        assert _read_section(db_path, "dynamic_routing") == {"enabled": True, "cost_bias": 0.4}

    def test_dry_run_writes_nothing(self, tmp_path):
        db_path = _db_path(tmp_path)
        yaml_path = _write_yaml(tmp_path, _sample_yaml())
        summary = migrate_gateway_yaml(yaml_path, db_path, overwrite=True, dry_run=True)
        assert summary["profiles"] == "would_write"
        con = sqlite3.connect(db_path)
        try:
            count = con.execute(
                "SELECT COUNT(*) FROM settings WHERE key LIKE 'gateway_config:%'"
            ).fetchone()[0]
        finally:
            con.close()
        assert count == 0  # nothing written

    def test_missing_required_section_raises(self, tmp_path):
        db_path = _db_path(tmp_path)
        data = _sample_yaml()
        del data["profiles"]
        yaml_path = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="profiles"):
            migrate_gateway_yaml(yaml_path, db_path)

    def test_missing_file_raises(self, tmp_path):
        db_path = _db_path(tmp_path)
        with pytest.raises(ConfigError, match="not found"):
            migrate_gateway_yaml(str(tmp_path / "nope.yaml"), db_path)
