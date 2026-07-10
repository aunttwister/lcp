"""Tests for config.py — YAML config loading, validation, hot-reload."""

import os
import tempfile
import time

import pytest
import yaml

import sys
sys.path.insert(0, "/opt/lcp")

from src.config import Config, ConfigError


def _write_config(data: dict) -> str:
    """Write a dict as YAML to a temp file, return path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data, tmp)
    tmp.close()
    return tmp.name


def _base_config() -> dict:
    return {
        "server": {"port": 8734, "default_profile": "l2"},
        "profiles": {
            "l2": {"chain": [{"provider": "opencode", "model": "deepseek-v4-pro"},
                             {"provider": "deepseek", "model": "deepseek-v4-flash"}],
                   "forbidden_tools": []},
            "l1": {"chain": [{"provider": "deepseek", "model": "deepseek-v4-flash"}]},
        },
        "providers": {
            "openai": {"api_key_env": "OPENAI_KEY", "base_url": "https://api.openai.com/v1"},
            "deepseek": {"api_key_env": "DEEPSEEK_KEY", "base_url": "https://api.deepseek.com/v1"},
        },
        "pricing": [
            {"provider": "deepseek", "model": "deepseek-v4-pro", "prompt": 2.5, "completion": 10.0},
            {"provider": "deepseek", "model": "deepseek-v4-flash", "prompt": 0.27, "completion": 1.10},
        ],
        "circuit_breaker": {"failures_before_skip": 5, "cooldown_seconds": 60},
        "database": {"path": "/app/data/costs.db"},
    }


class TestConfigLoading:
    def test_loads_valid_config(self):
        path = _write_config(_base_config())
        try:
            cfg = Config(path)
            assert cfg.server["port"] == 8734
            assert cfg.server["default_profile"] == "l2"
        finally:
            os.unlink(path)

    def test_profiles_loaded(self):
        path = _write_config(_base_config())
        try:
            cfg = Config(path)
            assert "l2" in cfg.profiles
            assert "l1" in cfg.profiles
            assert len(cfg.profiles["l2"]["chain"]) == 2
        finally:
            os.unlink(path)

    def test_providers_loaded(self):
        path = _write_config(_base_config())
        try:
            cfg = Config(path)
            assert "openai" in cfg.providers
            assert "deepseek" in cfg.providers
        finally:
            os.unlink(path)

    def test_pricing_loaded(self):
        path = _write_config(_base_config())
        try:
            cfg = Config(path)
            pricing = cfg.get_pricing("deepseek", "deepseek-v4-pro")
            assert pricing["prompt"] == 2.5
            assert pricing["completion"] == 10.0
        finally:
            os.unlink(path)


class TestConfigValidation:
    def test_missing_section_raises(self):
        data = _base_config()
        del data["pricing"]
        path = _write_config(data)
        try:
            with pytest.raises(ConfigError, match="pricing"):
                Config(path)
        finally:
            os.unlink(path)

    def test_missing_server_port(self):
        data = _base_config()
        del data["server"]["port"]
        path = _write_config(data)
        try:
            with pytest.raises(ConfigError, match="port"):
                Config(path)
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        with pytest.raises(ConfigError, match="not found"):
            Config("/nonexistent/path/config.yaml")


class TestConfigAccessors:
    def test_get_profile(self):
        path = _write_config(_base_config())
        try:
            cfg = Config(path)
            p = cfg.get_profile("l2")
            assert p is not None
            assert len(p["chain"]) == 2
        finally:
            os.unlink(path)

    def test_get_profile_missing(self):
        path = _write_config(_base_config())
        try:
            cfg = Config(path)
            assert cfg.get_profile("nonexistent") is None
        finally:
            os.unlink(path)

    def test_circuit_breaker(self):
        path = _write_config(_base_config())
        try:
            cfg = Config(path)
            assert cfg.circuit_breaker["failures_before_skip"] == 5
        finally:
            os.unlink(path)

    def test_database(self):
        path = _write_config(_base_config())
        try:
            cfg = Config(path)
            assert cfg.database["path"] == "/app/data/costs.db"
        finally:
            os.unlink(path)
