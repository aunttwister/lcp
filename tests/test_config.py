"""Tests for config.py — YAML config loading, validation, hot-reload."""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


from src.api.config import Config, ConfigError


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

    def test_dynamic_routing_defaults_disabled(self):
        path = _write_config(_base_config())
        try:
            cfg = Config(path)
            dr = cfg.dynamic_routing
            assert dr["enabled"] is False
            assert dr["cost_bias"] == 0.15
        finally:
            os.unlink(path)

    def test_dynamic_routing_from_config(self):
        data = _base_config()
        data["dynamic_routing"] = {"enabled": True, "cost_bias": 0.3}
        path = _write_config(data)
        try:
            cfg = Config(path)
            assert cfg.dynamic_routing["enabled"] is True
            assert cfg.dynamic_routing["cost_bias"] == 0.3
        finally:
            os.unlink(path)


class TestConfigEdgeCases:
    """Tests for validation edge cases and lesser-tested methods."""

    def test_empty_config_raises(self):
        path = _write_config({})
        try:
            with pytest.raises(ConfigError):
                Config(path)
        finally:
            os.unlink(path)

    def test_empty_yaml_file_raises(self, temp_dir):
        path = temp_dir / "empty.yaml"
        path.write_text("")
        with pytest.raises(ConfigError, match="Empty"):
            Config(str(path))

    def test_missing_providers_section(self):
        data = _base_config()
        del data["providers"]
        path = _write_config(data)
        try:
            with pytest.raises(ConfigError, match="providers"):
                Config(path)
        finally:
            os.unlink(path)

    def test_missing_circuit_breaker_section(self):
        data = _base_config()
        del data["circuit_breaker"]
        path = _write_config(data)
        try:
            with pytest.raises(ConfigError, match="circuit_breaker"):
                Config(path)
        finally:
            os.unlink(path)

    def test_missing_database_section(self):
        data = _base_config()
        del data["database"]
        path = _write_config(data)
        try:
            with pytest.raises(ConfigError, match="database"):
                Config(path)
        finally:
            os.unlink(path)

    def test_missing_default_profile(self):
        data = _base_config()
        del data["server"]["default_profile"]
        path = _write_config(data)
        try:
            with pytest.raises(ConfigError, match="default_profile"):
                Config(path)
        finally:
            os.unlink(path)

    def test_default_profile_not_exists(self):
        data = _base_config()
        data["server"]["default_profile"] = "nope"
        path = _write_config(data)
        try:
            with pytest.raises(ConfigError, match="not found in profiles"):
                Config(path)
        finally:
            os.unlink(path)

    def test_profile_missing_chain(self):
        data = _base_config()
        del data["profiles"]["l2"]["chain"]
        path = _write_config(data)
        try:
            with pytest.raises(ConfigError, match="missing 'chain'"):
                Config(path)
        finally:
            os.unlink(path)

    def test_profile_empty_chain(self):
        data = _base_config()
        data["profiles"]["l2"]["chain"] = []
        path = _write_config(data)
        try:
            with pytest.raises(ConfigError, match="empty 'chain'"):
                Config(path)
        finally:
            os.unlink(path)

    def test_raw_property(self):
        path = _write_config(_base_config())
        try:
            cfg = Config(path)
            assert cfg.raw["server"]["port"] == 8734
        finally:
            os.unlink(path)

    def test_get_provider_key_with_env(self):
        path = _write_config(_base_config())
        try:
            cfg = Config(path)
            with patch.dict(os.environ, {"OPENAI_KEY": "sk-test123"}):
                key = cfg.get_provider_key("openai")
                assert key == "sk-test123"
        finally:
            os.unlink(path)

    def test_get_provider_key_missing_env(self):
        path = _write_config(_base_config())
        try:
            cfg = Config(path)
            # Ensure env var is not set
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ConfigError, match="not set"):
                    cfg.get_provider_key("openai")
        finally:
            os.unlink(path)

    def test_get_pricing_not_found(self):
        path = _write_config(_base_config())
        try:
            cfg = Config(path)
            with pytest.raises(ConfigError, match="No pricing found"):
                cfg.get_pricing("unknown", "unknown")
        finally:
            os.unlink(path)

    def test_save_and_reload(self, temp_dir):
        data = _base_config()
        path = temp_dir / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        cfg = Config(str(path))
        # Modify and save
        cfg.raw["server"]["port"] = 9999
        cfg.save()
        # Reload from disk
        cfg2 = Config(str(path))
        assert cfg2.server["port"] == 9999
        # Clean up
        os.unlink(str(path))

    def test_check_reload_no_change(self, temp_dir):
        data = _base_config()
        path = temp_dir / "config2.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        cfg = Config(str(path))
        # No change on disk, should not reload
        assert cfg.check_reload() is False
        os.unlink(str(path))

    def test_check_reload_file_gone(self, temp_dir):
        data = _base_config()
        path = temp_dir / "config3.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        cfg = Config(str(path))
        os.unlink(str(path))  # Delete the file
        # File gone — should not raise
        assert cfg.check_reload() is False

    def test_check_reload_triggers_reload(self, temp_dir):
        """File modified externally triggers reload."""
        data = _base_config()
        path = temp_dir / "config5.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        cfg = Config(str(path))
        # Modify file externally to advance mtime
        data["server"]["port"] = 9999
        time.sleep(0.01)
        with open(path, "w") as f:
            yaml.dump(data, f)
        assert cfg.check_reload() is True
        assert cfg.server["port"] == 9999
        os.unlink(str(path))

    def test_check_reload_exception(self, temp_dir):
        """Exception during stat is caught and logged."""
        data = _base_config()
        path = temp_dir / "config6.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        cfg = Config(str(path))
        with patch.object(Path, 'stat', side_effect=PermissionError("denied")):
            assert cfg.check_reload() is False
        os.unlink(str(path))

    def test_get_provider_cache_config_default(self, temp_dir):
        """Non-existent provider returns default cache config."""
        data = _base_config()
        path = temp_dir / "config7.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        cfg = Config(str(path))
        result = cfg.get_provider_cache_config("nonexistent")
        assert result == {"strategy": "none", "savings": "none", "hit_field": None}
        os.unlink(str(path))

    def test_save_failure_cleans_up_tempfile(self, temp_dir):
        """If shutil.move fails, the temp file is still cleaned up."""
        data = _base_config()
        path = temp_dir / "config_fail.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        cfg = Config(str(path))
        cfg.raw["server"]["port"] = 9999
        with patch("shutil.move", side_effect=OSError("permission denied")):
            with pytest.raises(OSError):
                cfg.save()
        # The temp file created by NamedTemporaryFile should have been cleaned up
        os.unlink(str(path))

    def test_get_config_creates_when_none(self, temp_dir):
        """get_config creates a new Config when _config is None."""
        import src.api.config as cfg_mod
        data = _base_config()
        path = temp_dir / "get_config.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        cfg_mod._config = None
        with patch.dict(os.environ, {"LCP_CONFIG": str(path)}):
            cfg = cfg_mod.get_config()
            assert cfg.server["port"] == 8734
        os.unlink(str(path))

    def test_init_config_function(self, temp_dir):
        """init_config creates and returns a new Config."""
        import src.api.config as cfg_mod
        data = _base_config()
        path = temp_dir / "gateway2.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        cfg_mod._config = None  # reset global
        cfg = cfg_mod.init_config(str(path))
        assert cfg.server["port"] == 8734
        os.unlink(str(path))

    def test_init_config_with_env(self, temp_dir):
        data = _base_config()
        path = temp_dir / "gateway.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        with patch.dict(os.environ, {"LCP_CONFIG": str(path)}):
            from src.api.config import get_config
            cfg = get_config()
            assert cfg.server["port"] == 8734
        os.unlink(str(path))
