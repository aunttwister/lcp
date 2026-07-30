"""YAML configuration loader with validation and hot-reload."""

import os
import time
from pathlib import Path
from typing import Any

import structlog
import yaml

from .logging_config import get_logger

logger = get_logger("lcp.config")


class ConfigError(Exception):
    """Configuration validation error."""


class Config:
    """Loads and validates gateway.yaml. Supports hot-reload via file mtime tracking."""

    def __init__(self, path: str = "/app/config/gateway.yaml"):
        self._path = Path(path)
        self._data: dict[str, Any] = {}
        self._mtime: float = 0
        self._reload()

    def _reload(self) -> None:
        """Load and validate the config file."""
        if not self._path.exists():
            raise ConfigError(f"Config file not found: {self._path}")

        with open(self._path) as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raise ConfigError(f"Empty config file: {self._path}")

        self._validate(raw)
        self._data = raw
        self._mtime = self._path.stat().st_mtime
        logger.info("config_loaded", path=str(self._path))

    def _validate(self, raw: dict) -> None:
        """Validate required sections exist."""
        required = ["server", "profiles", "providers", "pricing", "circuit_breaker", "database"]
        for key in required:
            if key not in raw:
                raise ConfigError(f"Missing required config section: '{key}'")

        if "port" not in raw["server"]:
            raise ConfigError("Missing 'server.port'")
        if "default_profile" not in raw["server"]:
            raise ConfigError("Missing 'server.default_profile'")
        if raw["server"]["default_profile"] not in raw["profiles"]:
            raise ConfigError(
                f"Default profile '{raw['server']['default_profile']}' not found in profiles"
            )

        for name, prof in raw["profiles"].items():
            if "chain" not in prof:
                raise ConfigError(f"Profile '{name}' missing 'chain'")
            if not prof["chain"]:
                raise ConfigError(f"Profile '{name}' has empty 'chain'")

    def check_reload(self) -> bool:
        """Check if config file changed on disk. Returns True if reloaded."""
        try:
            mtime = self._path.stat().st_mtime
            if mtime > self._mtime:
                self._reload()
                return True
        except FileNotFoundError:
            pass
        except Exception:
            logger.warning("config_reload_failed", exc_info=True)
        return False

    # ── Accessors ──────────────────────────────────────────────────────────

    @property
    def server(self) -> dict:
        return self._data["server"]

    @property
    def profiles(self) -> dict:
        return self._data["profiles"]

    @property
    def providers(self) -> dict:
        return self._data["providers"]

    @property
    def pricing(self) -> list:
        return self._data["pricing"]

    @property
    def circuit_breaker(self) -> dict:
        return self._data["circuit_breaker"]

    @property
    def database(self) -> dict:
        return self._data["database"]

    def get_profile(self, name: str) -> dict | None:
        return self.profiles.get(name)

    def get_provider_key(self, provider_name: str) -> str:
        """Resolve API key from environment variable."""
        env_var = self.providers[provider_name]["api_key_env"]
        key = os.environ.get(env_var)
        if not key:
            raise ConfigError(f"Environment variable {env_var} not set for provider {provider_name}")
        return key

    def get_pricing(self, provider: str, model: str) -> dict:
        for p in self.pricing:
            if p["provider"] == provider and p["model"] == model:
                return p
        raise ConfigError(f"No pricing found for {provider}/{model}")

    def get_provider_cache_config(self, provider_name: str) -> dict:
        """Return cache config for a provider, or empty dict if not configured.

        Returns: {"strategy": "prefix"|"none", "savings": "cost"|"latency"|"none",
                   "hit_field": "prompt_cache_hit_tokens"|...|null}
        """
        p = self.providers.get(provider_name, {})
        return p.get("cache", {"strategy": "none", "savings": "none", "hit_field": None})

    @property
    def raw(self) -> dict:
        return self._data

    def save(self) -> None:
        import tempfile, shutil
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", dir=self._path.parent, delete=False)
        try:
            yaml.dump(self._data, tmp, default_flow_style=False, sort_keys=False, allow_unicode=True)
            tmp.flush()
            shutil.move(tmp.name, self._path)
            self._mtime = self._path.stat().st_mtime
            logger.info("config_saved", path=str(self._path))
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    def reload(self) -> None:
        self._reload()


# Global config instance — loaded at startup
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        config_path = os.environ.get("LCP_CONFIG", "/app/config/gateway.yaml")
        _config = Config(config_path)
    return _config


def init_config(path: str | None = None) -> Config:
    global _config
    config_path = path or os.environ.get("LCP_CONFIG", "/app/config/gateway.yaml")
    _config = Config(config_path)
    return _config
