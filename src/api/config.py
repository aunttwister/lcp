"""YAML configuration loader with validation and hot-reload."""

import os
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigError
from .logging_config import get_logger

logger = get_logger("lcp.config")


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

    # ── DB-backed runtime config sections ─────────────────────────────────
    # The runtime-tunable sections (dynamic_routing, retry, circuit_breaker,
    # model_limits) live in the settings DB (gateway_config:<section>) once
    # the store is available, so edits via the UI persist and survive
    # restarts. gateway.yaml remains only a seed: when no DB row exists the
    # YAML value is used. ``save()`` writes both so YAML stays a working
    # seed/backup.

    _DB_SECTIONS = ("dynamic_routing", "retry", "circuit_breaker", "model_limits")

    def _db_section(self, section: str, default: dict) -> dict:
        """Return a DB-backed section if the store has it, else *default*."""
        try:
            from .cost_cache import get_settings
            store = get_settings()
            if store is not None:
                stored = store.get_config_section(section, None)
                if isinstance(stored, dict) and stored:
                    return stored
        except Exception:  # noqa: BLE001 — never break config reads
            pass
        return default

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
        return self._db_section("circuit_breaker", self._data["circuit_breaker"])

    @property
    def retry(self) -> dict:
        """Per-provider retry config, with sensible defaults when absent."""
        default = self._data.get("retry", {
            "max_attempts": 3,
            "backoff_base": 0.5,
            "backoff_multiplier": 2,
            "max_backoff": 10,
            "jitter": True,
        })
        return self._db_section("retry", default)

    @property
    def database(self) -> dict:
        return self._data["database"]

    @property
    def dynamic_routing(self) -> dict:
        """Dynamic routing config, with sensible defaults when absent.

        ``enabled`` gates the CapabilityRouter; ``cost_bias`` controls how
        strongly cheaper models are favored (0.0 = pure capability). Prefers
        the DB-backed section so the UI toggle persists; falls back to the
        YAML seed (which defaults to disabled).
        """
        default = self._data.get("dynamic_routing", {
            "enabled": False,
            "cost_bias": 0.15,
        })
        return self._db_section("dynamic_routing", default)

    @property
    def model_limits(self) -> dict:
        return self._db_section("model_limits", self._data.get("model_limits", {}))

    @property
    def plugins(self) -> dict:
        """Plugin config blocks, e.g. ``plugins.memory``.

        Absent/partial config falls back to the memory plugin's defaults
        (enabled with the default embedding model). No validation here — the
        memory module tolerates a missing block.
        """
        return self._data.get("plugins", {})

    def get_model_limits(self, model_id: str) -> dict | None:
        """Return context_window, max_output_tokens, description for a model, or None."""
        return self.model_limits.get(model_id)

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
        # 1. Seed runtime-tunable sections into the settings DB only when a
        #    row is absent. The migrated sections are DB-owned at runtime (the
        #    UI/routing toggle writes them straight to the settings table), so
        #    we must NOT clobber an existing DB value with the (possibly
        #    stale) YAML ``_data`` on unrelated config saves.
        try:
            from .cost_cache import get_settings
            store = get_settings()
            if store is not None:
                for section in self._DB_SECTIONS:
                    if store.get_config_section(section, None) is None:
                        value = self._data.get(section)
                        if isinstance(value, dict) and value:
                            store.set_config_section(section, dict(value))
        except Exception:  # noqa: BLE001 — DB write must not break YAML save
            logger.warning("config_db_save_failed", error=True)
        # 2. Keep writing YAML as a working seed/backup (and for the sections
        #    that still live in YAML: server/profiles/providers/pricing/...).
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
