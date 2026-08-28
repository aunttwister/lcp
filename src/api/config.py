"""DB-backed gateway configuration.

The whole gateway config lives in the ``settings`` table as JSON blobs under
``gateway_config:<section>`` (server, profiles, providers, pricing,
circuit_breaker, retry, database, dynamic_routing, model_limits, plugins).
``gateway.yaml`` and the YAML hot-reload are obsolete: a Python ``SEED_CONFIG``
dict seeds a fresh DB on first boot, and edits via the UI (which mutate
``config.raw`` then call ``config.save()``) are written straight to the DB, so
they persist across restarts.
"""

import os
from typing import Any, Optional

from .exceptions import ConfigError
from .logging_config import get_logger

logger = get_logger("lcp.config")


# ─────────────────────────────────────────────────────────────────────────────
# Default seed — used ONLY to initialise a fresh DB (first boot) or when a
# section is missing. Once a section row exists in the DB it is the source of
# truth; edits to this dict do not affect a running gateway.
# ─────────────────────────────────────────────────────────────────────────────
SEED_CONFIG: dict[str, Any] = {
    "server": {
        "port": 8734,
        "default_profile": "l2",
    },
    "dynamic_routing": {
        "enabled": False,
        "cost_bias": 0.15,
    },
    "profiles": {
        "l2": {
            "forbidden_tools": ["write_file", "patch", "cronjob"],
            "chain": [
                {"provider": "opencode", "model": "deepseek-v4-pro",
                 "base_url": "https://opencode.ai/zen/go/v1"},
                {"provider": "deepseek", "model": "deepseek-v4-pro",
                 "base_url": "https://api.deepseek.com/v1"},
            ],
            "auth_required": False,
        },
        "l1": {
            "forbidden_tools": ["write_file", "patch", "terminal", "execute_code",
                                "cronjob", "process", "delegate_task", "memory",
                                "send_message", "vision_analyze"],
            "chain": [
                {"provider": "opencode", "model": "deepseek-v4-flash",
                 "base_url": "https://opencode.ai/zen/go/v1"},
                {"provider": "deepseek", "model": "deepseek-v4-flash",
                 "base_url": "https://api.deepseek.com/v1"},
            ],
            "auth_required": False,
        },
        "career": {
            "forbidden_tools": ["write_file", "patch", "terminal", "execute_code",
                                "cronjob", "process", "delegate_task", "memory",
                                "vision_analyze", "read_file", "search_files",
                                "skill_manage", "todo"],
            "chain": [
                {"provider": "deepseek", "model": "deepseek-v4-flash",
                 "base_url": "https://api.deepseek.com/v1"},
            ],
            "auth_required": False,
        },
        "cron": {
            "chain": [
                {"provider": "deepseek", "model": "deepseek-v4-flash",
                 "base_url": "https://api.deepseek.com/v1"},
            ],
            "forbidden_tools": None,
            "auth_required": False,
        },
        "coder": {
            "chain": [
                {"provider": "opencode", "model": "deepseek-v4-pro",
                 "base_url": "https://opencode.ai/zen/go/v1"},
                {"provider": "deepseek", "model": "deepseek-v4-flash",
                 "base_url": "https://api.deepseek.com/v1"},
            ],
            "forbidden_tools": [],
            "auth_required": True,
        },
    },
    "providers": {
        "opencode": {
            "api_key_env": "OPENCODE_API_KEY",
            "api_base": "https://opencode.ai/zen/go/v1",
            "models": ["deepseek-v4-pro", "deepseek-v4-flash"],
        },
        "deepseek": {
            "api_key_env": "DEEPSEEK_API_KEY",
            "api_base": "https://api.deepseek.com/v1",
            "models": ["deepseek-v4-pro", "deepseek-v4-flash"],
            "cache": {
                "strategy": "prefix",
                "savings": "cost",
                "hit_field": "prompt_cache_hit_tokens",
            },
        },
        "commandcode": {
            "api_base": "https://api.commandcode.ai/provider/v1",
            "models": ["deepseek-v4-pro", "deepseek-v4-flash", "claude-sonnet-5",
                       "gpt-5.6-luna", "kimi-k3", "minimax-m3", "qwen3.8-max"],
        },
    },
    "pricing": [
        {"provider": "deepseek", "model": "deepseek-v4-pro",
         "cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87},
        {"provider": "deepseek", "model": "deepseek-v4-flash",
         "cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28},
        {"provider": "opencode", "model": "deepseek-v4-pro",
         "cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87},
        {"provider": "opencode", "model": "deepseek-v4-flash",
         "cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28},
    ],
    "circuit_breaker": {
        "failures_degraded": 3,
        "failures_dead": 6,
        "degraded_cooldown_seconds": 30,
        "dead_cooldown_seconds": 120,
    },
    "retry": {
        "max_attempts": 3,
        "backoff_base": 0.5,
        "backoff_multiplier": 2,
        "max_backoff": 10,
        "jitter": True,
    },
    "model_limits": {
        "deepseek-v4-pro": {
            "context_window": 1000000,
            "max_output_tokens": 384000,
            "supports_vision": False,
            "supports_thinking": True,
            "description": "DeepSeek V4 Pro — flagship MoE for coding, reasoning, and agentic work",
        },
        "deepseek-v4-flash": {
            "context_window": 1000000,
            "max_output_tokens": 384000,
            "supports_vision": False,
            "supports_thinking": True,
            "description": "DeepSeek V4 Flash — fast lane for economical reasoning and long-context work",
        },
    },
    "database": {
        "path": "/app/data/costs.db",
        "wal_mode": True,
    },
    "plugins": {
        "memory": {
            "enabled": True,
            "auto_recall": False,   # opt-in: auto-inject recalled facts into requests
            "top_k": 3,
            "min_score": 0.0,
            "embedding": {"model": "BAAI/bge-small-en-v1.5", "dim": 384, "device": "cpu"},
        },
        # SEMANTIC ROUTING module — the sole task classifier (no keyword
        # fallback). Deps + model weights are baked into the image by the
        # Docker build (WITH_ROUTER=1); this block just pins the model and
        # the confidence gate.
        "router": {
            "enabled": True,
            "min_score": 0.35,
            "embedding": {"model": "BAAI/bge-small-en-v1.5", "dim": 384, "device": "cpu"},
        },
    },
}

# Sections stored in the DB. ``plugins`` is optional (memory module tolerates
# its absence), the rest are required for a valid config.
REQUIRED_SECTIONS = ("server", "profiles", "providers", "pricing",
                     "circuit_breaker", "database")
ALL_SECTIONS = ("server", "profiles", "providers", "pricing",
                "circuit_breaker", "retry", "database",
                "dynamic_routing", "model_limits", "plugins")


def _env_db_path() -> str:
    """Resolve the DB path from env (COST_DB) or the seed default."""
    return os.environ.get("COST_DB", SEED_CONFIG["database"]["path"])


def _env_port() -> int:
    """Resolve the listen port from env (LISTEN_PORT) or the seed default."""
    return int(os.environ.get("LISTEN_PORT", str(SEED_CONFIG["server"]["port"])))


def _validate(section: str, data: Any) -> None:
    """Validate a loaded section; raise ConfigError on structural problems."""
    if section == "server":
        if not isinstance(data, dict) or "port" not in data:
            raise ConfigError("Missing 'server.port'")
        if "default_profile" not in data:
            raise ConfigError("Missing 'server.default_profile'")
    elif section == "profiles":
        if not isinstance(data, dict):
            raise ConfigError("'profiles' must be a dict")
        for name, prof in data.items():
            if not isinstance(prof, dict) or "chain" not in prof:
                raise ConfigError(f"Profile '{name}' missing 'chain'")
            if not prof["chain"]:
                raise ConfigError(f"Profile '{name}' has empty 'chain'")
    elif section == "pricing":
        if not isinstance(data, list):
            raise ConfigError("'pricing' must be a list")
    elif section in ("providers", "circuit_breaker", "database"):
        if not isinstance(data, dict):
            raise ConfigError(f"'{section}' must be a dict")
    # dynamic_routing / retry / model_limits / plugins are optional; handled by
    # the accessors.


class Config:
    """DB-backed gateway configuration.

    Hydrated from the ``settings`` table (``gateway_config:<section>`` rows)
    at init, seeding any missing section from ``SEED_CONFIG``. ``raw`` returns
    the live in-memory dict (mutated by the CRUD endpoints); ``save()`` writes
    every section back to the DB. There is no YAML file and no hot-reload.
    """

    def __init__(self, store: Optional[Any] = None, seed: Optional[dict] = None):
        self._store = store
        self._seed = seed or SEED_CONFIG
        self._data: dict[str, Any] = {}
        self._hydrate()

    def _hydrate(self) -> None:
        """Load every section into ``_data``: DB row if present, else seed."""
        for section in ALL_SECTIONS:
            stored = None
            if self._store is not None:
                try:
                    stored = self._store.get_config_section(section, None)
                except Exception:  # noqa: BLE001 — never break config reads
                    stored = None
            if isinstance(stored, (dict, list)) and stored:
                self._data[section] = stored
            elif section in self._seed and self._seed[section] is not None:
                import copy
                self._data[section] = copy.deepcopy(self._seed[section])
            else:
                self._data[section] = {}
        # Validate required sections; fall back to seed on failure so boot and
        # read paths never hard-fail from a corrupt DB section.
        for section in REQUIRED_SECTIONS:
            try:
                _validate(section, self._data.get(section))
            except ConfigError as exc:
                logger.warning("config_section_invalid", section=section, error=str(exc))
                if section in self._seed:
                    import copy
                    self._data[section] = copy.deepcopy(self._seed[section])
        # Env overrides for the two bootstrap-critical values — only when the
        # env var is actually set (so a DB value is preserved otherwise).
        try:
            if os.environ.get("LISTEN_PORT"):
                self._data["server"]["port"] = _env_port()
        except Exception:  # noqa: BLE001
            pass
        try:
            if os.environ.get("COST_DB"):
                self._data["database"]["path"] = _env_db_path()
        except Exception:  # noqa: BLE001
            pass
        logger.info("config_loaded", sections=sorted(self._data.keys()))

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
        return self._data.get("circuit_breaker", {})

    @property
    def retry(self) -> dict:
        return self._data.get("retry", {
            "max_attempts": 3,
            "backoff_base": 0.5,
            "backoff_multiplier": 2,
            "max_backoff": 10,
            "jitter": True,
        })

    @property
    def database(self) -> dict:
        return self._data["database"]

    @property
    def dynamic_routing(self) -> dict:
        return self._data.get("dynamic_routing", {
            "enabled": False,
            "cost_bias": 0.15,
        })

    @property
    def model_limits(self) -> dict:
        return self._data.get("model_limits", {})

    @property
    def plugins(self) -> dict:
        """Plugin config blocks, e.g. ``plugins.memory``.

        Absent/partial config falls back to the memory plugin's defaults.
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
        """Return cache config for a provider, or empty dict if not configured."""
        p = self.providers.get(provider_name, {})
        return p.get("cache", {"strategy": "none", "savings": "none", "hit_field": None})

    @property
    def raw(self) -> dict:
        return self._data

    def save(self) -> None:
        """Persist every section to the settings DB (source of truth).

        This is the write path for all UI/config edits (the CRUD endpoints
        mutate ``config.raw`` then call ``save()``). Missing sections are
        seeded; present sections are fully replaced by the current ``_data``.
        """
        if self._store is None:
            logger.warning("config_save_no_store", error=True)
            return
        for section in ALL_SECTIONS:
            value = self._data.get(section)
            if value is None:
                continue
            try:
                self._store.set_config_section(section, value)
            except Exception:  # noqa: BLE001 — best-effort per section
                logger.warning("config_section_save_failed", section=section)
        logger.info("config_saved")


# Global config instance — loaded at startup
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        # Back-compat fallback: build from a store if one is already bound.
        from .cost_cache import get_settings
        _config = Config(store=get_settings())
    return _config


def init_config(store: Optional[Any] = None) -> Config:
    """Create the global Config bound to the settings store (DB-backed)."""
    global _config
    _config = Config(store=store)
    return _config
