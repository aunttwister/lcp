"""Abstract base class and registry for cost tracking plugins."""

from abc import ABC, abstractmethod
from typing import Optional, Any


class CostPlugin(ABC):
    """Abstract base for provider-specific cost tracking plugins.

    Each plugin handles one provider (e.g. deepseek, opencode, llamacpp).
    Default implementations return None/[] for operations the plugin
    does not support — override only the methods that apply.
    """

    # ── Identity ───────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Lowercase provider identifier, e.g. 'deepseek', 'opencode'."""
        ...

    # ── Model support ──────────────────────────────────────────────────────

    def get_supported_models(self) -> list[str]:
        """Return the list of model names this plugin handles.

        An empty list means 'all models' — the plugin provides global pricing
        or cost calculation covering every model the provider might serve.
        """
        return []

    def discover_models(self, api_base: str) -> list[dict] | None:
        """Query the provider's /models endpoint and return normalized model metadata.

        Args:
            api_base: The provider's API base URL (e.g. https://api.openai.com/v1).

        Returns:
            List of model dicts each with at minimum ``id``, plus optional
            metadata like ``context_length``, ``parameters``, ``quantization``.
            Return ``None`` to use the default generic HTTP-based discovery.
        """
        return None

    def get_api_model(self, model: str) -> str:
        """Translate a gateway model name to the provider's API model ID.

        Most providers accept the same name the gateway uses, so the default is
        a passthrough. Providers whose API uses a different model-ID scheme
        (e.g. Command Code's prefixed catalog IDs) override this.
        """
        return model

    # ── Pricing ────────────────────────────────────────────────────────────

    def get_pricing(self, model: str) -> Optional[dict]:
        """Return per-1M-token pricing for *model*.

        Return format::

            {"cache_hit": float, "cache_miss": float, "output": float}

        Return *None* when the plugin does not have pricing for *model*.
        """
        return None

    # ── Cost calculation ───────────────────────────────────────────────────

    def calculate_cost(self, model: str, usage: dict) -> Optional[float]:
        """Calculate total cost (USD) from a usage dict.

        *usage* typically contains ``prompt_tokens``, ``completion_tokens``,
        ``prompt_cache_hit_tokens``, ``prompt_cache_miss_tokens``.

        Return *None* to let the default pipeline apply generic pricing.
        """
        return None

    # ── Usage history (per-provider API or local storage) ──────────────────

    def fetch_usage(self,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> list[dict]:
        """Fetch historical usage data from the provider's API or local store.

        Returns a list of dicts, each with::

            date             str   YYYY-MM-DD
            model            str
            provider         str
            prompt_tokens    int
            completion_tokens int
            cache_hit_tokens int
            cache_miss_tokens int
            cost             float
            request_count    int

        Return an empty list when the source is unavailable or unsupported.
        """
        return []

    # ── Balance / credit check ─────────────────────────────────────────────

    def fetch_balance(self) -> Optional[dict]:
        """Fetch current account balance / remaining credits.

        Return format::

            {"balance": float, "currency": str, "total_granted": Optional[float]}

        Return *None* when balance info is unavailable.
        """
        return None

    # ── Rich summary (daily/weekly/monthly usage, balance, limits) ─────────

    def fetch_summary(self) -> Optional[dict]:
        """Return a rich provider summary for the dashboard.

        Each plugin returns a provider-specific shape:

        OpenCode (local DB)::

            {
                "daily":   {"tokens": int, "cost": float, "requests": int},
                "weekly":  {"tokens": int, "cost": float, "requests": int},
                "monthly": {"tokens": int, "cost": float, "requests": int},
            }

        DeepSeek (balance API)::

            {
                "balance": {
                    "available": float, "spent": float,
                    "total_granted": float, "currency": str,
                },
            }

        Return *None* when the summary is unavailable.
        """
        return None

    # ── Quick-add preset (provider config template) ────────────────────────

    @property
    def preset(self) -> Optional[dict]:
        """Recommended quick-add preset for this provider.

        Return format::

            {"api_base": str, "models": list[str]}

        Return *None* when there is no recommended preset.
        """
        return None

    # ── Subscription usage (provider web API, optional) ────────────────────

    def fetch_subscription(self) -> Optional[dict]:
        """Fetch subscription usage snapshot from the provider web API.

        Each plugin returns a provider-specific shape.  OpenCode example::

            {"rolling_pct": 17.0, "weekly_pct": 75.0,
             "rolling_reset_sec": 5944, "weekly_reset_sec": 278201}

        Return *None* when subscription data is unavailable.
        """
        return None

    # ── Lifecycle hooks (optional) ─────────────────────────────────────────

    def on_startup(self) -> None:
        """Called once when the plugin is first loaded."""

    def on_shutdown(self) -> None:
        """Called during graceful shutdown (not guaranteed)."""


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

class PluginRegistry:
    """Holds all registered CostPlugin instances and provides lookup helpers.

    Usage::

        registry = PluginRegistry()
        registry.register(DeepSeekPlugin())
        p = registry.for_provider("deepseek")   # -> DeepSeekPlugin instance
    """

    def __init__(self):
        self._plugins: dict[str, CostPlugin] = {}

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, plugin: CostPlugin) -> None:
        """Register a plugin instance."""
        name = plugin.provider_name
        if name in self._plugins:
            raise ValueError(f"Plugin for provider '{name}' is already registered")
        self._plugins[name] = plugin
        plugin.on_startup()

    # ── Lookup ────────────────────────────────────────────────────────────

    def for_provider(self, provider: str) -> Optional[CostPlugin]:
        """Get the plugin for *provider*, or None."""
        return self._plugins.get(provider)

    @property
    def all(self) -> list[CostPlugin]:
        """Return all registered plugins."""
        return list(self._plugins.values())

    @property
    def providers(self) -> list[str]:
        """Return provider names of all registered plugins."""
        return list(self._plugins.keys())

    @property
    def presets(self) -> dict[str, dict]:
        """Collect quick-add presets from all plugins that define one.

        Returns {provider_name: {api_base, models}}.
        """
        result: dict[str, dict] = {}
        for name, plugin in self._plugins.items():
            p = plugin.preset
            if p is not None:
                result[name] = p
        return result

    # ── Delegation helpers ────────────────────────────────────────────────

    def get_pricing(self, provider: str, model: str) -> Optional[dict]:
        """Ask the matching plugin for pricing. Falls back to None."""
        p = self.for_provider(provider)
        return p.get_pricing(model) if p else None

    def calculate_cost(self, provider: str, model: str,
                       usage: dict) -> Optional[float]:
        """Ask the matching plugin to calculate cost. Falls back to None."""
        p = self.for_provider(provider)
        return p.calculate_cost(model, usage) if p else None

    def fetch_all_usage(self,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> dict[str, list[dict]]:
        """Fetch usage from every plugin that has data.

        Returns {provider_name: [usage_dict, ...]}.
        """
        result: dict[str, list[dict]] = {}
        for name, plugin in self._plugins.items():
            data = plugin.fetch_usage(start_date, end_date)
            if data:
                result[name] = data
        return result

    def fetch_all_balances(self) -> dict[str, Optional[dict]]:
        """Fetch balances from every plugin that supports it.

        Returns {provider_name: balance_dict_or_None}.
        """
        result: dict[str, Optional[dict]] = {}
        for name, plugin in self._plugins.items():
            result[name] = plugin.fetch_balance()
        return result

    def fetch_all_summaries(self) -> dict[str, Optional[dict]]:
        """Fetch rich provider summaries from every plugin.

        Returns {provider_name: summary_dict_or_None}.
        """
        result: dict[str, Optional[dict]] = {}
        for name, plugin in self._plugins.items():
            result[name] = plugin.fetch_summary()
        return result

    def fetch_all_subscriptions(self) -> dict[str, Optional[dict]]:
        """Fetch subscription snapshots from every plugin.

        Returns {provider_name: subscription_dict_or_None}.
        """
        result: dict[str, Optional[dict]] = {}
        for name, plugin in self._plugins.items():
            result[name] = plugin.fetch_subscription()
        return result

    def shutdown_all(self) -> None:
        for plugin in self._plugins.values():
            try:
                plugin.on_shutdown()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════════════

_registry: Optional[PluginRegistry] = None


def get_registry() -> PluginRegistry:
    """Return the global plugin registry singleton."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry


def init_plugins(extra_plugins: Optional[list[CostPlugin]] = None,
                 engine: Any = None) -> PluginRegistry:
    """Initialize the global registry and register built-in plugins.

    May be called multiple times — subsequent calls are no-ops.
    Use *extra_plugins* to inject additional or test plugins.
    Pass *engine* (SQLAlchemy engine) to plugins that need gateway DB access.
    """
    global _registry
    if _registry is not None:
        # Ensure the registry has the built-in plugins loaded.
        # Module-level auto-registration may have been bypassed, so
        # we explicitly register them when the registry is empty.
        if not _registry.providers:
            from .deepseek import DeepSeekCostPlugin
            from .opencode import OpenCodeCostPlugin
            from .llamacpp import LlamaCppCostPlugin
            from .commandcode import CommandCodeCostPlugin
            _registry.register(DeepSeekCostPlugin())
            _registry.register(OpenCodeCostPlugin(engine=engine))
            _registry.register(LlamaCppCostPlugin())
            _registry.register(CommandCodeCostPlugin(engine=engine))
        elif engine is not None:
            # Inject engine into already-registered plugins that need it
            for _name, _plugin in _registry._plugins.items():
                if hasattr(_plugin, "set_engine"):
                    _plugin.set_engine(engine)
        return _registry

    _registry = PluginRegistry()
    from .deepseek import DeepSeekCostPlugin
    from .opencode import OpenCodeCostPlugin
    from .llamacpp import LlamaCppCostPlugin
    _registry.register(DeepSeekCostPlugin())
    _registry.register(OpenCodeCostPlugin(engine=engine))
    _registry.register(LlamaCppCostPlugin())

    if extra_plugins:
        for p in extra_plugins:
            _registry.register(p)

    return _registry
