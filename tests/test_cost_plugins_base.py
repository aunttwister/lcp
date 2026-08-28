"""Tests for the cost plugin base — CostPlugin ABC, PluginRegistry, singleton."""

import pytest

from src.api.cost_plugins.base import (
    CostPlugin,
    PluginRegistry,
    get_registry,
    init_plugins,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

class _DummyPlugin(CostPlugin):
    """Minimal concrete plugin for testing the registry."""

    @property
    def provider_name(self) -> str:
        return "dummy"

    def get_pricing(self, model: str):
        if model == "known":
            return {"cache_hit": 0.01, "cache_miss": 0.50, "output": 1.00}
        return None

    def calculate_cost(self, model: str, usage: dict):
        if model == "known":
            return 0.42
        return None

    def fetch_balance(self):
        return {"balance": 100.0, "currency": "USD"}

    def discover_models(self, api_base: str):
        return [{"id": "model-from-plugin", "context_length": 4096}]


class _AnotherPlugin(CostPlugin):
    """Second plugin for multi-registration tests."""

    @property
    def provider_name(self) -> str:
        return "another"

    def fetch_usage(self, start_date=None, end_date=None):
        return [{"date": "2026-01-01", "provider": "another", "cost": 5.0}]


# ═══════════════════════════════════════════════════════════════════════
# CostPlugin ABC
# ═══════════════════════════════════════════════════════════════════════

class _MinimalPlugin(CostPlugin):
    """Plugin that overrides NOTHING beyond provider_name.

    Used to verify that the base class defaults (return None / [])
    are exercised.
    """

    @property
    def provider_name(self) -> str:
        return "minimal"


class TestCostPluginABC:
    def test_cannot_instantiate_abc(self):
        """The ABC should not be instantiable without implementing provider_name."""
        with pytest.raises(TypeError):
            CostPlugin()

    def test_dummy_plugin_is_concrete(self):
        p = _DummyPlugin()
        assert p.provider_name == "dummy"

    def test_defaults_return_none_or_empty(self):
        """Un-overridden hooks should return sensible defaults."""
        p = _DummyPlugin()
        assert p.get_supported_models() == []
        assert p.fetch_usage() == []
        assert p.on_startup() is None
        assert p.on_shutdown() is None

    def test_default_discover_models_returns_none(self):
        """Plugins that don't override discover_models should return None."""
        p = _MinimalPlugin()
        assert p.discover_models("http://test.local/v1") is None

    def test_dummy_plugin_discover_models(self):
        p = _DummyPlugin()
        result = p.discover_models("http://test.local/v1")
        assert result == [{"id": "model-from-plugin", "context_length": 4096}]

    def test_minimal_plugin_returns_none_defaults(self):
        """Plugin that overrides nothing returns None for pricing/cost/preset/balance."""
        p = _MinimalPlugin()
        assert p.get_pricing("any") is None
        assert p.calculate_cost("any", {}) is None
        assert p.preset is None
        assert p.fetch_balance() is None
        assert p.fetch_summary() is None
        assert p.fetch_subscription() is None
        assert p.discover_models("http://x/v1") is None


# ═══════════════════════════════════════════════════════════════════════
# PluginRegistry
# ═══════════════════════════════════════════════════════════════════════

class TestPluginRegistry:
    def test_register_and_lookup(self):
        reg = PluginRegistry()
        p = _DummyPlugin()
        reg.register(p)
        assert reg.for_provider("dummy") is p
        assert reg.for_provider("nonexistent") is None

    def test_register_duplicate_raises(self):
        reg = PluginRegistry()
        reg.register(_DummyPlugin())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_DummyPlugin())

    def test_all_returns_plugins(self):
        reg = PluginRegistry()
        d = _DummyPlugin()
        a = _AnotherPlugin()
        reg.register(d)
        reg.register(a)
        assert set(reg.all) == {d, a}

    def test_providers_list(self):
        reg = PluginRegistry()
        reg.register(_DummyPlugin())
        reg.register(_AnotherPlugin())
        assert sorted(reg.providers) == ["another", "dummy"]

    def test_get_pricing_delegates(self):
        reg = PluginRegistry()
        reg.register(_DummyPlugin())
        assert reg.get_pricing("dummy", "known") == {"cache_hit": 0.01, "cache_miss": 0.50, "output": 1.00}
        assert reg.get_pricing("dummy", "unknown") is None
        assert reg.get_pricing("nonexistent", "known") is None

    def test_calculate_cost_delegates(self):
        reg = PluginRegistry()
        reg.register(_DummyPlugin())
        assert reg.calculate_cost("dummy", "known", {}) == 0.42
        assert reg.calculate_cost("dummy", "unknown", {}) is None
        assert reg.calculate_cost("nonexistent", "known", {}) is None

    def test_fetch_all_usage(self):
        reg = PluginRegistry()
        reg.register(_AnotherPlugin())
        reg.register(_DummyPlugin())
        result = reg.fetch_all_usage()
        assert "another" in result
        assert result["another"] == [{"date": "2026-01-01", "provider": "another", "cost": 5.0}]
        # Dummy hasn't overridden fetch_usage, so it should not appear
        assert "dummy" not in result

    def test_fetch_all_usage_empty_registry(self):
        reg = PluginRegistry()
        assert reg.fetch_all_usage() == {}

    def test_fetch_all_usage_with_date_params(self):
        reg = PluginRegistry()
        a = _AnotherPlugin()
        reg.register(a)
        result = reg.fetch_all_usage(start_date="2026-01-01")
        assert "another" in result
        result2 = reg.fetch_all_usage(end_date="2026-01-01")
        assert "another" in result2

    def test_presets_empty_without_registered_plugins(self):
        reg = PluginRegistry()
        assert reg.presets == {}

    def test_presets_only_non_none(self):
        """The presets property should only include plugins whose preset is not None."""
        reg = PluginRegistry()
        d = _DummyPlugin()
        a = _AnotherPlugin()
        reg.register(d)
        reg.register(a)
        presets = reg.presets
        # _DummyPlugin and _AnotherPlugin don't override preset, so they return None
        assert presets == {}

    def test_fetch_all_balances(self):
        reg = PluginRegistry()
        reg.register(_DummyPlugin())
        result = reg.fetch_all_balances()
        assert result["dummy"] == {"balance": 100.0, "currency": "USD"}

    def test_fetch_all_summaries(self):
        """fetch_all_summaries should collect summaries from all plugins."""
        reg = PluginRegistry()
        d = _DummyPlugin()
        # Override fetch_summary to return custom data
        d.fetch_summary = lambda: {"daily": {"tokens": 1000}}
        a = _AnotherPlugin()
        a.fetch_summary = lambda: {"balance": {"available": 50.0}}
        reg.register(d)
        reg.register(a)
        result = reg.fetch_all_summaries()
        assert result["dummy"] == {"daily": {"tokens": 1000}}
        assert result["another"] == {"balance": {"available": 50.0}}

    def test_fetch_all_subscriptions(self):
        """fetch_all_subscriptions collects snapshot dicts (or None) per plugin."""
        reg = PluginRegistry()
        d = _DummyPlugin()
        d.fetch_subscription = lambda: {"rolling_pct": 25.0}
        a = _AnotherPlugin()  # inherits base fetch_subscription -> None
        reg.register(d)
        reg.register(a)
        result = reg.fetch_all_subscriptions()
        assert result["dummy"] == {"rolling_pct": 25.0}
        assert result["another"] is None


# ═══════════════════════════════════════════════════════════════════════
# Singleton (get_registry / init_plugins)
# ═══════════════════════════════════════════════════════════════════════

class TestSingleton:
    def setup_method(self):
        # Reset the singleton before each test
        import src.api.cost_plugins.base as base_mod
        base_mod._registry = None

    def teardown_method(self):
        # Restore the populated baseline AFTER each test. A plain `_registry =
        # None` leaves later test files an EMPTY registry (the built-ins only
        # auto-register on the package's first import), so get_registry() would
        # miss deepseek/opencode/etc. — a cross-file test-isolation landmine.
        import src.api.cost_plugins.base as base_mod
        from src.api.cost_plugins.commandcode import CommandCodeCostPlugin
        from src.api.cost_plugins.deepseek import DeepSeekCostPlugin
        from src.api.cost_plugins.llamacpp import LlamaCppCostPlugin
        from src.api.cost_plugins.opencode import OpenCodeCostPlugin
        reg = base_mod.PluginRegistry()
        reg.register(DeepSeekCostPlugin())
        reg.register(OpenCodeCostPlugin())
        reg.register(LlamaCppCostPlugin())
        reg.register(CommandCodeCostPlugin())
        base_mod._registry = reg

    def test_get_registry_creates_on_first_call(self):
        reg = get_registry()
        assert isinstance(reg, PluginRegistry)
        # Second call returns same instance
        assert get_registry() is reg

    def test_init_plugins_is_idempotent(self):
        reg1 = init_plugins()
        reg2 = init_plugins()
        assert reg1 is reg2

    def test_init_plugins_with_extras(self):
        reg = init_plugins(extra_plugins=[_DummyPlugin()])
        assert reg.for_provider("dummy") is not None

    def test_init_plugins_does_not_override_existing(self):
        reg1 = init_plugins(extra_plugins=[_DummyPlugin()])
        # Second call with different extra should return same registry, not re-register
        reg2 = init_plugins(extra_plugins=[_AnotherPlugin()])
        assert reg1 is reg2
        # Dummy should still be registered
        assert reg1.for_provider("dummy") is not None

    def test_init_plugins_fills_empty_registry(self):
        """When _registry exists but is empty, init_plugins should fill it."""
        import src.api.cost_plugins.base as base_mod
        base_mod._registry = PluginRegistry()
        reg = init_plugins()
        # Should have found and registered the auto-registered plugins
        assert "deepseek" in reg.providers
        assert "opencode" in reg.providers
        assert "llamacpp" in reg.providers

    def test_init_plugins_injects_engine_into_registered(self):
        """When a registry already has providers, init_plugins(engine=...) calls set_engine."""
        import src.api.cost_plugins.base as base_mod
        from unittest.mock import MagicMock
        reg = PluginRegistry()
        plugin = _MinimalPlugin()
        plugin.set_engine = MagicMock()
        reg.register(plugin)
        base_mod._registry = reg
        try:
            result = init_plugins(engine="fake-engine")
            assert result is reg
            plugin.set_engine.assert_called_once_with("fake-engine")
        finally:
            # Restore the singleton so other test files see a clean registry
            base_mod._registry = None
