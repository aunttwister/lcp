"""Phase B pilot tests: cost_plugins as a runtime component."""

import pytest

from src.api.cost_plugins import (
    CostPluginsComponent,
    PluginRegistry,
    bind_runtime,
    get_registry,
)
from src.api.cost_plugins.base import CostPlugin
from src.api.runtime import Runtime, UndeclaredDependency


class _ShutdownRecorderPlugin(CostPlugin):
    """Fake plugin that records its on_shutdown calls."""

    def __init__(self, calls):
        self._calls = calls
        self.provider = "dummy"
        self.on_startup_calls = 0

    @property
    def provider_name(self) -> str:
        return self.provider

    def on_startup(self) -> None:
        self.on_startup_calls += 1

    def on_shutdown(self) -> None:
        self._calls.append("shutdown:dummy")


@pytest.fixture
def rt(monkeypatch):
    """A Runtime with the cost_plugins component started, runtime bound."""
    from src.api import cost_plugins as cp
    import src.api.cost_plugins.base as base_mod
    rt = Runtime(engine=object())
    rt.register(CostPluginsComponent())
    rt.start()
    cp.bind_runtime(rt)
    yield rt
    # Reset the facade global so other tests see the legacy singleton.
    monkeypatch.setattr(base_mod, "_runtime", None)


def test_component_registers_and_provides(rt):
    assert rt.is_active("cost_plugins") is True
    assert rt.resolve("cost_plugins") is rt.resolve("pricing")
    assert rt.resolve("cost_plugins").registry is not None


def test_component_builds_all_builtin_plugins(rt):
    reg = rt.resolve("cost_plugins").registry
    assert isinstance(reg, PluginRegistry)
    for provider in ("deepseek", "opencode", "llamacpp", "commandcode"):
        assert reg.for_provider(provider) is not None


def test_component_injects_engine_at_construction(rt):
    reg = rt.resolve("cost_plugins").registry
    # opencode/commandcode need the engine for DB queries — injected at
    # construction, NOT via the hasattr/set_engine probe.
    assert reg.for_provider("opencode")._engine is not None
    assert reg.for_provider("commandcode")._engine is not None


def test_get_registry_delegates_to_runtime_when_bound(rt):
    assert get_registry() is rt.resolve("cost_plugins").registry


def test_get_registry_falls_back_to_legacy_when_unbound(monkeypatch):
    import src.api.cost_plugins.base as base_mod
    monkeypatch.setattr(base_mod, "_runtime", None)
    base_mod._registry = None
    reg = get_registry()
    assert isinstance(reg, PluginRegistry)
    # No runtime bound → returns the module-level singleton (idempotent).
    assert get_registry() is reg


def test_shutdown_runs_plugin_on_shutdown_disposers(monkeypatch):
    """Runtime.shutdown must run each plugin's on_shutdown (e.g. llamacpp's
    persist) via the component's disposer, in LIFO order."""
    calls = []
    extra = _ShutdownRecorderPlugin(calls)
    rt = Runtime(engine=object())
    rt.register(CostPluginsComponent(extra_plugins=[extra]))
    rt.start()
    assert extra.on_startup_calls == 1
    rt.shutdown()
    assert "shutdown:dummy" in calls


def test_component_active_with_engine_none(monkeypatch):
    """'engine' is a ROOT key — always satisfiable. With no engine supplied,
    the component still starts and plugins operate engine-less (matching the
    legacy behavior of engine=None construction)."""
    rt = Runtime()
    rt.register(CostPluginsComponent())
    rt.start()
    assert rt.is_active("cost_plugins") is True
    reg = rt.resolve("cost_plugins").registry
    assert reg.for_provider("opencode")._engine is None


def test_bind_runtime_cleared_returns_legacy(monkeypatch):
    rt = Runtime(engine=object())
    rt.register(CostPluginsComponent())
    rt.start()
    monkeypatch.setattr("src.api.cost_plugins.base._runtime", None)
    # Legacy singleton used when no runtime bound.
    import src.api.cost_plugins.base as base_mod
    base_mod._registry = None
    assert isinstance(get_registry(), PluginRegistry)
