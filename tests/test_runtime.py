"""Tests for the declarative component runtime (src/api/runtime.py)."""

import pytest

from src.api.runtime import Runtime, UndeclaredDependency


class _Recorder:
    """Shared call log so tests can assert order across components."""

    def __init__(self):
        self.calls: list[str] = []


class _Comp:
    def __init__(self, name, requires=(), provides=(), recorder=None, disposer=None):
        self.name = name
        self.requires = list(requires)
        self.provides = list(provides)
        self.recorder = recorder
        self.disposer = disposer
        self.setup_calls = 0
        self.changed: list[str] = []

    def setup(self, rt):
        self.setup_calls += 1
        if self.recorder is not None:
            self.recorder.calls.append(f"setup:{self.name}")
        return self.disposer

    def on_dependency_change(self, key):
        self.changed.append(key)
        if self.recorder is not None:
            self.recorder.calls.append(f"changed:{self.name}:{key}")


def test_runtime_resolves_root_keys():
    cfg = object()
    engine = object()
    rt = Runtime(config=cfg, engine=engine, data_dir="/data")
    assert rt.resolve("config") is cfg
    assert rt.resolve("engine") is engine
    assert rt.resolve("data_dir") == "/data"


def test_runtime_register_collision_name():
    rt = Runtime()
    rt.register(_Comp("a"))
    with pytest.raises(ValueError):
        rt.register(_Comp("a"))


def test_runtime_register_collision_provides():
    rt = Runtime()
    rt.register(_Comp("a", provides=["x"]))
    with pytest.raises(ValueError):
        rt.register(_Comp("b", provides=["x"]))


def test_runtime_register_reserved_root_key():
    rt = Runtime()
    with pytest.raises(ValueError):
        rt.register(_Comp("config"))
    with pytest.raises(ValueError):
        rt.register(_Comp("c", provides=["engine"]))


def test_runtime_resolve_undeclared_raises():
    rt = Runtime()
    with pytest.raises(UndeclaredDependency):
        rt.resolve("nope")


def test_runtime_resolve_inactive_raises():
    rt = Runtime()
    rt.register(_Comp("a", requires=["missing"]))
    rt.start()
    with pytest.raises(UndeclaredDependency):
        rt.resolve("a")


def test_runtime_topo_order():
    rec = _Recorder()
    rt = Runtime()
    rt.register(_Comp("breaker", requires=["config", "engine"], recorder=rec))
    rt.register(_Comp("plugins", requires=["engine"], recorder=rec))
    rt.register(_Comp("router", requires=["engine"], recorder=rec))
    rt.start()
    assert rec.calls == ["setup:breaker", "setup:plugins", "setup:router"]
    assert rt.resolve("breaker").name == "breaker"


def test_runtime_unsatisfied_dep_marked_inactive():
    rt = Runtime()
    rt.register(_Comp("a", requires=["missing"]))
    rt.register(_Comp("b", requires=["engine"]))
    rt.start()
    assert rt.is_active("a") is False
    assert rt.is_active("b") is True


def test_runtime_cycle_leaves_inactive_not_hang():
    rt = Runtime()
    rt.register(_Comp("a", requires=["b"]))
    rt.register(_Comp("b", requires=["a"]))
    rt.start()  # must terminate, not hang
    assert rt.is_active("a") is False
    assert rt.is_active("b") is False


def test_runtime_shutdown_replays_disposers_lifo():
    rec = _Recorder()
    rt = Runtime()
    rt.register(_Comp("a", recorder=rec, disposer=lambda: rec.calls.append("dispose:a")))
    rt.register(_Comp("b", recorder=rec, disposer=lambda: rec.calls.append("dispose:b")))
    rt.start()
    rec.calls.clear()
    rt.shutdown()
    # LIFO: reverse of setup order.
    assert rec.calls == ["dispose:b", "dispose:a"]


def test_runtime_shutdown_without_disposers_is_safe():
    rt = Runtime()
    rt.register(_Comp("a"))
    rt.start()
    rt.shutdown()
    assert rt.is_active("a") is False


def test_runtime_reload_disposes_resetups_and_notifies():
    rec = _Recorder()
    rt = Runtime()
    rt.register(_Comp("breaker", recorder=rec, disposer=lambda: rec.calls.append("dispose:breaker")))
    rt.register(_Comp("consumer", requires=["breaker"], recorder=rec))
    rt.start()
    rec.calls.clear()
    rt.reload("breaker")
    # dispose, re-setup, then notify the dependent.
    assert rec.calls == ["dispose:breaker", "setup:breaker", "changed:consumer:breaker"]


def test_runtime_reload_unknown_raises():
    rt = Runtime()
    with pytest.raises(KeyError):
        rt.reload("nope")


def test_runtime_start_twice_raises():
    rt = Runtime()
    rt.register(_Comp("a"))
    rt.start()
    with pytest.raises(RuntimeError):
        rt.start()


def test_runtime_setup_failure_marks_inactive_continues():
    class _Boom(_Comp):
        def setup(self, rt):
            raise RuntimeError("boom")

    rt = Runtime()
    rt.register(_Boom("a"))
    rt.register(_Comp("b", requires=["engine"]))
    rt.start()  # must not raise; 'a' inactive, 'b' active
    assert rt.is_active("a") is False
    assert rt.is_active("b") is True


# ═══════════════════════════════════════════════════════════════════════════
# Active-runtime accessor (Phase F) — get_runtime() / resolve_service()
# ═══════════════════════════════════════════════════════════════════════════

class _ServiceComp:
    """Stub component that publishes a service via a ``service`` property."""

    def __init__(self, name, requires=(), provides=(), service=None):
        self.name = name
        self.requires = list(requires)
        self.provides = list(provides)
        self.service = service

    def setup(self, rt):
        return None


def test_get_runtime_unbound_returns_none():
    from src.api.runtime import get_runtime
    assert get_runtime() is None


def test_bind_active_runtime_records_single_runtime():
    from src.api.runtime import bind_active_runtime, get_runtime
    rt = Runtime()
    assert get_runtime() is None
    bind_active_runtime(rt)
    assert get_runtime() is rt


def test_resolve_service_returns_service_from_runtime():
    from src.api.runtime import bind_active_runtime, resolve_service
    svc = object()
    rt = Runtime()
    rt.register(_ServiceComp("a", provides=["thing"], service=svc))
    rt.start()
    bind_active_runtime(rt)
    assert resolve_service("thing") is svc


def test_resolve_service_falls_back_when_unbound():
    from src.api.runtime import resolve_service
    calls = []

    def _fallback():
        calls.append(1)
        return "legacy"

    assert resolve_service("thing", fallback=_fallback) == "legacy"
    assert calls == [1]


def test_resolve_service_falls_back_when_inactive():
    from src.api.runtime import bind_active_runtime, resolve_service
    rt = Runtime()
    rt.register(_ServiceComp("a", requires=["missing"], provides=["thing"],
                             service=object()))
    rt.start()  # 'a' is inactive (unsatisfied dep)
    bind_active_runtime(rt)
    assert resolve_service("thing", fallback=lambda: "legacy") == "legacy"


def test_resolve_service_returns_fallback_value_as_is():
    from src.api.runtime import resolve_service
    sentinel = object()
    assert resolve_service("nope", fallback=sentinel) is sentinel
