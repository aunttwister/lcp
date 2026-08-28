"""Phase C tests: dynamic_router as a runtime component."""

import pytest

from src.api.runtime import Runtime


@pytest.fixture(autouse=True)
def _reset_router_globals():
    import src.api.runtime as runtime
    yield
    runtime._active_runtime = None


@pytest.fixture
def rt(tmp_path):
    import src.api.router as r
    from src.api.router import RouterComponent
    from src.api.cost_cache import SettingsComponent
    rt = Runtime(engine=object(), data_dir=str(tmp_path))
    rt.register(SettingsComponent())
    rt.register(RouterComponent(db_path=str(tmp_path / "costs.db"), enabled=False))
    rt.start()
    r.bind_runtime(rt)
    return rt


def test_component_registers_and_provides(rt):
    assert rt.is_active("dynamic_router") is True
    router = rt.resolve("dynamic_router").router
    assert router is not None


def test_component_injects_db_path_and_enabled(rt):
    comp = rt.resolve("dynamic_router")
    assert comp.router.db_path.endswith("costs.db")
    assert comp.router.enabled is False


def test_facade_delegates_when_bound(rt):
    from src.api.router import get_dynamic_router
    assert get_dynamic_router() is rt.resolve("dynamic_router").router


def test_facade_falls_back_when_unbound(monkeypatch):
    import src.api.router as r
    from src.api.router import get_dynamic_router
    monkeypatch.setattr("src.api.runtime._active_runtime", None)
    # Legacy eager singleton returns the default router.
    assert get_dynamic_router() is r._dynamic_router
