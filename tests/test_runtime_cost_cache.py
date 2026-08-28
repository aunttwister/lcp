"""Phase C tests: settings/cost_cache/refresher as runtime components."""

import pytest

from src.api.runtime import Runtime


@pytest.fixture(autouse=True)
def _reset_cc_globals():
    """Force-reset the cost_cache module globals after every test so binding
    _runtime here never leaks into other test files."""
    import src.api.cost_cache as cc
    yield
    cc._runtime = None
    cc._settings_store = None
    cc._cost_cache = None
    cc._refresher = None


@pytest.fixture
def rt():
    """A Runtime with settings/cost_cache/refresher components started + bound."""
    import src.api.cost_cache as cc
    from src.api.cost_cache import SettingsComponent, CostCacheComponent, RefresherComponent
    rt = Runtime(engine=object())
    rt.register(SettingsComponent())
    rt.register(CostCacheComponent())
    rt.register(RefresherComponent())
    rt.start()
    cc.bind_runtime(rt)
    return rt


def test_components_register_and_provide(rt):
    assert rt.is_active("settings") is True
    assert rt.is_active("cost_cache") is True
    assert rt.is_active("refresher") is True
    assert rt.resolve("settings").store is not None
    assert rt.resolve("cost_cache").cache is not None
    assert rt.resolve("refresher").refresher is not None


def test_facades_delegate_when_bound(rt):
    from src.api.cost_cache import get_settings, get_cost_cache, get_refresher
    assert get_settings() is rt.resolve("settings").store
    assert get_cost_cache() is rt.resolve("cost_cache").cache
    assert get_refresher() is rt.resolve("refresher").refresher


def test_refresher_disposer_stops_thread(rt):
    from src.api.cost_cache import get_refresher
    refresher = rt.resolve("refresher").refresher
    assert refresher._thread is None or not refresher._thread.is_alive()
    rt.shutdown()  # disposer runs refresher.stop() — must not raise
    assert get_refresher() is None or get_refresher() is not refresher


def test_components_topo_order_engine_first():
    from src.api.cost_cache import SettingsComponent, CostCacheComponent, RefresherComponent
    rt = Runtime(engine=object())
    rt.register(SettingsComponent())
    rt.register(CostCacheComponent())
    rt.register(RefresherComponent())
    rt.start()
    # Refresher depends on cost_cache + settings → must come after them.
    order = rt._order
    assert order.index("settings") < order.index("refresher")
    assert order.index("cost_cache") < order.index("refresher")
