"""Phase C tests: alert_manager as a runtime component."""

import pytest

from src.api.runtime import Runtime


@pytest.fixture(autouse=True)
def _reset_am_globals():
    import src.api.alert_manager as am
    import src.api.runtime as runtime
    yield
    runtime._active_runtime = None
    am._alert_manager = None


@pytest.fixture
def rt():
    import src.api.alert_manager as am
    from src.api.alert_manager import AlertManagerComponent
    rt = Runtime(engine=object())
    rt.register(AlertManagerComponent())
    rt.start()
    am.bind_runtime(rt)
    return rt


def test_component_registers_and_provides(rt):
    assert rt.is_active("alert_manager") is True
    assert rt.resolve("alert_manager").manager is not None


def test_component_injects_engine(rt):
    assert rt.resolve("alert_manager").manager._engine is not None


def test_facade_delegates_when_bound(rt):
    from src.api.alert_manager import get_alert_manager
    assert get_alert_manager() is rt.resolve("alert_manager").manager


def test_facade_falls_back_when_unbound(monkeypatch):
    import src.api.alert_manager as am
    from src.api.alert_manager import get_alert_manager
    monkeypatch.setattr("src.api.runtime._active_runtime", None)
    am._alert_manager = None
    m = get_alert_manager()
    assert m is not None
    assert get_alert_manager() is m
