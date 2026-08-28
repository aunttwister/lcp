"""Phase C tests: circuit_breaker as a runtime component."""

import pytest

from src.api.circuit_breaker import CircuitBreakerComponent, get_circuit_breaker
from src.api.runtime import Runtime


@pytest.fixture(autouse=True)
def _reset_cb_globals():
    """Force-reset the circuit_breaker module globals after EVERY test in this
    module so binding _runtime here can never leak into other test files."""
    import src.api.circuit_breaker as cb_mod
    yield
    cb_mod._runtime = None
    cb_mod._circuit_breaker = None


@pytest.fixture
def rt():
    """A Runtime with the circuit_breaker component started, runtime bound."""
    import src.api.circuit_breaker as cb_mod
    rt = Runtime(config=object(), engine=object())
    rt.register(CircuitBreakerComponent())
    rt.start()
    cb_mod.bind_runtime(rt)
    yield rt


def test_component_registers_and_provides(rt):
    assert rt.is_active("circuit_breaker") is True
    breaker = rt.resolve("circuit_breaker")
    assert isinstance(breaker, CircuitBreakerComponent)
    assert breaker.breaker is not None


def test_component_injects_config_and_engine(rt):
    breaker = rt.resolve("circuit_breaker").breaker
    # config bound at construction; engine attached (no post-hoc attach step).
    assert breaker._config is not None
    assert breaker._engine is not None


def test_get_circuit_breaker_delegates_when_bound(rt):
    assert get_circuit_breaker() is rt.resolve("circuit_breaker").breaker


def test_get_circuit_breaker_falls_back_when_unbound(monkeypatch):
    import src.api.circuit_breaker as cb_mod
    monkeypatch.setattr(cb_mod, "_runtime", None)
    cb_mod._circuit_breaker = None
    b = get_circuit_breaker(config=object())
    assert b is not None
    assert get_circuit_breaker() is b


def test_component_active_with_root_keys_only():
    # config/engine are ROOT keys — with none supplied the breaker still
    # starts (config=None / engine=None, legacy-compatible).
    rt = Runtime()
    rt.register(CircuitBreakerComponent())
    rt.start()
    assert rt.is_active("circuit_breaker") is True
    assert rt.resolve("circuit_breaker").breaker._engine is None
