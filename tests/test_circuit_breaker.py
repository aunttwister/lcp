"""Tests for circuit_breaker.py — health checks, state transitions, cooldowns."""

import time
from unittest.mock import MagicMock, patch

import pytest
from src.api.circuit_breaker import CircuitBreaker, get_circuit_breaker


@pytest.fixture
def cb():
    """Fresh CircuitBreaker with mock config."""
    cfg = MagicMock()
    cfg.circuit_breaker = {
        "failures_degraded": 3,
        "failures_dead": 6,
        "degraded_cooldown_seconds": 2,
        "dead_cooldown_seconds": 5,
    }
    return CircuitBreaker(cfg)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the circuit breaker singleton between tests."""
    from src.api.circuit_breaker import _circuit_breaker
    import src.api.circuit_breaker as cb_module
    cb_module._circuit_breaker = None


class TestHealthTracking:
    def test_initial_state_healthy(self, cb):
        h = cb.get_health("test_prov", "https://test/v1", "l2")
        assert h["status"] == "healthy"
        assert h["consecutive_failures"] == 0

    def test_success_resets_failures(self, cb):
        # Record some failures first
        cb.record_failure("test_prov", "https://test/v1", "l2")
        cb.record_failure("test_prov", "https://test/v1", "l2")
        # Then record success
        cb.record_success("test_prov", "https://test/v1", "l2")
        h = cb.get_health("test_prov", "https://test/v1", "l2")
        assert h["consecutive_failures"] == 0
        assert h["status"] == "healthy"

    def test_failure_increments_count(self, cb):
        cb.record_failure("a", "https://x", "p1")
        h = cb.get_health("a", "https://x", "p1")
        assert h["consecutive_failures"] == 1
        assert h["last_failure"] is not None

    def test_different_providers_independent(self, cb):
        cb.record_failure("a", "https://x", "p1")
        cb.record_failure("a", "https://x", "p1")
        h2 = cb.get_health("b", "https://y", "p2")
        assert h2["consecutive_failures"] == 0


class TestDegradedState:
    def test_degraded_after_threshold(self, cb):
        for _ in range(3):
            cb.record_failure("p", "https://x", "l2")
        h = cb.get_health("p", "https://x", "l2")
        assert h["status"] == "degraded"
        assert h["tripped_until"] is not None

    def test_degraded_resets_after_cooldown(self, cb):
        for _ in range(3):
            cb.record_failure("q", "https://x", "l2")
        # Fast-forward time
        with patch("time.time") as mock_time:
            mock_time.return_value = time.time() + 10  # past cooldown
            # Next success should reset
            cb.record_success("q", "https://x", "l2")
            h = cb.get_health("q", "https://x", "l2")
            assert h["status"] == "healthy"


class TestDeadState:
    def test_dead_after_dead_threshold(self, cb):
        for _ in range(6):
            cb.record_failure("d", "https://x", "l2")
        h = cb.get_health("d", "https://x", "l2")
        assert h["status"] == "dead"

    def test_dead_provider_not_available(self, cb):
        for _ in range(6):
            cb.record_failure("dead_prov", "https://x", "l2")
        assert cb.is_available("dead_prov", "https://x", "l2") is False

    def test_healthy_provider_is_available(self, cb):
        assert cb.is_available("healthy", "https://x", "l2") is True


class TestGetAllHealth:
    def test_returns_all_providers(self, cb):
        cb.record_failure("a", "https://x", "l1")
        cb.record_failure("b", "https://y", "l2")
        all_h = cb.get_all_health()
        assert len(all_h) == 2

    def test_empty_initially(self, cb):
        assert cb.get_all_health() == {}


class TestSingleton:
    def test_get_circuit_breaker_requires_config(self):
        from src.api.circuit_breaker import _circuit_breaker
        import src.api.circuit_breaker as cb_module
        cb_module._circuit_breaker = None
        with pytest.raises(RuntimeError, match="not initialized"):
            get_circuit_breaker()

    def test_get_circuit_breaker_returns_same(self):
        cfg = MagicMock()
        cfg.circuit_breaker = {"failures_degraded": 2, "failures_dead": 4, "degraded_cooldown_seconds": 10, "dead_cooldown_seconds": 30}
        cb1 = get_circuit_breaker(cfg)
        cb2 = get_circuit_breaker()
        assert cb1 is cb2
