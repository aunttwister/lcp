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

    def test_available_after_cooldown_expired(self, cb):
        """is_available returns True when tripped_until is in the past (line 45)."""
        for _ in range(6):
            cb.record_failure("p", "https://x", "l2")
        assert cb.is_available("p", "https://x", "l2") is False  # tripped
        # Fast-forward past the cooldown — save real time before mocking
        real_now = time.time()
        with patch("time.time") as mock_time:
            mock_time.return_value = real_now + 60  # well past cooldown
            assert cb.is_available("p", "https://x", "l2") is True  # cooldown expired


class TestStats:
    def test_stats_counts(self, cb):
        cb.record_failure("h", "https://x", "l2")
        cb.record_failure("h", "https://x", "l2")
        cb.record_failure("h", "https://x", "l2")
        cb.record_failure("h", "https://x", "l2")  # still degraded
        s = cb.stats
        assert s["total"] == 1
        assert s["healthy"] == 0
        assert s["degraded"] == 1
        assert s["dead"] == 0

    def test_stats_all_states(self, cb):
        cb.record_failure("h1", "https://x", "l2")
        cb.record_failure("h1", "https://x", "l2")
        cb.record_failure("h1", "https://x", "l2")
        cb.record_failure("h1", "https://x", "l2")  # degraded
        for _ in range(6):
            cb.record_failure("dead-prov", "https://y", "l2")
        # Also have a healthy one
        assert cb.is_available("healthy-prov", "https://z", "l2")
        s = cb.stats
        assert s["total"] == 3
        assert s["healthy"] == 1
        assert s["degraded"] == 1
        assert s["dead"] == 1


class TestGetAllHealth:
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


class TestFailureWeighting:
    """Auth failures (weight 3) trip the breaker faster than transient errors."""

    def test_auth_failures_trip_dead_with_fewer_attempts(self, cb):
        # 2 auth failures × weight 3 = 6 >= failures_dead(6) → dead in 2 calls
        cb.record_failure("w", "https://x", "l2", error_type="ProviderAuthError")
        cb.record_failure("w", "https://x", "l2", error_type="ProviderAuthError")
        assert cb.status_of("w", "https://x", "l2") == "dead"

    def test_transient_failures_need_full_threshold(self, cb):
        # 2 transient failures × weight 1 = 2 < failures_degraded(3) → healthy
        cb.record_failure("w", "https://x", "l2", error_type="ProviderTimeoutError")
        cb.record_failure("w", "https://x", "l2", error_type="ProviderInternalError")
        assert cb.status_of("w", "https://x", "l2") == "healthy"

    def test_mixed_weights_accumulate(self, cb):
        # 1 auth (3) + 1 internal (1) = 4 >= failures_degraded(3) → degraded
        cb.record_failure("w", "https://x", "l2", error_type="ProviderAuthError")
        cb.record_failure("w", "https://x", "l2", error_type="ProviderInternalError")
        assert cb.status_of("w", "https://x", "l2") == "degraded"

    def test_unknown_error_type_uses_weight_one(self, cb):
        cb.record_failure("w", "https://x", "l2", error_type="SomeWeirdError")
        assert cb.status_of("w", "https://x", "l2") == "healthy"
        assert cb.get_health("w", "https://x", "l2")["consecutive_failures"] == 1


class TestHalfOpenLadder:
    """Dead → degraded (probe) → healthy on success, dead again on failure."""

    def _drive_to_dead(self, cb, provider="probe"):
        for _ in range(6):
            cb.record_failure(provider, "https://x", "l2")
        assert cb.status_of(provider, "https://x", "l2") == "dead"

    def test_dead_promotes_to_degraded_after_cooldown(self, cb):
        self._drive_to_dead(cb)
        real_now = time.time()
        with patch("time.time", return_value=real_now + 60):
            assert cb.is_available("probe", "https://x", "l2") is True
        assert cb.status_of("probe", "https://x", "l2") == "degraded"

    def test_degraded_probe_success_promotes_to_healthy(self, cb):
        self._drive_to_dead(cb)
        real_now = time.time()
        with patch("time.time", return_value=real_now + 60):
            cb.is_available("probe", "https://x", "l2")  # → degraded
        cb.record_success("probe", "https://x", "l2")
        assert cb.status_of("probe", "https://x", "l2") == "healthy"
        assert cb.get_health("probe", "https://x", "l2")["consecutive_failures"] == 0

    def test_degraded_probe_failure_retrips_to_dead(self, cb):
        self._drive_to_dead(cb)
        real_now = time.time()
        with patch("time.time", return_value=real_now + 60):
            cb.is_available("probe", "https://x", "l2")  # → degraded
        cb.record_failure("probe", "https://x", "l2", error_type="ProviderInternalError")
        assert cb.status_of("probe", "https://x", "l2") == "dead"

    def test_degraded_probe_remains_available_on_repeated_checks(self, cb):
        """A promoted probe provider stays available across is_available calls."""
        self._drive_to_dead(cb)
        real_now = time.time()
        with patch("time.time", return_value=real_now + 60):
            assert cb.is_available("probe", "https://x", "l2") is True
            assert cb.is_available("probe", "https://x", "l2") is True
        assert cb.status_of("probe", "https://x", "l2") == "degraded"

    def test_degraded_cooldown_expiry_promotes_to_healthy(self, cb):
        # 3 failures → degraded (cooldown 2s). Past cooldown → healthy.
        for _ in range(3):
            cb.record_failure("up", "https://x", "l2")
        assert cb.status_of("up", "https://x", "l2") == "degraded"
        real_now = time.time()
        with patch("time.time", return_value=real_now + 10):
            assert cb.is_available("up", "https://x", "l2") is True
        assert cb.status_of("up", "https://x", "l2") == "healthy"
