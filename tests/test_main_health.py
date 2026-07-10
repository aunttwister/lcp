"""Tests for main.py provider health functions."""
import time
import pytest
import sys
sys.path.insert(0, "/opt/lcp")
from unittest.mock import MagicMock

from src.main import (
    _health_key,
    _get_health,
    is_provider_available,
    record_provider_success,
    record_provider_failure,
)


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.circuit_breaker = {
        "failures_dead": 5,
        "dead_cooldown_seconds": 300,
        "failures_degraded": 3,
        "degraded_cooldown_seconds": 60,
    }
    return cfg


@pytest.fixture(autouse=True)
def reset_health():
    """Reset provider health between tests."""
    import src.main
    src.main._provider_health.clear()


class TestGetHealth:
    def test_creates_new_entry(self, mock_config):
        h = _get_health("deepseek", "url1", "l2", mock_config)
        assert h["status"] == "healthy"
        assert h["consecutive_failures"] == 0
        assert h["last_failure"] is None

    def test_returns_existing_entry(self, mock_config):
        h1 = _get_health("deepseek", "url1", "l2", mock_config)
        h1["consecutive_failures"] = 5
        h2 = _get_health("deepseek", "url1", "l2", mock_config)
        assert h2["consecutive_failures"] == 5

    def test_different_keys_independent(self, mock_config):
        h1 = _get_health("deepseek", "url1", "l2", mock_config)
        h2 = _get_health("openai", "url1", "l2", mock_config)
        assert h1 is not h2


class TestIsProviderAvailable:
    def test_healthy_provider(self, mock_config):
        h = _get_health("deepseek", "url1", "l2", mock_config)
        h["status"] = "healthy"
        assert is_provider_available("deepseek", "url1", "l2", mock_config)

    def test_degraded_but_cooldown_over(self, mock_config):
        h = _get_health("deepseek", "url1", "l2", mock_config)
        h["status"] = "degraded"
        h["tripped_until"] = time.time() - 10
        assert is_provider_available("deepseek", "url1", "l2", mock_config)

    def test_still_degraded(self, mock_config):
        h = _get_health("deepseek", "url1", "l2", mock_config)
        h["status"] = "degraded"
        h["tripped_until"] = time.time() + 3600
        assert not is_provider_available("deepseek", "url1", "l2", mock_config)


class TestRecordProviderSuccess:
    def test_resets_health(self, mock_config):
        h = _get_health("deepseek", "url1", "l2", mock_config)
        h["consecutive_failures"] = 3
        h["status"] = "degraded"
        h["tripped_until"] = "2026-01-01"
        record_provider_success("deepseek", "url1", "l2", mock_config)
        assert h["status"] == "healthy"
        assert h["consecutive_failures"] == 0
        assert h["tripped_until"] is None
        assert h["last_success"] is not None


class TestRecordProviderFailure:
    def test_increments_failures(self, mock_config):
        h = _get_health("deepseek", "url1", "l2", mock_config)
        record_provider_failure("deepseek", "url1", "l2", mock_config)
        assert h["consecutive_failures"] == 1
        assert h["last_failure"] is not None

    def test_degraded_after_threshold(self, mock_config):
        h = _get_health("deepseek", "url1", "l2", mock_config)
        h["consecutive_failures"] = 2  # threshold is 3
        record_provider_failure("deepseek", "url1", "l2", mock_config)
        assert h["status"] == "degraded"
        assert h["tripped_until"] is not None
        assert h["consecutive_failures"] == 3

    def test_dead_after_dead_threshold(self, mock_config):
        h = _get_health("deepseek", "url1", "l2", mock_config)
        h["consecutive_failures"] = 4  # dead threshold is 5
        record_provider_failure("deepseek", "url1", "l2", mock_config)
        assert h["status"] == "dead"
        assert h["consecutive_failures"] == 5
