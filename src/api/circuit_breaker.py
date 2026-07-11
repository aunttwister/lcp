"""Circuit breaker — provider health tracking and failure management.

Providers are monitored per (provider_name, base_url, profile) tuple.
Success/failure is recorded, and providers are automatically marked
healthy → degraded → dead based on consecutive failure thresholds.
"""

import time
from datetime import datetime, timezone

from .logging_config import get_logger

logger = get_logger("lcp.circuit_breaker")


class CircuitBreaker:
    """Tracks provider health with configurable thresholds."""

    def __init__(self, config):
        self._config = config
        self._health: dict = {}

    def _key(self, provider: str, base_url: str, profile: str) -> tuple:
        return (provider, base_url, profile)

    def get_health(self, provider: str, base_url: str, profile: str) -> dict:
        """Get or create health entry for a provider+profile."""
        key = self._key(provider, base_url, profile)
        if key not in self._health:
            self._health[key] = {
                "consecutive_failures": 0,
                "last_failure": None,
                "last_success": None,
                "status": "healthy",
                "tripped_until": None,
            }
        return self._health[key]

    def is_available(self, provider: str, base_url: str, profile: str) -> bool:
        """Check if provider is available (not tripped)."""
        h = self.get_health(provider, base_url, profile)
        if h["status"] == "healthy":
            return True
        if h["tripped_until"] and time.time() >= h["tripped_until"]:
            return True
        return False

    def record_success(self, provider: str, base_url: str, profile: str) -> None:
        """Record a successful request — resets failure count."""
        h = self.get_health(provider, base_url, profile)
        h["status"] = "healthy"
        h["consecutive_failures"] = 0
        h["last_success"] = datetime.now(timezone.utc).isoformat()
        h["tripped_until"] = None

    def record_failure(self, provider: str, base_url: str, profile: str) -> None:
        """Record a failed request — may trip circuit breaker."""
        cb_cfg = self._config.circuit_breaker
        h = self.get_health(provider, base_url, profile)
        h["consecutive_failures"] += 1
        h["last_failure"] = datetime.now(timezone.utc).isoformat()
        n = h["consecutive_failures"]
        if n >= cb_cfg["failures_dead"]:
            h["status"] = "dead"
            h["tripped_until"] = time.time() + cb_cfg["dead_cooldown_seconds"]
        elif n >= cb_cfg["failures_degraded"]:
            h["status"] = "degraded"
            h["tripped_until"] = time.time() + cb_cfg["degraded_cooldown_seconds"]

    def get_all_health(self) -> dict:
        """Return all tracked provider health entries keyed by (provider, url, profile)."""
        return dict(self._health)

    @property
    def stats(self) -> dict:
        """Return summary statistics across all tracked providers."""
        return {
            "total": len(self._health),
            "healthy": sum(1 for h in self._health.values() if h["status"] == "healthy"),
            "degraded": sum(1 for h in self._health.values() if h["status"] == "degraded"),
            "dead": sum(1 for h in self._health.values() if h["status"] == "dead"),
        }


# ── Module-level singleton ────────────────────────────────────────────────
_circuit_breaker: CircuitBreaker | None = None


def get_circuit_breaker(config=None) -> CircuitBreaker:
    """Get or create the circuit breaker singleton."""
    global _circuit_breaker
    if _circuit_breaker is None and config is not None:
        _circuit_breaker = CircuitBreaker(config)
    if _circuit_breaker is None:
        raise RuntimeError("CircuitBreaker not initialized — call with config first")
    return _circuit_breaker
