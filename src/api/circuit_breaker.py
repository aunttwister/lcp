"""Circuit breaker — provider health tracking and failure management.

Providers are monitored per (provider_name, base_url, profile) tuple.
Success/failure is recorded, and providers are automatically marked
healthy → degraded → dead based on consecutive failure thresholds.
"""

import time
from datetime import datetime, timezone

from .logging_config import get_logger

logger = get_logger("lcp.circuit_breaker")

# Consecutive-failure weight per error type. Auth failures are permanent
# (a rejected key won't self-heal) so they should trip the breaker faster.
_ERROR_WEIGHTS = {
    "ProviderAuthError": 3,
    "ProviderTimeoutError": 1,
    "ProviderRateLimitError": 1,
    "ProviderInternalError": 1,
    "ProviderBadRequestError": 1,
}


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

    def status_of(self, provider: str, base_url: str, profile: str) -> str:
        """Return the current status string ('healthy'|'degraded'|'dead')."""
        return self.get_health(provider, base_url, profile)["status"]

    def is_available(self, provider: str, base_url: str, profile: str) -> bool:
        """Check if provider is available (not tripped).

        Also implements the half-open ladder: when a cooldown expires the
        provider is promoted one level (dead → degraded → healthy) instead of
        jumping straight back to healthy. A single success during degraded
        promotes to healthy; a single failure re-trips to dead.
        """
        h = self.get_health(provider, base_url, profile)
        if h["status"] == "healthy":
            return True
        if h["tripped_until"] is not None and time.time() >= h["tripped_until"]:
            # Cooldown expired — promote one level and allow a probe request.
            if h["status"] == "dead":
                self._promote(h, "degraded")   # leaves tripped_until = None
            else:  # degraded → healthy after a quiet cooldown
                self._promote(h, "healthy")
            return True
        if h["status"] == "degraded" and h["tripped_until"] is None:
            # A promoted probe provider is available for its probe request.
            return True
        return False

    def _promote(self, h: dict, new_status: str) -> None:
        """Promote a provider one step up the health ladder (half-open probe)."""
        old_status = h["status"]
        h["status"] = new_status
        h["tripped_until"] = None
        logger.info(
            "circuit_breaker_probe",
            old_status=old_status,
            new_status=new_status,
        )

    def record_success(self, provider: str, base_url: str, profile: str) -> None:
        """Record a successful request — resets failure count."""
        h = self.get_health(provider, base_url, profile)
        old_status = h["status"]
        h["status"] = "healthy"
        h["consecutive_failures"] = 0
        h["last_success"] = datetime.now(timezone.utc).isoformat()
        h["tripped_until"] = None
        if old_status != "healthy":
            logger.info(
                "circuit_breaker_recovered",
                provider=provider,
                base_url=base_url,
                profile=profile,
                old_status=old_status,
                new_status="healthy",
            )

    def record_failure(self, provider: str, base_url: str, profile: str,
                       error_type: str | None = None) -> None:
        """Record a failed request — may trip circuit breaker.

        ``error_type`` is the exception class name (e.g. 'ProviderAuthError').
        It maps to a failure weight: permanent failures (auth) trip the breaker
        faster than transient ones.
        """
        cb_cfg = self._config.circuit_breaker
        h = self.get_health(provider, base_url, profile)
        old_status = h["status"]
        weight = _ERROR_WEIGHTS.get(error_type or "", 1)
        h["consecutive_failures"] += weight
        h["last_failure"] = datetime.now(timezone.utc).isoformat()
        n = h["consecutive_failures"]
        new_status = old_status
        if n >= cb_cfg["failures_dead"]:
            # A dead provider whose cooldown expired is promoted to 'degraded'
            # for a probe; its failure count is already ≥ threshold, so a single
            # probe failure naturally re-trips it to dead.
            h["status"] = "dead"
            h["tripped_until"] = time.time() + cb_cfg["dead_cooldown_seconds"]
            new_status = "dead"
        elif n >= cb_cfg["failures_degraded"]:
            h["status"] = "degraded"
            h["tripped_until"] = time.time() + cb_cfg["degraded_cooldown_seconds"]
            new_status = "degraded"
        if new_status != old_status:
            level = "error" if new_status == "dead" else "warning"
            logger_method = logger.error if level == "error" else logger.warning
            logger_method(
                "circuit_breaker_tripped",
                provider=provider,
                base_url=base_url,
                profile=profile,
                old_status=old_status,
                new_status=new_status,
                consecutive_failures=n,
                error_type=error_type,
            )

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
