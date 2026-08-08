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
    "ProviderCreditsError": 3,  # drained account won't self-heal — trip fast
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
        self._engine = None  # optional DB engine for failover event persistence

    def attach_engine(self, engine) -> None:
        """Attach a SQLAlchemy engine so failover events can be persisted."""
        self._engine = engine

    def _key(self, provider: str, base_url: str, profile: str) -> tuple:
        return (provider, base_url, profile)

    def get_health(self, provider: str, base_url: str, profile: str) -> dict:
        """Get or create health entry for a provider+profile."""
        key = self._key(provider, base_url, profile)
        if key not in self._health:
            self._health[key] = {
                "consecutive_failures": 0,
                "last_failure": None,
                "last_failure_reason": None,
                "last_success": None,
                "status": "healthy",
                "tripped_until": None,
                "manual_override": None,  # None | 'degraded' | 'dead'
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
        h["last_failure_reason"] = None
        h["manual_override"] = None
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
                       error_type: str | None = None,
                       error_reason: str | None = None) -> None:
        """Record a failed request — may trip circuit breaker.

        ``error_type`` is the exception class name (e.g. 'ProviderAuthError').
        It maps to a failure weight: permanent failures (auth) trip the breaker
        faster than transient ones.

        ``error_reason`` is a human-readable description of the error
        (e.g. 'HTTP 503 Service Unavailable') stored for diagnostics.
        """
        cb_cfg = self._config.circuit_breaker
        h = self.get_health(provider, base_url, profile)
        old_status = h["status"]
        weight = _ERROR_WEIGHTS.get(error_type or "", 1)
        h["consecutive_failures"] += weight
        h["last_failure"] = datetime.now(timezone.utc).isoformat()
        if error_reason:
            h["last_failure_reason"] = error_reason
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
                error_reason=error_reason,
            )

    def get_all_health(self) -> dict:
        """Return all tracked provider health entries keyed by (provider, url, profile)."""
        return dict(self._health)

    def reset(self, provider: str, base_url: str, profile: str) -> None:
        """Force-reset a provider back to healthy, clearing failures and cooldown."""
        h = self.get_health(provider, base_url, profile)
        h["status"] = "healthy"
        h["consecutive_failures"] = 0
        h["last_failure"] = None
        h["last_failure_reason"] = None
        h["tripped_until"] = None
        h["manual_override"] = None
        logger.info(
            "circuit_breaker_reset",
            provider=provider,
            base_url=base_url,
            profile=profile,
        )

    def force_status(self, provider: str, base_url: str, profile: str,
                     action: str) -> str:
        """Manually force a provider into a circuit-breaker state.

        ``action`` is one of:
          - 'degrade' — force degraded with the configured degraded cooldown
          - 'kill'    — force dead indefinitely (manual resume required)
          - 'resume'  — force back to healthy, clear failures + cooldown

        Returns the resulting status string.
        """
        h = self.get_health(provider, base_url, profile)
        cb_cfg = self._config.circuit_breaker
        if action == "degrade":
            h["status"] = "degraded"
            h["tripped_until"] = time.time() + cb_cfg["degraded_cooldown_seconds"]
            h["manual_override"] = "degraded"
            logger.info("circuit_breaker_manual_degrade",
                        provider=provider, base_url=base_url, profile=profile,
                        cooldown_seconds=cb_cfg["degraded_cooldown_seconds"])
        elif action == "kill":
            h["status"] = "dead"
            h["tripped_until"] = None  # indefinite — no auto-promotion
            h["manual_override"] = "dead"
            logger.info("circuit_breaker_manual_kill",
                        provider=provider, base_url=base_url, profile=profile)
        elif action == "resume":
            h["status"] = "healthy"
            h["consecutive_failures"] = 0
            h["last_failure"] = None
            h["last_failure_reason"] = None
            h["tripped_until"] = None
            h["manual_override"] = None
            logger.info("circuit_breaker_manual_resume",
                        provider=provider, base_url=base_url, profile=profile)
        else:
            raise ValueError(f"unknown circuit breaker action: {action}")
        return h["status"]

    def record_failover(self, profile: str, from_provider: str, to_provider: str,
                        reason: str, error_message: str | None = None,
                        request_id: int | None = None) -> None:
        """Persist a failover event (chain fallback) to the database.

        Best-effort: failures to persist are logged, never raised, so a DB
        problem can't break request handling.
        """
        if self._engine is None:
            return
        try:
            from .models import FailoverEvent, get_session
            with get_session(self._engine) as session:
                session.add(FailoverEvent(
                    profile=profile,
                    from_provider=from_provider,
                    to_provider=to_provider,
                    reason=reason,
                    error_message=error_message,
                    request_id=request_id,
                ))
                session.commit()
        except Exception as exc:
            logger.warning("failover_persist_failed",
                           profile=profile, from_provider=from_provider,
                           to_provider=to_provider, error=str(exc))

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
