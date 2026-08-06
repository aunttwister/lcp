"""Alert manager — webhook notifications and alert history.

Handles configurable alert rules, webhook dispatch, and alert lifecycle
(firing, resolving, acknowledging).
"""

import json
import threading
import time
import urllib.request
import urllib.error
import ssl
from datetime import datetime, timezone

from .logging_config import get_logger
from .models import Alert, get_session

logger = get_logger("lcp.alerts")


class AlertManager:
    """Manages alert rules and webhook dispatch."""

    def __init__(self, engine=None):
        self._engine = engine
        self._config: dict = {
            "webhook_url": "",
            "webhook_secret": "",
            "enabled": False,
            "rules": {
                "budget_breach": {"enabled": True, "min_severity": "warning"},
                "provider_dead": {"enabled": True, "min_severity": "critical"},
                "provider_degraded": {"enabled": True, "min_severity": "warning"},
                "error_spike": {"enabled": False, "min_severity": "warning", "threshold": 10, "window_minutes": 5},
                "circuit_breaker_trip": {"enabled": True, "min_severity": "warning"},
                "circuit_breaker_recovery": {"enabled": True, "min_severity": "info"},
            },
        }
        self._alert_history: list[dict] = []  # in-memory cache (bounded)
        self._active_alerts: dict[str, dict] = {}  # dedup_key -> alert
        self._last_fire: dict[str, float] = {}  # dedup_key -> timestamp for cooldown
        self._cooldown_seconds = 300  # 5 minutes
        # Error spike tracking
        self._error_counts: list[tuple[float, int]] = []  # (timestamp, count)

    @property
    def config(self) -> dict:
        return self._config

    def update_config(self, updates: dict) -> dict:
        """Update alert configuration. Returns new config."""
        if "webhook_url" in updates:
            self._config["webhook_url"] = updates["webhook_url"]
        if "webhook_secret" in updates:
            self._config["webhook_secret"] = updates["webhook_secret"]
        if "enabled" in updates:
            self._config["enabled"] = updates["enabled"]
        if "rules" in updates:
            for rule_name, rule_updates in updates["rules"].items():
                if rule_name in self._config["rules"]:
                    self._config["rules"][rule_name].update(rule_updates)
        if "cooldown_seconds" in updates:
            self._cooldown_seconds = int(updates["cooldown_seconds"])
        logger.info("alert_config_updated")
        return self._config

    def list_alerts(self, limit: int = 50, status: str | None = None) -> list[dict]:
        """List alert history, newest first.

        Reads from DB when an engine is bound; falls back to the in-memory
        cache otherwise (backward compatible with engine-less usage).
        """
        if self._engine is None:
            alerts = self._alert_history
            if status:
                alerts = [a for a in alerts if a.get("status") == status]
            return sorted(alerts, key=lambda a: a.get("timestamp", ""), reverse=True)[:limit]
        try:
            with get_session(self._engine) as session:
                query = session.query(Alert).order_by(Alert.timestamp.desc())
                if status:
                    query = query.filter(Alert.status == status)
                rows = query.limit(limit).all()
                return [_alert_to_dict(r) for r in rows]
        except Exception as e:
            logger.error("alert_list_failed", error=str(e))
            return []

    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an alert by its dedup key."""
        now = datetime.now(timezone.utc).isoformat()
        if alert_id in self._active_alerts:
            self._active_alerts[alert_id]["acknowledged"] = True
            self._active_alerts[alert_id]["acknowledged_at"] = now
        if self._engine is not None:
            try:
                with get_session(self._engine) as session:
                    alert = session.query(Alert).filter(Alert.dedup_key == alert_id).first()
                    if alert:
                        alert.acknowledged = 1
                        alert.acknowledged_at = now
                        session.commit()
                        return True
            except Exception as e:
                logger.error("alert_acknowledge_failed", error=str(e))
                return False
        return alert_id in self._active_alerts

    # ── Alert Firing ──────────────────────────────────────────────────────

    def fire(
        self,
        rule: str,
        severity: str,
        title: str,
        message: str,
        dedup_key: str = "",
        metadata: dict | None = None,
    ) -> dict | None:
        """Fire an alert if the rule is enabled and severity meets threshold.
        Returns the alert dict if fired, None if suppressed."""
        rule_cfg = self._config["rules"].get(rule, {})
        if not rule_cfg.get("enabled", True):
            return None

        # Check severity threshold
        severities = {"info": 0, "warning": 1, "critical": 2}
        min_sev = severities.get(rule_cfg.get("min_severity", "info"), 0)
        current_sev = severities.get(severity, 0)
        if current_sev < min_sev:
            return None

        # Cooldown check
        if dedup_key:
            now = time.time()
            last = self._last_fire.get(dedup_key, 0)
            if now - last < self._cooldown_seconds:
                logger.debug("alert_suppressed", rule=rule, dedup_key=dedup_key,
                             cooldown_remaining=int(self._cooldown_seconds - (now - last)))
                return None
            self._last_fire[dedup_key] = now

        timestamp = datetime.now(timezone.utc).isoformat()
        alert = {
            "dedup_key": dedup_key or f"{rule}:{title}:{int(time.time())}",
            "rule": rule,
            "severity": severity,
            "title": title,
            "message": message,
            "metadata": metadata or {},
            "status": "firing",
            "acknowledged": False,
            "timestamp": timestamp,
        }

        self._alert_history.append(alert)
        self._active_alerts[alert["dedup_key"]] = alert

        # Persist to DB
        self._persist_alert(alert)

        # Keep history bounded (in-memory cap only — DB handles its own)
        if len(self._alert_history) > 500:
            self._alert_history = self._alert_history[-500:]

        logger.info("alert_fired", rule=rule, severity=severity, title=title)
        self._dispatch_webhook(alert)
        return alert

    def resolve(self, dedup_key: str) -> bool:
        """Resolve an active alert."""
        now = datetime.now(timezone.utc).isoformat()
        found = dedup_key in self._active_alerts
        if found:
            alert = self._active_alerts.pop(dedup_key)
            alert["status"] = "resolved"
            alert["resolved_at"] = now
        # Update in DB
        if self._engine is not None:
            try:
                with get_session(self._engine) as session:
                    db_alert = session.query(Alert).filter(Alert.dedup_key == dedup_key).first()
                    if db_alert:
                        db_alert.status = "resolved"
                        db_alert.resolved_at = now
                        session.commit()
                        logger.info("alert_resolved", dedup_key=dedup_key)
                        self._dispatch_webhook(
                            {k: v for k, v in db_alert.__dict__.items() if not k.startswith("_")}
                        )
                        return True
                    # Not found in DB — fall back to in-memory result
                    return found
            except Exception as e:
                logger.error("alert_resolve_failed", error=str(e))
                return False
        return found

    def _persist_alert(self, alert: dict) -> None:
        """Write alert to SQLite. Non-blocking — failures are logged not raised."""
        if self._engine is None:
            return
        try:
            with get_session(self._engine) as session:
                db_alert = Alert(
                    timestamp=alert["timestamp"],
                    dedup_key=alert["dedup_key"],
                    rule=alert["rule"],
                    severity=alert["severity"],
                    title=alert["title"],
                    message=alert["message"],
                    metadata_json=json.dumps(alert.get("metadata", {})),
                    status=alert["status"],
                    acknowledged=1 if alert.get("acknowledged") else 0,
                )
                session.add(db_alert)
                session.commit()
        except Exception as e:
            logger.error("alert_persist_failed", error=str(e), dedup_key=alert.get("dedup_key"))

    def get_active_alerts(self) -> list[dict]:
        """Get currently active (firing) alerts."""
        return list(self._active_alerts.values())

    # ── Budget Breach Alert Helper ────────────────────────────────────────

    def fire_budget_breach(self, budget_name: str, threshold: int, spend_pct: float, budget_id: int) -> None:
        """Convenience method for budget threshold alerts."""
        severity = "critical" if spend_pct >= 100 else "warning" if spend_pct >= 80 else "info"
        self.fire(
            rule="budget_breach",
            severity=severity,
            title=f"Budget '{budget_name}' at {spend_pct:.1f}%",
            message=f"Budget '{budget_name}' has reached {spend_pct:.1f}% of its limit "
                    f"(threshold: {threshold}%). Current spend: ${spend_pct * 0.01 * 0:.2f}.",
            dedup_key=f"budget:{budget_id}:t{threshold}",
            metadata={"budget_id": budget_id, "threshold": threshold, "spend_pct": spend_pct},
        )

    # ── Provider Alert Helpers ────────────────────────────────────────────

    def fire_provider_status(self, provider: str, profile: str, old_status: str, new_status: str) -> None:
        """Alert on provider status changes."""
        if new_status == "dead":
            rule = "provider_dead"
            severity = "critical"
        elif new_status == "degraded":
            rule = "provider_degraded"
            severity = "warning"
        else:
            rule = "circuit_breaker_recovery"
            severity = "info"

        self.fire(
            rule=rule,
            severity=severity,
            title=f"Provider {provider}/{profile}: {old_status} → {new_status}",
            message=f"Provider '{provider}' in profile '{profile}' changed status "
                    f"from '{old_status}' to '{new_status}'.",
            dedup_key=f"provider:{provider}:{profile}:status",
            metadata={"provider": provider, "profile": profile, "old_status": old_status, "new_status": new_status},
        )

    def track_error(self) -> None:
        """Track an error occurrence for error spike detection."""
        now = time.time()
        self._error_counts.append((now, 1))
        # Prune old entries
        window = self._config["rules"]["error_spike"].get("window_minutes", 5) * 60
        cutoff = now - window
        self._error_counts = [(t, c) for t, c in self._error_counts if t > cutoff]

        total = sum(c for _, c in self._error_counts)
        threshold = self._config["rules"]["error_spike"].get("threshold", 10)
        if total >= threshold:
            logger.warning(
                "error_spike_detected",
                error_count=total,
                threshold=threshold,
                window_minutes=window // 60,
            )
            self.fire(
                rule="error_spike",
                severity="warning",
                title=f"Error spike: {total} errors in {window // 60}min",
                message=f"Detected {total} errors in the last {window // 60} minutes "
                        f"(threshold: {threshold}).",
                dedup_key="error_spike",
                metadata={"error_count": total, "window_minutes": window // 60},
            )

    # ── Webhook Dispatch ──────────────────────────────────────────────────

    def _dispatch_webhook(self, alert: dict) -> None:
        """Send alert as webhook POST in a background thread."""
        if not self._config.get("enabled"):
            return
        url = self._config.get("webhook_url", "").strip()
        if not url:
            return

        secret = self._config.get("webhook_secret", "")
        payload = json.dumps({
            "event": "alert",
            "alert": alert,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        def _send():
            try:
                req = urllib.request.Request(
                    url,
                    data=payload.encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "X-LCP-Webhook-Secret": secret,
                        "User-Agent": "LCP-AlertManager/1.0",
                    },
                )
                ctx = ssl.create_default_context()
                with urllib.request.urlopen(req, timeout=5, context=ctx) as resp:
                    logger.info("webhook_sent", status=resp.status, dedup_key=alert.get("dedup_key"))
            except Exception as e:
                logger.error("webhook_failed", error=str(e), dedup_key=alert.get("dedup_key"))

        t = threading.Thread(target=_send, daemon=True)
        t.start()

    def test_webhook(self) -> dict:
        """Send a test webhook to verify configuration."""
        test_alert = {
            "dedup_key": "test",
            "rule": "test",
            "severity": "info",
            "title": "Test Alert",
            "message": "This is a test alert from LCP webhook.",
            "status": "firing",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._dispatch_webhook(test_alert)
        return {"ok": True, "webhook_url": self._config.get("webhook_url", "")}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _alert_to_dict(alert: "Alert") -> dict:
    """Convert an Alert ORM object to a dict matching the legacy in-memory format."""
    return {
        "id": alert.id,
        "dedup_key": alert.dedup_key,
        "rule": alert.rule,
        "severity": alert.severity,
        "title": alert.title,
        "message": alert.message,
        "metadata": json.loads(alert.metadata_json) if alert.metadata_json else {},
        "status": alert.status,
        "acknowledged": bool(alert.acknowledged),
        "acknowledged_at": alert.acknowledged_at,
        "resolved_at": alert.resolved_at,
        "timestamp": alert.timestamp,
    }


# ── Module-level singleton ────────────────────────────────────────────────
_alert_manager: AlertManager | None = None


def get_alert_manager(engine=None) -> AlertManager:
    """Get or create the alert manager singleton."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager(engine)
    elif engine is not None and _alert_manager._engine is None:
        _alert_manager._engine = engine
    return _alert_manager


def init_alert_manager(engine) -> AlertManager:
    """Force-initialize the alert manager with an engine."""
    global _alert_manager
    _alert_manager = AlertManager(engine)
    return _alert_manager
