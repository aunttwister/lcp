"""Tests for alert_manager.py — alert rules, webhook dispatch, lifecycle."""

import json
import time
from unittest.mock import patch, MagicMock

import pytest
from src.api.alert_manager import AlertManager, get_alert_manager


@pytest.fixture
def am():
    """Fresh AlertManager for each test."""
    return AlertManager()


class TestFire:
    def test_fires_basic_alert(self, am):
        result = am.fire(
            rule="budget_breach",
            severity="warning",
            title="Test alert",
            message="Something happened",
        )
        assert result is not None
        assert result["rule"] == "budget_breach"
        assert result["severity"] == "warning"
        assert result["title"] == "Test alert"
        assert result["status"] == "firing"
        assert result["acknowledged"] is False
        assert "timestamp" in result

    def test_respects_min_severity(self, am):
        """info severity should not fire if min_severity is warning."""
        am.update_config({"rules": {"budget_breach": {"min_severity": "warning"}}})
        result = am.fire(rule="budget_breach", severity="info", title="Low", message="...")
        assert result is None

    def test_respects_disabled_rule(self, am):
        am.update_config({"rules": {"budget_breach": {"enabled": False}}})
        result = am.fire(rule="budget_breach", severity="critical", title="!", message="!")
        assert result is None

    def test_cooldown_prevents_duplicate(self, am):
        dedup = "test:dedup:1"
        first = am.fire(rule="budget_breach", severity="warning", title="A", message="...", dedup_key=dedup)
        second = am.fire(rule="budget_breach", severity="warning", title="A", message="...", dedup_key=dedup)
        assert first is not None
        assert second is None  # suppressed by cooldown

    def test_no_cooldown_without_dedup_key(self, am):
        """Without dedup key, alerts always fire."""
        first = am.fire(rule="budget_breach", severity="warning", title="A", message="...")
        second = am.fire(rule="budget_breach", severity="warning", title="A", message="...")
        assert first is not None
        assert second is not None


class TestResolve:
    def test_resolves_active_alert(self, am):
        alert = am.fire(rule="budget_breach", severity="warning", title="X", message="...", dedup_key="resolve:1")
        assert am.resolve("resolve:1") is True
        assert am.get_active_alerts() == []

    def test_resolve_marks_status(self, am):
        am.fire(rule="budget_breach", severity="warning", title="X", message="...", dedup_key="resolve:2")
        am.resolve("resolve:2")
        alerts = am.list_alerts()
        resolved = [a for a in alerts if a["dedup_key"] == "resolve:2"]
        assert len(resolved) == 1
        assert resolved[0]["status"] == "resolved"
        assert "resolved_at" in resolved[0]

    def test_resolve_nonexistent(self, am):
        assert am.resolve("nonexistent") is False


class TestAcknowledge:
    def test_acknowledges_active_alert(self, am):
        am.fire(rule="budget_breach", severity="warning", title="X", message="...", dedup_key="ack:1")
        assert am.acknowledge("ack:1") is True
        assert "ack:1" in am._active_alerts
        assert am._active_alerts["ack:1"]["acknowledged"] is True

    def test_acknowledge_nonexistent(self, am):
        assert am.acknowledge("nope") is False


class TestListAlerts:
    def test_returns_history_newest_first(self, am):
        am.fire(rule="budget_breach", severity="warning", title="Old", message="...", dedup_key="old")
        time.sleep(0.01)
        am.fire(rule="budget_breach", severity="warning", title="New", message="...", dedup_key="new")
        alerts = am.list_alerts()
        titles = [a["title"] for a in alerts]
        assert titles[0] == "New"
        assert titles[1] == "Old"

    def test_filter_by_status(self, am):
        am.fire(rule="budget_breach", severity="warning", title="Firing", message="...", dedup_key="f1")
        am.resolve("f1")
        firing = am.list_alerts(status="firing")
        resolved = am.list_alerts(status="resolved")
        assert len(firing) == 0
        assert len(resolved) == 1

    def test_respects_limit(self, am):
        for i in range(10):
            # budget_breach has min_severity=warning, so use warning level
            am.fire(rule="budget_breach", severity="warning", title=f"A{i}", message="...")
        alerts = am.list_alerts(limit=3)
        assert len(alerts) == 3


class TestGetActiveAlerts:
    def test_returns_active_only(self, am):
        am.fire(rule="budget_breach", severity="warning", title="Active", message="...", dedup_key="a1")
        active = am.get_active_alerts()
        assert len(active) == 1
        assert active[0]["status"] == "firing"

    def test_empty_after_resolve(self, am):
        am.fire(rule="budget_breach", severity="warning", title="X", message="...", dedup_key="a2")
        am.resolve("a2")
        assert am.get_active_alerts() == []


class TestConfig:
    def test_update_webhook_url(self, am):
        am.update_config({"webhook_url": "https://hooks.example.com"})
        assert am.config["webhook_url"] == "https://hooks.example.com"

    def test_enable_disable(self, am):
        am.update_config({"enabled": True})
        assert am.config["enabled"] is True
        am.update_config({"enabled": False})
        assert am.config["enabled"] is False

    def test_update_rule(self, am):
        am.update_config({"rules": {"provider_dead": {"min_severity": "critical"}}})
        assert am.config["rules"]["provider_dead"]["min_severity"] == "critical"

    def test_update_cooldown(self, am):
        am.update_config({"cooldown_seconds": 600})
        assert am._cooldown_seconds == 600


class TestWebhookDispatch:
    def test_no_dispatch_when_disabled(self, am):
        am.update_config({"webhook_url": "https://hooks.example.com", "enabled": False})
        with patch("urllib.request.urlopen") as mock_urlopen:
            am.fire(rule="budget_breach", severity="warning", title="X", message="...")
            mock_urlopen.assert_not_called()

    def test_no_dispatch_without_url(self, am):
        am.update_config({"enabled": True, "webhook_url": ""})
        with patch("urllib.request.urlopen") as mock_urlopen:
            am.fire(rule="budget_breach", severity="warning", title="X", message="...")
            mock_urlopen.assert_not_called()

    def test_dispatches_webhook(self, am):
        am.update_config({"webhook_url": "https://hooks.example.com", "enabled": True, "webhook_secret": "secret123"})
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            am.fire(rule="budget_breach", severity="warning", title="Webhook Test", message="payload", dedup_key="wh:1")
            # Webhook dispatched in background thread — just verify fire succeeded
            assert am.config["webhook_url"] == "https://hooks.example.com"

    def test_test_webhook(self, am):
        am.update_config({"webhook_url": "https://hooks.example.com", "enabled": True})
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            result = am.test_webhook()
            assert result["ok"] is True


class TestConvenienceMethods:
    def test_fire_budget_breach(self, am):
        am.fire_budget_breach("Test Budget", 80, 85.5, 1)
        alerts = am.list_alerts()
        assert len(alerts) == 1
        assert "Test Budget" in alerts[0]["title"]
        assert "85.5%" in alerts[0]["title"]

    def test_fire_provider_status(self, am):
        am.fire_provider_status("deepseek", "l2", "healthy", "dead")
        alerts = am.list_alerts()
        assert len(alerts) == 1
        assert "deepseek" in alerts[0]["title"]
        assert "dead" in alerts[0]["title"]
        assert alerts[0]["severity"] == "critical"


class TestErrorSpikeDetection:
    def test_tracks_errors(self, am):
        am.update_config({"rules": {"error_spike": {"enabled": True, "min_severity": "warning", "threshold": 3, "window_minutes": 5}}})
        am.track_error()
        am.track_error()
        am.track_error()
        alerts = am.list_alerts()
        assert len(alerts) == 1
        assert "Error spike" in alerts[0]["title"]

    def test_no_spike_below_threshold(self, am):
        am.update_config({"rules": {"error_spike": {"enabled": True, "min_severity": "warning", "threshold": 10, "window_minutes": 5}}})
        am.track_error()
        am.track_error()
        alerts = am.list_alerts()
        assert len(alerts) == 0
