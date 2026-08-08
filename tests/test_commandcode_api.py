"""Tests for the Command Code billing API client (commandcode_api.py)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.api.cost_plugins.commandcode_api import (
    CommandCodeSubscriptionSnapshot,
    fetch_subscription_snapshot,
    fetch_subscription_snapshot_dict,
    _parse_credits,
    _parse_subscription,
)


def _credits_payload(**overrides):
    """A realistic /internal/billing/credits response."""
    payload = {
        "credits": {
            "belowThreshold": False,
            "creditThreshold": 0,
            "monthlyCredits": 8.7784,
            "purchasedCredits": 0.0,
            "premiumMonthlyCredits": 0.0,
            "opensourceMonthlyCredits": 8.7784,
            "fiveHourLimit": {"used": 32, "limit": 100, "resetInSeconds": 11520},
            "weeklyLimit": {"used": 41, "limit": 100, "resetInSeconds": 172800},
        }
    }
    return payload


def _subscription_payload(**overrides):
    payload = {
        "success": True,
        "data": {
            "id": "sub_1TTzt3DSZgxV3MJKG4ClCWpn",
            "status": "active",
            "planID": "go",
            "currentPeriodEnd": "2026-09-08T00:00:00Z",
        },
    }
    payload["data"].update(overrides)
    return payload


class TestParseCredits:
    def test_full_credits(self):
        result = _parse_credits(_credits_payload())
        assert result["monthly_credits_remaining"] == 8.7784
        assert result["purchased_credits"] == 0.0
        assert result["premium_monthly_credits"] == 0.0
        assert result["opensource_monthly_credits"] == 8.7784
        assert result["five_hour_pct"] == 32.0
        assert result["weekly_pct"] == 41.0
        assert result["five_hour_reset_sec"] == 11520
        assert result["weekly_reset_sec"] == 172800

    def test_zero_credits(self):
        payload = {"credits": {
            "monthlyCredits": 0, "purchasedCredits": 0,
            "premiumMonthlyCredits": 0, "opensourceMonthlyCredits": 0,
            "fiveHourLimit": {"used": 0, "limit": 0, "resetInSeconds": 0},
            "weeklyLimit": {"used": 0, "limit": 0, "resetInSeconds": 0},
        }}
        result = _parse_credits(payload)
        assert result["monthly_credits_remaining"] == 0.0
        assert result["five_hour_pct"] == 0.0
        assert result["weekly_pct"] == 0.0

    def test_missing_credits_key(self):
        result = _parse_credits({})
        assert result["monthly_credits_remaining"] == 0.0
        assert result["five_hour_pct"] == 0.0

    def test_partial_windows(self):
        payload = {"credits": {
            "monthlyCredits": 5.0,
            "fiveHourLimit": {"used": 10, "limit": 20, "resetInSeconds": 60},
        }}
        result = _parse_credits(payload)
        assert result["monthly_credits_remaining"] == 5.0
        assert result["five_hour_pct"] == 50.0
        assert result["weekly_pct"] == 0.0

    def test_non_numeric_values(self):
        payload = {"credits": {
            "monthlyCredits": "abc",  # bad
            "fiveHourLimit": {"used": "x", "limit": "y", "resetInSeconds": "z"},
        }}
        result = _parse_credits(payload)
        assert result["monthly_credits_remaining"] == 0.0
        assert result["five_hour_pct"] == 0.0
        assert result["five_hour_reset_sec"] == 0


class TestParseSubscription:
    def test_full_subscription(self):
        plan_id, status, period = _parse_subscription(_subscription_payload())
        assert plan_id == "go"
        assert status == "active"
        assert period == "2026-09-08T00:00:00Z"

    def test_null_data(self):
        plan_id, status, period = _parse_subscription({"success": True, "data": None})
        assert plan_id is None
        assert status is None
        assert period is None

    def test_missing_data_key(self):
        plan_id, status, period = _parse_subscription({"success": True})
        assert plan_id is None
        assert status is None
        assert period is None

    def test_plan_id_variants(self):
        for key in ("planID", "planId", "plan"):
            payload = _subscription_payload()
            # Drop the default planID so only the variant key carries the value
            payload["data"].pop("planID", None)
            payload["data"][key] = "pro"
            plan_id, _, _ = _parse_subscription(payload)
            assert plan_id == "pro"


class TestFetchSubscriptionSnapshot:
    def _mock_json(self, payload):
        """A fake urlopen response returning payload as JSON.

        Must support the ``with urlopen(...) as resp:`` context-manager
        protocol — MagicMock's default ``__enter__`` returns a *different*
        mock, so pin it to return self.
        """
        resp = MagicMock()
        resp.read.return_value = json.dumps(payload).encode()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def test_cookie_missing_returns_none(self):
        assert fetch_subscription_snapshot(None) is None
        assert fetch_subscription_snapshot("") is None
        assert fetch_subscription_snapshot("   ") is None

    def test_full_fetch(self):
        responses = [
            self._mock_json(_credits_payload()),
            self._mock_json(_subscription_payload()),
        ]
        with patch("src.api.cost_plugins.commandcode_api.urlopen", side_effect=responses):
            snap = fetch_subscription_snapshot("session=valid")
        assert isinstance(snap, CommandCodeSubscriptionSnapshot)
        assert snap.monthly_credits_remaining == pytest.approx(8.7784)
        assert snap.five_hour_pct == 32.0
        assert snap.weekly_pct == 41.0
        assert snap.five_hour_reset_sec == 11520
        assert snap.weekly_reset_sec == 172800
        assert snap.plan_id == "go"
        assert snap.plan_status == "active"
        assert snap.billing_period_end == "2026-09-08T00:00:00Z"

    def test_subscription_endpoint_down_keeps_credits(self):
        """Subscription enrichment failure should not break the credits fetch."""
        from urllib.error import HTTPError
        responses = [
            self._mock_json(_credits_payload()),
            HTTPError("url", 503, "Service Unavailable", {}, None),
        ]
        # The HTTPError will be raised from urlopen — patch to raise for 2nd call
        def _fake_urlopen(req, timeout=15):
            call = _fake_urlopen.calls
            _fake_urlopen.calls += 1
            if call == 0:
                return responses[0]
            raise responses[1]
        _fake_urlopen.calls = 0

        with patch("src.api.cost_plugins.commandcode_api.urlopen", side_effect=_fake_urlopen):
            snap = fetch_subscription_snapshot("session=valid")
        assert snap is not None
        assert snap.monthly_credits_remaining == pytest.approx(8.7784)
        assert snap.plan_id is None  # enrichment failed but credits survived

    def test_credits_endpoint_auth_failure(self):
        from urllib.error import HTTPError
        err = HTTPError("url", 401, "Unauthorized", {}, None)
        err.read = MagicMock(return_value=b"{}")
        with patch("src.api.cost_plugins.commandcode_api.urlopen", side_effect=err):
            assert fetch_subscription_snapshot("session=bad") is None

    def test_credits_endpoint_http_error(self):
        from urllib.error import HTTPError
        err = HTTPError("url", 500, "Server Error", {}, None)
        err.read = MagicMock(return_value=b"{}")
        with patch("src.api.cost_plugins.commandcode_api.urlopen", side_effect=err):
            assert fetch_subscription_snapshot("session=valid") is None

    def test_network_error(self):
        from urllib.error import URLError
        with patch("src.api.cost_plugins.commandcode_api.urlopen", side_effect=URLError("no route")):
            assert fetch_subscription_snapshot("session=valid") is None

    def test_parse_failure(self):
        resp = MagicMock()
        resp.read.return_value = b"not json at all"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        with patch("src.api.cost_plugins.commandcode_api.urlopen", return_value=resp):
            assert fetch_subscription_snapshot("session=valid") is None


class TestFetchSubscriptionSnapshotDict:
    def test_returns_dict(self):
        resp = MagicMock()
        resp.read.return_value = json.dumps(_credits_payload()).encode()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        with patch("src.api.cost_plugins.commandcode_api.urlopen", return_value=resp):
            result = fetch_subscription_snapshot_dict("session=valid")
        assert isinstance(result, dict)
        assert result["monthly_credits_remaining"] == pytest.approx(8.7784)
        assert "five_hour_pct" in result
        assert "plan_id" in result

    def test_none_when_fetch_fails(self):
        from urllib.error import URLError
        with patch("src.api.cost_plugins.commandcode_api.urlopen", side_effect=URLError("boom")):
            assert fetch_subscription_snapshot_dict("session=valid") is None

    def test_none_when_cookie_missing(self):
        assert fetch_subscription_snapshot_dict(None) is None
