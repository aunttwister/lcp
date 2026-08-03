"""Tests for src/api/cost_plugins/opencode_api.py — SSR parser and HTTP client."""
from unittest.mock import MagicMock, patch

import pytest

from src.api.cost_plugins.opencode_api import (
    SubscriptionSnapshot,
    _base_headers,
    _extract_workspace_ids,
    _parse_ssr_subscription,
    fetch_subscription_dict,
)

# ═══════════════════════════════════════════════════════════════════════
# _base_headers
# ═══════════════════════════════════════════════════════════════════════

class TestBaseHeaders:
    """Tests for _base_headers()."""

    def test_returns_expected_keys(self):
        headers = _base_headers("my-cookie-123")
        assert headers["Cookie"] == "my-cookie-123"
        assert headers["User-Agent"].startswith("Mozilla")
        assert headers["Accept"] == "text/html,application/xhtml+xml"
        assert headers["Origin"] == "https://opencode.ai"
        assert headers["Referer"] == "https://opencode.ai"

    def test_strips_cookie_whitespace(self):
        headers = _base_headers("  cookie-with-spaces  ")
        assert headers["Cookie"] == "cookie-with-spaces"


# ═══════════════════════════════════════════════════════════════════════
# _extract_workspace_ids
# ═══════════════════════════════════════════════════════════════════════

class TestExtractWorkspaceIds:
    """Tests for _extract_workspace_ids()."""

    def test_extracts_single_id(self):
        text = 'id:"wrk_abc123DEF"'
        ids = _extract_workspace_ids(text)
        assert ids == ["wrk_abc123DEF"]

    def test_extracts_multiple_unique_ids(self):
        text = 'id:"wrk_aaa", id : "wrk_bbb", id:"wrk_aaa"'
        ids = _extract_workspace_ids(text)
        assert sorted(ids) == ["wrk_aaa", "wrk_bbb"]

    def test_returns_empty_list_for_no_match(self):
        assert _extract_workspace_ids("no workspace here") == []

    def test_case_insensitive(self):
        text = 'ID:"WRK_UPPERCASE"'
        ids = _extract_workspace_ids(text)
        assert ids == ["WRK_UPPERCASE"]


# ═══════════════════════════════════════════════════════════════════════
# _parse_ssr_subscription
# ═══════════════════════════════════════════════════════════════════════

# Realistic SSR fragment mimicking OpenCode's SolidJS $R output.
_SSR_TEMPLATE = (
    'lite.subscription.get["wrk_abc123"]={{{{...}}}};\n'
    "rollingUsage:$R[0]={{{{resetInSec: 14213, usagePercent: 6.0}}}};\n"
    "weeklyUsage:$R[1]={{{{usagePercent: 26.0, resetInSec: 197701}}}};\n"
    "monthlyUsage:$R[2]={{{{resetIn: 2483586, usage: 13.0}}}};\n"
)


class TestParseSsrSubscription:
    """Tests for _parse_ssr_subscription()."""

    def test_parses_all_windows(self):
        """All three windows (rolling, weekly, monthly) are extracted."""
        result = _parse_ssr_subscription(_SSR_TEMPLATE)
        assert result is not None
        assert result["rolling_pct"] == 6.0
        assert result["rolling_reset_sec"] == 14213
        assert result["weekly_pct"] == 26.0
        assert result["weekly_reset_sec"] == 197701
        assert result["monthly_pct"] == 13.0
        assert result["monthly_reset_sec"] == 2483586

    def test_parses_without_subscription_get_block(self):
        """Works even when the lite.subscription.get preamble is absent."""
        text = (
            "rollingUsage:$R[0]={{{{resetInSec: 100, usagePercent: 5.5}}}};\n"
        )
        result = _parse_ssr_subscription(text)
        assert result is not None
        assert result["rolling_pct"] == 5.5

    def test_partial_windows_return_none(self):
        """If rolling window is missing, returns None."""
        text = "weeklyUsage:$R[1]={{{{resetInSec: 100, usagePercent: 10}}}};\n"
        result = _parse_ssr_subscription(text)
        assert result is None

    def test_returns_none_for_empty_text(self):
        assert _parse_ssr_subscription("") is None

    def test_returns_none_for_garbage(self):
        assert _parse_ssr_subscription("<html>no ssr data</html>") is None

    def test_handles_variable_field_order(self):
        """Field order inside {{{...}}} shouldn't matter."""
        text = (
            "rollingUsage:$R[0]={{{{usagePct: 88.2, resetIn: 9999}}}};\n"
            "weeklyUsage:$R[1]={{{{resetInSec: 111, usage: 22}}}};\n"
        )
        result = _parse_ssr_subscription(text)
        assert result is not None
        assert result["rolling_pct"] == 88.2
        assert result["rolling_reset_sec"] == 9999
        # weekly has no "usagePercent" key — uses "usage" regex fallback
        assert result["weekly_pct"] == 22
        assert result["weekly_reset_sec"] == 111

    def test_missing_reset_in_block_is_skipped(self):
        """A block with no resetInSec is ignored (both fields needed)."""
        text = (
            "rollingUsage:$R[0]={{{{usagePercent: 50}}}};\n"
        )
        result = _parse_ssr_subscription(text)
        assert result is None

    def test_missing_pct_in_block_is_skipped(self):
        """A block with no usagePercent is ignored."""
        text = (
            "rollingUsage:$R[0]={{{{resetInSec: 5000}}}};\n"
        )
        result = _parse_ssr_subscription(text)
        assert result is None

    def test_scoped_to_subscription_block(self):
        """When lite.subscription.get block exists, parsing is scoped to it."""
        # Put valid data after the subscription block — should still be found
        text = (
            'lite.subscription.get["wrk_xyz"]={{{{stuff}}}};\n'
            "rollingUsage:$R[0]={{{{resetInSec: 1, usagePercent: 99}}}};\n"
        )
        result = _parse_ssr_subscription(text)
        assert result is not None
        assert result["rolling_pct"] == 99


# ═══════════════════════════════════════════════════════════════════════
# SubscriptionSnapshot
# ═══════════════════════════════════════════════════════════════════════

class TestSubscriptionSnapshot:
    """Tests for the SubscriptionSnapshot dataclass."""

    def test_defaults(self):
        snap = SubscriptionSnapshot()
        assert snap.rolling_pct == 0.0
        assert snap.weekly_pct == 0.0
        assert snap.monthly_pct == 0.0
        assert snap.rolling_reset_sec == 0
        assert snap.workspace_id is None

    def test_full_construction(self):
        snap = SubscriptionSnapshot(
            rolling_pct=6.0,
            weekly_pct=26.0,
            monthly_pct=13.0,
            rolling_reset_sec=14213,
            weekly_reset_sec=197701,
            monthly_reset_sec=2483586,
            workspace_id="wrk_test",
        )
        assert snap.rolling_pct == 6.0
        assert snap.workspace_id == "wrk_test"


# ═══════════════════════════════════════════════════════════════════════
# fetch_subscription_dict
# ═══════════════════════════════════════════════════════════════════════

class TestFetchSubscriptionDict:
    """Tests for fetch_subscription_dict()."""

    def test_returns_none_when_fetch_returns_none(self):
        with patch("src.api.cost_plugins.opencode_api.fetch_subscription",
                   return_value=None):
            result = fetch_subscription_dict("cookie")
            assert result is None

    def test_returns_dict_with_reset_at(self):
        """The dict includes human-readable reset_at timestamps."""
        snap = SubscriptionSnapshot(
            rolling_pct=6.0, weekly_pct=26.0, monthly_pct=13.0,
            rolling_reset_sec=3600, weekly_reset_sec=7200,
            monthly_reset_sec=86400, workspace_id="wrk_1",
        )
        with patch("src.api.cost_plugins.opencode_api.fetch_subscription",
                   return_value=snap):
            result = fetch_subscription_dict("cookie")
            assert result is not None
            assert result["rolling_pct"] == 6.0
            assert result["weekly_pct"] == 26.0
            assert result["monthly_pct"] == 13.0
            assert result["rolling_reset_sec"] == 3600
            assert result["rolling_reset_at"] != ""  # has a formatted date
            assert result["weekly_reset_at"] != ""
            assert result["monthly_reset_at"] != ""
            assert result["workspace_id"] == "wrk_1"

    def test_reset_at_empty_when_sec_is_zero(self):
        snap = SubscriptionSnapshot(
            rolling_reset_sec=0, weekly_reset_sec=0, monthly_reset_sec=0,
        )
        with patch("src.api.cost_plugins.opencode_api.fetch_subscription",
                   return_value=snap):
            result = fetch_subscription_dict("cookie")
            assert result["rolling_reset_at"] == ""
            assert result["weekly_reset_at"] == ""
            assert result["monthly_reset_at"] == ""
