"""Tests for src/api/cost_plugins/opencode_api.py — SSR parser and HTTP client."""
from unittest.mock import MagicMock, patch

import pytest

from src.api.cost_plugins.opencode_api import (
    SubscriptionSnapshot,
    _base_headers,
    _extract_workspace_ids,
    _http_get,
    _parse_ssr_subscription,
    fetch_subscription,
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


# ═══════════════════════════════════════════════════════════════════════
# fetch_subscription
# ═══════════════════════════════════════════════════════════════════════

class TestFetchSubscription:
    """Tests for fetch_subscription() — the full HTTP client flow."""

    def test_missing_cookie_returns_none(self):
        assert fetch_subscription(None) is None
        assert fetch_subscription("") is None
        assert fetch_subscription("   ") is None

    def test_workspace_id_from_arg(self):
        page = '<script>id:"wrk_env123"</script>'
        with patch("src.api.cost_plugins.opencode_api._http_get", return_value=page) as mock_get:
            with patch("src.api.cost_plugins.opencode_api._parse_ssr_subscription",
                       return_value={"rolling_pct": 6.0, "weekly_pct": 26.0, "monthly_pct": 13.0}):
                snap = fetch_subscription(" cookie ", workspace_id="wrk_env123")
        assert snap is not None
        assert snap.rolling_pct == 6.0
        assert snap.workspace_id == "wrk_env123"
        # Only the /go page is fetched (no dashboard discovery)
        urls = [c[0][0] for c in mock_get.call_args_list]
        assert len(urls) == 1
        assert "/go" in urls[0]

    def test_workspace_discovered_from_dashboard(self):
        with patch("src.api.cost_plugins.opencode_api._http_get") as mock_get:
            mock_get.side_effect = [
                'id:"wrk_abc"',  # dashboard page -> workspace discovery
                "<html>go page</html>",  # /go page
            ]
            with patch("src.api.cost_plugins.opencode_api._parse_ssr_subscription",
                       return_value={"rolling_pct": 10.0}):
                snap = fetch_subscription("cookie")
        assert snap is not None
        assert snap.workspace_id == "wrk_abc"
        assert snap.rolling_pct == 10.0

    def test_dashboard_fetch_failure_returns_none(self):
        from urllib.error import URLError
        with patch("src.api.cost_plugins.opencode_api._http_get",
                   side_effect=URLError("down")):
            assert fetch_subscription("cookie") is None

    def test_no_workspace_found_returns_none(self):
        with patch("src.api.cost_plugins.opencode_api._http_get",
                   return_value="no ids here"):
            assert fetch_subscription("cookie") is None

    def test_go_page_fetch_failure_returns_none(self):
        from urllib.error import URLError
        with patch("src.api.cost_plugins.opencode_api._http_get",
                   side_effect=URLError("down")):
            assert fetch_subscription("cookie", workspace_id="wrk_1") is None

    def test_parse_failure_returns_none(self):
        with patch("src.api.cost_plugins.opencode_api._http_get",
                   return_value="<html>no data</html>"):
            with patch("src.api.cost_plugins.opencode_api._parse_ssr_subscription",
                       return_value=None):
                assert fetch_subscription("cookie", workspace_id="wrk_1") is None

    def test_fraction_percentages_scaled(self):
        """Values in (0,1) are treated as fractions and multiplied by 100."""
        with patch("src.api.cost_plugins.opencode_api._http_get", return_value="<html/>"):
            with patch("src.api.cost_plugins.opencode_api._parse_ssr_subscription",
                       return_value={"rolling_pct": 0.5, "weekly_pct": 0.01, "monthly_pct": 1.0}):
                snap = fetch_subscription("cookie", workspace_id="wrk_1")
        assert snap is not None
        assert snap.rolling_pct == 50.0  # 0.5 * 100
        assert snap.weekly_pct == 1.0    # 0.01 * 100
        assert snap.monthly_pct == 1.0   # 1.0 left as-is

    def test_success_rounds_and_carries_resets(self):
        with patch("src.api.cost_plugins.opencode_api._http_get", return_value="<html/>"):
            with patch("src.api.cost_plugins.opencode_api._parse_ssr_subscription",
                       return_value={
                           "rolling_pct": 6.0, "weekly_pct": 26.4, "monthly_pct": 13.04,
                           "rolling_reset_sec": 3600, "weekly_reset_sec": 7200,
                           "monthly_reset_sec": 10800,
                           }):
                    snap = fetch_subscription("cookie", workspace_id="wrk_1")
        assert snap is not None
        assert snap.rolling_pct == 6.0
        assert snap.weekly_pct == 26.4
        assert snap.monthly_pct == 13.0
        assert snap.rolling_reset_sec == 3600
        assert snap.weekly_reset_sec == 7200
        assert snap.monthly_reset_sec == 10800


# ═══════════════════════════════════════════════════════════════════════
# _http_get
# ═══════════════════════════════════════════════════════════════════════

class TestHttpGet:
    def test_returns_decoded_body(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<html>hello</html>"
        mock_resp.__enter__.return_value = mock_resp
        with patch("src.api.cost_plugins.opencode_api.urlopen", return_value=mock_resp):
            body = _http_get("https://x", {"Cookie": "c"})
        assert body == "<html>hello</html>"


# ═══════════════════════════════════════════════════════════════════════
# Billing (credits) parsing + fetch
# ═══════════════════════════════════════════════════════════════════════

from src.api.cost_plugins.opencode_api import (
    BillingSnapshot,
    _parse_ssr_billing,
    fetch_billing,
    fetch_billing_dict,
)

_BILLING_TEMPLATE = (
    'lite.billing.get["wrk_abc123"]={{{{...}}}};\n'
    "availableCredits:$R[0]={{{{available: 25.75, plan: \"pro\"}}}};\n"
    "credits:$R[1]={{{{totalCredits: 100, balance: 25.75}}}};\n"
)


class TestParseSsrBilling:
    def test_parses_available_credits(self):
        result = _parse_ssr_billing(_BILLING_TEMPLATE)
        assert result is not None
        assert result["available_credits"] == 25.75
        assert result["currency"] == "USD"
        assert result["plan"] == "pro"

    def test_returns_none_for_empty(self):
        assert _parse_ssr_billing("") is None

    def test_returns_none_for_garbage(self):
        assert _parse_ssr_billing("<html>no ssr</html>") is None

    def test_balances_at_top_level(self):
        """A bare top-level balance: X (not in $R block) is still found."""
        text = 'lite.billing.get["wrk_1"]={{{balance: 42.5}}};\n'
        result = _parse_ssr_billing(text)
        assert result is not None
        assert result["available_credits"] == 42.5

    def test_prefers_available_over_total(self):
        """'available' beats 'totalCredits' as the balance source."""
        text = (
            "credits:$R[0]={{{{totalCredits: 100, availableCredits: 12.5}}}};\n"
        )
        result = _parse_ssr_billing(text)
        assert result is not None
        assert result["available_credits"] == 12.5

    def test_ignores_implausible_generic_balance(self):
        """A huge generic 'balance' (token ledger) must not override real credits.

        Regression: prod returned $949,260,397 instead of $9.49 because the
        parser took a large non-credit ``balance`` value.
        """
        text = (
            "billing:$R[0]={{{{balance: 949260397, plan: \"pro\"}}}};\n"
            "credits:$R[1]={{{{availableCredits: 9.49}}}};\n"
        )
        result = _parse_ssr_billing(text)
        assert result is not None
        assert result["available_credits"] == 9.49

    def test_implausible_generic_only_falls_back_to_plausible(self):
        """When only generic keys exist, implausible ones are skipped."""
        text = (
            "billing:$R[0]={{{{balance: 949260397, credits: 12.34}}}};\n"
        )
        result = _parse_ssr_billing(text)
        assert result is not None
        assert result["available_credits"] == 12.34


class TestBillingSnapshot:
    def test_defaults(self):
        snap = BillingSnapshot()
        assert snap.available_credits is None
        assert snap.currency == "USD"
        assert snap.plan is None
        assert snap.workspace_id is None


class TestFetchBilling:
    def test_missing_cookie_returns_none(self):
        assert fetch_billing(None) is None
        assert fetch_billing("") is None
        assert fetch_billing("   ") is None

    def test_uses_workspace_id_from_arg(self):
        with patch("src.api.cost_plugins.opencode_api._http_get", return_value="<html/>") as mock_get:
            with patch("src.api.cost_plugins.opencode_api._parse_ssr_billing",
                       return_value={"available_credits": 9.99, "currency": "USD", "plan": "pro"}):
                snap = fetch_billing(" cookie ", workspace_id="wrk_bill")
        assert snap is not None
        assert snap.available_credits == 9.99
        assert snap.plan == "pro"
        assert snap.workspace_id == "wrk_bill"
        urls = [c[0][0] for c in mock_get.call_args_list]
        assert len(urls) == 1
        assert "/billing" in urls[0]

    def test_parse_failure_returns_none(self):
        with patch("src.api.cost_plugins.opencode_api._http_get", return_value="<html/>"):
            with patch("src.api.cost_plugins.opencode_api._parse_ssr_billing", return_value=None):
                assert fetch_billing("cookie", workspace_id="wrk_1") is None


class TestFetchBillingDict:
    def test_returns_none_when_fetch_returns_none(self):
        with patch("src.api.cost_plugins.opencode_api.fetch_billing", return_value=None):
            assert fetch_billing_dict("cookie") is None

    def test_returns_contract_dict(self):
        snap = BillingSnapshot(available_credits=12.34, plan="pro",
                               workspace_id="wrk_1")
        with patch("src.api.cost_plugins.opencode_api.fetch_billing", return_value=snap):
            result = fetch_billing_dict("cookie")
        assert result["balance"] == 12.34
        assert result["available_credits"] == 12.34
        assert result["currency"] == "USD"
        assert result["plan"] == "pro"
        assert result["workspace_id"] == "wrk_1"
