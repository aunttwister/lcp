"""HTTP client for Command Code's internal billing API — subscription tracking.

Fetches credits + subscription data from ``api.commandcode.ai``'s internal
billing endpoints and parses the JSON. Authenticates via the browser session
cookie from a logged-in commandcode.ai session (same pattern as OpenCode).

Endpoints (discovered from the CodexBar project / the web app):

    GET /internal/billing/credits        -> credits summary + rolling limits
    GET /internal/billing/subscriptions  -> active subscription (plan/status)

Usage::

    from .commandcode_api import fetch_subscription_snapshot
    snap = fetch_subscription_snapshot(cookie)
    # => {"monthly_credits_remaining": 8.78, "purchased_credits": 0.0,
    #     "five_hour_pct": 32.0, "weekly_pct": 41.0,
    #     "five_hour_reset_sec": 11520, "weekly_reset_sec": 172800,
    #     "plan_id": "go", "plan_status": "active", ...}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger("lcp.cost.commandcode_api")

_COMMANDCODE_BASE = "https://api.commandcode.ai"
_WEB_ORIGIN = "https://commandcode.ai"

_CREDITS_PATH = "/internal/billing/credits"
_SUBSCRIPTIONS_PATH = "/internal/billing/subscriptions"

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/143.0.0.0 Safari/537.36"
)

_TIMEOUT_SECONDS = 15


@dataclass
class CommandCodeSubscriptionSnapshot:
    """Parsed billing/usage state from the Command Code web API."""
    monthly_credits_remaining: float = 0.0   # USD left in the monthly grant
    purchased_credits: float = 0.0           # top-up / pay-as-you-go balance
    premium_monthly_credits: float = 0.0     # USD left in premium grant
    opensource_monthly_credits: float = 0.0  # USD left in open-source grant
    five_hour_pct: float = 0.0               # 0-100 rolling 5h window usage
    weekly_pct: float = 0.0                  # 0-100 rolling weekly usage
    five_hour_reset_sec: int = 0             # seconds until 5h window resets
    weekly_reset_sec: int = 0                # seconds until weekly resets
    plan_id: Optional[str] = None
    plan_status: Optional[str] = None
    billing_period_end: Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────────

def _base_headers(cookie: str) -> dict[str, str]:
    """Return common headers for billing API requests."""
    return {
        "Cookie": cookie.strip(),
        "User-Agent": _USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Origin": _WEB_ORIGIN,
        "Referer": _WEB_ORIGIN + "/",
    }


def _http_get_json(url: str, headers: dict[str, str],
                   timeout: int = _TIMEOUT_SECONDS) -> dict:
    """Thin wrapper around urllib GET that parses a JSON object."""
    req = Request(url, headers=headers, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object, got {type(data).__name__}")
    return data


# ── Credits parsing ────────────────────────────────────────────────────────

def _parse_credits(payload: dict) -> dict:
    """Extract credits fields from the /internal/billing/credits response.

    Expected shape::

        {"credits": {
            "monthlyCredits": 8.78,
            "purchasedCredits": 0.0,
            "premiumMonthlyCredits": 0.0,
            "opensourceMonthlyCredits": 8.78,
            "belowThreshold": false,
            "creditThreshold": 0,
            "fiveHourLimit": {"used": 32, "limit": 100, "resetInSeconds": 11520},
            "weeklyLimit":   {"used": 41, "limit": 100, "resetInSeconds": 172800},
        }}
    """
    credits = payload.get("credits") or {}

    def _f(key: str, default: float = 0.0) -> float:
        try:
            val = credits.get(key)
            return float(val) if val is not None else default
        except (TypeError, ValueError):
            return default

    def _window(key: str) -> tuple[float, float, int]:
        win = credits.get(key) or {}
        used = _f_win(win, "used")
        limit = _f_win(win, "limit")
        reset = _i_win(win, "resetInSeconds") or _i_win(win, "resetIn") or 0
        pct = (used / limit * 100) if limit and limit > 0 else 0.0
        return pct, used, reset

    def _f_win(win: dict, key: str, default: float = 0.0) -> float:
        try:
            val = win.get(key)
            return float(val) if val is not None else default
        except (TypeError, ValueError):
            return default

    def _i_win(win: dict, key: str, default: int = 0) -> int:
        try:
            val = win.get(key)
            return int(val) if val is not None else default
        except (TypeError, ValueError):
            return default

    five_hour_pct, _five_hour_used, five_hour_reset = _window("fiveHourLimit")
    weekly_pct, _weekly_used, weekly_reset = _window("weeklyLimit")

    return {
        "monthly_credits_remaining": _f("monthlyCredits"),
        "purchased_credits": _f("purchasedCredits"),
        "premium_monthly_credits": _f("premiumMonthlyCredits"),
        "opensource_monthly_credits": _f("opensourceMonthlyCredits"),
        "five_hour_pct": round(five_hour_pct, 1),
        "weekly_pct": round(weekly_pct, 1),
        "five_hour_reset_sec": five_hour_reset,
        "weekly_reset_sec": weekly_reset,
    }


# ── Subscription parsing ───────────────────────────────────────────────────

def _parse_subscription(payload: dict) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract (plan_id, status, current_period_end) from /subscriptions.

    Expected shape::

        {"success": true, "data": {
            "id": "sub_...", "status": "active",
            "planID": "go", "currentPeriodEnd": "2026-09-08T00:00:00Z",
        }}

    ``data`` may be null when the user has no active subscription (free tier) —
    in that case all three fields are None and no error is raised.
    """
    data = payload.get("data")
    if not isinstance(data, dict):
        return None, None, None
    plan_id = data.get("planID") or data.get("planId") or data.get("plan")
    status = data.get("status")
    period_end = data.get("currentPeriodEnd") or data.get("current_period_end")
    return plan_id, status, period_end


# ── Public API ─────────────────────────────────────────────────────────────

def fetch_subscription_snapshot(cookie: Optional[str]) -> Optional[CommandCodeSubscriptionSnapshot]:
    """Fetch Command Code billing/usage data from the internal API.

    Returns a ``CommandCodeSubscriptionSnapshot`` or ``None`` when the cookie
    is missing, unreachable, or the response cannot be parsed.
    """
    if not cookie or not cookie.strip():
        logger.debug("commandcode_cookie_missing")
        return None

    cookie = cookie.strip()
    headers = _base_headers(cookie)

    # ── Fetch credits (primary) ────────────────────────────────────────
    try:
        credits_url = _COMMANDCODE_BASE + _CREDITS_PATH
        credits_payload = _http_get_json(credits_url, headers)
    except HTTPError as exc:
        if exc.code in (401, 403):
            logger.warning("commandcode_auth_failed status=%s", exc.code)
            return None
        logger.warning("commandcode_credits_http_error status=%s", exc.code)
        return None
    except (URLError, OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("commandcode_credits_fetch_failed: %s", str(exc))
        return None

    try:
        credits = _parse_credits(credits_payload)
    except Exception as exc:
        logger.warning("commandcode_credits_parse_failed: %s", str(exc))
        return None

    # ── Fetch subscription (best-effort enrichment) ────────────────────
    plan_id = plan_status = period_end = None
    try:
        sub_url = _COMMANDCODE_BASE + _SUBSCRIPTIONS_PATH
        sub_payload = _http_get_json(sub_url, headers)
        plan_id, plan_status, period_end = _parse_subscription(sub_payload)
    except Exception as exc:
        # Subscription enrichment is optional — credits still usable.
        logger.warning("commandcode_subscription_fetch_failed: %s", str(exc))

    return CommandCodeSubscriptionSnapshot(
        monthly_credits_remaining=credits["monthly_credits_remaining"],
        purchased_credits=credits["purchased_credits"],
        premium_monthly_credits=credits["premium_monthly_credits"],
        opensource_monthly_credits=credits["opensource_monthly_credits"],
        five_hour_pct=credits["five_hour_pct"],
        weekly_pct=credits["weekly_pct"],
        five_hour_reset_sec=credits["five_hour_reset_sec"],
        weekly_reset_sec=credits["weekly_reset_sec"],
        plan_id=plan_id,
        plan_status=plan_status,
        billing_period_end=period_end,
    )


def fetch_subscription_snapshot_dict(cookie: Optional[str]) -> Optional[dict]:
    """Same as ``fetch_subscription_snapshot`` but returns a plain dict (or None)."""
    snap = fetch_subscription_snapshot(cookie)
    if snap is None:
        return None
    now = datetime.now(timezone.utc)

    def _reset_at(sec: int) -> str:
        if sec:
            return (now + timedelta(seconds=sec)).strftime("%b %d, %H:%M")
        return ""

    return {
        "monthly_credits_remaining": snap.monthly_credits_remaining,
        "purchased_credits": snap.purchased_credits,
        "premium_monthly_credits": snap.premium_monthly_credits,
        "opensource_monthly_credits": snap.opensource_monthly_credits,
        "five_hour_pct": snap.five_hour_pct,
        "weekly_pct": snap.weekly_pct,
        "five_hour_reset_sec": snap.five_hour_reset_sec,
        "weekly_reset_sec": snap.weekly_reset_sec,
        "five_hour_reset_at": _reset_at(snap.five_hour_reset_sec),
        "weekly_reset_at": _reset_at(snap.weekly_reset_sec),
        "plan_id": snap.plan_id,
        "plan_status": snap.plan_status,
        "billing_period_end": snap.billing_period_end,
    }
