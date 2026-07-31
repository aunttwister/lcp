"""HTTP client for OpenCode's web API — subscription & usage tracking.

Calls ``opencode.ai/_server`` to fetch workspace ID and subscription usage
(rolling 5-hour / weekly percentages with reset countdowns).  Authenticates
via the ``auth`` browser cookie from a logged-in session.

Usage::

    from .opencode_api import fetch_subscription
    snapshot = fetch_subscription(os.environ.get("OPENCODE_COOKIE"))
    # => {"rolling_pct": 17.0, "weekly_pct": 75.0,
    #     "rolling_reset_sec": 5944, "weekly_reset_sec": 278201}
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger("lcp.cost.opencode_api")

# ── Server function IDs ────────────────────────────────────────────────────
_WORKSPACE_FN = "def39973159c7f0483d8793a822b8dbb10d067e12c65455fcb4608459ba0234f"
_SUBSCRIPTION_FN = "7abeebee372f304e050aaaf92be863f4a86490e382f8c79db68fd94040d691b4"

_OPENCODE_BASE = "https://opencode.ai"

# ── User-Agent (matches what a real browser sends) ─────────────────────────
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/143.0.0.0 Safari/537.36"
)


@dataclass
class SubscriptionSnapshot:
    """Parsed subscription usage from the OpenCode web API."""
    rolling_pct: float          # 0-100
    weekly_pct: float           # 0-100
    rolling_reset_sec: int      # seconds until 5-hour window resets
    weekly_reset_sec: int       # seconds until weekly window resets
    workspace_id: Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_headers(cookie: str, fn_id: str) -> dict[str, str]:
    """Return common headers for every _server request."""
    return {
        "Cookie": cookie.strip(),
        "X-Server-Id": fn_id,
        "X-Server-Instance": f"server-fn:{uuid.uuid4()}",
        "User-Agent": _USER_AGENT,
        "Origin": _OPENCODE_BASE,
        "Referer": _OPENCODE_BASE,
        "Accept": "text/javascript, application/json;q=0.9, */*;q=0.8",
    }


def _http_get(url: str, headers: dict[str, str], timeout: int = 15) -> str:
    """Thin wrapper around urllib GET.  Raises on non-200 or network error."""
    req = Request(url, headers=headers, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _http_post(url: str, headers: dict[str, str], body: str,
               timeout: int = 15) -> str:
    """Thin wrapper around urllib POST."""
    data = body.encode("utf-8")
    req = Request(url, data=data, headers=headers, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


# ── Parsing ────────────────────────────────────────────────────────────────

_WRK_RE = re.compile(r'id\s*:\s*"(wrk_[^"]+)"', re.IGNORECASE)

# Percent: usagePercent, usedPercent, percentUsed, percent, …
_PCT_RE = re.compile(
    r'(?:usagePercent|usedPercent|percentUsed|percent|usage_percent|'
    r'used_percent|utilization|utilizationPercent|utilization_percent|usage)'
    r'\s*:\s*([0-9]+(?:\.[0-9]+)?)'
)
# Reset seconds
_RESET_RE = re.compile(
    r'(?:resetInSec|resetInSeconds|resetSeconds|reset_sec|reset_in_sec|'
    r'resetsInSec|resetsInSeconds|resetIn|resetSec)'
    r'\s*:\s*([0-9]+)'
)


def _extract_workspace_id(text: str) -> Optional[str]:
    """Extract the first ``wrk_*`` workspace ID from the response."""
    # Try JSON first
    try:
        obj = json.loads(text)
        ids = _scan_for_wrk(obj)
        if ids:
            return ids[0]
    except (json.JSONDecodeError, TypeError):
        pass

    # Regex fallback
    m = _WRK_RE.search(text)
    if m:
        return m.group(1)

    return None


def _scan_for_wrk(obj, found: Optional[list] = None) -> list[str]:
    """Recursively scan a JSON tree for ``wrk_*`` strings."""
    if found is None:
        found = []
    if isinstance(obj, str) and obj.startswith("wrk_"):
        found.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _scan_for_wrk(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _scan_for_wrk(v, found)
    return found


# ── Regex-based subscription parsing (fallback when not JSON) ──────────────

def _parse_js_subscription(text: str) -> Optional[dict]:
    """Extract rolling/weekly usage and reset from JavaScript-serialized text."""
    rolling_pct: Optional[float] = None
    weekly_pct: Optional[float] = None
    rolling_reset: Optional[int] = None
    weekly_reset: Optional[int] = None

    # Find rollingUsage block
    roll_match = re.search(r'rollingUsage[^}]*?usagePercent\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if roll_match:
        rolling_pct = float(roll_match.group(1))
    roll_reset_match = re.search(r'rollingUsage[^}]*?resetInSec\s*:\s*([0-9]+)', text)
    if roll_reset_match:
        rolling_reset = int(roll_reset_match.group(1))

    # Find weeklyUsage block
    week_match = re.search(r'weeklyUsage[^}]*?usagePercent\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if week_match:
        weekly_pct = float(week_match.group(1))
    week_reset_match = re.search(r'weeklyUsage[^}]*?resetInSec\s*:\s*([0-9]+)', text)
    if week_reset_match:
        weekly_reset = int(week_reset_match.group(1))

    if rolling_pct is None or rolling_reset is None:
        return None

    return {
        "rolling_pct": rolling_pct,
        "weekly_pct": weekly_pct,
        "rolling_reset_sec": rolling_reset,
        "weekly_reset_sec": weekly_reset,
    }


# ── JSON-based subscription parsing (tried first) ──────────────────────────

def _parse_json_subscription(obj) -> Optional[dict]:
    """Walk a JSON tree looking for rollingUsage / weeklyUsage objects."""

    def _walk(o, path: str = "") -> list[dict]:
        results: list[dict] = []
        if isinstance(o, dict):
            # Check if this object looks like a usage window
            if "usagePercent" in o and "resetInSec" in o:
                results.append({"path": path, "obj": o})
            for k, v in o.items():
                results.extend(_walk(v, f"{path}.{k}" if path else k))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                results.extend(_walk(v, f"{path}[{i}]"))
        return results

    candidates = _walk(obj)
    rolling = None
    weekly = None

    for c in candidates:
        p_lower = c["path"].lower()
        o = c["obj"]
        pct = float(o.get("usagePercent", o.get("usedPercent",
               o.get("percentUsed", o.get("percent", o.get("usage_percent",
               o.get("utilization", o.get("utilizationPercent", 0))))))))
        reset_sec = int(o.get("resetInSec", o.get("resetInSeconds",
                            o.get("resetSeconds", o.get("reset_sec",
                            o.get("reset_in_sec", o.get("resetsInSec",
                            o.get("resetIn", 0))))))))

        if any(kw in p_lower for kw in ("rolling", "hour", "5h", "5-hour")):
            rolling = {"rolling_pct": pct, "rolling_reset_sec": reset_sec}
        elif any(kw in p_lower for kw in ("week",)):
            weekly = {"weekly_pct": pct, "weekly_reset_sec": reset_sec}

    if rolling is None:
        return None

    result = dict(rolling)
    if weekly:
        result.update(weekly)
    return result


# ── Public API ─────────────────────────────────────────────────────────────

def fetch_subscription(cookie: Optional[str]) -> Optional[SubscriptionSnapshot]:
    """Fetch OpenCode subscription usage snapshot.

    Returns a ``SubscriptionSnapshot`` or ``None`` when the cookie is
    missing, invalid, or the API is unreachable.
    """
    if not cookie or not cookie.strip():
        logger.debug("opencode_cookie_missing")
        return None

    cookie = cookie.strip()

    # ── Step 1: get workspace ID ───────────────────────────────────────
    try:
        headers = _build_headers(cookie, _WORKSPACE_FN)
        url = f"{_OPENCODE_BASE}/_server?id={_WORKSPACE_FN}"
        raw = _http_get(url, headers)
    except (URLError, OSError) as exc:
        logger.warning("opencode_workspace_fetch_failed", error=str(exc))
        return None

    # Detect signed-out / invalid credentials
    raw_lower = raw.lower()
    if any(needle in raw_lower for needle in (
        '"login"', '"sign in"', '"auth/authorize"',
        '"not associated with an account"',
        '"actor of type \\"public\\""',
    )):
        logger.warning("opencode_not_authenticated")
        return None

    workspace_id = _extract_workspace_id(raw)
    if not workspace_id:
        logger.warning("opencode_no_workspace_found")
        return None

    # ── Step 2: get subscription usage ──────────────────────────────────
    try:
        headers = _build_headers(cookie, _SUBSCRIPTION_FN)
        headers["Referer"] = f"{_OPENCODE_BASE}/workspace/{workspace_id}/billing"
        args_json = json.dumps([workspace_id])
        url = f"{_OPENCODE_BASE}/_server?id={_SUBSCRIPTION_FN}&args={args_json}"
        raw = _http_get(url, headers)
    except (URLError, OSError) as exc:
        logger.warning("opencode_subscription_fetch_failed", error=str(exc))
        return None

    # Detect null response
    if raw.strip() in ("null", ""):
        logger.warning("opencode_subscription_null")
        return None

    # Parse: try JSON first, then regex
    result: Optional[dict] = None
    try:
        obj = json.loads(raw)
        result = _parse_json_subscription(obj)
    except (json.JSONDecodeError, TypeError):
        pass

    if result is None:
        result = _parse_js_subscription(raw)

    if result is None:
        logger.warning("opencode_subscription_parse_failed")
        return None

    rolling_pct = result.get("rolling_pct", 0)
    # Heuristic: if percent ≤ 1 and is a direct field (not from used/limit),
    # multiply by 100 (treating fraction like 0.17 → 17%)
    if 0 < rolling_pct <= 1:
        rolling_pct *= 100

    weekly_pct = result.get("weekly_pct", 0)
    if 0 < weekly_pct <= 1:
        weekly_pct *= 100

    return SubscriptionSnapshot(
        rolling_pct=round(rolling_pct, 1),
        weekly_pct=round(weekly_pct, 1),
        rolling_reset_sec=result.get("rolling_reset_sec", 0),
        weekly_reset_sec=result.get("weekly_reset_sec", 0),
        workspace_id=workspace_id,
    )


def fetch_subscription_dict(cookie: Optional[str]) -> Optional[dict]:
    """Same as ``fetch_subscription`` but returns a plain dict (or None)."""
    snap = fetch_subscription(cookie)
    if snap is None:
        return None
    return {
        "rolling_pct": snap.rolling_pct,
        "weekly_pct": snap.weekly_pct,
        "rolling_reset_sec": snap.rolling_reset_sec,
        "weekly_reset_sec": snap.weekly_reset_sec,
        "workspace_id": snap.workspace_id,
    }
