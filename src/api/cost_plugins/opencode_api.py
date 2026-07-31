"""HTTP client for OpenCode's web API — subscription & usage tracking.

Fetches the OpenCode /go page (SolidJS SSR) and extracts subscription usage
(rolling 5-hour / weekly / monthly percentages with reset countdowns).
Authenticates via the ``auth`` browser cookie from a logged-in session.

Usage::

    from .opencode_api import fetch_subscription
    snapshot = fetch_subscription(os.environ.get("OPENCODE_COOKIE"))
    # => {"rolling_pct": 6.0, "weekly_pct": 26.0, "monthly_pct": 13.0,
    #     "rolling_reset_sec": 14213, "weekly_reset_sec": 197701,
    #     "monthly_reset_sec": 2483586}
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger("lcp.cost.opencode_api")

_OPENCODE_BASE = "https://opencode.ai"

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/143.0.0.0 Safari/537.36"
)


@dataclass
class SubscriptionSnapshot:
    """Parsed subscription usage from the OpenCode web API."""
    rolling_pct: float = 0.0        # 0-100
    weekly_pct: float = 0.0         # 0-100
    monthly_pct: float = 0.0        # 0-100
    rolling_reset_sec: int = 0      # seconds until 5-hour window resets
    weekly_reset_sec: int = 0       # seconds until weekly window resets
    monthly_reset_sec: int = 0      # seconds until monthly window resets
    workspace_id: Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────────

def _base_headers(cookie: str) -> dict[str, str]:
    """Return common headers for page requests."""
    return {
        "Cookie": cookie.strip(),
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Origin": _OPENCODE_BASE,
        "Referer": _OPENCODE_BASE,
    }


def _http_get(url: str, headers: dict[str, str], timeout: int = 15) -> str:
    """Thin wrapper around urllib GET."""
    req = Request(url, headers=headers, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _is_authenticated(raw: str) -> bool:
    """Check whether the page returned signed-in content."""
    raw_lower = raw.lower()
    needles = (
        '"login"', '"sign in"', '"auth/authorize"',
        '"not associated with an account"',
        '"actor of type \\"public\\""',
    )
    if any(n in raw_lower for n in needles):
        return False
    # If we got redirected to auth, the page is tiny
    if len(raw) < 2000:
        return False
    return True


# ── SSR parsing ────────────────────────────────────────────────────────────

_WRK_RE = re.compile(r'wrk_[a-zA-Z0-9]+')

# Match the full lite.subscription.get data in the $R SSR block.
# Format: lite.subscription.get["wrk_..."]...rollingUsage:$R[N]={status:"ok",...}
_SUB_COMBINED_RE = re.compile(
    r'lite\.subscription\.get\["wrk_[^"]+"\]'
    r'.+?'
    r'rollingUsage:\$R\[\d+\]=\{status:"[^"]*",resetInSec:(\d+),usagePercent:(\d+(?:\.\d+)?)\}'
    r'.+?'
    r'weeklyUsage:\$R\[\d+\]=\{status:"[^"]*",resetInSec:(\d+),usagePercent:(\d+(?:\.\d+)?)\}'
    r'.+?'
    r'monthlyUsage:\$R\[\d+\]=\{status:"[^"]*",resetInSec:(\d+),usagePercent:(\d+(?:\.\d+)?)\}',
    re.DOTALL,
)

# Per-window fallback regex
_USAGE_BLOCK_RE = re.compile(
    r'(rollingUsage|weeklyUsage|monthlyUsage)'
    r':\$R\[\d+\]=\{status:"([^"]*)",resetInSec:(\d+),usagePercent:(\d+(?:\.\d+)?)\}',
)


def _extract_workspace_ids(text: str) -> list[str]:
    """Extract all ``wrk_*`` workspace IDs from SSR data."""
    return list(set(_WRK_RE.findall(text)))


def _parse_ssr_subscription(text: str) -> Optional[dict]:
    """Parse the SolidJS SSR $R block for subscription usage data.

    Looks for ``lite.subscription.get`` query result containing
    rollingUsage, weeklyUsage, monthlyUsage objects.
    """
    # Approach 1: single combined regex
    m = _SUB_COMBINED_RE.search(text)
    if m:
        return {
            "rolling_reset_sec": int(m.group(1)),
            "rolling_pct": float(m.group(2)),
            "weekly_reset_sec": int(m.group(3)),
            "weekly_pct": float(m.group(4)),
            "monthly_reset_sec": int(m.group(5)),
            "monthly_pct": float(m.group(6)),
        }

    # Approach 2: per-window blocks individually
    result: dict = {}
    for m in _USAGE_BLOCK_RE.finditer(text):
        window = m.group(1)  # rollingUsage, weeklyUsage, monthlyUsage
        reset_sec = int(m.group(3))
        pct = float(m.group(4))

        if "rolling" in window:
            result["rolling_reset_sec"] = reset_sec
            result["rolling_pct"] = pct
        elif "weekly" in window:
            result["weekly_reset_sec"] = reset_sec
            result["weekly_pct"] = pct
        elif "monthly" in window:
            result["monthly_reset_sec"] = reset_sec
            result["monthly_pct"] = pct

    if "rolling_pct" in result and "rolling_reset_sec" in result:
        return result

    return None


# ── Public API ─────────────────────────────────────────────────────────────

def fetch_subscription(cookie: Optional[str]) -> Optional[SubscriptionSnapshot]:
    """Fetch OpenCode subscription usage from the /go page SSR data.

    ``https://opencode.ai/go`` works without a workspace ID — it resolves
    to the default workspace and exposes the ID in the SSR block.  A second
    request to ``/workspace/{wid}/go`` fetches the full subscription data.

    Returns a ``SubscriptionSnapshot`` or ``None`` when the cookie is
    missing, invalid, or no subscription data is found.
    """
    if not cookie or not cookie.strip():
        logger.debug("opencode_cookie_missing")
        return None

    cookie = cookie.strip()
    headers = _base_headers(cookie)
    workspace_id = os.environ.get("OPENCODE_WORKSPACE_ID", "").strip() or None

    # ── Step 1: discover workspace ID from /go (no ID in URL) ──────────
    if not workspace_id:
        try:
            raw = _http_get(f"{_OPENCODE_BASE}/go", headers)
            if not _is_authenticated(raw):
                logger.warning("opencode_not_authenticated")
                return None
            ids = _extract_workspace_ids(raw)
            if ids:
                workspace_id = ids[0]
        except (URLError, OSError) as exc:
            logger.warning("opencode_go_discovery_failed", error=str(exc))
            return None

    if not workspace_id:
        logger.warning("opencode_no_workspace_found")
        return None

    # ── Step 2: fetch /workspace/{wid}/go for subscription data ────────
    try:
        go_url = f"{_OPENCODE_BASE}/workspace/{workspace_id}/go"
        headers["Referer"] = f"{_OPENCODE_BASE}/workspace/{workspace_id}"
        raw = _http_get(go_url, headers)
    except (URLError, OSError) as exc:
        logger.warning("opencode_go_page_fetch_failed", error=str(exc))
        return None

    if not _is_authenticated(raw):
        logger.warning("opencode_not_authenticated_go")
        return None

    # ── Parse subscription data from SSR ───────────────────────────────
    result = _parse_ssr_subscription(raw)
    if result is None:
        logger.warning("opencode_subscription_parse_failed")
        return None

    # Heuristic: if percent ≤ 1, multiply by 100 (fraction → percentage)
    for key in ("rolling_pct", "weekly_pct", "monthly_pct"):
        val = result.get(key, 0)
        if 0 < val <= 1:
            result[key] = val * 100

    return SubscriptionSnapshot(
        rolling_pct=round(result.get("rolling_pct", 0), 1),
        weekly_pct=round(result.get("weekly_pct", 0), 1),
        monthly_pct=round(result.get("monthly_pct", 0), 1),
        rolling_reset_sec=result.get("rolling_reset_sec", 0),
        weekly_reset_sec=result.get("weekly_reset_sec", 0),
        monthly_reset_sec=result.get("monthly_reset_sec", 0),
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
        "monthly_pct": snap.monthly_pct,
        "rolling_reset_sec": snap.rolling_reset_sec,
        "weekly_reset_sec": snap.weekly_reset_sec,
        "monthly_reset_sec": snap.monthly_reset_sec,
        "workspace_id": snap.workspace_id,
    }
