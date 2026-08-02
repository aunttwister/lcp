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
from datetime import datetime, timedelta, timezone
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


# ── SSR parsing ────────────────────────────────────────────────────────────

_WRK_RE = re.compile(r'id\s*:\s*"(wrk_[a-zA-Z0-9]+)"', re.IGNORECASE)

# Match the full lite.subscription.get data in the $R SSR block.
_SUBSCRIBE_BLOCK_START_RE = re.compile(
    r'lite\.subscription\.get\["wrk_[^"]+"\]',
    re.DOTALL,
)

# Step 1: find each (rollingUsage|weeklyUsage|monthlyUsage):$R[N]={...} block.
_BLOCK_RE = re.compile(
    r'(rollingUsage|weeklyUsage|monthlyUsage)'
    r':\$R\[\d+\]=\{(.+?)\}',
)

# Step 2: extract resetInSec / usagePercent from inside a block (any order).
_RESET_RE = re.compile(r'(?:resetInSec|resetIn)\s*:\s*(-?\d+)')
_USAGE_PCT_RE = re.compile(
    r'(?:usagePercent|usagePct|usage|pct)\s*:\s*(\d+(?:\.\d+)?)',
)


def _extract_workspace_ids(text: str) -> list[str]:
    """Extract all ``wrk_*`` workspace IDs from SSR data."""
    return list(set(_WRK_RE.findall(text)))


def _parse_ssr_subscription(text: str) -> Optional[dict]:
    """Parse the SolidJS SSR $R block for subscription usage data.

    Looks for ``lite.subscription.get`` query result containing
    rollingUsage, weeklyUsage, monthlyUsage objects.
    """
    # Find the subscription.get block to scope our search.
    sub_start = _SUBSCRIBE_BLOCK_START_RE.search(text)
    if sub_start:
        end_pos = min(len(text), sub_start.end() + 20000)
        block = text[sub_start.start():end_pos]
    else:
        block = text

    # Collect per-window matches — two-step: find blocks, then extract values.
    result: dict = {}
    for m in _BLOCK_RE.finditer(block):
        window = m.group(1)  # rollingUsage, weeklyUsage, monthlyUsage
        body = m.group(2)    # content inside { ... }

        reset_m = _RESET_RE.search(body)
        pct_m = _USAGE_PCT_RE.search(body)
        if not reset_m or not pct_m:
            continue

        reset_sec = int(reset_m.group(1))
        pct = float(pct_m.group(1))

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

    # Debug: log a snippet around any usage-like patterns for diagnosis
    _debug_ssr_failure(block)
    return None


def _debug_ssr_failure(text: str) -> None:
    """Log snippets of the SSR text for diagnosis — inlined into the message
    because structlog JSON output does not render ``extra`` keys."""
    for kw in ("rollingUsage", "weeklyUsage", "monthlyUsage", "usagePercent",
               "resetInSec", "subscription"):
        idx = text.find(kw)
        if idx >= 0:
            start = max(0, idx - 100)
            end = min(len(text), idx + 300)
            snippet = text[start:end].replace("\n", "\\n").replace("\r", "")
            logger.warning(
                "opencode_ssr_context keyword=%s snippet=%s",
                kw, snippet[:500],
            )
            return

    # Nothing found at all — dump first and last chunks of the page.
    head = text[:1500].replace("\n", "\\n")
    tail = text[-1500:].replace("\n", "\\n")
    logger.warning(
        "opencode_ssr_no_usage_found head=%s tail=%s",
        head, tail,
    )


# ── Public API ─────────────────────────────────────────────────────────────

def fetch_subscription(cookie: Optional[str]) -> Optional[SubscriptionSnapshot]:
    """Fetch OpenCode subscription usage from the /go page SSR data.

    Returns a ``SubscriptionSnapshot`` or ``None`` when the cookie is
    missing, unreachable, or no subscription data is found in the page.
    """
    if not cookie or not cookie.strip():
        logger.debug("opencode_cookie_missing")
        return None

    cookie = cookie.strip()
    headers = _base_headers(cookie)

    # ── Step 1: discover workspace ID ──────────────────────────────────
    workspace_id: Optional[str] = os.environ.get("OPENCODE_WORKSPACE_ID", "").strip() or None

    if not workspace_id:
        try:
            raw = _http_get(_OPENCODE_BASE, headers)
            ids = _extract_workspace_ids(raw)
            if ids:
                workspace_id = ids[0]
        except (URLError, OSError) as exc:
            logger.warning("opencode_dashboard_fetch_failed", error=str(exc))
            return None

    if not workspace_id:
        logger.warning("opencode_no_workspace_found")
        return None

    # ── Step 2: fetch /go page and parse SSR subscription data ──────────
    try:
        headers["Referer"] = f"{_OPENCODE_BASE}/workspace/{workspace_id}"
        url = f"{_OPENCODE_BASE}/workspace/{workspace_id}/go"
        raw = _http_get(url, headers)
    except (URLError, OSError) as exc:
        logger.warning("opencode_go_page_fetch_failed", error=str(exc))
        return None

    result = _parse_ssr_subscription(raw)
    if result is None:
        head = raw[:2000].replace("\n", "\\n")
        tail = raw[-2000:].replace("\n", "\\n")
        logger.warning(
            "opencode_subscription_parse_failed len=%d head=%s tail=%s",
            len(raw), head, tail,
        )
        return None

    # Heuristic: if percent < 1, multiply by 100 (fraction → percentage).
    # OpenCode's ``usagePercent`` is already a percentage integer/float;
    # true fractions like 0.5 or 0.01 need scaling, but 1.0 is just 1%.
    for key in ("rolling_pct", "weekly_pct", "monthly_pct"):
        val = result.get(key, 0)
        if 0 < val < 1:
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
    now = datetime.now(timezone.utc)
    def _reset_at(sec: int) -> str:
        if sec:
            return (now + timedelta(seconds=sec)).strftime("%b %d, %H:%M")
        return ""
    return {
        "rolling_pct": snap.rolling_pct,
        "weekly_pct": snap.weekly_pct,
        "monthly_pct": snap.monthly_pct,
        "rolling_reset_sec": snap.rolling_reset_sec,
        "weekly_reset_sec": snap.weekly_reset_sec,
        "monthly_reset_sec": snap.monthly_reset_sec,
        "rolling_reset_at": _reset_at(snap.rolling_reset_sec),
        "weekly_reset_at": _reset_at(snap.weekly_reset_sec),
        "monthly_reset_at": _reset_at(snap.monthly_reset_sec),
        "workspace_id": snap.workspace_id,
    }
