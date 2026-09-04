"""HTTP client for OpenCode's web API — subscription & usage tracking.

Fetches the OpenCode /go page (SolidJS SSR) and extracts subscription usage
(rolling 5-hour / weekly / monthly percentages with reset countdowns).
Authenticates via the ``auth`` browser cookie from a logged-in session.

Usage::

    from .opencode_api import fetch_subscription
    snapshot = fetch_subscription(cookie, workspace_id=workspace_id)
    # => {"rolling_pct": 6.0, "weekly_pct": 26.0, "monthly_pct": 13.0,
    #     "rolling_reset_sec": 14213, "weekly_reset_sec": 197701,
    #     "monthly_reset_sec": 2483586}
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from ..logging_config import get_logger

logger = get_logger("lcp.cost.opencode_api")

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


@dataclass
class BillingSnapshot:
    """Parsed credits / balance from the OpenCode billing page."""
    available_credits: Optional[float] = None   # USD available now
    currency: str = "USD"
    plan: Optional[str] = None                  # e.g. "pro" / "go"
    workspace_id: Optional[str] = None
    fetched_at: Optional[str] = None


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

    return None


# ── Absence classification ─────────────────────────────────────────────────
# When the /go page loads but yields no usage SSR data, these helpers decide
# WHY, so the caller can log a concise reason (and surface a specific message)
# instead of dumping the raw page.

# A rendered (authenticated) workspace shell that has NO active subscription —
# the subscription object is explicitly null on the OpenCode/Anomaly page.
_SUBSCRIPTION_NULL_MARKERS = (
    "subscription:null",
    "subscriptionid:null",
    "subscriptionplan:null",
    '"subscription":null',
    "litesubscriptionid:null",
)
# SolidJS SSR session/user markers present on an authenticated app shell.
_SSR_SESSION_MARKERS = ("session.get", "useremail", "workspaces[", "_$HY")
# Markers typical of a signed-out marketing / login wall.
_LOGIN_PAGE_MARKERS = (
    "sign in", "log in", "log-in", "login", "anoma.ly", "social-share.png",
)


def _classify_subscription_absence(text: str) -> tuple[str, str]:
    """Classify why a /go page contained no subscription usage data.

    Returns ``(reason, detail)`` with ``reason`` one of:
      - ``no_subscription`` — authenticated shell, subscription object is null
        (expired / not renewed); renew to restore usage bars.
      - ``auth``            — page looks like a signed-out / login wall; the
        auth cookie is missing or no longer valid.
      - ``parse``           — layout drift; nothing we recognise (default).
    """
    low = text.lower()
    if any(m in low for m in _SUBSCRIPTION_NULL_MARKERS):
        return (
            "no_subscription",
            "No active OpenCode subscription (expired or not renewed). "
            "Renew at opencode.ai to restore usage tracking.",
        )
    has_ssr_session = any(m in low for m in _SSR_SESSION_MARKERS)
    looks_like_login = any(m in low for m in _LOGIN_PAGE_MARKERS)
    if not has_ssr_session and looks_like_login:
        return (
            "auth",
            "OpenCode session is invalid or expired — refresh the auth "
            "cookie in the Usage tab.",
        )
    return (
        "parse",
        "OpenCode changed its usage page layout — subscription usage could "
        "not be read.",
    )


class OpenCodeSubscriptionUnavailable(Exception):
    """Raised when the /go page loaded but subscription data is absent.

    Carries a machine-readable ``reason`` (``no_subscription`` / ``auth`` /
    ``parse``) and a user-facing ``detail`` string.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


# ── Billing SSR parsing ────────────────────────────────────────────────────

# Field names seen/most-likely on the billing page's $R blocks. The exact
# shape can vary; we scan broadly and normalise below.
_BILLING_BLOCK_START_RE = re.compile(
    r'lite\.(?:billing|credits|workspace|billing\.get)\.get\["wrk_[^"]+"\]',
    re.DOTALL,
)

# Match `$R[N]={ ... }` blocks (SolidJS SSR). Multi-line, non-greedy.
_R_BLOCK_RE = re.compile(r'\$R\[\d+\]=\{(.*?)\}\s*;', re.DOTALL)

# Credit/balance-ish keys → captured numeric value (any order). Captures the
# field NAME so we can prefer unambiguous credit fields over generic ones.
_CREDIT_VALUE_RE = re.compile(
    r'(availableCredits|available_credits|creditsAvailable|creditBalance'
    r'|balance|available|totalCredits|total_credits|remainingCredits'
    r'|remaining|monthlyCredits|monthly_credits|credits)\s*:\s*'
    r'([+-]?\d+(?:\.\d+)?)'
)

# Plan name (string or quoted).
_PLAN_RE = re.compile(
    r'(?:plan|planId|plan_name|tier|currentPlan)\s*:\s*"?([A-Za-z0-9_ .-]+)"?'
)

# Unambiguous "available credits" field names — these win over generic
# ``balance``/``available``/``credits`` which may hold token counts, ledger
# totals, or other large numbers on the billing page.
_CREDIT_SPECIFIC_KEYS = frozenset({
    "availablecredits", "available_credits", "creditsavailable", "creditbalance",
})

# The live OpenCode billing page stores the available balance in the
# ``billing.get`` object as an INTEGER in 8-decimal fixed-point:
#   balance: 949260397  ==  $9.49260397  (949260397 * 1e-8)
# ``balance`` is therefore the real credit figure — just scaled — NOT a
# token/ledger number, so we scale it by 1e-8 into USD.
_BALANCE_FIXED_POINT = 1e-8

# Plausible upper bound for USD credit balances. Generic keys (available,
# credits) on the OpenCode billing page can carry huge non-USD numbers (token
# ledgers etc.); anything above this is treated as not-credits. (``balance``
# is handled separately — it is scaled, not rejected.)
_MAX_PLAUSIBLE_CREDITS = 1_000_000.0


def _parse_ssr_billing(text: str) -> Optional[dict]:
    """Parse the SolidJS SSR $R blocks for billing/credit data.

    Returns ``{"available_credits": float, "currency": "USD", "plan": str}``
    or ``None`` when nothing credit-like is found.

    Candidate values are collected across all ``$R[N]`` blocks (scoped to the
    billing block when present), then:
      1. unambiguous credit fields (``availableCredits``/``creditBalance``/…)
         are preferred,
      2. ``balance`` (the billing page's 8-decimal fixed-point credit balance)
         is scaled by 1e-8 and treated as a strong candidate,
      3. generic keys (``available``/``credits``) are a final fallback, with
         implausibly-large values (token ledgers) skipped.
    """
    if not text:
        return None

    # Scope the search to the billing block when present, else whole page.
    block_start = _BILLING_BLOCK_START_RE.search(text)
    if block_start:
        end_pos = min(len(text), block_start.end() + 20000)
        block = text[block_start.start():end_pos]
    else:
        block = text

    specific: list[float] = []
    balance_scaled: list[float] = []
    generic: list[float] = []
    skipped: list[tuple[str, float]] = []
    plan: Optional[str] = None

    def _scan(segment: str) -> None:
        nonlocal plan
        for m in _CREDIT_VALUE_RE.finditer(segment):
            key = m.group(1).lower()
            val = float(m.group(2))
            if val <= 0:
                continue
            if key in _CREDIT_SPECIFIC_KEYS:
                specific.append(val)
            elif key == "balance":
                # billing.get: 8-decimal fixed point → USD.
                balance_scaled.append(val * _BALANCE_FIXED_POINT)
            elif val > _MAX_PLAUSIBLE_CREDITS:
                skipped.append((key, val))
                continue
            else:
                generic.append(val)
        if plan is None:
            pm = _PLAN_RE.search(segment)
            if pm:
                plan = pm.group(1).strip()

    for m in _R_BLOCK_RE.finditer(block):
        _scan(m.group(1))
    # Top-level (non-$R) keys — some SSR shapes inline them.
    _scan(block)

    if skipped:
        logger.warning(
            "opencode_billing_skipped_implausible",
            keys=[(k, v) for k, v in skipped],
        )

    available: Optional[float] = None
    if specific:
        available = specific[-1]
    elif balance_scaled:
        available = balance_scaled[-1]
    elif generic:
        available = generic[-1]

    if available is None:
        _debug_billing_failure(block)
        return None

    # Currency is always USD on opencode.ai.
    return {
        "available_credits": round(available, 2),
        "currency": "USD",
        "plan": plan,
    }


def _debug_billing_failure(text: str) -> None:
    """Log a short snippet of the billing SSR text for diagnosis (debug only)."""
    for kw in ("availableCredits", "creditBalance", "balance", "credits",
               "plan", "billing"):
        idx = text.find(kw)
        if idx >= 0:
            start = max(0, idx - 120)
            snippet = text[start:idx + 200].replace("\n", "\\n").replace("\r", "")
            logger.debug("opencode_billing_ssr_context", keyword=kw, snippet=snippet)
            return
    logger.debug("opencode_billing_ssr_no_credits_found")


# ── Public API ─────────────────────────────────────────────────────────────

def fetch_billing(cookie: Optional[str], workspace_id: Optional[str] = None) -> Optional[BillingSnapshot]:
    """Fetch OpenCode available credits from the /workspace/{id}/billing page.

    ``workspace_id``, if provided, skips the discover step (preferred when
    the ID is already known, e.g. from the dashboard or env var).

    Returns a ``BillingSnapshot`` or ``None`` when the cookie is missing,
    unreachable, or no credit data is found.
    """
    if not cookie or not cookie.strip():
        logger.debug("opencode_cookie_missing")
        return None

    cookie = cookie.strip()
    headers = _base_headers(cookie)

    # ── Step 1: discover workspace ID ──────────────────────────────────
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

    # ── Step 2: fetch /billing page and parse SSR credit data ──────────
    try:
        headers["Referer"] = f"{_OPENCODE_BASE}/workspace/{workspace_id}"
        url = f"{_OPENCODE_BASE}/workspace/{workspace_id}/billing"
        raw = _http_get(url, headers)
    except (URLError, OSError) as exc:
        logger.warning("opencode_billing_page_fetch_failed", error=str(exc))
        return None

    result = _parse_ssr_billing(raw)
    if result is None:
        logger.warning(
            "opencode_billing_parse_failed",
            page_len=len(raw),
        )
        return None

    return BillingSnapshot(
        available_credits=result.get("available_credits"),
        currency=result.get("currency", "USD"),
        plan=result.get("plan"),
        workspace_id=workspace_id,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


def fetch_billing_dict(cookie: Optional[str], workspace_id: Optional[str] = None) -> Optional[dict]:
    """Same as ``fetch_billing`` but returns a plain dict (or None)."""
    snap = fetch_billing(cookie, workspace_id=workspace_id)
    if snap is None:
        return None
    return {
        "available_credits": snap.available_credits,
        "balance": snap.available_credits,  # generic contract: {balance, currency}
        "currency": snap.currency,
        "plan": snap.plan,
        "workspace_id": snap.workspace_id,
        "fetched_at": snap.fetched_at,
    }

def fetch_subscription(cookie: Optional[str], workspace_id: Optional[str] = None) -> Optional[SubscriptionSnapshot]:
    """Fetch OpenCode subscription usage from the /go page SSR data.

    ``workspace_id``, if provided, skips the discover step (preferred when
    the ID is already known, e.g. from the dashboard or env var).

    Returns a ``SubscriptionSnapshot`` or ``None`` when the cookie is
    missing, unreachable, or no subscription data is found in the page.
    """
    if not cookie or not cookie.strip():
        logger.debug("opencode_cookie_missing")
        return None

    cookie = cookie.strip()
    headers = _base_headers(cookie)

    # ── Step 1: discover workspace ID ──────────────────────────────────
    # Priority: explicit argument → scrape as a fallback
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
        reason, detail = _classify_subscription_absence(raw)
        logger.warning(
            "opencode_subscription_unavailable",
            reason=reason,
            page_len=len(raw),
        )
        if reason in ("no_subscription", "auth"):
            raise OpenCodeSubscriptionUnavailable(reason, detail)
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


def fetch_subscription_dict(cookie: Optional[str], workspace_id: Optional[str] = None) -> Optional[dict]:
    """Same as ``fetch_subscription`` but returns a plain dict (or None)."""
    snap = fetch_subscription(cookie, workspace_id=workspace_id)
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
