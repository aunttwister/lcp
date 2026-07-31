"""Cost tracking plugin for OpenCode.

Uses the gateway's own ``requests`` table for cost history (every routed
request is logged there with ``provider='opencode'``) and optionally polls
the OpenCode web API for subscription usage (5-hour / weekly percentages
and reset countdowns) via ``OPENCODE_COOKIE`` env var.

Pricing is the same as DeepSeek (OpenCode uses deepseek models under the
hood).  The plugin also reports the free OpenCode-hosted models as zero-cost.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func

from ..logging_config import get_logger
from .base import CostPlugin

logger = get_logger("lcp.cost.opencode")

# ── Pricing ─────────────────────────────────────────────────────────────────
_OPENCODE_PRICING: dict[str, dict[str, float]] = {
    "deepseek-v4-pro": {
        "cache_hit": 0.003625,
        "cache_miss": 0.435,
        "output": 0.87,
    },
    "deepseek-v4-flash": {
        "cache_hit": 0.0028,
        "cache_miss": 0.14,
        "output": 0.28,
    },
}

_FREE_MODELS = frozenset({
    "qwen3-coder",
    "glm-4.7-free",
    "minimax-m2.1-free",
})


class OpenCodeCostPlugin(CostPlugin):
    """Cost tracking for OpenCode.

    Cost history comes from the gateway ``requests`` table (single source
    of truth for every routed request).  Subscription usage (usage % and
    reset countdowns) comes from the OpenCode web API when the
    ``OPENCODE_COOKIE`` env var is set.

    Requires a SQLAlchemy *engine* for gateway DB queries — pass it to the
    constructor or call ``set_engine()`` before any query methods are used.
    """

    def __init__(self, engine: Any = None) -> None:
        self._engine = engine

    def set_engine(self, engine: Any) -> None:
        """Bind the gateway SQLAlchemy engine for DB queries."""
        self._engine = engine

    # ── Identity ───────────────────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return "opencode"

    @property
    def preset(self) -> Optional[dict]:
        return {
            "api_base": "https://opencode.ai/zen/go/v1",
            "models": self.get_supported_models(),
        }

    def get_supported_models(self) -> list[str]:
        return list(_OPENCODE_PRICING.keys()) + sorted(_FREE_MODELS)

    # ── Pricing ────────────────────────────────────────────────────────────

    def get_pricing(self, model: str) -> Optional[dict]:
        if model in _FREE_MODELS:
            return {"cache_hit": 0.0, "cache_miss": 0.0, "output": 0.0}
        return _OPENCODE_PRICING.get(model)

    def calculate_cost(self, model: str, usage: dict) -> Optional[float]:
        """Calculate cost using OpenCode/DeepSeek pricing."""
        if model in _FREE_MODELS:
            return 0.0

        pricing = _OPENCODE_PRICING.get(model)
        if pricing is None:
            return None

        cache_hit = usage.get("prompt_cache_hit_tokens", 0)
        cache_miss = usage.get(
            "prompt_cache_miss_tokens",
            usage.get("prompt_tokens", 0) - cache_hit,
        )
        output = usage.get("completion_tokens", 0)

        if cache_hit == 0 and cache_miss == 0:
            cache_miss = usage.get("prompt_tokens", 0)

        cost = (
            (cache_hit / 1_000_000) * pricing["cache_hit"]
            + (cache_miss / 1_000_000) * pricing["cache_miss"]
            + (output / 1_000_000) * pricing["output"]
        )
        return round(cost, 8)

    # ── Database helpers ───────────────────────────────────────────────────

    def _ensure_engine(self) -> Any:
        """Return the engine or raise a clear error."""
        if self._engine is None:
            raise RuntimeError(
                "OpenCode plugin has no gateway engine — call set_engine() "
                "or pass engine= to the constructor before querying."
            )
        return self._engine

    def _gw_session(self):
        """Context manager returning a session bound to the gateway engine."""
        from ..models import get_session
        return get_session(self._ensure_engine())

    # ── Usage history (from gateway requests table) ────────────────────────

    def fetch_usage(self,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> list[dict]:
        """Return daily aggregates for opencode from the gateway DB."""
        if self._engine is None:
            return []

        from ..models import Request as RequestModel

        try:
            with self._gw_session() as session:
                q = session.query(
                    func.substr(RequestModel.timestamp, 1, 10).label("day"),
                    RequestModel.model,
                    RequestModel.provider,
                    func.coalesce(func.sum(RequestModel.prompt_tokens), 0).label("prompt_tokens"),
                    func.coalesce(func.sum(RequestModel.completion_tokens), 0).label("completion_tokens"),
                    func.coalesce(func.sum(RequestModel.cache_hit_tokens), 0).label("cache_hit_tokens"),
                    func.coalesce(func.sum(RequestModel.cache_miss_tokens), 0).label("cache_miss_tokens"),
                    func.coalesce(func.sum(RequestModel.cost), 0).label("cost"),
                    func.count(RequestModel.id).label("request_count"),
                ).filter(
                    RequestModel.provider == "opencode",
                    RequestModel.success == 1,
                )

                if start_date:
                    q = q.filter(RequestModel.timestamp >= start_date)
                if end_date:
                    q = q.filter(RequestModel.timestamp <= (end_date + "T23:59:59"))

                q = q.group_by("day").order_by("day")
                rows = q.all()

            result: list[dict] = []
            for r in rows:
                result.append({
                    "date": r.day,
                    "model": r.model or "unknown",
                    "provider": r.provider or "opencode",
                    "prompt_tokens": int(r.prompt_tokens),
                    "completion_tokens": int(r.completion_tokens),
                    "cache_hit_tokens": int(r.cache_hit_tokens),
                    "cache_miss_tokens": int(r.cache_miss_tokens),
                    "cost": round(float(r.cost), 8),
                    "request_count": int(r.request_count),
                })

            logger.debug("usage_fetched", days=len(result))
            return result
        except Exception as exc:
            logger.warning("usage_query_failed", error=str(exc))
            return []

    # ── Balance (not available) ────────────────────────────────────────────

    def fetch_balance(self) -> Optional[dict]:
        return None

    # ── Rich summary — daily / weekly / monthly from gateway DB ────────────

    def fetch_summary(self) -> Optional[dict]:
        """Return daily, weekly, and monthly cost aggregates from the gateway DB."""
        if self._engine is None:
            return None

        from ..models import Request as RequestModel

        try:
            with self._gw_session() as session:
                now = datetime.now(timezone.utc)
                thresholds: dict[str, str] = {
                    "daily": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "weekly": (now - timedelta(days=7)).strftime("%Y-%m-%d"),
                    "monthly": now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%S"),
                }

                result: dict[str, dict] = {}
                for period_label, cutoff in thresholds.items():
                    row = session.query(
                        func.coalesce(
                            func.sum(RequestModel.prompt_tokens + RequestModel.completion_tokens),
                            0,
                        ).label("tokens"),
                        func.coalesce(func.sum(RequestModel.cost), 0).label("cost"),
                        func.count(RequestModel.id).label("requests"),
                    ).filter(
                        RequestModel.provider == "opencode",
                        RequestModel.success == 1,
                        RequestModel.timestamp >= cutoff,
                    ).first()

                    if row:
                        result[period_label] = {
                            "tokens": int(row.tokens),
                            "cost": round(float(row.cost), 8),
                            "requests": int(row.requests),
                        }
                    else:
                        result[period_label] = {"tokens": 0, "cost": 0.0, "requests": 0}

            return result
        except Exception as exc:
            logger.warning("summary_query_failed", error=str(exc))
            return None

    # ── Subscription (from OpenCode web API) ───────────────────────────────

    def fetch_subscription(self) -> Optional[dict]:
        """Fetch subscription usage snapshot from the OpenCode web API.

        Requires ``OPENCODE_COOKIE`` env var.  Returns::

            {"rolling_pct": 17.0, "weekly_pct": 75.0,
             "rolling_reset_sec": 5944, "weekly_reset_sec": 278201}

        Returns ``None`` when the cookie is missing, invalid, or the API
        is unreachable.
        """
        try:
            from .opencode_api import fetch_subscription_dict
            cookie = os.environ.get("OPENCODE_COOKIE")
            if not cookie:
                logger.debug("opencode_cookie_not_configured")
                return None
            return fetch_subscription_dict(cookie)
        except Exception as exc:
            logger.warning("subscription_fetch_failed", error=str(exc))
            return None

