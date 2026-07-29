"""Cost tracking plugin for DeepSeek API.

Provides:
  - Model-specific pricing (deepseek-v4-pro, deepseek-v4-flash)
  - Cost calculation from token usage
  - Account balance query via the /user/balance endpoint
  - Usage history via a local cache table (populated by the pipeline)
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from .base import CostPlugin, get_registry

# ── Official pricing (per 1M tokens, USD) ─────────────────────────────────
# Source: https://api-docs.deepseek.com/quick_start/pricing (verified June 2026)
_PRICING: dict[str, dict[str, float]] = {
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

_BALANCE_URL = "https://api.deepseek.com/user/balance"

# How much to subtract from the balance response timestamp before we
# consider the cached value stale (seconds).
_BALANCE_CACHE_TTL = 300  # 5 minutes


class DeepSeekCostPlugin(CostPlugin):
    """Cost tracking for DeepSeek official API."""

    @property
    def provider_name(self) -> str:
        return "deepseek"

    @property
    def preset(self) -> Optional[dict]:
        return {
            "api_base": "https://api.deepseek.com/v1",
            "models": self.get_supported_models(),
        }

    def get_supported_models(self) -> list[str]:
        return list(_PRICING.keys())

    def get_pricing(self, model: str) -> Optional[dict]:
        return _PRICING.get(model)

    def calculate_cost(self, model: str, usage: dict) -> Optional[float]:
        pricing = _PRICING.get(model)
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

    # ── Balance query ────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._balance_cache: Optional[dict] = None
        self._balance_cached_at: float = 0.0

    def _api_key(self) -> Optional[str]:
        return os.environ.get("DEEPSEEK_API_KEY")

    def fetch_balance(self) -> Optional[dict]:
        now = time.time()
        if self._balance_cache and (now - self._balance_cached_at) < _BALANCE_CACHE_TTL:
            return self._balance_cache

        api_key = self._api_key()
        if not api_key:
            return None

        try:
            req = Request(
                _BALANCE_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                },
            )
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (URLError, OSError, json.JSONDecodeError, ValueError) as exc:
            from ..logging_config import get_logger
            get_logger("lcp.cost.deepseek").warning("balance_query_failed", error=str(exc))
            return None

        # Normalise: the API may return "balance" or "total_balance"
        balance = data.get("balance") or data.get("total_balance") or 0.0
        result = {
            "balance": float(balance),
            "currency": data.get("currency", "USD"),
            "total_granted": data.get("total_granted"),
            "raw": data,
        }
        self._balance_cache = result
        self._balance_cached_at = now
        return result


# ── Auto-register ──────────────────────────────────────────────────────────
_registry = get_registry()
_registry.register(DeepSeekCostPlugin())
