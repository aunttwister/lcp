"""Cost tracking plugin for OpenCode.

Reads OpenCode's local SQLite database (``~/.local/share/opencode/opencode.db``)
to extract token usage and cost data.  OpenCode stores per-message metadata
in its ``message`` table — the ``data`` column contains a JSON payload with
``tokens``, ``cost``, and ``model`` fields.

Pricing is the same as DeepSeek (OpenCode uses deepseek models under the hood).
The plugin also reports the free OpenCode-hosted models (qwen3-coder, etc.)
as zero-cost.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .base import CostPlugin, get_registry

# ── Pricing ─────────────────────────────────────────────────────────────────
# OpenCode's paid models are DeepSeek under the hood; free hosted models cost $0.
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

# Free OpenCode-hosted models
_FREE_MODELS = frozenset({
    "qwen3-coder",
    "glm-4.7-free",
    "minimax-m2.1-free",
})


def _default_db_path() -> str:
    """Resolve the default OpenCode SQLite path (~/.local/share/opencode/opencode.db)."""
    data_home = os.environ.get(
        "XDG_DATA_HOME",
        os.path.join(os.path.expanduser("~"), ".local", "share"),
    )
    return os.path.join(data_home, "opencode", "opencode.db")


class OpenCodeCostPlugin(CostPlugin):
    """Cost tracking for OpenCode — reads local SQLite store."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or _default_db_path()

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
        """Calculate cost using OpenCode/DeepSeek pricing.

        Free models always return 0.0; paid models follow DeepSeek rates.
        """
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

    # ── Usage history (from local SQLite) ──────────────────────────────────

    def fetch_usage(self,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> list[dict]:
        """Read aggregated usage from the local OpenCode SQLite database.

        Returns daily aggregates for assistant messages that have token data.
        """
        db_path = self._db_path
        if not os.path.isfile(db_path):
            return []

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Build filter clause
            filters: list[str] = []
            params: list[Any] = []
            if start_date:
                filters.append("DATE(m.time_created, 'unixepoch') >= ?")
                params.append(start_date)
            if end_date:
                filters.append("DATE(m.time_created, 'unixepoch') <= ?")
                params.append(end_date)

            where = " AND ".join(filters) if filters else "1=1"

            query = f"""
                SELECT
                    DATE(m.time_created, 'unixepoch') AS day,
                    m.data
                FROM message m
                WHERE {where}
                ORDER BY m.time_created ASC
            """
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
        except (sqlite3.Error, OSError) as exc:
            from ..logging_config import get_logger
            get_logger("lcp.cost.opencode").warning(
                "db_read_failed", path=db_path, error=str(exc)
            )
            return []

        # Aggregate by day (only assistant messages with token data)
        daily: dict[str, dict[str, Any]] = {}
        for row in rows:
            try:
                msg = json.loads(row["data"])
            except (json.JSONDecodeError, TypeError):
                continue

            if msg.get("role") != "assistant":
                continue

            tokens = msg.get("tokens")
            if not tokens:
                continue

            day: str = row["day"]
            model_id: str = (
                msg.get("model", {}).get("modelID")
                or msg.get("modelID")
                or "unknown"
            )
            provider_id: str = (
                msg.get("model", {}).get("providerID")
                or msg.get("providerID")
                or "opencode"
            )
            msg_cost: float = msg.get("cost") or self._calc_msg_cost(
                tokens, model_id
            )

            if day not in daily:
                daily[day] = {
                    "date": day,
                    "model": model_id,
                    "provider": provider_id,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cache_hit_tokens": 0,
                    "cache_miss_tokens": 0,
                    "cost": 0.0,
                    "request_count": 0,
                }

            d = daily[day]
            input_tok = tokens.get("input", 0)
            output_tok = tokens.get("output", 0)
            cache_read = tokens.get("cache", {}).get("read", 0)
            cache_write = tokens.get("cache", {}).get("write", 0)

            d["prompt_tokens"] += input_tok + cache_write
            d["completion_tokens"] += output_tok
            d["cache_hit_tokens"] += cache_read
            # cache_miss is prompt_tokens minus cache_hit (approximate)
            d["cache_miss_tokens"] += input_tok
            d["cost"] += msg_cost
            d["request_count"] += 1

        # Round costs for output
        for d in daily.values():
            d["cost"] = round(d["cost"], 8)

        return list(daily.values())

    @staticmethod
    def _calc_msg_cost(tokens: dict, model_id: str) -> float:
        """Cost for a single message's tokens."""
        pricing = _OPENCODE_PRICING.get(model_id)
        if not pricing:
            return 0.0
        input_tok = tokens.get("input", 0)
        output_tok = tokens.get("output", 0)
        cache_read = tokens.get("cache", {}).get("read", 0)
        return (
            (input_tok / 1_000_000) * pricing["cache_miss"]
            + (cache_read / 1_000_000) * pricing["cache_hit"]
            + (output_tok / 1_000_000) * pricing["output"]
        )

    # ── Balance (not available — OpenCode doesn't expose API balance) ──────

    def fetch_balance(self) -> Optional[dict]:
        return None


# ── Auto-register ──────────────────────────────────────────────────────────
_registry = get_registry()
_registry.register(OpenCodeCostPlugin())
