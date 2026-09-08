"""Cost tracking plugin for llama.cpp (local inference).

llama.cpp is a self-hosted local inference engine — there is no external
billing API to query.  The plugin acts as a local token tracker:

  - Records every request's token counts in memory and optionally persists
    to a local JSON file for continuity across restarts.
  - Reports zero monetary cost (local hardware).
  - Provides an accumulated ``fetch_usage()`` view for the dashboard.

Persistent state file location (if ``persist_path`` is set):
  ``~/.local/share/lcp/llamacpp-usage.json``
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

from .base import CostPlugin, get_registry


def _default_persist_path() -> str:
    data_home = os.environ.get(
        "XDG_DATA_HOME",
        os.path.join(os.path.expanduser("~"), ".local", "share"),
    )
    return os.path.join(data_home, "lcp", "llamacpp-usage.json")


def _fmt_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    return str(n)


class LlamaCppCostPlugin(CostPlugin):
    """Local token tracker for llama.cpp.

    All costs are zero (self-hosted).  The plugin accumulates daily token
    counts and exposes them via ``fetch_usage()`` so the dashboard can
    display "tokens served locally" alongside paid-provider costs.

    Thread-safe via internal lock.
    """

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self._persist_path = persist_path or _default_persist_path()
        self._lock = Lock()
        # In-memory daily accumulator: { "YYYY-MM-DD": { model: usage_dict } }
        self._daily: dict[str, dict[str, dict]] = {}
        self._load_persisted()

    @property
    def provider_name(self) -> str:
        return "llamacpp"

    @property
    def preset(self) -> Optional[dict]:
        return {
            "api_base": "http://localhost:8080/v1",
            "models": [],  # llama.cpp can serve any model
        }

    def get_supported_models(self) -> list[str]:
        # llama.cpp can serve any model — return empty = "all models"
        return []

    def get_pricing(self, model: str) -> Optional[dict]:
        # All local — zero cost
        return {"cache_hit": 0.0, "cache_miss": 0.0, "output": 0.0}

    def calculate_cost(self, model: str, usage: dict) -> Optional[float]:
        return 0.0

    def discover_models(self, api_base: str) -> list[dict] | None:
        """Query llama.cpp's /v1/models and extract metadata from 'meta' sub-object."""
        import json, urllib.request, ssl
        base = api_base.rstrip("/")
        urls = [f"{base}/models"]
        if "/v1" not in base.lower():
            urls.append(f"{base}/v1/models")
        for url in urls:
            try:
                req = urllib.request.Request(url)
                ctx = ssl.create_default_context()
                with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                    raw = json.loads(resp.read().decode())
                    models_raw = raw.get("data") or raw.get("models") or []
                    models = []
                    for m in models_raw:
                        if not isinstance(m, dict):
                            models.append({"id": str(m)})
                            continue
                        entry = {"id": m.get("id") or m.get("name") or str(m)}
                        for f in ("created", "owned_by", "object"):
                            if m.get(f):
                                entry[f] = m[f]
                        meta = m.get("meta", {})
                        if isinstance(meta, dict):
                            if meta.get("n_ctx"):
                                entry["context_length"] = meta["n_ctx"]
                            if meta.get("n_ctx_train"):
                                entry["context_train"] = meta["n_ctx_train"]
                            if meta.get("n_params"):
                                entry["parameters"] = _fmt_params(meta["n_params"])
                            if meta.get("ftype"):
                                entry["quantization"] = meta["ftype"]
                            if meta.get("size"):
                                entry["size_bytes"] = meta["size"]
                        models.append(entry)
                    return models
            except Exception:
                continue
        return None

    # ── Token recording ───────────────────────────────────────────────────

    def record_tokens(self,
                      model: str,
                      prompt_tokens: int,
                      completion_tokens: int,
                      cache_hit_tokens: int = 0,
                      latency_ms: int = 0) -> None:
        """Record a request's token counts and latency."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            day_models = self._daily.setdefault(today, {})
            entry = day_models.setdefault(model, {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cache_hit_tokens": 0,
                "request_count": 0,
                # Latency accumulation for tokens/sec (total TPS):
                #   tps = (prompt + completion) / (total_latency_ms / 1000)
                "total_latency_ms": 0,
            })
            entry["prompt_tokens"] += prompt_tokens
            entry["completion_tokens"] += completion_tokens
            entry["cache_hit_tokens"] += cache_hit_tokens
            entry["request_count"] += 1
            entry["total_latency_ms"] += latency_ms
        self._persist()

    # ── Usage history ─────────────────────────────────────────────────────

    def _entry(self, entry: dict) -> dict:
        """Normalize a raw stored entry to include latency fields (default 0).

        Backward-compatible with JSON files persisted before latency tracking
        existed (no ``total_latency_ms`` key).
        """
        out = dict(entry)
        out.setdefault("total_latency_ms", 0)
        return out

    def fetch_usage(self,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> list[dict]:
        """Return daily token usage for all models."""
        result: list[dict] = []
        with self._lock:
            for day, models in sorted(self._daily.items()):
                if start_date and day < start_date:
                    continue
                if end_date and day > end_date:
                    continue
                for model, raw_entry in models.items():
                    entry = self._entry(raw_entry)
                    result.append({
                        "date": day,
                        "model": model,
                        "provider": "llamacpp",
                        "prompt_tokens": entry["prompt_tokens"],
                        "completion_tokens": entry["completion_tokens"],
                        "cache_hit_tokens": entry["cache_hit_tokens"],
                        "cache_miss_tokens": entry["prompt_tokens"],
                        "cost": 0.0,
                        "request_count": entry["request_count"],
                        "total_latency_ms": entry["total_latency_ms"],
                        "total_tokens": (entry["prompt_tokens"]
                                         + entry["completion_tokens"]),
                    })
        from ..logging_config import get_logger
        get_logger("lcp.cost.llamacpp").debug("usage_fetched", days=len(result))
        return result

    def fetch_metrics(self,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> dict:
        """Return aggregate local-inference metrics (token-centric, no cost).

        Computed from the recorded daily usage:
          - total tokens in / out / cache-hit
          - total requests
          - average latency per request (ms)
          - TOTAL throughput in tokens/sec
            = (prompt_tokens + completion_tokens) / (total_latency_sec)

        Includes a per-model breakdown and the date range actually covered.
        All cost fields are zero (local hardware) — the dashboard for this
        provider is token/throughput focused.
        """
        rows = self.fetch_usage(start_date=start_date, end_date=end_date)
        prompt = sum(r["prompt_tokens"] for r in rows)
        completion = sum(r["completion_tokens"] for r in rows)
        cache_hit = sum(r["cache_hit_tokens"] for r in rows)
        requests = sum(r["request_count"] for r in rows)
        latency_ms = sum(r["total_latency_ms"] for r in rows)
        total_tokens = prompt + completion

        per_model: dict[str, dict] = {}
        for r in rows:
            m = per_model.setdefault(r["model"], {
                "prompt_tokens": 0, "completion_tokens": 0,
                "cache_hit_tokens": 0, "request_count": 0,
                "total_latency_ms": 0,
            })
            m["prompt_tokens"] += r["prompt_tokens"]
            m["completion_tokens"] += r["completion_tokens"]
            m["cache_hit_tokens"] += r["cache_hit_tokens"]
            m["request_count"] += r["request_count"]
            m["total_latency_ms"] += r["total_latency_ms"]

        latency_sec = latency_ms / 1000.0
        tps = round(total_tokens / latency_sec, 1) if latency_sec > 0 else 0.0
        return {
            "provider": "llamacpp",
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total_tokens,
            "cache_hit_tokens": cache_hit,
            "requests": requests,
            "total_latency_ms": latency_ms,
            "avg_latency_ms": round(latency_ms / requests, 1) if requests else 0,
            "tokens_per_sec": tps,
            "per_model": per_model,
        }

    def fetch_balance(self) -> Optional[dict]:
        # Local inference — no balance to query
        return None

    # ── Persistence ───────────────────────────────────────────────────────

    def _persist(self) -> None:
        if not self._persist_path:
            return
        try:
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with open(path, "w") as f:
                    json.dump(self._daily, f, indent=2)
        except OSError as exc:
            from ..logging_config import get_logger
            get_logger("lcp.cost.llamacpp").warning(
                "persist_failed", path=self._persist_path, error=str(exc)
            )

    def _load_persisted(self) -> None:
        if not self._persist_path:
            return
        path = Path(self._persist_path)
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            with self._lock:
                self._daily = data
        except (OSError, json.JSONDecodeError) as exc:
            from ..logging_config import get_logger
            get_logger("lcp.cost.llamacpp").warning(
                "load_persisted_failed", path=self._persist_path, error=str(exc)
            )

    def on_shutdown(self) -> None:
        self._persist()


# ── Auto-register ──────────────────────────────────────────────────────────
_registry = get_registry()
_registry.register(LlamaCppCostPlugin())
