"""LLM Control Plane — HTTP server entry point.

Routes:
  /{profile}/chat/completions  — profile-scoped
  /{profile}/v1/chat/completions
  /health                       — health check + provider status
  /v1/models                    — model list (OpenAI compat)
  /errors                       — recent errors
  /cache/stats                  — prompt cache statistics
  /metrics                      — Prometheus-compatible metrics
  /export                       — CSV cost data export
  /                             — dashboard
"""

import json
import os
import re
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from .config import get_config, init_config
from .logging_config import setup_logging, get_logger
from .exceptions import (
    AllProvidersFailedError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ToolBlockedError,
)
from .models import get_engine, get_session, Request as RequestModel
from .cost_estimator import estimate_from_request
from .prompt_cache import get_prompt_cache
from .token_verifier import get_token_verifier
from .router import get_dynamic_router

# ── Initialize ───────────────────────────────────────────────────────────────

logger = get_logger("lcp.server")

# Tool names that Hermes agents can request
KNOWN_HERMES_TOOLS = {
    "read_file", "write_file", "patch", "search_files", "terminal",
    "execute_code", "memory", "session_search", "process", "delegate_task",
    "send_message", "skill_manage", "todo", "vision_analyze", "web",
    "cronjob", "text_to_speech",
}


# ── Tool Stripping ───────────────────────────────────────────────────────────

def strip_forbidden_tools(body: dict, forbidden: list[str] | None) -> tuple[dict, list[str]]:
    """Remove forbidden tools from the request body. Returns (modified_body, blocked_tools)."""
    if forbidden is None:
        # ALL tools forbidden — strip everything
        blocked = []
        if "tools" in body and body["tools"]:
            blocked = [t.get("function", {}).get("name", "unknown") for t in body["tools"]]
            body["tools"] = []
        return body, blocked

    if not forbidden or "tools" not in body or not body["tools"]:
        return body, []

    blocked = []
    kept = []
    for tool in body["tools"]:
        name = tool.get("function", {}).get("name", "")
        if name in forbidden:
            blocked.append(name)
        else:
            kept.append(tool)

    body["tools"] = kept
    return body, blocked


# ── Cost Calculation ─────────────────────────────────────────────────────────

def calculate_cost(provider: str, model: str, body: dict, response_body: dict | None,
                   config) -> dict:
    """Calculate token usage and cost from request+response."""
    pricing = config.get_pricing(provider, model)

    usage = response_body.get("usage", {}) if response_body else {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    cache_hit = usage.get("prompt_cache_hit_tokens", 0)
    cache_miss = usage.get("prompt_cache_miss_tokens", prompt_tokens)

    cache_hit_cost = (cache_hit / 1_000_000) * pricing["cache_hit"]
    cache_miss_cost = (cache_miss / 1_000_000) * pricing["cache_miss"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_hit_tokens": cache_hit,
        "cache_miss_tokens": cache_miss,
        "cost": round(cache_hit_cost + cache_miss_cost + output_cost, 8),
    }


# ── Provider Health / Circuit Breaker ────────────────────────────────────────

_provider_health: dict = {}


def _health_key(provider: str, base_url: str, profile: str) -> tuple:
    return (provider, base_url, profile)


def _get_health(provider: str, base_url: str, profile: str, config) -> dict:
    key = _health_key(provider, base_url, profile)
    if key not in _provider_health:
        _provider_health[key] = {
            "consecutive_failures": 0,
            "last_failure": None,
            "last_success": None,
            "status": "healthy",
            "tripped_until": None,
        }
    return _provider_health[key]


def is_provider_available(provider: str, base_url: str, profile: str, config) -> bool:
    h = _get_health(provider, base_url, profile, config)
    if h["status"] == "healthy":
        return True
    if h["tripped_until"] and time.time() >= h["tripped_until"]:
        return True
    return False


def record_provider_success(provider: str, base_url: str, profile: str, config) -> None:
    h = _get_health(provider, base_url, profile, config)
    h["status"] = "healthy"
    h["consecutive_failures"] = 0
    h["last_success"] = datetime.now(timezone.utc).isoformat()
    h["tripped_until"] = None


def record_provider_failure(provider: str, base_url: str, profile: str, config) -> None:
    cb = config.circuit_breaker
    h = _get_health(provider, base_url, profile, config)
    h["consecutive_failures"] += 1
    h["last_failure"] = datetime.now(timezone.utc).isoformat()
    n = h["consecutive_failures"]
    if n >= cb["failures_dead"]:
        h["status"] = "dead"
        h["tripped_until"] = time.time() + cb["dead_cooldown_seconds"]
    elif n >= cb["failures_degraded"]:
        h["status"] = "degraded"
        h["tripped_until"] = time.time() + cb["degraded_cooldown_seconds"]


# ── Request Forwarding ───────────────────────────────────────────────────────

def forward_request(provider_cfg: dict, body: dict, config) -> tuple[dict, int]:
    """Forward a request to a provider. Returns (response_body_dict, status_code)."""
    api_key = os.environ.get(provider_cfg.get("api_key_env", ""))
    if not api_key:
        # Try resolving via config
        provider_name = provider_cfg["provider"]
        api_key = config.get_provider_key(provider_name)

    url = f"{provider_cfg['base_url']}/chat/completions"
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "LLMControlPlane/1.0",
    }

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            response_body = json.loads(resp.read().decode("utf-8"))
            return response_body, resp.status
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")[:500]
        status = e.code
        if status == 401 or status == 403:
            raise ProviderAuthError(f"Provider {provider_cfg['provider']} rejected auth: {status}")
        elif status == 429:
            raise ProviderRateLimitError(f"Provider {provider_cfg['provider']} rate limited")
        raise ProviderAuthError(f"Provider {provider_cfg['provider']} HTTP {status}: {error_body}")
    except urllib.error.URLError as e:
        raise ProviderTimeoutError(f"Provider {provider_cfg['provider']} unreachable: {e.reason}")


def try_chain(profile_name: str, profile_cfg: dict, body: dict, config) -> tuple[dict, int, str, str]:
    """Try each provider in the chain. Returns (response, status, provider, model)."""
    errors = []
    for step in profile_cfg["chain"]:
        provider_name = step["provider"]
        base_url = step.get("base_url") or config.providers.get(provider_name, {}).get("api_base", "")
        model = step["model"]

        # Check circuit breaker
        if not is_provider_available(provider_name, base_url, profile_name, config):
            logger.warning(
                "provider_skipped_circuit_breaker",
                provider=provider_name,
                profile=profile_name,
            )
            errors.append(f"{provider_name}: circuit breaker open")
            continue

        # Set model in body
        body["model"] = model

        # Add API key env to step config
        step_with_key = {**step, "api_key_env": config.providers[provider_name]["api_key_env"]}

        try:
            resp, status = forward_request(step_with_key, body, config)
            record_provider_success(provider_name, base_url, profile_name, config)
            return resp, status, provider_name, model
        except (ProviderTimeoutError, ProviderAuthError, ProviderRateLimitError) as e:
            record_provider_failure(provider_name, base_url, profile_name, config)
            errors.append(f"{provider_name}: {e}")
            logger.error("provider_failed", provider=provider_name, error=str(e))

    raise AllProvidersFailedError(f"All providers failed for {profile_name}: {'; '.join(errors)}")


# ── Cost Recording ───────────────────────────────────────────────────────────

def record_cost(engine, profile: str, model: str, provider: str, cost_info: dict,
                success: bool, error_type: str | None, tools_blocked: list[str]) -> None:
    """Record cost data to SQLite."""
    with get_session(engine) as session:
        req = RequestModel(
            timestamp=datetime.now(timezone.utc).isoformat(),
            profile=profile,
            model=model,
            provider=provider,
            prompt_tokens=cost_info.get("prompt_tokens", 0),
            completion_tokens=cost_info.get("completion_tokens", 0),
            cache_hit_tokens=cost_info.get("cache_hit_tokens", 0),
            cache_miss_tokens=cost_info.get("cache_miss_tokens", 0),
            cost=cost_info.get("cost", 0),
            latency_ms=cost_info.get("latency_ms", 0),
            success=1 if success else 0,
            error_type=error_type,
            tools_blocked=",".join(tools_blocked) if tools_blocked else None,
        )
        session.add(req)
        session.commit()


# ── HTTP Handler ─────────────────────────────────────────────────────────────

class LCPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for LLM Control Plane."""

    # Class-level references set after server init
    config = None
    engine = None

    def log_message(self, format, *args):
        """Suppress default http.server logging — we use structlog."""
        pass

    def _send_json(self, data: dict, status: int = 200):
        """Send a JSON response."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _resolve_profile(self) -> str | None:
        """Extract profile name from URL path. Returns None for non-profile routes."""
        path = self.path.rstrip("/")
        parts = path.split("/")
        if len(parts) >= 2:
            candidate = parts[1]
            if candidate in self.config.profiles:
                return candidate
        return None

    def _read_body(self) -> dict:
        """Read and parse JSON request body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        raw = self.rfile.read(content_length)
        return json.loads(raw)

    # ── Routes ────────────────────────────────────────────────────────────

    def do_GET(self):
        if self.path == "/" or self.path == "/dashboard":
            self._serve_dashboard()
        elif self.path.endswith("/dashboard"):
            # Per-profile: /l2/dashboard, /l1/dashboard, etc.
            profile = self._resolve_profile()
            if profile:
                self._serve_dashboard(profile_filter=profile)
            else:
                self._serve_dashboard()
        elif self.path == "/health":
            self._serve_health()
        elif self.path == "/v1/models":
            self._serve_models()
        elif self.path.startswith("/errors"):
            self._serve_errors()
        elif self.path == "/cache/stats":
            self._serve_cache_stats()
        elif self.path == "/metrics":
            self._serve_metrics()
        elif self.path == "/export" or self.path.startswith("/export?"):
            self._serve_export()
        elif self.path == "/api/daily-costs":
            self._serve_daily_costs_api()
        elif self.path == "/api/recent-requests":
            self._serve_recent_requests_api()
        elif self.path == "/api/providers":
            self._serve_providers_list()
        elif self.path == "/api/providers/presets":
            self._serve_provider_presets()
        elif self.path == "/api/profiles":
            self._serve_profiles_list()
        elif self.path == "/api/keys":
            self._serve_keys_list()
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        # Config hot-reload check
        self.config.check_reload()

        # Provider API routes
        if self.path == "/api/providers":
            self._serve_provider_create()
            return
        elif self.path == "/api/providers/test":
            self._serve_provider_test()
            return
        elif self.path == "/api/profiles":
            self._serve_profile_create()
            return
        elif self.path == "/api/keys":
            self._serve_key_create()
            return

        # Only handle chat completions
        if "/chat/completions" not in self.path:
            self._send_json({"error": "not found"}, 404)
            return

        profile = self._resolve_profile()
        if profile is None:
            self._send_json({"error": f"unknown profile in path: {self.path}"}, 400)
            return

        profile_cfg = self.config.get_profile(profile)
        if profile_cfg is None:
            self._send_json({"error": f"profile not found: {profile}"}, 400)
            return

        try:
            body = self._read_body()

            # Validate body has messages
            if not isinstance(body.get("messages"), list) or len(body.get("messages", [])) == 0:
                self._send_json({"error": "missing required field: messages"}, 400)
                return

            # Tool stripping
            body, blocked_tools = strip_forbidden_tools(body, profile_cfg.get("forbidden_tools"))

            # Pre-request cost estimation
            pricing = self.config.get_pricing(
                profile_cfg["chain"][0]["provider"], profile_cfg["chain"][0]["model"]
            )
            estimation = estimate_from_request(
                profile_cfg["chain"][0]["model"],
                body.get("messages", []),
                body.get("tools"),
                body.get("max_tokens", 1024),
                pricing,
            )

            # Prompt cache check
            cache = get_prompt_cache()
            primary_model = profile_cfg["chain"][0]["model"]
            cached = cache.get(profile, primary_model, body)
            if cached is not None:
                latency_ms = 1  # near-instant
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-LCP-Cache", "HIT")
                self.send_header("X-Estimated-Cost", str(estimation["estimated_total_cost"]))
                self.end_headers()
                self.wfile.write(json.dumps(cached).encode("utf-8"))
                logger.info("cache_hit_served", profile=profile, model=primary_model)
                return

            # Set estimated cost header (will be sent with response headers later — patched below)
            self._pending_headers = {
                "X-LCP-Cache": "MISS",
                "X-Estimated-Cost": str(estimation["estimated_total_cost"]),
            }

            t0 = time.time()

            # Try provider chain
            response_body, status, provider, model = try_chain(
                profile, profile_cfg, body, self.config
            )

            latency_ms = int((time.time() - t0) * 1000)

            # Cache the response
            cache.set(profile, model, body, response_body)

            # Token verification
            verifier = get_token_verifier()
            verification = verifier.verify(body.get("messages", []), response_body.get("usage", {}))
            if verification["suspicious"]:
                self._pending_headers["X-LCP-Token-Warning"] = (
                    f"suspicious: provider={verification['provider_prompt_tokens']} "
                    f"estimated={verification['estimated_prompt_tokens']} "
                    f"pct={verification['prompt_discrepancy_pct']}"
                )

            # Calculate cost
            cost_info = calculate_cost(provider, model, body, response_body, self.config)
            cost_info["latency_ms"] = latency_ms

            # Record cost
            record_cost(self.engine, profile, model, provider, cost_info, True, None, blocked_tools)

            # Send response with custom headers
            body_bytes = json.dumps(response_body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body_bytes))
            for hdr_name, hdr_val in self._pending_headers.items():
                self.send_header(hdr_name, hdr_val)
            self.end_headers()
            self.wfile.write(body_bytes)

            logger.info(
                "request_complete",
                profile=profile,
                provider=provider,
                model=model,
                cost=round(cost_info["cost"], 6),
                latency_ms=latency_ms,
                tools_blocked=len(blocked_tools),
                cache="MISS",
            )

        except ToolBlockedError as e:
            self._send_json({"error": str(e)}, 403)
        except AllProvidersFailedError as e:
            cost_info = {"prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0,
                         "cache_miss_tokens": 0, "cost": 0, "latency_ms": 0}
            record_cost(self.engine, profile, "unknown", "unknown", cost_info, False, "all_providers_failed", [])
            # Send error response with cost estimate header if available
            body_bytes = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body_bytes))
            if hasattr(self, '_pending_headers') and self._pending_headers:
                estimation_cost = self._pending_headers.get("X-Estimated-Cost", "0")
                self.send_header("X-Estimated-Cost", estimation_cost)
                self.send_header("X-LCP-Cache", "MISS")
            self.end_headers()
            self.wfile.write(body_bytes)
        except Exception as e:
            logger.error("unhandled_error", error=str(e), traceback=traceback.format_exc()[-500:])
            self._send_json({"error": "internal error"}, 500)

    def do_PUT(self):
        self.config.check_reload()
        if self.path.startswith("/api/providers/") and len(self.path.split("/")) == 4:
            provider_name = self.path.split("/")[3]
            self._serve_provider_update(provider_name)
        elif self.path.startswith("/api/chains/") and len(self.path.split("/")) == 4:
            profile = self.path.split("/")[3]
            self._serve_chain_reorder(profile)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_DELETE(self):
        self.config.check_reload()
        if self.path.startswith("/api/providers/") and len(self.path.split("/")) == 4:
            provider_name = self.path.split("/")[3]
            self._serve_provider_delete(provider_name)
        elif self.path.startswith("/api/keys/") and len(self.path.split("/")) == 4:
            key_id = self.path.split("/")[3]
            self._serve_key_delete(key_id)
        else:
            self._send_json({"error": "not found"}, 404)

    # ── Static Endpoints ───────────────────────────────────────────────────

    def _serve_health(self):
        provider_status = {}
        for key, h in _provider_health.items():
            provider, url, profile = key
            tripped_until = None
            if h.get("tripped_until"):
                tripped_until = datetime.fromtimestamp(h["tripped_until"], tz=timezone.utc).isoformat()
            provider_status[f"{provider}/{profile}"] = {
                "status": h["status"],
                "failures": h["consecutive_failures"],
                "last_success": h["last_success"],
                "last_failure": h["last_failure"],
                "base_url": url,
                "tripped_until": tripped_until,
            }
        self._send_json({
            "status": "ok",
            "profiles": list(self.config.profiles.keys()),
            "providers": provider_status,
        })

    def _serve_models(self):
        models = []
        seen = set()
        for prof_name, prof_cfg in self.config.profiles.items():
            for step in prof_cfg["chain"]:
                mid = step["model"]
                if mid not in seen:
                    seen.add(mid)
                    models.append({"id": mid, "object": "model", "owned_by": step["provider"]})
        self._send_json({"object": "list", "data": models})

    def _serve_errors(self):
        """Show recent errors from the database."""
        try:
            with get_session(self.engine) as session:
                recents = (
                    session.query(RequestModel)
                    .filter(RequestModel.success == 0)
                    .order_by(RequestModel.id.desc())
                    .limit(50)
                    .all()
                )
                errors_list = [
                    {
                        "timestamp": r.timestamp,
                        "profile": r.profile,
                        "error_type": r.error_type,
                    }
                    for r in recents
                ]
            self._send_json({"errors": errors_list})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _serve_cache_stats(self):
        """Prompt cache hit/miss statistics."""
        cache = get_prompt_cache()
        self._send_json(cache.stats)

    def _serve_metrics(self):
        """Prometheus-compatible metrics endpoint."""
        try:
            with get_session(self.engine) as session:
                from sqlalchemy import func
                total_req = session.query(func.count(RequestModel.id)).scalar() or 0
                total_cost = session.query(func.sum(RequestModel.cost)).scalar() or 0
                failed = (
                    session.query(func.count(RequestModel.id))
                    .filter(RequestModel.success == 0)
                    .scalar() or 0
                )
        except Exception:
            total_req = 0
            total_cost = 0
            failed = 0

        cache = get_prompt_cache()
        verifier = get_token_verifier()

        lines = [
            "# HELP lcp_requests_total Total requests processed",
            "# TYPE lcp_requests_total counter",
            f"lcp_requests_total {total_req}",
            "# HELP lcp_cost_total Total cost in USD",
            "# TYPE lcp_cost_total gauge",
            f"lcp_cost_total {total_cost}",
            "# HELP lcp_errors_total Total failed requests",
            "# TYPE lcp_errors_total counter",
            f"lcp_errors_total {failed}",
            "# HELP lcp_cache_hits_total Cache hits",
            "# TYPE lcp_cache_hits_total counter",
            f"lcp_cache_hits_total {cache.stats['hits']}",
            "# HELP lcp_cache_misses_total Cache misses",
            "# TYPE lcp_cache_misses_total counter",
            f"lcp_cache_misses_total {cache.stats['misses']}",
            "# HELP lcp_cache_entries Current cache entries",
            "# TYPE lcp_cache_entries gauge",
            f"lcp_cache_entries {cache.stats['entries']}",
            "# HELP lcp_token_warnings_total Token discrepancy warnings",
            "# TYPE lcp_token_warnings_total counter",
            f"lcp_token_warnings_total {verifier.stats['warnings']}",
        ]

        body = "\n".join(lines) + "\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def _serve_export(self):
        """CSV export of cost data."""
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        limit = int(params.get("limit", [1000])[0])
        profile_filter = params.get("profile", [None])[0]

        try:
            with get_session(self.engine) as session:
                q = session.query(RequestModel).order_by(RequestModel.id.desc())
                if profile_filter:
                    q = q.filter(RequestModel.profile == profile_filter)
                rows = q.limit(min(limit, 10000)).all()

            csv_lines = [
                "timestamp,profile,model,provider,prompt_tokens,completion_tokens,"
                "cache_hit_tokens,cache_miss_tokens,cost,latency_ms,success,error_type"
            ]
            for r in rows:
                csv_lines.append(
                    f"{r.timestamp},{r.profile},{r.model},{r.provider},"
                    f"{r.prompt_tokens},{r.completion_tokens},{r.cache_hit_tokens},"
                    f"{r.cache_miss_tokens},{r.cost},{r.latency_ms},{r.success},"
                    f"{r.error_type or ''}"
                )

            body = "\n".join(csv_lines)
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition", "attachment; filename=lcp-export.csv")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _serve_dashboard(self, profile_filter: str | None = None):
        """Server-rendered shadcn-themed dashboard with provider health, daily costs,
        recent requests/errors, Chart.js time-series, and Phase 5/6 metrics."""
        from sqlalchemy import func, case

        try:
            with get_session(self.engine) as session:
                # ── Summary query ──
                summary = session.query(
                    func.coalesce(func.sum(RequestModel.cost), 0).label("total_cost"),
                    func.count(RequestModel.id).label("total_requests"),
                    func.coalesce(func.sum(RequestModel.cache_hit_tokens), 0).label("cache_hits"),
                    func.coalesce(func.sum(RequestModel.cache_miss_tokens), 0).label("cache_misses"),
                    func.coalesce(func.sum(RequestModel.prompt_tokens), 0).label(
                        "prompt_tokens"
                    ),
                    func.coalesce(func.sum(RequestModel.completion_tokens), 0).label(
                        "output_tokens"
                    ),
                )
                if profile_filter:
                    summary = summary.filter(
                        RequestModel.profile == profile_filter
                    )
                summary = summary.first()

                fallback_count = (
                    session.query(func.count(RequestModel.id))
                    .filter(RequestModel.success == 1)
                )
                if profile_filter:
                    fallback_count = fallback_count.filter(
                        RequestModel.profile == profile_filter
                    )
                fallback_count = (
                    fallback_count.filter(RequestModel.provider != "error").scalar() or 0
                )
                total_success = (
                    session.query(func.count(RequestModel.id))
                    .filter(RequestModel.success == 1)
                )
                if profile_filter:
                    total_success = total_success.filter(
                        RequestModel.profile == profile_filter
                    )
                total_success = total_success.scalar() or 0

                cache_hit_tokens = summary.cache_hits or 0
                cache_miss_tokens = summary.cache_misses or 0
                cache_total = cache_hit_tokens + cache_miss_tokens
                cache_hit_rate = (
                    (cache_hit_tokens / cache_total * 100) if cache_total > 0 else 0
                )

                # ── Active days ──
                active_days = (
                    session.query(
                        func.count(func.distinct(func.substr(RequestModel.timestamp, 1, 10)))
                    ).scalar() or 0
                )

                # ── Per-profile cost ──
                profile_rows = (
                    session.query(
                        RequestModel.profile,
                        func.sum(RequestModel.cost).label("total_cost"),
                        func.count(RequestModel.id).label("count"),
                    )
                    .filter(RequestModel.success == 1)
                    .group_by(RequestModel.profile)
                    .all()
                )

                # ── Daily Costs (all time, grouped by date+profile+model+provider) ──
                daily_rows = (
                    session.query(
                        func.substr(RequestModel.timestamp, 1, 10).label("date"),
                        RequestModel.profile,
                        RequestModel.model,
                        RequestModel.provider,
                        func.count(RequestModel.id).label("reqs"),
                        func.sum(
                            case(
                                (RequestModel.provider != "error", 1), else_=0
                            )
                        ).label("fb_count"),
                        func.coalesce(func.sum(RequestModel.cache_hit_tokens), 0).label(
                            "cache_hit"
                        ),
                        func.coalesce(func.sum(RequestModel.cache_miss_tokens), 0).label(
                            "cache_miss"
                        ),
                        func.coalesce(func.sum(RequestModel.completion_tokens), 0).label(
                            "output"
                        ),
                        func.coalesce(func.sum(RequestModel.cost), 0).label("cost"),
                    )
                )
                if profile_filter:
                    daily_rows = daily_rows.filter(
                        RequestModel.profile == profile_filter
                    )
                daily_rows = (
                    daily_rows.group_by(
                        func.substr(RequestModel.timestamp, 1, 10),
                        RequestModel.profile,
                        RequestModel.model,
                        RequestModel.provider,
                    )
                    .order_by(
                        func.substr(RequestModel.timestamp, 1, 10).desc(),
                        RequestModel.profile,
                    )
                    .limit(30)
                    .all()
                )

                # ── Recent Requests ──
                recent_q = (
                    session.query(RequestModel)
                    .order_by(RequestModel.id.desc())
                )
                if profile_filter:
                    recent_q = recent_q.filter(
                        RequestModel.profile == profile_filter
                    )
                recent_rows = recent_q.limit(50).all()

                # ── Recent Errors ──
                error_q = (
                    session.query(RequestModel)
                    .filter(RequestModel.success == 0)
                    .order_by(RequestModel.id.desc())
                )
                if profile_filter:
                    error_q = error_q.filter(
                        RequestModel.profile == profile_filter
                    )
                error_rows = error_q.limit(20).all()

                # ── Time-series data for Chart.js (last 14 days, daily cost) ──
                ts_rows = (
                    session.query(
                        func.substr(RequestModel.timestamp, 1, 10).label("date"),
                        func.coalesce(func.sum(RequestModel.cost), 0).label("cost"),
                        func.coalesce(func.avg(RequestModel.latency_ms), 0).label("avg_lat"),
                        func.count(RequestModel.id).label("count"),
                    )
                )
                if profile_filter:
                    ts_rows = ts_rows.filter(
                        RequestModel.profile == profile_filter
                    )
                ts_rows = (
                    ts_rows.filter(RequestModel.success == 1)
                    .group_by(func.substr(RequestModel.timestamp, 1, 10))
                    .order_by(func.substr(RequestModel.timestamp, 1, 10).asc())
                    .limit(14)
                    .all()
                )
                ts_dates = [r.date for r in ts_rows]
                ts_costs = [float(r.cost) for r in ts_rows]
                ts_lats = [float(r.avg_lat) for r in ts_rows]

                # Per-profile time series (grouped bar chart data)
                pp_rows = (
                    session.query(
                        func.substr(RequestModel.timestamp, 1, 10).label("date"),
                        RequestModel.profile,
                        func.coalesce(func.sum(RequestModel.cost), 0).label("cost"),
                        func.coalesce(func.avg(RequestModel.latency_ms), 0).label("avg_lat"),
                    )
                    .filter(RequestModel.success == 1)
                    .group_by(func.substr(RequestModel.timestamp, 1, 10), RequestModel.profile)
                    .order_by(func.substr(RequestModel.timestamp, 1, 10).asc())
                    .limit(14 * len(self.config.profiles))
                    .all()
                )
                # Pivot into {dates: [...], profiles: {name: {costs:[], lats:[]}}}
                pp_data = {"dates": sorted(set(r.date for r in pp_rows)), "profiles": {}}
                for p in self.config.profiles.keys():
                    pp_data["profiles"][p] = {"costs": [], "lats": []}
                date_to_idx = {d: i for i, d in enumerate(pp_data["dates"])}
                for p in self.config.profiles.keys():
                    pp_data["profiles"][p]["costs"] = [0.0] * len(pp_data["dates"])
                    pp_data["profiles"][p]["lats"] = [0.0] * len(pp_data["dates"])
                for r in pp_rows:
                    idx = date_to_idx[r.date]
                    pp_data["profiles"][r.profile]["costs"][idx] = float(r.cost)
                    pp_data["profiles"][r.profile]["lats"][idx] = float(r.avg_lat)

                # Per-model time series (grouped by model)
                pm_rows = (
                    session.query(
                        func.substr(RequestModel.timestamp, 1, 10).label("date"),
                        RequestModel.model,
                        func.coalesce(func.sum(RequestModel.cost), 0).label("cost"),
                        func.coalesce(func.avg(RequestModel.latency_ms), 0).label("avg_lat"),
                    )
                    .filter(RequestModel.success == 1)
                    .group_by(func.substr(RequestModel.timestamp, 1, 10), RequestModel.model)
                    .order_by(func.substr(RequestModel.timestamp, 1, 10).asc())
                    .all()
                )
                pm_data = {"dates": sorted(set(r.date for r in pm_rows)), "models": {}}
                pm_date_to_idx = {d: i for i, d in enumerate(pm_data["dates"])}
                for r in pm_rows:
                    if r.model not in pm_data["models"]:
                        pm_data["models"][r.model] = {"costs": [0.0] * len(pm_data["dates"]), "lats": [0.0] * len(pm_data["dates"])}
                    idx = pm_date_to_idx[r.date]
                    pm_data["models"][r.model]["costs"][idx] = float(r.cost)
                    pm_data["models"][r.model]["lats"][idx] = float(r.avg_lat)
        except Exception:
            import traceback as _tb
            _tb.print_exc()
            summary = type("S", (), {"total_cost": 0, "total_requests": 0, "cache_hits": 0, "cache_misses": 0, "prompt_tokens": 0, "output_tokens": 0})()
            fallback_count = 0
            total_success = 0
            cache_hit_rate = 0
            cache_hit_tokens = 0
            cache_miss_tokens = 0
            active_days = 0
            profile_rows = []
            daily_rows = []
            recent_rows = []
            error_rows = []
            ts_dates = []
            ts_costs = []
            ts_lats = []
            pp_data = {"dates": [], "profiles": {}}
            pm_data = {"dates": [], "models": {}}

        cache = get_prompt_cache()
        cache_stats = cache.stats

        # ── Helper: format large numbers ──
        def _fmt_num(n):
            if n >= 1_000_000:
                return f"{n/1_000_000:.1f}M"
            if n >= 1_000:
                return f"{n/1_000:.1f}K"
            return str(int(n))

        def _fmt_cost(c):
            return f"${float(c):.6f}" if c else "$0.000000"

        # ── Build HTML sections ──

        # Provider Health — compact rows, click to expand
        prov_html = ""
        for prof_name, prof_cfg in self.config.profiles.items():
            if profile_filter and prof_name != profile_filter:
                continue
            for step in prof_cfg["chain"]:
                pn = step["provider"]
                bu = step["base_url"]
                h = _get_health(pn, bu, prof_name, self.config)
                status = h["status"]
                dot_class = (
                    "dot-healthy" if status == "healthy"
                    else "dot-degraded" if status == "degraded"
                    else "dot-dead"
                )
                tripped = ""
                if h.get("tripped_until"):
                    tt = datetime.fromtimestamp(h["tripped_until"], tz=timezone.utc)
                    tripped = f" · Tripped until {tt.strftime('%H:%M:%S')}"
                last_suc = h.get("last_success") or "never"
                last_fail = h.get("last_failure") or "never"
                import html as _html
                detail_data = _html.escape(json.dumps({
                    "name": pn, "profile": prof_name, "status": status,
                    "url": bu, "failures": h["consecutive_failures"],
                    "tripped": tripped, "last_success": last_suc, "last_failure": last_fail
                }, default=str))
                prov_html += (
                    f'<div class="ph-row" data-detail="{detail_data}" onclick="showPhDetail(event, this)">'
                    f'<span class="ph-dot {dot_class}"></span>'
                    f'<span class="ph-name">{pn}</span>'
                    f'<span class="ph-profile">{prof_name}</span>'
                    '</div>'
                )

        # Sidebar navigation — tree: Profiles → Providers → Models
        def _active(p): return " active" if profile_filter == p else ""
        dash_active = _active(None)
        host = self.headers.get("Host", "localhost:8735")
        scheme = "https" if self.headers.get("X-Forwarded-Proto") == "https" else "http"
        host_url = f"{scheme}://{host}"
        sidebar_nav = (
            '<aside class="sidebar" id="sidebar">\n'
            '  <div class="sidebar-brand">⚡ LCP</div>\n'
            '  <nav class="sidebar-nav">\n'
            f'    <a href="/dashboard" class="{dash_active}">Dashboard</a>\n'
            '    <div class="nav-label">Profiles</div>\n'
        )
        for p, pcfg in self.config.profiles.items():
            active = " active" if profile_filter == p else ""
            sidebar_nav += (
                f'\n    <div class="sb-tree">'
                f'\n      <div class="sb-folder sb-profile-row" data-profile="{p}">'
                f'\n        <div class="sb-swipe-actions">'
                f'\n          <button class="sb-action-btn" onclick="event.stopPropagation();openProfileConfig(event,\u0027{p}\u0027)">Settings</button>'
                f'\n        </div>'
                f'\n        <div class="sb-folder-content">'
                f'\n          <span class="sb-chevron" onclick="toggleSbFolder(this.closest(\u0027.sb-folder\u0027))">▸</span>'
                f'\n          <a href="/{p}/dashboard" class="{active} sb-folder-link">{p.upper()}</a>'
                f'\n          <a href="#" class="sb-settings-link" onclick="openProfileConfig(event,\u0027{p}\u0027);return false">settings</a>'
                f'\n        </div>'
                f'\n      </div>'
                f'\n      <div class="sb-children">'
            )
            for step in pcfg.get("chain", []):
                pn = step["provider"]
                bu = step.get("base_url", "")
                pdata = self.config.providers.get(pn, {})
                models = pdata.get("models", [])
                h = _get_health(pn, bu, p, self.config)
                status = h["status"]
                dot_class = "dot-healthy" if status == "healthy" else "dot-degraded" if status == "degraded" else "dot-dead"
                sidebar_nav += (
                    f'\n        <div class="sb-tree">'
                    f'\n          <div class="sb-folder sb-provider" data-profile="{p}" data-provider="{pn}" data-url="{bu}" data-models="{",".join(models)}" data-status="{status}" data-keyenv="{pdata.get("api_key_env", "")}">'
                    f'\n            <span class="sb-chevron" onclick="toggleProvider(event, this.parentElement)">▸</span>'
                    f'\n            <span class="ph-dot {dot_class}" style="display:inline-block;margin-right:4px"></span>'
                    f'\n            <span class="sb-name" onclick="toggleProvider(event, this.parentElement)">{pn}</span>'
                    f'\n            <span class="sb-edit-btn" onclick="editProviderFromSidebar(event, this.parentElement)" title="Edit provider">⚙</span>'
                    f'\n          </div>'
                    f'\n          <div class="sb-children sb-models">'
                )
                for m in models:
                    sidebar_nav += f'\n            <div class="sb-leaf"><span class="sb-model">{m}</span></div>'
                if not models:
                    sidebar_nav += '\n            <div class="sb-leaf" style="color:hsl(var(--muted-foreground));font-style:italic">no models</div>'
                sidebar_nav += (
                    f'\n          </div>'
                    f'\n        </div>'
                )
            if not pcfg.get("chain"):
                sidebar_nav += '\n        <div class="sb-leaf" style="color:hsl(var(--muted-foreground));font-style:italic">no providers</div>'
            sidebar_nav += (
                f'\n      </div>'
                f'\n      <div class="sb-leaf" style="padding-top:0.125rem">'
                f'\n        <span class="sb-url" id="url-{p}" title="Gateway URL">/{p}/chat/completions</span>'
                f'\n        <button class="sb-copy-btn" onclick="copyUrl(\u0027{host_url}/{p}/chat/completions\u0027)" title="Copy URL">copy</button>'
                f'\n      </div>'
                f'\n    </div>'
            )
        sidebar_nav += (
            '\n    <div class="sb-leaf" style="padding-top:0.375rem">'
            '\n      <button class="btn-sm btn-primary" onclick="addProfile()" style="font-size:0.6875rem;width:100%">+ Add Profile</button>'
            '\n    </div>'
            '\n    <div class="nav-label">Configuration</div>'
            '\n    <a href="#" onclick="openProviderModal();return false">Providers</a>'

            '\n  </nav>\n</aside>'
        )

        # Old-style tabs (kept for reference, not rendered)
        tabs_html = ""

        # Per-profile summary cards
        profile_card_html = ""
        for row in profile_rows:
            profile_card_html += (
                f'<div class="card"><div class="label">{row.profile} · '
                f'{row.count} reqs</div><div class="value">${row.total_cost:.4f}</div></div>\n'
            )

        # Daily Costs table
        daily_html = ""
        for r in daily_rows:
            daily_html += (
                f'<tr><td class="mono">{r.date}</td><td>{r.profile}</td>'
                f'<td class="mono">{r.model}</td><td>{r.provider}</td>'
                f'<td class="cost">{r.reqs}</td><td class="cost">{r.fb_count}</td>'
                f'<td class="cost">{_fmt_num(r.cache_hit)}</td>'
                f'<td class="cost">{_fmt_num(r.cache_miss)}</td>'
                f'<td class="cost">{_fmt_num(r.output)}</td>'
                f'<td class="cost mono">${float(r.cost):.6f}</td></tr>\n'
            )

        # Recent Requests table
        recent_html = ""
        for r in recent_rows:
            time_str = r.timestamp[11:19] if r.timestamp and "T" in r.timestamp else str(r.timestamp)[:19]
            status_badges = ""
            if r.success:
                status_badges += '<span class="badge badge-success">ok</span> '
            else:
                status_badges += f'<span class="badge badge-error">{r.error_type or "error"}</span> '
            if r.provider != "error" and r.success and r.provider:
                # Check if this was a fallback (first chain step != provider used)
                prof_cfg = self.config.profiles.get(r.profile, {})
                chain = prof_cfg.get("chain", [])
                if chain and chain[0]["provider"] != r.provider:
                    status_badges += '<span class="badge badge-fallback">FB</span>'
            recent_html += (
                f'<tr><td class="mono">{time_str}</td><td>{r.profile}</td>'
                f'<td class="mono">{r.model}</td><td>{r.provider}</td>'
                f'<td class="cost">{status_badges}</td>'
                f'<td class="cost">{r.latency_ms/1000:.1f}s</td>'
                f'<td class="cost mono">${r.cost:.6f}</td></tr>\n'
            )

        # Recent Errors table
        error_html = ""
        for r in error_rows:
            time_str = r.timestamp[:19] if r.timestamp else ""
            error_html += (
                f'<tr><td class="mono">{time_str}</td><td>{r.profile}</td>'
                f'<td>{r.provider}</td><td class="cost">{r.error_type}</td>'
                f'<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;'
                f'white-space:nowrap" title=""></td></tr>\n'
            )

        # Fallback rate
        fb_pct = (fallback_count / total_success * 100) if total_success > 0 else 0

        # Phase 5/6 extra cards
        verifier = get_token_verifier()
        v_stats = verifier.stats if hasattr(verifier, "stats") else {}
        router = get_dynamic_router()
        r_conf = getattr(router, "config", {})

        phase56_cards = (
            f'<div class="card"><div class="label">Token Mismatches</div>'
            f'<div class="value">{v_stats.get("mismatches", 0)}</div>'
            f'<div class="sub">Token verification diffs</div></div>\n'
            f'<div class="card"><div class="label">Routing Threshold</div>'
            f'<div class="value">{r_conf.get("flash_threshold_tokens", 4096)}</div>'
            f'<div class="sub">Tokens → flash model</div></div>\n'
            f'<div class="card"><div class="label">Cache Entries</div>'
            f'<div class="value">{cache_stats["entries"]}</div>'
            f'<div class="sub">Max: {cache_stats.get("max_entries", "N/A")}</div></div>\n'
        )

        # ── Full HTML ──
        filter_title = f" — {profile_filter.upper()}" if profile_filter else ""
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        ts_dates_json = json.dumps(ts_dates)
        ts_costs_json = json.dumps(ts_costs)
        ts_lats_json = json.dumps(ts_lats)
        pp_data_json = json.dumps(pp_data)
        pm_data_json = json.dumps(pm_data)

        html = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LCP Dashboard{filter_title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/sortablejs@1.15.0/Sortable.min.js"></script>
<style>
:root {{
  --background: 222.2 84% 4.9%;
  --foreground: 210 40% 98%;
  --card: 222.2 84% 4.9%;
  --card-border: 217.2 32.6% 17.5%;
  --primary: 210 40% 98%;
  --primary-foreground: 222.2 47.4% 11.2%;
  --secondary: 217.2 32.6% 17.5%;
  --secondary-foreground: 210 40% 98%;
  --muted: 217.2 32.6% 17.5%;
  --muted-foreground: 215 20.2% 65.1%;
  --destructive: 0 62.8% 30.6%;
  --destructive-foreground: 210 40% 98%;
  --border: 217.2 32.6% 17.5%;
  --radius: 0.5rem;
  --green-bg: 142.1 76.2% 6%;
  --green-fg: 142.1 70.6% 45.3%;
  --amber-bg: 26 83.3% 7%;
  --amber-fg: 43.3 96.4% 56.3%;
  --red-bg: 0 74.2% 7%;
  --red-fg: 0 83.2% 60.2%;
  --cyan-bg: 190 80% 6%;
  --cyan-fg: 190 80% 50%;
}}
/* ── Sidebar + Layout ── */
:root {{
  --sidebar-width: 240px;
  --sidebar-bg: 222.2 50% 8%;
}}
.sidebar {{
  position: fixed; top: 0; left: 0; bottom: 0; z-index: 100;
  width: var(--sidebar-width);
  background: hsl(var(--sidebar-bg));
  border-right: 1px solid hsl(var(--card-border));
  display: flex; flex-direction: column;
  overflow-y: auto;
  transform: translateX(0);
  transition: transform 0.25s cubic-bezier(0.4,0,0.2,1);
}}
.sidebar.collapsed {{ transform: translateX(-100%); }}
.sidebar-brand {{
  padding: 1.25rem 1rem 1rem; font-size: 1.125rem; font-weight: 700;
  letter-spacing: -0.02em; color: hsl(var(--foreground));
  border-bottom: 1px solid hsl(var(--card-border));
}}
.sidebar-nav {{ padding: 0.75rem; flex: 1; overflow-x: hidden; }}
.sidebar-nav a {{
  display: flex; align-items: center; gap: 0.5rem;
  padding: 0.5rem 0.75rem; border-radius: var(--radius);
  font-size: 0.8125rem; font-weight: 500; text-decoration: none;
  color: hsl(var(--muted-foreground)); transition: all 0.15s;
  margin-bottom: 0.125rem;
}}
.sidebar-nav a:hover {{ background: hsl(var(--secondary)); color: hsl(var(--foreground)); }}
.sidebar-nav a.active {{
  background: hsl(var(--primary)); color: hsl(var(--primary-foreground));
}}
.sidebar-nav .nav-label {{
  font-size: 0.625rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.08em; color: hsl(var(--muted-foreground) / 0.6);
  padding: 1rem 0.75rem 0.375rem;
}}
.sidebar-toggle {{
  position: fixed; top: 0.75rem; left: calc(var(--sidebar-width) + 0.75rem); z-index: 110;
  width: 36px; height: 36px; border-radius: var(--radius);
  background: hsl(var(--card)); border: 1px solid hsl(var(--card-border));
  color: hsl(var(--foreground)); cursor: pointer; display: flex;
  align-items: center; justify-content: center; transition: left 0.25s cubic-bezier(0.4,0,0.2,1);
  font-size: 1rem;
}}
.sidebar.collapsed ~ .sidebar-toggle {{ left: 0.75rem; }}
.main-content {{
  margin-left: var(--sidebar-width);
  padding: 2rem;
  transition: margin-left 0.25s cubic-bezier(0.4,0,0.2,1);
}}
.sidebar.collapsed ~ .main-content {{ margin-left: 0; }}
/* overlay on mobile */
@media (max-width: 768px) {{
  .sidebar {{ transform: translateX(-100%); }}
  .sidebar.open {{ transform: translateX(0); }}
  .sidebar.collapsed {{ transform: translateX(-100%); }}
  .sidebar-toggle {{ left: 0.75rem !important; }}
  .sidebar.open ~ .sidebar-toggle {{ left: calc(var(--sidebar-width) + 0.75rem) !important; }}
  .main-content {{ margin-left: 0 !important; padding: 1rem; padding-top: 3rem; }}
  .sidebar-overlay {{
    display: none; position: fixed; inset: 0; z-index: 99;
    background: rgba(0,0,0,0.5);
  }}
  .sidebar-overlay.show {{ display: block; }}
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: hsl(var(--background));
  color: hsl(var(--foreground));
  padding: 2rem;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}}
h1 {{ font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.25rem; }}
.subtitle {{ color: hsl(var(--muted-foreground)); font-size: 0.8125rem; margin-bottom: 2rem; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.75rem; }}
.card {{
  background: hsl(var(--card));
  border: 1px solid hsl(var(--card-border));
  border-radius: var(--radius);
  padding: 1.25rem;
  transition: border-color 0.15s;
}}
.card:hover {{ border-color: hsl(var(--border) / 0.5); }}
.card .label {{
  font-size: 0.6875rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.05em; color: hsl(var(--muted-foreground));
}}
.card .value {{ font-size: 1.75rem; font-weight: 700; margin-top: 0.375rem; font-variant-numeric: tabular-nums; }}
.card .sub {{ font-size: 0.75rem; color: hsl(var(--muted-foreground)); margin-top: 0.25rem; }}
details.dashboard-section {{
  margin-bottom: 1.5rem;
  border: 1px solid hsl(var(--card-border));
  border-radius: var(--radius);
  overflow: hidden;
  background: hsl(var(--card));
}}
details.dashboard-section > summary {{
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.75rem 1.25rem; font-size: 0.8125rem; font-weight: 600;
  cursor: pointer; user-select: none; color: hsl(var(--foreground));
  list-style: none;
}}
details.dashboard-section > summary::-webkit-details-marker {{ display: none; }}
details.dashboard-section[open] > summary {{
  border-bottom: 1px solid hsl(var(--card-border));
  background: hsl(var(--secondary));
}}
details.dashboard-section > summary .chevron {{
  font-size: 0.6875rem; transition: transform 0.2s;
  color: hsl(var(--muted-foreground));
}}
details.dashboard-section[open] > summary .chevron {{ transform: rotate(180deg); }}
details.dashboard-section .section-content {{ padding: 1rem 1.25rem; }}
/* ── Provider Health — compact horizontal tags ── */
.ph-row {{
  display: inline-flex; align-items: center; gap: 0.375rem;
  padding: 0.25rem 0.5rem; border-radius: var(--radius); cursor: pointer;
  border: 1px solid hsl(var(--card-border)); transition: all 0.15s;
  margin-right: 0.375rem; margin-bottom: 0.375rem;
  background: hsl(var(--secondary) / 0.3);
}}
.ph-row:hover {{ border-color: hsl(var(--primary) / 0.4); background: hsl(var(--secondary) / 0.6); }}
.ph-dot {{
  width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
}}
.dot-healthy {{ background: hsl(var(--green-fg)); box-shadow: 0 0 3px hsl(var(--green-fg) / 0.5); }}
.dot-degraded {{ background: hsl(var(--amber-fg)); box-shadow: 0 0 3px hsl(var(--amber-fg) / 0.5); }}
.dot-dead {{ background: hsl(var(--red-fg)); box-shadow: 0 0 3px hsl(var(--red-fg) / 0.5); }}
.ph-name {{ font-weight: 600; font-size: 0.75rem; font-family: 'SF Mono', 'Fira Code', monospace; white-space: nowrap; }}
.ph-profile {{
  font-size: 0.5625rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.05em; color: hsl(var(--muted-foreground));
  background: hsl(var(--secondary)); padding: 0.0625rem 0.3125rem;
  border-radius: 9999px; white-space: nowrap;
}}
/* ── Provider Health Detail Modal ── */
.ph-modal-overlay {{
  display: none; position: fixed; inset: 0; z-index: 300;
  background: rgba(0,0,0,0.6); backdrop-filter: blur(3px);
  align-items: center; justify-content: center;
}}
.ph-modal-overlay.open {{ display: flex; }}
.ph-modal {{
  background: hsl(var(--card)); border: 1px solid hsl(var(--card-border));
  border-radius: 12px; width: min(420px, 90vw);
  box-shadow: 0 16px 48px rgba(0,0,0,0.5);
  animation: modalIn 0.15s ease;
}}
.ph-modal-header {{
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.75rem 1rem; border-bottom: 1px solid hsl(var(--card-border));
}}
.ph-modal-title {{ font-weight: 700; font-size: 0.875rem; }}
.ph-modal-body {{
  padding: 0.75rem 1rem; font-size: 0.75rem; line-height: 1.7;
  color: hsl(var(--muted-foreground));
}}
.ph-modal-body .phm-label {{
  font-size: 0.625rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.06em; color: hsl(var(--muted-foreground));
  margin-top: 0.5rem;
}}
.ph-modal-body .phm-value {{
  font-size: 0.8125rem; color: hsl(var(--foreground));
  font-family: 'SF Mono', 'Fira Code', monospace;
  word-break: break-all;
}}
/* keep old provider-card / status-pill for provider panels in modal */
.table-wrap {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.8125rem; }}
thead th {{
  text-align: left; padding: 0.5rem 0.75rem; color: hsl(var(--muted-foreground));
  font-size: 0.6875rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.05em; border-bottom: 1px solid hsl(var(--card-border));
}}
tbody td {{ padding: 0.5rem 0.75rem; border-bottom: 1px solid hsl(var(--card-border) / 0.35); }}
tbody tr:hover td {{ background: hsl(var(--secondary) / 0.4); }}
.cost {{ text-align: right; }}
.mono {{ font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace; font-size: 0.75rem; }}
.badge {{
  display: inline-block; padding: 0.0625rem 0.5rem; border-radius: 9999px;
  font-size: 0.6875rem; font-weight: 600; letter-spacing: 0.01em;
}}
.badge-fallback {{ background: hsl(var(--amber-bg)); color: hsl(var(--amber-fg)); }}
.badge-error {{ background: hsl(var(--red-bg)); color: hsl(var(--red-fg)); }}
.badge-success {{ background: hsl(var(--green-bg)); color: hsl(var(--green-fg)); }}
.badge-cache {{ background: hsl(var(--cyan-bg)); color: hsl(var(--cyan-fg)); }}
.refresh {{
  font-size: 0.75rem; color: hsl(var(--muted-foreground));
  text-align: center; margin-top: 2rem; padding-top: 1rem;
  border-top: 1px solid hsl(var(--card-border));
}}
.empty {{ text-align: center; color: hsl(var(--muted-foreground)); padding: 1.5rem; font-size: 0.8125rem; }}
.green {{ color: hsl(var(--green-fg)); }}
.amber {{ color: hsl(var(--amber-fg)); }}
.red {{ color: hsl(var(--red-fg)); }}
.view-toggle {{
  padding: 0.15rem 0.5rem; border-radius: 9999px; font-size: 0.6875rem;
  font-weight: 600; border: 1px solid hsl(var(--card-border));
  background: hsl(var(--secondary)); color: hsl(var(--muted-foreground));
  cursor: pointer; transition: all 0.15s;
}}
.view-toggle.active {{
  background: hsl(var(--primary)); color: hsl(var(--primary-foreground));
  border-color: hsl(var(--primary));
}}
/* ── Provider Management ── */
.provider-mgmt {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
.provider-mgmt .prov-list {{  }}
.prov-item {{
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.625rem 0.75rem; margin-bottom: 0.375rem;
  background: hsl(var(--card)); border: 1px solid hsl(var(--card-border));
  border-radius: var(--radius);
}}
.prov-item .prov-name {{ font-weight: 600; font-size: 0.8125rem; }}
.prov-item .prov-detail {{ font-size: 0.6875rem; color: hsl(var(--muted-foreground)); }}
.prov-item .prov-actions {{ display: flex; gap: 0.375rem; }}
.prov-form {{
  background: hsl(var(--card)); border: 1px solid hsl(var(--card-border));
  border-radius: var(--radius); padding: 1rem;
}}
.prov-form label {{
  display: block; font-size: 0.6875rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.05em; color: hsl(var(--muted-foreground)); margin-bottom: 0.25rem;
}}
.prov-form input, .prov-form select {{
  width: 100%; padding: 0.375rem 0.5rem; margin-bottom: 0.625rem;
  background: hsl(var(--background)); border: 1px solid hsl(var(--card-border));
  border-radius: var(--radius); color: hsl(var(--foreground));
  font-size: 0.8125rem; font-family: inherit;
}}
.prov-form button, .btn-sm {{
  padding: 0.3rem 0.75rem; border-radius: var(--radius); font-size: 0.75rem;
  font-weight: 600; cursor: pointer; border: 1px solid hsl(var(--card-border));
  background: hsl(var(--secondary)); color: hsl(var(--foreground));
  transition: all 0.1s;
}}
.btn-primary {{ background: hsl(var(--primary)); color: hsl(var(--primary-foreground)); }}
.btn-danger {{ background: hsl(var(--destructive)); color: hsl(var(--destructive-foreground)); border-color: hsl(var(--destructive)); }}
.btn-success {{ background: hsl(var(--green-bg)); color: hsl(var(--green-fg)); border-color: hsl(var(--green-fg)); }}
.btn-sm:hover {{ opacity: 0.85; }}
.chain-list {{ list-style: none; padding: 0; }}
.chain-item {{
  display: flex; align-items: center; gap: 0.5rem;
  padding: 0.375rem 0.5rem; margin-bottom: 0.25rem;
  background: hsl(var(--secondary)); border-radius: var(--radius);
  font-size: 0.75rem; cursor: grab;
}}
.chain-item:active {{ cursor: grabbing; }}
.chain-item .drag-handle {{ color: hsl(var(--muted-foreground)); margin-right: 0.25rem; }}
.chain-item .provider-badge {{
  padding: 0.0625rem 0.375rem; border-radius: 9999px;
  background: hsl(var(--primary)); color: hsl(var(--primary-foreground));
  font-size: 0.625rem; font-weight: 700;
}}
.chain-item .model-badge {{
  color: hsl(var(--muted-foreground)); font-size: 0.6875rem;
}}
.test-result {{
  font-size: 0.75rem; margin-top: 0.375rem; padding: 0.375rem 0.5rem;
  border-radius: var(--radius);
}}
@media (max-width: 768px) {{ .provider-mgmt {{ grid-template-columns: 1fr; }} }}

/* ── Sidebar Tree ── */
.sb-tree {{ }}
.sb-folder {{
  display: flex; align-items: center; gap: 0.25rem;
  padding: 0.25rem 0.5rem; border-radius: var(--radius);
  cursor: pointer; transition: all 0.1s;
  font-size: 0.75rem; color: hsl(var(--muted-foreground));
}}
.sb-folder:hover {{ background: hsl(var(--secondary) / 0.5); color: hsl(var(--foreground)); }}

/* ── Swipe-to-reveal (mobile) ── */
.sb-profile-row {{
  overflow: hidden; padding: 0 !important;
  position: relative;
}}
.sb-folder-content {{
  display: flex; align-items: center; gap: 0.25rem;
  padding: 0.25rem 0.5rem; border-radius: var(--radius);
  background: hsl(var(--background));
  position: relative; z-index: 1;
  transition: transform 0.2s ease;
  width: 100%; box-sizing: border-box;
}}
.sb-profile-row.swiped .sb-folder-content {{
  transform: translateX(-72px);
  border-radius: var(--radius) 0 0 var(--radius);
}}
.sb-swipe-actions {{
  position: absolute; right: 0; top: 0; bottom: 0;
  display: flex; align-items: stretch; z-index: 0;
}}
.sb-action-btn {{
  background: hsl(var(--primary)); color: hsl(var(--primary-foreground));
  border: none; border-radius: 0 var(--radius) var(--radius) 0;
  padding: 0 0.625rem; font-size: 0.6875rem; font-weight: 600;
  cursor: pointer; white-space: nowrap;
  display: flex; align-items: center;
  height: 100%;
}}
.sb-action-btn:active {{ filter: brightness(0.85); }}
@media (min-width: 769px) {{
  .sb-profile-row {{ overflow: visible; }}
  .sb-swipe-actions {{ display: none; }}
  .sb-folder-content {{ transform: none !important; }}
}}
.sb-folder.sb-provider.selected {{
  background: hsl(var(--primary) / 0.15); color: hsl(var(--foreground));
  border: 1px solid hsl(var(--primary) / 0.3);
}}
.sb-chevron {{
  font-size: 0.5rem; transition: transform 0.15s; flex-shrink: 0; width: 10px; text-align: center;
}}
.sb-folder.open > .sb-chevron {{ transform: rotate(90deg); }}
.sb-folder-link {{ padding: 0; background: none; flex: 1; }}
.sb-folder-link:hover {{ background: none; }}
.sb-settings-link {{
  font-size: 0.5625rem; color: hsl(var(--muted-foreground) / 0.45);
  text-decoration: none; margin-left: auto; padding: 0.125rem 0.375rem;
  border-radius: 3px; flex-shrink: 0;
}}
.sb-settings-link:hover {{ color: hsl(var(--foreground)); background: hsl(var(--secondary) / 0.5); }}
.sb-name {{ font-family: 'SF Mono', 'Fira Code', monospace; font-weight: 600; font-size: 0.75rem; }}
.sb-children {{ display: none; padding-left: 1rem; }}
.sb-folder.open + .sb-children {{ display: block; }}
.sb-leaf {{
  padding: 0.2rem 0.5rem; font-size: 0.6875rem;
  color: hsl(var(--muted-foreground)); display: flex; align-items: center; gap: 0.25rem;
}}
.sb-model {{
  font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.625rem;
  color: hsl(var(--muted-foreground));
}}
.sb-models {{ padding-left: 1.5rem; }}
.sb-edit-btn {{
  margin-left: auto; font-size: 0.625rem; cursor: pointer;
  padding: 0.125rem 0.25rem; border-radius: 4px; opacity: 0;
  transition: opacity 0.1s; color: hsl(var(--muted-foreground));
}}
.sb-folder:hover .sb-edit-btn {{ opacity: 1; }}
.sb-edit-btn:hover {{ background: hsl(var(--secondary)); color: hsl(var(--foreground)); }}
.sb-url {{
  font-size: 0.5625rem; font-family: 'SF Mono', 'Fira Code', monospace;
  color: hsl(var(--muted-foreground) / 0.6); overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; max-width: 160px;
  display: inline-block; vertical-align: middle;
}}
.sb-copy-btn {{
  background: none; border: none; cursor: pointer; font-size: 0.5625rem;
  padding: 0 0.25rem; opacity: 0.5; vertical-align: middle;
}}
.sb-copy-btn:hover {{ opacity: 1; }}

/* ── Provider Modal ── */
.modal-overlay {{
  display: none; position: fixed; inset: 0; z-index: 200;
  background: rgba(0,0,0,0.7); backdrop-filter: blur(4px);
  align-items: center; justify-content: center;
}}
.modal-overlay.open {{ display: flex; }}
.modal {{
  background: hsl(var(--card)); border: 1px solid hsl(var(--card-border));
  border-radius: 14px; width: min(900px, 95vw); max-height: 90vh;
  display: flex; flex-direction: column; box-shadow: 0 20px 60px rgba(0,0,0,0.5);
  animation: modalIn 0.2s ease;
}}
@keyframes modalIn {{ from {{ opacity: 0; transform: translateY(-20px) scale(0.97); }} to {{ opacity: 1; transform: translateY(0) scale(1); }} }}
.modal-header {{
  display: flex; align-items: center; justify-content: space-between;
  padding: 1rem 1.25rem; border-bottom: 1px solid hsl(var(--card-border));
  flex-shrink: 0;
}}
.modal-header h2 {{ font-size: 1rem; font-weight: 700; margin: 0; }}
.modal-close {{
  background: none; border: 1px solid hsl(var(--card-border)); border-radius: var(--radius);
  color: hsl(var(--foreground)); font-size: 1.125rem; cursor: pointer;
  width: 32px; height: 32px; display: flex; align-items: center; justify-content: center;
  transition: all 0.1s;
}}
.modal-close:hover {{ background: hsl(var(--destructive)); border-color: hsl(var(--destructive)); }}
.modal-tabs {{
  display: flex; gap: 0; border-bottom: 1px solid hsl(var(--card-border));
  padding: 0 1rem; flex-shrink: 0; overflow-x: auto;
}}
.tab-btn {{
  background: none; border: none; border-bottom: 2px solid transparent;
  color: hsl(var(--muted-foreground)); font-size: 0.8125rem; font-weight: 600;
  padding: 0.625rem 1rem; cursor: pointer; transition: all 0.15s;
  white-space: nowrap;
}}
.tab-btn:hover {{ color: hsl(var(--foreground)); }}
.tab-btn.active {{
  color: hsl(var(--foreground)); border-bottom-color: hsl(var(--primary));
}}
.tab-dropdown {{ display: none; }}
@media (max-width: 640px) {{
  .modal-tabs {{ display: none; }}
  .tab-dropdown {{
    display: block; margin: 0.75rem 1rem; flex-shrink: 0;
  }}
  .tab-dropdown select {{
    width: 100%; padding: 0.5rem; border-radius: var(--radius);
    background: hsl(var(--secondary)); border: 1px solid hsl(var(--card-border));
    color: hsl(var(--foreground)); font-size: 0.875rem; font-family: inherit;
  }}
}}
.modal-body {{
  flex: 1; overflow-y: auto; padding: 1rem 1.25rem;
  -webkit-overflow-scrolling: touch;
}}
.modal-footer {{
  display: flex; gap: 0.5rem; padding: 0.75rem 1.25rem;
  border-top: 1px solid hsl(var(--card-border)); flex-shrink: 0;
  justify-content: flex-end;
}}
.tab-panel {{ display: none; }}
.tab-panel.active {{ display: block; }}
.modal-prov-form {{
  background: hsl(var(--secondary)); border: 1px solid hsl(var(--card-border));
  border-radius: var(--radius); padding: 1rem; margin-top: 1rem;
}}
.modal-prov-form label {{
  display: block; font-size: 0.6875rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.05em; color: hsl(var(--muted-foreground)); margin-bottom: 0.25rem;
}}
.modal-prov-form input, .modal-prov-form select {{
  width: 100%; padding: 0.375rem 0.5rem; margin-bottom: 0.625rem;
  background: hsl(var(--background)); border: 1px solid hsl(var(--card-border));
  border-radius: var(--radius); color: hsl(var(--foreground));
  font-size: 0.8125rem; font-family: inherit;
}}
.section-label {{
  font-size: 0.6875rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.06em; color: hsl(var(--muted-foreground));
  margin-bottom: 0.5rem;
}}

.chart-container {{ position: relative; min-height: 280px; width: 100%; }}
.endpoints {{
  font-size: 0.75rem; color: hsl(var(--muted-foreground));
  margin-top: 2rem; padding-top: 1rem;
  border-top: 1px solid hsl(var(--card-border));
  text-align: center;
}}
</style>
</head>
<body>
{sidebar_nav}
<div class="sidebar-overlay" id="sidebarOverlay" onclick="closeSidebar()"></div>
<button class="sidebar-toggle" id="sidebarToggle" onclick="toggleSidebar()" title="Toggle sidebar">☰</button>
<div class="main-content">
<h1>LLM Control Plane{filter_title}</h1>
<p class="subtitle">Cost tracking · Request history · Phase 5/6 intelligence</p>

<!-- ── Provider Edit Modal (opened from sidebar) ── -->
<div class="modal-overlay" id="provEditModal">
<div class="modal" style="width:min(560px,95vw)">
  <div class="modal-header">
    <h2 id="pemTitle"></h2>
    <button class="modal-close" onclick="closeProvEditModal()">✕</button>
  </div>
  <div class="modal-body">
    <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem">
      <span style="font-size:0.75rem;font-weight:600" id="pemStatus"></span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem">
      <div>
        <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">API Base URL</label>
        <input id="pemUrl" style="width:100%;padding:0.375rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem;font-family:inherit">
      </div>
      <div>
        <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">API Key Env Var</label>
        <input id="pemKeyEnv" style="width:100%;padding:0.375rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem;font-family:inherit">
      </div>
      <div style="grid-column:1/-1">
        <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Models (comma-separated)</label>
        <input id="pemModels" style="width:100%;padding:0.375rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem;font-family:inherit">
      </div>
    </div>
    <div style="display:flex;gap:0.5rem;margin-top:0.75rem">
      <button class="btn-sm btn-success" id="pemTestBtn" onclick="testPemProvider()">Test Connection</button>
      <button class="btn-sm btn-primary" id="pemSaveBtn" onclick="savePemProvider()">Save Provider</button>
    </div>
    <div id="pemTestResult" class="test-result" style="display:none;margin-top:0.5rem"></div>
  </div>
</div>
</div>

<!-- ── API Keys Modal ── -->
<div class="modal-overlay" id="keysModal">
<div class="modal" style="width:min(500px,95vw)">
  <div class="modal-header">
    <h2>API Keys</h2>
    <button class="modal-close" onclick="closeKeysModal()">✕</button>
  </div>
  <div class="modal-body">
    <div id="keysList"><div class="empty">Loading...</div></div>
  </div>
  <div class="modal-footer">
    <button class="btn-sm btn-primary" onclick="generateKey()">+ Generate Key</button>
    <button class="btn-sm" onclick="closeKeysModal()">Close</button>
  </div>

<!-- ── Profile Config Modal ── -->
<div class="modal-overlay" id="profileConfigModal">
<div class="modal" style="width:min(520px,95vw)">
  <div class="modal-header">
    <h2 id="pcmTitle">Profile: L2</h2>
    <button class="modal-close" onclick="closeProfileConfig()">✕</button>
  </div>
  <div class="modal-tabs" id="pcmTabs">
    <button class="modal-tab active" data-tab="pcm-apikeys">API Keys</button>
    <button class="modal-tab" data-tab="pcm-url">Copyable URL</button>
    <!-- future tabs here -->
  </div>
  <div class="modal-body" id="pcmBody">
    <!-- Tab: API Keys -->
    <div class="tab-panel active" id="panel-pcm-apikeys">
      <div id="pcmKeysList"><div class="empty">Loading...</div></div>
      <div style="margin-top:0.75rem">
        <button class="btn-sm btn-primary" onclick="generateProfileKey()">+ Generate Key</button>
      </div>
    </div>
    <!-- Tab: Copyable URL -->
    <div class="tab-panel" id="panel-pcm-url">
      <div class="phm-label">Gateway URL</div>
      <div style="display:flex;align-items:center;gap:0.5rem;margin-top:0.25rem">
        <code id="pcmUrl" style="flex:1;background:hsl(var(--secondary));padding:0.375rem 0.5rem;border-radius:var(--radius);font-size:0.8125rem;word-break:break-all"></code>
        <button class="btn-sm btn-primary" onclick="copyProfileUrl()">Copy</button>
      </div>
      <div class="phm-label" style="margin-top:1rem">Usage</div>
      <pre id="pcmCurl" style="background:hsl(var(--secondary));padding:0.5rem;border-radius:var(--radius);font-size:0.75rem;overflow-x:auto;margin-top:0.25rem"></pre>
    </div>
  </div>
  <div class="modal-footer">
    <button class="btn-sm" onclick="closeProfileConfig()">Close</button>
  </div>
</div>
</div>
</div>
</div>

<details class="dashboard-section" open>
<summary><span>Summary</span><span class="chevron">▾</span></summary>
<div class="section-content cards">
<div class="card">
  <div class="label">Total Cost</div>
  <div class="value">{_fmt_cost(summary.total_cost)}</div>
  <div class="sub">{active_days} active days</div>
</div>
<div class="card">
  <div class="label">Total Requests</div>
  <div class="value">{summary.total_requests:,}</div>
  <div class="sub">{fallback_count:,} fallbacks ({fb_pct:.1f}%)</div>
</div>
<div class="card">
  <div class="label">Cache Hit Ratio</div>
  <div class="value good">{cache_hit_rate:.1f}%</div>
  <div class="sub">{_fmt_num(cache_hit_tokens)} hit / {_fmt_num(cache_miss_tokens)} miss</div>
</div>
<div class="card">
  <div class="label">Output Tokens</div>
  <div class="value">{_fmt_num(summary.output_tokens or 0)}</div>
  <div class="sub">prompt: {_fmt_num(summary.prompt_tokens or 0)}</div>
</div>
{profile_card_html}
{phase56_cards}
</div>
</details>

<details class="dashboard-section" open>
<summary>
  <span>Daily Cost Trend (14-day)</span>
  <span style="display:flex;gap:0.25rem;align-items:center">
    <button onclick="switchView('aggregate')" id="btnAgg" class="view-toggle active">Aggregate</button>
    <button onclick="switchView('per-profile')" id="btnPP" class="view-toggle">Per Profile</button>
    <button onclick="switchView('per-model')" id="btnPM" class="view-toggle">Per Model</button>
    <span class="chevron">▾</span>
  </span>
</summary>
<div class="section-content">
<div class="chart-container">
<canvas id="costChart"></canvas>
</div>
<div class="chart-container">
<canvas id="latencyChart"></canvas>
</div>
</div>
</details>

<details class="dashboard-section">
<summary><span>Daily Costs</span><span class="chevron">▾</span></summary>
<div class="section-content">
<div class="table-wrap">
<table>
<thead><tr><th>Date</th><th>Profile</th><th>Model</th><th>Provider</th><th class="cost">Reqs</th><th class="cost">Fb</th><th class="cost">Cache Hit</th><th class="cost">Cache Miss</th><th class="cost">Output</th><th class="cost">Cost</th></tr></thead>
<tbody>
{daily_html or '<tr><td colspan="10" class="empty">No data yet</td></tr>'}
</tbody>
</table>
</div>
</div>
</details>

<details class="dashboard-section">
<summary><span>Recent Requests</span><span class="chevron">▾</span></summary>
<div class="section-content">
<div class="table-wrap">
<table>
<thead><tr><th>Time</th><th>Profile</th><th>Model</th><th>Provider</th><th class="cost">Status</th><th class="cost">Dur</th><th class="cost">Cost</th></tr></thead>
<tbody>
{recent_html or '<tr><td colspan="7" class="empty">No requests yet</td></tr>'}
</tbody>
</table>
</div>
</div>
</details>

<details class="dashboard-section">
<summary><span>Recent Errors</span><span class="chevron">▾</span></summary>
<div class="section-content">
<div class="table-wrap">
<table>
<thead><tr><th>Time</th><th>Profile</th><th>Provider</th><th class="cost">Error</th><th>Message</th></tr></thead>
<tbody>
{error_html or '<tr><td colspan="5" class="empty">No errors</td></tr>'}
</tbody>
</table>
</div>
</div>
</details>

<p class="endpoints">
<a href="/health" style="color:inherit">/health</a> ·
<a href="/v1/models" style="color:inherit">/v1/models</a> ·
<a href="/metrics" style="color:inherit">/metrics</a> ·
<a href="/cache/stats" style="color:inherit">/cache/stats</a> ·
<a href="/export?limit=500" style="color:inherit">/export</a> ·
<a href="/errors" style="color:inherit">/errors</a>
</p>

<p class="refresh">Generated {now_utc} · v{__import__('src').__version__} · Refresh page to update</p>

<script>
/* ── Sidebar toggle ── */
function toggleSidebar() {{
  var sb = document.getElementById('sidebar');
  var ov = document.getElementById('sidebarOverlay');
  if (window.innerWidth <= 768) {{
    // mobile: overlay mode
    sb.classList.toggle('open');
    ov.classList.toggle('show');
  }} else {{
    // desktop: push mode
    sb.classList.toggle('collapsed');
    localStorage.setItem('lcp-sidebar', sb.classList.contains('collapsed') ? 'collapsed' : 'pinned');
  }}
}}
function closeSidebar() {{
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebarOverlay').classList.remove('show');
}}
// restore state
(function() {{
  if (window.innerWidth <= 768) return;
  if (localStorage.getItem('lcp-sidebar') === 'collapsed') {{
    document.getElementById('sidebar').classList.add('collapsed');
  }}
}})();

/* ── Swipe-to-reveal for profile rows (mobile) ── */
var _swipedRow = null;
var _swipeStartX = 0;
var _swipeCurrentX = 0;
var _swipeThreshold = 40;

function closeAllSwipes() {{
  if (_swipedRow) {{
    _swipedRow.classList.remove('swiped');
    _swipedRow.querySelector('.sb-folder-content').style.transform = '';
    _swipedRow = null;
  }}
}}

document.addEventListener('touchstart', function(e) {{
  var row = e.target.closest('.sb-profile-row');
  if (!row) {{ closeAllSwipes(); return; }}
  if (row !== _swipedRow) closeAllSwipes();
  _swipeStartX = e.touches[0].clientX;
  _swipeCurrentX = _swipeStartX;
}}, {{passive: true}});

document.addEventListener('touchmove', function(e) {{
  var row = e.target.closest('.sb-profile-row');
  if (!row) return;
  _swipeCurrentX = e.touches[0].clientX;
  var dx = _swipeCurrentX - _swipeStartX;
  if (dx > 0) {{ return; }} // only left swipe
  dx = Math.max(dx, -72); // cap at action button width
  row.querySelector('.sb-folder-content').style.transform = 'translateX(' + dx + 'px)';
  row.querySelector('.sb-folder-content').style.transition = 'none';
}}, {{passive: true}});

document.addEventListener('touchend', function(e) {{
  var row = e.target.closest('.sb-profile-row');
  if (!row) return;
  var dx = _swipeCurrentX - _swipeStartX;
  row.querySelector('.sb-folder-content').style.transition = '';
  if (dx < -_swipeThreshold) {{
    row.classList.add('swiped');
    _swipedRow = row;
  }} else {{
    row.querySelector('.sb-folder-content').style.transform = '';
    row.classList.remove('swiped');
    if (_swipedRow === row) _swipedRow = null;
  }}
}});

// Close swipe on click outside
document.addEventListener('click', function(e) {{
  if (!e.target.closest('.sb-profile-row')) closeAllSwipes();
}});

var ppData = {pp_data_json};
var pmData = {pm_data_json};
var currentView = 'aggregate';
var costChart = null, latChart = null;
var profileColors = {{
  'l2': {{bg: 'hsla(142.1, 70.6%, 45.3%, 0.4)', border: 'hsl(142.1, 70.6%, 45.3%)'}},
  'l1': {{bg: 'hsla(190, 80%, 50%, 0.4)', border: 'hsl(190, 80%, 50%)'}},
  'career': {{bg: 'hsla(43.3, 96.4%, 56.3%, 0.4)', border: 'hsl(43.3, 96.4%, 56.3%)'}},
  'cron': {{bg: 'hsla(0, 83.2%, 60.2%, 0.4)', border: 'hsl(0, 83.2%, 60.2%)'}}
}};

function buildCharts(view) {{
  if (costChart) {{ costChart.destroy(); costChart = null; }}
  if (latChart) {{ latChart.destroy(); latChart = null; }}

  var darkOpts = {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#a1a1aa' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#a1a1aa', maxTicksLimit: 10 }} }},
      y: {{ ticks: {{ color: '#a1a1aa' }} }}
    }}
  }};

  var costDatasets, latDatasets, labels;

  if (view === 'aggregate') {{
    labels = {ts_dates_json};
    costDatasets = [{{
      label: 'Daily Cost ($)',
      data: {ts_costs_json},
      backgroundColor: 'hsla(142.1, 70.6%, 45.3%, 0.4)',
      borderColor: 'hsl(142.1, 70.6%, 45.3%)',
      borderWidth: 1
    }}];
    latDatasets = [{{
      label: 'Avg Latency (ms)',
      data: {ts_lats_json},
      borderColor: 'hsl(190, 80%, 50%)',
      backgroundColor: 'hsla(190, 80%, 50%, 0.1)',
      fill: true, tension: 0.3
    }}];
  }} else if (view === 'per-profile') {{
    labels = ppData.dates;
    costDatasets = [];
    latDatasets = [];
    var profs = Object.keys(ppData.profiles);
    for (var i = 0; i < profs.length; i++) {{
      var p = profs[i];
      var c = profileColors[p] || {{bg: 'hsla(0,0%,50%,0.4)', border: 'hsl(0,0%,50%)'}};
      costDatasets.push({{
        label: p.toUpperCase() + ' Cost ($)',
        data: ppData.profiles[p].costs,
        backgroundColor: c.bg,
        borderColor: c.border,
        borderWidth: 1
      }});
      latDatasets.push({{
        label: p.toUpperCase() + ' Latency (ms)',
        data: ppData.profiles[p].lats,
        borderColor: c.border,
        backgroundColor: c.bg,
        fill: false, tension: 0.3
      }});
    }}
  }} else if (view === 'per-model') {{
    labels = pmData.dates;
    costDatasets = [];
    latDatasets = [];
    var modelHues = ['142.1', '190', '43.3', '0', '280', '30'];
    var models = Object.keys(pmData.models);
    for (var i = 0; i < models.length; i++) {{
      var m = models[i];
      var hue = modelHues[i % modelHues.length];
      costDatasets.push({{
        label: m + ' Cost ($)',
        data: pmData.models[m].costs,
        backgroundColor: 'hsla(' + hue + ', 70%, 50%, 0.4)',
        borderColor: 'hsl(' + hue + ', 70%, 50%)',
        borderWidth: 1
      }});
      latDatasets.push({{
        label: m + ' Latency (ms)',
        data: pmData.models[m].lats,
        borderColor: 'hsl(' + hue + ', 70%, 50%)',
        backgroundColor: 'hsla(' + hue + ', 70%, 50%, 0.1)',
        fill: false, tension: 0.3
      }});
    }}
  }}

  var ctx1 = document.getElementById('costChart');
  if (ctx1) {{
    costChart = new Chart(ctx1, {{
      type: 'bar',
      data: {{ labels: labels, datasets: costDatasets }},
      options: darkOpts
    }});
  }}
  var ctx2 = document.getElementById('latencyChart');
  if (ctx2) {{
    latChart = new Chart(ctx2, {{
      type: 'line',
      data: {{ labels: labels, datasets: latDatasets }},
      options: darkOpts
    }});
  }}
}}

function switchView(view) {{
  currentView = view;
  document.getElementById('btnAgg').className = view === 'aggregate' ? 'view-toggle active' : 'view-toggle';
  document.getElementById('btnPP').className = view === 'per-profile' ? 'view-toggle active' : 'view-toggle';
  document.getElementById('btnPM').className = view === 'per-model' ? 'view-toggle active' : 'view-toggle';
  buildCharts(view);
}}

buildCharts('aggregate');

/* ── Provider Management ── */
var provData = {{}};
var provPresets = {{}};
var _provTestPassed = false;
var _dirtyChains = {{}};
var _activeTab = '';
window._hostUrl = '{host_url}';

function api(method, url, body) {{
  var opts = {{ method: method, headers: {{'Content-Type':'application/json'}} }};
  if (body) opts.body = JSON.stringify(body);
  return fetch(url, opts).then(function(r) {{ return r.json(); }});
}}

/* ── Modal Open / Close ── */
function openProviderModal() {{
  // Collapse sidebar
  var sidebar = document.getElementById('sidebar');
  if (sidebar && !sidebar.classList.contains('collapsed')) {{
    sidebar.classList.add('collapsed');
  }}
  document.getElementById('provModal').classList.add('open');
  _activeTab = '';
  loadProviders();
}}

function closeProviderModal() {{
  document.getElementById('provModal').classList.remove('open');
  // Restore sidebar
  var sidebar = document.getElementById('sidebar');
  if (sidebar && sidebar.classList.contains('collapsed')) {{
    sidebar.classList.remove('collapsed');
  }}
}}

/* ── Tab Switching ── */
function switchTab(profile) {{
  _activeTab = profile;
  document.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
  document.querySelectorAll('.tab-panel').forEach(function(p) {{ p.classList.remove('active'); }});
  var btn = document.getElementById('tab-'+profile);
  var panel = document.getElementById('panel-'+profile);
  if (btn) btn.classList.add('active');
  if (panel) panel.classList.add('active');
  document.getElementById('tabDropdown').value = profile;
}}

/* ── Load & Render ── */
function loadProviders() {{
  api('GET', '/api/providers').then(function(d) {{
    provData = d;
    _dirtyChains = JSON.parse(JSON.stringify(d.profile_chains || {{}}));
    renderTabsAndPanels();
    renderProvList();
  }});
  api('GET', '/api/providers/presets').then(function(d) {{
    provPresets = d.presets;
    var sel = document.getElementById('provPreset');
    sel.innerHTML = '<option value="">-- Custom --</option>';
    Object.keys(provPresets).forEach(function(k) {{
      sel.innerHTML += '<option value="'+k+'">'+k+'</option>';
    }});
  }});
}}

function renderTabsAndPanels() {{
  var chains = provData.profile_chains || {{}};
  var profiles = Object.keys(chains);

  // Tabs
  var tabsHtml = '';
  profiles.forEach(function(p) {{
    tabsHtml += '<button class="tab-btn" id="tab-'+p+'" onclick="switchTab(\\''+p+'\\')">'+p.toUpperCase()+'</button>';
  }});
  document.getElementById('modalTabs').innerHTML = tabsHtml;

  // Mobile dropdown
  var dd = document.getElementById('tabDropdown');
  dd.innerHTML = profiles.map(function(p) {{ return '<option value="'+p+'">'+p.toUpperCase()+'</option>'; }}).join('');

  // Panels
  var panelsHtml = '';
  profiles.forEach(function(pname) {{
    var chain = _dirtyChains[pname] || chains[pname] || [];
    panelsHtml += '<div class="tab-panel" id="panel-'+pname+'">';
    panelsHtml += '<div class="section-label" style="display:flex;justify-content:space-between;align-items:center">';
    panelsHtml += '<span>Fallback Chain — '+pname.toUpperCase()+'</span>';
    panelsHtml += '<button class="btn-sm btn-primary" onclick="addChainItem(\\''+pname+'\\')">+ Add Step</button>';
    panelsHtml += '</div>';
    if (chain.length === 0) {{
      panelsHtml += '<div class="empty" style="padding:1rem;font-size:0.75rem">No providers in chain. Add one.</div>';
    }} else {{
      panelsHtml += '<ul class="chain-list" id="chain-'+pname+'">';
      chain.forEach(function(c, i) {{
        panelsHtml += '<li class="chain-item" data-idx="'+i+'">';
        panelsHtml += '<span class="drag-handle">⋮⋮</span>';
        panelsHtml += '<select class="chain-prov-select" onchange="updateChainItem(\\''+pname+'\\','+i+', this.value, null)" style="background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:4px;color:hsl(var(--foreground));padding:0.125rem 0.375rem;font-size:0.6875rem">';
        Object.keys(provData.providers || {{}}).forEach(function(pn) {{
          var sel = c.provider === pn ? ' selected' : '';
          panelsHtml += '<option value="'+pn+'"'+sel+'>'+pn+'</option>';
        }});
        panelsHtml += '</select>';
        panelsHtml += '<select class="chain-model-select" onchange="updateChainItem(\\''+pname+'\\','+i+', null, this.value)" style="background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:4px;color:hsl(var(--foreground));padding:0.125rem 0.375rem;font-size:0.6875rem">';
        var models = (provData.providers || {{}})[c.provider]?.models || [];
        models.forEach(function(m) {{
          var sel = c.model === m ? ' selected' : '';
          panelsHtml += '<option value="'+m+'"'+sel+'>'+m+'</option>';
        }});
        panelsHtml += '</select>';
        panelsHtml += '<button class="btn-sm" style="margin-left:auto" onclick="removeChainItem(\\''+pname+'\\','+i+')">✕</button>';
        panelsHtml += '</li>';
      }});
      panelsHtml += '</ul>';
    }}
    panelsHtml += '</div>';
  }});
  document.getElementById('modalBody').innerHTML = panelsHtml;

  // Init SortableJS
  profiles.forEach(function(pname) {{
    var listEl = document.getElementById('chain-'+pname);
    if (listEl) {{
      new Sortable(listEl, {{
        animation: 150, handle: '.drag-handle',
        onEnd: function() {{ rebuildDirtyChain(pname); }}
      }});
    }}
  }});

  // Select first tab
  if (profiles.length > 0 && !_activeTab) switchTab(profiles[0]);
  else if (_activeTab) switchTab(_activeTab);
}}

function rebuildDirtyChain(profile) {{
  var items = document.querySelectorAll('#chain-'+profile+' .chain-item');
  var chain = [];
  items.forEach(function(item) {{
    var pSel = item.querySelector('.chain-prov-select');
    var mSel = item.querySelector('.chain-model-select');
    chain.push({{provider: pSel.value, model: mSel.value}});
  }});
  _dirtyChains[profile] = chain;
}}

function updateChainItem(profile, idx, newProvider, newModel) {{
  if (!_dirtyChains[profile]) _dirtyChains[profile] = JSON.parse(JSON.stringify(provData.profile_chains[profile] || []));
  if (newProvider !== null && newProvider !== undefined) _dirtyChains[profile][idx].provider = newProvider;
  if (newModel !== null && newModel !== undefined) _dirtyChains[profile][idx].model = newModel;
}}

function addChainItem(profile) {{
  var provs = Object.keys(provData.providers || {{}});
  if (provs.length === 0) {{ alert('Add a provider first'); return; }}
  if (!_dirtyChains[profile]) _dirtyChains[profile] = [];
  var p = provs[0];
  var m = (provData.providers[p]?.models || [])[0] || 'default';
  _dirtyChains[profile].push({{provider: p, model: m}});
  renderTabsAndPanels();
  switchTab(profile);
}}

function removeChainItem(profile, idx) {{
  if (!_dirtyChains[profile]) _dirtyChains[profile] = JSON.parse(JSON.stringify(provData.profile_chains[profile] || []));
  _dirtyChains[profile].splice(idx, 1);
  renderTabsAndPanels();
  switchTab(profile);
}}

function saveAllChains() {{
  var statusEl = document.getElementById('saveStatus');
  statusEl.textContent = 'Saving...';
  statusEl.style.color = 'hsl(var(--amber-fg))';
  var promises = [];
  Object.keys(_dirtyChains).forEach(function(profile) {{
    promises.push(api('PUT', '/api/chains/'+profile, {{chain: _dirtyChains[profile]}}));
  }});
  Promise.all(promises).then(function(results) {{
    var allOk = results.every(function(r) {{ return r.ok; }});
    if (allOk) {{
      provData.profile_chains = JSON.parse(JSON.stringify(_dirtyChains));
      statusEl.textContent = 'All chains saved';
      statusEl.style.color = 'hsl(var(--green-fg))';
      setTimeout(function() {{ statusEl.textContent = ''; }}, 2000);
    }} else {{
      statusEl.textContent = 'Some saves failed';
      statusEl.style.color = 'hsl(var(--red-fg))';
    }}
  }}).catch(function(e) {{
    statusEl.textContent = 'Error: '+e;
    statusEl.style.color = 'hsl(var(--red-fg))';
  }});
}}

/* ── Provider CRUD ── */
function renderProvList() {{
  var el = document.getElementById('provList');
  var provs = provData.providers || {{}};
  var names = Object.keys(provs);
  if (names.length === 0) {{ el.innerHTML = '<div class="empty">No providers. Add one below.</div>'; return; }}
  var h = '';
  names.forEach(function(n) {{
    var p = provs[n];
    h += '<div class="prov-item">';
    h += '<div><div class="prov-name">'+n+'</div><div class="prov-detail">'+p.api_base+'</div></div>';
    h += '<div class="prov-actions">';
    h += '<button class="btn-sm" onclick="editProvider(\\''+n+'\\')">Edit</button>';
    h += '<button class="btn-sm btn-danger" onclick="deleteProvider(\\''+n+'\\')">Del</button>';
    h += '</div></div>';
  }});
  el.innerHTML = h;
}}

function showAddProvForm() {{
  document.getElementById('provForm').style.display = 'block';
  document.getElementById('provName').value = '';
  document.getElementById('provUrl').value = '';
  document.getElementById('provKeyEnv').value = '';
  document.getElementById('provModels').value = '';
  document.getElementById('testResult').style.display = 'none';
  document.getElementById('btnSave').disabled = true;
  _provTestPassed = false;
}}

function hideAddProvForm() {{
  document.getElementById('provForm').style.display = 'none';
  _provTestPassed = false;
  document.getElementById('btnSave').disabled = true;
}}

function loadPreset() {{
  var key = document.getElementById('provPreset').value;
  if (!key || !provPresets[key]) return;
  var p = provPresets[key];
  document.getElementById('provName').value = key;
  document.getElementById('provUrl').value = p.api_base;
  document.getElementById('provModels').value = (p.models||[]).join(', ');
  document.getElementById('provKeyEnv').value = 'LCP_'+key.toUpperCase()+'_API_KEY';
}}

function editProvider(name) {{
  var p = provData.providers[name];
  if (!p) return;
  showAddProvForm();
  document.getElementById('provName').value = name;
  document.getElementById('provUrl').value = p.api_base || '';
  document.getElementById('provKeyEnv').value = p.api_key_env || '';
  document.getElementById('provModels').value = (p.models||[]).join(', ');
  // For editing existing, test not required but available
  _provTestPassed = true;
  document.getElementById('btnSave').disabled = false;
  document.getElementById('testResult').style.display = 'none';
}}

function testProvider() {{
  var url = document.getElementById('provUrl').value.trim();
  var keyEnv = document.getElementById('provKeyEnv').value.trim();
  var models = document.getElementById('provModels').value.split(',')[0].trim();
  var resultEl = document.getElementById('testResult');
  var btnSave = document.getElementById('btnSave');
  var btnTest = document.getElementById('btnTest');
  resultEl.style.display = 'block';
  resultEl.textContent = 'Testing...';
  resultEl.style.background = 'hsl(var(--secondary))';
  resultEl.style.color = 'hsl(var(--foreground))';
  btnTest.disabled = true;
  var apiKey = prompt('Enter API key for test (not stored):');
  if (!apiKey) {{ resultEl.textContent = 'Test cancelled'; btnTest.disabled = false; return; }}
  api('POST', '/api/providers/test', {{api_base:url, api_key:apiKey, model:models}}).then(function(d) {{
    btnTest.disabled = false;
    if (d.ok) {{
      resultEl.innerHTML = 'Connection OK - model: '+d.model+' (HTTP '+d.status+')';
      resultEl.style.background = 'hsl(var(--green-bg))';
      resultEl.style.color = 'hsl(var(--green-fg))';
      _provTestPassed = true;
      btnSave.disabled = false;
    }} else {{
      resultEl.innerHTML = 'FAILED: '+(d.error||'HTTP '+d.status);
      resultEl.style.background = 'hsl(var(--red-bg))';
      resultEl.style.color = 'hsl(var(--red-fg))';
      _provTestPassed = false;
      btnSave.disabled = true;
    }}
  }}).catch(function(e) {{
    btnTest.disabled = false;
    resultEl.textContent = 'Network error: '+e;
    resultEl.style.background = 'hsl(var(--red-bg))';
    resultEl.style.color = 'hsl(var(--red-fg))';
    _provTestPassed = false;
    btnSave.disabled = true;
  }});
}}

function saveProvider() {{
  var name = document.getElementById('provName').value.trim();
  var url = document.getElementById('provUrl').value.trim();
  var keyEnv = document.getElementById('provKeyEnv').value.trim();
  var models = document.getElementById('provModels').value.split(',').map(function(s){{return s.trim()}}).filter(Boolean);
  if (!name) {{ alert('Provider name required'); return; }}

  // For NEW providers (not editing existing), require test to pass
  var isNew = !provData.providers || !provData.providers[name];
  if (isNew && !_provTestPassed) {{
    alert('You must test the connection successfully before saving a new provider.');
    return;
  }}

  api('POST', '/api/providers', {{name:name, api_base:url, api_key_env:keyEnv, models:models}}).then(function(d) {{
    if (d.ok) {{
      hideAddProvForm();
      _provTestPassed = false;
      loadProviders();
    }} else {{
      alert('Error: '+JSON.stringify(d));
    }}
  }});
}}

function deleteProvider(name) {{
  if (!confirm('Delete provider '+name+'? This removes it from all chains.')) return;
  api('DELETE', '/api/providers/'+name).then(function(d) {{
    if (d.ok) loadProviders();
    else alert('Error: '+JSON.stringify(d));
  }});
}}

loadProviders();

/* ── Provider Health Detail Modal ── */
function showPhDetail(event, el) {{
  event.stopPropagation();
  var data = JSON.parse(el.dataset.detail);
  var statusClass = data.status === 'healthy' ? 'dot-healthy' : data.status === 'degraded' ? 'dot-degraded' : 'dot-dead';
  var statusColor = data.status === 'healthy' ? 'var(--green-fg)' : data.status === 'degraded' ? 'var(--amber-fg)' : 'var(--red-fg)';
  document.getElementById('phModalTitle').innerHTML = '<span class="ph-dot '+statusClass+'" style="display:inline-block;vertical-align:middle;margin-right:0.375rem"></span>'+data.name+' <span class="ph-profile">'+data.profile+'</span>';
  document.getElementById('phModalBody').innerHTML =
    '<div class="phm-label">Status</div><div style="color:'+statusColor+';font-weight:600">'+data.status.toUpperCase()+'</div>'+
    '<div class="phm-label">Base URL</div><div class="phm-value">'+data.url+'</div>'+
    '<div class="phm-label">Consecutive Failures</div><div class="phm-value">'+data.failures+data.tripped+'</div>'+
    '<div class="phm-label">Last Success</div><div class="phm-value">'+data.last_success+'</div>'+
    '<div class="phm-label">Last Failure</div><div class="phm-value">'+data.last_failure+'</div>';
  document.getElementById('phDetailModal').classList.add('open');
}}
function closePhDetail(event) {{
  if (event && event.target !== document.getElementById('phDetailModal')) return;
  document.getElementById('phDetailModal').classList.remove('open');
}}

/* ── Sidebar Tree ── */
function toggleSbFolder(el) {{
  el.classList.toggle('open');
  el.parentElement.classList.toggle('open');
}}
function toggleProvider(event, el) {{
  event.stopPropagation();
  toggleSbFolder(el);
  document.querySelectorAll('.sb-provider.selected').forEach(function(s) {{ s.classList.remove('selected'); }});
  el.classList.add('selected');
}}
function editProviderFromSidebar(event, el) {{
  event.stopPropagation();
  toggleSbFolder(el);  // expand to show models too
  document.querySelectorAll('.sb-provider.selected').forEach(function(s) {{ s.classList.remove('selected'); }});
  el.classList.add('selected');
  var ds = el.dataset;
  var h = ds.status;
  var dotClass = h === 'healthy' ? 'dot-healthy' : h === 'degraded' ? 'dot-degraded' : 'dot-dead';
  var statusColor = h === 'healthy' ? 'var(--green-fg)' : h === 'degraded' ? 'var(--amber-fg)' : 'var(--red-fg)';
  document.getElementById('pemTitle').innerHTML = '<span class="ph-dot '+dotClass+'" style="display:inline-block;margin-right:6px"></span>'+ds.provider;
  document.getElementById('pemStatus').innerHTML = '<span style="color:'+statusColor+';font-weight:700">'+h.toUpperCase()+'</span> <span class="ph-profile" style="font-size:0.625rem">'+ds.profile+'</span>';
  document.getElementById('pemUrl').value = ds.url;
  document.getElementById('pemKeyEnv').value = ds.keyenv || '';
  document.getElementById('pemModels').value = ds.models || '';
  document.getElementById('pemTestResult').style.display = 'none';
  document.getElementById('provEditModal').classList.add('open');
}}
function closeProvEditModal() {{
  document.getElementById('provEditModal').classList.remove('open');
}}
var _pemTestPassed = true;
function testPemProvider() {{
  var url = document.getElementById('pemUrl').value.trim();
  var models = document.getElementById('pemModels').value.split(',')[0].trim();
  var resultEl = document.getElementById('pemTestResult');
  var btn = document.getElementById('pemTestBtn');
  resultEl.style.display = 'block';
  resultEl.textContent = 'Testing...';
  resultEl.style.background = 'hsl(var(--secondary))';
  resultEl.style.color = 'hsl(var(--foreground))';
  btn.disabled = true;
  var apiKey = prompt('Enter API key for test (not stored):');
  if (!apiKey) {{ resultEl.textContent = 'Test cancelled'; btn.disabled = false; return; }}
  api('POST', '/api/providers/test', {{api_base:url, api_key:apiKey, model:models}}).then(function(d) {{
    btn.disabled = false;
    if (d.ok) {{
      resultEl.innerHTML = 'Connection OK - model: '+d.model+' (HTTP '+d.status+')';
      resultEl.style.background = 'hsl(var(--green-bg))';
      resultEl.style.color = 'hsl(var(--green-fg))';
      _pemTestPassed = true;
    }} else {{
      resultEl.innerHTML = 'FAILED: '+(d.error||'HTTP '+d.status);
      resultEl.style.background = 'hsl(var(--red-bg))';
      resultEl.style.color = 'hsl(var(--red-fg))';
      _pemTestPassed = false;
    }}
  }}).catch(function(e) {{
    btn.disabled = false;
    resultEl.textContent = 'Network error: '+e;
    resultEl.style.background = 'hsl(var(--red-bg))';
    resultEl.style.color = 'hsl(var(--red-fg))';
    _pemTestPassed = false;
  }});
}}
function savePemProvider() {{
  var titleEl = document.getElementById('pemTitle');
  var name = titleEl.textContent.replace(/^\\s+/, '').split(' ').pop() || '';
  var url = document.getElementById('pemUrl').value.trim();
  var keyEnv = document.getElementById('pemKeyEnv').value.trim();
  var models = document.getElementById('pemModels').value.split(',').map(function(s){{return s.trim()}}).filter(Boolean);
  if (!name) {{ alert('Provider name missing'); return; }}
  api('POST', '/api/providers', {{name:name, api_base:url, api_key_env:keyEnv, models:models}}).then(function(d) {{
    if (d.ok) {{
      document.getElementById('pemTestResult').style.display = 'block';
      document.getElementById('pemTestResult').textContent = 'Saved. Refresh to see changes.';
      document.getElementById('pemTestResult').style.background = 'hsl(var(--green-bg))';
      document.getElementById('pemTestResult').style.color = 'hsl(var(--green-fg))';
    }} else {{
      alert('Error: '+JSON.stringify(d));
    }}
  }});
}}
// Default expand only top-level profile folders (not providers)
(function() {{
  document.querySelectorAll('.sidebar-nav > .sb-tree > .sb-folder').forEach(function(f) {{ toggleSbFolder(f); }});
}})();

/* ── Profiles & Keys ── */
function addProfile() {{
  var name = prompt('New profile name (lowercase, e.g. "admin"):');
  if (!name) return;
  name = name.trim().toLowerCase();
  if (!/^[a-z0-9_-]+$/.test(name)) {{ alert('Invalid name. Use lowercase letters, numbers, hyphens, underscores.'); return; }}
  api('POST', '/api/profiles', {{name: name}}).then(function(d) {{
    if (d.ok) {{ location.reload(); }}
    else {{ alert('Error: '+JSON.stringify(d)); }}
  }});
}}
function copyUrl(url) {{
  navigator.clipboard.writeText(url).then(function() {{
    // brief flash
  }}).catch(function() {{
    prompt('Copy this URL:', url);
  }});
}}
/* ── Profile Config Modal ── */
var _pcmProfile = null;

function openProfileConfig(event, profile) {{
  event.stopPropagation();
  _pcmProfile = profile;
  document.getElementById('pcmTitle').textContent = 'Profile: ' + profile.toUpperCase();
  document.getElementById('pcmUrl').textContent = window._hostUrl + '/' + profile + '/chat/completions';
  document.getElementById('pcmCurl').textContent = 'curl -H "Authorization: Bearer lcp_YOUR_KEY" ' + window._hostUrl + '/' + profile + '/chat/completions';
  switchPcmTab('pcm-apikeys');
  loadProfileKeys(profile);
  var modal = document.getElementById('profileConfigModal');
  if (modal) {{ modal.classList.add('open'); }}
}}

function closeProfileConfig() {{
  document.getElementById('profileConfigModal').classList.remove('open');
}}

function switchPcmTab(tabId) {{
  document.querySelectorAll('#pcmTabs .modal-tab').forEach(function(t) {{
    t.classList.toggle('active', t.dataset.tab === tabId);
  }});
  document.querySelectorAll('#pcmBody .tab-panel').forEach(function(p) {{
    p.classList.toggle('active', p.id === 'panel-' + tabId);
  }});
}}

function loadProfileKeys(profile) {{
  api('GET', '/api/keys').then(function(d) {{
    var el = document.getElementById('pcmKeysList');
    var keys = (d.keys || []).filter(function(k) {{ return k.profile === profile; }});
    if (keys.length === 0) {{ el.innerHTML = '<div class="empty">No API keys for this profile.</div>'; return; }}
    var h = '';
    keys.forEach(function(k) {{
      h += '<div class="prov-item">';
      h += '<div><div class="prov-name">'+k.label+'</div><div class="prov-detail">Created: '+(k.created||'').slice(0,10)+(k.last_used ? ' | Last used: '+k.last_used.slice(0,10) : '')+'</div></div>';
      h += '<div class="prov-actions">';
      h += '<button class="btn-sm btn-danger" onclick="deleteProfileKey(\\''+k.id+'\\')">Revoke</button>';
      h += '</div></div>';
    }});
    el.innerHTML = h;
  }});
}}

function generateProfileKey() {{
  if (!_pcmProfile) return;
  var label = prompt('Label (e.g. "L2 agent"):', 'Key for ' + _pcmProfile);
  if (!label) return;
  api('POST', '/api/keys', {{profile: _pcmProfile, label: label}}).then(function(d) {{
    if (d.ok) {{
      var copied = false;
      try {{ navigator.clipboard.writeText(d.key); copied = true; }} catch(e) {{}}
      alert('API Key (copied to clipboard):\\n\\n'+d.key+'\\n\\n'+(copied ? 'Copied! Save it now — it won\\'t be shown again.' : 'SAVE THIS KEY — it won\\'t be shown again.'));
      loadProfileKeys(_pcmProfile);
    }} else {{
      alert('Error: '+JSON.stringify(d));
    }}
  }});
}}

function deleteProfileKey(id) {{
  if (!confirm('Revoke this API key? It will stop working immediately.')) return;
  api('DELETE', '/api/keys/'+id).then(function(d) {{
    if (d.ok) loadProfileKeys(_pcmProfile);
    else alert('Error: '+JSON.stringify(d));
  }});
}}

function copyProfileUrl() {{
  var url = document.getElementById('pcmUrl').textContent;
  navigator.clipboard.writeText(url).then(function() {{
    // brief flash
  }}).catch(function() {{
    prompt('Copy this URL:', url);
  }});
}}

// Profile config modal tab clicks
document.addEventListener('DOMContentLoaded', function() {{
  // Profile config modal tabs
  var pcmTabs = document.getElementById('pcmTabs');
  if (pcmTabs) {{
    pcmTabs.addEventListener('click', function(e) {{
      var tab = e.target.closest('.modal-tab');
      if (tab && tab.dataset.tab) switchPcmTab(tab.dataset.tab);
    }});
  }}
  // Backdrop click to close profile config modal
  var pcm = document.getElementById('profileConfigModal');
  if (pcm) {{
    pcm.addEventListener('click', function(e) {{
      if (e.target === pcm) closeProfileConfig();
    }});
  }}
  // Backdrop click to close keys modal
  var km = document.getElementById('keysModal');
  if (km) {{
    km.addEventListener('click', function(e) {{
      if (e.target === km) closeKeysModal();
    }});
  }}
}});

function openKeysModal() {{
  loadKeys();
  document.getElementById('keysModal').classList.add('open');
}}
function closeKeysModal() {{
  document.getElementById('keysModal').classList.remove('open');
}}
function loadKeys() {{
  api('GET', '/api/keys').then(function(d) {{
    var el = document.getElementById('keysList');
    var keys = d.keys || [];
    if (keys.length === 0) {{ el.innerHTML = '<div class="empty">No API keys yet.</div>'; return; }}
    var h = '';
    keys.forEach(function(k) {{
      h += '<div class="prov-item">';
      h += '<div><div class="prov-name">'+k.label+'</div><div class="prov-detail">Profile: '+k.profile+' | Created: '+(k.created||'').slice(0,10)+'</div></div>';
      h += '<div class="prov-actions">';
      h += '<button class="btn-sm btn-danger" onclick="deleteKey(\\''+k.id+'\\')">Revoke</button>';
      h += '</div></div>';
    }});
    el.innerHTML = h;
  }});
}}
function generateKey() {{
  var profiles = Object.keys(provData.profile_chains || {{}});
  if (profiles.length === 0) {{ alert('No profiles exist. Create one first.'); return; }}
  var profile = prompt('Profile for this key ('+profiles.join(', ')+'):', profiles[0]);
  if (!profile) return;
  var label = prompt('Label (e.g. "L2 agent"):', 'Key for '+profile);
  if (!label) return;
  api('POST', '/api/keys', {{profile: profile, label: label}}).then(function(d) {{
    if (d.ok) {{
      var copied = false;
      try {{ navigator.clipboard.writeText(d.key); copied = true; }} catch(e) {{}}
      alert('API Key (copied to clipboard):\\n\\n'+d.key+'\\n\\n'+(copied ? 'Copied! Save it now — it won\\'t be shown again.' : 'SAVE THIS KEY — it won\\'t be shown again.'));
      loadKeys();
    }} else {{
      alert('Error: '+JSON.stringify(d));
    }}
  }});
}}
function deleteKey(id) {{
  if (!confirm('Revoke this API key? It will stop working immediately.')) return;
  api('DELETE', '/api/keys/'+id).then(function(d) {{
    if (d.ok) loadKeys();
    else alert('Error: '+JSON.stringify(d));
  }});
}}
</script>
</div><!-- .main-content -->

<!-- ── Provider Modal ── -->
<div class="modal-overlay" id="provModal">
<div class="modal">
  <div class="modal-header">
    <h2>Provider Configuration</h2>
    <button class="modal-close" onclick="closeProviderModal()">✕</button>
  </div>
  <!-- Desktop tabs -->
  <div class="modal-tabs" id="modalTabs"></div>
  <!-- Mobile dropdown -->
  <div class="tab-dropdown">
    <select id="tabDropdown" onchange="switchTab(this.value)"></select>
  </div>
  <div class="modal-body" id="modalBody">
    <!-- Tab panels injected by JS -->
  </div>
  <!-- Provider add form (below tabs, shared) -->
  <div class="modal-body" style="border-top:1px solid hsl(var(--card-border));padding-top:0.75rem" id="provFormSection">
    <div class="section-label" style="display:flex;justify-content:space-between;align-items:center">
      <span>All Providers</span>
      <button class="btn-sm btn-primary" onclick="showAddProvForm()">+ Add Provider</button>
    </div>
    <div id="provList" class="prov-list" style="margin-top:0.5rem">Loading...</div>
    <div id="provForm" class="modal-prov-form" style="display:none">
      <label>Provider Name</label>
      <input id="provName" placeholder="e.g. opencode">
      <label>API Base URL</label>
      <input id="provUrl" placeholder="https://api.example.com/v1">
      <label>API Key Env Var</label>
      <input id="provKeyEnv" placeholder="e.g. LCP_OPENCODE_API_KEY">
      <label>Models (comma-separated)</label>
      <input id="provModels" placeholder="model-a, model-b">
      <label>Load Preset</label>
      <select id="provPreset" onchange="loadPreset()">
        <option value="">-- Custom --</option>
      </select>
      <div style="display:flex;gap:0.5rem;margin-top:0.75rem">
        <button class="btn-sm btn-success" id="btnTest" onclick="testProvider()">Test Connection</button>
        <button class="btn-sm btn-primary" id="btnSave" onclick="saveProvider()" disabled>Save Provider</button>
        <button class="btn-sm" onclick="hideAddProvForm()">Cancel</button>
      </div>
      <div id="testResult" class="test-result" style="display:none"></div>
    </div>
  </div>
  <div class="modal-footer">
    <span style="font-size:0.6875rem;color:hsl(var(--muted-foreground));margin-right:auto;align-self:center" id="saveStatus"></span>
    <button class="btn-sm" onclick="closeProviderModal()">Close</button>
    <button class="btn-sm btn-primary" id="btnSaveAll" onclick="saveAllChains()">Save All Chains</button>
  </div>
</div>
</div>

<!-- ── Provider Health Detail Modal ── -->
<div class="ph-modal-overlay" id="phDetailModal" onclick="closePhDetail(event)">
<div class="ph-modal" onclick="event.stopPropagation()">
  <div class="ph-modal-header">
    <span class="ph-modal-title" id="phModalTitle"></span>
    <button class="modal-close" onclick="closePhDetail()">✕</button>
  </div>
  <div class="ph-modal-body" id="phModalBody"></div>
</div>
</div>

</body>
</html>"""

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _serve_daily_costs_api(self):
        """API endpoint: daily costs as JSON."""
        from sqlalchemy import func
        try:
            with get_session(self.engine) as session:
                rows = (
                    session.query(
                        func.substr(RequestModel.timestamp, 1, 10).label("date"),
                        func.coalesce(func.sum(RequestModel.cost), 0).label("cost"),
                        func.count(RequestModel.id).label("requests"),
                    )
                    .group_by(func.substr(RequestModel.timestamp, 1, 10))
                    .order_by(func.substr(RequestModel.timestamp, 1, 10).desc())
                    .limit(90)
                    .all()
                )
            data = [{"date": r.date, "cost": float(r.cost), "requests": r.requests} for r in rows]
            self._send_json({"daily_costs": data})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _serve_recent_requests_api(self):
        """API endpoint: last 100 requests as JSON."""
        try:
            with get_session(self.engine) as session:
                rows = (
                    session.query(RequestModel)
                    .order_by(RequestModel.id.desc())
                    .limit(100)
                    .all()
                )
            data = [
                {
                    "timestamp": r.timestamp,
                    "profile": r.profile,
                    "model": r.model,
                    "provider": r.provider,
                    "cost": r.cost,
                    "latency_ms": r.latency_ms,
                    "success": bool(r.success),
                    "error_type": r.error_type,
                }
                for r in rows
            ]
            self._send_json({"requests": data})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    # ── Provider Configuration API ─────────────────────────────────────────

    def _serve_providers_list(self):
        cfg = self.config
        providers = {}
        for name, pdata in cfg.providers.items():
            providers[name] = {
                "api_key_env": pdata.get("api_key_env", ""),
                "api_base": pdata.get("api_base", ""),
                "models": pdata.get("models", []),
            }
        profile_chains = {}
        for pname, pcfg in cfg.profiles.items():
            profile_chains[pname] = pcfg.get("chain", [])
        self._send_json({"providers": providers, "profile_chains": profile_chains})

    def _serve_provider_presets(self):
        presets = {
            "deepseek": {"api_base": "https://api.deepseek.com/v1", "models": ["deepseek-v4-pro", "deepseek-v4-flash", "deepseek-chat"]},
            "opencode": {"api_base": "https://opencode.ai/zen/go/v1", "models": ["deepseek-v4-pro", "deepseek-v4-flash"]},
            "openai": {"api_base": "https://api.openai.com/v1", "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1"]},
            "anthropic": {"api_base": "https://api.anthropic.com/v1", "models": ["claude-sonnet-4-20250514", "claude-haiku-3-5-20241022"]},
            "groq": {"api_base": "https://api.groq.com/openai/v1", "models": ["llama-4-maverick-17b", "llama-4-scout-17b"]},
            "xai": {"api_base": "https://api.x.ai/v1", "models": ["grok-3"]},
        }
        self._send_json({"presets": presets})

    def _serve_provider_create(self):
        try:
            body = self._read_body()
        except Exception:
            self._send_json({"error": "invalid JSON body"}, 400)
            return
        name = body.get("name")
        if not name:
            self._send_json({"error": "missing 'name' field"}, 400)
            return
        cfg = self.config
        cfg.raw.setdefault("providers", {})[name] = {
            "api_key_env": body.get("api_key_env", f"LCP_{name.upper()}_API_KEY"),
            "api_base": body.get("api_base", ""),
            "models": body.get("models", []),
        }
        cfg.save()
        self._send_json({"ok": True, "provider": name})

    def _serve_provider_update(self, name: str):
        try:
            body = self._read_body()
        except Exception:
            self._send_json({"error": "invalid JSON body"}, 400)
            return
        cfg = self.config
        if name not in cfg.providers:
            self._send_json({"error": f"provider '{name}' not found"}, 404)
            return
        pdata = cfg.raw["providers"][name]
        if "api_key_env" in body:
            pdata["api_key_env"] = body["api_key_env"]
        if "api_base" in body:
            pdata["api_base"] = body["api_base"]
        if "models" in body:
            pdata["models"] = body["models"]
        cfg.save()
        self._send_json({"ok": True, "provider": name})

    def _serve_provider_delete(self, name: str):
        cfg = self.config
        if name not in cfg.providers:
            self._send_json({"error": f"provider '{name}' not found"}, 404)
            return
        del cfg.raw["providers"][name]
        for pname, pcfg in cfg.raw["profiles"].items():
            pcfg["chain"] = [c for c in pcfg.get("chain", []) if c.get("provider") != name]
        cfg.save()
        self._send_json({"ok": True, "deleted": name})

    def _serve_provider_test(self):
        try:
            body = self._read_body()
        except Exception:
            self._send_json({"error": "invalid JSON body"}, 400)
            return
        api_base = body.get("api_base", "").rstrip("/")
        api_key = body.get("api_key", "")
        model = body.get("model", "")
        if not api_base or not api_key:
            self._send_json({"error": "missing 'api_base' or 'api_key'"}, 400)
            return
        import urllib.request, ssl
        url = f"{api_base}/chat/completions"
        test_body = json.dumps({
            "model": model or "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        }).encode()
        req = urllib.request.Request(url, data=test_body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        })
        try:
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                result = json.loads(resp.read().decode())
                model_used = result.get("model", "unknown")
                self._send_json({"ok": True, "model": model_used, "status": resp.status})
        except urllib.error.HTTPError as e:
            err_body = e.read().decode()[:300] if e.fp else ""
            self._send_json({"ok": False, "status": e.code, "error": err_body})
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)})

    def _serve_chain_reorder(self, profile: str):
        try:
            body = self._read_body()
        except Exception:
            self._send_json({"error": "invalid JSON body"}, 400)
            return
        cfg = self.config
        if profile not in cfg.profiles:
            self._send_json({"error": f"profile '{profile}' not found"}, 404)
            return
        new_chain = body.get("chain")
        if not isinstance(new_chain, list):
            self._send_json({"error": "missing 'chain' list"}, 400)
            return
        # Preserve base_url from existing chain entries
        old_chain = cfg.raw["profiles"][profile].get("chain", [])
        old_by_prov = {}
        for s in old_chain:
            old_by_prov[(s.get("provider"), s.get("model"))] = s.get("base_url", "")
        merged = []
        for s in new_chain:
            entry = {"provider": s.get("provider"), "model": s.get("model")}
            key = (entry["provider"], entry["model"])
            if key in old_by_prov and old_by_prov[key]:
                entry["base_url"] = old_by_prov[key]
            merged.append(entry)
        cfg.raw["profiles"][profile]["chain"] = merged
        cfg.save()
        self._send_json({"ok": True, "profile": profile, "chain": merged})

    # ── Profile Management API ────────────────────────────────────────────

    def _serve_profiles_list(self):
        """Return all profiles with their gateway URLs."""
        profiles = {}
        host = self.headers.get("Host", "localhost:8735")
        scheme = "https" if self.headers.get("X-Forwarded-Proto") == "https" else "http"
        base = f"{scheme}://{host}"
        for pname, pcfg in self.config.profiles.items():
            profiles[pname] = {
                "chain": pcfg.get("chain", []),
                "url": f"{base}/{pname}/chat/completions",
                "forbidden": pcfg.get("forbidden_tools", []),
            }
        self._send_json({"profiles": profiles})

    def _serve_profile_create(self):
        try:
            body = self._read_body()
        except Exception:
            self._send_json({"error": "invalid JSON body"}, 400)
            return
        name = body.get("name", "").strip().lower()
        if not name:
            self._send_json({"error": "missing 'name' field"}, 400)
            return
        cfg = self.config
        if name in cfg.profiles:
            self._send_json({"error": f"profile '{name}' already exists"}, 409)
            return
        cfg.raw["profiles"][name] = {
            "chain": [],
            "forbidden_tools": [],
        }
        cfg.save()
        self._send_json({"ok": True, "profile": name})

    # ── API Key Management ────────────────────────────────────────────────

    def _load_keys(self) -> dict:
        import os as _os
        path = _os.environ.get("COST_DB", "/app/data/costs.db").rsplit("/", 1)[0] + "/api_keys.json"
        if _os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {"keys": []}

    def _save_keys(self, data: dict):
        import os as _os
        path = _os.environ.get("COST_DB", "/app/data/costs.db").rsplit("/", 1)[0] + "/api_keys.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _serve_keys_list(self):
        data = self._load_keys()
        # Return keys without hashes
        safe = []
        for k in data.get("keys", []):
            safe.append({
                "id": k["id"],
                "profile": k.get("profile", ""),
                "label": k.get("label", ""),
                "created": k.get("created", ""),
                "last_used": k.get("last_used"),
            })
        self._send_json({"keys": safe})

    def _serve_key_create(self):
        try:
            body = self._read_body()
        except Exception:
            self._send_json({"error": "invalid JSON body"}, 400)
            return
        profile = body.get("profile", "").strip()
        label = body.get("label", "").strip()
        if not profile:
            self._send_json({"error": "missing 'profile' field"}, 400)
            return
        import hashlib, secrets, uuid
        raw_key = "lcp_" + secrets.token_hex(24)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        data = self._load_keys()
        entry = {
            "id": str(uuid.uuid4())[:8],
            "profile": profile,
            "label": label or f"Key for {profile}",
            "hash": key_hash,
            "created": datetime.now(timezone.utc).isoformat(),
            "last_used": None,
        }
        data.setdefault("keys", []).append(entry)
        self._save_keys(data)
        self._send_json({"ok": True, "key": raw_key, "id": entry["id"], "profile": profile, "label": entry["label"]})

    def _serve_key_delete(self, key_id: str):
        data = self._load_keys()
        before = len(data.get("keys", []))
        data["keys"] = [k for k in data.get("keys", []) if k["id"] != key_id]
        if len(data["keys"]) == before:
            self._send_json({"error": "key not found"}, 404)
            return
        self._save_keys(data)
        self._send_json({"ok": True, "deleted": key_id})


# ── Server Bootstrap ─────────────────────────────────────────────────────────

def main():
    """Entry point — start the HTTP server."""
    # Config
    config = init_config()
    cfg = config.server

    # Logging
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    setup_logging(log_level)

    # Database
    db_path = os.environ.get("COST_DB", config.database.get("path", "/app/data/costs.db"))
    engine = get_engine(db_path)

    # Ensure tables exist
    from .models import Base
    Base.metadata.create_all(engine)

    # Server
    port = int(os.environ.get("LISTEN_PORT", str(cfg.get("port", 8734))))
    server = ThreadingHTTPServer(("0.0.0.0", port), LCPHandler)

    # Inject config + engine into handler class
    LCPHandler.config = config
    LCPHandler.engine = engine

    logger.info("server_starting", port=port, profiles=list(config.profiles.keys()))
    print(f"LLM Control Plane v{__import__('src').__version__} listening on :{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("server_shutdown")
        server.shutdown()


if __name__ == "__main__":
    main()
