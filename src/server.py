"""HTTP server and request handler for the smallm gateway.

Contains the LCPHandler class that dispatches requests to the appropriate
pipeline, dashboard, and API modules.
"""

import json
import os
import time
import traceback
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Any

from .api.config import get_config
from .api.logging_config import get_logger
from .api.circuit_breaker import get_circuit_breaker
from .ui.dashboard import render_dashboard
from .api.cost_plugins import get_registry, init_plugins
from .api.request_pipeline import (
    strip_forbidden_tools,
    calculate_cost,
    try_chain,
    record_cost,
)
from .api.cost_estimator import estimate_from_request
from .api.prompt_cache import get_prompt_cache
from .api.token_verifier import get_token_verifier
from .api.key_manager import get_key_manager, init_key_manager
from .api.alert_manager import get_alert_manager
from .api.models import get_session, Request as RequestModel
from .api.exceptions import ToolBlockedError, AllProvidersFailedError

logger = get_logger("lcp.server")

# ── SSE helpers ──────────────────────────────────────────────────────────────

def _extract_last_sse_chunk(raw_bytes):
    """Parse the last data chunk from an SSE response buffer."""
    try:
        text = raw_bytes.decode("utf-8", errors="replace")
        last_data = None
        for line in text.split("\n"):
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str and data_str != "[DONE]":
                    last_data = json.loads(data_str)
        return last_data
    except Exception:
        return None


def _estimate_cost_from_tokens(provider, model, cost_info, config):
    """Calculate cost from token counts using configured pricing or plugins."""
    # Try plugin registry first
    usage_for_plugin = {
        "prompt_tokens": cost_info.get("prompt_tokens", 0)
                        + cost_info.get("cache_miss_tokens", 0),
        "completion_tokens": cost_info.get("completion_tokens", 0),
        "prompt_cache_hit_tokens": cost_info.get("cache_hit_tokens", 0),
        "prompt_cache_miss_tokens": cost_info.get("cache_miss_tokens", 0),
    }
    plugin_cost = get_registry().calculate_cost(provider, model, usage_for_plugin)
    if plugin_cost is not None:
        return round(plugin_cost, 8)

    # Fall back to config-based pricing
    pricing = config.get_pricing(provider, model)
    cache_hit = cost_info.get("cache_hit_tokens", 0)
    cache_miss = cost_info.get("cache_miss_tokens", 0)
    output = cost_info.get("completion_tokens", 0)
    return round(
        (cache_hit / 1_000_000) * pricing["cache_hit"]
        + (cache_miss / 1_000_000) * pricing["cache_miss"]
        + (output / 1_000_000) * pricing["output"],
        8,
    )


class LCPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for smallm gateway."""

    # Class-level references set after server init
    config: Any = None
    engine: Any = None

    def log_message(self, format, *args):
        """Suppress default http.server logging — we use structlog."""
        pass

    def _send_json(self, data: dict, status: int = 200):
        """Send a JSON response."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
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
        elif self.path == "/keys" or self.path == "/keys/dashboard":
            self._serve_keys_dashboard()
        elif self.path == "/providers":
            self._serve_providers_page()
        elif self.path == "/profiles":
            self._serve_profiles_page()
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
        elif self.path.startswith("/api/keys/") and len(self.path.split("/")) == 4:
            key_id = self.path.split("/")[3]
            self._serve_key_detail(key_id)
        elif self.path == "/api/alerts":
            self._serve_alerts_list()
        elif self.path == "/api/alerts/config":
            self._serve_alerts_config()
        elif self.path == "/api/alerts/active":
            self._serve_alerts_active()
        elif self.path == "/api/cost-plugins/usage":
            self._serve_plugin_usage()
        elif self.path == "/api/cost-plugins/balances":
            self._serve_plugin_balances()
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
        elif self.path.startswith("/api/keys/") and self.path.endswith("/rotate"):
            key_id = self.path.split("/")[3]
            self._serve_key_rotate(key_id)
            return
        elif self.path == "/api/alerts/webhook/test":
            self._serve_alerts_test_webhook()
            return
        elif self.path.startswith("/api/alerts/") and self.path.endswith("/acknowledge"):
            alert_id = self.path.split("/")[3]
            self._serve_alert_acknowledge(alert_id)
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

        # Auth check — if profile requires API key, validate Authorization header
        if profile_cfg.get("auth_required", True):
            auth_header = self.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                self._send_json({"error": "API key required for this profile. Use Authorization: Bearer <key>"}, 401)
                return
            raw_key = auth_header[7:]
            km = get_key_manager()
            if km:
                key_info = km.validate_key(raw_key)
                if key_info is None:
                    self._send_json({"error": "invalid or revoked API key"}, 401)
                    return
                # Check profile access
                allowed = key_info.get("allowed_profiles")
                if allowed:
                    allowed_list = [p.strip() for p in allowed.split(",") if p.strip()]
                    if profile not in allowed_list:
                        self._send_json({"error": f"key does not have access to profile '{profile}'"}, 403)
                        return
                # Check spend limit
                limit = key_info.get("spend_limit", 0)
                spent = key_info.get("total_spend", 0)
                if limit > 0 and spent >= limit:
                    self._send_json({"error": f"spend limit exceeded (${spent:.2f} / ${limit:.2f})"}, 429)
                    return
                self._current_key_id = key_info.get("id")

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

            # Prompt cache check (skip cache for streaming requests - cached JSON cannot satisfy SSE)
            cache = get_prompt_cache()
            primary_model = profile_cfg["chain"][0]["model"]
            cached = cache.get(profile, primary_model, body) if not body.get("stream", False) else None
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

            streaming = body.get("stream", False)

            # Try provider chain
            response_body, status, provider, model = try_chain(
                profile, profile_cfg, body, self.config
            )

            latency_ms = int((time.time() - t0) * 1000)

            # ── Streaming response: forward SSE chunks in real time ──
            if streaming:
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("X-LCP-Cache", "MISS")
                self.send_header("X-Estimated-Cost", str(estimation["estimated_total_cost"]))
                self.end_headers()

                # Stream chunks from upstream to client as they arrive.
                # response_body is a generator yielding raw bytes from the provider.
                sse_parts = []
                for chunk in response_body:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    sse_parts.append(chunk)

                # Connection will be closed after this handler returns
                # (Connection: close header).

                # Extract usage from last SSE chunk for cost tracking
                full_sse = b"".join(sse_parts)
                last_chunk = _extract_last_sse_chunk(full_sse)
                if last_chunk and "usage" in last_chunk:
                    cost_info = {
                        "prompt_tokens": last_chunk["usage"].get("prompt_tokens", 0),
                        "completion_tokens": last_chunk["usage"].get("completion_tokens", 0),
                        "cache_hit_tokens": last_chunk["usage"].get("prompt_cache_hit_tokens", 0),
                        "cache_miss_tokens": last_chunk["usage"].get("prompt_cache_miss_tokens", 0),
                        "cost": 0,
                        "latency_ms": latency_ms,
                    }
                    cost_info["cost"] = _estimate_cost_from_tokens(
                        provider, model, cost_info, self.config
                    )
                    record_cost(self.engine, profile, model, provider, cost_info, True, None, blocked_tools)

                logger.info(
                    "request_complete",
                    profile=profile,
                    provider=provider,
                    model=model,
                    latency_ms=latency_ms,
                    tools_blocked=len(blocked_tools),
                    cache="MISS",
                    stream=True,
                )
                return

            # ── Non-streaming: buffer and return JSON ──
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

            # Track spend on the API key and check limits
            if hasattr(self, '_current_key_id') and self._current_key_id:
                try:
                    km = get_key_manager()
                    if km:
                        breach = km.record_spend(self._current_key_id, cost_info["cost"])
                        if breach:
                            from .api.alert_manager import get_alert_manager
                            am = get_alert_manager()
                            am.fire(
                                rule="budget_breach",
                                severity="warning" if breach["threshold"] < 100 else "critical",
                                title=f"Key '{breach['key_name']}' at {breach['spend_pct']}%",
                                message=f"Key '{breach['key_name']}' has used {breach['spend_pct']}% of its ${breach['limit']:.2f} limit (${breach['current_spend']:.4f} spent).",
                                dedup_key=f"key:{self._current_key_id}:t{breach['threshold']}",
                                metadata=breach,
                            )
                except Exception:
                    pass

            # Send response with custom headers
            body_bytes = json.dumps(response_body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body_bytes)))
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
            self.send_header("Content-Length", str(len(body_bytes)))
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
        elif self.path.startswith("/api/profiles/") and len(self.path.split("/")) == 4:
            profile = self.path.split("/")[3]
            self._serve_profile_update(profile)
        elif self.path == "/api/alerts/config":
            self._serve_alerts_config_update()
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
        elif self.path.startswith("/api/profiles/") and len(self.path.split("/")) == 4:
            profile = self.path.split("/")[3]
            self._serve_profile_delete(profile)
        else:
            self._send_json({"error": "not found"}, 404)

    # ── Static Endpoints ───────────────────────────────────────────────────

    def _serve_health(self):
        provider_status = {}
        cb = get_circuit_breaker()
        for key, h in cb.get_all_health().items():
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
        self.send_header("Content-Length", str(len(body)))
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
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _serve_dashboard(self, profile_filter: str | None = None):
        """Server-rendered dashboard."""
        host = self.headers.get("Host", "localhost:8734")
        scheme = "https" if (
            self.headers.get("X-Forwarded-Proto", "").split(",")[0].strip() == "https"
            or self.headers.get("X-Forwarded-Scheme") == "https"
        ) else "http"
        host_url = f"{scheme}://{host}"
        html = render_dashboard(self.config, self.engine, {"Host": host, "X-Forwarded-Proto": scheme}, profile_filter)
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
        presets = get_registry().presets
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
        import urllib.request, urllib.error, ssl
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
            # Prefer base_url from incoming request, fall back to existing chain
            if s.get("base_url"):
                entry["base_url"] = s["base_url"]
            else:
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
        scheme = "https" if (
            self.headers.get("X-Forwarded-Proto", "").split(",")[0].strip() == "https"
            or self.headers.get("X-Forwarded-Scheme") == "https"
        ) else "http"
        base = f"{scheme}://{host}"
        for pname, pcfg in self.config.profiles.items():
            profiles[pname] = {
                "chain": pcfg.get("chain", []),
                "url": f"{base}/{pname}/chat/completions",
                "forbidden": pcfg.get("forbidden_tools", []),
                "auth_required": pcfg.get("auth_required", True),
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

    def _serve_profile_update(self, name: str):
        try:
            body = self._read_body()
        except Exception:
            self._send_json({"error": "invalid JSON body"}, 400)
            return
        cfg = self.config
        if name not in cfg.profiles:
            self._send_json({"error": f"profile '{name}' not found"}, 404)
            return
        pcfg = cfg.raw["profiles"][name]
        if "forbidden_tools" in body:
            pcfg["forbidden_tools"] = body["forbidden_tools"]
        if "chain" in body:
            pcfg["chain"] = body["chain"]
        if "auth_required" in body:
            pcfg["auth_required"] = body["auth_required"]
        cfg.save()
        self._send_json({"ok": True, "profile": name})

    def _serve_profile_delete(self, name: str):
        cfg = self.config
        if name not in cfg.profiles:
            self._send_json({"error": f"profile '{name}' not found"}, 404)
            return
        del cfg.raw["profiles"][name]
        cfg.save()
        self._send_json({"ok": True, "deleted": name})

    # ── API Key Management ────────────────────────────────────────────────

    def _serve_keys_list(self):
        km = get_key_manager()
        keys = km.list_keys() if km else []
        self._send_json({"keys": keys})

    def _serve_key_detail(self, key_id: str):
        km = get_key_manager()
        if not km:
            self._send_json({"error": "key manager not initialized"}, 500)
            return
        try:
            kid = int(key_id)
        except ValueError:
            self._send_json({"error": "invalid key id"}, 400)
            return
        key = km.get_key(kid)
        if key:
            self._send_json({"key": key})
        else:
            self._send_json({"error": "key not found"}, 404)

    def _serve_key_create(self):
        try:
            body = self._read_body()
        except Exception:
            self._send_json({"error": "invalid JSON body"}, 400)
            return
        km = get_key_manager()
        if not km:
            self._send_json({"error": "key manager not initialized"}, 500)
            return
        result = km.create_key(
            name=body.get("name", ""),
            allowed_profiles=body.get("allowed_profiles", ""),
            spend_limit=float(body.get("spend_limit", 0) or 0),
            expires_at=body.get("expires_at", ""),
            metadata_tags=body.get("metadata_tags", ""),
        )
        self._send_json(result)

    def _serve_key_rotate(self, key_id: str):
        km = get_key_manager()
        if not km:
            self._send_json({"error": "key manager not initialized"}, 500)
            return
        try:
            kid = int(key_id)
        except ValueError:
            self._send_json({"error": "invalid key id"}, 400)
            return
        result = km.rotate_key(kid)
        if result:
            self._send_json(result)
        else:
            self._send_json({"error": "key not found"}, 404)

    def _serve_key_delete(self, key_id: str):
        km = get_key_manager()
        if not km:
            self._send_json({"error": "key manager not initialized"}, 500)
            return
        try:
            kid = int(key_id)
        except ValueError:
            self._send_json({"error": "invalid key id"}, 400)
            return
        ok = km.revoke_key(kid)
        if ok:
            self._send_json({"ok": True, "deleted": kid})
        else:
            self._send_json({"error": "key not found"}, 404)

    # ── Alert Management ──────────────────────────────────────────────────

    def _serve_alerts_list(self):
        am = get_alert_manager()
        limit = 100
        status = None
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if "limit" in params:
            limit = int(params["limit"][0])
        if "status" in params:
            status = params["status"][0]
        alerts = am.list_alerts(limit=limit, status=status)
        self._send_json({"alerts": alerts})

    def _serve_alerts_active(self):
        am = get_alert_manager()
        self._send_json({"alerts": am.get_active_alerts()})

    # ── Cost Plugin API ──────────────────────────────────────────────────

    def _serve_plugin_usage(self):
        """Return aggregated usage from all cost tracking plugins."""
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(self.path).query)
        start = qs.get("start", [None])[0]
        end = qs.get("end", [None])[0]
        data = get_registry().fetch_all_usage(start_date=start, end_date=end)
        self._send_json({"plugin_usage": data})

    def _serve_plugin_balances(self):
        """Return account balances from all cost tracking plugins."""
        data = get_registry().fetch_all_balances()
        self._send_json({"plugin_balances": data})

    def _serve_alerts_config(self):
        am = get_alert_manager()
        self._send_json({"config": am.config})

    def _serve_alerts_config_update(self):
        try:
            body = self._read_body()
        except Exception:
            self._send_json({"error": "invalid JSON body"}, 400)
            return
        am = get_alert_manager()
        config = am.update_config(body)
        self._send_json({"ok": True, "config": config})

    def _serve_alert_acknowledge(self, alert_id: str):
        am = get_alert_manager()
        ok = am.acknowledge(alert_id)
        if ok:
            self._send_json({"ok": True, "acknowledged": alert_id})
        else:
            self._send_json({"error": "alert not found"}, 404)

    def _serve_alerts_test_webhook(self):
        am = get_alert_manager()
        result = am.test_webhook()
        self._send_json(result)

    # ── Dashboard Pages ──────────────────────────────────────────────────

    def _serve_keys_dashboard(self):
        """Server-rendered API Keys management page."""
        html = _render_keys_page(self.config, self.engine)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _serve_providers_page(self):
        """Server-rendered Providers management page."""
        html = _render_providers_page(self.config)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _serve_profiles_page(self):
        """Server-rendered Profiles management page."""
        html = _render_profiles_page(self.config)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))


def _render_providers_page(config) -> str:
    """Render the Providers management page."""
    from pathlib import Path
    _templates_dir = Path(__file__).parent / "ui" / "templates"
    css = ""
    try:
        css = (_templates_dir / "dashboard.css").read_text()
    except Exception:
        pass

    prov_rows = ""
    for name, pdata in config.providers.items():
        models = ", ".join(pdata.get("models", []))
        prov_rows += (
            f'<tr>'
            f'<td><b>{name}</b></td>'
            f'<td class="mono" style="max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{pdata.get("api_base", "—")}</td>'
            f'<td>{pdata.get("api_key_env", "—")}</td>'
            f'<td>{models or "—"}</td>'
            f'<td><button class="btn-sm" onclick="editProvider(\'{name}\')">Edit</button> '
            f'<button class="btn-sm btn-danger" onclick="deleteProvider(\'{name}\')">Del</button></td>'
            f'</tr>'
        )
    if not prov_rows:
        prov_rows = '<tr><td colspan="5" class="empty">No providers configured.</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>smallm — Providers</title>
<style>{css}</style>
</head>
<body>
{_render_sidebar_html(config, "providers")}
<div class="sidebar-overlay" id="sidebarOverlay" onclick="closeSidebar()"></div>
<button class="sidebar-toggle" id="sidebarToggle" onclick="toggleSidebar()">☰</button>
<div class="main-content">
<h1>Providers</h1>
<p class="subtitle">Manage LLM API providers and their models</p>

<div style="display:flex;gap:0.5rem;margin-bottom:1rem">
  <button class="btn-sm btn-primary" onclick="showAddProvForm()">+ Add Provider</button>
  <select id="provPreset" onchange="loadPreset()" style="padding:0.3rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.75rem">
    <option value="">-- Quick-add preset --</option>
  </select>
</div>

<div class="prov-form" id="provForm" style="display:none">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem">
    <div>
      <label>Provider Name</label>
      <input id="provName" placeholder="e.g. openai">
    </div>
    <div>
      <label>API Base URL</label>
      <input id="provUrl" placeholder="https://api.openai.com/v1">
    </div>
    <div>
      <label>API Key Env Var</label>
      <input id="provKeyEnv" placeholder="OPENAI_API_KEY">
    </div>
    <div>
      <label>Models (comma-separated)</label>
      <input id="provModels" placeholder="gpt-4o, gpt-4o-mini">
    </div>
  </div>
  <div style="display:flex;gap:0.5rem;margin-top:0.75rem">
    <button class="btn-sm btn-success" id="provTestBtn" onclick="testProvider()">Test Connection</button>
    <button class="btn-sm btn-primary" id="provSaveBtn" onclick="saveProvider()" disabled>Save Provider</button>
    <button class="btn-sm" onclick="hideAddProvForm()">Cancel</button>
  </div>
  <div id="testResult" class="test-result" style="display:none"></div>
</div>

<div class="table-wrap" style="margin-top:0.75rem">
<table>
<thead><tr>
  <th>Name</th><th>Base URL</th><th>Key Env</th><th>Models</th><th>Actions</th>
</tr></thead>
<tbody id="providersBody">{prov_rows}</tbody>
</table>
</div>
</div>

<script>
function api(method, url, body) {{
  var opts = {{method:method, headers:{{'Content-Type':'application/json'}}}};
  if (body) opts.body = JSON.stringify(body);
  return fetch(url, opts).then(function(r) {{ return r.json(); }});
}}

function loadPresets() {{
  api('GET', '/api/providers/presets').then(function(d) {{
    var sel = document.getElementById('provPreset');
    Object.keys(d.presets || {{}}).forEach(function(k) {{
      sel.innerHTML += '<option value="' + k + '">' + k + '</option>';
    }});
  }});
}}

function loadPreset() {{
  var key = document.getElementById('provPreset').value;
  if (!key) return;
  showAddProvForm();
  api('GET', '/api/providers/presets').then(function(d) {{
    var p = (d.presets || {{}})[key];
    if (!p) return;
    document.getElementById('provName').value = key;
    document.getElementById('provUrl').value = p.api_base || '';
    document.getElementById('provModels').value = (p.models || []).join(', ');
    document.getElementById('provKeyEnv').value = 'LCP_' + key.toUpperCase() + '_API_KEY';
    document.getElementById('provSaveBtn').disabled = false;
  }});
}}

function showAddProvForm() {{
  document.getElementById('provForm').style.display = 'block';
  document.getElementById('provName').value = '';
  document.getElementById('provUrl').value = '';
  document.getElementById('provKeyEnv').value = '';
  document.getElementById('provModels').value = '';
  document.getElementById('testResult').style.display = 'none';
  document.getElementById('provSaveBtn').disabled = true;
}}

function hideAddProvForm() {{
  document.getElementById('provForm').style.display = 'none';
}}

function testProvider() {{
  var resultEl = document.getElementById('testResult');
  resultEl.style.display = 'block';
  resultEl.textContent = 'Testing...';
  resultEl.style.color = 'hsl(var(--amber-fg))';
  api('POST', '/api/providers/test', {{
    api_base: document.getElementById('provUrl').value,
    api_key: '',
    model: (document.getElementById('provModels').value || 'default').split(',')[0].trim()
  }}).then(function(d) {{
    if (d.ok) {{
      resultEl.textContent = 'Connected — model: ' + d.model;
      resultEl.style.color = 'hsl(var(--green-fg))';
      document.getElementById('provSaveBtn').disabled = false;
    }} else {{
      resultEl.textContent = 'Failed: ' + (d.error || 'unknown');
      resultEl.style.color = 'hsl(var(--red-fg))';
    }}
  }});
}}

function editProvider(name) {{
  api('GET', '/api/providers').then(function(d) {{
    var p = (d.providers || {{}})[name];
    if (!p) return;
    showAddProvForm();
    document.getElementById('provName').value = name;
    document.getElementById('provUrl').value = p.api_base || '';
    document.getElementById('provKeyEnv').value = p.api_key_env || '';
    document.getElementById('provModels').value = (p.models || []).join(', ');
    document.getElementById('provSaveBtn').disabled = false;
  }});
}}

function saveProvider() {{
  var name = document.getElementById('provName').value.trim();
  if (!name) {{ alert('Provider name is required'); return; }}
  var body = {{
    name: name,
    api_base: document.getElementById('provUrl').value.trim(),
    api_key_env: document.getElementById('provKeyEnv').value.trim(),
    models: document.getElementById('provModels').value.split(',').map(function(s) {{ return s.trim(); }}).filter(Boolean)
  }};
  api('POST', '/api/providers', body).then(function(d) {{
    if (d.ok) {{ location.reload(); }}
    else {{ alert('Error: ' + (d.error || 'unknown')); }}
  }});
}}

function deleteProvider(name) {{
  if (!confirm('Delete provider "' + name + '"? This removes it from all profile chains.')) return;
  api('DELETE', '/api/providers/' + name).then(function(d) {{
    if (d.ok) location.reload();
    else alert('Error: ' + (d.error || 'unknown'));
  }});
}}

function toggleSidebar() {{
  var sb = document.getElementById('sidebar');
  if (window.innerWidth <= 768) {{
    sb.classList.toggle('open');
    document.getElementById('sidebarOverlay').classList.toggle('show');
  }} else {{
    sb.classList.toggle('collapsed');
  }}
}}
function closeSidebar() {{
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebarOverlay').classList.remove('show');
}}

loadPresets();
</script>
</body>
</html>"""


def _render_profiles_page(config) -> str:
    """Render the Profiles management page."""
    from pathlib import Path
    _templates_dir = Path(__file__).parent / "ui" / "templates"
    css = ""
    try:
        css = (_templates_dir / "dashboard.css").read_text()
    except Exception:
        pass

    profile_rows = ""
    for pname, pcfg in config.profiles.items():
        chain = pcfg.get("chain", [])
        steps = " → ".join(f"{s['provider']}/{s['model']}" for s in chain) if chain else "—"
        forbidden = ", ".join(pcfg.get("forbidden_tools", []) or []) or "none"
        auth_required = pcfg.get("auth_required", True)
        auth_badge = "key" if auth_required else "public"
        profile_rows += (
            f'<tr>'
            f'<td><b>{pname.upper()}</b></td>'
            f'<td style="font-size:0.6875rem">{auth_badge}</td>'
            f'<td style="font-size:0.75rem">{steps}</td>'
            f'<td style="font-size:0.6875rem;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{forbidden}">{forbidden}</td>'
            f'<td class="mono" style="font-size:0.6875rem">/{pname}/chat/completions</td>'
            f'<td><button class="btn-sm" onclick="editProfile(\'{pname}\')">Edit</button> '
            f'<button class="btn-sm btn-danger" onclick="deleteProfile(\'{pname}\')">Del</button></td>'
            f'</tr>'
        )

    return f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>smallm — Profiles</title>
<style>{css}</style>
<script src="https://cdn.jsdelivr.net/npm/sortablejs@1.15.0/Sortable.min.js"></script>
</head>
<body>
{_render_sidebar_html(config, "profiles")}
<div class="sidebar-overlay" id="sidebarOverlay" onclick="closeSidebar()"></div>
<button class="sidebar-toggle" id="sidebarToggle" onclick="toggleSidebar()">☰</button>
<div class="main-content">
<h1>Profiles</h1>
<p class="subtitle">Manage routing profiles, fallback chains, and tool restrictions</p>

<div style="display:flex;gap:0.5rem;margin-bottom:1rem">
  <button class="btn-sm btn-primary" onclick="addProfile()">+ Add Profile</button>
</div>

<div class="table-wrap">
<table>
<thead><tr>
  <th>Profile</th><th>Auth</th><th>Chain</th><th>Blocked Tools</th><th>Gateway URL</th><th>Actions</th>
</tr></thead>
<tbody id="profilesBody">{profile_rows}</tbody>
</table>
</div>
</div>

<!-- Profile Edit Modal -->
<div class="modal-overlay" id="profileEditModal">
<div class="modal" style="width:min(600px,95vw)">
  <div class="modal-header">
    <h2 id="pemProfileTitle">Edit Profile</h2>
    <button class="modal-close" onclick="closeProfileEdit()">✕</button>
  </div>
  <div class="modal-body">
    <div style="margin-bottom:0.75rem">
      <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Access Control</label>
      <label style="display:flex;align-items:center;gap:0.5rem;cursor:pointer;font-size:0.8125rem">
        <input type="checkbox" id="pemAuthRequired" onchange="this.nextElementSibling.textContent = this.checked ? 'API key required' : 'No key required (public)'">
        <span>API key required</span>
      </label>
    </div>
    <div style="margin-bottom:0.75rem">
      <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Blocked Tools</label>
      <div id="pemToolsList" style="display:flex;flex-wrap:wrap;gap:0.25rem;margin-bottom:0.375rem"></div>
      <div style="display:flex;gap:0.375rem">
        <input id="pemNewTool" placeholder="tool_name" style="flex:1;padding:0.3rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.75rem">
        <button class="btn-sm btn-primary" onclick="addBlockedTool()">+ Add</button>
      </div>
    </div>
    <div>
      <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Fallback Chain (drag to reorder)</label>
      <ul class="chain-list" id="pemChainList"></ul>
      <button class="btn-sm btn-primary" onclick="addChainStep()" style="margin-top:0.25rem">+ Add Step</button>
    </div>
  </div>
  <div class="modal-footer">
    <span id="pemSaveStatus" style="font-size:0.6875rem;color:hsl(var(--muted-foreground));margin-right:auto"></span>
    <button class="btn-sm btn-primary" onclick="saveProfileEdit()">Save</button>
    <button class="btn-sm" onclick="closeProfileEdit()">Cancel</button>
  </div>
</div>
</div>

<script>
function api(method, url, body) {{
  var opts = {{method:method, headers:{{'Content-Type':'application/json'}}}};
  if (body) opts.body = JSON.stringify(body);
  return fetch(url, opts).then(function(r) {{ return r.json(); }});
}}

function addProfile() {{
  var name = prompt('Profile name (lowercase, e.g. "admin"):');
  if (!name) return;
  api('POST', '/api/profiles', {{ name: name }}).then(function(d) {{
    if (d.ok) location.reload();
    else alert('Error: ' + (d.error || 'unknown'));
  }});
}}

function deleteProfile(name) {{
  if (!confirm('Delete profile "' + name + '"? This cannot be undone.')) return;
  api('DELETE', '/api/profiles/' + name).then(function(d) {{
    if (d.ok) location.reload();
    else alert('Error: ' + (d.error || 'unknown'));
  }});
}}

var _editProfileName = '';
var _editProfileTools = [];
var _editProfileChain = [];
var _allProviders = {{}};
var _allChains = {{}};

function editProfile(name) {{
  _editProfileName = name;
  api('GET', '/api/providers').then(function(d) {{
    _allProviders = d.providers || {{}};
    _allChains = d.profile_chains || {{}};
    _editProfileChain = JSON.parse(JSON.stringify(_allChains[name] || []));
    api('GET', '/api/profiles').then(function(pd) {{
      var prof = (pd.profiles || {{}})[name] || {{}};
      _editProfileTools = (prof.forbidden || []).slice();
      document.getElementById('pemProfileTitle').textContent = 'Edit Profile: ' + name.toUpperCase();
      var authReq = prof.auth_required !== false; // default true
      document.getElementById('pemAuthRequired').checked = authReq;
      document.getElementById('pemAuthRequired').nextElementSibling.textContent = authReq ? 'API key required' : 'No key required (public)';
      renderPemTools();
      renderPemChain();
      document.getElementById('profileEditModal').classList.add('open');
    }});
  }});
}}

function closeProfileEdit() {{
  document.getElementById('profileEditModal').classList.remove('open');
}}

function renderPemTools() {{
  var html = '';
  _editProfileTools.forEach(function(t, i) {{
    html += '<span style="display:inline-flex;align-items:center;gap:0.1875rem;padding:0.125rem 0.5rem;background:hsl(var(--red-bg));color:hsl(var(--red-fg));border-radius:9999px;font-size:0.6875rem;font-weight:600">' +
      t +
      '<button onclick="removeBlockedTool(' + i + ')" style="background:none;border:none;color:inherit;cursor:pointer;font-size:0.75rem;padding:0;line-height:1">✕</button>' +
    '</span>';
  }});
  if (!html) html = '<span style="font-size:0.6875rem;color:hsl(var(--muted-foreground))">no tools blocked</span>';
  document.getElementById('pemToolsList').innerHTML = html;
}}

function addBlockedTool() {{
  var t = document.getElementById('pemNewTool').value.trim();
  if (!t) return;
  if (_editProfileTools.indexOf(t) >= 0) return;
  _editProfileTools.push(t);
  document.getElementById('pemNewTool').value = '';
  renderPemTools();
}}

function removeBlockedTool(idx) {{
  _editProfileTools.splice(idx, 1);
  renderPemTools();
}}

function renderPemChain() {{
  var provNames = Object.keys(_allProviders);
  var html = '';
  if (_editProfileChain.length === 0) {{
    html = '<li class="empty" style="padding:0.5rem;font-size:0.6875rem">No providers in chain</li>';
  }} else {{
    _editProfileChain.forEach(function(s, i) {{
      html += '<li class="chain-item" data-idx="' + i + '">';
      html += '<span class="drag-handle">⋮⋮</span>';
      html += '<select onchange="_editProfileChain[' + i + '].provider = this.value; renderPemChain();" style="background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:4px;color:hsl(var(--foreground));padding:0.125rem 0.375rem;font-size:0.6875rem">';
      provNames.forEach(function(pn) {{
        var sel = s.provider === pn ? ' selected' : '';
        html += '<option value="' + pn + '"' + sel + '>' + pn + '</option>';
      }});
      html += '</select>';
      var models = _allProviders[s.provider]?.models || [];
      html += '<select onchange="_editProfileChain[' + i + '].model = this.value" style="background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:4px;color:hsl(var(--foreground));padding:0.125rem 0.375rem;font-size:0.6875rem">';
      models.forEach(function(m) {{
        var sel = s.model === m ? ' selected' : '';
        html += '<option value="' + m + '"' + sel + '>' + m + '</option>';
      }});
      html += '</select>';
      html += '<button class="btn-sm" style="margin-left:auto" onclick="_editProfileChain.splice(' + i + ',1);renderPemChain();">✕</button>';
      html += '</li>';
    }});
  }}
  document.getElementById('pemChainList').innerHTML = html;
  var listEl = document.getElementById('pemChainList');
  if (listEl && typeof Sortable !== 'undefined') {{
    new Sortable(listEl, {{
      animation: 150, handle: '.drag-handle',
      onEnd: function() {{
        var items = document.querySelectorAll('#pemChainList .chain-item');
        var newChain = [];
        items.forEach(function(item) {{
          var idx = parseInt(item.getAttribute('data-idx'));
          newChain.push(_editProfileChain[idx]);
        }});
        _editProfileChain = newChain;
        renderPemChain();
      }}
    }});
  }}
}}

function addChainStep() {{
  var provs = Object.keys(_allProviders);
  if (provs.length === 0) {{ alert('Add a provider first'); return; }}
  var p = provs[0];
  var m = (_allProviders[p]?.models || [])[0] || 'default';
  _editProfileChain.push({{provider: p, model: m}});
  renderPemChain();
}}

function saveProfileEdit() {{
  var statusEl = document.getElementById('pemSaveStatus');
  statusEl.textContent = 'Saving...';
  statusEl.style.color = 'hsl(var(--amber-fg))';
  var chain = _editProfileChain.map(function(s) {{
    var bu = '';
    var old = _allChains[_editProfileName] || [];
    old.forEach(function(o) {{
      if (o.provider === s.provider && o.model === s.model) bu = o.base_url || '';
    }});
    return {{provider: s.provider, model: s.model, base_url: bu}};
  }});
  // Save chain first, then profile — sequentially to avoid race on config file
  api('PUT', '/api/chains/' + _editProfileName, {{chain: chain}}).then(function(chainResult) {{
    return api('PUT', '/api/profiles/' + _editProfileName, {{
      forbidden_tools: _editProfileTools,
      auth_required: document.getElementById('pemAuthRequired').checked
    }});
  }}).then(function(profResult) {{
    if (profResult.ok) {{
      statusEl.textContent = 'Saved';
      statusEl.style.color = 'hsl(var(--green-fg))';
      setTimeout(function() {{ location.reload(); }}, 800);
    }} else {{
      statusEl.textContent = 'Save failed';
      statusEl.style.color = 'hsl(var(--red-fg))';
    }}
  }}).catch(function(e) {{
    statusEl.textContent = 'Save failed: ' + (e.message || e);
    statusEl.style.color = 'hsl(var(--red-fg))';
  }});
}}

function toggleSidebar() {{
  var sb = document.getElementById('sidebar');
  if (window.innerWidth <= 768) {{
    sb.classList.toggle('open');
    document.getElementById('sidebarOverlay').classList.toggle('show');
  }} else {{
    sb.classList.toggle('collapsed');
  }}
}}
function closeSidebar() {{
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebarOverlay').classList.remove('show');
}}
</script>
</body>
</html>"""


def _render_keys_page(config, engine) -> str:
    """Render the API Keys management page."""
    from pathlib import Path
    _templates_dir = Path(__file__).parent / "ui" / "templates"
    css = ""
    try:
        css = (_templates_dir / "dashboard.css").read_text()
    except Exception:
        pass

    return f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>smallm — API Keys</title>
<style>{css}</style>
</head>
<body>
{_render_sidebar_html(config, "keys")}
<div class="sidebar-overlay" id="sidebarOverlay" onclick="closeSidebar()"></div>
<button class="sidebar-toggle" id="sidebarToggle" onclick="toggleSidebar()">☰</button>
<div class="main-content">
<h1>API Keys</h1>
<p class="subtitle">Manage virtual keys for API access</p>

<div style="display:flex;gap:0.5rem;margin-bottom:1rem">
  <button class="btn-sm btn-primary" onclick="showCreateKeyModal()">+ Create Key</button>
  <input type="text" id="keySearch" placeholder="Search keys..." oninput="filterKeys()"
    style="padding:0.3rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem;width:200px">
</div>

<div class="table-wrap">
<table id="keysTable">
<thead><tr>
  <th>Name</th><th>Prefix</th><th>Profiles</th><th>Spend</th><th>Limit</th><th>Status</th><th>Created</th><th>Actions</th>
</tr></thead>
<tbody id="keysBody"><tr><td colspan="8" class="empty">Loading...</td></tr></tbody>
</table>
</div>
</div>

<!-- Create Key Modal -->
<div class="modal-overlay" id="createKeyModal">
<div class="modal" style="width:min(500px,95vw)">
  <div class="modal-header">
    <h2>Create API Key</h2>
    <button class="modal-close" onclick="closeCreateKeyModal()">✕</button>
  </div>
  <div class="modal-body">
    <div style="display:grid;gap:0.75rem">
      <div>
        <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Key Name</label>
        <input id="newKeyName" placeholder="e.g. Production Key" style="width:100%;padding:0.375rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem">
      </div>
      <div>
        <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Allowed Profiles</label>
        <div id="newKeyProfilesPills" style="display:flex;flex-wrap:wrap;gap:0.375rem"></div>
        <div style="font-size:0.625rem;color:hsl(var(--muted-foreground));margin-top:0.25rem">Click to toggle · none selected = all profiles</div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem">
        <div>
          <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Spend Limit ($, 0=unlimited)</label>
          <input id="newKeyLimit" type="number" step="0.01" min="0" value="0" style="width:100%;padding:0.375rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem">
        </div>
        <div>
          <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Expires At (optional)</label>
          <input id="newKeyExpires" type="date" style="width:100%;padding:0.375rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem">
        </div>
      </div>
    </div>
  </div>
  <div class="modal-footer">
    <button class="btn-sm btn-primary" onclick="createKey()">Create</button>
    <button class="btn-sm" onclick="closeCreateKeyModal()">Cancel</button>
  </div>
</div>
</div>

<!-- Show Key Modal -->
<div class="modal-overlay" id="showKeyModal">
<div class="modal" style="width:min(450px,95vw)">
  <div class="modal-header">
    <h2>Key Created</h2>
    <button class="modal-close" onclick="closeShowKeyModal()">✕</button>
  </div>
  <div class="modal-body">
    <div class="phm-label">API Key (copy now — won't be shown again)</div>
    <div style="display:flex;align-items:center;gap:0.5rem;margin-top:0.25rem">
      <code id="shownKey" style="flex:1;background:hsl(var(--secondary));padding:0.5rem;border-radius:var(--radius);font-size:0.8125rem;word-break:break-all;user-select:all"></code>
      <button class="btn-sm btn-primary" onclick="copyShownKey()">Copy</button>
    </div>
    <div id="shownKeyInfo" style="margin-top:0.5rem;font-size:0.75rem;color:hsl(var(--muted-foreground))"></div>
  </div>
  <div class="modal-footer">
    <button class="btn-sm" onclick="closeShowKeyModal();loadKeys()">Done</button>
  </div>
</div>
</div>

<script>
var _hostUrl = window.location.origin;

function api(method, url, body) {{
  var opts = {{method:method, headers:{{'Content-Type':'application/json'}}}};
  if (body) opts.body = JSON.stringify(body);
  return fetch(url, opts).then(function(r) {{ return r.json(); }});
}}

function loadKeys() {{
  api('GET', '/api/keys').then(function(d) {{
    var keys = d.keys || [];
    var html = '';
    if (keys.length === 0) {{
      html = '<tr><td colspan="8" class="empty">No keys yet. Create one above.</td></tr>';
    }} else {{
      keys.forEach(function(k) {{
        var spend = '$' + (k.total_spend || 0).toFixed(4);
        var limit = k.spend_limit > 0 ? '$' + k.spend_limit.toFixed(2) : '∞';
        var statusBadge = k.status === 'active'
          ? '<span class="badge badge-success">active</span>'
          : '<span class="badge badge-error">revoked</span>';
        html += '<tr data-search="' + (k.name||'') + ' ' + (k.key_prefix||'') + '">' +
          '<td>' + (k.name || '—') + '</td>' +
          '<td class="mono">' + (k.key_prefix || '—') + '</td>' +
          '<td>' + (k.allowed_profiles || 'all') + '</td>' +
          '<td class="cost mono">' + spend + '</td>' +
          '<td class="cost mono">' + limit + '</td>' +
          '<td>' + statusBadge + '</td>' +
          '<td class="mono" style="font-size:0.6875rem">' + (k.created_at||'').slice(0,10) + '</td>' +
          '<td>' +
            (k.status === 'active'
              ? '<button class="btn-sm" onclick="rotateKey(' + k.id + ')" title="Rotate">↻</button> '
                + '<button class="btn-sm btn-danger" onclick="revokeKey(' + k.id + ')" title="Revoke">✕</button>'
              : '<span style="font-size:0.6875rem;color:hsl(var(--muted-foreground))">' + (k.revoked_at||'').slice(0,10) + '</span>'
            ) +
          '</td>' +
        '</tr>';
      }});
    }}
    document.getElementById('keysBody').innerHTML = html;
  }});
}}

function showCreateKeyModal() {{ 
  document.getElementById('createKeyModal').classList.add('open');
  _selectedProfiles = [];
  loadProfilesForKeys();
}}
function closeCreateKeyModal() {{ 
  document.getElementById('createKeyModal').classList.remove('open');
  _selectedProfiles = [];
}}
function closeShowKeyModal() {{ document.getElementById('showKeyModal').classList.remove('open'); }}

function loadProfilesForKeys() {{
  api('GET', '/api/profiles').then(function(d) {{
    var profiles = Object.keys(d.profiles || {{}});
    var html = '';
    profiles.forEach(function(p) {{
      html += '<span class="profile-pill" data-profile="' + p + '" onclick="toggleProfilePill(this)" style="padding:0.25rem 0.625rem;border-radius:9999px;font-size:0.75rem;font-weight:600;cursor:pointer;border:1px solid hsl(var(--card-border));background:hsl(var(--secondary)/0.3);color:hsl(var(--muted-foreground));transition:all 0.15s;user-select:none">' + p.toUpperCase() + '</span>';
    }});
    if (!html) html = '<span style="font-size:0.6875rem;color:hsl(var(--muted-foreground))">No profiles configured</span>';
    document.getElementById('newKeyProfilesPills').innerHTML = html;
  }});
}}

var _selectedProfiles = [];

function toggleProfilePill(el) {{
  var p = el.getAttribute('data-profile');
  var idx = _selectedProfiles.indexOf(p);
  if (idx >= 0) {{
    _selectedProfiles.splice(idx, 1);
    el.style.background = 'hsl(var(--secondary)/0.3)';
    el.style.color = 'hsl(var(--muted-foreground))';
    el.style.borderColor = 'hsl(var(--card-border))';
  }} else {{
    _selectedProfiles.push(p);
    el.style.background = 'hsl(var(--primary))';
    el.style.color = 'hsl(var(--primary-foreground))';
    el.style.borderColor = 'hsl(var(--primary))';
  }}
}}

function createKey() {{
  var name = document.getElementById('newKeyName').value.trim();
  var limit = parseFloat(document.getElementById('newKeyLimit').value) || 0;
  var expires = document.getElementById('newKeyExpires').value;
  api('POST', '/api/keys', {{
    name: name || 'API Key',
    allowed_profiles: _selectedProfiles.join(','),
    spend_limit: limit,
    expires_at: expires ? expires + 'T00:00:00' : ''
  }}).then(function(d) {{
    if (d.key) {{
      document.getElementById('shownKey').textContent = d.key;
      document.getElementById('shownKeyInfo').innerHTML =
        'Name: <b>' + d.name + '</b> · ID: ' + d.id + '<br>' +
        'Profiles: ' + (d.allowed_profiles || 'all') + ' · Limit: $' + (d.spend_limit || 0);
      closeCreateKeyModal();
      document.getElementById('showKeyModal').classList.add('open');
    }} else {{
      alert('Error: ' + (d.error || 'unknown'));
    }}
  }});
}}

function copyShownKey() {{
  navigator.clipboard.writeText(document.getElementById('shownKey').textContent).then(function() {{
    alert('Key copied to clipboard');
  }});
}}

function rotateKey(id) {{
  if (!confirm('Rotate this key? The old key will be revoked and a new one generated.')) return;
  api('POST', '/api/keys/' + id + '/rotate').then(function(d) {{
    if (d.key) {{
      document.getElementById('shownKey').textContent = d.key;
      document.getElementById('shownKeyInfo').innerHTML =
        'Rotated from ID: ' + d.old_id + ' → <b>' + d.id + '</b><br>' +
        'Name: <b>' + d.name + '</b>';
      document.getElementById('showKeyModal').classList.add('open');
    }} else {{
      alert('Error: ' + (d.error || 'unknown'));
    }}
  }});
}}

function revokeKey(id) {{
  if (!confirm('Revoke this key? It will no longer work for API calls.')) return;
  api('DELETE', '/api/keys/' + id).then(function(d) {{
    if (d.ok) loadKeys();
    else alert('Error: ' + (d.error || 'unknown'));
  }});
}}

function filterKeys() {{
  var q = document.getElementById('keySearch').value.toLowerCase();
  document.querySelectorAll('#keysBody tr').forEach(function(tr) {{
    var txt = (tr.getAttribute('data-search') || '').toLowerCase();
    tr.style.display = txt.includes(q) ? '' : 'none';
  }});
}}

function toggleSidebar() {{
  var sb = document.getElementById('sidebar');
  if (window.innerWidth <= 768) {{
    sb.classList.toggle('open');
    document.getElementById('sidebarOverlay').classList.toggle('show');
  }} else {{
    sb.classList.toggle('collapsed');
  }}
}}
function closeSidebar() {{
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebarOverlay').classList.remove('show');
}}

loadKeys();
</script>
</body>
</html>"""


def _render_sidebar_html(config, active_page: str = "") -> str:
    """Render the sidebar navigation for standalone pages."""
    dash_active = ' class="active"' if active_page == "dashboard" else ""
    keys_active = ' class="active"' if active_page == "keys" else ""
    providers_active = ' class="active"' if active_page == "providers" else ""
    profiles_active = ' class="active"' if active_page == "profiles" else ""

    sidebar = (
        '<aside class="sidebar" id="sidebar">\n'
        '  <div class="sidebar-brand">smallm</div>\n'
        '  <nav class="sidebar-nav">\n'
        f'    <a href="/dashboard"{dash_active}>Dashboard</a>\n'
        f'    <a href="/keys"{keys_active}>API Keys</a>\n'
        f'    <a href="/providers"{providers_active}>Providers</a>\n'
        f'    <a href="/profiles"{profiles_active}>Profiles</a>\n'
        '    <div class="nav-label">Profiles</div>\n'
    )
    for p in config.profiles.keys():
        sidebar += f'    <a href="/{p}/dashboard">{p.upper()}</a>\n'
    sidebar += (
        '  </nav>\n</aside>'
    )
    return sidebar


def create_server(config, engine, port=8734):
    """Create and configure the HTTP server."""
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    import os

    class ConfiguredHandler(LCPHandler):
        pass

    ConfiguredHandler.config = config
    ConfiguredHandler.engine = engine

    # Initialize key manager with engine
    init_key_manager(engine, "data")

    server = ThreadingHTTPServer(("0.0.0.0", port), ConfiguredHandler)
    logger.info("server_created", port=port, profiles=list(config.profiles.keys()))
    return server
