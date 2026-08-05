"""LCPHandler — HTTP request handler with routing and chat completion logic."""

import json
import os
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
from typing import Any

from ..api.config import get_config
from ..api.logging_config import get_logger
from ..api.circuit_breaker import get_circuit_breaker
from ..api.request_pipeline import (
    strip_forbidden_tools,
    calculate_cost,
    try_chain,
    record_cost,
)
from ..api.cost_estimator import estimate_from_request
from ..api.prompt_cache import get_prompt_cache
from ..api.token_verifier import get_token_verifier
from ..api.key_manager import get_key_manager
from ..api.exceptions import ToolBlockedError, AllProvidersFailedError, ProviderBadRequestError
from .sse_helpers import extract_last_sse_chunk, estimate_cost_from_tokens
from .endpoints import (
    HealthEndpoints,
    ProviderEndpoints,
    ProfileEndpoints,
    KeyEndpoints,
    AlertEndpoints,
    PluginEndpoints,
    UsageEndpoints,
    DashboardEndpoints,
)

logger = get_logger("lcp.server")


class LCPHandler(
    HealthEndpoints,
    ProviderEndpoints,
    ProfileEndpoints,
    KeyEndpoints,
    AlertEndpoints,
    PluginEndpoints,
    UsageEndpoints,
    DashboardEndpoints,
    BaseHTTPRequestHandler,
):
    """HTTP request handler for LCP gateway."""

    # Class-level references set after server init
    config: Any = None
    engine: Any = None

    def log_message(self, format, *args):
        """Suppress default http.server logging — we use structlog."""
        pass

    def _send_json(self, data: dict, status: int = 200):
        """Send a JSON response. Logs but does not crash on client disconnect."""
        body = json.dumps(data).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logger.debug("client_disconnected", path=self.path, status=status)

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

    def _serve_static(self):
        """Serve static files (JS, CSS, etc.) from the Jinja2 templates/static dir."""
        from pathlib import Path

        relative = self.path[len("/static/"):].split("?")[0]
        if ".." in relative or relative.startswith("/"):
            self._send_json({"error": "forbidden"}, 403)
            return

        static_dir = Path(__file__).resolve().parent.parent / "ui" / "templates" / "jinja" / "static"
        file_path = (static_dir / relative).resolve()
        if not str(file_path).startswith(str(static_dir.resolve())):
            self._send_json({"error": "forbidden"}, 403)
            return

        content_type = {
            ".js": "application/javascript",
            ".css": "text/css",
            ".html": "text/html",
            ".svg": "image/svg+xml",
            ".png": "image/png",
        }.get(file_path.suffix, "application/octet-stream")

        if not file_path.is_file():
            self._send_json({"error": "not found"}, 404)
            return

        data = file_path.read_bytes()
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache, max-age=0")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logger.debug("client_disconnected", path=self.path, static_file=relative)

    # ── Routes ────────────────────────────────────────────────────────────

    # Configurable models endpoint paths (env: LCP_MODELS_PATHS, default: /v1/models,/models)
    _models_paths = set(
        p.strip() for p in os.environ.get("LCP_MODELS_PATHS", "/v1/models,/models").split(",") if p.strip()
    )

    def do_GET(self):
        logger.debug("request_start", method="GET", path=self.path,
                     client_ip=self.client_address[0])
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
        elif self.path in self._models_paths:
            self._serve_models()
        elif any(self.path.endswith("/" + p.lstrip("/")) for p in self._models_paths):
            # Per-profile: /coder/v1/models or /coder/models
            profile = self._resolve_profile()
            if profile and profile in self.config.profiles:
                self._serve_models(profile=profile)
            else:
                self._send_json({"error": "not found"}, 404)
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
        elif self.path == "/api/logs" or self.path.startswith("/api/logs?"):
            self._serve_logs_api()
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
        elif self.path == "/api/cost-plugins/summary":
            self._serve_plugin_summary()
        elif self.path == "/api/cost-plugins/subscriptions":
            self._serve_plugin_subscriptions()
        elif self.path == "/api/usage/stats" or self.path.startswith("/api/usage/stats?"):
            self._serve_usage_stats_api()
        elif self.path == "/api/usage/totals" or self.path.startswith("/api/usage/totals?"):
            self._serve_usage_totals_api()
        elif self.path.startswith("/static/"):
            self._serve_static()
        elif self.path == "/usage":
            self._serve_usage_page()
        elif self.path == "/logs":
            self._serve_logs_page()
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        # Config hot-reload check
        self.config.check_reload()

        profile = self._resolve_profile()
        logger.debug("request_start", method="POST", path=self.path,
                     client_ip=self.client_address[0], profile=profile or "none")

        # Provider API routes
        if self.path == "/api/providers":
            self._serve_provider_create()
            return
        elif self.path == "/api/providers/test":
            self._serve_provider_test()
            return
        elif self.path == "/api/providers/discover":
            self._serve_provider_discover()
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
            logger.warning("invalid_route", method="POST", path=self.path,
                           client_ip=self.client_address[0])
            self._send_json({"error": "not found"}, 404)
            return

        # Read body early — needed for model→profile fallback and validation
        try:
            body = self._read_body()
        except Exception:
            self._send_json({"error": "invalid JSON body"}, 400)
            return

        profile = self._resolve_profile()
        if profile is None:
            # Fallback: resolve profile from "model" field in request body
            model_name = (body.get("model") or "").strip()
            if model_name in self.config.profiles:
                profile = model_name
                logger.info("profile_from_model", model=model_name, path=self.path)
            else:
                logger.warning("unknown_profile", path=self.path,
                               client_ip=self.client_address[0], model=model_name)
                self._send_json({"error": f"unknown profile in path: {self.path}. Use /PROFILE/chat/completions or set model to a profile name."}, 400)
                return

        profile_cfg = self.config.get_profile(profile)
        if profile_cfg is None:
            self._send_json({"error": f"profile not found: {profile}"}, 400)
            return

        # Auth check — if profile requires API key, validate Authorization header
        if profile_cfg.get("auth_required", True):
            auth_header = self.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                logger.warning("auth_failed", reason="missing_bearer_token",
                               profile=profile, client_ip=self.client_address[0])
                self._send_json({"error": "API key required for this profile. Use Authorization: Bearer <key>"}, 401)
                return
            raw_key = auth_header[7:]
            km = get_key_manager()
            if km:
                key_info = km.validate_key(raw_key)
                if key_info is None:
                    logger.warning("auth_failed", reason="invalid_or_revoked_key",
                                   profile=profile, client_ip=self.client_address[0])
                    self._send_json({"error": "invalid or revoked API key"}, 401)
                    return
                # Check profile access
                allowed = key_info.get("allowed_profiles")
                if allowed:
                    allowed_list = [p.strip() for p in allowed.split(",") if p.strip()]
                    if profile not in allowed_list:
                        logger.warning("auth_failed", reason="profile_access_denied",
                                       profile=profile, client_ip=self.client_address[0],
                                       key_id=key_info.get("id"))
                        self._send_json({"error": f"key does not have access to profile '{profile}'"}, 403)
                        return
                # Check spend limit
                limit = key_info.get("spend_limit", 0)
                spent = key_info.get("total_spend", 0)
                if limit > 0 and spent >= limit:
                    logger.warning("auth_failed", reason="spend_limit_exceeded",
                                   profile=profile, client_ip=self.client_address[0],
                                   key_id=key_info.get("id"), spent=round(spent, 2),
                                   limit=round(limit, 2))
                    self._send_json({"error": f"spend limit exceeded (${spent:.2f} / ${limit:.2f})"}, 429)
                    return
                self._current_key_id = key_info.get("id")

        try:
            # Validate body has messages (after auth so we return 401 first if unauthenticated)
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
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("X-LCP-Cache", "HIT")
                    self.send_header("X-Estimated-Cost", str(estimation["estimated_total_cost"]))
                    self.end_headers()
                    self.wfile.write(json.dumps(cached).encode("utf-8"))
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    logger.debug("client_disconnected", path=self.path, cache="HIT")
                logger.info("cache_hit_served", profile=profile, model=primary_model)
                return

            # Set estimated cost header
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

            # ── Streaming response ──
            if streaming:
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("X-LCP-Cache", "MISS")
                self.send_header("X-Estimated-Cost", str(estimation["estimated_total_cost"]))
                self.end_headers()

                sse_parts = []
                try:
                    for chunk in response_body:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        sse_parts.append(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    logger.debug("client_disconnected", path=self.path, stream="SSE")

                full_sse = b"".join(sse_parts)
                last_chunk = extract_last_sse_chunk(full_sse)
                if last_chunk and last_chunk.get("usage"):
                    cost_info = {
                        "prompt_tokens": last_chunk["usage"].get("prompt_tokens", 0),
                        "completion_tokens": last_chunk["usage"].get("completion_tokens", 0),
                        "cache_hit_tokens": last_chunk["usage"].get("prompt_cache_hit_tokens", 0),
                        "cache_miss_tokens": last_chunk["usage"].get("prompt_cache_miss_tokens", 0),
                        "cost": 0,
                        "latency_ms": latency_ms,
                    }
                    cost_info["cost"] = estimate_cost_from_tokens(
                        provider, model, cost_info, self.config
                    )
                else:
                    # SSE stream without usage — fall back to pre-flight estimation
                    cost_info = {
                        "prompt_tokens": estimation["input_tokens"],
                        "completion_tokens": 0,
                        "cache_hit_tokens": 0,
                        "cache_miss_tokens": estimation["input_tokens"],
                        "cost": estimation["estimated_total_cost"],
                        "latency_ms": latency_ms,
                    }
                record_cost(self.engine, profile, model, provider, cost_info, True, None, blocked_tools)

                total_wall_ms = int((time.time() - t0) * 1000)
                logger.info(
                    "request_complete",
                    profile=profile,
                    provider=provider,
                    model=model,
                    latency_ms=latency_ms,
                    total_wall_ms=total_wall_ms,
                    tools_blocked=len(blocked_tools),
                    cache="MISS",
                    stream=True,
                )
                return

            # ── Non-streaming ──
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
                            from ..api.alert_manager import get_alert_manager
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
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body_bytes)))
                for hdr_name, hdr_val in self._pending_headers.items():
                    self.send_header(hdr_name, hdr_val)
                self.end_headers()
                self.wfile.write(body_bytes)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                logger.debug("client_disconnected", path=self.path, stream=False)

            total_wall_ms = int((time.time() - t0) * 1000)
            logger.info(
                "request_complete",
                profile=profile,
                provider=provider,
                model=model,
                cost=round(cost_info["cost"], 6),
                latency_ms=latency_ms,
                total_wall_ms=total_wall_ms,
                tools_blocked=len(blocked_tools),
                cache="MISS",
            )

        except ToolBlockedError as e:
            logger.warning("tool_blocked", profile=profile, error=str(e))
            self._send_json({"error": str(e)}, 403)
        except ProviderBadRequestError as e:
            # The provider rejected the request body as invalid (HTTP 400).
            # Report the exact error to the client so they can fix their request.
            logger.error("provider_bad_request", profile=profile, error=str(e))
            cost_info = {"prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0,
                         "cache_miss_tokens": 0, "cost": 0, "latency_ms": 0}
            record_cost(self.engine, profile, "unknown", "unknown", cost_info, False,
                       "provider_bad_request", [], error_detail=str(e))
            self._send_json({"error": str(e)}, 400)
        except AllProvidersFailedError as e:
            logger.error("all_providers_failed", profile=profile, error=str(e))
            cost_info = {"prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0,
                         "cache_miss_tokens": 0, "cost": 0, "latency_ms": 0}
            record_cost(self.engine, profile, "unknown", "unknown", cost_info, False,
                       "all_providers_failed", [], error_detail=str(e))
            self._send_json({"error": str(e)}, 502)
        except Exception as e:
            logger.error("unhandled_error", error=str(e), traceback=traceback.format_exc()[-500:])
            self._send_json({"error": "internal error: {}".format(str(e))}, 500)

    def do_PUT(self):
        self.config.check_reload()
        logger.debug("request_start", method="PUT", path=self.path,
                     client_ip=self.client_address[0])
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
        logger.debug("request_start", method="DELETE", path=self.path,
                     client_ip=self.client_address[0])
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
