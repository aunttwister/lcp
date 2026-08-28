"""LCPHandler — HTTP request handler with routing and chat completion logic."""

import json
import os
import re
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
from typing import Any

from ..api.logging_config import get_logger
from ..api.request_pipeline import (
    strip_forbidden_tools,
    calculate_cost,
    try_chain,
    record_cost,
    capture_reasoning_from_response,
    capture_reasoning_from_sse,
)
from ..api.cost_estimator import estimate_from_request
from ..api.prompt_cache import get_prompt_cache
from ..api.token_verifier import get_token_verifier
from ..api.key_manager import get_key_manager
from ..api.alert_manager import get_alert_manager
from ..api.runtime import resolve_service
from ..api.models import Budget, ApiKey, get_session
from sqlalchemy import or_
from ..api.exceptions import (
    AllProvidersFailedError,
    AuthError,
    CreditExhaustedError,
    ForbiddenError,
    LCPError,
    ProviderBadRequestError,
    ProviderError,
    ToolBlockedError,
)
from .sse_helpers import extract_last_sse_chunk, estimate_cost_from_tokens
from .endpoints import (
    HealthEndpoints,
    ProviderEndpoints,
    ProfileEndpoints,
    KeyEndpoints,
    AlertEndpoints,
    BudgetEndpoints,
    PluginEndpoints,
    UsageEndpoints,
    DashboardEndpoints,
    SetupEndpoints,
    SettingsEndpoints,
    MemoryEndpoints,
)

logger = get_logger("lcp.server")

# Redact things that look like API keys / bearer tokens before surfacing any
# provider error text to a client.
_SENSITIVE_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_-]{6,}|Bearer\s+[A-Za-z0-9._~+/=-]{8,})",
    re.IGNORECASE,
)
_MAX_CLIENT_ERROR_LEN = 300


def _sanitize_message(msg: str) -> str:
    """Redact secrets and truncate a message for client-facing responses."""
    if not isinstance(msg, str):
        msg = str(msg)
    msg = _SENSITIVE_PATTERN.sub("[REDACTED]", msg)
    if len(msg) > _MAX_CLIENT_ERROR_LEN:
        msg = msg[:_MAX_CLIENT_ERROR_LEN] + "..."
    return msg


def _resolve_pricing(config, provider: str, model: str) -> dict | None:
    """Resolve pricing for cost estimation without hard-failing.

    Order: config (gateway.yaml) → cost-plugin registry (e.g. Command Code's
    built-in pricing) → None (``estimate_from_request`` then uses default rates).

    Returns ``None`` instead of raising so a missing pricing entry (e.g.
    commandcode not in gateway.yaml's ``pricing:`` section) does NOT turn a
    request into a 500 before it ever reaches the provider chain.
    """
    try:
        return config.get_pricing(provider, model)
    except Exception:
        pass
    try:
        from ..api.cost_plugins import get_registry
        p = resolve_service("pricing", fallback=get_registry).get_pricing(provider, model)
        if p is not None:
            return p
    except Exception:
        pass
    logger.warning(
        "pricing_resolved_fallback",
        provider=provider,
        model=model,
        reason="no config or plugin pricing — using default estimate rates",
    )
    return None


class LCPHandler(
    HealthEndpoints,
    ProviderEndpoints,
    ProfileEndpoints,
    KeyEndpoints,
    AlertEndpoints,
    BudgetEndpoints,
    PluginEndpoints,
    UsageEndpoints,
    DashboardEndpoints,
    SetupEndpoints,
    SettingsEndpoints,
    MemoryEndpoints,
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

    def _send_error(self, exc: Exception, status: int | None = None,
                    message: str | None = None) -> None:
        """Send a sanitized, structured error response.

        Format: ``{"error": {"code": "LCP-XXXX", "message": "..."}}``.

        The client-facing message never includes sensitive provider internals:
        exception ``client_message`` overrides win, provider-key patterns are
        redacted, and long messages are truncated. Full details are logged
        server-side by the caller.
        """
        if isinstance(exc, LCPError):
            payload = exc.to_dict()
            if message is not None:
                payload["message"] = _sanitize_message(message)
            else:
                payload["message"] = _sanitize_message(payload["message"])
            status = status if status is not None else exc.status_code
        else:
            payload = {"code": "LCP-5001", "message": "internal error"}
            status = status if status is not None else 500
        self._send_json({"error": payload}, status)

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
        elif self.path == "/models":
            self._serve_models_page()
        elif self.path == "/setup":
            self._serve_setup_page()
        elif self.path == "/api/settings" or self.path.startswith("/api/settings?"):
            self._serve_settings_api()
        elif self.path == "/api/routing/status" or self.path.startswith("/api/routing/status?"):
            self._serve_routing_status_api()
        elif self.path == "/api/setup" or self.path.startswith("/api/setup?"):
            self._serve_setup_api()
        elif self.path == "/api/setup/progress" or self.path.startswith("/api/setup/progress?"):
            self._serve_setup_progress_api()
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
        elif self.path == "/api/providers/health" or self.path.startswith("/api/providers/health?"):
            self._serve_providers_health_api()
        elif self.path == "/api/providers/failovers" or self.path.startswith("/api/providers/failovers?"):
            self._serve_providers_failovers_api()
        elif self.path.split("?")[0].startswith("/api/providers/") and self.path.split("?")[0].endswith("/failures"):
            # GET /api/providers/{name}/failures?window=...
            provider_name = self.path.split("?")[0].split("/")[3]
            self._serve_provider_failures_api(provider_name)
        elif self.path == "/api/profiles":
            self._serve_profiles_list()
        elif self.path.startswith("/api/profiles/") and self.path.endswith("/budget"):
            # GET /api/profiles/{name}/budget
            parts = self.path.split("/")
            self._serve_profile_budget(parts[3])
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
        elif self.path == "/api/budgets":
            self._serve_budgets_list()
        elif self.path == "/api/budgets/status":
            self._serve_budgets_status()
        elif self.path == "/api/cost-plugins/usage":
            self._serve_plugin_usage()
        elif self.path == "/api/cost-plugins/balances":
            self._serve_plugin_balances()
        elif self.path == "/api/cost-plugins/summary":
            self._serve_plugin_summary()
        elif self.path == "/api/cost-plugins/subscriptions":
            self._serve_plugin_subscriptions()
        elif self.path.startswith("/api/cost-plugins/cookie/") and len(self.path.split("/")) == 5:
            # GET /api/cost-plugins/cookie/{provider}
            provider = self.path.split("/")[4]
            self._serve_plugin_cookie_get(provider)
        elif self.path.startswith("/api/cost-plugins/workspace-id/") and len(self.path.split("/")) == 5:
            # GET /api/cost-plugins/workspace-id/{provider}
            provider = self.path.split("/")[4]
            self._serve_plugin_workspace_id_get(provider)
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
        elif self.path == "/alerts":
            self._serve_alerts_page()
        elif self.path == "/api/models/capability" or self.path.startswith("/api/models/capability?"):
            self._serve_capability_api()
        elif self.path == "/api/models/registry" or self.path.startswith("/api/models/registry?"):
            self._serve_registry_api()
        elif self.path == "/api/models/benchmark" or self.path.startswith("/api/models/benchmark?"):
            self._serve_benchmark_list_api()
        elif self.path == "/api/models/benchmark/status":
            self._serve_benchmark_status_api()
        elif self.path.startswith("/api/models/benchmark/") and self.path.endswith("/log"):
            # GET /api/models/benchmark/{id}/log
            parts = self.path.split("/")
            if len(parts) == 6:
                self._serve_benchmark_log_api(parts[4])
            else:
                self._send_json({"error": "not found"}, 404)
        elif self.path.startswith("/api/models/benchmark/") and len(self.path.split("/")) == 5:
            run_id = self.path.split("/")[4]
            self._serve_benchmark_detail_api(run_id)
        else:
            # GET /{profile}/memory/count
            profile = self._memory_profile()
            if profile:
                parts = self.path.rstrip("/").split("/")
                if len(parts) == 4 and parts[3] == "count":
                    self._serve_memory_api(profile, "count")
                    return
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
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
        elif self.path == "/api/budgets":
            self._serve_budget_create()
            return
        elif self.path == "/api/setup/skip":
            self._serve_setup_skip_api()
            return
        elif self.path == "/api/settings":
            self._serve_settings_update_api()
            return
        elif self.path == "/api/settings/cache/refresh":
            self._serve_settings_refresh_api()
            return
        elif self.path == "/api/settings/cache/clear":
            self._serve_settings_cache_clear()
            return
        elif self.path == "/api/routing/policy":
            self._serve_routing_policy_api()
            return
        elif self.path == "/api/routing/rules":
            self._serve_routing_rules_api()
            return
        elif self.path.startswith("/api/setup/install/") and len(self.path.split("/")) == 6:
            # POST /api/setup/install/{kind}/{name}
            kind = self.path.split("/")[4]
            name = self.path.split("/")[5]
            self._serve_setup_install_api(kind, name)
            return
        elif self.path.startswith("/api/circuit-breaker/reset"):
            self._serve_circuit_breaker_reset()
            return
        elif self.path == "/api/models/registry":
            self._serve_registry_upsert_api()
            return
        elif self.path == "/api/models/capability/manual":
            self._serve_capability_manual_api()
            return
        elif self.path == "/api/models/capability/import":
            self._serve_capability_import_api()
            return
        elif self.path == "/api/models/benchmark":
            self._serve_benchmark_create_api()
            return
        elif self.path.startswith("/api/providers/") and self.path.endswith("/toggle"):
            # POST /api/providers/{name}/toggle
            provider_name = self.path.split("/")[3]
            self._serve_provider_toggle(provider_name)
            return
        elif self.path.startswith("/api/cost-plugins/cookie/") and len(self.path.split("/")) == 5:
            # POST /api/cost-plugins/cookie/{provider}
            provider = self.path.split("/")[4]
            self._serve_plugin_cookie_set(provider)
            return
        elif self.path.startswith("/api/cost-plugins/workspace-id/") and len(self.path.split("/")) == 5:
            # POST /api/cost-plugins/workspace-id/{provider}
            provider = self.path.split("/")[4]
            self._serve_plugin_workspace_id_set(provider)
            return

        # POST /{profile}/memory/{retain|recall|forget}
        profile = self._memory_profile()
        if profile:
            parts = self.path.rstrip("/").split("/")
            if len(parts) == 4 and parts[3] in ("retain", "recall", "forget"):
                self._serve_memory_api(profile, parts[3])
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
        try:
            if profile_cfg.get("auth_required", True):
                auth_header = self.headers.get("Authorization", "")
                if not auth_header.startswith("Bearer "):
                    logger.warning("auth_failed", reason="missing_bearer_token",
                                   profile=profile, client_ip=self.client_address[0])
                    raise AuthError("API key required for this profile. Use Authorization: Bearer <key>")
                raw_key = auth_header[7:]
                km = resolve_service("key_manager", fallback=get_key_manager)
                if km:
                    key_info = km.validate_key(raw_key)
                    if key_info is None:
                        logger.warning("auth_failed", reason="invalid_or_revoked_key",
                                       profile=profile, client_ip=self.client_address[0])
                        raise AuthError("invalid or revoked API key")
                    # Check profile access
                    allowed = key_info.get("allowed_profiles")
                    if allowed:
                        allowed_list = [p.strip() for p in allowed.split(",") if p.strip()]
                        if profile not in allowed_list:
                            logger.warning("auth_failed", reason="profile_access_denied",
                                           profile=profile, client_ip=self.client_address[0],
                                           key_id=key_info.get("id"))
                            raise ForbiddenError(f"key does not have access to profile '{profile}'")
                    # Check budget-enforced spend limit (budgets are single source of truth)
                    if key_info.get("id"):
                        blocked = self._check_budget_block(profile, key_info.get("id"))
                        if blocked:
                            logger.warning("auth_failed", reason="budget_exceeded",
                                           profile=profile, client_ip=self.client_address[0],
                                           key_id=key_info.get("id"), budget=blocked)
                            raise CreditExhaustedError(
                                f"Budget '{blocked}' has been exceeded for this key/profile"
                            )
                    self._current_key_id = key_info.get("id")
        except (AuthError, CreditExhaustedError, ForbiddenError) as e:
            self._send_error(e)
            return

        # ── Budget enforcement (before LLM call) ──
        blocked_budget = self._check_budget_block(
            profile, getattr(self, '_current_key_id', None)
        )
        if blocked_budget:
            self._send_json({
                "error": {
                    "code": "LCP-4290",
                    "message": f"Budget '{blocked_budget}' has been exceeded. Further requests are blocked.",
                }
            }, 429)
            return

        try:
            # Validate body has messages (after auth so we return 401 first if unauthenticated)
            if not isinstance(body.get("messages"), list) or len(body.get("messages", [])) == 0:
                self._send_json({"error": "missing required field: messages"}, 400)
                return

            # Tool stripping
            body, blocked_tools = strip_forbidden_tools(body, profile_cfg.get("forbidden_tools"))

            # Pre-request cost estimation (pricing falls back to the plugin
            # registry or default rates instead of hard-failing → 500)
            primary_step = profile_cfg["chain"][0]
            pricing = _resolve_pricing(
                self.config, primary_step["provider"], primary_step["model"]
            )
            try:
                estimation = estimate_from_request(
                    primary_step["model"],
                    body.get("messages", []),
                    body.get("tools"),
                    body.get("max_tokens", 1024),
                    pricing,
                )
            except Exception:  # noqa: BLE001 — estimation must never break a request
                estimation = {
                    "input_tokens": 0,
                    "estimated_output_tokens": 0,
                    "estimated_input_cost": 0.0,
                    "estimated_output_cost": 0.0,
                    "estimated_total_cost": 0.0,
                    "currency": "USD",
                }

            # Prompt cache check (skip cache for streaming requests - cached JSON cannot satisfy SSE)
            cache = resolve_service("prompt_cache", fallback=get_prompt_cache)
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
                # Capture reasoning_content from the stream so it can be
                # re-attached to later requests (DeepSeek thinking mode).
                try:
                    capture_reasoning_from_sse(full_sse)
                except Exception:
                    pass
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

                # Increment budget spend and fire alerts
                try:
                    self._track_budget_spend(
                        profile, cost_info["cost"],
                        getattr(self, '_current_key_id', None)
                    )
                except Exception:
                    pass

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
            # Capture reasoning_content so it can be re-attached to later
            # requests (DeepSeek thinking mode requires it on tool-call turns).
            try:
                capture_reasoning_from_response(response_body)
            except Exception:
                pass
            cache.set(profile, model, body, response_body)

            # Token verification
            verifier = resolve_service("token_verifier", fallback=get_token_verifier)
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

            # Budget spend tracking — unified: increments profile + key budgets
            # and syncs ApiKey.total_spend with key-scoped budgets.
            try:
                key_id = getattr(self, '_current_key_id', None)
                self._track_budget_spend(profile, cost_info["cost"], key_id)
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
            self._send_error(e)
        except ProviderBadRequestError as e:
            # The provider rejected the request body as invalid (HTTP 400).
            # Report the exact error to the client so they can fix their request.
            logger.error("provider_bad_request", profile=profile, error=str(e))
            cost_info = {"prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0,
                         "cache_miss_tokens": 0, "cost": 0, "latency_ms": 0}
            record_cost(self.engine, profile, "unknown", "unknown", cost_info, False,
                       "provider_bad_request", [], error_detail=str(e))
            self._send_error(e)
        except AllProvidersFailedError as e:
            logger.error("all_providers_failed", profile=profile, error=str(e))
            cost_info = {"prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0,
                         "cache_miss_tokens": 0, "cost": 0, "latency_ms": 0}
            record_cost(self.engine, profile, "unknown", "unknown", cost_info, False,
                       "all_providers_failed", [], error_detail=str(e))
            self._send_error(e)
        except ProviderError as e:
            # Any other typed provider error (timeout, rate limit, upstream
            # auth, 5xx) that escaped the chain fallback — surface it cleanly.
            logger.error("provider_error", profile=profile, error=str(e))
            cost_info = {"prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0,
                         "cache_miss_tokens": 0, "cost": 0, "latency_ms": 0}
            record_cost(self.engine, profile, "unknown", "unknown", cost_info, False,
                       e.code, [], error_detail=str(e))
            self._send_error(e)
        except LCPError as e:
            logger.error("lcp_error", profile=profile, error=str(e))
            self._send_error(e)
        except Exception as e:
            logger.error("unhandled_error", error=str(e), traceback=traceback.format_exc()[-500:])
            self._send_error(e)

    def do_PUT(self):
        logger.debug("request_start", method="PUT", path=self.path,
                     client_ip=self.client_address[0])
        if self.path.startswith("/api/providers/") and len(self.path.split("/")) == 4:
            provider_name = self.path.split("/")[3]
            self._serve_provider_update(provider_name)
        elif self.path.startswith("/api/chains/") and len(self.path.split("/")) == 4:
            profile = self.path.split("/")[3]
            self._serve_chain_reorder(profile)
        elif self.path.startswith("/api/profiles/") and self.path.endswith("/budget"):
            # PUT /api/profiles/{name}/budget
            parts = self.path.split("/")
            self._serve_profile_budget_update(parts[3])
        elif self.path.startswith("/api/profiles/") and len(self.path.split("/")) == 4:
            profile = self.path.split("/")[3]
            self._serve_profile_update(profile)
        elif self.path == "/api/alerts/config":
            self._serve_alerts_config_update()
        elif self.path.startswith("/api/budgets/") and len(self.path.split("/")) == 4:
            budget_id = self.path.split("/")[3]
            self._serve_budget_update(budget_id)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_DELETE(self):
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
        elif self.path.startswith("/api/budgets/") and len(self.path.split("/")) == 4:
            budget_id = self.path.split("/")[3]
            self._serve_budget_delete(budget_id)
        elif self.path.startswith("/api/models/registry/") and len(self.path.split("/")) == 5:
            logical = self.path.split("/")[4]
            self._serve_registry_delete_api(logical)
        elif self.path.startswith("/api/setup/") and len(self.path.split("/")) == 5:
            # DELETE /api/setup/{kind}/{name}
            kind = self.path.split("/")[3]
            name = self.path.split("/")[4]
            self._serve_setup_remove_api(kind, name)
        else:
            self._send_json({"error": "not found"}, 404)

    # ── Budget Enforcement ────────────────────────────────────────────────

    def _check_budget_block(self, profile: str, key_id: int | None = None) -> str | None:
        """Check if any active budget with action=block is exceeded.

        Returns the budget name if blocked, or None if allowed.
        """
        try:
            with get_session(self.engine) as session:
                query = session.query(Budget).filter(
                    Budget.action == "block",
                    Budget.status.in_(["active", "exceeded"]),
                )
                # Match budgets for this profile, or global (null profile)
                query = query.filter(
                    or_(Budget.profile == profile, Budget.profile.is_(None))
                )
                if key_id is not None:
                    query = query.filter(
                        or_(Budget.key_id == key_id, Budget.key_id.is_(None))
                    )
                else:
                    query = query.filter(Budget.key_id.is_(None))

                for budget in query.all():
                    if budget.amount > 0 and budget.current_spend >= budget.amount:
                        logger.warning(
                            "budget_blocked",
                            budget_name=budget.name,
                            profile=profile,
                            spend=budget.current_spend,
                            limit=budget.amount,
                        )
                        return budget.name
        except Exception as e:
            logger.error("budget_check_failed", error=str(e))
        return None

    def _increment_budget_spend(self, profile: str, cost: float, key_id: int | None = None) -> list[dict]:
        """Increment spend on matching budgets and return any threshold breaches."""
        breaches = []
        try:
            with get_session(self.engine) as session:
                query = session.query(Budget).filter(
                    Budget.status == "active",
                )
                query = query.filter(
                    or_(Budget.profile == profile, Budget.profile.is_(None))
                )
                if key_id is not None:
                    query = query.filter(
                        or_(Budget.key_id == key_id, Budget.key_id.is_(None))
                    )
                else:
                    query = query.filter(Budget.key_id.is_(None))

                for budget in query.all():
                    old_spend = budget.current_spend
                    budget.current_spend = round(old_spend + cost, 6)

                    # Check thresholds
                    if budget.amount > 0:
                        old_pct = (old_spend / budget.amount) * 100
                        new_pct = (budget.current_spend / budget.amount) * 100
                        for t in [int(t) for t in budget.threshold_pct.split(",") if t]:
                            if old_pct < t <= new_pct:
                                breaches.append({
                                    "budget_id": budget.id,
                                    "budget_name": budget.name,
                                    "threshold": t,
                                    "spend_pct": round(new_pct, 1),
                                    "limit": budget.amount,
                                    "current_spend": budget.current_spend,
                                })
                    if budget.current_spend >= budget.amount and budget.amount > 0:
                        budget.status = "exceeded"
                        budget.last_alert_at = datetime.now(timezone.utc).isoformat()

                    # Sync ApiKey.total_spend with key-scoped budget
                    if budget.key_id is not None:
                        key = session.query(ApiKey).filter(ApiKey.id == budget.key_id).first()
                        if key:
                            key.total_spend = budget.current_spend

                session.commit()
                return breaches
        except Exception as e:
            logger.error("budget_increment_failed", error=str(e))
        return []

    def _track_budget_spend(self, profile: str, cost: float, key_id: int | None = None) -> None:
        """Increment budgets and fire alerts for any threshold breaches."""
        breaches = self._increment_budget_spend(profile, cost, key_id)
        am = resolve_service("alert_manager", fallback=get_alert_manager)
        for breach in breaches:
            severity = "critical" if breach["spend_pct"] >= 100 else "warning" if breach["spend_pct"] >= 80 else "info"
            am.fire(
                rule="budget_breach",
                severity=severity,
                title=f"Budget '{breach['budget_name']}' at {breach['spend_pct']}%",
                message=f"Budget '{breach['budget_name']}' has reached {breach['spend_pct']}% of its ${breach['limit']:.2f} limit (${breach['current_spend']:.4f} spent).",
                dedup_key=f"budget:{breach['budget_id']}:t{breach['threshold']}",
                metadata=breach,
            )
