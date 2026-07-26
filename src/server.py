"""HTTP server and request handler for the LLM Control Plane.

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
from .api.budget_manager import get_budget_manager, init_budget_manager
from .api.alert_manager import get_alert_manager
from .api.models import get_session, Request as RequestModel
from .api.exceptions import ToolBlockedError, AllProvidersFailedError

logger = get_logger("lcp.server")

class LCPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for LLM Control Plane."""

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
        elif self.path == "/budgets" or self.path == "/budgets/dashboard":
            self._serve_budgets_dashboard()
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
        elif self.path == "/api/budgets":
            self._serve_budgets_list()
        elif self.path == "/api/budgets/status":
            self._serve_budgets_status()
        elif self.path == "/api/alerts":
            self._serve_alerts_list()
        elif self.path == "/api/alerts/config":
            self._serve_alerts_config()
        elif self.path == "/api/alerts/active":
            self._serve_alerts_active()
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
        elif self.path == "/api/budgets":
            self._serve_budget_create()
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
        elif self.path.startswith("/api/budgets/") and len(self.path.split("/")) == 4:
            budget_id = self.path.split("/")[3]
            self._serve_budget_update(budget_id)
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
        elif self.path.startswith("/api/budgets/") and len(self.path.split("/")) == 4:
            budget_id = self.path.split("/")[3]
            self._serve_budget_delete(budget_id)
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
        scheme = "https" if self.headers.get("X-Forwarded-Proto") == "https" else "http"
        host_url = f"{scheme}://{host}"
        html = render_dashboard(self.config, self.engine, {"Host": host}, profile_filter)
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

    # ── Budget Management ──────────────────────────────────────────────────

    def _serve_budgets_list(self):
        bm = get_budget_manager()
        budgets = bm.list_budgets() if bm else []
        self._send_json({"budgets": budgets})

    def _serve_budgets_status(self):
        bm = get_budget_manager()
        status = bm.get_budget_status() if bm else []
        self._send_json({"budgets": status})

    def _serve_budget_create(self):
        try:
            body = self._read_body()
        except Exception:
            self._send_json({"error": "invalid JSON body"}, 400)
            return
        bm = get_budget_manager()
        if not bm:
            self._send_json({"error": "budget manager not initialized"}, 500)
            return
        if not body.get("name") or not body.get("amount"):
            self._send_json({"error": "missing 'name' or 'amount'"}, 400)
            return
        result = bm.create_budget(
            name=body["name"],
            amount=float(body["amount"]),
            key_id=body.get("key_id"),
            profile=body.get("profile"),
            period=body.get("period", "monthly"),
            threshold_pct=body.get("threshold_pct", "50,80,90"),
            action=body.get("action", "log"),
        )
        self._send_json(result)

    def _serve_budget_update(self, budget_id: str):
        try:
            body = self._read_body()
        except Exception:
            self._send_json({"error": "invalid JSON body"}, 400)
            return
        bm = get_budget_manager()
        if not bm:
            self._send_json({"error": "budget manager not initialized"}, 500)
            return
        try:
            bid = int(budget_id)
        except ValueError:
            self._send_json({"error": "invalid budget id"}, 400)
            return
        result = bm.update_budget(bid, body)
        if result:
            self._send_json(result)
        else:
            self._send_json({"error": "budget not found"}, 404)

    def _serve_budget_delete(self, budget_id: str):
        bm = get_budget_manager()
        if not bm:
            self._send_json({"error": "budget manager not initialized"}, 500)
            return
        try:
            bid = int(budget_id)
        except ValueError:
            self._send_json({"error": "invalid budget id"}, 400)
            return
        ok = bm.delete_budget(bid)
        if ok:
            self._send_json({"ok": True, "deleted": bid})
        else:
            self._send_json({"error": "budget not found"}, 404)

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

    def _serve_budgets_dashboard(self):
        """Server-rendered Budgets management page."""
        html = _render_budgets_page(self.config, self.engine)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))


def _render_keys_page(config, engine) -> str:
    """Render the API Keys management page."""
    # Load CSS
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
<title>LCP — API Keys</title>
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
        <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Allowed Profiles (comma-separated, blank=all)</label>
        <input id="newKeyProfiles" placeholder="l2,l1" style="width:100%;padding:0.375rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem">
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

<!-- Show Key Modal (one-time view) -->
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

function showCreateKeyModal() {{ document.getElementById('createKeyModal').classList.add('open'); }}
function closeCreateKeyModal() {{ document.getElementById('createKeyModal').classList.remove('open'); }}
function closeShowKeyModal() {{ document.getElementById('showKeyModal').classList.remove('open'); }}

function createKey() {{
  var name = document.getElementById('newKeyName').value.trim();
  var profiles = document.getElementById('newKeyProfiles').value.trim();
  var limit = parseFloat(document.getElementById('newKeyLimit').value) || 0;
  var expires = document.getElementById('newKeyExpires').value;
  api('POST', '/api/keys', {{
    name: name || 'API Key',
    allowed_profiles: profiles,
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
  var el = document.getElementById('shownKey');
  navigator.clipboard.writeText(el.textContent).then(function() {{
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

// Sidebar toggle
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


def _render_budgets_page(config, engine) -> str:
    """Render the Budgets management page."""
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
<title>LCP — Budgets</title>
<style>{css}</style>
</head>
<body>
{_render_sidebar_html(config, "budgets")}
<div class="sidebar-overlay" id="sidebarOverlay" onclick="closeSidebar()"></div>
<button class="sidebar-toggle" id="sidebarToggle" onclick="toggleSidebar()">☰</button>
<div class="main-content">
<h1>Budgets</h1>
<p class="subtitle">Set spending limits and get alerts</p>

<div style="display:flex;gap:0.5rem;margin-bottom:1rem">
  <button class="btn-sm btn-primary" onclick="showCreateBudgetModal()">+ Create Budget</button>
</div>

<div class="table-wrap">
<table id="budgetsTable">
<thead><tr>
  <th>Name</th><th>Key/Profile</th><th>Spend</th><th>Limit</th><th>%</th><th>Period</th><th>Action</th><th>Status</th><th>Actions</th>
</tr></thead>
<tbody id="budgetsBody"><tr><td colspan="9" class="empty">Loading...</td></tr></tbody>
</table>
</div>
</div>

<!-- Create Budget Modal -->
<div class="modal-overlay" id="createBudgetModal">
<div class="modal" style="width:min(500px,95vw)">
  <div class="modal-header">
    <h2>Create Budget</h2>
    <button class="modal-close" onclick="closeCreateBudgetModal()">✕</button>
  </div>
  <div class="modal-body">
    <div style="display:grid;gap:0.75rem">
      <div>
        <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Budget Name</label>
        <input id="newBudgetName" placeholder="e.g. Monthly L2 Budget" style="width:100%;padding:0.375rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem">
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem">
        <div>
          <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Amount ($)</label>
          <input id="newBudgetAmount" type="number" step="0.01" min="0.01" value="100" style="width:100%;padding:0.375rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem">
        </div>
        <div>
          <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Period</label>
          <select id="newBudgetPeriod" style="width:100%;padding:0.375rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem">
            <option value="monthly">Monthly</option>
            <option value="total">Total (lifetime)</option>
          </select>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem">
        <div>
          <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">API Key (blank=global)</label>
          <select id="newBudgetKey" style="width:100%;padding:0.375rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem">
            <option value="">-- All Keys (Global) --</option>
          </select>
        </div>
        <div>
          <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Profile (blank=all)</label>
          <input id="newBudgetProfile" placeholder="l2" style="width:100%;padding:0.375rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem">
        </div>
        <div>
          <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">On Exceed</label>
          <select id="newBudgetAction" style="width:100%;padding:0.375rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem">
            <option value="log">Log only (warn)</option>
            <option value="block">Block requests</option>
          </select>
        </div>
      </div>
      <div>
        <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Alert Thresholds (%)</label>
        <input id="newBudgetThresholds" value="50,80,90" placeholder="50,80,90" style="width:100%;padding:0.375rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem">
      </div>
    </div>
  </div>
  <div class="modal-footer">
    <button class="btn-sm btn-primary" onclick="createBudget()">Create</button>
    <button class="btn-sm" onclick="closeCreateBudgetModal()">Cancel</button>
  </div>
</div>
</div>

<script>
function api(method, url, body) {{
  var opts = {{method:method, headers:{{'Content-Type':'application/json'}}}};
  if (body) opts.body = JSON.stringify(body);
  return fetch(url, opts).then(function(r) {{ return r.json(); }});
}}

function loadBudgets() {{
  // Load keys first to build a name map
  api('GET', '/api/keys').then(function(kd) {{
    var keyMap = {{}};
    (kd.keys || []).forEach(function(k) {{ keyMap[k.id] = k.name; }});
    // Now load budgets
    api('GET', '/api/budgets').then(function(d) {{
      var budgets = d.budgets || [];
      var html = '';
      if (budgets.length === 0) {{
        html = '<tr><td colspan="9" class="empty">No budgets yet. Create one above.</td></tr>';
      }} else {{
        budgets.forEach(function(b) {{
          var pct = b.spend_pct || 0;
          var pctColor = pct >= 100 ? 'red' : pct >= 80 ? 'amber' : 'green';
          var statusBadge = b.status === 'exceeded'
            ? '<span class="badge badge-error">exceeded</span>'
            : b.status === 'active'
              ? '<span class="badge badge-success">active</span>'
              : '<span class="badge">' + b.status + '</span>';
          var scope;
          if (b.key_id) {{
            scope = '🔑 ' + (keyMap[b.key_id] || 'Key #' + b.key_id);
          }} else if (b.profile) {{
            scope = '📊 ' + b.profile;
          }} else {{
            scope = '🌐 Global';
          }}
          html += '<tr>' +
            '<td><b>' + (b.name || '—') + '</b></td>' +
            '<td>' + scope + '</td>' +
            '<td class="cost mono">$' + (b.current_spend || 0).toFixed(4) + '</td>' +
            '<td class="cost mono">$' + (b.amount || 0).toFixed(2) + '</td>' +
            '<td class="cost"><span class="' + pctColor + '">' + pct.toFixed(1) + '%</span></td>' +
            '<td>' + (b.period || '—') + '</td>' +
            '<td>' + (b.action === 'block' ? '🔒 Block' : '📋 Log') + '</td>' +
            '<td>' + statusBadge + '</td>' +
            '<td><button class="btn-sm btn-danger" onclick="deleteBudget(' + b.id + ')">Del</button></td>' +
          '</tr>';
        }});
      }}
      document.getElementById('budgetsBody').innerHTML = html;
    }});
  }});
}}

function showCreateBudgetModal() {{ 
  document.getElementById('createBudgetModal').classList.add('open');
  loadKeysForBudget();
}}
function closeCreateBudgetModal() {{ document.getElementById('createBudgetModal').classList.remove('open'); }}

function loadKeysForBudget() {{
  api('GET', '/api/keys').then(function(d) {{
    var keys = d.keys || [];
    var sel = document.getElementById('newBudgetKey');
    // Keep first "All Keys" option, clear rest
    sel.innerHTML = '<option value="">-- All Keys (Global) --</option>';
    keys.forEach(function(k) {{
      if (k.status === 'active') {{
        sel.innerHTML += '<option value="' + k.id + '">' + k.name + ' (' + k.key_prefix + ')</option>';
      }}
    }});
  }});
}}

function createBudget() {{
  var name = document.getElementById('newBudgetName').value.trim();
  var amount = parseFloat(document.getElementById('newBudgetAmount').value);
  var period = document.getElementById('newBudgetPeriod').value;
  var keyId = document.getElementById('newBudgetKey').value;
  var profile = document.getElementById('newBudgetProfile').value.trim();
  var action = document.getElementById('newBudgetAction').value;
  var thresholds = document.getElementById('newBudgetThresholds').value.trim();
  if (!name || !amount) {{ alert('Name and amount are required'); return; }}
  api('POST', '/api/budgets', {{
    name: name, amount: amount, period: period,
    key_id: keyId ? parseInt(keyId) : null,
    profile: profile || null, action: action, threshold_pct: thresholds
  }}).then(function(d) {{
    if (d.ok) {{ closeCreateBudgetModal(); loadBudgets(); }}
    else {{ alert('Error: ' + (d.error || 'unknown')); }}
  }});
}}

function deleteBudget(id) {{
  if (!confirm('Delete this budget?')) return;
  api('DELETE', '/api/budgets/' + id).then(function(d) {{
    if (d.ok) loadBudgets();
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

loadBudgets();
</script>
</body>
</html>"""


def _render_sidebar_html(config, active_page: str = "") -> str:
    """Render the sidebar navigation for standalone pages."""
    host_url = ""
    dash_active = ' class="active"' if active_page == "dashboard" else ""
    keys_active = ' class="active"' if active_page == "keys" else ""
    budgets_active = ' class="active"' if active_page == "budgets" else ""

    sidebar = (
        '<aside class="sidebar" id="sidebar">\n'
        '  <div class="sidebar-brand">⚡ LCP</div>\n'
        '  <nav class="sidebar-nav">\n'
        f'    <a href="/dashboard"{dash_active}>Dashboard</a>\n'
        f'    <a href="/keys"{keys_active}>API Keys</a>\n'
        f'    <a href="/budgets"{budgets_active}>Budgets</a>\n'
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

    # Initialize key manager and budget manager with engine
    init_key_manager(engine, "data")
    init_budget_manager(engine)

    server = ThreadingHTTPServer(("0.0.0.0", port), ConfiguredHandler)
    logger.info("server_created", port=port, profiles=list(config.profiles.keys()))
    return server
