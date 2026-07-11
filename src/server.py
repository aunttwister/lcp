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
from .api.key_manager import get_key_manager
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
        keys = get_key_manager().list_keys()
        self._send_json({"keys": keys})

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
        result = get_key_manager().create_key(profile, label)
        self._send_json(result)

    def _serve_key_delete(self, key_id: str):
        ok = get_key_manager().revoke_key(key_id)
        if ok:
            self._send_json({"ok": True, "deleted": key_id})
        else:
            self._send_json({"error": "key not found"}, 404)


def create_server(config, engine, port=8734):
    """Create and configure the HTTP server."""
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    import os

    class ConfiguredHandler(LCPHandler):
        pass

    ConfiguredHandler.config = config
    ConfiguredHandler.engine = engine

    server = ThreadingHTTPServer(("0.0.0.0", port), ConfiguredHandler)
    logger.info("server_created", port=port, profiles=list(config.profiles.keys()))
    return server
