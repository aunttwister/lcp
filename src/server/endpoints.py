"""Endpoint mixin classes for LCPHandler.

Each class groups related _serve_* methods by domain.
LCPHandler inherits from all of them via multiple inheritance.
"""

import json
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, parse_qs

from ..api.logging_config import get_logger
from ..api.circuit_breaker import get_circuit_breaker
from ..api.key_manager import get_key_manager
from ..api.alert_manager import get_alert_manager
from ..api.cost_plugins import get_registry
from ..api.models import get_session, Request as RequestModel
from ..api.prompt_cache import get_prompt_cache
from ..api.token_verifier import get_token_verifier
from ..ui.dashboard import render_dashboard

logger = get_logger("lcp.server")


# ── Health / Monitoring Endpoints ────────────────────────────────────────────

class HealthEndpoints:
    """Health, models, errors, cache stats, metrics, export."""

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

    def _serve_models(self, profile: str | None = None):
        # Collect all providers per model: {model_id: [provider_names]}
        model_providers = {}
        model_provider_seen = {}  # {model_id: set(provider_names)} for dedup
        model_owned_by = {}

        model_limits = self.config.model_limits

        profiles_iter = (
            [(profile, self.config.profiles[profile])]
            if profile
            else self.config.profiles.items()
        )

        for _prof_name, prof_cfg in profiles_iter:
            for step in prof_cfg["chain"]:
                mid = step["model"]
                provider = step["provider"]

                if mid not in model_provider_seen:
                    model_provider_seen[mid] = set()
                    model_providers[mid] = []

                if provider not in model_provider_seen[mid]:
                    model_provider_seen[mid].add(provider)
                    model_providers[mid].append(provider)

                # First-seen provider becomes owned_by
                if mid not in model_owned_by:
                    model_owned_by[mid] = provider

        models = []
        for mid, providers in model_providers.items():
            limits = model_limits.get(mid, {})
            context_len = limits.get("context_window")

            entry = {
                "id": mid,
                "object": "model",
                "owned_by": model_owned_by[mid],
                "kind": "model",
            }
            if context_len:
                entry["context_window"] = context_len
                entry["max_model_len"] = context_len
                entry["context_length"] = context_len
            if limits.get("max_output_tokens"):
                entry["max_output_tokens"] = limits["max_output_tokens"]
            if limits.get("description"):
                entry["description"] = limits["description"]

            entry["providers"] = [
                {
                    "provider": p,
                    "context_length": context_len or 128000,
                    "supports_tools": True,
                }
                for p in providers
            ]
            models.append(entry)

        # ── Emit profile entries as virtual models ──
        for prof_name, prof_cfg in profiles_iter:
            chain = prof_cfg.get("chain", [])
            if not chain:
                continue

            profile_providers = []
            max_context = 0
            profile_description = ""

            for step in chain:
                mid = step["model"]
                provider = step["provider"]
                limits = model_limits.get(mid, {})
                ctx = limits.get("context_window", 128000)
                profile_providers.append({
                    "provider": provider,
                    "model": mid,
                    "context_length": ctx,
                    "supports_tools": True,
                })
                if ctx > max_context:
                    max_context = ctx
                # Use first model's description as fallback
                if not profile_description and limits.get("description"):
                    profile_description = limits["description"]

            models.append({
                "id": prof_name,
                "object": "model",
                "owned_by": "lcp",
                "kind": "profile",
                "context_window": max_context,
                "max_model_len": max_context,
                "context_length": max_context,
                "description": profile_description or f"Profile with {len(chain)} provider(s) in chain",
                "providers": profile_providers,
            })

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
        cache = get_prompt_cache()  # local import in original; use top-level
        self._send_json(cache.stats)

    def _serve_metrics(self):
        """Prometheus-compatible metrics endpoint."""
        from sqlalchemy import func

        try:
            with get_session(self.engine) as session:
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
        try:
            with get_session(self.engine) as session:
                q = session.query(RequestModel).order_by(RequestModel.id.desc())
                rows = q.limit(min(1000, 10000)).all()

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


# ── Provider Configuration Endpoints ─────────────────────────────────────────

class ProviderEndpoints:
    """Provider CRUD, testing, chain reorder."""

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
        import urllib.request
        import urllib.error
        import ssl

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


# ── Profile Management Endpoints ─────────────────────────────────────────────

class ProfileEndpoints:
    """Profile CRUD."""

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


# ── API Key Management ───────────────────────────────────────────────────────

class KeyEndpoints:
    """API key CRUD."""

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


# ── Alert Management ─────────────────────────────────────────────────────────

class AlertEndpoints:
    """Alert listing, config, webhook testing."""

    def _serve_alerts_list(self):
        am = get_alert_manager()
        limit = 100
        status = None
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


# ── Cost Plugin API ──────────────────────────────────────────────────────────

class PluginEndpoints:
    """Cost tracking plugin endpoints."""

    def _serve_plugin_usage(self):
        """Return aggregated usage from all cost tracking plugins."""
        qs = parse_qs(urlparse(self.path).query)
        start = qs.get("start", [None])[0]
        end = qs.get("end", [None])[0]
        data = get_registry().fetch_all_usage(start_date=start, end_date=end)
        self._send_json({"plugin_usage": data})

    def _serve_plugin_balances(self):
        """Return account balances from all cost tracking plugins."""
        data = get_registry().fetch_all_balances()
        self._send_json({"plugin_balances": data})

    def _serve_plugin_summary(self):
        """Return rich provider summaries (usage limits, balance, etc.)."""
        data = get_registry().fetch_all_summaries()
        self._send_json({"plugin_summaries": data})

    def _serve_plugin_subscriptions(self):
        """Return subscription usage snapshots from all cost plugins."""
        data = get_registry().fetch_all_subscriptions()
        self._send_json({"plugin_subscriptions": data})


# ── Usage Stats API ──────────────────────────────────────────────────────────

class UsageEndpoints:
    """Usage statistics and usage page."""

    def _serve_usage_stats_api(self):
        """Return per-provider aggregates: daily spending, by-model, by-profile.

        Accepts ?provider=X&days=N or ?provider=X&start=YYYY-MM-DD&end=YYYY-MM-DD.
        """
        from sqlalchemy import func

        qs = parse_qs(urlparse(self.path).query)
        provider = qs.get("provider", [None])[0]
        days = int(qs.get("days", ["30"])[0])
        start_str = qs.get("start", [None])[0]
        end_str = qs.get("end", [None])[0]

        try:
            with get_session(self.engine) as session:
                base_filter = [RequestModel.success == 1]
                if provider:
                    base_filter.append(RequestModel.provider == provider)

                # Date range: use start/end if provided, else use days limit
                if start_str and end_str:
                    base_filter.append(RequestModel.timestamp >= start_str)
                    base_filter.append(RequestModel.timestamp <= end_str + "T23:59:59")
                    daily_date_filter = RequestModel.timestamp.between(start_str, end_str + "T23:59:59")
                else:
                    daily_date_filter = True

                # Daily spending trend
                daily_q = (
                    session.query(
                        func.substr(RequestModel.timestamp, 1, 10).label("date"),
                        func.coalesce(func.sum(RequestModel.cost), 0).label("cost"),
                        func.count(RequestModel.id).label("requests"),
                    )
                )
                for f in base_filter:
                    daily_q = daily_q.filter(f)
                if not (start_str and end_str):
                    daily_q = daily_q.filter(daily_date_filter)
                daily_rows = (
                    daily_q
                    .group_by(func.substr(RequestModel.timestamp, 1, 10))
                    .order_by(func.substr(RequestModel.timestamp, 1, 10).desc())
                    .limit(days)
                    .all()
                )

                # Build lookup of existing data keyed by date
                daily_map = {r.date: {"cost": float(r.cost), "requests": r.requests} for r in daily_rows}

                # Determine full date range to fill gaps with zero-usage days
                if start_str and end_str:
                    date_end = datetime.strptime(end_str, "%Y-%m-%d").date()
                    date_start = datetime.strptime(start_str, "%Y-%m-%d").date()
                else:
                    date_end = datetime.utcnow().date()
                    date_start = date_end - timedelta(days=days - 1)

                # Generate every date in the range, inserting zeros for missing days
                daily = []
                d = date_start
                while d <= date_end:
                    ds = d.strftime("%Y-%m-%d")
                    if ds in daily_map:
                        daily.append({"date": ds, "cost": daily_map[ds]["cost"], "requests": daily_map[ds]["requests"]})
                    else:
                        daily.append({"date": ds, "cost": 0, "requests": 0})
                    d += timedelta(days=1)

                # Per-model aggregates (with cache tokens)
                model_q = (
                    session.query(
                        RequestModel.model,
                        func.coalesce(func.sum(RequestModel.cost), 0).label("cost"),
                        func.count(RequestModel.id).label("requests"),
                        func.coalesce(func.sum(RequestModel.prompt_tokens), 0).label("prompt_tokens"),
                        func.coalesce(func.sum(RequestModel.completion_tokens), 0).label("completion_tokens"),
                        func.coalesce(func.sum(RequestModel.cache_hit_tokens), 0).label("cache_hit_tokens"),
                        func.coalesce(func.sum(RequestModel.cache_miss_tokens), 0).label("cache_miss_tokens"),
                    )
                )
                for f in base_filter:
                    model_q = model_q.filter(f)
                model_rows = model_q.group_by(RequestModel.model).order_by(func.sum(RequestModel.cost).desc()).all()

                # Compute cache savings from gateway.yaml pricing
                total_cache_savings = 0.0
                for r in model_rows:
                    try:
                        pricing = self.config.get_pricing(provider or "deepseek", r.model)
                        if pricing and r.cache_hit_tokens > 0:
                            savings = (r.cache_hit_tokens / 1_000_000) * (
                                pricing["cache_miss"] - pricing["cache_hit"]
                            )
                            total_cache_savings += savings
                    except Exception:
                        pass

                by_model = {
                    r.model: {
                        "cost": float(r.cost),
                        "requests": r.requests,
                        "prompt_tokens": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens,
                        "cache_hit_tokens": r.cache_hit_tokens,
                        "cache_miss_tokens": r.cache_miss_tokens,
                    }
                    for r in model_rows
                }

                # Per-profile aggregates
                profile_q = (
                    session.query(
                        RequestModel.profile,
                        func.coalesce(func.sum(RequestModel.cost), 0).label("cost"),
                        func.count(RequestModel.id).label("requests"),
                        func.coalesce(func.sum(RequestModel.prompt_tokens), 0).label("prompt_tokens"),
                        func.coalesce(func.sum(RequestModel.completion_tokens), 0).label("completion_tokens"),
                    )
                )
                for f in base_filter:
                    profile_q = profile_q.filter(f)
                profile_rows = profile_q.group_by(RequestModel.profile).order_by(func.sum(RequestModel.cost).desc()).all()
                by_profile = {
                    r.profile: {
                        "cost": float(r.cost),
                        "requests": r.requests,
                        "prompt_tokens": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens,
                    }
                    for r in profile_rows
                }

                # Totals (respect date range)
                total_q = (
                    session.query(
                        func.coalesce(func.sum(RequestModel.cost), 0).label("cost"),
                        func.count(RequestModel.id).label("requests"),
                    )
                )
                for f in base_filter:
                    total_q = total_q.filter(f)
                total_row = total_q.first()
                total_cost = float(total_row.cost) if total_row else 0
                total_requests = total_row.requests if total_row else 0

                # Aggregate cache stats
                cache_hit = sum(r.cache_hit_tokens for r in model_rows)
                cache_miss = sum(r.cache_miss_tokens for r in model_rows)

            self._send_json({
                "provider": provider,
                "daily": daily,
                "by_model": by_model,
                "by_profile": by_profile,
                "totals": {"cost": total_cost, "requests": total_requests},
                "cache": {
                    "hit_tokens": cache_hit,
                    "miss_tokens": cache_miss,
                    "savings": round(total_cache_savings, 6),
                },
            })
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _serve_usage_totals_api(self):
        """Return just the totals for a date range — lightweight endpoint for date-filter pills."""
        from sqlalchemy import func

        qs = parse_qs(urlparse(self.path).query)
        provider = qs.get("provider", [None])[0]
        start_str = qs.get("start", [None])[0]
        end_str = qs.get("end", [None])[0]

        try:
            with get_session(self.engine) as session:
                filters = [RequestModel.success == 1]
                if provider:
                    filters.append(RequestModel.provider == provider)
                if start_str and end_str:
                    filters.append(RequestModel.timestamp >= start_str)
                    filters.append(RequestModel.timestamp <= end_str + "T23:59:59")

                total_q = (
                    session.query(
                        func.coalesce(func.sum(RequestModel.cost), 0).label("cost"),
                        func.count(RequestModel.id).label("requests"),
                        func.coalesce(func.sum(RequestModel.prompt_tokens + RequestModel.completion_tokens), 0).label("tokens"),
                    )
                )
                for f in filters:
                    total_q = total_q.filter(f)
                row = total_q.first()
                result = {
                    "provider": provider,
                    "start": start_str,
                    "end": end_str,
                    "cost": float(row.cost) if row else 0,
                    "requests": row.requests if row else 0,
                    "tokens": int(row.tokens) if row and row.tokens else 0,
                }
            self._send_json(result)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _serve_usage_page(self):
        """Server-rendered Usage & Spending page."""
        from ..ui.pages import render_usage_page
        html = render_usage_page(self.config, self.engine)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))


# ── Dashboard / Page Endpoints ───────────────────────────────────────────────

class DashboardEndpoints:
    """Dashboard, daily costs, recent requests, and management page rendering."""

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

    def _serve_keys_dashboard(self):
        """Server-rendered API Keys management page."""
        from ..ui.pages import render_keys_page
        html = render_keys_page(self.config, self.engine)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _serve_providers_page(self):
        """Server-rendered Providers management page."""
        from ..ui.pages import render_providers_page
        html = render_providers_page(self.config, self.engine)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _serve_profiles_page(self):
        """Server-rendered Profiles management page."""
        from ..ui.pages import render_profiles_page
        html = render_profiles_page(self.config, self.engine)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))
