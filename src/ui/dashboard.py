"""Dashboard renderer for the smallm gateway.

Generates the full server-rendered HTML dashboard.
"""
import json
from datetime import datetime, timezone
from typing import Any, cast
from sqlalchemy import func, case
from ..api.models import Request as RequestModel, get_session
from ..api.prompt_cache import get_prompt_cache
from ..api.token_verifier import get_token_verifier
from ..api.router import get_dynamic_router
from ..api.circuit_breaker import get_circuit_breaker

# ── Load dashboard CSS from template file ────────────────────────────────
from pathlib import Path
_templates_dir = Path(__file__).parent / "templates"
_DASHBOARD_CSS = (_templates_dir / "dashboard.css").read_text()


def render_dashboard(config, engine, headers, profile_filter=None):
    """Generate the full dashboard HTML page and return it as a string."""

    # Get circuit breaker for provider health dots
    cb = get_circuit_breaker()
    _get_health = cb.get_health

    # The original method body follows:
    """Server-rendered shadcn-themed dashboard with provider health, daily costs,
    recent requests/errors, Chart.js time-series, and Phase 5/6 metrics."""
    from sqlalchemy import func, case

    try:
        with get_session(engine) as session:
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

            # Unpack summary to plain Python types (avoids Column typing issues)
            _s = summary
            summary_total_cost: float = float(_s.total_cost) if _s else 0.0
            summary_total_requests: int = int(_s.total_requests) if _s else 0
            summary_prompt_tokens: int = int(_s.prompt_tokens) if _s else 0
            summary_output_tokens: int = int(_s.output_tokens) if _s else 0
            cache_hit_tokens: int = int(_s.cache_hits) if _s else 0
            cache_miss_tokens: int = int(_s.cache_misses) if _s else 0

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

            cache_total = cache_hit_tokens + cache_miss_tokens
            cache_hit_rate = (
                (cache_hit_tokens / cache_total * 100) if cache_total > 0 else 0
            )

            # ── Active days ──
            active_days_q = session.query(
                func.count(func.distinct(func.substr(RequestModel.timestamp, 1, 10)))
            )
            if profile_filter:
                active_days_q = active_days_q.filter(
                    RequestModel.profile == profile_filter
                )
            active_days = active_days_q.scalar() or 0

            # ── Per-profile cost ──
            profile_rows_q = session.query(
                RequestModel.profile,
                func.sum(RequestModel.cost).label("total_cost"),
                func.count(RequestModel.id).label("count"),
            ).filter(RequestModel.success == 1)
            if profile_filter:
                profile_rows_q = profile_rows_q.filter(
                    RequestModel.profile == profile_filter
                )
            profile_rows = (
                profile_rows_q.group_by(RequestModel.profile).all()
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
            pp_rows_q = (
                session.query(
                    func.substr(RequestModel.timestamp, 1, 10).label("date"),
                    RequestModel.profile,
                    func.coalesce(func.sum(RequestModel.cost), 0).label("cost"),
                    func.coalesce(func.avg(RequestModel.latency_ms), 0).label("avg_lat"),
                )
                .filter(RequestModel.success == 1)
            )
            if profile_filter:
                pp_rows_q = pp_rows_q.filter(
                    RequestModel.profile == profile_filter
                )
            pp_rows = (
                pp_rows_q.group_by(func.substr(RequestModel.timestamp, 1, 10), RequestModel.profile)
                .order_by(func.substr(RequestModel.timestamp, 1, 10).asc())
                .limit(14 * len(config.profiles))
                .all()
            )
            # Pivot into {dates: [...], profiles: {name: {costs:[], lats:[]}}}
            pp_data = {"dates": sorted(set(r.date for r in pp_rows)), "profiles": {}}
            for p in config.profiles.keys():
                pp_data["profiles"][p] = {"costs": [], "lats": []}
            date_to_idx = {d: i for i, d in enumerate(pp_data["dates"])}
            for p in config.profiles.keys():
                pp_data["profiles"][p]["costs"] = [0.0] * len(pp_data["dates"])
                pp_data["profiles"][p]["lats"] = [0.0] * len(pp_data["dates"])
            for r in pp_rows:
                idx = date_to_idx[r.date]
                pp_data["profiles"][r.profile]["costs"][idx] = float(r.cost)
                pp_data["profiles"][r.profile]["lats"][idx] = float(r.avg_lat)

            # Per-model time series (grouped by model)
            pm_rows_q = (
                session.query(
                    func.substr(RequestModel.timestamp, 1, 10).label("date"),
                    RequestModel.model,
                    func.coalesce(func.sum(RequestModel.cost), 0).label("cost"),
                    func.coalesce(func.avg(RequestModel.latency_ms), 0).label("avg_lat"),
                )
                .filter(RequestModel.success == 1)
            )
            if profile_filter:
                pm_rows_q = pm_rows_q.filter(
                    RequestModel.profile == profile_filter
                )
            pm_rows = (
                pm_rows_q.group_by(func.substr(RequestModel.timestamp, 1, 10), RequestModel.model)
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
        summary_total_cost = 0.0
        summary_total_requests = 0
        summary_output_tokens = 0
        summary_prompt_tokens = 0
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

    # ── Cache savings helper ──
    # Estimate dollars saved via provider prefix caching.
    # cache_hit tokens are billed at ~0.8% of miss rate.
    def _savings_for_model(model: str, hit_tokens: int) -> float:
        if hit_tokens <= 0:
            return 0.0
        for p_name, p_cfg in config.providers.items():
            if model in p_cfg.get("models", []):
                try:
                    pricing = config.get_pricing(p_name, model)
                    return (hit_tokens / 1_000_000) * (
                        pricing["cache_miss"] - pricing["cache_hit"]
                    )
                except Exception:
                    pass
        return 0.0

    total_cache_savings = sum(
        _savings_for_model(cast(Any, r).model, cast(Any, r).cache_hit)
        for r in daily_rows
    ) if daily_rows else 0.0

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
    for prof_name, prof_cfg in config.profiles.items():
        if profile_filter and prof_name != profile_filter:
            continue
        for step in prof_cfg["chain"]:
            pn = step["provider"]
            bu = step.get("base_url") or ""
            h = _get_health(pn, bu, prof_name)
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

    # Sidebar navigation — clean, minimal
    def _active(p): return " active" if profile_filter == p else ""
    dash_active = _active(None)
    host = headers.get("Host", "localhost:8735")
    scheme = "https" if (
        headers.get("X-Forwarded-Proto", "").split(",")[0].strip() == "https"
        or headers.get("X-Forwarded-Scheme") == "https"
    ) else "http"
    host_url = f"{scheme}://{host}"
    sidebar_nav = (
        '<aside class="sidebar" id="sidebar">\n'
        '  <div class="sidebar-brand">smallm</div>\n'
        '  <nav class="sidebar-nav">\n'
        f'    <a href="/dashboard" class="{dash_active}">Dashboard</a>\n'
        f'    <a href="/keys">API Keys</a>\n'
        f'    <a href="/providers">Providers</a>\n'
        f'    <a href="/usage">Usage</a>\n'
        '    <div class="sb-provider-rows" id="providerPluginRows"></div>\n'
        f'    <a href="/profiles">Profiles</a>\n'
        '    <div class="nav-label">Profiles</div>\n'
    )
    for p, pcfg in config.profiles.items():
        active = " active" if profile_filter == p else ""
        chain = pcfg.get("chain", [])
        providers_str = ", ".join(s["provider"] for s in chain[:2])
        if len(chain) > 2:
            providers_str += f" +{len(chain)-2}"
        sidebar_nav += (
            f'\n    <a href="/{p}/dashboard" class="{active} sb-profile-link" title="{providers_str}">'
            f'<span>{p.upper()}</span>'
            f'<button class="sb-copy-btn" onclick="event.preventDefault();event.stopPropagation();copyUrl(\u0027{host_url}/{p}/chat/completions\u0027)" title="Copy gateway URL">/{p}/chat/completions</button>'
            f'</a>'
        )
    sidebar_nav += (
        '\n  </nav>'
        '\n</aside>'
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
        r = cast(Any, r)
        daily_saved = _savings_for_model(r.model, r.cache_hit)
        daily_html += (
            f'<tr><td class="mono">{r.date}</td><td>{r.profile}</td>'
            f'<td class="mono">{r.model}</td><td>{r.provider}</td>'
            f'<td class="cost">{r.reqs}</td><td class="cost">{r.fb_count}</td>'
            f'<td class="cost">{_fmt_num(r.cache_hit)}</td>'
            f'<td class="cost">{_fmt_num(r.cache_miss)}</td>'
            f'<td class="cost">{_fmt_num(r.output)}</td>'
            f'<td class="cost mono">${float(r.cost):.6f}</td>'
            f'<td class="cost mono saved">${daily_saved:.6f}</td></tr>\n'
        )

    # Recent Requests table
    recent_html = ""
    for r in recent_rows:
        r = cast(Any, r)
        time_str = r.timestamp[11:19] if r.timestamp and "T" in r.timestamp else str(r.timestamp)[:19]
        status_badges = ""
        if r.success:
            status_badges += '<span class="badge badge-success">ok</span> '
        else:
            status_badges += f'<span class="badge badge-error">{r.error_type or "error"}</span> '
        if r.provider != "error" and r.success and r.provider:
            # Check if this was a fallback (first chain step != provider used)
            prof_cfg = config.profiles.get(r.profile, {})
            chain = prof_cfg.get("chain", [])
            if chain and chain[0]["provider"] != r.provider:
                status_badges += '<span class="badge badge-fallback">FB</span>'
        req_saved = _savings_for_model(r.model, r.cache_hit_tokens)
        recent_html += (
            f'<tr><td class="mono">{time_str}</td><td>{r.profile}</td>'
            f'<td class="mono">{r.model}</td><td>{r.provider}</td>'
            f'<td class="cost">{status_badges}</td>'
            f'<td class="cost">{r.latency_ms/1000:.1f}s</td>'
            f'<td class="cost mono">${r.cost:.6f}</td>'
            f'<td class="cost mono saved">${req_saved:.6f}</td></tr>\n'
        )

    # Recent Errors table
    error_html = ""
    for r in error_rows:
        r = cast(Any, r)
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

    # ── Latest responding (current) provider ──
    _latest_provider = None
    with get_session(engine) as _s:
        _latest_prov_row = _s.query(RequestModel.provider).filter(
            RequestModel.success == 1,
            RequestModel.provider != None,
            RequestModel.provider != 'error',
        )
        if profile_filter:
            _latest_prov_row = _latest_prov_row.filter(
                RequestModel.profile == profile_filter
            )
        _latest_prov_row = _latest_prov_row.order_by(
            RequestModel.timestamp.desc()
        ).first()
        if _latest_prov_row:
            _latest_provider = _latest_prov_row[0]

    # ── Monthly usage per provider (for OpenCode display) ──
    from datetime import date as _date
    _first_of_month = _date.today().replace(day=1).isoformat()
    with get_session(engine) as _s:
        _monthly_rows = _s.query(
            RequestModel.provider,
            func.count(RequestModel.id).label("m_reqs"),
            func.coalesce(func.sum(RequestModel.completion_tokens + RequestModel.prompt_tokens), 0).label("m_tokens"),
            func.coalesce(func.sum(RequestModel.cost), 0).label("m_cost"),
        ).filter(
            RequestModel.timestamp >= _first_of_month,
            RequestModel.success == 1,
        )
        if profile_filter:
            _monthly_rows = _monthly_rows.filter(
                RequestModel.profile == profile_filter
            )
        _monthly_rows = _monthly_rows.group_by(RequestModel.provider).all()
    _monthly_data = {}
    for r in _monthly_rows:
        _monthly_data[r.provider] = {
            "reqs": int(r.m_reqs),
            "tokens": int(r.m_tokens),
            "cost": float(r.m_cost),
        }

    # Serialize for JS
    _latest_provider_json = json.dumps(_latest_provider)
    _monthly_data_json = json.dumps(_monthly_data)
    _configured_providers_json = json.dumps(sorted(config.providers.keys()))

    # Template snippet for header badge — key stat per active provider
    plugin_header_info = (
        "var latestProvider = " + _latest_provider_json + ";\n"
        "var summaries = pluginSummaries || {};\n"
        "if (latestProvider && hdrText) {\n"
        "  if (hdrDot) hdrDot.className = 'header-plugin-dot on';\n"
        "  var sum = summaries[latestProvider];\n"
        "  var cur, hdrLabel = latestProvider;\n"
        "  if (latestProvider === 'deepseek' && sum && sum.balance) {\n"
        "    cur = sum.balance.currency || 'USD';\n"
        "    hdrLabel += ' \\u00b7 ' + cur + ' ' + sum.balance.available.toFixed(2) + ' available';\n"
        "  } else if (latestProvider === 'opencode') {\n"
        "    var om = (sum && sum.monthly) ? sum.monthly : (monthly[latestProvider] || {});\n"
        "    var now = new Date();\n"
        "    var start = new Date(now.getFullYear(), now.getMonth(), 1);\n"
        "    var end = new Date(now.getFullYear(), now.getMonth() + 1, 1);\n"
        "    var moPct = Math.round(((now - start) / (end - start)) * 100);\n"
        "    hdrLabel += ' \\u00b7 ' + moPct + '% of month';\n"
        "    if (om.tokens) hdrLabel += ' \\u00b7 ' + formatTokens(om.tokens) + ' tok';\n"
        "    if (om.cost) hdrLabel += ' \\u00b7 $' + om.cost.toFixed(4);\n"
        "  } else if (latestProvider === 'llamacpp') {\n"
        "    var mt = (monthly[latestProvider] || {}).tokens || 0;\n"
        "    hdrLabel += ' \\u00b7 ' + formatTokens(mt) + ' tokens';\n"
        "  } else {\n"
        "    var usg = allUsage[latestProvider] || [];\n"
        "    var totalCost = usg.reduce(function(s,r){return s + r.cost}, 0);\n"
        "    hdrLabel += ' \\u00b7 $' + totalCost.toFixed(4);\n"
        "  }\n"
        "  hdrText.textContent = hdrLabel;\n"
        "  if (hdrDot) hdrDot.title = latestProvider + ' \\u2014 latest successful provider';\n"
        "} else {\n"
        "  if (hdrText) hdrText.textContent = 'No requests yet';\n"
        "  if (hdrDot) hdrDot.className = 'header-plugin-dot off';\n"
        "}"
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
    <title>smallm Dashboard{filter_title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/sortablejs@1.15.0/Sortable.min.js"></script>
    <style>
    {_DASHBOARD_CSS}
    </style>
    </head>
    <body>
    {sidebar_nav}
    <div class="sidebar-overlay" id="sidebarOverlay" onclick="closeSidebar()"></div>
    <button class="sidebar-toggle" id="sidebarToggle" onclick="toggleSidebar()" title="Toggle sidebar">☰</button>
    <div class="main-content">
    <div class="header-row">
      <div>
        <h1>smallm gateway{filter_title}</h1>
        <p class="subtitle">Cost tracking · Request history · Phase 5/6 intelligence</p>
      </div>
      <div class="header-plugin-badge" id="pluginHeaderBadge">
        <span class="header-plugin-dot" id="pluginHeaderDot"></span>
        <span class="header-plugin-text" id="pluginHeaderText">Plugins…</span>
      </div>
    </div>

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
    </div>
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

    <details class="dashboard-section" open>
    <summary><span>Summary</span><span class="chevron">▾</span></summary>
    <div class="section-content cards">
    <div class="card">
      <div class="label">Total Cost</div>
      <div class="value">{_fmt_cost(summary_total_cost)}</div>
      <div class="sub">{active_days} active days</div>
    </div>
    <div class="card">
      <div class="label">Total Requests</div>
      <div class="value">{summary_total_requests:,}</div>
      <div class="sub">{fallback_count:,} fallbacks ({fb_pct:.1f}%)</div>
    </div>
    <div class="card">
      <div class="label">Cache Hit Ratio</div>
      <div class="value good">{cache_hit_rate:.1f}%</div>
      <div class="sub">{_fmt_num(cache_hit_tokens)} hit / {_fmt_num(cache_miss_tokens)} miss</div>
    </div>
    <div class="card">
      <div class="label">Cache Savings</div>
      <div class="value good">${total_cache_savings:.4f}</div>
      <div class="sub">prefix caching discount</div>
    </div>
    <div class="card">
      <div class="label">Output Tokens</div>
      <div class="value">{_fmt_num(summary_output_tokens)}</div>
      <div class="sub">prompt: {_fmt_num(summary_prompt_tokens)}</div>
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
    <thead><tr><th>Time</th><th>Profile</th><th>Model</th><th>Provider</th><th class="cost">Status</th><th class="cost">Dur</th><th class="cost">Cost</th><th class="cost">Saved</th></tr></thead>
    <tbody>
    {recent_html or '<tr><td colspan="8" class="empty">No requests yet</td></tr>'}
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
      dx = Math.max(dx, -44); // cap at action button width
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

    <script>
    // Monthly usage per provider (from DB — injected server-side)
    var monthly = {_monthly_data_json};
    var configuredProviders = {_configured_providers_json};

    function loadPluginStatus() {{
      var hdrDot = document.getElementById('pluginHeaderDot');
      var hdrText = document.getElementById('pluginHeaderText');

      Promise.all([
        fetch('/api/cost-plugins/usage').then(function(r){{return r.json()}}).catch(function(){{return {{plugin_usage:{{}}}}}}),
        fetch('/api/cost-plugins/balances').then(function(r){{return r.json()}}).catch(function(){{return {{plugin_balances:{{}}}}}}),
        fetch('/api/cost-plugins/summary').then(function(r){{return r.json()}}).catch(function(){{return {{plugin_summaries:{{}}}}}})
      ]).then(function(results) {{
        var allUsage = results[0].plugin_usage || {{}};
        var balances = results[1].plugin_balances || {{}};
        var pluginSummaries = results[2].plugin_summaries || {{}};

        var allProviders = Object.keys(allUsage).concat(Object.keys(balances));
        var uniqueProvs = allProviders.filter(function(v,i,a){{return a.indexOf(v)===i}}).filter(function(v){{return configuredProviders.indexOf(v) !== -1}});

        // ── Sidebar: plugin info under Providers nav link ──
        var provRows = document.getElementById('providerPluginRows');
        if (provRows) {{
          if (uniqueProvs.length === 0) {{
            provRows.innerHTML = '<div class="sb-provider-empty">No plugins active</div>';
          }} else {{
            var rows = '';
            uniqueProvs.forEach(function(prov) {{
              var bal = balances[prov];
              var usg = allUsage[prov] || [];
              var totalCost = usg.reduce(function(s,r){{return s + r.cost}}, 0);
              var totalTokens = usg.reduce(function(s,r){{return s + r.prompt_tokens + r.completion_tokens}}, 0);
              var sum = pluginSummaries[prov];

              var detailLine = '';
              if (prov === 'opencode') {{
                var om = (sum && sum.monthly) ? sum.monthly : (monthly[prov] || {});
                var now = new Date();
                var moStart = new Date(now.getFullYear(), now.getMonth(), 1);
                var moEnd = new Date(now.getFullYear(), now.getMonth() + 1, 1);
                var moPct = Math.round(((now - moStart) / (moEnd - moStart)) * 100);
                detailLine = '<span class="sb-provider-detail">' +
                  moPct + '% of month';
                if (om.tokens) detailLine += ' \\u00b7 ' + formatTokens(om.tokens) + ' tok';
                if (om.cost) detailLine += ' \\u00b7 $' + om.cost.toFixed(4);
                detailLine += '</span>';
              }} else if (prov === 'deepseek' && sum && sum.balance) {{
                var cur = sum.balance.currency || 'USD';
                detailLine = '<span class="sb-provider-detail">' + cur + ' ' + sum.balance.available.toFixed(2) + ' available';
                if (sum.balance.spent !== null) detailLine += ' \\u00b7 spent ' + cur + ' ' + sum.balance.spent.toFixed(2);
                detailLine += '</span>';
              }} else if (bal && bal.balance !== null && bal.balance !== undefined) {{
                var currency = bal.currency || 'USD';
                detailLine = '<span class="sb-provider-detail">' + currency + ' ' + bal.balance.toFixed(2) + ' balance</span>';
              }} else if (prov === 'llamacpp') {{
                detailLine = '<span class="sb-provider-detail">' + formatTokens(totalTokens) + ' tokens \\u00b7 local</span>';
              }} else {{
                var m = monthly[prov] || {{}};
                var mr = m.reqs || 0;
                var mt = m.tokens || 0;
                if (mr > 0) detailLine = '<span class="sb-provider-detail">' + mr + ' req \\u00b7 ' + formatTokens(mt) + ' tok this month</span>';
              }}

              rows += '<div class="sb-provider-row">' +
                '<div class="sb-provider-top">' +
                '<span class="sb-provider-name">' + prov + '</span>' +
                '<span class="sb-provider-cost">$' + totalCost.toFixed(4) + '</span>' +
                '</div>' +
                (detailLine ? detailLine : '') +
                '</div>';
            }});
            provRows.innerHTML = rows;
          }}
        }}

        // ── Header badge: active provider summary ──
        {plugin_header_info}
      }});
    }}

    function formatTokens(n) {{
      if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
      if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
      return String(n);
    }}

    // Load plugin status on page load and every 60 seconds
    loadPluginStatus();
    setInterval(loadPluginStatus, 60000);
    </script>

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



    return html
