"""Standalone page-rendering functions (HTML templates).

Each function returns a complete HTML page as a string.
Called from endpoint mixins in src.server.endpoints.
"""

import json
from datetime import date as _date
from pathlib import Path
from sqlalchemy import func


def render_providers_page(config, engine=None) -> str:
    """Render the Providers management page (Jinja2)."""
    from .render import render_page
    return render_page("pages/providers.html", config, engine, active_page="providers")


def render_profiles_page(config, engine=None) -> str:
    """Render the Profiles management page (Jinja2)."""
    from .render import render_page
    return render_page("pages/profiles.html", config, engine, active_page="profiles")


def render_keys_page(config, engine) -> str:
    """Render the API Keys management page (Jinja2)."""
    from .render import render_page
    return render_page("pages/keys.html", config, engine, active_page="keys")


def render_sidebar_html(config, active_page: str = "") -> str:
    """Render the sidebar navigation for standalone pages."""
    dash_active = ' class="active"' if active_page == "dashboard" else ""
    keys_active = ' class="active"' if active_page == "keys" else ""
    providers_active = ' class="active"' if active_page == "providers" else ""
    profiles_active = ' class="active"' if active_page == "profiles" else ""
    usage_active = ' class="active"' if active_page == "usage" else ""

    sidebar = (
        '<aside class="sidebar" id="sidebar">\n'
        '  <div class="sidebar-brand">LCP</div>\n'
        '  <nav class="sidebar-nav">\n'
        f'    <a href="/dashboard"{dash_active}>Dashboard</a>\n'
        f'    <a href="/keys"{keys_active}>API Keys</a>\n'
        f'    <a href="/providers"{providers_active}>Providers</a>\n'
        f'    <a href="/usage"{usage_active}>Usage</a>\n'
        '    <div class="sb-provider-rows" id="providerPluginRows"></div>\n'
        f'    <a href="/profiles"{profiles_active}>Profiles</a>\n'
        '    <div class="nav-label">Profiles</div>\n'
    )
    for p in config.profiles.keys():
        sidebar += f'    <a href="/{p}/dashboard">{p.upper()}</a>\n'
    sidebar += (
        '  </nav>\n</aside>'
    )
    return sidebar


def render_sidebar_plugin_js(config, engine=None) -> str:
    """Shared JS for sidebar plugin status rows (loadPluginStatus, toggle, etc.).

    Injects ``monthly`` gateway data and ``configuredProviders`` lists so
    that the sidebar plugin rows can fall back to the gateway DB when the
    provider plugin has no local data (e.g. OpenCode DB not on server).

    Requires ``engine`` (SQLAlchemy) to query the gateway DB for monthly
    totals.  When ``engine`` is None the monthly fallback is silently empty.
    """
    from datetime import date as _date
    from sqlalchemy import func

    # Monthly data from gateway DB
    _first_of_month = _date.today().replace(day=1).isoformat()
    monthly_data = {}
    if engine is not None:
        try:
            from ..api.models import get_session, Request as RequestModel
            with get_session(engine) as _s:
                _monthly_rows = _s.query(
                    RequestModel.provider,
                    func.count(RequestModel.id).label("m_reqs"),
                    func.coalesce(func.sum(RequestModel.completion_tokens + RequestModel.prompt_tokens), 0).label("m_tokens"),
                    func.coalesce(func.sum(RequestModel.cost), 0).label("m_cost"),
                ).filter(
                    RequestModel.timestamp >= _first_of_month,
                    RequestModel.success == 1,
                ).group_by(RequestModel.provider).all()
                for r in _monthly_rows:
                    monthly_data[r.provider] = {
                        "reqs": int(r.m_reqs),
                        "tokens": int(r.m_tokens),
                        "cost": float(r.m_cost),
                    }
        except Exception:
            pass

    monthly_json = json.dumps(monthly_data)
    configured_json = json.dumps(sorted(config.providers.keys()))

    return f"""<script>
function toggleSidebar() {{
  var sb = document.getElementById('sidebar');
  var ov = document.getElementById('sidebarOverlay');
  if (window.innerWidth <= 768) {{
    sb.classList.toggle('open');
    ov.classList.toggle('show');
  }} else {{
    sb.classList.toggle('collapsed');
    localStorage.setItem('lcp-sidebar', sb.classList.contains('collapsed') ? 'collapsed' : 'pinned');
  }}
}}
function closeSidebar() {{
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebarOverlay').classList.remove('show');
}}
(function() {{
  if (window.innerWidth <= 768) return;
  if (localStorage.getItem('lcp-sidebar') === 'collapsed') {{
    document.getElementById('sidebar').classList.add('collapsed');
  }}
}})();

function formatTokens(n) {{
  if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
  return String(n);
}}

var monthly = {monthly_json};
var configuredProviders = {configured_json};

function loadPluginStatus() {{
  var provRows = document.getElementById('providerPluginRows');
  if (!provRows) return;

  Promise.all([
    fetch('/api/cost-plugins/usage').then(function(r){{return r.json()}}).catch(function(){{return {{plugin_usage:{{}}}}}}),
    fetch('/api/cost-plugins/balances').then(function(r){{return r.json()}}).catch(function(){{return {{plugin_balances:{{}}}}}}),
    fetch('/api/cost-plugins/summary').then(function(r){{return r.json()}}).catch(function(){{return {{plugin_summaries:{{}}}}}}),
    fetch('/api/cost-plugins/subscriptions').then(function(r){{return r.json()}}).catch(function(){{return {{plugin_subscriptions:{{}}}}}})
  ]).then(function(results) {{
    var allUsage = results[0].plugin_usage || {{}};
    var balances = results[1].plugin_balances || {{}};
    var pluginSummaries = results[2].plugin_summaries || {{}};
    var subscriptions = results[3].plugin_subscriptions || {{}};

    var allProviders = Object.keys(allUsage).concat(Object.keys(balances));
    var uniqueProvs = allProviders.filter(function(v,i,a){{return a.indexOf(v)===i}}).filter(function(v){{return configuredProviders.indexOf(v) !== -1}});

    if (uniqueProvs.length === 0) {{
      provRows.innerHTML = '<div class="sb-provider-empty">No plugins active</div>';
    }} else {{
      var rows = '';
      uniqueProvs.forEach(function(prov) {{
        var bal = balances[prov];
        var usg = allUsage[prov] || [];
        var totalCost = usg.reduce(function(s,r){{return s + r.cost}}, 0);
        var totalTokens = usg.reduce(function(s,r){{return s + r.prompt_tokens + r.completion_tokens}}, 0);
        // Fallback to gateway DB when plugin has no usage data
        if (totalCost === 0 && monthly[prov]) {{
          totalCost = monthly[prov].cost || 0;
        }}
        if (totalTokens === 0 && monthly[prov]) {{
          totalTokens = monthly[prov].tokens || 0;
        }}
        var sum = pluginSummaries[prov];

        var detailLine = '';
        if (prov === 'opencode') {{
          var om = (sum && sum.monthly) ? sum.monthly : (monthly[prov] || {{}});
          detailLine = '<span class="sb-provider-detail">' +
            'month: ' + formatTokens(om.tokens || totalTokens) + ' tok';
          if (om.cost) detailLine += ' \\u00b7 $' + om.cost.toFixed(4);
          // Subscription data from OpenCode web API
          var sub = subscriptions[prov];
          if (sub) {{
            detailLine += '<br><span class="sb-sub-detail">';
            var parts = [];
            if (sub.rolling_pct != null) {{
              var resetMin = Math.floor(sub.rolling_reset_sec / 60);
              var r = '5h: ' + sub.rolling_pct.toFixed(0) + '% used';
              if (resetMin > 0) r += ' \\u00b7 resets in ' + resetMin + 'm';
              parts.push(r);
            }}
            if (sub.weekly_pct != null) {{
              var wkResetHr = Math.floor(sub.weekly_reset_sec / 3600);
              var w = 'week: ' + sub.weekly_pct.toFixed(0) + '% used';
              if (wkResetHr > 0) w += ' \\u00b7 resets in ' + wkResetHr + 'h';
              parts.push(w);
            }}
            if (sub.monthly_pct != null) {{
              var moResetDay = Math.floor(sub.monthly_reset_sec / 86400);
              var m = 'month: ' + sub.monthly_pct.toFixed(0) + '% used';
              if (moResetDay > 0) m += ' \\u00b7 resets in ' + moResetDay + 'd';
              parts.push(m);
            }}
            detailLine += parts.join('<br>');
            detailLine += '</span>';
          }}
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
  }});
}}

loadPluginStatus();
setInterval(loadPluginStatus, 60000);
</script>"""


def render_usage_page(config, engine=None) -> str:
    """Render the Usage & Spending page (Jinja2)."""
    from .render import render_page
    return render_page("pages/usage.html", config, engine, active_page="usage")
