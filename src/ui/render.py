"""Jinja2 template renderer for the LCP gateway UI.

Replaces the f-string-based HTML generation in pages.py and dashboard.py
with proper Jinja2 template files — giving syntax highlighting, auto-escaping,
and shared partials for sidebar/JS without any build step.
"""

import json
from datetime import date as _date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from sqlalchemy import func

_templates_dir = Path(__file__).parent / "templates" / "jinja"
_env = Environment(loader=FileSystemLoader(str(_templates_dir)), autoescape=True, extensions=["jinja2.ext.do"])


def _fmt_num(n) -> str:
    """Format a large number for display: 1.5M, 42.3K, 7."""
    if n is None:
        return "0"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(int(n))


def _fmt_cost(c) -> str:
    """Format a cost value as a dollar string."""
    if c is None:
        return "$0.000000"
    return f"${float(c):.6f}"


_env.filters["fmt_num"] = _fmt_num
_env.filters["fmt_cost"] = _fmt_cost


def _compute_monthly(engine) -> dict:
    """Query gateway DB for current-month per-provider totals.

    Returns a dict like ``{"deepseek": {"reqs": 42, "tokens": 12345, "cost": 1.23}}``.
    When ``engine`` is None (e.g. tests), returns an empty dict.
    """
    from ..api.models import get_session, Request as RequestModel

    _first_of_month = _date.today().replace(day=1).isoformat()
    monthly: dict = {}
    if engine is None:
        return monthly
    try:
        with get_session(engine) as s:
            rows = (
                s.query(
                    RequestModel.provider,
                    func.count(RequestModel.id).label("m_reqs"),
                    func.coalesce(
                        func.sum(
                            RequestModel.completion_tokens + RequestModel.prompt_tokens
                        ),
                        0,
                    ).label("m_tokens"),
                    func.coalesce(func.sum(RequestModel.cost), 0).label("m_cost"),
                )
                .filter(
                    RequestModel.timestamp >= _first_of_month,
                    RequestModel.success == 1,
                )
                .group_by(RequestModel.provider)
                .all()
            )
        for r in rows:
            monthly[r.provider] = {
                "reqs": int(r.m_reqs),
                "tokens": int(r.m_tokens),
                "cost": float(r.m_cost),
            }
    except Exception:
        pass
    return monthly


def render_page(template_name: str, config, engine=None, **kwargs) -> str:
    """Render a standalone page template with common context injected.

    All templates receive:
      * ``config`` — the gateway config object
      * ``monthly`` / ``monthly_json`` / ``configured_providers_json`` — sidebar data
      * ``profiles`` — list of profile names for sidebar nav

    Usage::

        html = render_page("pages/providers.html", config=self.config, engine=self.engine)
    """
    monthly = _compute_monthly(engine)
    provider_names = sorted(config.providers.keys()) if config is not None and hasattr(config, 'providers') else []
    profiles_list = list(config.profiles.keys()) if config is not None and hasattr(config, 'profiles') else []
    ctx = {
        "config": config,
        "monthly": monthly,
        "monthly_json": json.dumps(monthly),
        "configured_providers_json": json.dumps(provider_names),
        "providers": config.providers if config is not None and hasattr(config, 'providers') else {},
        "profiles": profiles_list,
        "profiles_dict": config.profiles if config is not None and hasattr(config, 'profiles') else {},
    }
    # Let caller-supplied kwargs override defaults (e.g. profile-filtered monthly)
    ctx.update(kwargs)
    return _env.get_template(template_name).render(**ctx)
