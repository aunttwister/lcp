"""Standalone page-rendering functions (HTML templates).

Each function returns a complete HTML page as a string.
Called from endpoint mixins in src.server.endpoints.
"""


def render_providers_page(config, engine=None) -> str:
    """Render the Providers management page (Jinja2)."""
    from .render import render_page
    return render_page("pages/providers.html", config, engine, active_page="providers")


def render_profiles_page(config, engine=None) -> str:
    """Render the Profiles management page (Jinja2)."""
    from .render import render_page
    # Include profile budget data
    profile_budgets = {}
    if engine is not None:
        try:
            from ..api.models import Budget, get_session as _gs
            with _gs(engine) as s:
                for b in s.query(Budget).filter(Budget.key_id.is_(None), Budget.profile.isnot(None)).all():
                    profile_budgets[b.profile] = {
                        "id": b.id, "name": b.name, "amount": b.amount,
                        "current_spend": b.current_spend, "period": b.period,
                        "threshold_pct": b.threshold_pct, "action": b.action, "status": b.status,
                        "spend_pct": round((b.current_spend / b.amount * 100) if b.amount > 0 else 0, 1),
                    }
        except Exception:
            pass
    return render_page("pages/profiles.html", config, engine,
                       active_page="profiles", profile_budgets=profile_budgets)


def render_keys_page(config, engine) -> str:
    """Render the API Keys management page (Jinja2)."""
    from .render import render_page
    return render_page("pages/keys.html", config, engine, active_page="keys")


def render_usage_page(config, engine=None) -> str:
    """Render the Usage & Spending page (Jinja2)."""
    from .render import render_page
    return render_page("pages/usage.html", config, engine, active_page="usage")


def render_logs_page(config, engine=None) -> str:
    """Render the Logs page (Jinja2)."""
    from .render import render_page
    return render_page("pages/logs.html", config, engine, active_page="logs")


def render_alerts_page(config, engine=None) -> str:
    """Render the Alerts page (Jinja2)."""
    from .render import render_page
    return render_page("pages/alerts.html", config, engine, active_page="alerts")
