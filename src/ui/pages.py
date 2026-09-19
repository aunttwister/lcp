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


def render_models_page(config, engine=None) -> str:
    """Render the Models capability matrix page (Jinja2)."""
    from .render import render_page
    return render_page("pages/models.html", config, engine, active_page="models")


def render_setup_page(config, engine=None) -> str:
    """Render the first-run setup wizard page (Jinja2)."""
    from .render import render_page
    return render_page("pages/setup.html", config, engine, active_page="setup")


def render_work_decisions_page(config, engine=None) -> str:
    """Render the Work > Decisions page (Jinja2).

    The Work section is the work layer merged into LCP as a module: the moments
    the board recorded, attributed to the actor that decided each one.
    """
    from .render import render_page
    from ..api import work as work_api
    try:
        view = work_api.decisions_view()
    except Exception as e:  # never blank the page on a data error
        view = {
            "available": False,
            "empty": {"reason": "could not read the decisions ledger",
                      "hint": "%s: %s" % (type(e).__name__, e)},
            "funnel": None,
            "ledger": None,
        }
    return render_page("pages/work_decisions.html", config, engine,
                       active_page="work_decisions", view=view)


def render_work_tasks_page(config, engine=None) -> str:
    """Render the Work > Tasks page (Jinja2).

    A task's state is its directory, so this view reads the tree rather than a
    status field -- the state cannot drift from where the item actually lives.
    """
    from .render import render_page
    from ..api import work_tasks
    try:
        view = work_tasks.tasks_view()
    except Exception as e:  # never blank the page on a data error
        view = {
            "available": False,
            "empty": {"reason": "could not read the task tree",
                      "hint": "%s: %s" % (type(e).__name__, e)},
            "counts": {}, "total": 0, "tasks": [], "todos": None, "conflicts": [],
        }
    return render_page("pages/work_tasks.html", config, engine,
                       active_page="work_tasks", view=view)


def render_work_fleet_page(config, engine=None) -> str:
    """Render the Work > Fleet page (Jinja2).

    Reads LCP's own provider/profile APIs, so the chain shown is the chain LCP
    will actually use -- not a second copy that can drift from it.
    """
    from .render import render_page
    from ..api import work_fleet
    try:
        view = work_fleet.fleet_view()
    except Exception as e:  # never blank the page on a data error
        view = {
            "available": False,
            "empty": {"reason": "could not read the fleet",
                      "hint": "%s: %s" % (type(e).__name__, e)},
            "summary": None, "flaky": [], "profiles": [],
            "failover_moments": [], "failover_stats": None, "routing": None,
        }
    return render_page("pages/work_fleet.html", config, engine,
                       active_page="work_fleet", view=view)



