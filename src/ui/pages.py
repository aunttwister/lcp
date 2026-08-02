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
    return render_page("pages/profiles.html", config, engine, active_page="profiles")


def render_keys_page(config, engine) -> str:
    """Render the API Keys management page (Jinja2)."""
    from .render import render_page
    return render_page("pages/keys.html", config, engine, active_page="keys")


def render_usage_page(config, engine=None) -> str:
    """Render the Usage & Spending page (Jinja2)."""
    from .render import render_page
    return render_page("pages/usage.html", config, engine, active_page="usage")
