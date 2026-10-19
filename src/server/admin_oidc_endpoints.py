"""OIDC SSO endpoints: /admin/oidc/start and /admin/callback.

Both are public paths (under the /admin/ prefix that Phase 1 classifies as
public) — they ARE the login. Every other admin path still requires the
Phase 1 session cookie, which this module mints through the same
``admin_auth.issue_cookie`` on success, so the guard itself does not change.

Flow:
  GET /admin/oidc/start[?next=/x]  → set state cookie, 302 to the IdP
  GET /admin/callback?code&state   → verify state, exchange code, fetch
                                     userinfo, decide, 302 with session
                                     cookie (or a denial page)
"""

from __future__ import annotations

import html
import secrets
from urllib.parse import parse_qs, quote

import httpx

from ..api import admin_auth, admin_oidc
from ..api.logging_config import get_logger

logger = get_logger("lcp.admin_oidc_ep")

_DENIED_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>LCP — not authorized</title>
<style>body {{ font: 15px/1.5 system-ui, sans-serif; background: #0f1115;
color: #e7e9ee; display: grid; place-items: center; min-height: 100vh; }}
main {{ background: #171a21; border: 1px solid #5d2b33; border-radius: 10px;
padding: 1.75rem; width: min(92vw, 26rem); }}
a {{ color: #4c7dff; }}</style></head><body><main>
<h1>Not authorized</h1>
<p>You authenticated with Pocket ID as <strong>{who}</strong>, but that identity
is not on this deployment's admin allow-list.</p>
<p><a href="/admin/login">Back to token login</a></p>
</main></body></html>"""

_FAILED_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>LCP — sign-in failed</title>
<style>body {{ font: 15px/1.5 system-ui, sans-serif; background: #0f1115;
color: #e7e9ee; display: grid; place-items: center; min-height: 100vh; }}
main {{ background: #171a21; border: 1px solid #5d2b33; border-radius: 10px;
padding: 1.75rem; width: min(92vw, 26rem); }}
a {{ color: #4c7dff; }}</style></head><body><main>
<h1>Sign-in failed</h1><p>{reason}</p>
<p><a href="/admin/login">Back to token login</a></p>
</main></body></html>"""


def _safe_next(raw: str | None) -> str:
    from urllib.parse import unquote, urlparse
    if not raw:
        return "/"
    candidate = unquote(raw).strip()
    if not candidate.startswith("/") or candidate.startswith("//"):
        return "/"
    parsed = urlparse(candidate)
    if parsed.scheme or parsed.netloc:
        return "/"
    return candidate


class AdminOIDCEndpoints:
    """Mixin: SSO start + callback. Renderers reuse AdminLoginEndpoints' HTML."""

    # ── GET /admin/oidc/start ────────────────────────────────────────────────
    def _serve_admin_oidc_start(self) -> None:
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        nxt = _safe_next((parse_qs(query).get("next") or ["/"])[0])
        try:
            cfg = admin_oidc.config()
        except admin_oidc.OIDCConfigError as exc:
            logger.warning("oidc_start_refused", reason=str(exc))
            return self._admin_send_html(_FAILED_HTML.format(
                reason="SSO is misconfigured on this deployment.", ), 503)
        if cfg is None:
            return self._admin_send_html(_FAILED_HTML.format(
                reason="SSO is not configured on this deployment."), 503)

        state = secrets.token_urlsafe(24)
        verifier, challenge = admin_oidc.new_pkce()
        cookie = admin_oidc.issue_state_cookie(state, verifier, nxt,
                                               data_dir=self._oidc_data_dir())
        try:
            url = admin_oidc.authorization_url(cfg, state, challenge,
                                               client=self._oidc_http())
        except Exception:  # noqa: BLE001 — discovery failure = generic page
            logger.exception("oidc_discovery_failed", issuer=cfg["issuer"])
            return self._admin_send_html(_FAILED_HTML.format(
                reason="Could not reach the identity provider."), 503)
        self.send_response(302)
        self.send_header("Location", url)
        self._admin_send_oidc_cookie(cookie)
        self.send_header("Content-Length", "0")
        self.end_headers()

    # ── GET /admin/callback ──────────────────────────────────────────────────
    def _serve_admin_oidc_callback(self) -> None:
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        params = parse_qs(query)
        code = (params.get("code") or [""])[0]
        state = (params.get("state") or [""])[0]

        cookie_value = self._oidc_read_flow_cookie()
        flow = admin_oidc.verify_state_cookie(cookie_value,
                                              data_dir=self._oidc_data_dir())
        if flow is None or not code or not state or \
                not secrets.compare_digest(str(flow.get("state")), state):
            logger.warning("oidc_callback_state_rejected")
            return self._admin_send_html(_FAILED_HTML.format(
                reason="The sign-in flow expired or was tampered with. "
                       "Start again."), 400)
        nxt = _safe_next(str(flow.get("next") or "/"))

        cfg = admin_oidc.config() or {}
        try:
            token = admin_oidc.exchange_code(cfg, code,
                                             str(flow["verifier"]),
                                             client=self._oidc_http())
            claims = admin_oidc.fetch_userinfo(cfg, token["access_token"],
                                               client=self._oidc_http())
        except admin_oidc.OIDCLoginError as exc:
            logger.warning("oidc_login_failed", reason=str(exc))
            return self._admin_send_html(_FAILED_HTML.format(
                reason="The identity provider refused the sign-in."), 502)
        except Exception:  # noqa: BLE001
            logger.exception("oidc_unexpected_failure")
            return self._admin_send_html(_FAILED_HTML.format(
                reason="Sign-in failed unexpectedly."), 502)

        actor = admin_oidc.decide_session(claims)
        if actor is None:
            who = html.escape(str(claims.get("preferred_username") or "?"))
            return self._admin_send_html(_DENIED_HTML.format(who=who), 403)

        session_cookie = admin_auth.issue_cookie(actor=actor,
                                                 data_dir=self._oidc_data_dir())
        logger.info("admin_login_ok", actor=actor,
                    client_ip=self.client_address[0] if self.client_address else "-")
        # Expire the flow cookie; the session cookie takes over.
        self._admin_redirect_oidc(nxt, session_cookie)

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _oidc_data_dir() -> str:
        return "data"

    @staticmethod
    def _oidc_http() -> httpx.Client | None:
        # None → the module builds its own short-lived client. Overridden in
        # tests with a MockTransport.
        return None

    def _oidc_read_flow_cookie(self) -> str | None:
        header = self.headers.get("Cookie", "") or ""
        for part in header.split(";"):
            name, _, value = part.strip().partition("=")
            if name == admin_oidc.STATE_COOKIE and value:
                return value
        return None

    def _admin_send_oidc_cookie(self, value: str) -> None:
        attrs = [
            f"{admin_oidc.STATE_COOKIE}={value}",
            "Path=/admin",
            "HttpOnly",
            "SameSite=Lax",
            f"Max-Age={admin_oidc.STATE_TTL_SECONDS}",
        ]
        if self._is_tls(self):
            attrs.append("Secure")
        self.send_header("Set-Cookie", "; ".join(attrs))

    def _admin_redirect_oidc(self, location: str, session_cookie: str) -> None:
        """302 that sets the session and expires the flow cookie."""
        self.send_response(302)
        self.send_header("Location", location)
        self._admin_send_cookie(session_cookie)
        attrs = [
            f"{admin_oidc.STATE_COOKIE}=",
            "Path=/admin",
            "HttpOnly",
            "SameSite=Lax",
            "Max-Age=0",
            "Expires=Thu, 01 Jan 1970 00:00:00 GMT",
        ]
        if self._is_tls(self):
            attrs.append("Secure")
        self.send_header("Set-Cookie", "; ".join(attrs))
        self.send_header("Content-Length", "0")
        self.end_headers()
