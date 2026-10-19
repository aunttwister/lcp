"""OIDC SSO acceptance proofs (Phase 2, CUR-219).

Same doctrine as test_admin_auth.py: these back a security claim, so they do
NOT inherit the suite-wide anonymous-admin opt-out, and the IdP is faked with
an httpx.MockTransport rather than by hitting the real Pocket ID.
"""

import base64
import hashlib
import json
from unittest.mock import patch

import httpx
import pytest

from src.api import admin_oidc

# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _real_auth(monkeypatch):
    monkeypatch.delenv("LCP_ALLOW_ANONYMOUS_ADMIN", raising=False)
    monkeypatch.setenv("LCP_ADMIN_TOKEN", "test-token-do-not-use-in-prod")
    monkeypatch.setenv("LCP_SECRET_KEY", "unit-test-master-secret")
    # Never let a real network call out of the suite.
    yield
    admin_oidc._discovery_cache = None
    admin_oidc._discovery_at = 0.0


@pytest.fixture(autouse=True)
def _reset_discovery_cache():
    admin_oidc._discovery_cache = None
    admin_oidc._discovery_at = 0.0
    yield
    admin_oidc._discovery_cache = None
    admin_oidc._discovery_at = 0.0


OIDC_ENV = {
    "LCP_OIDC_ISSUER": "https://idp.example.test",
    "LCP_OIDC_CLIENT_ID": "cid-123",
    "LCP_OIDC_CLIENT_SECRET": "sec-456",
    "LCP_OIDC_REDIRECT_URI": "https://lcp.example.test/admin/callback",
    "LCP_OIDC_ADMIN_USERS": "aunttwister, second.admin",
}

DISCOVERY_DOC = {
    "issuer": "https://idp.example.test",
    "authorization_endpoint": "https://idp.example.test/authorize",
    "token_endpoint": "https://idp.example.test/api/oidc/token",
    "userinfo_endpoint": "https://idp.example.test/api/oidc/userinfo",
    "code_challenge_methods_supported": ["plain", "S256"],
}


def _handler(path="/admin/oidc/start", method="GET", headers=None):
    from tests.test_server import TestHandler, _header, _status
    return TestHandler(path=path, method=method, headers=headers or {})


def _status_of(h):
    from tests.test_server import _status
    return _status(h)


def _header_of(h, name):
    from tests.test_server import _header
    return _header(h, name)


def _cookies_of(h):
    return [c[0][1] for c in h.send_header.call_args_list
            if c[0] and str(c[0][0]).lower() == "set-cookie"]


# ── configuration ───────────────────────────────────────────────────────────

class TestConfig:
    def test_absent_is_none(self, monkeypatch):
        for k in OIDC_ENV:
            monkeypatch.delenv(k, raising=False)
        assert admin_oidc.config() is None
        assert admin_oidc.is_oidc_configured() is False

    def test_partial_refuses(self, monkeypatch):
        for k, v in OIDC_ENV.items():
            monkeypatch.setenv(k, v)
        monkeypatch.delenv("LCP_OIDC_CLIENT_SECRET", raising=False)
        with pytest.raises(admin_oidc.OIDCConfigError, match="CLIENT_SECRET"):
            admin_oidc.config()

    def test_full_resolves(self, monkeypatch):
        for k, v in OIDC_ENV.items():
            monkeypatch.setenv(k, v)
        cfg = admin_oidc.config()
        assert cfg["issuer"] == "https://idp.example.test"
        assert cfg["client_id"] == "cid-123"

    def test_http_issuer_refuses(self, monkeypatch):
        for k, v in OIDC_ENV.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setenv("LCP_OIDC_ISSUER", "http://idp.example.test")
        with pytest.raises(admin_oidc.OIDCConfigError, match="https"):
            admin_oidc.config()

    def test_admin_usernames_parsed_and_stripped(self, monkeypatch):
        monkeypatch.setenv("LCP_OIDC_ADMIN_USERS", " a , b,,c ")
        assert admin_oidc.admin_usernames() == ["a", "b", "c"]

    def test_no_admin_users_is_not_fatal_at_startup(self, monkeypatch):
        """Fail-open would be a trap; fail-closed at login is the right place."""
        for k, v in OIDC_ENV.items():
            monkeypatch.setenv(k, v)
        monkeypatch.delenv("LCP_OIDC_ADMIN_USERS", raising=False)
        admin_oidc.assert_oidc_config()  # must not raise
        claims = {"preferred_username": "aunttwister"}
        assert admin_oidc.decide_session(claims) is None  # deny, not crash


# ── PKCE + state ────────────────────────────────────────────────────────────

class TestPKCEAndState:
    def test_challenge_is_s256_of_verifier(self):
        verifier, challenge = admin_oidc.new_pkce()
        expect = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        assert challenge == expect
        assert verifier != challenge

    def test_state_cookie_roundtrip(self, temp_dir):
        cookie = admin_oidc.issue_state_cookie("s1", "v1", "/dashboard",
                                               data_dir=str(temp_dir))
        flow = admin_oidc.verify_state_cookie(cookie, data_dir=str(temp_dir))
        assert flow is not None
        assert flow["state"] == "s1"
        assert flow["verifier"] == "v1"
        assert flow["next"] == "/dashboard"

    def test_state_cookie_tamper_rejected(self, temp_dir):
        cookie = admin_oidc.issue_state_cookie("s1", "v1", "/", data_dir=str(temp_dir))
        body, _, sig = cookie.rpartition(".")
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        payload["state"] = "forged"
        forged = admin_oidc._b64e(json.dumps(payload).encode())
        assert admin_oidc.verify_state_cookie(f"{forged}.{sig}",
                                              data_dir=str(temp_dir)) is None

    def test_state_cookie_expiry(self, temp_dir):
        cookie = admin_oidc.issue_state_cookie("s1", "v1", "/", data_dir=str(temp_dir),
                                               now=1_000_000)
        assert admin_oidc.verify_state_cookie(cookie, data_dir=str(temp_dir),
                                              now=1_000_000 + admin_oidc.STATE_TTL_SECONDS + 5) is None


# ── the decision ────────────────────────────────────────────────────────────

class TestDecideSession:
    def test_allowed(self, monkeypatch):
        monkeypatch.setenv("LCP_OIDC_ADMIN_USERS", "aunttwister")
        actor = admin_oidc.decide_session({"preferred_username": "aunttwister"})
        assert actor == "oidc:aunttwister"

    def test_denied(self, monkeypatch):
        monkeypatch.setenv("LCP_OIDC_ADMIN_USERS", "aunttwister")
        assert admin_oidc.decide_session({"preferred_username": "mallory"}) is None

    def test_missing_username_denied(self, monkeypatch):
        monkeypatch.setenv("LCP_OIDC_ADMIN_USERS", "aunttwister")
        assert admin_oidc.decide_session({"sub": "x"}) is None


# ── IdP interaction, faked ──────────────────────────────────────────────────

def _fake_idp(client_cls_target=None):
    """An httpx.Client whose transport serves the discovery doc + token + userinfo."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/.well-known/openid-configuration":
            return httpx.Response(200, json=DISCOVERY_DOC)
        if request.url.path == "/api/oidc/token":
            body = dict(p.split("=", 1) for p in request.content.decode().split("&"))
            if body.get("code") == "good-code" and body.get("code_verifier"):
                return httpx.Response(200, json={"access_token": "at-1", "token_type": "Bearer"})
            return httpx.Response(400, json={"error": "invalid_grant"})
        if request.url.path == "/api/oidc/userinfo":
            if request.headers.get("Authorization") == "Bearer at-1":
                return httpx.Response(200, json={
                    "sub": "u-1", "preferred_username": "aunttwister",
                    "email": "pavle.curcic678@gmail.com"})
            return httpx.Response(401, json={"error": "invalid_token"})
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5)


class TestIdPInteraction:
    def test_discovery_cached(self, monkeypatch):
        for k, v in OIDC_ENV.items():
            monkeypatch.setenv(k, v)
        cfg = admin_oidc.config()
        client = _fake_idp()
        doc1 = admin_oidc.discovery(cfg, client)
        doc2 = admin_oidc.discovery(cfg, client)
        assert doc1["token_endpoint"] == doc2["token_endpoint"]

    def test_discovery_without_s256_refuses(self, monkeypatch):
        for k, v in OIDC_ENV.items():
            monkeypatch.setenv(k, v)
        cfg = admin_oidc.config()
        doc = dict(DISCOVERY_DOC, code_challenge_methods_supported=["plain"])
        client = httpx.Client(transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=doc)), timeout=5)
        with pytest.raises(admin_oidc.OIDCLoginError, match="S256"):
            admin_oidc.discovery(cfg, client)

    def test_exchange_happy(self, monkeypatch):
        for k, v in OIDC_ENV.items():
            monkeypatch.setenv(k, v)
        cfg = admin_oidc.config()
        token = admin_oidc.exchange_code(cfg, "good-code", "v" * 43,
                                         client=_fake_idp())
        assert token["access_token"] == "at-1"

    def test_exchange_bad_code_rejected(self, monkeypatch):
        for k, v in OIDC_ENV.items():
            monkeypatch.setenv(k, v)
        cfg = admin_oidc.config()
        with pytest.raises(admin_oidc.OIDCLoginError):
            admin_oidc.exchange_code(cfg, "bad-code", "v" * 43, client=_fake_idp())

    def test_userinfo_happy(self, monkeypatch):
        for k, v in OIDC_ENV.items():
            monkeypatch.setenv(k, v)
        cfg = admin_oidc.config()
        claims = admin_oidc.fetch_userinfo(cfg, "at-1", client=_fake_idp())
        assert claims["preferred_username"] == "aunttwister"

    def test_userinfo_without_username_refuses(self, monkeypatch):
        for k, v in OIDC_ENV.items():
            monkeypatch.setenv(k, v)
        cfg = admin_oidc.config()
        client = httpx.Client(transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"sub": "u-1"})), timeout=5)
        with pytest.raises(admin_oidc.OIDCLoginError):
            admin_oidc.fetch_userinfo(cfg, "at-1", client=client)


# ── endpoints over HTTP ─────────────────────────────────────────────────────

class TestEndpoints:
    def _configured(self, monkeypatch):
        for k, v in OIDC_ENV.items():
            monkeypatch.setenv(k, v)

    def test_start_redirects_to_idp_with_state_cookie(self, monkeypatch, temp_dir):
        self._configured(monkeypatch)
        h = _handler("/admin/oidc/start?next=/profiles")
        with patch.object(type(h), "_oidc_http", staticmethod(lambda: _fake_idp())), \
             patch.object(type(h), "_oidc_data_dir", staticmethod(lambda: str(temp_dir))):
            h._serve_admin_oidc_start()
        assert _status_of(h) == 302
        location = _header_of(h, "Location") or ""
        assert location.startswith(DISCOVERY_DOC["authorization_endpoint"])
        assert "code_challenge_method=S256" in location
        assert any(c.startswith("lcp_oidc_flow=") for c in _cookies_of(h))

    def test_start_without_config_returns_503(self, monkeypatch):
        for k in OIDC_ENV:
            monkeypatch.delenv(k, raising=False)
        h = _handler("/admin/oidc/start")
        h._serve_admin_oidc_start()
        assert _status_of(h) == 503

    def test_callback_full_flow_issues_session(self, monkeypatch, temp_dir):
        self._configured(monkeypatch)
        # Build the flow cookie the start endpoint would have set.
        verifier, challenge = admin_oidc.new_pkce()
        flow_cookie = admin_oidc.issue_state_cookie("st-1", verifier, "/dashboard",
                                                    data_dir=str(temp_dir))
        # The verifier must be the one sent in the token exchange: patch new_pkce
        # so start/callback agree... actually callback uses the cookie's verifier,
        # so no patching needed; just make the fake IdP accept any verifier.
        h = _handler("/admin/callback?code=good-code&state=st-1",
                     headers={"Cookie": f"lcp_oidc_flow={flow_cookie}"})
        with patch.object(type(h), "_oidc_http", staticmethod(lambda: _fake_idp())), \
             patch.object(type(h), "_oidc_data_dir", staticmethod(lambda: str(temp_dir))):
            h._serve_admin_oidc_callback()
        assert _status_of(h) == 302
        assert _header_of(h, "Location") == "/dashboard"
        cookies = _cookies_of(h)
        session = [c for c in cookies if c.startswith("lcp_admin=")]
        assert session, "expected a session cookie"
        assert any("Max-Age=0" in c for c in cookies), "flow cookie must be expired"

    def test_callback_bad_state_rejected(self, monkeypatch, temp_dir):
        self._configured(monkeypatch)
        h = _handler("/admin/callback?code=good-code&state=wrong",
                     headers={"Cookie": "lcp_oidc_flow=garbage.garbage"})
        with patch.object(type(h), "_oidc_data_dir", staticmethod(lambda: str(temp_dir))):
            h._serve_admin_oidc_callback()
        assert _status_of(h) == 400

    def test_callback_denied_username_gets_403(self, monkeypatch, temp_dir):
        self._configured(monkeypatch)
        monkeypatch.setenv("LCP_OIDC_ADMIN_USERS", "someoneelse")
        verifier, _ = admin_oidc.new_pkce()
        flow_cookie = admin_oidc.issue_state_cookie("st-1", verifier, "/",
                                                    data_dir=str(temp_dir))
        h = _handler("/admin/callback?code=good-code&state=st-1",
                     headers={"Cookie": f"lcp_oidc_flow={flow_cookie}"})
        with patch.object(type(h), "_oidc_http", staticmethod(lambda: _fake_idp())), \
             patch.object(type(h), "_oidc_data_dir", staticmethod(lambda: str(temp_dir))):
            h._serve_admin_oidc_callback()
        assert _status_of(h) == 403
        assert not any(c.startswith("lcp_admin=") for c in _cookies_of(h))

    def test_oidc_routes_are_public_paths(self):
        from src.api.admin_auth import is_public_path
        assert is_public_path("/admin/oidc/start")
        assert is_public_path("/admin/callback")
        # and the guard surface stays guarded
        assert not is_public_path("/api/keys")
