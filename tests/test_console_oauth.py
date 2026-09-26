"""Console OAuth (PKCE) flow + credits mapping.

The credits route (``/api/billing/account``) only accepts a console session, and
their OAuth server supports exactly ``authorization_code`` + ``refresh_token``
(no device grant).  These tests pin the flow mechanics, the storage/refresh
behaviour, and the mapping of the console's micro-cent money format.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time

import pytest
from urllib.error import HTTPError

from api.cost_plugins import console_oauth as co
from api.cost_plugins.opencode import OpenCodeCostPlugin, _map_account_credits


def _expected_challenge(verifier: str) -> str:
    return base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()


# ── PKCE ───────────────────────────────────────────────────────────────────

def test_new_pkce_challenge_is_s256_of_verifier():
    verifier, challenge = co.new_pkce()
    assert challenge == _expected_challenge(verifier)
    assert 43 <= len(verifier) <= 128
    assert "=" not in verifier and "=" not in challenge


def test_new_pkce_is_random_per_call():
    assert co.new_pkce()[0] != co.new_pkce()[0]


def test_authorize_url_carries_every_required_parameter():
    url, state = co.build_authorize_url("oac_test", "verifier-abc")
    assert url.startswith(co.AUTHORIZE_URL + "?")
    for fragment in (
        "client_id=oac_test",
        "response_type=code",
        "code_challenge_method=S256",
        f"code_challenge={_expected_challenge('verifier-abc')}",
        "redirect_uri=http%3A%2F%2F127.0.0.1%3A8791%2Fcallback",
    ):
        assert fragment in url
    assert f"state={state}" in url
    assert state


# ── callback parsing ───────────────────────────────────────────────────────

def test_parse_callback_accepts_a_full_redirect_url():
    assert co.parse_callback(
        "http://127.0.0.1:8791/callback?code=abc123&state=xyz"
    ) == "abc123"


def test_parse_callback_accepts_extra_params_and_quoting():
    assert co.parse_callback(
        '"http://127.0.0.1:8791/callback?state=xyz&code=abc123"'
    ) == "abc123"


def test_parse_callback_accepts_a_bare_code():
    assert co.parse_callback("  abc123  ") == "abc123"


@pytest.mark.parametrize("value", ["", "   ", "http://127.0.0.1:8791/callback",
                                   "http://127.0.0.1:8791/callback?state=xyz"])
def test_parse_callback_rejects_non_codes(value):
    with pytest.raises(ValueError):
        co.parse_callback(value)


# ── error classification ───────────────────────────────────────────────────

def test_token_error_is_classified_not_swallowed(monkeypatch):
    body = json.dumps({"_tag": "OAuthTokenError", "error": "invalid_grant",
                       "error_description": "Unknown or already used code"}).encode()

    def fake_urlopen(req, timeout=30):
        raise HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(body))

    monkeypatch.setattr(co, "urlopen", fake_urlopen)
    with pytest.raises(co.OAuthFlowError) as exc:
        co.exchange_code("code", "verifier", "client")
    assert exc.value.status == 400
    assert exc.value.error == "invalid_grant"
    assert "Unknown or already used" in exc.value.description


def test_registration_requires_a_client_id(monkeypatch):
    monkeypatch.setattr(co, "_post_json", lambda url, payload: {"error": "nope"})
    with pytest.raises(co.OAuthFlowError):
        co.register_client()


def test_register_client_uses_loopback_redirect_and_no_secret(monkeypatch):
    seen = {}

    def fake_post(url, payload):
        seen["url"] = url
        seen["payload"] = payload
        return {"client_id": "oac_x"}

    monkeypatch.setattr(co, "_post_json", fake_post)
    assert co.register_client() == "oac_x"
    assert seen["url"] == co.REGISTER_URL
    # http redirects are rejected off-loopback, and there is no device grant
    assert seen["payload"]["redirect_uris"] == ["http://127.0.0.1:8791/callback"]
    assert seen["payload"]["grant_types"] == ["authorization_code", "refresh_token"]
    assert seen["payload"]["token_endpoint_auth_method"] == "none"


# ── token storage / refresh ────────────────────────────────────────────────

def test_merge_keeps_the_existing_refresh_token_when_omitted():
    state = {"client_id": "c", "refresh_token": "r-old", "access_token": "a-old"}
    merged = co.merge_tokens(state, {"access_token": "a-new", "expires_in": 3600})
    assert merged["refresh_token"] == "r-old"
    assert merged["access_token"] == "a-new"
    assert merged["expires_at"] > time.time()


def test_merge_prefers_a_rotated_refresh_token():
    merged = co.merge_tokens({"refresh_token": "r-old"}, {"refresh_token": "r-new"})
    assert merged["refresh_token"] == "r-new"


def test_current_access_token_is_empty_when_nothing_is_stored(monkeypatch):
    monkeypatch.setattr(co, "load_tokens", lambda: None)
    assert co.current_access_token() == ""


def test_current_access_token_returns_a_fresh_token_without_refreshing(monkeypatch):
    monkeypatch.setattr(co, "load_tokens", lambda: {
        "client_id": "c", "access_token": "a", "refresh_token": "r",
        "expires_at": time.time() + 3600,
    })

    def boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("should not refresh a fresh token")

    monkeypatch.setattr(co, "refresh", boom)
    assert co.current_access_token() == "a"


def test_current_access_token_refreshes_a_near_expiry_token(monkeypatch):
    monkeypatch.setattr(co, "load_tokens", lambda: {
        "client_id": "c", "access_token": "a-old", "refresh_token": "r",
        "expires_at": time.time() + 5,  # inside the skew window
    })
    saved = {}
    monkeypatch.setattr(co, "save_tokens", lambda s: saved.update(s))
    monkeypatch.setattr(co, "refresh",
                        lambda cid, rt: {"access_token": "a-new", "expires_in": 3600})
    assert co.current_access_token() == "a-new"
    assert saved["access_token"] == "a-new"
    assert saved["refresh_token"] == "r"


def test_current_access_token_survives_a_failed_refresh(monkeypatch):
    monkeypatch.setattr(co, "load_tokens", lambda: {
        "client_id": "c", "access_token": "a-old", "refresh_token": "r",
        "expires_at": time.time() - 10,
    })

    def boom(cid, rt):
        raise co.OAuthFlowError(400, "invalid_grant", "The refresh token is invalid")

    monkeypatch.setattr(co, "refresh", boom)
    saved = []
    monkeypatch.setattr(co, "save_tokens", lambda s: saved.append(s))
    # the stale token is still returned: the caller's 403 handling is explicit
    assert co.current_access_token() == "a-old"
    assert saved == []


def test_current_access_token_without_refresh_token_returns_what_it_has(monkeypatch):
    monkeypatch.setattr(co, "load_tokens", lambda: {
        "access_token": "a", "expires_at": time.time() - 10,
    })
    assert co.current_access_token() == "a"


# ── credits mapping ────────────────────────────────────────────────────────

def test_map_account_credits_reads_micro_cents():
    mapped = _map_account_credits({
        "availableMicroCents": "1234000000",  # $12.34
        "currency": "USD", "plan": "go", "workspaceId": "wrk_1",
    })
    assert mapped["available_credits"] == 12.34
    assert mapped["balance"] == 12.34
    assert mapped["plan"] == "go"
    assert mapped["workspace_id"] == "wrk_1"


def test_map_account_credits_unwraps_a_nested_account():
    mapped = _map_account_credits({"account": {"availableMicroCents": 500000000}})
    assert mapped["available_credits"] == 5.0


def test_map_account_credits_falls_back_to_balance():
    mapped = _map_account_credits({"balanceMicroCents": 250000000})
    assert mapped["available_credits"] == 2.5


def test_map_account_credits_derives_plan_from_subscription():
    mapped = _map_account_credits({
        "availableMicroCents": 100000000, "subscription": {"plan": "pro"},
    })
    assert mapped["plan"] == "pro"


@pytest.mark.parametrize("payload", [{}, {"foo": "bar"}, None, [], {"account": {}}])
def test_map_account_credits_returns_none_when_unmappable(payload):
    """A shape change must report 'unrecognised', never a fabricated balance."""
    assert _map_account_credits(payload) is None


# ── plugin wiring ──────────────────────────────────────────────────────────

def test_fetch_balance_is_quiet_without_any_credential(monkeypatch):
    plugin = OpenCodeCostPlugin()
    monkeypatch.setattr(plugin, "_token", lambda **kw: "", raising=False)
    monkeypatch.setattr(co, "current_access_token", lambda force_refresh=False: "")
    assert plugin.fetch_balance() is None


def test_fetch_balance_states_that_a_session_is_needed(monkeypatch):
    plugin = OpenCodeCostPlugin()
    monkeypatch.setattr(plugin, "_token", lambda **kw: "oc_sk_all", raising=False)
    monkeypatch.setattr(co, "current_access_token", lambda force_refresh=False: "")
    result = plugin.fetch_balance()
    assert result["_error"] == "auth_failed"
    assert "console_oauth start" in result["detail"]


def test_fetch_balance_uses_the_console_session(monkeypatch):
    plugin = OpenCodeCostPlugin()
    monkeypatch.setattr(plugin, "_token", lambda **kw: "oc_sk_all", raising=False)
    monkeypatch.setattr(co, "current_access_token", lambda force_refresh=False: "sess")
    from api.cost_plugins import opencode_api as oa

    seen = {}

    def fake_credits(token):
        seen["token"] = token
        return {"availableMicroCents": 300000000}

    monkeypatch.setattr(oa, "fetch_account_credits", fake_credits)
    result = plugin.fetch_balance()
    assert seen["token"] == "sess"
    assert result["available_credits"] == 3.0


def test_fetch_balance_reports_a_403_distinctly(monkeypatch):
    plugin = OpenCodeCostPlugin()
    monkeypatch.setattr(plugin, "_token", lambda **kw: "oc_sk_all", raising=False)
    monkeypatch.setattr(co, "current_access_token", lambda force_refresh=False: "sess")
    from api.cost_plugins import opencode_api as oa

    def forbidden(token):
        raise oa.ConsoleApiError(403, "/api/billing/account", "Forbidden")

    monkeypatch.setattr(oa, "fetch_account_credits", forbidden)
    result = plugin.fetch_balance()
    assert result["_error"] == "auth_failed"
    assert "403" in result["detail"]


def test_fetch_balance_flags_an_unknown_payload_shape(monkeypatch):
    plugin = OpenCodeCostPlugin()
    monkeypatch.setattr(plugin, "_token", lambda **kw: "oc_sk_all", raising=False)
    monkeypatch.setattr(co, "current_access_token", lambda force_refresh=False: "sess")
    from api.cost_plugins import opencode_api as oa

    monkeypatch.setattr(oa, "fetch_account_credits", lambda token: {"surprise": 1})
    result = plugin.fetch_balance()
    assert result["_error"] == "api_error"
    assert "unrecognised" in result["detail"]
