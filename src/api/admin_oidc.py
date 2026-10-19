"""Pocket ID OIDC SSO for the LCP admin surface (Phase 2, CUR-219).

Design, again measured rather than assumed:

1. **Authorization-code + PKCE(S256), confidential client.** The client secret
   never reaches the browser; the code exchange happens server-side. PKCE is
   enabled on the Pocket ID client and ``S256`` is what its discovery document
   advertises (verified 2026-10-19: ``code_challenge_methods_supported``).

2. **Identity comes from ``/userinfo``, not from decoding the ID token.** The
   tree has no jose/pyjwt dependency and adding one for one RS256 check is
   unnecessary attack surface; a server-to-server userinfo GET with the access
   token answers "who is this" with the same authority and zero crypto code.

3. **The first admin is a configured username, not a first-login race.**
   ``LCP_OIDC_ADMIN_USERS`` is a comma-separated allow-list of
   ``preferred_username`` values (A4 decision 2026-10-19: preferred_username,
   value ``aunttwister``). A successful OIDC login whose username is not on the
   list gets a clear denial page and NO session. Without a list, the flow
   denies everyone — fail closed, same as Phase 1.

4. **The state cookie is signed and bound to the flow.** state + PKCE verifier
   + next-path travel in a short-lived HMAC cookie under the same derived key
   family as the session cookie. A callback whose state does not verify is
   dropped before any network call is made.

5. **Fail closed on partial configuration.** If any OIDC variable is set but
   the set is incomplete, startup refuses (same doctrine as
   ``admin_auth.assert_configured``) — a half-wired SSO that silently falls
   back to token-only login is exactly the kind of quiet drift this surface
   must not have.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any
from urllib.parse import urlencode

import httpx

from .logging_config import get_logger

logger = get_logger("lcp.admin_oidc")

# ── env names ────────────────────────────────────────────────────────────────
ISSUER_ENV = "LCP_OIDC_ISSUER"
CLIENT_ID_ENV = "LCP_OIDC_CLIENT_ID"
CLIENT_SECRET_ENV = "LCP_OIDC_CLIENT_SECRET"
REDIRECT_URI_ENV = "LCP_OIDC_REDIRECT_URI"
ADMIN_USERS_ENV = "LCP_OIDC_ADMIN_USERS"
ADMIN_CLAIM_ENV = "LCP_OIDC_ADMIN_CLAIM"
DEFAULT_ADMIN_CLAIM = "preferred_username"

SCOPE = "openid profile email"
STATE_COOKIE = "lcp_oidc_flow"
STATE_TTL_SECONDS = 600  # 10 minutes to complete a login round-trip

_STATE_INFO_LABEL = b"lcp-oidc-flow-v1"


class OIDCConfigError(Exception):
    """OIDC env is partially set. Startup must refuse, per doctrine."""


class OIDCLoginError(Exception):
    """A login attempt failed. Rendered as a denial page, never a session."""


# ── configuration ────────────────────────────────────────────────────────────

def _env(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def config() -> dict[str, str] | None:
    """Return the resolved OIDC configuration, or None when fully unset."""
    issuer = _env(ISSUER_ENV)
    if issuer is None and _env(CLIENT_ID_ENV) is None and \
       _env(CLIENT_SECRET_ENV) is None and _env(REDIRECT_URI_ENV) is None:
        return None  # feature entirely absent — plain token login only
    # Some part is configured: the whole set is required.
    missing = [n for n, v in (
        (ISSUER_ENV, issuer),
        (CLIENT_ID_ENV, _env(CLIENT_ID_ENV)),
        (CLIENT_SECRET_ENV, _env(CLIENT_SECRET_ENV)),
        (REDIRECT_URI_ENV, _env(REDIRECT_URI_ENV)),
    ) if v is None]
    if missing:
        raise OIDCConfigError(
            "OIDC is partially configured; missing: " + ", ".join(missing)
        )
    if issuer is None:
        raise OIDCConfigError(f"{ISSUER_ENV} is missing while other OIDC vars are set")
    if not issuer.startswith("https://"):
        # The client secret would cross it. localhost http is the only
        # tolerable exception and this estate has no localhost IdP.
        raise OIDCConfigError(f"{ISSUER_ENV} must be https:// (got {issuer!r})")
    return {
        "issuer": issuer.rstrip("/"),
        "client_id": _env(CLIENT_ID_ENV) or "",
        "client_secret": _env(CLIENT_SECRET_ENV) or "",
        "redirect_uri": _env(REDIRECT_URI_ENV) or "",
    }


def admin_claim() -> str:
    """The claim that identifies a user for the allow-list.

    ``preferred_username`` (Pocket ID, and most IdPs) by default; override for
    e.g. Keycloak's ``preferred_username`` variants or an email-based scheme.
    """
    return _env(ADMIN_CLAIM_ENV) or DEFAULT_ADMIN_CLAIM


def admin_usernames() -> list[str]:
    raw = _env(ADMIN_USERS_ENV) or ""
    return [u.strip() for u in raw.split(",") if u.strip()]


def is_oidc_configured() -> bool:
    try:
        return config() is not None
    except OIDCConfigError:
        return False


def assert_oidc_config() -> None:
    """Called from main() next to admin_auth.assert_configured()."""
    cfg = config()
    if cfg is None:
        return
    if not admin_usernames():
        logger.warning(
            "oidc_no_admin_users",
            consequence="OIDC logins will authenticate but never be granted a session",
            fix=f"set {ADMIN_USERS_ENV}=preferred_username[,more]",
        )


# ── discovery (cached for the process lifetime) ─────────────────────────────

_discovery_cache: dict[str, Any] | None = None
_discovery_at: float = 0.0
_DISCOVERY_TTL = 3600


def discovery(cfg: dict[str, str], client: httpx.Client | None = None) -> dict[str, Any]:
    global _discovery_cache, _discovery_at
    now = time.time()
    if _discovery_cache is not None and now - _discovery_at < _DISCOVERY_TTL:
        return _discovery_cache
    url = cfg["issuer"] + "/.well-known/openid-configuration"
    own = client is None
    if own:
        client = httpx.Client(timeout=10, verify=True)
    try:
        resp = client.get(url)
        resp.raise_for_status()
        doc = resp.json()
    finally:
        if own:
            client.close()
    for key in ("authorization_endpoint", "token_endpoint", "userinfo_endpoint"):
        if not doc.get(key):
            raise OIDCLoginError(f"discovery document lacks {key}")
    if "S256" not in (doc.get("code_challenge_methods_supported") or []):
        raise OIDCLoginError("IdP does not advertise S256 PKCE")
    _discovery_cache = doc
    _discovery_at = now
    return doc


# ── PKCE + state ─────────────────────────────────────────────────────────────

def new_pkce() -> tuple[str, str]:
    """Return (verifier, challenge). verifier travels only server→IdP."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(48)).decode().rstrip("=")
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()).decode().rstrip("=")
    return verifier, challenge


def _flow_key(data_dir: str = "data") -> bytes:
    # Reuse the Phase 1 derivation shape with a distinct label.
    from .crypto import get_secret_key
    master = get_secret_key(data_dir)
    if isinstance(master, str):
        master = master.encode("utf-8")
    return hmac.new(master, _STATE_INFO_LABEL, hashlib.sha256).digest()


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def issue_state_cookie(state: str, verifier: str, next_path: str,
                       data_dir: str = "data", now: float | None = None) -> str:
    issued = int(now if now is not None else time.time())
    payload = {"state": state, "verifier": verifier, "next": next_path,
               "iat": issued, "exp": issued + STATE_TTL_SECONDS}
    body = _b64e(json.dumps(payload, separators=(",", ":"),
                            sort_keys=True).encode("utf-8"))
    sig = hmac.new(_flow_key(data_dir), body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_b64e(sig)}"


def verify_state_cookie(value: str | None, data_dir: str = "data",
                        now: float | None = None) -> dict[str, Any] | None:
    if not value or "." not in value:
        return None
    body, _, sig_b64 = value.rpartition(".")
    if not body or not sig_b64:
        return None
    try:
        provided = _b64d(sig_b64)
    except (ValueError, base64.binascii.Error):  # type: ignore[attr-defined]
        return None
    if not hmac.compare_digest(
            provided, hmac.new(_flow_key(data_dir), body.encode("ascii"),
                               hashlib.sha256).digest()):
        return None
    try:
        payload = json.loads(_b64d(body))
    except (ValueError, base64.binascii.Error):  # type: ignore[attr-defined]
        return None
    if not isinstance(payload, dict):
        return None
    if (now if now is not None else time.time()) > payload.get("exp", 0):
        return None
    return payload


# ── the flow ─────────────────────────────────────────────────────────────────

def authorization_url(cfg: dict[str, str], state: str, challenge: str,
                      client: httpx.Client | None = None) -> str:
    doc = discovery(cfg, client)
    params = {
        "response_type": "code",
        "client_id": cfg["client_id"],
        "redirect_uri": cfg["redirect_uri"],
        "scope": SCOPE,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return doc["authorization_endpoint"] + "?" + urlencode(params)


def exchange_code(cfg: dict[str, str], code: str, verifier: str,
                  client: httpx.Client | None = None) -> dict[str, Any]:
    doc = discovery(cfg, client)
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": cfg["redirect_uri"],
        "client_id": cfg["client_id"],
        "client_secret": cfg["client_secret"],
        "code_verifier": verifier,
    }
    own = client is None
    if own:
        client = httpx.Client(timeout=10, verify=True)
    try:
        resp = client.post(doc["token_endpoint"], data=data)
        if resp.status_code != 200:
            # Log the IdP's error class, never the body (it can echo the code).
            logger.warning("oidc_token_exchange_failed", status=resp.status_code)
            raise OIDCLoginError("token exchange rejected")
        token = resp.json()
    finally:
        if own:
            client.close()
    if "access_token" not in token:
        raise OIDCLoginError("token response had no access_token")
    return token


def fetch_userinfo(cfg: dict[str, str], access_token: str,
                   client: httpx.Client | None = None) -> dict[str, Any]:
    doc = discovery(cfg, client)
    own = client is None
    if own:
        client = httpx.Client(timeout=10, verify=True)
    try:
        resp = client.get(doc["userinfo_endpoint"],
                          headers={"Authorization": f"Bearer {access_token}"})
        if resp.status_code != 200:
            logger.warning("oidc_userinfo_failed", status=resp.status_code)
            raise OIDCLoginError("userinfo rejected")
        claims = resp.json()
    finally:
        if own:
            client.close()
    if not isinstance(claims, dict):
        raise OIDCLoginError("userinfo was not a claim set")
    identity = claims.get(admin_claim()) or claims.get(DEFAULT_ADMIN_CLAIM)
    if not identity:
        raise OIDCLoginError(f"userinfo lacked {admin_claim()}")
    return claims


def decide_session(claims: dict[str, Any]) -> str | None:
    """Return the session actor for this identity, or None to deny.

    The deny is silent-by-page: the operator sees "not authorized", the log
    sees who, and nothing about the allow-list shape is exposed.
    """
    username = str(claims.get(admin_claim())
                   or claims.get(DEFAULT_ADMIN_CLAIM) or "")
    if username in admin_usernames():
        return f"oidc:{username}"
    logger.warning("oidc_login_denied", preferred_username=username)
    return None
