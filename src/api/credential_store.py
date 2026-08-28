"""Credential store — encrypted persistence of upstream provider API keys.

Keys entered via the UI are encrypted with Fernet (src.api.crypto) and stored
in the ``provider_credentials`` table. This module provides a singleton store
that mirrors the KeyManager pattern (init with engine at server startup).

Precedence when resolving a provider key at request time (in forward_request):
  1. env var named by ``api_key_env`` (container/deployment-managed)
  2. encrypted credential stored via the UI
  3. ``config.get_provider_key`` (env var again, raises if unset)
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from .crypto import encrypt_secret, decrypt_secret
from .logging_config import get_logger
from .models import ProviderCredential, get_session

if TYPE_CHECKING:  # pragma: no cover — runtime import only for type hints
    from .component import Component
    from .runtime import Runtime
else:
    from .component import Component
    from .runtime import Runtime

logger = get_logger("lcp.credentials")


class CredentialStore:
    """Persists encrypted provider credentials in SQLite."""

    def __init__(self, engine, data_dir: str = "data"):
        self._engine = engine
        self._data_dir = data_dir

    def set(self, provider: str, plaintext: str) -> None:
        """Upsert an encrypted credential for a provider. Empty clears it."""
        now = datetime.now(timezone.utc).isoformat()
        with get_session(self._engine) as session:
            row = session.query(ProviderCredential).filter(
                ProviderCredential.provider == provider
            ).first()
            if not plaintext:
                if row is not None:
                    session.delete(row)
                    session.commit()
                return
            token = encrypt_secret(plaintext, self._data_dir)
            if row is None:
                session.add(ProviderCredential(
                    provider=provider,
                    encrypted_key=token,
                    created_at=now,
                    updated_at=now,
                ))
            else:
                row.encrypted_key = token
                row.updated_at = now
            session.commit()
        logger.info("credential_upserted", provider=provider)

    def get(self, provider: str) -> str | None:
        """Return the decrypted plaintext key for a provider, or None."""
        with get_session(self._engine) as session:
            row = session.query(ProviderCredential).filter(
                ProviderCredential.provider == provider
            ).first()
        if row is None:
            return None
        return decrypt_secret(row.encrypted_key, self._data_dir) or None

    def has(self, provider: str) -> bool:
        """Return True if a credential exists for this provider."""
        with get_session(self._engine) as session:
            return session.query(ProviderCredential).filter(
                ProviderCredential.provider == provider
            ).first() is not None

    # ── Cookies (e.g. OpenCode browser session) ───────────────────────────

    @staticmethod
    def _cookie_key(provider: str) -> str:
        """Namespaced key so cookies never collide with API keys."""
        return f"cookie:{provider}"

    def set_cookie(self, provider: str, cookie: str) -> None:
        """Upsert an encrypted cookie for a provider. Empty clears it."""
        self.set(self._cookie_key(provider), cookie)

    def get_cookie(self, provider: str) -> str | None:
        """Return the decrypted cookie for a provider, or None."""
        return self.get(self._cookie_key(provider))

    def has_cookie(self, provider: str) -> bool:
        """Return True if a cookie exists for this provider."""
        return self.has(self._cookie_key(provider))

    # ── Workspace IDs (e.g. OpenCode wrk_*) ────────────────────────────────

    @staticmethod
    def _ws_id_key(provider: str) -> str:
        """Namespaced key so workspace IDs never collide with API keys or cookies."""
        return f"wsid:{provider}"

    def set_workspace_id(self, provider: str, ws_id: str) -> None:
        """Upsert a workspace ID for a provider. Empty clears it."""
        self.set(self._ws_id_key(provider), ws_id)

    def get_workspace_id(self, provider: str) -> str | None:
        """Return the decrypted workspace ID for a provider, or None."""
        return self.get(self._ws_id_key(provider))

    def has_workspace_id(self, provider: str) -> bool:
        """Return True if a workspace ID exists for this provider."""
        return self.has(self._ws_id_key(provider))


_credential_store: CredentialStore | None = None


def get_credential_store(engine=None, data_dir: str = "data") -> CredentialStore | None:
    """Get the active credential store (None if not initialized).

    Delegates to the runtime's CredentialStoreComponent when bound; otherwise
    the legacy singleton (lazy-created with *engine*).
    """
    global _runtime
    if _runtime is not None:
        try:
            comp = _runtime.resolve("credential_store")
        except Exception:  # noqa: BLE001 — inactive/unbound → legacy
            comp = None
        if comp is not None and getattr(comp, "store", None) is not None:
            return comp.store
    global _credential_store
    if _credential_store is None and engine is not None:
        _credential_store = CredentialStore(engine, data_dir)
    return _credential_store


def init_credential_store(engine, data_dir: str = "data") -> CredentialStore:
    """Initialize the credential store singleton with the engine."""
    global _credential_store
    _credential_store = CredentialStore(engine, data_dir)
    return _credential_store


# ── Component-runtime adapter (Phase C) ────────────────────────────────────
_runtime: Optional["Runtime"] = None


def bind_runtime(rt: "Runtime") -> None:
    """Bind an active Runtime so ``get_credential_store()`` delegates to it."""
    global _runtime
    _runtime = rt


class CredentialStoreComponent(Component):
    """The encrypted credential store as a runtime component.

    ``requires=["engine", "data_dir"]`` — data_dir selects the on-disk fallback
    key used by the crypto layer.
    """

    name = "credential_store"
    requires = ["engine", "data_dir"]
    provides = ["credential_store"]

    def __init__(self) -> None:
        super().__init__()
        self.store: Optional[CredentialStore] = None

    def setup(self, rt: "Runtime") -> Optional[Any]:
        self.store = CredentialStore(rt.resolve("engine"), rt.resolve("data_dir") or "data")
        return None
