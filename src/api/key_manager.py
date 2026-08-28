"""API key manager — DB-backed persistent storage and validation.

Keys are stored in the SQLite database via SQLAlchemy.
Uses SHA-256 hashing for key storage (raw keys shown only once).
Migrates legacy JSON keys on first load.
"""

import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .logging_config import get_logger
from .models import ApiKey, get_session

if TYPE_CHECKING:  # pragma: no cover — runtime import only for type hints
    from .component import Component
    from .runtime import Runtime
else:
    from .component import Component
    from .runtime import Runtime

logger = get_logger("lcp.keys")


class KeyManager:
    """Manages API keys with SQLAlchemy DB persistence."""

    def __init__(self, engine, data_dir: str = "data"):
        self._engine = engine
        self._data_dir = Path(data_dir)
        self._migrate_legacy_keys()

    def _migrate_legacy_keys(self):
        """Migrate keys from legacy JSON file to DB if any exist."""
        legacy_path = self._data_dir / "api_keys.json"
        if not legacy_path.exists():
            return
        try:
            with open(legacy_path) as f:
                legacy = json.load(f)
            keys = legacy.get("keys", [])
            if not keys:
                return
            with get_session(self._engine) as session:
                for k in keys:
                    existing = session.query(ApiKey).filter(
                        ApiKey.key_hash == k.get("hash", "")
                    ).first()
                    if existing:
                        continue
                    entry = ApiKey(
                        key_hash=k.get("hash", ""),
                        key_prefix=k.get("id", "")[:8],
                        name=k.get("label", f"Key for {k.get('profile', '')}"),
                        allowed_profiles=k.get("profile", ""),
                        status="active",
                        created_at=k.get("created", datetime.now(timezone.utc).isoformat()),
                        last_used_at=k.get("last_used"),
                    )
                    session.add(entry)
                session.commit()
            # Rename legacy file after successful migration
            legacy_path.rename(legacy_path.with_suffix(".json.bak"))
            logger.info("key_migration_complete", count=len(keys))
        except Exception as e:
            logger.error("key_migration_failed", error=str(e))

    def list_keys(self) -> list[dict]:
        """List all keys with stats."""
        with get_session(self._engine) as session:
            keys = session.query(ApiKey).order_by(ApiKey.id.desc()).all()
            logger.debug("key_listed", count=len(keys))
            return [
                {
                    "id": k.id,
                    "key_prefix": k.key_prefix,
                    "name": k.name,
                    "allowed_profiles": k.allowed_profiles,
                    "spend_limit": k.spend_limit,
                    "total_spend": k.total_spend,
                    "status": k.status,
                    "created_at": k.created_at,
                    "last_used_at": k.last_used_at,
                    "expires_at": k.expires_at,
                    "revoked_at": k.revoked_at,
                    "metadata_tags": k.metadata_tags,
                }
                for k in keys
            ]

    def create_key(
        self,
        name: str = "",
        allowed_profiles: str = "",
        spend_limit: float = 0.0,
        expires_at: str = "",
        metadata_tags: str = "",
    ) -> dict:
        """Create a new API key. Returns dict with raw key (one-time view)."""
        raw_key = "lcp_" + secrets.token_hex(24)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_prefix = raw_key[:12]  # "lcp_" + 8 hex chars

        with get_session(self._engine) as session:
            entry = ApiKey(
                key_hash=key_hash,
                key_prefix=key_prefix,
                name=name or f"Key {key_prefix}",
                allowed_profiles=allowed_profiles or None,
                spend_limit=spend_limit or 0.0,
                status="active",
                created_at=datetime.now(timezone.utc).isoformat(),
                expires_at=expires_at or None,
                metadata_tags=metadata_tags or None,
            )
            session.add(entry)
            session.commit()
            key_id = entry.id

        logger.info("key_created", key_id=key_id, name=name)
        return {
            "ok": True,
            "key": raw_key,
            "id": key_id,
            "key_prefix": key_prefix,
            "name": entry.name,
            "allowed_profiles": entry.allowed_profiles,
            "spend_limit": entry.spend_limit,
        }

    def rotate_key(self, key_id: int) -> dict | None:
        """Rotate a key: revoke old, create new with same permissions. Returns new key info."""
        with get_session(self._engine) as session:
            old = session.query(ApiKey).filter(ApiKey.id == key_id).first()
            if not old:
                return None

            # Revoke old key
            old.status = "revoked"
            old.revoked_at = datetime.now(timezone.utc).isoformat()

            # Create new key with same settings
            raw_key = "lcp_" + secrets.token_hex(24)
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
            key_prefix = raw_key[:12]

            new_key = ApiKey(
                key_hash=key_hash,
                key_prefix=key_prefix,
                name=old.name,
                allowed_profiles=old.allowed_profiles,
                spend_limit=old.spend_limit,
                status="active",
                created_at=datetime.now(timezone.utc).isoformat(),
                expires_at=old.expires_at,
                metadata_tags=old.metadata_tags,
            )
            session.add(new_key)
            session.commit()
            new_id = new_key.id

        logger.info("key_rotated", old_id=key_id, new_id=new_id)
        return {
            "ok": True,
            "key": raw_key,
            "id": new_id,
            "key_prefix": key_prefix,
            "name": new_key.name,
            "old_id": key_id,
        }

    def revoke_key(self, key_id: int) -> bool:
        """Soft-revoke a key by ID. Returns True if found."""
        with get_session(self._engine) as session:
            key = session.query(ApiKey).filter(ApiKey.id == key_id).first()
            if not key:
                return False
            key.status = "revoked"
            key.revoked_at = datetime.now(timezone.utc).isoformat()
            session.commit()
        logger.info("key_revoked", key_id=key_id)
        return True

    def get_key(self, key_id: int) -> dict | None:
        """Get a single key by ID."""
        with get_session(self._engine) as session:
            k = session.query(ApiKey).filter(ApiKey.id == key_id).first()
            if not k:
                return None
            return {
                "id": k.id,
                "key_prefix": k.key_prefix,
                "name": k.name,
                "allowed_profiles": k.allowed_profiles,
                "spend_limit": k.spend_limit,
                "total_spend": k.total_spend,
                "status": k.status,
                "created_at": k.created_at,
                "last_used_at": k.last_used_at,
                "expires_at": k.expires_at,
                "revoked_at": k.revoked_at,
                "metadata_tags": k.metadata_tags,
            }

    def validate_key(self, raw_key: str) -> dict | None:
        """Validate an API key. Returns key info dict if valid, None otherwise."""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        with get_session(self._engine) as session:
            k = session.query(ApiKey).filter(
                ApiKey.key_hash == key_hash,
                ApiKey.status == "active",
            ).first()
            if not k:
                # Check if it exists at all (for better error logging)
                any_key = session.query(ApiKey).filter(
                    ApiKey.key_hash == key_hash
                ).first()
                if any_key:
                    logger.warning("key_validation_failed", reason="revoked_or_inactive",
                                   key_prefix=any_key.key_prefix, key_id=any_key.id)
                else:
                    logger.warning("key_validation_failed", reason="not_found")
                return None
            # Check expiry
            if k.expires_at:
                try:
                    exp = datetime.fromisoformat(k.expires_at)
                    if exp.tzinfo is None:
                        exp = exp.replace(tzinfo=timezone.utc)
                    if datetime.now(timezone.utc) > exp:
                        logger.warning("key_validation_failed", reason="expired",
                                       key_prefix=k.key_prefix, key_id=k.id)
                        return None
                except ValueError:
                    pass
            # Update last used
            k.last_used_at = datetime.now(timezone.utc).isoformat()
            session.commit()
            return {
                "id": k.id,
                "name": k.name,
                "allowed_profiles": k.allowed_profiles,
                "spend_limit": k.spend_limit,
                "total_spend": k.total_spend,
            }

# ── Module-level singleton ────────────────────────────────────────────────
_key_manager: KeyManager | None = None


def get_key_manager(engine=None, data_dir: str = "data") -> KeyManager:
    """Get the active key manager.

    Delegates to the runtime's KeyManagerComponent when bound; otherwise the
    legacy singleton (lazy-created with *engine*).
    """
    from .runtime import get_runtime
    rt = get_runtime()
    if rt is not None:
        try:
            comp = rt.resolve("key_manager")
        except Exception:  # noqa: BLE001 — inactive/unbound → legacy
            comp = None
        if comp is not None and getattr(comp, "manager", None) is not None:
            return comp.manager
    global _key_manager
    if _key_manager is None and engine is not None:
        _key_manager = KeyManager(engine, data_dir)
    return _key_manager


# ── Component-runtime adapter (Phase C) ────────────────────────────────────


def bind_runtime(rt: "Runtime") -> None:
    """Bind an active Runtime so ``get_key_manager()`` delegates to it."""
    from .runtime import bind_active_runtime
    bind_active_runtime(rt)


class KeyManagerComponent(Component):
    """The API-key manager as a runtime component.

    ``requires=["engine", "data_dir"]`` — both declared; the data_dir only
    feeds legacy JSON-key migration.
    """

    name = "key_manager"
    requires = ["engine", "data_dir"]
    provides = ["key_manager"]

    def __init__(self) -> None:
        super().__init__()
        self.manager: Optional[KeyManager] = None

    @property
    def service(self) -> Optional[KeyManager]:
        return self.manager

    def setup(self, rt: "Runtime") -> Optional[Any]:
        self.manager = KeyManager(rt.resolve("engine"), rt.resolve("data_dir") or "data")
        return None

