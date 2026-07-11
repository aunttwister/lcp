"""API key manager — persistent storage and validation.

Keys are stored in a JSON file alongside the SQLite database.
Uses SHA-256 hashing for key storage (raw keys shown only once).
"""

import hashlib
import json
import os
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .logging_config import get_logger

logger = get_logger("lcp.keys")


class KeyManager:
    """Manages API keys with JSON file persistence."""

    def __init__(self, data_dir: str = "data"):
        self._path = Path(data_dir) / "api_keys.json"
        self._data: dict | None = None

    def _load(self) -> dict:
        if self._data is None:
            if self._path.exists():
                with open(self._path) as f:
                    self._data = json.load(f)
            else:
                self._data = {"keys": []}
        return self._data

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    def list_keys(self) -> list[dict]:
        """List all keys (without hashes)."""
        data = self._load()
        safe = []
        for k in data.get("keys", []):
            safe.append({
                "id": k["id"],
                "profile": k.get("profile", ""),
                "label": k.get("label", ""),
                "created": k.get("created", ""),
                "last_used": k.get("last_used"),
            })
        return safe

    def create_key(self, profile: str, label: str = "") -> dict:
        """Create a new API key. Returns dict with raw key (one-time view)."""
        raw_key = "lcp_" + secrets.token_hex(24)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        data = self._load()
        entry = {
            "id": str(uuid.uuid4())[:8],
            "profile": profile,
            "label": label or f"Key for {profile}",
            "hash": key_hash,
            "created": datetime.now(timezone.utc).isoformat(),
            "last_used": None,
        }
        data.setdefault("keys", []).append(entry)
        self._save()
        logger.info("key_created", profile=profile, key_id=entry["id"])
        return {"ok": True, "key": raw_key, "id": entry["id"], "profile": profile, "label": entry["label"]}

    def revoke_key(self, key_id: str) -> bool:
        """Revoke (delete) a key by ID. Returns True if found and deleted."""
        data = self._load()
        before = len(data.get("keys", []))
        data["keys"] = [k for k in data.get("keys", []) if k["id"] != key_id]
        if len(data["keys"]) == before:
            return False
        self._save()
        logger.info("key_revoked", key_id=key_id)
        return True

    def validate_key(self, raw_key: str) -> dict | None:
        """Validate an API key. Returns key entry dict if valid, None otherwise."""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        data = self._load()
        for k in data.get("keys", []):
            if k.get("hash") == key_hash:
                return k
        return None


# ── Module-level singleton ────────────────────────────────────────────────
_key_manager: KeyManager | None = None


def get_key_manager(data_dir: str = "data") -> KeyManager:
    """Get or create the key manager singleton."""
    global _key_manager
    if _key_manager is None:
        _key_manager = KeyManager(data_dir)
    return _key_manager
