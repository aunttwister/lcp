"""Secret encryption/decryption helpers for provider API keys.

Provider API keys entered via the UI are encrypted at rest using Fernet
(symmetric AES-128-CBC + HMAC) and stored in the SQLite database — never in
plaintext and never in the git-tracked gateway.yaml.

The master encryption key comes from the ``LCP_SECRET_KEY`` environment
variable. When it is unset, we fall back to a per-install random key persisted
next to the config file so the feature works out-of-the-box, while logging a
clear warning that the key is on disk.

This module intentionally has NO global state: every function takes an
explicit ``secret_key`` so callers (and tests) control the key.
"""

import base64
import hashlib
import os
from pathlib import Path

from .logging_config import get_logger

logger = get_logger("lcp.crypto")

# Env var that holds the master secret used to derive the Fernet key.
SECRET_KEY_ENV = "LCP_SECRET_KEY"

# Fallback file (gitignored) holding a random key when LCP_SECRET_KEY is unset.
_FALLBACK_KEY_FILE = "data/.lcp_secret_key"


def derive_fernet_key(secret: str | bytes) -> bytes:
    """Derive a url-safe 32-byte Fernet key from an arbitrary secret string."""
    if isinstance(secret, str):
        secret = secret.encode("utf-8")
    # SHA-256 of the secret → exactly 32 bytes, url-safe b64-encoded for Fernet.
    digest = hashlib.sha256(secret).digest()
    return base64.urlsafe_b64encode(digest)


def _load_or_create_fallback_key(data_dir: str | Path) -> bytes | None:
    """Load the on-disk fallback key, creating it if missing.

    Returns None only on unexpected I/O errors (logged); otherwise always
    returns a key so encryption works even without LCP_SECRET_KEY.
    """
    path = Path(data_dir) / Path(_FALLBACK_KEY_FILE).name
    try:
        if path.exists():
            # Read the EXACT bytes: the writer stores raw os.urandom(32), and
            # the first/last byte can legitimately be whitespace (\r, \n, \t,
            # space). Stripping would corrupt such keys and break round-trip
            # (the persisted key would differ from the one first generated).
            return path.read_bytes()
        key = os.urandom(32)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write with 0o600 so only the owner can read it.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, key)
        finally:
            os.close(fd)
        logger.warning(
            "crypto_fallback_key_created",
            path=str(path),
            hint=f"Set {SECRET_KEY_ENV} to control the master key instead.",
        )
        return key
    except Exception as e:
        logger.error("crypto_fallback_key_failed", error=str(e))
        return None


def get_secret_key(data_dir: str | Path = "data") -> bytes:
    """Return the master secret bytes used to derive the Fernet key.

    Priority: ``LCP_SECRET_KEY`` env var → on-disk fallback key.
    """
    env = os.environ.get(SECRET_KEY_ENV)
    if env:
        return env.encode("utf-8")
    key = _load_or_create_fallback_key(data_dir)
    if key is None:
        raise RuntimeError(
            f"Unable to obtain a secret key: set {SECRET_KEY_ENV} or ensure the "
            "fallback key file is writable."
        )
    return key


def encrypt_secret(plaintext: str, data_dir: str | Path = "data") -> str:
    """Encrypt a plaintext secret, returning a url-safe base64 Fernet token."""
    from cryptography.fernet import Fernet, InvalidToken

    if not plaintext:
        return ""
    try:
        fernet = Fernet(derive_fernet_key(get_secret_key(data_dir)))
        return fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")
    except InvalidToken:  # pragma: no cover - defensive
        logger.error("crypto_encrypt_invalid_token")
        raise


def decrypt_secret(token: str, data_dir: str | Path = "data") -> str:
    """Decrypt a Fernet token back to plaintext. Returns '' for empty/None."""
    from cryptography.fernet import Fernet, InvalidToken

    if not token:
        return ""
    try:
        fernet = Fernet(derive_fernet_key(get_secret_key(data_dir)))
        return fernet.decrypt(token.encode("utf-8")).decode("utf-8")
    except (InvalidToken, Exception) as e:
        logger.error("crypto_decrypt_failed", error=str(e))
        return ""
