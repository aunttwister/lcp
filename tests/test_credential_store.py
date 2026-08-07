"""Tests for src.api.credential_store — encrypted provider credential persistence."""

import os
from unittest.mock import patch

import pytest

from src.api.credential_store import CredentialStore, init_credential_store, get_credential_store
from src.api.models import ProviderCredential, get_session


@pytest.fixture
def store(temp_db, temp_dir):
    """CredentialStore backed by the temp DB + isolated data dir."""
    db_path, engine = temp_db
    return CredentialStore(engine, data_dir=str(temp_dir))


@pytest.fixture(autouse=True)
def _isolate_secret():
    """Use a fixed master key so encrypt/decrypt is deterministic across tests."""
    with patch.dict(os.environ, {"LCP_SECRET_KEY": "test-master-key"}, clear=False):
        yield


class TestCredentialStore:
    def test_set_and_get(self, store):
        store.set("deepseek", "sk-ds-123")
        assert store.get("deepseek") == "sk-ds-123"

    def test_stores_ciphertext_not_plaintext(self, store, temp_db):
        db_path, engine = temp_db
        store.set("deepseek", "sk-plaintext")
        with get_session(engine) as session:
            row = session.query(ProviderCredential).filter(
                ProviderCredential.provider == "deepseek"
            ).first()
        assert row is not None
        assert row.encrypted_key != "sk-plaintext"
        assert "sk-plaintext" not in row.encrypted_key

    def test_has(self, store):
        assert store.has("opencode") is False
        store.set("opencode", "sk-oc")
        assert store.has("opencode") is True

    def test_update_overwrites(self, store):
        store.set("p", "key-1")
        store.set("p", "key-2")
        assert store.get("p") == "key-2"

    def test_clear_with_empty_removes(self, store, temp_db):
        db_path, engine = temp_db
        store.set("p", "key-1")
        store.set("p", "")
        assert store.get("p") is None
        assert store.has("p") is False
        with get_session(engine) as session:
            assert session.query(ProviderCredential).filter(
                ProviderCredential.provider == "p"
            ).first() is None

    def test_get_missing_returns_none(self, store):
        assert store.get("nope") is None


class TestCookieStorage:
    def test_set_and_get_cookie(self, store):
        store.set_cookie("opencode", "auth=abc123")
        assert store.get_cookie("opencode") == "auth=abc123"

    def test_cookie_namespace_does_not_collide_with_key(self, store):
        """A provider API key and its cookie live in separate namespaces."""
        store.set("opencode", "sk-apikey")
        store.set_cookie("opencode", "auth=abc123")
        assert store.get("opencode") == "sk-apikey"
        assert store.get_cookie("opencode") == "auth=abc123"

    def test_has_cookie(self, store):
        assert store.has_cookie("opencode") is False
        store.set_cookie("opencode", "auth=abc123")
        assert store.has_cookie("opencode") is True

    def test_clear_cookie(self, store):
        store.set_cookie("opencode", "auth=abc123")
        store.set_cookie("opencode", "")
        assert store.get_cookie("opencode") is None
        assert store.has_cookie("opencode") is False


class TestCredentialStoreSingleton:
    def test_init_and_get(self, temp_db, temp_dir):
        db_path, engine = temp_db
        s = init_credential_store(engine, data_dir=str(temp_dir))
        assert get_credential_store() is s

    def test_uninitialized_returns_none(self):
        import src.api.credential_store as cs
        cs._credential_store = None
        assert get_credential_store() is None
