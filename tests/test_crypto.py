"""Tests for src.api.crypto — encryption/decryption of provider secrets."""

import os
from unittest.mock import patch

import pytest

from src.api import crypto


@pytest.fixture
def temp_data_dir(tmp_path):
    """Isolated data dir so the on-disk fallback key never touches real data/."""
    return str(tmp_path / "data")


class TestDeriveKey:
    def test_derive_fernet_key_is_32_bytes_b64(self):
        key = crypto.derive_fernet_key("some-secret")
        import base64
        decoded = base64.urlsafe_b64decode(key)
        assert len(decoded) == 32

    def test_derive_is_deterministic(self):
        assert crypto.derive_fernet_key("abc") == crypto.derive_fernet_key("abc")
        assert crypto.derive_fernet_key("abc") != crypto.derive_fernet_key("abd")


class TestEncryptDecrypt:
    def test_roundtrip(self, temp_data_dir):
        with patch.dict(os.environ, {"LCP_SECRET_KEY": "master-key-123"}):
            token = crypto.encrypt_secret("sk-supersecret", temp_data_dir)
            assert token != "sk-supersecret"
            assert crypto.decrypt_secret(token, temp_data_dir) == "sk-supersecret"

    def test_empty_plaintext_returns_empty(self, temp_data_dir):
        with patch.dict(os.environ, {"LCP_SECRET_KEY": "master-key-123"}):
            assert crypto.encrypt_secret("", temp_data_dir) == ""
            assert crypto.decrypt_secret("", temp_data_dir) == ""

    def test_decrypt_wrong_key_returns_empty(self, temp_data_dir):
        with patch.dict(os.environ, {"LCP_SECRET_KEY": "key-A"}):
            token = crypto.encrypt_secret("sk-x", temp_data_dir)
        with patch.dict(os.environ, {"LCP_SECRET_KEY": "key-B"}):
            assert crypto.decrypt_secret(token, temp_data_dir) == ""

    def test_ciphertext_is_randomized(self, temp_data_dir):
        with patch.dict(os.environ, {"LCP_SECRET_KEY": "master-key-123"}):
            t1 = crypto.encrypt_secret("sk-same", temp_data_dir)
            t2 = crypto.encrypt_secret("sk-same", temp_data_dir)
            assert t1 != t2


class TestSecretKeyResolution:
    def test_env_var_wins(self, temp_data_dir):
        with patch.dict(os.environ, {"LCP_SECRET_KEY": "from-env"}):
            assert crypto.get_secret_key(temp_data_dir) == b"from-env"

    def test_fallback_key_created_and_reused(self, temp_data_dir):
        import os.path as _osp
        with patch.dict(os.environ, {}, clear=True):
            k1 = crypto.get_secret_key(temp_data_dir)
            k2 = crypto.get_secret_key(temp_data_dir)
            assert k1 == k2  # persisted and reused
            assert _osp.exists(_osp.join(temp_data_dir, ".lcp_secret_key"))
