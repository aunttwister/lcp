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

    def test_fallback_key_leading_whitespace_byte_roundtrips(self, temp_data_dir):
        """Regression: os.urandom(32) may start with an ASCII whitespace byte
        (\r, \n, \t, space, ...). read_bytes().strip() removed it on read-back,
        so the persisted key did not equal the generated key (flaky failure).
        The raw bytes must survive the round-trip untouched."""
        import os
        import os.path as _osp
        key = b"\r\n\t " + b"x" * 28  # leading whitespace bytes
        os.makedirs(temp_data_dir, exist_ok=True)
        path = _osp.join(temp_data_dir, ".lcp_secret_key")
        with open(path, "wb") as fh:
            fh.write(key)
        with patch.dict(os.environ, {}, clear=True):
            assert crypto.get_secret_key(temp_data_dir) == key


class TestCryptoFailClosed:
    """CWE-310 hardening: the fallback key path must fail loudly and
    deterministically, never hand out an absent/falsy key."""

    def test_fallback_io_error_raises_not_none_key(self, temp_data_dir):
        import errno
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.api.crypto.os.open", side_effect=OSError(errno.EACCES, "permission denied")):
                with pytest.raises(RuntimeError):
                    crypto.get_secret_key(temp_data_dir)

    def test_fallback_loader_returns_none_but_get_secret_key_raises(self, temp_data_dir):
        import errno
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.api.crypto.os.open", side_effect=OSError(errno.EACCES, "permission denied")):
                assert crypto._load_or_create_fallback_key(temp_data_dir) is None
                with pytest.raises(RuntimeError):
                    crypto.get_secret_key(temp_data_dir)

    def test_fallback_created_key_matches_persisted_file(self, temp_data_dir):
        import os.path as _osp
        with patch.dict(os.environ, {}, clear=True):
            k = crypto.get_secret_key(temp_data_dir)
            with open(_osp.join(temp_data_dir, ".lcp_secret_key"), "rb") as fh:
                persisted = fh.read()
        assert k == persisted

    def test_decrypt_bad_tag_logs_distinct_event(self, temp_data_dir):
        """A present-but-tampered token must log crypto_decrypt_bad_tag, not the
        catch-all crypto_decrypt_failed, so 'no token' and 'bad tag' differ."""
        with patch.dict(os.environ, {"LCP_SECRET_KEY": "master-key-123"}):
            token = crypto.encrypt_secret("sk-x", temp_data_dir)
            tampered = token[:-6] + "AAAAAA" + token[-6:]
            with patch("src.api.crypto.logger.warning") as mw:
                out = crypto.decrypt_secret(tampered, temp_data_dir)
            with patch("src.api.crypto.logger.error") as me:
                crypto.decrypt_secret(tampered, temp_data_dir)
        assert out == ""
        warn_events = [c.args[0] for c in mw.call_args_list if c.args]
        err_events = [c.args[0] for c in me.call_args_list if c.args]
        assert any("crypto_decrypt_bad_tag" in e for e in warn_events)
        assert not any("crypto_decrypt_failed" in e for e in err_events)

    def test_decrypt_empty_token_no_log(self, temp_data_dir):
        with patch.dict(os.environ, {"LCP_SECRET_KEY": "master-key-123"}):
            with patch("src.api.crypto.logger.warning") as mw, patch("src.api.crypto.logger.error") as me:
                assert crypto.decrypt_secret("", temp_data_dir) == ""
        assert mw.call_count == 0
        assert me.call_count == 0
