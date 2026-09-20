"""SSRF destination guard tests for the provider test/discover endpoints (CWE-918).

Covers the ``_validate_api_base_destination`` helper (unit) and the two
handlers' reject/allow behavior (handler-level, mirroring test_precision_gaps).
"""

import ipaddress
import json
import socket
from unittest.mock import MagicMock, patch

import pytest

from src.server.endpoints import (
    _PRIVATE_NETWORKS,
    _ssrf_allowlist,
    _validate_api_base_destination,
)


@pytest.fixture
def temp_db():
    import os
    import tempfile as _t
    from src.api.models import get_engine, Base
    fd, path = _t.mkstemp(suffix=".db")
    os.close(fd)
    engine = get_engine(path)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()
    for ext in ["", "-wal", "-shm"]:
        try:
            os.unlink(path + ext)
        except FileNotFoundError:
            pass


@pytest.fixture(autouse=True)
def _setup_handler_config(temp_db):
    from src.server import LCPHandler
    from src.api.key_manager import KeyManager
    import src.api.key_manager as key_manager_mod
    key_manager_mod._key_manager = KeyManager(temp_db, "data")
    cfg = MagicMock()
    cfg.server = {"port": 8734, "default_profile": "l2"}
    cfg.profiles = {
        "l2": {
            "forbidden_tools": [],
            "chain": [{"provider": "deepseek", "model": "deepseek-v4-pro", "base_url": "https://t/v1"}],
            "auth_required": False,
        },
    }
    cfg.providers = {"deepseek": {"api_base": "https://t/v1", "models": ["deepseek-v4-pro"]}}
    cfg.pricing = [{"provider": "deepseek", "model": "deepseek-v4-pro", "cache_hit": 0.01, "cache_miss": 0.5, "output": 1.0}]
    cfg.circuit_breaker = {"failures_dead": 5, "dead_cooldown_seconds": 300, "failures_degraded": 3, "degraded_cooldown_seconds": 60}
    cfg.database = {"path": "/tmp/test.db", "wal_mode": True}
    cfg.model_limits = {}
    cfg.get_profile = lambda name: cfg.profiles.get(name)
    cfg.get_pricing = lambda provider, model: cfg.pricing[0]
    cfg.get_provider_key = lambda name: "test-key"
    cfg.check_reload = MagicMock()
    cfg.raw = {"providers": dict(cfg.providers), "profiles": dict(cfg.profiles)}
    cfg.save = MagicMock()
    LCPHandler.config = cfg
    LCPHandler.engine = temp_db


def _addrinfo(*addrs):
    """Build a socket.getaddrinfo-style result list from (ip, port) pairs."""
    out = []
    for ip, port in addrs:
        family = ipaddress.ip_address(ip).version
        socktype = socket.SOCK_STREAM
        out.append((socket.AF_INET if family == 4 else socket.AF_INET6, socktype, 6, "", (ip, port)))
    return out


class TestValidateApiBaseDestination:
    def test_http_public_ip_allowed(self):
        with patch("src.server.endpoints.socket.getaddrinfo", return_value=_addrinfo(("8.8.8.8", 80))):
            assert _validate_api_base_destination("http://8.8.8.8") is None

    def test_https_public_hostname_allowed(self):
        with patch("src.server.endpoints.socket.getaddrinfo", return_value=_addrinfo(("104.18.0.1", 443), ("2606:4700::1", 443))):
            assert _validate_api_base_destination("https://api.example.com/v1") is None

    def test_loopback_ip_rejected(self):
        with patch("src.server.endpoints.socket.getaddrinfo", return_value=_addrinfo(("127.0.0.1", 8734))):
            reason = _validate_api_base_destination("http://127.0.0.1:8734")
        assert reason is not None
        assert "127.0.0.1" in reason

    def test_localhost_hostname_rejected(self):
        with patch("src.server.endpoints.socket.getaddrinfo", return_value=_addrinfo(("127.0.0.1", 80))):
            assert _validate_api_base_destination("http://localhost") is not None

    def test_metadata_ip_rejected(self):
        with patch("src.server.endpoints.socket.getaddrinfo", return_value=_addrinfo(("169.254.169.254", 80))):
            reason = _validate_api_base_destination("http://169.254.169.254/latest/meta-data")
        assert reason is not None

    def test_rfc1918_rejected(self):
        for ip in ("10.0.0.5", "172.16.0.1", "192.168.1.107"):
            with patch("src.server.endpoints.socket.getaddrinfo", return_value=_addrinfo((ip, 8000))):
                reason = _validate_api_base_destination(f"http://{ip}:8000")
            assert reason is not None

    def test_ipv6_loopback_rejected(self):
        with patch("src.server.endpoints.socket.getaddrinfo", return_value=_addrinfo(("::1", 80))):
            assert _validate_api_base_destination("http://[::1]") is not None

    def test_ipv6_ula_rejected(self):
        with patch("src.server.endpoints.socket.getaddrinfo", return_value=_addrinfo(("fd00::1", 80))):
            assert _validate_api_base_destination("http://[fd00::1]") is not None

    def test_unresolvable_host_allowed_through(self):
        with patch("src.server.endpoints.socket.getaddrinfo", side_effect=socket.gaierror("name not known")):
            assert _validate_api_base_destination("http://no-such-host.invalid") is None

    def test_unsupported_scheme_rejected(self):
        assert _validate_api_base_destination("file:///etc/passwd") is not None
        assert _validate_api_base_destination("gopher://127.0.0.1:70") is not None

    def test_missing_host_rejected(self):
        assert _validate_api_base_destination("http://") is not None

    def test_invalid_port_rejected(self):
        assert _validate_api_base_destination("http://8.8.8.8:99999") is not None

    def test_mixed_public_and_private_rejected(self):
        with patch("src.server.endpoints.socket.getaddrinfo", return_value=_addrinfo(("8.8.8.8", 80), ("127.0.0.1", 80))):
            assert _validate_api_base_destination("http://mixed.example.com") is not None

    def test_allowlist_permits_private(self, monkeypatch):
        monkeypatch.setenv("LCP_SSRF_ALLOWLIST", "127.0.0.0/8,192.168.0.0/16")
        with patch("src.server.endpoints.socket.getaddrinfo", return_value=_addrinfo(("192.168.1.107", 18300))):
            assert _validate_api_base_destination("http://192.168.1.107:18300") is None
        monkeypatch.delenv("LCP_SSRF_ALLOWLIST", raising=False)

    def test_allowlist_does_not_cover_unlisted_private(self, monkeypatch):
        monkeypatch.setenv("LCP_SSRF_ALLOWLIST", "10.0.0.0/8")
        with patch("src.server.endpoints.socket.getaddrinfo", return_value=_addrinfo(("127.0.0.1", 80))):
            assert _validate_api_base_destination("http://127.0.0.1") is not None
        monkeypatch.delenv("LCP_SSRF_ALLOWLIST", raising=False)

    def test_invalid_allowlist_entries_skipped(self, monkeypatch):
        monkeypatch.setenv("LCP_SSRF_ALLOWLIST", "not-a-cidr,127.0.0.0/8")
        assert _ssrf_allowlist() == [ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_network("::1/128")]
        monkeypatch.delenv("LCP_SSRF_ALLOWLIST", raising=False)

    def test_allowlist_loopback_permits_dualstack(self, monkeypatch):
        monkeypatch.setenv("LCP_SSRF_ALLOWLIST", "127.0.0.0/8")
        with patch("src.server.endpoints.socket.getaddrinfo", return_value=_addrinfo(("127.0.0.1", 8080), ("::1", 8080))):
            assert _validate_api_base_destination("http://localhost:8080") is None
        assert ipaddress.ip_network("::1/128") in _ssrf_allowlist()
        monkeypatch.delenv("LCP_SSRF_ALLOWLIST", raising=False)

    def test_private_networks_cover_metadata_and_loopback(self):
        nets = {str(n) for n in _PRIVATE_NETWORKS}
        assert "127.0.0.0/8" in nets
        assert "169.254.0.0/16" in nets
        assert "192.168.0.0/16" in nets
        assert "10.0.0.0/8" in nets
        assert "::1/128" in nets
        assert "fc00::/7" in nets


class TestProviderEndpointsSsrGuard:
    def test_provider_test_loopback_rejected_400(self, temp_db):
        from tests.test_server import TestHandler, _status, _json_body
        body = json.dumps({"api_base": "http://127.0.0.1:8734", "api_key": "x", "model": "m"})
        h = TestHandler(path="/api/providers/test", method="POST", engine=temp_db, body=body)
        with patch("src.server.endpoints.socket.getaddrinfo", return_value=_addrinfo(("127.0.0.1", 8734))):
            h.do_POST()
        assert _status(h) == 400
        assert "not allowed" in _json_body(h)["error"]

    def test_provider_discover_metadata_rejected_400(self, temp_db):
        from tests.test_server import TestHandler, _status, _json_body
        body = json.dumps({"api_base": "http://169.254.169.254/latest/meta-data"})
        h = TestHandler(path="/api/providers/discover", method="POST", engine=temp_db, body=body)
        with patch("src.server.endpoints.socket.getaddrinfo", return_value=_addrinfo(("169.254.169.254", 80))):
            h.do_POST()
        assert _status(h) == 400
        assert "not allowed" in _json_body(h)["error"]

    def test_provider_test_private_rejected_before_urlopen(self, temp_db):
        """The fetch must never happen — urlopen stays unmocked and would
        raise if called; the guard returns 400 before it."""
        from tests.test_server import TestHandler, _status
        body = json.dumps({"api_base": "http://192.168.1.50:81", "api_key": "x", "model": "m"})
        h = TestHandler(path="/api/providers/test", method="POST", engine=temp_db, body=body)
        with patch("src.server.endpoints.socket.getaddrinfo", return_value=_addrinfo(("192.168.1.50", 81))):
            h.do_POST()
        assert _status(h) == 400

    def test_provider_test_public_allowed(self, temp_db):
        from tests.test_server import TestHandler, _status, _json_body
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({"model": "m1"}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        body = json.dumps({"api_base": "https://api.example.com/v1", "api_key": "k", "model": "m"})
        h = TestHandler(path="/api/providers/test", method="POST", engine=temp_db, body=body)
        with patch("src.server.endpoints.socket.getaddrinfo", return_value=_addrinfo(("104.18.0.1", 443))), \
             patch("urllib.request.urlopen", return_value=mock_resp):
            h.do_POST()
        assert _status(h) == 200
        assert _json_body(h)["ok"] is True