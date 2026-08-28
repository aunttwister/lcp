"""Final precision batch: discover commandcode enrichment, provider-test
cf-ray header path, profile create/update/delete 404 branches, key-manager
not-initialized branches, and metrics error branch."""

import json
from unittest.mock import MagicMock, patch

import pytest


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


class TestDiscoverCommandCode:
    @patch("urllib.request.urlopen")
    def test_discover_commandcode_enriches_plan(self, mock_urlopen, temp_db):
        from tests.test_server import TestHandler, _json_body
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"data": [{"id": "m1"}]}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        reg = MagicMock()
        plugin = MagicMock()
        plugin.discover_models.return_value = None  # fall through to HTTP
        plugin.fetch_subscription.return_value = {"plan_id": "go", "plan_status": "active"}
        reg.for_provider.return_value = plugin
        body = json.dumps({"api_base": "https://api.commandcode.ai/provider/v1", "provider": "commandcode", "api_key": "k"})
        with patch("src.server.endpoints.get_registry", return_value=reg):
            h = TestHandler(path="/api/providers/discover", method="POST", engine=temp_db, body=body)
            h.do_POST()
        assert h.send_response.call_args[0][0] == 200
        result = _json_body(h)
        assert result["ok"] is True
        assert result["plan_id"] == "go"
        assert result["plan_status"] == "active"

    @patch("urllib.request.urlopen")
    def test_discover_commandcode_plan_error(self, mock_urlopen, temp_db):
        from tests.test_server import TestHandler, _json_body
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"data": [{"id": "m1"}]}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        reg = MagicMock()
        plugin = MagicMock()
        plugin.discover_models.return_value = None
        plugin.fetch_subscription.return_value = {"_error": "auth_failed", "detail": "cookie missing"}
        reg.for_provider.return_value = plugin
        body = json.dumps({"api_base": "https://api.commandcode.ai/provider/v1", "provider": "commandcode", "api_key": "k"})
        with patch("src.server.endpoints.get_registry", return_value=reg):
            h = TestHandler(path="/api/providers/discover", method="POST", engine=temp_db, body=body)
            h.do_POST()
        result = _json_body(h)
        assert result["plan_error"] == "cookie missing"


class TestProviderTestCfRay:
    def test_http_error_with_cf_ray_header(self, temp_db):
        import urllib.error
        from tests.test_server import TestHandler, _json_body
        err = urllib.error.HTTPError("url", 403, "Forbidden", {"cf-ray": "ray123"}, None)
        err.read = MagicMock(return_value=b"error code: 1010")
        body = json.dumps({"api_base": "https://api.commandcode.ai/provider/v1", "api_key": "k", "model": "m"})
        h = TestHandler(path="/api/providers/test", method="POST", engine=temp_db, body=body)
        with patch("urllib.request.urlopen", side_effect=err):
            h.do_POST()
        result = _json_body(h)
        assert result["ok"] is False
        assert result["cf_ray"] == "ray123"


class TestProfileBranches:
    def test_profile_create_invalid_body(self, temp_db):
        from tests.test_server import TestHandler
        h = TestHandler(path="/api/profiles", method="POST", engine=temp_db)
        h.rfile.read = MagicMock(side_effect=json.JSONDecodeError("bad", "doc", 0))
        h.headers["Content-Length"] = "5"
        h.do_POST()
        assert h.send_response.call_args[0][0] == 400

    def test_profile_update_missing_profile(self, temp_db):
        from tests.test_server import TestHandler
        h = TestHandler(path="/api/profiles/nope", method="PUT", engine=temp_db, body="{}")
        h.do_PUT()
        assert h.send_response.call_args[0][0] == 404

    def test_profile_delete_missing(self, temp_db):
        from tests.test_server import TestHandler
        h = TestHandler(path="/api/profiles/nope", method="DELETE", engine=temp_db)
        h.do_DELETE()
        assert h.send_response.call_args[0][0] == 404


class TestKeyManagerBranches:
    def test_key_detail_manager_none(self, temp_db):
        from tests.test_server import TestHandler
        with patch("src.server.endpoints.get_key_manager", return_value=None):
            h = TestHandler(path="/api/keys/1", engine=temp_db)
            h.do_GET()
        assert h.send_response.call_args[0][0] == 500

    def test_key_create_manager_none(self, temp_db):
        from tests.test_server import TestHandler
        with patch("src.server.endpoints.get_key_manager", return_value=None):
            h = TestHandler(path="/api/keys", method="POST", engine=temp_db, body="{}")
            h.do_POST()
        assert h.send_response.call_args[0][0] == 500

    def test_key_rotate_manager_none(self, temp_db):
        from tests.test_server import TestHandler
        with patch("src.server.endpoints.get_key_manager", return_value=None):
            h = TestHandler(path="/api/keys/1/rotate", method="POST", engine=temp_db)
            h.do_POST()
        assert h.send_response.call_args[0][0] == 500

    def test_key_delete_manager_none(self, temp_db):
        from tests.test_server import TestHandler
        with patch("src.server.endpoints.get_key_manager", return_value=None):
            h = TestHandler(path="/api/keys/1", method="DELETE", engine=temp_db)
            h.do_DELETE()
        assert h.send_response.call_args[0][0] == 500


class TestMetricsErrorBranch:
    def test_metrics_error_branch(self, temp_db):
        from tests.test_server import TestHandler
        with patch("src.server.endpoints.get_session", side_effect=RuntimeError("db down")):
            h = TestHandler(path="/metrics", engine=temp_db)
            h.do_GET()
        assert h.send_response.call_args[0][0] == 200  # metrics degrades to zeros
