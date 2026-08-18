"""Remaining handler.py coverage: do_PUT/do_DELETE fallbacks, invalid POST
routes, _sanitize_message, _resolve_pricing fallback, _serve_static edges."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.server import LCPHandler


class _TestHandler(LCPHandler):
    """Subclass that skips auto-handle on init so we control invocation."""

    def __init__(self, path="/", method="GET", engine=None, body=b"{}"):
        self.path = path
        self.command = method
        self.headers = {"Content-Length": str(len(body))} if body else {}
        self.request_version = "HTTP/1.1"
        self.requestline = f"{method} {path} HTTP/1.1"
        self.raw_requestline = f"{method} {path} HTTP/1.1".encode()
        self.client_address = ("127.0.0.1", 0)
        self.send_response = MagicMock()
        self.send_header = MagicMock()
        self.end_headers = MagicMock()
        self.wfile = MagicMock()
        self.wfile.write = MagicMock()
        self.rfile = MagicMock()
        self.rfile.read = MagicMock(return_value=body)
        self._write_chunk = MagicMock()
        self.engine = engine
        self.log_error = MagicMock()


def _status(handler):
    return handler.send_response.call_args[0][0] if handler.send_response.call_args else None


@pytest.fixture(autouse=True)
def _setup_config(temp_db):
    from src.api.key_manager import init_key_manager
    init_key_manager(temp_db, "data")
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


@pytest.fixture
def temp_db():
    import os
    import tempfile as _t
    from src.api.models import get_engine, Base
    fd, db_path = _t.mkstemp(suffix=".db")
    os.close(fd)
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()
    for ext in ["", "-wal", "-shm"]:
        try:
            os.unlink(db_path + ext)
        except FileNotFoundError:
            pass


class TestHandlerEdges:
    def test_put_unknown_route_404(self, temp_db):
        h = _TestHandler("/api/nope", method="PUT", engine=temp_db)
        h.do_PUT()
        assert _status(h) == 404

    def test_delete_unknown_route_404(self, temp_db):
        h = _TestHandler("/api/nope", method="DELETE", engine=temp_db)
        h.do_DELETE()
        assert _status(h) == 404

    def test_post_invalid_route_404(self, temp_db):
        h = _TestHandler("/api/nope", method="POST", engine=temp_db)
        h.do_POST()
        assert _status(h) == 404

    def test_put_provider_update(self, temp_db):
        LCPHandler.config.raw["providers"]["deepseek"] = {"api_base": "https://t/v1", "models": []}
        h = _TestHandler("/api/providers/deepseek", method="PUT", engine=temp_db,
                         body=json.dumps({"api_base": "https://new/v1"}).encode())
        h.do_PUT()
        assert _status(h) == 200

    def test_put_chain_reorder(self, temp_db):
        LCPHandler.config.raw["profiles"]["l2"]["chain"] = [
            {"provider": "deepseek", "model": "deepseek-v4-pro", "base_url": "https://old/v1"}
        ]
        h = _TestHandler("/api/chains/l2", method="PUT", engine=temp_db,
                         body=json.dumps({"chain": [{"provider": "deepseek", "model": "deepseek-v4-pro"}]}).encode())
        h.do_PUT()
        assert _status(h) == 200

    def test_put_profile_budget(self, temp_db):
        h = _TestHandler("/api/profiles/l2/budget", method="PUT", engine=temp_db,
                         body=json.dumps({"amount": 50.0}).encode())
        h.do_PUT()
        assert _status(h) == 200

    def test_delete_provider(self, temp_db):
        LCPHandler.config.providers["delco"] = {"api_base": "https://d/v1", "models": []}
        LCPHandler.config.raw["providers"]["delco"] = {"api_base": "https://d/v1", "models": []}
        h = _TestHandler("/api/providers/delco", method="DELETE", engine=temp_db)
        h.do_DELETE()
        assert _status(h) == 200

    def test_delete_key(self, temp_db):
        from src.api.key_manager import get_key_manager
        km = get_key_manager()
        created = km.create_key(name="ToDelete")
        h = _TestHandler(f"/api/keys/{created['id']}", method="DELETE", engine=temp_db)
        h.do_DELETE()
        assert _status(h) == 200

    def test_delete_profile(self, temp_db):
        LCPHandler.config.profiles["tempprof"] = {"chain": []}
        LCPHandler.config.raw["profiles"]["tempprof"] = {"chain": []}
        h = _TestHandler("/api/profiles/tempprof", method="DELETE", engine=temp_db)
        h.do_DELETE()
        assert _status(h) == 200

    def test_delete_budget(self, temp_db):
        from src.api.models import Budget, get_session
        with get_session(temp_db) as s:
            b = Budget(name="B", key_id=None, profile="l2", amount=10.0, period="monthly",
                       threshold_pct="80", action="log", status="active")
            s.add(b)
            s.commit()
            bid = b.id
        h = _TestHandler(f"/api/budgets/{bid}", method="DELETE", engine=temp_db)
        h.do_DELETE()
        assert _status(h) == 200


# ── _sanitize_message + _resolve_pricing ─────────────────────────────────────

class TestHandlerHelpers:
    def test_sanitize_message_truncates(self):
        from src.server.handler import _sanitize_message
        long = "x" * 500
        out = _sanitize_message(long)
        assert len(out) <= 304  # 300 + "..."
        assert out.endswith("...")

    def test_sanitize_message_redacts(self):
        from src.server.handler import _sanitize_message
        out = _sanitize_message("key sk-abc123def leaked")
        assert "sk-abc123def" not in out

    def test_resolve_pricing_config_ok(self):
        from src.server.handler import _resolve_pricing
        cfg = MagicMock()
        cfg.get_pricing.return_value = {"cache_miss": 1.0}
        assert _resolve_pricing(cfg, "p", "m") == {"cache_miss": 1.0}

    def test_resolve_pricing_plugin_fallback(self):
        from src.server.handler import _resolve_pricing
        cfg = MagicMock()
        cfg.get_pricing.side_effect = RuntimeError("no pricing")
        reg = MagicMock()
        reg.get_pricing.return_value = {"cache_miss": 2.0}
        with patch("src.api.cost_plugins.get_registry", return_value=reg):
            assert _resolve_pricing(cfg, "commandcode", "m") == {"cache_miss": 2.0}

    def test_resolve_pricing_none(self):
        from src.server.handler import _resolve_pricing
        cfg = MagicMock()
        cfg.get_pricing.side_effect = RuntimeError("no pricing")
        reg = MagicMock()
        reg.get_pricing.return_value = None
        with patch("src.api.cost_plugins.get_registry", return_value=reg):
            assert _resolve_pricing(cfg, "p", "m") is None


# ── _serve_static edges ──────────────────────────────────────────────────────

class TestStaticEdges:
    def test_static_forbidden_path_traversal(self, temp_db):
        h = _TestHandler("/static/../secret", engine=temp_db)
        h.do_GET()
        assert _status(h) == 403

    def test_static_missing_file(self, temp_db):
        h = _TestHandler("/static/nonexistent.css", engine=temp_db)
        h.do_GET()
        assert _status(h) == 404

    def test_static_serves_css(self, temp_db):
        h = _TestHandler("/static/dashboard.css", engine=temp_db)
        h.do_GET()
        assert _status(h) == 200
        # CSS bytes were written.
        written = b"".join(c[0][0] for c in h.wfile.write.call_args_list if isinstance(c[0][0], bytes))
        assert b".models-layout" in written or b"body" in written or b"--" in written
