"""Tests for the memory HTTP endpoints (/{profile}/memory/...)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.server import LCPHandler
from src.api.memory.lancedb_backend import LanceDBMemoryBackend

VOCAB = {
    "gpu": 0, "rtx": 1, "3090": 2, "tesla": 3, "p40": 4, "hardware": 5,
    "basement": 6, "location": 7, "wifi": 8, "password": 9, "node01": 10,
    "server": 11, "rack": 12,
}


def fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * 32
        for w in t.lower().split():
            idx = VOCAB.get(w.strip(".,:;"))
            if idx is not None:
                v[idx % 32] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / norm for x in v])
    return out


class TestHandler(LCPHandler):
    """Subclass that skips BaseHTTPRequestHandler.__init__ for direct testing."""

    def __init__(self, path="/", method="GET", engine=None, headers=None, body=None):
        self.path = path
        self.command = method
        self.headers = headers or {}
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
        body_bytes = (body or b"{}") if isinstance(body or b"{}", bytes) else (body or "{}").encode()
        self.rfile.read = MagicMock(return_value=body_bytes)
        if body:
            self.headers["Content-Length"] = str(len(body_bytes))
        self._write_chunk = MagicMock()
        self.engine = engine
        self.log_error = MagicMock()


def _json_body(handler):
    for call in handler.wfile.write.call_args_list:
        try:
            return json.loads(call[0][0])
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return {}


@pytest.fixture
def handler(monkeypatch, tmp_path):
    """A TestHandler wired to a real LanceDB backend (auth disabled)."""
    backend = LanceDBMemoryBackend(str(tmp_path / "mem"), fake_embed, dim=32)
    monkeypatch.setattr("src.api.memory.get_memory", lambda: backend)
    monkeypatch.setattr("src.api.key_manager.get_key_manager", lambda: None)

    class _Cfg:
        profiles = {"l2": {"auth_required": False}, "career": {"auth_required": False}}

        def get_profile(self, name):
            return self.profiles.get(name)

        def check_reload(self):
            return False

    _Cfg.engine = MagicMock()

    h = TestHandler(path="/l2/memory/retain", method="POST", body="{}")
    h.config = _Cfg()
    return h


class TestMemoryEndpoints:
    def test_retain_roundtrip(self, handler, tmp_path):
        body = json.dumps({"content": "node01 has an RTX 3090 GPU", "tags": ["gpu"]})
        h = TestHandler(path="/l2/memory/retain", method="POST", engine=MagicMock(), body=body)
        h.config = handler.config
        h.do_POST()
        assert h.send_response.call_args[0][0] == 200
        mid = _json_body(h).get("memory_id")
        assert mid

        # recall returns it
        h2 = TestHandler(path="/l2/memory/recall", method="POST", engine=MagicMock(),
                         body=json.dumps({"query": "which gpu is in node01"}))
        h2.config = handler.config
        h2.do_POST()
        body2 = _json_body(h2)
        assert body2["results"]
        assert body2["results"][0]["content"] == "node01 has an RTX 3090 GPU"

    def test_count(self, handler, tmp_path):
        backend = handler._memory_backend_or_501()
        backend.retain("a", profile="l2")
        backend.retain("b", profile="l2")
        h = TestHandler(path="/l2/memory/count", method="GET", engine=MagicMock())
        h.config = handler.config
        h.do_GET()
        assert _json_body(h) == {"count": 2}

    def test_forget(self, handler, tmp_path):
        backend = handler._memory_backend_or_501()
        mid = backend.retain("node01 has a Tesla P40", profile="l2")
        h = TestHandler(path="/l2/memory/forget", method="POST", engine=MagicMock(),
                        body=json.dumps({"memory_id": mid}))
        h.config = handler.config
        h.do_POST()
        assert _json_body(h) == {"deleted": True}
        assert backend.count("l2") == 0

    def test_missing_module_returns_501(self, handler, monkeypatch):
        monkeypatch.setattr("src.api.memory.get_memory", lambda: None)
        h = TestHandler(path="/l2/memory/retain", method="POST",
                        body=json.dumps({"content": "x"}))
        h.config = handler.config
        h.do_POST()
        assert h.send_response.call_args[0][0] == 501
        assert _json_body(h)["error"]["code"] == "LCP-5010"

    def test_unknown_action_404(self, handler):
        h = TestHandler(path="/l2/memory/frobnicate", method="POST", body="{}")
        h.config = handler.config
        h.do_POST()
        assert h.send_response.call_args[0][0] == 404

    def test_retain_missing_content_400(self, handler):
        h = TestHandler(path="/l2/memory/retain", method="POST", body=json.dumps({}))
        h.config = handler.config
        h.do_POST()
        assert h.send_response.call_args[0][0] == 400

    def test_per_profile_isolation(self, handler, tmp_path):
        backend = handler._memory_backend_or_501()
        backend.retain("wifi password", profile="l2")
        h = TestHandler(path="/career/memory/count", method="GET", engine=MagicMock())
        h.config = handler.config
        h.do_GET()
        assert _json_body(h) == {"count": 0}

    def test_auth_required(self, handler):
        handler.config.profiles["l2"]["auth_required"] = True
        h = TestHandler(path="/l2/memory/retain", method="POST",
                        body=json.dumps({"content": "x"}))
        h.config = handler.config
        h.do_POST()
        assert h.send_response.call_args[0][0] == 401

    def test_auth_valid_key(self, handler):
        handler.config.profiles["l2"]["auth_required"] = True
        fake_km = MagicMock()
        fake_km.validate_key.return_value = {"id": 1, "allowed_profiles": "l2"}
        with patch("src.api.key_manager.get_key_manager", return_value=fake_km):
            h = TestHandler(path="/l2/memory/count", method="GET",
                            headers={"Authorization": "Bearer sk-test"})
            h.config = handler.config
            h.do_GET()
        assert h.send_response.call_args[0][0] == 200

    def test_auth_wrong_profile_forbidden(self, handler):
        handler.config.profiles["l2"]["auth_required"] = True
        fake_km = MagicMock()
        fake_km.validate_key.return_value = {"id": 1, "allowed_profiles": "career"}
        with patch("src.api.key_manager.get_key_manager", return_value=fake_km):
            h = TestHandler(path="/l2/memory/count", method="GET",
                            headers={"Authorization": "Bearer sk-test"})
            h.config = handler.config
            h.do_GET()
        assert h.send_response.call_args[0][0] == 403


class TestMemoryRouteDispatch:
    def test_get_dispatch_count(self, handler, monkeypatch):
        called = {}
        def spy(self, profile, action):
            called["profile"] = profile
            called["action"] = action
        monkeypatch.setattr(handler.__class__, "_serve_memory_api", spy)
        h = TestHandler(path="/l2/memory/count", method="GET", engine=MagicMock())
        h.config = handler.config
        h.do_GET()
        assert called == {"profile": "l2", "action": "count"}

    def test_post_dispatch_retain(self, handler, monkeypatch):
        called = {}
        def spy(self, profile, action):
            called["profile"] = profile
            called["action"] = action
        monkeypatch.setattr(handler.__class__, "_serve_memory_api", spy)
        h = TestHandler(path="/l2/memory/retain", method="POST",
                        body=json.dumps({"content": "x"}))
        h.config = handler.config
        h.do_POST()
        assert called == {"profile": "l2", "action": "retain"}

    def test_non_memory_path_unaffected(self, handler, monkeypatch):
        # /l2/chat/completions must NOT route to memory
        called = {}
        def spy(self, profile, action):
            called["profile"] = profile
            called["action"] = action
        monkeypatch.setattr(handler.__class__, "_serve_memory_api", spy)
        h = TestHandler(path="/l2/chat/completions", method="POST",
                        body=json.dumps({"messages": [{"role": "user", "content": "hi"}]}))
        h.config = handler.config
        # This would 400 on missing messages etc. — we only assert memory wasn't called.
        try:
            h.do_POST()
        except Exception:
            pass
        assert called == {}
