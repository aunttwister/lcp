"""Batch F3 — src/server/handler.py final coverage gaps."""
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.server import LCPHandler
from src.api.models import get_engine, Base


class _TH(LCPHandler):
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
        if isinstance(body, dict):
            body = json.dumps(body)
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
def temp_db():
    fd, db_path = tempfile.mkstemp(suffix=".db")
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


def _chat_cfg():
    cfg = MagicMock()
    cfg.profiles = {"l2": {"chain": [{"provider": "test", "model": "test-model",
                                      "base_url": "http://t"}],
                           "forbidden_tools": [], "auth_required": False}}
    cfg.providers = {"test": {"base_url": "http://t"}}
    cfg.get_profile = lambda name: cfg.profiles.get(name)
    cfg.get_pricing = MagicMock(return_value={"cache_hit": 0.01,
                                              "cache_miss": 0.5, "output": 1.0})
    LCPHandler.config = cfg
    return cfg


def _chat_handler(temp_db, stream=False):
    # Unique content → the process-wide prompt-cache singleton never hits.
    body = {"messages": [{"role": "user",
                          "content": f"probe-{os.urandom(8).hex()}"}]}
    if stream:
        body["stream"] = True
    bb = json.dumps(body).encode()
    h = _TH("/l2/chat/completions", method="POST", engine=temp_db)
    h.rfile.read = MagicMock(return_value=bb)
    h.headers = {"Content-Length": str(len(bb))}
    _chat_cfg()
    return h


class TestSanitizeAndPricing:
    def test_sanitize_message_non_string(self):
        from src.server.handler import _sanitize_message
        out = _sanitize_message(ValueError("boom"))     # 69
        assert "boom" in out

    def test_resolve_pricing_both_fail(self):
        from src.server.handler import _resolve_pricing
        cfg = MagicMock()
        cfg.get_pricing.side_effect = RuntimeError("no yaml")
        reg = MagicMock()
        reg.get_pricing.side_effect = RuntimeError("no plugin")   # 95-96
        with patch("src.server.handler.resolve_service", return_value=reg):
            assert _resolve_pricing(cfg, "p", "m") is None


class TestSendPaths:
    def test_send_json_broken_pipe(self, temp_db):
        h = _TH("/health", engine=temp_db)
        h.wfile.write = MagicMock(side_effect=BrokenPipeError())
        h._send_json({"a": 1})                            # 140-141

    def test_send_error_custom_message(self, temp_db):
        from src.api.exceptions import ProviderTimeoutError
        h = _TH("/health", engine=temp_db)
        exc = ProviderTimeoutError("timed out")
        h._send_error(exc, message="masked {exc} text")   # 157
        body = _json_body(h)
        assert "masked" in body["error"]["message"]

    def test_log_message_suppressed(self, temp_db):
        h = _TH("/", engine=temp_db)
        assert h.log_message("x", 1) is None              # 129

    def test_static_traversal_forbidden(self, temp_db):
        h = _TH("/static/../../../etc/passwd", engine=temp_db)
        h.path = "/static/../../../etc/passwd"
        h._serve_static()                                 # 196-197
        assert h.send_response.call_args[0][0] == 403

    def test_static_broken_pipe(self, temp_db):
        h = _TH("/static/dashboard.css", engine=temp_db)
        h.path = "/static/dashboard.css"
        h.wfile.write = MagicMock(side_effect=ConnectionResetError())
        h._serve_static()                                 # 219-220


class TestDoGetRoutes:
    def test_dashboard_no_profile(self, temp_db):
        LCPHandler.config = MagicMock()
        LCPHandler.config.pricing = []
        LCPHandler.config.providers = {}
        LCPHandler.config.profiles = {"l2": {"chain": []}}
        LCPHandler.config.get_pricing = MagicMock(return_value=None)
        LCPHandler.config.get_profile = MagicMock(return_value=None)
        h = _TH("/xyz/dashboard", engine=temp_db)
        with patch("src.server.handler.LCPHandler._resolve_profile",
                   return_value=None):
            h.do_GET()                                    # 246
        assert h.send_response.called

    def test_benchmark_log_deep_path_404(self, temp_db):
        h = _TH("/api/models/benchmark/1/2/log", engine=temp_db)
        h.do_GET()                                        # 359
        assert h.send_response.call_args[0][0] == 404


class TestDoPostRoutes:
    def test_alert_acknowledge_route(self, temp_db):
        h = _TH("/api/alerts/7/acknowledge", method="POST", engine=temp_db)
        with patch("src.server.handler.LCPHandler._serve_alert_acknowledge") as ack:
            h.do_POST()                                   # 402-404
        ack.assert_called_once_with("7")

    def test_budget_create_route(self, temp_db):
        h = _TH("/api/budgets", method="POST", engine=temp_db)
        with patch("src.server.handler.LCPHandler._serve_budget_create") as bc:
            h.do_POST()                                   # 406-407
        bc.assert_called_once()


class TestChatFlowEdges:
    def test_estimation_crash_falls_back(self, temp_db):
        h = _chat_handler(temp_db)
        result = {"id": "x", "choices": [], "usage": {}}
        with patch("src.server.handler.estimate_from_request",
                   side_effect=RuntimeError("est boom")), \
             patch("src.server.handler.try_chain",
                   return_value=(result, 200, "test", "test-model")):
            h.do_POST()                                   # 582-583
        assert h.send_response.call_args[0][0] == 200

    def test_cache_hit_broken_pipe(self, temp_db):
        h = _chat_handler(temp_db)
        cache = MagicMock()
        cache.get.return_value = {"id": "cached"}
        h.wfile.write = MagicMock(side_effect=ConnectionAbortedError())
        with patch("src.server.handler.resolve_service",
                   side_effect=lambda k, fallback=None: cache if k == "prompt_cache" else None):
            h.do_POST()                                   # 605-606
        # cache hit path returns after logging the disconnect

    def test_sse_capture_crash_swallowed(self, temp_db):
        h = _chat_handler(temp_db, stream=True)

        def chunks():
            yield b"data: {}\n\n"
        with patch("src.server.handler.try_chain",
                   return_value=(chunks(), 200, "test", "test-model")), \
             patch("src.server.handler.capture_reasoning_from_sse",
                   side_effect=RuntimeError("cap boom")):
            h.do_POST()                                   # 662-663
        assert h.send_response.called

    def test_stream_budget_track_crash_swallowed(self, temp_db):
        h = _chat_handler(temp_db, stream=True)

        def chunks():
            yield (b'data: {"choices": [], "usage": {"prompt_tokens": 1, '
                   b'"completion_tokens": 1, "total_tokens": 2}}\n\n')
        with patch("src.server.handler.try_chain",
                   return_value=(chunks(), 200, "test", "test-model")), \
             patch("src.server.handler.extract_last_sse_chunk",
                   return_value={"usage": {"prompt_tokens": 1,
                                           "completion_tokens": 1}}), \
             patch("src.server.handler.get_token_verifier",
                   return_value=None), \
             patch("src.server.handler.resolve_service", return_value=None), \
             patch("src.server.handler.LCPHandler._track_budget_spend",
                   side_effect=RuntimeError("track boom")):
            h.do_POST()                                   # 695-696
        assert h.send_response.called

    def test_nonstream_capture_crash_swallowed(self, temp_db):
        h = _chat_handler(temp_db)
        response = {"id": "x", "choices": [],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        with patch("src.server.handler.try_chain",
                   return_value=(response, 200, "test", "test-model")), \
             patch("src.server.handler.capture_reasoning_from_response",
                   side_effect=RuntimeError("cap boom")):
            h.do_POST()                                   # 717-718
        assert h.send_response.call_args[0][0] == 200

    def test_nonstream_budget_track_crash_swallowed(self, temp_db):
        h = _chat_handler(temp_db)
        response = {"id": "x", "choices": [],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        with patch("src.server.handler.try_chain",
                   return_value=(response, 200, "test", "test-model")), \
             patch("src.server.handler.LCPHandler._track_budget_spend",
                   side_effect=RuntimeError("track boom")):
            h.do_POST()                                   # 743-744
        assert h.send_response.call_args[0][0] == 200

    def test_nonstream_write_broken_pipe(self, temp_db):
        h = _chat_handler(temp_db)
        response = {"id": "x", "choices": [],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        h.wfile.write = MagicMock(side_effect=BrokenPipeError())
        with patch("src.server.handler.try_chain",
                   return_value=(response, 200, "test", "test-model")):
            h.do_POST()                                   # 756-757
        assert h.send_response.call_args[0][0] == 200

    def test_tool_blocked_error_path(self, temp_db):
        from src.api.exceptions import ToolBlockedError
        h = _chat_handler(temp_db)
        with patch("src.server.handler.strip_forbidden_tools",
                   side_effect=ToolBlockedError("tool 'write_file' blocked")):
            h.do_POST()                                   # 773-774
        assert h.send_response.call_args[0][0] == 403

    def test_lcp_error_generic_path(self, temp_db):
        from src.api.exceptions import AuthError
        h = _chat_handler(temp_db)
        # AuthError raised INSIDE the try block (not the pre-auth one) —
        # trigger via try_chain so the LCPError except-arm runs.
        with patch("src.server.handler.try_chain",
                   side_effect=AuthError("nope")):
            h.do_POST()                                   # 801-802
        assert h.send_response.call_args[0][0] == 401
