"""In-process tests for LCPHandler — covers do_GET, do_POST, _serve_*.

Uses a handle-safe subclass that skips the BaseHTTPRequestHandler auto-handle
so we can call do_GET/do_POST directly with pytest-cov measuring coverage.
"""
import json
import os
import tempfile
import pytest
import sys

from unittest.mock import patch, MagicMock

from src.server import LCPHandler
from src.api.models import get_engine, Base


class _TestHandler(LCPHandler):
    """Subclass that skips auto-handle on init so we control invocation."""

    def __init__(self, path="/", method="GET", engine=None):
        # Skip socketserver.BaseRequestHandler.__init__ completely
        self.path = path
        self.command = method
        self.headers = {}
        self.request_version = "HTTP/1.1"
        self.requestline = f"{method} {path} HTTP/1.1"
        self.raw_requestline = f"{method} {path} HTTP/1.1".encode()
        self.client_address = ("127.0.0.1", 0)

        # Mock response methods
        self.send_response = MagicMock()
        self.send_header = MagicMock()
        self.end_headers = MagicMock()
        self.wfile = MagicMock()
        self.wfile.write = MagicMock()
        self.rfile = MagicMock()
        self.rfile.read = MagicMock(return_value=b"{}")

        # Set up handlers for stream chunks
        self._write_chunk = MagicMock()

        # Set up internal refs that methods expect
        self.engine = engine

        # Log error helper
        self.log_error = MagicMock()


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


def _get_written_bytes(handler):
    """Extract all bytes written to wfile."""
    written = []
    for call in handler.wfile.write.call_args_list:
        arg = call[0][0]
        if isinstance(arg, bytes):
            written.append(arg)
        elif isinstance(arg, str):
            written.append(arg.encode())
        elif isinstance(arg, (bytearray, memoryview)):
            written.append(bytes(arg))
    return b"".join(written)


def _json_body(handler):
    """Parse the last JSON body written to wfile."""
    raw = _get_written_bytes(handler)
    return json.loads(raw.decode("utf-8"))


# ═══════════════════════════════════════════════════════════════════════
# do_GET tests
# ═══════════════════════════════════════════════════════════════════════

class TestDoGet:
    def test_root_serves_dashboard(self, temp_db):
        LCPHandler.config = MagicMock()
        LCPHandler.config.pricing = []
        LCPHandler.config.providers = {}
        LCPHandler.config.profiles = {"l2": {"chain": [], "forbidden_tools": []}, "l1": {"chain": [], "forbidden_tools": []}}
        LCPHandler.config.circuit_breaker = {
            "failures_dead": 5, "dead_cooldown_seconds": 300,
            "failures_degraded": 3, "degraded_cooldown_seconds": 60,
        }
        LCPHandler.config.get_pricing = MagicMock(return_value=None)
        LCPHandler.config.get_profile = MagicMock(return_value={"chain": [], "forbidden_tools": []})
        LCPHandler.engine = temp_db

        h = _TestHandler("/", engine=temp_db)
        h.do_GET()
        assert h.send_response.called
        assert h.send_response.call_args[0][0] == 200
        combined = _get_written_bytes(h)
        assert b"LCP" in combined

    def test_health(self, temp_db):
        h = _TestHandler("/health", engine=temp_db)
        h.do_GET()
        assert h.send_response.called

    def test_models(self, temp_db):
        LCPHandler.config.profiles = {"l2": {"chain": []}, "l1": {"chain": []}}
        h = _TestHandler("/v1/models", engine=temp_db)
        h.do_GET()
        assert h.send_response.called
        combined = _get_written_bytes(h)
        assert b'"data"' in combined

    def test_cache_stats(self, temp_db):
        h = _TestHandler("/cache/stats", engine=temp_db)
        h.do_GET()
        assert h.send_response.called

    def test_metrics(self, temp_db):
        h = _TestHandler("/metrics", engine=temp_db)
        h.do_GET()
        assert h.send_response.called

    def test_export(self, temp_db):
        h = _TestHandler("/export", engine=temp_db)
        h.do_GET()
        assert h.send_response.called

    def test_export_with_limit(self, temp_db):
        h = _TestHandler("/export?limit=3", engine=temp_db)
        h.do_GET()
        assert h.send_response.called

    def test_errors(self, temp_db):
        h = _TestHandler("/errors", engine=temp_db)
        h.do_GET()
        assert h.send_response.called

    def test_404(self, temp_db):
        h = _TestHandler("/nonexistent", engine=temp_db)
        h.do_GET()
        assert h.send_response.call_args[0][0] == 404


# ═══════════════════════════════════════════════════════════════════════
# do_POST tests
# ═══════════════════════════════════════════════════════════════════════

class TestDoPost:
    def test_non_chat_path(self, temp_db):
        h = _TestHandler("/health", method="POST", engine=temp_db)
        h.do_POST()
        assert h.send_response.called
        assert h.send_response.call_args[0][0] == 404

    def test_unknown_profile(self, temp_db):
        LCPHandler.config.get_profile = MagicMock(return_value=None)
        h = _TestHandler("/no-such/chat/completions", method="POST", engine=temp_db)
        h.do_POST()
        assert h.send_response.called

    def test_read_body_error(self, temp_db):
        h = _TestHandler("/l2/chat/completions", method="POST", engine=temp_db)
        h.rfile.read = MagicMock(side_effect=Exception("IO error"))
        h.headers = {"Content-Length": "10"}
        h.do_POST()
        assert h.send_response.called

    def test_chat_flow(self, temp_db):
        """Test through do_POST with a minimal valid body."""
        body = {"messages": [{"role": "user", "content": "test"}]}
        body_bytes = json.dumps(body).encode()

        h = _TestHandler("/l2/chat/completions", method="POST", engine=temp_db)
        h.rfile.read = MagicMock(return_value=body_bytes)
        h.headers = {"Content-Length": str(len(body_bytes))}

        LCPHandler.config.get_profile = MagicMock(return_value={
            "chain": [{"provider": "test", "model": "test-model", "base_url": "http://t"}],
            "forbidden_tools": [],
        })
        LCPHandler.config.providers = {"test": {"base_url": "http://t"}}
        LCPHandler.config.get_pricing = MagicMock(return_value=None)

        h.do_POST()
        assert h.send_response.called


# ═══════════════════════════════════════════════════════════════════════
# Structured error response tests
# ═══════════════════════════════════════════════════════════════════════

class TestErrorResponses:
    """do_POST error paths return structured {code, message} payloads."""

    def _chat_handler(self, temp_db):
        body = {"messages": [{"role": "user", "content": "test"}]}
        body_bytes = json.dumps(body).encode()
        h = _TestHandler("/l2/chat/completions", method="POST", engine=temp_db)
        h.rfile.read = MagicMock(return_value=body_bytes)
        h.headers = {"Content-Length": str(len(body_bytes))}
        LCPHandler.config.get_profile = MagicMock(return_value={
            "chain": [{"provider": "test", "model": "test-model", "base_url": "http://t"}],
            "forbidden_tools": [],
            "auth_required": False,
        })
        LCPHandler.config.providers = {"test": {"base_url": "http://t"}}
        LCPHandler.config.get_pricing = MagicMock(return_value=None)
        return h

    def test_provider_bad_request_returns_structured_400(self, temp_db):
        from src.api.exceptions import ProviderBadRequestError
        h = self._chat_handler(temp_db)
        with patch("src.server.handler.try_chain",
                   side_effect=ProviderBadRequestError("model 'x' not found")):
            h.do_POST()
        assert h.send_response.call_args[0][0] == 400
        err = _json_body(h)["error"]
        assert err["code"] == "LCP-2004"
        assert "model 'x' not found" in err["message"]

    def test_bad_request_message_redacts_api_keys(self, temp_db):
        from src.api.exceptions import ProviderBadRequestError
        h = self._chat_handler(temp_db)
        with patch("src.server.handler.try_chain",
                   side_effect=ProviderBadRequestError(
                       "HTTP 400: sk-abcdef1234567890 is invalid")):
            h.do_POST()
        err = _json_body(h)["error"]
        assert err["code"] == "LCP-2004"
        assert "sk-abcdef1234567890" not in err["message"]
        assert "[REDACTED]" in err["message"]

    def test_all_providers_failed_does_not_leak_internals(self, temp_db):
        from src.api.exceptions import AllProvidersFailedError
        h = self._chat_handler(temp_db)
        with patch("src.server.handler.try_chain",
                   side_effect=AllProvidersFailedError(
                       "All providers failed for l2: secretco: HTTP 500: sk-abc123secret")):
            h.do_POST()
        assert h.send_response.call_args[0][0] == 502
        err = _json_body(h)["error"]
        assert err["code"] == "LCP-3001"
        assert "secretco" not in err["message"]
        assert "sk-abc123secret" not in err["message"]
        assert "providers failed" in err["message"]

    def test_provider_timeout_returns_504(self, temp_db):
        from src.api.exceptions import ProviderTimeoutError
        h = self._chat_handler(temp_db)
        with patch("src.server.handler.try_chain",
                   side_effect=ProviderTimeoutError("upstream unreachable")):
            h.do_POST()
        assert h.send_response.call_args[0][0] == 504
        err = _json_body(h)["error"]
        assert err["code"] == "LCP-2001"

    def test_unhandled_exception_returns_500_structured(self, temp_db):
        h = self._chat_handler(temp_db)
        with patch("src.server.handler.try_chain", side_effect=RuntimeError("boom")):
            h.do_POST()
        assert h.send_response.call_args[0][0] == 500
        err = _json_body(h)["error"]
        assert err["code"] == "LCP-5001"
        assert "internal error" in err["message"]

    def test_missing_messages_returns_400(self, temp_db):
        h = self._chat_handler(temp_db)
        h.rfile.read = MagicMock(return_value=b"{}")
        h.headers = {"Content-Length": "2"}
        h.do_POST()
        assert h.send_response.call_args[0][0] == 400
        # Client-input validation errors keep the flat format (not structured).
        assert "missing required field" in _json_body(h)["error"]

    def test_spend_limit_exceeded_returns_429(self, temp_db):
        from src.api.exceptions import CreditExhaustedError
        km = MagicMock()
        km.validate_key.return_value = {
            "id": 1,
            "allowed_profiles": None,
            "spend_limit": 10,
            "total_spend": 12,
        }
        with patch("src.server.handler.get_key_manager", return_value=km):
            h = self._chat_handler(temp_db)
            # Re-enable auth for this profile so the spend-limit path runs.
            LCPHandler.config.get_profile.return_value["auth_required"] = True
            h.headers["Authorization"] = "Bearer somekey123"
            h.do_POST()
        assert h.send_response.call_args[0][0] == 429
        err = _json_body(h)["error"]
        assert err["code"] == "LCP-1002"
        assert "spend limit" in err["message"]
