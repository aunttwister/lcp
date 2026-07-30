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
        assert b"smallm gateway" in combined

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
