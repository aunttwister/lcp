"""Tests for provider health / failover / failure-breakdown endpoints.

Covers:
- GET /api/providers/health
- GET /api/providers/{name}/failures
- GET /api/providers/failovers
- POST /api/providers/{name}/toggle
"""
import json
import tempfile
import os
from unittest.mock import MagicMock

import pytest

from src.server import LCPHandler
from src.api.models import get_engine, Base, Request as RequestModel, FailoverEvent, get_session
from src.api.circuit_breaker import get_circuit_breaker


class _TestHandler(LCPHandler):
    """In-process handler that skips socketserver auto-handle."""

    def __init__(self, path="/", method="GET", engine=None):
        self.path = path
        self.command = method
        self.headers = {}
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
        self.rfile.read = MagicMock(return_value=b"{}")
        self._write_chunk = MagicMock()
        self.engine = engine
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


@pytest.fixture(autouse=True)
def _setup_handler_config():
    """Configure LCPHandler with a mock config + fresh circuit breaker."""
    from src.api import circuit_breaker as cb_module
    # Reset the singleton at setup AND recreate with our config so tests are
    # isolated regardless of conftest fixture ordering.
    cb_module._circuit_breaker = None
    cfg = MagicMock()
    cfg.circuit_breaker = {
        "failures_dead": 5, "dead_cooldown_seconds": 300,
        "failures_degraded": 3, "degraded_cooldown_seconds": 60,
    }
    cfg.profiles = {
        "l2": {
            "chain": [
                {"provider": "opencode", "model": "m", "base_url": "https://oc/v1"},
                {"provider": "deepseek", "model": "m", "base_url": "https://ds/v1"},
            ],
        }
    }
    cfg.providers = {"opencode": {"api_base": "https://oc/v1"},
                     "deepseek": {"api_base": "https://ds/v1"}}
    cfg.get_profile = MagicMock(return_value=cfg.profiles["l2"])
    get_circuit_breaker(cfg)
    LCPHandler.config = cfg
    LCPHandler.engine = None
    yield
    cb_module._circuit_breaker = None


def _written(handler):
    data = b""
    for call in handler.wfile.write.call_args_list:
        arg = call[0][0]
        data += arg.encode() if isinstance(arg, str) else bytes(arg)
    return data


def _json(handler):
    return json.loads(_written(handler))


class TestProvidersHealth:
    def test_empty_health(self, temp_db):
        h = _TestHandler("/api/providers/health", engine=temp_db)
        h.do_GET()
        assert h.send_response.called
        data = _json(h)
        assert data["summary"] == {"total": 0, "healthy": 0, "degraded": 0, "dead": 0}
        assert data["providers"] == {}

    def test_health_with_populated_circuit_breaker(self, temp_db):
        cb = get_circuit_breaker()
        cb.record_success("opencode", "https://oc/v1", "l2")
        cb.record_failure("deepseek", "https://ds/v1", "l2",
                          error_type="ProviderTimeoutError",
                          error_reason="HTTP 504")
        cb.record_failure("deepseek", "https://ds/v1", "l2",
                          error_type="ProviderTimeoutError")
        cb.record_failure("deepseek", "https://ds/v1", "l2",
                          error_type="ProviderTimeoutError")

        h = _TestHandler("/api/providers/health", engine=temp_db)
        h.do_GET()
        data = _json(h)
        assert data["summary"]["total"] == 2
        assert data["summary"]["healthy"] == 1
        assert data["summary"]["degraded"] == 1
        key = "opencode/l2"
        p = data["providers"][key]
        assert p["status"] == "healthy"
        assert p["base_url"] == "https://oc/v1"
        assert p["manual_override"] is None
        # Uptime defaults to 100 with no request rows
        assert p["uptime_24h"] == 100.0

    def test_uptime_computed_from_requests(self, temp_db):
        cb = get_circuit_breaker()
        cb.record_success("opencode", "https://oc/v1", "l2")
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with get_session(temp_db) as s:
            s.add(RequestModel(timestamp=now, profile="l2", model="m",
                               provider="opencode", success=1))
            s.add(RequestModel(timestamp=now, profile="l2", model="m",
                               provider="opencode", success=1))
            s.add(RequestModel(timestamp=now, profile="l2", model="m",
                               provider="opencode", success=0,
                               error_type="ProviderTimeoutError"))
            s.commit()

        h = _TestHandler("/api/providers/health", engine=temp_db)
        h.do_GET()
        data = _json(h)
        p = data["providers"]["opencode/l2"]
        # 2/3 success = 66.67%
        assert p["uptime_24h"] == pytest.approx(66.67, abs=0.01)
        assert p["failures_24h"].get("ProviderTimeoutError") == 1


class TestProviderFailures:
    def test_failures_empty(self, temp_db):
        h = _TestHandler("/api/providers/opencode/failures?window=24h", engine=temp_db)
        h.do_GET()
        data = _json(h)
        assert data["provider"] == "opencode"
        assert data["total"] == 0
        assert data["buckets"]["timeout"] == 0

    def test_failures_breakdown(self, temp_db):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with get_session(temp_db) as s:
            for et in ["ProviderTimeoutError", "ProviderTimeoutError",
                       "ProviderAuthError", "ProviderRateLimitError",
                       "ProviderInternalError", "ProviderBadRequestError"]:
                s.add(RequestModel(timestamp=now, profile="l2", model="m",
                                   provider="opencode", success=0, error_type=et))
            # Non-matching provider should be excluded
            s.add(RequestModel(timestamp=now, profile="l2", model="m",
                               provider="deepseek", success=0,
                               error_type="ProviderTimeoutError"))
            s.commit()

        h = _TestHandler("/api/providers/opencode/failures?window=24h", engine=temp_db)
        h.do_GET()
        data = _json(h)
        assert data["total"] == 6
        assert data["buckets"]["timeout"] == 2
        assert data["buckets"]["auth"] == 1
        assert data["buckets"]["rate_limit"] == 1
        assert data["buckets"]["internal_error"] == 1
        assert data["buckets"]["bad_request"] == 1
        assert data["buckets"]["other"] == 0

    def test_failures_profile_filter(self, temp_db):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with get_session(temp_db) as s:
            s.add(RequestModel(timestamp=now, profile="l2", model="m",
                               provider="opencode", success=0,
                               error_type="ProviderTimeoutError"))
            s.add(RequestModel(timestamp=now, profile="l1", model="m",
                               provider="opencode", success=0,
                               error_type="ProviderAuthError"))
            s.commit()

        h = _TestHandler("/api/providers/opencode/failures?window=24h&profile=l2",
                         engine=temp_db)
        h.do_GET()
        data = _json(h)
        assert data["total"] == 1
        assert data["buckets"]["timeout"] == 1
        assert data["buckets"]["auth"] == 0

    def test_failures_bucket_accepts_legacy_short_names(self, temp_db):
        """Legacy error_type strings (rate_limit, timeout, auth_error) group correctly."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with get_session(temp_db) as s:
            for et in ["timeout", "timeout", "rate_limit", "auth_error",
                       "internal_error", "bad_request"]:
                s.add(RequestModel(timestamp=now, profile="l2", model="m",
                                   provider="opencode", success=0, error_type=et))
            s.commit()

        h = _TestHandler("/api/providers/opencode/failures?window=24h", engine=temp_db)
        h.do_GET()
        data = _json(h)
        assert data["total"] == 6
        assert data["buckets"]["timeout"] == 2
        assert data["buckets"]["rate_limit"] == 1
        assert data["buckets"]["auth"] == 1
        assert data["buckets"]["internal_error"] == 1
        assert data["buckets"]["bad_request"] == 1
        assert data["buckets"]["other"] == 0


class TestFailoversEndpoint:
    def test_failovers_empty(self, temp_db):
        h = _TestHandler("/api/providers/failovers", engine=temp_db)
        h.do_GET()
        data = _json(h)
        assert data["failovers"] == []

    def test_failovers_listed_newest_first(self, temp_db):
        from datetime import datetime, timezone
        with get_session(temp_db) as s:
            for i in range(3):
                s.add(FailoverEvent(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    profile="l2",
                    from_provider="opencode",
                    to_provider="deepseek",
                    reason="ProviderTimeoutError",
                    error_message=f"timeout {i}",
                ))
            s.commit()

        h = _TestHandler("/api/providers/failovers?limit=10", engine=temp_db)
        h.do_GET()
        data = _json(h)
        assert len(data["failovers"]) == 3
        # Newest (highest id) first
        assert data["failovers"][0]["to_provider"] == "deepseek"
        assert data["failovers"][0]["reason"] == "ProviderTimeoutError"

    def test_failovers_filter_by_provider(self, temp_db):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with get_session(temp_db) as s:
            s.add(FailoverEvent(timestamp=now, profile="l2",
                                from_provider="opencode", to_provider="deepseek",
                                reason="ProviderTimeoutError"))
            s.add(FailoverEvent(timestamp=now, profile="l2",
                                from_provider="deepseek", to_provider="opencode",
                                reason="ProviderInternalError"))
            s.commit()

        h = _TestHandler("/api/providers/failovers?from=opencode", engine=temp_db)
        h.do_GET()
        data = _json(h)
        assert len(data["failovers"]) == 1
        assert data["failovers"][0]["from_provider"] == "opencode"


class TestProviderToggle:
    def _post(self, path, body, temp_db):
        body_bytes = json.dumps(body).encode()
        h = _TestHandler(path, method="POST", engine=temp_db)
        h.rfile.read = MagicMock(return_value=body_bytes)
        h.headers = {"Content-Length": str(len(body_bytes))}
        h.do_POST()
        return h

    def test_toggle_degrade(self, temp_db):
        h = self._post("/api/providers/opencode/toggle",
                       {"profile": "l2", "action": "degrade"}, temp_db)
        data = _json(h)
        assert data["ok"] is True
        assert data["status"] == "degraded"
        cb = get_circuit_breaker()
        assert cb.status_of("opencode", "https://oc/v1", "l2") == "degraded"

    def test_toggle_kill(self, temp_db):
        h = self._post("/api/providers/opencode/toggle",
                       {"profile": "l2", "action": "kill"}, temp_db)
        data = _json(h)
        assert data["ok"] is True
        assert data["status"] == "dead"
        cb = get_circuit_breaker()
        assert cb.is_available("opencode", "https://oc/v1", "l2") is False

    def test_toggle_resume(self, temp_db):
        cb = get_circuit_breaker()
        cb.force_status("opencode", "https://oc/v1", "l2", "kill")
        h = self._post("/api/providers/opencode/toggle",
                       {"profile": "l2", "action": "resume"}, temp_db)
        data = _json(h)
        assert data["ok"] is True
        assert data["status"] == "healthy"

    def test_toggle_missing_fields(self, temp_db):
        h = self._post("/api/providers/opencode/toggle", {"action": "kill"}, temp_db)
        assert h.send_response.call_args[0][0] == 400
        assert "profile" in _json(h)["error"]

    def test_toggle_invalid_action(self, temp_db):
        h = self._post("/api/providers/opencode/toggle",
                       {"profile": "l2", "action": "nonsense"}, temp_db)
        assert h.send_response.call_args[0][0] == 400
        assert "degrade" in _json(h)["error"]

    def test_toggle_provider_not_in_profile(self, temp_db):
        h = self._post("/api/providers/unknown/toggle",
                       {"profile": "l2", "action": "kill"}, temp_db)
        assert h.send_response.call_args[0][0] == 404
