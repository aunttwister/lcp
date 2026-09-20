"""CWE-200: the unauthenticated /health payload must not disclose provider
base_url or last_failure_reason (the private provider endpoint map that Strix
captured via this route on 09-10).
"""

import json
from unittest.mock import MagicMock, patch

from src.server.endpoints import HealthEndpoints


class TestHealthRedaction:
    def test_health_redacts_base_url_and_failure_reason(self):
        h = MagicMock()
        h.config.profiles = {"l1": {}, "l2": {}}
        cb = MagicMock()
        cb.get_all_health.return_value = {
            ("local-zgx", "http://192.168.1.107:18300/v1", "l1"): {
                "status": "healthy",
                "consecutive_failures": 0,
                "last_success": "2026-09-20T07:00:00+00:00",
                "last_failure": None,
                "last_failure_reason": "Provider local-zgx unreachable: [Errno -3]",
                "tripped_until": None,
            },
            ("deepseek", "https://api.deepseek.com/v1", "l2"): {
                "status": "degraded",
                "consecutive_failures": 2,
                "last_success": "2026-09-20T06:00:00+00:00",
                "last_failure": "2026-09-20T06:30:00+00:00",
                "last_failure_reason": "chain timeout",
                "tripped_until": 1779999999,
            },
        }
        with patch("src.server.endpoints.resolve_service", return_value=cb):
            HealthEndpoints._serve_health(h)
        payload = h._send_json.call_args[0][0]
        assert isinstance(payload, dict)
        assert payload["status"] == "ok"
        serialized = json.dumps(payload)
        assert "192.168.1.107" not in serialized
        assert "18300" not in serialized
        assert "api.deepseek.com" not in serialized
        entry = payload["providers"]["local-zgx/l1"]
        assert entry["status"] == "healthy"
        assert entry["failures"] == 0
        assert "base_url" not in entry
        assert "last_failure_reason" not in entry
        # Health consumers (badges, uptime) still get everything they need.
        assert "last_success" in entry
        assert "tripped_until" in entry
        assert "tripped_until" in payload["providers"]["deepseek/l2"]

    def test_health_keeps_shape_when_no_health_records(self):
        h = MagicMock()
        h.config.profiles = {"l1": {}}
        cb = MagicMock()
        cb.get_all_health.return_value = {}
        with patch("src.server.endpoints.resolve_service", return_value=cb):
            HealthEndpoints._serve_health(h)
        payload = h._send_json.call_args[0][0]
        assert payload["providers"] == {}
        assert payload["profiles"] == ["l1"]