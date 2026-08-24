"""Helper-function + error-branch coverage for src/server/endpoints.py.

Covers the module-level helpers (_fmt_params, _savings_for_model,
_browser_headers, _friendly_provider_error) and the health/metrics/export
error branches that the route-level tests don't reach.
"""

from unittest.mock import MagicMock, patch


from src.server.endpoints import (
    _fmt_params,
    _savings_for_model,
    _browser_headers,
    _friendly_provider_error,
)


# ── _fmt_params ──────────────────────────────────────────────────────────────

class TestFmtParams:
    def test_billions(self):
        assert _fmt_params(27_320_697_856) == "27.3B"

    def test_millions(self):
        assert _fmt_params(8_000_000) == "8.0M"

    def test_small(self):
        assert _fmt_params(1234) == "1234"


# ── _savings_for_model ───────────────────────────────────────────────────────

class TestSavingsForModel:
    def test_zero_hits_no_savings(self):
        cfg = MagicMock()
        assert _savings_for_model(cfg, "m", 0) == 0.0

    def test_computes_savings(self):
        cfg = MagicMock()
        cfg.providers = {"deepseek": {"models": ["deepseek-v4-pro"]}}
        cfg.get_pricing.return_value = {"cache_miss": 1.0, "cache_hit": 0.5}
        assert _savings_for_model(cfg, "deepseek-v4-pro", 1_000_000) == 0.5

    def test_pricing_error_returns_zero(self):
        cfg = MagicMock()
        cfg.providers = {"deepseek": {"models": ["m"]}}
        cfg.get_pricing.side_effect = RuntimeError("boom")
        assert _savings_for_model(cfg, "m", 1000) == 0.0

    def test_model_not_in_providers(self):
        cfg = MagicMock()
        cfg.providers = {"deepseek": {"models": ["other"]}}
        assert _savings_for_model(cfg, "m", 1000) == 0.0


# ── _browser_headers ─────────────────────────────────────────────────────────

class TestBrowserHeaders:
    def test_with_origin(self):
        h = _browser_headers("https://api.deepseek.com/v1")
        assert h["Origin"] == "https://api.deepseek.com"
        assert h["Referer"] == "https://api.deepseek.com/"
        assert h["User-Agent"].startswith("Mozilla/5.0")

    def test_no_origin(self):
        h = _browser_headers("")
        assert "Origin" not in h
        assert "Referer" not in h

    def test_extra_headers(self):
        h = _browser_headers("https://x/v1", {"X-Custom": "1"})
        assert h["X-Custom"] == "1"


# ── _friendly_provider_error ─────────────────────────────────────────────────

class TestFriendlyError:
    def test_unsupported_model(self):
        msg, code, etype = _friendly_provider_error(
            "commandcode", 400,
            '{"error": {"message": "Model must be called via /provider/v1/messages", "type": "invalid_request_error", "code": "unsupported_model"}}',
        )
        assert "Anthropic" in msg or "messages" in msg
        assert code == "unsupported_model"

    def test_model_not_in_plan(self):
        msg, code, etype = _friendly_provider_error(
            "commandcode", 403,
            '{"error": {"message": "not in plan", "type": "permission_error", "code": "MODEL_NOT_IN_PLAN"}}',
        )
        assert "not available on your current plan" in msg
        assert code == "MODEL_NOT_IN_PLAN"

    def test_auth_error(self):
        msg, code, etype = _friendly_provider_error(
            "commandcode", 401, '{"error": {"message": "invalid api key", "code": "authentication_error"}}'
        )
        assert "Authentication failed" in msg
        assert code == "authentication_error"

    def test_upgrade_required(self):
        msg, code, etype = _friendly_provider_error(
            "commandcode", 403, '{"error": {"message": "upgrade_required"}}'
        )
        assert "Provider plan or higher" in msg
        assert code == "upgrade_required"

    def test_rate_limit(self):
        msg, code, etype = _friendly_provider_error(
            "commandcode", 429, '{"error": {"message": "rate limit", "code": "rate_limit_error"}}'
        )
        assert "Rate limited" in msg
        assert code == "rate_limit_error"

    def test_plain_status_401(self):
        msg, code, etype = _friendly_provider_error("commandcode", 401, "raw body")
        assert code == "authentication_error"

    def test_plain_status_403(self):
        msg, code, etype = _friendly_provider_error("commandcode", 403, "raw body")
        assert code == "upgrade_required"

    def test_fallback_raw_body(self):
        msg, code, etype = _friendly_provider_error("commandcode", 500, "some raw error")
        assert msg == "some raw error"
        assert code == ""


# ── _uptime_for / _failure_breakdown exception branches ─────────────────────

class TestHealthHelpers:
    def test_uptime_for_exception_returns_100(self):
        from src.server.endpoints import HealthEndpoints
        e = HealthEndpoints()
        e.engine = MagicMock()
        with patch("src.server.endpoints.get_session", side_effect=RuntimeError("db down")):
            assert e._uptime_for("deepseek", "l2", 24) == 100.0

    def test_uptime_for_empty_returns_100(self, temp_db):
        from src.server.endpoints import HealthEndpoints
        e = HealthEndpoints()
        e.engine = temp_db
        assert e._uptime_for("deepseek", "l2", 24) == 100.0

    def test_failure_breakdown_exception_returns_empty(self):
        from src.server.endpoints import HealthEndpoints
        e = HealthEndpoints()
        e.engine = MagicMock()
        with patch("src.server.endpoints.get_session", side_effect=RuntimeError("db down")):
            assert e._failure_breakdown("deepseek", "l2", 24) == {}
