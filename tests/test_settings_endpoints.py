"""Tests for the admin Settings API + cache-read cost-plugin endpoints."""

import json
from unittest.mock import MagicMock

import pytest

from src.api.cost_cache import (
    _reset_singletons,
    get_cost_cache,
    get_refresher,
    get_settings,
    init_cost_cache,
    init_refresher,
    init_settings,
)
from src.api.models import Base, get_engine
from src.server import LCPHandler


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


def _status(handler):
    return handler.send_response.call_args[0][0] if handler.send_response.call_args else None


def _json_body(handler):
    return json.loads(handler.wfile.write.call_args[0][0].decode("utf-8"))


@pytest.fixture
def engine(tmp_path):
    db_path = str(tmp_path / "test.db")
    e = get_engine(db_path)
    Base.metadata.create_all(e)
    return e


@pytest.fixture(autouse=True)
def _cleanup_singletons():
    yield
    _reset_singletons()


class TestSettingsApi:
    def test_settings_page_renders(self, engine, mock_config):
        init_settings(engine)
        h = TestHandler(path="/settings", engine=engine)
        h.config = mock_config
        h.do_GET()
        assert _status(h) == 200
        html = h.wfile.write.call_args[0][0].decode("utf-8")
        assert "Cost data refresh" in html
        assert "Cache" in html

    def test_settings_api_default(self, engine):
        init_settings(engine)
        init_cost_cache(engine)
        h = TestHandler(path="/api/settings", engine=engine)
        h.do_GET()
        body = _json_body(h)
        assert body["ttl_minutes"] == 30
        assert body["entries"] == []

    def test_settings_update_persists(self, engine, mock_config):
        init_settings(engine)
        h = TestHandler(path="/api/settings", method="POST", engine=engine,
                        body=json.dumps({"ttl_minutes": 7}))
        h.config = mock_config
        h.do_POST()
        assert _status(h) == 200
        assert _json_body(h)["ttl_minutes"] == 7
        # Persisted + readable via a fresh store.
        assert get_settings().get_ttl_minutes() == 7

    def test_settings_update_rejects_bad_ttl(self, engine, mock_config):
        init_settings(engine)
        h = TestHandler(path="/api/settings", method="POST", engine=engine,
                        body=json.dumps({"ttl_minutes": "nope"}))
        h.config = mock_config
        h.do_POST()
        assert _status(h) == 400

    def test_settings_refresh_and_clear(self, engine, mock_config):
        init_settings(engine)
        init_cost_cache(engine)
        init_refresher(get_cost_cache(), get_settings())
        h = TestHandler(path="/api/settings/cache/refresh", method="POST", engine=engine, body="{}")
        h.config = mock_config
        h.do_POST()
        assert _status(h) == 200
        assert _json_body(h)["refreshing"] is True

        get_cost_cache().set("opencode", "subscription", {"a": 1})
        h = TestHandler(path="/api/settings/cache/clear", method="POST", engine=engine, body="{}")
        h.config = mock_config
        h.do_POST()
        assert _status(h) == 200
        assert get_cost_cache().entries() == []


class TestPluginEndpointsCacheRead:
    def test_subscriptions_served_from_cache_without_fetch(self, engine, monkeypatch):
        init_settings(engine)
        cache = init_cost_cache(engine)
        cache.set("opencode", "subscription", {"monthly_pct": 42.0})

        from src.api.cost_plugins import get_registry
        # If the endpoint tried to scrape, this would blow up.
        monkeypatch.setattr(
            get_registry(), "fetch_all_subscriptions",
            lambda: (_ for _ in ()).throw(AssertionError("live fetch called!")),
        )

        h = TestHandler(path="/api/cost-plugins/subscriptions", engine=engine)
        h.do_GET()
        body = _json_body(h)
        assert body["plugin_subscriptions"]["opencode"] == {"monthly_pct": 42.0, "fetched_at": cache.get("opencode", "subscription")["fetched_at"]}

    def test_subscriptions_stale_flag(self, engine):
        init_settings(engine)
        cache = init_cost_cache(engine)
        cache.set("opencode", "subscription", {"monthly_pct": 42.0}, stale_error="boom")

        h = TestHandler(path="/api/cost-plugins/subscriptions", engine=engine)
        h.do_GET()
        body = _json_body(h)
        sub = body["plugin_subscriptions"]["opencode"]
        assert sub["monthly_pct"] == 42.0
        assert sub["_stale"] is True
        assert sub["_stale_error"] == "boom"

    def test_subscriptions_unsupported_provider_is_none(self, engine):
        init_settings(engine)
        init_cost_cache(engine)
        h = TestHandler(path="/api/cost-plugins/subscriptions", engine=engine)
        h.do_GET()
        body = _json_body(h)
        # llamacpp does not support subscriptions → None (matches old shape).
        assert body["plugin_subscriptions"]["llamacpp"] is None

    def test_balances_served_from_cache(self, engine):
        init_settings(engine)
        cache = init_cost_cache(engine)
        cache.set("deepseek", "balance", {"balance": 11.5, "currency": "USD"})
        h = TestHandler(path="/api/cost-plugins/balances", engine=engine)
        h.do_GET()
        body = _json_body(h)
        assert body["plugin_balances"]["deepseek"]["balance"] == 11.5

    def test_cookie_set_invalidates_cache_and_requests_refresh(self, engine, monkeypatch, mock_config):
        init_settings(engine)
        cache = init_cost_cache(engine)
        init_refresher(cache, get_settings())
        cache.set("opencode", "subscription", {"monthly_pct": 42.0})

        # Stub the credential store so the handler has one to write to.
        from src.api import credential_store as cred_mod
        fake_store = MagicMock()
        fake_store.has_cookie.return_value = True
        monkeypatch.setattr(cred_mod, "get_credential_store", lambda *a, **k: fake_store)

        h = TestHandler(path="/api/cost-plugins/cookie/opencode", method="POST",
                        engine=engine, body=json.dumps({"cookie": "auth=xyz"}))
        h.config = mock_config
        h.do_POST()
        assert _status(h) == 200
        # Cache for opencode was invalidated; the refresh was requested.
        assert cache.get("opencode", "subscription") is None
