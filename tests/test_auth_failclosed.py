"""CWE-287: chat-completion auth must fail CLOSED when the key-manager
service is unresolved — never answer a completion unauthenticated.
"""

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
            "auth_required": True,
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


class TestAuthFailsClosed:
    def test_unresolved_key_manager_denies_instead_of_answering(self, temp_db):
        from tests.test_server import TestHandler, _status
        import src.server.handler as handler_mod
        real_resolve = handler_mod.resolve_service

        def fake_resolve(name, **kwargs):
            if name == "key_manager":
                return None  # the vuln-0003 state: service unresolved
            return real_resolve(name, **kwargs)

        body = json.dumps({"messages": [{"role": "user", "content": "hi"}]})
        h = TestHandler(path="/l2/chat/completions", method="POST", engine=temp_db, body=body)
        h.headers["Authorization"] = "Bearer some-key"
        with patch.object(handler_mod, "resolve_service", side_effect=fake_resolve):
            h.do_POST()
        # Fail CLOSED: 401, never a 200 completion.
        assert _status(h) == 401

    def test_valid_bearer_with_resolved_manager_uses_normal_invalid_key_path(self, temp_db):
        """With a working key manager, an unknown key hits the normal
        invalid-or-revoked 401 — the new else-branch isn't the only gate."""
        from tests.test_server import TestHandler, _status
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}]})
        h = TestHandler(path="/l2/chat/completions", method="POST", engine=temp_db, body=body)
        h.headers["Authorization"] = "Bearer bogus-key"
        h.do_POST()
        assert _status(h) == 401