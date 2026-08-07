"""Tests for request_pipeline helper functions.

Covers the pure message/tool/cost helpers that were previously untested:
  - normalize_messages_for_cache / normalize_tools_for_cache
  - has_image_content
  - sanitize_messages (dangling/orphaned tool calls)
  - compute_cache_savings / read_cache_hit_tokens
  - calculate_cost (plugin + config-pricing paths)
  - forward_request streaming + HTTPError branches
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_db():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from src.api.models import get_engine, Base
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()
    for ext in ["", "-wal", "-shm"]:
        try:
            os.unlink(db_path + ext)
        except FileNotFoundError:
            pass

from src.api.request_pipeline import (
    calculate_cost,
    compute_cache_savings,
    forward_request,
    has_image_content,
    normalize_messages_for_cache,
    normalize_tools_for_cache,
    read_cache_hit_tokens,
    record_cost,
    sanitize_messages,
    strip_forbidden_tools,
    try_chain,
)
from src.api.exceptions import (
    AllProvidersFailedError,
    ConfigError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderInternalError,
)


@pytest.fixture(autouse=True)
def _fresh_circuit_breaker():
    """Reset the circuit-breaker singleton between tests."""
    import src.api.circuit_breaker as cb_module
    cb_module._circuit_breaker = None
    yield
    cb_module._circuit_breaker = None


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.pricing = []
    return cfg


@pytest.fixture
def chain_config(mock_config):
    """A config with a 2-step chain and providers, ready for try_chain."""
    from src.api.circuit_breaker import get_circuit_breaker
    mock_config.circuit_breaker = {
        "failures_dead": 5, "dead_cooldown_seconds": 300,
        "failures_degraded": 3, "degraded_cooldown_seconds": 60,
    }
    mock_config.profiles = {"l2": {"chain": [
        {"provider": "first", "base_url": "https://first.com/v1", "model": "m1"},
        {"provider": "second", "base_url": "https://second.com/v1", "model": "m2"},
    ], "forbidden_tools": []}}
    mock_config.providers = {
        "first": {"api_key_env": "KEY", "base_url": "https://first.com/v1"},
        "second": {"api_key_env": "KEY", "base_url": "https://second.com/v1"},
    }
    mock_config.model_limits = {}
    mock_config.get_provider_key.return_value = None
    # try_chain() calls get_circuit_breaker() with no args, so pre-initialize it
    get_circuit_breaker(mock_config)
    return mock_config


# ── normalize_messages_for_cache ─────────────────────────────────────────

class TestNormalizeMessagesForCache:
    def test_system_whitespace_stripped(self):
        out = normalize_messages_for_cache([
            {"role": "system", "content": "  be terse  \n"},
        ])
        assert out[0]["content"] == "  be terse"

    def test_tool_message_preserves_tool_call_id(self):
        out = normalize_messages_for_cache([
            {"role": "tool", "content": "result", "tool_call_id": "call_1"},
        ])
        assert out[0] == {"role": "tool", "content": "result", "tool_call_id": "call_1"}

    def test_assistant_with_tool_calls_preserved(self):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        }
        out = normalize_messages_for_cache([msg])
        assert out[0]["tool_calls"] == msg["tool_calls"]

    def test_assistant_reasoning_and_name_preserved(self):
        msg = {
            "role": "assistant",
            "content": "think",
            "reasoning_content": "secret chain-of-thought",
            "name": "helper",
        }
        out = normalize_messages_for_cache([msg])
        assert out[0]["reasoning_content"] == "secret chain-of-thought"
        assert out[0]["name"] == "helper"

    def test_tool_message_without_id_gets_empty(self):
        out = normalize_messages_for_cache([{"role": "tool", "content": "x"}])
        assert out[0]["tool_call_id"] == ""


# ── normalize_tools_for_cache ────────────────────────────────────────────

class TestNormalizeToolsForCache:
    def test_empty_returns_same(self):
        assert normalize_tools_for_cache([]) == []

    def test_sorts_by_function_name(self):
        tools = [
            {"function": {"name": "zeta", "arguments": "1"}},
            {"function": {"name": "alpha", "arguments": "2"}},
            {"function": {"name": "mid", "arguments": "3"}},
        ]
        names = [t["function"]["name"] for t in normalize_tools_for_cache(tools)]
        assert names == ["alpha", "mid", "zeta"]


# ── has_image_content ────────────────────────────────────────────────────

class TestHasImageContent:
    def test_false_without_images(self):
        assert has_image_content([
            {"role": "user", "content": "text only"},
        ]) is False

    def test_true_with_image_block(self):
        assert has_image_content([
            {"role": "user", "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ]},
        ]) is True

    def test_string_content_never_flagged(self):
        assert has_image_content([{"role": "user", "content": "no list"}]) is False


# ── sanitize_messages ────────────────────────────────────────────────────

class TestSanitizeMessages:
    def test_empty_messages_unchanged(self):
        assert sanitize_messages([]) == []

    def test_clean_history_unchanged(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
        assert sanitize_messages(msgs) == msgs

    def test_dangling_assistant_tool_call_removed(self):
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_x", "type": "function", "function": {"name": "f"}},
            ]},
        ]
        out = sanitize_messages(msgs)
        # Assistant message with only dangling calls and no content is dropped
        assert out == []

    def test_dangling_call_with_content_kept_but_calls_nulled(self):
        msgs = [
            {"role": "assistant", "content": "partial", "tool_calls": [
                {"id": "call_x", "type": "function", "function": {"name": "f"}},
            ]},
        ]
        out = sanitize_messages(msgs)
        assert len(out) == 1
        assert out[0]["tool_calls"] is None
        assert out[0]["content"] == "partial"

    def test_orphaned_tool_message_converted_to_user(self):
        msgs = [
            {"role": "tool", "tool_call_id": "ghost", "content": "orphaned result"},
        ]
        out = sanitize_messages(msgs)
        assert out == [{"role": "user", "content": "orphaned result"}]

    def test_dangling_and_orphaned_both_fixed(self):
        msgs = [
            {"role": "assistant", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f"}}]},
            {"role": "tool", "tool_call_id": "b", "content": "orphan"},
            {"role": "assistant", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f"}}]},
            {"role": "tool", "tool_call_id": "a", "content": "kept"},
        ]
        out = sanitize_messages(msgs)
        # id 'a' is declared AND answered -> kept intact
        # id 'b' is orphaned -> converted to a user message
        assert len(out) == 4
        assert out[0]["tool_calls"][0]["id"] == "a"
        assert out[1] == {"role": "user", "content": "orphan"}
        assert out[2]["tool_calls"][0]["id"] == "a"
        assert out[3] == {"role": "tool", "tool_call_id": "a", "content": "kept"}

    def test_keeps_referenced_tool_calls(self):
        msgs = [
            {"role": "assistant", "tool_calls": [{"id": "ok1", "type": "function", "function": {"name": "f"}}]},
            {"role": "tool", "tool_call_id": "ok1", "content": "good"},
        ]
        out = sanitize_messages(msgs)
        assert len(out) == 2
        assert out[0]["tool_calls"][0]["id"] == "ok1"


# ── compute_cache_savings ────────────────────────────────────────────────

class TestComputeCacheSavings:
    def test_zero_tokens_returns_zero(self, mock_config):
        assert compute_cache_savings("deepseek", "m", 0, mock_config) == 0.0

    def test_non_cost_savings_type_returns_zero(self, mock_config):
        mock_config.get_provider_cache_config.return_value = {"savings": "latency"}
        assert compute_cache_savings("deepseek", "m", 1000, mock_config) == 0.0

    def test_cost_savings_computed(self, mock_config):
        mock_config.get_provider_cache_config.return_value = {"savings": "cost"}
        mock_config.get_pricing.return_value = {"cache_hit": 0.01, "cache_miss": 0.5}
        # 1_000_000 tokens * (0.5 - 0.01) = 0.49
        assert compute_cache_savings("deepseek", "m", 1_000_000, mock_config) == pytest.approx(0.49)

    def test_pricing_error_returns_zero(self, mock_config):
        mock_config.get_provider_cache_config.return_value = {"savings": "cost"}
        mock_config.get_pricing.side_effect = KeyError("no pricing")
        assert compute_cache_savings("deepseek", "m", 1000, mock_config) == 0.0


# ── read_cache_hit_tokens ────────────────────────────────────────────────

class TestReadCacheHitTokens:
    def test_none_body_returns_zero(self, mock_config):
        assert read_cache_hit_tokens("deepseek", None, mock_config) == 0

    def test_default_field(self, mock_config):
        resp = {"usage": {"prompt_cache_hit_tokens": 42}}
        mock_config.get_provider_cache_config.return_value = {}
        assert read_cache_hit_tokens("deepseek", resp, mock_config) == 42

    def test_custom_hit_field(self, mock_config):
        resp = {"usage": {"cache_read_input_tokens": 7}}
        mock_config.get_provider_cache_config.return_value = {"hit_field": "cache_read_input_tokens"}
        assert read_cache_hit_tokens("llamacpp", resp, mock_config) == 7


# ── calculate_cost ───────────────────────────────────────────────────────

class TestCalculateCost:
    def test_plugin_path_wins(self, mock_config):
        resp = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}
        with patch("src.api.request_pipeline.get_registry") as mock_reg:
            mock_reg.return_value.calculate_cost.return_value = 0.123
            result = calculate_cost("deepseek", "m", {}, resp, mock_config)
        assert result["cost"] == 0.123
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50

    def test_fallback_config_pricing(self, mock_config):
        resp = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 1000,
                "prompt_cache_hit_tokens": 500,
                # no prompt_cache_miss_tokens -> derived from prompt - hit
            },
        }
        mock_config.get_provider_cache_config.return_value = {}
        mock_config.get_pricing.return_value = {"cache_hit": 0.01, "cache_miss": 0.5, "output": 1.0}
        with patch("src.api.request_pipeline.get_registry") as mock_reg:
            mock_reg.return_value.calculate_cost.return_value = None
            result = calculate_cost("deepseek", "m", {}, resp, mock_config)
        # hit: 500 * 0.01 /1M = 0.000005
        # miss: 500 * 0.5 /1M  = 0.00025
        # out:  1000 * 1.0 /1M = 0.001
        assert result["cache_hit_tokens"] == 500
        assert result["cache_miss_tokens"] == 500
        assert result["cost"] == pytest.approx(0.001255)

    def test_missing_usage(self, mock_config):
        mock_config.get_provider_cache_config.return_value = {}
        mock_config.get_pricing.return_value = {"cache_hit": 0.01, "cache_miss": 0.5, "output": 1.0}
        with patch("src.api.request_pipeline.get_registry") as mock_reg:
            mock_reg.return_value.calculate_cost.return_value = None
            result = calculate_cost("deepseek", "m", {}, None, mock_config)
        assert result["prompt_tokens"] == 0
        assert result["completion_tokens"] == 0
        assert result["cost"] == 0.0


# ── forward_request ──────────────────────────────────────────────────────

class TestForwardRequestStreaming:
    def _cfg(self, **overrides):
        cfg = {"provider": "testco", "api_key_env": "TEST_KEY", "base_url": "https://api.example.com/v1"}
        cfg.update(overrides)
        return cfg

    def test_streaming_chunk_reader(self, mock_config):
        body = {"messages": [], "stream": True}
        chunks = [b"data: {}\n\n", b"data: {}\n\n", b""]
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.side_effect = chunks
        with patch.dict("os.environ", {"TEST_KEY": "sk"}):
            with patch("urllib.request.urlopen", return_value=mock_resp):
                reader, status = forward_request(self._cfg(), body, mock_config)
        assert status == 200
        out = list(reader)
        assert out == [b"data: {}\n\n", b"data: {}\n\n"]
        mock_resp.close.assert_called_once()

    def test_api_key_from_config_when_env_missing(self, mock_config):
        body = {"messages": [], "stream": False}
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"ok": true}'
        mock_config.get_provider_key.return_value = "cfg-key"
        with patch.dict("os.environ", {}, clear=False):
            with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
                with patch.dict("os.environ", {"TEST_KEY": ""}):
                    result, status = forward_request(self._cfg(), body, mock_config)
        assert status == 200
        req = mock_open.call_args[0][0]
        assert req.headers["Authorization"] == "Bearer cfg-key"

    def test_http_400_raises_bad_request(self, mock_config):
        import urllib.error
        body = {"messages": [], "stream": False}
        err = urllib.error.HTTPError("url", 400, "Bad Request", {}, None)
        err.read = MagicMock(return_value=b'{"error":"bad"}')
        mock_config.get_provider_key.return_value = None
        with patch.dict("os.environ", {"TEST_KEY": "sk"}):
            with patch("urllib.request.urlopen", side_effect=err):
                with pytest.raises(ProviderBadRequestError):
                    forward_request(self._cfg(), body, mock_config)

    def test_http_500_raises_internal_error(self, mock_config):
        import urllib.error
        body = {"messages": [], "stream": False}
        err = urllib.error.HTTPError("url", 502, "Bad Gateway", {}, None)
        err.read = MagicMock(return_value=b'{"error":"down"}')
        mock_config.get_provider_key.return_value = None
        with patch.dict("os.environ", {"TEST_KEY": "sk"}):
            with patch("urllib.request.urlopen", side_effect=err):
                with pytest.raises(ProviderInternalError):
                    forward_request(self._cfg(), body, mock_config)

    def test_http_401_raises_auth_error(self, mock_config):
        import urllib.error
        body = {"messages": [], "stream": False}
        err = urllib.error.HTTPError("url", 401, "Unauthorized", {}, None)
        err.read = MagicMock(return_value=b'{"error":"nope"}')
        mock_config.get_provider_key.return_value = None
        with patch.dict("os.environ", {"TEST_KEY": "sk"}):
            with patch("urllib.request.urlopen", side_effect=err):
                with pytest.raises(ProviderAuthError):
                    forward_request(self._cfg(), body, mock_config)

    def test_api_key_from_credential_store(self, mock_config, temp_db, tmp_path):
        """When env is unset and a credential is stored, use the decrypted key."""
        from src.api.credential_store import CredentialStore
        import src.api.credential_store as cs_module

        with patch.dict("os.environ", {"LCP_SECRET_KEY": "test-master"}, clear=False):
            store = CredentialStore(temp_db, data_dir=str(tmp_path))
            cs_module._credential_store = store
            store.set("testco", "sk-stored")

            body = {"messages": [], "stream": False}
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.read.return_value = b'{"ok": true}'
            mock_config.get_provider_key.return_value = "cfg-key"  # should NOT be used
            with patch.dict("os.environ", {"TEST_KEY": ""}, clear=False):
                with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
                    result, status = forward_request(self._cfg(), body, mock_config)
            assert status == 200
            req = mock_open.call_args[0][0]
            assert req.headers["Authorization"] == "Bearer sk-stored"


# ── strip_forbidden_tools ───────────────────────────────────────────────

class TestStripForbiddenTools:
    def test_forbidden_none_strips_all(self):
        body = {"tools": [
            {"function": {"name": "read"}},
            {"function": {"name": "write"}},
        ]}
        new_body, blocked = strip_forbidden_tools(body, None)
        assert new_body["tools"] == []
        assert blocked == ["read", "write"]

    def test_forbidden_none_with_no_tools(self):
        body = {"messages": []}
        new_body, blocked = strip_forbidden_tools(body, None)
        assert new_body is body
        assert "tools" not in new_body
        assert blocked == []

    def test_strips_specific_tools_keeps_rest(self):
        body = {"tools": [
            {"function": {"name": "safe"}},
            {"function": {"name": "danger"}},
        ]}
        new_body, blocked = strip_forbidden_tools(body, ["danger"])
        assert blocked == ["danger"]
        assert [t["function"]["name"] for t in new_body["tools"]] == ["safe"]

    def test_empty_forbidden_returns_unchanged(self):
        body = {"tools": [{"function": {"name": "safe"}}]}
        new_body, blocked = strip_forbidden_tools(body, [])
        assert blocked == []
        assert new_body is body


# ── record_cost plugin hooks ─────────────────────────────────────────────

class TestRecordCostPluginHooks:
    def test_plugin_record_tokens_called(self, temp_db):
        cost_info = {
            "prompt_tokens": 100, "completion_tokens": 50,
            "cache_hit_tokens": 0, "cache_miss_tokens": 100,
            "cost": 0.001, "latency_ms": 10,
        }
        plugin = MagicMock()
        plugin.record_tokens = MagicMock()
        with patch("src.api.request_pipeline.get_registry") as mock_reg:
            mock_reg.return_value.for_provider.return_value = plugin
            record_cost(temp_db, "l2", "model-x", "llamacpp", cost_info,
                        True, None, [])
        plugin.record_tokens.assert_called_once()
        args = plugin.record_tokens.call_args
        # model positional + prompt/completion/cache_hit kwargs
        assert args[0][0] == "model-x"
        assert args[1]["prompt_tokens"] == 200  # prompt + cache_miss
        assert args[1]["completion_tokens"] == 50

    def test_plugin_record_tokens_failure_logged_not_raised(self, temp_db):
        cost_info = {
            "prompt_tokens": 1, "completion_tokens": 1,
            "cache_hit_tokens": 0, "cache_miss_tokens": 1,
            "cost": 0.0, "latency_ms": 1,
        }
        plugin = MagicMock()
        plugin.record_tokens = MagicMock(side_effect=RuntimeError("boom"))
        with patch("src.api.request_pipeline.get_registry") as mock_reg:
            mock_reg.return_value.for_provider.return_value = plugin
            record_cost(temp_db, "l2", "m", "llamacpp", cost_info, True, None, [])
        # Should not raise
        plugin.record_tokens.assert_called_once()


# ── try_chain error paths ────────────────────────────────────────────────

class TestTryChainErrorPaths:
    def _body(self):
        return {"messages": [{"role": "user", "content": "hi"}]}

    def test_bad_request_raises_immediately(self, chain_config):
        """ProviderBadRequestError is NOT retried/fallen back — wrapped and raised."""
        with patch("src.api.request_pipeline.forward_request",
                   side_effect=ProviderBadRequestError("model x missing")):
            with pytest.raises(AllProvidersFailedError) as exc:
                try_chain("l2", chain_config.profiles["l2"], self._body(), chain_config)
        assert "rejected the request as invalid" in str(exc.value)

    def test_config_error_falls_back_to_next_provider(self, chain_config):
        """ConfigError on the first provider falls back to the second."""
        good_resp = MagicMock()
        good_resp.status = 200
        good_resp.read.return_value = b'{"choices":[{"message":{"content":"ok"}}]}'

        def fake_forward(provider_cfg, body, config):
            if provider_cfg["provider"] == "first":
                raise ConfigError("missing api_key_env")
            return json.loads(good_resp.read()), 200

        with patch("src.api.request_pipeline.forward_request", side_effect=fake_forward):
            result_body, status, provider, model = try_chain(
                "l2", chain_config.profiles["l2"], self._body(), chain_config)
        assert status == 200
        assert provider == "second"
        assert model == "m2"

    def test_transient_failure_falls_back(self, chain_config):
        """A transient provider failure falls back to the next provider."""
        good_resp = MagicMock()
        good_resp.status = 200
        good_resp.read.return_value = b'{"choices":[{"message":{"content":"ok"}}]}'

        def fake_forward(provider_cfg, body, config):
            if provider_cfg["provider"] == "first":
                from src.api.exceptions import ProviderTimeoutError
                raise ProviderTimeoutError("unreachable")
            return json.loads(good_resp.read()), 200

        with patch("src.api.request_pipeline.forward_request", side_effect=fake_forward):
            result_body, status, provider, model = try_chain(
                "l2", chain_config.profiles["l2"], self._body(), chain_config)
        assert provider == "second"
        assert status == 200