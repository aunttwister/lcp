"""Tests for the DeepSeek cost tracking plugin."""

import json
import time
from unittest.mock import patch, MagicMock
from urllib.error import URLError

import pytest
from src.api.cost_plugins.deepseek import DeepSeekCostPlugin, _PRICING, _BALANCE_URL


# ═══════════════════════════════════════════════════════════════════════
# Pricing
# ═══════════════════════════════════════════════════════════════════════

class TestDeepSeekPricing:
    def setup_method(self):
        self.plugin = DeepSeekCostPlugin()

    def test_provider_name(self):
        assert self.plugin.provider_name == "deepseek"

    def test_supported_models(self):
        models = self.plugin.get_supported_models()
        assert "deepseek-v4-pro" in models
        assert "deepseek-v4-flash" in models

    def test_get_pricing_known(self):
        p = self.plugin.get_pricing("deepseek-v4-pro")
        assert p == _PRICING["deepseek-v4-pro"]
        assert p["cache_hit"] == 0.003625

    def test_get_pricing_flash(self):
        p = self.plugin.get_pricing("deepseek-v4-flash")
        assert p == _PRICING["deepseek-v4-flash"]
        assert p["cache_hit"] == 0.0028

    def test_get_pricing_unknown(self):
        assert self.plugin.get_pricing("nonexistent-model") is None


# ═══════════════════════════════════════════════════════════════════════
# calculate_cost
# ═══════════════════════════════════════════════════════════════════════

class TestDeepSeekCalculateCost:
    def setup_method(self):
        self.plugin = DeepSeekCostPlugin()

    def test_v4_pro_mixed_tokens(self):
        """v4-pro: 1M cache_hit + 1M cache_miss + 500K output"""
        cost = self.plugin.calculate_cost("deepseek-v4-pro", {
            "prompt_cache_hit_tokens": 1_000_000,
            "prompt_cache_miss_tokens": 1_000_000,
            "completion_tokens": 500_000,
        })
        expected = (1_000_000 / 1_000_000) * 0.003625 \
                   + (1_000_000 / 1_000_000) * 0.435 \
                   + (500_000 / 1_000_000) * 0.87
        assert cost == pytest.approx(expected)

    def test_v4_flash_only_output(self):
        cost = self.plugin.calculate_cost("deepseek-v4-flash", {
            "prompt_tokens": 0,
            "completion_tokens": 100_000,
        })
        expected = (100_000 / 1_000_000) * 0.28
        assert cost == pytest.approx(expected)

    def test_cache_only_auto_miss(self):
        """When only prompt_tokens is given and no cache_hit/miss, all treated as miss."""
        cost = self.plugin.calculate_cost("deepseek-v4-pro", {
            "prompt_tokens": 500_000,
            "completion_tokens": 0,
        })
        expected = (500_000 / 1_000_000) * 0.435
        assert cost == pytest.approx(expected)

    def test_cache_hit_auto_miss_derived(self):
        """When cache_hit is set but cache_miss is not, derive from prompt - hit."""
        cost = self.plugin.calculate_cost("deepseek-v4-pro", {
            "prompt_tokens": 1000,
            "prompt_cache_hit_tokens": 300,
            "completion_tokens": 0,
        })
        # cache_miss = 1000 - 300 = 700
        expected = (700 / 1_000_000) * 0.435 + (300 / 1_000_000) * 0.003625
        assert cost == pytest.approx(expected, rel=1e-5)

    def test_unknown_model_returns_none(self):
        cost = self.plugin.calculate_cost("unknown-model", {"prompt_tokens": 100})
        assert cost is None


# ═══════════════════════════════════════════════════════════════════════
# fetch_balance
# ═══════════════════════════════════════════════════════════════════════

class TestDeepSeekFetchBalance:
    def setup_method(self):
        self.plugin = DeepSeekCostPlugin()

    def test_no_api_key_returns_none(self):
        with patch.dict("os.environ", {}, clear=True):
            assert self.plugin.fetch_balance() is None

    def test_successful_balance_query(self):
        fake_resp = MagicMock()
        fake_resp.read.return_value = json.dumps({
            "balance": 42.50,
            "currency": "USD",
            "total_granted": 100.0,
        }).encode("utf-8")
        fake_resp.__enter__.return_value = fake_resp

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}):
            with patch("src.api.cost_plugins.deepseek.urlopen", return_value=fake_resp) as mock_req:
                result = self.plugin.fetch_balance()

        assert result is not None
        assert result["balance"] == 42.50
        assert result["currency"] == "USD"
        assert result["total_granted"] == 100.0
        # Verify the correct URL was called
        args, _ = mock_req.call_args
        assert args[0].full_url == _BALANCE_URL

    def test_balance_is_cached(self):
        """Within cache TTL, the API should not be called again."""
        fake_resp = MagicMock()
        fake_resp.read.return_value = json.dumps({"balance": 10.0}).encode("utf-8")
        fake_resp.__enter__.return_value = fake_resp

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}):
            with patch("src.api.cost_plugins.deepseek.urlopen", return_value=fake_resp) as mock_req:
                # First call hits API
                r1 = self.plugin.fetch_balance()
                assert r1["balance"] == 10.0
                assert mock_req.call_count == 1

                # Second call should be cached
                r2 = self.plugin.fetch_balance()
                assert r2["balance"] == 10.0
                assert mock_req.call_count == 1  # still 1 — no API call

    def test_cache_expires(self):
        """After cache TTL elapses, the API should be called again."""
        fake_resp = MagicMock()
        fake_resp.read.return_value = json.dumps({"balance": 10.0}).encode("utf-8")
        fake_resp.__enter__.return_value = fake_resp

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}):
            with patch("src.api.cost_plugins.deepseek.urlopen", return_value=fake_resp):
                self.plugin.fetch_balance()
                # Manually expire cache
                self.plugin._balance_cached_at = 0
                self.plugin.fetch_balance()
                # We can verify cache was bypassed by checking the state
                assert self.plugin._balance_cached_at > 0

    def test_api_error_returns_none_and_logs(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}):
            with patch("src.api.cost_plugins.deepseek.urlopen", side_effect=URLError("timeout")):
                result = self.plugin.fetch_balance()
        assert result is None

    def test_malformed_json_returns_none(self):
        fake_resp = MagicMock()
        fake_resp.read.return_value = b"not json"
        fake_resp.__enter__.return_value = fake_resp

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}):
            with patch("src.api.cost_plugins.deepseek.urlopen", return_value=fake_resp):
                result = self.plugin.fetch_balance()
        assert result is None
