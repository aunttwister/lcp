"""Tests for cost_estimator.py"""
import pytest
from src.api.cost_estimator import count_tokens, estimate_tokens, estimate_from_request, estimate_cost

def test_count_tokens_simple():
    n = count_tokens([{"role": "user", "content": "hello world"}])
    assert n >= 2
    assert n <= 20

def test_count_tokens_empty():
    n = count_tokens([{"role": "user", "content": ""}])
    assert n >= 4

def test_count_tokens_multiple_msgs():
    msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    n = count_tokens(msgs)
    assert n >= 8

def test_count_tokens_with_tools():
    msgs = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    n_with = count_tokens(msgs, tools)
    n_without = count_tokens(msgs, None)
    assert n_with > n_without

def test_estimate_tokens_basic():
    msgs = [{"role": "user", "content": "hello world"}]
    result = estimate_tokens(msgs)
    assert "total" in result
    assert "messages" in result
    assert result["total"] >= 2

def test_estimate_tokens_with_tools():
    msgs = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    result = estimate_tokens(msgs, tools)
    assert result["tools"] > 0

def test_estimate_from_request():
    result = estimate_from_request("deepseek-v4-pro", [{"role": "user", "content": "hello"}], max_tokens=100)
    assert "estimated_total_cost" in result
    assert result["currency"] == "USD"

def test_estimate_from_request_unknown_model():
    """Unknown model falls back to deepseek pricing (lines 33-37)."""
    result = estimate_from_request("nonexistent-model", [{"role": "user", "content": "hello"}], max_tokens=100)
    assert "estimated_total_cost" in result
    assert result["estimated_total_cost"] > 0

def test_count_tokens_vision_content():
    """Vision content blocks (content as list of dicts)."""
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
    }]
    n = count_tokens(msgs)
    assert n >= 4


# ═══════════════════════════════════════════════════════════════════════
# estimate_cost
# ═══════════════════════════════════════════════════════════════════════

class TestEstimateCost:
    """Tests for estimate_cost()."""

    def test_basic_estimation(self):
        result = estimate_cost("deepseek-v4-pro", 1000, max_tokens=500)
        assert result["input_tokens"] == 1000
        assert result["estimated_output_tokens"] == 500
        assert result["estimated_input_cost"] > 0
        assert result["estimated_output_cost"] > 0
        assert result["estimated_total_cost"] > 0
        assert result["currency"] == "USD"
        # total = input + output
        assert (result["estimated_total_cost"]
                == round(result["estimated_input_cost"] + result["estimated_output_cost"], 8))

    def test_known_model_uses_correct_pricing(self):
        """deepseek-v4-pro uses 0.435 input / 0.87 output per 1M tokens."""
        result = estimate_cost("deepseek-v4-pro", 1_000_000, max_tokens=1_000_000)
        assert result["estimated_input_cost"] == 0.435
        assert result["estimated_output_cost"] == 0.87
        assert result["estimated_total_cost"] == 1.305

    def test_unknown_model_falls_back_to_default(self):
        """Unknown models fall back to 0.435/0.87 pricing."""
        result = estimate_cost("nonexistent-model", 1_000_000, max_tokens=1_000_000)
        assert result["estimated_input_cost"] == 0.435
        assert result["estimated_output_cost"] == 0.87

    def test_custom_pricing(self):
        """Custom pricing dict overrides defaults."""
        custom = {"cache_miss": 1.50, "output": 3.00}
        result = estimate_cost("irrelevant", 1_000_000, max_tokens=500_000, pricing=custom)
        assert result["estimated_input_cost"] == 1.50
        assert result["estimated_output_cost"] == 1.50  # 500k / 1M * 3.00

    def test_pricing_with_input_key(self):
        """Pricing dict using 'input' instead of 'cache_miss' works too."""
        custom = {"input": 2.00, "output": 4.00}
        result = estimate_cost("irrelevant", 1_000_000, max_tokens=1_000_000, pricing=custom)
        assert result["estimated_input_cost"] == 2.00

    def test_zero_tokens(self):
        result = estimate_cost("deepseek-v4-pro", 0, max_tokens=0)
        assert result["estimated_input_cost"] == 0.0
        assert result["estimated_output_cost"] == 0.0
        assert result["estimated_total_cost"] == 0.0
