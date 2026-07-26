"""Tests for cost_estimator.py"""
import pytest
import sys
from src.api.cost_estimator import count_tokens, estimate_tokens, estimate_from_request

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
