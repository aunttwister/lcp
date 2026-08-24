"""Tests for prompt_cache.py"""
import time
from unittest.mock import patch

from src.api.prompt_cache import PromptCache, get_prompt_cache


def _body(msg: str) -> dict:
    return {"messages": [{"role": "user", "content": msg}]}


def test_set_and_get():
    cache = PromptCache(max_entries=100, ttl_seconds=3600)
    body = _body("hello")
    response = {"choices": [{"message": {"content": "hi"}}]}
    cache.set("l2", "deepseek-v4-pro", body, response)
    result = cache.get("l2", "deepseek-v4-pro", body)
    assert result == response

def test_miss_different_profile():
    cache = PromptCache(max_entries=100, ttl_seconds=3600)
    cache.set("l2", "deepseek-v4-pro", _body("test"), {"x": 1})
    assert cache.get("l1", "deepseek-v4-pro", _body("test")) is None

def test_miss_different_body():
    cache = PromptCache(max_entries=100, ttl_seconds=3600)
    cache.set("l2", "deepseek-v4-pro", _body("a"), {"x": 1})
    assert cache.get("l2", "deepseek-v4-pro", _body("b")) is None

def test_expired_entry():
    """get returns None for entries past TTL (lines 47-49)."""
    cache = PromptCache(max_entries=100, ttl_seconds=3600)
    body = _body("expire-me")
    cache.set("l2", "deepseek-v4-pro", body, {"x": 1})
    # Fast-forward time past TTL — save real time before mocking
    real_now = time.time()
    with patch("time.time") as mock_time:
        mock_time.return_value = real_now + 7200  # 2 hours later
        assert cache.get("l2", "deepseek-v4-pro", body) is None

def test_set_eviction():
    """set evicts oldest entry when at capacity (lines 59-60)."""
    cache = PromptCache(max_entries=1, ttl_seconds=3600)
    cache.set("l2", "deepseek-v4-pro", _body("first"), {"data": 1})
    cache.set("l2", "deepseek-v4-pro", _body("second"), {"data": 2})
    # First entry should be evicted, second should exist
    assert cache.get("l2", "deepseek-v4-pro", _body("first")) is None
    assert cache.get("l2", "deepseek-v4-pro", _body("second")) == {"data": 2}

def test_stats():
    cache = PromptCache(max_entries=100, ttl_seconds=3600)
    cache.set("l2", "deepseek-v4-pro", _body("hello"), {"x": 1})
    cache.get("l2", "deepseek-v4-pro", _body("hello"))
    cache.get("l2", "deepseek-v4-pro", _body("no"))
    s = cache.stats
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["entries"] == 1
    assert s["hit_rate"] == 0.5

def test_clear():
    cache = PromptCache(max_entries=100, ttl_seconds=3600)
    cache.set("l2", "deepseek-v4-pro", _body("hello"), {"x": 1})
    cache.clear()
    assert cache.stats["entries"] == 0

def test_singleton():
    assert get_prompt_cache() is get_prompt_cache()
