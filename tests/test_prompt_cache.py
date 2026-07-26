"""Tests for prompt_cache.py"""
import pytest
import sys
from src.api.prompt_cache import PromptCache, get_prompt_cache

def test_set_and_get():
    cache = PromptCache(max_entries=100, ttl_seconds=3600)
    body = {"messages": [{"role": "user", "content": "hello"}]}
    response = {"choices": [{"message": {"content": "hi"}}]}
    cache.set("l2", "deepseek-v4-pro", body, response)
    result = cache.get("l2", "deepseek-v4-pro", body)
    assert result == response

def test_miss_different_profile():
    cache = PromptCache(max_entries=100, ttl_seconds=3600)
    body = {"messages": [{"role": "user", "content": "test"}]}
    cache.set("l2", "deepseek-v4-pro", body, {"x": 1})
    assert cache.get("l1", "deepseek-v4-pro", body) is None

def test_miss_different_body():
    cache = PromptCache(max_entries=100, ttl_seconds=3600)
    body1 = {"messages": [{"role": "user", "content": "a"}]}
    body2 = {"messages": [{"role": "user", "content": "b"}]}
    cache.set("l2", "deepseek-v4-pro", body1, {"x": 1})
    assert cache.get("l2", "deepseek-v4-pro", body2) is None

def test_stats():
    cache = PromptCache(max_entries=100, ttl_seconds=3600)
    body = {"messages": [{"role": "user", "content": "hello"}]}
    cache.set("l2", "deepseek-v4-pro", body, {"x": 1})
    cache.get("l2", "deepseek-v4-pro", body)
    cache.get("l2", "deepseek-v4-pro", {"messages": [{"role": "user", "content": "no"}]})
    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 1
    assert cache.stats["entries"] == 1

def test_clear():
    cache = PromptCache(max_entries=100, ttl_seconds=3600)
    body = {"messages": [{"role": "user", "content": "hello"}]}
    cache.set("l2", "deepseek-v4-pro", body, {"x": 1})
    cache.clear()
    assert cache.stats["entries"] == 0

def test_singleton():
    assert get_prompt_cache() is get_prompt_cache()
