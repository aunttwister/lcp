"""Tests for router.py"""
import pytest
import sys
sys.path.insert(0, "/opt/lcp")
from src.router import DynamicRouter, get_dynamic_router

def test_disabled_returns_pro():
    router = DynamicRouter(enabled=False)
    msgs = [{"role": "user", "content": "hi"}]
    assert router.get_recommended_model(msgs, max_tokens=100) == "deepseek-v4-pro"

def test_short_prompt_flash():
    router = DynamicRouter(enabled=True)
    msgs = [{"role": "user", "content": "hello"}]
    assert router.get_recommended_model(msgs, max_tokens=100) == "deepseek-v4-flash"

def test_long_prompt_pro():
    router = DynamicRouter(enabled=True)
    huge = [{"role": "user", "content": "hello " * 2000}]
    assert router.get_recommended_model(huge) == "deepseek-v4-pro"

def test_many_tools_pro():
    router = DynamicRouter(enabled=True)
    tools = [{"type": "function", "function": {"name": "t" + str(i)}} for i in range(5)]
    msgs = [{"role": "user", "content": "hi"}]
    assert router.get_recommended_model(msgs, tools=tools, max_tokens=100) == "deepseek-v4-pro"

def test_long_max_tokens_pro():
    router = DynamicRouter(enabled=True)
    msgs = [{"role": "user", "content": "hi"}]
    assert router.get_recommended_model(msgs, max_tokens=4096) == "deepseek-v4-pro"

def test_singleton():
    assert get_dynamic_router() is get_dynamic_router()
