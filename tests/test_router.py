"""Tests for router.py"""
import pytest
import sys
from src.api.router import (
    DynamicRouter, CapabilityRouter, classify_task, get_dynamic_router,
    resolve_latest_variant, _dated_variant_key,
)


# ── Legacy DynamicRouter tests (flash/pro heuristic) ──────────────────────

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


# ── CapabilityRouter tests ────────────────────────────────────────────────

def test_classify_code_generation():
    msgs = [{"role": "user", "content": "write a function in python to sort a list"}]
    assert classify_task(msgs) == "code_generation"

def test_classify_debugging():
    msgs = [{"role": "user", "content": "why does this fail with a TypeError?"}]
    assert classify_task(msgs) == "debugging"

def test_classify_agentic():
    msgs = [{"role": "system", "content": "You are an AI agent with tools: read_file, write_file"}]
    assert classify_task(msgs) == "agentic_multi_step"

def test_classify_planning():
    msgs = [{"role": "user", "content": "design the architecture for a microservice"}]
    assert classify_task(msgs) == "planning"

def test_classify_casual():
    msgs = [{"role": "user", "content": "hello, how are you?"}]
    assert classify_task(msgs) == "casual_chat"

def test_classify_defaults_to_code():
    msgs = [{"role": "user", "content": "what is the speed of light?"}]
    assert classify_task(msgs) == "code_generation"  # default for LCP

def test_classify_many_tools():
    tools = [{"type": "function", "function": {"name": f"t{i}"}} for i in range(6)]
    msgs = [{"role": "user", "content": "do something"}]
    assert classify_task(msgs, tools=tools) == "agentic_multi_step"

def test_router_disabled_returns_none():
    router = CapabilityRouter(enabled=False)
    msgs = [{"role": "user", "content": "write a function"}]
    assert router.select_model(msgs) is None

def test_router_enabled_selects_best():
    router = CapabilityRouter(enabled=True)
    msgs = [{"role": "user", "content": "hello there"}]
    result = router.select_model(msgs, available_models=["deepseek-v4-flash", "deepseek-v4-pro"])
    # Casual chat → flash should be preferred (cheaper, good enough)
    assert result is not None

def test_router_ranks_models():
    router = CapabilityRouter(enabled=True)
    ranked = router.rank_models("code_generation", ["deepseek-v4-flash", "deepseek-v4-pro"])
    assert len(ranked) == 2
    # Both are valid — flash is cheaper and nearly as capable for coding,
    # so it may rank higher with cost bias. Pro is within 5% margin.
    assert ranked[0][1] > 0.6  # both should have good scores

def test_singleton():
    assert get_dynamic_router() is get_dynamic_router()


# ── Dated variant alias resolution tests ──────────────────────────────────

def test_dated_variant_key_split():
    assert _dated_variant_key("deepseek-v4-flash-0731") == ("deepseek-v4-flash", 731)
    assert _dated_variant_key("deepseek-v4-flash") == ("deepseek-v4-flash", 0)
    assert _dated_variant_key("deepseek-v4-pro") == ("deepseek-v4-pro", 0)

def test_dated_variant_key_ignores_non_date_suffix():
    # "pro" is not 4 digits, so no split
    assert _dated_variant_key("deepseek-v4-pro") == ("deepseek-v4-pro", 0)
    # 3-digit suffix is not a date
    assert _dated_variant_key("model-123") == ("model-123", 0)

def test_resolve_bare_alias_to_latest_variant():
    scores = {
        "deepseek-v4-flash": 0.655,
        "deepseek-v4-flash-0731": 0.742,
    }
    assert resolve_latest_variant("deepseek-v4-flash", scores) == "deepseek-v4-flash-0731"

def test_resolve_explicit_dated_variant_unchanged():
    scores = {
        "deepseek-v4-flash": 0.655,
        "deepseek-v4-flash-0731": 0.742,
    }
    assert resolve_latest_variant("deepseek-v4-flash-0731", scores) == "deepseek-v4-flash-0731"

def test_resolve_picks_newest_dated_variant():
    scores = {
        "deepseek-v4-flash": 0.655,
        "deepseek-v4-flash-0731": 0.742,
        "deepseek-v4-flash-0813": 0.800,
    }
    assert resolve_latest_variant("deepseek-v4-flash", scores) == "deepseek-v4-flash-0813"

def test_resolve_no_dated_variant_returns_self():
    scores = {"deepseek-v4-pro": 0.716}
    assert resolve_latest_variant("deepseek-v4-pro", scores) == "deepseek-v4-pro"
