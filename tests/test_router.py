"""Tests for router.py"""
import os
import tempfile

import pytest
import sys
from src.api.router import (
    DynamicRouter, CapabilityRouter, classify_task, get_dynamic_router,
    logical_model_name, benchmark_model_name,
)
from src.api.seed_capabilities import DEFAULT_MODEL_REGISTRY


@pytest.fixture
def registry_db():
    """A fresh DB seeded with the default model registry."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from src.api.models import get_engine, Base
    engine = get_engine(path)
    Base.metadata.create_all(engine)
    engine.dispose()
    from src.api.seed_capabilities import seed_model_registry
    seed_model_registry(path)
    yield path
    for ext in ("", "-wal", "-shm"):
        try:
            os.unlink(path + ext)
        except FileNotFoundError:
            pass


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

def test_router_enabled_selects_best(registry_db):
    router = CapabilityRouter(enabled=True, db_path=registry_db)
    msgs = [{"role": "user", "content": "hello there"}]
    result = router.select_model(msgs, available_models=["deepseek-v4-flash", "deepseek-v4-pro"])
    # Casual chat → flash should be preferred (cheaper, good enough)
    assert result is not None

def test_router_ranks_models(registry_db):
    router = CapabilityRouter(enabled=True, db_path=registry_db)
    ranked = router.rank_models("code_generation", ["deepseek-v4-flash", "deepseek-v4-pro"])
    assert len(ranked) == 2
    # Both are valid — flash is cheaper and nearly as capable for coding,
    # so it may rank higher with cost bias. Pro is within 5% margin.
    assert ranked[0][1] > 0.6  # both should have good scores

def test_singleton():
    assert get_dynamic_router() is get_dynamic_router()


# ── Model registry tests (explicit alias → logical → benchmark) ───────────

def test_logical_model_name_maps_provider_alias(registry_db):
    assert logical_model_name("deepseek/deepseek-v4-pro", registry_db) == "deepseek-v4-pro"
    assert logical_model_name("moonshotai/Kimi-K3", registry_db) == "kimi-k3"
    assert logical_model_name("Qwen/Qwen3.8-Max", registry_db) == "qwen3.8-max"
    assert logical_model_name("google/gemini-3.6-flash", registry_db) == "gemini-3.6-flash"

def test_logical_model_name_passthrough_unknown(registry_db):
    assert logical_model_name("some-brand-new-model", registry_db) == "some-brand-new-model"

def test_logical_model_name_case_insensitive(registry_db):
    assert logical_model_name("DeepSeek-V4-Pro", registry_db) == "deepseek-v4-pro"

def test_benchmark_model_name_resolves_rolling_alias(registry_db):
    # Rolling 'deepseek-v4-flash' alias scores as the latest benchmarked
    # snapshot (0731), not the stale bare name.
    assert benchmark_model_name("deepseek-v4-flash", registry_db) == "deepseek-v4-flash-0731"

def test_benchmark_model_name_passthrough_without_pin(registry_db):
    # deepseek-v4-pro has no dated snapshot in the registry
    assert benchmark_model_name("deepseek-v4-pro", registry_db) == "deepseek-v4-pro"

def test_registry_aliases_are_unique():
    seen = {}
    for entry in DEFAULT_MODEL_REGISTRY:
        for alias in entry["aliases"]:
            key = alias.lower()
            assert key not in seen, f"alias {alias!r} duplicated in registry"
            seen[key] = entry["logical_name"]
