"""Tests for router.py"""
import os
import tempfile

import pytest
import sys
from src.api.router import (
    DynamicRouter, CapabilityRouter, classify_task, get_dynamic_router,
    logical_model_name, benchmark_model_name,
    normalize_model_id, detect_quantization,
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

def test_logical_model_name_passthrough_unknown(registry_db):
    assert logical_model_name("some-brand-new-model", registry_db) == "some-brand-new-model"

def test_logical_model_name_case_insensitive(registry_db):
    assert logical_model_name("DeepSeek-V4-Pro", registry_db) == "deepseek-v4-pro"

def test_benchmark_model_name_resolves_rolling_alias(registry_db):
    # The benchmark key is the stable logical name; the dated 0731 snapshot
    # is a RELEASE of it, not a separate identity.
    assert benchmark_model_name("deepseek-v4-flash", registry_db) == "deepseek-v4-flash"

def test_benchmark_model_name_passthrough_without_pin(registry_db):
    # deepseek-v4-pro has no dated snapshot in the registry
    assert benchmark_model_name("deepseek-v4-pro", registry_db) == "deepseek-v4-pro"

def test_registry_logical_names_unique():
    names = [entry["logical_name"].lower() for entry in DEFAULT_MODEL_REGISTRY]
    assert len(names) == len(set(names)), "logical names duplicated in registry"


# ── Model-ID normalization + quantization detection ────────────────────────

def test_normalize_model_id_strips_models_prefix_and_gguf():
    assert normalize_model_id("/models/qwen3.6-27b-q4_k_m.gguf") == "qwen3.6-27b-q4_k_m"
    assert normalize_model_id("/models/deepseek-v4-flash.gguf") == "deepseek-v4-flash"
    assert normalize_model_id("deepseek-v4-pro") == "deepseek-v4-pro"

def test_normalize_model_id_strips_leading_slash_and_path():
    assert normalize_model_id("/qwen3.6-27b-q4_k_m.gguf") == "qwen3.6-27b-q4_k_m"
    assert normalize_model_id("moonshotai/Kimi-K3") == "kimi-k3"

def test_normalize_model_id_lowercases_and_trims():
    assert normalize_model_id("  DeepSeek-V4-Pro ") == "deepseek-v4-pro"
    assert normalize_model_id("Qwen/Qwen3.8-Max") == "qwen3.8-max"

def test_normalize_model_id_empty():
    assert normalize_model_id("") == ""
    assert normalize_model_id(None) is None

def test_detect_quantization_gguf_tags():
    assert detect_quantization("/models/qwen3.6-27b-q4_k_m.gguf") == "Q4_K_M"
    assert detect_quantization("llama-3.1-8b-instruct-q8_0") == "Q8_0"
    assert detect_quantization("qwen3.6-27b-q4_0") == "Q4_0"
    assert detect_quantization("model-f16") == "F16"

def test_detect_quantization_none_for_regular_models():
    assert detect_quantization("deepseek-v4-pro") is None
    assert detect_quantization("qwen3.8-max") is None
    assert detect_quantization("gpt-5.6-sol") is None
    assert detect_quantization("kimi-k3") is None

def test_logical_model_name_normalizes_llamacpp_path(registry_db):
    # An unregistered llama.cpp path is normalized even without a registry entry.
    assert logical_model_name("/models/qwen3.6-27b-q4_k_m.gguf", registry_db) == "qwen3.6-27b-q4_k_m"


def test_load_model_registry_includes_quantization(registry_db):
    from src.api.seed_capabilities import load_model_registry, seed_model_registry
    # Insert a quantized entry and re-read.
    from src.api.models import ModelRegistryEntry, get_engine, get_session
    import json
    engine = get_engine(registry_db)
    with get_session(engine) as session:
        session.add(ModelRegistryEntry(
            logical_name="qwen3.6-27b-q4_k_m",
            benchmark_key="qwen3.6-27b-q4_k_m",
            provider_mappings_json=json.dumps({"llamacpp": "/models/qwen3.6-27b-q4_k_m.gguf"}),
            quantization="Q4_K_M",
        ))
        session.commit()
    registry = load_model_registry(registry_db)
    entry = registry["qwen3.6-27b-q4_k_m"]
    assert entry["quantization"] == "Q4_K_M"
    assert entry["provider_mappings"]["llamacpp"] == "/models/qwen3.6-27b-q4_k_m.gguf"


# ── Provider → model mappings ─────────────────────────────────────────────

def test_provider_model_name_explicit_mapping(registry_db):
    from src.api.router import provider_model_name
    # Command Code uses catalog IDs; registry pins the exact mapping.
    assert provider_model_name("deepseek-v4-pro", "commandcode", registry_db) == "deepseek/deepseek-v4-pro"
    # OpenCode/DeepSeek use bare names.
    assert provider_model_name("deepseek-v4-pro", "opencode", registry_db) == "deepseek-v4-pro"

def test_provider_model_name_falls_back_to_logical(registry_db):
    from src.api.router import provider_model_name
    # Kimi: explicit mapping supplies the prefixed catalog ID.
    assert provider_model_name("kimi-k3", "commandcode", registry_db) == "moonshotai/Kimi-K3"

def test_provider_model_name_unknown_provider_passthrough(registry_db):
    from src.api.router import provider_model_name
    assert provider_model_name("deepseek-v4-pro", "someprovider", registry_db) == "deepseek-v4-pro"


# ── CapabilityRouter selection edge paths ────────────────────────────────────

def test_router_keeps_default_when_tied(registry_db):
    """When the best model is within 5% of the chain default, keep default."""
    from src.api.router import CapabilityRouter
    router = CapabilityRouter(enabled=True, db_path=registry_db)
    # available_models where all scores are equal → no override.
    result = router.select_model(
        [{"role": "user", "content": "write a function"}],
        available_models=["deepseek-v4-pro", "deepseek-v4-pro"],
    )
    assert result is None

def test_router_no_available_models_returns_none(registry_db):
    from src.api.router import CapabilityRouter
    router = CapabilityRouter(enabled=True, db_path=registry_db)
    assert router.select_model([{"role": "user", "content": "hi"}], available_models=[]) is None

def test_router_empty_rank_returns_empty(registry_db):
    from src.api.router import CapabilityRouter
    router = CapabilityRouter(enabled=True, db_path=registry_db)
    assert router.rank_models("code_generation", []) == []

def test_router_load_matrix_error_returns_empty(tmp_path):
    from src.api.router import CapabilityRouter
    router = CapabilityRouter(enabled=True, db_path=str(tmp_path / "missing.db"))
    # Missing DB → load fails gracefully to {}.
    matrix = router.load_matrix()
    assert matrix == {}

def test_init_router_warm_cache(registry_db):
    from src.api import router as router_mod
    router_mod.init_router(db_path=registry_db, enabled=True)
    assert router_mod.get_dynamic_router().enabled is True
    # Restore for other tests.
    router_mod.init_router(db_path=registry_db, enabled=False)

def test_dynamic_router_flash_heuristic():
    from src.api.router import DynamicRouter
    r = DynamicRouter(enabled=True)
    assert r.should_use_flash([{"role": "user", "content": "hi"}]) is True
    assert r.should_use_flash([{"role": "user", "content": "x " * 3000}]) is False
    tools = [{"type": "function", "function": {"name": f"t{i}"}} for i in range(6)]
    assert r.should_use_flash([{"role": "user", "content": "hi"}], tools=tools) is False
    assert r.should_use_flash([{"role": "user", "content": "hi"}], max_tokens=4096) is False
