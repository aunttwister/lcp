"""Tests for router.py"""
import os
import tempfile

import pytest
from src.api.router import (
    CapabilityRouter, classify_task, get_dynamic_router,
    init_router, logical_model_name, benchmark_model_name,
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

def test_classify_agentic_system_prompt_does_not_mask_planning():
    """An agentic system prompt must not mask a concrete planning user msg."""
    msgs = [
        {"role": "system", "content": "You are an AI agent with tools: read_file, write_file, terminal"},
        {"role": "user", "content": "design the architecture for a microservice"},
    ]
    assert classify_task(msgs) == "planning"

def test_classify_agentic_system_prompt_does_not_mask_debugging():
    msgs = [
        {"role": "system", "content": "You are a coding agent. tools: terminal, patch"},
        {"role": "user", "content": "debug why this returns a TypeError"},
    ]
    assert classify_task(msgs) == "debugging"

def test_classify_agentic_system_prompt_does_not_mask_unit_tests():
    msgs = [
        {"role": "system", "content": "You are a coding agent with tools: write_file"},
        {"role": "user", "content": "write unit tests for the payment module"},
    ]
    assert classify_task(msgs) == "unit_tests"

def test_classify_user_intent_beats_assistant_test_chatter():
    """Regression: assistant/tool messages that mention 'test' must NOT hijack
    the classification. The USER message is the actual intent signal."""
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "can we plan the next feature in features folder?"},
        {"role": "assistant", "content": "Sure, let me look at the existing test suite first."},
        {"role": "tool", "content": "Ran 12 tests, 3 failed. The test file is features/test_plan.py"},
    ]
    assert classify_task(msgs) == "planning"

def test_classify_user_intent_beats_tool_test_output():
    """A user debugging request must stay debugging even when tool output
    mentions running tests."""
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "why does this endpoint return a 500? debug the traceback"},
        {"role": "assistant", "content": "I ran the test suite to reproduce"},
        {"role": "tool", "content": "test_plan.py: 3 tests passed, 1 failed"},
    ]
    assert classify_task(msgs) == "debugging"

def test_classify_user_mentioning_tests_still_unit_tests():
    """If the USER message itself asks for tests, it's still unit_tests despite
    any assistant/tool chatter."""
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "write pytest unit tests for the auth module with mocking"},
        {"role": "assistant", "content": "Let me check the current test coverage"},
    ]
    assert classify_task(msgs) == "unit_tests"

def test_classify_first_user_msg_beats_tool_result_as_user():
    """Regression: some clients send tool results as role='user'. The classifier
    must use the FIRST user message (the original instruction), not the
    accumulated user text or the latest tool echo."""
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "can we plan the next feature in features folder?"},
        {"role": "assistant", "content": "Let me look at the current state."},
        {"role": "user", "content": "[tool result] Ran 12 tests, 3 failed. test_plan.py: FAILED"},
    ]
    assert classify_task(msgs) == "planning"

def test_classify_first_user_msg_beats_later_test_echoes():
    """A multi-turn session where the first user turn was planning and later
    tool-result user turns mention tests must stay planning."""
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "design the architecture for the new feature"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "[tool] running tests..."},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "[tool] 10 tests passed"},
    ]
    assert classify_task(msgs) == "planning"

# ── Newest genuine user instruction (intent extraction) ───────────────────

def test_classify_mid_session_instruction_reclassifies():
    """A session that started with debugging must RE-CLASSIFY when the user
    later issues a new planning instruction (regression for sticky-debugging:
    the old first-user-message behavior froze intent for the whole session)."""
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "fix this error in the code"},
        {"role": "assistant", "content": "Let me reproduce."},
        {"role": "user", "content": "[tool result] Ran 12 tests, 3 failed. test_plan.py: FAILED"},
        {"role": "assistant", "content": "I see the bug."},
        {"role": "user", "content": "can we plan the next feature in features folder?"},
    ]
    assert classify_task(msgs) == "planning"

def test_classify_continuation_keeps_earlier_intent():
    """A trailing 'continue' carries no new intent — it must NOT reset a
    debugging session; the intent stays with the last real instruction."""
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "why does this endpoint return a 500? debug the traceback"},
        {"role": "assistant", "content": "Running the endpoint..."},
        {"role": "user", "content": "[tool] Exit code: 1, traceback below"},
        {"role": "assistant", "content": "Found the failing line."},
        {"role": "user", "content": "continue"},
    ]
    assert classify_task(msgs) == "debugging"

def test_classify_short_instruction_not_treated_as_continuation():
    """Short genuine instructions ('fix it', 'make it work') are intent, not
    continuations."""
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "fix this"},
    ]
    assert classify_task(msgs) == "debugging"

def test_classify_structural_tool_result_after_assistant_tool_calls():
    """OpenAI-style: assistant.tool_calls followed by a bare user message is a
    tool result and must be skipped in favour of the real instruction."""
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "add unit tests for the auth module with mocking"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "run_tests", "arguments": "{}"}}]},
        {"role": "user", "content": '{"output": "3 tests passed, 1 failed"}'},
    ]
    assert classify_task(msgs) == "unit_tests"

def test_extract_intent_text_meta():
    """_extract_intent_text reports which source won and how many messages were
    skipped (feeds the future observability work)."""
    from src.api.router import _extract_intent_text
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "why does this test fail"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "[tool result] Ran 12 tests, 3 failed"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "continue"},
    ]
    text, meta = _extract_intent_text(msgs)
    assert text == "why does this test fail"
    assert meta["source"] == "last_instruction"
    assert meta["skipped_tool"] == 1
    assert meta["skipped_cont"] == 1

def test_extract_intent_text_pure_tool_conversation_falls_back():
    """Edge: a conversation with no genuine instruction falls back to the first
    user message instead of erroring."""
    from src.api.router import _extract_intent_text
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "[tool result] Ran 12 tests, 3 failed"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "[tool] command finished with exit code 1"},
    ]
    text, meta = _extract_intent_text(msgs)
    assert meta["source"] == "first_user_fallback"
    assert "Ran 12 tests" in text

def test_extract_intent_text_skips_attachment_context():
    """A trailing client-injected <attachments> wrapper must not become the
    intent — the real user instruction behind it wins."""
    from src.api.router import _extract_intent_text
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "write pytest tests for the router"},
        {"role": "assistant", "content": "on it"},
        {"role": "user", "content":
         "<attachments>\n<attachment id=\"Browser Pages\">\nNo browser pages "
         "are currently shared with you.\n</attachment>"},
    ]
    text, meta = _extract_intent_text(msgs)
    assert text == "write pytest tests for the router"
    assert meta["source"] == "last_instruction"
    assert meta["skipped_context"] == 1

def test_extract_intent_text_skips_context_and_tool_echoes():
    """Client-context wrappers and tool echoes both skipped; the walk finds
    the last genuine instruction."""
    from src.api.router import _extract_intent_text
    msgs = [
        {"role": "system", "content": "You are a coding agent. tools: terminal"},
        {"role": "user", "content": "debug this traceback"},
        {"role": "assistant", "content": "running",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}}]},
        {"role": "user", "content": "[tool result] Ran 12 tests, 3 failed", "tool_call_id": "c1"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "<attachments> <attachment id='X'> no pages"},
    ]
    text, meta = _extract_intent_text(msgs)
    assert text == "debug this traceback"
    assert meta["source"] == "last_instruction"
    assert meta["skipped_tool"] == 1
    assert meta["skipped_context"] == 1

def test_summarize_preserves_attachment_skip():
    """End-trimmed summary keeps the <attachments> prefix so replay skips it
    and the round-trip intent is unchanged."""
    from src.api.router import _summarize_conversation, _extract_intent_text
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "debug this traceback"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "<attachments> <attachment id='X'> " + "z" * 300},
    ]
    s = _summarize_conversation(msgs)
    assert _extract_intent_text(s)[0] == "debug this traceback"

def test_classify_agentic_system_prompt_only_still_agentic():
    """With no concrete user task, the agentic system prompt still wins."""
    msgs = [{"role": "system", "content": "You are an AI agent with tools: read_file, write_file"}]
    assert classify_task(msgs) == "agentic_multi_step"

def test_classify_system_prompt_planning_keywords_do_not_force_planning():
    """Regression: a system prompt full of incidental planning keywords (plan,
    strategy, recommend, suggest, best practice) must NOT classify a code
    request as planning. Classification must use conversation content, not the
    fixed agent preamble."""
    msgs = [
        {"role": "system", "content": (
            "You are a coding assistant. Plan your approach, recommend best "
            "practices, suggest a strategy, and design clean architecture.")},
        {"role": "user", "content": "implement a sorting algorithm in python"},
    ]
    assert classify_task(msgs) == "code_generation"

def test_classify_planning_keywords_in_user_message_win():
    """Planning keywords in the USER message (actual intent) still classify as
    planning even when the system prompt is generic."""
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "design the architecture for a microservice"},
    ]
    assert classify_task(msgs) == "planning"

def test_classify_planning():
    msgs = [{"role": "user", "content": "design the architecture for a microservice"}]
    assert classify_task(msgs) == "planning"

def test_classify_recommend_not_planning():
    """Bare 'recommend'/'suggest' must NOT force planning (too generic)."""
    msgs = [{"role": "user", "content": "recommend a good book to read"}]
    assert classify_task(msgs) != "planning"

def test_classify_implement_not_planning_even_with_long_max_tokens():
    """A code request with a large output budget must NOT be classified planning."""
    msgs = [{"role": "user", "content": "implement a sorting algorithm in python"}]
    assert classify_task(msgs, max_tokens=16384) == "code_generation"

def test_classify_long_max_tokens_no_longer_forces_planning():
    """Long max_tokens alone is not a planning signal."""
    msgs = [{"role": "user", "content": "write me a report about whales"}]
    assert classify_task(msgs, max_tokens=16384) != "planning"

def test_classify_casual():
    msgs = [{"role": "user", "content": "hello, how are you?"}]
    assert classify_task(msgs) == "casual_chat"

def test_classify_defaults_to_code():
    """A generic factual question is NOT routed to code once semantics are active.

    The semantic classifier (when sentence-transformers is installed) classifies
    by meaning — a factual query lands on reasoning_chain rather than the
    keyword default of code_generation.
    """
    msgs = [{"role": "user", "content": "what is the speed of light?"}]
    # Semantics active: meaning beats the code_generation keyword default.
    assert classify_task(msgs) == "reasoning_chain"

def test_classify_many_tools():
    """Tool-count still signals agentic, but semantic classification runs first.

    With the semantic layer active, a vague 'do something' is casual_chat even
    when many tools are present (semantics run before the tool-count heuristic).
    """
    tools = [{"type": "function", "function": {"name": f"t{i}"}} for i in range(6)]
    msgs = [{"role": "user", "content": "do something"}]
    # Semantics active: meaning beats the tool-count agentic heuristic.
    assert classify_task(msgs, tools=tools) == "casual_chat"

def test_classify_unit_tests():
    msgs = [{"role": "user", "content": "write unit tests for the payment module"}]
    assert classify_task(msgs) == "unit_tests"

def test_classify_pytest_suite():
    msgs = [{"role": "user", "content": "add a pytest test suite with mocks for the auth service"}]
    assert classify_task(msgs) == "unit_tests"

def test_classify_unit_tests_beats_code_gen_class_kw():
    # "class " is a code_generation signal, but unit tests must win first-match.
    msgs = [{"role": "user", "content": "add unit tests to this class"}]
    assert classify_task(msgs) == "unit_tests"

def test_classify_code_gen_not_unit_tests():
    msgs = [{"role": "user", "content": "implement a sorting algorithm in python"}]
    assert classify_task(msgs) == "code_generation"

def test_singleton():
    assert get_dynamic_router() is get_dynamic_router()

def test_init_router_passes_cost_bias(registry_db):
    init_router(registry_db, enabled=True, cost_bias=0.4)
    try:
        r = get_dynamic_router()
        assert r.enabled is True
        assert r.cost_bias == 0.4
    finally:
        # restore the default so other tests aren't affected
        init_router(enabled=False)

def test_sync_router_enabled_from_settings(registry_db, monkeypatch):
    """Boot seeds enabled=False from the yaml baseline; the persisted UI toggle
    must re-sync the global router to enabled=True."""
    init_router(registry_db, enabled=False)
    try:
        class FakeSettings:
            def get_routing_enabled(self, default=None):
                return True
        monkeypatch.setattr("src.api.cost_cache.get_settings", lambda: FakeSettings())
        from src.api.router import sync_router_enabled_from_settings
        assert sync_router_enabled_from_settings() is True
        assert get_dynamic_router().enabled is True
    finally:
        init_router(enabled=False)

def test_get_model_score_resolves_debugging_via_matrix(registry_db):
    """debugging is derived from code_generation, so scores must resolve."""
    from src.api.seed_capabilities import seed_livebench, load_capability_matrix
    seed_livebench(registry_db)
    router = CapabilityRouter(enabled=True, db_path=registry_db)
    matrix = load_capability_matrix(registry_db)
    assert "debugging" in matrix
    # A debugging prompt should resolve to a real score (not the 0.5 default).
    score = router.get_model_score("deepseek-v4-pro", "debugging")
    assert score > 0.5

def test_get_model_score_falls_back_for_unknown_model(registry_db):
    """A model with no debugging row falls back to the 0.5 default."""
    from src.api.seed_capabilities import seed_livebench
    seed_livebench(registry_db)
    router = CapabilityRouter(enabled=True, db_path=registry_db)
    assert router.get_model_score("some-unknown-model", "debugging") == 0.5

# ── Provider-aware selection (Phase 2) ───────────────────────────────────

def test_score_step_uses_capability_and_cost(registry_db):
    from src.api.seed_capabilities import seed_livebench
    seed_livebench(registry_db)
    router = CapabilityRouter(enabled=True, db_path=registry_db)
    # No breaker/cost-cache configured → pure capability + cost-bias boost.
    s = router.score_step({"provider": "opencode", "model": "deepseek-v4-pro",
                           "base_url": "https://opencode.ai"}, "reasoning_chain")
    assert s > 0.5
    # Same model on the same step scores identically without health input.
    s2 = router.score_step({"provider": "deepseek", "model": "deepseek-v4-pro",
                            "base_url": "https://deepseek.com"}, "reasoning_chain")
    assert abs(s - s2) < 1e-9  # health/credit tiebreakers both absent

def test_health_bonus_penalizes_degraded(registry_db):
    from src.api.seed_capabilities import seed_livebench
    from src.api.circuit_breaker import get_circuit_breaker
    seed_livebench(registry_db)
    router = CapabilityRouter(enabled=True, db_path=registry_db)

    class _Cfg:
        providers = {"deepseek": {"api_base": "https://deepseek.com"},
                     "opencode": {"api_base": "https://opencode.ai"}}
    cfg = _Cfg()
    breaker_cfg = {"failures_dead": 5, "dead_cooldown_seconds": 60,
                   "failures_degraded": 3, "degraded_cooldown_seconds": 60}
    get_circuit_breaker(breaker_cfg)

    healthy_step = {"provider": "deepseek", "model": "deepseek-v4-pro"}
    degraded_step = {"provider": "opencode", "model": "deepseek-v4-pro"}
    # Mark opencode degraded for this profile/base_url.
    get_circuit_breaker().get_health("opencode", "https://opencode.ai", "l2")["status"] = "degraded"

    sh = router.score_step(healthy_step, "reasoning_chain", profile="l2", config=cfg)
    sd = router.score_step(degraded_step, "reasoning_chain", profile="l2", config=cfg)
    assert sh > sd  # healthy provider step scores higher

def test_credit_bonus_penalizes_low_credits(registry_db, monkeypatch):
    from src.api.seed_capabilities import seed_livebench
    seed_livebench(registry_db)
    router = CapabilityRouter(enabled=True, db_path=registry_db)

    class FakeCache:
        def get(self, provider, kind):
            if provider == "commandcode" and kind == "subscription":
                return {"payload": {"monthly_credits_remaining": 1.0}}
            return None
    monkeypatch.setattr("src.api.cost_cache.get_cost_cache", lambda: FakeCache())

    step = {"provider": "commandcode", "model": "deepseek-v4-pro", "base_url": "x"}
    base = router.score_step(step, "reasoning_chain")
    # Same provider with plenty of credits scores higher (no penalty).
    class RichCache:
        def get(self, provider, kind):
            if provider == "commandcode" and kind == "subscription":
                return {"payload": {"monthly_credits_remaining": 50.0}}
            return None
    monkeypatch.setattr("src.api.cost_cache.get_cost_cache", lambda: RichCache())
    rich = router.score_step(step, "reasoning_chain")
    assert rich > base

def test_select_step_reorders_best_first(registry_db, monkeypatch):
    router = CapabilityRouter(enabled=True, db_path=registry_db)
    # Force known scores: second step clearly better.
    monkeypatch.setattr(router, "score_step",
                        lambda step, task, profile=None, config=None, bias=None:
                        0.9 if step["model"] == "deepseek-v4-pro" else 0.6)
    chain = [{"provider": "opencode", "model": "deepseek-v4-flash"},
             {"provider": "deepseek", "model": "deepseek-v4-pro"}]
    out = router.select_step([{"role": "user", "content": "hi"}], chain=chain)
    assert out is not None
    assert out[0]["model"] == "deepseek-v4-pro"  # best first

def test_select_step_keeps_order_within_hysteresis(registry_db, monkeypatch):
    router = CapabilityRouter(enabled=True, db_path=registry_db)
    # Both steps ~equal → no reorder (avoids flapping).
    monkeypatch.setattr(router, "score_step",
                        lambda step, task, profile=None, config=None, bias=None:
                        0.81 if step["model"] == "deepseek-v4-pro" else 0.80)
    chain = [{"provider": "opencode", "model": "deepseek-v4-flash"},
             {"provider": "deepseek", "model": "deepseek-v4-pro"}]
    assert router.select_step([{"role": "user", "content": "hi"}], chain=chain) is None

def test_select_step_disabled_or_empty_returns_none(registry_db):
    router = CapabilityRouter(enabled=False, db_path=registry_db)
    assert router.select_step([{"role": "user", "content": "hi"}], chain=[{"provider": "a", "model": "m"}]) is None
    router.enabled = True
    assert router.select_step([{"role": "user", "content": "hi"}], chain=[]) is None


# ── Policy + decisions + matrix invalidation (Phase 3) ───────────────────

def test_effective_policy_from_settings(registry_db, monkeypatch):
    router = CapabilityRouter(enabled=True, db_path=registry_db)

    class FakeSettings:
        def get_routing_policy(self, default="eager"):
            return "cost_first"
        def get_routing_min_score(self, default=0.0):
            return 0.5
    monkeypatch.setattr("src.api.cost_cache.get_settings", lambda: FakeSettings())
    policy, min_score = router._effective_policy(None)
    assert policy == "cost_first"
    assert min_score == 0.5

def test_effective_policy_config_fallback(registry_db, monkeypatch):
    router = CapabilityRouter(enabled=True, db_path=registry_db)
    # No runtime settings → config.dynamic_routing used.
    monkeypatch.setattr("src.api.cost_cache.get_settings", lambda: None)

    class _Cfg:
        dynamic_routing = {"policy": "explore", "min_score": 0.4}
    policy, min_score = router._effective_policy(_Cfg())
    assert policy == "explore"
    assert min_score == 0.4

def test_is_enabled_runtime_setting_wins(registry_db, monkeypatch):
    router = CapabilityRouter(enabled=False, db_path=registry_db)

    class FakeSettings:
        def get_routing_enabled(self, default=None):
            return True
    monkeypatch.setattr("src.api.cost_cache.get_settings", lambda: FakeSettings())
    assert router.is_enabled() is True

def test_is_enabled_falls_back_to_boot_value(registry_db, monkeypatch):
    monkeypatch.setattr("src.api.cost_cache.get_settings", lambda: None)
    assert CapabilityRouter(enabled=True, db_path=registry_db).is_enabled() is True
    assert CapabilityRouter(enabled=False, db_path=registry_db).is_enabled() is False

def test_select_step_disabled_via_runtime_toggle(registry_db, monkeypatch):
    """select_step honors the runtime disable even when the router boots enabled."""
    router = CapabilityRouter(enabled=True, db_path=registry_db)

    class FakeSettings:
        def get_routing_enabled(self, default=None):
            return False
    monkeypatch.setattr("src.api.cost_cache.get_settings", lambda: FakeSettings())
    chain = [{"provider": "opencode", "model": "deepseek-v4-pro"},
             {"provider": "deepseek", "model": "deepseek-v4-flash"}]
    assert router.select_step([{"role": "user", "content": "hi"}], chain=chain) is None

def test_select_step_min_score_floor(registry_db, monkeypatch):
    router = CapabilityRouter(enabled=True, db_path=registry_db)
    monkeypatch.setattr(router, "_effective_policy", lambda config: ("eager", 0.99))
    # Best score is ~0.9 → below the 0.99 floor → no reorder + decision recorded.
    monkeypatch.setattr(router, "score_step",
                        lambda step, task, profile=None, config=None, bias=None:
                        0.9 if step["model"] == "deepseek-v4-pro" else 0.6)
    chain = [{"provider": "opencode", "model": "deepseek-v4-flash"},
             {"provider": "deepseek", "model": "deepseek-v4-pro"}]
    assert router.select_step([{"role": "user", "content": "hi"}], chain=chain) is None
    assert router.recent_decisions()[-1]["action"] == "below_min_score"

def test_select_step_cost_first_policy(registry_db, monkeypatch):
    """cost_first boosts cheap models so a cheaper-but-close step can win."""
    router = CapabilityRouter(enabled=True, db_path=registry_db)
    monkeypatch.setattr(router, "_effective_policy", lambda config: ("cost_first", 0.0))
    monkeypatch.setattr(router, "score_step", router.score_step)  # real scoring
    chain = [{"provider": "opencode", "model": "deepseek-v4-pro"},
             {"provider": "deepseek", "model": "deepseek-v4-flash"}]
    # Flash is much cheaper; under cost_first it should rank first for a
    # task where the models are close (e.g. casual_chat).
    out = router.select_step(
        [{"role": "user", "content": "hello, how are you?"}], chain=chain)
    assert out is None or out[0]["model"] == "deepseek-v4-flash"

def test_select_step_explore_records_decision(registry_db, monkeypatch):
    router = CapabilityRouter(enabled=True, db_path=registry_db)
    monkeypatch.setattr(router, "_effective_policy", lambda config: ("explore", 0.0))
    monkeypatch.setattr(router, "score_step",
                        lambda step, task, profile=None, config=None, bias=None:
                        0.95 if step["model"] == "deepseek-v4-pro" else 0.93)
    chain = [{"provider": "opencode", "model": "deepseek-v4-flash"},
             {"provider": "deepseek", "model": "deepseek-v4-pro"}]
    out = router.select_step([{"role": "user", "content": "hi"}], chain=chain)
    # Either a reorder (explore) or keep_default — both record a decision.
    assert router.recent_decisions()
    if out is not None:
        assert out[0]["model"] == "deepseek-v4-pro"

def test_decision_buffer_bounded(registry_db, monkeypatch):
    router = CapabilityRouter(enabled=True, db_path=registry_db)
    monkeypatch.setattr(router, "_effective_policy", lambda config: ("eager", 0.0))
    monkeypatch.setattr(router, "score_step",
                        lambda step, task, profile=None, config=None, bias=None:
                        0.9 if step["model"] == "deepseek-v4-pro" else 0.6)
    chain = [{"provider": "opencode", "model": "deepseek-v4-flash"},
             {"provider": "deepseek", "model": "deepseek-v4-pro"}]
    for _ in range(60):
        router.select_step([{"role": "user", "content": "hi"}], chain=chain)
    assert len(router._decisions) <= 50

def test_invalidate_matrix(registry_db):
    router = CapabilityRouter(enabled=True, db_path=registry_db)
    router.load_matrix()
    assert router._matrix is not None
    router.invalidate_matrix()
    assert router._matrix is None
    # reloads on next access
    assert router.load_matrix() is not None

def test_routing_status(registry_db, monkeypatch):
    from src.api.router import routing_status, init_router
    init_router(registry_db, enabled=True)
    monkeypatch.setattr("src.api.cost_cache.get_settings", lambda: None)
    router = get_dynamic_router()
    router._record_decision({"ts": "t", "profile": "l2", "task": "debugging",
                             "policy": "eager", "action": "reorder",
                             "model": "m", "provider": "p", "score": 0.9})
    st = routing_status(None)
    assert st["enabled"] is True
    assert st["policy"] == "eager"
    assert "recent_decisions" in st
    assert st["recent_decisions"][0]["action"] == "reorder"
    init_router(enabled=False)  # restore


def test_decisions_persist_across_restart(registry_db):
    """Routing decisions survive a rebuild: a fresh router on the same DB reads
    back the persisted routing_decisions rows."""
    from src.api.models import Base, get_engine, RoutingDecision, get_session

    # Ensure the routing_decisions table exists on the shared registry DB.
    engine = get_engine(registry_db)
    Base.metadata.create_all(engine)

    r1 = CapabilityRouter(enabled=True, db_path=registry_db)
    r1._record_decision({
        "ts": "2026-08-25T20:59:15Z", "profile": "coder", "task": "planning",
        "policy": "eager", "action": "prefer", "provider": "commandcode",
        "model": "deepseek/deepseek-v4-flash", "score": 0.778,
        "rules": ["prefer"], "note": "fired: prefer",
    })

    # Simulate a rebuild: a FRESH router instance on the SAME db_path.
    r2 = CapabilityRouter(enabled=True, db_path=registry_db)
    decs = r2.recent_decisions(25)
    assert any(d["task"] == "planning" and d["action"] == "prefer" for d in decs)
    # The row is actually in the DB table.
    with get_session(engine) as s:
        assert s.query(RoutingDecision).filter(RoutingDecision.task == "planning").count() >= 1


# ── Routing rules (Phase: UI-defined overrides) ─────────────────────────

def _rule_router(registry_db, monkeypatch, rules, config=None):
    router = CapabilityRouter(enabled=True, db_path=registry_db)

    class FakeSettings:
        def get_routing_rules(self):
            return rules
        def get_routing_policy(self, default="eager"):
            return "eager"
        def get_routing_min_score(self, default=0.0):
            return 0.0
    monkeypatch.setattr("src.api.cost_cache.get_settings", lambda: FakeSettings())
    return router


def _prov_config(providers):
    """A minimal config stub with provider model lists (for prefer expansion)."""
    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.providers = providers
    return cfg


REAL_PROVIDERS = {
    "commandcode": {"models": ["deepseek-v4-pro", "deepseek-v4-flash"]},
    "deepseek": {"models": ["deepseek-v4-pro", "deepseek-v4-flash"]},
    "opencode": {"models": ["deepseek-v4-pro", "deepseek-v4-flash", "ox-alpha-free"]},
}

def test_apply_rules_block_removes_provider(registry_db, monkeypatch):
    router = _rule_router(registry_db, monkeypatch,
                          [{"task": "*", "action": "block", "provider": "opencode"}])
    chain = [{"provider": "opencode", "model": "m1"},
             {"provider": "deepseek", "model": "m2"}]
    candidates, fired = router._apply_rules(chain, "code_generation", "l2")
    assert [s["provider"] for s in candidates] == ["deepseek"]
    assert fired and fired[0]["action"] == "block"

def test_apply_rules_prefer_moves_to_front(registry_db, monkeypatch):
    router = _rule_router(registry_db, monkeypatch,
                          [{"task": "debugging", "action": "prefer",
                            "provider": "deepseek", "model": "deepseek-v4-pro"}])
    chain = [{"provider": "opencode", "model": "deepseek-v4-flash"},
             {"provider": "deepseek", "model": "deepseek-v4-pro"}]
    candidates, fired = router._apply_rules(chain, "debugging", "l2")
    assert candidates[0]["provider"] == "deepseek"
    assert fired[0]["action"] == "prefer"
    # Non-matching task → no change.
    candidates2, _ = router._apply_rules(chain, "casual_chat", "l2")
    assert candidates2[0]["provider"] == "opencode"

def test_apply_rules_prefer_min_score_gate(registry_db, monkeypatch):
    router = _rule_router(registry_db, monkeypatch,
                          [{"task": "debugging", "action": "prefer",
                            "provider": "deepseek", "model": "deepseek-v4-pro",
                            "min_score": 0.99}])
    chain = [{"provider": "opencode", "model": "deepseek-v4-flash"},
             {"provider": "deepseek", "model": "deepseek-v4-pro"}]
    # deepseek-v4-pro's debugging score is ~0.77 < 0.99 → prefer skipped.
    candidates, fired = router._apply_rules(chain, "debugging", "l2")
    assert candidates[0]["provider"] == "opencode"
    assert fired and fired[0]["action"] == "prefer_skipped_low_score"

def test_apply_rules_model_only_prefer(registry_db, monkeypatch):
    """A rule with only a model (provider '*' wildcard) expands to every
    provider that serves the model, in chain order."""
    router = _rule_router(registry_db, monkeypatch,
                          [{"task": "*", "action": "prefer",
                            "provider": "*", "model": "deepseek-v4-pro"}])
    chain = [{"provider": "opencode", "model": "deepseek-v4-flash"},
             {"provider": "deepseek", "model": "deepseek-v4-pro"}]
    candidates, fired = router._apply_rules(chain, "code_generation", "l2",
                                            _prov_config(REAL_PROVIDERS))
    assert candidates[0]["model"] == "deepseek-v4-pro"
    assert candidates[0]["provider"] == "opencode"  # chain order: opencode first
    assert fired and fired[0]["action"] == "prefer"

def test_apply_rules_model_only_block(registry_db, monkeypatch):
    """A block rule with only a model removes it from every provider."""
    router = _rule_router(registry_db, monkeypatch,
                          [{"task": "*", "action": "block",
                            "provider": "*", "model": "deepseek-v4-flash"}])
    chain = [{"provider": "opencode", "model": "deepseek-v4-flash"},
             {"provider": "deepseek", "model": "deepseek-v4-pro"}]
    candidates, fired = router._apply_rules(chain, "code_generation", "l2")
    assert [s["model"] for s in candidates] == ["deepseek-v4-pro"]
    assert fired and fired[0]["action"] == "block"

def test_apply_rules_both_wildcards_matches_nothing(registry_db, monkeypatch):
    """provider='*' AND model='*' has no concrete target → no-op."""
    router = _rule_router(registry_db, monkeypatch,
                          [{"task": "*", "action": "prefer",
                            "provider": "*", "model": "*"}])
    chain = [{"provider": "opencode", "model": "deepseek-v4-flash"},
             {"provider": "deepseek", "model": "deepseek-v4-pro"}]
    candidates, fired = router._apply_rules(chain, "code_generation", "l2")
    assert candidates[0]["provider"] == "opencode"
    assert not fired

def test_rule_target_normalizes_provider_side_model_id(registry_db, monkeypatch):
    """A rule written with the logical model name matches a chain step whose
    model is a provider-side ID (commandcode: deepseek/deepseek-v4-pro)."""
    router = _rule_router(registry_db, monkeypatch, [])
    rule = {"provider": "*", "model": "deepseek-v4-pro"}
    step = {"provider": "commandcode", "model": "deepseek/deepseek-v4-pro"}
    assert router._rule_target(rule, step) is True

def test_apply_rules_prefer_first_match_wins(registry_db, monkeypatch):
    """The first matching prefer (in rule order) wins; later prefers don't
    override it."""
    router = _rule_router(registry_db, monkeypatch, [
        {"task": "*", "action": "prefer", "provider": "*", "model": "deepseek-v4-pro"},
        {"task": "*", "action": "prefer", "provider": "*", "model": "deepseek-v4-flash"},
    ])
    chain = [{"provider": "opencode", "model": "deepseek-v4-flash"},
             {"provider": "deepseek", "model": "deepseek-v4-pro"}]
    candidates, fired = router._apply_rules(chain, "code_generation", "l2")
    assert candidates[0]["model"] == "deepseek-v4-pro"
    assert [f["action"] for f in fired] == ["prefer"]

def test_apply_rules_prefer_expands_across_providers(registry_db, monkeypatch):
    """A prefer rule expands the preferred model to EVERY provider that serves
    it, in chain order, before the chain's other models. So a degraded provider
    falls to the NEXT provider of the SAME model — not to a cheaper one."""
    router = _rule_router(registry_db, monkeypatch,
                          [{"task": "*", "action": "prefer",
                            "provider": "*", "model": "deepseek-v4-pro"}])
    chain = [
        {"provider": "commandcode", "model": "deepseek/deepseek-v4-pro"},
        {"provider": "deepseek", "model": "deepseek-v4-flash"},
        {"provider": "opencode", "model": "deepseek-v4-pro"},
        {"provider": "opencode", "model": "ox-alpha-free"},
    ]
    candidates, fired = router._apply_rules(chain, "planning", "coder",
                                            _prov_config(REAL_PROVIDERS))
    models = [c["model"] for c in candidates]
    # commandcode-pro, deepseek-pro, opencode-pro, then fallbacks.
    assert models == ["deepseek/deepseek-v4-pro", "deepseek-v4-pro",
                      "deepseek-v4-pro", "deepseek-v4-flash", "ox-alpha-free"]
    assert [c["provider"] for c in candidates] == [
        "commandcode", "deepseek", "opencode", "deepseek", "opencode",
    ]
    assert fired[0]["action"] == "prefer"
    assert fired[0].get("steps") == 3

def test_select_step_prefer_is_mandatory(registry_db, monkeypatch):
    """A fired prefer expands the preferred model across providers: even if
    scoring favors flash, the router returns the preferred model first, tried on
    every provider that serves it, before any other model."""
    rules = [{"task": "planning", "profile": "*", "action": "prefer",
              "provider": "*", "model": "deepseek-v4-pro"}]
    router = _rule_router(registry_db, monkeypatch, rules)
    monkeypatch.setattr(router, "_effective_policy", lambda config: ("eager", 0.0))
    # Flash would outscore pro on planning — prefer must still win.
    monkeypatch.setattr(router, "score_step",
                        lambda step, task, profile=None, config=None, bias=None:
                        0.9 if step["model"] == "deepseek-v4-flash" else 0.7)
    chain = [{"provider": "deepseek", "model": "deepseek-v4-flash"},
             {"provider": "commandcode", "model": "deepseek/deepseek-v4-pro"}]
    cfg = _prov_config(REAL_PROVIDERS)
    out = router.select_step([{"role": "user", "content": "design the architecture"}],
                             chain=chain, profile="coder", config=cfg)
    assert out is not None
    # pro first on deepseek (chain order), then pro on commandcode, then flash.
    assert out[0]["provider"] == "deepseek"
    assert out[0]["model"] == "deepseek-v4-pro"
    assert out[1]["provider"] == "commandcode"
    assert out[1]["model"] == "deepseek/deepseek-v4-pro"
    assert out[2]["model"] == "deepseek-v4-flash"
    dec = router.recent_decisions()[-1]
    assert dec["action"] == "prefer"
    assert dec["rules"] == ["prefer"]
    assert dec["task"] == "planning"

def test_select_step_prefer_gate_skip_still_scores(registry_db, monkeypatch):
    """When a prefer is skipped by its min_score gate (no prefer fires), the
    router falls through to normal scoring."""
    rules = [{"task": "planning", "profile": "*", "action": "prefer",
              "provider": "*", "model": "deepseek-v4-pro", "min_score": 0.99}]
    router = _rule_router(registry_db, monkeypatch, rules)
    monkeypatch.setattr(router, "_effective_policy", lambda config: ("eager", 0.0))
    monkeypatch.setattr(router, "score_step",
                        lambda step, task, profile=None, config=None, bias=None:
                        0.9 if step["model"] == "deepseek-v4-flash" else 0.7)
    chain = [{"provider": "deepseek", "model": "deepseek-v4-flash"},
             {"provider": "commandcode", "model": "deepseek/deepseek-v4-pro"}]
    out = router.select_step([{"role": "user", "content": "design the architecture"}],
                             chain=chain, profile="coder")
    # No prefer fired → eager keeps the default (flash is already first/highest).
    # The router returns None to keep the chain order.
    assert out is None
    dec = router.recent_decisions()[-1]
    assert dec["rules"] == ["prefer_skipped_low_score"]
    assert dec["action"] == "keep_default"

def test_select_step_agentic_system_prompt_still_fires_planning_prefer(registry_db, monkeypatch):
    """Regression: an agentic system prompt must not mask a planning request —
    the planning prefer rule must fire and pin pro even though the system
    prompt is agentic and flash scores higher on planning."""
    rules = [{"task": "planning", "profile": "*", "action": "prefer",
              "provider": "*", "model": "deepseek-v4-pro"}]
    router = _rule_router(registry_db, monkeypatch, rules)
    monkeypatch.setattr(router, "_effective_policy", lambda config: ("eager", 0.0))
    # Flash out-scores pro on planning — prefer must still win.
    monkeypatch.setattr(router, "score_step",
                        lambda step, task, profile=None, config=None, bias=None:
                        0.9 if step["model"] == "deepseek-v4-flash" else 0.7)
    chain = [{"provider": "deepseek", "model": "deepseek-v4-flash"},
             {"provider": "commandcode", "model": "deepseek/deepseek-v4-pro"}]
    msgs = [
        {"role": "system", "content": "You are an AI agent with tools: read_file, write_file, terminal"},
        {"role": "user", "content": "design the architecture for a microservice"},
    ]
    out = router.select_step(msgs, chain=chain, profile="coder",
                             config=_prov_config(REAL_PROVIDERS))
    assert out is not None
    # Preferred model expanded: deepseek-pro first (chain order), not flash.
    assert out[0]["model"] == "deepseek-v4-pro"
    dec = router.recent_decisions()[-1]
    assert dec["task"] == "planning"
    assert dec["action"] == "prefer"
    assert dec["rules"] == ["prefer"]

def test_select_step_filters_unavailable_provider(registry_db, monkeypatch):
    """The circuit breaker only gates PROVIDERS: when a provider is unavailable,
    the router drops its steps entirely, so the ordering never proposes it — and
    a prefer rule still expands the preferred model across the remaining
    providers."""
    rules = [{"task": "planning", "profile": "*", "action": "prefer",
              "provider": "*", "model": "deepseek-v4-pro"}]
    router = _rule_router(registry_db, monkeypatch, rules)
    monkeypatch.setattr(router, "_effective_policy", lambda config: ("eager", 0.0))
    # commandcode unavailable → its pro step is dropped; deepseek + opencode pro survive.
    monkeypatch.setattr(router, "_provider_available",
                        lambda step, profile=None, config=None:
                        step["provider"] != "commandcode")
    chain = [
        {"provider": "commandcode", "model": "deepseek/deepseek-v4-pro"},
        {"provider": "deepseek", "model": "deepseek-v4-flash"},
        {"provider": "opencode", "model": "deepseek-v4-pro"},
    ]
    msgs = [{"role": "user", "content": "design the architecture"}]
    out = router.select_step(msgs, chain=chain, profile="coder",
                             config=_prov_config(REAL_PROVIDERS))
    assert out is not None
    # commandcode dropped; pro expanded to deepseek + opencode, then flash.
    assert [s["provider"] for s in out] == ["deepseek", "opencode", "deepseek"]
    assert [s["model"] for s in out] == ["deepseek-v4-pro", "deepseek-v4-pro", "deepseek-v4-flash"]
    assert router.recent_decisions()[-1]["action"] == "prefer"

def test_select_step_audit_note_when_rule_matches_scope_but_not_fired(registry_db, monkeypatch):
    """When a rule matches the task scope but no action fires (e.g. the only
    matching rule is a policy override), the decision carries an audit note."""
    rules = [{"profile": "cron", "action": "policy", "policy": "cost_first"}]
    router = _rule_router(registry_db, monkeypatch, rules)
    monkeypatch.setattr(router, "_effective_policy", lambda config: ("eager", 0.0))
    monkeypatch.setattr(router, "score_step",
                        lambda step, task, profile=None, config=None, bias=None:
                        0.9 if step["model"] == "deepseek-v4-pro" else 0.6)
    chain = [{"provider": "opencode", "model": "deepseek-v4-flash"},
             {"provider": "deepseek", "model": "deepseek-v4-pro"}]
    # cron profile → policy rule flips policy but doesn't "fire" a prefer/block.
    router.select_step([{"role": "user", "content": "hi"}], chain=chain, profile="cron")
    dec = router.recent_decisions()[-1]
    assert dec["note"] is not None
    assert "matched scope" in dec["note"]

def test_select_step_policy_rule_override(registry_db, monkeypatch):
    rules = [{"profile": "cron", "action": "policy", "policy": "cost_first"}]
    router = _rule_router(registry_db, monkeypatch, rules)
    monkeypatch.setattr(router, "_effective_policy", lambda config: ("eager", 0.0))
    monkeypatch.setattr(router, "score_step",
                        lambda step, task, profile=None, config=None, bias=None:
                        0.9 if step["model"] == "deepseek-v4-pro" else 0.6)
    chain = [{"provider": "opencode", "model": "deepseek-v4-flash"},
             {"provider": "deepseek", "model": "deepseek-v4-pro"}]
    # cron profile → policy rule flips to cost_first (bias > 0) but the outcome
    # is a reorder regardless; assert the decision recorded the policy.
    router.select_step([{"role": "user", "content": "hi"}], chain=chain, profile="cron")
    assert router.recent_decisions()[-1]["policy"] == "cost_first"

def test_routing_status_includes_rules(registry_db, monkeypatch):
    from src.api.router import routing_status, init_router
    rules = [{"task": "debugging", "action": "prefer", "provider": "deepseek"}]
    init_router(registry_db, enabled=True)
    try:
        _rule_router(registry_db, monkeypatch, rules)
        st = routing_status(None)
        assert st["rules"] == rules
    finally:
        init_router(enabled=False)

# ── Per-profile routing (enable/policy/min_score/rules overrides) ─────────

class _PerProfileSettings:
    """A duck-typed settings store that supports per-profile routing keys."""

    def __init__(self, store):
        self._store = store

    def get_routing_enabled(self, default=None, profile=None):
        return self._store.get_routing_enabled(default=default, profile=profile)

    def get_routing_policy(self, default="eager", profile=None):
        return self._store.get_routing_policy(default=default, profile=profile)

    def get_routing_min_score(self, default=0.0, profile=None):
        return self._store.get_routing_min_score(default=default, profile=profile)

    def get_routing_rules(self, default=None, profile=None):
        return self._store.get_routing_rules(default=default, profile=profile)


def test_per_profile_enabled_wins_over_global(registry_db, monkeypatch):
    import tempfile, os
    from src.api.cost_cache import SettingsStore
    from src.api.models import get_engine, Base
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = get_engine(path)
    Base.metadata.create_all(engine)
    try:
        store = SettingsStore(engine)
        store.set_routing_enabled(False)            # global off
        store.set_routing_enabled(True, profile="l2")  # l2 on
        monkeypatch.setattr("src.api.cost_cache.get_settings", lambda: store)

        router = CapabilityRouter(enabled=True, db_path=registry_db)
        assert router.is_enabled(None) is False          # global off
        assert router.is_enabled(None, "l2") is True     # l2 override on
        assert router.is_enabled(None, "career") is False  # falls back to global
    finally:
        engine.dispose()
        for ext in ("", "-wal", "-shm"):
            try:
                os.unlink(path + ext)
            except FileNotFoundError:
                pass


def test_per_profile_policy_and_rules(registry_db, monkeypatch):
    import tempfile, os
    from src.api.cost_cache import SettingsStore
    from src.api.models import get_engine, Base
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = get_engine(path)
    Base.metadata.create_all(engine)
    try:
        store = SettingsStore(engine)
        store.set_routing_policy("cost_first")                     # global
        store.set_routing_policy("explore", profile="l2")          # l2 override
        store.set_routing_rules([{"task": "planning", "action": "prefer",
                                  "model": "deepseek-v4-pro"}], profile="l2")
        monkeypatch.setattr("src.api.cost_cache.get_settings", lambda: store)

        router = CapabilityRouter(enabled=True, db_path=registry_db)
        pol, ms = router._effective_policy(None, "l2")
        assert pol == "explore"
        pol_g, _ = router._effective_policy(None)
        assert pol_g == "cost_first"
        assert len(router._rules(None, "l2")) == 1
        assert router._rules(None) == []  # global rules untouched
    finally:
        engine.dispose()
        for ext in ("", "-wal", "-shm"):
            try:
                os.unlink(path + ext)
            except FileNotFoundError:
                pass


def test_routing_status_includes_per_profile(registry_db, monkeypatch):
    from src.api.router import routing_status, init_router, _status_for_profile
    init_router(registry_db, enabled=True)
    monkeypatch.setattr("src.api.cost_cache.get_settings", lambda: None)
    try:
        class _Cfg:
            profiles = {"l2": {"chain": [{"provider": "deepseek", "model": "deepseek-v4-pro"}]},
                        "career": {"chain": [{"provider": "deepseek", "model": "deepseek-v4-flash"}]}}
            dynamic_routing = {}
        st = routing_status(_Cfg())
        assert "per_profile" in st
        assert set(st["per_profile"].keys()) == {"l2", "career"}
        assert st["per_profile"]["l2"]["enabled"] is True
        # _status_for_profile returns the same shape for a direct profile query.
        router = get_dynamic_router()
        block = _status_for_profile(router, _Cfg(), "l2")
        assert block["enabled"] is True
        assert "rules" in block
    finally:
        init_router(enabled=False)


def test_routing_status_restricts_to_selected_models(registry_db, monkeypatch):
    """Per-task recommendations only include models referenced by a chain."""
    from src.api.seed_capabilities import seed_livebench
    from src.api.router import routing_status, init_router
    seed_livebench(registry_db)
    init_router(registry_db, enabled=True)
    try:
        # Chain selects ONLY deepseek-v4-flash — recommendations must not
        # mention models like gpt-5.6-sol / claude-fable-5 that aren't selected.
        class _Cfg:
            profiles = {"l2": {"chain": [
                {"provider": "deepseek", "model": "deepseek-v4-flash"},
            ]}}
            dynamic_routing = {}
        st = routing_status(_Cfg())
        for task, rec in st["per_task"].items():
            assert rec["model"] == "deepseek-v4-flash", f"{task} -> {rec['model']} (not selected)"
    finally:
        init_router(enabled=False)

def test_routing_status_falls_back_without_config(registry_db, monkeypatch):
    """Without config (tests), the top model per task is still shown."""
    from src.api.seed_capabilities import seed_livebench
    from src.api.router import routing_status, init_router
    seed_livebench(registry_db)
    init_router(registry_db, enabled=True)
    try:
        st = routing_status(None)
        assert st["per_task"], "expected per_task populated without config"
        # The top overall model appears for at least one task.
        assert any(rec["model"] for rec in st["per_task"].values())
    finally:
        init_router(enabled=False)


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
    from src.api.seed_capabilities import load_model_registry
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


# ── Observability + judgment: classify_task_detail, summarize, rationale ────

def test_classify_task_detail_keyword_path():
    from src.api.router import classify_task_detail
    msgs = [{"role": "user", "content": "debug why this returns a TypeError"}]
    detail = classify_task_detail(msgs)
    assert detail.task == "debugging"
    assert detail.path == "keyword:debugging"
    assert detail.keyword and detail.keyword in ("debug", "error", "exception")
    assert "TypeError" in detail.intent_text
    assert detail.intent_meta is not None


def test_classify_task_detail_default_path(monkeypatch):
    # Deterministic: disable the semantic path so we exercise the fallback
    # chain (with the real embedder installed, semantics usually win).
    from src.api.router import classify_task_detail
    from src.api import task_classifier as tc
    monkeypatch.setattr(tc, "get_semantic_classifier", lambda: None)
    msgs = [{"role": "user", "content": "review the current state of things in this workspace"}]
    detail = classify_task_detail(msgs)
    assert detail.task == "code_generation"
    assert detail.path == "default"
    assert detail.semantic is None
    assert detail.sem_available is False


def test_classify_task_detail_agentic_prompt_path():
    from src.api.router import classify_task_detail
    msgs = [{"role": "system",
             "content": "You are an AI agent with tools: read_file, write_file, terminal, patch, bash, curl"}]
    detail = classify_task_detail(msgs)
    assert detail.task == "agentic_multi_step"
    assert detail.path == "agentic_prompt"


def test_classify_task_detail_semantic_path(monkeypatch):
    from src.api.router import classify_task_detail
    from src.api import task_classifier as tc

    class FakeClf:
        min_score = 0.3
        def top_scores(self, text, k=5):
            return [("unit_tests", 0.9), ("code_generation", 0.4)]

    monkeypatch.setattr(tc, "get_semantic_classifier", lambda: FakeClf())
    detail = classify_task_detail([{"role": "user", "content": "zzz qqq some tests"}])
    assert detail.task == "unit_tests"
    assert detail.path == "semantic"
    assert detail.sem_available is True
    assert detail.min_score == 0.3
    assert detail.semantic == [("unit_tests", 0.9), ("code_generation", 0.4)]


def test_classify_task_is_thin_wrapper():
    from src.api.router import classify_task_detail
    msgs = [{"role": "user", "content": "debug this traceback please"}]
    assert classify_task(msgs) == classify_task_detail(msgs).task


def test_summarize_conversation_preserves_tool_shape():
    from src.api.router import _summarize_conversation
    msgs = [
        {"role": "system", "content": "You are a coding agent. tools: terminal, patch"},
        {"role": "user", "content": "debug this failing test"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "run_tests", "arguments": '{"cmd": "pytest"}'}}]},
        {"role": "user", "content": "[tool result] Ran 12 tests, 3 failed. " + "x" * 500,
         "tool_call_id": "call_1"},
    ]
    s = _summarize_conversation(msgs)
    assert [m["role"] for m in s] == ["system", "user", "assistant", "user"]
    assert s[2]["tool_calls"][0]["id"] == "call_1"
    assert s[2]["tool_calls"][0]["function"]["name"] == "run_tests"
    assert s[3]["tool_call_id"] == "call_1"
    # Trimmed from the END keeps the leading tool-result marker.
    assert s[3]["content"].startswith("[tool result]")
    assert "… [" in s[3]["content"] and "chars omitted]" in s[3]["content"]


def test_summarize_conversation_roundtrip_intent():
    from src.api.router import _summarize_conversation, _extract_intent_text
    msgs = [
        {"role": "system", "content": "You are a coding agent. tools: terminal"},
        {"role": "user", "content": "debug this error in the code"},
        {"role": "assistant", "content": "Let me reproduce.",
         "tool_calls": [{"id": "call_9", "type": "function", "function": {"name": "run", "arguments": "{}"}}]},
        {"role": "user", "content": "[tool result] Ran 12 tests, 3 failed", "tool_call_id": "call_9"},
    ]
    summarized = _summarize_conversation(msgs)
    assert _extract_intent_text(msgs) == _extract_intent_text(summarized)


def test_summarize_conversation_drops_oldest_when_huge():
    import json
    from src.api.router import _summarize_conversation
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "y" * 2000},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "write pytest tests now"},
    ]
    s = _summarize_conversation(msgs, max_content=50, max_total=800)
    blob = json.dumps(s)
    assert "pytest tests" in blob  # newest intent survives
    assert any(isinstance(m.get("content"), str) and "omitted" in m["content"] for m in s)


def test_record_decision_persists_rationale_and_conversation(registry_db):
    import json
    from src.api.router import CapabilityRouter, classify_task_detail, _extract_intent_text
    router = CapabilityRouter(enabled=True, db_path=registry_db)
    msgs = [{"role": "user", "content": "debug this traceback"}]
    detail = classify_task_detail(msgs)
    router._record_decision({
        **router._decision_base(detail, msgs, "l2", "eager", "note here", ["prefer"]),
        "action": "prefer", "model": "deepseek-v4-pro", "provider": "commandcode",
        "score": 0.85,
    })
    decs = router.recent_decisions(5)
    assert decs and decs[0]["task"] == "debugging"
    assert decs[0]["path"] == "keyword:debugging"
    assert decs[0]["keyword"] and decs[0]["intent_text"]
    assert decs[0]["note"] == "note here"
    assert decs[0]["profile"] == "l2"
    assert decs[0]["conversation_json"]
    conv = json.loads(decs[0]["conversation_json"])
    assert _extract_intent_text(conv)[0] == "debug this traceback"


# ── Deterministic routing: rule path == scoring path (unified resolver) ─────

def test_flash_prefer_and_scoring_agree(registry_db, monkeypatch):
    """HEADLINE: the prefer (rule) path and the scoring (reorder) path must
    resolve the SAME provider for the same target model. Regression for the
    live divergence: prefer picked commandcode/…flash while reorder picked
    deepseek/…flash."""
    chain = [
        {"provider": "commandcode", "model": "deepseek/deepseek-v4-pro"},
        {"provider": "deepseek", "model": "deepseek-v4-flash"},
        {"provider": "opencode", "model": "deepseek-v4-pro"},
        {"provider": "opencode", "model": "ox-alpha-free"},
    ]
    cfg = _prov_config(REAL_PROVIDERS)

    # Rule path: a prefer-flash rule fires for code_generation.
    rule_router = _rule_router(
        registry_db, monkeypatch,
        [{"task": "code_generation", "profile": "*", "action": "prefer",
          "provider": "*", "model": "deepseek-v4-flash"}])
    rule_router._effective_policy = lambda config: ("eager", 0.0)
    rule_out = rule_router.select_step(
        [{"role": "user", "content": "write a helper script in python"}],
        chain=chain, profile="coder", config=cfg)
    assert rule_out is not None

    # Scoring path: no prefer rule; flash out-scores pro so scoring picks flash.
    score_router = _rule_router(registry_db, monkeypatch, [])
    score_router._effective_policy = lambda config: ("eager", 0.0)
    score_router.score_step = (
        lambda step, task, profile=None, config=None, bias=None:
        0.9 if step["model"] == "deepseek-v4-flash" else
        0.7 if step["model"] == "deepseek-v4-pro" else 0.4)
    score_out = score_router.select_step(
        [{"role": "user", "content": "this endpoint returns a 500"}],
        chain=chain, profile="coder", config=cfg)
    assert score_out is not None

    # BOTH must resolve the target model on the SAME first provider (chain order).
    assert (rule_out[0]["provider"], rule_out[0]["model"]) == \
           (score_out[0]["provider"], score_out[0]["model"])
    assert rule_out[0]["provider"] == "commandcode"
    assert rule_out[0]["model"] == "deepseek/deepseek-v4-flash"


def test_degraded_target_provider_falls_to_healthy_same_model(registry_db, monkeypatch):
    """Among providers serving the target model, healthy providers lead a
    degraded one — but the model stays the same (no cheaper-model fallback)."""
    from src.api.circuit_breaker import get_circuit_breaker
    get_circuit_breaker({"failures_dead": 5, "dead_cooldown_seconds": 60,
                         "failures_degraded": 3, "degraded_cooldown_seconds": 60})
    get_circuit_breaker().get_health("commandcode", "https://cmd.example/v1", "l2")["status"] = "degraded"
    rules = [{"task": "*", "action": "prefer", "provider": "*", "model": "deepseek-v4-flash"}]
    router = _rule_router(registry_db, monkeypatch, rules)
    router._effective_policy = lambda config: ("eager", 0.0)
    chain = [
        {"provider": "commandcode", "model": "deepseek/deepseek-v4-pro",
         "base_url": "https://cmd.example/v1"},
        {"provider": "deepseek", "model": "deepseek-v4-flash"},
        {"provider": "opencode", "model": "deepseek-v4-pro"},
    ]
    out = router.select_step([{"role": "user", "content": "design the architecture"}],
                             chain=chain, profile="l2", config=_prov_config(REAL_PROVIDERS))
    assert out is not None
    # Flash target expanded across providers; healthy deepseek/opencode lead
    # the degraded commandcode.
    assert [s["provider"] for s in out[:3]] == ["deepseek", "opencode", "commandcode"]
    assert out[0]["model"] == "deepseek-v4-flash"


def test_provider_only_prefer_is_tiebreak_not_mandate(registry_db, monkeypatch):
    """A provider-only prefer does NOT force a model — it is a provider
    tiebreak: the model is still chosen by scoring, and the provider leads only
    because it serves the chosen model."""
    rules = [{"task": "*", "action": "prefer", "provider": "opencode"}]
    router = _rule_router(registry_db, monkeypatch, rules)
    router._effective_policy = lambda config: ("eager", 0.0)
    router.score_step = (
        lambda step, task, profile=None, config=None, bias=None:
        0.9 if step["model"] == "deepseek-v4-flash" else 0.7)
    chain = [
        {"provider": "commandcode", "model": "deepseek/deepseek-v4-pro"},
        {"provider": "deepseek", "model": "deepseek-v4-flash"},
        {"provider": "opencode", "model": "deepseek-v4-pro"},
    ]
    out = router.select_step([{"role": "user", "content": "hi"}], chain=chain,
                             config=_prov_config(REAL_PROVIDERS))
    assert out is not None
    # Model still chosen by scoring = flash; opencode serves flash so it leads.
    assert out[0]["model"] == "deepseek-v4-flash"
    assert out[0]["provider"] == "opencode"
    assert router.recent_decisions()[-1]["action"] == "prefer"


def test_prefer_model_served_by_no_provider_falls_through(registry_db, monkeypatch):
    """A prefer whose model no provider serves must fall through to scoring
    (no provider is force-picked), with an audit trail."""
    rules = [{"task": "*", "action": "prefer", "provider": "*",
              "model": "does-not-exist-9000"}]
    router = _rule_router(registry_db, monkeypatch, rules)
    router._effective_policy = lambda config: ("eager", 0.0)
    router.score_step = (
        lambda step, task, profile=None, config=None, bias=None:
        0.9 if step["model"] == "deepseek-v4-flash" else 0.7)
    chain = [
        {"provider": "commandcode", "model": "deepseek/deepseek-v4-pro"},
        {"provider": "deepseek", "model": "deepseek-v4-flash"},
        {"provider": "opencode", "model": "deepseek-v4-pro"},
    ]
    out = router.select_step([{"role": "user", "content": "hi"}], chain=chain,
                             config=_prov_config(REAL_PROVIDERS))
    assert out is not None
    # Scoring picks flash (no prefer mandate) on the first serving provider.
    from src.api.router import logical_model_name
    assert logical_model_name(out[0]["model"], registry_db) == "deepseek-v4-flash"
    dec = router.recent_decisions()[-1]
    assert dec["action"] == "reorder"
    assert "prefer_unserved" in dec["rules"]


def test_build_chain_for_model_deterministic_order(registry_db):
    """The shared chain-builder emits one target-model step per serving
    provider (chain order), then non-target steps as fallbacks."""
    from src.api.router import CapabilityRouter, provider_model_name
    router = CapabilityRouter(enabled=True, db_path=registry_db)
    chain = [
        {"provider": "commandcode", "model": "deepseek/deepseek-v4-pro"},
        {"provider": "deepseek", "model": "deepseek-v4-flash"},
        {"provider": "opencode", "model": "deepseek-v4-pro"},
        {"provider": "opencode", "model": "ox-alpha-free"},
    ]
    out = router._build_chain_for_model(chain, "deepseek-v4-flash",
                                        config=_prov_config(REAL_PROVIDERS))
    heads = out[:3]
    assert [(s["provider"], s["model"]) for s in heads] == [
        ("commandcode", provider_model_name("deepseek-v4-flash", "commandcode", registry_db)),
        ("deepseek", "deepseek-v4-flash"),
        ("opencode", "deepseek-v4-flash"),
    ]
    # Non-target original steps remain as fallbacks.
    assert any(s["model"] == "ox-alpha-free" for s in out[3:])
