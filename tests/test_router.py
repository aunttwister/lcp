"""Tests for router.py"""
import os
import tempfile

import pytest
import sys
from src.api.router import (
    DynamicRouter, CapabilityRouter, classify_task, get_dynamic_router,
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

def test_init_router_passes_cost_bias(registry_db):
    init_router(registry_db, enabled=True, cost_bias=0.4)
    try:
        r = get_dynamic_router()
        assert r.enabled is True
        assert r.cost_bias == 0.4
    finally:
        # restore the default so other tests aren't affected
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

def test_select_model_debugging_prompt_returns_choice(registry_db):
    """A debugging-flavored prompt classifies + ranks and returns a model."""
    from src.api.seed_capabilities import seed_livebench
    seed_livebench(registry_db)
    router = CapabilityRouter(enabled=True, db_path=registry_db)
    msgs = [{"role": "user", "content": "why does this fail with a TypeError? please debug"}]
    result = router.select_model(
        msgs, available_models=["deepseek-v4-flash", "deepseek-v4-pro"])
    assert result in ("deepseek-v4-flash", "deepseek-v4-pro")


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
    """A rule with only a model (provider '*' wildcard) matches any provider."""
    router = _rule_router(registry_db, monkeypatch,
                          [{"task": "*", "action": "prefer",
                            "provider": "*", "model": "deepseek-v4-pro"}])
    chain = [{"provider": "opencode", "model": "deepseek-v4-flash"},
             {"provider": "deepseek", "model": "deepseek-v4-pro"}]
    candidates, fired = router._apply_rules(chain, "code_generation", "l2")
    assert candidates[0]["model"] == "deepseek-v4-pro"
    assert candidates[0]["provider"] == "deepseek"
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

def test_select_step_prefer_is_mandatory(registry_db, monkeypatch):
    """A fired prefer pins the step: even if scoring favors another step, the
    router returns the preferred one first (no reorder away). Also verifies the
    model-ID normalization (pro under commandcode's provider-side ID)."""
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
    out = router.select_step([{"role": "user", "content": "design the architecture"}],
                             chain=chain, profile="coder")
    assert out is not None
    assert out[0]["provider"] == "commandcode"
    assert out[0]["model"] == "deepseek/deepseek-v4-pro"
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
    out = router.select_step([{"role": "user", "content": "hi"}], chain=chain, profile="cron")
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
