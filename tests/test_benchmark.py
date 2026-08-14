"""Tests for the LiveBench benchmark runner."""
import json
import sys

import pytest

from src.api.benchmark import (
    build_livebench_commands,
    parse_livebench_csv,
    livebench_dir,
    LIVEBENCH_CATEGORIES,
)


# ── Command construction ────────────────────────────────────────────────────

def test_build_commands_full_suite(tmp_path):
    runner = tmp_path / "run_livebench.py"
    shower = tmp_path / "show_livebench_result.py"
    runner.write_text("")
    shower.write_text("")

    commands = build_livebench_commands(
        model="deepseek-v4-pro",
        api_base="https://api.deepseek.com/v1",
        api_key="sk-test",
        categories=None,
        livebench_path=str(tmp_path),
    )
    # categories=None → one scope per non-Docker category (2 cmds each).
    n_categories = len(LIVEBENCH_CATEGORIES)
    assert len(commands) == n_categories * 2
    run_cmd = commands[0]
    assert "run_livebench.py" in run_cmd[1]
    assert "--model" in run_cmd and "deepseek-v4-pro" in run_cmd
    assert "--api-base" in run_cmd and "https://api.deepseek.com/v1" in run_cmd
    assert "--api-key" in run_cmd and "sk-test" in run_cmd
    # No bare live_bench scope (that would include Docker agentic coding).
    scopes = [c for cmd in commands for c in cmd if c.startswith("live_bench")]
    assert all(s.startswith("live_bench/") for s in scopes)
    assert "live_bench/agentic_coding" not in scopes


def test_build_commands_subset(tmp_path):
    runner = tmp_path / "run_livebench.py"
    shower = tmp_path / "show_livebench_result.py"
    runner.write_text("")
    shower.write_text("")

    commands = build_livebench_commands(
        model="m", api_base="u", api_key="k",
        categories=["coding", "math"],
        livebench_path=str(tmp_path),
    )
    # Two categories → 2 scopes × 2 commands = 4
    assert len(commands) == 4
    scopes = [c for cmd in commands if "run_livebench.py" in cmd[1]
              for c in cmd if c.startswith("live_bench/")]
    assert scopes == ["live_bench/coding", "live_bench/math"]


def test_build_commands_missing_checkout_raises(monkeypatch):
    monkeypatch.delenv("LCP_LIVEBENCH_DIR", raising=False)
    with pytest.raises(RuntimeError, match="LiveBench checkout not found"):
        build_livebench_commands(
            model="m", api_base="u", api_key="k",
            categories=None, livebench_path=None,
        )


def test_livebench_categories_exclude_agentic():
    assert "agentic_coding" not in LIVEBENCH_CATEGORIES
    assert len(LIVEBENCH_CATEGORIES) == 6


# ── CSV parsing ─────────────────────────────────────────────────────────────

SAMPLE_CSV = (
    "model,reasoning,coding,agentic_coding,math,data_analysis,language,instruction_following\n"
    "deepseek-v4-pro,82.7,70.0,42.6,90.7,74.5,78.1,62.4\n"
    "gpt-5.6-sol,91.7,83.9,56.2,96.2,79.8,87.7,71.8\n"
)


def test_parse_csv_finds_model_row():
    scores = parse_livebench_csv(SAMPLE_CSV, "deepseek-v4-pro")
    assert scores["coding"] == 70.0
    assert scores["math"] == 90.7
    assert set(scores.keys()) == set(LIVEBENCH_CATEGORIES)


def test_parse_csv_model_not_found():
    assert parse_livebench_csv(SAMPLE_CSV, "unknown-model") == {}


def test_parse_csv_skips_non_numeric():
    csv = (
        "model,coding,math\n"
        "m,70.0,N/A\n"
    )
    scores = parse_livebench_csv(csv, "m")
    assert "coding" in scores
    assert "math" not in scores  # N/A skipped


def test_parse_csv_empty():
    assert parse_livebench_csv("", "m") == {}


# ── Queue + run flow (with mocked subprocess) ───────────────────────────────

def test_queue_and_run_flow(tmp_path, monkeypatch):
    from src.api.benchmark import queue_benchmark, get_run
    from src.api.models import Base, get_engine, get_session

    engine = get_engine(str(tmp_path / "bench.db"))
    Base.metadata.create_all(engine)

    # Fake config with a provider + env key.
    class FakeConfig:
        providers = {
            "deepseek": {"api_base": "https://api.deepseek.com/v1", "models": []},
        }
        def get_provider_key(self, provider):
            return "sk-env"

    # Fake credential store returning None (falls back to env key).
    monkeypatch.setattr(
        "src.api.credential_store.get_credential_store", lambda: None
    )
    # Fake plugin registry returning no plugin (passthrough model).
    class FakeRegistry:
        def for_provider(self, provider):
            return None
    monkeypatch.setattr("src.api.cost_plugins.get_registry", lambda: FakeRegistry())

    # Mock subprocess: first call (run) returns 0, second (show) writes CSV.
    import subprocess as sp
    real_run = sp.run

    def fake_run(cmd, cwd, capture_output, text, timeout):
        if "show_livebench_result.py" in str(cmd):
            import os
            with open(os.path.join(cwd, "all_groups.csv"), "w") as f:
                f.write(
                    "model,coding,math\n"
                    "deepseek-v4-pro,70.0,90.7\n"
                )
        return real_run(
            ["true"], cwd=cwd, capture_output=True, text=True, timeout=10,
        )

    monkeypatch.setattr("src.api.benchmark.subprocess.run", fake_run)
    monkeypatch.setattr("src.api.benchmark.livebench_dir", lambda: str(tmp_path))

    run = queue_benchmark(
        engine, FakeConfig(),
        target_kind="provider",
        target={"provider": "deepseek", "model": "deepseek-v4-pro"},
        categories=["coding", "math"],
    )
    assert run["status"] == "queued"

    # The worker is a background thread; give it a moment to process.
    import time
    for _ in range(50):
        result = get_run(engine, run["id"])
        if result["status"] in ("done", "failed"):
            break
        time.sleep(0.1)

    result = get_run(engine, run["id"])
    assert result["status"] == "done", result
    assert result["result"]["categories"]["coding"] == 70.0

    # The scores should be in model_capabilities as lcp_benchmark.
    from src.api.models import ModelCapability
    with get_session(engine) as session:
        rows = session.query(ModelCapability).filter_by(
            model="deepseek-v4-pro", source="lcp_benchmark"
        ).all()
        by_task = {r.task_type: r.score for r in rows}
        assert by_task["code_generation"] == pytest.approx(0.70)
        assert by_task["reasoning_chain"] == pytest.approx(0.907)
