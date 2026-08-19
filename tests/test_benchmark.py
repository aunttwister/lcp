"""Tests for the LiveBench benchmark runner."""
import json
import sys

import pytest

from src.api.benchmark import (
    build_livebench_commands,
    parse_livebench_csv,
    livebench_dir,
    validate_categories,
    benchmark_status,
    _redact_cmd,
    _resolve_provider_target,
    _upsert_scores,
    LIVEBENCH_CATEGORIES,
)


# ── Availability / status (benchmark "plugin") ──────────────────────────────

def test_benchmark_status_unavailable_when_no_checkout(monkeypatch):
    monkeypatch.delenv("LCP_MODULES_DIR", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr("src.api.benchmark._valid_checkout", lambda p: False)
    status = benchmark_status()
    assert status["available"] is False
    assert status["reason"] is not None
    assert status["categories"] == LIVEBENCH_CATEGORIES


def test_benchmark_status_available_with_checkout(tmp_path, monkeypatch):
    root = tmp_path / "modules" / "livebench"
    pkg = root / "livebench"
    pkg.mkdir(parents=True)
    (root / "pyproject.toml").write_text("")
    (pkg / "run_livebench.py").write_text("")
    monkeypatch.setenv("LCP_MODULES_DIR", str(tmp_path / "modules"))
    status = benchmark_status()
    assert status["available"] is True
    assert status["livebench_dir"] == str(pkg)
    # coding_supported is False unless tensorflow is importable — don't assert
    # a specific value here, just that the key exists.
    assert "coding_supported" in status


def test_empty_modules_dir_is_not_a_checkout(tmp_path, monkeypatch):
    # An empty $LCP_MODULES_DIR/livebench directory must NOT count as installed.
    checkout = tmp_path / "modules" / "livebench"
    checkout.mkdir(parents=True)
    monkeypatch.setenv("LCP_MODULES_DIR", str(tmp_path / "modules"))
    monkeypatch.setattr("shutil.which", lambda _: None)
    status = benchmark_status()
    assert status["available"] is False


def test_benchmark_status_coding_detection(monkeypatch):
    import src.api.benchmark as bm

    monkeypatch.setattr("src.api.benchmark.coding_deps_available", lambda: True)
    assert bm.coding_deps_available() is True
    monkeypatch.setattr("src.api.benchmark.coding_deps_available", lambda: False)
    assert bm.coding_deps_available() is False


# ── Category validation ─────────────────────────────────────────────────────

def test_validate_categories_none_returns_all():
    cats = validate_categories(None)
    assert cats == LIVEBENCH_CATEGORIES


def test_validate_categories_empty_returns_all():
    assert validate_categories([]) == LIVEBENCH_CATEGORIES


def test_validate_categories_normalizes_and_dedupes():
    assert validate_categories(["coding", "Coding", " math "]) == ["coding", "math"]


def test_validate_categories_rejects_agentic_coding():
    with pytest.raises(ValueError, match="agentic_coding"):
        validate_categories(["agentic_coding"])


def test_validate_categories_rejects_unknown():
    with pytest.raises(ValueError, match="bogus"):
        validate_categories(["coding", "bogus"])


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
    monkeypatch.delenv("LCP_MODULES_DIR", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr("src.api.benchmark._valid_checkout", lambda p: False)
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


def test_parse_tasks_csv_groups_by_category():
    from src.api.benchmark import parse_livebench_tasks_csv
    csv = (
        "model,theory_of_mind,zebra_puzzle,spatial,logic_with_navigation\n"
        "qwen3.6-27b,65.4,53.8,100.0,62.0\n"
    )
    tasks = parse_livebench_tasks_csv(csv, "qwen3.6-27b")
    assert set(tasks.keys()) == {"reasoning"}
    assert tasks["reasoning"]["theory_of_mind"] == 65.4
    assert tasks["reasoning"]["spatial"] == 100.0


def test_parse_tasks_csv_unknown_task_goes_to_all():
    from src.api.benchmark import parse_livebench_tasks_csv
    csv = (
        "model,some_brand_new_task\n"
        "m,42.0\n"
    )
    tasks = parse_livebench_tasks_csv(csv, "m")
    assert tasks["_all"]["some_brand_new_task"] == 42.0


def test_parse_tasks_csv_skips_non_numeric_and_missing():
    from src.api.benchmark import parse_livebench_tasks_csv
    csv = (
        "model,code_generation,code_completion\n"
        "m,75.0,N/A\n"
    )
    tasks = parse_livebench_tasks_csv(csv, "m")
    assert tasks["coding"]["code_generation"] == 75.0
    assert "code_completion" not in tasks["coding"]


def test_parse_tasks_csv_model_not_found():
    from src.api.benchmark import parse_livebench_tasks_csv
    csv = "model,theory_of_mind\nother-model,1.0\n"
    assert parse_livebench_tasks_csv(csv, "m") == {}


# ── Log file persistence + stale-run recovery ───────────────────────────────

def test_log_written_to_file_and_read_back(tmp_path, monkeypatch):
    import src.api.benchmark as bm

    engine = None
    bm._bind_log_engine(None)
    monkeypatch.setattr(bm, "_log_dir", str(tmp_path))
    # Clear any prior in-memory buffer.
    bm._run_logs.clear()

    bm._log(42, "first line")
    bm._log(42, "second line")

    path = tmp_path / "run-42.log"
    assert path.is_file()
    assert "first line\nsecond line\n" == path.read_text()

    # Simulate a restart: clear the live buffer; file should still be read.
    bm._run_logs.clear()
    lines = bm.get_run_log(None, 42)
    assert lines == ["first line", "second line"]


def test_recover_stale_runs(tmp_path):
    from src.api.benchmark import recover_stale_runs, get_run
    from src.api.models import Base, get_engine, get_session, BenchmarkRun

    engine = get_engine(str(tmp_path / "stale.db"))
    Base.metadata.create_all(engine)

    with get_session(engine) as session:
        session.add(BenchmarkRun(target_kind="provider", target_json='{"model":"m"}', status="running"))
        session.add(BenchmarkRun(target_kind="provider", target_json='{"model":"m"}', status="queued"))
        session.add(BenchmarkRun(target_kind="provider", target_json='{"model":"m"}', status="done"))
        session.commit()

    recovered = recover_stale_runs(engine)
    assert recovered == 2

    for run_id in (1, 2):
        run = get_run(engine, run_id)
        assert run["status"] == "failed"
        assert "restart" in run["error"]
    assert get_run(engine, 3)["status"] == "done"


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

    # Mock subprocess.Popen: first call (run) returns 0, second (show) writes CSV.
    class FakeProc:
        def __init__(self, cmd, cwd, **kwargs):
            self.cmd = cmd
            self.cwd = cwd

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        @property
        def stdout(self):
            import io
            return io.StringIO("")

        def wait(self):
            if "show_livebench_result.py" in str(self.cmd):
                import os
                with open(os.path.join(self.cwd, "all_groups.csv"), "w") as f:
                    f.write(
                        "model,coding,math\n"
                        "deepseek-v4-pro,70.0,90.7\n"
                    )
            return 0

    monkeypatch.setattr("src.api.benchmark.subprocess.Popen", FakeProc)
    monkeypatch.setattr("src.api.benchmark.livebench_dir", lambda: str(tmp_path))
    monkeypatch.setattr("src.api.benchmark.core_deps_available", lambda: True)

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


# ── API key redaction in logs ───────────────────────────────────────────────

def test_redact_cmd_masks_api_key():
    cmd = ["python", "run_livebench.py", "--model", "m", "--api-key", "sk-secret-123"]
    out = _redact_cmd(cmd)
    assert "sk-secret-123" not in out
    assert "--api-key ***" in out


def test_redact_cmd_no_api_key_unchanged():
    cmd = ["python", "run_livebench.py", "--model", "m"]
    assert _redact_cmd(cmd) == "python run_livebench.py --model m"


# ── Provider target resolution ──────────────────────────────────────────────

class _FakeConfig:
    def __init__(self, providers, key="sk-env-key"):
        self.providers = providers
        self._key = key

    def get_provider_key(self, provider):
        return self._key


class _FakeStore:
    def __init__(self, key=None):
        self._key = key

    def get(self, provider):
        return self._key


class _FakePlugin:
    def __init__(self, api_model=None):
        self._api_model = api_model

    def get_api_model(self, model):
        return self._api_model or model


class _FakeRegistry:
    def __init__(self, plugin=None):
        self._plugin = plugin

    def for_provider(self, provider):
        return self._plugin


def test_resolve_provider_target_env_key(monkeypatch):
    config = _FakeConfig({"deepseek": {"api_base": "https://api.deepseek.com/v1"}})
    monkeypatch.setattr("src.api.credential_store.get_credential_store", lambda: None)
    monkeypatch.setattr("src.api.cost_plugins.get_registry", lambda: _FakeRegistry())

    model, base, key = _resolve_provider_target(
        None, config, {"provider": "deepseek", "model": "deepseek-v4-pro"}
    )
    assert model == "deepseek-v4-pro"
    assert base == "https://api.deepseek.com/v1"
    assert key == "sk-env-key"


def test_resolve_provider_target_credential_store_wins(monkeypatch):
    config = _FakeConfig({"deepseek": {"api_base": "https://api.deepseek.com/v1"}})
    monkeypatch.setattr(
        "src.api.credential_store.get_credential_store",
        lambda: _FakeStore("sk-store-key"),
    )
    monkeypatch.setattr("src.api.cost_plugins.get_registry", lambda: _FakeRegistry())

    _, _, key = _resolve_provider_target(
        None, config, {"provider": "deepseek", "model": "deepseek-v4-pro"}
    )
    assert key == "sk-store-key"


def test_resolve_provider_target_translates_model(monkeypatch):
    config = _FakeConfig({"commandcode": {"api_base": "https://api.commandcode.ai/provider/v1"}})
    monkeypatch.setattr("src.api.credential_store.get_credential_store", lambda: None)
    plugin = _FakePlugin("deepseek/deepseek-v4-pro")
    monkeypatch.setattr("src.api.cost_plugins.get_registry", lambda: _FakeRegistry(plugin))

    model, _, _ = _resolve_provider_target(
        None, config, {"provider": "commandcode", "model": "deepseek-v4-pro"}
    )
    assert model == "deepseek/deepseek-v4-pro"


def test_resolve_provider_target_unknown_provider_raises():
    config = _FakeConfig({})
    with pytest.raises(ValueError, match="unknown provider"):
        _resolve_provider_target(None, config, {"provider": "nope", "model": "m"})


def test_resolve_provider_target_missing_fields_raises():
    config = _FakeConfig({"deepseek": {"api_base": "x"}})
    with pytest.raises(ValueError, match="requires 'provider' and 'model'"):
        _resolve_provider_target(None, config, {"provider": "deepseek"})


def test_resolve_provider_target_missing_api_base_raises():
    config = _FakeConfig({"deepseek": {}})
    with pytest.raises(ValueError, match="no api_base"):
        _resolve_provider_target(None, config, {"provider": "deepseek", "model": "m"})


# ── Score upsert ────────────────────────────────────────────────────────────

def test_upsert_scores_normalizes_and_inserts(tmp_path):
    from src.api.models import Base, get_engine, get_session, ModelCapability

    engine = get_engine(str(tmp_path / "upsert.db"))
    Base.metadata.create_all(engine)

    _upsert_scores(
        engine,
        {"model": "m1"},
        {"coding": 70.0, "math": 90.0},
    )

    with get_session(engine) as session:
        rows = session.query(ModelCapability).filter_by(model="m1", source="lcp_benchmark").all()
        by_task = {r.task_type: (r.score, r.raw_score, r.benchmark_category) for r in rows}
        assert by_task["code_generation"] == (pytest.approx(0.70), pytest.approx(70.0), "coding")
        assert by_task["reasoning_chain"] == (pytest.approx(0.90), pytest.approx(90.0), "math")
        # Derived: debugging mirrors code_generation (a coding subskill).
        assert by_task["debugging"] == (pytest.approx(0.70), pytest.approx(70.0), "coding")


def test_upsert_scores_overwrites_existing(tmp_path):
    from src.api.models import Base, get_engine, get_session, ModelCapability

    engine = get_engine(str(tmp_path / "upsert2.db"))
    Base.metadata.create_all(engine)

    # First grade: coding 60.0
    _upsert_scores(engine, {"model": "m1"}, {"coding": 60.0})
    # Regrade: coding 80.0 (should overwrite, not duplicate)
    _upsert_scores(engine, {"model": "m1"}, {"coding": 80.0})

    with get_session(engine) as session:
        rows = session.query(ModelCapability).filter_by(
            model="m1", source="lcp_benchmark", task_type="code_generation"
        ).all()
        assert len(rows) == 1
        assert rows[0].score == pytest.approx(0.80)
        # Debugging also updated in place, single row.
        dbg = session.query(ModelCapability).filter_by(
            model="m1", source="lcp_benchmark", task_type="debugging"
        ).all()
        assert len(dbg) == 1
        assert dbg[0].score == pytest.approx(0.80)


# ── Queue-time validation ───────────────────────────────────────────────────

def test_queue_rejects_unknown_category(tmp_path):
    from src.api.benchmark import queue_benchmark
    from src.api.models import Base, get_engine

    engine = get_engine(str(tmp_path / "q.db"))
    Base.metadata.create_all(engine)

    with pytest.raises(ValueError, match="bogus"):
        queue_benchmark(
            engine, _FakeConfig({}),
            target_kind="provider",
            target={"provider": "deepseek", "model": "m"},
            categories=["bogus"],
        )


def test_queue_rejects_missing_provider_target(tmp_path):
    from src.api.benchmark import queue_benchmark
    from src.api.models import Base, get_engine

    engine = get_engine(str(tmp_path / "q2.db"))
    Base.metadata.create_all(engine)

    with pytest.raises(ValueError, match="requires 'provider' and 'model'"):
        queue_benchmark(
            engine, _FakeConfig({}),
            target_kind="provider",
            target={"model": "m"},
        )


def test_queue_rejects_invalid_kind(tmp_path):
    from src.api.benchmark import queue_benchmark
    from src.api.models import Base, get_engine

    engine = get_engine(str(tmp_path / "q3.db"))
    Base.metadata.create_all(engine)

    with pytest.raises(ValueError, match="invalid target_kind"):
        queue_benchmark(
            engine, _FakeConfig({}),
            target_kind="bogus",
            target={},
        )


def test_queue_stores_normalized_categories(tmp_path):
    from src.api.benchmark import queue_benchmark
    from src.api.models import Base, get_engine, get_session, BenchmarkRun

    engine = get_engine(str(tmp_path / "q4.db"))
    Base.metadata.create_all(engine)

    # Queue with categories=None → stored as full normalized list.
    queue_benchmark(
        engine, _FakeConfig({}),
        target_kind="provider",
        target={"provider": "deepseek", "model": "m"},
        categories=None,
    )
    with get_session(engine) as session:
        run = session.query(BenchmarkRun).first()
        assert json.loads(run.categories_json) == LIVEBENCH_CATEGORIES
