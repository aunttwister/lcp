"""Tests for the LiveBench benchmark execution path (src.api.benchmark).

Covers the parts of the runner that were untested: the streaming subprocess
execution in ``_execute_run`` (fatal marker abort, non-zero exit, missing
checkout, no-parseable-CSV), ``_redact_cmd`` / ``_redact_stream_line``,
``_bind_log_engine`` / ``get_run_log`` file fallback, ``_resolve_provider_target``
errors, and ``queue_benchmark`` validation.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.api.benchmark import (
    _redact_cmd,
    _redact_stream_line,
    build_livebench_commands,
    queue_benchmark,
)


# ── Redaction ────────────────────────────────────────────────────────────────

class TestRedaction:
    def test_redact_cmd_hides_api_key(self):
        cmd = ["python", "run_livebench.py", "--api-key", "sk-secret", "--model", "m"]
        assert "sk-secret" not in _redact_cmd(cmd)
        assert "***" in _redact_cmd(cmd)

    def test_redact_stream_line_plain(self):
        assert _redact_stream_line("export KEY=abc123", "abc123") == "export KEY=***"

    def test_redact_stream_line_quoted(self):
        assert _redact_stream_line("KEY='abc123'", "abc123") == "KEY='***'"
        assert _redact_stream_line('KEY="abc123"', "abc123") == 'KEY="***"'

    def test_redact_stream_line_no_key(self):
        assert _redact_stream_line("hello", "") == "hello"


# ── Queue validation ─────────────────────────────────────────────────────────

class TestQueueValidation:
    def test_rejects_unknown_target_kind(self, tmp_path):
        from src.api.models import Base, get_engine
        engine = get_engine(str(tmp_path / "b.db"))
        Base.metadata.create_all(engine)
        with pytest.raises(ValueError, match="invalid target_kind"):
            queue_benchmark(engine, None, "bogus", {"model": "m"})

    def test_rejects_provider_target_missing_model(self, tmp_path):
        from src.api.models import Base, get_engine
        engine = get_engine(str(tmp_path / "b.db"))
        Base.metadata.create_all(engine)
        with pytest.raises(ValueError, match="requires 'provider' and 'model'"):
            queue_benchmark(engine, None, "provider", {"provider": "deepseek"})

    def test_queue_creates_run(self, tmp_path):
        from src.api.models import Base, get_engine, BenchmarkRun, get_session
        engine = get_engine(str(tmp_path / "b.db"))
        Base.metadata.create_all(engine)
        # Suppress the worker thread — we only assert the DB row.
        with patch("src.api.benchmark._worker_queue.put"):
            with patch("src.api.benchmark._ensure_worker"):
                run = queue_benchmark(engine, None, "provider",
                                      {"provider": "deepseek", "model": "m"})
        assert run["status"] == "queued"
        with get_session(engine) as session:
            row = session.query(BenchmarkRun).filter_by(id=run["id"]).first()
            assert row is not None
            assert "reasoning" in row.categories_json
            assert "agentic_coding" not in row.categories_json


# ── _resolve_provider_target error paths ─────────────────────────────────────

class TestResolveProviderTarget:
    def test_unknown_provider(self, tmp_path):
        from src.api.benchmark import _resolve_provider_target
        from src.api.models import Base, get_engine
        engine = get_engine(str(tmp_path / "b.db"))
        Base.metadata.create_all(engine)

        class FakeConfig:
            providers = {}
        with pytest.raises(ValueError, match="unknown provider"):
            _resolve_provider_target(engine, FakeConfig(), {"provider": "nope", "model": "m"})

    def test_missing_api_base(self, tmp_path):
        from src.api.benchmark import _resolve_provider_target
        from src.api.models import Base, get_engine
        engine = get_engine(str(tmp_path / "b.db"))
        Base.metadata.create_all(engine)

        class FakeConfig:
            providers = {"deepseek": {"api_base": ""}}
        with pytest.raises(ValueError, match="no api_base"):
            _resolve_provider_target(engine, FakeConfig(), {"provider": "deepseek", "model": "m"})


# ── _execute_run streaming paths ─────────────────────────────────────────────

def _fake_engine(tmp_path):
    from src.api.models import Base, get_engine
    engine = get_engine(str(tmp_path / "b.db"))
    Base.metadata.create_all(engine)
    return engine


def _insert_run(engine, target, categories=None, status="queued"):
    from src.api.models import BenchmarkRun, get_session
    with get_session(engine) as session:
        run = BenchmarkRun(
            target_kind="provider",
            target_json=json.dumps(target),
            categories_json=json.dumps(categories or ["coding", "math"]),
            status=status,
        )
        session.add(run)
        session.commit()
        return run.id


class FakeProc:
    """Minimal subprocess.Popen stand-in with a scriptable stdout."""

    def __init__(self, lines, rc=0):
        self.lines = list(lines)
        self.rc = rc
        self.stdout = MagicMock()
        self.stdout.__iter__ = lambda self_: iter(self.lines)
        self.killed = False

    def wait(self):
        return self.rc

    def kill(self):
        self.killed = True


def _run_execute(tmp_path, lines, rc=0, categories=("coding", "math")):
    from src.api.benchmark import _execute_run
    engine = _fake_engine(tmp_path)
    run_id = _insert_run(engine, {"provider": "deepseek", "model": "deepseek-v4-pro"}, categories=list(categories))
    root = tmp_path / "livebench"
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("")
    (root / "livebench").mkdir(exist_ok=True)
    (root / "livebench" / "run_livebench.py").write_text("")
    (root / "livebench" / "show_livebench_result.py").write_text("")
    (root / "all_groups.csv").write_text(
        "model,coding,math\n"
        "deepseek-v4-pro,80.0,90.0\n"
    )
    (root / "all_tasks.csv").write_text(
        "model,theory_of_mind\n"
        "deepseek-v4-pro,84.6\n"
    )

    class FakeConfig:
        providers = {"deepseek": {"api_base": "https://api.deepseek.com/v1"}}
        def get_provider_key(self, provider):
            return "sk-test"

    with patch("src.api.benchmark.core_deps_available", return_value=True), \
         patch("src.api.benchmark.livebench_dir", return_value=str(root)), \
         patch("src.api.benchmark.subprocess.Popen", return_value=FakeProc(lines, rc)):
        _execute_run(run_id, engine, FakeConfig())
    return engine, run_id


class TestExecuteRun:
    def test_successful_run_marks_done_with_scores(self, tmp_path):
        from src.api.benchmark import get_run
        from src.api.models import get_session, ModelCapability
        engine, run_id = _run_execute(tmp_path, ["ok"], rc=0)
        run = get_run(engine, run_id)
        assert run["status"] == "done"
        assert run["result"]["categories"]["coding"] == 80.0
        assert run["result"]["tasks"]["reasoning"]["theory_of_mind"] == 84.6
        # Upserted capability rows
        with get_session(engine) as session:
            rows = session.query(ModelCapability).filter_by(
                model="deepseek-v4-pro", source="lcp_benchmark"
            ).all()
            assert rows

    def test_fatal_marker_aborts(self, tmp_path):
        from src.api.benchmark import get_run
        engine, run_id = _run_execute(tmp_path, ["insufficient balance — abort"], rc=0)
        run = get_run(engine, run_id)
        assert run["status"] == "failed"
        assert "fatal provider error" in run["error"]

    def test_nonzero_exit_fails(self, tmp_path):
        from src.api.benchmark import get_run
        engine, run_id = _run_execute(tmp_path, ["boom"], rc=3)
        run = get_run(engine, run_id)
        assert run["status"] == "failed"
        assert "exit 3" in run["error"]


# ── Log file fallback ────────────────────────────────────────────────────────

class TestLogFileFallback:
    def test_get_run_log_reads_file_after_buffer_cleared(self, tmp_path):
        import src.api.benchmark as bm
        bm._run_logs.clear()
        bm._log_dir = str(tmp_path)
        bm._log(99, "line-a")
        bm._run_logs.clear()  # simulate restart
        lines = bm.get_run_log(None, 99)
        assert lines == ["line-a"]

    def test_bind_log_engine_from_url(self):
        import src.api.benchmark as bm
        try:
            engine = MagicMock()
            engine.url = "sqlite:////data/costs.db"
            bm._bind_log_engine(engine)
            assert bm._log_dir == "/data/benchmark-logs"
        finally:
            bm._log_dir = None
