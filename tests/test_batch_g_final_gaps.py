"""Batch G: final gap sweep — benchmark.py and other small residual gaps.

Covers the last uncovered statements identified by term-missing at 99% TOTAL.
"""

import json
import os
import queue
from unittest.mock import MagicMock, patch

import pytest


# ── benchmark.py: CSV parser edge branches ───────────────────────────────────

class TestParseCsvGaps:
    def test_groups_csv_bad_float_skipped(self):
        # 334-335: float(raw) ValueError → continue
        from src.api.benchmark import parse_livebench_csv
        csv_text = "model,coding,math\nm-1,abc,90.0\n"
        assert parse_livebench_csv(csv_text, "m-1") == {"math": 90.0}

    def test_tasks_csv_empty_text(self):
        # 350-351: fieldnames None → {}
        from src.api.benchmark import parse_livebench_tasks_csv
        assert parse_livebench_tasks_csv("", "m-1") == {}

    def test_tasks_csv_bad_float_skipped(self):
        # 365-366: float(raw) ValueError → continue
        from src.api.benchmark import parse_livebench_tasks_csv
        csv_text = "model,theory_of_mind,zebra_puzzle\nm-1,zzz,71.5\n"
        out = parse_livebench_tasks_csv(csv_text, "m-1")
        assert out == {"reasoning": {"zebra_puzzle": 71.5}}


# ── benchmark.py: _bind_log_engine exception branch ──────────────────────────

class TestBindLogEngineExcept:
    def test_url_side_effect_swallowed(self, monkeypatch):
        # 404-405: engine.url property raises → except Exception: pass
        import src.api.benchmark as bm

        class Exploder:
            # getattr(engine, "url", "") propagates non-AttributeError raises
            # → the except-Exception branch in _bind_log_engine swallows it.
            @property
            def url(self):
                raise RuntimeError("engine is dead")

        monkeypatch.setattr(bm, "_log_dir", None)
        bm._bind_log_engine(Exploder())  # must not raise
        assert bm._log_dir is None


# ── benchmark.py: _log / get_run_log file branches ───────────────────────────

class TestLogHelpers:
    def test_log_trims_buffer(self, monkeypatch):
        # 422: buffer exceeds _LOG_MAX_LINES → oldest trimmed
        import src.api.benchmark as bm
        monkeypatch.setattr(bm, "_LOG_MAX_LINES", 2)
        monkeypatch.setattr(bm, "_log_dir", None)
        try:
            bm._log(9001, "a")
            bm._log(9001, "b")
            bm._log(9001, "c")
            assert bm._run_logs[9001] == ["b", "c"]
        finally:
            bm._run_logs.pop(9001, None)

    def test_log_oserror_on_write(self, monkeypatch, tmp_path):
        # 430-431: OSError while persisting log line → pass
        import src.api.benchmark as bm
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x")
        monkeypatch.setattr(bm, "_log_dir", str(blocker))
        bm._log(9002, "hello")  # must not raise
        assert bm._run_logs.get(9002) == ["hello"]
        bm._run_logs.pop(9002, None)

    def test_get_run_log_read_oserror(self, monkeypatch, tmp_path):
        # 449-450: OSError while reading persisted log → []
        import src.api.benchmark as bm
        monkeypatch.setattr(bm, "_log_dir", str(tmp_path))
        bm._run_logs.pop(9003, None)
        with patch("src.api.benchmark.os.path.isfile", return_value=True), \
             patch("builtins.open", side_effect=OSError("nope")):
            assert bm.get_run_log(None, 9003) == []


# ── benchmark.py: worker loop crash branch ───────────────────────────────────

class TestWorkerLoopCrash:
    def test_execute_exception_marks_failed(self, tmp_path):
        # 470-472: _execute_run raises inside worker → logged + _mark_failed
        import src.api.benchmark as bm
        from src.api.models import Base, get_engine, BenchmarkRun, get_session
        engine = get_engine(str(tmp_path / "w.db"))
        Base.metadata.create_all(engine)
        with get_session(engine) as session:
            run = BenchmarkRun(target_kind="provider",
                               target_json=json.dumps({"provider": "p", "model": "m"}),
                               categories_json="[]", status="running")
            session.add(run)
            session.commit()
            rid = run.id

        q = queue.Queue()
        q.put((rid, engine, None))
        items = iter([q.get_nowait()])

        def get_side_effect():
            try:
                return next(items)
            except StopIteration:
                raise RuntimeError("stop-loop")

        with patch.object(bm, "_worker_queue") as wq:
            wq.get.side_effect = get_side_effect
            with patch.object(bm, "_execute_run", side_effect=RuntimeError("kaboom")):
                with pytest.raises(RuntimeError, match="stop-loop"):
                    bm._worker_loop()
            assert wq.task_done.called
        with get_session(engine) as session:
            row = session.query(BenchmarkRun).filter_by(id=rid).first()
            assert row.status == "failed"
            assert "kaboom" in row.error


# ── benchmark.py: list_runs model filter ─────────────────────────────────────

class TestListRunsFilter:
    def test_filter_by_model(self, tmp_path):
        # 573-574: model filter branch
        from src.api.benchmark import list_runs
        from src.api.models import Base, get_engine, BenchmarkRun, get_session
        engine = get_engine(str(tmp_path / "lr.db"))
        Base.metadata.create_all(engine)
        with get_session(engine) as session:
            for m in ("alpha", "beta"):
                session.add(BenchmarkRun(
                    target_kind="provider",
                    target_json=json.dumps({"provider": "p", "model": m}),
                    categories_json="[]", status="done"))
            session.commit()
        out = list_runs(engine, model="alpha")
        assert out["total"] == 1
        assert out["runs"][0]["target"]["model"] == "alpha"


# ── benchmark.py: _resolve_provider_target url.database exc ─────────────────

class TestResolveUrlExc:
    def test_database_property_raises(self, tmp_path):
        # 656-657: str(engine.url.database) raises → fallback db_path
        from src.api.benchmark import _resolve_provider_target
        engine = MagicMock()

        class U:
            @property
            def database(self):
                raise RuntimeError("no database")

        engine.url = U()

        class Cfg:
            providers = {"deepseek": {"api_base": "https://x/v1"}}
            def get_provider_key(self, p):
                return "sk"

        with patch("src.api.credential_store.get_credential_store", return_value=None), \
             patch("src.api.router.logical_model_name", return_value="m") as lm, \
             patch("src.api.router.provider_model_name", return_value="api-m"):
            out = _resolve_provider_target(engine, Cfg(),
                                           {"provider": "deepseek", "model": "m"})
        assert out == ("api-m", "https://x/v1", "sk")


# ── benchmark.py: _execute_run early exits ───────────────────────────────────

class TestExecuteRunEarly:
    def test_missing_run_returns(self, tmp_path):
        # 683: run row absent → immediate return, no crash
        from src.api.benchmark import _execute_run
        from src.api.models import Base, get_engine
        engine = get_engine(str(tmp_path / "er.db"))
        Base.metadata.create_all(engine)
        _execute_run(12345, engine, None)  # silent no-op

    def test_checkout_missing_fails(self, tmp_path):
        # 705 + 709-710: livebench_dir() falsy → RuntimeError recorded
        from src.api.benchmark import _execute_run, get_run
        from src.api.models import Base, get_engine, BenchmarkRun, get_session
        engine = get_engine(str(tmp_path / "er2.db"))
        Base.metadata.create_all(engine)
        with get_session(engine) as session:
            run = BenchmarkRun(target_kind="provider",
                               target_json=json.dumps({"provider": "p", "model": "m"}),
                               categories_json=None, status="queued")
            session.add(run)
            session.commit()
            rid = run.id

        class Cfg:
            providers = {"p": {"api_base": "https://x/v1"}}
            def get_provider_key(self, p):
                return "sk"

        with patch("src.api.benchmark.core_deps_available", return_value=True), \
             patch("src.api.benchmark.livebench_dir", return_value=None), \
             patch("src.api.benchmark._resolve_provider_target",
                   return_value=("m", "https://x/v1", "sk")):
            _execute_run(rid, engine, Cfg())
        run = get_run(engine, rid)
        assert run["status"] == "failed"
        assert "checkout not found" in run["error"]


# ── benchmark.py: _upsert_scores unknown category ────────────────────────────

class TestUpsertUnknownCategory:
    def test_unknown_lb_category_skipped(self, tmp_path):
        # 868: LB_TO_LCP.get(lb_cat) is None → continue
        from src.api.benchmark import _upsert_scores
        from src.api.models import Base, get_engine, ModelCapability, get_session
        engine = get_engine(str(tmp_path / "us.db"))
        Base.metadata.create_all(engine)
        _upsert_scores(engine, {"model": "m-x"}, {"zzz_unknown_cat": 50.0})
        with get_session(engine) as session:
            assert session.query(ModelCapability).count() == 0
