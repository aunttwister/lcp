"""Tests for LiveBench benchmark parallelism plumbing (src.api.benchmark).

Covers:
  - ``build_livebench_commands`` parallel → ``--parallel-requests N``
  - ``queue_benchmark`` parallel validation + target_json carry-through
  - ``_serve_benchmark_create_api`` parallel validation + forwarding
  - ``_execute_run`` site-aware deps gate (persistent site dir, not main env)
  - ``benchmark_status`` site-aware ``core_installed``
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.api.benchmark import build_livebench_commands, queue_benchmark


# ── build_livebench_commands ────────────────────────────────────────────────

class TestBuildCommandsParallel:
    def _setup(self, tmp_path):
        runner = tmp_path / "run_livebench.py"
        shower = tmp_path / "show_livebench_result.py"
        runner.write_text("")
        shower.write_text("")
        return runner, shower

    def test_default_sequential_no_flag(self, tmp_path):
        self._setup(tmp_path)
        commands = build_livebench_commands(
            model="m", api_base="u", api_key="k",
            categories=["math"],
            livebench_path=str(tmp_path),
        )
        run_cmd = commands[0]
        assert "--parallel-requests" not in run_cmd
        assert "--api-key" in run_cmd

    def test_parallel_one_no_flag(self, tmp_path):
        self._setup(tmp_path)
        commands = build_livebench_commands(
            model="m", api_base="u", api_key="k",
            categories=["math"],
            livebench_path=str(tmp_path),
            parallel=1,
        )
        assert "--parallel-requests" not in commands[0]

    def test_parallel_eight_adds_flag(self, tmp_path):
        self._setup(tmp_path)
        commands = build_livebench_commands(
            model="m", api_base="u", api_key="k",
            categories=["math"],
            livebench_path=str(tmp_path),
            parallel=8,
        )
        run_cmd = commands[0]
        assert "--parallel-requests" in run_cmd
        assert run_cmd[run_cmd.index("--parallel-requests") + 1] == "8"
        # The show-result command does NOT get the parallel flag.
        shower_cmd = commands[1]
        assert "--parallel-requests" not in shower_cmd

    def test_parallel_string_coerced(self, tmp_path):
        self._setup(tmp_path)
        commands = build_livebench_commands(
            model="m", api_base="u", api_key="k",
            categories=["math"],
            livebench_path=str(tmp_path),
            parallel="4",
        )
        run_cmd = commands[0]
        assert run_cmd[run_cmd.index("--parallel-requests") + 1] == "4"

    def test_parallel_applies_to_each_category_scope(self, tmp_path):
        self._setup(tmp_path)
        commands = build_livebench_commands(
            model="m", api_base="u", api_key="k",
            categories=["coding", "math"],
            livebench_path=str(tmp_path),
            parallel=6,
        )
        # 2 categories → 4 commands (run, show per scope)
        assert len(commands) == 4
        for i, cmd in enumerate(commands):
            if "run_livebench.py" in cmd[1]:
                assert "--parallel-requests" in cmd
            else:
                assert "--parallel-requests" not in cmd


# ── queue_benchmark validation ──────────────────────────────────────────────

class TestQueueParallel:
    def _engine(self, tmp_path):
        from src.api.models import Base, get_engine
        engine = get_engine(str(tmp_path / "b.db"))
        Base.metadata.create_all(engine)
        return engine

    def test_rejects_non_int(self, tmp_path):
        engine = self._engine(tmp_path)
        with pytest.raises(ValueError, match="invalid 'parallel'"):
            queue_benchmark(engine, None, "provider",
                            {"provider": "p", "model": "m"}, parallel="abc")

    def test_rejects_zero(self, tmp_path):
        engine = self._engine(tmp_path)
        with pytest.raises(ValueError, match="between 1 and 64"):
            queue_benchmark(engine, None, "provider",
                            {"provider": "p", "model": "m"}, parallel=0)

    def test_rejects_over_64(self, tmp_path):
        engine = self._engine(tmp_path)
        with pytest.raises(ValueError, match="between 1 and 64"):
            queue_benchmark(engine, None, "provider",
                            {"provider": "p", "model": "m"}, parallel=65)

    def test_parallel_carried_in_target_json(self, tmp_path, monkeypatch):
        from src.api.models import Base, get_engine, BenchmarkRun, get_session
        engine = self._engine(tmp_path)
        # Avoid actually dispatching to the worker thread.
        monkeypatch.setattr("src.api.benchmark._worker_queue", MagicMock())
        monkeypatch.setattr("src.api.benchmark._ensure_worker", lambda: None)

        run = queue_benchmark(engine, None, "provider",
                              {"provider": "zgx", "model": "m"}, parallel=12)
        assert run["target"]["parallel"] == 12
        assert run["status"] == "queued"

        with get_session(engine) as session:
            row = session.query(BenchmarkRun).filter(BenchmarkRun.id == run["id"]).first()
            parsed = json.loads(row.target_json)
            assert parsed["parallel"] == 12
            assert parsed["provider"] == "zgx"


# ── endpoint validation (POST /api/models/benchmark) ────────────────────────

class TestEndpointParallel:
    def _run(self, temp_db, body):
        from tests.test_batch_f2_endpoints_apis import TestHandler, _json_body, _status
        h = TestHandler(path="/api/models/benchmark", method="POST", engine=temp_db,
                        body=body)
        h._serve_benchmark_create_api()
        return h, _status(h), _json_body(h)

    def test_accepts_parallel_and_forwards(self, temp_db):
        from tests.test_batch_f2_endpoints_apis import TestHandler, _json_body
        captured = {}
        def fake_queue(engine, config, target_kind, target, categories=None, parallel=None):
            captured["parallel"] = parallel
            captured["target"] = target
            return {"id": 99, "status": "queued"}
        h = TestHandler(path="/api/models/benchmark", method="POST", engine=temp_db,
                        body={"provider": "p", "model": "m", "parallel": 8})
        with patch("src.api.benchmark.queue_benchmark", side_effect=fake_queue):
            h._serve_benchmark_create_api()
        assert _json_body(h)["run"]["id"] == 99
        assert captured["parallel"] == 8
        assert captured["target"] == {"provider": "p", "model": "m"}

    def test_rejects_bool_parallel(self, temp_db):
        h, st, body = self._run(temp_db, {"provider": "p", "model": "m", "parallel": True})
        assert st == 400
        assert "integer" in body["error"]

    def test_rejects_parallel_below_range(self, temp_db):
        h, st, body = self._run(temp_db, {"provider": "p", "model": "m", "parallel": 0})
        assert st == 400
        assert "between 1 and 64" in body["error"]

    def test_rejects_parallel_above_range(self, temp_db):
        h, st, body = self._run(temp_db, {"provider": "p", "model": "m", "parallel": 99})
        assert st == 400
        assert "between 1 and 64" in body["error"]


# ── site-aware deps gate ────────────────────────────────────────────────────

class TestSiteAwareGate:
    def test_execute_run_uses_site_for_core_deps(self, tmp_path, monkeypatch):
        """_execute_run must call core_deps_available(site=...) so deps in the
        persistent <modules_dir>/site dir pass the gate (they're lost from the
        image layer on rebuild)."""
        from src.api.models import Base, get_engine, BenchmarkRun, get_session
        engine = get_engine(str(tmp_path / "b.db"))
        Base.metadata.create_all(engine)
        with get_session(engine) as session:
            row = BenchmarkRun(
                target_kind="provider",
                target_json=json.dumps({"provider": "zgx", "model": "m", "parallel": 4}),
                categories_json=json.dumps(["math"]),
                status="queued",
            )
            session.add(row)
            session.commit()
            run_id = row.id

        (tmp_path / "run_livebench.py").write_text("")
        (tmp_path / "show_livebench_result.py").write_text("")

        from src.api import benchmark as bm
        with patch.object(bm, "core_deps_available", return_value=True) as mock_gate, \
             patch.object(bm, "_resolve_provider_target",
                          return_value=("m", "http://x/v1", "k")), \
             patch.object(bm, "livebench_dir", return_value=str(tmp_path)), \
             patch.object(bm, "_log", lambda *a, **k: None), \
             patch.object(bm, "build_livebench_commands") as mock_build, \
             patch.object(bm, "parse_livebench_csv", return_value={}), \
             patch.object(bm, "subprocess") as mock_sp:
            proc = MagicMock()
            proc.stdout = []
            proc.wait.return_value = 0
            mock_sp.Popen.return_value = proc
            bm._execute_run(run_id, engine, None)

        # The deps gate must have been called with the persistent site dir.
        assert "site" in mock_gate.call_args.kwargs
        assert mock_gate.call_args.kwargs["site"] is not None
        # The parallel value from the run target flows into the command builder.
        assert mock_build.call_args.kwargs["parallel"] == 4

    def test_benchmark_status_core_installed_site_aware(self, monkeypatch):
        from src.api import benchmark as bm
        monkeypatch.setattr(bm, "livebench_dir", lambda: "/mods/livebench/livebench")
        monkeypatch.setattr(bm, "coding_deps_available", lambda: True)
        captured = {}
        def fake_gate(site=None):
            captured["site"] = site
            return True
        monkeypatch.setattr(bm, "core_deps_available", fake_gate)
        monkeypatch.setattr("src.api.setup.livebench_site", lambda: "/mods/site")

        status = bm.benchmark_status()
        assert status["available"] is True
        assert status["core_installed"] is True
        assert captured["site"] == "/mods/site"