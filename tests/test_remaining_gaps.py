"""Remaining branch coverage: setup._stream, commandcode subscription error,
benchmark _execute_run no-csv + non-provider target, core_deps/livebench_root."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest


# ── setup._stream ────────────────────────────────────────────────────────────

class TestSetupStream:
    def test_stream_success(self, monkeypatch):
        from src.api import setup as setup_mod

        monkeypatch.setattr(setup_mod, "_bench_install", {
            "status": "running", "progress": 0.0, "detail": "", "log": [],
        })
        proc = MagicMock()
        proc.stdout.__iter__ = lambda self_: iter(["a", "b", "c"])
        proc.wait.return_value = 0
        with patch("subprocess.Popen", return_value=proc):
            setup_mod._stream(["echo", "hi"], cwd="/tmp", start=2.0, end=25.0, status_msg="x")
        state = setup_mod._bench_install
        assert state["progress"] == 25.0
        assert state["detail"] == "c"
        monkeypatch.undo()

    def test_stream_nonzero_exit(self, monkeypatch):
        import subprocess
        from src.api import setup as setup_mod

        monkeypatch.setattr(setup_mod, "_bench_install", {
            "status": "running", "progress": 0.0, "detail": "", "log": [],
        })
        proc = MagicMock()
        proc.stdout.__iter__ = lambda self_: iter([])
        proc.wait.return_value = 1
        with patch("subprocess.Popen", return_value=proc):
            with pytest.raises(subprocess.CalledProcessError):
                setup_mod._stream(["echo"], cwd=None, start=0, end=10, status_msg="y")
        monkeypatch.undo()

    def test_start_livebench_install_joins_inflight(self, monkeypatch):
        from src.api import setup as setup_mod

        inflight = {"status": "running", "progress": 0.0, "detail": "", "log": []}
        monkeypatch.setattr(setup_mod, "_bench_install", inflight)
        monkeypatch.setattr(setup_mod.shutil, "which", lambda _: "/usr/bin/git")
        result = setup_mod.start_livebench_install(None)
        assert result is inflight
        monkeypatch.undo()


# ── commandcode subscription api_error path ──────────────────────────────────

class TestCommandCodeApiError:
    def test_subscription_exception_returns_api_error(self):
        from src.api.cost_plugins.commandcode import CommandCodeCostPlugin
        plugin = CommandCodeCostPlugin(engine=None)
        fake_store = MagicMock()
        fake_store.get_cookie.return_value = "session=valid"
        with patch.dict(os.environ, {}, clear=False):
            with patch("src.api.credential_store.get_credential_store", return_value=fake_store):
                with patch(
                    "src.api.cost_plugins.commandcode_api.fetch_subscription_snapshot_dict",
                    side_effect=RuntimeError("boom"),
                ):
                    result = plugin.fetch_subscription()
        assert result["_error"] == "api_error"
        assert "boom" in result["detail"]


# ── benchmark: _execute_run non-provider target + no-CSV ─────────────────────

def _engine(tmp_path):
    from src.api.models import Base, get_engine
    engine = get_engine(str(tmp_path / "b.db"))
    Base.metadata.create_all(engine)
    return engine


class TestExecuteRunMore:
    def test_non_provider_target_fails(self, tmp_path):
        from src.api.benchmark import _execute_run, get_run
        from src.api.models import BenchmarkRun, get_session

        engine = _engine(tmp_path)
        with get_session(engine) as session:
            run = BenchmarkRun(
                target_kind="profile",
                target_json=json.dumps({"profile": "l2"}),
                categories_json=json.dumps(["coding"]),
                status="queued",
            )
            session.add(run)
            session.commit()
            run_id = run.id
        _execute_run(run_id, engine, None)
        run = get_run(engine, run_id)
        assert run["status"] == "failed"
        assert "not implemented" in run["error"]

    def test_no_csv_raises(self, tmp_path):
        from src.api.benchmark import _execute_run, get_run, _log
        from src.api.models import BenchmarkRun, get_session

        engine = _engine(tmp_path)
        with get_session(engine) as session:
            run = BenchmarkRun(
                target_kind="provider",
                target_json=json.dumps({"provider": "deepseek", "model": "m"}),
                categories_json=json.dumps(["coding"]),
                status="queued",
            )
            session.add(run)
            session.commit()
            run_id = run.id

        root = tmp_path / "livebench"
        root.mkdir(parents=True)
        (root / "pyproject.toml").write_text("")
        (root / "livebench").mkdir()
        (root / "livebench" / "run_livebench.py").write_text("")
        (root / "livebench" / "show_livebench_result.py").write_text("")
        # NO all_groups.csv → "no parseable category scores"

        class FakeProc:
            def __init__(self, *a, **k):
                self.stdout = iter([])
            def wait(self):
                return 0
            def kill(self):
                pass

        class FakeConfig:
            providers = {"deepseek": {"api_base": "https://api.deepseek.com/v1"}}
            def get_provider_key(self, provider):
                return "sk"

        with patch("src.api.benchmark.core_deps_available", return_value=True), \
             patch("src.api.benchmark.livebench_dir", return_value=str(root)), \
             patch("src.api.benchmark.subprocess.Popen", FakeProc):
            _execute_run(run_id, engine, FakeConfig())
        run = get_run(engine, run_id)
        assert run["status"] == "failed"
        assert "no parseable category scores" in run["error"]


# ── benchmark: livebench_root fallback + core_deps ───────────────────────────

class TestCheckoutResolution:
    def test_livebench_root_fallback_path(self, tmp_path, monkeypatch):
        from src.api import benchmark as bm
        alt = tmp_path / "opt" / "livebench"
        alt.mkdir(parents=True)
        (alt / "pyproject.toml").write_text("")
        (alt / "livebench").mkdir()
        (alt / "livebench" / "run_livebench.py").write_text("")
        monkeypatch.delenv("LCP_MODULES_DIR", raising=False)
        # livebench_root only probes the real /opt/livebench; patch it via env-free
        # _valid_checkout returning True only for that path is wrong. Instead,
        # verify the LCP_MODULES_DIR candidate branch.
        monkeypatch.setattr(bm, "_valid_checkout", lambda p: p == str(alt))
        monkeypatch.setenv("LCP_MODULES_DIR", str(tmp_path / "opt"))
        assert bm.livebench_root() == str(alt)

    def test_livebench_dir_falls_back_to_which(self, tmp_path, monkeypatch):
        from src.api import benchmark as bm
        script_dir = tmp_path / "bin"
        script_dir.mkdir()
        runner = script_dir / "run_livebench.py"
        runner.write_text("")
        monkeypatch.setattr(bm, "livebench_root", lambda: None)
        monkeypatch.setattr("shutil.which", lambda name: str(runner) if name == "run_livebench.py" else None)
        assert bm.livebench_dir() == str(script_dir)

    def test_core_deps_available_subprocess_ok(self, monkeypatch):
        from src.api import benchmark as bm
        result = MagicMock()
        result.returncode = 0
        with patch("subprocess.run", return_value=result) as mock_run:
            assert bm.core_deps_available() is True
        # Called with the probe script.
        assert "find_spec" in mock_run.call_args[0][0][2]

    def test_core_deps_available_exception(self, monkeypatch):
        from src.api import benchmark as bm
        with patch("subprocess.run", side_effect=OSError("no")) as mock_run:
            assert bm.core_deps_available() is False


# ── benchmark: list_runs clamping + run_to_dict JSON fallback ───────────────

class TestRunDict:
    def test_list_runs_clamps_limit(self, tmp_path):
        from src.api.benchmark import list_runs
        engine = _engine(tmp_path)
        # limit=0 falls back to default 50 (int(0 or 50)); offset clamps to >=0.
        result = list_runs(engine, limit=0, offset=-5)
        assert result["limit"] == 50
        assert result["offset"] == 0
        # Oversized limit clamps to 200.
        result2 = list_runs(engine, limit=9999)
        assert result2["limit"] == 200

    def test_run_to_dict_bad_json_fallback(self):
        from src.api.benchmark import _run_to_dict
        run = MagicMock()
        run.id = 1
        run.target_kind = "provider"
        run.target_json = "{bad json"
        run.categories_json = None
        run.status = "done"
        run.started_at = None
        run.finished_at = None
        run.result_json = None
        run.error = None
        run.created_at = "now"
        d = _run_to_dict(run)
        assert d["target"] == {}
        assert d["categories"] is None
