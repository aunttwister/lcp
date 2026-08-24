"""Tests for the memory module Setup integration (src/api/setup memory parts)."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.api.models import get_engine, Base
from src.api import setup as setup_mod


@pytest.fixture
def temp_db():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()
    for ext in ["", "-wal", "-shm"]:
        try:
            os.unlink(db_path + ext)
        except FileNotFoundError:
            pass


@pytest.fixture(autouse=True)
def _reset_mem_state():
    """Reset the module-level in-flight/terminal install state between tests."""
    setup_mod._mem_install = None
    setup_mod._mem_last = None
    yield
    setup_mod._mem_install = None
    setup_mod._mem_last = None


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.providers = {}
    cfg.raw = {"providers": {}}
    cfg.save = MagicMock()
    return cfg


class TestMemoryPaths:
    def test_memory_site_uses_modules_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LCP_MODULES_DIR", str(tmp_path / "mods"))
        assert setup_mod.memory_site() == str(tmp_path / "mods" / "memory")

    def test_memory_models_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LCP_MODULES_DIR", str(tmp_path / "mods"))
        assert setup_mod.memory_models_dir() == str(tmp_path / "mods" / "models" / "memory")


class TestMemoryStep:
    def test_memory_step_manifest(self, monkeypatch):
        monkeypatch.setenv("LCP_MODULES_DIR", "/tmp/nowhere-memory")
        step = setup_mod.memory_step()
        assert step["kind"] == "module"
        assert step["name"] == "memory"
        assert step["installed"] is False
        assert step["installing"] is None

    def test_manifest_includes_memory(self, mock_config):
        m = setup_mod.manifest(mock_config)
        names = {mod["name"] for mod in m["modules"]}
        assert {"livebench", "memory"} <= names

    def test_memory_step_reflects_inflight(self, monkeypatch):
        inflight = {"status": "running", "progress": 10.0}
        with patch.object(setup_mod, "_mem_install", inflight), \
             patch("src.api.memory.memory_status", return_value={"available": False}):
            step = setup_mod.memory_step()
        assert step["installing"] is inflight


class TestMemoryInstallCoordinator:
    def test_mem_progress_idle_by_default(self):
        assert setup_mod.mem_progress() is None
        assert setup_mod.mem_last() is None

    def test_start_memory_install_spawns_thread(self, temp_db):
        with patch.object(setup_mod, "_run_memory_install", lambda engine: None):
            state = setup_mod.start_memory_install(temp_db)
        assert state["status"] in ("queued", "running")

    def test_mem_finish_marks_terminal(self, monkeypatch):
        monkeypatch.setattr(setup_mod, "_mem_install", {
            "status": "running", "progress": 42.0, "detail": "x", "log": ["a"],
        })
        setup_mod._mem_finish("done", "Memory installed.")
        assert setup_mod.mem_progress() is None
        last = setup_mod.mem_last()
        assert last is not None
        assert last["status"] == "done"
        assert last["progress"] == 100.0

    def test_mem_update_clamps_progress(self, monkeypatch):
        monkeypatch.setattr(setup_mod, "_mem_install", {
            "status": "running", "progress": 0.0, "detail": "", "log": [],
        })
        setup_mod._mem_update(None, progress=999.0)
        assert setup_mod._mem_install["progress"] == 100.0

    def test_run_memory_install_success(self, temp_db, monkeypatch, tmp_path):
        monkeypatch.setenv("LCP_MODULES_DIR", str(tmp_path / "mods"))
        monkeypatch.setattr(setup_mod.os, "makedirs", lambda *a, **k: None)
        monkeypatch.setattr(setup_mod, "_stream_mem", lambda *a, **k: None)
        monkeypatch.setattr("src.api.memory.memory_available", lambda site=None: True)
        # Skip the model pre-download subprocess.
        def _no_popen(*a, **k):
            raise FileNotFoundError("skip model")
        monkeypatch.setattr(setup_mod.subprocess, "Popen", _no_popen)
        setup_mod._mem_install = {
            "status": "running", "progress": 0.0, "detail": "", "log": [],
        }

        setup_mod._run_memory_install(temp_db)
        assert setup_mod.mem_last()["status"] == "done"
        assert setup_mod.load_state(temp_db)["module:memory"]["status"] == "done"

    def test_run_memory_install_failure(self, temp_db, monkeypatch, tmp_path):
        monkeypatch.setenv("LCP_MODULES_DIR", str(tmp_path / "mods"))
        monkeypatch.setattr(setup_mod.os, "makedirs", lambda *a, **k: None)
        def _boom(*a, **k):
            raise Exception("pip exploded")
        monkeypatch.setattr(setup_mod, "_stream_mem", _boom)
        setup_mod._mem_install = {
            "status": "running", "progress": 0.0, "detail": "", "log": [],
        }

        setup_mod._run_memory_install(temp_db)
        assert setup_mod.mem_last()["status"] == "failed"


class TestRemoveMemory:
    def test_remove_memory_deletes_dirs(self, temp_db, monkeypatch, tmp_path):
        mods = tmp_path / "mods"
        site = mods / "memory"
        models = mods / "models" / "memory"
        site.mkdir(parents=True)
        models.mkdir(parents=True)
        (site / "x").write_text("")
        monkeypatch.setenv("LCP_MODULES_DIR", str(mods))

        result = setup_mod.remove_memory(temp_db)
        assert result["removed"] is True
        assert not site.exists()
        assert not models.exists()
        assert setup_mod.load_state(temp_db)["module:memory"]["status"] == "removed"

    def test_remove_memory_noop_when_absent(self, temp_db, monkeypatch, tmp_path):
        monkeypatch.setenv("LCP_MODULES_DIR", str(tmp_path / "mods"))
        result = setup_mod.remove_memory(temp_db)
        assert result["removed"] is True
        assert result["paths"] == []


class TestMemoryAvailableProbe:
    def test_memory_available_false_when_missing(self, monkeypatch):
        monkeypatch.setattr(setup_mod.subprocess, "run",
                            lambda *a, **k: type("R", (), {"returncode": 1})())
        from src.api import memory as mem_mod
        assert mem_mod.memory_available() is False

    def test_memory_available_true_when_importable(self, monkeypatch):
        monkeypatch.setattr(setup_mod.subprocess, "run",
                            lambda *a, **k: type("R", (), {"returncode": 0})())
        from src.api import memory as mem_mod
        assert mem_mod.memory_available() is True
