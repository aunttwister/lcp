"""Small-module coverage gaps: crypto fallback-key error path, reasoning_store
singleton, benchmark_import materialization edge cases, and setup install
progress/failure paths."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest


# ── crypto: fallback-key error path ──────────────────────────────────────────

class TestCryptoExtra:
    def test_fallback_key_io_error_returns_none(self, tmp_path, monkeypatch):
        from src.api import crypto
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.open", side_effect=OSError("denied")):
                key = crypto._load_or_create_fallback_key(str(tmp_path))
        assert key is None

    def test_get_secret_key_raises_when_fallback_fails(self, tmp_path, monkeypatch):
        from src.api import crypto
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(crypto, "_load_or_create_fallback_key", return_value=None):
                with pytest.raises(RuntimeError, match="Unable to obtain a secret key"):
                    crypto.get_secret_key(str(tmp_path))


# ── reasoning_store: singleton lifecycle ─────────────────────────────────────

class TestReasoningStoreSingleton:
    def test_singleton_reset(self):
        from src.api import reasoning_store
        reasoning_store.reset_reasoning_store()
        a = reasoning_store.get_reasoning_store()
        b = reasoning_store.get_reasoning_store()
        assert a is b
        reasoning_store.reset_reasoning_store()
        c = reasoning_store.get_reasoning_store()
        assert c is not a
        reasoning_store.reset_reasoning_store()


# ── benchmark_import: materialization edge cases ─────────────────────────────

@pytest.fixture
def db_path():
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from src.api.models import get_engine, Base
    engine = get_engine(path)
    Base.metadata.create_all(engine)
    engine.dispose()
    yield path
    for ext in ("", "-wal", "-shm"):
        try:
            os.unlink(path + ext)
        except FileNotFoundError:
            pass


class TestMaterializeEdgeCases:
    def test_materialize_capability_updates_existing(self, db_path):
        """Re-importing a metric updates the existing typed row (upsert)."""
        from src.api.benchmark_import import import_payload
        from src.api.models import ModelCapability, get_engine, get_session

        payload = {
            "schema_id": "livebench",
            "release_label": "2026-06-25",
            "models": {"m": {"releases": {"2026-06-25": {"coding": 70.0}}}},
        }
        import_payload(db_path, payload)
        import_payload(db_path, payload)  # idempotent upsert

        engine = get_engine(db_path)
        with get_session(engine) as session:
            rows = session.query(ModelCapability).filter_by(
                model="m", source="livebench"
            ).all()
            assert len(rows) == 1
            assert rows[0].score == 0.7

    def test_materialize_subtasks_updates_existing(self, db_path):
        from src.api.benchmark_import import import_payload
        from src.api.models import ModelCapabilitySubtask, get_engine, get_session

        payload = {
            "schema_id": "livebench",
            "release_label": "2026-06-25",
            "models": {"m": {"subtasks": {"reasoning": {"theory_of_mind": 80.0}}}},
        }
        import_payload(db_path, payload, materialize_capabilities=False)
        import_payload(db_path, payload, materialize_capabilities=False)

        engine = get_engine(db_path)
        with get_session(engine) as session:
            rows = session.query(ModelCapabilitySubtask).filter_by(model="m").all()
            assert len(rows) == 1
            assert rows[0].score == 0.8

    def test_unknown_category_skipped(self, db_path):
        from src.api.benchmark_import import import_payload
        from src.api.models import ModelCapability, get_engine, get_session

        payload = {
            "schema_id": "livebench",
            "release_label": "2026-06-25",
            "models": {"m": {"releases": {"2026-06-25": {"not_a_lb_cat": 90.0}}}},
        }
        import_payload(db_path, payload)
        engine = get_engine(db_path)
        with get_session(engine) as session:
            assert session.query(ModelCapability).filter_by(model="m").count() == 0

    def test_discover_files_invalid_json_skipped(self, tmp_path, monkeypatch):
        from src.api import benchmark_import
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        monkeypatch.setattr(benchmark_import, "BUNDLED_DATA_DIR", str(tmp_path))
        files = benchmark_import.discover_files()
        assert str(bad) in files
        # import_bundled skips it gracefully.
        with patch("src.api.benchmark_import.logger") as mock_log:
            n = benchmark_import.import_bundled("data/costs.db")
        mock_log.warning.assert_called()


# ── setup: install failure paths ─────────────────────────────────────────────

class TestSetupFailurePaths:
    def test_run_install_clone_failure(self, tmp_path, monkeypatch):
        import subprocess
        from src.api import setup as setup_mod
        from src.api.models import get_engine, Base

        engine = get_engine(str(tmp_path / "s.db"))
        Base.metadata.create_all(engine)

        # Save pristine globals so _bench_finish writes don't leak across tests.
        orig_install = setup_mod._bench_install
        orig_last = setup_mod._bench_last
        try:
            monkeypatch.setattr(setup_mod.os, "environ", {"LCP_MODULES_DIR": str(tmp_path / "mods")})
            monkeypatch.setattr("os.makedirs", lambda *a, **k: None)
            monkeypatch.setattr("os.path.isdir", lambda _: False)
            monkeypatch.setattr("shutil.rmtree", lambda *a, **k: None)
            monkeypatch.setattr(setup_mod, "_bench_install", {
                "status": "running", "progress": 0.0, "detail": "", "log": ["cloning..."],
            })

            def fake_stream(cmd, cwd=None, start=0, end=0, status_msg=""):
                raise subprocess.CalledProcessError(128, cmd)

            monkeypatch.setattr(setup_mod, "_stream", fake_stream)
            setup_mod._run_livebench_install(engine)
            last = setup_mod.bench_last()
            assert last is not None and last["status"] == "failed"
        finally:
            setup_mod._bench_install = orig_install
            setup_mod._bench_last = orig_last

    def test_run_install_core_not_importable(self, tmp_path, monkeypatch):
        import subprocess
        from src.api import setup as setup_mod
        from src.api.models import get_engine, Base

        engine = get_engine(str(tmp_path / "s.db"))
        Base.metadata.create_all(engine)

        orig_install = setup_mod._bench_install
        orig_last = setup_mod._bench_last
        try:
            monkeypatch.setattr(setup_mod.os, "environ", {"LCP_MODULES_DIR": str(tmp_path / "mods")})
            monkeypatch.setattr("os.makedirs", lambda *a, **k: None)
            monkeypatch.setattr("os.path.isdir", lambda _: False)
            monkeypatch.setattr("os.path.isfile", lambda p: p.endswith("run_livebench.py") or p.endswith("pyproject.toml"))
            monkeypatch.setattr("shutil.rmtree", lambda *a, **k: None)
            monkeypatch.setattr(setup_mod, "_bench_install", {
                "status": "running", "progress": 0.0, "detail": "", "log": ["cloning..."],
            })

            def fake_stream(cmd, cwd=None, start=0, end=0, status_msg=""):
                pass

            monkeypatch.setattr(setup_mod, "_stream", fake_stream)
            monkeypatch.setattr("src.api.benchmark.core_deps_available", lambda site=None: False)
            setup_mod._run_livebench_install(engine)
            last = setup_mod.bench_last()
            assert last is not None and last["status"] == "failed"
            combined = (last.get("detail") or "") + " " + " ".join(last.get("log") or [])
            assert "LiveBench core install did not take effect" in combined
        finally:
            setup_mod._bench_install = orig_install
            setup_mod._bench_last = orig_last


# ── commandcode: _load_catalog TTL/cooldown branches ─────────────────────────

class TestCommandCodeCatalog:
    def test_catalog_ttl_hit_returns_cache(self):
        import src.api.cost_plugins.commandcode as cc
        import time as _t
        cc._catalog_cache["by_last_seg"] = {"cached": "cached-model"}
        cc._catalog_cache["loaded_ts"] = _t.time()  # fresh → TTL hit
        cc._catalog_cache["failed_ts"] = 0.0
        with patch("urllib.request.urlopen") as mock_open:
            idx = cc._load_catalog()
            mock_open.assert_not_called()
        assert idx == {"cached": "cached-model"}

    def test_catalog_failure_cooldown_returns_cache(self):
        import src.api.cost_plugins.commandcode as cc
        import time as _t
        cc._catalog_cache["by_last_seg"] = {"cached": "cached-model"}
        cc._catalog_cache["loaded_ts"] = 0.0
        cc._catalog_cache["failed_ts"] = _t.time()  # recent failure → cooldown
        with patch("urllib.request.urlopen") as mock_open:
            idx = cc._load_catalog()
            mock_open.assert_not_called()
        assert idx == {"cached": "cached-model"}
