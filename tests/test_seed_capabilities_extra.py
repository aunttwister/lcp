"""Tests for seed_capabilities CLI + cleanup, router registry/select paths,
and benchmark_import edge cases (extending the earlier targeted suites)."""

import json
import os
import tempfile

import pytest

from src.api.seed_capabilities import (
    LIVEBENCH_RELEASE,
    load_capability_matrix,
    seed_livebench,
    seed_livebench_tasks,
    seed_model_registry,
    _cleanup_legacy_capabilities,
)


@pytest.fixture
def db_path():
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


# ── seed_model_registry / cleanup ────────────────────────────────────────────

class TestSeedRegistry:
    def test_seed_and_sync(self, db_path):
        n = seed_model_registry(db_path)
        assert n == len(__import__("src.api.seed_capabilities", fromlist=["DEFAULT_MODEL_REGISTRY"]).DEFAULT_MODEL_REGISTRY)
        # Idempotent on second run.
        n2 = seed_model_registry(db_path)
        assert n2 == 0
        # sync re-applies defaults (still idempotent count-wise on unchanged data).
        n3 = seed_model_registry(db_path, sync=True)
        assert n3 == len(__import__("src.api.seed_capabilities", fromlist=["DEFAULT_MODEL_REGISTRY"]).DEFAULT_MODEL_REGISTRY)

    def test_load_registry_roundtrip(self, db_path):
        seed_model_registry(db_path)
        from src.api.seed_capabilities import load_model_registry
        reg = load_model_registry(db_path)
        assert "deepseek-v4-pro" in reg
        assert reg["deepseek-v4-pro"]["benchmark_key"] == "deepseek-v4-pro"
        assert reg["deepseek-v4-pro"]["provider_mappings"]["commandcode"] == "deepseek/deepseek-v4-pro"

    def test_cleanup_removes_legacy_rows(self, db_path):
        """Legacy unversioned + superseded rows are removed after cleanup."""
        from src.api.models import ModelCapability, get_engine, get_session
        from src.api.seed_capabilities import LIVEBENCH_DATA, LB_TO_LCP

        engine = get_engine(db_path)
        now = "2026-08-17T00:00:00"
        with get_session(engine) as session:
            # Simulate a stale row that should be dropped.
            session.add(ModelCapability(
                model="deepseek-v4-flash-0731", task_type="code_generation",
                score=0.5, source="livebench", release_label=None, updated_at=now,
            ))
            session.add(ModelCapability(
                model="deepseek-v4-pro", task_type="code_generation",
                score=0.4, source="livebench", release_label=None, updated_at=now,  # unversioned → dropped
            ))
            session.commit()

        # Run the import pipeline (writes metrics) then cleanup.
        seed_livebench(db_path)
        _cleanup_legacy_capabilities(db_path)

        with get_session(engine) as session:
            stale = session.query(ModelCapability).filter_by(
                model="deepseek-v4-flash-0731"
            ).first()
            assert stale is None
            unversioned = session.query(ModelCapability).filter_by(
                model="deepseek-v4-pro", release_label=None
            ).all()
            assert unversioned == []


# ── load_capability_matrix resolution ────────────────────────────────────────

class TestCapabilityMatrix:
    def test_matrix_resolves_active_release(self, db_path):
        seed_model_registry(db_path)
        seed_livebench(db_path)
        matrix = load_capability_matrix(db_path)
        # deepseek-v4-pro should be present under reasoning_chain.
        assert "reasoning_chain" in matrix
        assert "deepseek-v4-pro" in matrix["reasoning_chain"]

    def test_matrix_release_filter(self, db_path):
        seed_livebench(db_path)
        matrix = load_capability_matrix(db_path, release=LIVEBENCH_RELEASE)
        assert "gpt-5.6-luna" in matrix.get("casual_chat", {})


# ── benchmark_import edge cases ──────────────────────────────────────────────

class TestImportEdgeCases:
    def test_import_bundled_dry_run_no_write(self, db_path):
        from src.api.benchmark_import import import_bundled
        from src.api.models import CapabilityMetric, get_engine, get_session
        n = import_bundled(db_path, dry_run=True)
        assert n > 0
        engine = get_engine(db_path)
        with get_session(engine) as session:
            assert session.query(CapabilityMetric).count() == 0

    def test_parse_payload_invalid_types(self):
        from src.api.benchmark_import import parse_payload
        # Bad nested types are skipped gracefully.
        schema, rel, rows = parse_payload({
            "schema_id": "x", "release_label": "r",
            "models": {"m": {"releases": "not-a-dict", "subtasks": 5}},
        })
        assert rows == []

    def test_import_module_override_wins(self, db_path, tmp_path, monkeypatch):
        """A module-provided dataset with the same schema_id overrides bundled."""
        from src.api.benchmark_import import import_bundled
        mod_data = tmp_path / "modules" / "mymod" / "data"
        mod_data.mkdir(parents=True)
        (mod_data / "livebench.json").write_text(json.dumps({
            "schema_id": "livebench",
            "release_label": "2026-06-25",
            "models": {"only-in-module": {"releases": {"2026-06-25": {"coding": 99.0}}}},
        }))
        monkeypatch.setenv("LCP_MODULES_DIR", str(tmp_path / "modules"))
        import_bundled(db_path)
        from src.api.models import ModelCapability, get_engine, get_session
        engine = get_engine(db_path)
        with get_session(engine) as session:
            rows = session.query(ModelCapability).filter_by(model="only-in-module").all()
            assert rows, "module-provided model should be imported"
