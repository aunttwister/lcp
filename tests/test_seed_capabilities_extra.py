"""Tests for seed_capabilities CLI + cleanup, router registry/select paths,
and benchmark_import edge cases (extending the earlier targeted suites)."""

import os
import tempfile

import pytest

from src.api.seed_capabilities import (
    DERIVED_TASKS,
    LIVEBENCH_RELEASE,
    load_capability_matrix,
    seed_livebench,
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

    def test_matrix_includes_derived_coding_subskills(self, db_path):
        seed_model_registry(db_path)
        seed_livebench(db_path)
        matrix = load_capability_matrix(db_path)
        # debugging + unit_tests are derived from code_generation (coding
        # subskills), so they must be present with the same per-model scores.
        for derived in ("debugging", "unit_tests"):
            assert derived in matrix
            assert matrix[derived]["deepseek-v4-pro"] == matrix["code_generation"]["deepseek-v4-pro"]


# ── derived tasks registry ───────────────────────────────────────────────────

def test_derived_tasks_registry():
    # debugging + unit_tests are derived from code_generation.
    assert DERIVED_TASKS == {
        "debugging": "code_generation",
        "unit_tests": "code_generation",
    }


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

    def test_parse_csv_invalid_types_skipped(self):
        from src.api.benchmark_import import parse_livebench_csv
        # Non-numeric values are skipped gracefully.
        schema, rel, rows = parse_livebench_csv(
            "model,code_generation\ngpt-5.5-xhigh,not-a-number\n")
        assert rows == []

    def test_import_module_override_wins(self, db_path, tmp_path, monkeypatch):
        """A module-provided CSV overrides the bundled CSV."""
        from src.api.benchmark_import import import_bundled
        mod_data = tmp_path / "modules" / "mymod" / "data"
        mod_data.mkdir(parents=True)
        (mod_data / "table_2026_06_25.csv").write_text(
            "model,code_generation\nonly-in-module,99.0\n"
            "gpt-5.5-xhigh,82.609\n")
        monkeypatch.setenv("LCP_MODULES_DIR", str(tmp_path / "modules"))
        import_bundled(db_path)
        from src.api.models import ModelCapability, get_engine, get_session
        engine = get_engine(db_path)
        with get_session(engine) as session:
            rows = session.query(ModelCapability).filter_by(model="gpt-5.5-thinking").all()
            assert rows, "module-provided CSV should be imported (bundled overridden)"
