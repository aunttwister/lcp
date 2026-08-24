"""Final targeted gaps: derive second-pass, load_model_registry bad-JSON,
benchmark_import overall branch + CLI file print, load_capability_matrix
source-priority, and the remaining setup/benchmark small branches."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


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


# ── seed_capabilities: derive second-pass (unknown category) ────────────────

class TestDeriveSecondPass:
    def test_derive_unknown_category_appended(self):
        """A subtask category not in CATEGORY_ORDER is still averaged."""
        from src.api.seed_capabilities import derive_category_scores
        result = derive_category_scores({"brand_new_cat": {"task_a": 80.0, "task_b": 60.0}})
        assert result["brand_new_cat"] == 70.0
        assert result["overall"] == 70.0


# ── load_model_registry: bad provider_mappings JSON ─────────────────────────

class TestLoadRegistryBadJson:
    def test_bad_mappings_falls_back_empty(self, db_path):
        from src.api.seed_capabilities import load_model_registry
        from src.api.models import ModelRegistryEntry, get_engine, get_session
        engine = get_engine(db_path)
        with get_session(engine) as s:
            s.add(ModelRegistryEntry(
                logical_name="m", benchmark_key="m",
                provider_mappings_json="{not valid json", updated_at="x",
            ))
            s.commit()
        reg = load_model_registry(db_path)
        assert reg["m"]["provider_mappings"] == {}


# ── benchmark_import: overall branch + CLI file print ───────────────────────

class TestImportOverallBranch:
    def test_import_subtask_only_derives_overall(self, db_path):
        from src.api.benchmark_import import import_csv_string
        from src.api.models import CapabilityMetric, get_engine, get_session
        csv_text = "model,theory_of_mind\ngpt-5.5-xhigh,100.0\n"
        import_csv_string(db_path, csv_text, materialize_capabilities=False)
        engine = get_engine(db_path)
        with get_session(engine) as session:
            overall = session.query(CapabilityMetric).filter_by(
                model="gpt-5.5-thinking", category="overall", task=None
            ).first()
            assert overall is not None
            assert overall.value == 100.0

    def test_cli_file_print(self, db_path, tmp_path, capsys):
        from src.api.benchmark_import import main
        jf = tmp_path / "d.csv"
        jf.write_text("model,code_generation\ngpt-5.5-xhigh,90.0\n")
        with patch("sys.argv", ["benchmark_import", "--db", db_path, "--file", str(jf)]):
            main()
        out = capsys.readouterr().out
        assert "Imported" in out


# ── load_capability_matrix: source priority ─────────────────────────────────

class TestMatrixSourcePriority:
    def test_source_priority_picks_livebench_over_lcp(self, db_path):
        from src.api.seed_capabilities import load_capability_matrix
        from src.api.models import ModelCapability, get_engine, get_session
        engine = get_engine(db_path)
        with get_session(engine) as s:
            s.add(ModelCapability(model="m", task_type="code_generation", score=0.5,
                                  source="lcp_benchmark", release_label="r1", updated_at="x"))
            s.add(ModelCapability(model="m", task_type="code_generation", score=0.9,
                                  source="livebench", release_label="r2", updated_at="x"))
            s.commit()
        matrix = load_capability_matrix(db_path, release="r1")
        # lcp_benchmark (priority 2) wins over livebench (priority 3) for r1.
        assert matrix["code_generation"]["m"] == 0.5


# ── benchmark: remaining small branches ─────────────────────────────────────

class TestBenchmarkSmall:
    def test_coding_deps_available_true(self):
        import src.api.benchmark as bm
        with patch.dict("sys.modules", {"tensorflow": MagicMock()}):
            assert bm.coding_deps_available() is True

    def test_coding_deps_available_false(self):
        import src.api.benchmark as bm
        import sys
        saved = sys.modules.get("tensorflow")
        sys.modules.pop("tensorflow", None)
        try:
            assert bm.coding_deps_available() is False
        finally:
            if saved is not None:
                sys.modules["tensorflow"] = saved

    def test_log_noop_on_empty_line(self):
        import src.api.benchmark as bm
        bm._run_logs.clear()
        bm._log_dir = None
        bm._log(1, "")  # empty → no buffer entry
        assert bm.get_run_log(None, 1) == []

    def test_get_run_log_no_file_returns_empty(self, tmp_path):
        import src.api.benchmark as bm
        bm._run_logs.clear()
        bm._log_dir = str(tmp_path)
        assert bm.get_run_log(None, 999) == []
        bm._log_dir = None


# ── setup: remaining branches ────────────────────────────────────────────────

class TestSetupRemaining:
    def test_is_complete_no_required_steps(self, temp_db):
        from src.api import setup as setup_mod
        cfg = MagicMock()
        cfg.providers = {}
        with patch.object(setup_mod, "provider_steps", return_value=[]):
            assert setup_mod.is_complete(temp_db, cfg) is True
