"""CLI + branch coverage for seed_capabilities and benchmark_import."""

import json
import os
import tempfile
from unittest.mock import patch

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


# ── seed_capabilities CLI ────────────────────────────────────────────────────

class TestSeedCli:
    def test_cli_registry_only(self, db_path, capsys):
        from src.api.seed_capabilities import main
        with patch("sys.argv", ["seed_capabilities", "--db", db_path, "--registry-only"]):
            main()
        out = capsys.readouterr().out
        assert "registry entries" in out

    def test_cli_livebench_only(self, db_path, capsys):
        from src.api.seed_capabilities import main
        with patch("sys.argv", ["seed_capabilities", "--db", db_path, "--livebench-only"]):
            main()
        out = capsys.readouterr().out
        assert "LiveBench capability rows" in out

    def test_cli_default(self, db_path, capsys):
        from src.api.seed_capabilities import main
        with patch("sys.argv", ["seed_capabilities", "--db", db_path]):
            main()
        out = capsys.readouterr().out
        assert "registry entries" in out
        assert "LiveBench capability rows" in out

    def test_cli_release_filter(self, db_path, capsys):
        from src.api.seed_capabilities import main
        with patch("sys.argv", ["seed_capabilities", "--db", db_path, "--release", "2026-06-25"]):
            main()
        out = capsys.readouterr().out
        assert "release=2026-06-25" in out


# ── benchmark_import CLI ─────────────────────────────────────────────────────

class TestImportCli:
    def test_cli_file(self, db_path, capsys, tmp_path):
        from src.api.benchmark_import import main
        jf = tmp_path / "d.json"
        jf.write_text(json.dumps({
            "schema_id": "x", "release_label": "r",
            "models": {"m": {"releases": {"r": {"coding": 90.0}}}},
        }))
        with patch("sys.argv", ["benchmark_import", "--db", db_path, "--file", str(jf)]):
            main()
        out = capsys.readouterr().out
        assert "Imported" in out
        assert "materialized" in out

    def test_cli_bundled(self, db_path, capsys):
        from src.api.benchmark_import import main
        with patch("sys.argv", ["benchmark_import", "--db", db_path, "--dry-run"]):
            main()
        out = capsys.readouterr().out
        assert "dataset file(s)" in out


# ── benchmark_import parse edge cases (remaining branches) ──────────────────

class TestParseBranches:
    def test_parse_missing_release_label(self):
        from src.api.benchmark_import import parse_payload
        with pytest.raises(ValueError, match="release_label"):
            parse_payload({"schema_id": "x", "models": {}})

    def test_parse_models_not_dict(self):
        from src.api.benchmark_import import parse_payload
        with pytest.raises(ValueError, match="must be an object"):
            parse_payload({"schema_id": "x", "release_label": "r", "models": "oops"})

    def test_parse_non_numeric_skipped(self):
        from src.api.benchmark_import import parse_payload
        schema, rel, rows = parse_payload({
            "schema_id": "x", "release_label": "r",
            "models": {"m": {"releases": {"r": {"coding": "not-a-number"}}}},
        })
        assert rows == []

    def test_parse_empty_model_skipped(self):
        from src.api.benchmark_import import parse_payload
        schema, rel, rows = parse_payload({
            "schema_id": "x", "release_label": "r",
            "models": {"  ": {"releases": {"r": {"coding": 1.0}}}},
        })
        assert rows == []

    def test_parse_bad_release_dict_skipped(self):
        from src.api.benchmark_import import parse_payload
        schema, rel, rows = parse_payload({
            "schema_id": "x", "release_label": "r",
            "models": {"m": {"releases": {"r": "not-a-dict"}}},
        })
        assert rows == []


# ── _cleanup_legacy_capabilities superseded-release branch ──────────────────

class TestCleanupBranch:
    def test_cleanup_drops_superseded_release(self, db_path):
        """A model with two release snapshots keeps only the newest."""
        from src.api.seed_capabilities import _cleanup_legacy_capabilities, seed_livebench
        from src.api.models import (
            CapabilityMetric, ModelCapability, get_engine, get_session,
        )
        from src.api.seed_capabilities import LB_TO_LCP

        # Seed the pipeline so metrics exist, then add a superseded capability row.
        seed_livebench(db_path)
        engine = get_engine(db_path)
        with get_session(engine) as session:
            # A hand-typed stale row for deepseek-v4-pro at an OLD release.
            session.add(ModelCapability(
                model="deepseek-v4-pro", task_type="code_generation",
                score=0.4, source="livebench", release_label="2026-06-25",
                updated_at="x",
            ))
            session.commit()

        _cleanup_legacy_capabilities(db_path)
        with get_session(engine) as session:
            stale = session.query(ModelCapability).filter_by(
                model="deepseek-v4-pro", release_label="2026-06-25"
            ).all()
            # deepseek-v4-pro's active release is 2026-08-13 → the 06-25 row is dropped.
            assert stale == []
