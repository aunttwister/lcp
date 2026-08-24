"""CLI + branch coverage for seed_capabilities and benchmark_import."""

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
        jf = tmp_path / "d.csv"
        jf.write_text("model,code_generation,theory_of_mind\ngpt-5.5-xhigh,82.609,84.615\n")
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


# ── benchmark_import CSV parse edge cases (remaining branches) ───────────────

class TestParseBranches:
    def test_parse_empty_csv(self):
        from src.api.benchmark_import import parse_livebench_csv
        schema, rel, rows = parse_livebench_csv("model,foo\n")
        assert schema == "livebench"
        assert rows == []

    def test_parse_non_numeric_skipped(self):
        from src.api.benchmark_import import parse_livebench_csv
        schema, rel, rows = parse_livebench_csv(
            "model,code_generation\ngpt-5.5-xhigh,not-a-number\n")
        assert rows == []

    def test_parse_unmapped_model_skipped(self):
        from src.api.benchmark_import import parse_livebench_csv
        schema, rel, rows = parse_livebench_csv(
            "model,code_generation\nunmapped-model,90.0\n")
        assert rows == []

    def test_parse_unknown_column_skipped(self):
        from src.api.benchmark_import import parse_livebench_csv
        schema, rel, rows = parse_livebench_csv(
            "model,not_a_lb_task\ngpt-5.5-xhigh,90.0\n")
        assert rows == []

    def test_parse_no_header(self):
        from src.api.benchmark_import import parse_livebench_csv
        schema, rel, rows = parse_livebench_csv("")
        assert schema == "livebench"
        assert rows == []


# ── _cleanup_legacy_capabilities superseded-release branch ──────────────────

class TestCleanupBranch:
    def test_cleanup_drops_superseded_release(self, db_path):
        """A model with two release snapshots keeps only the newest."""
        from src.api.seed_capabilities import _cleanup_legacy_capabilities, seed_livebench
        from src.api.models import (
            ModelCapability,
            get_engine,
            get_session,
        )

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
