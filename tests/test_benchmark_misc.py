"""Final: validate_categories empty, effective_releases newest, livebench_root
/opt fallback, core_deps_available with site PYTHONPATH."""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def db_path():
    import tempfile as _t
    fd, path = _t.mkstemp(suffix=".db")
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


class TestValidateCategories:
    def test_empty_strings_skipped(self):
        from src.api.benchmark import validate_categories
        # Leading whitespace entries are skipped (line 94).
        assert validate_categories(["coding", "   ", "math"]) == ["coding", "math"]


class TestEffectiveReleasesNewest:
    def test_pinned_missing_uses_newest(self):
        from src.api.seed_capabilities import effective_releases
        from src.api.models import ModelCapability
        from datetime import datetime, timezone

        def row(model, release):
            return ModelCapability(model=model, task_type="t", score=0.5,
                                   source="livebench", release_label=release,
                                   updated_at=datetime.now(timezone.utc).isoformat())

        rows = [row("m", "2026-06-25"), row("m", "2026-08-13")]
        registry = {"m": {"benchmark_key": "m", "active_release": "2099-01-01",
                          "provider_mappings": {}}}
        out = effective_releases(rows, registry)
        assert out["m"] == "2026-08-13"  # pinned missing → newest


class TestLivebenchRootFallback:
    def test_fallback_to_opt_livebench(self, tmp_path, monkeypatch):
        import src.api.benchmark as bm
        monkeypatch.delenv("LCP_MODULES_DIR", raising=False)
        monkeypatch.setattr(bm, "_valid_checkout", lambda p: p == "/opt/livebench")
        assert bm.livebench_root() == "/opt/livebench"

    def test_no_root_returns_none(self, monkeypatch):
        import src.api.benchmark as bm
        monkeypatch.delenv("LCP_MODULES_DIR", raising=False)
        monkeypatch.setattr(bm, "_valid_checkout", lambda p: False)
        assert bm.livebench_root() is None


class TestCoreDepsWithSite:
    def test_site_prepends_pythonpath(self, monkeypatch):
        import src.api.benchmark as bm
        result = MagicMock()
        result.returncode = 0
        with patch("subprocess.run", return_value=result) as mock_run:
            with patch("src.api.setup.livebench_pythonpath", return_value="/mods/site:/mods/livebench"):
                assert bm.core_deps_available(site="/mods/site") is True
        env = mock_run.call_args[1].get("env", {})
        assert "/mods/site" in env.get("PYTHONPATH", "")


class TestBenchmarkStatusBranches:
    def test_status_unavailable_no_modules_dir(self, monkeypatch):
        import src.api.benchmark as bm
        monkeypatch.delenv("LCP_MODULES_DIR", raising=False)
        monkeypatch.setattr(bm, "livebench_dir", lambda: None)
        status = bm.benchmark_status()
        assert status["available"] is False
        assert "WITH_BENCH" in status["reason"]
