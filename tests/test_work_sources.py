"""work_sources tests — validation, roundtrip, resolved defaults, fallbacks."""

import json

import pytest

from src.api import work_sources


@pytest.fixture
def srcdir(tmp_path, monkeypatch):
    monkeypatch.setenv("LCP_CRON_OPS_DIR", str(tmp_path / "cron-ops"))
    monkeypatch.setenv("LCP_WORK_SOURCES", str(tmp_path / "work-sources.json"))
    return tmp_path


class TestValidation:
    def test_valid_payload(self):
        out = work_sources.validate({
            "version": 1,
            "hermes_profiles_dir": "/srv/profiles",
            "tasks_root": "/srv/profiles/pv/work/tasks",
            "ops_dir": "/srv/ops",
            "profiles": {"pv": {"legacy": True, "hidden": True}},
        })
        assert out["hermes_profiles_dir"] == "/srv/profiles"
        assert out["profiles"]["pv"]["legacy"] is True
        assert out["profiles"]["pv"]["hidden"] is True

    def test_rejects_bad_version(self):
        with pytest.raises(ValueError):
            work_sources.validate({"version": 2})

    def test_rejects_relative_path(self):
        with pytest.raises(ValueError):
            work_sources.validate({"version": 1,
                                   "hermes_profiles_dir": "../etc"})

    def test_rejects_shell_chars(self):
        with pytest.raises(ValueError):
            work_sources.validate({"version": 1,
                                   "hermes_profiles_dir": "/a; rm -rf /"})

    def test_rejects_bad_profile_name(self):
        with pytest.raises(ValueError):
            work_sources.validate({"version": 1,
                                   "profiles": {"bad/name": {}}})

    def test_rejects_unknown_profile_key(self):
        with pytest.raises(ValueError):
            work_sources.validate({"version": 1,
                                   "profiles": {"pv": {"evil": 1}}})

    def test_ignores_reserved_extras_noted(self):
        out = work_sources.validate({"version": 1, "future_field": 1})
        assert out["notes"]["ignored_keys"] == ["future_field"]


class TestSaveLoad:
    def test_roundtrip(self, srcdir):
        payload = {
            "version": 1,
            "hermes_profiles_dir": "/srv/profiles",
            "tasks_root": "/srv/profiles/pv/work/tasks",
            "profiles": {"pv": {"cron_store": "/srv/stores/pv"}},
        }
        saved = work_sources.save_sources(payload)
        assert saved["updated_at"]
        loaded = work_sources.load_sources()
        assert loaded["tasks_root"] == "/srv/profiles/pv/work/tasks"
        assert loaded["profiles"]["pv"]["cron_store"] == "/srv/stores/pv"

    def test_missing_file_is_none(self, srcdir, monkeypatch):
        monkeypatch.setenv("LCP_WORK_SOURCES", str(srcdir / "nope.json"))
        assert work_sources.load_sources() is None

    def test_corrupt_file_is_none(self, srcdir):
        (srcdir / "work-sources.json").write_text("{bad", encoding="utf-8")
        assert work_sources.load_sources() is None

    def test_save_rejects_invalid(self, srcdir):
        with pytest.raises(ValueError):
            work_sources.save_sources({"version": 1,
                                       "hermes_profiles_dir": "relative"})
        assert work_sources.load_sources() is None  # nothing written


class TestResolvedView:
    def _default_snapshot(self, srcdir):
        layers = srcdir / ".work-layers"
        layers.mkdir(exist_ok=True)
        snap = {
            "generated_at": "2026-09-19T10:00:00Z",
            "counts": {"total": 1, "active": 1, "paused": 0, "disabled": 0, "error": 0},
            "profiles": [
                {"profile": "homelab-expert-l2", "legacy": False, "jobs": []},
                {"profile": "homelab-expert-L1", "legacy": True, "jobs": []},
            ],
        }
        (layers / "cron-jobs.json").write_text(json.dumps(snap), encoding="utf-8")
        import os
        os.environ["LCP_WORK_TASKS_DIR"] = str(srcdir)

    def test_unconfigured_defaults(self, srcdir):
        self._default_snapshot(srcdir)
        v = work_sources.resolved_view()
        assert v["configured"] is False
        assert v["hermes_profiles_dir"].startswith("/root/")
        profs = v["profiles"]
        assert "homelab-expert-l2" in profs
        assert profs["homelab-expert-L1"]["legacy"] is True  # snapshot legacy

    def test_configured_overrides(self, srcdir):
        self._default_snapshot(srcdir)
        work_sources.save_sources({
            "version": 1,
            "hermes_profiles_dir": "/srv/profiles",
            "tasks_root": "/srv/profiles/pv/work/tasks",
            "profiles": {"homelab-expert-l2": {"tasks_root": "/srv/alt/tasks"}},
        })
        v = work_sources.resolved_view()
        assert v["configured"] is True
        assert v["tasks_root"] == "/srv/profiles/pv/work/tasks"
        assert v["profiles"]["homelab-expert-l2"]["tasks_root"] == "/srv/alt/tasks"
        assert v["profiles"]["homelab-expert-l2"]["configured"] is True
        assert v["profiles"]["homelab-expert-L1"]["configured"] is False