"""Cron view tests — snapshot payload, defensive degradation, decoration."""

import json
import os
import time

import pytest

from src.api import work_cron


@pytest.fixture
def snap_dir(tmp_path, monkeypatch):
    """Point tasks_dir() at a temp tree with a cron snapshot present."""
    layers = tmp_path / ".work-layers"
    layers.mkdir(parents=True)
    ops = tmp_path / "cron-ops"
    ops.mkdir(parents=True)
    now = time.time()
    snap = {
        "generated_at_ts": now - 300,
        "generated_at": "2026-09-19T08:00:00Z",
        "counts": {"total": 3, "active": 2, "paused": 1, "disabled": 0, "error": 1},
        "profiles": [
            {"profile": "homelab-expert-l2", "legacy": False, "error": None, "jobs": [
                {"id": "a1", "name": "Infra Daily Report", "schedule": "0 2 * * *",
                 "status": "active", "next_run_at": "2026-09-20T02:30:00",
                 "last_run_at": "2026-09-19T02:31:00", "last_status": "ok",
                 "last_error": None, "last_delivery_error": None, "script": None,
                 "no_agent": False, "deliver": "origin",
                 "last_execution": {"status": "completed", "finished_at": "2026-09-19T02:31:05"}},
                {"id": "a2", "name": "Broken Watch", "schedule": "0 * * * *",
                 "status": "active", "next_run_at": "2026-09-19T09:00:00",
                 "last_run_at": "2026-09-19T08:00:00", "last_status": "failed",
                 "last_error": "boom", "last_delivery_error": None, "script": None,
                 "no_agent": True, "deliver": "local", "last_execution": None},
            ]},
            {"profile": "homelab-expert-L1", "legacy": True, "error": None, "jobs": [
                {"id": "a3", "name": "Old Weekly", "schedule": "0 5 * * 1",
                 "status": "paused", "next_run_at": None, "last_run_at": "2026-05-30T05:31:00",
                 "last_status": "ok", "last_error": None, "last_delivery_error": None,
                 "script": "/your/data/x/y.sh", "no_agent": False, "deliver": "email",
                 "last_execution": {"status": "completed", "finished_at": "2026-05-30T05:31:10"}},
            ]},
        ],
    }
    (layers / "cron-jobs.json").write_text(json.dumps(snap), encoding="utf-8")
    monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(tmp_path))
    monkeypatch.setenv("LCP_CRON_OPS_DIR", str(ops))
    return tmp_path


class TestCronView:
    def test_present_snapshot(self, snap_dir):
        v = work_cron.cron_view()
        assert v["available"] is True
        assert v["counts"]["total"] == 3
        assert v["counts"]["active"] == 2

    def test_missing_snapshot_degrades(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(tmp_path))
        v = work_cron.cron_view()
        assert v["available"] is False
        assert "has not" in v["hint"].lower() or "missing" in v["hint"].lower()

    def test_corrupt_snapshot_degrades(self, snap_dir):
        (snap_dir / ".work-layers" / "cron-jobs.json").write_text("{not json", encoding="utf-8")
        v = work_cron.cron_view()
        assert v["available"] is False
        assert v["counts"]["total"] == 0

    def test_profile_decoration(self, snap_dir):
        v = work_cron.cron_view()
        l2 = next(p for p in v["profiles"] if p["profile"] == "homelab-expert-l2")
        assert l2["legacy"] is False
        assert l2["counts"]["active"] == 2
        legacy = next(p for p in v["profiles"] if p["profile"] == "homelab-expert-L1")
        assert legacy["legacy"] is True

    def test_job_warn_flags(self, snap_dir):
        v = work_cron.cron_view()
        l2 = next(p for p in v["profiles"] if p["profile"] == "homelab-expert-l2")
        by_name = {j["name"]: j for j in l2["jobs"]}
        assert by_name["Broken Watch"]["warn"] == "error"   # last_error present
        assert by_name["Infra Daily Report"]["warn"] is None  # healthy
        assert by_name["Infra Daily Report"]["last_execution_status"] == "completed"

    def test_relative_time_decorations(self, snap_dir):
        v = work_cron.cron_view()
        l2 = next(p for p in v["profiles"] if p["profile"] == "homelab-expert-l2")
        job = l2["jobs"][0]
        assert job["next_rel"]  # "in X" string
        assert job["last_rel"]
        assert v["generated_rel"]  # snapshot freshness


class TestCronOps:
    def test_submit_create(self, snap_dir):
        op = work_cron.submit_cron_op({
            "action": "create", "profile": "homelab-expert-l2",
            "name": "Watchdog", "schedule": "0 * * * *", "prompt": "run the scan",
            "deliver": "local"})
        assert op["status"] == "pending"
        f = snap_dir / "cron-ops" / "ops" / (op["id"] + ".json")
        assert f.exists()
        stored = json.loads(f.read_text(encoding="utf-8"))
        assert stored["action"] == "create"
        assert stored["no_agent"] is False

    def test_submit_no_agent_script(self, snap_dir):
        op = work_cron.submit_cron_op({
            "action": "create", "profile": "homelab-expert-l2",
            "name": "Watch", "schedule": "30m", "script": "watch.sh",
            "no_agent": True, "deliver": "local"})
        assert op["status"] == "pending"
        assert op["script"] == "watch.sh"

    def test_reject_unknown_action(self, snap_dir):
        with pytest.raises(ValueError):
            work_cron.submit_cron_op({"action": "explode", "profile": "homelab-expert-l2"})

    def test_reject_unknown_profile(self, snap_dir):
        with pytest.raises(ValueError):
            work_cron.submit_cron_op({"action": "create", "profile": "nope",
                                      "name": "x", "schedule": "30m", "prompt": "p"})

    def test_reject_create_without_prompt_or_script(self, snap_dir):
        with pytest.raises(ValueError):
            work_cron.submit_cron_op({"action": "create", "profile": "homelab-expert-l2",
                                      "name": "x", "schedule": "30m"})

    def test_reject_script_with_slash(self, snap_dir):
        with pytest.raises(ValueError):
            work_cron.submit_cron_op({"action": "create", "profile": "homelab-expert-l2",
                                      "name": "x", "schedule": "30m",
                                      "script": "../evil.sh"})

    def test_edit_known_job_ok(self, snap_dir):
        op = work_cron.submit_cron_op({"action": "edit", "profile": "homelab-expert-l2",
                                       "job_id": "a1", "fields": {"schedule": "0 3 * * *"}})
        assert op["fields"]["schedule"] == "0 3 * * *"

    def test_edit_unknown_job_rejected(self, snap_dir):
        with pytest.raises(ValueError):
            work_cron.submit_cron_op({"action": "edit", "profile": "homelab-expert-l2",
                                      "job_id": "zzzz", "fields": {"schedule": "0 3 * * *"}})

    def test_edit_empty_fields_rejected(self, snap_dir):
        with pytest.raises(ValueError):
            work_cron.submit_cron_op({"action": "edit", "profile": "homelab-expert-l2",
                                      "job_id": "a1", "fields": {}})

    def test_ops_view_empty_then_pending(self, snap_dir):
        v = work_cron.cron_ops_view()
        assert v["pending"] == [] and v["done"] == []
        work_cron.submit_cron_op({"action": "pause", "profile": "homelab-expert-l2",
                                  "job_id": "a1"})
        v = work_cron.cron_ops_view()
        assert len(v["pending"]) == 1
        assert v["pending"][0]["job_id"] == "a1"