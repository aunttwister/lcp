"""Unit tests for the Tasks-view document extraction (work_tasks.py).

Covers what the expander displays: the plan summary derivation, the
capped plan/RESULTS text with explicit truncation flags, and the
artifacts list. Everything runs against temp dirs -- never the real
task tree.
"""

import os

import pytest

from src.api import work_tasks as wt


def _tree(tmp_path, state="completed", slug="demo-task", plan=None,
          results=None, extra_files=()):
    """Build one task dir under a state dir, return (tasks_root, task_dir)."""
    tdir = tmp_path / state / slug
    tdir.mkdir(parents=True)
    if plan is not None:
        (tdir / "PLAN.md").write_text(plan, encoding="utf-8")
    if results is not None:
        (tdir / "RESULTS.md").write_text(results, encoding="utf-8")
    for f in extra_files:
        (tdir / f).write_text("x", encoding="utf-8")
    return tmp_path, tdir


class TestPlanSummary:
    def test_skips_frontmatter_and_headings(self):
        plan = (
            "# Task: Fix the thing\n"
            "**Created:** 2026-09-19 | **Status:** in_progress\n"
            "**Branch:** feat/x\n"
            "## Problem\n"
            "The widget crashes on empty state.\n"
        )
        assert wt._plan_summary(plan) == "The widget crashes on empty state."

    def test_skips_meta_lines_addressing_variants(self):
        plan = (
            "**Status:** completed\n"
            "**Owner:** homelab-expert-l2\n"
            "**User ask (verbatim):** *\"do it\"*\n"
            "First real sentence about what happened.\n"
        )
        assert wt._plan_summary(plan) == "First real sentence about what happened."

    def test_returns_none_for_garbage(self):
        assert wt._plan_summary("") is None
        assert wt._plan_summary("###\n##\n#\n") is None

    def test_caps_at_320_chars(self):
        plan = "word " * 200
        s = wt._plan_summary(plan)
        assert s is not None
        assert len(s) == 320


class TestMomentPayload:
    def test_plan_summary_and_text_present(self, tmp_path, monkeypatch):
        root, _ = _tree(tmp_path, plan="# T\n\nBody line one.\nSecond line.\n")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        (m,) = [m for m in wt.task_moments()]
        assert m["payload"]["plan_summary"] == "Body line one."
        assert m["payload"]["plan_text"] == "# T\n\nBody line one.\nSecond line.\n"
        assert m["payload"]["plan_truncated"] is False
        assert m["payload"]["has_plan"] is True

    def test_results_read_when_present(self, tmp_path, monkeypatch):
        root, _ = _tree(tmp_path, results="# RESULTS\n\nShipped.\n")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        (m,) = wt.task_moments()
        assert m["payload"]["results_text"] == "# RESULTS\n\nShipped.\n"
        assert m["payload"]["results_truncated"] is False

    def test_results_absent_when_missing(self, tmp_path, monkeypatch):
        root, _ = _tree(tmp_path, plan="# T\n")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        (m,) = wt.task_moments()
        assert m["payload"]["results_text"] is None

    def test_artifacts_exclude_plan(self, tmp_path, monkeypatch):
        root, _ = _tree(tmp_path, plan="# T\n", results="# R\n",
                        extra_files=("phase-1-research.md",))
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        (m,) = wt.task_moments()
        assert set(m["payload"]["artifacts"]) == {"RESULTS.md", "phase-1-research.md"}

    def test_long_plan_is_truncated_and_flagged(self, tmp_path, monkeypatch):
        root, _ = _tree(tmp_path, plan="# T\n" + ("x" * 9000))
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        (m,) = wt.task_moments()
        assert m["payload"]["plan_truncated"] is True
        assert len(m["payload"]["plan_text"]) <= wt._PLAN_CAP

    def test_long_results_are_truncated_and_flagged(self, tmp_path, monkeypatch):
        root, _ = _tree(tmp_path, results=("x" * 2500))
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        (m,) = wt.task_moments()
        assert m["payload"]["results_truncated"] is True
        assert len(m["payload"]["results_text"]) <= wt._RESULTS_CAP

    def test_corrupt_plan_never_raises(self, tmp_path, monkeypatch):
        tdir = tmp_path / "completed" / "bad"
        tdir.mkdir(parents=True)
        (tdir / "PLAN.md").write_bytes(b"\xff\xfe\x00garbage")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(tmp_path))
        (m,) = wt.task_moments()
        # errors="replace" keeps the text readable; the moment must exist.
        assert m["payload"]["has_plan"] is True