"""Unit tests for the Tasks-view document extraction (work_tasks.py).

Covers what the expander displays: the plan summary derivation, the
capped plan/RESULTS text with explicit truncation flags, and the
artifacts list. Everything runs against temp dirs -- never the real
task tree.
"""

import json
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


class TestClassificationAndSummary:
    def _with_index(self, tmp_path, monkeypatch, index_json):
        root, _ = _tree(tmp_path, state="in_progress", slug="demo-task", plan="# T\n\nBody.\n")
        wl = tmp_path / ".work-layers"
        wl.mkdir()
        (wl / "classifications.json").write_text(index_json, encoding="utf-8")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        return root

    def test_classification_attached_when_index_present(self, tmp_path, monkeypatch):
        idx = json.dumps({"by_label": {"infrastructure": 1},
                          "labels": {"infrastructure": {"description": "x"}},
                          "tasks": {"demo-task": {"label": "infrastructure", "score": 0.71,
                                                  "state": "in_progress"}}})
        self._with_index(tmp_path, monkeypatch, idx)
        (m,) = wt.task_moments()
        assert m["payload"]["classification"] == {"label": "infrastructure", "score": 0.71}

    def test_no_classification_without_label(self, tmp_path, monkeypatch):
        idx = json.dumps({"tasks": {"demo-task": {"label": None, "score": 0.2}}})
        self._with_index(tmp_path, monkeypatch, idx)
        (m,) = wt.task_moments()
        assert m["payload"]["classification"] is None

    def test_view_exposes_by_label_rollup(self, tmp_path, monkeypatch):
        idx = json.dumps({"by_label": {"infrastructure": 1, "automation": 0},
                          "labels": {"infrastructure": {"description": "x"},
                                     "automation": {"description": "y"}},
                          "tasks": {"demo-task": {"label": "infrastructure", "score": 0.71}}})
        self._with_index(tmp_path, monkeypatch, idx)
        v = wt.tasks_view()
        assert v["by_label"]["infrastructure"] == 1
        assert v["classified"] == 1
        assert v["labels_meta"]["infrastructure"]["description"] == "x"

    def test_corrupt_index_is_ignored(self, tmp_path, monkeypatch):
        self._with_index(tmp_path, monkeypatch, "{not json")
        (m,) = wt.task_moments()
        assert m["payload"]["classification"] is None
        v = wt.tasks_view()
        assert v["by_label"] == {}
        assert v["classified"] == 0

    def test_state_summary_rendered_from_file(self, tmp_path, monkeypatch):
        root, tdir = _tree(tmp_path, state="in_progress", slug="demo-task", plan="# T\n")
        (tdir / "STATE-SUMMARY.md").write_text(
            "# STATE-SUMMARY\n\n**PLAN claims** x\n\n**Verdict** healthy\n", encoding="utf-8")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        (m,) = wt.task_moments()
        p = m["payload"]["state_summary"]
        assert p["present"] is True
        assert "**PLAN claims** x" in p["text"]
        assert "<strong>PLAN claims</strong> x" in p["html"]
        # STATE-SUMMARY.md is a work-layers artifact, not a task artifact.
        assert "STATE-SUMMARY.md" not in m["payload"]["artifacts"]

    def test_state_summary_absent_when_missing(self, tmp_path, monkeypatch):
        root, _ = _tree(tmp_path, state="in_progress", slug="demo-task", plan="# T\n")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        (m,) = wt.task_moments()
        assert m["payload"]["state_summary"]["present"] is False
        assert m["payload"]["state_summary"]["html"] is None

    def test_session_facts_rendered_from_file(self, tmp_path, monkeypatch):
        root, tdir = _tree(tmp_path, state="in_progress", slug="demo-task", plan="# T\n")
        (tdir / "SESSION-FACTS.md").write_text(
            "### 2026-09-19 06:00 UTC\n\nPhase 2 done in session.\n", encoding="utf-8")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        (m,) = wt.task_moments()
        p = m["payload"]["session_facts"]
        assert p["present"] is True
        assert "Phase 2 done in session." in p["text"]
        assert "Phase 2 done in session." in p["html"]
        assert "SESSION-FACTS.md" not in m["payload"]["artifacts"]

    def test_session_facts_absent_when_missing(self, tmp_path, monkeypatch):
        root, _ = _tree(tmp_path, state="in_progress", slug="demo-task", plan="# T\n")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        (m,) = wt.task_moments()
        assert m["payload"]["session_facts"]["present"] is False

    def test_long_session_facts_truncated_and_flagged(self, tmp_path, monkeypatch):
        root, tdir = _tree(tmp_path, state="in_progress", slug="demo-task", plan="# T\n")
        (tdir / "SESSION-FACTS.md").write_text("x" * 6000, encoding="utf-8")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        (m,) = wt.task_moments()
        p = m["payload"]["session_facts"]
        assert len(p["text"]) <= 4000
        assert p["truncated"] is True


class TestAssessmentFeed:
    def test_feed_empty_when_no_ledger(self, tmp_path, monkeypatch):
        root, _ = _tree(tmp_path, state="in_progress", slug="demo-task", plan="# T\n")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        assert wt._assessment_feed() == []

    def test_feed_skips_bad_lines_and_orders_newest(self, tmp_path, monkeypatch):
        root, _ = _tree(tmp_path, state="in_progress", slug="demo-task", plan="# T\n")
        wl = tmp_path / ".work-layers"
        wl.mkdir()
        (wl / "assessments.jsonl").write_text(
            "{not json}\n"
            + json.dumps({"ts": 100.0, "action": "complete", "slug": "a",
                          "summary": "s", "applied": True, "reason": None}) + "\n"
            + json.dumps({"ts": 200.0, "action": "create", "slug": "b",
                          "summary": "t", "applied": False, "reason": "dup"}) + "\n",
            encoding="utf-8")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        feed = wt._assessment_feed()
        assert len(feed) == 2
        assert feed[0]["slug"] == "b"  # newest first
        assert feed[0]["ts_iso"].startswith("1970")
        assert feed[1]["applied"] is True

    def test_tasks_view_no_longer_carries_assessments(self, tmp_path, monkeypatch):
        """The ledger moved to its own tab — the task-tree payload must not keep
        paying for it on every lazy fetch."""
        root, _ = _tree(tmp_path, state="in_progress", slug="demo-task", plan="# T\n")
        wl = tmp_path / ".work-layers"
        wl.mkdir()
        (wl / "assessments.jsonl").write_text(
            json.dumps({"ts": 300.0, "action": "note", "slug": "demo-task",
                        "summary": "touched", "applied": True, "reason": None}) + "\n",
            encoding="utf-8")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        assert "assessments" not in wt.tasks_view()

    def test_assessments_view_rollup_and_rich_fields(self, tmp_path, monkeypatch):
        root, _ = _tree(tmp_path, state="in_progress", slug="demo-task", plan="# T\n")
        wl = tmp_path / ".work-layers"
        wl.mkdir()
        (wl / "assessments.jsonl").write_text(
            json.dumps({"ts": 300.0, "round_actor": "homelab-expert-l1",
                        "action": "complete", "slug": "demo-task",
                        "title": "Demo task", "summary": "finding " * 400,
                        "evidence": ["sess-1", "sess-2"], "applied": True}) + "\n"
            + json.dumps({"ts": 200.0, "round_actor": "homelab-expert-l1",
                          "action": "complete", "slug": "other",
                          "title": "Other", "summary": "dup", "evidence": [],
                          "applied": False, "reason": "duplicate"}) + "\n",
            encoding="utf-8")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        v = wt.assessments_view()
        assert v["total"] == 2 and v["applied"] == 1 and v["skipped"] == 1
        assert v["by_action"] == {"complete": 2}
        newest = v["records"][0]                    # newest first
        assert newest["round_actor"] == "homelab-expert-l1"
        assert newest["title"] == "Demo task"
        assert newest["evidence"] == ["sess-1", "sess-2"]
        assert newest["n_evidence"] == 2
        assert newest["summary_html"]               # markdown-rendered finding
        assert len(newest["summary"]) == wt._ASSESS_SUMMARY_CAP
        assert newest["summary_truncated"] is True
        assert v["records"][1]["reason"] == "duplicate"

    def test_assessments_view_tolerates_scalar_evidence(self, tmp_path, monkeypatch):
        """One malformed record must not blank the tab."""
        root, _ = _tree(tmp_path, state="in_progress", slug="demo-task", plan="# T\n")
        wl = tmp_path / ".work-layers"
        wl.mkdir()
        (wl / "assessments.jsonl").write_text(
            json.dumps({"ts": 1.0, "action": "note", "slug": "d",
                        "evidence": "sess-solo", "applied": True}) + "\n",
            encoding="utf-8")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        feed = wt.assessments_view()["records"]
        assert feed[0]["evidence"] == ["sess-solo"]
        assert feed[0]["n_evidence"] == 1
        assert feed[0]["title"] == ""

    def test_assessments_view_empty_when_no_ledger(self, tmp_path, monkeypatch):
        root, _ = _tree(tmp_path, state="in_progress", slug="demo-task", plan="# T\n")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        v = wt.assessments_view()
        assert v["records"] == [] and v["total"] == 0 and v["by_action"] == {}


class TestTasksPageTabs:
    """The page renders two tabs off one route; these are the smoke tests that
    catch a Jinja-level break (the 500 class of bug that unit-testing the API
    alone cannot see)."""

    @staticmethod
    def _cfg():
        from unittest.mock import MagicMock
        cfg = MagicMock()
        cfg._data = {}
        return cfg

    def test_assessments_tab_renders_ledger_rows(self, tmp_path, monkeypatch):
        root, _ = _tree(tmp_path, state="in_progress", slug="demo-task", plan="# T\n")
        wl = tmp_path / ".work-layers"
        wl.mkdir()
        (wl / "assessments.jsonl").write_text(
            json.dumps({"ts": 300.0, "action": "complete", "slug": "demo-task",
                        "title": "Demo", "summary": "did a thing",
                        "evidence": ["s1"], "applied": True,
                        "round_actor": "homelab-expert-l1"}) + "\n",
            encoding="utf-8")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        from src.ui.pages import render_work_tasks_page
        html = render_work_tasks_page(self._cfg(), None, {"view": "assessments"})
        assert 'id="assess-body"' in html
        assert "did a thing" in html
        assert "homelab-expert-l1" in html
        assert 'class="tab-btn active"' in html
        # the task link must widen the state filter: the tasks view defaults to
        # in_progress, so a slug sitting in new/ would otherwise match nothing.
        assert '/work/tasks?q=demo-task&amp;states=all' in html
        # the tasks-tab init blob reads view.filter, which the ledger view does
        # not have — it must not be emitted here (undefined -> tojson raises).
        assert 'id="tasks-init"' not in html

    def test_tasks_tab_hides_the_conflicts_section(self, tmp_path, monkeypatch):
        root, _ = _tree(tmp_path, state="completed", slug="demo-task",
                        plan="# T\n\n**Status:** in_progress\n")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        from src.ui.pages import render_work_tasks_page
        html = render_work_tasks_page(self._cfg(), None, {})
        assert wt.tasks_view()["conflicts"]          # the API still reports it
        assert "Status conflicts" not in html        # the page no longer shows it
        assert 'id="tasks-init"' in html

    def test_unknown_tab_falls_back_to_tasks(self, tmp_path, monkeypatch):
        root, _ = _tree(tmp_path, state="in_progress", slug="demo-task", plan="# T\n")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        from src.ui.pages import render_work_tasks_page
        html = render_work_tasks_page(self._cfg(), None, {"view": "../../etc/passwd"})
        assert 'id="tasks-init"' in html and 'id="assess-body"' not in html


class TestMarkdown:
    def test_renders_headings_bold_code_lists(self):
        html = wt._md_to_html("# T\n\n**bold** and `code`\n\n- a\n- b\n")
        assert "<h1>T</h1>" in html
        assert "<strong>bold</strong>" in html
        assert "<code>code</code>" in html
        assert "<li>a</li>" in html

    def test_escapes_raw_html(self):
        html = wt._md_to_html("hello <script>alert(1)</script>\n")
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_link_is_not_auto_linked(self):
        html = wt._md_to_html("see https://example.com\n")
        assert "<a href" not in html

    def test_payload_carries_rendered_html(self, tmp_path, monkeypatch):
        root, _ = _tree(tmp_path, plan="# T\n\nBody line one.\n", results="# R\n\nDone.\n")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(root))
        (m,) = wt.task_moments()
        assert m["payload"]["plan_html"] == "<h1>T</h1>\n<p>Body line one.</p>\n"
        assert m["payload"]["results_html"] == "<h1>R</h1>\n<p>Done.</p>\n"