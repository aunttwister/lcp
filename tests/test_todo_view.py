"""todo.md overview tests — timeline, sections, tables skipped, truncation."""

import textwrap

import pytest

from src.api import work_tasks as wt

SAMPLE = textwrap.dedent('''\
    # Pending Tasks — Phase-by-Phase

    > Updated: 2026-09-18 — NEW task **`decision-pipeline-taskboard`** per aunttwister
    > Updated: 2026-09-17 — BUILT AND LIVE — the dashboard is at http://192.168.1.198:8090

    ## 🔴 In Progress (2)

    ### mimo-style-run-dashboard (CREATED 2026-09-18 — aunttwister)
    | col | val |
    |---|---|
    | state | in_progress |
    | priority | high |

    ### apple-ai-compute-thesis (created 2026-09-13)
    - [ ] research phase
    - [x] outline

    ## 🟡 New / Pending (33)

    ### dashboard-ui-components (pending item)

    ## 🆕 Queued (2)

    ### zgx-night-batch-pipeline (queued)

    ## ✅ Completed (89)

    ### zgx-fp8kv-cache — ✅ COMPLETED
    - [x] shipped
''')


@pytest.fixture
def todo_tree(tmp_path, monkeypatch):
    tree = tmp_path / "work" / "tasks"
    tree.mkdir(parents=True)
    (tmp_path / "work" / "todo.md").write_text(SAMPLE, encoding="utf-8")
    monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(tree))
    return tmp_path


class TestTodoView:
    def test_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(tmp_path / "nope"))
        assert wt._todo_view() is None

    def test_summary_stats(self, todo_tree):
        v = wt._todo_view()
        assert v["lines"] == SAMPLE.count("\n") + 1
        assert v["open_boxes"] == 1
        assert v["done_boxes"] == 2
        assert v["updated"] and "2026-09-18" in v["updated"]

    def test_timeline_dates_and_text(self, todo_tree):
        v = wt._todo_view()
        assert len(v["timeline"]) == 2
        assert v["timeline"][0]["date"] == "2026-09-18"
        assert "decision-pipeline-taskboard" in v["timeline"][0]["text"]
        assert v["timeline"][0]["html"].startswith("<p>")  # markdown-rendered

    def test_sections_skip_table_duplicates(self, todo_tree):
        v = wt._todo_view()
        titles = [s["title"] for s in v["sections"]]
        # the state groups duplicated by the All Tasks filter are dropped, and so
        # is Queued — a dispatch buffer, not task state (operator, 2026-09-20).
        # SAMPLE has no other group, so nothing survives.
        assert "In Progress (2)" not in titles
        assert "New / Pending (33)" not in titles
        assert "Completed (89)" not in titles
        assert not any(t.lower().startswith("queued") for t in titles)
        assert titles == []

    def test_non_state_section_still_renders(self, tmp_path, monkeypatch):
        """Skipping state + Queued groups must not swallow a real one."""
        tree = tmp_path / "work" / "tasks"
        tree.mkdir(parents=True)
        (tmp_path / "work" / "todo.md").write_text(textwrap.dedent('''\
            # T
            ## Blocked on hardware (1)

            ### nas-audit-phase7 (blocked)
        '''), encoding="utf-8")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(tree))
        v = wt._todo_view()
        assert [s["title"] for s in v["sections"]] == ["Blocked on hardware (1)"]
        assert [e["text"] for e in v["sections"][0]["entries"]] == \
            ["nas-audit-phase7 (blocked)"]

    def test_truncation_flagged(self, todo_tree, monkeypatch):
        import os
        long = textwrap.dedent('''\
            # T
            > Updated: 2026-09-01 — ''' + ("word " * 300) + '''
        ''')
        p = todo_tree / "work" / "todo.md"
        p.write_text(long, encoding="utf-8")
        v = wt._todo_view()
        assert v["timeline"][0]["truncated"] is True
        assert len(v["timeline"][0]["html"]) < 4000

    def test_timeline_capped(self, todo_tree):
        v = wt._todo_view()
        assert len(v["timeline"]) <= wt._TODO_TIMELINE_CAP

    def test_h1_and_blank_lines_ignored(self, todo_tree):
        v = wt._todo_view()
        assert all(not s["title"].startswith("# Pending") for s in v["sections"])

    def test_emoji_stripped_from_entries(self, tmp_path, monkeypatch):
        tree = tmp_path / "work" / "tasks"
        tree.mkdir(parents=True)
        (tmp_path / "work" / "todo.md").write_text(textwrap.dedent('''\
            # T
            ## Blocked on hardware (2)

            ### serbian-tts-stt — ⏸ PARKED 2026-09-06 → next step
            ### zgx-fp8kv-cache — ✅ COMPLETED
        '''), encoding="utf-8")
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(tree))
        v = wt._todo_view()
        texts = [e["text"] for e in v["sections"][0]["entries"]]
        assert "serbian-tts-stt — PARKED 2026-09-06 → next step" in texts
        assert "zgx-fp8kv-cache — COMPLETED" in texts
        # arrows are prose and must survive
        assert "→" in texts[0]


class TestTasksViewParams:
    def _moments(self, tmp_path):
        root = tmp_path / "tasks"
        for state in ("in_progress", "new", "completed"):
            for slug in ("alpha", "bravo", "charlie"):
                d = root / state / slug
                d.mkdir(parents=True)
                (d / "PLAN.md").write_text("# %s\nplan text for %s" % (slug, slug),
                                            encoding="utf-8")
        import os
        os.environ["LCP_WORK_TASKS_DIR"] = str(root)
        os.environ.pop("LCP_WORK_TODO", None)
        # tag one task via a fake classification index
        layers = root / ".work-layers"
        layers.mkdir(exist_ok=True)
        import json
        (layers / "classifications.json").write_text(json.dumps({
            "by_label": {"infrastructure": 1},
            "labels": {"infrastructure": {"label": "infrastructure", "exemplars": []}},
            "tasks": {"alpha": {"label": "infrastructure", "score": 0.9}},
        }), encoding="utf-8")

    def test_default_only_in_progress(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(tmp_path / "tasks"))
        self._moments(tmp_path)
        v = wt.tasks_view()
        assert v["filter"]["states"] == ["in_progress"]
        assert len(v["tasks"]) == 3  # alpha/bravo/charlie in_progress
        assert v["filter"]["total_filtered"] == 3
        assert v["filter"]["pages"] == 1

    def test_multiselect_states(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(tmp_path / "tasks"))
        self._moments(tmp_path)
        v = wt.tasks_view({"states": "in_progress,new"})
        assert v["filter"]["total_filtered"] == 6

    def test_states_all(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(tmp_path / "tasks"))
        self._moments(tmp_path)
        v = wt.tasks_view({"states": "all"})
        assert v["filter"]["total_filtered"] == 9

    def test_search_matches_plan_text(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(tmp_path / "tasks"))
        self._moments(tmp_path)
        v = wt.tasks_view({"states": "all", "q": "plan text for bravo"})
        # bravo exists in every state -> 3 matches, all named bravo
        assert v["filter"]["total_filtered"] == 3
        assert {m["subject"] for m in v["tasks"]} == {"bravo"}

    def test_tag_filter(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(tmp_path / "tasks"))
        self._moments(tmp_path)
        v = wt.tasks_view({"states": "all", "tag": "infrastructure"})
        # the index labels slug "alpha", which exists in every state
        assert v["filter"]["total_filtered"] == 3
        assert {m["subject"] for m in v["tasks"]} == {"alpha"}

    def test_pagination(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(tmp_path / "tasks"))
        self._moments(tmp_path)
        v = wt.tasks_view({"states": "all", "per": "4", "page": "2"})
        assert v["filter"]["pages"] == 3
        assert v["filter"]["page"] == 2
        assert len(v["tasks"]) == 4

    def test_per_all(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(tmp_path / "tasks"))
        self._moments(tmp_path)
        v = wt.tasks_view({"states": "all", "per": "all"})
        assert v["filter"]["pages"] == 1
        assert len(v["tasks"]) == 9


class TestLiteAndDetail:
    def test_lite_rows_carry_no_plan_payload(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(tmp_path / "tasks"))
        TestTasksViewParams._moments(TestTasksViewParams(), tmp_path)
        v = wt.tasks_view({"states": "in_progress", "lite": "1"})
        row = v["tasks"][0]
        assert "plan_html" not in row["payload"]
        assert "results_html" not in row["payload"]
        assert "state_summary" not in row["payload"]
        assert row["key"] == "in_progress/alpha"

    def test_detail_returns_heavy_payload(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(tmp_path / "tasks"))
        TestTasksViewParams._moments(TestTasksViewParams(), tmp_path)
        d = wt.task_detail("in_progress/alpha")
        assert d["subject"] == "alpha"
        assert d["payload"]["state"] == "in_progress"
        assert d["payload"]["has_plan"] is True
        assert d["payload"]["plan_html"]

    def test_detail_rejects_bad_keys(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LCP_WORK_TASKS_DIR", str(tmp_path / "tasks"))
        TestTasksViewParams._moments(TestTasksViewParams(), tmp_path)
        with pytest.raises(ValueError):
            wt.task_detail("../etc/passwd")
        with pytest.raises(ValueError):
            wt.task_detail("unknown/alpha")
        with pytest.raises(FileNotFoundError):
            wt.task_detail("in_progress/does-not-exist")