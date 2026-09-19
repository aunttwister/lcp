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

    def test_sections_and_items_tables_skipped(self, todo_tree):
        v = wt._todo_view()
        titles = [s["title"] for s in v["sections"]]
        assert titles == ["In Progress (2)", "Completed (89)"]
        inprog = v["sections"][0]
        assert inprog["expanded"] is True
        assert v["sections"][1]["expanded"] is False
        # h3 items + table data rows (header + separator dropped)
        assert [e["text"] for e in inprog["entries"]] == [
            "mimo-style-run-dashboard (CREATED 2026-09-18 — aunttwister)",
            "state · in_progress",
            "priority · high",
            "apple-ai-compute-thesis (created 2026-09-13)",
        ]
        assert inprog["entries"][1]["table_row"] is True
        # no row carries the pipe char itself
        assert all("|" not in i["text"] for s in v["sections"] for i in s["entries"])

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