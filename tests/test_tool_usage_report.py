"""Per-tool usage over LCP's own conversation store.

The feature's contract: given a window, report which tools ran, how often, and how
many bytes each returned into the conversation. No external service, no second
database, no pre-computed rollup that can drift from the rows.

Two shapes matter and both are exercised here because production contains both:
- a direct tool call (name on the assistant turn, result named from the map), and
- a deferred-catalog wrapper whose real names live in ``arguments.calls[].name``.
"""

import json
import sqlite3

import pytest

from src.api import conv_store


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("LCP_COSTS_DB", str(tmp_path / "costs.db"))
    conv_store.ensure_store()
    return tmp_path / "costs.db"


def _seed(cid, profile, calls, results, ts="2026-11-20T10:00:00+00:00"):
    """Write a conversation directly, so the read path is tested against rows
    shaped exactly like the ones already in prod (including the wrapper shape)."""
    con = sqlite3.connect(str(conv_store.db_path()))
    con.execute(
        "INSERT OR REPLACE INTO conversations (conversation_id, profile, first_seen, last_seen) "
        "VALUES (?,?,?,?)", (cid, profile, ts, ts))
    seq = 0
    for name, arguments in calls:
        con.execute(
            "INSERT INTO conv_messages (conversation_id, seq, role, content, content_len, "
            "tool_calls, ts) VALUES (?,?,?,?,?,?,?)",
            (cid, seq, "assistant", "", 0, json.dumps([{
                "id": f"c{seq}", "type": "function",
                "function": {"name": name, "arguments": arguments}}]), ts))
        seq += 1
    for tool_name, length in results:
        con.execute(
            "INSERT INTO conv_messages (conversation_id, seq, role, content, content_len, "
            "tool_name, ts) VALUES (?,?,?,?,?,?,?)",
            (cid, seq, "tool", "x" * 10, length, tool_name, ts))
        seq += 1
    con.commit()
    con.close()


def test_counts_direct_calls_and_sums_result_bytes(db):
    _seed("c1", "l2", [("terminal", '{"command":"ls"}')], [("terminal", 1200)])
    _seed("c2", "l2", [("terminal", '{"command":"pwd"}')], [("terminal", 800)])
    out = conv_store.tool_usage({})
    row = next(r for r in out["tools"] if r["tool"] == "terminal")
    assert row["calls"] == 2
    assert row["result_bytes"] == 2000
    assert row["avg_bytes"] == 1000
    assert row["max_bytes"] == 1200
    assert row["conversations"] == 2


def test_unwraps_a_deferred_catalog_call(db):
    wrapped = json.dumps({"calls": [{"name": "mcp__linear__list_issues", "arguments": {}}]})
    _seed("c3", "l2", [("tool_call", wrapped)], [("mcp__linear__list_issues", 49650)])
    out = conv_store.tool_usage({})
    row = next(r for r in out["tools"] if r["tool"] == "mcp__linear__list_issues")
    assert row["calls"] == 1
    assert row["result_bytes"] == 49650
    # the wrapper name is not a tool and must not appear as one
    assert not any(r["tool"] == "tool_call" for r in out["tools"])


def test_multi_call_wrapper_counts_each_tool(db):
    wrapped = json.dumps({"calls": [{"name": "mcp__a__one"}, {"name": "mcp__a__two"}]})
    _seed("c4", "l2", [("tool_call", wrapped)], [("mcp__a__one,mcp__a__two", 100)])
    out = conv_store.tool_usage({})
    names = {r["tool"] for r in out["tools"]}
    assert {"mcp__a__one", "mcp__a__two"} <= names


def test_reports_the_unreadable_residue_instead_of_hiding_it(db):
    _seed("c5", "l2", [("tool_call", "{not json")], [("tool_call", 10)])
    out = conv_store.tool_usage({})
    assert out["wrapper_named_result_rows"] == 1


def test_filters_by_tool_prefix(db):
    _seed("c6", "l2",
          [("terminal", "{}"), ("tool_call", json.dumps({"calls": [{"name": "mcp__linear__get_issue"}]}))],
          [("terminal", 10), ("mcp__linear__get_issue", 20)])
    out = conv_store.tool_usage({"tool": "mcp__*"})
    assert [r["tool"] for r in out["tools"]] == ["mcp__linear__get_issue"]


def test_filters_by_profile(db):
    _seed("c7", "l2", [("terminal", "{}")], [("terminal", 5)])
    _seed("c8", "blog", [("patch", "{}")], [("patch", 7)])
    l2 = conv_store.tool_usage({"profile": "l2"})
    assert [r["tool"] for r in l2["tools"]] == ["terminal"]


def test_window_excludes_out_of_range_rows(db):
    _seed("c9", "l2", [("terminal", "{}")], [("terminal", 5)], ts="2026-01-01T00:00:00+00:00")
    out = conv_store.tool_usage({"start": "2026-06-01", "end": "2026-12-31"})
    assert out["tools"] == []


def test_a_call_without_a_stored_result_is_still_counted(db):
    """A tool that crashed left no result row; the call is still a fact."""
    _seed("c10", "l2", [("terminal", "{}")], [])
    out = conv_store.tool_usage({})
    row = next(r for r in out["tools"] if r["tool"] == "terminal")
    assert row["calls"] == 1
    assert row["result_bytes"] == 0


def test_sorted_by_calls_then_bytes(db):
    _seed("c11", "l2", [("a", "{}"), ("a", "{}"), ("b", "{}")], [("a", 1), ("a", 1), ("b", 99999)])
    out = conv_store.tool_usage({})
    assert [r["tool"] for r in out["tools"]][:2] == ["a", "b"]
