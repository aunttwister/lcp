"""Deferred-catalog wrapper unwrapping: the tool that ran must be the tool recorded.

A deferred catalog (tool_search / tool_describe / a single ``tool_call`` dispatcher)
exposes ONE function; the real call lives in its ``arguments`` under ``calls[].name``.
Before this, every such call was stored under the wrapper name, so per-tool
accounting returned zero for the whole era while tens of thousands of real calls
sat in the same table.

These tests go through the shapes the live capture path actually hands over — a raw
client message (arguments as a JSON string) and an already-serialised batch — because
reading only one of them is what produced the earlier "unnamed rows" defect.
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


def _rows(db, cid):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    out = [dict(r) for r in con.execute(
        "SELECT seq, role, tool_name, tool_call_id FROM conv_messages "
        "WHERE conversation_id=? ORDER BY seq", (cid,))]
    con.close()
    return out


# ── the helper itself ────────────────────────────────────────────────────────

def test_unwrap_returns_none_for_a_direct_call():
    assert conv_store._unwrap_deferred_tool_args('{"command": "ls"}') is None


def test_unwrap_reads_one_wrapped_call():
    args = json.dumps({"calls": [{"name": "mcp__linear__save_issue", "arguments": {}}]})
    assert conv_store._unwrap_deferred_tool_args(args) == "mcp__linear__save_issue"


def test_unwrap_joins_a_multi_call_wrapper_without_duplicates():
    args = json.dumps({"calls": [
        {"name": "mcp__linear__save_issue"},
        {"name": "mcp__linear__save_issue"},
        {"name": "mcp__linear__save_comment"},
    ]})
    assert conv_store._unwrap_deferred_tool_args(args) == (
        "mcp__linear__save_issue,mcp__linear__save_comment")


def test_unwrap_accepts_an_already_parsed_dict():
    assert conv_store._unwrap_deferred_tool_args(
        {"calls": [{"name": "mcp__x__y"}]}) == "mcp__x__y"


def test_unwrap_accepts_double_encoded_calls_string():
    """The live store really contains rows whose ``calls`` is a JSON STRING inside
    the JSON — 282 of them measured. They must resolve, not be skipped."""
    args = json.dumps({"calls": json.dumps([{"name": "mcp__x__z"}])})
    assert conv_store._unwrap_deferred_tool_args(args) == "mcp__x__z"


def test_unwrap_leaves_malformed_arguments_alone():
    assert conv_store._unwrap_deferred_tool_args("{not json") is None
    assert conv_store._unwrap_deferred_tool_args(None) is None
    assert conv_store._unwrap_deferred_tool_args([1, 2, 3]) is None


def test_unwrap_ignores_a_calls_list_with_no_names():
    assert conv_store._unwrap_deferred_tool_args(json.dumps({"calls": [{"x": 1}]})) is None


# ── through the real capture path ────────────────────────────────────────────

def test_capture_records_the_real_tool_name_behind_a_wrapper(db):
    wrapper_call = [{
        "id": "call_w1", "type": "function",
        "function": {
            "name": "tool_call",
            "arguments": json.dumps({"calls": [
                {"name": "mcp__linear__list_issues", "arguments": {"limit": 5}}]}),
        },
    }]
    conv_store.capture_request("cw1", {"messages": [
        {"role": "user", "content": "list issues"},
        {"role": "assistant", "content": "", "tool_calls": wrapper_call},
        {"role": "tool", "tool_call_id": "call_w1", "content": '{"issues": []}'},
    ]})
    rows = _rows(db, "cw1")
    tool_rows = [r for r in rows if r["role"] == "tool"]
    assert len(tool_rows) == 1
    assert tool_rows[0]["tool_name"] == "mcp__linear__list_issues"


def test_capture_still_records_a_direct_call_unchanged(db):
    direct = [{"id": "call_d1", "type": "function",
               "function": {"name": "terminal", "arguments": "{\"command\":\"ls\"}"}}]
    conv_store.capture_request("cd1", {"messages": [
        {"role": "user", "content": "run"},
        {"role": "assistant", "content": "", "tool_calls": direct},
        {"role": "tool", "tool_call_id": "call_d1", "content": "ok"},
    ]})
    tool_rows = [r for r in _rows(db, "cd1") if r["role"] == "tool"]
    assert tool_rows[0]["tool_name"] == "terminal"


def test_capture_keeps_the_wrapper_row_when_arguments_are_unparseable(db):
    """A row must never be dropped because its arguments were malformed."""
    bad = [{"id": "call_b1", "type": "function",
            "function": {"name": "tool_call", "arguments": "{not json"}}]
    conv_store.capture_request("cb1", {"messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": bad},
        {"role": "tool", "tool_call_id": "call_b1", "content": "whatever"},
    ]})
    tool_rows = [r for r in _rows(db, "cb1") if r["role"] == "tool"]
    assert len(tool_rows) == 1
    assert tool_rows[0]["tool_name"] == "tool_call"
