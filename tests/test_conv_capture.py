"""Conversation capture: SSE/text extraction and the request-path store writes.

Covers the two extractors that recover the assistant's words from a response
(the reply is not in the request body) and ``conv_store.capture_request``, which
is the live half of "LCP owns its conversation history".
"""

import json

import pytest

from src.api import conv_store
from src.server.sse_helpers import extract_message_text, extract_sse_text


def _sse(*chunks):
    return ("".join("data: %s\n\n" % json.dumps(c) for c in chunks)
            + "data: [DONE]\n\n").encode("utf-8")


def _delta(content):
    return {"choices": [{"delta": {"content": content}}]}


@pytest.fixture
def store(tmp_path, monkeypatch):
    db = tmp_path / "costs.db"
    monkeypatch.setenv("LCP_COSTS_DB", str(db))
    conv_store.ensure_store()
    return db


class TestExtractSseText:
    def test_joins_delta_fragments(self):
        raw = _sse(_delta("Hel"), _delta("lo "), _delta("world"))
        assert extract_sse_text(raw) == "Hello world"

    def test_ignores_reasoning_content(self):
        raw = _sse({"choices": [{"delta": {"reasoning_content": "thinking…"}}]},
                   _delta("answer"))
        assert extract_sse_text(raw) == "answer"

    def test_ignores_tool_call_argument_deltas(self):
        raw = _sse({"choices": [{"delta": {"tool_calls": [
            {"function": {"arguments": '{"a":1}'}}]}}]}, _delta("done"))
        assert extract_sse_text(raw) == "done"

    def test_stops_at_done_marker(self):
        assert extract_sse_text(b"data: [DONE]\n\n") == ""

    def test_malformed_lines_are_skipped_not_fatal(self):
        raw = (b"data: {not json}\n\n" + b"data: \n\n"
               + b": comment\n\n" + b"data: " + json.dumps(_delta("ok")).encode() + b"\n\n")
        assert extract_sse_text(raw) == "ok"

    def test_non_string_content_ignored(self):
        raw = _sse({"choices": [{"delta": {"content": None}}]}, _delta("x"))
        assert extract_sse_text(raw) == "x"

    def test_empty_and_garbage_yield_empty_string(self):
        assert extract_sse_text(b"") == ""
        assert extract_sse_text(b"\xff\xfe not utf8") == ""


class TestExtractMessageText:
    def test_reads_assistant_content(self):
        assert extract_message_text(
            {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}) == "hi"

    def test_absent_shape_yields_empty(self):
        assert extract_message_text({}) == ""
        assert extract_message_text({"choices": []}) == ""
        assert extract_message_text({"choices": [{"message": None}]}) == ""

    def test_non_string_content_yields_empty(self):
        assert extract_message_text(
            {"choices": [{"message": {"content": [{"type": "text"}]}}]}) == ""


class TestCaptureRequest:
    BODY = {"messages": [
        {"role": "system", "content": "you are a helper"},
        {"role": "user", "content": "what is 2+2"},
    ]}

    def test_stores_prompt_and_reply(self, store):
        n = conv_store.capture_request("sess-1", self.BODY, reply="4",
                                       profile="l2", model="m1")
        assert n == 2  # user + assistant; the system turn is not conversation
        rows = conv_store.read_messages("sess-1")
        assert [r["role"] for r in rows] == ["user", "assistant"]
        assert rows[0]["content"] == "what is 2+2"
        assert rows[1]["content"] == "4"

    def test_system_message_not_stored(self, store):
        conv_store.capture_request("sess-2", self.BODY, reply="x")
        assert all(r["role"] != "system" for r in conv_store.read_messages("sess-2"))

    def test_second_call_appends_only_the_new_turn(self, store):
        conv_store.capture_request("sess-3", self.BODY, reply="4")
        body2 = {"messages": self.BODY["messages"] + [
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "and 3+3"},
        ]}
        assert conv_store.capture_request("sess-3", body2, reply="6") == 2
        assert [r["content"] for r in conv_store.read_messages("sess-3")] == [
            "what is 2+2", "4", "and 3+3", "6"]

    def test_replay_of_same_request_writes_nothing(self, store):
        conv_store.capture_request("sess-4", self.BODY, reply="4")
        assert conv_store.capture_request("sess-4", self.BODY, reply="4") == 0

    def test_reply_is_the_last_turn_so_a_different_reply_is_new(self, store):
        conv_store.capture_request("sess-5", self.BODY, reply="4")
        assert conv_store.capture_request("sess-5", self.BODY, reply="five") == 1

    def test_rollup_records_profile_model_and_counts(self, store):
        conv_store.capture_request("sess-6", self.BODY, reply="4",
                                   profile="l2", model="m1")
        con = conv_store._connect()
        row = con.execute(
            "SELECT profile, model, source, message_count, active_message_count "
            "FROM conversations WHERE conversation_id=?", ("sess-6",)).fetchone()
        con.close()
        assert tuple(row) == ("l2", "m1", "live", 2, 2)

    def test_unidentified_request_is_a_noop(self, store):
        assert conv_store.capture_request(None, self.BODY, reply="4") == 0

    def test_empty_message_list_is_a_noop(self, store):
        assert conv_store.capture_request("sess-7", {"messages": []}) == 0

    def test_a_store_failure_never_raises(self, store, monkeypatch):
        """Capture is bookkeeping — it must not be able to fail a request."""
        def boom(*a, **k):
            raise RuntimeError("disk full")
        monkeypatch.setattr(conv_store, "append_messages", boom)
        assert conv_store.capture_request("sess-8", self.BODY, reply="4") == 0

    def test_capture_without_a_timestamp_still_records_an_activity_window(self, store):
        """The handler passes no ts; a NULL window made live rows unlistable."""
        conv_store.capture_request("sess-9", self.BODY, reply="4", profile="l2")
        con = conv_store._connect()
        first_seen, last_seen, tools = con.execute(
            "SELECT first_seen, last_seen, tool_call_count FROM conversations "
            "WHERE conversation_id=?", ("sess-9",)).fetchone()
        con.close()
        assert first_seen and last_seen          # not NULL
        assert first_seen <= last_seen
        assert tools == 0                        # no tool turns in this body

    def test_tool_turns_counted_in_the_rollup(self, store):
        body = {"messages": [{"role": "user", "content": "go"}]}
        conv_store.capture_request("sess-10", body, reply="ok", profile="l2")
        conv_store.append_messages("sess-10", [
            {"role": "tool", "tool_name": "read_file", "content": "a"},
            {"role": "tool", "tool_name": "read_file", "content": "b"},
        ], origin="capture")
        conv_store.recount_conversation("sess-10")
        con = conv_store._connect()
        n = con.execute("SELECT tool_call_count FROM conversations WHERE "
                        "conversation_id=?", ("sess-10",)).fetchone()[0]
        con.close()
        assert n == 2
