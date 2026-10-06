"""Conversations history table + the audit view over LCP's own store."""

import pytest

from src.api import conv_store


def _seed(store):
    """Three conversations with distinct shape: sizes, tools, profiles."""
    conv_store.capture_request(
        "c-big", {"messages": [{"role": "user", "content": "refactor the auth layer"}]},
        reply="on it", profile="l2", model="m1", ts="2026-10-01T10:00:00+00:00")
    conv_store.append_messages("c-big", [
        {"role": "tool", "tool_name": "read_file", "content": "x" * 10, "ts": "2026-10-01T10:00:01+00:00"},
        {"role": "tool", "tool_name": "read_file", "content": "y", "ts": "2026-10-01T10:00:02+00:00"},
        {"role": "tool", "tool_name": "patch", "content": "ok", "ts": "2026-10-01T10:00:03+00:00"},
        # A compacted-away turn: counted in the transcript, not in the live window.
        {"role": "user", "content": "old context", "active": 0, "ts": "2026-10-01T10:00:04+00:00"},
    ], origin="capture")
    conv_store.upsert_conversation("c-big", name="Refactor the auth layer")
    conv_store.recount_conversation("c-big")

    conv_store.capture_request(
        "c-small", {"messages": [{"role": "user", "content": "what is 2+2"}]},
        reply="4", profile="l1", model="m2", ts="2026-10-05T10:00:00+00:00")

    conv_store.capture_request(
        "c-mid", {"messages": [{"role": "user", "content": "summarise the logs"}]},
        reply="done", profile="l2", model="m1", ts="2026-10-03T10:00:00+00:00")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LCP_COSTS_DB", str(tmp_path / "costs.db"))
    conv_store.ensure_store()
    _seed(store)
    return store


class TestListConversations:
    def test_returns_every_conversation_newest_first(self, store):
        view = conv_store.list_conversations({"per": "20"})
        assert view["available"] is True
        assert view["total"] == 3
        assert [c["id"] for c in view["conversations"]] == ["c-small", "c-mid", "c-big"]

    def test_oldest_sort_is_the_reverse(self, store):
        view = conv_store.list_conversations({"sort": "oldest"})
        assert [c["id"] for c in view["conversations"]] == ["c-big", "c-mid", "c-small"]

    def test_messages_sort_puts_the_biggest_first(self, store):
        view = conv_store.list_conversations({"sort": "messages"})
        assert view["conversations"][0]["id"] == "c-big"

    def test_pagination_is_20_by_default_and_windows(self, store):
        view = conv_store.list_conversations({})
        assert view["filter"]["per"] == "20"
        assert view["filter"]["page"] == 1
        page2 = conv_store.list_conversations({"per": "2", "page": "2"})
        assert len(page2["conversations"]) == 1
        assert page2["filter"]["pages"] == 2

    def test_profile_filter_narrows_and_is_echoed(self, store):
        view = conv_store.list_conversations({"profile": "l2"})
        assert view["total"] == 2
        assert {c["profile"] for c in view["conversations"]} == {"l2"}
        assert view["filter"]["profile"] == "l2"
        assert sorted(view["filter"]["profiles"]) == ["l1", "l2"]

    def test_text_search_matches_a_substring_not_just_the_whole_value(self, store):
        # 'auth' appears only inside the generated name, and 'ig' only inside an
        # id — a LIKE/equality mix-up shows up on both.
        assert conv_store.list_conversations({"q": "auth"})["total"] == 1
        assert conv_store.list_conversations({"q": "ig"})["total"] == 1
        assert conv_store.list_conversations({"q": "nothing-matches-this"})["total"] == 0

    def test_counts_come_from_the_stored_transcript(self, store):
        view = conv_store.list_conversations({"q": "c-big"})
        c = view["conversations"][0]
        assert c["messages"] == 6           # user + reply + 3 tools + 1 compacted
        assert c["active_messages"] == 5    # the compacted turn is out of the window
        assert c["tool_calls"] == 3
        assert c["source"] == "live"


class TestConversationAudit:
    def test_reports_the_tools_that_ran_most_first(self, store):
        audit = conv_store.conversation_audit("c-big")
        assert audit["available"] is True
        assert audit["tools"][0] == {"tool": "read_file", "count": 2}
        assert {"tool": "patch", "count": 1} in audit["tools"]

    def test_tool_calls_total_is_the_sum_of_the_breakdown(self, store):
        audit = conv_store.conversation_audit("c-big")
        assert audit["tool_calls"] == sum(t["count"] for t in audit["tools"]) == 3

    def test_reports_turn_counts_and_the_opening_ask(self, store):
        audit = conv_store.conversation_audit("c-big")
        assert audit["user_turns"] == 2      # the original ask + the compacted turn
        assert audit["assistant_turns"] == 1
        assert audit["first_ask"] == "refactor the auth layer"

    def test_carries_profile_model_and_window(self, store):
        audit = conv_store.conversation_audit("c-big")
        assert audit["profile"] == "l2"
        assert audit["model"] == "m1"
        assert audit["started_at"] == "2026-10-01T10:00:00+00:00"
        assert audit["last_at"] == "2026-10-01T10:00:04+00:00"

    def test_unknown_conversation_reports_unavailable_not_an_error(self, store):
        assert conv_store.conversation_audit("nope") == {"available": False, "id": "nope"}

    def test_audit_does_not_dump_tool_bodies(self, store):
        """The audit view describes the work; bodies are fetched on demand."""
        audit = conv_store.conversation_audit("c-big")
        blob = repr(audit)
        assert "x" * 10 not in blob
