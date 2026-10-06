"""Conversations — the page that merges the live set with LCP's own history.

The page exists because two different stores answer "what is LCP working on":
the in-process registry (a running call has no `requests` row yet) and LCP's own
conversation store. Three things here can rot without raising:

1. **the history silently vanishing** — a failing store query must degrade the
   history half, never blank a page whose live half is still useful;
2. **a transcript presented as complete when it was clipped** — the message fetch
   applies a limit, and the view has to say so;
3. **the retired realm coming back** — the old Logs tab still routes here.
"""

import json
from unittest.mock import MagicMock

import pytest

from src.api import conv_store
from src.api import inflight as fl


@pytest.fixture(autouse=True)
def _clean_registry():
    fl.clear()
    yield
    fl.clear()


@pytest.fixture
def cfg():
    c = MagicMock()
    c.providers = {}
    c.profiles = {}
    return c


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LCP_COSTS_DB", str(tmp_path / "costs.db"))
    conv_store.ensure_store()
    conv_store.capture_request(
        "sess-a", {"messages": [{"role": "user", "content": "check the sparks"}]},
        reply="they are fine", profile="l2", model="m1",
        ts="2026-10-06T10:00:00+00:00")
    conv_store.upsert_conversation("sess-a", name="Spark thermals")
    conv_store.recount_conversation("sess-a")
    return tmp_path / "costs.db"


class TestPayload:
    def test_carries_both_halves(self, cfg, store):
        from src.ui.pages import conversations_payload
        p = conversations_payload(cfg, None, {})
        assert p["live"]["count"] == 0
        assert p["history"]["total"] == 1
        assert p["history"]["conversations"][0]["id"] == "sess-a"

    def test_live_half_reflects_the_registry(self, cfg, store):
        from src.ui.pages import conversations_payload
        fl.begin("l2", "m1", "local-sparks", prompt_tokens=10)
        p = conversations_payload(cfg, None, {})
        assert p["live"]["count"] == 1
        assert p["live"]["items"][0]["profile"] == "l2"

    def test_a_broken_store_degrades_history_not_the_page(self, cfg, monkeypatch):
        from src.ui.pages import conversations_payload
        from src.api import conv_store as cs

        def boom(*a, **k):
            raise RuntimeError("db is gone")
        monkeypatch.setattr(cs, "list_conversations", boom)

        p = conversations_payload(cfg, None, {})
        assert p["history"]["available"] is False
        assert p["history"]["error"] == "db is gone"
        assert p["history"]["filter"]["per"] == "20"
        assert p["live"]["count"] == 0      # the live half still rendered


class TestPage:
    def test_renders_live_section_and_history_rows(self, cfg, store):
        from src.ui.pages import render_conversations_page
        fl.begin("l2", "glm-5.3-flash", "local-sparks", prompt_tokens=42, tools=3)
        html = render_conversations_page(cfg, None, {})
        assert "<h1>Conversations</h1>" in html
        assert "In flight" in html
        assert "Spark thermals" in html          # the history row's name
        assert 'data-cid="sess-a"' in html
        # The live row itself: this is where `payload.live.items` would resolve to
        # the dict METHOD in Jinja and kill the render.
        assert "local-sparks" in html
        assert "untagged" in html                # no x-opencode-session on this one

    def test_tool_output_is_collapsed_by_default(self, cfg, store):
        """The transcript's tool bodies live inside <details>, not in the page."""
        from src.ui.pages import render_conversations_page
        html = render_conversations_page(cfg, None, {})
        assert "conv-tool-body" in html          # the collapse style is shipped
        assert "they are fine" not in html       # bodies are fetched on demand

    def test_page_survives_a_broken_store(self, cfg, monkeypatch):
        from src.ui.pages import render_conversations_page
        from src.api import conv_store as cs
        monkeypatch.setattr(cs, "list_conversations",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")))
        html = render_conversations_page(cfg, None, {})
        assert "History unavailable" in html
        assert "In flight" in html


class TestMessagesView:
    def test_reports_a_clipped_transcript_as_truncated(self, store):
        conv_store.append_messages("sess-a", [
            {"role": "user", "content": "x%d" % i} for i in range(20)
        ], origin="capture")
        view = conv_store.messages_view("sess-a", limit=5)
        assert len(view["messages"]) == 5
        assert view["truncated"] is True
        assert view["total"] > 5

    def test_a_complete_transcript_is_not_flagged(self, store):
        view = conv_store.messages_view("sess-a", limit=500)
        assert view["truncated"] is False
        assert view["total"] == len(view["messages"])


class TestRetiredRealm:
    def test_the_logs_tab_no_longer_offers_conversations(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "src", "ui",
                            "templates", "jinja", "sections", "sec_logs.html")
        body = open(path, encoding="utf-8").read()
        assert "tab_link('conversations'" not in body
        assert "tab_link('requests'" in body

    def test_the_logs_view_default_is_not_the_retired_realm(self):
        from src.ui.pages import _logs_view
        assert _logs_view({})["tab"] == "requests"

    def test_the_retired_realm_says_where_it_went(self):
        from src.ui.pages import _logs_view
        view = _logs_view({"view": "conversations"})
        assert "Conversations page" in view["tab_error"]


class TestRoutes:
    def test_the_new_routes_are_registered(self):
        from src.server.handler import LCPHandler
        rt = LCPHandler._route_table()
        names = [str(r[0] if isinstance(r, tuple) else getattr(r, "name", r))
                 for r in rt.rules]
        for expected in ("page.conversations", "api.conversations",
                         "api.conversation.messages", "api.conversation.audit"):
            assert any(expected in n for n in names), (expected, names)

    def test_the_retired_page_route_still_resolves(self):
        """A bookmark to /in-progress must still land somewhere real."""
        from src.server.handler import LCPHandler
        rt = LCPHandler._route_table()
        names = [str(r[0] if isinstance(r, tuple) else getattr(r, "name", r))
                 for r in rt.rules]
        assert any("page.in_progress" in n for n in names), names
