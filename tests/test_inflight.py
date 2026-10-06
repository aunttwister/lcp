"""In-progress — the registry, the payload, and the guards that keep them honest.

``record_cost()`` writes the ``requests`` row (and its ``latency_ms``) only after a
reply has left, so a call that is still running appears NOWHERE in the database.
The In-progress tab reads the in-process registry (``src/api/inflight.py``)
instead. Four things here can rot without raising, and each has a guard below:

1. **a ghost** — an entry registered and never cleared, so the page shows a
   request that finished or died hours ago;
2. **the aid failing the request it observes** — ``begin``/``end`` raising into
   the proxy path is strictly worse than having no registry at all;
3. **a silent task join** — the conversation lookup returning nothing turns
   tagged traffic into "untagged" with no error anywhere;
4. **the surface disappearing** — the nav entry or the route going missing.
"""

import os
import re
import sqlite3
import threading

import pytest

from src.api import inflight as fl


@pytest.fixture(autouse=True)
def _clean_registry():
    """The registry is process-global; every test starts and ends empty."""
    fl.clear()
    yield
    fl.clear()


@pytest.fixture
def conv_db(tmp_path, monkeypatch):
    """A conversations table with one known row, wired to the registry's lookup."""
    path = tmp_path / "costs.db"
    monkeypatch.setenv("LCP_COSTS_DB", str(path))
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE conversations (conversation_id TEXT PRIMARY KEY, name TEXT, "
        "summary TEXT, profile TEXT, model TEXT)"
    )
    con.execute(
        "INSERT INTO conversations (conversation_id, name, profile, model) "
        "VALUES ('abc123', 'Box is rock solid from the LAN', 'l2', 'deepseek-flash')"
    )
    con.commit()
    con.close()
    return str(path)


# ── the registry ─────────────────────────────────────────────────────────────

def test_begin_then_end_round_trip():
    tok = fl.begin("l1", "glm-5.3-flash-exl3", "local-sparks", prompt_tokens=3877)
    assert fl.count() == 1
    row = fl.snapshot()[0]
    assert row["profile"] == "l1"
    assert row["model"] == "glm-5.3-flash-exl3"
    assert row["provider"] == "local-sparks"
    assert row["prompt_tokens"] == 3877
    assert row["elapsed_s"] >= 0
    fl.end(tok)
    assert fl.count() == 0, "end() left the entry registered — that is the ghost"


def test_end_tolerates_none_and_unknown_tokens():
    """Cleanup runs in a `finally`, including on paths that never registered."""
    fl.end(None)
    fl.end("not-a-real-token")
    fl.end("")
    assert fl.count() == 0


def test_each_begin_gets_a_distinct_token():
    toks = {fl.begin("l1", "m", "p") for _ in range(50)}
    assert len(toks) == 50, "tokens collide, so one end() clears the wrong request"


def test_snapshot_is_ordered_longest_running_first():
    """Longest-running first — NOT insertion order.

    The older entry is inserted SECOND on purpose. A dict keeps insertion order,
    so an oldest-first implementation would return [fresh, stale] and a naive
    assertion comparing against insertion order would pass either way; only a
    real sort by elapsed time puts `stale` first.
    """
    fresh = fl.begin("l1", "m", "p")
    stale = fl.begin("l2", "m", "p")
    with fl._LOCK:
        fl._ENTRIES[stale]["started_at"] -= 30
    rows = fl.snapshot()
    assert [r["id"] for r in rows] == [stale, fresh], rows
    assert rows[0]["elapsed_s"] > rows[1]["elapsed_s"]


def test_sweep_drops_only_entries_past_the_age_limit():
    """A thread that dies between begin() and end() must not leak forever."""
    fl.begin("l1", "m", "p")
    aged = fl.begin("l2", "m", "p")
    with fl._LOCK:
        fl._ENTRIES[aged]["started_at"] -= fl.MAX_AGE_S + 60
    assert fl.sweep() == 1, "sweep did not drop the stale entry"
    assert fl.count() == 1, "sweep dropped a live entry too"


def test_sweep_keeps_an_entry_just_inside_the_limit():
    """The boundary matters: the limit must exceed the longest real call."""
    tok = fl.begin("l1", "m", "p")
    with fl._LOCK:
        fl._ENTRIES[tok]["started_at"] -= fl.MAX_AGE_S - 5
    assert fl.sweep() == 0, "sweep is cutting short a request that is still running"
    assert fl.count() == 1


def test_concurrent_begin_and_end_leave_the_registry_empty():
    """The server is a ThreadingHTTPServer, so the lock is the whole guard."""
    errors = []

    def worker():
        try:
            for _ in range(200):
                fl.end(fl.begin("l1", "m", "p"))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert fl.count() == 0, "%d entries leaked under concurrency" % fl.count()


def test_begin_never_raises_on_junk_input():
    """A registry that can 500 the request it observes is worse than none."""
    tok = fl.begin(None, None, None, prompt_tokens=None, tools=None)
    assert fl.count() == 1
    row = fl.snapshot()[0]
    assert row["profile"] == "unknown"
    assert row["model"] == "unknown"
    assert row["provider"] == "unknown"
    assert row["prompt_tokens"] == 0
    assert row["tools"] == 0
    fl.end(tok)


# ── the payload ──────────────────────────────────────────────────────────────

def test_payload_joins_the_conversation_task_name(conv_db):
    """The task name is what makes the view answer "what", not just "how much"."""
    from src.ui.pages import in_progress_payload
    fl.begin("l1", "glm-5.3-flash-exl3", "local-sparks", conversation_id="abc123")
    p = in_progress_payload(None)
    assert p["count"] == 1
    assert p["items"][0]["task"] == "Box is rock solid from the LAN"


def test_payload_task_is_none_when_untagged(conv_db):
    """No conversation id means no task — the page must say so, not invent one."""
    from src.ui.pages import in_progress_payload
    fl.begin("l1", "glm-5.3-flash-exl3", "local-sparks")
    assert in_progress_payload(None)["items"][0]["task"] is None


def test_payload_task_is_none_for_an_unknown_conversation(conv_db):
    """A stale id must not raise, and must not borrow another conversation's name."""
    from src.ui.pages import in_progress_payload
    fl.begin("l2", "deepseek-flash", "deepseek", conversation_id="no-such-id")
    assert in_progress_payload(None)["items"][0]["task"] is None


def test_payload_survives_an_unreadable_conversations_table(tmp_path, monkeypatch):
    """A missing DB must cost the task labels, not the page."""
    from src.ui.pages import in_progress_payload
    monkeypatch.setenv("LCP_COSTS_DB", str(tmp_path / "absent.db"))
    fl.begin("l1", "m", "p", conversation_id="abc123")
    p = in_progress_payload(None)
    assert p["count"] == 1, "an unreadable conversations table killed the payload"
    assert p["items"][0]["task"] is None


def test_payload_reports_profiles_and_providers_distinctly(conv_db):
    from src.ui.pages import in_progress_payload
    fl.begin("l1", "glm-5.3-flash-exl3", "local-sparks")
    fl.begin("l2", "deepseek-flash", "deepseek")
    p = in_progress_payload(None)
    assert p["profiles"] == ["l1", "l2"]
    assert p["providers"] == ["deepseek", "local-sparks"]
    assert p["items"][0]["task"] is None


# ── the page and its surfaces ────────────────────────────────────────────────

def test_page_renders_the_table_and_the_poll_url(conv_db):
    from src.ui.pages import render_in_progress_page
    fl.begin("l1", "glm-5.3-flash-exl3", "local-sparks", conversation_id="abc123",
             prompt_tokens=3877, tools=2)
    html = render_in_progress_page(None, None, None)
    assert "/api/in-progress" in html, "the refresh loop has no endpoint"
    assert "glm-5.3-flash-exl3" in html
    assert "local-sparks" in html
    assert "Box is rock solid from the LAN" in html, "the task name is not rendered"


def test_page_renders_an_empty_state_without_error():
    from src.ui.pages import render_in_progress_page
    html = render_in_progress_page(None, None, None)
    assert "Nothing in flight" in html


def test_page_does_not_dot_access_the_items_key():
    """`payload.items` is the dict METHOD in Jinja, not the key.

    Dotted access renders "builtin_function_or_method object has no element 0" at
    request time — a 500 on a page that looked fine in review. The template must
    use bracket access, and this pins that so a tidy-up cannot undo it.
    """
    path = os.path.join(os.path.dirname(__file__), "..", "src", "ui", "templates",
                        "jinja", "pages", "in_progress.html")
    src = open(path, encoding="utf-8").read()
    # Strip Jinja comments first: the header explains this trap in prose, and the
    # prose necessarily names the bad form. The guard is about the markup.
    code = re.sub(r"\{#.*?#\}", "", src, flags=re.S)
    assert 'payload["items"]' in code, "bracket access to items is missing"
    # Only the Jinja form is a bug. The page's JS reads `p.items` off a
    # JSON-parsed object, where the dot is correct and must stay.
    assert "payload.items" not in code, (
        "`payload.items` is the dict METHOD in Jinja, not the key")


def test_navbar_links_the_in_progress_page():
    path = os.path.join(os.path.dirname(__file__), "..", "src", "ui", "templates",
                        "jinja", "_sidebar.html")
    nav = open(path, encoding="utf-8").read()
    assert 'href="/in-progress"' in nav, "the navbar entry is gone"
    assert "active_page == 'in_progress'" in nav, "the nav entry never highlights"


def test_routes_are_registered():
    from src.server.handler import LCPHandler
    rt = LCPHandler._route_table()
    names = [str(r[0] if isinstance(r, tuple) else getattr(r, "name", r)) for r in rt.rules]
    assert any("page.in_progress" in n for n in names), names
    assert any("api.in_progress" in n for n in names), names


def test_the_chat_handler_clears_the_registry_on_the_success_path():
    """The registration must be undone by the `finally`, not left to chance.

    Asserted on the source rather than by driving a real request: driving one
    would need a live provider, and the property under test is structural — that
    cleanup sits in a `finally` covering every exit path, not at the end of the
    happy path.
    """
    path = os.path.join(os.path.dirname(__file__), "..", "src", "server", "handler.py")
    src = open(path, encoding="utf-8").read()
    assert "in_flight.begin(" in src, "nothing registers an in-flight request"
    assert "in_flight.end(_inflight_token)" in src, "nothing deregisters it"
    begin_at = src.index("in_flight.begin(")
    end_at = src.index("in_flight.end(_inflight_token)")
    finally_at = src.rindex("\n        finally:", 0, end_at)
    assert finally_at > begin_at, "cleanup is not in a finally block"
    # and the token is initialised before the try, or `finally` can NameError
    init_at = src.index("_inflight_token = None")
    assert init_at < begin_at, "the token is not initialised before use"
