"""The board-decisions ledger pages through the shared table module.

Two properties are load-bearing here and both are pinned below:

1. The ledger pages and sorts like every other log (newest event first).
2. The **funnel is computed over the whole ledger, never over one page** — a
   funnel that changed as you paged would be a lie, and the invariant
   ``sum(by_label) == considered`` would stop meaning anything.
"""

import os
import sqlite3

import pytest

from src.api import work


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """A minimal decisions DB: the `decisions` table + the `latest_verdict` view."""
    path = tmp_path / "decisions.db"
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE decisions (id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT);
        CREATE TABLE verdicts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT, run TEXT, event_t TEXT, event_kind TEXT,
            asked_at TEXT, engine TEXT, model TEXT, question_hash TEXT,
            label TEXT, criteria TEXT, confidence REAL, latency_ms INTEGER,
            raw TEXT);
        CREATE VIEW latest_verdict AS
            SELECT id, event_id, run, event_t, event_kind, asked_at, engine,
                   model, question_hash, label, criteria, confidence,
                   latency_ms, raw
              FROM verdicts;
    """)
    for i in range(12):
        con.execute(
            "INSERT INTO verdicts (event_id, run, event_t, event_kind, asked_at, "
            "engine, model, label, confidence) VALUES (?,?,?,?,?,?,?,?,?)",
            ("e%d" % i, "run%d" % i, "2026-09-%02dT00:00:00+00:00" % (i + 1),
             "reset", "2026-09-%02dT01:00:00+00:00" % (i + 1),
             "engine%d" % (i % 2), "m", "publish" if i % 3 == 0 else "suppress",
             0.9))
    con.execute("INSERT INTO decisions (payload) VALUES ('{}')")
    con.commit()
    con.close()
    monkeypatch.setenv("LCP_WORK_DECISIONS_DB", str(path))
    return path


class TestDecisionsLedgerPaging:
    def test_no_params_returns_the_whole_ledger(self, ledger):
        # Back-compat: the JSON API without paging params is unchanged.
        v = work.decisions_view()
        assert v["available"] is True
        assert v["ledger"]["count"] == 12
        assert len(v["ledger"]["rows"]) == 12

    def test_params_page_the_ledger(self, ledger):
        v = work.decisions_view(params={"per": "5", "page": "2"})
        assert len(v["ledger"]["rows"]) == 5
        assert v["filter"]["pages"] == 3
        assert (v["filter"]["first"], v["filter"]["last"]) == (6, 10)

    def test_ledger_total_is_the_ledger_size_not_the_page_size(self, ledger):
        # The server-rendered heading prints ledger.total; before this the paged
        # payload reported the page size as the ledger size, so the heading
        # contradicted the table under it.
        v = work.decisions_view(params={"per": "5", "page": "2"})
        assert v["ledger"]["count"] == 5        # rows in this payload
        assert v["ledger"]["total"] == 12       # size of the ledger they came from

    def test_pages_do_not_overlap(self, ledger):
        seen = []
        for page in ("1", "2", "3"):
            v = work.decisions_view(params={"per": "5", "page": page})
            seen += [m["id"] for m in v["ledger"]["rows"]]
        assert len(seen) == 12 and len(set(seen)) == 12

    def test_default_sort_is_newest_event_first(self, ledger):
        v = work.decisions_view(params={"per": "3"})
        assert v["filter"]["sort"] == "newest"
        assert [m["t"] for m in v["ledger"]["rows"]] == [
            "2026-09-12T00:00:00+00:00", "2026-09-11T00:00:00+00:00",
            "2026-09-10T00:00:00+00:00"]

    def test_sort_oldest_flips_it(self, ledger):
        v = work.decisions_view(params={"per": "3", "sort": "oldest"})
        assert [m["t"] for m in v["ledger"]["rows"]] == [
            "2026-09-01T00:00:00+00:00", "2026-09-02T00:00:00+00:00",
            "2026-09-03T00:00:00+00:00"]

    def test_unknown_sort_key_falls_back_to_default(self, ledger):
        v = work.decisions_view(params={"per": "2", "sort": "id; DROP TABLE verdicts"})
        assert v["filter"]["sort"] == "newest"
        assert len(work.decision_moments(str(ledger))) == 12

    def test_funnel_covers_the_whole_ledger_not_the_page(self, ledger):
        page = work.decisions_view(params={"per": "3", "page": "1"})
        assert len(page["ledger"]["rows"]) == 3
        assert page["funnel"]["considered"] == 12
        assert sum(page["funnel"]["by_label"].values()) == 12

    def test_funnel_invariant_holds_on_every_page(self, ledger):
        for page in ("1", "2", "3"):
            v = work.decisions_view(params={"per": "5", "page": page})
            assert sum(v["funnel"]["by_label"].values()) == v["funnel"]["considered"]

    def test_page_clamps_past_the_end(self, ledger):
        v = work.decisions_view(params={"per": "5", "page": "99"})
        assert v["filter"]["page"] == 3
        assert len(v["ledger"]["rows"]) == 2

    def test_unavailable_ledger_shape(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LCP_WORK_DECISIONS_DB", str(tmp_path / "nope.db"))
        v = work.decisions_view(params={"per": "5"})
        assert v["available"] is False
        assert v["ledger"] is None
        assert v["filter"] is None


class TestDecisionMomentsWindow:
    def test_default_call_is_the_whole_ledger(self, ledger):
        assert len(work.decision_moments(str(ledger))) == 12

    def test_order_and_window_are_honoured(self, ledger):
        newest = work.DECISION_SORTS[0][2]
        rows = work.decision_moments(str(ledger), order_sql=newest, per=4, offset=4)
        assert [m["t"] for m in rows] == [
            "2026-09-08T00:00:00+00:00", "2026-09-07T00:00:00+00:00",
            "2026-09-06T00:00:00+00:00", "2026-09-05T00:00:00+00:00"]

    def test_foreign_order_fragment_is_refused(self, ledger):
        # The SQL builder re-checks the allow-list itself, not just the caller.
        rows = work.decision_moments(str(ledger), order_sql="1; DROP TABLE verdicts--")
        assert [m["t"] for m in rows][0] == "2026-09-12T00:00:00+00:00"
        assert len(work.decision_moments(str(ledger))) == 12
