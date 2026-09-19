"""work_conversations tests — schema, backfill grouping, stamping, detail."""

import json
import os
import sqlite3

import pytest

from src.api import work_conversations as wc


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "costs.db"
    monkeypatch.setenv("LCP_COSTS_DB", str(path))
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
            profile TEXT NOT NULL DEFAULT 'unknown', model TEXT NOT NULL,
            provider TEXT NOT NULL, prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0, cache_hit_tokens INTEGER DEFAULT 0,
            cache_miss_tokens INTEGER DEFAULT 0, cost REAL DEFAULT 0.0,
            latency_ms INTEGER DEFAULT 0, success INTEGER DEFAULT 1,
            error_type TEXT, tools_blocked TEXT, error_detail TEXT);
        CREATE TABLE routing_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts VARCHAR NOT NULL,
            profile VARCHAR NOT NULL, task VARCHAR NOT NULL, policy VARCHAR NOT NULL,
            action VARCHAR NOT NULL, provider VARCHAR, model VARCHAR, score FLOAT,
            rules_json TEXT, from_provider VARCHAR, from_model VARCHAR, note TEXT,
            path VARCHAR, keyword VARCHAR, intent_text TEXT, semantic_json TEXT,
            min_score FLOAT, sem_available BOOLEAN, conversation_json TEXT);
        CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT);
    """)
    con.commit()
    return con


def _req(con, ts, profile="l2", model="m", provider="p", **kw):
    con.execute(
        "INSERT INTO requests (timestamp, profile, model, provider, success, "
        "prompt_tokens, completion_tokens, cost, latency_ms) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (ts, profile, model, provider, kw.get("success", 1),
         kw.get("pt", 10), kw.get("ct", 5), kw.get("cost", 0.01),
         kw.get("lat", 100)))
    return con.execute("SELECT last_insert_rowid()").fetchone()[0]


def _route(con, ts, profile="l2", task="research", action="keep_default", **kw):
    con.execute(
        "INSERT INTO routing_decisions (ts, profile, task, policy, action, "
        "provider, model, score, conversation_json) VALUES (?,?,?,?,?,?,?,?,?)",
        (ts, profile, task, "policy", action, kw.get("provider", "opencode"),
         kw.get("model", "m"), kw.get("score", 0.8),
         kw.get("conv", '[]')))


class TestBackfill:
    def test_burst_groups_by_profile_model_gap(self, db):
        _req(db, "2026-09-19T10:00:00+00:00", profile="l2", model="a")
        _req(db, "2026-09-19T10:01:00+00:00", profile="l2", model="a")   # same burst
        _req(db, "2026-09-19T10:20:00+00:00", profile="l2", model="a")   # new burst (>5m)
        _req(db, "2026-09-19T10:02:00+00:00", profile="l1", model="a")   # diff profile
        db.commit()
        got = wc.backfill_conversations()
        assert got["requests"] == 4
        rows = db.execute("SELECT profile, model, conversation_id FROM requests "
                          "ORDER BY id").fetchall()
        # three distinct conversations
        ids = {r[2] for r in rows}
        assert len(ids) == 3
        assert all(r[2] for r in rows)
        # same conversation for the first two l2/a rows
        assert rows[0][2] == rows[1][2]

    def test_idempotent(self, db):
        _req(db, "2026-09-19T10:00:00+00:00")
        _req(db, "2026-09-19T10:01:00+00:00")
        db.commit()
        assert wc.backfill_conversations()["requests"] == 2
        assert wc.backfill_conversations()["requests"] == 0  # nothing new

    def test_routing_joins_request_window(self, db):
        rid = _req(db, "2026-09-19T10:00:00+00:00")
        _route(db, "2026-09-19T10:00:30+00:00", conv="[]")
        db.commit()
        got = wc.backfill_conversations()
        assert got["requests"] == 1 and got["routing"] == 1
        r = db.execute("SELECT conversation_id FROM requests WHERE id=?", (rid,)).fetchone()[0]
        rt = db.execute("SELECT conversation_id FROM routing_decisions").fetchone()[0]
        assert r == rt

    def test_orphan_routing_gets_synthetic(self, db):
        _route(db, "2026-09-19T10:00:00+00:00")
        db.commit()
        got = wc.backfill_conversations()
        assert got["routing"] == 1
        cid = db.execute("SELECT conversation_id FROM routing_decisions").fetchone()[0]
        assert cid.startswith("c")


class TestStampNew:
    def test_stamp_request_and_routing_window(self, db):
        rid = _req(db, "2026-09-19T10:00:00+00:00", lat=100)
        _route(db, "2026-09-19T10:00:05+00:00")
        _route(db, "2026-09-19T10:30:00+00:00")  # outside window — untouched
        db.commit()
        wc.stamp_new(request_id=rid, conversation_id="sess-abc",
                     profile="l2", ts="2026-09-19T10:00:00+00:00")
        assert db.execute("SELECT conversation_id FROM requests WHERE id=?",
                          (rid,)).fetchone()[0] == "sess-abc"
        rt = [r[0] for r in db.execute(
            "SELECT conversation_id FROM routing_decisions ORDER BY id").fetchall()]
        assert rt[0] == "sess-abc"
        assert rt[1] is None  # outside the ±30s window

    def test_does_not_overwrite_existing(self, db):
        wc.ensure_schema()
        rid = _req(db, "2026-09-19T10:00:00+00:00")
        db.execute("UPDATE requests SET conversation_id=? WHERE id=?",
                   ("existing", rid))
        db.commit()
        wc.stamp_new(rid, "new", "l2", "2026-09-19T10:00:00+00:00")
        assert db.execute("SELECT conversation_id FROM requests WHERE id=?",
                          (rid,)).fetchone()[0] == "existing"


class TestSync:
    def test_generates_and_persists_name_summary(self, db):
        wc.ensure_schema()
        _req(db, "2026-09-19T10:00:00+00:00", profile="l2", model="a")
        _route(db, "2026-09-19T10:00:30+00:00", conv=json.dumps(
            [{"role": "user", "content": "Please analyze the zgx metrics now"}])
        )
        db.commit()
        wc.backfill_conversations()
        got = wc.sync_conversations()
        assert got["created"] == 1 and got["updated"] == 0
        row = db.execute("SELECT name, summary, calls FROM conversations").fetchone()
        assert row["name"] == "Please analyze the zgx metrics now"
        assert "1 calls" in row["summary"]
        # second sync only updates
        got2 = wc.sync_conversations()
        assert got2["created"] == 0 and got2["updated"] == 1
        # name/summary were NOT overwritten
        row2 = db.execute("SELECT name FROM conversations").fetchone()
        assert row2["name"] == "Please analyze the zgx metrics now"

    def test_fallback_name(self, db):
        wc.ensure_schema()
        _req(db, "2026-09-19T10:00:00+00:00", profile="l2", model="a")
        db.commit()
        wc.backfill_conversations()
        wc.sync_conversations()
        row = db.execute("SELECT name, summary FROM conversations").fetchone()
        assert row["name"] == "Conversation %s" % (
            db.execute("SELECT conversation_id FROM conversations").fetchone()[0])
        assert "no user turns captured" in row["summary"]


class TestViews:
    def test_conversations_view_and_detail(self, db):
        _req(db, "2026-09-19T10:00:00+00:00", profile="l2", model="a")
        _req(db, "2026-09-19T10:01:00+00:00", profile="l2", model="a")
        _req(db, "2026-09-19T11:00:00+00:00", profile="l1", model="b")
        _route(db, "2026-09-19T10:00:30+00:00", profile="l2", task="research",
               conv=json.dumps([{"role": "user", "content": "hello there long prompt"}]))
        db.commit()
        wc.backfill_conversations()
        v = wc.conversations_view({"per": "10"})
        # two conversations: the l2/a burst and the l1/b row
        assert v["total"] == 2
        assert v["filter"]["pages"] == 1
        # newest first
        assert v["conversations"][0]["profile"] == "l1"
        l2 = next(c for c in v["conversations"] if c["profile"] == "l2")
        assert l2["calls"] == 2
        assert l2["summary"].startswith("hello there")
        d = wc.conversation_detail(l2["id"])
        kinds = [e["kind"] for e in d["events"]]
        # chronological: first call (10:00:00) → routing (10:00:30) → second call (10:01:00)
        assert kinds == ["request", "routing", "request"]

    def test_extract_first_user(self):
        assert wc._extract_first_user(
            '[{"role":"system","content":"s"},{"role":"user","content":"the ask"}]'
        ) == "the ask"
        assert wc._extract_first_user('[{"role":"assistant","content":"x"}]') is None
        assert wc._extract_first_user("not json") is None