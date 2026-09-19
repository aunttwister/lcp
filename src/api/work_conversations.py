"""Unified conversation-centered log view.

Three log realms currently sit in separate tables under the same costs.db:

* ``requests``           — every chat-completions call (model, provider, tokens,
                           cost, latency, success/error)
* ``routing_decisions``  — the router's provider decision per request (task
                           label, policy/action, and a summary of the
                           conversation content that drove it)
* (external) the board decisions ledger, joined by profile+time later

Until now nothing tied them together. This module introduces the missing
correlation key — ``conversation_id`` — on both tables:

* NEW calls are stamped at write time with the client's real per-conversation
  header (``x-opencode-session``, sent by Hermes >= 0.21). Rows written by
  clients WITHOUT that header (or pre-date this change) get NULL and are
  covered by a one-time BACKFILL: same (profile, model), gaps <= 5 minutes
  form one conversation.
* The backfill is deterministic and idempotent (fills NULLs only).

The view then renders conversations newest-first: a short summary (first user
message + counts) per conversation, expanding into the chronological sequence
of request + routing events. Determinstic summaries in v1; an LLM summary
pass is a later, optional refinement.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional

# conversation_id values are opaque strings: either the client's session
# header or a deterministic backfill id like "c<10 hex>".
_BURST_GAP_SECONDS = 300
_CONV_SQLITE = "data/costs.db"


def db_path() -> str:
    return os.environ.get("LCP_COSTS_DB", _CONV_SQLITE)


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(db_path(), timeout=10)
    con.row_factory = sqlite3.Row
    return con


def _burstable(row: sqlite3.Row, prev: Optional[sqlite3.Row],
               gap: float) -> bool:
    if prev is None:
        return True
    if row["profile"] != prev["profile"] or row["model"] != prev["model"]:
        return True
    return gap > _BURST_GAP_SECONDS


def _backfill_id(profile: str, model: str, first_ts: str) -> str:
    import hashlib
    return "c" + hashlib.sha1(
        ("%s|%s|%s" % (profile, model, first_ts)).encode()).hexdigest()[:10]


def ensure_schema() -> None:
    """Idempotent ALTER: add conversation_id to both tables."""
    con = _connect()
    try:
        for table in ("requests", "routing_decisions"):
            cols = {r[1] for r in con.execute("PRAGMA table_info(%s)" % table)}
            if "conversation_id" not in cols:
                con.execute("ALTER TABLE %s ADD COLUMN conversation_id TEXT" % table)
        con.commit()
    finally:
        con.close()


def _last_backfill_ts() -> Optional[str]:
    try:
        con = _connect()
        row = con.execute(
            "SELECT value FROM settings WHERE key='convo_backfill_ts'").fetchone()
        con.close()
        return row["value"] if row else None
    except sqlite3.Error:
        return None


def _set_last_backfill_ts(ts: str) -> None:
    con = _connect()
    try:
        con.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
        try:
            # Some installs have settings.updated_at NOT NULL — satisfy it,
            # else fall back to the minimal row shape.
            con.execute(
                "INSERT OR REPLACE INTO settings(key, value, updated_at) "
                "VALUES(?,?,?)", ("convo_backfill_ts", ts, ts))
        except sqlite3.Error:
            con.execute(
                "INSERT OR REPLACE INTO settings(key, value) VALUES(?,?)",
                ("convo_backfill_ts", ts))
        con.commit()
    finally:
        con.close()


def backfill_conversations(vacuum: bool = False) -> Dict[str, int]:
    """Assign conversation_id to legacy rows (NULL only). Returns counts.

    Requests: burst grouping over (profile, model) ordered by timestamp.
    Routing decisions: joined to whichever conversation is active for their
    profile within the burst window; orphaned rows get their own ids.
    """
    ensure_schema()
    stamp_requests = 0
    stamp_routing = 0

    con = _connect()
    try:
        rows = con.execute(
            "SELECT id, timestamp, profile, model FROM requests "
            "WHERE conversation_id IS NULL ORDER BY profile, model, timestamp"
        ).fetchall()
        prev: Optional[sqlite3.Row] = None
        cur_id: Optional[str] = None
        for r in rows:
            gap = 0.0
            if prev is not None:
                try:
                    gap = (__import__("datetime").datetime.fromisoformat(
                        r["timestamp"].replace("Z", "+00:00")) -
                        __import__("datetime").datetime.fromisoformat(
                            prev["timestamp"].replace("Z", "+00:00"))).total_seconds()
                except ValueError:
                    gap = _BURST_GAP_SECONDS + 1
            if _burstable(r, prev, gap):
                cur_id = _backfill_id(r["profile"], r["model"], r["timestamp"])
            con.execute("UPDATE requests SET conversation_id=? WHERE id=?",
                        (cur_id, r["id"]))
            stamp_requests += 1
            prev = r

        con.commit()

        # routing decisions: attach to the conversation whose request window
        # covers them (same profile). Requests are backfilled now, so look up
        # by nearest request conversation within a sliding join.
        rt = con.execute(
            "SELECT id, ts, profile FROM routing_decisions "
            "WHERE conversation_id IS NULL ORDER BY profile, ts"
        ).fetchall()
        for r in rt:
            match = con.execute(
                """
                SELECT conversation_id FROM requests
                WHERE profile = ? AND conversation_id IS NOT NULL
                  AND datetime(timestamp) BETWEEN
                      datetime(?, '-10 minutes') AND datetime(?, '+10 minutes')
                ORDER BY abs(julianday(timestamp) - julianday(?)) LIMIT 1
                """, (r["profile"], r["ts"], r["ts"], r["ts"])).fetchone()
            conv = match["conversation_id"] if match else None
            if conv is None:
                # synthetic conversation for the orphan
                first_ts = r["ts"]
                conv = _backfill_id(r["profile"], "__routing__", first_ts)
            con.execute("UPDATE routing_decisions SET conversation_id=? WHERE id=?",
                        (conv, r["id"]))
            stamp_routing += 1
        con.commit()
    finally:
        con.close()

    _set_last_backfill_ts(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return {"requests": stamp_requests, "routing": stamp_routing}


def stamp_new(request_id: int, conversation_id: str, profile: str,
              ts: str) -> None:
    """Stamp a JUST-inserted request + the routing decisions of the same call.

    Called by the proxy at the end of a request when the client sent a real
    per-conversation header. The request row is matched by its primary key;
    the routing rows by the request's timestamp window (±30s), which keeps
    the stamp from touching unrelated traffic.
    """
    ensure_schema()
    con = _connect()
    try:
        con.execute(
            "UPDATE requests SET conversation_id = ? WHERE id = ? "
            "AND conversation_id IS NULL",
            (conversation_id, request_id))
        delta = 30  # seconds of slack for the routing rows of this call
        con.execute(
            """
            UPDATE routing_decisions SET conversation_id = ?
            WHERE conversation_id IS NULL
              AND profile = ?
              AND datetime(ts) BETWEEN datetime(?, '-%d seconds')
                                     AND datetime(?, '+%d seconds')
            """ % (delta, delta),
            (conversation_id, profile, ts, ts))
        con.commit()
    finally:
        con.close()


def _extract_first_user(conversation_json: Optional[str]) -> Optional[str]:
    if not conversation_json:
        return None
    try:
        msgs = json.loads(conversation_json)
    except (ValueError, TypeError):
        return None
    if not isinstance(msgs, list):
        return None
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                return re.sub(r"\s+", " ", c).strip()[:_CONV_SUMMARY_CAP]
            if isinstance(c, list):
                parts = [p.get("text", "") for p in c if isinstance(p, dict)]
                joined = " ".join(parts).strip()
                if joined:
                    return joined[:_CONV_SUMMARY_CAP]
    return None


_CONV_SUMMARY_CAP = 240


def conversations_view(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Paginated conversation list + per-conversation summary."""
    params = params or {}
    per_raw = str(params.get("per") or "20")
    if per_raw == "all":
        per = "all"
    else:
        try:
            per = max(1, min(500, int(per_raw)))
        except (TypeError, ValueError):
            per = 20
    try:
        page = max(1, int(params.get("page") or 1))
    except (TypeError, ValueError):
        page = 1
    profile_filter = str(params.get("profile") or "").strip()

    ensure_schema()
    con = _connect()
    try:
        where = "WHERE r.conversation_id IS NOT NULL"
        args: List[Any] = []
        if profile_filter:
            where += " AND r.profile = ?"
            args.append(profile_filter)

        total = con.execute(
            "SELECT COUNT(DISTINCT conversation_id) FROM requests r " + where,
            args).fetchone()[0]

        base = """
            SELECT r.conversation_id AS cid,
                   r.profile AS profile,
                   r.model AS model,
                   COUNT(*) AS calls,
                   SUM(r.prompt_tokens + r.completion_tokens) AS tokens,
                   SUM(r.cost) AS cost,
                   SUM(CASE WHEN r.success = 0 THEN 1 ELSE 0 END) AS errors,
                   MIN(r.timestamp) AS started_at,
                   MAX(r.timestamp) AS last_at
            FROM requests r
            {where}
            GROUP BY r.conversation_id, r.profile, r.model
            ORDER BY last_at DESC, cid
        """
        if per == "all":
            rows = con.execute(base.format(where=where), args).fetchall()
            page = 1
            pages = 1
        else:
            pages = max(1, -(-total // per))
            page = min(page, pages)
            rows = con.execute(base.format(where=where) + " LIMIT ? OFFSET ?",
                               args + [per, (page - 1) * per]).fetchall()

        conversations = []
        for r in rows:
            cid = r["cid"]
            # routing context for this conversation: task labels + a slice of
            # the conversation content that drove routing (for the summary)
            route = con.execute(
                """
                SELECT task, action, conversation_json FROM routing_decisions
                WHERE conversation_id = ? ORDER BY ts DESC LIMIT 1
                """, (cid,)).fetchone()
            first_user = None
            if route:
                first_user = _extract_first_user(route["conversation_json"])
            conversations.append({
                "id": cid,
                "profile": r["profile"],
                "model": r["model"],
                "calls": r["calls"],
                "tokens": r["tokens"] or 0,
                "cost": round(r["cost"] or 0.0, 6),
                "errors": r["errors"] or 0,
                "started_at": r["started_at"],
                "last_at": r["last_at"],
                "task": route["task"] if route else None,
                "summary": first_user,
            })
        con.close()
    except sqlite3.Error:
        con.close()
        raise

    return {
        "available": True,
        "conversations": conversations,
        "total": total,
        "filter": {
            "per": str(per), "page": page, "pages": pages,
            "profile": profile_filter,
            "profiles": _profiles(),
        },
    }


def _profiles() -> List[str]:
    con = _connect()
    try:
        return [r[0] for r in con.execute(
            "SELECT DISTINCT profile FROM requests WHERE conversation_id IS NOT NULL "
            "ORDER BY profile")]
    finally:
        con.close()


def conversation_detail(cid: str, limit: int = 500) -> Dict[str, Any]:
    """The chronological sequence of a conversation: requests + routing."""
    ensure_schema()
    con = _connect()
    try:
        events: List[Dict[str, Any]] = []
        for r in con.execute(
                "SELECT * FROM requests WHERE conversation_id=? ORDER BY timestamp",
                (cid,)):
            events.append({
                "kind": "request", "ts": r["timestamp"],
                "provider": r["provider"], "model": r["model"],
                "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                "cost": r["cost"], "latency_ms": r["latency_ms"],
                "success": r["success"], "error_type": r["error_type"],
                "error_detail": r["error_detail"],
            })
        for r in con.execute(
                "SELECT * FROM routing_decisions WHERE conversation_id=? "
                "ORDER BY ts", (cid,)):
            events.append({
                "kind": "routing", "ts": r["ts"],
                "task": r["task"], "policy": r["policy"], "action": r["action"],
                "provider": r["provider"], "model": r["model"], "score": r["score"],
                "note": r["note"],
            })
        events.sort(key=lambda e: e["ts"])
        totals = con.execute(
            "SELECT COUNT(*) AS calls, SUM(cost) AS cost, "
            "SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) AS errors "
            "FROM requests WHERE conversation_id=?", (cid,)).fetchone()
        con.close()
    finally:
        pass
    return {
        "id": cid,
        "events": events[:limit],
        "truncated": len(events) > limit,
        "calls": totals["calls"], "cost": round(totals["cost"] or 0.0, 6),
        "errors": totals["errors"] or 0,
    }