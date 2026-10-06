#!/usr/bin/env python3
"""Import the Hermes session corpus into LCP's own conversation store.

This is the one-off backfill behind "LCP owns its conversation history". It reads
each Hermes profile's ``state.db`` — which is a *migration source*, not a runtime
dependency — and writes the conversations and their messages into LCP's
``conversations`` / ``conv_messages`` tables.

It is idempotent and re-runnable. Every write goes through
``conv_store.append_messages``, which is positional: it compares the incoming
messages against what is already stored and appends only the unmatched suffix. So
running it twice imports nothing the second time, and running it again after new
sessions have ended picks up exactly those sessions.

Usage::

    # from the repo root, with the app venv
    python scripts/migrate_conversations.py --dry-run
    python scripts/migrate_conversations.py
    python scripts/migrate_conversations.py --profile homelab-expert-l2

    # inside the container (profiles are mounted read-only at /app/profiles)
    docker exec lcp python /app/scripts/migrate_conversations.py

Exit status is 0 on success, 1 if any profile failed to read, so a scheduled run
cannot silently import nothing.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Iterator, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import conv_store  # noqa: E402
from src.api.work_conversations import db_path  # noqa: E402

#: Message roles that belong to the conversation. ``session_meta`` rows are the
#: store's own bookkeeping, not something anyone said.
_CONVERSATION_ROLES = ("user", "assistant", "tool")

#: Where the Hermes profiles live inside the container and on the host.
_PROFILE_ROOTS = ("/app/profiles", "/root/.hermes/profiles")


def discover_stores(root: Optional[str] = None) -> list[tuple[str, str]]:
    """Return ``(profile_name, state.db path)`` for every readable Hermes store.

    ``root`` overrides the search location — used by tests to point at a frozen
    copy of a store so a re-run can be measured against an unchanging source.
    """
    roots = (root,) if root else _PROFILE_ROOTS
    found: list[tuple[str, str]] = []
    for r in roots:
        if not os.path.isdir(r):
            continue
        for name in sorted(os.listdir(r)):
            path = os.path.join(r, name, "state.db")
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                found.append((name, path))
        if found and not root:
            break  # first root that yields anything wins (container vs host)
    return found


def _iso(epoch: Optional[float]) -> Optional[str]:
    if not epoch:
        return None
    try:
        return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _sessions(src: sqlite3.Connection) -> list[sqlite3.Row]:
    cols = {r[1] for r in src.execute("PRAGMA table_info(sessions)")}
    wanted = [c for c in ("id", "profile_name", "model", "title", "started_at",
                          "ended_at", "message_count", "tool_call_count",
                          "input_tokens", "output_tokens", "api_call_count")
              if c in cols]
    return list(src.execute("SELECT %s FROM sessions" % ", ".join(wanted)))


def _messages(src: sqlite3.Connection, session_id: str) -> Iterator[dict]:
    """Stream one session's messages in conversation order (oldest first)."""
    cols = {r[1] for r in src.execute("PRAGMA table_info(messages)")}
    picked = [c for c in ("role", "content", "tool_name", "tool_call_id",
                          "tool_calls", "reasoning", "timestamp", "active",
                          "compacted") if c in cols]
    sql = ("SELECT %s FROM messages WHERE session_id=? AND role IN (%s) "
           "ORDER BY id" % (", ".join(picked),
                            ", ".join("?" * len(_CONVERSATION_ROLES))))
    for row in src.execute(sql, [session_id, *_CONVERSATION_ROLES]):
        m = dict(row)
        m["ts"] = _iso(m.pop("timestamp", None))
        yield m


def migrate_profile(name: str, path: str, *, dry_run: bool = False,
                    limit: Optional[int] = None) -> dict:
    src = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    src.row_factory = sqlite3.Row
    stats = {"profile": name, "sessions": 0, "messages": 0, "skipped": 0}
    try:
        sessions = _sessions(src)
        for i, s in enumerate(sessions):
            if limit is not None and i >= limit:
                break
            sid = s["id"]
            msgs = list(_messages(src, sid))
            if not msgs:
                stats["skipped"] += 1
                continue
            profile = s["profile_name"] if "profile_name" in s.keys() else None
            profile = profile or name
            if not dry_run:
                conv_store.append_messages(sid, msgs, origin="migration")
                # Counts come from what was actually stored, not from Hermes'
                # ``sessions.message_count`` — that column is the post-compaction
                # live window, not the transcript (see alembic 024).
                conv_store.upsert_conversation(
                    sid,
                    profile=profile,
                    model=s["model"] if "model" in s.keys() else None,
                    name=s["title"] if "title" in s.keys() else None,
                    source="hermes",
                    started_at=_iso(s["started_at"]) if "started_at" in s.keys() else None,
                    ended_at=_iso(s["ended_at"]) if "ended_at" in s.keys() else None,
                    tool_call_count=s["tool_call_count"] if "tool_call_count" in s.keys() else None,
                    input_tokens=s["input_tokens"] if "input_tokens" in s.keys() else None,
                    output_tokens=s["output_tokens"] if "output_tokens" in s.keys() else None,
                )
                conv_store.recount_conversation(sid)
            stats["sessions"] += 1
            stats["messages"] += len(msgs)
    finally:
        src.close()
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="read and count only; write nothing")
    ap.add_argument("--profile", action="append", default=None,
                    help="limit to this profile (repeatable)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N sessions per profile (for a smoke run)")
    ap.add_argument("--root", default=None,
                    help="read profiles from this directory instead of the defaults")
    args = ap.parse_args()

    stores = discover_stores(args.root)
    if args.profile:
        wanted = set(args.profile)
        stores = [s for s in stores if s[0] in wanted]
    if not stores:
        print("no Hermes session stores found under %s" % ", ".join(_PROFILE_ROOTS),
              file=sys.stderr)
        return 1

    if not args.dry_run:
        conv_store.ensure_store()

    print("LCP store : %s" % db_path())
    print("mode      : %s" % ("DRY RUN (no writes)" if args.dry_run else "import"))
    print("sources   : %d profile(s)" % len(stores))
    print()

    failed = 0
    t0 = time.time()
    totals = {"sessions": 0, "messages": 0, "skipped": 0}
    for name, path in stores:
        try:
            st = migrate_profile(name, path, dry_run=args.dry_run, limit=args.limit)
        except sqlite3.DatabaseError as exc:
            print("  %-24s FAILED: %s" % (name, exc), file=sys.stderr)
            failed += 1
            continue
        for k in totals:
            totals[k] += st[k]
        print("  %-24s %5d conversations  %7d messages  (%d empty)"
              % (name, st["sessions"], st["messages"], st["skipped"]))

    print()
    print("total: %d conversations, %d messages in %.1fs"
          % (totals["sessions"], totals["messages"], time.time() - t0))
    if not args.dry_run:
        print("stored: %d messages now in conv_messages"
              % _stored_count())
    return 1 if failed else 0


def _stored_count() -> int:
    con = conv_store._connect()
    try:
        return con.execute("SELECT COUNT(*) FROM conv_messages").fetchone()[0]
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
