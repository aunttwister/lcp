#!/usr/bin/env python3
"""Backfill: rewrite wrapper-named tool RESULT rows to the tool that actually ran.

Measured shape of the real store (2026-11-23): the wrapper name is on the **result**
rows (`role='tool'`, 82,651 of them), while assistant rows carry no `tool_name` at all.
So the repair is a lookup, not a re-parse of the row itself: a tool result carries only
`tool_call_id`, and the tool's real name is inside the caller assistant turn's
`tool_calls[].function.arguments` under `calls[].name`.

Idempotent: a second run changes nothing (no wrapper-named rows remain to match).
Batched by rowid so a 3.7 M-row table is never held in memory. Reports the residue it
could not resolve — that number is part of the output, not a silent loss.

Usage:
  python3 scripts/backfill_deferred_tool_names.py --db /app/data/costs.db --dry-run
  python3 scripts/backfill_deferred_tool_names.py --db /app/data/costs.db --apply
"""
import argparse
import json
import sqlite3
import sys

WRAPPERS = ("tool_call",)


#: The ONLY name a deferred catalog leaves on a row. `process_manage` is a real
#: inner tool name, not a wrapper — listing it here made the backfill rewrite
#: legitimate rows to themselves and never converge (measured 2026-11-23).
def unwrap(arguments):
    """Mirror of conv_store._unwrap_deferred_tool_args — same rules, no import, so
    this runs against a database whose app version may differ."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, TypeError):
            return None
    if not isinstance(arguments, dict):
        return None
    calls = arguments.get("calls")
    if isinstance(calls, str):
        try:
            calls = json.loads(calls)
        except (ValueError, TypeError):
            return None
    if not isinstance(calls, list):
        return None
    names = []
    for c in calls:
        if isinstance(c, dict):
            n = c.get("name")
            if isinstance(n, str) and n and n not in names:
                names.append(n)
    return ",".join(names) if names else None


def caller_map(con, conversation_id, tool_call_ids):
    """tool_call_id -> real tool name, resolved from this conversation's assistant turns."""
    if not tool_call_ids:
        return {}
    want = set(tool_call_ids)
    out = {}
    rows = con.execute(
        "SELECT tool_calls FROM conv_messages "
        "WHERE conversation_id=? AND role='assistant' AND tool_calls IS NOT NULL",
        (conversation_id,))
    for (raw,) in rows:
        try:
            tcs = json.loads(raw) if raw else []
        except (ValueError, TypeError):
            continue
        for tc in tcs if isinstance(tcs, list) else []:
            if not isinstance(tc, dict):
                continue
            tid = tc.get("id")
            if tid not in want or tid in out:
                continue
            fn = tc.get("function") or {}
            inner = unwrap(fn.get("arguments") if isinstance(fn, dict) else None)
            if not inner and tc.get("calls"):
                inner = unwrap(tc.get("calls"))
            if inner:
                out[tid] = inner
            if len(out) == len(want):
                return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", type=int, default=5000)
    args = ap.parse_args()
    if not (args.apply or args.dry_run):
        sys.exit("choose --dry-run or --apply")

    con = sqlite3.connect(args.db, timeout=60)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")

    placeholders = ",".join("?" for _ in WRAPPERS)
    have = {(r[0], r[1]) for r in con.execute(f"PRAGMA table_info(conv_messages)")}
    del have

    total = con.execute(
        f"SELECT COUNT(*) FROM conv_messages WHERE role='tool' "
        f"AND tool_name IN ({placeholders})", WRAPPERS).fetchone()[0]
    print(f"result rows named as a wrapper: {total}")

    fixed = unresolved = 0
    last_rowid = 0
    cache_cid, cache_map = None, {}
    while True:
        rows = con.execute(
            f"SELECT rowid AS rid, conversation_id, tool_call_id FROM conv_messages "
            f"WHERE rowid > ? AND role='tool' AND tool_name IN ({placeholders}) "
            f"ORDER BY rowid LIMIT ?", (last_rowid, *WRAPPERS, args.batch)).fetchall()
        if not rows:
            break
        last_rowid = rows[-1]["rid"]
        # one conversation's caller map serves every row from that conversation
        by_cid = {}
        for r in rows:
            by_cid.setdefault(r["conversation_id"], []).append(r)
        for cid, group in by_cid.items():
            if cid != cache_cid:
                cache_cid = cid
                cache_map = caller_map(con, cid, [g["tool_call_id"] for g in group
                                                  if g["tool_call_id"]])
            for g in group:
                name = cache_map.get(g["tool_call_id"]) if g["tool_call_id"] else None
                if not name:
                    unresolved += 1
                    continue
                fixed += 1
                if args.apply:
                    con.execute("UPDATE conv_messages SET tool_name=? WHERE rowid=?",
                                (name, g["rid"]))
        if args.apply:
            con.commit()
        print(f"  scanned to rowid {last_rowid}: fixed={fixed} unresolved={unresolved}",
              flush=True)

    print("\n--- summary ---")
    print(f"  result rows rewritten    : {fixed}")
    print(f"  unresolved (no caller / unparseable / no id): {unresolved}")
    print(f"  mode: {'APPLIED' if args.apply else 'DRY RUN — nothing written'}")
    con.close()


if __name__ == "__main__":
    main()
