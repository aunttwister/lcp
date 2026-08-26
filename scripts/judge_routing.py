#!/usr/bin/env python3
"""Judge dynamic routing decisions — replay, review, report.

Pulls recorded ``routing_decisions`` (with the captured conversation shape +
classification rationale) and either re-runs the CURRENT classifier over them
(replay, flagging SAME/DRIFT/NODATA) or lets a human review them interactively
(review), building the ``routing_judgments`` labeled dataset.

Modes
-----
    replay   (default)
        Re-run the current classifier over captured decisions and flag
        SAME / DRIFT / NODATA, with rationale + aggregates.
    review
        Interactive batch review of unjudged decisions; writes routing_judgments.
    pending
        List unjudged decisions (review queue).
    report
        Aggregate the judged set: verdict distribution, per-task/per-path
        accuracy, expected x actual confusion.

Examples
--------
    .venv/bin/python scripts/judge_routing.py replay --limit 50
    .venv/bin/python scripts/judge_routing.py pending
    .venv/bin/python scripts/judge_routing.py review --limit 20
    .venv/bin/python scripts/judge_routing.py report --out /tmp/routing-report.md

DB resolution: --db PATH -> $COST_DB -> seed default (/app/data/costs.db).
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.api.router import classify_task_detail  # noqa: E402


def _resolve_db(args) -> str:
    if getattr(args, "db", None):
        return args.db
    # Match the gateway: $COST_DB wins exactly (the container sets it to
    # /app/data/costs.db). Never silently create a fresh empty DB.
    env = os.environ.get("COST_DB")
    if env:
        return env
    # Local dev: fall back to the repo's data dir only if a real DB exists.
    for cand in ("data/costs.db", "data/lcp.db", "lcp.db", "costs.db"):
        p = os.path.join(ROOT, cand)
        if os.path.exists(p):
            return p
    raise SystemExit(
        "No routing DB found. Run with --db PATH, or set COST_DB. "
        "Inside the container: docker compose exec -e COST_DB=/app/data/costs.db "
        "lcp python3 scripts/judge_routing.py replay"
    )


def _session(db_path):
    from src.api.models import get_engine, get_session
    return get_session(get_engine(db_path))


def _decisions(db_path, limit, since=None, profile=None):
    """Return the newest decisions as dicts (id desc), with filters."""
    from src.api.models import RoutingDecision
    with _session(db_path) as s:
        q = s.query(RoutingDecision).order_by(RoutingDecision.id.desc())
        if since:
            q = q.filter(RoutingDecision.ts >= since)
        if profile:
            q = q.filter(RoutingDecision.profile == profile)
        rows = q.limit(limit).all()
        return [{
            "id": r.id, "ts": r.ts, "profile": r.profile, "task": r.task,
            "path": r.path, "action": r.action,
            "provider": r.provider, "model": r.model, "score": r.score,
            "intent_text": r.intent_text, "semantic_json": r.semantic_json,
            "conversation_json": r.conversation_json, "note": r.note,
        } for r in rows]


def _judged_ids(db_path):
    from src.api.models import RoutingJudgment
    with _session(db_path) as s:
        return {j.decision_id for j in s.query(RoutingJudgment.decision_id).all()}


# ── replay ──────────────────────────────────────────────────────────────────

def cmd_replay(args) -> int:
    decs = _decisions(args.db, args.limit, args.since, args.profile)
    if not decs:
        print("No routing decisions found.")
        return 1
    flags = Counter()
    tasks = Counter()
    paths = Counter()
    print(f"{'id':>5}  {'ts':<22} {'task':<14} {'path':<15} {'flag':<8} intent")
    print("-" * 104)
    for d in decs:
        conv = json.loads(d["conversation_json"]) if d["conversation_json"] else None
        re_task = re_path = None
        if conv is None:
            flag = "NODATA"
        else:
            try:
                detail = classify_task_detail(conv)
                re_task, re_path = detail.task, detail.path
                if not detail.intent_text:
                    flag = "NOINTENT"
                else:
                    flag = "SAME" if detail.task == d["task"] else "DRIFT"
            except Exception:  # noqa: BLE001 — a bad row must not abort the replay
                flag = "NODATA"
        flags[flag] += 1
        tasks[d["task"]] += 1
        paths[d["path"] or "?"] += 1
        if re_task:
            shown_task = f"{d['task']}→{re_task}" if flag == "DRIFT" else d["task"]
        else:
            shown_task = d["task"]
        rec_path = d["path"] or "?"
        shown_path = f"{rec_path}→{re_path}" if (re_path and re_path != rec_path) else rec_path
        intent = (d["intent_text"] or "").replace("\n", " ")[:52]
        print(f"{d['id']:>5}  {d['ts'][:22]:<22} {shown_task:<14} {shown_path:<15} {flag:<8} {intent}")
    print("-" * 104)
    print("flags: " + ", ".join(f"{k}={v}" for k, v in flags.most_common()))
    print("tasks: " + ", ".join(f"{k}={v}" for k, v in tasks.most_common()))
    print("paths: " + ", ".join(f"{k}={v}" for k, v in paths.most_common()))
    if flags.get("DRIFT"):
        print("\nDRIFT rows are decisions the CURRENT code would classify differently —")
        print("re-run replay after a router change to spot regressions.")
    return 0


# ── pending ─────────────────────────────────────────────────────────────────

def cmd_pending(args) -> int:
    from src.api.models import RoutingDecision
    with _session(args.db) as s:
        total = s.query(RoutingDecision).count()
    decs = _decisions(args.db, args.limit, args.since, args.profile)
    judged = _judged_ids(args.db)
    pending = [d for d in decs if d["id"] not in judged]
    print(f"{len(pending)} unjudged of {len(decs)} shown (limit={args.limit}); "
          f"{total} decisions total; {len(judged)} judged.")
    for d in pending:
        print(f"  #{d['id']} {d['ts'][:19]} profile={d['profile']!r} "
              f"task={d['task']!r} path={d['path'] or '?'} action={d['action']}")
    return 0


# ── review ──────────────────────────────────────────────────────────────────

def cmd_review(args) -> int:
    from src.api.models import RoutingJudgment
    decs = _decisions(args.db, args.limit, args.since, args.profile)
    judged = _judged_ids(args.db)
    pending = [d for d in decs if d["id"] not in judged]
    if not pending:
        print("No unjudged decisions to review.")
        return 0
    written = 0
    for d in pending:
        print("\n" + "=" * 100)
        print(f"decision #{d['id']}  {d['ts']}  profile={d['profile']!r}")
        print(f"  recorded : task={d['task']!r} path={d['path'] or '?'} "
              f"action={d['action']} provider={d['provider']} model={d['model']} "
              f"score={d['score']}")
        print(f"  intent   : {(d['intent_text'] or '').strip()[:160]!r}")
        if d["semantic_json"]:
            sem = json.loads(d["semantic_json"])
            print("  semantic : " + ", ".join(f"{t}={s}" for t, s in sem))
        if d["conversation_json"]:
            conv = json.loads(d["conversation_json"])
            roles = " > ".join(m.get("role", "?") for m in conv)
            print(f"  messages : {roles}")
        verdict = input("  [c]orrect  [w]rong  [a]mbiguous  [s]kip  [q]uit ? ").strip().lower()
        if verdict in ("q", "quit"):
            break
        if verdict in ("s", "skip", ""):
            continue
        v = {"c": "correct", "correct": "correct",
             "w": "wrong", "wrong": "wrong",
             "a": "ambiguous", "ambiguous": "ambiguous"}.get(verdict)
        if v is None:
            print("  (unrecognized — skipped)")
            continue
        expected = None
        note = None
        if v in ("wrong", "ambiguous"):
            expected = input("  expected task (blank to skip): ").strip() or None
            if v == "wrong":
                note = input("  note: ").strip() or None
        with _session(args.db) as s:
            s.add(RoutingJudgment(
                decision_id=d["id"], profile=d["profile"], task=d["task"],
                path=d["path"], verdict=v, expected_task=expected, note=note,
            ))
            s.commit()
        written += 1
        print(f"  → judged {v}")
    print(f"\n{written} judgment(s) written.")
    return 0


# ── report ──────────────────────────────────────────────────────────────────

def _accuracy(c: Counter):
    tot = c.get("correct", 0) + c.get("wrong", 0)
    return (c.get("correct", 0) / tot) if tot else None


def cmd_report(args) -> int:
    from src.api.models import RoutingDecision, RoutingJudgment
    with _session(args.db) as s:
        rows = s.query(RoutingJudgment).order_by(RoutingJudgment.id.asc()).all()
        dec_count = s.query(RoutingDecision).count()
    if not rows:
        print("No judgments recorded yet. Run `judge_routing.py review` first.")
        return 0
    verdicts = Counter(r.verdict for r in rows)
    per_task = defaultdict(Counter)
    per_path = defaultdict(Counter)
    confusion = Counter()
    for r in rows:
        per_task[r.task][r.verdict] += 1
        per_path[r.path or "?"][r.verdict] += 1
        if r.verdict == "wrong" and r.expected_task:
            confusion[(r.task, r.expected_task)] += 1

    lines = [
        f"# Routing judgment report ({len(rows)} judgments, {dec_count} decisions)",
        "",
        "## Verdict distribution",
    ]
    for v in ("correct", "wrong", "ambiguous"):
        lines.append(f"- {v}: {verdicts.get(v, 0)}")
    lines += ["", "## Per-task accuracy"]
    for task in sorted(per_task, key=lambda t: -sum(per_task[t].values())):
        c = per_task[task]
        acc = _accuracy(c)
        lines.append(
            f"- {task}: {f'{acc:.0%}' if acc is not None else '-'}  "
            f"({c.get('correct', 0)}c / {c.get('wrong', 0)}w / {c.get('ambiguous', 0)}a)"
        )
    lines += ["", "## Per-path accuracy"]
    for path in sorted(per_path, key=lambda p: -sum(per_path[p].values())):
        c = per_path[path]
        acc = _accuracy(c)
        lines.append(
            f"- {path}: {f'{acc:.0%}' if acc is not None else '-'}  "
            f"({c.get('correct', 0)}c / {c.get('wrong', 0)}w / {c.get('ambiguous', 0)}a)"
        )
    lines += ["", "## Confusion (actual task → expected, wrong judgments)"]
    if confusion:
        for (actual, expected), n in sorted(confusion.items(), key=lambda x: -x[1]):
            lines.append(f"- {actual} → {expected}: {n}")
    else:
        lines.append("- (none)")
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"Wrote {args.out}")
    else:
        print(text)
    return 0


# ── main ────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Judge dynamic routing decisions.")
    p.add_argument("mode", nargs="?", default="replay",
                   choices=["replay", "review", "pending", "report"],
                   help="what to do (default: replay)")
    p.add_argument("--db", help="SQLite DB path (default: $COST_DB or seed default)")
    p.add_argument("--limit", type=int, default=50, help="max decisions to consider")
    p.add_argument("--since", help="only decisions with ts >= this ISO string")
    p.add_argument("--profile", help="only decisions for this profile")
    p.add_argument("--out", help="write the report to this file")
    args = p.parse_args(argv)
    cmds = {"replay": cmd_replay, "review": cmd_review,
            "pending": cmd_pending, "report": cmd_report}
    return cmds[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
