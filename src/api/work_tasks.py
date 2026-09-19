"""Work-layer: Tasks and Fleet views.

Same moment model as the Decisions view -- but the sources are the filesystem
(task directories) and the gateway health endpoint, not the board ledger.

A task directory is a small structured record whose *state is its location*
(new -> in_progress -> completed / cancelled). That makes ``moved`` a first-class
moment: the directory's mtime IS the transition time, and the destination IS the
new state. No extra bookkeeping, and it cannot drift from reality because the
location is the state.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

# Where the task trees live. Overridable so LCP can point at any profile.
DEFAULT_TASKS_DIR = "/root/.hermes/profiles/homelab-expert-l2/work/tasks"

STATES = ("new", "in_progress", "completed", "cancelled")

# How much of a PLAN.md / RESULTS.md to carry into the view. The plan is the
# task's document of record; the panel shows its head as a summary and the
# full (capped) text inside the expander. Truncation is flagged, not silent.
_PLAN_CAP = 8000
_RESULTS_CAP = 2000


def _md_to_html(text: str) -> str:
    """CommonMark -> HTML with raw HTML escaped (html=False).

    Task files are agent-written, but they are still untrusted input as far
    as the browser is concerned; markdown-it with html=False renders any raw
    HTML inside the document as escaped text instead of pasting it through.
    """
    from markdown_it import MarkdownIt

    md = MarkdownIt("commonmark", {"html": False, "linkify": False})
    return md.render(text)


def _plan_summary(plan_text: str) -> Optional[str]:
    """The first substantive line of a PLAN.md, minus plumbing.

    Skips headings and the front-matter meta lines (**Created:**, **Status:**,
    **Branch:**, **Owner:**) so the summary is what the task is actually about
    -- for most plans that is the first user-ask or problem sentence.
    """
    skip_meta = ("created:", "status:", "branch:", "owner:", "user ask")
    for raw in plan_text.splitlines():
        line = raw.strip().lstrip("*_`~ ").strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        low = line.lower()
        if low.startswith(skip_meta):
            continue
        return line[:320]
    return None


def tasks_dir() -> str:
    """Resolve the task tree root: env override, then configured sources."""
    env = os.environ.get("LCP_WORK_TASKS_DIR")
    if env:
        return env
    try:
        from . import work_sources
        src = work_sources.load_sources()
        if src and src.get("tasks_root"):
            return src["tasks_root"]
    except Exception:  # noqa: BLE001
        pass
    return DEFAULT_TASKS_DIR


def todos_path() -> str:
    """The sibling todo.md for the same profile as the task tree."""
    root = tasks_dir()
    return os.environ.get("LCP_WORK_TODO", os.path.join(os.path.dirname(root), "todo.md"))


def _classification_index() -> Optional[Dict[str, Any]]:
    """The work-layers classification index, if the batch has run.

    ``<tasks_root>/.work-layers/classifications.json`` is written by the
    deterministic classifier (bge-small centroids over a hand-authored
    taxonomy). Defensive: a missing or corrupt index simply means the panel
    shows no chips.
    """
    root = tasks_dir()
    p = os.path.join(root, ".work-layers", "classifications.json")
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _state_summary(tdir: str) -> Dict[str, Any]:
    """STATE-SUMMARY.md (PLAN-vs-REAL, written by the summarizer batch)."""
    p = os.path.join(tdir, "STATE-SUMMARY.md")
    if not os.path.isfile(p):
        return {"text": None, "html": None, "present": False}
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read(4000)
    except OSError:
        return {"text": None, "html": None, "present": False}
    return {"text": text, "html": _md_to_html(text), "present": True}


_FACTS_CAP = 4000


def _session_facts(tdir: str) -> Dict[str, Any]:
    """SESSION-FACTS.md (mined from sessions by the 4h assessment round)."""
    p = os.path.join(tdir, "SESSION-FACTS.md")
    if not os.path.isfile(p):
        return {"text": None, "html": None, "present": False, "truncated": False}
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read(_FACTS_CAP)
        truncated = os.path.getsize(p) > _FACTS_CAP
    except OSError:
        return {"text": None, "html": None, "present": False, "truncated": False}
    return {"text": text, "html": _md_to_html(text), "present": True, "truncated": truncated}


def _assessment_feed(limit: int = 40) -> List[Dict[str, Any]]:
    """Newest decisions from .work-layers/assessments.jsonl (append-only).

    The ledger is written by the 4h session-assessment round; a corrupt or
    absent file must never crash the view, so record-level tolerance is
    mandatory (one bad line is skipped, not fatal).
    """
    root = tasks_dir()
    p = os.path.join(root, ".work-layers", "assessments.jsonl")
    if not os.path.isfile(p):
        return []
    records = []
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    records.sort(key=lambda r: r.get("ts", 0) or 0, reverse=True)
    out = []
    for r in records[:limit]:
        out.append({
            "ts": r.get("ts"),
            "ts_iso": time.strftime("%Y-%m-%d %H:%M",
                                    time.gmtime(r.get("ts", 0) or 0)),
            "action": r.get("action"),
            "slug": r.get("slug"),
            "summary": (r.get("summary") or "")[:200],
            "applied": bool(r.get("applied")),
            "reason": r.get("reason"),
        })
    return out


def _read_status_line(plan_path: str) -> Optional[str]:
    """Pull the bolded **Status:** line out of a PLAN.md, if present.

    PLAN.md files are prose, not a schema -- but every one written to the task
    standard carries a ``**Status:**`` line. Reading that one line is enough to
    show what a task believes about itself, and when it disagrees with the
    directory it lives in, that disagreement is worth showing rather than hiding.
    """
    try:
        with open(plan_path, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(4000)
    except OSError:
        return None
    m = re.search(r"\*\*Status:\*\*\s*(.+)", head)
    if not m:
        return None
    return m.group(1).strip().rstrip("*").strip()


def task_moments() -> List[Dict[str, Any]]:
    """Every task directory as a moment, ordered newest transition first."""
    root = tasks_dir()
    if not os.path.isdir(root):
        return []

    cls_idx = _classification_index()
    cls_tasks = (cls_idx or {}).get("tasks") or {}

    moments: List[Dict[str, Any]] = []
    for state in STATES:
        sdir = os.path.join(root, state)
        if not os.path.isdir(sdir):
            continue
        for name in sorted(os.listdir(sdir)):
            tdir = os.path.join(sdir, name)
            if not os.path.isdir(tdir):
                continue
            try:
                st = os.stat(tdir)
            except OSError:
                continue

            files = []
            try:
                files = sorted(os.listdir(tdir))
            except OSError:
                pass

            plan = os.path.join(tdir, "PLAN.md")
            claimed = _read_status_line(plan) if os.path.isfile(plan) else None

            # Task document of record (capped + flagged, never silently cut).
            plan_text = None
            plan_truncated = False
            plan_summary = None
            plan_html = None
            if os.path.isfile(plan):
                try:
                    with open(plan, "r", encoding="utf-8", errors="replace") as fh:
                        plan_text = fh.read(_PLAN_CAP)
                    plan_truncated = os.path.getsize(plan) > _PLAN_CAP
                except OSError:
                    plan_text = None
                plan_summary = _plan_summary(plan_text) if plan_text else None
                plan_html = _md_to_html(plan_text) if plan_text else None

            # Completed tasks carry their output as RESULTS.md (when the agent
            # wrote one); the expander surfaces it so "review the output" does
            # not mean opening the terminal.
            results_path = os.path.join(tdir, "RESULTS.md")
            results_text = None
            results_truncated = False
            results_html = None
            if os.path.isfile(results_path):
                try:
                    with open(results_path, "r", encoding="utf-8", errors="replace") as fh:
                        results_text = fh.read(_RESULTS_CAP)
                    results_truncated = os.path.getsize(results_path) > _RESULTS_CAP
                except OSError:
                    results_text = None
                results_html = _md_to_html(results_text) if results_text else None

            artifacts = [f for f in files
                 if f not in ("PLAN.md", "STATE-SUMMARY.md", "SESSION-FACTS.md")]

            # Layer-1 classification (deterministic batch) + Layer-2 summary.
            classification = None
            cls = cls_tasks.get(name)
            if cls and cls.get("label"):
                classification = {
                    "label": cls["label"],
                    "score": cls.get("score"),
                }
            st_summary = _state_summary(tdir)
            facts = _session_facts(tdir)

            moments.append({
                "id": "task:%s" % name,
                "t": st.st_mtime,
                "computed_at": st.st_mtime,
                "subject": name,
                "actor": "filesystem",
                "kind": "task.%s" % state,
                "payload": {
                    "state": state,
                    "files": files,
                    "n_files": len(files),
                    "has_plan": os.path.isfile(plan),
                    "claimed_status": claimed,
                    "classification": classification,
                    "state_summary": st_summary,
                    "session_facts": facts,
                    "plan_summary": plan_summary,
                    "plan_text": plan_text,
                    "plan_html": plan_html,
                    "plan_truncated": plan_truncated,
                    "results_text": results_text,
                    "results_html": results_html,
                    "results_truncated": results_truncated,
                    "artifacts": artifacts,
                    # A task whose PLAN says one thing while sitting in another
                    # directory is a real inconsistency, not a formatting nit.
                    "status_conflict": bool(claimed) and _conflicts(claimed, state),
                },
                "provenance": {"path": tdir},
            })

    moments.sort(key=lambda m: m["t"] or 0, reverse=True)
    return moments


def _conflicts(claimed: str, state: str) -> bool:
    """Does a PLAN's own status line contradict the directory it sits in?

    Deliberately conservative -- only flags a clear contradiction, and only when
    the claim sits at the START of the status line. Substring matching anywhere
    in the line produces false positives: "in_progress — Phases 1-3 done" is a
    consistent status that happens to contain the word "done", and a detector
    that cries wolf on ten tasks is one nobody reads.
    """
    c = claimed.strip().lower()
    # Strip leading emphasis/markers so the comparison sees the actual claim.
    c = c.lstrip("*_`~ ").strip()

    if state == "completed":
        return c.startswith("in progress") or c.startswith("in_progress") or c.startswith("pending")
    if state == "in_progress":
        return c.startswith("completed") or c.startswith("done") or c.startswith("cancelled")
    if state == "cancelled":
        return c.startswith("completed") or c.startswith("done")
    if state == "new":
        return c.startswith("completed") or c.startswith("cancelled")
    return False


def tasks_view() -> Dict[str, Any]:
    """Assemble the Tasks view: counts per state plus the moment list."""
    root = tasks_dir()
    moments = task_moments()
    cls_idx = _classification_index()

    if not moments:
        return {
            "available": os.path.isdir(root),
            "empty": {
                "reason": "no task directories found at %s" % root,
                "hint": "Set LCP_WORK_TASKS_DIR to the profile's work/tasks tree.",
            },
            "counts": {s: 0 for s in STATES},
            "total": 0,
            "tasks": [],
            "todos": None,
        }

    counts = {s: 0 for s in STATES}
    for m in moments:
        counts[m["payload"]["state"]] = counts.get(m["payload"]["state"], 0) + 1

    conflicts = [m for m in moments if m["payload"]["status_conflict"]]

    # Classification rollup: counts per label sibling to the per-state counts.
    by_label = (cls_idx or {}).get("by_label") or {}
    labels_meta = (cls_idx or {}).get("labels") or {}
    classified = sum(1 for p in (cls_idx or {}).get("tasks", {}).values() if p.get("label"))
    total_tasks = len(moments)

    return {
        "available": True,
        "empty": None,
        "counts": counts,
        "total": total_tasks,
        "classified": classified,
        "by_label": by_label,
        "labels_meta": labels_meta,
        "conflicts": conflicts,
        "assessments": _assessment_feed(),
        "tasks": moments,
        "todos": _todo_view(),
    }


def _todo_view() -> Optional[Dict[str, Any]]:
    """The profile's todo.md, parsed into a renderable overview.

    The ledger file is never rewritten (task-management standard), so this is
    purely a presentation parse: the ``> Updated:`` blockquote run becomes a
    timeline, ``##`` headings become sections, and each ``###`` task line
    becomes a bulleted item. Tables are skipped for the overview — the full
    content stays in the file. Truncation is flagged, never silent.
    """
    p = todos_path()
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None
    m = re.search(r"Updated:\s*([^\n]+)", text)
    out = {
        "path": p,
        "lines": text.count("\n") + 1,
        "updated": m.group(1).strip() if m else None,
        "open_boxes": text.count("- [ ]"),
        "done_boxes": text.count("- [x]"),
        "timeline": [],
        "sections": [],
    }
    section: Optional[Dict[str, Any]] = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("> "):
            if len(out["timeline"]) >= _TODO_TIMELINE_CAP:
                continue
            body = s[2:].strip(" \t")
            # separator: em/en dash (or hyphen) followed by whitespace —
            # the date itself contains hyphens, so `—` without the \s+ would
            # swallow "2026-09-18" down to "2026".
            pm = re.match(r"^Updated[:：]?[\s　]*(.*?)[\s　]*[—–-][\s　]+(.*)$", body)
            if pm and pm.group(1):
                date = pm.group(1).strip()
                piece = pm.group(2).strip()
            else:
                date, piece = None, body
            out["timeline"].append({
                "date": date or "update",
                "text": piece,
                "html": _md_to_html(piece[:_TODO_ENTRY_CAP]),
                "truncated": len(piece) > _TODO_ENTRY_CAP,
            })
        elif s.startswith("## "):
            # NOTE: key is `entries`, not `items` — Jinja's attribute access
            # would resolve `sec.items` to the dict METHOD first.
            raw_title = s[3:].strip()
            # The section markers (🆕🔴🟡📋✅ …) are decorative; the overview
            # shows clean titles. `expanded` remembers the 🔴 marker because
            # "In Progress" is the one group that should open by default.
            expanded = raw_title.startswith("🔴")
            title = raw_title.lstrip("🆕🔴🟡📋✅⚪🟢").strip()
            section = {"title": title, "expanded": expanded, "entries": []}
            out["sections"].append(section)
        elif s.startswith("### ") and section is not None:
            t = s[4:].strip()
            section["entries"].append({
                "text": t,
                "html": _md_to_html(t[:_TODO_ITEM_CAP]),
                "truncated": len(t) > _TODO_ITEM_CAP,
            })
        elif s.startswith("|") and section is not None:
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not cells or all(re.fullmatch(r":?-+:?", c) for c in cells):
                # |---|---| separator: the row just before it was the table
                # header — drop it so it never becomes an overview bullet.
                if section["entries"] and section["entries"][-1].get("table_row"):
                    section["entries"].pop()
                continue
            t = " · ".join(c for c in cells if c)
            if not t:
                continue
            section["entries"].append({
                "text": t,
                "html": _md_to_html(t[:_TODO_ITEM_CAP]),
                "truncated": len(t) > _TODO_ITEM_CAP,
                "table_row": True,
            })
    return out


_TODO_TIMELINE_CAP = 50
_TODO_ENTRY_CAP = 500
_TODO_ITEM_CAP = 180
