"""Work-layer: Tasks and Fleet views.

Same moment model as the Decisions view -- but the sources are the filesystem
(task directories) and the gateway health endpoint, not the board ledger.

A task directory is a small structured record whose *state is its location*
(new -> in_progress -> completed / cancelled). That makes ``moved`` a first-class
moment: the directory's mtime IS the transition time, and the destination IS the
new state. No extra bookkeeping, and it cannot drift from reality because the
location is the state.
"""

import os
import re
from typing import Any, Dict, List, Optional

# Where the task trees live. Overridable so LCP can point at any profile.
DEFAULT_TASKS_DIR = "/root/.hermes/profiles/homelab-expert-l2/work/tasks"

STATES = ("new", "in_progress", "completed", "cancelled")

# How much of a PLAN.md / RESULTS.md to carry into the view. The plan is the
# task's document of record; the panel shows its head as a summary and the
# full (capped) text inside the expander. Truncation is flagged, not silent.
_PLAN_CAP = 8000
_RESULTS_CAP = 2000


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
    """Resolve the task tree root, allowing an env override."""
    return os.environ.get("LCP_WORK_TASKS_DIR", DEFAULT_TASKS_DIR)


def todos_path() -> str:
    """The sibling todo.md for the same profile as the task tree."""
    root = tasks_dir()
    return os.environ.get("LCP_WORK_TODO", os.path.join(os.path.dirname(root), "todo.md"))


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
            if os.path.isfile(plan):
                try:
                    with open(plan, "r", encoding="utf-8", errors="replace") as fh:
                        plan_text = fh.read(_PLAN_CAP)
                    plan_truncated = os.path.getsize(plan) > _PLAN_CAP
                except OSError:
                    plan_text = None
                plan_summary = _plan_summary(plan_text) if plan_text else None

            # Completed tasks carry their output as RESULTS.md (when the agent
            # wrote one); the expander surfaces it so "review the output" does
            # not mean opening the terminal.
            results_path = os.path.join(tdir, "RESULTS.md")
            results_text = None
            results_truncated = False
            if os.path.isfile(results_path):
                try:
                    with open(results_path, "r", encoding="utf-8", errors="replace") as fh:
                        results_text = fh.read(_RESULTS_CAP)
                    results_truncated = os.path.getsize(results_path) > _RESULTS_CAP
                except OSError:
                    results_text = None

            artifacts = [f for f in files if f != "PLAN.md"]

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
                    "plan_summary": plan_summary,
                    "plan_text": plan_text,
                    "plan_truncated": plan_truncated,
                    "results_text": results_text,
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

    return {
        "available": True,
        "empty": None,
        "counts": counts,
        "total": len(moments),
        "conflicts": conflicts,
        "tasks": moments,
        "todos": _todo_summary(),
    }


def _todo_summary() -> Optional[Dict[str, Any]]:
    """Line count + last-updated line from the profile's todo.md."""
    p = todos_path()
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None
    m = re.search(r"Updated:\s*([^\n]+)", text)
    return {
        "path": p,
        "lines": text.count("\n") + 1,
        "updated": m.group(1).strip() if m else None,
        "open_boxes": text.count("- [ ]"),
        "done_boxes": text.count("- [x]"),
    }
