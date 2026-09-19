"""Work-layer: Cron visibility view.

A read-only window into every Hermes gateway profile's scheduled jobs. The
source is the host-side snapshot ``<tasks_root>/.work-layers/cron-jobs.json``
written by the ``work_layers.collect_cron`` batch (systemd timer, every 15
minutes) -- the LCP container cannot see ``/root/.hermes/profiles``, so the
snapshot is the single source of truth for "what is scheduled right now".

The view is deliberately defensive the same way the rest of the module is:
a missing or corrupt snapshot degrades to ``available: False`` with a hint
instead of an empty-looking page. Nothing here calls a model or writes back
to Hermes; pause/resume/run stay CLI-side by design.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

DEFAULT_TASKS_DIR = "/root/.hermes/profiles/homelab-expert-l2/work/tasks"

# Statuses that mean "this job is not actually doing its thing right now".
_WARN_STATUSES = {"disabled", "paused"}


def tasks_dir() -> str:
    """Resolve the task tree root, allowing an env override."""
    return os.environ.get("LCP_WORK_TASKS_DIR", DEFAULT_TASKS_DIR)


def _snapshot_path() -> str:
    return os.path.join(tasks_dir(), ".work-layers", "cron-jobs.json")


def _parse_iso(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    try:
        s = str(ts)
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _relative(ts: Any, now: float) -> Optional[str]:
    """Human relative time for a job timestamp (None if unparseable)."""
    dt = _parse_iso(ts)
    if dt is None:
        return None
    try:
        secs = int(now - dt.timestamp())
    except (OverflowError, OSError, ValueError):
        return None
    future = secs < 0
    secs = abs(secs)
    if secs < 60:
        unit = "just now" if not future else "under 1m"
    elif secs < 3600:
        unit = "%dm" % (secs // 60)
    elif secs < 86400:
        unit = "%dh" % (secs // 3600)
    else:
        unit = "%dd" % (secs // 86400)
    return ("in " if future else "") + unit


def _decorate_job(job: Dict[str, Any], now: float) -> Dict[str, Any]:
    out = dict(job)
    out["next_rel"] = _relative(job.get("next_run_at"), now)
    out["last_rel"] = _relative(job.get("last_run_at"), now)
    lex = job.get("last_execution") or {}
    out["last_execution_status"] = lex.get("status")
    out["last_execution_finished_rel"] = _relative(lex.get("finished_at"), now)
    if job["status"] in _WARN_STATUSES:
        out["warn"] = job["status"]
    elif job.get("last_error") or job.get("last_delivery_error"):
        out["warn"] = "error"
    elif lex.get("status") == "failed":
        out["warn"] = "error"
    else:
        out["warn"] = None
    return out


def cron_view() -> Dict[str, Any]:
    """The Cron page payload: snapshot + decorated rows + module liveness."""
    now = time.time()
    try:
        with open(_snapshot_path(), "r", encoding="utf-8") as fh:
            snap = json.load(fh)
    except (OSError, ValueError) as exc:
        return {
            "available": False,
            "hint": ("cron snapshot missing; the host-side collector has not "
                     "run yet (systemd work-layers-cron.timer)"),
            "error": str(exc),
            "generated_at": None,
            "generated_rel": None,
            "counts": {"total": 0, "active": 0, "paused": 0, "disabled": 0, "error": 0},
            "profiles": [],
        }

    counts = snap.get("counts", {})
    profiles = []
    for p in snap.get("profiles", []):
        prof = {
            "profile": p.get("profile"),
            "legacy": bool(p.get("legacy")),
            "error": p.get("error"),
            "jobs": [_decorate_job(j, now) for j in (p.get("jobs") or [])],
        }
        prof["counts"] = {
            "total": len(prof["jobs"]),
            "active": sum(1 for j in prof["jobs"] if j["status"] == "active"),
            "warned": sum(1 for j in prof["jobs"] if j.get("warn")),
        }
        profiles.append(prof)

    return {
        "available": True,
        "hint": None,
        "error": None,
        "generated_at": snap.get("generated_at"),
        "generated_ts": snap.get("generated_at_ts"),
        "generated_rel": _relative(snap.get("generated_at"), now),
        "counts": {
            "total": counts.get("total", 0),
            "active": counts.get("active", 0),
            "paused": counts.get("paused", 0),
            "disabled": counts.get("disabled", 0),
            "error": counts.get("error", 0),
        },
        "profiles": profiles,
    }