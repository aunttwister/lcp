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
import re
import time
import uuid
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


# ── Cron op spool (the only writable surface of the work layer) ─────────────
#
# The Cron page cannot see /root/.hermes/profiles, so a mutation cannot be
# applied from inside the container. Instead the panel spools an *intent*: a
# validated op file in the dedicated writable mount, which a host-side
# executor (work-layers-cron-executor.timer) applies through the real hermes
# CLI and then refreshes the snapshot. LCP never edits jobs.json directly.

DEFAULT_OPS_DIR = "/app/cron-ops"

ACTIONS = ("create", "edit", "pause", "resume", "run", "remove")
_SCHEDULE_RE = re.compile(r"^[A-Za-z0-9 ,*/:+-]{1,64}$")
_SCRIPT_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
_BASENAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
_DELIVERS = ("origin", "local", "all")


def cron_ops_dir() -> str:
    """Resolve the op spool root (env override; tests point at tmp dirs)."""
    return os.environ.get("LCP_CRON_OPS_DIR", DEFAULT_OPS_DIR)


def _ops_subdir(name: str) -> str:
    d = os.path.join(cron_ops_dir(), name)
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return d


def _profile_names() -> set:
    v = cron_view()
    return {p["profile"] for p in v["profiles"]} if v["available"] else set()


def _job_ids(profile: str) -> set:
    v = cron_view()
    if not v["available"]:
        return set()
    for p in v["profiles"]:
        if p["profile"] == profile:
            return {j["id"] for j in (p.get("jobs") or [])}
    return set()


def _validate_op(payload: dict) -> dict:
    """Validate a spool op against the current snapshot. Raises ValueError."""
    action = (payload.get("action") or "").strip().lower()
    if action not in ACTIONS:
        raise ValueError("action must be one of %s" % ", ".join(ACTIONS))
    profile = (payload.get("profile") or "").strip()
    if not profile or len(profile) > 64:
        raise ValueError("profile is required")
    known = _profile_names()
    if known and profile not in known:
        raise ValueError("unknown profile %r (known: %s)"
                         % (profile, ", ".join(sorted(known)) or "none"))

    op = {"action": action, "profile": profile}
    if action == "create":
        name = (payload.get("name") or "").strip()
        if not name or len(name) > 80 or "\n" in name:
            raise ValueError("name is required (<= 80 chars, one line)")
        schedule = (payload.get("schedule") or "").strip()
        if not schedule or len(schedule) > 64:
            raise ValueError("schedule is required (e.g. '30m', '0 9 * * *')")
        prompt = (payload.get("prompt") or "").strip()
        script = (payload.get("script") or "").strip()
        if not prompt and not script:
            raise ValueError("provide a prompt or a script for the job")
        if prompt and len(prompt) > 8000:
            raise ValueError("prompt too long (max 8000 chars)")
        if script and not _SCRIPT_RE.fullmatch(script):
            raise ValueError("script must be a bare filename (no slashes)")
        if payload.get("no_agent") and not script:
            raise ValueError("no-agent jobs need a script")
        op.update(name=name, schedule=schedule,
                  prompt=prompt or None, script=script or None,
                  no_agent=bool(payload.get("no_agent")))
        deliver = (payload.get("deliver") or "").strip() or "origin"
        if deliver not in _DELIVERS and not deliver.startswith(
                ("telegram:", "discord:", "sms:", "signal:")):
            raise ValueError("deliver must be origin|local|all or a platform:… id")
        op["deliver"] = deliver
        repeat = payload.get("repeat")
        if repeat is not None:
            try:
                repeat = int(repeat)
            except (TypeError, ValueError):
                raise ValueError("repeat must be an integer")  # noqa: B904
            if repeat < 0 or repeat > 1000:
                raise ValueError("repeat must be 0..1000")
            op["repeat"] = repeat
        skill = (payload.get("skill") or "").strip()
        if skill and (len(skill) > 64 or not _BASENAME_RE.fullmatch(skill)):
            raise ValueError("skill must be a bare skill name")
        if skill:
            op["skill"] = skill
        workdir = (payload.get("workdir") or "").strip()
        if workdir and not workdir.startswith("/your/data/"):
            raise ValueError("workdir must live under /your/data/")
        if workdir:
            op["workdir"] = workdir
        return op

    # edit / pause / resume / run / remove all target an existing job
    job_id = (payload.get("job_id") or "").strip()
    if not job_id or len(job_id) > 20:
        raise ValueError("job_id is required")
    known_ids = _job_ids(profile)
    if known_ids and job_id not in known_ids:
        raise ValueError("job %s not found under profile %s (snapshot may be "
                         "stale — refresh)" % (job_id, profile))
    op["job_id"] = job_id
    if action == "edit":
        new = {}
        # The UI sends changed fields under `fields`; accept the top-level
        # form too (a caller that spreads the whole payload).
        fields = payload.get("fields")
        if not isinstance(fields, dict):
            fields = payload
        name = (fields.get("name") or "").strip()
        if name:
            if "\n" in name or len(name) > 80:
                raise ValueError("name must be one line, <= 80 chars")
            new["name"] = name
        schedule = (fields.get("schedule") or "").strip()
        if schedule:
            if not _SCHEDULE_RE.fullmatch(schedule):
                raise ValueError("schedule looks invalid")
            new["schedule"] = schedule
        prompt = (fields.get("prompt") or "").strip()
        if prompt:
            if len(prompt) > 8000:
                raise ValueError("prompt too long (max 8000 chars)")
            new["prompt"] = prompt
        script = (fields.get("script") or "").strip()
        if script:
            if not _SCRIPT_RE.fullmatch(script):
                raise ValueError("script must be a bare filename")
            new["script"] = script
        deliver = (fields.get("deliver") or "").strip()
        if deliver:
            if deliver not in _DELIVERS and not deliver.startswith(
                    ("telegram:", "discord:", "sms:", "signal:")):
                raise ValueError("deliver must be origin|local|all or platform:…")
            new["deliver"] = deliver
        if fields.get("no_agent") is not None:
            new["no_agent"] = bool(fields.get("no_agent"))
        if not new:
            raise ValueError("edit needs at least one field to change")
        op["fields"] = new
    return op


def submit_cron_op(payload: dict) -> dict:
    """Validate + spool an op. Returns the op record (id, …)."""
    op = _validate_op(payload)
    op_id = uuid.uuid4().hex[:12]
    op["id"] = op_id
    op["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    op["status"] = "pending"
    path = os.path.join(_ops_subdir("ops"), op_id + ".json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(op, fh, indent=2)
    os.replace(tmp, path)
    return op


def cron_ops_view(limit: int = 20) -> dict:
    """Pending + recent executed ops for the panel's ops section."""
    ops_dir = _ops_subdir("ops")
    done_dir = _ops_subdir("done")
    items = []
    for name in sorted(os.listdir(ops_dir), reverse=True):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(ops_dir, name), encoding="utf-8") as fh:
                items.append(json.load(fh))
        except (OSError, ValueError):
            continue
    done = []
    for name in sorted(os.listdir(done_dir), reverse=True)[:limit]:
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(done_dir, name), encoding="utf-8") as fh:
                done.append(json.load(fh))
        except (OSError, ValueError):
            continue
    return {"pending": items, "done": done}