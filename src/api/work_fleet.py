"""Work-layer: Fleet view.

Which profile is served by which provider/model chain, and what has failed over.

The interesting property here is that the failover list and the chain list are
the *same* data seen two ways: the chain is what we asked for, the failovers are
what we actually got. Showing only the chain makes a fleet look static and
healthy; showing only failovers hides the configured intent. Both, side by side.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

DEFAULT_BASE = "http://127.0.0.1:8734"


def base_url() -> str:
    """LCP's own base URL, for reading its own API."""
    return os.environ.get("LCP_SELF_BASE", DEFAULT_BASE).rstrip("/")


def _get(path: str, timeout: float = 10.0) -> Optional[Any]:
    """GET a JSON endpoint on ourselves. Returns None on any failure."""
    try:
        with urllib.request.urlopen(base_url() + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError):
        return None


def failover_moments(failovers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Failovers as moments.

    ``computed_at`` is when we rendered; ``t`` is when the failover happened.
    For a failover those are genuinely different questions and the second one is
    the one that matters -- "this fleet failed over 40 minutes ago" is the fact.
    """
    moments = []
    for f in failovers or []:
        ts = f.get("timestamp")
        epoch = None
        if ts:
            try:
                from datetime import datetime
                epoch = datetime.fromisoformat(ts).timestamp()
            except (TypeError, ValueError):
                epoch = None
        moments.append({
            "id": "failover:%s" % f.get("id"),
            "t": epoch,
            "computed_at": None,
            "subject": f.get("profile") or "?",
            "actor": f.get("from_provider") or "?",
            "kind": "failover",
            "payload": {
                "from": f.get("from_provider"),
                "to": f.get("to_provider"),
                "reason": f.get("reason"),
                "error": f.get("error_message"),
                # A failover that lands on the same provider is a retry, not a
                # failover. Counting them together overstates redundancy.
                "same_provider": f.get("from_provider") == f.get("to_provider"),
                "raw_ts": ts,
            },
            "provenance": {"id": f.get("id")},
        })
    moments.sort(key=lambda m: m["t"] or 0, reverse=True)
    return moments


def fleet_view() -> Dict[str, Any]:
    """Assemble the Fleet view from LCP's own provider/profile/routing APIs."""
    health = _get("/api/providers/health")
    failovers = _get("/api/providers/failovers")
    profiles = _get("/api/profiles")
    routing = _get("/api/routing/status")

    if health is None and profiles is None:
        return {
            "available": False,
            "empty": {
                "reason": "could not read LCP's own provider/profile APIs",
                "hint": "Check LCP_SELF_BASE (currently %s)." % base_url(),
            },
            "summary": None, "profiles": [], "failover_moments": [],
            "failover_stats": None, "routing": None,
        }

    summary = (health or {}).get("summary") or {}
    provs = (health or {}).get("providers") or {}

    # Providers that have ever failed, worst first -- the ones worth looking at.
    flaky = sorted(
        (p for p in provs.values() if (p.get("failures") or 0) > 0
         or (p.get("status") or "healthy") != "healthy"),
        key=lambda p: (-(p.get("failures") or 0), p.get("provider") or ""),
    )

    fms = failover_moments((failovers or {}).get("failovers") or [])
    retries = [m for m in fms if m["payload"]["same_provider"]]
    real = [m for m in fms if not m["payload"]["same_provider"]]

    # Group failovers by the profile they hit.
    by_profile: Dict[str, int] = {}
    by_reason: Dict[str, int] = {}
    for m in fms:
        by_profile[m["subject"]] = by_profile.get(m["subject"], 0) + 1
        r = m["payload"]["reason"] or "unknown"
        by_reason[r] = by_reason.get(r, 0) + 1

    prof_list = []
    for name, p in sorted(((profiles or {}).get("profiles") or {}).items()):
        chain = p.get("chain") or []
        prof_list.append({
            "name": name,
            "chain": chain,
            "chain_len": len(chain),
            "url": p.get("url"),
            "forbidden": p.get("forbidden") or [],
            "auth_required": bool(p.get("auth_required")),
            # A single-entry chain has no fallback: the third column of a
            # redundancy table that reads 1 is the finding, not the length.
            "no_fallback": len(chain) <= 1,
            "failovers": by_profile.get(name, 0),
        })

    return {
        "available": True,
        "empty": None,
        "summary": summary,
        "flaky": flaky,
        "profiles": prof_list,
        "failover_moments": fms,
        "failover_stats": {
            "total": len(fms),
            "real": len(real),
            "retries": len(retries),
            "by_profile": by_profile,
            "by_reason": by_reason,
        },
        "routing": routing,
    }
