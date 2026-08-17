"""Model capability storage and the model registry.

Capability scores are produced only by running LiveBench against a model
(direct-to-provider) via ``src.api.benchmark`` — there is no bulk seeding
from public datasets. The model registry maps each logical model name to its
stable benchmark key and provider-side model IDs; the provider keys double as
the model's provider list in the UI.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

# Map LiveBench categories → LCP task types
LB_TO_LCP: dict[str, str] = {
    "reasoning": "reasoning_chain",
    "coding": "code_generation",
    "agentic_coding": "agentic_multi_step",
    "math": "reasoning_chain",
    "data_analysis": "research_deep",
    "language": "casual_chat",
    "instruction_following": "planning",
}


# ── Bulk-seeded LiveBench snapshots (opt-in) ─────────────────────────────────
# Hand-typed leaderboard snapshots (source: livebench.ai). Kept as an OPT-IN
# convenience for the setup wizard / a CLI: this is NOT loaded on boot.
#
# Shape: {logical_model: {release_label: {category: raw_0_100}}}. Only the
# LATEST snapshot is kept per model — ``release_label`` is the model VERSION
# (``2026-08-13`` = DeepSeek V4 Pro 0813, ``2026-07-31`` = Flash 0731) for
# DeepSeek, and the leaderboard date for models without a dated build. The
# leaderboard snapshot date itself lives in ``benchmark_release`` on the
# registry entry, so version and benchmark date stay separate.
LIVEBENCH_RELEASE = "2026-06-25"

LIVEBENCH_DATA: dict[str, dict[str, dict[str, float]]] = {
    "deepseek-v4-pro": {
        "2026-08-13": {
            "reasoning": 85.8, "coding": 77.2, "agentic_coding": 54.9,
            "math": 95.1, "data_analysis": 79.2, "language": 82.1,
            "instruction_following": 67.7, "overall": 77.4,
        },
    },
    "deepseek-v4-flash": {
        "2026-07-31": {
            "reasoning": 86.6, "coding": 75.0, "agentic_coding": 46.8,
            "math": 86.8, "data_analysis": 79.3, "language": 79.2,
            "instruction_following": 65.5, "overall": 74.2,
        },
    },
    "claude-fable-5": {
        "2026-06-25": {
            "reasoning": 89.7, "coding": 86.0, "agentic_coding": 62.2,
            "math": 96.0, "data_analysis": 80.5, "language": 90.7,
            "instruction_following": 75.8, "overall": 83.0,
        },
    },
    "claude-sonnet-5": {
        "2026-06-25": {
            "reasoning": 88.7, "coding": 80.7, "agentic_coding": 59.4,
            "math": 92.9, "data_analysis": 71.7, "language": 75.0,
            "instruction_following": 63.9, "overall": 76.0,
        },
    },
    "claude-opus-5": {
        "2026-06-25": {
            "reasoning": 91.2, "coding": 81.4, "agentic_coding": 65.2,
            "math": 95.7, "data_analysis": 74.6, "language": 88.7,
            "instruction_following": 63.8, "overall": 80.1,
        },
    },
    "gpt-5.6-sol": {
        "2026-06-25": {
            "reasoning": 91.7, "coding": 83.9, "agentic_coding": 56.2,
            "math": 96.2, "data_analysis": 79.8, "language": 87.7,
            "instruction_following": 71.8, "overall": 81.0,
        },
    },
    "gpt-5.5-thinking": {
        "2026-06-25": {
            "reasoning": 89.7, "coding": 82.1, "agentic_coding": 54.0,
            "math": 95.9, "data_analysis": 81.6, "language": 87.4,
            "instruction_following": 70.7, "overall": 80.2,
        },
    },
    "claude-5-opus-thinking": {
        "2026-06-25": {
            "reasoning": 91.2, "coding": 81.4, "agentic_coding": 65.2,
            "math": 95.7, "data_analysis": 74.6, "language": 88.7,
            "instruction_following": 63.8, "overall": 80.1,
        },
    },
    "smaug-agentic": {
        "2026-06-25": {
            "reasoning": 90.3, "coding": 82.5, "agentic_coding": 64.6,
            "math": 83.9, "data_analysis": 79.9, "language": 84.4,
            "instruction_following": 71.0, "overall": 79.5,
        },
    },
    "kimi-k3": {
        "2026-06-25": {
            "reasoning": 90.7, "coding": 81.4, "agentic_coding": 62.2,
            "math": 84.4, "data_analysis": 78.7, "language": 85.5,
            "instruction_following": 71.4, "overall": 79.2,
        },
    },
    "qwen-3.8-max": {
        "2026-06-25": {
            "reasoning": 88.2, "coding": 72.9, "agentic_coding": 64.6,
            "math": 91.3, "data_analysis": 78.4, "language": 79.7,
            "instruction_following": 74.1, "overall": 78.5,
        },
    },
    "gemini-3.6-flash": {
        "2026-06-25": {
            "reasoning": 85.1, "coding": 77.9, "agentic_coding": 43.4,
            "math": 86.4, "data_analysis": 63.0, "language": 83.9,
            "instruction_following": 75.4, "overall": 73.6,
        },
    },
    "grok-4.5": {
        "2026-06-25": {
            "reasoning": 87.2, "coding": 68.6, "agentic_coding": 56.5,
            "math": 90.8, "data_analysis": 73.0, "language": 82.8,
            "instruction_following": 71.5, "overall": 75.8,
        },
    },
}


def seed_livebench(db_path: str, release: Optional[str] = None) -> int:
    """Seed model_capabilities from the bulk LiveBench snapshots (opt-in).

    Rows are tagged ``source="livebench"`` and ``release_label=<that
    release>``. When ``release`` is given, only that release is seeded;
    otherwise every release in ``LIVEBENCH_DATA`` is seeded.

    Idempotent: replaces prior rows for the same (model, task, source,
    release), and removes rows seeded under the retired dated snapshot key
    (``deepseek-v4-flash-0731``) that predates release-aware seeding.
    """
    from src.api.models import ModelCapability

    session = _get_session(db_path)
    now = datetime.now(timezone.utc).isoformat()
    count = 0

    for model, releases in LIVEBENCH_DATA.items():
        for rel, categories in releases.items():
            if release is not None and rel != release:
                continue
            for lb_cat, raw in categories.items():
                if lb_cat == "overall":
                    continue
                lcp_task = LB_TO_LCP.get(lb_cat, "casual_chat")
                normalized = round(raw / 100.0, 4)

                session.query(ModelCapability).filter_by(
                    model=model, task_type=lcp_task, source="livebench",
                    release_label=rel,
                ).delete()
                session.add(ModelCapability(
                    model=model,
                    task_type=lcp_task,
                    score=normalized,
                    source="livebench",
                    benchmark_category=lb_cat,
                    raw_score=raw,
                    release_label=rel,
                    updated_at=now,
                ))
                count += 1

    # Migration: drop rows keyed to the old dated snapshot name. They now live
    # under the logical name with release_label "2026-07-31".
    retired_snapshot_keys = ("deepseek-v4-flash-0731",)
    session.query(ModelCapability).filter(
        ModelCapability.model.in_(retired_snapshot_keys),
        ModelCapability.source == "livebench",
    ).delete(synchronize_session=False)

    # Migration: drop LEGACY unversioned livebench rows (release_label NULL)
    # for every seeded model. They predate release-aware seeding and would
    # otherwise tie-break against the freshly-tagged rows (same source
    # priority, arbitrary row order).
    seeded_models = list(LIVEBENCH_DATA.keys())
    session.query(ModelCapability).filter(
        ModelCapability.model.in_(seeded_models),
        ModelCapability.source == "livebench",
        ModelCapability.release_label.is_(None),
    ).delete(synchronize_session=False)

    # Migration: drop dated snapshots that are no longer the latest release.
    # We keep ONLY the newest snapshot per model — historical rows (e.g.
    # deepseek-v4-pro@2026-06-25 now superseded by @2026-08-13) are removed.
    for model, releases in LIVEBENCH_DATA.items():
        keep = list(releases.keys())
        session.query(ModelCapability).filter(
            ModelCapability.model == model,
            ModelCapability.source == "livebench",
            ModelCapability.release_label.notin_(keep),
        ).delete(synchronize_session=False)

    session.commit()
    session.close()
    return count


def seed_livebench_tasks(db_path: str, release: str = LIVEBENCH_RELEASE) -> int:
    """Seed model_capability_subtasks from the LiveBench 2026-06-25 table.

    Stores per-subtask scores (e.g. theory_of_mind, zebra_puzzle) under
    ``source="livebench"`` + ``release_label=<release>``, keyed by each
    model's ``benchmark_key`` (logical). Idempotent per (model, category,
    task, release).
    """
    from src.api.models import ModelCapabilitySubtask
    from .livebench_tasks import LIVEBENCH_TASKS

    session = _get_session(db_path)
    now = datetime.now(timezone.utc).isoformat()
    count = 0

    for model, categories in LIVEBENCH_TASKS.items():
        for category, tasks in categories.items():
            for task, raw in tasks.items():
                normalized = round(raw / 100.0, 4)
                session.query(ModelCapabilitySubtask).filter_by(
                    model=model, category=category, task=task,
                    source="livebench", release_label=release,
                ).delete()
                session.add(ModelCapabilitySubtask(
                    model=model,
                    category=category,
                    task=task,
                    score=normalized,
                    source="livebench",
                    raw_score=raw,
                    release_label=release,
                    updated_at=now,
                ))
                count += 1

    session.commit()
    session.close()
    return count


def _get_session(db_path: str):
    from src.api.models import get_engine, get_session, Base

    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return get_session(engine)


def _default_db_path() -> str:
    """Resolve the DB path the same way the gateway does (COST_DB or data/costs.db)."""
    import os
    return os.environ.get("COST_DB", "data/costs.db")


def main() -> None:
    """CLI: seed the model registry + bulk LiveBench snapshots.

    Usage:
        python -m src.api.seed_capabilities                  # seed registry + all releases
        python -m src.api.seed_capabilities --release 2026-06-25
        python -m src.api.seed_capabilities --registry-only
        python -m src.api.seed_capabilities --livebench-only --db /app/data/costs.db
        python -m src.api.seed_capabilities --sync           # migrate identity → release split
    """
    import argparse

    parser = argparse.ArgumentParser(description="Seed LCP model data")
    parser.add_argument("--db", default=_default_db_path(), help="path to SQLite DB")
    parser.add_argument("--release", default=None, help="release label for LiveBench snapshot (default: every release)")
    parser.add_argument("--registry-only", action="store_true", help="only seed the model registry")
    parser.add_argument("--livebench-only", action="store_true", help="only seed the LiveBench snapshot")
    parser.add_argument("--sync", action="store_true", help="re-apply curated registry defaults to existing rows (migrate identity → release split)")
    args = parser.parse_args()

    db_path = args.db

    if args.livebench_only:
        n = seed_livebench(db_path, release=args.release)
        t = seed_livebench_tasks(db_path, release=args.release or LIVEBENCH_RELEASE)
        print(f"Seeded {n} LiveBench capability rows + {t} subtask rows (release={args.release or 'all'})")
        return

    if args.registry_only:
        n = seed_model_registry(db_path, sync=args.sync)
        print(f"Seeded {n} registry entries")
        return

    r = seed_model_registry(db_path, sync=args.sync)
    l = seed_livebench(db_path, release=args.release)
    t = seed_livebench_tasks(db_path, release=args.release or LIVEBENCH_RELEASE)
    print(f"Seeded {r} registry entries + {l} LiveBench capability rows + {t} subtask rows (release={args.release or 'all'})")


def resolve_active_rows(rows, registry: dict, release: Optional[str] = None):
    """Filter capability rows to each model's active release.

    Shared by ``load_capability_matrix`` (router) and the capability API
    (UI) so both always agree on which release's scores are live:

      * ``release`` given → exact ``release_label`` filter (+ legacy NULL rows);
      * ``active_release`` pinned on the model → that release (+ legacy);
      * otherwise → newest available ``release_label`` (+ legacy).

    Legacy rows (``release_label`` NULL) are always kept as a fallback so
    models with only unversioned scores still participate.
    """
    if release is not None:
        return [r for r in rows if r.release_label == release or r.release_label is None]

    # Reverse index: capability row model key → registry entry. Rows are keyed
    # by the model's benchmark_key (and sometimes by a provider-side model ID),
    # so map every reachable spelling back to its entry.
    by_key: dict[str, dict] = {}
    for entry in registry.values():
        keys = [entry.get("benchmark_key"), *((entry.get("provider_mappings") or {}).values())]
        for key in keys:
            if key:
                by_key.setdefault(key.lower(), entry)

    by_model: dict[str, dict[Optional[str], list]] = {}
    for row in rows:
        by_model.setdefault(row.model, {}).setdefault(row.release_label, []).append(row)

    out: list = []
    for model, buckets in by_model.items():
        entry = by_key.get(model.lower(), registry.get(model, {}))
        active = entry.get("active_release")
        dated = [rel for rel in buckets if rel is not None]
        if active in buckets:
            selected = active
        elif active in (None, "latest", "") or not dated:
            selected = max(dated) if dated else None
        else:
            # Pinned release has no rows yet — fall back to the newest anyway.
            selected = max(dated)
        out.extend(buckets.get(selected, []))
        out.extend(buckets.get(None, []))
    return out


def effective_releases(rows, registry: dict) -> dict[str, str]:
    """Return {model_key: release_label} for the release actually in effect.

    Mirrors ``resolve_active_rows``: the pinned ``active_release`` when rows
    exist for it, otherwise the newest available ``release_label``. Models
    with only legacy (NULL) rows are omitted.
    """
    by_key: dict[str, dict] = {}
    for entry in registry.values():
        keys = [entry.get("benchmark_key"), *((entry.get("provider_mappings") or {}).values())]
        for key in keys:
            if key:
                by_key.setdefault(key.lower(), entry)

    by_model: dict[str, set] = {}
    for row in rows:
        if row.release_label:
            by_model.setdefault(row.model, set()).add(row.release_label)

    out: dict[str, str] = {}
    for model, releases in by_model.items():
        if not releases:
            continue
        entry = by_key.get(model.lower(), registry.get(model, {}))
        active = entry.get("active_release")
        if active in releases:
            out[model] = active
        else:
            out[model] = max(releases)
    return out


def load_capability_matrix(db_path: str, release: Optional[str] = None) -> dict[str, dict[str, float]]:
    """Load the capability matrix from the DB for the ACTIVE release per model.

    ``release`` is treated as an exact ``release_label`` filter when given;
    otherwise each model resolves to its own release via
    ``resolve_active_rows`` (``active_release`` pin → newest release).

    Returns {task_type: {model_name: score}}.
    Sources priority: gateway_yaml > manual > lcp_benchmark > livebench > arena
    """
    from src.api.models import ModelCapability

    session = _get_session(db_path)
    SOURCE_PRIORITY = {"gateway_yaml": 0, "manual": 1, "lcp_benchmark": 2, "livebench": 3, "arena": 4}

    rows = session.query(ModelCapability).all()
    session.close()

    registry = load_model_registry(db_path) if release is None else {}
    candidate_rows = resolve_active_rows(rows, registry, release)

    # Pick the best source per (model, task).
    best: dict[tuple[str, str], dict] = {}
    for row in candidate_rows:
        key = (row.task_type, row.model)
        existing = best.get(key)
        new_prio = SOURCE_PRIORITY.get(row.source, 99)
        if existing is not None and SOURCE_PRIORITY.get(existing["source"], 99) <= new_prio:
            continue
        best[key] = {"score": row.score, "source": row.source}

    matrix: dict[str, dict[str, float]] = defaultdict(dict)
    for (task, model), data in best.items():
        matrix[task][model] = data["score"]
    return dict(matrix)


# ── Model registry (logical → benchmark → providers) ────────────────────────

# Curated default registry. Seeded into the model_registry table on first run;
# subsequent changes are made via the DB (dashboard / API), NOT this dict.
DEFAULT_MODEL_REGISTRY: list[dict] = [
    {
        "logical_name": "deepseek-v4-pro",
        "benchmark_key": "deepseek-v4-pro",
        "provider_mappings": {
            "deepseek": "deepseek-v4-pro",
            "opencode": "deepseek-v4-pro",
            "commandcode": "deepseek/deepseek-v4-pro",
        },
        "active_release": "2026-08-13",
        "benchmark_release": "2026-06-25",
    },
    {
        "logical_name": "deepseek-v4-flash",
        "benchmark_key": "deepseek-v4-flash",
        "provider_mappings": {
            "deepseek": "deepseek-v4-flash",
            "opencode": "deepseek-v4-flash",
            "commandcode": "deepseek/deepseek-v4-flash",
        },
        "active_release": "2026-07-31",
        "benchmark_release": "2026-06-25",
    },
    {
        "logical_name": "claude-sonnet-5",
        "benchmark_key": "claude-sonnet-5",
        "provider_mappings": {"commandcode": "claude-sonnet-5"},
    },
    {
        "logical_name": "claude-fable-5",
        "benchmark_key": "claude-fable-5",
        "provider_mappings": {"commandcode": "claude-fable-5"},
    },
    {
        "logical_name": "claude-opus-5",
        "benchmark_key": "claude-opus-5",
        "provider_mappings": {"commandcode": "claude-opus-5"},
    },
    {
        "logical_name": "gpt-5.6-sol",
        "benchmark_key": "gpt-5.6-sol",
        "provider_mappings": {},
    },
    {
        "logical_name": "gpt-5.6-terra",
        "benchmark_key": "gpt-5.6-terra",
        "provider_mappings": {"commandcode": "gpt-5.6-terra"},
    },
    {
        "logical_name": "gpt-5.6-luna",
        "benchmark_key": "gpt-5.6-luna",
        "provider_mappings": {"commandcode": "gpt-5.6-luna"},
    },
    {
        "logical_name": "kimi-k3",
        "benchmark_key": "kimi-k3",
        "provider_mappings": {"commandcode": "moonshotai/Kimi-K3"},
    },
    {
        "logical_name": "minimax-m3",
        "benchmark_key": "minimax-m3",
        "provider_mappings": {"commandcode": "MiniMaxAI/MiniMax-M3"},
    },
    {
        "logical_name": "qwen3.8-max",
        "benchmark_key": "qwen-3.8-max",
        "provider_mappings": {"commandcode": "Qwen/Qwen3.8-Max"},
    },
    {
        "logical_name": "grok-4.5",
        "benchmark_key": "grok-4.5",
        "provider_mappings": {},
    },
    {
        "logical_name": "gemini-3.6-flash",
        "benchmark_key": "gemini-3.6-flash",
        "provider_mappings": {},
    },
]


def seed_model_registry(db_path: str, sync: bool = False) -> int:
    """Seed the model_registry table from DEFAULT_MODEL_REGISTRY.

    Insert-only by default: existing rows are NOT overwritten — the DB is the
    source of truth once seeded. With ``sync=True``, curated defaults are
    re-applied to existing rows (provider_mappings, benchmark_key,
    active_release, benchmark_release) so a code-level identity→release
    migration can be pushed out without wiping admin edits to unmanaged
    columns.
    """
    from src.api.models import ModelRegistryEntry

    session = _get_session(db_path)
    now = datetime.now(timezone.utc).isoformat()
    count = 0

    for entry in DEFAULT_MODEL_REGISTRY:
        existing = session.query(ModelRegistryEntry).filter_by(
            logical_name=entry["logical_name"]
        ).first()
        if existing:
            if sync:
                existing.benchmark_key = entry["benchmark_key"]
                existing.provider_mappings_json = json.dumps(entry.get("provider_mappings", {}))
                existing.active_release = entry.get("active_release")
                existing.benchmark_release = entry.get("benchmark_release")
                existing.quantization = entry.get("quantization")
                existing.updated_at = now
                count += 1
            continue
        session.add(ModelRegistryEntry(
            logical_name=entry["logical_name"],
            benchmark_key=entry["benchmark_key"],
            provider_mappings_json=json.dumps(entry.get("provider_mappings", {})),
            active_release=entry.get("active_release"),
            benchmark_release=entry.get("benchmark_release"),
            quantization=entry.get("quantization"),
            updated_at=now,
        ))
        count += 1

    session.commit()
    session.close()
    print(f"Seeded {count} model registry entries")
    return count


def load_model_registry(db_path: str) -> dict[str, dict]:
    """Load the model registry as {logical_name: {...}}.

    Each entry: {benchmark_key, provider_mappings, active_release,
    benchmark_release, quantization}. ``active_release`` is the CURRENT model
    version (e.g. ``2026-08-13``); ``benchmark_release`` is the leaderboard
    snapshot date the scores came from (e.g. ``2026-06-25``).
    """
    from src.api.models import ModelRegistryEntry

    session = _get_session(db_path)
    registry: dict[str, dict] = {}

    for row in session.query(ModelRegistryEntry).all():
        try:
            provider_mappings = json.loads(row.provider_mappings_json or "{}")
        except (json.JSONDecodeError, TypeError):
            provider_mappings = {}
        registry[row.logical_name] = {
            "benchmark_key": row.benchmark_key,
            "provider_mappings": provider_mappings,
            "active_release": row.active_release,
            "benchmark_release": row.benchmark_release,
            "quantization": row.quantization,
        }

    session.close()
    return registry


if __name__ == "__main__":
    main()

