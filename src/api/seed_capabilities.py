"""Model capability storage and the model registry.

Capability scores are produced only by running LiveBench against a model
(direct-to-provider) via ``src.api.benchmark`` — there is no bulk seeding
from public datasets. The model registry maps each logical model name to its
benchmark snapshot key and provider-side aliases.
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


# ── Bulk-seeded LiveBench snapshot (opt-in) ──────────────────────────────────
# Hand-typed 2026-06-25 leaderboard snapshot (source: livebench.ai). Kept as an
# OPT-IN convenience for the setup wizard / a CLI: this is NOT loaded on boot.
# Each entry is tagged with the release it represents so the router can keep
# dated snapshots side-by-side (e.g. deepseek-v4-pro@2026-06-25 vs @2026-08-13).
LIVEBENCH_RELEASE = "2026-06-25"

LIVEBENCH_DATA: dict[str, dict[str, float]] = {
    "deepseek-v4-pro": {
        "reasoning": 82.7, "coding": 70.0, "agentic_coding": 42.6,
        "math": 90.7, "data_analysis": 74.5, "language": 78.1,
        "instruction_following": 62.4, "overall": 71.6,
    },
    "deepseek-v4-flash": {
        "reasoning": 70.6, "coding": 69.2, "agentic_coding": 37.6,
        "math": 79.6, "data_analysis": 68.0, "language": 70.1,
        "instruction_following": 63.1, "overall": 65.5,
    },
    "deepseek-v4-flash-0731": {
        "reasoning": 86.6, "coding": 75.0, "agentic_coding": 46.8,
        "math": 86.8, "data_analysis": 79.3, "language": 79.2,
        "instruction_following": 65.5, "overall": 74.2,
    },
    "claude-fable-5": {
        "reasoning": 89.7, "coding": 86.0, "agentic_coding": 62.2,
        "math": 96.0, "data_analysis": 80.5, "language": 90.7,
        "instruction_following": 75.8, "overall": 83.0,
    },
    "gpt-5.6-sol": {
        "reasoning": 91.7, "coding": 83.9, "agentic_coding": 56.2,
        "math": 96.2, "data_analysis": 79.8, "language": 87.7,
        "instruction_following": 71.8, "overall": 81.0,
    },
    "gpt-5.5-thinking": {
        "reasoning": 89.7, "coding": 82.1, "agentic_coding": 54.0,
        "math": 95.9, "data_analysis": 81.6, "language": 87.4,
        "instruction_following": 70.7, "overall": 80.2,
    },
    "claude-5-opus-thinking": {
        "reasoning": 91.2, "coding": 81.4, "agentic_coding": 65.2,
        "math": 95.7, "data_analysis": 74.6, "language": 88.7,
        "instruction_following": 63.8, "overall": 80.1,
    },
    "smaug-agentic": {
        "reasoning": 90.3, "coding": 82.5, "agentic_coding": 64.6,
        "math": 83.9, "data_analysis": 79.9, "language": 84.4,
        "instruction_following": 71.0, "overall": 79.5,
    },
    "kimi-k3": {
        "reasoning": 90.7, "coding": 81.4, "agentic_coding": 62.2,
        "math": 84.4, "data_analysis": 78.7, "language": 85.5,
        "instruction_following": 71.4, "overall": 79.2,
    },
    "qwen-3.8-max": {
        "reasoning": 88.2, "coding": 72.9, "agentic_coding": 64.6,
        "math": 91.3, "data_analysis": 78.4, "language": 79.7,
        "instruction_following": 74.1, "overall": 78.5,
    },
    "gemini-3.6-flash": {
        "reasoning": 85.1, "coding": 77.9, "agentic_coding": 43.4,
        "math": 86.4, "data_analysis": 63.0, "language": 83.9,
        "instruction_following": 75.4, "overall": 73.6,
    },
    "grok-4.5": {
        "reasoning": 87.2, "coding": 68.6, "agentic_coding": 56.5,
        "math": 90.8, "data_analysis": 73.0, "language": 82.8,
        "instruction_following": 71.5, "overall": 75.8,
    },
}


def seed_livebench(db_path: str, release: Optional[str] = None) -> int:
    """Seed model_capabilities from the bulk LiveBench snapshot (opt-in).

    Rows are tagged ``source="livebench"`` and ``release_label=<release or
    LIVEBENCH_RELEASE>``. Idempotent: replaces prior rows for the same
    (model, task, release).
    """
    from src.api.models import ModelCapability

    label = release or LIVEBENCH_RELEASE
    session = _get_session(db_path)
    now = datetime.now(timezone.utc).isoformat()
    count = 0

    for model, categories in LIVEBENCH_DATA.items():
        for lb_cat, raw in categories.items():
            if lb_cat == "overall":
                continue
            lcp_task = LB_TO_LCP.get(lb_cat, "casual_chat")
            normalized = round(raw / 100.0, 4)

            session.query(ModelCapability).filter_by(
                model=model, task_type=lcp_task, source="livebench",
                release_label=label,
            ).delete()
            session.add(ModelCapability(
                model=model,
                task_type=lcp_task,
                score=normalized,
                source="livebench",
                benchmark_category=lb_cat,
                raw_score=raw,
                release_label=label,
                updated_at=now,
            ))
            count += 1

    session.commit()
    session.close()
    return count


def _get_session(db_path: str):
    from src.api.models import get_engine, get_session

    engine = get_engine(db_path)
    return get_session(engine)


def load_capability_matrix(db_path: str, release: Optional[str] = None) -> dict[str, dict[str, float]]:
    """Load the capability matrix from the DB for the ACTIVE release per model.

    ``release`` is treated as a filter when given (an exact release_label),
    otherwise each model resolves to its own active release (see
    ``load_model_registry``). Legacy rows (release_label NULL) are used as a
    fallback when a model has no explicitly-selected release.

    Returns {task_type: {model_name: score}}.
    Sources priority: gateway_yaml > manual > lcp_benchmark > livebench > arena
    """
    from src.api.models import ModelCapability

    session = _get_session(db_path)
    SOURCE_PRIORITY = {"gateway_yaml": 0, "manual": 1, "lcp_benchmark": 2, "livebench": 3, "arena": 4}

    if release is None:
        registry = load_model_registry(db_path)

    # Collect all rows, then pick the best source per (model, task).
    best: dict[tuple[str, str], dict] = {}

    for row in session.query(ModelCapability).all():
        if release is not None:
            # Exact-release filter.
            if row.release_label != release:
                continue
        else:
            active = registry.get(row.model, {}).get("active_release")
            if active and row.release_label is not None and row.release_label != active:
                # This model has an active release and this row is a different,
                # non-legacy release — skip it.
                continue

        key = (row.task_type, row.model)
        existing = best.get(key)
        new_prio = SOURCE_PRIORITY.get(row.source, 99)
        if existing is not None and SOURCE_PRIORITY.get(existing["source"], 99) <= new_prio:
            continue
        best[key] = {"score": row.score, "source": row.source}

    session.close()

    matrix: dict[str, dict[str, float]] = defaultdict(dict)
    for (task, model), data in best.items():
        matrix[task][model] = data["score"]
    return dict(matrix)


# ── Model registry (alias → logical → benchmark) ────────────────────────────

# Curated default registry. Seeded into the model_registry table on first run;
# subsequent changes are made via the DB (dashboard / API), NOT this dict.
DEFAULT_MODEL_REGISTRY: list[dict] = [
    {
        "logical_name": "deepseek-v4-pro",
        "benchmark_key": "deepseek-v4-pro",
        "aliases": ["deepseek-v4-pro", "deepseek/deepseek-v4-pro"],
        "provider_mappings": {
            "deepseek": "deepseek-v4-pro",
            "opencode": "deepseek-v4-pro",
            "commandcode": "deepseek/deepseek-v4-pro",
        },
    },
    {
        "logical_name": "deepseek-v4-flash",
        "benchmark_key": "deepseek-v4-flash-0731",
        "aliases": ["deepseek-v4-flash", "deepseek/deepseek-v4-flash", "deepseek-v4-flash-0731"],
        "provider_mappings": {
            "deepseek": "deepseek-v4-flash",
            "opencode": "deepseek-v4-flash",
            "commandcode": "deepseek/deepseek-v4-flash",
        },
    },
    {
        "logical_name": "claude-sonnet-5",
        "benchmark_key": "claude-sonnet-5",
        "aliases": ["claude-sonnet-5"],
    },
    {
        "logical_name": "claude-fable-5",
        "benchmark_key": "claude-fable-5",
        "aliases": ["claude-fable-5"],
    },
    {
        "logical_name": "claude-opus-5",
        "benchmark_key": "claude-opus-5",
        "aliases": ["claude-opus-5"],
    },
    {
        "logical_name": "gpt-5.6-sol",
        "benchmark_key": "gpt-5.6-sol",
        "aliases": ["gpt-5.6-sol"],
    },
    {
        "logical_name": "gpt-5.6-terra",
        "benchmark_key": "gpt-5.6-terra",
        "aliases": ["gpt-5.6-terra"],
    },
    {
        "logical_name": "gpt-5.6-luna",
        "benchmark_key": "gpt-5.6-luna",
        "aliases": ["gpt-5.6-luna"],
    },
    {
        "logical_name": "kimi-k3",
        "benchmark_key": "kimi-k3",
        "aliases": ["kimi-k3", "moonshotai/Kimi-K3"],
    },
    {
        "logical_name": "minimax-m3",
        "benchmark_key": "minimax-m3",
        "aliases": ["minimax-m3", "MiniMaxAI/MiniMax-M3"],
    },
    {
        "logical_name": "qwen3.8-max",
        "benchmark_key": "qwen-3.8-max",
        "aliases": ["qwen3.8-max", "Qwen/Qwen3.8-Max"],
    },
    {
        "logical_name": "grok-4.5",
        "benchmark_key": "grok-4.5",
        "aliases": ["grok-4.5", "xai/grok-4.5"],
    },
    {
        "logical_name": "gemini-3.6-flash",
        "benchmark_key": "gemini-3.6-flash",
        "aliases": ["gemini-3.6-flash", "google/gemini-3.6-flash"],
    },
]


def seed_model_registry(db_path: str) -> int:
    """Seed the model_registry table from DEFAULT_MODEL_REGISTRY (insert-only).

    Existing rows are NOT overwritten — the DB is the source of truth once
    seeded. This makes the first-run seed safe against clobbering admin edits.
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
            continue
        session.add(ModelRegistryEntry(
            logical_name=entry["logical_name"],
            benchmark_key=entry["benchmark_key"],
            aliases_json=json.dumps(entry["aliases"]),
            provider_mappings_json=json.dumps(entry.get("provider_mappings", {})),
            updated_at=now,
        ))
        count += 1

    session.commit()
    session.close()
    print(f"Seeded {count} model registry entries")
    return count


def load_model_registry(db_path: str) -> dict[str, dict]:
    """Load the model registry as {logical_name: {...}}.

    Each entry: {benchmark_key, aliases, provider_mappings, active_release}.
    ``active_release`` is None until an admin pins a release; the router then
    falls back to the latest-scored release for that model.
    """
    from src.api.models import ModelRegistryEntry

    session = _get_session(db_path)
    registry: dict[str, dict] = {}

    for row in session.query(ModelRegistryEntry).all():
        try:
            aliases = json.loads(row.aliases_json or "[]")
        except (json.JSONDecodeError, TypeError):
            aliases = []
        try:
            provider_mappings = json.loads(row.provider_mappings_json or "{}")
        except (json.JSONDecodeError, TypeError):
            provider_mappings = {}
        registry[row.logical_name] = {
            "benchmark_key": row.benchmark_key,
            "aliases": aliases,
            "provider_mappings": provider_mappings,
            "active_release": row.active_release,
        }

    session.close()
    return registry

