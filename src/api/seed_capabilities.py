"""Seed the model_capabilities table from public benchmark data.

Sources (in priority order):
  1. LiveBench 2026-06-25 leaderboard (livebench.ai) — objective per-category scores
  2. LMSYS Chatbot Arena 55k (HF) — human preference Elo for legacy models
  3. gateway.yaml dynamic_routing.capability_overrides — admin hand-tuned

Usage:
    python -m src.api.seed_capabilities              # seed from LiveBench + Arena
    python -m src.api.seed_capabilities --source livebench  # LiveBench only
    python -m src.api.seed_capabilities --source arena      # Arena only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

# ── LiveBench 2026-06-25 per-category scores (source: livebench.ai) ────────
# Sourced 2026-08-11. Refresh when LiveBench releases new data (every ~6 months).
# Scores are 0-100.

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


def _get_session(db_path: str):
    from src.api.models import get_engine, get_session

    engine = get_engine(db_path)
    return get_session(engine)


def seed_livebench(db_path: str) -> int:
    """Seed model_capabilities from LiveBench leaderboard data."""
    from src.api.models import ModelCapability

    session = _get_session(db_path)
    now = datetime.now(timezone.utc).isoformat()
    count = 0

    for model, categories in LIVEBENCH_DATA.items():
        for lb_cat, raw_score in categories.items():
            if lb_cat == "overall":
                continue
            lcp_task = LB_TO_LCP.get(lb_cat, "casual_chat")
            normalized = round(raw_score / 100.0, 4)

            # Upsert: delete existing, insert new
            session.query(ModelCapability).filter_by(
                model=model, task_type=lcp_task, source="livebench"
            ).delete()

            session.add(ModelCapability(
                model=model,
                task_type=lcp_task,
                score=normalized,
                source="livebench",
                benchmark_category=lb_cat,
                raw_score=raw_score,
                updated_at=now,
            ))
            count += 1

    session.commit()
    session.close()
    print(f"Seeded {count} LiveBench capability rows for {len(LIVEBENCH_DATA)} models")
    return count


def seed_arena(db_path: str, max_rows: int = 10000) -> int:
    """Seed model_capabilities from Chatbot Arena Elo ratings."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Skipping Arena: datasets not installed. Run: pip install datasets")
        return 0

    from src.api.arena_capability import (
        compute_elo_per_task,
        normalize_to_capability,
        classify_prompt,
    )
    from src.api.models import ModelCapability

    print(f"Loading Arena dataset (first {max_rows} rows)...")
    ds = load_dataset(
        "lmsys/lmsys-arena-human-preference-55k", split="train", streaming=True
    )

    rows = []
    it = iter(ds)
    for i, row in enumerate(it):
        if i >= max_rows:
            break
        if row.get("winner_model_a") or row.get("winner_model_b"):
            rows.append(row)

    # Compute global Elo (task-specific classification too noisy on Arena chat data)
    import src.api.arena_capability as ac
    original_fn = ac.classify_prompt
    ac.classify_prompt = lambda x: "all"  # type: ignore[method-assign]
    elo = compute_elo_per_task(rows, min_count=20)
    arena_cap = normalize_to_capability(elo)
    ac.classify_prompt = original_fn  # type: ignore[method-assign]

    global_scores = arena_cap.get("all", {})
    if not global_scores:
        print("Arena: no models with sufficient battles")
        return 0

    session = _get_session(db_path)
    now = datetime.now(timezone.utc).isoformat()
    count = 0

    # Arena is global (no per-task breakdown), so add to ALL task types
    all_tasks = list(set(LB_TO_LCP.values()))

    for model, score in global_scores.items():
        # Skip if already in LiveBench (LiveBench is more recent/objective)
        existing = session.query(ModelCapability).filter_by(
            model=model, source="livebench"
        ).first()
        if existing:
            continue

        for task in all_tasks:
            session.query(ModelCapability).filter_by(
                model=model, task_type=task, source="arena"
            ).delete()
            session.add(ModelCapability(
                model=model,
                task_type=task,
                score=round(score, 4),
                source="arena",
                updated_at=now,
            ))
            count += 1

    session.commit()
    session.close()
    print(f"Seeded {count} Arena capability rows for {len(global_scores)} legacy models")
    return count


def load_capability_matrix(db_path: str) -> dict[str, dict[str, float]]:
    """Load the current capability matrix from the DB.

    Returns {task_type: {model_name: score}} suitable for the router.
    Models with scores from multiple sources use the highest-priority source.
    Priority: gateway_yaml > lcp_benchmark > livebench > arena
    """
    from src.api.models import ModelCapability

    session = _get_session(db_path)
    SOURCE_PRIORITY = {"gateway_yaml": 0, "lcp_benchmark": 1, "livebench": 2, "arena": 3}

    matrix: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    for row in session.query(ModelCapability).all():
        existing = matrix[row.task_type].get(row.model)
        if existing is not None:
            # Keep the higher-priority source
            existing_prio = SOURCE_PRIORITY.get(existing.get("source", "arena"), 99)
            new_prio = SOURCE_PRIORITY.get(row.source, 99)
            if new_prio >= existing_prio:
                continue
        matrix[row.task_type][row.model] = {"score": row.score, "source": row.source}

    session.close()

    return {
        task: {model: data["score"] for model, data in models.items()}
        for task, models in matrix.items()
    }


# ── Model registry (alias → logical → benchmark) ────────────────────────────

# Curated default registry. Seeded into the model_registry table on first run;
# subsequent changes are made via the DB (dashboard / API), NOT this dict.
DEFAULT_MODEL_REGISTRY: list[dict] = [
    {
        "logical_name": "deepseek-v4-pro",
        "benchmark_key": "deepseek-v4-pro",
        "aliases": ["deepseek-v4-pro", "deepseek/deepseek-v4-pro"],
    },
    {
        "logical_name": "deepseek-v4-flash",
        "benchmark_key": "deepseek-v4-flash-0731",
        "aliases": ["deepseek-v4-flash", "deepseek/deepseek-v4-flash", "deepseek-v4-flash-0731"],
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
            updated_at=now,
        ))
        count += 1

    session.commit()
    session.close()
    print(f"Seeded {count} model registry entries")
    return count


def load_model_registry(db_path: str) -> dict[str, dict]:
    """Load the model registry as {logical_name: {benchmark_key, aliases}}.

    Also builds a reverse alias index for fast lookup.
    """
    from src.api.models import ModelRegistryEntry

    session = _get_session(db_path)
    registry: dict[str, dict] = {}

    for row in session.query(ModelRegistryEntry).all():
        try:
            aliases = json.loads(row.aliases_json or "[]")
        except (json.JSONDecodeError, TypeError):
            aliases = []
        registry[row.logical_name] = {
            "benchmark_key": row.benchmark_key,
            "aliases": aliases,
        }

    session.close()
    return registry


def main():
    parser = argparse.ArgumentParser(description="Seed model capability scores")
    parser.add_argument("--source", choices=["livebench", "arena", "registry", "all"], default="all")
    parser.add_argument("--db-path", default="data/costs.db")
    parser.add_argument("--arena-rows", type=int, default=10000)
    args = parser.parse_args()

    sys.path.insert(0, os.getcwd())

    total = 0
    if args.source in ("livebench", "all"):
        total += seed_livebench(args.db_path)
    if args.source in ("arena", "all"):
        total += seed_arena(args.db_path, max_rows=args.arena_rows)
    if args.source in ("registry", "all"):
        total += seed_model_registry(args.db_path)

    # Show summary
    matrix = load_capability_matrix(args.db_path)
    print(f"\n=== Capability matrix ({len(matrix)} task types) ===")
    for task in sorted(matrix.keys()):
        top = sorted(matrix[task].items(), key=lambda x: -x[1])[:5]
        print(f"\n--- {task} ---")
        for model, score in top:
            print(f"  {model:35s} {score:.4f}")


if __name__ == "__main__":
    main()
