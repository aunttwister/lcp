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

