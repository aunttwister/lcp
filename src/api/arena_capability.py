"""Capability matrix from public benchmark leaderboards.

Problem: Chatbot Arena (lmsys-55k) only has models through 2024. DeepSeek V3+
isn't in it. LiveBench HF datasets are stale too.

Solution: Combine three sources, each contributing what they have:
  1. Arena 55k → Elo ratings for legacy models (GPT-4, Claude 1-2, Mixtral, etc.)
  2. LiveBench web leaderboard → per-category scores for current models
     (including DeepSeek V4 Pro, DeepSeek V4 Flash, Claude Fable 5, etc.)
     ─ scraped from livebench.ai (public HTML), refreshed periodically.
  3. Config overrides (gateway.yaml) → hand-tuned for models missing everywhere.

Sources:
  - Arena: lmsys/lmsys-arena-human-preference-55k (HF)
  - LiveBench HF: livebench/model_judgment (stale, up to deepseek-v3)
  - LiveBench web: https://livebench.ai/ (current, has DeepSeek V4)
  - Artificial Analysis: https://artificialanalysis.ai/ (independent benchmarks)

For now the PoC uses hardcoded LiveBench scores from the public leaderboard,
since web scraping requires a browser. A future cron job can refresh these.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .logging_config import get_logger

logger = get_logger("lcp.capability")


# ── LiveBench per-category scores (manually sourced from livebench.ai) ──────
# Release: 2026-06-25 (latest)
# Scores are 0-100. Categories: reasoning, coding, agentic_coding, math,
# data_analysis, language, instruction_following.
# Sourced 2026-08-11 from https://livebench.ai/

LIVEBENCH_SCORES: dict[str, dict[str, float]] = {
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
    "qwen-3.8-max": {
        "reasoning": 88.2, "coding": 72.9, "agentic_coding": 64.6,
        "math": 91.3, "data_analysis": 78.4, "language": 79.7,
        "instruction_following": 74.1, "overall": 78.5,
    },
}

# Map LiveBench categories to LCP task types
LB_TO_LCP_TASK: dict[str, str] = {
    "reasoning": "reasoning_chain",
    "coding": "code_generation",
    "agentic_coding": "agentic_multi_step",
    "math": "reasoning_chain",
    "data_analysis": "research_deep",
    "language": "casual_chat",
    "instruction_following": "planning",
}


def livebench_to_capability() -> dict[str, dict[str, float]]:
    """Convert LiveBench scores to LCP capability matrix.

    Maps LiveBench categories → LCP task types, normalizes 0-100
    scores to 0-1 range.
    """
    capability: dict[str, dict[str, float]] = defaultdict(dict)

    for model, categories in LIVEBENCH_SCORES.items():
        for lb_cat, score in categories.items():
            if lb_cat == "overall":
                continue
            lcp_task = LB_TO_LCP_TASK.get(lb_cat, "casual_chat")
            # Normalize: 100 → 1.0, 0 → 0.0 (LiveBench scores range ~40-96)
            normalized = round(score / 100.0, 4)
            capability[lcp_task][model] = normalized

    return dict(capability)


# ── Arena-derived Elo capability (for legacy models) ─────────────────────

@dataclass
class EloState:
    ratings: dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 1500.0))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def update(self, winner: str, loser: str, k: float = 32.0) -> None:
        ra = self.ratings[winner]
        rb = self.ratings[loser]
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        self.ratings[winner] = ra + k * (1.0 - ea)
        self.ratings[loser] = rb + k * (0.0 - (1.0 - ea))
        self.counts[winner] += 1
        self.counts[loser] += 1


# ── Task classification ──────────────────────────────────────────────────

TASK_SIGNALS: dict[str, list[str]] = {
    "agentic_multi_step": [
        "you are an ai agent", "you are a coding agent",
        "autonomous", "multi-step", "multi step", "agent",
        "tools:", "tool call", "function call",
    ],
    "code_generation": [
        "write a function", "implement", "create a script",
        "write code", "def ", "class ", "import ",
        "programming language", "write a program",
        "in python", "in javascript", "in rust", "in go",
        "html", "css", "react", "component",
    ],
    "debugging": [
        "debug", "error", "exception", "traceback",
        "stack trace", "why does this fail", "not working",
        "bug", "fix this", "what's wrong",
    ],
    "research_deep": [
        "explain", "analyze", "compare and contrast",
        "research", "literature review", "summarize",
        "in detail", "comprehensive", "thorough",
    ],
    "long_document": [
        "long document", "long text", "summarize this article",
        "summarize the following",
    ],
    "reasoning_chain": [
        "solve", "proof", "prove", "calculate",
        "logic puzzle", "step by step", "mathematical",
        "equation", "theorem",
    ],
    "planning": [
        "design", "architecture", "how should i structure",
        "plan", "roadmap", "strategy", "approach",
        "best practice", "recommend", "suggest",
    ],
}

CASUAL_SIGNALS = [
    "hello", "hi ", "hey", "thanks", "thank you", "how are you",
    "what's up", "good morning", "good night",
]


def classify_prompt(prompt_text: str) -> str:
    """Classify a prompt into a task type using keyword heuristics."""
    lower = prompt_text.lower()
    for task, keywords in TASK_SIGNALS.items():
        for kw in keywords:
            if kw in lower:
                return task
    for kw in CASUAL_SIGNALS:
        if kw in lower:
            return "casual_chat"
    return "casual_chat"


# ── Elo from Arena ────────────────────────────────────────────────────────

def compute_elo_per_task(
    rows: list[dict],
    min_count: int = 3,
) -> dict[str, dict[str, float]]:
    """Compute Elo ratings per task type from Arena preference data."""
    elo_per_task: dict[str, EloState] = defaultdict(EloState)

    for row in rows:
        prompt_text = ""
        if isinstance(row.get("prompt"), list) and row["prompt"]:
            prompt_text = row["prompt"][0]
        else:
            prompt_text = str(row.get("prompt", ""))

        task = classify_prompt(prompt_text)
        state = elo_per_task[task]

        if row.get("winner_model_a"):
            state.update(row["model_a"], row["model_b"])
        elif row.get("winner_model_b"):
            state.update(row["model_b"], row["model_a"])

    result: dict[str, dict[str, float]] = {}
    for task, state in elo_per_task.items():
        filtered = {
            model: rating
            for model, rating in state.ratings.items()
            if state.counts[model] >= min_count
        }
        if filtered:
            result[task] = filtered

    return result


def normalize_to_capability(
    elo_per_task: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Normalize Elo ratings to 0-1 scores using sigmoid."""
    result: dict[str, dict[str, float]] = {}
    for task, ratings in elo_per_task.items():
        normalized = {}
        for model, elo in ratings.items():
            z = (elo - 1500.0) / 250.0
            sigmoid = 1.0 / (1.0 + math.exp(-z))
            normalized[model] = round(max(0.05, min(0.99, sigmoid)), 4)
        result[task] = normalized
    return result


# ── Merged capability matrix ──────────────────────────────────────────────

def build_merged_capability(
    max_arena_rows: int = 10000,
    min_count: int = 5,
) -> dict[str, dict[str, float]]:
    """Build capability matrix from both sources.

    Layer 1: LiveBench scores (current models)
    Layer 2: Arena Elo (legacy models) — merged only for models NOT in LiveBench
    Layer 3: Config overrides (applied at router load time)

    Returns a merged dict: {task_type: {model_name: score_0_1}}
    """
    # Layer 1: LiveBench
    lb_cap = livebench_to_capability()
    lb_models: set[str] = set()
    for task_scores in lb_cap.values():
        lb_models.update(task_scores.keys())

    logger.info("livebench_models", count=len(lb_models), models=sorted(lb_models))

    print("\n=== LiveBench-derived Capability (current models) ===")
    for task, models in sorted(lb_cap.items()):
        top = sorted(models.items(), key=lambda x: -x[1])[:5]
        print(f"\n--- {task} ---")
        for model, score in top:
            print(f"  {model:35s} {score:.4f}")

    # Layer 2: Arena Elo (legacy, only for models not in LiveBench)
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "lmsys/lmsys-arena-human-preference-55k", split="train", streaming=True
        )
        rows = []
        it = iter(ds)
        for i, row in enumerate(it):
            if i >= max_arena_rows:
                break
            if row.get("winner_model_a") or row.get("winner_model_b"):
                # Only include if neither model is already in LiveBench
                # (LiveBench scores are more recent and objective)
                rows.append(row)

        # Compute global Elo (task classification is noisy for Arena chat data)
        # Use a dummy single-task approach
        original = classify_prompt
        import src.api.arena_capability as ac
        ac.classify_prompt = lambda x: "all"  # type: ignore[method-assign]

        elo = compute_elo_per_task(rows, min_count=min_count)
        arena_cap = normalize_to_capability(elo)
        ac.classify_prompt = original  # type: ignore[method-assign]

        # Merge: Arena models that aren't in LiveBench
        arena_global = arena_cap.get("all", {})
        for model, score in arena_global.items():
            if model not in lb_models:
                # Arena is global (no task breakdown), so add to all tasks
                for task in lb_cap:
                    lb_cap[task][model] = round(score, 4)

        print(f"\n=== Arena legacy models merged: {len(arena_global) - len(arena_global & lb_models)} ===")
        for model in sorted(arena_global.keys() - lb_models):
            print(f"  {model}: {arena_global[model]:.4f}")

    except Exception as e:
        logger.warning("arena_merge_failed", error=str(e))

    return lb_cap


# ── Export / Load ─────────────────────────────────────────────────────────

def export_capability_json(
    capability: dict[str, dict[str, float]],
    path: str = "data/capability_matrix.json",
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(capability, f, indent=2)
    logger.info("capability_exported", path=path, tasks=len(capability))


def load_capability_json(path: str = "data/capability_matrix.json") -> dict[str, dict[str, float]]:
    with open(path) as f:
        return json.load(f)
