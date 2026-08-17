"""Benchmark data import pipeline — JSON datasets → SQLite.

Reads benchmark datasets from JSON files and materializes them into SQLite:

  ``capability_metrics``            — source-of-truth metric rows
  ``model_capabilities``            — typed top-level scores (router + UI)
  ``model_capability_subtasks``     — typed per-subtask scores (Models page)

JSON files are discovered from two locations:

1. **Bundled** datasets: ``src/api/data/*.json`` (ship with the gateway).
2. **Installed modules**: ``$LCP_MODULES_DIR/**/data/*.json``. A module that
   ships a dataset with the same ``schema_id`` OVERRIDES the bundled one, so
   third-party benchmark plugins can drop in their own data without forking.

A dataset payload looks like::

    {
      "schema_id": "livebench",
      "release_label": "2026-06-25",
      "models": {
        "deepseek-v4-pro": {
          "releases": {"2026-08-13": {"reasoning": 85.8, ...}},
          "subtasks": {"reasoning": {"theory_of_mind": 84.6, ...}}
        }
      }
    }

CLI::

    python -m src.api.benchmark_import [--db path] [--file path] [--dry-run]
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone
from typing import Optional

from .logging_config import get_logger
from .seed_capabilities import LB_TO_LCP, derive_category_scores

logger = get_logger("lcp.benchmark_import")

BUNDLED_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODULES_DIR_ENV = "LCP_MODULES_DIR"
DEFAULT_MODULES_DIR = "/opt/lcp-modules"

# Canonical leaderboard order. Materialization iterates metrics in this order
# so that categories mapping to the SAME LCP task (reasoning + math →
# reasoning_chain) keep the hand-typed "last writer wins" behavior.
CATEGORY_ORDER = (
    "reasoning", "coding", "agentic_coding", "math",
    "data_analysis", "language", "instruction_following",
)


def modules_dir() -> str:
    """Return the configured module install root (env or default)."""
    return os.environ.get(MODULES_DIR_ENV, "").strip() or DEFAULT_MODULES_DIR


def discover_files(modules_root: Optional[str] = None) -> list[str]:
    """Return dataset JSON paths — bundled first, then module-provided.

    Bundled files are listed first; module files later so that (in
    ``import_bundled``) a module dataset with the same ``schema_id``
    overrides the bundled one.
    """
    files: list[str] = []
    if os.path.isdir(BUNDLED_DATA_DIR):
        files.extend(sorted(glob.glob(os.path.join(BUNDLED_DATA_DIR, "*.json"))))
    root = modules_root if modules_root is not None else modules_dir()
    if root and os.path.isdir(root):
        files.extend(sorted(glob.glob(os.path.join(root, "**", "data", "*.json"), recursive=True)))
    return files


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_payload(payload: dict) -> tuple[str, str, list[dict]]:
    """Validate a dataset payload and flatten it into metric rows.

    Returns ``(schema_id, release_label, rows)`` where each row is a dict:
    ``{schema_id, release_label, model, category, task, value, source}``.

    * Top-level category scores → ``task=None``.
    * Per-subtask scores → ``task=<task>``.
    * ``release_label`` on a top-level row is the MODEL VERSION (mirrors
      ``model_capabilities.release_label``); on a subtask row it is the
      dataset snapshot date (mirrors ``model_capability_subtasks``).
    * Models that have subtasks but no ``releases`` get top-level scores
      DERIVED from their subtask rows.
    """
    schema_id = str(payload.get("schema_id") or "").strip()
    release_label = str(payload.get("release_label") or "").strip()
    if not schema_id:
        raise ValueError("dataset payload missing 'schema_id'")
    if not release_label:
        raise ValueError("dataset payload missing 'release_label'")

    models = payload.get("models") or {}
    if not isinstance(models, dict):
        raise ValueError("dataset payload 'models' must be an object")

    rows: list[dict] = []

    def add(model: str, rel: str, category: str, task: Optional[str], value: float) -> None:
        rows.append({
            "schema_id": schema_id,
            "release_label": rel,
            "model": model,
            "category": category,
            "task": task,
            "value": value,
            "source": "livebench",
        })

    for model, entry in models.items():
        model = str(model).strip()
        if not model or not isinstance(entry, dict):
            continue

        releases = entry.get("releases") or {}
        if not isinstance(releases, dict):
            releases = {}
        for rel, categories in releases.items():
            rel = str(rel).strip()
            if not isinstance(categories, dict):
                continue
            for category, raw in categories.items():
                value = _to_float(raw)
                if value is None:
                    continue
                add(model, rel, str(category), None, value)

        subtasks = entry.get("subtasks") or {}
        if not isinstance(subtasks, dict):
            subtasks = {}
        for category, tasks in subtasks.items():
            if not isinstance(tasks, dict):
                continue
            for task, raw in tasks.items():
                value = _to_float(raw)
                if value is None:
                    continue
                add(model, release_label, str(category), str(task), value)

        # Models with subtasks but no releases get top-level scores derived
        # from their subtask averages.
        if subtasks and not releases:
            for category, value in derive_category_scores(subtasks).items():
                if category == "overall":
                    add(model, release_label, "overall", None, value)
                else:
                    add(model, release_label, category, None, value)

    return schema_id, release_label, rows


def _engine(db_path: str):
    from .models import Base, get_engine

    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine


def import_payload(
    db_path: str,
    payload: dict,
    release: Optional[str] = None,
    materialize_capabilities: bool = True,
    materialize_subtasks: bool = True,
    dry_run: bool = False,
) -> int:
    """Import one parsed dataset payload into SQLite.

    Writes ``capability_metrics`` (the whole payload snapshot, replacing prior
    rows for the same ``schema_id``) and then materializes the typed query
    tables. Returns the number of materialized typed rows.
    """
    from .models import CapabilityMetric, get_session

    schema_id, _, rows = parse_payload(payload)
    now = datetime.now(timezone.utc).isoformat()

    engine = _engine(db_path)
    if dry_run:
        return len(rows)

    with get_session(engine) as session:
        session.query(CapabilityMetric).filter(
            CapabilityMetric.schema_id == schema_id
        ).delete(synchronize_session=False)
        for row in rows:
            session.add(CapabilityMetric(updated_at=now, **row))
        session.commit()

    count = 0
    if materialize_capabilities:
        count += materialize_capability_rows(engine, schema_id=schema_id, release=release)
    if materialize_subtasks:
        count += materialize_subtask_rows(engine, schema_id=schema_id, release=release)
    return count


def import_file(
    db_path: str,
    path: str,
    release: Optional[str] = None,
    materialize_capabilities: bool = True,
    materialize_subtasks: bool = True,
    dry_run: bool = False,
) -> int:
    """Import a single JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return import_payload(
        db_path, payload,
        release=release,
        materialize_capabilities=materialize_capabilities,
        materialize_subtasks=materialize_subtasks,
        dry_run=dry_run,
    )


def import_bundled(
    db_path: str,
    release: Optional[str] = None,
    materialize_capabilities: bool = True,
    materialize_subtasks: bool = True,
    dry_run: bool = False,
) -> int:
    """Import all discovered datasets (bundled + module-provided).

    When a module dataset shares a ``schema_id`` with a bundled one, the
    module's file wins (it is discovered later and overrides).
    """
    by_schema: dict[str, dict] = {}
    for path in discover_files():
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("benchmark_import_skip", path=path, error=str(exc))
            continue
        schema_id = str(payload.get("schema_id") or "").strip()
        if not schema_id:
            logger.warning("benchmark_import_no_schema", path=path)
            continue
        by_schema[schema_id] = payload

    total = 0
    for payload in by_schema.values():
        total += import_payload(
            db_path, payload,
            release=release,
            materialize_capabilities=materialize_capabilities,
            materialize_subtasks=materialize_subtasks,
            dry_run=dry_run,
        )
    return total


def _metric_rows(engine, schema_id: Optional[str], release: Optional[str],
                 top_level: bool) -> list:
    from .models import CapabilityMetric, get_session

    with get_session(engine) as session:
        q = session.query(CapabilityMetric)
        if schema_id:
            q = q.filter(CapabilityMetric.schema_id == schema_id)
        if release:
            q = q.filter(CapabilityMetric.release_label == release)
        if top_level:
            q = q.filter(CapabilityMetric.task.is_(None))
        else:
            q = q.filter(CapabilityMetric.task.isnot(None))
        rows = q.all()
    return rows


def materialize_capability_rows(engine, schema_id: Optional[str] = None,
                                release: Optional[str] = None) -> int:
    """Write typed ``model_capabilities`` rows from top-level metrics.

    Iterates in canonical category order so ``reasoning_chain`` ends up with
    the ``math`` value (matching the hand-typed seed order).
    """
    from .models import ModelCapability, get_session

    rows = _metric_rows(engine, schema_id, release, top_level=True)
    order = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    rows.sort(key=lambda r: order.get(r.category, len(order)))

    now = datetime.now(timezone.utc).isoformat()
    count = 0
    with get_session(engine) as session:
        for r in rows:
            task_type = LB_TO_LCP.get(r.category)
            if task_type is None:
                continue  # "overall" + unknown categories
            score = round(r.value / 100.0, 4)
            existing = session.query(ModelCapability).filter_by(
                model=r.model, task_type=task_type, source="livebench",
                release_label=r.release_label,
            ).first()
            if existing is not None:
                existing.score = score
                existing.raw_score = r.value
                existing.benchmark_category = r.category
                existing.updated_at = now
            else:
                session.add(ModelCapability(
                    model=r.model,
                    task_type=task_type,
                    score=score,
                    source="livebench",
                    benchmark_category=r.category,
                    raw_score=r.value,
                    release_label=r.release_label,
                    updated_at=now,
                ))
            count += 1
        session.commit()
    return count


def materialize_subtask_rows(engine, schema_id: Optional[str] = None,
                             release: Optional[str] = None) -> int:
    """Write typed ``model_capability_subtasks`` rows from per-subtask metrics."""
    from .models import ModelCapabilitySubtask, get_session

    rows = _metric_rows(engine, schema_id, release, top_level=False)
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    with get_session(engine) as session:
        for r in rows:
            score = round(r.value / 100.0, 4)
            existing = session.query(ModelCapabilitySubtask).filter_by(
                model=r.model, category=r.category, task=r.task,
                source="livebench", release_label=r.release_label,
            ).first()
            if existing is not None:
                existing.score = score
                existing.raw_score = r.value
                existing.updated_at = now
            else:
                session.add(ModelCapabilitySubtask(
                    model=r.model,
                    category=r.category,
                    task=r.task,
                    score=score,
                    source="livebench",
                    raw_score=r.value,
                    release_label=r.release_label,
                    updated_at=now,
                ))
            count += 1
        session.commit()
    return count


def _default_db_path() -> str:
    return os.environ.get("COST_DB", "data/costs.db")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Import benchmark datasets from JSON into SQLite")
    parser.add_argument("--db", default=_default_db_path(), help="path to SQLite DB")
    parser.add_argument("--file", default=None, help="import a single JSON file (default: all discovered datasets)")
    parser.add_argument("--release", default=None, help="only materialize this release label")
    parser.add_argument("--dry-run", action="store_true", help="parse + report without writing")
    args = parser.parse_args()

    if args.file:
        count = import_file(args.db, args.file, release=args.release, dry_run=args.dry_run)
        print(f"Imported {count} materialized rows from {args.file}")
        return

    count = import_bundled(args.db, release=args.release, dry_run=args.dry_run)
    files = discover_files()
    print(f"Imported {len(files)} dataset file(s) → {count} materialized rows (release={args.release or 'all'})")


if __name__ == "__main__":
    main()
