"""LiveBench benchmark runner — queue and execute benchmark runs in the background.

LCP benchmarks the **raw model directly against its provider** (never through
LCP's own routing, which would contaminate the very scores the dynamic router
depends on). The target abstraction accepts a ``provider`` kind today and a
``profile`` kind later (to benchmark council / dynamic-routed profiles
end-to-end through LCP).

LiveBench is invoked as a subprocess against a local checkout (path from
``LCP_LIVEBENCH_DIR``). Only the six non-Docker categories are run — agentic
coding is excluded by design (no Docker).

Results are parsed from LiveBench's ``all_groups.csv`` (per-category accuracy)
and upserted into ``model_capabilities`` with ``source="lcp_benchmark"``.
"""

from __future__ import annotations

import csv
import io
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from .logging_config import get_logger

logger = get_logger("lcp.benchmark")

# LiveBench categories LCP benchmarks. These map 1:1 to columns in
# all_groups.csv and to LCP task types via seed_capabilities.LB_TO_LCP.
# ``agentic_coding`` is intentionally excluded — it requires Docker and is not
# run (per project decision to skip Docker tests).
LIVEBENCH_CATEGORIES = [
    "reasoning", "coding", "math",
    "data_analysis", "language", "instruction_following",
]

# LiveBench question release to evaluate. The latest public release is
# 2024-11-25; the website shows 2026-06-25 but not all questions are public.
DEFAULT_LIVEBENCH_RELEASE = os.environ.get("LCP_LIVEBENCH_RELEASE", "2024-11-25")


def validate_categories(categories: Optional[list[str]]) -> list[str]:
    """Validate a requested category list against ``LIVEBENCH_CATEGORIES``.

    Returns a normalized (deduped, order-preserving) list, or raises
    ``ValueError`` for unknown categories — including ``agentic_coding``,
    which is deliberately unsupported.
    """
    if not categories:
        return list(LIVEBENCH_CATEGORIES)
    normalized: list[str] = []
    for c in categories:
        c = (c or "").strip().lower()
        if not c:
            continue
        if c not in LIVEBENCH_CATEGORIES:
            raise ValueError(
                f"unknown benchmark category {c!r} — supported: "
                f"{', '.join(LIVEBENCH_CATEGORIES)}"
            )
        if c not in normalized:
            normalized.append(c)
    return normalized or list(LIVEBENCH_CATEGORIES)


# ── Pure command construction (testable) ────────────────────────────────────

def livebench_dir() -> Optional[str]:
    """Return the path to a LiveBench checkout, or None if not configured.

    Resolution order: ``LCP_LIVEBENCH_DIR`` env var → ``livebench`` on PATH
    (a directory containing ``run_livebench.py``).
    """
    env = os.environ.get("LCP_LIVEBENCH_DIR", "").strip()
    if env and os.path.isdir(env):
        return env
    found = shutil.which("run_livebench.py")
    if found:
        return os.path.dirname(found)
    return None


def _redact_cmd(cmd: list[str]) -> str:
    """Render a command for logging with the API key redacted."""
    out = list(cmd)
    for i, arg in enumerate(out):
        if i > 0 and out[i - 1] == "--api-key":
            out[i] = "***"
    return " ".join(out)


def build_livebench_commands(
    model: str,
    api_base: str,
    api_key: str,
    categories: Optional[list[str]] = None,
    livebench_path: Optional[str] = None,
    release: str = DEFAULT_LIVEBENCH_RELEASE,
) -> list[list[str]]:
    """Build the LiveBench subprocess argv lists for a model run.

    Returns one or two commands per requested scope:
      - ``run_livebench.py`` — generates answers + ground-truth judgments
      - ``show_livebench_result.py`` — writes ``all_groups.csv`` for parsing

    ``categories=None`` (or empty) benchmarks every non-Docker category
    (``LIVEBENCH_CATEGORIES``); otherwise one scope per requested category
    (``live_bench/<category>``). The full ``live_bench`` scope is never used
    because it would pull in Docker-requiring agentic coding.
    """
    path = livebench_path or livebench_dir()
    if not path:
        raise RuntimeError(
            "LiveBench checkout not found — set LCP_LIVEBENCH_DIR to a "
            "directory containing run_livebench.py, or put it on PATH."
        )

    runner = os.path.join(path, "run_livebench.py")
    shower = os.path.join(path, "show_livebench_result.py")

    chosen = validate_categories(categories)
    scopes = [f"live_bench/{c}" for c in chosen]

    commands: list[list[str]] = []
    for scope in scopes:
        commands.append([
            "python", runner,
            "--model", model,
            "--api-base", api_base,
            "--api-key", api_key,
            "--bench-name", scope,
            "--livebench-release-option", release,
        ])
        commands.append([
            "python", shower,
            "--model-list", model,
            "--bench-name", scope,
            "--livebench-release-option", release,
        ])
    return commands


def parse_livebench_csv(csv_text: str, model: str) -> dict[str, float]:
    """Parse LiveBench's ``all_groups.csv`` into ``{category: score_0_100}``.

    The CSV has a ``model`` column plus one column per category. Scores are
    0-100 floats. Returns an empty dict when the model row is absent.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None:
        return {}

    normalized_fields = {f.strip().lower(): f for f in reader.fieldnames}

    for row in reader:
        if (row.get("model") or "").strip().lower() != model.strip().lower():
            continue
        scores: dict[str, float] = {}
        for cat in LIVEBENCH_CATEGORIES:
            col = normalized_fields.get(cat)
            if col is None or col not in row:
                continue
            raw = (row[col] or "").strip()
            if raw in ("", "-", "N/A"):
                continue
            try:
                scores[cat] = float(raw)
            except ValueError:
                continue
        return scores
    return {}


# ── Background worker (single thread, serialized) ───────────────────────────

_worker_thread: Optional[threading.Thread] = None
_worker_queue: "queue.Queue[tuple[int, Any, Any]]" = queue.Queue()
_worker_lock = threading.Lock()


def _ensure_worker() -> None:
    """Start the single background worker thread if it isn't running."""
    global _worker_thread
    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
            _worker_thread.start()
            logger.info("benchmark_worker_started")


def _worker_loop() -> None:
    """Consume queued benchmark runs one at a time."""
    while True:
        run_id, engine, config = _worker_queue.get()
        try:
            _execute_run(run_id, engine, config)
        except Exception as exc:  # noqa: BLE001 — worker must never die
            logger.error("benchmark_worker_crash", run_id=run_id, error=str(exc))
            _mark_failed(run_id, engine, f"worker crash: {exc}")
        finally:
            _worker_queue.task_done()


def _mark_failed(run_id: int, engine, error: str) -> None:
    from .models import BenchmarkRun, get_session
    with get_session(engine) as session:
        run = session.query(BenchmarkRun).filter(BenchmarkRun.id == run_id).first()
        if run is not None:
            run.status = "failed"
            run.error = error[:4000]
            run.finished_at = datetime.now(timezone.utc).isoformat()
            session.commit()


# ── Public API ───────────────────────────────────────────────────────────────

def queue_benchmark(
    engine,
    config,
    target_kind: str,
    target: dict,
    categories: Optional[list[str]] = None,
) -> dict:
    """Queue a benchmark run. Returns the run record (status=queued).

    Validates the target kind, target shape, and categories up front so a bad
    request fails synchronously (the caller sees the error) instead of
    producing a silently-failed background run.
    """
    from .models import BenchmarkRun, get_session

    if target_kind not in ("provider", "profile"):
        raise ValueError(f"invalid target_kind: {target_kind!r}")

    # Category validation now (fail fast), not in the background worker.
    validated = validate_categories(categories)

    if target_kind == "provider":
        if not isinstance(target, dict) or not target.get("provider") or not target.get("model"):
            raise ValueError("provider target requires 'provider' and 'model'")

    with get_session(engine) as session:
        run = BenchmarkRun(
            target_kind=target_kind,
            target_json=json.dumps(target),
            categories_json=json.dumps(validated),
            status="queued",
        )
        session.add(run)
        session.commit()
        run_id = run.id

    _worker_queue.put((run_id, engine, config))
    _ensure_worker()
    logger.info("benchmark_queued", run_id=run_id, target_kind=target_kind, target=target)
    return get_run(engine, run_id)


def list_runs(engine, limit: int = 50) -> list[dict]:
    from .models import BenchmarkRun, get_session
    with get_session(engine) as session:
        rows = session.query(BenchmarkRun).order_by(BenchmarkRun.id.desc()).limit(limit).all()
        return [_run_to_dict(r) for r in rows]


def get_run(engine, run_id: int) -> Optional[dict]:
    from .models import BenchmarkRun, get_session
    with get_session(engine) as session:
        run = session.query(BenchmarkRun).filter(BenchmarkRun.id == run_id).first()
        return _run_to_dict(run) if run else None


def _run_to_dict(run) -> dict:
    def _json(s, default):
        try:
            return json.loads(s) if s else default
        except json.JSONDecodeError:
            return default

    return {
        "id": run.id,
        "target_kind": run.target_kind,
        "target": _json(run.target_json, {}),
        "categories": _json(run.categories_json, None),
        "status": run.status,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "result": _json(run.result_json, None),
        "error": run.error,
        "created_at": run.created_at,
    }


# ── Execution ───────────────────────────────────────────────────────────────

def _resolve_provider_target(engine, config, target: dict) -> tuple[str, str, str]:
    """Resolve a provider-kind target to (provider_api_model, api_base, api_key).

    Returns the provider-side model ID (translated via the cost-plugin registry
    when the provider uses a different ID scheme, e.g. Command Code prefixes).
    """
    provider = target.get("provider")
    model = target.get("model")
    if not provider or not model:
        raise ValueError("provider target requires 'provider' and 'model'")

    provider_cfg = config.providers.get(provider)
    if provider_cfg is None:
        raise ValueError(f"unknown provider: {provider!r}")

    api_base = provider_cfg.get("api_base", "")
    if not api_base:
        raise ValueError(f"provider {provider!r} has no api_base")

    # API key: credential store first (sole UI-managed source), then env var.
    from .credential_store import get_credential_store
    store = get_credential_store()
    api_key = ""
    if store is not None:
        api_key = store.get(provider) or ""
    if not api_key:
        api_key = config.get_provider_key(provider)

    # Translate logical model → provider API model ID.
    api_model = model
    from .cost_plugins import get_registry
    plugin = get_registry().for_provider(provider)
    if plugin is not None:
        api_model = plugin.get_api_model(model)

    return api_model, api_base, api_key


def _execute_run(run_id: int, engine, config) -> None:
    from .models import BenchmarkRun, get_session

    # Load the run record.
    with get_session(engine) as session:
        run = session.query(BenchmarkRun).filter(BenchmarkRun.id == run_id).first()
        if run is None:
            return
        target = json.loads(run.target_json)
        categories = json.loads(run.categories_json) if run.categories_json else None
        target_kind = run.target_kind
        run.status = "running"
        run.started_at = datetime.now(timezone.utc).isoformat()
        session.commit()

    try:
        if target_kind != "provider":
            raise RuntimeError(
                f"target_kind {target_kind!r} is not implemented yet "
                f"(only 'provider' benchmarks the raw model directly)"
            )

        api_model, api_base, api_key = _resolve_provider_target(engine, config, target)

        commands = build_livebench_commands(
            model=api_model,
            api_base=api_base,
            api_key=api_key,
            categories=categories,
        )

        workdir = tempfile.mkdtemp(prefix="lcp-bench-")
        category_scores: dict[str, float] = {}

        for cmd in commands:
            logger.info("benchmark_subprocess", run_id=run_id, cmd=_redact_cmd(cmd))
            proc = subprocess.run(
                cmd, cwd=workdir, capture_output=True, text=True, timeout=3600,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"LiveBench command failed (exit {proc.returncode}): "
                    f"{proc.stderr[-2000:] or proc.stdout[-2000:]}"
                )

        # Parse all_groups.csv written by show_livebench_result.py into cwd.
        csv_path = os.path.join(workdir, "all_groups.csv")
        if os.path.isfile(csv_path):
            with open(csv_path) as f:
                category_scores = parse_livebench_csv(f.read(), api_model)
        else:
            logger.warning("benchmark_no_csv", run_id=run_id, path=csv_path)

        if not category_scores:
            raise RuntimeError("LiveBench produced no parseable category scores")

        _upsert_scores(engine, target, category_scores)

        now = datetime.now(timezone.utc).isoformat()
        with get_session(engine) as session:
            run = session.query(BenchmarkRun).filter(BenchmarkRun.id == run_id).first()
            run.status = "done"
            run.finished_at = now
            run.result_json = json.dumps({
                "model": api_model,
                "categories": category_scores,
            })
            run.error = None
            session.commit()
        logger.info("benchmark_done", run_id=run_id, categories=category_scores)

    except Exception as exc:  # noqa: BLE001 — record any failure, keep worker alive
        logger.error("benchmark_failed", run_id=run_id, error=str(exc))
        _mark_failed(run_id, engine, str(exc))


def _upsert_scores(engine, target: dict, category_scores: dict[str, float]) -> None:
    """Normalize 0-100 category scores to 0-1 and upsert into model_capabilities."""
    from .seed_capabilities import LB_TO_LCP
    from .models import ModelCapability, get_session

    logical_model = target.get("model", "")
    now = datetime.now(timezone.utc).isoformat()

    with get_session(engine) as session:
        for lb_cat, raw in category_scores.items():
            task_type = LB_TO_LCP.get(lb_cat)
            if task_type is None:
                continue
            # Upsert: replace any prior lcp_benchmark row for this model+task.
            existing = session.query(ModelCapability).filter_by(
                model=logical_model, task_type=task_type, source="lcp_benchmark"
            ).first()
            score = round(raw / 100.0, 4)
            if existing is not None:
                existing.score = score
                existing.raw_score = raw
                existing.benchmark_category = lb_cat
                existing.updated_at = now
            else:
                session.add(ModelCapability(
                    model=logical_model,
                    task_type=task_type,
                    score=score,
                    source="lcp_benchmark",
                    benchmark_category=lb_cat,
                    raw_score=raw,
                    updated_at=now,
                ))
        session.commit()
