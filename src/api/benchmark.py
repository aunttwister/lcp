"""LiveBench benchmark runner — queue and execute benchmark runs in the background.

LCP benchmarks the **raw model directly against its provider** (never through
LCP's own routing, which would contaminate the very scores the dynamic router
depends on). The target abstraction accepts a ``provider`` kind today and a
``profile`` kind later (to benchmark council / dynamic-routed profiles
end-to-end through LCP).

LiveBench is invoked as a subprocess against a local checkout (path from
``LCP_MODULES_DIR``). Only the six non-Docker categories are run — agentic
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
import sys
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

# LiveBench task name → category. Used to group all_tasks.csv subtask scores.
# Based on the 2026-06-25 question release; unknown tasks fall under "_all".
LIVEBENCH_TASK_CATEGORIES: dict[str, str] = {
    "theory_of_mind": "reasoning",
    "zebra_puzzle": "reasoning",
    "spatial": "reasoning",
    "logic_with_navigation": "reasoning",
    "web_of_lies_v2": "reasoning",
    "house_traversal": "reasoning",
    "code_generation": "coding",
    "code_completion": "coding",
    "lcb_generation": "coding",
    "javascript": "agentic_coding",
    "typescript": "agentic_coding",
    "python": "agentic_coding",
    "amps_hard": "math",
    "integrals_with_game": "math",
    "math_comp": "math",
    "olympiad": "math",
    "consecutive_events": "data_analysis",
    "table_join": "data_analysis",
    "table_reformat": "data_analysis",
    "cta": "data_analysis",
    "connections": "language",
    "plot_unscrambling": "language",
    "typos": "language",
    "paraphrase": "instruction_following",
    "simplify": "instruction_following",
    "story_generation": "instruction_following",
    "summarize": "instruction_following",
}

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

def _valid_checkout(path: str) -> bool:
    """Return True when *path* is a checkout ROOT.

    A checkout root contains ``pyproject.toml`` AND a ``livebench/`` package
    directory holding ``run_livebench.py``. The repository clones to
    ``<LCP_MODULES_DIR>/livebench`` (root), and the actual Python package is
    ``<LCP_MODULES_DIR>/livebench/livebench``.
    """
    return (
        os.path.isfile(os.path.join(path, "pyproject.toml"))
        and os.path.isfile(os.path.join(path, "livebench", "run_livebench.py"))
    )


def livebench_root() -> Optional[str]:
    """Return the LiveBench checkout root (contains pyproject.toml)."""
    modules_root = os.environ.get("LCP_MODULES_DIR", "").strip()
    if modules_root:
        candidate = os.path.join(modules_root, "livebench")
        if _valid_checkout(candidate):
            return candidate
    for candidate in ("/opt/livebench",):
        if _valid_checkout(candidate):
            return candidate
    return None


def livebench_dir() -> Optional[str]:
    """Return the LiveBench package dir containing ``run_livebench.py``.

    This is both the script directory AND the required working directory —
    LiveBench's scripts import ``from livebench...`` and read ``data/...``
    relative to it.
    """
    root = livebench_root()
    if root:
        return os.path.join(root, "livebench")
    found = shutil.which("run_livebench.py")
    if found:
        return os.path.dirname(found)
    return None


def coding_deps_available() -> bool:
    """Return True when the 'coding' category can be graded.

    Grading generated code needs LiveBench's ``code_runner`` extra
    (TensorFlow, scipy, etc.). We probe for one heavyweight marker package;
    the absence only disables the ``coding`` category, not the whole runner.
    """
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


def core_deps_available(site: Optional[str] = None) -> bool:
    """Return True when LiveBench's core package is importable.

    ``show_livebench_result.py`` does ``from livebench.common import ...`` so
    the ``livebench`` package must be importable, and ``run_livebench.py``
    imports ``libtmux`` first.

    IMPORTANT: we probe with a FRESH python subprocess rather than importing
    in the long-lived LCP process — editable installs (``pip install -e .``)
    register a ``.pth`` finder that is only picked up at interpreter startup,
    so an in-process ``import livebench`` would falsely fail right after a
    successful editable install. When *site* is given (the persistent
    ``--target`` dir), it is prepended to ``PYTHONPATH`` along with the repo
    root for the probe.
    """
    env = dict(os.environ)
    if site:
        from .setup import livebench_pythonpath
        module_path = livebench_pythonpath()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = module_path if not existing else f"{module_path}{os.pathsep}{existing}"
    probe = (
        "import importlib.util, sys;"
        "sys.exit(0 if importlib.util.find_spec('libtmux') and "
        "importlib.util.find_spec('livebench') else 1)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, timeout=60, env=env,
        )
        return result.returncode == 0
    except Exception:  # noqa: BLE001 — treat as unavailable
        return False


def benchmark_status() -> dict:
    """Report whether the benchmark runner is usable, and at what capacity.

    Returns ``{"available": bool, "reason": str|None, "categories": [...],
    "coding_supported": bool}``. The runner is optional ("plugin") — the base
    gateway stays lean and this simply reports what's installed.
    """
    path = livebench_dir()
    if not path:
        modules_root = os.environ.get("LCP_MODULES_DIR", "").strip()
        return {
            "available": False,
            "reason": (
                "LiveBench checkout not found — install it from the Setup "
                "page (it clones into $LCP_MODULES_DIR/livebench), or set "
                "LCP_MODULES_DIR to a directory that already contains a "
                "livebench checkout." if modules_root else
                "LiveBench checkout not found — install it from the Setup "
                "page, or build the image with WITH_BENCH=1."
            ),
            "categories": list(LIVEBENCH_CATEGORIES),
            "coding_supported": False,
        }
    return {
        "available": True,
        "reason": None,
        "livebench_dir": path,
        "categories": list(LIVEBENCH_CATEGORIES),
        "coding_supported": coding_deps_available(),
        "core_installed": core_deps_available(),
    }


def _redact_cmd(cmd: list[str]) -> str:
    """Render a command for logging with the API key redacted."""
    out = list(cmd)
    for i, arg in enumerate(out):
        if i > 0 and out[i - 1] == "--api-key":
            out[i] = "***"
    return " ".join(out)


def _redact_stream_line(line: str, api_key: str) -> str:
    """Redact a raw API key that LiveBench itself echoes in its own output.

    LiveBench prints the full command string including
    ``export LIVEBENCH_API_KEY='...'``. This scrubs that value before it
    reaches the live log, regardless of quoting style.
    """
    if not api_key:
        return line
    line = line.replace(api_key, "***")
    # Also catch the env-export form with single or double quotes.
    for quoted in (f"'{api_key}'", f'"{api_key}"'):
        line = line.replace(quoted, "'***'")
    return line


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
            "LiveBench checkout not found — set LCP_MODULES_DIR to a "
            "directory containing a livebench checkout, or put run_livebench.py on PATH."
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


def parse_livebench_tasks_csv(csv_text: str, model: str) -> dict[str, dict[str, float]]:
    """Parse LiveBench's ``all_tasks.csv`` into subtask scores.

    ``all_tasks.csv`` has a ``model`` column plus one column per TASK
    (e.g. ``theory_of_mind``, ``zebra_puzzle``). Scores are 0-100 floats,
    averaged per task. Returns ``{category: {task_name: score}}`` when a
    category→task mapping is known (via ``LIVEBENCH_TASK_CATEGORIES``), else
    ``{"_all": {task: score}}``. Returns ``{}`` when the model row is absent.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None:
        return {}

    for row in reader:
        if (row.get("model") or "").strip().lower() != model.strip().lower():
            continue
        by_cat: dict[str, dict[str, float]] = {}
        for task, col in row.items():
            if task.lower() == "model":
                continue
            raw = (col or "").strip()
            if raw in ("", "-", "N/A"):
                continue
            try:
                score = float(raw)
            except ValueError:
                continue
            cat = LIVEBENCH_TASK_CATEGORIES.get(task.lower(), "_all")
            by_cat.setdefault(cat, {})[task.lower()] = score
        return by_cat
    return {}


# ── Background worker (single thread, serialized) ───────────────────────────

_worker_thread: Optional[threading.Thread] = None
_worker_queue: "queue.Queue[tuple[int, Any, Any]]" = queue.Queue()
_worker_lock = threading.Lock()

# Per-run streaming output. The worker appends subprocess stdout/stderr lines
# as they arrive; the UI polls GET /api/models/benchmark/{id}/log for live
# progress. Lines are also appended to a per-run log FILE under the data dir
# so they survive process restarts without bloating the DB.
_run_logs: dict[int, list[str]] = {}
_LOG_MAX_LINES = 1000

# Directory + engine resolved lazily from the first queue_benchmark call.
_log_dir: Optional[str] = None


def _bind_log_engine(engine: Any) -> None:
    """Resolve the log directory from the engine's DB URL (best-effort).

    Logs live next to the database (e.g. /app/data/benchmark-logs/) so they
    ride the same bind mount as costs.db.
    """
    global _log_dir
    try:
        url = str(getattr(engine, "url", "") or "")
        # sqlite:///path/costs.db → path/
        path = url.split("///", 1)[1] if "///" in url else ""
        if path:
            parent = os.path.dirname(path)
            _log_dir = os.path.join(parent, "benchmark-logs") if parent else "benchmark-logs"
    except Exception:
        pass


def _log_path(run_id: int) -> Optional[str]:
    if not _log_dir:
        return None
    return os.path.join(_log_dir, f"run-{run_id}.log")


def _log(run_id: int, line: str) -> None:
    """Append a line to a run's live buffer and its log file."""
    clean = (line.rstrip("\n") if isinstance(line, str) else str(line)).strip("\r")
    if not clean:
        return
    buf = _run_logs.setdefault(run_id, [])
    buf.append(clean)
    if len(buf) > _LOG_MAX_LINES:
        del buf[: len(buf) - _LOG_MAX_LINES]

    path = _log_path(run_id)
    if path:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                f.write(clean + "\n")
        except OSError:
            pass


def get_run_log(engine, run_id: int) -> list[str]:
    """Return the run log: live buffer, else the persisted log file.

    The live buffer is authoritative while the worker holds it in memory; on a
    restart the buffer is empty and we read the file instead (bounded tail).
    """
    if run_id in _run_logs and _run_logs[run_id]:
        return list(_run_logs[run_id])
    if _log_dir is None and engine is not None:
        _bind_log_engine(engine)
    path = _log_path(run_id)
    if path and os.path.isfile(path):
        try:
            with open(path, errors="replace") as f:
                return [ln.rstrip("\n") for ln in f.readlines()][-_LOG_MAX_LINES:]
        except OSError:
            pass
    return []


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


def recover_stale_runs(engine) -> int:
    """Mark orphaned 'queued'/'running' runs as failed at startup.

    The worker is an in-process daemon thread; after a restart any run left in
    queued/running will never progress. Mark them failed with a clear message
    so the UI doesn't show a phantom 'running' row forever. Returns the count.
    """
    from .models import BenchmarkRun, get_session

    now = datetime.now(timezone.utc).isoformat()
    recovered = 0
    with get_session(engine) as session:
        stale = session.query(BenchmarkRun).filter(
            BenchmarkRun.status.in_(["queued", "running"])
        ).all()
        for run in stale:
            run.status = "failed"
            run.error = (
                "Run was interrupted by a gateway restart — its in-memory "
                "worker no longer exists. Re-run it to benchmark again."
            )
            run.finished_at = now
            recovered += 1
        session.commit()
    if recovered:
        logger.info("benchmark_stale_recovered", count=recovered)
    return recovered


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


def list_runs(engine, limit: int = 50, offset: int = 0, model: Optional[str] = None) -> dict:
    """Return a page of benchmark runs (newest first) plus total count.

    ``model`` (optional) filters to runs whose target model/profile equals it.
    Returns ``{"runs": [...], "total": int, "limit": int, "offset": int}`` so
    the UI can render pagination. ``limit`` is clamped to 1..200.
    """
    from .models import BenchmarkRun, get_session
    limit = max(1, min(int(limit or 50), 200))
    offset = max(0, int(offset or 0))
    with get_session(engine) as session:
        q = session.query(BenchmarkRun)
        if model:
            q = q.filter(BenchmarkRun.target_json.like(f'%"{model}"%'))
        total = q.count()
        rows = (
            q.order_by(BenchmarkRun.id.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return {
            "runs": [_run_to_dict(r) for r in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }


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
    # Priority: explicit registry provider_mappings > cost-plugin heuristic.
    api_model = model
    if engine is not None:
        db_path = "data/costs.db"
        try:
            if getattr(engine, "url", None) is not None:
                db_path = str(engine.url.database) or db_path
        except Exception:
            pass
        from .router import provider_model_name, logical_model_name
        logical = logical_model_name(model, db_path)
        api_model = provider_model_name(logical, provider, db_path)

    from .cost_plugins import get_registry
    plugin = get_registry().for_provider(provider)
    if plugin is not None:
        translated = plugin.get_api_model(model)
        # Plugin heuristic wins when there's no engine/registry, or when the
        # registry had no explicit mapping for this provider (api_model == model).
        if api_model == model:
            api_model = translated

    return api_model, api_base, api_key


def _execute_run(run_id: int, engine, config) -> None:
    from .models import BenchmarkRun, get_session

    _bind_log_engine(engine)

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

        if not core_deps_available():
            raise RuntimeError(
                "LiveBench core package is not installed (missing libtmux). "
                "Run the Setup → LiveBench install to pip install the "
                "checkout, or run: pip install -e \"$LCP_MODULES_DIR/livebench\""
            )

        api_model, api_base, api_key = _resolve_provider_target(engine, config, target)

        checkout = livebench_dir()
        if not checkout:
            raise RuntimeError("LiveBench checkout not found")

        # Dependencies are installed into the persistent bind mount
        # (<modules_dir>/site) and the livebench package resolves via the repo
        # root. Prepend both to PYTHONPATH for the benchmark subprocesses.
        env = dict(os.environ)
        from .setup import livebench_pythonpath
        module_path = livebench_pythonpath()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = module_path if not existing else f"{module_path}{os.pathsep}{existing}"

        commands = build_livebench_commands(
            model=api_model,
            api_base=api_base,
            api_key=api_key,
            categories=categories,
            livebench_path=checkout,
        )

        # LiveBench's scripts are path-sensitive: run_livebench.py invokes
        # gen_api_answer.py / gen_ground_truth_judgment.py relative to CWD, and
        # show_livebench_result.py does `from livebench.common import ...` plus
        # reads `data/...` relative to CWD. They MUST run from the checkout
        # root, not a temp dir.
        workdir = checkout
        category_scores: dict[str, float] = {}

        # Fatal provider/auth errors in the streamed output. When these appear
        # there's no point grinding through all 150 questions with retries —
        # abort the run immediately with an actionable message.
        fatal_markers = (
            "insufficient balance", "creditserror", "authenticationerror",
            "invalid api key", "not in plan", "model_not_in_plan",
            "upgrade_required", "permission_error",
        )

        def _is_fatal(line: str) -> bool:
            low = line.lower()
            return any(m in low for m in fatal_markers)

        for cmd in commands:
            _log(run_id, f"$ {_redact_cmd(cmd)}")
            logger.info("benchmark_subprocess", run_id=run_id, cmd=_redact_cmd(cmd))
            proc = subprocess.Popen(
                cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, errors="replace", env=env,
            )
            assert proc.stdout is not None
            fatal = None
            for line in proc.stdout:
                clean = _redact_stream_line(line, api_key)
                _log(run_id, clean)
                if fatal is None and _is_fatal(clean):
                    fatal = clean.strip()[-500:]
            rc = proc.wait()
            if fatal is not None:
                proc.kill()
                raise RuntimeError(
                    f"LiveBench aborted — fatal provider error: {fatal}"
                )
            if rc != 0:
                tail = "\n".join(get_run_log(engine, run_id)[-40:])
                raise RuntimeError(
                    f"LiveBench command failed (exit {rc}): {_redact_stream_line(tail[-2000:], api_key)}"
                )

        # Parse all_groups.csv written by show_livebench_result.py into CWD
        # (the checkout root).
        csv_path = os.path.join(workdir, "all_groups.csv")
        if os.path.isfile(csv_path):
            with open(csv_path) as f:
                category_scores = parse_livebench_csv(f.read(), api_model)
        else:
            logger.warning("benchmark_no_csv", run_id=run_id, path=csv_path)

        # Also parse all_tasks.csv (subtask breakdown) when present.
        task_scores: dict[str, dict[str, float]] = {}
        tasks_csv_path = os.path.join(workdir, "all_tasks.csv")
        if os.path.isfile(tasks_csv_path):
            with open(tasks_csv_path) as f:
                task_scores = parse_livebench_tasks_csv(f.read(), api_model)

        if not category_scores:
            raise RuntimeError("LiveBench produced no parseable category scores")

        release_label = target.get("release") or None
        _upsert_scores(engine, target, category_scores, release_label=release_label)
        # New scores landed in model_capabilities → drop the router's cached
        # matrix so live routing picks up the refreshed scores immediately.
        try:
            from .router import invalidate_router_matrix
            invalidate_router_matrix()
        except Exception:  # noqa: BLE001 — never block benchmark completion
            pass

        now = datetime.now(timezone.utc).isoformat()
        with get_session(engine) as session:
            run = session.query(BenchmarkRun).filter(BenchmarkRun.id == run_id).first()
            run.status = "done"
            run.finished_at = now
            run.result_json = json.dumps({
                "model": api_model,
                "categories": category_scores,
                "tasks": task_scores,
            })
            run.error = None
            session.commit()
        logger.info("benchmark_done", run_id=run_id, categories=category_scores)

    except Exception as exc:  # noqa: BLE001 — record any failure, keep worker alive
        logger.error("benchmark_failed", run_id=run_id, error=str(exc))
        _log(run_id, f"[LCP] benchmark failed: {exc}")
        _mark_failed(run_id, engine, str(exc))


def _upsert_scores(engine, target: dict, category_scores: dict[str, float],
                   release_label: Optional[str] = None) -> None:
    """Normalize 0-100 category scores to 0-1 and upsert into model_capabilities.

    Scores are tagged with ``release_label`` (defaults to today's date when not
    provided) so re-benchmarking the same model on a new release keeps history
    instead of overwriting it. Only rows for the SAME (model, task, source,
    release) are replaced.
    """
    from .seed_capabilities import LB_TO_LCP, DERIVED_TASKS
    from .models import ModelCapability, get_session

    logical_model = target.get("model", "")
    label = release_label or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now = datetime.now(timezone.utc).isoformat()

    def _upsert(task_type, raw, category):
        existing = session.query(ModelCapability).filter_by(
            model=logical_model, task_type=task_type, source="lcp_benchmark",
            release_label=label,
        ).first()
        score = round(raw / 100.0, 4)
        if existing is not None:
            existing.score = score
            existing.raw_score = raw
            existing.benchmark_category = category
            existing.updated_at = now
        else:
            session.add(ModelCapability(
                model=logical_model,
                task_type=task_type,
                score=score,
                source="lcp_benchmark",
                benchmark_category=category,
                raw_score=raw,
                release_label=label,
                updated_at=now,
            ))

    with get_session(engine) as session:
        code_raw = None
        for lb_cat, raw in category_scores.items():
            task_type = LB_TO_LCP.get(lb_cat)
            if task_type is None:
                continue
            _upsert(task_type, raw, lb_cat)
            if task_type == "code_generation":
                code_raw = raw
        # Derived tasks (debugging mirrors code_generation — a coding subskill).
        if code_raw is not None:
            for derived in DERIVED_TASKS:
                _upsert(derived, code_raw, "coding")
        session.commit()
