"""First-run setup wizard — installable module manifest + install coordinator.

LCP ships a small number of self-contained "plugins" that are each installed
(registered) once:

  - **Provider cost plugins** — ``deepseek``, ``opencode``, ``commandcode``,
    ``llamacpp``. "Installation" means adding the provider to ``gateway.yaml``
    (reusing the same preset + provider-create machinery as the Providers
    page) and — for API-keyed providers — storing the key encrypted in the
    credential store. Local/credential-free steps (``llamacpp``) or steps that
    only need a cookie/workspace-id are config-apply only.
  - **LiveBench benchmark module** — a runtime install (clone + pip install
    into the running container) that can run in the background and reports
    its progress in real time.

State is persisted in the ``setup_state`` table so the wizard knows which
steps are done, skipped, or failed. The Setup page reports the manifest
(kind, name, title, installed, required, etc.) and drives the
``POST /api/setup/install/{kind}/{name}`` + ``GET /api/setup/progress``
handshake.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from typing import Optional

from .logging_config import get_logger

logger = get_logger("lcp.setup")

LIVEBENCH_REPO = "https://github.com/LiveBench/LiveBench.git"
LIVEBENCH_DIR = os.environ.get("LCP_LIVEBENCH_DIR", "/opt/livebench").strip() or "/opt/livebench"
LIVEBENCH_EVAL_REQUIREMENTS = "code_runner/requirements_eval.txt"

# Env knob that disables the HTTP redirect gate (used by the test suite).
DISABLE_ENV = "LCP_DISABLE_SETUP_WIZARD"


class SetupError(Exception):
    """Raised for synchronous, user-facing setup failures."""


# ── Manifest ────────────────────────────────────────────────────────────────

def provider_steps(config) -> list[dict]:
    """Build the provider manifest from configured providers + plugin presets.

    A provider is "installed" when it exists in ``config.providers`` AND (for
    providers with an API-keyed plugin preset) a credential or cookie/workspace
    id is present. ``llamacpp`` has no key/cookie — config presence is enough.
    """
    from .credential_store import get_credential_store

    configured = set(config.providers.keys()) if config and hasattr(config, "providers") else set()
    store = get_credential_store()

    steps: list[dict] = []
    for name in ("deepseek", "opencode", "commandcode", "llamacpp"):
        has_cred = bool(store and (store.has(name) or store.has_cookie(name) or store.has_workspace_id(name)))
        steps.append({
            "kind": "provider",
            "name": name,
            "title": _PROVIDER_TITLES.get(name, name),
            "description": _PROVIDER_DESCRIPTIONS.get(name, ""),
            "required": name in ("deepseek", "opencode", "commandcode"),
            "needs_key": name in ("deepseek", "opencode", "commandcode"),
            "configured": name in configured,
            "has_credential": has_cred,
            "installed": name in configured and (not _provider_needs_key(name) or has_cred),
        })
    return steps


def benchmark_step() -> dict:
    """Build the LiveBench benchmark module manifest entry."""
    from .benchmark import benchmark_status

    status = benchmark_status()
    return {
        "kind": "module",
        "name": "livebench",
        "title": "LiveBench benchmarks",
        "description": (
            "Grades provider models into the capability matrix that drives "
            "dynamic routing. Clone + pip install at runtime."
        ),
        "required": False,
        "installed": bool(status.get("available")),
        "status": status,
        "installing": _bench_install,
    }


def manifest(config) -> dict:
    """Return the full setup manifest (provider steps + benchmark module)."""
    return {
        "steps": provider_steps(config),
        "modules": [benchmark_step()],
    }


# ── State helpers (kept in src.api.setup so handlers don't touch models) ────

def load_state(engine) -> dict[str, dict]:
    """Return {key: {status, updated_at, detail}} from the setup_state table."""
    from .models import SetupState, get_session

    try:
        with get_session(engine) as session:
            rows = session.query(SetupState).all()
            return {
                r.key: {
                    "status": r.status,
                    "updated_at": r.updated_at,
                    "detail": r.detail,
                }
                for r in rows
            }
    except Exception as exc:  # noqa: BLE001 — table may not exist yet
        logger.warning("setup_state_read_failed", error=str(exc))
        return {}


def set_state(engine, key: str, status: str, detail: Optional[str] = None) -> None:
    """Upsert one setup state record."""
    from .models import SetupState, get_session

    now = datetime.now(timezone.utc).isoformat()
    with get_session(engine) as session:
        row = session.query(SetupState).filter(SetupState.key == key).first()
        if row is None:
            session.add(SetupState(key=key, status=status, detail=detail, updated_at=now))
        else:
            row.status = status
            row.detail = detail
            row.updated_at = now
        session.commit()


def mark_skipped(engine) -> bool:
    """Persist wizard completion (skip) and return True when newly marked."""
    if load_state(engine).get("wizard", {}).get("status") == "skipped":
        return False
    # Record a marker for the gate's skip check.
    set_state(engine, "wizard", "skipped")
    return True


def is_complete(engine, config) -> bool:
    """True when the wizard may be bypassed (all required steps done or skipped)."""
    if load_state(engine).get("wizard", {}).get("status") == "skipped":
        return True
    required = [s for s in provider_steps(config) if s.get("required")]
    if not required:
        return True
    return all(s["installed"] for s in required)


# ── Provider install ────────────────────────────────────────────────────────

_PROVIDER_TITLES = {
    "deepseek": "DeepSeek",
    "opencode": "OpenCode",
    "commandcode": "Command Code",
    "llamacpp": "llama.cpp",
}
_PROVIDER_DESCRIPTIONS = {
    "deepseek": "Official DeepSeek API — pricing, cost + balance tracking.",
    "opencode": "OpenCode gateway — cost history + subscription usage.",
    "commandcode": "Command Code — billing API + gateway cost tracking.",
    "llamacpp": "Self-hosted local inference — zero-cost token tracking.",
}


def _provider_needs_key(name: str) -> bool:
    return name in ("deepseek", "opencode", "commandcode")


def _provider_preset(name: str) -> dict:
    """Return the quick-add preset for a provider (from the plugin registry)."""
    from .cost_plugins import get_registry

    return get_registry().presets.get(name, {}) or {}


def install_provider(engine, config, name: str, body: dict) -> dict:
    """Install one provider plugin into the running config.

    Mirrors ``_serve_provider_create`` (same write path + credential handling)
    but is idempotent and returns a structured result.
    """
    if name not in ("deepseek", "opencode", "commandcode", "llamacpp"):
        raise SetupError(f"unknown provider: {name}")

    preset = _provider_preset(name)
    api_base = (body.get("api_base") or preset.get("api_base") or "").strip()
    models = body.get("models") or preset.get("models") or []
    if not isinstance(models, list):
        raise SetupError("'models' must be a list")

    provider_data = {"api_base": api_base, "models": models}
    if not api_base:
        raise SetupError(f"missing api_base for {name} (no preset available)")

    # Write provider into gateway.yaml (same path as the Providers page).
    cfg_raw = config.raw
    existing = cfg_raw.get("providers", {}).get(name, {})
    merged = dict(existing)
    if api_base:
        merged["api_base"] = api_base
    if models:
        merged["models"] = models
    # Preserve gateway.yaml-only keys like api_key_env / cache settings.
    cfg_raw.setdefault("providers", {})[name] = merged
    config.save()

    # Store the API key encrypted (never in gateway.yaml) when provided.
    from .credential_store import get_credential_store

    store = get_credential_store()
    if store is None:
        raise SetupError("credential store not initialized")

    api_key = (body.get("api_key") or "").strip()
    if _provider_needs_key(name):
        if not api_key and not store.has(name):
            raise SetupError(f"missing api_key for {name}")

    if api_key:
        store.set(name, api_key)
    if body.get("cookie"):
        store.set_cookie(name, body["cookie"])
    if body.get("workspace_id"):
        store.set_workspace_id(name, body["workspace_id"])

    set_state(engine, f"provider:{name}", "done")
    logger.info("setup_provider_installed", provider=name)
    return {"installed": True, "provider": name}


# ── LiveBench runtime install (background + progress) ───────────────────────

_bench_lock = threading.Lock()
_bench_install: Optional[dict] = None  # in-flight {status, progress, detail, log, ...}
_bench_last: Optional[dict] = None     # terminal result (done/failed) for the UI

_LOG_MAX_LINES = 300


def _bench_update(msg: Optional[str], progress: Optional[float] = None,
                  status: Optional[str] = None) -> None:
    """Mutate the shared install state (no-op when nothing is in flight)."""
    global _bench_install
    if _bench_install is None:
        return
    if status is not None:
        _bench_install["status"] = status
    if progress is not None:
        _bench_install["progress"] = round(min(max(progress, 0.0), 100.0), 1)
    if msg is not None:
        clean = (msg.rstrip("\n") if isinstance(msg, str) else str(msg)).strip("\r")
        if clean:
            _bench_install["detail"] = clean[-200:]
            log = _bench_install.setdefault("log", [])
            log.append(clean)
            if len(log) > _LOG_MAX_LINES:
                del log[: len(log) - _LOG_MAX_LINES]
    _bench_install["updated_at"] = datetime.now(timezone.utc).isoformat()


def _bench_finish(status: str, detail: str) -> None:
    """Mark the install terminal and move its state into ``_bench_last``."""
    global _bench_install, _bench_last
    _bench_update(detail, status=status)
    if status == "done":
        _bench_update(None, progress=100.0)
    if _bench_install is not None:
        _bench_install["finished_at"] = datetime.now(timezone.utc).isoformat()
        _bench_last = dict(_bench_install)
    _bench_install = None


def _stream(cmd: list[str], cwd: Optional[str], start: float, end: float,
            status_msg: str) -> None:
    """Run a subprocess streaming its output into the shared install log.

    Progress eases from ``start`` to ``end`` as output lines arrive.
    Raises ``subprocess.CalledProcessError`` on a non-zero exit code.
    """
    _bench_update(status_msg, progress=start, status="running")
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, errors="replace",
    )
    seen = 0
    assert proc.stdout is not None
    for line in proc.stdout:
        seen += 1
        _bench_update(line)
        if seen % 3 == 0:
            # Ease toward the phase ceiling; the final wait pins it exactly.
            frac = min(0.9, seen / 90.0)
            _bench_update(None, progress=start + (end - start) * frac)
    rc = proc.wait()
    _bench_update(None, progress=end)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def bench_progress() -> Optional[dict]:
    """Return the in-flight LiveBench install state (or None when idle)."""
    return _bench_install


def bench_last() -> Optional[dict]:
    """Return the most recent terminal install result (or None)."""
    return _bench_last


def start_livebench_install(engine) -> dict:
    """Start (or join) the LiveBench runtime install and return its state."""
    global _bench_install, _bench_last

    if shutil.which("git") is None:
        raise SetupError(
            "git is not installed in this environment — rebuild the image with "
            "WITH_BENCH=1, or install git in the container."
        )

    with _bench_lock:
        if _bench_install is not None and _bench_install.get("status") in ("queued", "running"):
            return _bench_install
        _bench_last = None
        _bench_install = {
            "status": "queued",
            "progress": 0.0,
            "detail": "Waiting to start…",
            "log": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        state = dict(_bench_install)
        thread = threading.Thread(target=_run_livebench_install, args=(engine,), daemon=True)
        thread.start()
        return state


def _run_livebench_install(engine) -> None:
    """Background install: clone LiveBench + pip install core + eval extras."""
    try:
        if os.path.isdir(LIVEBENCH_DIR):
            shutil.rmtree(LIVEBENCH_DIR, ignore_errors=True)

        _stream(
            ["git", "clone", "--depth", "1", LIVEBENCH_REPO, LIVEBENCH_DIR],
            cwd=None, start=2.0, end=25.0, status_msg="Cloning LiveBench…",
        )
        _stream(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-e", "."],
            cwd=LIVEBENCH_DIR, start=25.0, end=60.0, status_msg="Installing LiveBench core…",
        )
        _stream(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", LIVEBENCH_EVAL_REQUIREMENTS],
            cwd=LIVEBENCH_DIR, start=60.0, end=100.0,
            status_msg="Installing eval extras (TensorFlow + scientific stack)…",
        )

        # Make the running process see the checkout (benchmark.livebench_dir()
        # reads LCP_LIVEBENCH_DIR first).
        os.environ["LCP_LIVEBENCH_DIR"] = LIVEBENCH_DIR
        set_state(engine, "module:livebench", "done")
        _bench_finish("done", "LiveBench installed.")
    except subprocess.CalledProcessError as exc:
        _bench_finish("failed", f"Install failed: {exc}")
    except FileNotFoundError as exc:
        _bench_finish("failed", f"Missing tool: {exc}")
    except Exception as exc:  # noqa: BLE001 — background thread must not die
        _bench_finish("failed", f"Install failed: {exc}")
