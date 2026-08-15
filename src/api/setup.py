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
LIVEBENCH_EVAL_REQUIREMENTS = "code_runner/requirements_eval.txt"

# Root directory where runtime-installed modules are cloned. Override with
# ``LCP_MODULES_DIR`` (defaults to ``/opt/lcp-modules``, resolving to the
# per-install fallback ``~/.local/share/lcp/modules`` when /opt isn't writable).
MODULES_DIR_ENV = "LCP_MODULES_DIR"
DEFAULT_MODULES_DIR = "/opt/lcp-modules"

# Env var that pins the LiveBench checkout location explicitly. When unset,
# the checkout lives at ``<modules_dir>/livebench``. This still wins if set —
# it's the most specific override (backward compatible with older setups).
LIVEBENCH_DIR_ENV = "LCP_LIVEBENCH_DIR"


class SetupError(Exception):
    """Raised for synchronous, user-facing setup failures."""


def modules_dir() -> str:
    """Return the configured module install root (env or default)."""
    return os.environ.get(MODULES_DIR_ENV, "").strip() or DEFAULT_MODULES_DIR


def livebench_dir() -> str:
    """Return the LiveBench checkout path used by the runtime installer.

    Precedence:
      1. ``LCP_LIVEBENCH_DIR`` (explicit override, backward compatible)
      2. ``<LCP_MODULES_DIR>/livebench``

    This is the install *target*; it deliberately does NOT require the
    directory to exist yet.
    """
    explicit = os.environ.get(LIVEBENCH_DIR_ENV, "").strip()
    if explicit:
        return explicit
    return os.path.join(modules_dir(), "livebench")


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
        preset = _provider_preset(name)
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
            "api_base": preset.get("api_base", ""),
        })
    return steps


def benchmark_step() -> dict:
    """Build the LiveBench benchmark module manifest entry."""
    from .benchmark import benchmark_status

    status = benchmark_status()
    # Expose an in-flight install, or the most recent terminal *failure* so
    # the UI can keep showing what went wrong. A successful terminal state is
    # represented by ``installed=True`` and needs no ``installing`` payload.
    installing = _bench_install
    if installing is None and _bench_last is not None and _bench_last.get("status") == "failed":
        installing = _bench_last
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
        "install_path": livebench_dir(),
        "installing": installing,
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


def remove_provider(engine, config, name: str) -> dict:
    """Remove a provider plugin from the running config + credential store.

    Mirrors ``_serve_provider_delete`` but also clears the setup state marker.
    Returns ``{"removed": True, "provider": name}`` or raises ``SetupError``.
    """
    if name not in ("deepseek", "opencode", "commandcode", "llamacpp"):
        raise SetupError(f"unknown provider: {name}")

    # Remove the provider entry from gateway.yaml.
    cfg_raw = config.raw
    cfg_raw.setdefault("providers", {}).pop(name, None)
    # Drop it from every profile chain (same as the Providers page).
    for pcfg in cfg_raw.get("profiles", {}).values():
        chain = pcfg.get("chain")
        if isinstance(chain, list):
            pcfg["chain"] = [c for c in chain if c.get("provider") != name]
    config.save()

    # Clear encrypted credentials / cookies / workspace IDs.
    from .credential_store import get_credential_store

    store = get_credential_store()
    if store is not None:
        store.set(name, "")
        store.set_cookie(name, "")
        store.set_workspace_id(name, "")

    set_state(engine, f"provider:{name}", "removed")
    logger.info("setup_provider_removed", provider=name)
    return {"removed": True, "provider": name}


def remove_livebench(engine) -> dict:
    """Remove the LiveBench checkout and clear its setup state.

    Removes BOTH the configured install target (e.g. ``<modules_dir>/livebench``
    or an explicit ``LCP_LIVEBENCH_DIR``) AND any well-known fallback paths,
    so a stale checkout can't resurrect the module after removal.
    """
    targets: list[str] = []

    # 1. Explicit override / configured target.
    target = livebench_dir()
    targets.append(target)

    # 2. Well-known fallbacks the runtime installer may have written earlier.
    targets.append("/opt/livebench")
    targets.append(os.path.join(DEFAULT_MODULES_DIR, "livebench"))

    removed: list[str] = []
    seen: set[str] = set()
    for path in targets:
        if not path or path in seen:
            continue
        seen.add(path)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)

    # Drop the env override so a later install can re-clone cleanly.
    os.environ.pop(LIVEBENCH_DIR_ENV, None)

    set_state(engine, "module:livebench", "removed")
    logger.info("setup_livebench_removed", removed=removed)
    return {"removed": True, "module": "livebench", "paths": removed}


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
    """Background install: clone LiveBench + pip install core + eval extras.

    The ``code_runner/requirements_eval.txt`` extras (TensorFlow + scientific
    stack) are only needed to grade the ``coding`` category. When they fail
    (e.g. no compatible wheel for the Python version) the install still
    succeeds with a warning — the other five categories keep working.
    """
    try:
        target = livebench_dir()
        _bench_update(f"Target directory: {target}")
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)

        _stream(
            ["git", "clone", "--depth", "1", LIVEBENCH_REPO, target],
            cwd=None, start=2.0, end=25.0, status_msg="Cloning LiveBench…",
        )
        _stream(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-e", "."],
            cwd=target, start=25.0, end=60.0, status_msg="Installing LiveBench core…",
        )

        coding_note = None
        try:
            _stream(
                [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", LIVEBENCH_EVAL_REQUIREMENTS],
                cwd=target, start=60.0, end=100.0,
                status_msg="Installing eval extras (TensorFlow + scientific stack)…",
            )
        except subprocess.CalledProcessError as exc:
            # Non-fatal: grading the `coding` category will simply be disabled.
            _bench_update(
                f"Eval extras failed ({exc}) — LiveBench works, but the "
                f"'coding' category will be unavailable."
            )
            coding_note = "eval extras not installed; coding category unavailable"
            _bench_update(None, progress=100.0)

        # Make the running process see the checkout (benchmark.livebench_dir()
        # reads LCP_LIVEBENCH_DIR first).
        os.environ[LIVEBENCH_DIR_ENV] = target
        set_state(engine, "module:livebench", "done")
        if coding_note:
            _bench_finish("done", f"LiveBench installed — {coding_note}.")
        else:
            _bench_finish("done", "LiveBench installed.")
    except subprocess.CalledProcessError as exc:
        _bench_finish("failed", _tail_detail(f"Install failed: {exc}"))
    except FileNotFoundError as exc:
        _bench_finish("failed", f"Missing tool: {exc}")
    except Exception as exc:  # noqa: BLE001 — background thread must not die
        _bench_finish("failed", _tail_detail(f"Install failed: {exc}"))


def _tail_detail(fallback: str, lines: int = 3) -> str:
    """Return *fallback* plus the last few log lines (the real pip error)."""
    global _bench_install
    if _bench_install is None:
        return fallback
    log = _bench_install.get("log") or []
    if not log:
        return fallback
    tail = "\n".join(log[-lines:]).strip()
    if not tail:
        return fallback
    return f"{fallback}\n{tail[-400:]}"
