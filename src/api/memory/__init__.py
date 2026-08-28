"""Memory plugin — module-level API.

``init_memory(config)`` builds a :class:`LanceDBMemoryBackend` when the memory
module is installed (lancedb importable) AND ``plugins.memory.enabled`` is not
False. It never fails boot: a missing module / disabled config degrades to
``get_memory() -> None`` and the HTTP layer returns 501 with a Setup hint.

The embedder is resolved lazily: the real ``sentence-transformers`` model is
only loaded on first use, and only when the module has been installed via the
Setup page.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

from ..logging_config import get_logger
from .base import MemoryBackend
from .embeddings import EmbeddingModel, embedder_from_config

logger = get_logger("lcp.memory")

_backend: Optional[MemoryBackend] = None


# ── Install-time paths (mirror src.api.setup naming) ────────────────────────

def memory_site() -> str:
    """Return the persistent site-packages dir for memory deps.

    ``<LCP_MODULES_DIR>/memory`` — pip installs ``--target`` here so lancedb +
    sentence-transformers survive container recreation, and ``remove_memory``
    can delete it without touching LiveBench's shared ``site``.
    """
    root = os.environ.get("LCP_MODULES_DIR", "").strip() or "/opt/lcp-modules"
    return os.path.join(root, "memory")


def memory_models() -> str:
    """Return the directory used to cache the embedding model weights."""
    root = os.environ.get("LCP_MODULES_DIR", "").strip() or "/opt/lcp-modules"
    return os.path.join(root, "models", "memory")


def router_site() -> str:
    """Return the site-packages dir for the SEMANTIC ROUTING module deps.

    ``<LCP_MODULES_DIR>/router`` — pip installs sentence-transformers + torch
    ``--target`` here (independent of the memory plugin's install). The Docker
    build (WITH_ROUTER=1) and the Setup-page installer both use this dir.
    """
    root = os.environ.get("LCP_MODULES_DIR", "").strip() or "/opt/lcp-modules"
    return os.path.join(root, "router")


def router_models() -> str:
    """Return the directory used to cache the router embedding model weights.

    Prefers the baked image path (``/app/models/router``, populated by the
    Docker build's WITH_ROUTER=1 step) when present; otherwise falls back to
    the persistent modules dir used by the Setup-page runtime installer. The
    image path matters because ``/opt/lcp-modules`` and ``/app/data`` are
    bind-mounted at runtime and shadow any image content baked there.
    """
    baked = "/app/models/router"
    if os.path.isdir(baked):
        return baked
    root = os.environ.get("LCP_MODULES_DIR", "").strip() or "/opt/lcp-modules"
    return os.path.join(root, "models", "router")


def memory_available(site: Optional[str] = None) -> bool:
    """Return True when the memory module's Python deps are importable.

    Probes with a FRESH subprocess (like ``core_deps_available``) so an
    editable/``--target`` install that only registers a finder at interpreter
    startup is detected correctly. When *site* is given it is prepended to
    ``PYTHONPATH`` for the probe.
    """
    env = dict(os.environ)
    if site:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = site if not existing else f"{site}{os.pathsep}{existing}"
    probe = (
        "import importlib.util, sys;"
        "sys.exit(0 if importlib.util.find_spec('lancedb') and "
        "importlib.util.find_spec('sentence_transformers') else 1)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, timeout=60, env=env,
        )
        return result.returncode == 0
    except Exception:  # noqa: BLE001 — treat as unavailable
        return False


# ── Runtime state ───────────────────────────────────────────────────────────

def init_memory(config=None) -> bool:
    """Initialize the memory backend from config. Returns True when active.

    Never raises: any failure logs and leaves ``get_memory()`` returning None
    (endpoints report 501 with a Setup hint).
    """
    global _backend
    try:
        plugins = (getattr(config, "plugins", None) or {}) if config is not None else {}
        mem_cfg = plugins.get("memory") or {}
        if not mem_cfg.get("enabled", True):
            logger.info("memory_disabled_by_config")
            _backend = None
            return False

        # A non-string storage_path (e.g. a MagicMock from a test or a dubious
        # config) must NEVER reach a path API — os.makedirs would silently
        # create a directory tree named after the mock (MagicMock/<chain>/<id>)
        # instead of raising. Treat "configured but not a string" as a
        # misconfiguration: disable memory rather than guess; the default path
        # is only used when storage_path is absent/empty (init_memory(None)).
        raw_storage = mem_cfg.get("storage_path")
        if raw_storage is not None and not isinstance(raw_storage, str):
            logger.warning(
                "memory_storage_path_invalid",
                detail=f"storage_path must be a string, got {type(raw_storage).__name__}",
            )
            _backend = None
            return False
        storage = raw_storage.strip() if isinstance(raw_storage, str) else ""
        if not storage:
            data_dir = None
            try:
                data_dir = os.environ.get("COST_DB", "")
            except Exception:
                data_dir = ""
            base = os.path.dirname(data_dir) if data_dir else "data"
            storage = os.path.join(base, "memory")

        embedder: Optional[EmbeddingModel] = None
        try:
            embedder = embedder_from_config(mem_cfg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory_embedder_init_failed", error=str(exc))

        from .lancedb_backend import LanceDBMemoryBackend

        dim = int((mem_cfg.get("embedding") or {}).get("dim", 384) or 384)
        _backend = LanceDBMemoryBackend(
            storage,
            embed=lambda texts: embedder.embed(texts) if embedder else _noop_embed(texts),
            dim=dim,
        )
        logger.info("memory_initialized", storage=storage, dim=dim)
        return True
    except Exception as exc:  # noqa: BLE001 — never block boot
        logger.warning("memory_init_failed", error=str(exc))
        _backend = None
        return False


def _noop_embed(texts: list[str]) -> list[list[float]]:
    """Fallback embedder used only when the model can't be built.

    Produces a deterministic zero-ish vector per text so retain/recall never
    hard-fail before the module is installed; recall will simply be low-signal.
    """
    vec = [0.0] * 384
    return [list(vec) for _ in texts]


def get_memory() -> Optional[MemoryBackend]:
    """Return the active memory backend (or None when unavailable)."""
    return _backend


def memory_status() -> dict:
    """Return a status dict for the UI / manifest.

    ``available`` reflects the installed module (importable deps);
    ``active`` reflects the runtime backend (config-enabled + initialized).
    """
    available = memory_available(memory_site())
    return {
        "available": available,
        "active": _backend is not None,
        "site": memory_site(),
        "models_dir": memory_models(),
    }


# ── Semantic routing module status ─────────────────────────────────────────
# The embedding-based task classifier is its OWN installable module ("router"),
# independent of the memory plugin. It reuses the same availability probe
# (sentence-transformers importable) but against the router deps dir.

def router_available(site: Optional[str] = None) -> bool:
    """Return True when the router module's Python deps are importable.

    Probes with a FRESH subprocess (like ``memory_available``) so a --target
    install is detected correctly. When *site* is given it is prepended to
    ``PYTHONPATH`` for the probe.
    """
    env = dict(os.environ)
    if site:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = site if not existing else f"{site}{os.pathsep}{existing}"
    probe = (
        "import importlib.util, sys;"
        "sys.exit(0 if importlib.util.find_spec('sentence_transformers') else 1)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, timeout=60, env=env,
        )
        return result.returncode == 0
    except Exception:  # noqa: BLE001 — treat as unavailable
        return False


def router_status() -> dict:
    """Return a status dict for the semantic routing module (UI / manifest)."""
    site = os.path.join(os.environ.get("LCP_MODULES_DIR", "").strip() or "/opt/lcp-modules", "router")
    available = router_available(site)
    return {
        "available": available,
        "active": available,
        "site": site,
        "models_dir": os.path.join(os.path.dirname(site), "models", "router"),
    }


def shutdown_memory() -> None:
    """Release the backend and its embedder (best-effort)."""
    global _backend
    _backend = None
    logger.info("memory_shutdown")
