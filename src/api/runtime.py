"""Declarative component runtime for the LCP gateway.

Owns the shared dependencies (``config``, ``engine``, ``data_dir``) and the
component graph. Components declare what they need (``requires``) and publish
(``provides``); :meth:`Runtime.start` topologically orders them, calls each
``setup`` in dependency order, and stacks the returned disposers.
:meth:`Runtime.shutdown` replays those disposers in LIFO order, so teardown is
the exact inverse of setup — clean removal without hand-sequenced ``finally``
blocks.

Boot never breaks: a component whose declared dependencies cannot be satisfied
(missing key, or a dependency cycle) is logged and marked INACTIVE, and the
rest of the runtime starts. Request-time ``resolve`` of an unsatisfied key
raises :class:`UndeclaredDependency`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover — runtime import only for type hints
    from .component import Component

logger = get_logger("lcp.runtime")


class UndeclaredDependency(Exception):
    """Raised when a key is requested from the runtime but not provided."""


class Runtime:
    """The context. Owns shared deps and the component graph."""

    # Root keys owned directly by the runtime (not by any component).
    ROOT_KEYS = ("config", "engine", "data_dir")

    def __init__(self, config: Any = None, engine: Any = None,
                 data_dir: str = ""):
        self._config = config
        self._engine = engine
        self._data_dir = data_dir
        # name -> Component
        self._components: dict[str, Component] = {}
        # published key -> component name that provides it
        self._provides: dict[str, str] = {}
        self._order: list[str] = []          # names in setup order
        self._disposers: dict[str, Any] = {}  # name -> disposer (or None)
        self._active: dict[str, bool] = {}    # name -> active flag
        self._started = False

    # ── Registration ────────────────────────────────────────────────────

    def register(self, comp: "Component") -> None:
        """Register a component. Collision on name or provides-key is an error.

        Root keys (``config``/``engine``/``data_dir``) are reserved and cannot
        be claimed as a component's ``provides``.
        """
        if comp.name in self._components:
            raise ValueError(f"Component '{comp.name}' already registered")
        if comp.name in self.ROOT_KEYS:
            raise ValueError(f"Component name '{comp.name}' is a reserved root key")
        for key in comp.provides:
            if key in self.ROOT_KEYS:
                raise ValueError(f"Component '{comp.name}' provides reserved root key '{key}'")
            if key in self._provides:
                raise ValueError(
                    f"Key '{key}' already provided by '{self._provides[key]}'"
                )
        self._components[comp.name] = comp
        self._active[comp.name] = False
        # A component's own name resolves to its instance.
        self._provides[comp.name] = comp.name
        for key in comp.provides:
            self._provides[key] = comp.name

    # ── Resolution ──────────────────────────────────────────────────────

    def resolve(self, key: str) -> Any:
        """Resolve a provided key.

        Root keys return the runtime's shared deps. A component's name or any
        of its ``provides`` keys return the component instance. Raises
        :class:`UndeclaredDependency` when the key is not provided, is
        provided by an inactive component, or was never registered.
        """
        if key in self.ROOT_KEYS:
            return getattr(self, f"_{key}")
        owner = self._provides.get(key)
        if owner is None:
            raise UndeclaredDependency(
                f"'{key}' is not provided by any registered component"
            )
        if not self._active.get(owner, False):
            raise UndeclaredDependency(
                f"'{key}' is provided by '{owner}', which is inactive"
            )
        return self._components[owner]

    def is_active(self, name: str) -> bool:
        """True when the named component is active (started, deps satisfied)."""
        return self._active.get(name, False)

    def components(self) -> list["Component"]:
        """Return all registered components (setup order when started)."""
        return [self._components[n] for n in self._order] + [
            c for n, c in self._components.items() if n not in self._order
        ]

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Topo-sort components by requires/provides and setup each.

        A component is set up only when every declared dependency is already
        satisfied (a root key, or a key provided by an already-started
        component). Components whose deps can't be satisfied (missing key or a
        cycle) are logged and left INACTIVE — boot never breaks.
        """
        if self._started:
            raise RuntimeError("Runtime already started")
        pending = dict(self._components)
        ready = set(self.ROOT_KEYS)
        progressed = True
        while pending and progressed:
            progressed = False
            for name in list(pending):
                comp = pending[name]
                if all(r in ready or r in self.ROOT_KEYS for r in comp.requires):
                    self._setup_one(comp)
                    self._order.append(name)
                    ready.add(name)
                    ready.update(comp.provides)
                    pending.pop(name, None)
                    progressed = True
        for name, comp in pending.items():
            missing = [r for r in comp.requires if r not in ready]
            logger.warning(
                "component_inactive",
                name=name,
                reason="dependencies unsatisfied",
                missing=missing,
            )
            self._active[name] = False
        self._started = True
        logger.info(
            "runtime_started",
            order=self._order,
            inactive=sorted(n for n, a in self._active.items() if not a),
        )

    def reload(self, name: str) -> None:
        """Dispose + re-setup one component and notify its dependents.

        The component's disposer runs (undo), then ``setup`` runs again and a
        fresh disposer is stacked in the original position. Every component
        that depends on this one is notified via ``on_dependency_change``.
        """
        comp = self._components.get(name)
        if comp is None:
            raise KeyError(f"Component '{name}' is not registered")
        if not self._active.get(name, False):
            raise RuntimeError(f"Component '{name}' is not active")
        self._dispose_one(name)
        self._setup_one(comp, notify=False)
        for dep in self._dependents(name):
            dep.on_dependency_change(name)
        logger.info("component_reloaded", name=name)

    def shutdown(self) -> None:
        """Replay all tracked disposers in LIFO order (reverse of setup)."""
        for name in reversed(self._order):
            self._dispose_one(name)
        self._order = []
        self._disposers = {}
        for name in self._active:
            self._active[name] = False
        self._started = False
        logger.info("runtime_shutdown")

    # ── Internals ───────────────────────────────────────────────────────

    def _setup_one(self, comp: "Component", notify: bool = True) -> None:
        try:
            disposer = comp.setup(self)
            self._disposers[comp.name] = disposer
            self._active[comp.name] = True
            logger.info("component_started", name=comp.name)
        except Exception as exc:  # noqa: BLE001 — never break boot
            logger.warning("component_start_failed", name=comp.name, error=str(exc)[:300])
            self._disposers[comp.name] = None
            self._active[comp.name] = False

    def _dispose_one(self, name: str) -> None:
        disposer = self._disposers.get(name)
        if disposer is not None:
            try:
                disposer()
            except Exception as exc:  # noqa: BLE001 — dispose must not break teardown
                logger.warning("component_dispose_failed", name=name, error=str(exc)[:300])
        self._disposers[name] = None
        self._active[name] = False

    def _dependents(self, name: str) -> list["Component"]:
        """Components whose ``requires`` mention *name* or any key it provides."""
        provided = {name}
        comp = self._components.get(name)
        if comp is not None:
            provided.update(comp.provides)
        return [
            c for c in self._components.values()
            if any(r in provided for r in c.requires)
        ]


# ═══════════════════════════════════════════════════════════════════════════
# Active-runtime accessor (Phase F)
# ═══════════════════════════════════════════════════════════════════════════
# Every module facade's ``bind_runtime(rt)`` also records the runtime here, so
# "the runtime" is a single concept instead of one ``_runtime`` global per
# module. The request path resolves services through :func:`get_runtime` /
# :func:`resolve_service` and falls back to the legacy module singletons when
# no runtime is bound (tests, standalone).
_active_runtime: Optional["Runtime"] = None


def bind_active_runtime(rt: "Runtime") -> None:
    """Record *rt* as the active runtime (called by every facade's bind)."""
    global _active_runtime
    _active_runtime = rt


def get_runtime() -> Optional["Runtime"]:
    """Return the active Runtime, or None when none is bound."""
    return _active_runtime


def resolve_service(key: str, fallback: Any = None) -> Any:
    """Resolve a provided key's published SERVICE from the active runtime.

    Returns the component's ``service`` (the object it publishes, e.g. the
    breaker, the settings store, the key manager) when a runtime is bound and
    that component is active. Otherwise returns *fallback* — invoked lazily
    when it is callable, returned as-is otherwise.

    The request path uses this so it reads the runtime-owned instance when one
    is active and the legacy module singleton otherwise — identical behavior to
    the ``get_*()`` facades, but explicit about the source.
    """
    rt = _active_runtime
    if rt is not None:
        try:
            return rt.resolve(key).service
        except Exception:  # noqa: BLE001 — inactive/unbound/attr → fallback
            pass
    if callable(fallback):
        return fallback()
    return fallback
