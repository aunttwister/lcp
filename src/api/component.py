"""Component contract for the LCP runtime.

A :class:`Component` is a unit of composition in the gateway. It declares the
shared dependencies it READS (``requires``) and the keys it PUBLISHES
(``provides``) on the runtime context, and returns its own cleanup
(``dispose``) from :meth:`setup`. The runtime topologically orders components,
calls ``setup`` in dependency order, and replays disposers in LIFO order on
teardown.

This is the spatiotemporal-composability model (Cordis) applied to the gateway:
a component is active only while its declared dependencies are satisfied, and
its effects on the shared environment are revertible (each setup returns its
own inverse).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:  # pragma: no cover — runtime import only for type hints
    from .runtime import Runtime

# A cleanup closure returned by ``setup`` — the inverse of every effect the
# component performed while starting.
Disposer = Callable[[], None]


class Component(ABC):
    """A unit of composition. Declares deps (coeffects) and its own undo."""

    # Stable id — the runtime registry key.
    name: str = ""

    # Keys this component reads from the runtime (coeffects).
    requires: list[str] = []

    # Keys this component publishes to the runtime (effects).
    provides: list[str] = []

    def __init__(self) -> None:
        if not self.name:
            raise TypeError(
                f"{type(self).__name__} must define a non-empty 'name'"
            )

    @abstractmethod
    def setup(self, rt: "Runtime") -> Optional[Disposer]:
        """Acquire resources and bind dependencies.

        ``rt.resolve(key)`` reads declared dependencies (from ``requires``).
        Return an optional cleanup closure — the INVERSE of every effect this
        component performed. The runtime stacks these and replays them in LIFO
        order on :meth:`Runtime.shutdown`.
        """
        raise NotImplementedError

    def on_dependency_change(self, key: str) -> None:
        """Called when a declared dependency's provider changes (reactive
        coeffect). Default: no-op. Components that must react to a provider
        swap override this; the runtime may reload them if the provider
        differs.
        """
        return None
