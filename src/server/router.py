"""Declarative route table for the LCP HTTP handler.

This replaces the hand-rolled ``if/elif`` chain that used to dispatch every
request in ``src/server/handler.py`` — a single method with 147 branches of
string comparison (60 ``if path ==``, 58 ``elif path ==``, 25 ``startswith``,
plus suffix and membership tests). That chain could not be listed, counted or
tested without executing the handler, and every new endpoint had to be
inserted into a method that had grown to 438 lines.

The contract is the same one the chain had, made explicit: **rules are matched
in registration order, and the first match wins.** Specific paths must
therefore be registered before generic ones that would also match them. The
difference is that the table is now data — it can be enumerated, asserted
against, and printed.

Stdlib only: no framework, no dependency. The server stays a
``ThreadingHTTPServer``; this only replaces how it picks a handler.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

# A matcher receives the raw path and the same path with any query string
# stripped, and returns the extracted params dict on a match, or None.
Matcher = Callable[[str, str], "dict[str, str] | None"]
# An action receives the handler instance and the matcher's params.
Action = Callable[[object, "dict[str, str]"], None]

_MISS = None


# ── Matcher factories ───────────────────────────────────────────────────────
# Each returns a Matcher. They are deliberately tiny and side-effect free so
# the table can be built at import time and reused across requests.

def exact(*paths: str) -> Matcher:
    """Match one of ``paths`` exactly. Query strings are ignored.

    Replaces both ``path == X`` and the ``path == X or path.startswith(X + "?")``
    idiom the chain used for endpoints that accept query parameters.
    """
    wanted = frozenset(paths)

    def _match(path: str, path_no_query: str) -> dict[str, str] | None:
        return {} if path_no_query in wanted else _MISS

    return _match


def prefix(pre: str) -> Matcher:
    """Match any path starting with ``pre`` (query string ignored)."""
    def _match(path: str, path_no_query: str) -> dict[str, str] | None:
        return {} if path_no_query.startswith(pre) else _MISS

    return _match


def suffix(suf: str) -> Matcher:
    """Match any path ending with ``suf`` (query string ignored)."""
    def _match(path: str, path_no_query: str) -> dict[str, str] | None:
        return {} if path_no_query.endswith(suf) else _MISS

    return _match


def prefix_suffix(pre: str, suf: str) -> Matcher:
    """Match a path that both starts with ``pre`` and ends with ``suf``.

    Replaces ``startswith(X) and endswith(Y)`` — the chain used this for
    sub-resource routes like ``/api/providers/{name}/toggle``.
    """
    def _match(path: str, path_no_query: str) -> dict[str, str] | None:
        if path_no_query.startswith(pre) and path_no_query.endswith(suf):
            return {}
        return _MISS

    return _match


def regex(pattern: str) -> Matcher:
    """Match a full path against ``pattern``, exposing named groups as params.

    Named groups are required for anything the action needs::

        regex(r"^/api/providers/(?P<name>[^/]+)/failures$")

    Use ``(?P<name>[^/]+)`` rather than ``(.+)`` so a trailing segment cannot
    swallow slashes and accidentally match a deeper path.
    """
    compiled = re.compile(pattern)

    def _match(path: str, path_no_query: str) -> dict[str, str] | None:
        m = compiled.match(path_no_query)
        return m.groupdict() if m else _MISS

    return _match


def any_of(*matchers: Matcher) -> Matcher:
    """Match if any of ``matchers`` match. First match's params win."""
    def _match(path: str, path_no_query: str) -> dict[str, str] | None:
        for m in matchers:
            got = m(path, path_no_query)
            if got is not None:
                return got
        return _MISS

    return _match


# ── The table ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Rule:
    method: str
    name: str
    matcher: Matcher
    action: Action
    note: str = ""


class RouteTable:
    """An ordered list of (method, matcher, action) rules.

    ``add`` appends; ``dispatch`` walks in order and stops at the first match,
    which mirrors the ``if/elif`` semantics this replaces.
    """

    def __init__(self) -> None:
        self._rules: list[Rule] = []

    def add(self, method: str, name: str, matcher: Matcher, action: Action,
            note: str = "") -> "RouteTable":
        self._rules.append(Rule(method.upper(), name, matcher, action, note))
        return self

    def get(self, name: str, matcher: Matcher, action: Action, note: str = ""):
        return self.add("GET", name, matcher, action, note)

    def post(self, name: str, matcher: Matcher, action: Action, note: str = ""):
        return self.add("POST", name, matcher, action, note)

    def put(self, name: str, matcher: Matcher, action: Action, note: str = ""):
        return self.add("PUT", name, matcher, action, note)

    def delete(self, name: str, matcher: Matcher, action: Action, note: str = ""):
        return self.add("DELETE", name, matcher, action, note)

    # ── introspection: the whole point of the exercise ──
    @property
    def rules(self) -> tuple[Rule, ...]:
        return tuple(self._rules)

    def describe(self, method: str | None = None) -> list[str]:
        """Return ``['METHOD  name', ...]`` — a readable routing inventory."""
        return ["%-6s %s" % (r.method, r.name) for r in self._rules
                if method is None or r.method == method.upper()]

    def count(self, method: str | None = None) -> int:
        return sum(1 for r in self._rules
                   if method is None or r.method == method.upper())

    # ── dispatch ──
    def dispatch(self, handler, method: str) -> bool:
        """Run the first matching rule for ``method``.

        Returns True if a rule handled the request, False if nothing matched
        (the caller decides what a miss means — the handler sends a 404).
        """
        path = handler.path
        path_no_query = path.split("?", 1)[0]
        method = method.upper()
        for rule in self._rules:
            if rule.method != method:
                continue
            params = rule.matcher(path, path_no_query)
            if params is not None:
                rule.action(handler, params)
                return True
        return False
