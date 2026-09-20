"""The shared table module — one definition of ordering + pagination for LCP logs.

Every log surface in LCP (conversations, raw requests, provider routing
decisions, the board decisions ledger) resolves its page window and its ORDER BY
through this module, and renders through ``static/js/logtable.js``. It exists so
that "what order are these rows in, and can I page through them" is answered in
exactly one place instead of once per view.

Two invariants this module owns:

1. **Newest first by default.** Every surface declares its sorts as named keys;
   the first one is the default and it is always the newest-to-oldest ordering.
   A view that says nothing about sorting still comes back newest-first, and a
   tie-break keeps the order stable across pages (an unstable ORDER BY silently
   duplicates and drops rows once you paginate).
2. **Sort keys are an allow-list, never SQL.** A caller sends ``?sort=oldest``;
   the module looks the key up in the view's own table of
   ``(key, label, order fragment)`` and uses the fragment authored here. An
   unknown or hostile key falls back to the default — it is never interpolated.

The payload the views expose as ``filter`` is what the client module renders its
controls from, so a new surface gets pagination + a sort control by declaring its
sorts and calling :func:`state` — no template or JavaScript changes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# ``per`` values offered in the UI. ``all`` is honoured by the API (and by the
# module) but deliberately NOT offered as a button: a log with 54k rows must not
# be one click away from a 20 MB page.
PER_CHOICES: Tuple[str, ...] = ("20", "50", "100")
DEFAULT_PER = "20"
PER_MAX = 500

# A sort spec is (key, label, order fragment). The fragment is a static string
# authored in each view module — it is the allow-list, not user input.
SortSpec = Tuple[str, str, str]


def parse_per(raw: Any) -> Any:
    """``?per=`` → an int page size, or the string ``"all"``.

    Unknown/garbage values fall back to the default rather than raising: a bad
    query string must degrade the page, never 500 it.
    """
    value = str(raw if raw is not None else DEFAULT_PER).strip().lower()
    if value == "all":
        return "all"
    try:
        return max(1, min(PER_MAX, int(value)))
    except (TypeError, ValueError):
        return int(DEFAULT_PER)


def parse_page(raw: Any) -> int:
    """``?page=`` → a 1-based page number (clamped later against ``pages``)."""
    try:
        return max(1, int(str(raw or 1).strip()))
    except (TypeError, ValueError):
        return 1


def resolve_sort(sorts: Sequence[SortSpec], requested: Any) -> SortSpec:
    """Pick the sort spec for ``requested``, defaulting to the first (newest)."""
    keys = [s[0] for s in sorts]
    key = str(requested or "").strip()
    for spec in sorts:
        if spec[0] == key:
            return spec
    return sorts[0]


def paginate(total: int, per: Any, page: int) -> Tuple[int, int, int]:
    """``(page, pages, offset)`` for a total row count.

    ``per == "all"`` is one page holding everything. The page number is clamped
    to the real range, so ``?page=9999`` returns the last page rather than an
    empty table.
    """
    total = max(0, int(total or 0))
    if per == "all":
        return 1, 1, 0
    pages = max(1, -(-total // per))  # ceil
    page = max(1, min(page, pages))
    return page, pages, (page - 1) * per


def row_window(total: int, per: Any, page: int) -> Tuple[int, int]:
    """The inclusive 1-based row range on this page — the "Showing 21–40 of N"."""
    total = max(0, int(total or 0))
    if total == 0:
        return 0, 0
    offset = 1 + (page - 1) * per if per != "all" else 1
    last = total if per == "all" else min(total, page * per)
    return offset, last


def state(
    params: Optional[Dict[str, Any]] = None,
    sorts: Sequence[SortSpec] = (("newest", "Newest first", "id DESC"),),
    total: int = 0,
) -> Dict[str, Any]:
    """Resolve one request's table state.

    Returns the ordering fragment (``order_sql``) plus the pagination window the
    caller needs to build ``LIMIT ? OFFSET ?``. The same shape is mirrored into
    the ``filter`` payload by :func:`filter_payload`.
    """
    params = params or {}
    per = parse_per(params.get("per"))
    page = parse_page(params.get("page"))
    sort_key, sort_label, order_sql = resolve_sort(sorts, params.get("sort"))
    page, pages, offset = paginate(total, per, page)
    first, last = row_window(total, per, page)
    return {
        "per": per,
        "page": page,
        "pages": pages,
        "offset": offset,
        "sort": sort_key,
        "sort_label": sort_label,
        "order_sql": order_sql,
        "sorts": [{"key": k, "label": lbl} for k, lbl, _ in sorts],
        "per_choices": list(PER_CHOICES),
        "total": int(total or 0),
        "first": first,
        "last": last,
    }


def filter_payload(st: Dict[str, Any], total: int, **extra: Any) -> Dict[str, Any]:
    """The ``filter`` dict every log view returns and the client renders from.

    ``per`` stays a string (``"20"`` / ``"all"``) because it round-trips straight
    back into the query string.
    """
    payload: Dict[str, Any] = {
        "per": str(st["per"]),
        "page": st["page"],
        "pages": st["pages"],
        "total": int(total or 0),
        "first": st["first"],
        "last": st["last"],
        "sort": st["sort"],
        "sort_label": st["sort_label"],
        "sorts": st["sorts"],
        "per_choices": st["per_choices"],
    }
    payload.update(extra)
    return payload


def limit_clause(st: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """``(" LIMIT ? OFFSET ?", [per, offset])`` — or nothing at all for ``all``."""
    if st["per"] == "all":
        return "", []
    return " LIMIT ? OFFSET ?", [st["per"], st["offset"]]
