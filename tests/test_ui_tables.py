"""Tests for the shared log-table module (``src/ui/tables.py``).

The module owns two things every log view depends on: the page window, and the
ORDER BY. These tests pin the invariants that make it safe to be the *only*
place those are decided — newest-first by default, an allow-list instead of
interpolated SQL, and a page number that clamps rather than returns nothing.
"""

import pytest

from src.ui import tables

SORTS = (
    ("newest", "Newest first", "ts DESC, id DESC"),
    ("oldest", "Oldest first", "ts ASC, id DESC"),
)


class TestParsePer:
    def test_default(self):
        assert tables.parse_per(None) == 20

    def test_valid(self):
        assert tables.parse_per("50") == 50
        assert tables.parse_per(100) == 100

    def test_all(self):
        assert tables.parse_per("all") == "all"
        assert tables.parse_per("ALL") == "all"

    def test_garbage_falls_back(self):
        # A bad query string must degrade the page, never 500 it.
        assert tables.parse_per("bogus") == 20
        assert tables.parse_per("") == 20

    def test_clamped_to_range(self):
        assert tables.parse_per("9999") == tables.PER_MAX
        assert tables.parse_per("0") == 1
        assert tables.parse_per("-5") == 1


class TestParsePage:
    def test_default_and_valid(self):
        assert tables.parse_page(None) == 1
        assert tables.parse_page("7") == 7

    def test_garbage_and_negative(self):
        assert tables.parse_page("nope") == 1
        assert tables.parse_page("-3") == 1


class TestResolveSort:
    def test_named_key(self):
        assert tables.resolve_sort(SORTS, "oldest")[0] == "oldest"

    def test_default_is_first_entry(self):
        assert tables.resolve_sort(SORTS, None)[0] == "newest"
        assert tables.resolve_sort(SORTS, "")[0] == "newest"

    def test_unknown_key_falls_back_to_default(self):
        assert tables.resolve_sort(SORTS, "nonsense")[0] == "newest"

    def test_order_fragment_is_the_allow_listed_one(self):
        # The fragment that reaches SQL is the module's, not the caller's.
        assert tables.resolve_sort(SORTS, "ts DESC; DROP TABLE requests")[2] == \
            "ts DESC, id DESC"


class TestPaginate:
    def test_empty(self):
        assert tables.paginate(0, 20, 1) == (1, 1, 0)

    def test_exact_multiple(self):
        assert tables.paginate(40, 20, 2) == (2, 2, 20)

    def test_partial_last_page(self):
        assert tables.paginate(41, 20, 3) == (3, 3, 40)

    def test_page_clamps_to_last(self):
        assert tables.paginate(25, 20, 99) == (2, 2, 20)

    def test_all_is_one_page(self):
        assert tables.paginate(54244, "all", 1) == (1, 1, 0)


class TestRowWindow:
    def test_zero_total(self):
        assert tables.row_window(0, 20, 1) == (0, 0)

    def test_middle_page(self):
        assert tables.row_window(95, 20, 2) == (21, 40)

    def test_last_page_truncates(self):
        assert tables.row_window(95, 20, 5) == (81, 95)

    def test_all(self):
        assert tables.row_window(95, "all", 1) == (1, 95)


class TestState:
    def test_defaults_to_newest_first(self):
        st = tables.state({}, SORTS, 100)
        assert st["sort"] == "newest"
        assert st["order_sql"] == "ts DESC, id DESC"
        assert st["per"] == 20
        assert st["page"] == 1
        assert st["pages"] == 5

    def test_page_and_per_from_params(self):
        st = tables.state({"per": "10", "page": "3"}, SORTS, 100)
        assert (st["per"], st["page"], st["offset"]) == (10, 3, 20)
        assert (st["first"], st["last"]) == (21, 30)

    def test_sort_key_selects_its_fragment(self):
        st = tables.state({"sort": "oldest"}, SORTS, 100)
        assert st["order_sql"] == "ts ASC, id DESC"

    def test_sorts_exposed_for_the_client(self):
        st = tables.state({}, SORTS, 0)
        assert st["sorts"] == [{"key": "newest", "label": "Newest first"},
                               {"key": "oldest", "label": "Oldest first"}]

    def test_per_choices_exclude_all(self):
        # "all" is honoured by the API but must never be a one-click 20 MB page.
        assert "all" not in tables.state({}, SORTS, 0)["per_choices"]

    def test_total_coerced_from_none(self):
        assert tables.state({}, SORTS, None)["total"] == 0


class TestFilterPayload:
    def test_shape(self):
        st = tables.state({"per": "50", "sort": "oldest"}, SORTS, 120)
        f = tables.filter_payload(st, 120, profile="l2")
        assert f["per"] == "50"
        assert f["sort"] == "oldest"
        assert f["total"] == 120
        assert (f["first"], f["last"]) == (1, 50)
        assert f["profile"] == "l2"
        assert [s["key"] for s in f["sorts"]] == ["newest", "oldest"]

    def test_per_all_round_trips_as_string(self):
        st = tables.state({"per": "all"}, SORTS, 9)
        assert tables.filter_payload(st, 9)["per"] == "all"


class TestLimitClause:
    def test_paged(self):
        st = tables.state({"per": "20", "page": "3"}, SORTS, 100)
        sql, args = tables.limit_clause(st)
        assert sql == " LIMIT ? OFFSET ?"
        assert args == [20, 40]

    def test_all_emits_nothing(self):
        st = tables.state({"per": "all"}, SORTS, 100)
        assert tables.limit_clause(st) == ("", [])
