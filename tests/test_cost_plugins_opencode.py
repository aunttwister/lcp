"""Tests for the OpenCode cost tracking plugin.

Creates a temporary SQLite database mirroring opencode.db's schema,
injects sample messages, and verifies the plugin reads them correctly.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from src.api.cost_plugins.opencode import OpenCodeCostPlugin, _OPENCODE_PRICING, _FREE_MODELS


# ── Helpers ─────────────────────────────────────────────────────────────────

def _create_opencode_db(path: str) -> None:
    """Create an opencode.db with session + message tables and sample rows."""
    import sqlite3
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE session (
            id TEXT PRIMARY KEY,
            project_id TEXT,
            slug TEXT,
            directory TEXT,
            title TEXT,
            version TEXT,
            time_created INTEGER,
            time_updated INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE message (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            time_created INTEGER,
            time_updated INTEGER,
            data TEXT
        )
    """)
    conn.execute("""
        INSERT INTO session (id, project_id, slug, title, time_created)
        VALUES ('sess-1', 'proj-a', 'test-session', 'Test Session', 1760000000)
    """)
    conn.commit()
    conn.close()


def _insert_message(
    db_path: str,
    msg_id: str,
    session_id: str,
    timestamp: int,
    role: str,
    model_id: str,
    provider_id: str,
    input_tok: int = 0,
    output_tok: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
    reasoning: int = 0,
    cost: float | None = None,
) -> None:
    """Insert a single message into the test DB."""
    import sqlite3
    data = {
        "id": msg_id,
        "sessionID": session_id,
        "role": role,
        "model": {"providerID": provider_id, "modelID": model_id},
        "tokens": {
            "input": input_tok,
            "output": output_tok,
            "reasoning": reasoning,
            "cache": {"read": cache_read, "write": cache_write},
        },
        "time": {"created": timestamp, "completed": timestamp},
    }
    if cost is not None:
        data["cost"] = cost

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?, ?, ?, ?, ?)",
        (msg_id, session_id, timestamp, timestamp, json.dumps(data)),
    )
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def opencode_db(tmp_path):
    """Create a populated opencode.db and return its path."""
    db_path = str(tmp_path / "opencode.db")
    _create_opencode_db(db_path)
    return db_path


@pytest.fixture
def plugin(opencode_db):
    """Return an OpenCode plugin pointing at the test DB."""
    return OpenCodeCostPlugin(db_path=opencode_db)


# ═══════════════════════════════════════════════════════════════════════
# Plugin identity & pricing
# ═══════════════════════════════════════════════════════════════════════

class TestOpenCodeIdentity:
    def test_provider_name(self, plugin):
        assert plugin.provider_name == "opencode"

    def test_supported_models_includes_all(self, plugin):
        models = plugin.get_supported_models()
        assert "deepseek-v4-pro" in models
        assert "deepseek-v4-flash" in models
        for free in _FREE_MODELS:
            assert free in models

    def test_get_pricing_pro(self, plugin):
        p = plugin.get_pricing("deepseek-v4-pro")
        assert p == _OPENCODE_PRICING["deepseek-v4-pro"]

    def test_get_pricing_flash(self, plugin):
        p = plugin.get_pricing("deepseek-v4-flash")
        assert p == _OPENCODE_PRICING["deepseek-v4-flash"]

    def test_get_pricing_free_models(self, plugin):
        for free in _FREE_MODELS:
            p = plugin.get_pricing(free)
            assert p == {"cache_hit": 0.0, "cache_miss": 0.0, "output": 0.0}

    def test_get_pricing_unknown(self, plugin):
        assert plugin.get_pricing("nonexistent") is None


# ═══════════════════════════════════════════════════════════════════════
# calculate_cost
# ═══════════════════════════════════════════════════════════════════════

class TestOpenCodeCalculateCost:
    def test_v4_pro_cost(self, plugin):
        cost = plugin.calculate_cost("deepseek-v4-pro", {
            "prompt_cache_hit_tokens": 500_000,
            "prompt_cache_miss_tokens": 1_000_000,
            "completion_tokens": 200_000,
        })
        expected = (
            (500_000 / 1_000_000) * 0.003625
            + (1_000_000 / 1_000_000) * 0.435
            + (200_000 / 1_000_000) * 0.87
        )
        assert cost == pytest.approx(expected)

    def test_free_model_zero_cost(self, plugin):
        cost = plugin.calculate_cost("qwen3-coder", {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 500_000,
        })
        assert cost == 0.0

    def test_unknown_model_returns_none(self, plugin):
        cost = plugin.calculate_cost("unknown", {"prompt_tokens": 100})
        assert cost is None

    def test_cache_miss_fallback(self, plugin):
        """When cache_hit and cache_miss are both 0/absent, fall back to prompt_tokens."""
        cost = plugin.calculate_cost("deepseek-v4-pro", {
            "completion_tokens": 200_000,
        })
        # cache_hit=0, cache_miss=0 → cache_miss=usage.get("prompt_tokens", 0)=0
        # cost = (0)*0.003625 + (0)*0.435 + (200K/1M)*0.87 = 0.174
        assert cost == pytest.approx(0.174)


# ═══════════════════════════════════════════════════════════════════════
# fetch_usage (reading from SQLite)
# ═══════════════════════════════════════════════════════════════════════

class TestOpenCodeFetchUsage:
    def test_empty_db_returns_empty(self, plugin):
        assert plugin.fetch_usage() == []

    def test_single_message(self, plugin, opencode_db):
        _insert_message(
            opencode_db, "msg-1", "sess-1",
            timestamp=1760100000,  # 2025-10-10
            role="assistant", model_id="deepseek-v4-pro",
            provider_id="opencode",
            input_tok=1000, output_tok=500,
        )
        result = plugin.fetch_usage()
        assert len(result) == 1
        row = result[0]
        assert row["date"] == "2025-10-10"
        assert row["model"] == "deepseek-v4-pro"
        assert row["provider"] == "opencode"
        assert row["prompt_tokens"] == 1000
        assert row["completion_tokens"] == 500
        assert row["request_count"] == 1

    def test_multiple_messages_same_day(self, plugin, opencode_db):
        ts = 1760100000  # 2025-10-10
        _insert_message(opencode_db, "msg-1", "sess-1", ts,
                        "assistant", "deepseek-v4-pro", "opencode",
                        input_tok=1000, output_tok=200)
        _insert_message(opencode_db, "msg-2", "sess-1", ts + 60,
                        "assistant", "deepseek-v4-pro", "opencode",
                        input_tok=500, output_tok=100)
        result = plugin.fetch_usage()
        assert len(result) == 1
        row = result[0]
        assert row["prompt_tokens"] == 1500  # 1000 + 500
        assert row["completion_tokens"] == 300  # 200 + 100
        assert row["request_count"] == 2

    def test_user_messages_ignored(self, plugin, opencode_db):
        _insert_message(opencode_db, "msg-1", "sess-1", 1760100000,
                        "user", "deepseek-v4-pro", "opencode",
                        input_tok=100, output_tok=0)
        _insert_message(opencode_db, "msg-2", "sess-1", 1760100000,
                        "assistant", "deepseek-v4-pro", "opencode",
                        input_tok=200, output_tok=100)
        result = plugin.fetch_usage()
        assert len(result) == 1
        assert result[0]["prompt_tokens"] == 200

    def test_free_model_no_cost(self, plugin, opencode_db):
        _insert_message(opencode_db, "msg-1", "sess-1", 1760100000,
                        "assistant", "qwen3-coder", "opencode",
                        input_tok=5000, output_tok=2000)
        result = plugin.fetch_usage()
        assert result[0]["cost"] == 0.0

    def test_date_filtering(self, plugin, opencode_db):
        _insert_message(opencode_db, "msg-1", "sess-1", 1760000000,  # 2025-10-09
                        "assistant", "deepseek-v4-pro", "opencode",
                        input_tok=100, output_tok=10)
        _insert_message(opencode_db, "msg-2", "sess-1", 1760100000,  # 2025-10-10
                        "assistant", "deepseek-v4-pro", "opencode",
                        input_tok=200, output_tok=20)
        result = plugin.fetch_usage(start_date="2025-10-10")
        assert len(result) == 1
        assert result[0]["date"] == "2025-10-10"

        result2 = plugin.fetch_usage(end_date="2025-10-09")
        assert len(result2) == 1
        assert result2[0]["date"] == "2025-10-09"

    def test_no_db_file_returns_empty(self, tmp_path):
        p = OpenCodeCostPlugin(db_path=str(tmp_path / "nonexistent.db"))
        assert p.fetch_usage() == []

    def test_missing_data_column_handled_gracefully(self, plugin, opencode_db):
        """Messages with no tokens field should be skipped."""
        import sqlite3
        conn = sqlite3.connect(opencode_db)
        conn.execute(
            "INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?, ?, ?, ?, ?)",
            ("bad-msg", "sess-1", 1760100000, 1760100000,
             json.dumps({"id": "bad-msg", "role": "assistant"})),
        )
        conn.commit()
        conn.close()
        # Should not crash, and return empty since no usable tokens
        result = plugin.fetch_usage()
        assert len(result) == 0

    def test_fallback_provider_id(self, plugin, opencode_db):
        """Messages with modelID at top level should still work."""
        import sqlite3
        conn = sqlite3.connect(opencode_db)
        conn.execute(
            "INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?, ?, ?, ?, ?)",
            ("msg-legacy", "sess-1", 1760100000, 1760100000,
             json.dumps({
                 "id": "msg-legacy",
                 "sessionID": "sess-1",
                 "role": "assistant",
                 "modelID": "deepseek-v4-flash",
                 "providerID": "opencode",
                 "tokens": {"input": 50, "output": 25, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                 "time": {"created": 1760100000},
             })),
        )
        conn.commit()
        conn.close()
        result = plugin.fetch_usage()
        assert len(result) == 1
        assert result[0]["model"] == "deepseek-v4-flash"


# ═══════════════════════════════════════════════════════════════════════
# fetch_balance
# ═══════════════════════════════════════════════════════════════════════

    def test_db_read_failure_returns_empty(self, tmp_path):
        """Database read failure (sqlite3.Error) returns empty list."""
        import sqlite3
        db_path = tmp_path / "opencode.db"
        # Create a valid SQLite file so os.path.isfile passes
        conn = sqlite3.connect(str(db_path))
        conn.close()
        plugin = OpenCodeCostPlugin(db_path=str(db_path))
        with patch("sqlite3.connect", side_effect=sqlite3.Error("corrupt")):
            result = plugin.fetch_usage()
            assert result == []

    def test_bad_json_data_is_skipped(self, plugin, opencode_db):
        """Messages with invalid JSON in data column are skipped."""
        import sqlite3
        conn = sqlite3.connect(opencode_db)
        conn.execute(
            "INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?, ?, ?, ?, ?)",
            ("bad-json", "sess-1", 1760100000, 1760100000, "not valid json {{{"),
        )
        conn.commit()
        conn.close()
        result = plugin.fetch_usage()
        # Only the bad JSON message exists (no valid assistant messages with tokens)
        assert result == []


class TestOpenCodeFetchBalance:
    def test_balance_always_none(self, plugin):
        assert plugin.fetch_balance() is None


class TestOpenCodeFetchSummary:
    """Tests for the rich summary (daily/weekly/monthly) from local DB."""

    def test_summary_empty_db(self, plugin):
        """Empty DB returns zeros for all periods."""
        result = plugin.fetch_summary()
        assert result is not None
        for period in ("daily", "weekly", "monthly"):
            assert result[period]["tokens"] == 0
            assert result[period]["cost"] == 0.0
            assert result[period]["requests"] == 0

    def test_summary_with_data(self, opencode_db):
        """Messages from today should appear in daily/weekly/monthly aggregates."""
        import time as _time
        now_ts = int(_time.time())
        _insert_message(
            opencode_db, "msg-1", "sess-1", now_ts - 3600,  # 1 hour ago
            "assistant", "deepseek-v4-pro", "opencode",
            input_tok=1000, output_tok=500, cache_read=200,
        )
        _insert_message(
            opencode_db, "msg-2", "sess-1", now_ts - 1800,  # 30 min ago
            "assistant", "deepseek-v4-pro", "opencode",
            input_tok=2000, output_tok=1000,
            cost=0.005,
        )
        plugin = OpenCodeCostPlugin(db_path=opencode_db)
        result = plugin.fetch_summary()
        assert result is not None

        # Daily should capture both messages
        # msg-1: _calc_msg_cost = 1000/1M*0.435 + 200/1M*0.003625 + 500/1M*0.87 = 0.000870725
        # msg-2: explicit cost = 0.005
        # total = 0.005870725 → rounded to 8 decimal places
        assert result["daily"]["tokens"] == 4500  # 1000+500+2000+1000
        assert result["daily"]["cost"] == pytest.approx(0.00587073, rel=1e-5)
        assert result["daily"]["requests"] == 2

        # Weekly should be same as daily (both within 7 days)
        assert result["weekly"]["tokens"] >= 4500
        assert result["weekly"]["requests"] >= 2

        # Monthly should be same
        assert result["monthly"]["tokens"] >= 4500

    def test_summary_ignores_non_assistant(self, opencode_db):
        """User messages should not be counted in summary."""
        import time as _time
        now_ts = int(_time.time())
        _insert_message(
            opencode_db, "msg-1", "sess-1", now_ts - 60,
            "user", "deepseek-v4-pro", "opencode",
            input_tok=5000, output_tok=0,
        )
        plugin = OpenCodeCostPlugin(db_path=opencode_db)
        result = plugin.fetch_summary()
        assert result is not None
        assert result["daily"]["tokens"] == 0
        assert result["daily"]["requests"] == 0

    def test_summary_none_when_db_missing(self, tmp_path):
        """Should return None when the DB file doesn't exist."""
        plugin = OpenCodeCostPlugin(db_path=str(tmp_path / "no_such.db"))
        assert plugin.fetch_summary() is None

    def test_summary_none_on_db_error(self, opencode_db):
        """Corrupt/inaccessible DB should return None gracefully."""
        import sqlite3
        with patch("src.api.cost_plugins.opencode.sqlite3.connect",
                   side_effect=sqlite3.Error("boom")):
            plugin = OpenCodeCostPlugin(db_path=opencode_db)
            result = plugin.fetch_summary()
            assert result is None
