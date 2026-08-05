"""Tests for the llama.cpp cost tracking plugin.

LlamaCppCostPlugin tracks tokens locally via an in-memory dict
persisted to a JSON file. All costs are $0.
"""

import json
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest
from src.api.cost_plugins.llamacpp import LlamaCppCostPlugin


class TestLlamaCppIdentity:
    def test_provider_name(self):
        plugin = LlamaCppCostPlugin()
        assert plugin.provider_name == "llamacpp"

    def test_supported_models_empty(self):
        """llama.cpp can serve any model, so it returns an empty list."""
        plugin = LlamaCppCostPlugin()
        assert plugin.get_supported_models() == []

    def test_get_pricing_zero(self):
        plugin = LlamaCppCostPlugin()
        assert plugin.get_pricing("any-model") == {
            "cache_hit": 0.0, "cache_miss": 0.0, "output": 0.0,
        }

    def test_calculate_cost_always_zero(self):
        plugin = LlamaCppCostPlugin()
        assert plugin.calculate_cost("default", {}) == 0.0
        assert plugin.calculate_cost("default",
            {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}) == 0.0


class TestLlamaCppRecordTokens:
    """record_tokens accumulates tokens in memory and persists to JSON."""

    def test_record_tokens_accumulates(self, tmp_path):
        persist = tmp_path / "usage.json"
        plugin = LlamaCppCostPlugin(persist_path=str(persist))
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        plugin.record_tokens(model="default", prompt_tokens=100, completion_tokens=50)
        assert plugin._daily[today]["default"]["prompt_tokens"] == 100
        assert plugin._daily[today]["default"]["completion_tokens"] == 50
        assert plugin._daily[today]["default"]["request_count"] == 1

    def test_record_tokens_multiple_calls(self, tmp_path):
        persist = tmp_path / "usage.json"
        plugin = LlamaCppCostPlugin(persist_path=str(persist))
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        plugin.record_tokens(model="default", prompt_tokens=100, completion_tokens=50)
        plugin.record_tokens(model="default", prompt_tokens=30, completion_tokens=20)
        assert plugin._daily[today]["default"]["prompt_tokens"] == 130
        assert plugin._daily[today]["default"]["completion_tokens"] == 70
        assert plugin._daily[today]["default"]["request_count"] == 2

    def test_record_tokens_cache_hit(self, tmp_path):
        persist = tmp_path / "usage.json"
        plugin = LlamaCppCostPlugin(persist_path=str(persist))
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        plugin.record_tokens(model="default", prompt_tokens=100, completion_tokens=50,
                             cache_hit_tokens=30)
        assert plugin._daily[today]["default"]["cache_hit_tokens"] == 30

    def test_record_tokens_different_models(self, tmp_path):
        persist = tmp_path / "usage.json"
        plugin = LlamaCppCostPlugin(persist_path=str(persist))
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        plugin.record_tokens(model="llama3", prompt_tokens=100, completion_tokens=50)
        plugin.record_tokens(model="qwen2.5", prompt_tokens=200, completion_tokens=100)
        assert len(plugin._daily[today]) == 2
        assert plugin._daily[today]["llama3"]["prompt_tokens"] == 100
        assert plugin._daily[today]["qwen2.5"]["prompt_tokens"] == 200


class TestLlamaCppPersistence:
    """Verify that token data is written to and loaded from the JSON file."""

    def test_persists_to_json(self, tmp_path):
        persist = tmp_path / "usage.json"
        plugin = LlamaCppCostPlugin(persist_path=str(persist))
        plugin.record_tokens(model="default", prompt_tokens=100, completion_tokens=50)
        assert persist.exists()
        data = json.loads(persist.read_text())
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert data[today]["default"]["prompt_tokens"] == 100

    def test_loads_from_json_on_restart(self, tmp_path):
        persist = tmp_path / "usage.json"
        plugin1 = LlamaCppCostPlugin(persist_path=str(persist))
        plugin1.record_tokens(model="default", prompt_tokens=100, completion_tokens=50)

        plugin2 = LlamaCppCostPlugin(persist_path=str(persist))
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert plugin2._daily[today]["default"]["prompt_tokens"] == 100

    def test_corrupted_json_ignored(self, tmp_path):
        persist = tmp_path / "usage.json"
        persist.write_text("{corrupted")
        plugin = LlamaCppCostPlugin(persist_path=str(persist))
        # Should load gracefully with empty state
        assert plugin._daily == {}

    def test_no_file_starts_empty(self, tmp_path):
        persist = tmp_path / "nonexistent" / "usage.json"
        plugin = LlamaCppCostPlugin(persist_path=str(persist))
        assert plugin._daily == {}


class TestLlamaCppDiscoverModels:
    """discover_models queries llama.cpp /v1/models and extracts metadata."""

    @patch("urllib.request.urlopen")
    def test_extracts_id_and_meta_fields(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "data": [{
                "id": "qwen-27b.gguf",
                "created": 1700000000,
                "owned_by": "llamacpp",
                "object": "model",
                "meta": {
                    "n_ctx": 200192,
                    "n_ctx_train": 262144,
                    "n_params": 27320697856,
                    "ftype": "Q4_K - Medium",
                    "size": 17095778304,
                }
            }]
        }).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        plugin = LlamaCppCostPlugin()
        result = plugin.discover_models("http://localhost:8082/v1")
        assert result is not None
        assert len(result) == 1
        m = result[0]
        assert m["id"] == "qwen-27b.gguf"
        assert m["context_length"] == 200192
        assert m["context_train"] == 262144
        assert m["parameters"] == "27.3B"
        assert m["quantization"] == "Q4_K - Medium"
        assert m["size_bytes"] == 17095778304

    @patch("urllib.request.urlopen")
    def test_handles_models_format(self, mock_urlopen):
        """llama.cpp sometimes returns {"models": [...]} instead of {"data": [...]}."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "models": [{"name": "llama3.gguf", "meta": {"n_ctx": 8192}}]
        }).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        plugin = LlamaCppCostPlugin()
        result = plugin.discover_models("http://test.local/")
        assert result is not None
        assert result[0]["id"] == "llama3.gguf"
        assert result[0]["context_length"] == 8192

    @patch("urllib.request.urlopen")
    def test_falls_back_to_v1_models(self, mock_urlopen):
        """When base has no /v1, tries /models then /v1/models."""
        mock_urlopen.side_effect = Exception("fail")

        plugin = LlamaCppCostPlugin()
        result = plugin.discover_models("http://test.local/")
        assert result is None
        assert mock_urlopen.call_count == 2

    @patch("urllib.request.urlopen")
    def test_skips_v1_fallback_when_already_in_url(self, mock_urlopen):
        """Base already contains /v1 — only try /models on that base."""
        mock_urlopen.side_effect = Exception("fail")

        plugin = LlamaCppCostPlugin()
        plugin.discover_models("http://test.local/v1")
        assert mock_urlopen.call_count == 1

    @patch("urllib.request.urlopen")
    def test_handles_non_dict_entries(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "data": [{"id": "valid.gguf", "meta": {"n_ctx": 4096}}, "bare-string"]
        }).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        plugin = LlamaCppCostPlugin()
        result = plugin.discover_models("http://test.local/v1")
        assert len(result) == 2
        assert result[0]["id"] == "valid.gguf"
        assert result[1]["id"] == "bare-string"


class TestLlamaCppFetchUsage:
    def test_empty_returns_empty_list(self, tmp_path):
        plugin = LlamaCppCostPlugin(persist_path=str(tmp_path / "empty.json"))
        assert plugin.fetch_usage() == []

    def test_returns_recorded_data(self, tmp_path):
        persist = tmp_path / "usage.json"
        plugin = LlamaCppCostPlugin(persist_path=str(persist))
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        plugin.record_tokens(model="default", prompt_tokens=100, completion_tokens=50,
                             cache_hit_tokens=20)

        result = plugin.fetch_usage()
        assert len(result) == 1
        row = result[0]
        assert row["date"] == today
        assert row["model"] == "default"
        assert row["provider"] == "llamacpp"
        assert row["prompt_tokens"] == 100
        assert row["completion_tokens"] == 50
        assert row["cache_hit_tokens"] == 20
        # cache_miss_tokens is set to prompt_tokens in fetch_usage()
        assert row["cache_miss_tokens"] == 100
        assert row["cost"] == 0.0
        assert row["request_count"] == 1

    def test_date_filtering(self, tmp_path):
        persist = tmp_path / "usage.json"
        plugin = LlamaCppCostPlugin(persist_path=str(persist))

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

        plugin._daily[yesterday] = {
            "default": {"prompt_tokens": 50, "completion_tokens": 25,
                        "cache_hit_tokens": 0, "request_count": 1}
        }
        plugin._daily[today] = {
            "default": {"prompt_tokens": 100, "completion_tokens": 50,
                        "cache_hit_tokens": 0, "request_count": 2}
        }

        result_today = plugin.fetch_usage(start_date=today)
        assert len(result_today) == 1
        assert result_today[0]["date"] == today

        result_yesterday = plugin.fetch_usage(end_date=yesterday)
        assert len(result_yesterday) == 1
        assert result_yesterday[0]["date"] == yesterday


class TestLlamaCppFetchBalance:
    def test_balance_always_none(self):
        plugin = LlamaCppCostPlugin()
        assert plugin.fetch_balance() is None


class TestLlamaCppOnStartupShutdown:
    def test_persist_oserror_swallowed(self, tmp_path):
        """_persist should not crash when writing fails (OSError)."""
        from pathlib import Path
        with patch.object(Path, 'mkdir', side_effect=OSError("readonly")):
            plugin = LlamaCppCostPlugin(persist_path=str(tmp_path / "usage.json"))
            plugin.record_tokens(model="default", prompt_tokens=5, completion_tokens=5)
            # Should not raise

    def test_on_startup_noop(self, tmp_path):
        persist = tmp_path / "usage.json"
        plugin = LlamaCppCostPlugin(persist_path=str(persist))
        plugin.on_startup()  # should not raise
        assert plugin._daily == {}

    def test_on_shutdown_persists(self, tmp_path):
        """on_shutdown should persist the usage data to disk."""
        persist = tmp_path / "usage.json"
        plugin = LlamaCppCostPlugin(persist_path=str(persist))
        # Data in memory, no file yet
        assert not persist.exists()
        plugin.record_tokens(model="default", prompt_tokens=10, completion_tokens=5)
        # record_tokens calls _persist, so file exists now — clear it to test on_shutdown
        persist.unlink()
        plugin.on_shutdown()
        assert persist.exists()
        data = json.loads(persist.read_text())
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert data[today]["default"]["prompt_tokens"] == 10

    def test_on_shutdown_no_persist_path(self, tmp_path):
        """on_shutdown should not crash when no persist_path is set, using default."""
        plugin = LlamaCppCostPlugin(persist_path=str(tmp_path / "usage.json"))
        plugin.record_tokens(model="default", prompt_tokens=5, completion_tokens=5)
        plugin.on_shutdown()  # should not raise

    def test_on_shutdown_persist_failure_swallowed(self, tmp_path):
        """on_shutdown should not crash if persist fails (missing parent dir)."""
        plugin = LlamaCppCostPlugin(persist_path=str(tmp_path / "no_dir" / "usage.json"))
        plugin._daily["2025-01-01"] = {
            "default": {"prompt_tokens": 5, "completion_tokens": 5,
                        "cache_hit_tokens": 0, "request_count": 1}
        }
        plugin.on_shutdown()  # should not raise
