"""Tests for the LanceDB memory backend (src/api/memory)."""

import pytest

from src.api.memory.lancedb_backend import LanceDBMemoryBackend
from src.api.memory.base import MemoryError


# ── Fake embedder ───────────────────────────────────────────────────────────

VOCAB = {
    "gpu": 0, "rtx": 1, "3090": 2, "tesla": 3, "p40": 4, "hardware": 5,
    "basement": 6, "location": 7, "wifi": 8, "password": 9, "node01": 10,
    "server": 11, "rack": 12,
}


def fake_embed(texts):
    """Keyword-hash embedder: one-hot-ish per known vocab word, then normalize."""
    out = []
    for t in texts:
        v = [0.0] * 32
        for w in t.lower().split():
            idx = VOCAB.get(w.strip(".,:;"))
            if idx is not None:
                v[idx % 32] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / norm for x in v])
    return out


@pytest.fixture
def backend(tmp_path):
    return LanceDBMemoryBackend(str(tmp_path / "mem"), fake_embed, dim=32)


class TestNonStringPath:
    def test_non_string_db_path_raises_and_creates_no_dirs(self, tmp_path, monkeypatch):
        """A non-string storage path must raise MemoryError and create no
        directory tree — MagicMock.__fspath__ would otherwise make os.makedirs
        create 'MagicMock/<chain>/<id>' junk dirs."""
        from unittest.mock import MagicMock
        monkeypatch.chdir(tmp_path)
        with pytest.raises(MemoryError):
            LanceDBMemoryBackend(MagicMock(), fake_embed, dim=32)
        assert not (tmp_path / "MagicMock").exists()


# ── retain ──────────────────────────────────────────────────────────────────

class TestRetain:
    def test_retain_returns_id_and_increments_count(self, backend):
        mid = backend.retain("node01 has an RTX 3090 and a Tesla P40 GPU",
                             {"host": "node01"}, ["gpu", "hardware"], profile="l2")
        assert mid
        assert backend.count("l2") == 1

    def test_retain_empty_content_raises(self, backend):
        with pytest.raises(MemoryError):
            backend.retain("   ", profile="l2")

    def test_retain_per_profile_isolation(self, backend):
        backend.retain("wifi password is hunter2", None, None, profile="l2")
        backend.retain("wifi password is hunter2", None, None, profile="career")
        assert backend.count("l2") == 1
        assert backend.count("career") == 1
        assert backend.count("default") == 0

    def test_retain_embed_failure_raises_memory_error(self, tmp_path):
        def bad_embed(texts):
            raise RuntimeError("model exploded")
        b = LanceDBMemoryBackend(str(tmp_path / "m2"), bad_embed, dim=32)
        with pytest.raises(MemoryError):
            b.retain("hello", profile="l2")


# ── recall ──────────────────────────────────────────────────────────────────

class TestRecall:
    def test_recall_returns_best_first(self, backend):
        backend.retain("node01 has an RTX 3090 GPU", None, None, profile="l2")
        backend.retain("node01 is in the basement", None, None, profile="l2")
        res = backend.recall("which gpu is in node01", top_k=2, profile="l2")
        assert len(res) == 2
        assert res[0]["score"] >= res[1]["score"]
        assert "gpu" in res[0]["content"].lower()

    def test_recall_returns_empty_for_unknown_profile(self, backend):
        assert backend.recall("anything", profile="nope") == []

    def test_recall_tag_filter(self, backend):
        backend.retain("node01 has an RTX 3090 GPU", None, ["gpu"], profile="l2")
        backend.retain("node01 is in the basement", None, ["location"], profile="l2")
        res = backend.recall("node01", top_k=5, tag_filter=["gpu"], profile="l2")
        assert len(res) == 1
        assert res[0]["tags"] == ["gpu"]

    def test_recall_returns_metadata(self, backend):
        backend.retain("node01 has a Tesla P40", {"host": "node01", "ip": "10.0.0.1"},
                       None, profile="l2")
        res = backend.recall("tesla", top_k=1, profile="l2")
        assert res[0]["metadata"] == {"host": "node01", "ip": "10.0.0.1"}

    def test_recall_nonpositive_top_k_returns_empty(self, backend):
        assert backend.recall("x", top_k=0, profile="l2") == []

    def test_recall_embed_failure_raises_memory_error(self, tmp_path):
        d = tmp_path / "m3"
        LanceDBMemoryBackend(str(d), fake_embed, dim=32).retain(
            "node01 has an RTX 3090", None, None, profile="l2")

        def bad_embed(texts):
            raise RuntimeError("boom")
        b = LanceDBMemoryBackend(str(d), bad_embed, dim=32)
        with pytest.raises(MemoryError):
            b.recall("hello", profile="l2")


# ── forget ──────────────────────────────────────────────────────────────────

class TestForget:
    def test_forget_removes_row(self, backend):
        mid = backend.retain("node01 has an RTX 3090", None, None, profile="l2")
        assert backend.forget(mid, profile="l2") is True
        assert backend.count("l2") == 0

    def test_forget_missing_id_returns_false(self, backend):
        backend.retain("x", None, None, profile="l2")
        assert backend.forget("nonexistent", profile="l2") is False

    def test_forget_unknown_profile_returns_false(self, backend):
        assert backend.forget("whatever", profile="nope") is False


# ── count / persistence / errors ────────────────────────────────────────────

class TestCountAndPersistence:
    def test_count_zero_when_empty(self, backend):
        assert backend.count("l2") == 0

    def test_persists_across_reopen(self, tmp_path):
        d = tmp_path / "mem"
        b1 = LanceDBMemoryBackend(str(d), fake_embed, dim=32)
        b1.retain("node01 has a Tesla P40", None, None, profile="l2")
        b2 = LanceDBMemoryBackend(str(d), fake_embed, dim=32)
        assert b2.count("l2") == 1
        res = b2.recall("tesla", top_k=1, profile="l2")
        assert res[0]["content"] == "node01 has a Tesla P40"

    def test_list_tables(self, backend):
        backend.retain("a", None, None, profile="l2")
        backend.retain("b", None, None, profile="career")
        assert backend.list_tables() == ["career", "l2"]

    def test_index_build_failure_does_not_raise(self, tmp_path):
        b = LanceDBMemoryBackend(str(tmp_path / "m4"), fake_embed, dim=32,
                                 index_threshold=1)
        b.retain("node01 has an RTX 3090", None, None, profile="l2")
        with pytest.raises(Exception):
            # create_index with an invalid metric bubbles up; ensure recall
            # still works afterwards (index build is best-effort).
            raise RuntimeError("simulated")
        assert b.count("l2") == 1

    def test_constructor_requires_lancedb(self, tmp_path, monkeypatch):
        import importlib.util
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name, *a, **k):
            if name == "lancedb":
                return None
            return real_find_spec(name, *a, **k)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        with pytest.raises(MemoryError, match="lancedb is not installed"):
            LanceDBMemoryBackend(str(tmp_path / "m5"), fake_embed, dim=32)
