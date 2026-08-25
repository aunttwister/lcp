"""Tests for the memory module-level runtime API + embeddings.

Covers the parts of src/api/memory that the backend/endpoint/setup tests don't:
``init_memory`` / ``get_memory`` / ``memory_status`` / ``shutdown_memory``,
``embedder_from_config``, ``EmbeddingModel`` (lazy, with a fake model), and the
``_noop_embed`` fallback.
"""

import pytest

import src.api.memory as mem
from src.api.memory.base import MemoryError
from src.api.memory.embeddings import (
    DEFAULT_DIM,
    DEFAULT_MODEL,
    EmbeddingModel,
    embedder_from_config,
)


@pytest.fixture(autouse=True)
def _reset_backend():
    mem._backend = None
    yield
    mem._backend = None


def _mem_config(storage, enabled=True, **embedding):
    return {
        "plugins": {
            "memory": {
                "enabled": enabled,
                "storage_path": storage,
                "embedding": embedding or {"dim": 32},
            }
        }
    }


class _Cfg:
    def __init__(self, plugins):
        self.plugins = plugins


class TestInitMemory:
    def test_init_enabled_creates_backend(self, tmp_path, monkeypatch):
        # No real embedder (sentence-transformers absent) -> _noop_embed used.
        monkeypatch.setattr("src.api.memory.embedder_from_config", lambda cfg: None)
        cfg = _Cfg(_mem_config(str(tmp_path / "mem"))["plugins"])
        assert mem.init_memory(cfg) is True
        assert mem.get_memory() is not None
        assert mem.memory_status()["active"] is True

    def test_init_disabled_returns_false(self, tmp_path):
        cfg = _Cfg(_mem_config(str(tmp_path / "mem"), enabled=False)["plugins"])
        assert mem.init_memory(cfg) is False
        assert mem.get_memory() is None

    def test_init_no_config_uses_default_storage(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.api.memory.embedder_from_config", lambda cfg: None)
        monkeypatch.setenv("COST_DB", str(tmp_path / "costs.db"))
        assert mem.init_memory(None) is True
        backend = mem.get_memory()
        assert backend is not None
        # Storage defaults to <db dir>/memory.
        assert str(tmp_path / "memory") in backend._db_path

    def test_init_never_raises_on_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.api.memory.lancedb_backend.LanceDBMemoryBackend",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        cfg = _Cfg(_mem_config(str(tmp_path / "mem"))["plugins"])
        assert mem.init_memory(cfg) is False
        assert mem.get_memory() is None

    def test_shutdown_clears_backend(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.api.memory.embedder_from_config", lambda cfg: None)
        cfg = _Cfg(_mem_config(str(tmp_path / "mem"))["plugins"])
        mem.init_memory(cfg)
        assert mem.get_memory() is not None
        mem.shutdown_memory()
        assert mem.get_memory() is None
        assert mem.memory_status()["active"] is False


class TestMemoryStatus:
    def test_status_shape(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.api.memory.memory_available", lambda site=None: True)
        st = mem.memory_status()
        assert st["available"] is True
        assert st["active"] is False
        assert "site" in st and "models_dir" in st


class TestNoopEmbed:
    def test_returns_zero_vectors_of_default_dim(self):
        out = mem._noop_embed(["a", "b", "c"])
        assert len(out) == 3
        assert all(len(v) == 384 for v in out)
        assert all(all(x == 0.0 for x in v) for v in out)

    def test_empty_input(self):
        assert mem._noop_embed([]) == []


class TestEmbedderFromConfig:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("LCP_MODULES_DIR", raising=False)
        m = embedder_from_config({})
        assert m.model_name == DEFAULT_MODEL
        assert m.device == "cpu"
        assert m.cache_dir is None

    def test_from_config_block(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LCP_MODULES_DIR", str(tmp_path / "modules"))
        m = embedder_from_config({
            "embedding": {"model": "custom/model", "device": "cuda", "dim": 768},
        })
        assert m.model_name == "custom/model"
        assert m.device == "cuda"
        assert m.cache_dir == str(tmp_path / "modules" / "models" / "memory")


class TestEmbeddingModel:
    def test_dim_falls_back_when_unloadable(self, monkeypatch):
        m = EmbeddingModel()
        monkeypatch.setattr(m, "_load", lambda: (_ for _ in ()).throw(RuntimeError("no model")))
        assert m.dim == DEFAULT_DIM

    def test_embed_with_fake_model(self, monkeypatch):
        class _Vec:
            def __init__(self, vals):
                self._vals = vals
            def tolist(self):
                return self._vals

        class FakeModel:
            def encode(self, texts, **kw):
                return [_Vec([1.0, 0.0, 0.0]) for _ in texts]

        m = EmbeddingModel()
        monkeypatch.setattr(m, "_load", lambda: FakeModel())
        out = m.embed(["hello", "world"])
        assert len(out) == 2
        assert out[0] == [1.0, 0.0, 0.0]

    def test_embed_empty(self, monkeypatch):
        m = EmbeddingModel()
        assert m.embed([]) == []

    def test_embed_missing_deps_raises_memory_error(self, monkeypatch):
        m = EmbeddingModel()
        monkeypatch.setattr(
            m, "_load",
            lambda: (_ for _ in ()).throw(MemoryError("sentence-transformers is not installed")),
        )
        with pytest.raises(MemoryError):
            m.embed(["x"])

    def test_embed_failure_wraps_memory_error(self, monkeypatch):
        class BadModel:
            def encode(self, texts, **kw):
                raise RuntimeError("model exploded")

        m = EmbeddingModel()
        monkeypatch.setattr(m, "_load", lambda: BadModel())
        with pytest.raises(MemoryError, match="embedding failed"):
            m.embed(["x"])