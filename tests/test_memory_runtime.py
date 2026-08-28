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

    def test_init_mock_config_creates_no_dirs(self, tmp_path, monkeypatch):
        """A MagicMock config (plugins.get().get().strip() resolving to the
        storage path) must NOT create a 'MagicMock/...' junk directory tree.
        Regresses the accidental-junk bug where os.makedirs accepted the
        mock's __fspath__ and silently created real dirs."""
        from unittest.mock import MagicMock
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("COST_DB", raising=False)
        assert mem.init_memory(MagicMock()) is False
        assert not (tmp_path / "MagicMock").exists()

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

    def test_status_baked_not_removable(self, monkeypatch):
        """Baked image: deps importable from global site-packages too, so
        deleting the module dir (Remove) would be a no-op."""
        monkeypatch.setattr("src.api.memory.memory_available", lambda site=None: True)
        st = mem.memory_status()
        assert st["available"] is True
        assert st["removable"] is False

    def test_status_runtime_install_removable(self, monkeypatch):
        """Lean image after a runtime install: deps importable ONLY via the
        module --target dir, so the module can be removed/reinstalled."""
        monkeypatch.setattr(
            "src.api.memory.memory_available",
            lambda site=None: site is not None,
        )
        st = mem.memory_status()
        assert st["available"] is True
        assert st["removable"] is True

    def test_status_not_installed(self, monkeypatch):
        monkeypatch.setattr("src.api.memory.memory_available", lambda site=None: False)
        st = mem.memory_status()
        assert st["available"] is False
        assert st["removable"] is False


class TestRouterStatus:
    def test_router_status_shape(self, monkeypatch):
        monkeypatch.setattr("src.api.memory.router_available", lambda site=None: True)
        st = mem.router_status()
        assert st["available"] is True
        assert "site" in st and st["site"].endswith("router")
        assert "models_dir" in st

    def test_router_status_unavailable(self, monkeypatch):
        monkeypatch.setattr("src.api.memory.router_available", lambda site=None: False)
        st = mem.router_status()
        assert st["available"] is False
        assert st["active"] is False

    def test_router_status_baked_not_removable(self, monkeypatch):
        monkeypatch.setattr("src.api.memory.router_available", lambda site=None: True)
        st = mem.router_status()
        assert st["available"] is True
        assert st["removable"] is False

    def test_router_status_runtime_install_removable(self, monkeypatch):
        monkeypatch.setattr(
            "src.api.memory.router_available",
            lambda site=None: site is not None,
        )
        st = mem.router_status()
        assert st["available"] is True
        assert st["removable"] is True


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
        # cache_dir now always resolves to the shared model cache (baked image
        # path when present, else the persistent modules dir) so the memory
        # embedder and the semantic router share the same weights.
        assert m.cache_dir == mem.memory_models()

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