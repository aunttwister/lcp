"""Embedding model wrapper for the memory plugin.

The real model (``BAAI/bge-small-en-v1.5``, 384-dim) is only loaded when the
memory module is installed via Setup (it lives in ``$LCP_MODULES_DIR``, a
persistent bind mount). The wrapper is deliberately lazy so importing this
module never triggers a heavy ``sentence-transformers`` import, and embedding
failures degrade to a clear :class:`MemoryError` instead of crashing the
gateway.
"""

from __future__ import annotations

import os
from typing import Optional

from ..logging_config import get_logger
from .base import MemoryError

logger = get_logger("lcp.memory.embeddings")

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIM = 384


class EmbeddingModel:
    """Lazy wrapper around a sentence-transformers model.

    ``embed`` returns a list of vectors (one per input text), each a list of
    floats of length ``dim``. Embedding is deterministic enough for storage but
    is NOT expected to be exact — ANN recall tolerates small drift.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 device: str = "cpu", cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise MemoryError(
                "sentence-transformers is not installed — install the memory "
                "module from the Setup page"
            ) from exc
        kwargs: dict = {"device": self.device}
        if self.cache_dir:
            kwargs["cache_folder"] = self.cache_dir
        try:
            self._model = SentenceTransformer(self.model_name, **kwargs)
        except Exception:
            # The weights may already be cached while the HuggingFace Hub is
            # unreachable (e.g. an offline container). Retry offline so the
            # cached snapshot is used instead of failing on the hub check.
            if not (self.cache_dir and os.path.isdir(self.cache_dir)):
                raise
            logger.info("embedding_retry_local_files_only", model=self.model_name,
                        cache_dir=self.cache_dir)
            kwargs["local_files_only"] = True
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    @property
    def dim(self) -> int:
        """Embedding dimension (default 384 for bge-small; resolved lazily)."""
        try:
            model = self._load()
            return model.get_sentence_embedding_dimension()
        except Exception:  # noqa: BLE001 — fall back to the default dim
            return DEFAULT_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts (best effort: 2**14 token chunks).

        Returns one vector per input text, in order.
        """
        if not texts:
            return []
        try:
            model = self._load()
            return [v.tolist() for v in model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True)]
        except MemoryError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory_embed_failed", error=str(exc))
            raise MemoryError(f"embedding failed: {exc}") from exc


def embedder_from_config(cfg: dict) -> EmbeddingModel:
    """Build an :class:`EmbeddingModel` from a ``plugins.memory`` config block."""
    emb = (cfg or {}).get("embedding") or {}
    cache_dir = None
    # Prefer an explicit cache dir; default to $LCP_MODULES_DIR/models/memory.
    models_root = os.environ.get("LCP_MODULES_DIR", "").strip()
    if models_root:
        cache_dir = os.path.join(models_root, "models", "memory")
    return EmbeddingModel(
        model_name=emb.get("model", DEFAULT_MODEL),
        device=emb.get("device", "cpu"),
        cache_dir=cache_dir,
    )
