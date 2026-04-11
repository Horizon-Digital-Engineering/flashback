"""Sentence-transformers embedding wrapper.

The model is loaded once at process startup and kept in memory. Inference is
synchronous (CPU-bound); callers should run it in a thread pool if they need
to avoid blocking the event loop on large batches.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as _ST

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """Wraps ``sentence-transformers`` for synchronous text embedding."""

    def __init__(self) -> None:
        self._model: Optional[_ST] = None
        self.model_name: str = MODEL_NAME
        self.dimension: Optional[int] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the model from the local cache (or download on first run).

        Raises on any failure so the caller can decide whether to abort
        startup or continue in degraded mode.
        """
        from sentence_transformers import SentenceTransformer  # lazy — heavy import

        logger.info("Loading embedding model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)

        # Probe to confirm the model works and record the dimension.
        probe: np.ndarray = self._model.encode(["probe"], convert_to_numpy=True)
        self.dimension = int(probe.shape[1])
        logger.info("Embedding model ready — dimension: %d", self.dimension)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return a list of embedding vectors, one per input text.

        Args:
            texts: Non-empty list of strings.

        Returns:
            List of float lists, shape ``[len(texts), dimension]``.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if self._model is None:
            raise RuntimeError("Embedding model not loaded — call load() first.")
        vectors: np.ndarray = self._model.encode(texts, convert_to_numpy=True)
        return vectors.tolist()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
