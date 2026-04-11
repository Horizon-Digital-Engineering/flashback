"""Shared pytest fixtures.

Models are mocked so tests run without downloading any weights.
The lifespan skips loading when ``is_loaded`` is already ``True``,
so injecting pre-configured singletons before creating TestClient is enough.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Make the sidecar package importable from tests/.
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Embedder fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_embedder():
    """Embedder with model pre-loaded; encode returns zero vectors."""
    from embedder import Embedder

    emb = Embedder()
    emb.dimension = 384

    def _encode(texts, **kwargs):  # noqa: ANN001
        return np.zeros((len(texts), 384), dtype=np.float32)

    emb._model = MagicMock()
    emb._model.encode = MagicMock(side_effect=_encode)
    return emb


# ---------------------------------------------------------------------------
# Extractor fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_extractor():
    """Extractor with spaCy mocked; default doc has no entities or sents."""
    from extractor import Extractor, LocalExtractionBackend

    ext = Extractor()

    # Default doc: no entities, no sentences.
    default_doc = MagicMock()
    default_doc.ents = []
    default_doc.sents = []

    ext._nlp = MagicMock(return_value=default_doc)
    ext._backend = LocalExtractionBackend(ext._nlp)
    return ext


# ---------------------------------------------------------------------------
# TestClient fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(mock_embedder, mock_extractor):
    """FastAPI TestClient with both models mocked out."""
    import main

    # Inject mocks before lifespan runs so the is_loaded check prevents
    # the startup handler from trying to download real models.
    main._embedder = mock_embedder
    main._extractor = mock_extractor

    with TestClient(main.app) as c:
        yield c
