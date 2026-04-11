"""API endpoint tests (all models mocked)."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "flashback-sidecar"


def test_health_reports_models_loaded(client):
    body = client.get("/health").json()
    assert body["embedding_model"]["loaded"] is True
    assert body["embedding_model"]["name"] == "all-MiniLM-L6-v2"
    assert body["embedding_model"]["dimension"] == 384
    assert body["spacy_model"]["loaded"] is True
    assert body["spacy_model"]["name"] == "en_core_web_sm"


# ---------------------------------------------------------------------------
# /embed
# ---------------------------------------------------------------------------


def test_embed_single_text(client):
    resp = client.post("/embed", json={"text": "hello world"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert len(body["embeddings"]) == 1
    assert len(body["embeddings"][0]) == 384
    assert body["model"] == "all-MiniLM-L6-v2"
    assert body["dimension"] == 384


def test_embed_batch(client):
    resp = client.post("/embed", json={"text": ["hello", "world", "foo"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert len(body["embeddings"]) == 3
    for vec in body["embeddings"]:
        assert len(vec) == 384


def test_embed_503_when_model_not_loaded(mock_extractor):
    """503 is returned when the embedding model is not loaded."""
    from embedder import Embedder
    from unittest.mock import patch
    import main
    from fastapi.testclient import TestClient

    unloaded = Embedder()  # is_loaded == False
    orig = main._embedder
    main._embedder = unloaded
    main._extractor = mock_extractor

    # Patch load() to a no-op so lifespan doesn't try to download weights.
    with patch.object(unloaded, "load"):
        with TestClient(main.app) as c:
            resp = c.post("/embed", json={"text": "hi"})

    main._embedder = orig
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /extract
# ---------------------------------------------------------------------------


def test_extract_returns_empty_for_blank_doc(client):
    resp = client.post("/extract", json={"text": "Hello."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["entities"] == []
    assert body["facts"] == []
    assert body["relationships"] == []


def test_extract_entities(client, mock_extractor):
    """NER results are surfaced in the response."""
    ent = MagicMock()
    ent.text = "Leslie"
    ent.label_ = "PERSON"
    ent.start_char = 0
    ent.end_char = 6

    mock_doc = MagicMock()
    mock_doc.ents = [ent]
    mock_doc.sents = []
    mock_extractor._nlp.return_value = mock_doc

    resp = client.post(
        "/extract",
        json={
            "text": "Leslie lives in Wisconsin.",
            "extract_facts": False,
            "extract_relationships": False,
        },
    )
    assert resp.status_code == 200
    entities = resp.json()["entities"]
    assert len(entities) == 1
    assert entities[0]["text"] == "Leslie"
    assert entities[0]["label"] == "PERSON"
    assert entities[0]["start"] == 0
    assert entities[0]["end"] == 6


def test_extract_503_when_model_not_loaded(mock_embedder):
    """503 is returned when the spaCy model is not loaded."""
    from extractor import Extractor
    from unittest.mock import patch
    import main
    from fastapi.testclient import TestClient

    unloaded = Extractor()
    orig = main._extractor
    main._extractor = unloaded
    main._embedder = mock_embedder

    with patch.object(unloaded, "load"):
        with TestClient(main.app) as c:
            resp = c.post("/extract", json={"text": "hi"})

    main._extractor = orig
    assert resp.status_code == 503


def test_extract_use_llm_without_key_returns_400(client):
    """Requesting use_llm without an API key set returns 400."""
    import os

    os.environ.pop("EXTRACTION_LLM_API_KEY", None)
    resp = client.post("/extract", json={"text": "hello", "use_llm": True})
    assert resp.status_code == 400
    assert "EXTRACTION_LLM_API_KEY" in resp.json()["detail"]


def test_extract_selective_flags(client, mock_extractor):
    """extract_entities=False skips NER; the list is empty."""
    resp = client.post(
        "/extract",
        json={"text": "hello", "extract_entities": False, "extract_facts": False, "extract_relationships": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["entities"] == []
    assert body["facts"] == []
    assert body["relationships"] == []
