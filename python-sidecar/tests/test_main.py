"""
Unit tests for flashback-sidecar endpoints.
ML dependencies (sentence-transformers, spaCy) are mocked so tests run
without installing the actual models.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers to build fake spaCy docs
# ---------------------------------------------------------------------------

def _make_fake_doc(text: str):
    """Return a minimal spaCy-like doc object for testing."""
    doc = MagicMock()

    # Entities
    ent1 = MagicMock()
    ent1.text = "London"
    ent1.label_ = "GPE"
    ent1.start_char = text.find("London") if "London" in text else 0
    ent1.end_char = ent1.start_char + len("London")
    doc.ents = [ent1] if "London" in text else []

    # Dependency tokens for SVO extraction
    verb = MagicMock()
    verb.dep_ = "ROOT"
    verb.pos_ = "VERB"
    verb.text = "visited"

    subj = MagicMock()
    subj.dep_ = "nsubj"
    subj.text = "Alice"

    obj = MagicMock()
    obj.dep_ = "dobj"
    obj.text = "London"

    verb.children = [subj, obj]

    noun = MagicMock()
    noun.dep_ = "nsubj"
    noun.pos_ = "NOUN"
    noun.text = "Alice"
    noun.children = []

    doc.__iter__ = lambda self: iter([subj, verb, obj] if "visited" in text else [noun])
    return doc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """Return a TestClient with ML models mocked at import time."""
    fake_embed = MagicMock()
    fake_embed.encode.return_value = np.zeros(384, dtype=np.float32)

    fake_nlp = MagicMock()
    fake_nlp.side_effect = _make_fake_doc

    with (
        patch("main._embed_model", fake_embed),
        patch("main._nlp", fake_nlp),
        patch("main._model_errors", {}),
    ):
        # Import app after patches are in place
        from main import app  # noqa: PLC0415
        yield TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "flashback-sidecar"
    assert body["status"] in ("ok", "degraded")
    assert "embed" in body["models"]
    assert "extract" in body["models"]


# ---------------------------------------------------------------------------
# /embed
# ---------------------------------------------------------------------------

def test_embed_returns_vector(client):
    resp = client.post("/embed", json={"text": "hello world"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "all-MiniLM-L6-v2"
    assert body["dimensions"] == 384
    assert len(body["embedding"]) == 384
    assert all(isinstance(v, float) for v in body["embedding"])


def test_embed_503_when_model_missing():
    with (
        patch("main._embed_model", None),
        patch("main._model_errors", {"embed": "load failed"}),
    ):
        from main import app  # noqa: PLC0415
        c = TestClient(app, raise_server_exceptions=False)
        resp = c.post("/embed", json={"text": "hello"})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /extract
# ---------------------------------------------------------------------------

def test_extract_entities_and_facts(client):
    resp = client.post(
        "/extract",
        json={"text": "Alice visited London", "extract_entities": True, "extract_facts": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["entities"], list)
    assert isinstance(body["facts"], list)

    entity_texts = [e["text"] for e in body["entities"]]
    assert "London" in entity_texts

    fact_predicates = [f["predicate"] for f in body["facts"]]
    assert "visited" in fact_predicates


def test_extract_entities_only(client):
    resp = client.post(
        "/extract",
        json={"text": "Alice visited London", "extract_entities": True, "extract_facts": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["facts"] == []


def test_extract_503_when_model_missing():
    with (
        patch("main._nlp", None),
        patch("main._model_errors", {"extract": "load failed"}),
    ):
        from main import app  # noqa: PLC0415
        c = TestClient(app, raise_server_exceptions=False)
        resp = c.post("/extract", json={"text": "hello"})
        assert resp.status_code == 503
