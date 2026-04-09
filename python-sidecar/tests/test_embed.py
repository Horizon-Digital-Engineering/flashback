from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.array([0.1] * 384)

    mock_nlp = MagicMock()

    with (
        patch("app.main._embedder", mock_embedder),
        patch("app.main._nlp", mock_nlp),
    ):
        from app.main import app
        with TestClient(app) as c:
            yield c, mock_embedder


def test_embed_returns_vector(client):
    c, mock_embedder = client
    response = c.post("/embed", json={"text": "Hello world"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["embedding"]) == 384
    assert data["model"] == "all-MiniLM-L6-v2"
    assert data["dimensions"] == 384


def test_embed_calls_encode(client):
    c, mock_embedder = client
    c.post("/embed", json={"text": "test sentence"})
    mock_embedder.encode.assert_called_once_with(
        "test sentence", normalize_embeddings=True
    )


def test_embed_empty_text(client):
    c, _ = client
    response = c.post("/embed", json={"text": ""})
    assert response.status_code == 200
    assert response.json()["dimensions"] == 384


def test_health_models_loaded(client):
    c, _ = client
    response = c.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True
