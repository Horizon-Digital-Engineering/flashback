from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.zeros(384, dtype=np.float32)

    with patch("app.main._embedder", mock_embedder), patch("app.main._nlp", MagicMock()):
        from app.main import app
        yield TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_embed_returns_vector(client):
    resp = client.post("/embed", json={"text": "hello world"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "all-MiniLM-L6-v2"
    assert body["dim"] == 384
    assert len(body["embedding"]) == 384


def test_embed_calls_encoder(client):
    from app.main import _embedder
    client.post("/embed", json={"text": "test sentence"})
    _embedder.encode.assert_called_once_with("test sentence", convert_to_numpy=True)
