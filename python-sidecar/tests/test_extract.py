from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_ent(text, label, start, end):
    ent = MagicMock()
    ent.text = text
    ent.label_ = label
    ent.start_char = start
    ent.end_char = end
    return ent


@pytest.fixture()
def client():
    mock_nlp = MagicMock()
    mock_embedder = MagicMock()

    with patch("app.main._nlp", mock_nlp), patch("app.main._embedder", mock_embedder):
        from app.main import app
        yield TestClient(app), mock_nlp


def test_extract_no_entities(client):
    test_client, mock_nlp = client
    mock_nlp.return_value.ents = []

    resp = test_client.post("/extract", json={"text": "nothing to find here"})
    assert resp.status_code == 200
    assert resp.json()["entities"] == []


def test_extract_returns_entities(client):
    test_client, mock_nlp = client
    mock_doc = MagicMock()
    mock_doc.ents = [
        _make_ent("London", "GPE", 10, 16),
        _make_ent("Alice", "PERSON", 0, 5),
    ]
    mock_nlp.return_value = mock_doc

    resp = test_client.post("/extract", json={"text": "Alice visited London"})
    assert resp.status_code == 200
    entities = resp.json()["entities"]
    assert len(entities) == 2
    assert entities[0] == {"text": "London", "label": "GPE", "start": 10, "end": 16}
    assert entities[1] == {"text": "Alice", "label": "PERSON", "start": 0, "end": 5}


def test_extract_calls_nlp(client):
    test_client, mock_nlp = client
    mock_nlp.return_value.ents = []

    test_client.post("/extract", json={"text": "some input"})
    mock_nlp.assert_called_once_with("some input")
