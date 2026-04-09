from unittest.mock import MagicMock, patch, PropertyMock
import pytest
from fastapi.testclient import TestClient


def _make_ent(text, label_, start_char, end_char):
    ent = MagicMock()
    ent.text = text
    ent.label_ = label_
    ent.start_char = start_char
    ent.end_char = end_char
    return ent


def _make_doc(ents=None, tokens=None):
    doc = MagicMock()
    doc.ents = ents or []
    doc.__iter__ = MagicMock(return_value=iter(tokens or []))
    return doc


@pytest.fixture()
def client():
    mock_embedder = MagicMock()
    mock_nlp = MagicMock()

    with (
        patch("app.main._embedder", mock_embedder),
        patch("app.main._nlp", mock_nlp),
    ):
        from app.main import app
        with TestClient(app) as c:
            yield c, mock_nlp


def test_extract_entities(client):
    c, mock_nlp = client
    doc = _make_doc(
        ents=[
            _make_ent("Leslie", "PERSON", 0, 6),
            _make_ent("Anthropic", "ORG", 14, 23),
        ]
    )
    mock_nlp.return_value = doc

    response = c.post(
        "/extract",
        json={"text": "Leslie works at Anthropic", "extract_relationships": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["entities"]) == 2
    assert data["entities"][0] == {
        "text": "Leslie",
        "label": "PERSON",
        "start": 0,
        "end": 6,
    }
    assert data["entities"][1]["label"] == "ORG"


def test_extract_no_entities_when_disabled(client):
    c, mock_nlp = client
    doc = _make_doc(ents=[_make_ent("Leslie", "PERSON", 0, 6)])
    mock_nlp.return_value = doc

    response = c.post(
        "/extract",
        json={"text": "Leslie works here", "extract_entities": False, "extract_relationships": False},
    )
    assert response.status_code == 200
    assert response.json()["entities"] == []


def test_extract_relationships(client):
    c, mock_nlp = client

    # Build a mock token tree: "Leslie" --nsubj--> "works" (VERB) --dobj--> "code"
    verb_token = MagicMock()
    verb_token.dep_ = "ROOT"
    verb_token.pos_ = "VERB"
    verb_token.text = "works"

    obj_token = MagicMock()
    obj_token.dep_ = "dobj"
    obj_token.text = "code"
    obj_token.is_punct = False
    obj_token.subtree = [obj_token]

    verb_token.children = [obj_token]

    subj_token = MagicMock()
    subj_token.dep_ = "nsubj"
    subj_token.text = "Leslie"
    subj_token.head = verb_token

    doc = _make_doc(tokens=[subj_token, verb_token, obj_token])
    mock_nlp.return_value = doc

    response = c.post(
        "/extract",
        json={"text": "Leslie works code", "extract_entities": False},
    )
    assert response.status_code == 200
    rels = response.json()["relationships"]
    assert len(rels) == 1
    assert rels[0]["subject"] == "Leslie"
    assert rels[0]["predicate"] == "works"
    assert rels[0]["object"] == "code"


def test_extract_empty_text(client):
    c, mock_nlp = client
    mock_nlp.return_value = _make_doc()
    response = c.post("/extract", json={"text": ""})
    assert response.status_code == 200
    data = response.json()
    assert data["entities"] == []
    assert data["relationships"] == []
