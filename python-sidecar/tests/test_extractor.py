"""Unit tests for extractor helpers and backends."""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# _overlapping_entity helper
# ---------------------------------------------------------------------------


def test_overlapping_entity_exact_match():
    from extractor import _overlapping_entity

    assert _overlapping_entity("Leslie", {"Leslie", "Wisconsin"}) == "Leslie"


def test_overlapping_entity_substring():
    from extractor import _overlapping_entity

    # text contains the entity
    assert _overlapping_entity("Leslie Gutschow", {"Leslie"}) == "Leslie"


def test_overlapping_entity_reverse_substring():
    from extractor import _overlapping_entity

    # entity contains the text (abbreviation scenario)
    assert _overlapping_entity("HDE", {"Horizon Digital Engineering HDE"}) == "Horizon Digital Engineering HDE"


def test_overlapping_entity_no_match():
    from extractor import _overlapping_entity

    assert _overlapping_entity("unknown token", {"Leslie", "Wisconsin"}) is None


def test_overlapping_entity_none_input():
    from extractor import _overlapping_entity

    assert _overlapping_entity(None, {"Leslie"}) is None


# ---------------------------------------------------------------------------
# _noun_phrase helper
# ---------------------------------------------------------------------------


def test_noun_phrase_simple():
    from extractor import _noun_phrase

    token = MagicMock()
    token.text = "Wisconsin"
    token.lefts = []
    assert _noun_phrase(token) == "Wisconsin"


def test_noun_phrase_with_compound():
    from extractor import _noun_phrase

    left = MagicMock()
    left.text = "New"
    left.dep_ = "compound"

    token = MagicMock()
    token.text = "York"
    token.lefts = [left]
    assert _noun_phrase(token) == "New York"


# ---------------------------------------------------------------------------
# LocalExtractionBackend
# ---------------------------------------------------------------------------


def test_local_backend_empty_sentences(mock_extractor):
    """Empty doc produces empty facts and relationships."""
    mock_doc = MagicMock()
    mock_doc.sents = []
    mock_extractor._nlp.return_value = mock_doc

    facts, rels = mock_extractor.extract_facts_and_relationships("", [])
    assert facts == []
    assert rels == []


def test_local_backend_no_root_verb(mock_extractor):
    """Sentences without a verbal ROOT produce no facts."""
    token = MagicMock()
    token.dep_ = "ROOT"
    token.pos_ = "NOUN"  # not a VERB/AUX

    sent = MagicMock()
    sent.__iter__ = MagicMock(return_value=iter([token]))

    mock_doc = MagicMock()
    mock_doc.sents = [sent]
    mock_extractor._nlp.return_value = mock_doc

    facts, rels = mock_extractor.extract_facts_and_relationships("A noun.", [])
    assert facts == []


# ---------------------------------------------------------------------------
# LLMExtractionBackend
# ---------------------------------------------------------------------------


def test_llm_backend_raises_not_implemented():
    from extractor import LLMExtractionBackend

    backend = LLMExtractionBackend(api_key="fake-key")
    with pytest.raises(NotImplementedError):
        backend.extract_facts_and_relationships("text", [])


def test_extractor_use_llm_without_env_var_raises(mock_extractor):
    """ValueError is raised when use_llm=True but env var is absent."""
    os.environ.pop("EXTRACTION_LLM_API_KEY", None)
    with pytest.raises(ValueError, match="EXTRACTION_LLM_API_KEY"):
        mock_extractor.extract_facts_and_relationships("text", [], use_llm=True)


# ---------------------------------------------------------------------------
# Extractor.extract_entities
# ---------------------------------------------------------------------------


def test_extract_entities_maps_spacy_output(mock_extractor):
    ent = MagicMock()
    ent.text = "Wisconsin"
    ent.label_ = "GPE"
    ent.start_char = 16
    ent.end_char = 25

    mock_doc = MagicMock()
    mock_doc.ents = [ent]
    mock_extractor._nlp.return_value = mock_doc

    result = mock_extractor.extract_entities("Leslie lives in Wisconsin.")
    assert len(result) == 1
    assert result[0] == {
        "text": "Wisconsin",
        "label": "GPE",
        "start": 16,
        "end": 25,
    }


def test_extract_entities_raises_when_not_loaded():
    from extractor import Extractor

    ext = Extractor()  # never loaded
    with pytest.raises(RuntimeError, match="not loaded"):
        ext.extract_entities("some text")
