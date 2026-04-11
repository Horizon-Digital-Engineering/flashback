"""Text extraction: named entities, declarative facts, and relationships.

Architecture
------------
Entity extraction uses spaCy's ``en_core_web_sm`` NER pipeline — fast,
deterministic, and no network dependency at runtime.

Fact and relationship extraction is split behind a pluggable backend
interface (``ExtractionBackend``):

* ``LocalExtractionBackend``  — dependency parsing + heuristics (default).
  Zero external calls; good for simple declarative sentences.
* ``LLMExtractionBackend``    — forwards the text to an external LLM.
  Activated when ``use_llm=True`` is set on the request AND the env var
  ``EXTRACTION_LLM_API_KEY`` is present.  The implementation is intentionally
  left as a stub so the API contract can be established before committing to
  a specific model provider.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass  # spacy types imported lazily

logger = logging.getLogger(__name__)

SPACY_MODEL = "en_core_web_sm"

# ---------------------------------------------------------------------------
# Extraction backend interface
# ---------------------------------------------------------------------------


class ExtractionBackend(ABC):
    """Pluggable strategy for fact and relationship extraction."""

    @abstractmethod
    def extract_facts_and_relationships(
        self,
        text: str,
        entities: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return ``(facts, relationships)`` for *text*.

        *entities* is the NER output already computed by :class:`Extractor`
        so backends can use it to promote SVO triples to typed relationships
        without re-running NER.
        """
        ...


class LocalExtractionBackend(ExtractionBackend):
    """Deterministic extraction using spaCy dependency parsing.

    For each sentence the backend walks the dependency tree looking for a
    verbal ROOT with a nominal subject.  The resulting SVO triple becomes a
    fact.  If both the subject and object spans overlap with known named
    entities the triple is also promoted to a relationship.
    """

    def __init__(self, nlp: Any) -> None:
        self.nlp = nlp

    def extract_facts_and_relationships(
        self,
        text: str,
        entities: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        doc = self.nlp(text)
        entity_texts: set[str] = {e["text"] for e in entities}
        facts: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        for sent in doc.sents:
            for token in sent:
                if token.dep_ != "ROOT" or token.pos_ not in ("VERB", "AUX"):
                    continue

                subj_token: Optional[Any] = None
                obj_token: Optional[Any] = None

                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass") and subj_token is None:
                        subj_token = child
                    elif (
                        child.dep_ in ("dobj", "attr", "acomp") and obj_token is None
                    ):
                        obj_token = child
                    elif child.dep_ == "prep" and obj_token is None:
                        # Capture prepositional objects: "lives in Wisconsin"
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                obj_token = grandchild
                                break

                if subj_token is None:
                    continue

                subj_text = _noun_phrase(subj_token)
                obj_text = _noun_phrase(obj_token) if obj_token else None
                pred = token.lemma_

                facts.append(
                    {
                        "text": sent.text.strip(),
                        "subject": subj_text,
                        "predicate": pred,
                        "object": obj_text,
                        "confidence": 0.80,
                    }
                )

                # Promote to relationship only when both sides are named entities.
                subj_ent = _overlapping_entity(subj_text, entity_texts)
                obj_ent = (
                    _overlapping_entity(obj_text, entity_texts) if obj_text else None
                )
                if subj_ent and obj_ent:
                    relationships.append(
                        {
                            "subject": subj_ent,
                            "predicate": pred,
                            "object": obj_ent,
                            "confidence": 0.75,
                        }
                    )

        return facts, relationships


class LLMExtractionBackend(ExtractionBackend):
    """LLM-powered extraction.

    This backend is intentionally a stub.  To activate it:

    1. Set ``EXTRACTION_LLM_API_KEY`` in the environment.
    2. Implement :meth:`extract_facts_and_relationships` to call your chosen
       LLM API (Anthropic, OpenAI, …) with a structured prompt and parse the
       JSON response back into the ``(facts, relationships)`` tuple format.

    The stub raises ``NotImplementedError`` so that callers receive a clear
    HTTP 501 rather than silent empty results.
    """

    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307") -> None:
        self.api_key = api_key
        self.model = model

    def extract_facts_and_relationships(
        self,
        text: str,
        entities: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        raise NotImplementedError(
            "LLMExtractionBackend is not yet implemented. "
            "Add the API call in extractor.py and remove this error."
        )


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------


class Extractor:
    """spaCy-based NER with a pluggable fact/relationship extraction backend."""

    def __init__(self) -> None:
        self._nlp: Optional[Any] = None
        self._backend: Optional[ExtractionBackend] = None
        self.model_name: str = SPACY_MODEL

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the spaCy pipeline.  Raises on failure."""
        import spacy  # lazy import

        logger.info("Loading spaCy model: %s", self.model_name)
        self._nlp = spacy.load(self.model_name)
        self._backend = LocalExtractionBackend(self._nlp)

        if os.environ.get("EXTRACTION_LLM_API_KEY"):
            logger.info(
                "EXTRACTION_LLM_API_KEY present — LLM backend available via use_llm=true"
            )
        logger.info("spaCy model ready")

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Return a list of named-entity dicts using spaCy NER.

        Each dict has keys: ``text``, ``label``, ``start``, ``end``.
        """
        if self._nlp is None:
            raise RuntimeError("spaCy model not loaded — call load() first.")
        doc = self._nlp(text)
        return [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]

    # ------------------------------------------------------------------
    # Fact / relationship extraction
    # ------------------------------------------------------------------

    def extract_facts_and_relationships(
        self,
        text: str,
        entities: list[dict[str, Any]],
        use_llm: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return ``(facts, relationships)`` using the active backend.

        Args:
            text:      Source text.
            entities:  Pre-computed NER results (from :meth:`extract_entities`).
            use_llm:   If ``True``, route through the LLM backend instead of
                       the local dependency-parsing backend.

        Raises:
            RuntimeError:  Model not loaded.
            ValueError:    ``use_llm=True`` but no API key configured.
            NotImplementedError: LLM backend stub has not been implemented.
        """
        if self._nlp is None:
            raise RuntimeError("spaCy model not loaded — call load() first.")

        if use_llm:
            api_key = os.environ.get("EXTRACTION_LLM_API_KEY")
            if not api_key:
                raise ValueError(
                    "use_llm=true requires the EXTRACTION_LLM_API_KEY env var."
                )
            backend: ExtractionBackend = LLMExtractionBackend(api_key)
        else:
            backend = self._backend  # type: ignore[assignment]

        return backend.extract_facts_and_relationships(text, entities)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._nlp is not None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _noun_phrase(token: Any) -> str:
    """Return a short noun-phrase string anchored at *token*.

    Walks left children with dependency labels that are part of a compound
    or modified noun phrase (compound, amod, det, poss, nummod).
    """
    modifiers = [
        child.text
        for child in token.lefts
        if child.dep_ in ("compound", "amod", "det", "poss", "nummod")
    ]
    modifiers.append(token.text)
    return " ".join(modifiers)


def _overlapping_entity(text: Optional[str], entity_texts: set[str]) -> Optional[str]:
    """Return the entity string that best overlaps with *text*, or ``None``."""
    if not text:
        return None
    text_lower = text.lower()
    for ent in entity_texts:
        ent_lower = ent.lower()
        if ent_lower in text_lower or text_lower in ent_lower:
            return ent
    return None
