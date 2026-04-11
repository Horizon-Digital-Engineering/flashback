"""Pydantic request/response schemas for the flashback-sidecar API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /embed
# ---------------------------------------------------------------------------


class EmbedRequest(BaseModel):
    """Embed one text or a batch of texts."""

    text: str | list[str] = Field(
        ...,
        description="A single string or a list of strings to embed.",
    )


class EmbedResponse(BaseModel):
    embeddings: list[list[float]] = Field(
        ..., description="One embedding vector per input text."
    )
    model: str = Field(..., description="Name of the embedding model used.")
    dimension: int = Field(..., description="Embedding dimension.")
    count: int = Field(..., description="Number of embeddings returned.")


# ---------------------------------------------------------------------------
# /extract
# ---------------------------------------------------------------------------


class Entity(BaseModel):
    """A named entity identified in the source text."""

    text: str = Field(..., description="Surface form of the entity.")
    label: str = Field(
        ...,
        description="spaCy NER label (PERSON, ORG, GPE, DATE, …).",
    )
    start: int = Field(..., description="Character start offset in source text.")
    end: int = Field(..., description="Character end offset in source text.")


class Fact(BaseModel):
    """A declarative statement extracted from the source text."""

    text: str = Field(..., description="Source sentence containing the fact.")
    subject: Optional[str] = Field(None, description="Grammatical subject.")
    predicate: Optional[str] = Field(None, description="Verb / predicate lemma.")
    object: Optional[str] = Field(None, description="Object or complement.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Relationship(BaseModel):
    """A directed relationship between two named entities."""

    subject: str
    predicate: str
    object: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ExtractRequest(BaseModel):
    text: str = Field(..., description="Text to analyse.")
    extract_entities: bool = Field(True, description="Run NER.")
    extract_facts: bool = Field(True, description="Extract declarative facts.")
    extract_relationships: bool = Field(
        True, description="Extract entity-to-entity relationships."
    )
    use_llm: bool = Field(
        False,
        description=(
            "Route fact/relationship extraction through the LLM backend "
            "(requires EXTRACTION_LLM_API_KEY env var)."
        ),
    )


class ExtractResponse(BaseModel):
    entities: list[Entity]
    facts: list[Fact]
    relationships: list[Relationship]


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class ModelStatus(BaseModel):
    loaded: bool
    name: Optional[str] = None
    dimension: Optional[int] = None  # only meaningful for embedding models


class HealthResponse(BaseModel):
    status: str
    service: str
    embedding_model: ModelStatus
    spacy_model: ModelStatus
