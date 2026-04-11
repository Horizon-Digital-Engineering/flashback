"""Flashback Python sidecar — embedding and extraction service.

Endpoints
---------
GET  /health   — liveness + model status
POST /embed    — sentence-transformers text embedding (all-MiniLM-L6-v2)
POST /extract  — spaCy NER + dependency-parse fact/relationship extraction
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, status

from embedder import Embedder
from extractor import Extractor
from schemas import (
    EmbedRequest,
    EmbedResponse,
    Entity,
    ExtractRequest,
    ExtractResponse,
    Fact,
    HealthResponse,
    ModelStatus,
    Relationship,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons — loaded once, reused for every request.
# ---------------------------------------------------------------------------
_embedder = Embedder()
_extractor = Extractor()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    logger.info("Sidecar starting — loading NLP models…")

    if not _embedder.is_loaded:
        try:
            _embedder.load()
        except Exception:
            logger.exception("Failed to load embedding model — /embed will return 503")

    if not _extractor.is_loaded:
        try:
            _extractor.load()
        except Exception:
            logger.exception("Failed to load spaCy model — /extract will return 503")

    logger.info("Startup complete")
    yield
    logger.info("Sidecar shutting down")


app = FastAPI(
    title="flashback-sidecar",
    version="0.2.0",
    description=(
        "Embedding and NLP extraction sidecar for the Flashback episodic memory system."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Return service liveness and the load status of each NLP model."""
    return HealthResponse(
        status="ok",
        service="flashback-sidecar",
        embedding_model=ModelStatus(
            loaded=_embedder.is_loaded,
            name=_embedder.model_name if _embedder.is_loaded else None,
            dimension=_embedder.dimension if _embedder.is_loaded else None,
        ),
        spacy_model=ModelStatus(
            loaded=_extractor.is_loaded,
            name=_extractor.model_name if _extractor.is_loaded else None,
        ),
    )


@app.post("/embed", response_model=EmbedResponse, tags=["embedding"])
async def embed(request: EmbedRequest) -> EmbedResponse:
    """Generate embeddings using ``all-MiniLM-L6-v2``.

    Accepts a single string **or** a list of strings.  Always returns a list
    of embedding vectors (one per input text).
    """
    if not _embedder.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding model not loaded.",
        )

    texts: list[str] = (
        [request.text] if isinstance(request.text, str) else list(request.text)
    )
    if not texts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="text must be a non-empty string or list.",
        )

    try:
        vectors = _embedder.embed(texts)
    except Exception as exc:
        logger.exception("Embedding failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return EmbedResponse(
        embeddings=vectors,
        model=_embedder.model_name,
        dimension=_embedder.dimension or 0,
        count=len(vectors),
    )


@app.post("/extract", response_model=ExtractResponse, tags=["extraction"])
async def extract(request: ExtractRequest) -> ExtractResponse:
    """Extract entities, facts, and relationships from free text.

    By default uses the local dependency-parsing backend (no network calls).
    Pass ``use_llm=true`` to route through the LLM backend — requires the
    ``EXTRACTION_LLM_API_KEY`` environment variable to be set.
    """
    if not _extractor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="spaCy model not loaded.",
        )

    entities: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    try:
        if request.extract_entities:
            entities = _extractor.extract_entities(request.text)

        if request.extract_facts or request.extract_relationships:
            facts_raw, rels_raw = _extractor.extract_facts_and_relationships(
                request.text,
                entities,
                use_llm=request.use_llm,
            )
            if request.extract_facts:
                facts = facts_raw
            if request.extract_relationships:
                relationships = rels_raw

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Extraction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return ExtractResponse(
        entities=[Entity(**e) for e in entities],
        facts=[Fact(**f) for f in facts],
        relationships=[Relationship(**r) for r in relationships],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")
