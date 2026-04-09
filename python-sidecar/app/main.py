from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import logging

from .models import (
    EmbedRequest,
    EmbedResponse,
    ExtractRequest,
    ExtractResponse,
    Entity,
    Relationship,
    HealthResponse,
)

logger = logging.getLogger(__name__)

_embedder = None
_nlp = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _nlp

    logger.info("Loading sentence-transformers model...")
    from sentence_transformers import SentenceTransformer
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info("Loading spaCy model...")
    import spacy
    _nlp = spacy.load("en_core_web_sm")

    logger.info("Models loaded.")
    yield

    _embedder = None
    _nlp = None


app = FastAPI(title="flashback-sidecar", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        service="flashback-sidecar",
        models_loaded=(_embedder is not None and _nlp is not None),
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    if _embedder is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")

    embedding = _embedder.encode(request.text, normalize_embeddings=True)
    vector = embedding.tolist()

    return EmbedResponse(
        embedding=vector,
        model="all-MiniLM-L6-v2",
        dimensions=len(vector),
    )


@app.post("/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest):
    if _nlp is None:
        raise HTTPException(status_code=503, detail="NLP model not loaded")

    doc = _nlp(request.text)

    entities: list[Entity] = []
    if request.extract_entities:
        for ent in doc.ents:
            entities.append(
                Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                )
            )

    relationships: list[Relationship] = []
    if request.extract_relationships:
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
                subject = token.text
                verb = token.head.text
                for child in token.head.children:
                    if child.dep_ in ("dobj", "attr", "prep"):
                        obj = " ".join(
                            t.text for t in child.subtree
                            if not t.is_punct
                        )
                        relationships.append(
                            Relationship(
                                subject=subject,
                                predicate=verb,
                                object=obj,
                            )
                        )

    return ExtractResponse(entities=entities, relationships=relationships)
