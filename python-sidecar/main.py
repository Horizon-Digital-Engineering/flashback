from contextlib import asynccontextmanager
from typing import Optional

import spacy
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Models (loaded once at startup)
# ---------------------------------------------------------------------------

_embed_model: Optional[SentenceTransformer] = None
_nlp: Optional[spacy.language.Language] = None
_model_errors: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embed_model, _nlp
    try:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:
        _model_errors["embed"] = str(exc)

    try:
        _nlp = spacy.load("en_core_web_sm")
    except Exception as exc:
        _model_errors["extract"] = str(exc)

    yield


app = FastAPI(title="flashback-sidecar", version="0.1.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: list[float]
    model: str
    dimensions: int


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class Fact(BaseModel):
    subject: str
    predicate: str
    object: str


class ExtractRequest(BaseModel):
    text: str
    extract_entities: bool = True
    extract_facts: bool = True


class ExtractResponse(BaseModel):
    entities: list[Entity]
    facts: list[Fact]


class HealthResponse(BaseModel):
    status: str
    service: str
    models: dict[str, str]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    model_status: dict[str, str] = {}
    model_status["embed"] = "error" if "embed" in _model_errors else ("loaded" if _embed_model is not None else "not_loaded")
    model_status["extract"] = "error" if "extract" in _model_errors else ("loaded" if _nlp is not None else "not_loaded")

    overall = "ok" if all(v == "loaded" for v in model_status.values()) else "degraded"
    return HealthResponse(status=overall, service="flashback-sidecar", models=model_status)


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    if _embed_model is None:
        detail = _model_errors.get("embed", "Embedding model not loaded")
        raise HTTPException(status_code=503, detail=detail)

    vector = _embed_model.encode(request.text, convert_to_numpy=True)
    return EmbedResponse(
        embedding=vector.tolist(),
        model=EMBED_MODEL_NAME,
        dimensions=len(vector),
    )


@app.post("/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest):
    if _nlp is None:
        detail = _model_errors.get("extract", "NLP model not loaded")
        raise HTTPException(status_code=503, detail=detail)

    doc = _nlp(request.text)

    entities: list[Entity] = []
    if request.extract_entities:
        entities = [
            Entity(text=ent.text, label=ent.label_, start=ent.start_char, end=ent.end_char)
            for ent in doc.ents
        ]

    facts: list[Fact] = []
    if request.extract_facts:
        # Simple SVO triples from dependency parse
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
                objects = [c for c in token.children if c.dep_ in ("dobj", "pobj", "attr")]
                for subj in subjects:
                    for obj in objects:
                        facts.append(
                            Fact(
                                subject=subj.text,
                                predicate=token.text,
                                object=obj.text,
                            )
                        )

    return ExtractResponse(entities=entities, facts=facts)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
