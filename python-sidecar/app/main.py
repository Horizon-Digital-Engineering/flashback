from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
import spacy

from .models import EmbedRequest, EmbedResponse, ExtractRequest, ExtractResponse, Entity

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_SPACY_MODEL_NAME = "en_core_web_sm"

_embedder: SentenceTransformer | None = None
_nlp = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _nlp
    _embedder = SentenceTransformer(_EMBED_MODEL_NAME)
    _nlp = spacy.load(_SPACY_MODEL_NAME)
    yield


app = FastAPI(title="flashback-sidecar", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "flashback-sidecar"}


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    if _embedder is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    vector = _embedder.encode(request.text, convert_to_numpy=True)
    return EmbedResponse(
        embedding=vector.tolist(),
        model=_EMBED_MODEL_NAME,
        dim=len(vector),
    )


@app.post("/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest):
    if _nlp is None:
        raise HTTPException(status_code=503, detail="NLP model not loaded")
    doc = _nlp(request.text)
    entities = [
        Entity(text=ent.text, label=ent.label_, start=ent.start_char, end=ent.end_char)
        for ent in doc.ents
    ]
    return ExtractResponse(entities=entities)
