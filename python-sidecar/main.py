from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="flashback-sidecar", version="0.1.0")


class EmbedRequest(BaseModel):
    text: str
    model: Optional[str] = "text-embedding-3-small"


class EmbedResponse(BaseModel):
    embedding: list[float]
    model: str
    tokens: int


class ExtractRequest(BaseModel):
    text: str
    extract_entities: bool = True
    extract_facts: bool = True


class ExtractResponse(BaseModel):
    entities: list[dict]
    facts: list[dict]


@app.get("/health")
async def health():
    return {"status": "ok", "service": "flashback-sidecar"}


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """Generate embeddings for text. Stub — wire up to OpenAI or local model."""
    # TODO: call OpenAI / local embedding model
    stub_dim = 1536
    return EmbedResponse(
        embedding=[0.0] * stub_dim,
        model=request.model or "text-embedding-3-small",
        tokens=len(request.text.split()),
    )


@app.post("/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest):
    """Extract entities and facts from text. Stub — wire up to LLM."""
    # TODO: call LLM for entity/fact extraction
    return ExtractResponse(entities=[], facts=[])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
