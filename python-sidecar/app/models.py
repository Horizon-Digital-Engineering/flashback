from pydantic import BaseModel
from typing import Optional


class EmbedRequest(BaseModel):
    text: str
    model: Optional[str] = "all-MiniLM-L6-v2"


class EmbedResponse(BaseModel):
    embedding: list[float]
    model: str
    dimensions: int


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class Relationship(BaseModel):
    subject: str
    predicate: str
    object: str


class ExtractRequest(BaseModel):
    text: str
    extract_entities: bool = True
    extract_relationships: bool = True


class ExtractResponse(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]


class HealthResponse(BaseModel):
    status: str
    service: str
    models_loaded: bool
