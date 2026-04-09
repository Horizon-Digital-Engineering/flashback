from pydantic import BaseModel


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: list[float]
    model: str
    dim: int


class ExtractRequest(BaseModel):
    text: str


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class ExtractResponse(BaseModel):
    entities: list[Entity]
