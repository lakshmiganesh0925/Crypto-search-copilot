from typing import Literal, Optional

from pydantic import BaseModel, Field

SourceType = Literal["news", "governance", "whitepaper"]
Status = Literal["answered", "insufficient_information", "refused"]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    history: list[ChatMessage] = Field(default_factory=list)
    source_types: Optional[list[SourceType]] = None
    days: Optional[int] = Field(default=None, ge=1, le=3650)


class SourceOut(BaseModel):
    id: int  # matches the [n] citation in the answer
    title: str
    url: str
    source_type: SourceType
    source_name: str
    published_at: str
    snippet: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    status: Status
    sources: list[SourceOut] = Field(default_factory=list)
    disclaimer: str
    latency_ms: dict[str, int] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    source_types: list[SourceType] = Field(default_factory=lambda: ["news", "governance", "whitepaper"])
