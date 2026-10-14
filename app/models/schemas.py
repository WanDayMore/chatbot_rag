from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "model", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User prompt or question")
    turnstile_token: Optional[str] = Field(None, description="Cloudflare Turnstile token for verification")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Recent conversation turns")


class SourceDocument(BaseModel):
    source: str
    content_snippet: str
    score: Optional[float] = None


class ChatResponse(BaseModel):
    reply: str
    cached: bool
    sources: List[SourceDocument] = []
    latency_ms: float


class IngestResponse(BaseModel):
    documents_indexed: int
    chunks_created: int
    status: str
