from typing import Any, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., description="Chat conversation history")
    context: dict[str, Any] | None = Field(default=None, description="Optional active forensic analysis result/report context")
    api_key: str | None = Field(default=None, description="Optional client-provided OpenRouter API key")
    model: str | None = Field(default=None, description="OpenRouter model ID, e.g., google/gemini-2.0-flash-001")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=2048, ge=1, le=8192, description="Maximum completion tokens")


class ChatResponse(BaseModel):
    message: ChatMessage
    model: str
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


class ChatStatusResponse(BaseModel):
    configured: bool
    default_model: str
    available_models: list[dict[str, Any]]
