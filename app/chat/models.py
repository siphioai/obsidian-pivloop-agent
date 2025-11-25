"""OpenAI-compatible Pydantic models."""

from typing import Literal, Union

from pydantic import BaseModel, Field, field_validator


class ContentPart(BaseModel):
    """A content part in a multi-part message."""

    type: str
    text: str | None = None


class ChatMessage(BaseModel):
    """A single message in a chat conversation.

    Content can be either a string or an array of content parts (OpenAI format).
    """

    role: Literal["system", "user", "assistant"]
    content: Union[str, list[ContentPart]]

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, v: Union[str, list]) -> Union[str, list[ContentPart]]:
        """Normalize content to handle both string and array formats."""
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            # Convert list of dicts to ContentPart objects if needed
            return v
        return v

    def get_text_content(self) -> str:
        """Extract text content from message, handling both formats."""
        if isinstance(self.content, str):
            return self.content
        # Extract text from content parts
        texts = []
        for part in self.content:
            if isinstance(part, ContentPart) and part.type == "text" and part.text:
                texts.append(part.text)
            elif isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        return "\n".join(texts) if texts else ""


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = "pivloop-agent"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=2048, gt=0, le=16384)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False


class Choice(BaseModel):
    """A completion choice."""

    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] = "stop"


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
