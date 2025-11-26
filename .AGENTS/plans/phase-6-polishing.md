# Feature: Phase 6 - Polishing & Production Readiness

## Feature Description

Phase 6 refines the Obsidian PivLoop Agent for production readiness:
1. **True SSE Streaming**: Replace simulated streaming with real token-by-token streaming using PydanticAI's `run_stream_events()` API
2. **Error Handling Standardization**: Consistent error responses across all endpoints
3. **Test Coverage Expansion**: Target 90%+ coverage, especially `router.py` (currently 27%)
4. **Code Quality Polish**: Complete docstrings and mypy compliance

## User Story

As an Obsidian user using Co-Pilot, I want responses to stream in real-time so I see progress on longer operations, receive consistent error messages, and have confidence the system is stable.

## Problem Statement

1. **Simulated streaming**: Current implementation waits for full response then sends as single chunk
2. **Inconsistent errors**: Router raises HTTPException, tools return plain strings
3. **Low router coverage**: 27% test coverage on critical code path
4. **Missing edge case tests**: `get_text_content()`, streaming errors untested

## Feature Metadata

- **Feature Type**: Enhancement / Polish
- **Estimated Complexity**: Medium
- **Primary Systems Affected**: `app/chat/router.py`, `app/chat/models.py`, `app/tests/test_chat.py`
- **Dependencies**: pydantic-ai (run_stream_events API)

---

## CONTEXT REFERENCES

### Relevant Codebase Files - READ BEFORE IMPLEMENTING!

| File | Lines | Purpose |
|------|-------|---------|
| `app/chat/router.py` | 77-132 | Current streaming (simulated) - **MUST REPLACE** |
| `app/chat/router.py` | 23-74 | Non-streaming - reference patterns |
| `app/chat/models.py` | 1-84 | OpenAI models - **ADD streaming models** |
| `app/notes/tools.py` | 229-244 | Error handling pattern |
| `app/tests/test_chat.py` | 1-50 | Current tests - **EXPAND** |

### PydanticAI Streaming API

```python
async for event in agent.run_stream_events('prompt'):
    # PartDeltaEvent has .delta.content_delta with incremental text
    # AgentRunResultEvent signals completion
```

### OpenAI Streaming Format (SSE)

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"}}]}
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

---

## STEP-BY-STEP TASKS

### Task 1: UPDATE `app/chat/models.py` - Add streaming and error models

**ADD after `ChatCompletionResponse` class:**

```python
class DeltaContent(BaseModel):
    """Delta content in a streaming chunk."""
    role: str | None = None
    content: str | None = None

class StreamChoice(BaseModel):
    """A streaming chunk choice."""
    index: int = 0
    delta: DeltaContent
    finish_reason: str | None = None

class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk response."""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]

class ErrorDetail(BaseModel):
    """Structured error detail."""
    message: str = Field(..., description="Human-readable error message")
    type: Literal["invalid_request_error", "server_error", "authentication_error"] = "server_error"
    code: str | None = None

class ErrorResponse(BaseModel):
    """OpenAI-compatible error response wrapper."""
    error: ErrorDetail
```

**VALIDATE**: `uv run python -c "from app.chat.models import ChatCompletionChunk, ErrorResponse; print('OK')"`

---

### Task 2: UPDATE `app/chat/router.py` - Add imports

**REPLACE imports with:**

```python
"""FastAPI router for /v1/chat/completions endpoint."""
import json
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic_ai import AgentRunResultEvent
from pydantic_ai.messages import PartDeltaEvent

from app.chat.agent import chat_agent
from app.chat.models import (
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage, Choice, DeltaContent, ErrorDetail, ErrorResponse, StreamChoice,
)
from app.dependencies import ChatDependencies, VaultClient, get_vault_client, logger

router = APIRouter(prefix="/v1", tags=["chat"])
```

**GOTCHA**: Import path for `PartDeltaEvent` may vary - try `pydantic_ai.messages` or `pydantic_ai.agent`

**VALIDATE**: `uv run ruff check app/chat/router.py`

---

### Task 3: UPDATE `app/chat/router.py` - Standardize error responses

**REPLACE `process_chat_completion` function:**

```python
async def process_chat_completion(
    request: ChatCompletionRequest, deps: ChatDependencies
) -> ChatCompletionResponse:
    """Process non-streaming chat completion."""
    logger.info("processing_chat", extra={"trace_id": deps.trace_id, "messages": len(request.messages)})

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="No user message provided.",
                    type="invalid_request_error",
                    code="missing_user_message",
                )
            ).model_dump(),
        )

    try:
        user_content = user_messages[-1].get_text_content()
        result = await chat_agent.run(user_prompt=user_content, deps=deps)
        return ChatCompletionResponse(
            id=f"chatcmpl-{deps.trace_id}",
            created=int(time.time()),
            model=request.model,
            choices=[Choice(message=ChatMessage(role="assistant", content=result.output), finish_reason="stop")],
        )
    except Exception as e:
        logger.error("chat_failed", extra={"trace_id": deps.trace_id, "error": str(e)}, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(error=ErrorDetail(message=f"Agent error: {str(e)}", code="agent_error")).model_dump(),
        )
```

---

### Task 4: UPDATE `app/chat/router.py` - Implement true streaming

**REPLACE `stream_chat_completion` function:**

```python
async def stream_chat_completion(
    request: ChatCompletionRequest, deps: ChatDependencies
) -> AsyncGenerator[str, None]:
    """Stream chat completion using PydanticAI's run_stream_events()."""
    logger.info("processing_chat_stream", extra={"trace_id": deps.trace_id, "messages": len(request.messages)})

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        error = ErrorResponse(error=ErrorDetail(message="No user message provided.", type="invalid_request_error", code="missing_user_message"))
        yield f"data: {error.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        return

    try:
        user_content = user_messages[-1].get_text_content()
        chunk_id = f"chatcmpl-{deps.trace_id}"
        created = int(time.time())

        # Send initial role chunk
        role_chunk = ChatCompletionChunk(
            id=chunk_id, created=created, model=request.model,
            choices=[StreamChoice(delta=DeltaContent(role="assistant"))],
        )
        yield f"data: {role_chunk.model_dump_json()}\n\n"

        # Stream content deltas
        async for event in chat_agent.run_stream_events(user_prompt=user_content, deps=deps):
            if isinstance(event, PartDeltaEvent):
                if hasattr(event.delta, "content_delta") and event.delta.content_delta:
                    content_chunk = ChatCompletionChunk(
                        id=chunk_id, created=created, model=request.model,
                        choices=[StreamChoice(delta=DeltaContent(content=event.delta.content_delta))],
                    )
                    yield f"data: {content_chunk.model_dump_json()}\n\n"
            elif isinstance(event, AgentRunResultEvent):
                final_chunk = ChatCompletionChunk(
                    id=chunk_id, created=created, model=request.model,
                    choices=[StreamChoice(delta=DeltaContent(), finish_reason="stop")],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error("chat_stream_failed", extra={"trace_id": deps.trace_id, "error": str(e)}, exc_info=True)
        error = ErrorResponse(error=ErrorDetail(message=f"Streaming error: {str(e)}", code="streaming_error"))
        yield f"data: {error.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
```

**VALIDATE**: `uv run ruff check app/chat/router.py && uv run ruff format app/chat/router.py`

---

### Task 5: UPDATE `app/tests/test_chat.py` - Expand test coverage

**REPLACE entire file with comprehensive tests:**

```python
"""Tests for chat feature."""
from unittest.mock import AsyncMock, Mock, patch
import pytest
from fastapi.testclient import TestClient
from app.chat.models import (
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage, Choice, ContentPart, DeltaContent, ErrorDetail, ErrorResponse, StreamChoice,
)

class TestChatMessage:
    def test_string_content(self) -> None:
        msg = ChatMessage(role="user", content="Hello")
        assert msg.get_text_content() == "Hello"

    def test_content_parts_list(self) -> None:
        parts = [ContentPart(type="text", text="Hello"), ContentPart(type="text", text="World")]
        msg = ChatMessage(role="user", content=parts)
        assert msg.get_text_content() == "Hello\nWorld"

    def test_content_parts_filters_non_text(self) -> None:
        parts = [ContentPart(type="text", text="Hello"), ContentPart(type="image_url", text=None)]
        msg = ChatMessage(role="user", content=parts)
        assert msg.get_text_content() == "Hello"

    def test_empty_content(self) -> None:
        msg = ChatMessage(role="user", content="")
        assert msg.get_text_content() == ""

class TestChatCompletionRequest:
    def test_defaults(self) -> None:
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")])
        assert req.model == "pivloop-agent"
        assert req.max_tokens == 2048
        assert req.stream is False

    def test_max_tokens_validation(self) -> None:
        with pytest.raises(ValueError):
            ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")], max_tokens=0)

    def test_temperature_validation(self) -> None:
        with pytest.raises(ValueError):
            ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")], temperature=3.0)

class TestStreamingModels:
    def test_delta_content(self) -> None:
        assert DeltaContent(role="assistant").role == "assistant"
        assert DeltaContent(content="Hi").content == "Hi"
        assert DeltaContent().role is None

    def test_stream_choice(self) -> None:
        choice = StreamChoice(delta=DeltaContent(content="test"))
        assert choice.delta.content == "test"
        assert choice.finish_reason is None

    def test_chunk_serialization(self) -> None:
        chunk = ChatCompletionChunk(id="test", created=0, model="m", choices=[StreamChoice(delta=DeltaContent(content="Hi"))])
        assert "chat.completion.chunk" in chunk.model_dump_json()

class TestErrorModels:
    def test_error_detail_defaults(self) -> None:
        err = ErrorDetail(message="Test")
        assert err.type == "server_error"
        assert err.code is None

    def test_error_response(self) -> None:
        resp = ErrorResponse(error=ErrorDetail(message="Test", type="invalid_request_error", code="test_code"))
        assert resp.error.code == "test_code"

class TestEndpoints:
    def test_health(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_root(self, client: TestClient) -> None:
        r = client.get("/")
        assert "PivLoop" in r.json()["name"]

    def test_chat_requires_messages(self, client: TestClient) -> None:
        r = client.post("/v1/chat/completions", json={"model": "test"})
        assert r.status_code == 422

    def test_chat_requires_user_message(self, client: TestClient) -> None:
        r = client.post("/v1/chat/completions", json={"model": "test", "messages": [{"role": "system", "content": "Hi"}]})
        assert r.status_code == 400
        assert r.json()["detail"]["error"]["code"] == "missing_user_message"

    def test_streaming_no_user_message(self, client: TestClient) -> None:
        r = client.post("/v1/chat/completions", json={"model": "test", "messages": [{"role": "system", "content": "Hi"}], "stream": True})
        assert r.status_code == 200
        assert "invalid_request_error" in r.text

class TestChatIntegration:
    @pytest.fixture
    def mock_agent(self):
        with patch("app.chat.router.chat_agent") as mock:
            mock_result = Mock()
            mock_result.output = "Test response"
            mock.run = AsyncMock(return_value=mock_result)
            yield mock

    def test_successful_completion(self, client: TestClient, mock_agent) -> None:
        r = client.post("/v1/chat/completions", json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]})
        assert r.status_code == 200
        data = r.json()
        assert data["choices"][0]["message"]["content"] == "Test response"
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_response_has_id(self, client: TestClient, mock_agent) -> None:
        r = client.post("/v1/chat/completions", json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]})
        assert r.json()["id"].startswith("chatcmpl-")
```

**VALIDATE**: `uv run pytest app/tests/test_chat.py -v`

---

### Task 6: UPDATE `app/tests/conftest.py` - Add streaming fixture

**ADD after existing fixtures:**

```python
@pytest.fixture
def mock_stream_events():
    """Create mock async generator for streaming events."""
    from unittest.mock import Mock

    async def _mock_stream(content: str = "Test response"):
        try:
            from pydantic_ai import AgentRunResultEvent
            from pydantic_ai.messages import PartDeltaEvent
        except ImportError:
            PartDeltaEvent = type("PartDeltaEvent", (), {})
            AgentRunResultEvent = type("AgentRunResultEvent", (), {})

        for word in content.split():
            delta = Mock()
            delta.content_delta = word + " "
            event = Mock(spec=PartDeltaEvent)
            event.delta = delta
            yield event

        result_event = Mock(spec=AgentRunResultEvent)
        result_event.result = Mock()
        result_event.result.output = content
        yield result_event

    return _mock_stream
```

---

### Task 7: Run full validation

```bash
# Lint & format
uv run ruff check app/
uv run ruff format app/ --check

# Tests with coverage
uv run pytest app/tests/ -v --cov=app --cov-report=term-missing

# Import verification
uv run python -c "from app.chat.models import ChatCompletionChunk, ErrorResponse; print('OK')"
```

---

## TESTING STRATEGY

| Area | Coverage Target |
|------|-----------------|
| `app/chat/models.py` | 95%+ |
| `app/chat/router.py` | 80%+ (up from 27%) |

**Edge Cases**: Empty messages, no user message, agent exceptions, streaming errors, content parts variations

---

## VALIDATION COMMANDS

```bash
uv run ruff check app/
uv run ruff format app/ --check
uv run pytest app/tests/ -v --cov=app --cov-report=term-missing
uv run python -c "from app.chat.models import ChatCompletionChunk, ErrorResponse; print('OK')"
```

---

## ACCEPTANCE CRITERIA

- [ ] True SSE streaming with PydanticAI `run_stream_events()`
- [ ] Error responses use `ErrorResponse` model consistently
- [ ] Test coverage >= 85% overall, router >= 70%
- [ ] All validation commands pass
- [ ] No regressions

---

## COMPLETION CHECKLIST

- [ ] Task 1: Streaming and error models added
- [ ] Task 2: Imports updated
- [ ] Task 3: `process_chat_completion` uses ErrorResponse
- [ ] Task 4: `stream_chat_completion` implements true streaming
- [ ] Task 5: Test coverage expanded
- [ ] Task 6: Streaming fixtures added
- [ ] Task 7: All validations pass

---

## NOTES

### Design Decisions

1. **PydanticAI `run_stream_events()`**: Yields `PartDeltaEvent` with `.delta.content_delta` for incremental text
2. **OpenAI Compliance**: Streaming format matches OpenAI spec exactly
3. **Streaming Errors**: Sent as SSE data (not HTTP errors) since response has started

### Potential Issues

1. **Import paths**: `PartDeltaEvent` location may vary by PydanticAI version
2. **Event attributes**: Delta content may be `.content_delta` or `.content`
3. **Mock async generators**: Require careful setup for streaming tests

### Future Enhancements

- Token counting for streaming responses
- Request timeouts
- WebSocket support as SSE alternative
