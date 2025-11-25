
# Obsidian PivLoop Agent - Global Rules

**Version:** 1.0
**Architecture:** Vertical Slice Architecture
**Tech Stack:** PydanticAI + FastAPI + Obsidian Co-Pilot Integration

---

## 1. Core Principles

### Non-Negotiables

1. **Type Safety First**: All functions, variables, and parameters MUST have explicit type hints
2. **Pydantic Validation**: Use Pydantic models for all data structures, API requests/responses, and agent outputs
3. **Dependency Injection**: Use PydanticAI's RunContext system - NEVER use globals or singletons
4. **Structured Logging**: Every operation logs with JSON structure including trace_id, feature, action, status
5. **Documentation**: All public functions require docstrings (Google style with Args/Returns sections)
6. **Error Handling**: All external calls (LLM, file I/O, API) wrapped in try/except with specific error types
7. **Testing Required**: Every feature slice must have unit tests and integration tests
8. **OpenAI Compatibility**: All endpoints must conform to OpenAI's API specification

---

## 2. Tech Stack

### Backend
- **Framework**: FastAPI 0.115+
- **Language**: Python 3.11+
- **Package Manager**: uv (preferred) or pip
- **AI Framework**: PydanticAI (latest)
- **HTTP Client**: httpx (async)
- **Testing**: pytest + pytest-asyncio
- **Linting**: ruff
- **Formatting**: ruff format
- **Type Checking**: mypy --strict

### Integration
- **Obsidian Plugin**: Co-Pilot (community plugin)
- **API Protocol**: OpenAI-compatible /v1/chat/completions endpoint
- **File Operations**: Obsidian Vault file system access via API

### Task Management
- **Archon MCP**: Used for task tracking and RAG documentation queries
- **Documentation Sources**: PydanticAI docs, FastAPI docs, Obsidian API docs

---

## 3. Architecture

### Vertical Slice Structure

Each feature is a self-contained slice with its own folder containing the agent and tools:

```
app/
├── chat/
│   ├── __init__.py            # Exports router
│   ├── agent.py               # Agent definition only
│   ├── tools.py               # All agent tool calls
│   └── router.py              # FastAPI endpoints + service logic
├── notes/
│   ├── __init__.py
│   ├── agent.py
│   ├── tools.py
│   └── router.py
├── vault/
│   ├── __init__.py
│   ├── agent.py
│   ├── tools.py
│   └── router.py
├── dependencies.py            # Shared dependencies & config
├── main.py                    # FastAPI app entry point
└── tests/
    ├── test_chat.py
    ├── test_notes.py
    └── test_vault.py
```

### Feature Folder Structure

Each feature folder contains 3-4 focused files:

```python
# app/chat/agent.py - Agent definition and configuration
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from app.dependencies import VaultClient

@dataclass
class ChatDependencies:
    """Dependencies for chat agent."""
    vault: VaultClient
    trace_id: str

chat_agent = Agent(
    "openai:gpt-4",
    deps_type=ChatDependencies,
    retries=2,
    system_prompt="You are a helpful assistant integrated with Obsidian."
)

# app/chat/tools.py - All tool implementations
from pydantic_ai import RunContext
from .agent import chat_agent, ChatDependencies

@chat_agent.tool
async def read_note(ctx: RunContext[ChatDependencies], note_path: str) -> str:
    """Read a note from the vault."""
    return await ctx.deps.vault.read_file(note_path)

@chat_agent.tool
async def search_vault(ctx: RunContext[ChatDependencies], query: str) -> list[str]:
    """Search the vault for notes."""
    return await ctx.deps.vault.search(query)

# app/chat/router.py - API endpoints and service logic
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.dependencies import VaultClient, get_vault_client
from .agent import chat_agent, ChatDependencies

class ChatCompletionRequest(BaseModel):
    messages: list[dict]
    model: str = "default"

router = APIRouter(prefix="/v1", tags=["chat"])

@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    vault: VaultClient = Depends(get_vault_client)
):
    deps = ChatDependencies(vault=vault, trace_id=str(uuid.uuid4()))
    result = await chat_agent.run(request.messages[-1]["content"], deps=deps)
    return {"choices": [{"message": {"content": result.output}}]}
```

### File Responsibilities

- **agent.py**: Agent instance, dependencies dataclass, system prompts
- **tools.py**: All `@agent.tool` decorated functions for this feature
- **router.py**: Pydantic models, FastAPI routes, service logic
- **__init__.py**: Exports the router for main.py to import

### Key Patterns

1. **One Agent Per Folder**: Each feature has its own folder with agent + tools + router
2. **Clear Separation**: Agent config separate from tools separate from API layer
3. **Tools Registration**: Import agent in tools.py, decorate functions with `@agent.tool`
4. **Minimal Shared Code**: Only VaultClient and logger in dependencies.py

---

## 4. Code Style

### Naming Conventions

**Python Files & Modules**
- snake_case for all files: `agent.py`, `tools.py`, `router.py`
- Feature folders: lowercase, plural: `app/notes/`, `app/chat/`, `app/vault/`
- Standard files in each feature: `agent.py`, `tools.py`, `router.py`, `__init__.py`

**Classes**
- PascalCase for all classes: `ChatRequest`, `NoteMetadata`, `VaultDependencies`
- Pydantic models end with purpose: `ChatCompletionRequest`, `NoteUpdateResponse`
- Agents end with "Agent": `ChatAgent`, `NoteAgent`

**Functions & Variables**
- snake_case: `get_vault_notes()`, `update_note_content()`
- Private functions: `_build_system_prompt()`, `_validate_file_path()`
- Async functions: Always prefix with verb: `async def fetch_note()`, `async def create_note()`

**Constants**
- SCREAMING_SNAKE_CASE: `MAX_TOKENS`, `DEFAULT_MODEL`, `VAULT_BASE_PATH`

**Agent Tools**
- Verb-first, descriptive: `read_note()`, `search_vault()`, `update_note_metadata()`

### Code Examples

#### Complete Chat Feature

```python
# app/chat/agent.py - Agent definition and dependencies
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from app.dependencies import VaultClient, logger

@dataclass
class ChatDependencies:
    """Dependencies for chat agent."""
    vault: VaultClient
    trace_id: str

    async def get_vault_stats(self) -> dict:
        """Get current vault statistics."""
        return await self.vault.get_stats()

chat_agent = Agent(
    "openai:gpt-4",
    deps_type=ChatDependencies,
    retries=2,
    system_prompt="""You are a helpful assistant integrated with Obsidian.
    You can read, search, create, and update notes in the user's vault."""
)

@chat_agent.system_prompt
async def dynamic_system_prompt(ctx: RunContext[ChatDependencies]) -> str:
    """Generate system prompt with vault context."""
    vault_stats = await ctx.deps.get_vault_stats()
    return f"Vault has {vault_stats['note_count']} notes."


# app/chat/tools.py - All agent tools
from pydantic_ai import RunContext
from app.dependencies import logger
from .agent import chat_agent, ChatDependencies

@chat_agent.tool
async def read_note(ctx: RunContext[ChatDependencies], note_path: str) -> str:
    """Read the content of a note from the vault.

    Args:
        ctx: Context containing vault file system access
        note_path: Relative path to note (e.g., "folder/note.md")

    Returns:
        The full content of the note
    """
    logger.info("read_note_tool_called", extra={
        "note_path": note_path,
        "trace_id": ctx.deps.trace_id
    })
    return await ctx.deps.vault.read_file(note_path)

@chat_agent.tool
async def search_vault(
    ctx: RunContext[ChatDependencies],
    query: str,
    limit: int = 10
) -> list[str]:
    """Search notes in the vault.

    Args:
        ctx: Context with vault access
        query: Search query string
        limit: Maximum results to return

    Returns:
        List of matching note paths
    """
    logger.info("search_vault_tool_called", extra={
        "query": query,
        "limit": limit,
        "trace_id": ctx.deps.trace_id
    })
    return await ctx.deps.vault.search(query, limit=limit)


# app/chat/router.py - API endpoints and service logic
import time
import uuid
from typing import Literal
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Request
from pydantic_ai.usage import UsageLimits
from app.dependencies import VaultClient, get_vault_client, logger
from .agent import chat_agent, ChatDependencies

# Request/Response Models
class ChatMessage(BaseModel):
    """A single message in a chat conversation."""
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)

class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(default="default")
    messages: list[ChatMessage]
    max_tokens: int = Field(default=2048, gt=0, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False

class Choice(BaseModel):
    """A completion choice."""
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls"]

class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]

# Service Function
async def process_chat_completion(
    request: ChatCompletionRequest,
    deps: ChatDependencies
) -> ChatCompletionResponse:
    """Process a chat completion request using the agent."""
    logger.info("processing_chat_completion", extra={
        "trace_id": deps.trace_id,
        "message_count": len(request.messages),
        "model": request.model
    })

    try:
        result = await chat_agent.run(
            user_prompt=request.messages[-1].content,
            deps=deps,
            usage_limits=UsageLimits(request_tokens_limit=request.max_tokens)
        )

        logger.info("chat_completion_success", extra={"trace_id": deps.trace_id})
        return ChatCompletionResponse(
            id=deps.trace_id,
            created=int(time.time()),
            model=request.model,
            choices=[Choice(
                message=ChatMessage(role="assistant", content=result.output),
                finish_reason="stop"
            )]
        )
    except Exception as e:
        logger.error("chat_completion_failed", extra={
            "trace_id": deps.trace_id,
            "error": str(e)
        }, exc_info=True)
        raise

# Router
router = APIRouter(prefix="/v1", tags=["chat"])

@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    vault: VaultClient = Depends(get_vault_client),
    req: Request = None
) -> ChatCompletionResponse:
    """OpenAI-compatible chat completion endpoint."""
    trace_id = req.headers.get("X-Trace-Id", str(uuid.uuid4()))
    deps = ChatDependencies(vault=vault, trace_id=trace_id)
    return await process_chat_completion(request, deps)


# app/chat/__init__.py - Export router
from .router import router

__all__ = ["router"]
```

### Docstring Format (Google Style)

```python
async def search_notes(
    ctx: RunContext[VaultDependencies],
    query: str,
    limit: int = 10
) -> list[NoteSearchResult]:
    """Search notes in the vault by content or title.

    This tool performs full-text search across all notes in the vault
    and returns matching results sorted by relevance.

    Args:
        ctx: Context with vault file system access
        query: Search query string (2-100 characters)
        limit: Maximum number of results to return (1-50)

    Returns:
        List of matching notes with title, path, and excerpt

    Raises:
        VaultAccessError: If vault is inaccessible
        ValidationError: If query is empty or limit is invalid
    """
```

---

## 5. Logging

### Structured Logging Format

Use Python's structlog or standard logging with JSON formatter:

```python
# app/dependencies.py
import logging
import json
from typing import Any

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            **getattr(record, "extra", {})
        }
        return json.dumps(log_data)

logger = logging.getLogger("pivloop_agent")
```

### What to Log

**Every agent tool call:**
```python
logger.info("tool_called", extra={
    "tool_name": "read_note",
    "note_path": note_path,
    "trace_id": ctx.deps.trace_id
})
```

**Service operations:**
```python
logger.info("operation_started", extra={
    "operation": "chat_completion",
    "feature": "chat",
    "trace_id": trace_id
})
```

**Errors with context:**
```python
logger.error("file_read_failed", extra={
    "feature": "vault",
    "file_path": path,
    "error_type": type(e).__name__,
    "trace_id": trace_id
}, exc_info=True)
```

### Log Levels

- **DEBUG**: Agent prompt construction, tool parameter details
- **INFO**: Tool calls, operations, successful completions
- **WARNING**: Rate limits, retries, fallback behavior
- **ERROR**: Exceptions, failed operations, validation errors

---

## 6. Testing

### Testing Framework

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific feature tests
uv run pytest src/features/chat/tests/
```

### Test File Structure

```
app/tests/
├── conftest.py           # Shared fixtures
├── test_chat.py          # Chat feature tests
├── test_notes.py         # Notes feature tests
└── test_vault.py         # Vault feature tests
```

### Test Patterns

```python
# app/tests/test_chat.py - Testing chat feature
import pytest
from unittest.mock import Mock, AsyncMock
from pydantic_ai import RunContext
from app.chat.agent import ChatDependencies
from app.chat.tools import read_note
from app.chat.router import process_chat_completion
from app.dependencies import VaultClient

@pytest.fixture
def mock_vault_client():
    """Mock vault client for testing."""
    client = Mock(spec=VaultClient)
    client.read_file = AsyncMock(return_value="# Test Note\nContent")
    client.get_stats = AsyncMock(return_value={"note_count": 42})
    return client

@pytest.fixture
def chat_deps(mock_vault_client):
    """Create chat dependencies with mocked vault."""
    return ChatDependencies(vault=mock_vault_client, trace_id="test-123")

@pytest.mark.asyncio
async def test_read_note_tool(chat_deps):
    """Test read_note tool function."""
    ctx = Mock(spec=RunContext)
    ctx.deps = chat_deps

    result = await read_note(ctx, "test.md")

    assert result == "# Test Note\nContent"
    chat_deps.vault.read_file.assert_called_once_with("test.md")

@pytest.mark.asyncio
async def test_process_chat_completion(chat_deps):
    """Test chat completion processing."""
    from app.chat.router import ChatCompletionRequest, ChatMessage

    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Hello")]
    )

    response = await process_chat_completion(request, chat_deps)

    assert response.choices[0].message.role == "assistant"
    assert len(response.choices[0].message.content) > 0
    assert response.id == "test-123"

def test_chat_completions_endpoint(client: TestClient, mock_vault_client):
    """Test chat completions API endpoint."""
    from fastapi.testclient import TestClient

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Test"}]
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert data["choices"][0]["message"]["role"] == "assistant"
```

---

## 7. OpenAI API Contract

### Required Endpoints

**POST /v1/chat/completions**
- OpenAI-compatible chat completion
- Supports streaming with `stream: true`
- Returns OpenAI response format

### Request Format

```python
# Matches OpenAI spec
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
```

### Response Format

```python
class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: UsageInfo

class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"]

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

### Streaming Format

When `stream: true`, use Server-Sent Events (SSE):

```python
from fastapi.responses import StreamingResponse

@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(request),
            media_type="text/event-stream"
        )
    # Non-streaming response
```

---

## 8. Common Patterns

### Pattern 1: Complete Feature Folder Structure

```python
# app/notes/agent.py - Agent and dependencies
from dataclasses import dataclass
from pydantic_ai import Agent
from app.dependencies import VaultClient

@dataclass
class NoteDependencies:
    """Dependencies for notes agent."""
    vault: VaultClient
    trace_id: str

note_agent = Agent(
    "openai:gpt-4",
    deps_type=NoteDependencies,
    system_prompt="You are an expert at managing and organizing notes."
)


# app/notes/tools.py - Note management tools
from pydantic_ai import RunContext
from app.dependencies import logger
from .agent import note_agent, NoteDependencies

@note_agent.tool
async def create_note(
    ctx: RunContext[NoteDependencies],
    title: str,
    content: str,
    folder: str = ""
) -> str:
    """Create a new note in the vault.

    Args:
        ctx: Context with vault file system access
        title: Note title (used for filename)
        content: Markdown content for the note
        folder: Optional folder path (e.g., "daily/2025")

    Returns:
        Path to the created note
    """
    logger.info("create_note_tool", extra={
        "title": title,
        "folder": folder,
        "trace_id": ctx.deps.trace_id
    })

    # Sanitize filename
    filename = f"{title.replace(' ', '-').lower()}.md"
    full_path = f"{folder}/{filename}" if folder else filename

    await ctx.deps.vault.write_file(full_path, content)
    logger.info("note_created", extra={"path": full_path})
    return full_path

@note_agent.tool
async def update_note_content(
    ctx: RunContext[NoteDependencies],
    path: str,
    new_content: str
) -> str:
    """Update the content of an existing note.

    Args:
        ctx: Context with vault access
        path: Path to the note
        new_content: New markdown content

    Returns:
        Confirmation message
    """
    await ctx.deps.vault.write_file(path, new_content)
    return f"Updated {path} successfully"


# app/notes/router.py - API endpoints
import uuid
from datetime import datetime
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, status
from app.dependencies import VaultClient, get_vault_client, logger
from .agent import note_agent, NoteDependencies

# Models
class NoteUpdateRequest(BaseModel):
    path: str
    instruction: str

class NoteUpdateResponse(BaseModel):
    path: str
    content: str
    updated_at: datetime

# Service
async def update_note(
    request: NoteUpdateRequest,
    deps: NoteDependencies
) -> NoteUpdateResponse:
    """Update a note using AI agent."""
    logger.info("updating_note", extra={
        "path": request.path,
        "trace_id": deps.trace_id
    })

    try:
        result = await note_agent.run(
            user_prompt=f"Update note at {request.path}: {request.instruction}",
            deps=deps
        )

        return NoteUpdateResponse(
            path=request.path,
            content=result.output,
            updated_at=datetime.now()
        )
    except Exception as e:
        logger.error("note_update_failed", extra={
            "path": request.path,
            "trace_id": deps.trace_id,
            "error": str(e)
        }, exc_info=True)
        raise

# Router
router = APIRouter(prefix="/api/notes", tags=["notes"])

@router.post("/update", response_model=NoteUpdateResponse)
async def update_note_endpoint(
    request: NoteUpdateRequest,
    vault: VaultClient = Depends(get_vault_client)
) -> NoteUpdateResponse:
    """Update a note using AI instructions."""
    deps = NoteDependencies(vault=vault, trace_id=str(uuid.uuid4()))
    try:
        return await update_note(request, deps)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {e}"
        )


# app/notes/__init__.py - Export router
from .router import router

__all__ = ["router"]
```

### Pattern 2: Shared Dependencies File

```python
# app/dependencies.py - Shared utilities and dependencies
import logging
from dataclasses import dataclass
from typing import AsyncIterator
import httpx

# Logging setup
logger = logging.getLogger("pivloop_agent")

# Vault client
@dataclass
class VaultClient:
    """Client for interacting with Obsidian vault."""
    base_path: str
    http_client: httpx.AsyncClient

    async def read_file(self, path: str) -> str:
        """Read file from vault."""
        ...

    async def write_file(self, path: str, content: str) -> None:
        """Write file to vault."""
        ...

    async def get_stats(self) -> dict:
        """Get vault statistics."""
        ...

# Dependency providers
async def get_vault_client() -> AsyncIterator[VaultClient]:
    """FastAPI dependency for vault client."""
    async with httpx.AsyncClient() as client:
        yield VaultClient(
            base_path="/path/to/vault",
            http_client=client
        )
```

### Pattern 3: Main App File

```python
# app/main.py - FastAPI application entry point
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers from feature modules
from app.chat import router as chat_router
from app.notes import router as notes_router
from app.vault import router as vault_router

app = FastAPI(
    title="Obsidian PivLoop Agent",
    description="AI agent for Obsidian with OpenAI-compatible API",
    version="1.0.0"
)

# CORS for Obsidian plugin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["app://obsidian.md"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register feature routers
app.include_router(chat_router)
app.include_router(notes_router)
app.include_router(vault_router)

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Obsidian PivLoop Agent",
        "version": "1.0.0",
        "docs": "/docs"
    }
```

---

## 9. Development Commands

### Setup

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
uv pip install -e ".[dev]"

# Or use pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Development Server

```bash
# Run FastAPI server with hot reload
uv run uvicorn src.main:app --reload --port 8000

# With custom host and logging
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --log-level info
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=app --cov-report=html --cov-report=term

# Run specific feature tests
uv run pytest app/tests/test_chat.py -v

# Run with markers
uv run pytest -m "not integration"
```

### Linting & Formatting

```bash
# Format code
uv run ruff format app/

# Lint code
uv run ruff check app/

# Lint with auto-fix
uv run ruff check app/ --fix

# Type checking
uv run mypy app/
```

### Documentation

```bash
# Generate API docs (FastAPI auto-generates)
# Visit http://localhost:8000/docs after starting server

# Generate code documentation
uv run pdoc app/ --html --output-dir docs/
```

---

## 10. AI Coding Assistant Instructions

When working on this codebase, AI assistants MUST:

1. **Always check Archon MCP for active tasks** - Use `find_tasks()` before starting work
2. **Search PydanticAI docs via Archon** - Use `rag_search_knowledge_base()` for best practices
3. **One agent = one folder in app/** - Each feature gets folder with agent.py + tools.py + router.py
4. **Use RunContext for all agent tools** - Never access vault/services directly in tools
5. **Type hint everything** - Functions, variables, class attributes all need type hints
6. **Log all operations** - Use structured logging with trace_id and contextual fields
7. **Test before committing** - Run `uv run pytest` and `uv run ruff check` before any commit
8. **Validate OpenAI compatibility** - All chat endpoints must match OpenAI spec exactly
9. **Document your changes** - Update docstrings, add comments for complex logic
10. **Use Archon for task management** - Mark tasks in_progress/completed via `manage_task()`

### Before Implementing a Feature:
- Search Archon RAG for relevant documentation
- Check reference guides in `reference/` directory
- Review existing feature folders in `app/` for patterns
- Create task breakdown in Archon MCP

### During Implementation:
- Create feature folder: `app/<feature>/`
- Create 3-4 files: `agent.py`, `tools.py`, `router.py`, `__init__.py`
- **agent.py**: Dependencies dataclass + Agent instance + system prompts
- **tools.py**: Import agent from agent.py, define all `@agent.tool` functions
- **router.py**: Pydantic models + service functions + FastAPI router + endpoints
- **__init__.py**: Export router only (`from .router import router`)
- Log at each layer with trace_id
- Add type hints and docstrings as you write
- Write tests in `app/tests/test_<feature>.py`

### After Implementation:
- Run full test suite: `uv run pytest --cov`
- Lint and format: `uv run ruff check app/ && ruff format app/`
- Type check: `uv run mypy app/`
- Update Archon tasks to "review" status
- Test OpenAI compatibility with actual Obsidian Co-Pilot plugin

---

## Additional Resources

### Research Sources

**Vertical Slice Architecture:**
- [fastapi-vslice GitHub](https://github.com/massimobiagioli/fastapi-vslice)
- [FastAPI Best Practices](https://github.com/zhanymkanov/fastapi-best-practices)
- [Embracing Vertical Slices](https://leapcell.io/blog/embracing-vertical-slices-beyond-n-tier-architectures)

**OpenAI-Compatible APIs:**
- [Building OpenAI-Compatible API](https://github.com/ritun16/openai-compatible-fastapi)
- [How to build an OpenAI-compatible API](https://towardsdatascience.com/how-to-build-an-openai-compatible-api-87c8edea2f06/)
- [vLLM OpenAI Server](https://deepwiki.com/vllm-project/vllm/4.1-openai-compatible-api)

**Documentation:**
- PydanticAI: https://ai.pydantic.dev
- FastAPI: https://fastapi.tiangolo.com
- Obsidian API: https://docs.obsidian.md

---

**Last Updated:** 2025-11-23
**Maintained By:** Development Team
**Questions?** Check Archon MCP knowledge base or reference guides first
