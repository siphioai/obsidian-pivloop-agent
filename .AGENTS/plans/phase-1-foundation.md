# Phase 1: Foundation (Core Infrastructure)

## Overview

**Goal:** Establish FastAPI server with OpenAI-compatible endpoint and basic PydanticAI agent (no tools yet).

**Validation:** Obsidian Co-Pilot connects and has basic conversations with the agent.

**Key Deliverables:**
- FastAPI server with CORS for Obsidian (`app://obsidian.md`)
- `/v1/chat/completions` endpoint (OpenAI format)
- PydanticAI agent using `anthropic:claude-haiku-4-5`
- VaultClient prepared for future tools
- Structured JSON logging

---

## Vertical Slice Architecture

This project organizes code by **features** (vertical slices), not layers. Each slice is self-contained:

```
app/
├── chat/                    # Feature slice
│   ├── __init__.py          # Exports router
│   ├── agent.py             # Dependencies + Agent
│   ├── models.py            # Pydantic models
│   └── router.py            # API endpoints
├── config.py                # Shared settings
├── dependencies.py          # Shared utilities
└── main.py                  # App entry point
```

### File Responsibilities

| File | Contains |
|------|----------|
| `agent.py` | `@dataclass` Dependencies, Agent instance, system prompt |
| `models.py` | Pydantic request/response models |
| `router.py` | FastAPI routes, service functions |
| `__init__.py` | `from .router import router` |

### Key Patterns

**1. Dependencies Dataclass** - Inject services via PydanticAI's RunContext:
```python
@dataclass
class ChatDependencies:
    vault: VaultClient
    trace_id: str
```

**2. Agent Definition** - Module-level singleton:
```python
chat_agent = Agent("anthropic:claude-haiku-4-5", deps_type=ChatDependencies, ...)
```

**3. Tool Functions** - First param is always `ctx: RunContext[Deps]`:
```python
@chat_agent.tool
async def read_note(ctx: RunContext[ChatDependencies], path: str) -> str:
    return await ctx.deps.vault.read_file(path)
```

**4. Router Pattern** - Fresh deps per request:
```python
@router.post("/chat/completions")
async def chat(request: Request, vault: VaultClient = Depends(get_vault_client)):
    deps = ChatDependencies(vault=vault, trace_id=str(uuid.uuid4()))
    result = await chat_agent.run(prompt, deps=deps)
```

---

## Files to Create

```
obsidian-pivloop-agent/
├── pyproject.toml
├── .env.example
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── dependencies.py
│   ├── main.py
│   ├── chat/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── models.py
│   │   └── router.py
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py
│       └── test_chat.py
```

---

## Task 1: pyproject.toml

```toml
[project]
name = "obsidian-pivloop-agent"
version = "0.1.0"
description = "AI agent for Obsidian with OpenAI-compatible API"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "pydantic-ai>=0.1.0",
    "anthropic>=0.40.0",
    "httpx>=0.27.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-asyncio>=0.24.0", "pytest-cov>=5.0.0", "ruff>=0.7.0", "mypy>=1.13.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["app/tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
```

**Validate:** `uv sync`

---

## Task 2: .env.example

```env
ANTHROPIC_API_KEY=sk-ant-your-api-key-here
VAULT_PATH=/path/to/your/obsidian/vault
HOST=127.0.0.1
PORT=8000
LOG_LEVEL=INFO
ALLOWED_ORIGINS=app://obsidian.md
```

---

## Task 3: app/config.py

```python
"""Environment configuration using Pydantic Settings."""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    anthropic_api_key: str
    vault_path: Path
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    allowed_origins: str = "app://obsidian.md"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

---

## Task 4: app/dependencies.py

```python
"""Shared dependencies: VaultClient and structured logger."""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator
from app.config import get_settings

# JSON Logger
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {"timestamp": self.formatTime(record), "level": record.levelname,
                "message": record.getMessage(), "module": record.module}
        if extra := getattr(record, "extra", {}):
            data.update(extra)
        return json.dumps(data)

def setup_logging() -> logging.Logger:
    settings = get_settings()
    logger = logging.getLogger("pivloop_agent")
    logger.setLevel(getattr(logging, settings.log_level.upper()))
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger

logger = setup_logging()

# Vault Exceptions
class VaultError(Exception): pass
class VaultNotFoundError(VaultError): pass
class VaultSecurityError(VaultError): pass

# VaultClient
@dataclass
class VaultClient:
    vault_path: Path

    def _validate_path(self, relative_path: str) -> Path:
        full_path = (self.vault_path / relative_path).resolve()
        if not full_path.is_relative_to(self.vault_path.resolve()):
            raise VaultSecurityError(f"Path traversal detected: {relative_path}")
        return full_path

    async def read_file(self, path: str) -> str:
        full_path = self._validate_path(path)
        if not full_path.exists():
            raise VaultNotFoundError(f"File not found: {path}")
        return full_path.read_text(encoding="utf-8")

    async def write_file(self, path: str, content: str) -> None:
        full_path = self._validate_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    async def list_files(self, folder: str = "", pattern: str = "*.md") -> list[str]:
        base = self._validate_path(folder) if folder else self.vault_path
        return sorted(str(f.relative_to(self.vault_path)) for f in base.rglob(pattern) if f.is_file())

    async def get_stats(self) -> dict[str, str | int]:
        return {"vault_path": str(self.vault_path), "note_count": len(await self.list_files())}

# Dependency Provider
async def get_vault_client() -> AsyncIterator[VaultClient]:
    yield VaultClient(vault_path=get_settings().vault_path)
```

---

## Task 5: app/chat/agent.py

```python
"""PydanticAI agent definition for chat feature."""
from dataclasses import dataclass
from pydantic_ai import Agent
from app.dependencies import VaultClient

@dataclass
class ChatDependencies:
    """Dependencies injected into tools via RunContext."""
    vault: VaultClient
    trace_id: str

SYSTEM_PROMPT = """You are a helpful AI assistant integrated with Obsidian.
You help users manage their knowledge base through conversation.
Currently you can chat; vault tools will be added in future phases."""

chat_agent = Agent(
    "anthropic:claude-haiku-4-5",
    deps_type=ChatDependencies,
    retries=2,
    system_prompt=SYSTEM_PROMPT,
)
```

---

## Task 6: app/chat/models.py

```python
"""OpenAI-compatible Pydantic models."""
from typing import Literal
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)

class ChatCompletionRequest(BaseModel):
    model: str = "pivloop-agent"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=2048, gt=0, le=16384)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False

class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] = "stop"

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
```

---

## Task 7: app/chat/router.py

```python
"""FastAPI router for /v1/chat/completions endpoint."""
import time
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request, status
from app.chat.agent import ChatDependencies, chat_agent
from app.chat.models import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, UsageInfo
from app.dependencies import VaultClient, get_vault_client, logger

router = APIRouter(prefix="/v1", tags=["chat"])

async def process_chat_completion(request: ChatCompletionRequest, deps: ChatDependencies) -> ChatCompletionResponse:
    logger.info("processing_chat", extra={"trace_id": deps.trace_id, "messages": len(request.messages)})

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail={"error": {"message": "No user message", "type": "invalid_request"}})

    try:
        result = await chat_agent.run(user_prompt=user_messages[-1].content, deps=deps)
        return ChatCompletionResponse(
            id=f"chatcmpl-{deps.trace_id}",
            created=int(time.time()),
            model=request.model,
            choices=[Choice(message=ChatMessage(role="assistant", content=result.output), finish_reason="stop")],
        )
    except Exception as e:
        logger.error("chat_failed", extra={"trace_id": deps.trace_id, "error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail={"error": {"message": str(e), "type": "server_error"}})

@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, req: Request, vault: VaultClient = Depends(get_vault_client)):
    deps = ChatDependencies(vault=vault, trace_id=req.headers.get("X-Trace-Id", str(uuid.uuid4())))
    return await process_chat_completion(request, deps)
```

---

## Task 8: app/chat/__init__.py

```python
from app.chat.router import router
__all__ = ["router"]
```

---

## Task 9: app/main.py

```python
"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.chat import router as chat_router
from app.config import get_settings
from app.dependencies import logger

settings = get_settings()

app = FastAPI(title="Obsidian PivLoop Agent", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins_list,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(chat_router)

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "0.1.0", "vault_path": str(settings.vault_path)}

@app.get("/")
async def root():
    return {"name": "Obsidian PivLoop Agent", "version": "0.1.0", "docs": "/docs"}

logger.info("app_startup", extra={"host": settings.host, "port": settings.port})
```

---

## Task 10: app/__init__.py

```python
"""Obsidian PivLoop Agent."""
```

---

## Task 11: app/tests/conftest.py

```python
"""Shared pytest fixtures."""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.chat.agent import ChatDependencies
from app.dependencies import VaultClient
from app.main import app

@pytest.fixture
def mock_vault_path(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "test.md").write_text("# Test")
    return vault

@pytest.fixture
def mock_vault_client(mock_vault_path: Path) -> VaultClient:
    return VaultClient(vault_path=mock_vault_path)

@pytest.fixture
def chat_deps(mock_vault_client: VaultClient) -> ChatDependencies:
    return ChatDependencies(vault=mock_vault_client, trace_id="test-123")

@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
```

---

## Task 12: app/tests/test_chat.py

```python
"""Tests for chat feature."""
import pytest
from fastapi.testclient import TestClient
from app.chat.models import ChatCompletionRequest, ChatMessage

class TestModels:
    def test_chat_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"

    def test_empty_content_fails(self):
        with pytest.raises(ValueError):
            ChatMessage(role="user", content="")

class TestEndpoints:
    def test_health(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_root(self, client: TestClient):
        r = client.get("/")
        assert r.status_code == 200
        assert "PivLoop" in r.json()["name"]

    def test_chat_requires_messages(self, client: TestClient):
        r = client.post("/v1/chat/completions", json={"model": "test"})
        assert r.status_code == 422
```

---

## Task 13: app/tests/__init__.py

```python
"""Tests package."""
```

---

## Validation Commands

```bash
# Install dependencies
uv sync

# Lint & format
uv run ruff check app/ && uv run ruff format app/ --check

# Type check
uv run mypy app/ --ignore-missing-imports

# Run tests
uv run pytest app/tests/ -v

# Start server
uv run uvicorn app.main:app --reload --port 8000

# Test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

---

## Obsidian Co-Pilot Setup

1. Open Obsidian → Settings → Co-Pilot → Custom Models
2. Add Custom Model:
   - **Name:** `PivLoop Agent`
   - **Provider:** `3rd party (openai-format)`
   - **Base URL:** `http://localhost:8000`
   - **API Key:** (leave empty)
3. Select model and test with "Hello!"

---

## Acceptance Criteria

- [ ] Server starts without errors
- [ ] `/health` returns healthy status
- [ ] `/v1/chat/completions` accepts requests
- [ ] Response matches OpenAI format
- [ ] CORS allows `app://obsidian.md`
- [ ] All tests pass
- [ ] Co-Pilot connects and receives responses

---

## Next Phase

After validation, proceed to **Phase 2: note_operations tool** which adds CRUD operations using the same vertical slice pattern in `app/notes/`.
