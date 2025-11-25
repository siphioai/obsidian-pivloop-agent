# Obsidian PivLoop Agent - Product Requirements Document

**Version:** 1.1
**Created:** 2025-11-25
**Updated:** 2025-11-25
**Status:** Draft

---

## 1. Executive Summary

Obsidian PivLoop Agent is an AI-powered assistant that integrates seamlessly with Obsidian through the Co-Pilot community plugin. Built using PydanticAI, FastAPI, and the Claude API, the agent provides natural language control over an Obsidian vault—enabling users to create, read, update, delete, search, tag, link, and analyze their notes without leaving the Co-Pilot chat interface.

The agent exposes an OpenAI-compatible `/v1/chat/completions` endpoint, allowing it to function as a drop-in replacement for any OpenAI model within Co-Pilot's custom model configuration. This design ensures zero vendor lock-in while providing powerful, context-aware vault management capabilities.

**MVP Goal:** Deliver a fully functional AI agent that enables natural language vault management including CRUD operations, intelligent search, automated tagging with note connections via wikilinks, and analytical insights—all accessible through Obsidian Co-Pilot's chat interface.

**Implementation Approach:** Iterative, tool-by-tool development. Each tool is planned, executed, and validated independently before moving to the next, ensuring quality and allowing for learnings to inform subsequent development.

---

## 2. Mission

### Mission Statement

Empower Obsidian users with an intelligent, conversational assistant that transforms how they interact with their knowledge base—making note management, discovery, and insight generation as simple as having a conversation.

### Core Principles

1. **Local-First:** All vault operations happen on the user's local file system. Notes never leave their machine.
2. **Natural Language Native:** Users interact through conversation, not commands or complex syntax.
3. **Connection-Driven:** The agent actively creates meaningful connections between notes through intelligent tagging and wikilinks.
4. **Insight-Generating:** Beyond CRUD operations, the agent surfaces patterns, trends, and insights from the user's knowledge base.
5. **Standards-Compliant:** Full OpenAI API compatibility ensures seamless integration with existing tools and future portability.

---

## 3. Target Users

### Primary Persona: Knowledge Worker

**Profile:**
- Uses Obsidian for personal knowledge management, note-taking, project documentation, or journaling
- Comfortable with technology but prefers natural language over complex queries
- Values their time and wants to reduce friction in vault management
- Already uses or is interested in AI assistants for productivity

**Technical Comfort Level:** Intermediate
- Can install Obsidian plugins
- Can configure API endpoints and keys
- May not be familiar with programming or command-line tools

**Key Needs:**
- Quick note creation and updates without context-switching
- Finding relevant notes without remembering exact titles or locations
- Maintaining organization through consistent tagging
- Understanding patterns in their note-taking habits
- Discovering connections between disparate notes

**Pain Points:**
- Manual tagging is tedious and inconsistent
- Search requires knowing what to search for
- Notes become isolated "orphans" without connections
- No visibility into note-taking patterns or trends
- Existing AI tools don't understand their vault structure

---

## 4. MVP Scope

### In Scope

**Core Functionality**
- ✅ OpenAI-compatible `/v1/chat/completions` endpoint
- ✅ Natural language understanding for vault operations
- ✅ Note CRUD operations (create, read, update, delete)
- ✅ Note summarization
- ✅ Full-text vault search
- ✅ Tag-based search and filtering
- ✅ Wikilink-based search (find notes linking to/from a note)
- ✅ Date-range filtering for searches
- ✅ Tag management (add, remove, rename)
- ✅ Intelligent tag suggestions based on content
- ✅ Auto-tagging capability
- ✅ Wikilink creation for note connections
- ✅ Vault analytics (note counts, trends, distributions)
- ✅ Both narrative and structured analytics output

**Technical**
- ✅ PydanticAI agent with Claude API backend
- ✅ FastAPI server with async support
- ✅ Direct file system vault access
- ✅ YAML frontmatter parsing and manipulation
- ✅ Markdown content handling
- ✅ Structured JSON logging
- ✅ Type-safe implementation with Pydantic models
- ✅ Comprehensive error handling

**Integration**
- ✅ Obsidian Co-Pilot compatibility via OpenAI format
- ✅ CORS support for Obsidian plugin requests
- ✅ Streaming response support (SSE)
- ✅ Non-streaming response support

**Deployment**
- ✅ Local development server (uvicorn)
- ✅ Configuration via environment variables
- ✅ Health check endpoint

### Out of Scope

**Deferred Features**
- ❌ Obsidian Local REST API integration (future enhancement)
- ❌ Multi-vault support (single vault for MVP)
- ❌ Graph visualization of note connections
- ❌ Embedding-based semantic search
- ❌ Note templates and template application
- ❌ Scheduled/automated operations
- ❌ Backup and version history
- ❌ Plugin marketplace distribution
- ❌ Web-based configuration UI
- ❌ Mobile/sync considerations

**Technical Exclusions**
- ❌ Cloud deployment (local-only for MVP)
- ❌ User authentication (single-user local deployment)
- ❌ Database persistence (file system is source of truth)
- ❌ Rate limiting (local deployment)

---

## 5. User Stories

### Primary User Stories

**US-1: Quick Note Creation**
> As a knowledge worker, I want to create a new note by describing it in natural language, so that I can capture ideas without leaving my current context.

*Example:* "Create a note called 'Project Alpha Meeting Notes' in the Projects folder with a summary of today's standup discussion about the API redesign."

---

**US-2: Note Discovery**
> As a knowledge worker, I want to find notes using natural language queries, so that I don't need to remember exact titles or locations.

*Example:* "Find all notes about API design from last month" or "What notes mention authentication?"

---

**US-3: Intelligent Tagging**
> As a knowledge worker, I want the agent to suggest and apply relevant tags to my notes, so that my vault stays organized without manual effort.

*Example:* "Suggest tags for my note on React hooks" → Agent analyzes content and suggests `#programming`, `#react`, `#frontend`, `#hooks`

---

**US-4: Note Connections**
> As a knowledge worker, I want the agent to identify and create connections between related notes, so that my knowledge graph grows organically.

*Example:* "Connect this note to other related notes in my vault" → Agent finds related notes and adds `[[wikilinks]]` to relevant content.

---

**US-5: Content Updates**
> As a knowledge worker, I want to update notes through conversation, so that I can make changes without opening the note directly.

*Example:* "Add a new section to my 'Weekly Review' note about the completed marketing campaign"

---

**US-6: Vault Analytics**
> As a knowledge worker, I want to understand my note-taking patterns and trends, so that I can optimize my knowledge management habits.

*Example:* "Tell me my trends over the last month" → Agent provides narrative summary + structured data about note creation, tag usage, and activity patterns.

---

**US-7: Note Summarization**
> As a knowledge worker, I want to get summaries of long notes, so that I can quickly understand their content without reading everything.

*Example:* "Summarize my meeting notes from last week's strategy session"

---

**US-8: Tag Cleanup**
> As a knowledge worker, I want to rename or consolidate tags across my vault, so that I can maintain consistent taxonomy.

*Example:* "Rename all #dev tags to #development" or "What are my most used tags?"

---

### Technical User Stories

**US-T1: OpenAI Compatibility**
> As a system integrator, I want the agent to expose an OpenAI-compatible API, so that it works with any tool expecting OpenAI format.

---

**US-T2: Streaming Responses**
> As a user, I want responses to stream in real-time, so that I see progress on longer operations.

---

## 6. Core Architecture & Patterns

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Obsidian App                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Co-Pilot Plugin                       │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │  Custom Model: "PivLoop Agent"                  │    │    │
│  │  │  Base URL: http://localhost:8000                │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP (OpenAI Format)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PivLoop Agent Server                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   FastAPI Application                    │    │
│  │                                                          │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │    │
│  │  │    /v1/      │  │   /health    │  │     /api/    │   │    │
│  │  │    chat/     │  │              │  │    (future)  │   │    │
│  │  │ completions  │  │              │  │              │   │    │
│  │  └──────┬───────┘  └──────────────┘  └──────────────┘   │    │
│  │         │                                                │    │
│  │         ▼                                                │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │              PydanticAI Agent                    │    │    │
│  │  │                                                  │    │    │
│  │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐   │    │    │
│  │  │  │   note_    │ │   vault_   │ │    tag_    │   │    │    │
│  │  │  │ operations │ │   search   │ │ management │   │    │    │
│  │  │  └────────────┘ └────────────┘ └────────────┘   │    │    │
│  │  │  ┌────────────┐                                  │    │    │
│  │  │  │   vault_   │                                  │    │    │
│  │  │  │ analytics  │                                  │    │    │
│  │  │  └────────────┘                                  │    │    │
│  │  └──────────────────────┬──────────────────────────┘    │    │
│  └─────────────────────────┼────────────────────────────────┘    │
└────────────────────────────┼────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Local File System                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           /Users/hugemarley/Documents/Obsidian Vault     │    │
│  │                                                          │    │
│  │  ├── .obsidian/          (plugin configs)               │    │
│  │  ├── Coding System/      (project notes)                │    │
│  │  ├── copilot/            (conversation logs)            │    │
│  │  ├── Excalidraw/         (drawings)                     │    │
│  │  └── *.md                (notes with YAML frontmatter)  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │ API Calls
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Claude API (Anthropic)                      │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
obsidian-pivloop-agent/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI entry point
│   ├── config.py                  # Environment configuration (Pydantic Settings)
│   ├── dependencies.py            # Shared deps (VaultClient, logger)
│   │
│   ├── chat/                      # OpenAI-compatible chat endpoint
│   │   ├── __init__.py            # Exports: router
│   │   ├── agent.py               # ChatDependencies + chat_agent
│   │   ├── models.py              # OpenAI-compatible Pydantic models
│   │   ├── tools.py               # All @chat_agent.tool functions
│   │   └── router.py              # /v1/chat/completions endpoint
│   │
│   ├── notes/                     # Note operations feature slice
│   │   ├── __init__.py            # Exports: router, note_operations
│   │   ├── agent.py               # NoteDependencies + note_agent
│   │   ├── tools.py               # note_operations tool implementation
│   │   └── router.py              # Direct API routes (optional)
│   │
│   ├── search/                    # Vault search feature slice
│   │   ├── __init__.py            # Exports: router, vault_search
│   │   ├── agent.py               # SearchDependencies + search_agent
│   │   ├── tools.py               # vault_search tool implementation
│   │   └── router.py
│   │
│   ├── tags/                      # Tag management feature slice
│   │   ├── __init__.py            # Exports: router, tag_management
│   │   ├── agent.py               # TagDependencies + tag_agent
│   │   ├── tools.py               # tag_management tool implementation
│   │   └── router.py
│   │
│   ├── analytics/                 # Vault analytics feature slice
│   │   ├── __init__.py            # Exports: router, vault_analytics
│   │   ├── agent.py               # AnalyticsDependencies + analytics_agent
│   │   ├── tools.py               # vault_analytics tool implementation
│   │   └── router.py
│   │
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py            # Shared fixtures (mock_vault, test_client)
│       ├── test_chat.py
│       ├── test_notes.py
│       ├── test_search.py
│       ├── test_tags.py
│       └── test_analytics.py
│
├── .AGENTS/                       # Agent planning and documentation
│   ├── PRD.md
│   ├── plans/                     # Feature implementation plans
│   └── reference/                 # External documentation guides
│
├── .env.example
├── pyproject.toml
├── CLAUDE.md                      # Project coding standards
└── README.md
```

---

## 6.1 Vertical Slice Architecture Patterns

### Core Philosophy

The project follows **Vertical Slice Architecture** where each feature is a self-contained slice with its own folder containing agent, tools, and router. This contrasts with traditional layered architecture (controllers → services → repositories) by organizing code around features rather than technical concerns.

**Benefits:**
- Independent development and testing per feature
- Clear separation of concerns
- Easy feature addition/removal without touching other code
- Minimal merge conflicts in team development
- Each slice can evolve independently

### File Responsibilities Per Slice

Each feature folder contains **3-4 focused files** with specific responsibilities:

| File | Responsibility | Contains |
|------|----------------|----------|
| `agent.py` | Agent configuration | Dependencies dataclass, Agent instance, system prompts |
| `tools.py` | Tool implementations | All `@agent.tool` decorated functions |
| `router.py` | API layer | Pydantic models, FastAPI routes, service logic |
| `__init__.py` | Module exports | Exports router (and optionally tool functions) |

### Pattern 1: Agent Definition (`agent.py`)

```python
# app/notes/agent.py - Agent definition and dependencies
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from app.dependencies import VaultClient, logger

@dataclass
class NoteDependencies:
    """Dependencies injected into all note tools via RunContext."""
    vault: VaultClient
    trace_id: str

    async def get_vault_stats(self) -> dict:
        """Helper method available to tools."""
        return await self.vault.get_stats()

# System prompt defines agent's personality and capabilities
SYSTEM_PROMPT = """You are a helpful assistant for managing notes in an Obsidian vault.
You can create, read, update, delete, and summarize notes.
Always confirm destructive operations before executing."""

note_agent = Agent(
    "anthropic:claude-haiku-4-5",
    deps_type=NoteDependencies,
    retries=2,
    system_prompt=SYSTEM_PROMPT,
)

# Optional: Dynamic system prompt that adds runtime context
@note_agent.system_prompt
async def dynamic_context(ctx: RunContext[NoteDependencies]) -> str:
    """Add vault statistics to system prompt at runtime."""
    stats = await ctx.deps.get_vault_stats()
    return f"The vault currently contains {stats['note_count']} notes."
```

**Key Rules:**
- Dependencies dataclass holds all injected services (VaultClient, trace_id, etc.)
- Agent instance is created at module level (singleton per feature)
- System prompt is static string; use `@agent.system_prompt` decorator for dynamic context
- Never use globals or singletons for dependencies - always inject via dataclass

### Pattern 2: Tool Implementation (`tools.py`)

```python
# app/notes/tools.py - All tool implementations for this feature
from typing import Literal
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from app.dependencies import logger
from .agent import note_agent, NoteDependencies

# Pydantic models for complex tool parameters
class NoteOperationParams(BaseModel):
    """Parameters for note operations."""
    operation: Literal["create", "read", "update", "delete", "summarize"]
    path: str = Field(..., description="Relative path to note (e.g., 'Projects/API.md')")
    content: str | None = Field(None, description="Note content for create/update")
    confirm: bool = Field(False, description="Confirmation for delete operation")

@note_agent.tool
async def note_operations(
    ctx: RunContext[NoteDependencies],
    operation: str,
    path: str,
    content: str | None = None,
    confirm: bool = False,
) -> str:
    """Perform operations on notes in the vault.

    Args:
        ctx: Context with vault access and trace_id
        operation: One of 'create', 'read', 'update', 'delete', 'summarize'
        path: Relative path to the note file
        content: Content for create/update operations
        confirm: Must be True for delete operations

    Returns:
        Operation result message or note content
    """
    logger.info("note_operation_called", extra={
        "operation": operation,
        "path": path,
        "trace_id": ctx.deps.trace_id,
    })

    try:
        if operation == "create":
            await ctx.deps.vault.write_file(path, content or "")
            return f"Created note at {path}"

        elif operation == "read":
            return await ctx.deps.vault.read_file(path)

        elif operation == "update":
            existing = await ctx.deps.vault.read_file(path)
            updated = existing + "\n" + (content or "")
            await ctx.deps.vault.write_file(path, updated)
            return f"Updated note at {path}"

        elif operation == "delete":
            if not confirm:
                return "Please confirm deletion by setting confirm=True"
            await ctx.deps.vault.delete_file(path)
            return f"Deleted note at {path}"

        elif operation == "summarize":
            content = await ctx.deps.vault.read_file(path)
            # Agent will generate summary from returned content
            return f"Content to summarize:\n{content}"

        else:
            return f"Unknown operation: {operation}"

    except Exception as e:
        logger.error("note_operation_failed", extra={
            "operation": operation,
            "path": path,
            "error": str(e),
            "trace_id": ctx.deps.trace_id,
        }, exc_info=True)
        return f"Error: {str(e)}"
```

**Key Rules:**
- First parameter is ALWAYS `ctx: RunContext[DepsType]`
- Use Google-style docstrings with Args/Returns sections
- Docstrings become part of tool schema sent to LLM
- Log all operations with trace_id for debugging
- Handle errors gracefully and return user-friendly messages
- Use Pydantic models for complex parameter validation

### Pattern 3: Router/API Layer (`router.py`)

```python
# app/notes/router.py - API endpoints and service logic
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from app.dependencies import VaultClient, get_vault_client, logger
from .agent import note_agent, NoteDependencies

# Request/Response Pydantic models
class NoteCreateRequest(BaseModel):
    """Request to create a new note."""
    path: str = Field(..., description="Note path relative to vault")
    content: str = Field(..., description="Initial note content")
    tags: list[str] = Field(default_factory=list)

class NoteResponse(BaseModel):
    """Response containing note data."""
    path: str
    content: str
    success: bool = True

# Service function (business logic separate from route handler)
async def create_note_via_agent(
    request: NoteCreateRequest,
    deps: NoteDependencies,
) -> NoteResponse:
    """Create a note using the agent."""
    logger.info("creating_note", extra={
        "path": request.path,
        "trace_id": deps.trace_id,
    })

    result = await note_agent.run(
        user_prompt=f"Create a note at {request.path} with content: {request.content}",
        deps=deps,
    )

    return NoteResponse(path=request.path, content=result.output)

# Router definition
router = APIRouter(prefix="/api/notes", tags=["notes"])

@router.post("/", response_model=NoteResponse)
async def create_note(
    request: NoteCreateRequest,
    req: Request,
    vault: VaultClient = Depends(get_vault_client),
) -> NoteResponse:
    """Create a new note in the vault."""
    trace_id = req.headers.get("X-Trace-Id", str(uuid.uuid4()))
    deps = NoteDependencies(vault=vault, trace_id=trace_id)

    try:
        return await create_note_via_agent(request, deps)
    except Exception as e:
        logger.error("create_note_failed", extra={
            "trace_id": trace_id,
            "error": str(e),
        }, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": str(e), "type": "server_error"}},
        )
```

**Key Rules:**
- Define Pydantic models for all requests/responses
- Separate service functions from route handlers for testability
- Create dependencies (with trace_id) fresh for each request
- Use FastAPI's `Depends()` for dependency injection
- Handle errors and return structured error responses
- All routes should have type hints and docstrings

### Pattern 4: Module Export (`__init__.py`)

```python
# app/notes/__init__.py - Export router for main.py
from .router import router

__all__ = ["router"]
```

**Key Rules:**
- Only export what main.py needs (typically just router)
- Keep imports minimal to avoid circular dependencies

### Dependency Injection Flow

```
Request arrives at FastAPI endpoint
         │
         ▼
┌─────────────────────────────────────┐
│ Route handler receives request       │
│ - Injects VaultClient via Depends() │
│ - Creates trace_id                   │
│ - Builds Dependencies dataclass      │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ Service function processes request   │
│ - Receives Dependencies              │
│ - Calls agent.run() with deps        │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ Agent decides to call tool           │
│ - Tool receives RunContext[Deps]     │
│ - Accesses deps.vault, deps.trace_id │
│ - Performs operation                 │
└────────────────┬────────────────────┘
                 │
                 ▼
       Response returned to user
```

### Key Design Patterns

**1. Vertical Slice Architecture**
Each feature (notes, search, tags, analytics) is a self-contained slice with its own agent, tools, and router. This enables:
- Independent development and testing
- Clear separation of concerns
- Easy feature addition/removal

**2. Dependency Injection via RunContext**
All tools receive dependencies through PydanticAI's `RunContext[DepsType]` pattern:
```python
@dataclass
class NoteDependencies:
    vault: VaultClient
    trace_id: str

@note_agent.tool
async def create_note(ctx: RunContext[NoteDependencies], ...) -> str:
    return await ctx.deps.vault.write_file(...)
```

**3. OpenAI Compatibility Layer**
The chat router transforms between OpenAI format and PydanticAI:
- Incoming: Parse OpenAI `ChatCompletionRequest`
- Processing: Run PydanticAI agent
- Outgoing: Format as OpenAI `ChatCompletionResponse`

**4. Consolidated High-Impact Tools**
Rather than many specific tools, we use 4 multi-operation tools:
- Reduces context window usage
- Simplifies agent decision-making
- Allows complex operations via `operation` parameter

---

## 7. Tools/Features

### Tool 1: `note_operations`

**Purpose:** Handle all note CRUD operations and summarization in a single, consolidated tool.

**Operations:**

| Operation | Description | Parameters |
|-----------|-------------|------------|
| `create` | Create a new note | `path`, `content`, `tags?`, `frontmatter?` |
| `read` | Read note content | `path` |
| `update` | Update note content | `path`, `content?`, `append?`, `section?` |
| `delete` | Delete a note | `path`, `confirm` |
| `summarize` | Generate note summary | `path`, `max_length?` |

**Key Features:**
- Automatic YAML frontmatter generation with timestamps
- Support for folder creation if path doesn't exist
- Content appending vs. replacement modes
- Section-specific updates (by heading)
- Summarization using Claude's understanding

**Example Interactions:**
```
User: "Create a note called 'API Design Decisions' in Projects folder"
Tool Call: note_operations(operation="create", path="Projects/API Design Decisions.md", content="# API Design Decisions\n\nCreated: 2025-11-25")

User: "Add a section about authentication to my API notes"
Tool Call: note_operations(operation="update", path="Projects/API Design Decisions.md", append="## Authentication\n\n", section="end")
```

---

### Tool 2: `vault_search`

**Purpose:** Unified search across the vault with multiple search modes and filters.

**Operations:**

| Operation | Description | Parameters |
|-----------|-------------|------------|
| `fulltext` | Search note contents | `query`, `limit?`, `folder?` |
| `by_tag` | Find notes with specific tags | `tags`, `match_all?`, `limit?` |
| `by_link` | Find notes linking to/from | `note_path`, `direction?`, `limit?` |
| `by_date` | Find notes by date range | `start_date?`, `end_date?`, `date_field?` |
| `combined` | Multi-criteria search | `query?`, `tags?`, `date_range?`, `folder?` |

**Key Features:**
- Full-text search across all markdown content
- Tag filtering with AND/OR logic
- Backlink and forward-link discovery
- Date filtering by creation or modification
- Folder-scoped searches
- Result ranking by relevance
- Excerpt generation for context

**Example Interactions:**
```
User: "Find all notes about React from last month"
Tool Call: vault_search(operation="combined", query="React", date_range={"start": "2025-10-25", "end": "2025-11-25"})

User: "What notes link to my 'Architecture Overview' note?"
Tool Call: vault_search(operation="by_link", note_path="Architecture Overview.md", direction="backlinks")
```

---

### Tool 3: `tag_management`

**Purpose:** Comprehensive tag operations including intelligent suggestions and note connections.

**Operations:**

| Operation | Description | Parameters |
|-----------|-------------|------------|
| `add` | Add tags to a note | `path`, `tags` |
| `remove` | Remove tags from a note | `path`, `tags` |
| `rename` | Rename tag across vault | `old_tag`, `new_tag` |
| `list` | List all tags in vault | `sort_by?`, `limit?` |
| `suggest` | Suggest tags for content | `path` or `content` |
| `auto_tag` | Automatically tag a note | `path`, `confirm?` |
| `connect` | Create wikilinks to related notes | `path`, `limit?` |

**Key Features:**
- Frontmatter tag manipulation (YAML array format)
- Vault-wide tag renaming with atomic updates
- AI-powered tag suggestions based on content analysis
- Related note discovery and wikilink insertion
- Tag usage statistics
- Orphan note detection

**Example Interactions:**
```
User: "Suggest tags for my new meeting notes"
Tool Call: tag_management(operation="suggest", path="Meetings/2025-11-25 Standup.md")
Response: Suggested tags: #meeting, #standup, #team, #project-alpha

User: "Connect this note to related notes in my vault"
Tool Call: tag_management(operation="connect", path="Projects/API Design.md", limit=5)
Response: Added links to: [[Architecture Overview]], [[Authentication Flow]], [[Database Schema]]
```

---

### Tool 4: `vault_analytics`

**Purpose:** Generate insights about vault activity, patterns, and trends.

**Operations:**

| Operation | Description | Parameters |
|-----------|-------------|------------|
| `overview` | General vault statistics | None |
| `trends` | Activity trends over time | `period`, `metric?` |
| `tag_distribution` | Tag usage analysis | `limit?`, `include_orphans?` |
| `activity` | Recent vault activity | `days?`, `type?` |
| `insights` | AI-generated insights | `focus?` |

**Key Features:**
- Note count by folder, tag, and time period
- Creation/modification trends (daily, weekly, monthly)
- Tag usage distribution and growth
- Most active periods identification
- Orphan note identification (unlinked, untagged)
- Dual output: narrative summary + structured data

**Example Interactions:**
```
User: "Tell me my trends over the last month"
Tool Call: vault_analytics(operation="trends", period="month")
Response:
  Narrative: "Over the past month, you created 23 notes—a 40% increase from the previous month. Your most active day was Monday with an average of 4 notes. The #project-alpha tag saw the most growth with 12 new notes..."

  Data: {
    "period": "2025-10-25 to 2025-11-25",
    "notes_created": 23,
    "notes_modified": 45,
    "growth_percent": 40,
    "top_tags": ["project-alpha", "meeting", "dev"],
    "most_active_day": "Monday",
    "orphan_notes": 3
  }

User: "What are my most used tags?"
Tool Call: vault_analytics(operation="tag_distribution", limit=10)
```

---

## 8. Technology Stack

### Backend

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Language | Python | 3.11+ | Runtime |
| Framework | FastAPI | 0.115+ | API server |
| AI Framework | PydanticAI | Latest | Agent orchestration |
| LLM Provider | Anthropic Claude | claude-sonnet-4-20250514 | AI reasoning |
| HTTP Client | httpx | Latest | Async HTTP |
| Validation | Pydantic | 2.x | Data models |

### Development

| Component | Technology | Purpose |
|-----------|------------|---------|
| Package Manager | uv | Dependency management |
| Testing | pytest + pytest-asyncio | Test framework |
| Linting | ruff | Code quality |
| Formatting | ruff format | Code style |
| Type Checking | mypy --strict | Static analysis |

### Dependencies

**Core:**
```toml
[project]
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "pydantic>=2.0.0",
    "pydantic-ai>=0.1.0",
    "anthropic>=0.40.0",
    "httpx>=0.27.0",
    "python-frontmatter>=1.1.0",
    "pyyaml>=6.0.0",
    "python-dateutil>=2.9.0",
]
```

**Development:**
```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.7.0",
    "mypy>=1.13.0",
]
```

### Third-Party Integrations

| Integration | Purpose | Protocol |
|-------------|---------|----------|
| Obsidian Co-Pilot | User interface | OpenAI API format |
| Anthropic Claude API | LLM backend | Anthropic SDK |

---

## 9. Security & Configuration

### Configuration Management

**Environment Variables:**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | - | Claude API key |
| `VAULT_PATH` | Yes | - | Path to Obsidian vault |
| `HOST` | No | `127.0.0.1` | Server bind address |
| `PORT` | No | `8000` | Server port |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |
| `ALLOWED_ORIGINS` | No | `app://obsidian.md` | CORS origins |

**Example `.env`:**
```env
ANTHROPIC_API_KEY=sk-ant-...
VAULT_PATH=/Users/hugemarley/Documents/Obsidian Vault
HOST=127.0.0.1
PORT=8000
LOG_LEVEL=INFO
```

### Security Scope

**In Scope:**
- ✅ Input validation on all endpoints
- ✅ Path traversal prevention (vault-scoped file access)
- ✅ CORS configuration for Obsidian plugin
- ✅ Request size limits
- ✅ Structured logging for audit trail
- ✅ Environment-based secrets management

**Out of Scope (MVP):**
- ❌ Authentication/authorization (local single-user deployment)
- ❌ API key management for endpoint access
- ❌ Rate limiting
- ❌ Encryption at rest

### File Access Security

The `VaultClient` enforces vault-scoped access:
```python
def _validate_path(self, path: str) -> Path:
    """Ensure path is within vault boundaries."""
    resolved = (self.vault_path / path).resolve()
    if not resolved.is_relative_to(self.vault_path):
        raise SecurityError("Path traversal attempt detected")
    return resolved
```

---

## 10. API Specification

### Endpoint: POST `/v1/chat/completions`

**Purpose:** OpenAI-compatible chat completion endpoint for Co-Pilot integration.

**Request Format:**
```json
{
  "model": "pivloop-agent",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful Obsidian vault assistant."
    },
    {
      "role": "user",
      "content": "Create a note about today's meeting"
    }
  ],
  "max_tokens": 2048,
  "temperature": 0.7,
  "stream": false
}
```

**Response Format (Non-Streaming):**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1732550400,
  "model": "pivloop-agent",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I've created a new note at 'Meetings/2025-11-25 Meeting.md' with the following content..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 100,
    "total_tokens": 150
  }
}
```

**Response Format (Streaming):**
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1732550400,"model":"pivloop-agent","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1732550400,"model":"pivloop-agent","choices":[{"index":0,"delta":{"content":"I've"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1732550400,"model":"pivloop-agent","choices":[{"index":0,"delta":{"content":" created"},"finish_reason":null}]}

data: [DONE]
```

### Endpoint: GET `/health`

**Purpose:** Health check for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "vault_connected": true
}
```

### Error Responses

**Validation Error (400):**
```json
{
  "error": {
    "message": "Invalid request: messages array is required",
    "type": "invalid_request_error",
    "code": "validation_error"
  }
}
```

**Server Error (500):**
```json
{
  "error": {
    "message": "Internal server error",
    "type": "server_error",
    "code": "internal_error"
  }
}
```

---

## 11. Success Criteria

### MVP Success Definition

The MVP is successful when a user can:
1. Configure the agent as a custom model in Obsidian Co-Pilot
2. Have a natural conversation to manage their vault
3. Create, find, update, and delete notes through chat
4. Receive intelligent tag suggestions and apply them
5. See connections created between related notes
6. Get meaningful insights about their vault activity

### Functional Requirements

**Chat Integration:**
- ✅ Co-Pilot connects without errors
- ✅ Responses stream in real-time
- ✅ Conversation context is maintained
- ✅ Error messages are user-friendly

**Note Operations:**
- ✅ Notes created with correct frontmatter
- ✅ Notes readable with full content
- ✅ Updates preserve existing structure
- ✅ Deletions require confirmation
- ✅ Summaries are accurate and concise

**Search:**
- ✅ Full-text search returns relevant results
- ✅ Tag filtering works with AND/OR logic
- ✅ Date ranges filter correctly
- ✅ Results include helpful excerpts

**Tags & Connections:**
- ✅ Tags added to frontmatter correctly
- ✅ Tag suggestions are contextually relevant
- ✅ Wikilinks point to existing notes
- ✅ Vault-wide tag rename works atomically

**Analytics:**
- ✅ Statistics are accurate
- ✅ Trends reflect actual activity
- ✅ Narrative insights are meaningful
- ✅ Structured data is parseable

### Quality Indicators

| Metric | Target |
|--------|--------|
| Response latency (non-streaming) | < 3 seconds |
| Streaming first token | < 500ms |
| Search accuracy (relevant in top 5) | > 80% |
| Tag suggestion relevance | > 70% user acceptance |
| Test coverage | > 80% |
| Type check pass | 100% |

### User Experience Goals

- Natural, conversational interaction
- Helpful error messages with recovery suggestions
- Progress feedback for long operations
- Consistent response formatting

---

## 12. Implementation Phases

> **Implementation Philosophy:** This project follows an iterative, validation-driven approach. Each phase goes through a complete **Plan → Execute → Validate** cycle before the next phase begins. This ensures each component works correctly in isolation and integration before adding complexity.

### Development Workflow Per Phase

```
┌─────────────────────────────────────────────────────────────┐
│                    For Each Phase:                          │
│                                                             │
│  1. PLAN (/plan-feature)                                    │
│     └─→ Detailed implementation plan                        │
│     └─→ File structure and code design                      │
│     └─→ Test strategy                                       │
│                                                             │
│  2. EXECUTE (/execute)                                      │
│     └─→ Implement the planned features                      │
│     └─→ Write unit tests                                    │
│     └─→ Integration tests                                   │
│                                                             │
│  3. VALIDATE                                                │
│     └─→ Run test suite                                      │
│     └─→ Manual testing with Co-Pilot                        │
│     └─→ Fix any issues discovered                           │
│     └─→ Confirm ready for next phase                        │
│                                                             │
│  ✓ Phase Complete → Proceed to Next Phase                   │
└─────────────────────────────────────────────────────────────┘
```

---

### Phase 1: Foundation (Core Infrastructure)

**Goal:** Establish project structure, FastAPI server, and OpenAI-compatible endpoint with a basic agent (no tools yet).

**Deliverables:**
- ✅ Project scaffolding with vertical slice architecture
- ✅ FastAPI application with CORS support
- ✅ `/v1/chat/completions` endpoint (basic)
- ✅ `/health` endpoint
- ✅ VaultClient for file system access
- ✅ Pydantic models for OpenAI request/response
- ✅ Basic PydanticAI agent setup (conversational, no tools)
- ✅ Environment configuration
- ✅ Structured logging

**Validation Criteria:**
- [ ] Server starts without errors
- [ ] Health endpoint returns healthy status
- [ ] Co-Pilot can connect as custom model
- [ ] Agent responds to basic conversation
- [ ] CORS works correctly with Obsidian
- [ ] All tests pass

**Exit Criteria:** Co-Pilot can have a basic conversation with the agent (no vault operations yet).

---

### Phase 2: Tool 1 - `note_operations`

**Goal:** Implement the first tool enabling note CRUD operations and summarization.

**Deliverables:**
- ✅ `note_operations` tool with all operations:
  - `create` - Create new notes with frontmatter
  - `read` - Read note content
  - `update` - Update/append to notes
  - `delete` - Delete notes (with confirmation)
  - `summarize` - Generate note summaries
- ✅ Frontmatter parsing and generation (YAML)
- ✅ Folder creation if path doesn't exist
- ✅ Unit tests for all operations
- ✅ Integration tests with real vault

**Validation Criteria:**
- [ ] Create note via chat → note appears in vault with correct frontmatter
- [ ] Read note via chat → full content returned
- [ ] Update note via chat → changes persist correctly
- [ ] Delete note via chat → note removed (after confirmation)
- [ ] Summarize note via chat → accurate summary returned
- [ ] Edge cases handled (missing files, invalid paths, etc.)
- [ ] All tests pass

**Exit Criteria:** User can perform all note CRUD operations through Co-Pilot chat.

---

### Phase 3: Tool 2 - `vault_search`

**Goal:** Implement unified search capabilities across the vault.

**Deliverables:**
- ✅ `vault_search` tool with all operations:
  - `fulltext` - Search note contents
  - `by_tag` - Find notes by tags
  - `by_link` - Find notes by wikilinks (backlinks/forward links)
  - `by_date` - Find notes by date range
  - `combined` - Multi-criteria search
- ✅ Result ranking by relevance
- ✅ Excerpt generation for context
- ✅ Pagination support
- ✅ Unit and integration tests

**Validation Criteria:**
- [ ] Full-text search returns relevant results
- [ ] Tag search with AND/OR logic works
- [ ] Backlink discovery finds all linking notes
- [ ] Date range filtering is accurate
- [ ] Combined search applies all filters correctly
- [ ] Results include helpful excerpts
- [ ] All tests pass

**Exit Criteria:** User can find any note using natural language search queries.

---

### Phase 4: Tool 3 - `tag_management`

**Goal:** Implement tag operations with AI-powered suggestions and note connections.

**Deliverables:**
- ✅ `tag_management` tool with all operations:
  - `add` - Add tags to notes
  - `remove` - Remove tags from notes
  - `rename` - Rename tags vault-wide
  - `list` - List all tags with usage stats
  - `suggest` - AI-powered tag suggestions
  - `auto_tag` - Automatically tag based on content
  - `connect` - Create wikilinks to related notes
- ✅ Frontmatter tag manipulation
- ✅ Related note discovery algorithm
- ✅ Unit and integration tests

**Validation Criteria:**
- [ ] Tags added to frontmatter correctly
- [ ] Tags removed without affecting other metadata
- [ ] Vault-wide rename updates all notes atomically
- [ ] Tag suggestions are contextually relevant
- [ ] Auto-tag applies appropriate tags
- [ ] Related note connections create valid wikilinks
- [ ] All tests pass

**Exit Criteria:** User can manage tags and create note connections through conversation.

---

### Phase 5: Tool 4 - `vault_analytics`

**Goal:** Implement analytics and insights generation.

**Deliverables:**
- ✅ `vault_analytics` tool with all operations:
  - `overview` - General vault statistics
  - `trends` - Activity trends over time
  - `tag_distribution` - Tag usage analysis
  - `activity` - Recent vault activity
  - `insights` - AI-generated narrative insights
- ✅ Dual output format (narrative + structured data)
- ✅ Accurate date-based calculations
- ✅ Unit and integration tests

**Validation Criteria:**
- [ ] Statistics match actual vault state
- [ ] Trends accurately reflect activity over time
- [ ] Tag distribution counts are correct
- [ ] Narrative insights are meaningful and accurate
- [ ] Structured data is properly formatted
- [ ] All tests pass

**Exit Criteria:** User can ask "tell me my trends" and receive accurate, insightful analysis.

---

### Phase 6: Polish & Integration

**Goal:** Refine the overall experience, add streaming support, and ensure production readiness.

**Deliverables:**
- ✅ Streaming response support (SSE)
- ✅ Comprehensive error handling across all tools
- ✅ End-to-end testing suite
- ✅ Performance optimization
- ✅ Documentation and README
- ✅ Example conversations and use cases

**Validation Criteria:**
- [ ] Streaming works correctly with Co-Pilot
- [ ] All error cases return helpful messages
- [ ] E2E tests cover main user flows
- [ ] Response times meet quality indicators
- [ ] Documentation is complete and accurate
- [ ] All tests pass

**Exit Criteria:** Production-ready agent with all features working seamlessly.

---

### Implementation Order Summary

| Order | Phase | Focus | Depends On |
|-------|-------|-------|------------|
| 1 | Foundation | Server + Basic Agent | - |
| 2 | Tool 1 | `note_operations` | Foundation |
| 3 | Tool 2 | `vault_search` | Foundation |
| 4 | Tool 3 | `tag_management` | note_operations, vault_search |
| 5 | Tool 4 | `vault_analytics` | All previous tools |
| 6 | Polish | Streaming, Errors, Docs | All tools |

> **Note:** Each phase must pass all validation criteria before proceeding. Learnings from each phase inform the implementation of subsequent phases.

---

## 13. Future Considerations

### Post-MVP Enhancements

**Search & Discovery:**
- Embedding-based semantic search
- Fuzzy matching for typos
- Saved searches and filters

**Automation:**
- Scheduled daily notes creation
- Auto-tagging on note save (via file watcher)
- Template system with variables

**Visualization:**
- Graph view data export
- Tag hierarchy visualization
- Activity heatmaps

**Multi-Vault:**
- Support for multiple vaults
- Vault switching via chat
- Cross-vault search

### Integration Opportunities

- **Obsidian Local REST API:** Alternative vault access method
- **Dataview plugin:** Query-based note aggregation
- **Templater plugin:** Advanced template support
- **Calendar plugin:** Date-based note integration

### Advanced Features

- Voice input processing
- Image/PDF content extraction
- External knowledge base integration
- Collaborative features (shared vaults)

---

## 14. Risks & Mitigations

### Risk 1: OpenAI API Format Compliance

**Risk:** Co-Pilot may expect specific OpenAI response structures that we miss.

**Mitigation:**
- Study Co-Pilot source code for exact expectations
- Test with multiple message types and edge cases
- Implement comprehensive error responses matching OpenAI format
- Add request/response logging for debugging

---

### Risk 2: File System Race Conditions

**Risk:** Concurrent vault access (agent + Obsidian) may cause conflicts.

**Mitigation:**
- Use atomic file operations where possible
- Implement optimistic locking via frontmatter timestamps
- Handle file-not-found gracefully (user may have moved/deleted)
- Add retry logic for transient failures

---

### Risk 3: Large Vault Performance

**Risk:** Vaults with thousands of notes may have slow search/analytics.

**Mitigation:**
- Implement incremental indexing (cache file metadata)
- Use generators for large result sets
- Add pagination to search results
- Profile and optimize hot paths

---

### Risk 4: LLM Context Limits

**Risk:** Large notes or many search results may exceed Claude's context window.

**Mitigation:**
- Truncate note content with smart summarization
- Limit search results with relevance threshold
- Use tool output compression
- Implement conversation memory management

---

### Risk 5: Tag Suggestion Quality

**Risk:** AI tag suggestions may be irrelevant or inconsistent with user's taxonomy.

**Mitigation:**
- Learn from user's existing tag vocabulary
- Provide confidence scores with suggestions
- Allow user feedback loop ("not relevant")
- Default to suggesting existing tags over new ones

---

## 15. Appendix

### Related Documents

- [CLAUDE.md](/Users/hugemarley/Dev/Projects/obsidian-pivloop-agent/CLAUDE.md) - Project coding standards
- [Obsidian Co-Pilot Settings](https://www.obsidiancopilot.com/en/docs/settings) - Plugin configuration
- [PydanticAI Documentation](https://ai.pydantic.dev) - Agent framework docs
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference) - API format spec

### Key Dependencies

| Dependency | Documentation |
|------------|---------------|
| PydanticAI | https://ai.pydantic.dev |
| FastAPI | https://fastapi.tiangolo.com |
| Anthropic SDK | https://docs.anthropic.com |
| python-frontmatter | https://python-frontmatter.readthedocs.io |

### Vault Structure Reference

```
/Users/hugemarley/Documents/Obsidian Vault/
├── .obsidian/              # Plugin configs (ignore)
├── Coding System/          # Project documentation
│   ├── COMMANDS.md
│   └── WORKFLOW.md
├── copilot/                # Co-Pilot generated
│   ├── copilot-conversations/
│   └── copilot-custom-prompts/
├── Excalidraw/             # Drawings
└── Welcome.md              # Default note
```

### Frontmatter Schema

```yaml
---
created: 2025-11-25T10:30:00
modified: 2025-11-25T14:45:00
tags:
  - project
  - meeting
  - action-items
aliases:
  - "Weekly Sync"
related:
  - "[[Project Overview]]"
  - "[[Team Members]]"
---
```

---

---

*Document generated: 2025-11-25*
*Updated: 2025-11-25 - Revised to iterative, tool-by-tool implementation approach*
*Next review: After Phase 1 completion*
