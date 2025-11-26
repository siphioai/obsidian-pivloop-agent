# Feature: note_operations Tool (Phase 2)

> **Status**: Ready for Implementation
> **Created**: 2025-11-25
> **Complexity**: Medium
> **Estimated Tasks**: 12
> **Confidence Score**: 8/10

The following plan should be complete, but it's important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils, types, and models. Import from the right files etc.

---

## Table of Contents

1. [Feature Description](#feature-description)
2. [User Story](#user-story)
3. [Problem Statement](#problem-statement)
4. [Solution Statement](#solution-statement)
5. [Feature Metadata](#feature-metadata)
6. [Architecture Overview](#architecture-overview)
7. [Context References](#context-references)
8. [Implementation Plan](#implementation-plan)
9. [Step-by-Step Tasks](#step-by-step-tasks)
10. [Testing Strategy](#testing-strategy)
11. [Validation Commands](#validation-commands)
12. [Acceptance Criteria](#acceptance-criteria)
13. [Completion Checklist](#completion-checklist)
14. [Notes & Design Decisions](#notes--design-decisions)

---

## Feature Description

Implement the `note_operations` tool as the first agent tool for the Obsidian PivLoop Agent. This is a **consolidated multi-operation tool** that enables note CRUD operations (create, read, update, delete) and summarization through the chat interface.

### Why a Consolidated Tool?

The PRD explicitly chose a "one tool, multiple operations" pattern over individual tools like `create_note()`, `read_note()`, etc. for several reasons:

1. **Reduces context window usage** - Fewer tool definitions means less tokens spent on tool schemas
2. **Simplifies LLM decision-making** - The agent has one tool to consider, not five
3. **Allows complex operations** - The `operation` parameter provides flexibility
4. **Matches Obsidian's mental model** - Users think in terms of "note operations" not individual verbs

### Operations Supported

| Operation | Description | Required Parameters |
|-----------|-------------|---------------------|
| `create` | Create a new note with YAML frontmatter | `path`, `content` (optional) |
| `read` | Read note content and metadata | `path` |
| `update` | Update/replace note content | `path`, `content` |
| `delete` | Delete a note (with confirmation) | `path`, `confirm=True` |
| `summarize` | Get note content for LLM summarization | `path` |

---

## User Story

```
As an Obsidian user
I want to create, read, update, delete, and summarize notes through natural language conversation
So that I can manage my vault without leaving the Co-Pilot chat interface
```

### Example Conversations

**Creating a note:**
```
User: "Create a note called 'Meeting Notes' in the Projects folder about today's standup"
Agent: [calls note_operations(operation="create", path="Projects/Meeting Notes", content="...")]
Agent: "I've created the note at Projects/Meeting Notes.md with today's date and your standup summary."
```

**Reading a note:**
```
User: "What's in my API Design note?"
Agent: [calls note_operations(operation="read", path="API Design")]
Agent: "Here's the content of API Design.md: ..."
```

**Deleting with confirmation:**
```
User: "Delete the old draft note"
Agent: [calls note_operations(operation="delete", path="old-draft", confirm=False)]
Agent: "I found 'old-draft.md'. Are you sure you want to delete it? This cannot be undone."
User: "Yes, delete it"
Agent: [calls note_operations(operation="delete", path="old-draft", confirm=True)]
Agent: "Done. I've deleted old-draft.md from your vault."
```

---

## Problem Statement

Currently the PivLoop agent can only have basic conversations. Users cannot perform any vault operations through the chat interface. The agent has no tools registered, making it essentially a chatbot without Obsidian integration.

This is the **foundational tool** that enables all note management capabilities. Without it, the agent cannot fulfill its core purpose of being an Obsidian vault assistant.

---

## Solution Statement

Create a `note_operations` tool in a new `app/notes/` vertical slice that:

1. **Handles 5 operations**: create, read, update, delete, summarize
2. **Auto-generates YAML frontmatter** with timestamps on create
3. **Preserves existing frontmatter** on updates (only modifies `modified` timestamp)
4. **Implements two-stage delete confirmation** for safety
5. **Normalizes paths** (auto-append `.md`, create parent directories, strip slashes)
6. **Returns structured responses** optimized for LLM comprehension
7. **Logs all operations** with trace_id for debugging

The tool follows **vertical slice architecture** with its own dedicated folder while being registered with the existing `chat_agent` for use through the `/v1/chat/completions` endpoint.

---

## Feature Metadata

| Attribute | Value |
|-----------|-------|
| **Feature Type** | New Capability |
| **Estimated Complexity** | Medium |
| **Primary Systems Affected** | `app/notes/` (new), `app/chat/agent.py`, `app/dependencies.py` |
| **New Dependencies** | `python-frontmatter>=1.1.0` |
| **New Files** | 4 (`__init__.py`, `models.py`, `tools.py`, `test_notes.py`) |
| **Modified Files** | 3 (`pyproject.toml`, `dependencies.py`, `chat/agent.py`) |

---

## Architecture Overview

### Vertical Slice Structure

This project organizes code by **features** (vertical slices), not layers. Each slice is self-contained:

```
app/
├── chat/                    # Existing - OpenAI endpoint (MODIFIED)
│   ├── __init__.py
│   ├── agent.py             # ← Register note_operations here
│   ├── models.py
│   └── router.py
│
├── notes/                   # NEW - Note operations slice
│   ├── __init__.py          # Module exports
│   ├── models.py            # Pydantic models for frontmatter
│   └── tools.py             # note_operations tool implementation
│
├── dependencies.py          # ← Add delete_file, file_exists methods
├── config.py
├── main.py
└── tests/
    ├── test_chat.py
    └── test_notes.py        # NEW - Tests for note operations
```

### Data Flow

```
User Message (Co-Pilot)
         │
         ▼
┌─────────────────────────────────────────┐
│     /v1/chat/completions endpoint       │
│              (router.py)                │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│           chat_agent.run()              │
│         (PydanticAI Agent)              │
│                                         │
│   tools=[note_operations]  ◄────────────┼── Tool registered here
└────────────────┬────────────────────────┘
                 │
                 │ LLM decides to call tool
                 ▼
┌─────────────────────────────────────────┐
│        note_operations()                │
│          (notes/tools.py)               │
│                                         │
│   ctx.deps.vault.read_file()            │
│   ctx.deps.vault.write_file()           │
│   ctx.deps.vault.delete_file()          │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│            VaultClient                  │
│         (dependencies.py)               │
│                                         │
│   File system operations                │
│   Path validation & security            │
└─────────────────────────────────────────┘
                 │
                 ▼
         Obsidian Vault (filesystem)
```

### Why Register with chat_agent?

The `note_operations` tool lives in `app/notes/tools.py` but is **registered** with `chat_agent` in `app/chat/agent.py`. This is because:

1. **Single entry point**: Co-Pilot only talks to `/v1/chat/completions`
2. **Separation of concerns**: Tool implementation is in its own slice
3. **Reusability**: The tool function could be registered with other agents later
4. **Testability**: Tool can be tested independently of the chat agent

---

## Context References

### Relevant Codebase Files - YOU MUST READ THESE FILES BEFORE IMPLEMENTING!

| File | Lines | Why Read This |
|------|-------|---------------|
| `app/chat/agent.py` | 1-28 | Contains `ChatDependencies` dataclass and agent definition; tool will be registered here |
| `app/chat/router.py` | 1-75 | Shows how dependencies flow from router to agent via `ChatDependencies` |
| `app/dependencies.py` | 44-135 | `VaultClient` implementation; needs `delete_file` and `file_exists` methods added |
| `app/config.py` | 1-35 | Settings pattern for `vault_path` access |
| `app/tests/conftest.py` | 1-50 | Test fixture patterns for mock vault using `tmp_path` |
| `.AGENTS/reference/pydantic_ai_tools_guide.md` | Full file | PydanticAI tool patterns and best practices |

### New Files to Create

| File | Purpose |
|------|---------|
| `app/notes/__init__.py` | Module exports for `note_operations` tool |
| `app/notes/models.py` | Pydantic models for `NoteFrontmatter` and `NoteContent` |
| `app/notes/tools.py` | Main `note_operations` tool implementation with helper functions |
| `app/tests/test_notes.py` | Comprehensive unit tests (15+ test cases) |

### Relevant Documentation - YOU SHOULD READ THESE BEFORE IMPLEMENTING!

| Documentation | Section | Why |
|---------------|---------|-----|
| [python-frontmatter docs](https://python-frontmatter.readthedocs.io/en/latest/) | Basic usage, Post objects | Required for YAML frontmatter parsing and generation |
| [PydanticAI Tools Documentation](https://ai.pydantic.dev/tools/) | Registering via Decorator, Tool Schema | Correct tool registration and docstring patterns |

### Patterns to Follow

**Naming Conventions (from CLAUDE.md):**
```python
# Functions: snake_case with verb prefix
async def note_operations(...) -> str:
async def _create_note(...) -> str:  # Private helper with underscore

# Classes: PascalCase
class NoteFrontmatter(BaseModel):
class NoteContent(BaseModel):

# Constants: SCREAMING_SNAKE_CASE
DEFAULT_TAGS: list[str] = []
```

**Tool Function Signature Pattern (from pydantic_ai_tools_guide.md):**
```python
# CRITICAL: First parameter MUST be ctx: RunContext[DepsType]
# Docstrings become the tool schema - they're sent to the LLM!

async def note_operations(
    ctx: RunContext[ChatDependencies],  # ALWAYS first
    operation: str,                      # Required params next
    path: str,
    content: str | None = None,          # Optional params last
    confirm: bool = False,
) -> str:                                # Return type annotation required
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
```

**Error Handling Pattern (from dependencies.py):**
```python
# Return user-friendly strings, not exceptions
# The LLM needs to interpret errors and relay them to users

try:
    result = await ctx.deps.vault.read_file(path)
    return result
except VaultNotFoundError:
    return f"Note not found at {path}. Would you like me to create it?"
except VaultSecurityError:
    return "Invalid path: cannot access files outside the vault."
except Exception as e:
    logger.error("operation_failed", extra={
        "error": str(e),
        "trace_id": ctx.deps.trace_id
    })
    return f"Error: {str(e)}"
```

**Structured Logging Pattern (from dependencies.py):**
```python
# Always include trace_id for request correlation
# Use descriptive event names

logger.info("note_operation_called", extra={
    "operation": operation,
    "path": path,
    "trace_id": ctx.deps.trace_id,
})

logger.info("note_created", extra={
    "path": path,
    "trace_id": ctx.deps.trace_id,
})

logger.error("note_operation_failed", extra={
    "operation": operation,
    "path": path,
    "error": str(e),
    "trace_id": ctx.deps.trace_id,
}, exc_info=True)  # Include stack trace for errors
```

---

## Implementation Plan

### Phase 1: Foundation (Tasks 1-3)

**Goal**: Add new dependency and extend VaultClient with required methods.

**What we're doing**:
- Adding `python-frontmatter` library for YAML parsing
- Extending `VaultClient` with `delete_file()` and `file_exists()` methods
- These are prerequisites for the tool implementation

**Why this order**: The tool depends on these methods existing. We validate each step before proceeding.

### Phase 2: Core Implementation (Tasks 4-6)

**Goal**: Create the notes vertical slice with complete tool implementation.

**What we're doing**:
- Creating the `app/notes/` directory structure
- Implementing Pydantic models for frontmatter handling
- Implementing the `note_operations` tool with all 5 operations
- Implementing helper functions for path normalization and content parsing

**Why this order**: Models before tools (tools depend on models). Helper functions are defined before the main tool function that uses them.

### Phase 3: Integration (Task 7)

**Goal**: Register tool with chat_agent and update system prompt.

**What we're doing**:
- Importing `note_operations` into `chat/agent.py`
- Adding it to the `tools=[]` parameter of the Agent constructor
- Updating the system prompt to describe tool capabilities

**Why this order**: Integration comes after implementation is complete and tested in isolation.

### Phase 4: Testing & Validation (Tasks 8-12)

**Goal**: Comprehensive testing and validation of all operations.

**What we're doing**:
- Creating test fixtures for notes with and without frontmatter
- Writing unit tests for each operation and helper function
- Running linting and formatting checks
- Manual integration test via curl

**Why this order**: Tests validate the implementation. Manual testing confirms end-to-end functionality.

---

## Step-by-Step Tasks

> **IMPORTANT**: Execute every task in order, top to bottom. Each task is atomic and independently testable. Do not skip validation steps.

### Task 1: UPDATE pyproject.toml - Add python-frontmatter dependency

**What**: Add `python-frontmatter>=1.1.0` to the project dependencies.

**Why**: This library provides clean YAML frontmatter parsing and generation. It handles edge cases like notes without frontmatter gracefully.

**Implementation**:
```toml
# In pyproject.toml, add to dependencies list (around line 14):
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "pydantic-ai>=0.1.0",
    "anthropic>=0.40.0",
    "httpx>=0.27.0",
    "python-dotenv>=1.0.0",
    "python-frontmatter>=1.1.0",  # ← ADD THIS LINE
]
```

**Validation**:
```bash
uv sync && uv pip list | grep frontmatter
# Expected: python-frontmatter    1.1.0 (or higher)
```

---

### Task 2: UPDATE app/dependencies.py - Add delete_file method

**What**: Add `async def delete_file(self, path: str) -> None` method to `VaultClient`.

**Why**: The delete operation needs to remove files from the vault. This follows the same security pattern as `read_file` and `write_file`.

**Where**: Add after `write_file` method (around line 112).

**Implementation**:
```python
async def delete_file(self, path: str) -> None:
    """Delete a file from the vault.

    Args:
        path: Relative path to file

    Raises:
        VaultNotFoundError: If file does not exist
        VaultSecurityError: If path traversal detected
    """
    full_path = self._validate_path(path)
    if not full_path.exists():
        raise VaultNotFoundError(f"File not found: {path}")
    full_path.unlink()
```

**Validation**:
```bash
uv run pytest app/tests/test_chat.py -v
# Expected: All existing tests still pass
```

---

### Task 3: UPDATE app/dependencies.py - Add file_exists method

**What**: Add `async def file_exists(self, path: str) -> bool` method to `VaultClient`.

**Why**: Many operations need to check if a file exists before proceeding (create checks it doesn't exist, update/delete check it does). Returning bool instead of raising exception provides cleaner control flow.

**Where**: Add after `delete_file` method.

**Implementation**:
```python
async def file_exists(self, path: str) -> bool:
    """Check if a file exists in the vault.

    Args:
        path: Relative path to file

    Returns:
        True if file exists, False otherwise
    """
    try:
        full_path = self._validate_path(path)
        return full_path.exists()
    except VaultSecurityError:
        return False
```

**Validation**:
```bash
uv run pytest app/tests/test_chat.py -v
# Expected: All existing tests still pass
```

---

### Task 4: CREATE app/notes/__init__.py

**What**: Create the module init file that exports the `note_operations` tool.

**Why**: This makes the tool importable from `app.notes` and follows the existing pattern in `app/chat/__init__.py`.

**Implementation**:
```python
"""Notes feature module - note operations tool."""

from app.notes.tools import note_operations

__all__ = ["note_operations"]
```

**Validation**:
```bash
uv run python -c "from app.notes import note_operations; print('Import OK')"
# Note: This will fail until tools.py exists (Task 6)
```

---

### Task 5: CREATE app/notes/models.py

**What**: Create Pydantic models for frontmatter handling and parsed note content.

**Why**: Type-safe models ensure consistent frontmatter structure and make the code self-documenting. The `to_yaml_dict()` method handles datetime serialization for YAML.

**Implementation**:
```python
"""Pydantic models for note operations."""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class NoteFrontmatter(BaseModel):
    """YAML frontmatter metadata for notes.

    This model represents the structured metadata stored in the YAML
    frontmatter section of Obsidian notes. It automatically generates
    timestamps on creation.

    Attributes:
        created: When the note was originally created (UTC)
        modified: When the note was last modified (UTC)
        tags: List of tags for the note (without # prefix)

    Example frontmatter:
        ---
        created: 2025-11-25T10:30:00+00:00
        modified: 2025-11-25T14:45:00+00:00
        tags:
          - project
          - meeting
        ---
    """

    created: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the note was created"
    )
    modified: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the note was last modified"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="List of tags for the note"
    )

    def to_yaml_dict(self) -> dict[str, Any]:
        """Convert to dictionary suitable for YAML frontmatter.

        Returns:
            Dictionary with ISO-formatted datetime strings
        """
        return {
            "created": self.created.isoformat(),
            "modified": self.modified.isoformat(),
            "tags": self.tags,
        }


class NoteContent(BaseModel):
    """Parsed note with frontmatter and body separated.

    This model represents a fully parsed note, separating the YAML
    frontmatter metadata from the markdown body content.

    Attributes:
        frontmatter: Parsed frontmatter metadata (None if no frontmatter)
        body: The markdown content after the frontmatter
        raw: The original unparsed content
    """

    frontmatter: NoteFrontmatter | None = None
    body: str = ""
    raw: str = ""
```

**Validation**:
```bash
uv run python -c "
from app.notes.models import NoteFrontmatter, NoteContent
fm = NoteFrontmatter(tags=['test'])
print('Frontmatter:', fm.to_yaml_dict())
nc = NoteContent(body='Hello')
print('NoteContent:', nc.model_dump())
"
# Expected: Both print without errors
```

---

### Task 6: CREATE app/notes/tools.py

**What**: Create the main `note_operations` tool with all 5 operations and helper functions.

**Why**: This is the core implementation. The tool is a standalone async function that will be registered with the chat agent.

**Key Implementation Details**:

1. **`normalize_path()`**: Ensures consistent path handling
2. **`parse_note()`**: Parses content into frontmatter + body
3. **`create_note_content()`**: Generates new note with frontmatter
4. **`update_note_content()`**: Updates content while preserving frontmatter
5. **`note_operations()`**: Main tool function that dispatches to operation handlers
6. **`_create_note()`, `_read_note()`, etc.**: Private handlers for each operation

**Implementation**:
```python
"""Note operations tool for the chat agent.

This module implements the note_operations tool which provides CRUD
operations for Obsidian vault notes through the PydanticAI agent.

The tool follows a consolidated multi-operation pattern where a single
tool handles multiple operations via an `operation` parameter. This
reduces context window usage and simplifies LLM decision-making.

Example usage by the agent:
    note_operations(operation="create", path="Projects/API", content="# API Design")
    note_operations(operation="read", path="Projects/API")
    note_operations(operation="delete", path="Projects/API", confirm=True)
"""

from datetime import datetime, timezone

import frontmatter
from pydantic_ai import RunContext

from app.chat.agent import ChatDependencies
from app.dependencies import VaultNotFoundError, VaultSecurityError, logger
from app.notes.models import NoteContent, NoteFrontmatter


def normalize_path(path: str) -> str:
    """Normalize a note path for consistent handling.

    This function ensures all paths are in a consistent format:
    - Strips leading/trailing whitespace
    - Strips leading/trailing slashes
    - Appends .md extension if not present

    Args:
        path: Raw path input from user or LLM

    Returns:
        Normalized path with .md extension

    Examples:
        >>> normalize_path("test")
        'test.md'
        >>> normalize_path("/Projects/API Design/")
        'Projects/API Design.md'
        >>> normalize_path("note.md")
        'note.md'
    """
    path = path.strip().strip("/")
    if not path.endswith(".md"):
        path = f"{path}.md"
    return path


def parse_note(content: str) -> NoteContent:
    """Parse note content into frontmatter and body.

    Uses python-frontmatter to separate YAML frontmatter from
    markdown body. Gracefully handles notes without frontmatter.

    Args:
        content: Raw note content (may or may not have frontmatter)

    Returns:
        NoteContent with parsed frontmatter (if present) and body
    """
    try:
        post = frontmatter.loads(content)
        fm_data = dict(post.metadata) if post.metadata else None

        if fm_data:
            # Parse existing frontmatter into our model
            fm = NoteFrontmatter(
                created=fm_data.get("created", datetime.now(timezone.utc)),
                modified=fm_data.get("modified", datetime.now(timezone.utc)),
                tags=fm_data.get("tags", []),
            )
        else:
            fm = None

        return NoteContent(frontmatter=fm, body=post.content, raw=content)
    except Exception:
        # If parsing fails for any reason, treat as plain content
        return NoteContent(frontmatter=None, body=content, raw=content)


def create_note_content(body: str, tags: list[str] | None = None) -> str:
    """Create note content with YAML frontmatter.

    Generates a complete note with properly formatted YAML frontmatter
    including creation timestamp and optional tags.

    Args:
        body: Note body content (markdown)
        tags: Optional list of tags (without # prefix)

    Returns:
        Full note content with YAML frontmatter

    Example output:
        ---
        created: 2025-11-25T10:30:00+00:00
        modified: 2025-11-25T10:30:00+00:00
        tags:
          - project
        ---
        # Note Title

        Content here...
    """
    fm = NoteFrontmatter(tags=tags or [])
    post = frontmatter.Post(body)
    post.metadata = fm.to_yaml_dict()
    return frontmatter.dumps(post)


def update_note_content(existing: str, new_body: str | None, append: bool = False) -> str:
    """Update note content while preserving frontmatter.

    This function handles the complexity of updating notes while
    maintaining their metadata. It:
    - Preserves existing frontmatter structure
    - Updates the 'modified' timestamp
    - Either replaces or appends to the body content

    Args:
        existing: Existing note content
        new_body: New body content (or content to append)
        append: If True, append to existing body instead of replacing

    Returns:
        Updated note content with preserved/updated frontmatter
    """
    parsed = parse_note(existing)

    if parsed.frontmatter:
        # Update modified timestamp, preserve everything else
        fm_dict = parsed.frontmatter.to_yaml_dict()
        fm_dict["modified"] = datetime.now(timezone.utc).isoformat()
    else:
        # No existing frontmatter - don't add one (respect user's format)
        if append and new_body:
            return existing + "\n" + new_body
        return new_body or existing

    # Build updated content with frontmatter
    if append and new_body:
        body = parsed.body + "\n" + new_body
    else:
        body = new_body if new_body is not None else parsed.body

    post = frontmatter.Post(body)
    post.metadata = fm_dict
    return frontmatter.dumps(post)


# =============================================================================
# Main Tool Function
# =============================================================================

async def note_operations(
    ctx: RunContext[ChatDependencies],
    operation: str,
    path: str,
    content: str | None = None,
    confirm: bool = False,
) -> str:
    """Perform operations on notes in the Obsidian vault.

    This is the main tool function that the LLM calls to interact with
    notes. It supports five operations: create, read, update, delete,
    and summarize.

    Args:
        ctx: Context with vault access and trace_id for logging
        operation: One of 'create', 'read', 'update', 'delete', 'summarize'
        path: Relative path to the note (e.g., 'Projects/API Design.md')
        content: Content for create/update operations (optional for others)
        confirm: Must be True for delete operations (safety measure)

    Returns:
        Operation result message or note content. Returns user-friendly
        error messages instead of raising exceptions.

    Examples:
        Create: note_operations(ctx, "create", "Projects/API", "# API Design")
        Read:   note_operations(ctx, "read", "Projects/API")
        Update: note_operations(ctx, "update", "Projects/API", "# Updated")
        Delete: note_operations(ctx, "delete", "Projects/API", confirm=True)
        Summarize: note_operations(ctx, "summarize", "Projects/API")
    """
    # Normalize path for consistent handling
    path = normalize_path(path)

    # Log the operation for debugging and audit trail
    logger.info(
        "note_operation_called",
        extra={
            "operation": operation,
            "path": path,
            "has_content": content is not None,
            "confirm": confirm,
            "trace_id": ctx.deps.trace_id,
        },
    )

    try:
        # Dispatch to appropriate handler
        if operation == "create":
            return await _create_note(ctx, path, content)
        elif operation == "read":
            return await _read_note(ctx, path)
        elif operation == "update":
            return await _update_note(ctx, path, content)
        elif operation == "delete":
            return await _delete_note(ctx, path, confirm)
        elif operation == "summarize":
            return await _summarize_note(ctx, path)
        else:
            return (
                f"Unknown operation: {operation}. "
                f"Valid operations: create, read, update, delete, summarize"
            )

    except VaultSecurityError:
        return "Invalid path: cannot access files outside the vault."
    except VaultNotFoundError:
        return f"Note not found at {path}. Would you like me to create it?"
    except Exception as e:
        logger.error(
            "note_operation_failed",
            extra={
                "operation": operation,
                "path": path,
                "error": str(e),
                "trace_id": ctx.deps.trace_id,
            },
            exc_info=True,
        )
        return f"Error performing {operation}: {str(e)}"


# =============================================================================
# Operation Handlers (Private)
# =============================================================================

async def _create_note(
    ctx: RunContext[ChatDependencies],
    path: str,
    content: str | None
) -> str:
    """Create a new note with frontmatter.

    Creates a new note at the specified path with auto-generated YAML
    frontmatter including creation timestamp. Parent directories are
    created automatically by VaultClient.write_file().
    """
    # Check if note already exists
    if await ctx.deps.vault.file_exists(path):
        return f"Note already exists at {path}. Use 'update' to modify it."

    # Create note with frontmatter (empty content is allowed)
    note_content = create_note_content(content or "")
    await ctx.deps.vault.write_file(path, note_content)

    logger.info("note_created", extra={"path": path, "trace_id": ctx.deps.trace_id})
    return f"Created note at {path}"


async def _read_note(ctx: RunContext[ChatDependencies], path: str) -> str:
    """Read a note and return its content with metadata.

    Returns the note content along with parsed metadata (if frontmatter
    exists) in a format that's easy for the LLM to understand and relay
    to the user.
    """
    content = await ctx.deps.vault.read_file(path)
    parsed = parse_note(content)

    # Build response with metadata if available
    response_parts = []

    if parsed.frontmatter:
        response_parts.append(f"**Note: {path}**")
        response_parts.append(f"Created: {parsed.frontmatter.created.isoformat()}")
        response_parts.append(f"Modified: {parsed.frontmatter.modified.isoformat()}")
        if parsed.frontmatter.tags:
            response_parts.append(f"Tags: {', '.join(parsed.frontmatter.tags)}")
        response_parts.append("")  # Blank line before content
        response_parts.append(parsed.body)
    else:
        response_parts.append(f"**Note: {path}** (no frontmatter)")
        response_parts.append("")
        response_parts.append(content)

    return "\n".join(response_parts)


async def _update_note(
    ctx: RunContext[ChatDependencies],
    path: str,
    content: str | None
) -> str:
    """Update an existing note while preserving frontmatter.

    Replaces the note body with new content while preserving the
    existing frontmatter structure and updating the modified timestamp.
    """
    # Check if note exists
    if not await ctx.deps.vault.file_exists(path):
        return f"Note not found at {path}. Would you like me to create it instead?"

    # Read existing content
    existing = await ctx.deps.vault.read_file(path)

    # Update content (preserves frontmatter, updates modified timestamp)
    updated = update_note_content(existing, content, append=False)
    await ctx.deps.vault.write_file(path, updated)

    logger.info("note_updated", extra={"path": path, "trace_id": ctx.deps.trace_id})
    return f"Updated note at {path}"


async def _delete_note(
    ctx: RunContext[ChatDependencies],
    path: str,
    confirm: bool
) -> str:
    """Delete a note with two-stage confirmation.

    This implements a safety mechanism where deletion requires explicit
    confirmation. The first call (confirm=False) returns a warning,
    and only a second call with confirm=True actually deletes the file.
    """
    # Check if note exists
    if not await ctx.deps.vault.file_exists(path):
        return f"Note not found at {path}. Nothing to delete."

    # Require explicit confirmation
    if not confirm:
        return (
            f"To delete '{path}', please confirm. "
            f"This action cannot be undone. "
            f"Call again with confirm=True to proceed."
        )

    # Delete the file
    await ctx.deps.vault.delete_file(path)

    logger.info("note_deleted", extra={"path": path, "trace_id": ctx.deps.trace_id})
    return f"Deleted note at {path}"


async def _summarize_note(ctx: RunContext[ChatDependencies], path: str) -> str:
    """Return note content for summarization by the LLM.

    This operation reads the note and returns its body content with a
    prompt hint for the LLM to summarize it. The actual summarization
    is performed by the LLM, not this function.
    """
    content = await ctx.deps.vault.read_file(path)
    parsed = parse_note(content)

    return f"Content to summarize from {path}:\n\n{parsed.body}"
```

**Validation**:
```bash
uv run python -c "
from app.notes.tools import note_operations, normalize_path, parse_note
print('normalize_path:', normalize_path('test'))
print('parse_note:', parse_note('# Hello').body)
print('note_operations doc:', note_operations.__doc__[:100])
"
# Expected: All print without errors
```

---

### Task 7: UPDATE app/chat/agent.py - Register tool and update system prompt

**What**: Import `note_operations` and register it with `chat_agent` via the `tools=[]` parameter. Update system prompt to describe capabilities.

**Why**: The tool must be registered with the agent for the LLM to be able to call it. The system prompt guides the LLM on how to use the tool.

**Implementation**:
```python
"""PydanticAI agent definition for chat feature."""

from dataclasses import dataclass

from pydantic_ai import Agent

from app.dependencies import VaultClient
from app.notes.tools import note_operations


@dataclass
class ChatDependencies:
    """Dependencies injected into tools via RunContext."""

    vault: VaultClient
    trace_id: str


SYSTEM_PROMPT = """You are a helpful AI assistant integrated with Obsidian.
You help users manage their knowledge base through natural conversation.

You have access to the note_operations tool which allows you to:
- create: Create new notes with automatic timestamps and frontmatter
- read: Read note content and metadata (creation date, tags, etc.)
- update: Update existing notes (preserves frontmatter, updates modified time)
- delete: Delete notes (requires confirmation for safety)
- summarize: Get note content for summarization

Guidelines:
- When creating notes, use descriptive filenames and organize in appropriate folders
- Paths are relative to the vault root (e.g., 'Projects/API Design.md')
- The .md extension is added automatically if not provided
- When deleting notes, always inform the user what will be deleted first
- Wait for explicit user confirmation before deleting

Example interactions:
- "Create a note about X" → use create operation
- "What's in my note about Y?" → use read operation
- "Update my Z note with..." → use update operation
- "Delete the old draft" → use delete (will ask for confirmation)
- "Summarize my meeting notes" → use summarize operation"""

chat_agent = Agent(
    "anthropic:claude-haiku-4-5",
    deps_type=ChatDependencies,
    tools=[note_operations],
    retries=2,
    system_prompt=SYSTEM_PROMPT,
)
```

**Validation**:
```bash
uv run python -c "
from app.chat.agent import chat_agent
tools = list(chat_agent._function_tools.keys())
print('Registered tools:', tools)
"
# Expected: Registered tools: ['note_operations']
```

---

### Task 8: CREATE app/tests/test_notes.py

**What**: Create comprehensive unit tests for all note operations and helper functions.

**Why**: Tests ensure correctness and prevent regressions. They also serve as documentation for expected behavior.

**Test Coverage**:
- `normalize_path()` - 4 test cases
- `parse_note()` - 2 test cases
- `create_note_content()` - 1 test case
- `NoteFrontmatter` model - 2 test cases
- `note_operations` - 11 test cases (all operations + error cases)

**Implementation**:
```python
"""Tests for notes feature.

This module contains comprehensive tests for the note_operations tool
and its helper functions. Tests use pytest fixtures with temporary
directories to ensure isolation.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.chat.agent import ChatDependencies
from app.dependencies import VaultClient
from app.notes.models import NoteContent, NoteFrontmatter
from app.notes.tools import (
    create_note_content,
    normalize_path,
    note_operations,
    parse_note,
    update_note_content,
)


# =============================================================================
# Path Normalization Tests
# =============================================================================

class TestNormalizePath:
    """Tests for path normalization helper function."""

    def test_adds_md_extension(self) -> None:
        """Test that .md extension is added when missing."""
        assert normalize_path("test") == "test.md"
        assert normalize_path("folder/test") == "folder/test.md"

    def test_preserves_md_extension(self) -> None:
        """Test that existing .md extension is not duplicated."""
        assert normalize_path("test.md") == "test.md"
        assert normalize_path("folder/test.md") == "folder/test.md"

    def test_strips_whitespace(self) -> None:
        """Test that leading/trailing whitespace is stripped."""
        assert normalize_path("  test  ") == "test.md"
        assert normalize_path("\ttest\n") == "test.md"

    def test_strips_leading_slash(self) -> None:
        """Test that leading/trailing slashes are stripped."""
        assert normalize_path("/folder/test") == "folder/test.md"
        assert normalize_path("folder/test/") == "folder/test.md"
        assert normalize_path("/folder/test/") == "folder/test.md"


# =============================================================================
# Note Parsing Tests
# =============================================================================

class TestParsing:
    """Tests for note content parsing functions."""

    def test_parse_note_with_frontmatter(self) -> None:
        """Test parsing note with valid YAML frontmatter."""
        content = """---
created: 2025-01-01T00:00:00+00:00
modified: 2025-01-02T00:00:00+00:00
tags:
  - test
  - example
---
# Test Note

Body content here."""

        result = parse_note(content)

        assert result.frontmatter is not None
        assert result.frontmatter.tags == ["test", "example"]
        assert "Body content here" in result.body
        assert "# Test Note" in result.body

    def test_parse_note_without_frontmatter(self) -> None:
        """Test parsing note without any frontmatter."""
        content = "# Just a heading\n\nNo frontmatter here."

        result = parse_note(content)

        assert result.frontmatter is None
        assert result.body == content
        assert result.raw == content

    def test_create_note_content_with_tags(self) -> None:
        """Test creating note content with frontmatter and tags."""
        body = "# Test\n\nContent"
        result = create_note_content(body, tags=["test", "example"])

        assert "---" in result
        assert "created:" in result
        assert "modified:" in result
        assert "tags:" in result
        assert "- test" in result
        assert "- example" in result
        assert "# Test" in result

    def test_update_note_content_preserves_frontmatter(self) -> None:
        """Test that update preserves existing frontmatter structure."""
        existing = """---
created: 2025-01-01T00:00:00+00:00
modified: 2025-01-01T00:00:00+00:00
tags:
  - original
---
Original body"""

        result = update_note_content(existing, "New body")

        assert "created: 2025-01-01" in result  # Preserved
        assert "- original" in result  # Tags preserved
        assert "New body" in result


# =============================================================================
# Frontmatter Model Tests
# =============================================================================

class TestFrontmatterModel:
    """Tests for NoteFrontmatter Pydantic model."""

    def test_default_values(self) -> None:
        """Test that default timestamps are generated."""
        fm = NoteFrontmatter()

        assert fm.created is not None
        assert fm.modified is not None
        assert fm.tags == []
        # Should be recent (within last minute)
        assert (datetime.now(timezone.utc) - fm.created).seconds < 60

    def test_to_yaml_dict(self) -> None:
        """Test YAML dictionary conversion with ISO format."""
        fm = NoteFrontmatter(tags=["a", "b"])
        result = fm.to_yaml_dict()

        assert "created" in result
        assert "modified" in result
        assert result["tags"] == ["a", "b"]
        # Should be ISO format string
        assert "T" in result["created"]


# =============================================================================
# Note Operations Integration Tests
# =============================================================================

class TestNoteOperations:
    """Integration tests for note_operations tool function."""

    @pytest.fixture
    def mock_vault(self, tmp_path: Path) -> VaultClient:
        """Create a VaultClient with temporary directory."""
        return VaultClient(vault_path=tmp_path)

    @pytest.fixture
    def mock_ctx(self, mock_vault: VaultClient) -> Mock:
        """Create a mock RunContext with ChatDependencies."""
        ctx = Mock()
        ctx.deps = ChatDependencies(vault=mock_vault, trace_id="test-123")
        return ctx

    # -------------------------------------------------------------------------
    # Create Operation Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_create_note(self, mock_ctx: Mock, tmp_path: Path) -> None:
        """Test creating a new note successfully."""
        result = await note_operations(
            mock_ctx, operation="create", path="test", content="Hello world"
        )

        assert "Created note at test.md" in result
        assert (tmp_path / "test.md").exists()

        # Verify content has frontmatter
        content = (tmp_path / "test.md").read_text()
        assert "Hello world" in content
        assert "created:" in content
        assert "---" in content

    @pytest.mark.asyncio
    async def test_create_note_already_exists(
        self, mock_ctx: Mock, tmp_path: Path
    ) -> None:
        """Test that creating existing note returns error message."""
        (tmp_path / "existing.md").write_text("# Existing")

        result = await note_operations(
            mock_ctx, operation="create", path="existing", content="New content"
        )

        assert "already exists" in result
        # Original content unchanged
        assert (tmp_path / "existing.md").read_text() == "# Existing"

    @pytest.mark.asyncio
    async def test_create_note_with_nested_path(
        self, mock_ctx: Mock, tmp_path: Path
    ) -> None:
        """Test that parent directories are created automatically."""
        result = await note_operations(
            mock_ctx,
            operation="create",
            path="deep/nested/folder/note",
            content="Content",
        )

        assert "Created" in result
        assert (tmp_path / "deep" / "nested" / "folder" / "note.md").exists()

    # -------------------------------------------------------------------------
    # Read Operation Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_read_note_with_frontmatter(
        self, mock_ctx: Mock, tmp_path: Path
    ) -> None:
        """Test reading a note with frontmatter."""
        note_content = """---
created: 2025-01-01T00:00:00+00:00
modified: 2025-01-01T00:00:00+00:00
tags:
  - test
---
# Test Note

Content here."""
        (tmp_path / "test.md").write_text(note_content)

        result = await note_operations(mock_ctx, operation="read", path="test")

        assert "test.md" in result
        assert "Content here" in result
        assert "Tags: test" in result

    @pytest.mark.asyncio
    async def test_read_note_not_found(self, mock_ctx: Mock) -> None:
        """Test reading non-existent note returns helpful message."""
        result = await note_operations(mock_ctx, operation="read", path="nonexistent")

        assert "not found" in result.lower()
        assert "create it" in result.lower()

    # -------------------------------------------------------------------------
    # Update Operation Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_update_note(self, mock_ctx: Mock, tmp_path: Path) -> None:
        """Test updating an existing note."""
        (tmp_path / "test.md").write_text("# Original\n\nContent")

        result = await note_operations(
            mock_ctx, operation="update", path="test", content="# Updated\n\nNew content"
        )

        assert "Updated note at test.md" in result
        updated = (tmp_path / "test.md").read_text()
        assert "Updated" in updated
        assert "New content" in updated

    @pytest.mark.asyncio
    async def test_update_note_not_found(self, mock_ctx: Mock) -> None:
        """Test updating non-existent note returns helpful message."""
        result = await note_operations(
            mock_ctx, operation="update", path="nonexistent", content="content"
        )

        assert "not found" in result.lower()
        assert "create it" in result.lower()

    # -------------------------------------------------------------------------
    # Delete Operation Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_delete_without_confirm(
        self, mock_ctx: Mock, tmp_path: Path
    ) -> None:
        """Test that delete without confirm returns warning."""
        (tmp_path / "test.md").write_text("# Test")

        result = await note_operations(mock_ctx, operation="delete", path="test")

        assert "confirm" in result.lower()
        assert "cannot be undone" in result.lower()
        # File should NOT be deleted
        assert (tmp_path / "test.md").exists()

    @pytest.mark.asyncio
    async def test_delete_with_confirm(self, mock_ctx: Mock, tmp_path: Path) -> None:
        """Test that delete with confirm=True actually deletes."""
        (tmp_path / "test.md").write_text("# Test")

        result = await note_operations(
            mock_ctx, operation="delete", path="test", confirm=True
        )

        assert "Deleted" in result
        assert not (tmp_path / "test.md").exists()

    @pytest.mark.asyncio
    async def test_delete_not_found(self, mock_ctx: Mock) -> None:
        """Test deleting non-existent note."""
        result = await note_operations(
            mock_ctx, operation="delete", path="nonexistent", confirm=True
        )

        assert "not found" in result.lower()
        assert "nothing to delete" in result.lower()

    # -------------------------------------------------------------------------
    # Summarize Operation Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_summarize_note(self, mock_ctx: Mock, tmp_path: Path) -> None:
        """Test summarize operation returns content for LLM."""
        (tmp_path / "test.md").write_text("# Long Note\n\nThis is content to summarize.")

        result = await note_operations(mock_ctx, operation="summarize", path="test")

        assert "Content to summarize" in result
        assert "This is content to summarize" in result

    # -------------------------------------------------------------------------
    # Error Handling Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_invalid_operation(self, mock_ctx: Mock) -> None:
        """Test that invalid operation returns helpful error."""
        result = await note_operations(mock_ctx, operation="invalid", path="test")

        assert "Unknown operation" in result
        assert "invalid" in result
        assert "create, read, update, delete, summarize" in result
```

**Validation**:
```bash
uv run pytest app/tests/test_notes.py -v
# Expected: All tests pass (should be 18+ tests)
```

---

### Task 9: Run Full Test Suite

**What**: Verify all tests pass including existing chat tests.

**Validation**:
```bash
uv run pytest app/tests/ -v --tb=short
# Expected: All tests pass with no failures
```

---

### Task 10: Run Linting

**What**: Ensure code passes ruff linting.

**Validation**:
```bash
uv run ruff check app/
# Expected: No errors (exit code 0)
```

---

### Task 11: Run Formatting Check

**What**: Ensure code is properly formatted.

**Validation**:
```bash
uv run ruff format app/ --check
# Expected: No formatting issues (exit code 0)
# If issues found, run: uv run ruff format app/
```

---

### Task 12: Manual Integration Test

**What**: Test the complete flow via HTTP requests.

**Validation**:
```bash
# Start server (if not running)
uv run uvicorn app.main:app --reload --port 8000 &
sleep 3

# Test health endpoint
curl -s http://localhost:8000/health | head -1

# Test creating a note via chat
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Create a note called integration-test with content Hello from integration test"}
    ]
  }' | head -20

# Clean up (stop server if started in background)
pkill -f "uvicorn app.main:app" 2>/dev/null || true
```

---

## Testing Strategy

### Unit Tests (app/tests/test_notes.py)

| Category | Test Count | What's Tested |
|----------|------------|---------------|
| Path normalization | 4 | Extension adding, whitespace, slashes |
| Note parsing | 4 | With/without frontmatter, content generation |
| Frontmatter model | 2 | Defaults, YAML serialization |
| Create operation | 3 | Success, exists error, nested paths |
| Read operation | 2 | Success with metadata, not found |
| Update operation | 2 | Success, not found |
| Delete operation | 3 | Without confirm, with confirm, not found |
| Summarize operation | 1 | Returns content for LLM |
| Error handling | 1 | Invalid operation |
| **Total** | **22** | |

### Edge Cases Covered

- Empty content on create (allowed, creates note with empty body)
- Note without frontmatter (handled gracefully, no metadata shown)
- Path with nested directories (created automatically)
- Delete without confirmation (returns warning, file preserved)
- Invalid operation name (returns list of valid operations)

### Not Covered (Future Enhancement)

- Concurrent access (no locking for MVP)
- Very large notes (no size limits implemented)
- Special characters in paths (rely on OS/filesystem)

---

## Validation Commands

### Level 1: Syntax & Style

```bash
# Ruff linting
uv run ruff check app/
# Expected: Exit code 0, no errors

# Ruff formatting
uv run ruff format app/ --check
# Expected: Exit code 0, no changes needed
```

### Level 2: Unit Tests

```bash
# All tests with verbose output
uv run pytest app/tests/ -v

# With coverage report
uv run pytest app/tests/ --cov=app --cov-report=term-missing
# Expected: >80% coverage
```

### Level 3: Type Checking (Optional)

```bash
uv run mypy app/ --ignore-missing-imports
# Expected: No type errors
```

### Level 4: Manual Integration

```bash
# Health check
curl http://localhost:8000/health

# Create note
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Create a test note"}]}'
```

---

## Acceptance Criteria

- [ ] `python-frontmatter>=1.1.0` added to pyproject.toml
- [ ] `VaultClient.delete_file()` implemented with proper error handling
- [ ] `VaultClient.file_exists()` implemented returning bool
- [ ] `app/notes/__init__.py` exports `note_operations`
- [ ] `app/notes/models.py` contains `NoteFrontmatter` and `NoteContent`
- [ ] `app/notes/tools.py` implements all 5 operations
- [ ] Path normalization auto-appends `.md` and strips slashes
- [ ] Create operation generates YAML frontmatter with timestamps
- [ ] Update operation preserves frontmatter and updates modified timestamp
- [ ] Delete operation requires `confirm=True` (two-stage safety)
- [ ] Tool registered with `chat_agent` via `tools=[]`
- [ ] System prompt describes tool capabilities and guidelines
- [ ] All 22+ unit tests pass
- [ ] Linting passes with zero errors
- [ ] Formatting check passes
- [ ] Manual curl test successfully creates a note

---

## Completion Checklist

- [ ] Task 1: pyproject.toml updated
- [ ] Task 2: VaultClient.delete_file() added
- [ ] Task 3: VaultClient.file_exists() added
- [ ] Task 4: app/notes/__init__.py created
- [ ] Task 5: app/notes/models.py created
- [ ] Task 6: app/notes/tools.py created
- [ ] Task 7: chat/agent.py updated with tool + system prompt
- [ ] Task 8: test_notes.py created
- [ ] Task 9: Full test suite passes
- [ ] Task 10: Linting passes
- [ ] Task 11: Formatting passes
- [ ] Task 12: Manual integration test successful

---

## Notes & Design Decisions

### Why Consolidated Tool Pattern?

The PRD specifies using consolidated multi-operation tools rather than individual tools. This design:

1. **Reduces token usage** - One tool schema vs five
2. **Simplifies LLM reasoning** - Fewer tools to choose between
3. **Matches mental model** - "Note operations" is intuitive
4. **Enables future operations** - Easy to add `move`, `rename`, etc.

### Why Two-Stage Delete?

Delete is destructive and irreversible. The two-stage confirmation:

1. First call returns warning: "Are you sure? Call with confirm=True"
2. LLM relays this to user and waits for confirmation
3. Second call with `confirm=True` performs deletion

This prevents accidental deletions from ambiguous user requests.

### Why Preserve Existing Frontmatter Format?

Some users have notes without frontmatter, and we should respect their format:

- Create: Always adds frontmatter (new notes follow our standard)
- Update: Preserves whatever format existed (frontmatter or plain)
- This prevents surprise modifications to user's existing notes

### Potential Issues

1. **Circular import risk**: `tools.py` imports `ChatDependencies` from `chat/agent.py`. This works because we import the dataclass, not the agent instance.

2. **Frontmatter parsing edge cases**: Different note formats may have edge cases. The `try/except` in `parse_note()` handles gracefully by falling back to plain content.

3. **No file locking**: Concurrent access by Obsidian and the agent is not handled. Acceptable for MVP single-user deployment.

### Future Enhancements

- `append` mode for update operation (add content without replacing)
- `move` and `rename` operations
- Batch operations (create/update multiple notes)
- Note templates with variable substitution
- Tag operations integrated into note_operations
