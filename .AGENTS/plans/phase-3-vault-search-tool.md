# Feature: vault_search Tool

## Overview

Implement a unified `vault_search` tool providing search capabilities across the Obsidian vault with five operations: fulltext, by_tag, by_link, by_date, and combined.

**User Story**: As a knowledge worker, I want to find notes using natural language queries and filters, so I don't need to remember exact titles or manually browse folders.

**Feature Type**: New Capability | **Complexity**: Medium | **Systems**: `app/search/`, `app/chat/agent.py`

---

## MUST-READ FILES

| File | Lines | Purpose |
|------|-------|---------|
| `app/notes/tools.py` | 165-244 | Tool function pattern to follow |
| `app/notes/tools.py` | 30-87 | Helper functions pattern |
| `app/notes/models.py` | 9-71 | Pydantic model pattern |
| `app/chat/agent.py` | 1-38 | Tool registration pattern |
| `app/dependencies.py` | 62-164 | VaultClient methods |
| `app/tests/test_notes.py` | 157-190 | Test fixture pattern |

## Files to Create

- `app/search/__init__.py` - Export vault_search
- `app/search/models.py` - SearchResult, SearchResults models
- `app/search/tools.py` - vault_search tool implementation
- `app/tests/test_search.py` - Comprehensive tests

---

## KEY PATTERNS

**Tool Signature** (mirror `note_operations`):
```python
async def vault_search(
    ctx: RunContext[ChatDependencies],
    operation: str,
    query: str | None = None,
    # ... other params with defaults
) -> str:
```

**Logging**: Always include `trace_id`, operation, and relevant params
**Error Handling**: Return user-friendly strings, never raise exceptions
**Results**: Format as readable string for LLM response

---

## IMPLEMENTATION TASKS

### Task 1: CREATE `app/search/__init__.py`

```python
"""Vault search feature module."""
from .tools import vault_search

__all__ = ["vault_search"]
```

### Task 2: CREATE `app/search/models.py`

```python
"""Pydantic models for search results."""
from datetime import datetime
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single search result."""
    path: str = Field(..., description="Relative path to note")
    title: str = Field(..., description="Note title")
    excerpt: str = Field(default="", description="Match context")
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    created: datetime | None = None
    modified: datetime | None = None


class SearchResults(BaseModel):
    """Collection of search results."""
    results: list[SearchResult] = Field(default_factory=list)
    total: int = 0
    query: str = ""
    operation: str = ""
```

### Task 3: CREATE `app/search/tools.py`

**Imports and Constants:**
```python
"""Vault search tool for the chat agent."""
import re
from datetime import UTC, datetime

import frontmatter
from pydantic_ai import RunContext

from app.dependencies import ChatDependencies, VaultNotFoundError, logger
from app.notes.tools import normalize_path, parse_note
from app.search.models import SearchResult, SearchResults

WIKILINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
```

**Helper Functions to Implement:**

| Function | Purpose |
|----------|---------|
| `extract_wikilinks(content)` | Extract `[[links]]` from content |
| `extract_title(content, path)` | Get title from H1 or filename |
| `generate_excerpt(content, query, max_length=150)` | Create context excerpt around match |
| `calculate_relevance(content, query, title)` | Score 0.0-1.0 based on matches |
| `format_results(search_results)` | Convert SearchResults to LLM string |

**Private Operation Handlers:**

| Handler | Parameters | Logic |
|---------|------------|-------|
| `_fulltext_search` | query, folder, limit | Scan files for query matches, sort by relevance |
| `_search_by_tag` | tags, match_all, limit | Filter by frontmatter tags (AND/OR) |
| `_search_by_link` | note_path, direction, limit | Find backlinks or forward links |
| `_search_by_date` | start_date, end_date, date_field, limit | Filter by created/modified date |
| `_combined_search` | query, tags, start_date, end_date, folder, limit | Multi-criteria AND filter |

**Main Tool Function:**

```python
async def vault_search(
    ctx: RunContext[ChatDependencies],
    operation: str,
    query: str | None = None,
    tags: list[str] | None = None,
    note_path: str | None = None,
    direction: str = "backlinks",
    start_date: str | None = None,
    end_date: str | None = None,
    date_field: str = "created",
    folder: str | None = None,
    match_all: bool = False,
    limit: int = 10,
) -> str:
    """Search the Obsidian vault.

    Args:
        ctx: Context with vault access
        operation: 'fulltext' | 'by_tag' | 'by_link' | 'by_date' | 'combined'
        query: Text to search (fulltext/combined)
        tags: Tags to filter (by_tag/combined)
        note_path: Target note (by_link)
        direction: 'backlinks' or 'forward' (by_link)
        start_date: ISO date YYYY-MM-DD (by_date/combined)
        end_date: ISO date YYYY-MM-DD (by_date/combined)
        date_field: 'created' or 'modified' (by_date)
        folder: Limit to folder (fulltext/combined)
        match_all: Require ALL tags vs ANY (by_tag)
        limit: Max results (default 10)

    Returns:
        Formatted search results or error message
    """
    logger.info("vault_search_called", extra={
        "operation": operation, "query": query, "trace_id": ctx.deps.trace_id
    })

    try:
        if operation == "fulltext":
            if not query:
                return "Error: 'query' required for fulltext search"
            results = await _fulltext_search(ctx, query, folder, limit)
        elif operation == "by_tag":
            if not tags:
                return "Error: 'tags' required for tag search"
            results = await _search_by_tag(ctx, tags, match_all, limit)
        elif operation == "by_link":
            if not note_path:
                return "Error: 'note_path' required for link search"
            results = await _search_by_link(ctx, note_path, direction, limit)
        elif operation == "by_date":
            if not start_date and not end_date:
                return "Error: 'start_date' or 'end_date' required"
            results = await _search_by_date(ctx, start_date, end_date, date_field, limit)
        elif operation == "combined":
            if not any([query, tags, start_date, end_date, folder]):
                return "Error: at least one search criteria required"
            results = await _combined_search(ctx, query, tags, start_date, end_date, folder, limit)
        else:
            return f"Unknown operation: {operation}. Valid: fulltext, by_tag, by_link, by_date, combined"

        return format_results(results)

    except Exception as e:
        logger.error("vault_search_failed", extra={
            "operation": operation, "error": str(e), "trace_id": ctx.deps.trace_id
        }, exc_info=True)
        return f"Error performing {operation} search: {str(e)}"
```

### Task 4: UPDATE `app/chat/agent.py`

1. Add import: `from app.search import vault_search`
2. Update tools list: `tools=[note_operations, vault_search]`
3. Update SYSTEM_PROMPT to document search operations:

```python
SYSTEM_PROMPT = """You are a helpful AI assistant integrated with Obsidian.

You have access to note_operations tool:
- create, read, update, delete, summarize

You have access to vault_search tool:
- fulltext: Search note contents
- by_tag: Find notes by tags (AND/OR logic)
- by_link: Find backlinks/forward links
- by_date: Filter by date range
- combined: Multi-criteria search

Guidelines:
- Paths relative to vault root (e.g., 'Projects/API.md')
- .md extension added automatically
- Date format: YYYY-MM-DD
- For search, use fulltext for general queries, by_tag for organization"""
```

### Task 5: CREATE `app/tests/test_search.py`

**Test Structure:**
```python
"""Tests for search feature."""
from pathlib import Path
from unittest.mock import Mock
import pytest
from app.dependencies import ChatDependencies, VaultClient
from app.search.tools import (
    extract_wikilinks, extract_title, generate_excerpt,
    calculate_relevance, vault_search,
)


class TestExtractWikilinks:
    def test_simple_links(self) -> None: ...
    def test_aliased_links(self) -> None: ...
    def test_no_links(self) -> None: ...


class TestExtractTitle:
    def test_h1_heading(self) -> None: ...
    def test_filename_fallback(self) -> None: ...


class TestGenerateExcerpt:
    def test_centers_on_match(self) -> None: ...
    def test_strips_frontmatter(self) -> None: ...


class TestCalculateRelevance:
    def test_title_match_high(self) -> None: ...
    def test_score_capped(self) -> None: ...


class TestVaultSearch:
    @pytest.fixture
    def sample_vault(self, tmp_path: Path) -> Path:
        """Create vault with tagged, linked notes."""
        # Create 3-4 test notes with frontmatter, tags, wikilinks
        ...

    @pytest.fixture
    def mock_ctx(self, sample_vault: Path) -> Mock:
        ctx = Mock()
        ctx.deps = ChatDependencies(
            vault=VaultClient(vault_path=sample_vault),
            trace_id="test-123"
        )
        return ctx

    @pytest.mark.asyncio
    async def test_fulltext_finds_matches(self, mock_ctx): ...

    @pytest.mark.asyncio
    async def test_fulltext_requires_query(self, mock_ctx): ...

    @pytest.mark.asyncio
    async def test_tag_search_finds_notes(self, mock_ctx): ...

    @pytest.mark.asyncio
    async def test_link_search_backlinks(self, mock_ctx): ...

    @pytest.mark.asyncio
    async def test_date_search_range(self, mock_ctx): ...

    @pytest.mark.asyncio
    async def test_combined_search(self, mock_ctx): ...

    @pytest.mark.asyncio
    async def test_invalid_operation(self, mock_ctx): ...
```

---

## VALIDATION COMMANDS

```bash
# Lint & format
uv run ruff check app/search/
uv run ruff format app/search/ --check

# Tests
uv run pytest app/tests/test_search.py -v

# Full suite
uv run pytest app/tests/ -v

# Import check
uv run python -c "from app.chat.agent import chat_agent; print(f'Tools: {len(chat_agent._toolset.tools)}')"
```

---

## ACCEPTANCE CRITERIA

- [ ] `vault_search` implements all 5 operations
- [ ] Tool registered with chat agent
- [ ] System prompt documents search operations
- [ ] All functions have type hints and docstrings
- [ ] Logging includes trace_id
- [ ] Tests pass for all operations
- [ ] `uv run ruff check app/` passes
- [ ] `uv run pytest app/tests/` passes

---

## DESIGN NOTES

**Why single tool?** Follows `note_operations` pattern - reduces context window usage, simplifies agent decision-making.

**Why string returns?** LLM can naturally present formatted results in conversation.

**Relevance scoring:** Simple algorithm (title > content frequency > word boundaries). Can enhance with TF-IDF or embeddings post-MVP.

**Performance:** Reads all files for fulltext search. Acceptable for <500 notes. Future: build search index.
