# Feature: tag_management Tool (Phase 4)

> **Status**: Ready for Implementation
> **Complexity**: Medium-High | **Tasks**: 10 | **Confidence**: 8/10

---

## Overview

Implement `tag_management` tool - the **connection layer** that builds the knowledge graph through intelligent tagging and wikilink creation.

**User Story**: As an Obsidian user, I want intelligent tag suggestions and automatic note connections, so my knowledge graph grows organically without manual effort.

**Feature Type**: New Capability | **Systems**: `app/tags/`, `app/chat/agent.py`

### Operations

| Operation | Description | Required Params |
|-----------|-------------|-----------------|
| `add` | Add tags to note frontmatter | `path`, `tags` |
| `remove` | Remove tags from frontmatter | `path`, `tags` |
| `rename` | Rename tag vault-wide | `old_tag`, `new_tag` |
| `list` | List all tags with counts | `sort_by?`, `limit?` |
| `suggest` | AI tag suggestions from content | `path` |
| `auto_tag` | Apply suggestions (with confirm) | `path`, `confirm?` |
| `connect` | Create wikilinks to related notes | `path`, `limit?` |

---

## MUST-READ FILES

| File | Lines | Purpose |
|------|-------|---------|
| `app/notes/tools.py` | 30-87 | `normalize_path()`, `parse_note()` - reuse |
| `app/notes/tools.py` | 120-157 | `update_note_content()` - frontmatter pattern |
| `app/notes/tools.py` | 165-244 | Tool function pattern to follow |
| `app/search/tools.py` | 37-84 | `extract_wikilinks()`, `extract_title()` - reuse |
| `app/search/tools.py` | 136-181 | `calculate_relevance()` - reuse for scoring |
| `app/notes/models.py` | 9-54 | `NoteFrontmatter` - tags field pattern |
| `app/chat/agent.py` | 1-51 | Tool registration pattern |
| `app/tests/test_search.py` | 302-381 | Sample vault fixture pattern |

## Files to Create

- `app/tags/__init__.py` - Export tag_management
- `app/tags/models.py` - TagInfo, TagSuggestion, ConnectionResult
- `app/tags/tools.py` - tag_management tool implementation
- `app/tests/test_tags.py` - Comprehensive tests

---

## KEY PATTERNS

**Tool Signature** (mirror `note_operations`):
```python
async def tag_management(
    ctx: RunContext[ChatDependencies],
    operation: str,
    path: str | None = None,
    tags: list[str] | None = None,
    old_tag: str | None = None,
    new_tag: str | None = None,
    sort_by: str = "count",
    limit: int = 10,
    confirm: bool = False,
) -> str:
```

**Logging**: Always include `trace_id`, operation, path
**Error Handling**: Return user-friendly strings, never raise exceptions
**Tag Format**: Without # prefix (e.g., "project" not "#project")

---

## IMPLEMENTATION TASKS

### Task 1: CREATE `app/tags/__init__.py`

```python
"""Tags feature module - tag management and note connections."""
from .tools import tag_management

__all__ = ["tag_management"]
```

---

### Task 2: CREATE `app/tags/models.py`

```python
"""Pydantic models for tag management."""
from pydantic import BaseModel, Field


class TagInfo(BaseModel):
    """Tag with usage count."""
    name: str = Field(..., description="Tag name without #")
    count: int = Field(default=0, ge=0)
    notes: list[str] = Field(default_factory=list)


class TagSuggestion(BaseModel):
    """Suggested tag with confidence."""
    tag: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""
    existing: bool = False  # Exists in vault taxonomy


class ConnectionResult(BaseModel):
    """Result of connecting notes."""
    path: str
    connections_added: int = 0
    connected_notes: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
```

**Validate**: `uv run python -c "from app.tags.models import TagInfo, TagSuggestion"`

---

### Task 3: CREATE `app/tags/tools.py` - Imports and Helpers

```python
"""Tag management tool for the chat agent."""
import re
from collections import Counter
from datetime import UTC, datetime

import frontmatter
from pydantic_ai import RunContext

from app.dependencies import ChatDependencies, VaultNotFoundError, logger
from app.notes.tools import normalize_path, parse_note
from app.search.tools import extract_title, extract_wikilinks
from app.tags.models import TagInfo, TagSuggestion, ConnectionResult


async def get_all_vault_tags(ctx: RunContext[ChatDependencies]) -> dict[str, list[str]]:
    """Scan vault, return dict of tag -> list of note paths."""
    tag_to_notes: dict[str, list[str]] = {}
    files = await ctx.deps.vault.list_files()

    for file_path in files:
        try:
            content = await ctx.deps.vault.read_file(file_path)
            parsed = parse_note(content)
            if parsed.frontmatter and parsed.frontmatter.tags:
                for tag in parsed.frontmatter.tags:
                    tag_lower = tag.lower()
                    if tag_lower not in tag_to_notes:
                        tag_to_notes[tag_lower] = []
                    tag_to_notes[tag_lower].append(file_path)
        except Exception:
            continue
    return tag_to_notes


def extract_keywords(content: str, min_length: int = 4) -> list[str]:
    """Extract keywords from content, filtered by length and stopwords."""
    stopwords = {"the", "and", "for", "are", "but", "not", "you", "all", "can",
                 "had", "was", "one", "our", "has", "have", "been", "this", "that",
                 "with", "they", "from", "will", "would", "there", "their", "what",
                 "about", "which", "when", "make", "like", "time", "just", "know"}
    words = re.findall(r'\b[a-z]+\b', content.lower())
    filtered = [w for w in words if len(w) >= min_length and w not in stopwords]
    return [word for word, _ in Counter(filtered).most_common(20)]


def update_frontmatter_tags(content: str, new_tags: list[str]) -> str:
    """Update tags in frontmatter, preserve other fields, update modified."""
    try:
        post = frontmatter.loads(content)
        if post.metadata is None:
            post.metadata = {}
        post.metadata["tags"] = new_tags
        post.metadata["modified"] = datetime.now(UTC).isoformat()
        return frontmatter.dumps(post)
    except Exception:
        # Create frontmatter if missing
        fm = f"---\ncreated: {datetime.now(UTC).isoformat()}\nmodified: {datetime.now(UTC).isoformat()}\ntags:\n"
        for tag in new_tags:
            fm += f"  - {tag}\n"
        return fm + "---\n" + content


def insert_wikilinks(content: str, links: list[str]) -> str:
    """Insert wikilinks in Related Notes section at end of content."""
    if not links:
        return content
    section = "\n\n## Related Notes\n\n"
    for link in links:
        section += f"- {link}\n"
    return content.rstrip() + section
```

---

### Task 4: ADD Operation Handlers to `app/tags/tools.py`

```python
# =============================================================================
# Operation Handlers
# =============================================================================

async def _add_tags(ctx: RunContext[ChatDependencies], path: str, tags: list[str]) -> str:
    """Add tags to note frontmatter."""
    path = normalize_path(path)
    if not await ctx.deps.vault.file_exists(path):
        return f"Note not found at {path}."

    content = await ctx.deps.vault.read_file(path)
    parsed = parse_note(content)
    existing = parsed.frontmatter.tags if parsed.frontmatter else []
    existing_set = {t.lower() for t in existing}

    new_tags = [t for t in tags if t.lower() not in existing_set]
    if not new_tags:
        return f"Note already has all specified tags."

    updated = update_frontmatter_tags(content, existing + new_tags)
    await ctx.deps.vault.write_file(path, updated)
    logger.info("tags_added", extra={"path": path, "tags": new_tags, "trace_id": ctx.deps.trace_id})
    return f"Added {len(new_tags)} tag(s) to {path}: {', '.join(new_tags)}"


async def _remove_tags(ctx: RunContext[ChatDependencies], path: str, tags: list[str]) -> str:
    """Remove tags from note frontmatter."""
    path = normalize_path(path)
    if not await ctx.deps.vault.file_exists(path):
        return f"Note not found at {path}."

    content = await ctx.deps.vault.read_file(path)
    parsed = parse_note(content)
    if not parsed.frontmatter or not parsed.frontmatter.tags:
        return f"Note has no tags to remove."

    to_remove = {t.lower() for t in tags}
    remaining = [t for t in parsed.frontmatter.tags if t.lower() not in to_remove]
    removed_count = len(parsed.frontmatter.tags) - len(remaining)

    if removed_count == 0:
        return f"None of the specified tags exist on {path}."

    updated = update_frontmatter_tags(content, remaining)
    await ctx.deps.vault.write_file(path, updated)
    return f"Removed {removed_count} tag(s) from {path}"


async def _rename_tag(ctx: RunContext[ChatDependencies], old_tag: str, new_tag: str) -> str:
    """Rename tag across entire vault."""
    if old_tag.lower() == new_tag.lower():
        return "Old and new tag are the same."

    files = await ctx.deps.vault.list_files()
    updated_count = 0

    for file_path in files:
        try:
            content = await ctx.deps.vault.read_file(file_path)
            parsed = parse_note(content)
            if not parsed.frontmatter or not parsed.frontmatter.tags:
                continue

            if not any(t.lower() == old_tag.lower() for t in parsed.frontmatter.tags):
                continue

            new_tags = [new_tag if t.lower() == old_tag.lower() else t
                       for t in parsed.frontmatter.tags]
            updated = update_frontmatter_tags(content, new_tags)
            await ctx.deps.vault.write_file(file_path, updated)
            updated_count += 1
        except Exception:
            continue

    if updated_count == 0:
        return f"Tag '{old_tag}' not found in any notes."
    return f"Renamed #{old_tag} → #{new_tag} in {updated_count} note(s)"


async def _list_tags(ctx: RunContext[ChatDependencies], sort_by: str, limit: int) -> str:
    """List all vault tags with counts."""
    tag_to_notes = await get_all_vault_tags(ctx)
    if not tag_to_notes:
        return "No tags found in vault."

    tags = [TagInfo(name=t, count=len(n)) for t, n in tag_to_notes.items()]
    tags.sort(key=lambda x: x.count if sort_by == "count" else x.name, reverse=(sort_by == "count"))

    lines = [f"Found {len(tags)} tags:", ""]
    for i, t in enumerate(tags[:limit], 1):
        lines.append(f"{i}. **#{t.name}** ({t.count} notes)")
    if len(tags) > limit:
        lines.append(f"\n...and {len(tags) - limit} more")
    return "\n".join(lines)


async def _suggest_tags(ctx: RunContext[ChatDependencies], path: str) -> str:
    """Suggest tags based on content and existing taxonomy."""
    path = normalize_path(path)
    if not await ctx.deps.vault.file_exists(path):
        return f"Note not found at {path}."

    content = await ctx.deps.vault.read_file(path)
    parsed = parse_note(content)
    current = {t.lower() for t in (parsed.frontmatter.tags if parsed.frontmatter else [])}

    all_tags = await get_all_vault_tags(ctx)
    keywords = extract_keywords(parsed.body)

    suggestions = []
    for kw in keywords:
        if kw in all_tags and kw not in current:
            count = len(all_tags[kw])
            suggestions.append(TagSuggestion(
                tag=kw, confidence=min(0.9, 0.5 + count * 0.05),
                reason=f"Used by {count} notes", existing=True
            ))

    suggestions = sorted(suggestions, key=lambda x: x.confidence, reverse=True)[:10]
    if not suggestions:
        return f"No tag suggestions for {path}."

    lines = [f"Suggestions for **{path}**:", ""]
    for s in suggestions:
        lines.append(f"- **#{s.tag}** ({int(s.confidence*100)}%) - {s.reason}")
    lines.append("\nUse auto_tag with confirm=True to apply.")
    return "\n".join(lines)


async def _auto_tag(ctx: RunContext[ChatDependencies], path: str, confirm: bool) -> str:
    """Apply high-confidence tag suggestions."""
    path = normalize_path(path)
    if not await ctx.deps.vault.file_exists(path):
        return f"Note not found at {path}."

    content = await ctx.deps.vault.read_file(path)
    parsed = parse_note(content)
    current = {t.lower() for t in (parsed.frontmatter.tags if parsed.frontmatter else [])}

    all_tags = await get_all_vault_tags(ctx)
    keywords = extract_keywords(parsed.body)

    auto_tags = [kw for kw in keywords if kw in all_tags and kw not in current][:5]
    if not auto_tags:
        return f"No suitable tags for {path}."

    if not confirm:
        return f"Would add: {', '.join(auto_tags)}. Call with confirm=True to apply."

    existing = parsed.frontmatter.tags if parsed.frontmatter else []
    updated = update_frontmatter_tags(content, existing + auto_tags)
    await ctx.deps.vault.write_file(path, updated)
    return f"Auto-tagged {path}: {', '.join(auto_tags)}"


async def _connect_notes(ctx: RunContext[ChatDependencies], path: str, limit: int) -> str:
    """Find related notes and add wikilinks."""
    path = normalize_path(path)
    if not await ctx.deps.vault.file_exists(path):
        return f"Note not found at {path}."

    content = await ctx.deps.vault.read_file(path)
    parsed = parse_note(content)
    source_tags = set(t.lower() for t in (parsed.frontmatter.tags if parsed.frontmatter else []))
    existing_links = set(extract_wikilinks(content))

    files = await ctx.deps.vault.list_files()
    related = []

    for fp in files:
        if fp == path:
            continue
        fname = fp.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if any(fname in lnk or lnk in fp for lnk in existing_links):
            continue

        try:
            fc = await ctx.deps.vault.read_file(fp)
            fp_parsed = parse_note(fc)
            if not fp_parsed.frontmatter:
                continue

            note_tags = set(t.lower() for t in fp_parsed.frontmatter.tags)
            shared = source_tags & note_tags
            if len(shared) >= 2:
                title = extract_title(fc, fp)
                related.append((f"[[{title}]]", f"shares: {', '.join(shared)}", len(shared)))
        except Exception:
            continue

    related.sort(key=lambda x: x[2], reverse=True)
    related = related[:limit]

    if not related:
        return f"No related notes found for {path}."

    links = [r[0] for r in related]
    updated = insert_wikilinks(content, links)
    await ctx.deps.vault.write_file(path, updated)

    lines = [f"Connected {path} to {len(related)} note(s):", ""]
    for link, reason, _ in related:
        lines.append(f"- {link} - {reason}")
    return "\n".join(lines)
```

---

### Task 5: ADD Main Tool Function to `app/tags/tools.py`

```python
async def tag_management(
    ctx: RunContext[ChatDependencies],
    operation: str,
    path: str | None = None,
    tags: list[str] | None = None,
    old_tag: str | None = None,
    new_tag: str | None = None,
    sort_by: str = "count",
    limit: int = 10,
    confirm: bool = False,
) -> str:
    """Manage tags and note connections in the vault.

    Args:
        ctx: Context with vault access
        operation: 'add'|'remove'|'rename'|'list'|'suggest'|'auto_tag'|'connect'
        path: Note path for single-note operations
        tags: Tags to add/remove (without #)
        old_tag: Tag to rename from
        new_tag: Tag to rename to
        sort_by: 'count' or 'alpha' for list
        limit: Max results for list/connect
        confirm: Must be True for auto_tag

    Returns:
        Operation result or error message
    """
    logger.info("tag_management", extra={
        "operation": operation, "path": path, "trace_id": ctx.deps.trace_id
    })

    try:
        if operation == "add":
            if not path or not tags:
                return "Error: 'path' and 'tags' required for add."
            return await _add_tags(ctx, path, tags)
        elif operation == "remove":
            if not path or not tags:
                return "Error: 'path' and 'tags' required for remove."
            return await _remove_tags(ctx, path, tags)
        elif operation == "rename":
            if not old_tag or not new_tag:
                return "Error: 'old_tag' and 'new_tag' required."
            return await _rename_tag(ctx, old_tag, new_tag)
        elif operation == "list":
            return await _list_tags(ctx, sort_by, limit)
        elif operation == "suggest":
            if not path:
                return "Error: 'path' required for suggest."
            return await _suggest_tags(ctx, path)
        elif operation == "auto_tag":
            if not path:
                return "Error: 'path' required for auto_tag."
            return await _auto_tag(ctx, path, confirm)
        elif operation == "connect":
            if not path:
                return "Error: 'path' required for connect."
            return await _connect_notes(ctx, path, limit)
        else:
            return f"Unknown operation: {operation}. Valid: add, remove, rename, list, suggest, auto_tag, connect"
    except Exception as e:
        logger.error("tag_management_failed", extra={
            "operation": operation, "error": str(e), "trace_id": ctx.deps.trace_id
        }, exc_info=True)
        return f"Error: {str(e)}"
```

---

### Task 6: UPDATE `app/chat/agent.py`

1. Add import: `from app.tags import tag_management`
2. Update tools: `tools=[note_operations, vault_search, tag_management]`
3. Add to SYSTEM_PROMPT:

```python
"""
You have access to tag_management tool:
- add: Add tags to note frontmatter
- remove: Remove tags from frontmatter
- rename: Rename tag vault-wide
- list: List all tags with counts
- suggest: AI tag suggestions from content
- auto_tag: Apply suggestions (needs confirm=True)
- connect: Create [[wikilinks]] to related notes

Tags without # prefix. Use suggest before auto_tag.
"""
```

**Validate**:
```bash
uv run python -c "from app.chat.agent import chat_agent; print(list(chat_agent._function_tools.keys()))"
```

---

### Task 7: CREATE `app/tests/test_tags.py`

**Test Classes**:
- `TestExtractKeywords` - 3 tests (frequency, length, stopwords)
- `TestUpdateFrontmatterTags` - 2 tests (update existing, create new)
- `TestInsertWikilinks` - 2 tests (add section, empty)
- `TestTagModels` - 3 tests (defaults, bounds)
- `TestTagManagement` - 15 tests covering all operations

**Sample Vault Fixture** (reuse pattern from test_search.py):
```python
@pytest.fixture
def sample_vault(self, tmp_path: Path) -> Path:
    """Create vault with tagged notes."""
    (tmp_path / "API.md").write_text("""---
tags: [project, api]
---
# API Design""")
    (tmp_path / "Arch.md").write_text("""---
tags: [project]
---
# Architecture""")
    (tmp_path / "Meeting.md").write_text("""---
tags: [meeting]
---
# Meeting about API project""")
    return tmp_path
```

---

### Task 8: Run Tests

```bash
uv run pytest app/tests/test_tags.py -v
uv run pytest app/tests/ -v --tb=short
```

---

### Task 9: Lint and Format

```bash
uv run ruff check app/
uv run ruff format app/ --check
```

---

### Task 10: Manual Integration Test

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "List my tags"}]}'
```

---

## VALIDATION COMMANDS

```bash
# Lint & format
uv run ruff check app/tags/
uv run ruff format app/ --check

# Tests
uv run pytest app/tests/test_tags.py -v

# Full suite
uv run pytest app/tests/ -v

# Import check
uv run python -c "from app.chat.agent import chat_agent; assert 'tag_management' in chat_agent._function_tools"
```

---

## ACCEPTANCE CRITERIA

- [ ] `tag_management` implements all 7 operations
- [ ] Tool registered with chat_agent
- [ ] System prompt documents operations
- [ ] Add/remove manipulate frontmatter correctly
- [ ] Rename updates vault-wide atomically
- [ ] Suggest returns relevant existing tags
- [ ] Auto_tag requires confirm=True
- [ ] Connect inserts wikilinks in Related Notes section
- [ ] All tests pass
- [ ] `uv run ruff check app/` passes

---

## DESIGN NOTES

**Why prefer existing tags?** Maintains taxonomy consistency. Suggesting `#project` over `#proj` keeps the vault organized.

**Why two-stage auto_tag?** Like delete, modifies files. Preview first, apply with confirmation.

**Connection algorithm**: Score by shared tags (2+ tags = high score). Simple but effective for MVP.

**Performance**: Reads all files for vault-wide ops. Acceptable for <500 notes. Future: add caching.
