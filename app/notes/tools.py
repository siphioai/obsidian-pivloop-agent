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

from datetime import UTC, datetime

import frontmatter
from pydantic_ai import RunContext

from app.dependencies import (
    ChatDependencies,
    VaultNotFoundError,
    VaultSecurityError,
    logger,
)
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
                created=fm_data.get("created", datetime.now(UTC)),
                modified=fm_data.get("modified", datetime.now(UTC)),
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
        fm_dict["modified"] = datetime.now(UTC).isoformat()
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
    content: str | None,
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
    content: str | None,
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
    confirm: bool,
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
