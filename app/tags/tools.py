"""Tag management tool for the chat agent.

This module implements the tag_management tool which provides tag
operations and note connection features through the PydanticAI agent.

The tool follows a consolidated multi-operation pattern where a single
tool handles multiple operations via an `operation` parameter.

Example usage by the agent:
    tag_management(operation="add", path="Projects/API", tags=["backend", "api"])
    tag_management(operation="list", sort_by="count", limit=20)
    tag_management(operation="suggest", path="Projects/API")
    tag_management(operation="connect", path="Projects/API", limit=5)
"""

import re
from collections import Counter
from datetime import UTC, datetime

import frontmatter
from pydantic_ai import RunContext

from app.dependencies import ChatDependencies, logger
from app.notes.tools import normalize_path, parse_note
from app.search.tools import extract_title, extract_wikilinks
from app.tags.models import TagInfo, TagSuggestion

# =============================================================================
# Helper Functions
# =============================================================================


async def get_all_vault_tags(ctx: RunContext[ChatDependencies]) -> dict[str, list[str]]:
    """Scan vault and return dict of tag -> list of note paths.

    Args:
        ctx: Context with vault access

    Returns:
        Dictionary mapping lowercase tag names to lists of note paths
    """
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
    """Extract keywords from content, filtered by length and stopwords.

    Args:
        content: Note content to analyze
        min_length: Minimum word length to consider

    Returns:
        List of keywords sorted by frequency (most common first)
    """
    stopwords = {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "had",
        "was",
        "one",
        "our",
        "has",
        "have",
        "been",
        "this",
        "that",
        "with",
        "they",
        "from",
        "will",
        "would",
        "there",
        "their",
        "what",
        "about",
        "which",
        "when",
        "make",
        "like",
        "time",
        "just",
        "know",
    }
    words = re.findall(r"\b[a-z]+\b", content.lower())
    filtered = [w for w in words if len(w) >= min_length and w not in stopwords]
    return [word for word, _ in Counter(filtered).most_common(20)]


def update_frontmatter_tags(content: str, new_tags: list[str]) -> str:
    """Update tags in frontmatter, preserve other fields, update modified.

    Args:
        content: Note content with or without frontmatter
        new_tags: List of tags to set

    Returns:
        Updated note content with new tags in frontmatter
    """
    try:
        post = frontmatter.loads(content)
        if post.metadata is None:
            post.metadata = {}
        post.metadata["tags"] = new_tags
        post.metadata["modified"] = datetime.now(UTC).isoformat()
        return frontmatter.dumps(post)
    except Exception:
        # Create frontmatter if missing
        now = datetime.now(UTC).isoformat()
        fm = f"---\ncreated: {now}\nmodified: {now}\ntags:\n"
        for tag in new_tags:
            fm += f"  - {tag}\n"
        return fm + "---\n" + content


def insert_wikilinks(content: str, links: list[str]) -> str:
    """Insert wikilinks in Related Notes section at end of content.

    Args:
        content: Note content to append links to
        links: List of wikilinks to add (e.g., ["[[Note Title]]"])

    Returns:
        Content with Related Notes section appended
    """
    if not links:
        return content
    section = "\n\n## Related Notes\n\n"
    for link in links:
        section += f"- {link}\n"
    return content.rstrip() + section


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
        return "Note already has all specified tags."

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
        return "Note has no tags to remove."

    to_remove = {t.lower() for t in tags}
    remaining = [t for t in parsed.frontmatter.tags if t.lower() not in to_remove]
    removed_count = len(parsed.frontmatter.tags) - len(remaining)

    if removed_count == 0:
        return f"None of the specified tags exist on {path}."

    updated = update_frontmatter_tags(content, remaining)
    await ctx.deps.vault.write_file(path, updated)
    logger.info(
        "tags_removed",
        extra={"path": path, "count": removed_count, "trace_id": ctx.deps.trace_id},
    )
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

            new_tags = [
                new_tag if t.lower() == old_tag.lower() else t for t in parsed.frontmatter.tags
            ]
            updated = update_frontmatter_tags(content, new_tags)
            await ctx.deps.vault.write_file(file_path, updated)
            updated_count += 1
        except Exception:
            continue

    if updated_count == 0:
        return f"Tag '{old_tag}' not found in any notes."

    logger.info(
        "tag_renamed",
        extra={
            "old_tag": old_tag,
            "new_tag": new_tag,
            "count": updated_count,
            "trace_id": ctx.deps.trace_id,
        },
    )
    return f"Renamed #{old_tag} -> #{new_tag} in {updated_count} note(s)"


async def _list_tags(ctx: RunContext[ChatDependencies], sort_by: str, limit: int) -> str:
    """List all vault tags with counts."""
    tag_to_notes = await get_all_vault_tags(ctx)
    if not tag_to_notes:
        return "No tags found in vault."

    tags = [TagInfo(name=t, count=len(n)) for t, n in tag_to_notes.items()]
    tags.sort(
        key=lambda x: x.count if sort_by == "count" else x.name,
        reverse=(sort_by == "count"),
    )

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

    suggestions: list[TagSuggestion] = []
    for kw in keywords:
        if kw in all_tags and kw not in current:
            count = len(all_tags[kw])
            suggestions.append(
                TagSuggestion(
                    tag=kw,
                    confidence=min(0.9, 0.5 + count * 0.05),
                    reason=f"Used by {count} notes",
                    existing=True,
                )
            )

    suggestions = sorted(suggestions, key=lambda x: x.confidence, reverse=True)[:10]
    if not suggestions:
        return f"No tag suggestions for {path}."

    lines = [f"Suggestions for **{path}**:", ""]
    for s in suggestions:
        lines.append(f"- **#{s.tag}** ({int(s.confidence * 100)}%) - {s.reason}")
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
    logger.info(
        "auto_tagged",
        extra={"path": path, "tags": auto_tags, "trace_id": ctx.deps.trace_id},
    )
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
    related: list[tuple[str, str, int]] = []

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

    logger.info(
        "notes_connected",
        extra={"path": path, "count": len(related), "trace_id": ctx.deps.trace_id},
    )

    lines = [f"Connected {path} to {len(related)} note(s):", ""]
    for link, reason, _ in related:
        lines.append(f"- {link} - {reason}")
    return "\n".join(lines)


# =============================================================================
# Main Tool Function
# =============================================================================


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

    This is the main tool function that the LLM calls to manage tags
    and create connections between notes.

    Args:
        ctx: Context with vault access and trace_id for logging
        operation: One of 'add', 'remove', 'rename', 'list', 'suggest',
                   'auto_tag', 'connect'
        path: Note path for single-note operations
        tags: Tags to add/remove (without # prefix)
        old_tag: Tag to rename from (for rename operation)
        new_tag: Tag to rename to (for rename operation)
        sort_by: 'count' or 'alpha' for list operation
        limit: Max results for list/connect operations
        confirm: Must be True for auto_tag operation

    Returns:
        Operation result or error message. Returns user-friendly
        error messages instead of raising exceptions.

    Examples:
        add:      tag_management(ctx, "add", path="note.md", tags=["project"])
        remove:   tag_management(ctx, "remove", path="note.md", tags=["old"])
        rename:   tag_management(ctx, "rename", old_tag="proj", new_tag="project")
        list:     tag_management(ctx, "list", sort_by="count", limit=20)
        suggest:  tag_management(ctx, "suggest", path="note.md")
        auto_tag: tag_management(ctx, "auto_tag", path="note.md", confirm=True)
        connect:  tag_management(ctx, "connect", path="note.md", limit=5)
    """
    logger.info(
        "tag_management",
        extra={"operation": operation, "path": path, "trace_id": ctx.deps.trace_id},
    )

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
            return (
                f"Unknown operation: {operation}. "
                f"Valid: add, remove, rename, list, suggest, auto_tag, connect"
            )
    except Exception as e:
        logger.error(
            "tag_management_failed",
            extra={"operation": operation, "error": str(e), "trace_id": ctx.deps.trace_id},
            exc_info=True,
        )
        return f"Error: {str(e)}"
