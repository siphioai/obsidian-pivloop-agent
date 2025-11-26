"""Vault search tool for the chat agent.

This module implements the vault_search tool which provides search
capabilities across the Obsidian vault through the PydanticAI agent.

The tool follows a consolidated multi-operation pattern where a single
tool handles multiple search types via an `operation` parameter. This
reduces context window usage and simplifies LLM decision-making.

Example usage by the agent:
    vault_search(operation="fulltext", query="API design")
    vault_search(operation="by_tag", tags=["project", "active"])
    vault_search(operation="by_link", note_path="Projects/API.md", direction="backlinks")
    vault_search(operation="by_date", start_date="2025-01-01", end_date="2025-01-31")
    vault_search(operation="combined", query="meeting", tags=["work"], folder="Work")
"""

import re
from datetime import UTC, datetime

import frontmatter
from pydantic_ai import RunContext

from app.dependencies import ChatDependencies, logger
from app.notes.tools import normalize_path, parse_note
from app.search.models import SearchResult, SearchResults

# Regex pattern for extracting wikilinks: [[target]] or [[target|alias]]
WIKILINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


# =============================================================================
# Helper Functions
# =============================================================================


def extract_wikilinks(content: str) -> list[str]:
    """Extract wikilink targets from content.

    Finds all [[link]] and [[link|alias]] patterns and extracts
    the target note path (without the alias).

    Args:
        content: Note content to search

    Returns:
        List of unique link targets (normalized paths)

    Examples:
        >>> extract_wikilinks("See [[API Design]] and [[Projects/Plan|Plan]]")
        ['API Design.md', 'Projects/Plan.md']
    """
    matches = WIKILINK_PATTERN.findall(content)
    return list({normalize_path(match) for match in matches})


def extract_title(content: str, path: str) -> str:
    """Extract note title from content or path.

    Attempts to find an H1 heading (# Title) in the content.
    Falls back to the filename without extension.

    Args:
        content: Note content (may include frontmatter)
        path: File path as fallback

    Returns:
        The note title

    Examples:
        >>> extract_title("# My Note\\nContent", "notes/my-note.md")
        'My Note'
        >>> extract_title("No heading here", "notes/my-note.md")
        'my-note'
    """
    # Try to find H1 heading
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()

    # Fallback to filename without extension
    filename = path.rsplit("/", 1)[-1]
    return filename.rsplit(".", 1)[0]


def generate_excerpt(content: str, query: str, max_length: int = 150) -> str:
    """Generate an excerpt centered on the query match.

    Creates a context snippet around where the query appears in
    the content. Strips frontmatter and truncates to max_length.

    Args:
        content: Full note content
        query: Search query to center on
        max_length: Maximum excerpt length

    Returns:
        Context snippet with ellipsis if truncated

    Examples:
        >>> generate_excerpt("Long content with API design notes", "API", 30)
        '...nt with API design no...'
    """
    # Strip frontmatter if present
    try:
        post = frontmatter.loads(content)
        body = post.content
    except Exception:
        body = content

    # Find query position (case-insensitive)
    lower_body = body.lower()
    lower_query = query.lower()
    pos = lower_body.find(lower_query)

    if pos == -1:
        # Query not found, return start of content
        excerpt = body[:max_length].strip()
        return f"{excerpt}..." if len(body) > max_length else excerpt

    # Center the excerpt around the match
    half_length = max_length // 2
    start = max(0, pos - half_length)
    end = min(len(body), pos + len(query) + half_length)

    excerpt = body[start:end].strip()

    # Add ellipsis if truncated
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(body) else ""

    return f"{prefix}{excerpt}{suffix}"


def calculate_relevance(content: str, query: str, title: str) -> float:
    """Calculate relevance score for a search result.

    Scores based on:
    - Title match (highest weight)
    - Query frequency in content
    - Word boundary matches

    Args:
        content: Note body content
        query: Search query
        title: Note title

    Returns:
        Score between 0.0 and 1.0

    Examples:
        >>> calculate_relevance("API design doc", "API", "API Design Guide")
        0.95  # High score due to title match
    """
    score = 0.0
    lower_query = query.lower()
    lower_title = title.lower()
    lower_content = content.lower()

    # Title match is highest priority (0.5 base)
    if lower_query in lower_title:
        score += 0.5
        # Exact title match bonus
        if lower_query == lower_title:
            score += 0.3

    # Content frequency scoring (up to 0.3)
    count = lower_content.count(lower_query)
    if count > 0:
        # Diminishing returns for frequency
        frequency_score = min(0.3, 0.1 * count)
        score += frequency_score

    # Word boundary matches (0.1)
    word_pattern = rf"\b{re.escape(lower_query)}\b"
    if re.search(word_pattern, lower_content, re.IGNORECASE):
        score += 0.1

    # Cap at 1.0
    return min(1.0, score)


def format_results(search_results: SearchResults) -> str:
    """Format search results for LLM response.

    Converts SearchResults into a human-readable string that the
    LLM can present to the user.

    Args:
        search_results: SearchResults object to format

    Returns:
        Formatted string with results summary and details
    """
    if not search_results.results:
        return f"No results found for {search_results.operation} search."

    lines = [
        f"Found {search_results.total} result(s) for {search_results.operation} search",
    ]

    if search_results.query:
        lines[0] += f" (query: '{search_results.query}')"

    lines.append("")  # Blank line

    for i, result in enumerate(search_results.results, 1):
        lines.append(f"{i}. **{result.title}** ({result.path})")

        if result.score < 1.0:
            lines.append(f"   Relevance: {result.score:.0%}")

        if result.tags:
            lines.append(f"   Tags: {', '.join(result.tags)}")

        if result.excerpt:
            lines.append(f"   {result.excerpt}")

        lines.append("")  # Blank line between results

    return "\n".join(lines)


# =============================================================================
# Private Operation Handlers
# =============================================================================


async def _fulltext_search(
    ctx: RunContext[ChatDependencies],
    query: str,
    folder: str | None,
    limit: int,
) -> SearchResults:
    """Search notes by content.

    Scans all markdown files for query matches, calculates relevance
    scores, and returns sorted results.

    Args:
        ctx: Context with vault access
        query: Text to search for
        folder: Optional folder to limit search
        limit: Maximum results to return

    Returns:
        SearchResults with matching notes
    """
    results: list[SearchResult] = []
    files = await ctx.deps.vault.list_files(folder=folder or "")

    for file_path in files:
        try:
            content = await ctx.deps.vault.read_file(file_path)
            lower_content = content.lower()
            lower_query = query.lower()

            if lower_query not in lower_content:
                continue

            parsed = parse_note(content)
            title = extract_title(content, file_path)
            score = calculate_relevance(parsed.body, query, title)
            excerpt = generate_excerpt(content, query)

            result = SearchResult(
                path=file_path,
                title=title,
                excerpt=excerpt,
                score=score,
                tags=parsed.frontmatter.tags if parsed.frontmatter else [],
                created=parsed.frontmatter.created if parsed.frontmatter else None,
                modified=parsed.frontmatter.modified if parsed.frontmatter else None,
            )
            results.append(result)

        except Exception as e:
            logger.debug(
                "fulltext_search_file_error",
                extra={"path": file_path, "error": str(e), "trace_id": ctx.deps.trace_id},
            )
            continue

    # Sort by relevance score (descending)
    results.sort(key=lambda r: r.score, reverse=True)

    return SearchResults(
        results=results[:limit],
        total=len(results),
        query=query,
        operation="fulltext",
    )


async def _search_by_tag(
    ctx: RunContext[ChatDependencies],
    tags: list[str],
    match_all: bool,
    limit: int,
) -> SearchResults:
    """Search notes by frontmatter tags.

    Filters notes based on tag presence. Can require all tags (AND)
    or any tag (OR) depending on match_all parameter.

    Args:
        ctx: Context with vault access
        tags: Tags to search for
        match_all: If True, note must have ALL tags; if False, ANY tag
        limit: Maximum results to return

    Returns:
        SearchResults with matching notes
    """
    results: list[SearchResult] = []
    files = await ctx.deps.vault.list_files()
    search_tags = {t.lower() for t in tags}

    for file_path in files:
        try:
            content = await ctx.deps.vault.read_file(file_path)
            parsed = parse_note(content)

            if not parsed.frontmatter or not parsed.frontmatter.tags:
                continue

            note_tags = {t.lower() for t in parsed.frontmatter.tags}

            # Check tag match based on match_all setting
            if match_all:
                # All search tags must be present
                if not search_tags.issubset(note_tags):
                    continue
            else:
                # At least one search tag must be present
                if not search_tags.intersection(note_tags):
                    continue

            title = extract_title(content, file_path)

            result = SearchResult(
                path=file_path,
                title=title,
                tags=parsed.frontmatter.tags,
                created=parsed.frontmatter.created,
                modified=parsed.frontmatter.modified,
            )
            results.append(result)

        except Exception as e:
            logger.debug(
                "tag_search_file_error",
                extra={"path": file_path, "error": str(e), "trace_id": ctx.deps.trace_id},
            )
            continue

    return SearchResults(
        results=results[:limit],
        total=len(results),
        query=f"tags: {', '.join(tags)} ({'ALL' if match_all else 'ANY'})",
        operation="by_tag",
    )


async def _search_by_link(
    ctx: RunContext[ChatDependencies],
    note_path: str,
    direction: str,
    limit: int,
) -> SearchResults:
    """Search notes by wikilink relationships.

    Finds notes that link TO the target (backlinks) or notes that
    the target links TO (forward links).

    Args:
        ctx: Context with vault access
        note_path: Target note path
        direction: 'backlinks' or 'forward'
        limit: Maximum results to return

    Returns:
        SearchResults with linked notes
    """
    results: list[SearchResult] = []
    normalized_path = normalize_path(note_path)
    files = await ctx.deps.vault.list_files()

    if direction == "forward":
        # Find notes that this note links to
        try:
            content = await ctx.deps.vault.read_file(normalized_path)
            links = extract_wikilinks(content)

            for link_path in links[:limit]:
                try:
                    link_content = await ctx.deps.vault.read_file(link_path)
                    parsed = parse_note(link_content)
                    title = extract_title(link_content, link_path)

                    result = SearchResult(
                        path=link_path,
                        title=title,
                        tags=parsed.frontmatter.tags if parsed.frontmatter else [],
                        created=parsed.frontmatter.created if parsed.frontmatter else None,
                        modified=parsed.frontmatter.modified if parsed.frontmatter else None,
                    )
                    results.append(result)
                except Exception:
                    # Link target doesn't exist
                    results.append(
                        SearchResult(
                            path=link_path,
                            title=link_path.rsplit("/", 1)[-1].rsplit(".", 1)[0],
                            excerpt="(Note does not exist)",
                        )
                    )

            return SearchResults(
                results=results,
                total=len(links),
                query=f"forward links from: {normalized_path}",
                operation="by_link",
            )

        except Exception as e:
            logger.error(
                "forward_link_search_error",
                extra={"path": normalized_path, "error": str(e), "trace_id": ctx.deps.trace_id},
            )
            return SearchResults(
                results=[],
                total=0,
                query=f"forward links from: {normalized_path}",
                operation="by_link",
            )

    else:
        # Find notes that link to this note (backlinks)
        # Match both with and without .md extension
        target_name = normalized_path.rsplit(".md", 1)[0]

        for file_path in files:
            if file_path == normalized_path:
                continue

            try:
                content = await ctx.deps.vault.read_file(file_path)
                links = extract_wikilinks(content)

                # Check if any link points to target
                for link in links:
                    link_name = link.rsplit(".md", 1)[0]
                    if link_name == target_name or link == normalized_path:
                        parsed = parse_note(content)
                        title = extract_title(content, file_path)

                        result = SearchResult(
                            path=file_path,
                            title=title,
                            tags=parsed.frontmatter.tags if parsed.frontmatter else [],
                            created=parsed.frontmatter.created if parsed.frontmatter else None,
                            modified=parsed.frontmatter.modified if parsed.frontmatter else None,
                        )
                        results.append(result)
                        break

            except Exception as e:
                logger.debug(
                    "backlink_search_file_error",
                    extra={"path": file_path, "error": str(e), "trace_id": ctx.deps.trace_id},
                )
                continue

        return SearchResults(
            results=results[:limit],
            total=len(results),
            query=f"backlinks to: {normalized_path}",
            operation="by_link",
        )


async def _search_by_date(
    ctx: RunContext[ChatDependencies],
    start_date: str | None,
    end_date: str | None,
    date_field: str,
    limit: int,
) -> SearchResults:
    """Search notes by creation or modification date.

    Filters notes based on their frontmatter date fields within
    the specified range.

    Args:
        ctx: Context with vault access
        start_date: ISO date string (YYYY-MM-DD) for range start
        end_date: ISO date string (YYYY-MM-DD) for range end
        date_field: 'created' or 'modified'
        limit: Maximum results to return

    Returns:
        SearchResults with notes in date range
    """
    results: list[SearchResult] = []
    files = await ctx.deps.vault.list_files()

    # Parse date boundaries
    start_dt = None
    end_dt = None

    if start_date:
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
    if end_date:
        # End date is inclusive (end of day)
        end_dt = datetime.fromisoformat(f"{end_date}T23:59:59").replace(tzinfo=UTC)

    for file_path in files:
        try:
            content = await ctx.deps.vault.read_file(file_path)
            parsed = parse_note(content)

            if not parsed.frontmatter:
                continue

            # Get the appropriate date field
            if date_field == "modified":
                note_date = parsed.frontmatter.modified
            else:
                note_date = parsed.frontmatter.created

            if note_date is None:
                continue

            # Ensure timezone awareness
            if note_date.tzinfo is None:
                note_date = note_date.replace(tzinfo=UTC)

            # Check date range
            if start_dt and note_date < start_dt:
                continue
            if end_dt and note_date > end_dt:
                continue

            title = extract_title(content, file_path)

            result = SearchResult(
                path=file_path,
                title=title,
                tags=parsed.frontmatter.tags,
                created=parsed.frontmatter.created,
                modified=parsed.frontmatter.modified,
            )
            results.append(result)

        except Exception as e:
            logger.debug(
                "date_search_file_error",
                extra={"path": file_path, "error": str(e), "trace_id": ctx.deps.trace_id},
            )
            continue

    # Sort by date (most recent first)
    results.sort(
        key=lambda r: (getattr(r, date_field) or datetime.min.replace(tzinfo=UTC)),
        reverse=True,
    )

    date_range = []
    if start_date:
        date_range.append(f"from {start_date}")
    if end_date:
        date_range.append(f"to {end_date}")

    return SearchResults(
        results=results[:limit],
        total=len(results),
        query=f"{date_field} {' '.join(date_range)}",
        operation="by_date",
    )


async def _combined_search(
    ctx: RunContext[ChatDependencies],
    query: str | None,
    tags: list[str] | None,
    start_date: str | None,
    end_date: str | None,
    folder: str | None,
    limit: int,
) -> SearchResults:
    """Multi-criteria search with AND logic.

    Combines multiple search criteria (text, tags, dates, folder)
    with AND logic - results must match ALL specified criteria.

    Args:
        ctx: Context with vault access
        query: Optional text to search for
        tags: Optional tags to filter by
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        folder: Optional folder to limit search
        limit: Maximum results to return

    Returns:
        SearchResults with notes matching all criteria
    """
    results: list[SearchResult] = []
    files = await ctx.deps.vault.list_files(folder=folder or "")

    # Parse date boundaries if provided
    start_dt = None
    end_dt = None
    if start_date:
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
    if end_date:
        end_dt = datetime.fromisoformat(f"{end_date}T23:59:59").replace(tzinfo=UTC)

    # Prepare tag set if provided
    search_tags = {t.lower() for t in tags} if tags else None

    for file_path in files:
        try:
            content = await ctx.deps.vault.read_file(file_path)
            parsed = parse_note(content)

            # Text query filter
            if query:
                if query.lower() not in content.lower():
                    continue

            # Tag filter (any tag matches)
            if search_tags:
                if not parsed.frontmatter or not parsed.frontmatter.tags:
                    continue
                note_tags = {t.lower() for t in parsed.frontmatter.tags}
                if not search_tags.intersection(note_tags):
                    continue

            # Date filter
            if start_dt or end_dt:
                if not parsed.frontmatter:
                    continue
                note_date = parsed.frontmatter.created
                if note_date is None:
                    continue
                if note_date.tzinfo is None:
                    note_date = note_date.replace(tzinfo=UTC)
                if start_dt and note_date < start_dt:
                    continue
                if end_dt and note_date > end_dt:
                    continue

            title = extract_title(content, file_path)
            score = calculate_relevance(parsed.body, query, title) if query else 1.0
            excerpt = generate_excerpt(content, query) if query else ""

            result = SearchResult(
                path=file_path,
                title=title,
                excerpt=excerpt,
                score=score,
                tags=parsed.frontmatter.tags if parsed.frontmatter else [],
                created=parsed.frontmatter.created if parsed.frontmatter else None,
                modified=parsed.frontmatter.modified if parsed.frontmatter else None,
            )
            results.append(result)

        except Exception as e:
            logger.debug(
                "combined_search_file_error",
                extra={"path": file_path, "error": str(e), "trace_id": ctx.deps.trace_id},
            )
            continue

    # Sort by relevance if query provided, otherwise by date
    if query:
        results.sort(key=lambda r: r.score, reverse=True)
    else:
        results.sort(
            key=lambda r: r.created or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )

    # Build query description
    criteria = []
    if query:
        criteria.append(f"text: '{query}'")
    if tags:
        criteria.append(f"tags: {', '.join(tags)}")
    if start_date or end_date:
        date_range = []
        if start_date:
            date_range.append(f"from {start_date}")
        if end_date:
            date_range.append(f"to {end_date}")
        criteria.append(f"date {' '.join(date_range)}")
    if folder:
        criteria.append(f"folder: {folder}")

    return SearchResults(
        results=results[:limit],
        total=len(results),
        query=" AND ".join(criteria),
        operation="combined",
    )


# =============================================================================
# Main Tool Function
# =============================================================================


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

    This is the main tool function that the LLM calls to search notes.
    It supports five operation types: fulltext, by_tag, by_link,
    by_date, and combined.

    Args:
        ctx: Context with vault access and trace_id for logging
        operation: Search type - 'fulltext', 'by_tag', 'by_link', 'by_date', 'combined'
        query: Text to search for (fulltext/combined operations)
        tags: Tags to filter by (by_tag/combined operations)
        note_path: Target note for link search (by_link operation)
        direction: 'backlinks' or 'forward' (by_link operation)
        start_date: ISO date YYYY-MM-DD for range start (by_date/combined)
        end_date: ISO date YYYY-MM-DD for range end (by_date/combined)
        date_field: 'created' or 'modified' (by_date operation)
        folder: Limit search to folder (fulltext/combined operations)
        match_all: Require ALL tags vs ANY tag (by_tag operation)
        limit: Maximum results to return (default 10)

    Returns:
        Formatted search results or error message. Returns user-friendly
        error messages instead of raising exceptions.

    Examples:
        fulltext: vault_search(ctx, "fulltext", query="API design")
        by_tag:   vault_search(ctx, "by_tag", tags=["project"], match_all=False)
        by_link:  vault_search(ctx, "by_link", note_path="Projects/API", direction="backlinks")
        by_date:  vault_search(ctx, "by_date", start_date="2025-01-01", date_field="created")
        combined: vault_search(ctx, "combined", query="meeting", tags=["work"], folder="Work")
    """
    logger.info(
        "vault_search_called",
        extra={
            "operation": operation,
            "query": query,
            "tags": tags,
            "note_path": note_path,
            "trace_id": ctx.deps.trace_id,
        },
    )

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
                return "Error: 'start_date' or 'end_date' required for date search"
            results = await _search_by_date(ctx, start_date, end_date, date_field, limit)

        elif operation == "combined":
            if not any([query, tags, start_date, end_date, folder]):
                return "Error: at least one search criteria required for combined search"
            results = await _combined_search(ctx, query, tags, start_date, end_date, folder, limit)

        else:
            return (
                f"Unknown operation: {operation}. "
                f"Valid operations: fulltext, by_tag, by_link, by_date, combined"
            )

        logger.info(
            "vault_search_completed",
            extra={
                "operation": operation,
                "result_count": len(results.results),
                "total_count": results.total,
                "trace_id": ctx.deps.trace_id,
            },
        )

        return format_results(results)

    except Exception as e:
        logger.error(
            "vault_search_failed",
            extra={
                "operation": operation,
                "error": str(e),
                "trace_id": ctx.deps.trace_id,
            },
            exc_info=True,
        )
        return f"Error performing {operation} search: {str(e)}"
