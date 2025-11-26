"""Tests for search feature.

This module contains comprehensive tests for the vault_search tool
and its helper functions. Tests use pytest fixtures with temporary
directories to ensure isolation.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from app.dependencies import ChatDependencies, VaultClient
from app.search.models import SearchResult, SearchResults
from app.search.tools import (
    calculate_relevance,
    extract_title,
    extract_wikilinks,
    format_results,
    generate_excerpt,
    vault_search,
)

# =============================================================================
# Wikilink Extraction Tests
# =============================================================================


class TestExtractWikilinks:
    """Tests for wikilink extraction helper function."""

    def test_simple_links(self) -> None:
        """Test extraction of simple [[link]] patterns."""
        content = "See [[API Design]] and [[Projects/Plan]]"
        links = extract_wikilinks(content)

        assert "API Design.md" in links
        assert "Projects/Plan.md" in links
        assert len(links) == 2

    def test_aliased_links(self) -> None:
        """Test extraction of [[target|alias]] patterns."""
        content = "Check [[API Design|the API]] and [[Projects/Plan|plan doc]]"
        links = extract_wikilinks(content)

        assert "API Design.md" in links
        assert "Projects/Plan.md" in links
        assert len(links) == 2

    def test_no_links(self) -> None:
        """Test content without any wikilinks."""
        content = "This is plain text without any links."
        links = extract_wikilinks(content)

        assert links == []

    def test_duplicate_links(self) -> None:
        """Test that duplicate links are deduplicated."""
        content = "See [[Note]] and again [[Note]] and [[Note|alias]]"
        links = extract_wikilinks(content)

        assert len(links) == 1
        assert "Note.md" in links

    def test_mixed_links(self) -> None:
        """Test mixture of simple and aliased links."""
        content = """
        Start with [[Simple]]
        Then [[Path/To/Note|aliased]]
        And [[Another|with alias]]
        Plus [[Simple]] again
        """
        links = extract_wikilinks(content)

        assert "Simple.md" in links
        assert "Path/To/Note.md" in links
        assert "Another.md" in links
        assert len(links) == 3


# =============================================================================
# Title Extraction Tests
# =============================================================================


class TestExtractTitle:
    """Tests for title extraction helper function."""

    def test_h1_heading(self) -> None:
        """Test extraction from H1 heading."""
        content = "# My Note Title\n\nContent here"
        title = extract_title(content, "notes/my-note.md")

        assert title == "My Note Title"

    def test_h1_with_frontmatter(self) -> None:
        """Test extraction from H1 after frontmatter."""
        content = """---
created: 2025-01-01
---
# Real Title

Content"""
        title = extract_title(content, "notes/file.md")

        assert title == "Real Title"

    def test_filename_fallback(self) -> None:
        """Test fallback to filename when no H1."""
        content = "No heading here, just content."
        title = extract_title(content, "notes/my-note.md")

        assert title == "my-note"

    def test_filename_with_nested_path(self) -> None:
        """Test filename extraction from nested path."""
        content = "No heading"
        title = extract_title(content, "deep/nested/folder/note-name.md")

        assert title == "note-name"

    def test_h2_not_used(self) -> None:
        """Test that H2 is not mistaken for title."""
        content = "## This is H2\n\nNot a title"
        title = extract_title(content, "test.md")

        assert title == "test"


# =============================================================================
# Excerpt Generation Tests
# =============================================================================


class TestGenerateExcerpt:
    """Tests for excerpt generation helper function."""

    def test_centers_on_match(self) -> None:
        """Test that excerpt centers on query match."""
        content = "A" * 100 + "TARGET" + "B" * 100
        excerpt = generate_excerpt(content, "TARGET", max_length=50)

        assert "TARGET" in excerpt
        assert "..." in excerpt

    def test_strips_frontmatter(self) -> None:
        """Test that frontmatter is stripped from excerpt."""
        content = """---
created: 2025-01-01
tags: [test]
---
This is the body with KEYWORD here."""
        excerpt = generate_excerpt(content, "KEYWORD", max_length=100)

        assert "KEYWORD" in excerpt
        assert "---" not in excerpt
        assert "created:" not in excerpt

    def test_case_insensitive(self) -> None:
        """Test case-insensitive query matching."""
        content = "The API design document"
        excerpt = generate_excerpt(content, "api", max_length=50)

        assert "API" in excerpt

    def test_query_not_found(self) -> None:
        """Test behavior when query not in content."""
        content = "Some random content here"
        excerpt = generate_excerpt(content, "MISSING", max_length=50)

        # Should return start of content
        assert excerpt.startswith("Some")

    def test_short_content(self) -> None:
        """Test with content shorter than max_length."""
        content = "Short note"
        excerpt = generate_excerpt(content, "Short", max_length=100)

        assert excerpt == "Short note"
        assert "..." not in excerpt


# =============================================================================
# Relevance Calculation Tests
# =============================================================================


class TestCalculateRelevance:
    """Tests for relevance scoring helper function."""

    def test_title_match_high(self) -> None:
        """Test that title matches get high scores."""
        score = calculate_relevance("API documentation", "API", "API Design Guide")

        assert score >= 0.5  # Title match bonus

    def test_exact_title_match(self) -> None:
        """Test that exact title match gets highest score."""
        score = calculate_relevance("content", "test", "test")

        assert score >= 0.8  # Title match + exact match bonus

    def test_content_only(self) -> None:
        """Test scoring for content match without title match."""
        score = calculate_relevance("The API is documented here", "API", "Notes")

        assert 0.0 < score < 0.5

    def test_score_capped(self) -> None:
        """Test that score never exceeds 1.0."""
        # Many matches should still cap at 1.0
        content = "API " * 100
        score = calculate_relevance(content, "API", "API API API")

        assert score <= 1.0

    def test_no_match(self) -> None:
        """Test scoring when query doesn't match."""
        score = calculate_relevance("Unrelated content", "MISSING", "Other Title")

        assert score == 0.0

    def test_word_boundary_bonus(self) -> None:
        """Test word boundary matching bonus."""
        # "API" as whole word vs "APIs" containing "API"
        score_whole = calculate_relevance("The API is here", "API", "Note")
        score_partial = calculate_relevance("The APIs are here", "API", "Note")

        assert score_whole >= score_partial


# =============================================================================
# Results Formatting Tests
# =============================================================================


class TestFormatResults:
    """Tests for results formatting helper function."""

    def test_empty_results(self) -> None:
        """Test formatting of empty results."""
        results = SearchResults(
            results=[],
            total=0,
            query="test",
            operation="fulltext",
        )
        output = format_results(results)

        assert "No results found" in output
        assert "fulltext" in output

    def test_single_result(self) -> None:
        """Test formatting of single result."""
        results = SearchResults(
            results=[
                SearchResult(
                    path="notes/test.md",
                    title="Test Note",
                    excerpt="This is a test",
                    score=0.8,
                    tags=["project"],
                )
            ],
            total=1,
            query="test",
            operation="fulltext",
        )
        output = format_results(results)

        assert "1 result" in output
        assert "Test Note" in output
        assert "notes/test.md" in output
        assert "project" in output
        assert "80%" in output

    def test_multiple_results(self) -> None:
        """Test formatting of multiple results."""
        results = SearchResults(
            results=[
                SearchResult(path="a.md", title="First", score=0.9),
                SearchResult(path="b.md", title="Second", score=0.7),
            ],
            total=2,
            query="query",
            operation="fulltext",
        )
        output = format_results(results)

        assert "2 result" in output
        assert "1." in output
        assert "2." in output
        assert "First" in output
        assert "Second" in output


# =============================================================================
# Vault Search Integration Tests
# =============================================================================


class TestVaultSearch:
    """Integration tests for vault_search tool function."""

    @pytest.fixture
    def sample_vault(self, tmp_path: Path) -> Path:
        """Create a vault with test notes for searching."""
        # Note 1: API Design with tags
        (tmp_path / "Projects").mkdir()
        (tmp_path / "Projects" / "API.md").write_text(
            """---
created: 2025-01-15T10:00:00+00:00
modified: 2025-01-15T10:00:00+00:00
tags:
  - project
  - api
---
# API Design

This document describes the API design.
Links to [[Architecture]] and [[Database]].
"""
        )

        # Note 2: Architecture with backlink to API
        (tmp_path / "Architecture.md").write_text(
            """---
created: 2025-01-10T10:00:00+00:00
modified: 2025-01-10T10:00:00+00:00
tags:
  - project
---
# Architecture Overview

See [[Projects/API]] for API details.
"""
        )

        # Note 3: Database (no backlink)
        (tmp_path / "Database.md").write_text(
            """---
created: 2025-01-20T10:00:00+00:00
modified: 2025-01-20T10:00:00+00:00
tags:
  - database
---
# Database Schema

Schema documentation here.
"""
        )

        # Note 4: Meeting note with work tag
        (tmp_path / "Work").mkdir()
        (tmp_path / "Work" / "Meeting.md").write_text(
            """---
created: 2025-01-18T10:00:00+00:00
modified: 2025-01-18T10:00:00+00:00
tags:
  - meeting
  - work
---
# Team Meeting

Discussion about the API project.
"""
        )

        return tmp_path

    @pytest.fixture
    def mock_vault(self, sample_vault: Path) -> VaultClient:
        """Create VaultClient with sample vault."""
        return VaultClient(vault_path=sample_vault)

    @pytest.fixture
    def mock_ctx(self, mock_vault: VaultClient) -> Mock:
        """Create mock RunContext with ChatDependencies."""
        ctx = Mock()
        ctx.deps = ChatDependencies(vault=mock_vault, trace_id="test-search-123")
        return ctx

    # -------------------------------------------------------------------------
    # Fulltext Search Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fulltext_finds_matches(self, mock_ctx: Mock) -> None:
        """Test fulltext search finds matching notes."""
        result = await vault_search(mock_ctx, operation="fulltext", query="API")

        assert "API Design" in result
        assert "Projects/API.md" in result
        # Should also find Meeting note that mentions API
        assert "Meeting" in result

    @pytest.mark.asyncio
    async def test_fulltext_requires_query(self, mock_ctx: Mock) -> None:
        """Test that fulltext search requires query parameter."""
        result = await vault_search(mock_ctx, operation="fulltext")

        assert "Error" in result
        assert "query" in result.lower()

    @pytest.mark.asyncio
    async def test_fulltext_folder_filter(self, mock_ctx: Mock) -> None:
        """Test fulltext search limited to folder."""
        result = await vault_search(
            mock_ctx, operation="fulltext", query="API", folder="Projects"
        )

        assert "API Design" in result
        # Meeting note is in Work folder, should not appear
        assert "Meeting" not in result

    @pytest.mark.asyncio
    async def test_fulltext_no_matches(self, mock_ctx: Mock) -> None:
        """Test fulltext search with no matches."""
        result = await vault_search(
            mock_ctx, operation="fulltext", query="NONEXISTENT_TERM"
        )

        assert "No results found" in result

    # -------------------------------------------------------------------------
    # Tag Search Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_tag_search_finds_notes(self, mock_ctx: Mock) -> None:
        """Test tag search finds tagged notes."""
        result = await vault_search(mock_ctx, operation="by_tag", tags=["project"])

        assert "API Design" in result
        assert "Architecture" in result
        # Database has different tag
        assert "Database Schema" not in result

    @pytest.mark.asyncio
    async def test_tag_search_match_all(self, mock_ctx: Mock) -> None:
        """Test tag search with match_all=True."""
        result = await vault_search(
            mock_ctx, operation="by_tag", tags=["project", "api"], match_all=True
        )

        # Only API has both tags
        assert "API Design" in result
        assert "Architecture" not in result

    @pytest.mark.asyncio
    async def test_tag_search_match_any(self, mock_ctx: Mock) -> None:
        """Test tag search with match_all=False (any tag)."""
        result = await vault_search(
            mock_ctx, operation="by_tag", tags=["project", "database"], match_all=False
        )

        assert "API Design" in result
        assert "Architecture" in result
        assert "Database Schema" in result

    @pytest.mark.asyncio
    async def test_tag_search_requires_tags(self, mock_ctx: Mock) -> None:
        """Test that tag search requires tags parameter."""
        result = await vault_search(mock_ctx, operation="by_tag")

        assert "Error" in result
        assert "tags" in result.lower()

    # -------------------------------------------------------------------------
    # Link Search Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_link_search_backlinks(self, mock_ctx: Mock) -> None:
        """Test finding backlinks to a note."""
        result = await vault_search(
            mock_ctx,
            operation="by_link",
            note_path="Projects/API",
            direction="backlinks",
        )

        # Architecture links to Projects/API
        assert "Architecture" in result

    @pytest.mark.asyncio
    async def test_link_search_forward(self, mock_ctx: Mock) -> None:
        """Test finding forward links from a note."""
        result = await vault_search(
            mock_ctx,
            operation="by_link",
            note_path="Projects/API",
            direction="forward",
        )

        # API links to Architecture and Database
        assert "Architecture" in result
        assert "Database" in result

    @pytest.mark.asyncio
    async def test_link_search_requires_path(self, mock_ctx: Mock) -> None:
        """Test that link search requires note_path parameter."""
        result = await vault_search(mock_ctx, operation="by_link")

        assert "Error" in result
        assert "note_path" in result.lower()

    # -------------------------------------------------------------------------
    # Date Search Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_date_search_range(self, mock_ctx: Mock) -> None:
        """Test date search with date range."""
        result = await vault_search(
            mock_ctx,
            operation="by_date",
            start_date="2025-01-15",
            end_date="2025-01-20",
        )

        # API (Jan 15), Meeting (Jan 18), Database (Jan 20) in range
        # Architecture (Jan 10) outside range
        assert "API Design" in result
        assert "Database Schema" in result
        assert "Architecture" not in result

    @pytest.mark.asyncio
    async def test_date_search_start_only(self, mock_ctx: Mock) -> None:
        """Test date search with only start date."""
        result = await vault_search(
            mock_ctx, operation="by_date", start_date="2025-01-18"
        )

        # Meeting (Jan 18) and Database (Jan 20) from start
        assert "Meeting" in result
        assert "Database" in result

    @pytest.mark.asyncio
    async def test_date_search_requires_date(self, mock_ctx: Mock) -> None:
        """Test that date search requires at least one date."""
        result = await vault_search(mock_ctx, operation="by_date")

        assert "Error" in result
        assert "start_date" in result.lower() or "end_date" in result.lower()

    # -------------------------------------------------------------------------
    # Combined Search Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_combined_search(self, mock_ctx: Mock) -> None:
        """Test combined search with multiple criteria."""
        result = await vault_search(
            mock_ctx,
            operation="combined",
            query="design",
            tags=["project"],
        )

        # Must match both: has "design" AND has "project" tag
        assert "API Design" in result
        # Should only have 1 result (API Design matches both criteria)
        assert "1 result" in result
        # Meeting shouldn't match (has neither design text nor project tag)
        assert "Team Meeting" not in result

    @pytest.mark.asyncio
    async def test_combined_search_with_folder(self, mock_ctx: Mock) -> None:
        """Test combined search limited to folder."""
        result = await vault_search(
            mock_ctx,
            operation="combined",
            tags=["project"],
            folder="Projects",
        )

        # Only API is in Projects folder with project tag
        assert "API Design" in result
        assert "Architecture" not in result

    @pytest.mark.asyncio
    async def test_combined_requires_criteria(self, mock_ctx: Mock) -> None:
        """Test that combined search requires at least one criterion."""
        result = await vault_search(mock_ctx, operation="combined")

        assert "Error" in result
        assert "criteria" in result.lower()

    # -------------------------------------------------------------------------
    # Error Handling Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_invalid_operation(self, mock_ctx: Mock) -> None:
        """Test that invalid operation returns helpful error."""
        result = await vault_search(mock_ctx, operation="invalid")

        assert "Unknown operation" in result
        assert "invalid" in result
        assert "fulltext" in result
        assert "by_tag" in result
        assert "by_link" in result
        assert "by_date" in result
        assert "combined" in result


# =============================================================================
# Model Tests
# =============================================================================


class TestSearchModels:
    """Tests for search Pydantic models."""

    def test_search_result_defaults(self) -> None:
        """Test SearchResult default values."""
        result = SearchResult(path="test.md", title="Test")

        assert result.excerpt == ""
        assert result.score == 1.0
        assert result.tags == []
        assert result.created is None
        assert result.modified is None

    def test_search_result_score_bounds(self) -> None:
        """Test that score is bounded between 0 and 1."""
        result = SearchResult(path="test.md", title="Test", score=0.5)
        assert 0.0 <= result.score <= 1.0

        # Pydantic should enforce bounds
        with pytest.raises(ValueError):
            SearchResult(path="test.md", title="Test", score=1.5)

        with pytest.raises(ValueError):
            SearchResult(path="test.md", title="Test", score=-0.1)

    def test_search_results_defaults(self) -> None:
        """Test SearchResults default values."""
        results = SearchResults()

        assert results.results == []
        assert results.total == 0
        assert results.query == ""
        assert results.operation == ""
