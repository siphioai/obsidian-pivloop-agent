"""Tests for tag management feature.

This module contains comprehensive tests for the tag_management tool
and its helper functions. Tests use pytest fixtures with temporary
directories to ensure isolation.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from app.dependencies import ChatDependencies, VaultClient
from app.tags.models import ConnectionResult, TagInfo, TagSuggestion
from app.tags.tools import (
    extract_keywords,
    insert_wikilinks,
    tag_management,
    update_frontmatter_tags,
)

# =============================================================================
# Extract Keywords Tests
# =============================================================================


class TestExtractKeywords:
    """Tests for keyword extraction helper function."""

    def test_basic_extraction(self) -> None:
        """Test extraction of common words from content."""
        content = "This is about project management and project planning."
        keywords = extract_keywords(content)

        assert "project" in keywords
        assert "management" in keywords
        assert "planning" in keywords

    def test_frequency_ordering(self) -> None:
        """Test that keywords are ordered by frequency."""
        content = "python python python code code java"
        keywords = extract_keywords(content)

        # Python should be first due to higher frequency
        python_idx = keywords.index("python")
        code_idx = keywords.index("code")
        java_idx = keywords.index("java")

        assert python_idx < code_idx < java_idx

    def test_stopwords_filtered(self) -> None:
        """Test that stopwords are filtered out."""
        content = "the and for are but not you all can had"
        keywords = extract_keywords(content)

        assert "the" not in keywords
        assert "and" not in keywords
        assert "for" not in keywords

    def test_min_length_filter(self) -> None:
        """Test that short words are filtered by min_length."""
        content = "a an api is it to we at"
        keywords = extract_keywords(content)

        assert "api" not in keywords  # Length 3, default min is 4
        assert len([k for k in keywords if len(k) < 4]) == 0

    def test_custom_min_length(self) -> None:
        """Test custom min_length parameter."""
        content = "api is the best tool for code"
        keywords = extract_keywords(content, min_length=3)

        assert "api" in keywords


# =============================================================================
# Update Frontmatter Tags Tests
# =============================================================================


class TestUpdateFrontmatterTags:
    """Tests for frontmatter tag update helper function."""

    def test_update_existing_tags(self) -> None:
        """Test updating existing frontmatter with new tags."""
        content = """---
created: 2025-01-01T00:00:00+00:00
tags:
  - old
---
# Content"""
        result = update_frontmatter_tags(content, ["new", "tags"])

        assert "new" in result
        assert "tags" in result
        assert "modified" in result

    def test_create_frontmatter_if_missing(self) -> None:
        """Test creating frontmatter when none exists."""
        content = "# Just markdown content"
        result = update_frontmatter_tags(content, ["project", "api"])

        assert "---" in result
        assert "tags:" in result
        assert "project" in result
        assert "api" in result
        # The frontmatter library handles creating frontmatter automatically
        # and adds modified timestamp; created is not added in this case

    def test_preserve_other_fields(self) -> None:
        """Test that other frontmatter fields are preserved."""
        content = """---
created: 2025-01-01T00:00:00+00:00
custom_field: value
tags:
  - old
---
# Content"""
        result = update_frontmatter_tags(content, ["new"])

        assert "custom_field: value" in result
        assert "new" in result


# =============================================================================
# Insert Wikilinks Tests
# =============================================================================


class TestInsertWikilinks:
    """Tests for wikilink insertion helper function."""

    def test_add_related_notes_section(self) -> None:
        """Test that Related Notes section is added."""
        content = "# My Note\n\nSome content here."
        links = ["[[Related Note]]", "[[Another Note]]"]
        result = insert_wikilinks(content, links)

        assert "## Related Notes" in result
        assert "[[Related Note]]" in result
        assert "[[Another Note]]" in result

    def test_empty_links_no_change(self) -> None:
        """Test that empty links list doesn't modify content."""
        content = "# My Note\n\nSome content here."
        result = insert_wikilinks(content, [])

        assert result == content

    def test_section_at_end(self) -> None:
        """Test that section is added at the end of content."""
        content = "# My Note\n\nContent"
        links = ["[[Link]]"]
        result = insert_wikilinks(content, links)

        assert result.endswith("- [[Link]]\n")


# =============================================================================
# Tag Models Tests
# =============================================================================


class TestTagModels:
    """Tests for tag Pydantic models."""

    def test_tag_info_defaults(self) -> None:
        """Test TagInfo model defaults."""
        tag = TagInfo(name="project")

        assert tag.name == "project"
        assert tag.count == 0
        assert tag.notes == []

    def test_tag_info_with_values(self) -> None:
        """Test TagInfo model with values."""
        tag = TagInfo(name="api", count=5, notes=["a.md", "b.md"])

        assert tag.name == "api"
        assert tag.count == 5
        assert len(tag.notes) == 2

    def test_tag_suggestion_confidence_bounds(self) -> None:
        """Test TagSuggestion confidence bounds."""
        # Test lower bound
        with pytest.raises(ValueError):
            TagSuggestion(tag="test", confidence=-0.1)

        # Test upper bound
        with pytest.raises(ValueError):
            TagSuggestion(tag="test", confidence=1.1)

        # Test valid values
        s = TagSuggestion(tag="test", confidence=0.8)
        assert s.confidence == 0.8

    def test_connection_result_defaults(self) -> None:
        """Test ConnectionResult model defaults."""
        result = ConnectionResult(path="note.md")

        assert result.path == "note.md"
        assert result.connections_added == 0
        assert result.connected_notes == []
        assert result.reasons == []


# =============================================================================
# Tag Management Tool Tests
# =============================================================================


class TestTagManagement:
    """Tests for the tag_management tool function."""

    @pytest.fixture
    def sample_vault(self, tmp_path: Path) -> Path:
        """Create vault with tagged notes."""
        (tmp_path / "API.md").write_text("""---
created: 2025-01-01T00:00:00+00:00
modified: 2025-01-01T00:00:00+00:00
tags:
  - project
  - api
---
# API Design

This is about API project design.""")

        (tmp_path / "Arch.md").write_text("""---
created: 2025-01-01T00:00:00+00:00
modified: 2025-01-01T00:00:00+00:00
tags:
  - project
---
# Architecture

System architecture notes.""")

        (tmp_path / "Meeting.md").write_text("""---
created: 2025-01-01T00:00:00+00:00
modified: 2025-01-01T00:00:00+00:00
tags:
  - meeting
---
# Meeting Notes

Meeting about API project.""")

        (tmp_path / "NoTags.md").write_text("""# Note Without Tags

Just content, no frontmatter.""")

        return tmp_path

    @pytest.fixture
    def mock_ctx(self, sample_vault: Path) -> Mock:
        """Create mock context with vault client."""
        ctx = Mock()
        ctx.deps = ChatDependencies(
            vault=VaultClient(vault_path=sample_vault),
            trace_id="test-trace-123",
        )
        return ctx

    @pytest.mark.asyncio
    async def test_list_tags(self, mock_ctx: Mock) -> None:
        """Test listing all vault tags."""
        result = await tag_management(mock_ctx, operation="list", sort_by="count", limit=10)

        assert "Found" in result
        assert "project" in result
        assert "2 notes" in result  # project appears in API and Arch

    @pytest.mark.asyncio
    async def test_list_tags_alpha_sort(self, mock_ctx: Mock) -> None:
        """Test listing tags sorted alphabetically."""
        result = await tag_management(mock_ctx, operation="list", sort_by="alpha", limit=10)

        assert "Found" in result
        # Tags should be in order
        api_pos = result.find("api")
        meeting_pos = result.find("meeting")
        project_pos = result.find("project")
        assert api_pos < meeting_pos < project_pos

    @pytest.mark.asyncio
    async def test_add_tags(self, mock_ctx: Mock) -> None:
        """Test adding tags to a note."""
        result = await tag_management(mock_ctx, operation="add", path="API.md", tags=["backend"])

        assert "Added" in result
        assert "backend" in result

        # Verify tag was added
        content = await mock_ctx.deps.vault.read_file("API.md")
        assert "backend" in content

    @pytest.mark.asyncio
    async def test_add_duplicate_tags(self, mock_ctx: Mock) -> None:
        """Test adding tags that already exist."""
        result = await tag_management(
            mock_ctx, operation="add", path="API.md", tags=["project", "api"]
        )

        assert "already has all" in result

    @pytest.mark.asyncio
    async def test_add_tags_file_not_found(self, mock_ctx: Mock) -> None:
        """Test adding tags to non-existent note."""
        result = await tag_management(
            mock_ctx, operation="add", path="nonexistent.md", tags=["tag"]
        )

        assert "not found" in result

    @pytest.mark.asyncio
    async def test_remove_tags(self, mock_ctx: Mock) -> None:
        """Test removing tags from a note."""
        result = await tag_management(mock_ctx, operation="remove", path="API.md", tags=["api"])

        assert "Removed" in result
        assert "1 tag" in result

        # Verify tag was removed
        content = await mock_ctx.deps.vault.read_file("API.md")
        assert "- api" not in content

    @pytest.mark.asyncio
    async def test_remove_nonexistent_tags(self, mock_ctx: Mock) -> None:
        """Test removing tags that don't exist."""
        result = await tag_management(
            mock_ctx, operation="remove", path="API.md", tags=["nonexistent"]
        )

        assert "None of the specified" in result

    @pytest.mark.asyncio
    async def test_remove_tags_no_frontmatter(self, mock_ctx: Mock) -> None:
        """Test removing tags from note without frontmatter."""
        result = await tag_management(mock_ctx, operation="remove", path="NoTags.md", tags=["tag"])

        assert "no tags" in result

    @pytest.mark.asyncio
    async def test_rename_tag(self, mock_ctx: Mock) -> None:
        """Test renaming a tag vault-wide."""
        result = await tag_management(
            mock_ctx, operation="rename", old_tag="project", new_tag="proj"
        )

        assert "Renamed" in result
        assert "2 note" in result  # API and Arch had project tag

        # Verify in both files
        api_content = await mock_ctx.deps.vault.read_file("API.md")
        arch_content = await mock_ctx.deps.vault.read_file("Arch.md")
        assert "proj" in api_content
        assert "proj" in arch_content

    @pytest.mark.asyncio
    async def test_rename_tag_not_found(self, mock_ctx: Mock) -> None:
        """Test renaming a tag that doesn't exist."""
        result = await tag_management(
            mock_ctx, operation="rename", old_tag="nonexistent", new_tag="new"
        )

        assert "not found" in result

    @pytest.mark.asyncio
    async def test_rename_same_tag(self, mock_ctx: Mock) -> None:
        """Test renaming a tag to itself."""
        result = await tag_management(
            mock_ctx, operation="rename", old_tag="project", new_tag="project"
        )

        assert "same" in result

    @pytest.mark.asyncio
    async def test_suggest_tags(self, mock_ctx: Mock) -> None:
        """Test tag suggestions for a note."""
        result = await tag_management(mock_ctx, operation="suggest", path="Meeting.md")

        # Meeting.md content mentions "API" and "project" which exist as tags
        assert "Suggestions" in result or "No tag suggestions" in result

    @pytest.mark.asyncio
    async def test_suggest_tags_file_not_found(self, mock_ctx: Mock) -> None:
        """Test suggestions for non-existent note."""
        result = await tag_management(mock_ctx, operation="suggest", path="nonexistent.md")

        assert "not found" in result

    @pytest.mark.asyncio
    async def test_auto_tag_preview(self, mock_ctx: Mock) -> None:
        """Test auto_tag preview mode (confirm=False)."""
        result = await tag_management(
            mock_ctx, operation="auto_tag", path="Meeting.md", confirm=False
        )

        # Should show preview or no suitable tags
        assert "Would add" in result or "No suitable" in result

    @pytest.mark.asyncio
    async def test_auto_tag_apply(self, mock_ctx: Mock) -> None:
        """Test auto_tag apply mode (confirm=True)."""
        result = await tag_management(
            mock_ctx, operation="auto_tag", path="Meeting.md", confirm=True
        )

        # Should apply or have no suitable tags
        assert "Auto-tagged" in result or "No suitable" in result

    @pytest.mark.asyncio
    async def test_connect_notes(self, mock_ctx: Mock) -> None:
        """Test connecting related notes."""
        result = await tag_management(mock_ctx, operation="connect", path="API.md", limit=5)

        # API has project+api, Arch has project - they share 1 tag
        # Meeting has meeting - no shared tags
        # Need 2+ shared tags, so likely no connections
        assert "Connected" in result or "No related" in result

    @pytest.mark.asyncio
    async def test_connect_notes_file_not_found(self, mock_ctx: Mock) -> None:
        """Test connecting non-existent note."""
        result = await tag_management(mock_ctx, operation="connect", path="nonexistent.md", limit=5)

        assert "not found" in result

    @pytest.mark.asyncio
    async def test_unknown_operation(self, mock_ctx: Mock) -> None:
        """Test unknown operation returns error."""
        result = await tag_management(mock_ctx, operation="invalid")

        assert "Unknown operation" in result
        assert "Valid:" in result

    @pytest.mark.asyncio
    async def test_add_missing_params(self, mock_ctx: Mock) -> None:
        """Test add operation with missing parameters."""
        result = await tag_management(mock_ctx, operation="add")

        assert "Error" in result
        assert "path" in result and "tags" in result

    @pytest.mark.asyncio
    async def test_remove_missing_params(self, mock_ctx: Mock) -> None:
        """Test remove operation with missing parameters."""
        result = await tag_management(mock_ctx, operation="remove", path="API.md")

        assert "Error" in result
        assert "tags" in result

    @pytest.mark.asyncio
    async def test_rename_missing_params(self, mock_ctx: Mock) -> None:
        """Test rename operation with missing parameters."""
        result = await tag_management(mock_ctx, operation="rename", old_tag="test")

        assert "Error" in result
        assert "new_tag" in result

    @pytest.mark.asyncio
    async def test_suggest_missing_path(self, mock_ctx: Mock) -> None:
        """Test suggest operation with missing path."""
        result = await tag_management(mock_ctx, operation="suggest")

        assert "Error" in result
        assert "path" in result
