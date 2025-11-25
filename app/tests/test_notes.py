"""Tests for notes feature.

This module contains comprehensive tests for the note_operations tool
and its helper functions. Tests use pytest fixtures with temporary
directories to ensure isolation.
"""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.dependencies import ChatDependencies, VaultClient
from app.notes.models import NoteFrontmatter
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

        assert "2025-01-01" in result  # Created date preserved
        assert "original" in result  # Tags preserved
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
        assert (datetime.now(UTC) - fm.created).seconds < 60

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
    async def test_create_note_already_exists(self, mock_ctx: Mock, tmp_path: Path) -> None:
        """Test that creating existing note returns error message."""
        (tmp_path / "existing.md").write_text("# Existing")

        result = await note_operations(
            mock_ctx, operation="create", path="existing", content="New content"
        )

        assert "already exists" in result
        # Original content unchanged
        assert (tmp_path / "existing.md").read_text() == "# Existing"

    @pytest.mark.asyncio
    async def test_create_note_with_nested_path(self, mock_ctx: Mock, tmp_path: Path) -> None:
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
    async def test_read_note_with_frontmatter(self, mock_ctx: Mock, tmp_path: Path) -> None:
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
    async def test_delete_without_confirm(self, mock_ctx: Mock, tmp_path: Path) -> None:
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

        result = await note_operations(mock_ctx, operation="delete", path="test", confirm=True)

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
