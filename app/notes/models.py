"""Pydantic models for note operations."""

from datetime import UTC, datetime
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
        default_factory=lambda: datetime.now(UTC),
        description="When the note was created",
    )
    modified: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the note was last modified",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="List of tags for the note",
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
