"""Pydantic models for tag management."""

from pydantic import BaseModel, Field


class TagInfo(BaseModel):
    """Tag with usage count.

    Represents a tag found in the vault along with statistics
    about its usage across notes.

    Attributes:
        name: Tag name without # prefix
        count: Number of notes using this tag
        notes: List of note paths that have this tag
    """

    name: str = Field(..., description="Tag name without #")
    count: int = Field(default=0, ge=0)
    notes: list[str] = Field(default_factory=list)


class TagSuggestion(BaseModel):
    """Suggested tag with confidence.

    Represents an AI-suggested tag for a note based on content
    analysis and existing vault taxonomy.

    Attributes:
        tag: The suggested tag name
        confidence: Confidence score between 0.0 and 1.0
        reason: Explanation for why this tag was suggested
        existing: Whether this tag already exists in the vault taxonomy
    """

    tag: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""
    existing: bool = False  # Exists in vault taxonomy


class ConnectionResult(BaseModel):
    """Result of connecting notes via wikilinks.

    Represents the outcome of the connect operation which
    adds [[wikilinks]] to related notes.

    Attributes:
        path: Path to the note that was updated
        connections_added: Number of new wikilinks added
        connected_notes: List of notes that were linked
        reasons: Explanations for why each connection was made
    """

    path: str
    connections_added: int = 0
    connected_notes: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
