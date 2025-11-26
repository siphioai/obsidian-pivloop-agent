"""Pydantic models for search results.

This module defines the data structures for search results returned
by the vault_search tool. Models follow the same pattern as notes/models.py.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single search result.

    Represents one note that matched the search criteria. Contains
    both identifying information and match context.

    Attributes:
        path: Relative path to note in vault (e.g., 'Projects/API.md')
        title: Note title (from H1 heading or filename)
        excerpt: Context snippet around the match
        score: Relevance score from 0.0 to 1.0
        tags: List of tags from frontmatter
        created: Note creation timestamp (if available)
        modified: Note modification timestamp (if available)
    """

    path: str = Field(..., description="Relative path to note")
    title: str = Field(..., description="Note title")
    excerpt: str = Field(default="", description="Match context snippet")
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Relevance score")
    tags: list[str] = Field(default_factory=list, description="Tags from frontmatter")
    created: datetime | None = Field(default=None, description="Creation timestamp")
    modified: datetime | None = Field(default=None, description="Modification timestamp")


class SearchResults(BaseModel):
    """Collection of search results.

    Contains the matched results along with metadata about the search
    operation that produced them.

    Attributes:
        results: List of matching SearchResult objects
        total: Total number of results found
        query: The original search query or criteria
        operation: The search operation type that was performed
    """

    results: list[SearchResult] = Field(default_factory=list, description="List of results")
    total: int = Field(default=0, description="Total results found")
    query: str = Field(default="", description="Original search query")
    operation: str = Field(default="", description="Search operation type")
