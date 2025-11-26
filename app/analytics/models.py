"""Pydantic models for analytics results.

This module defines the data structures for vault analytics results
returned by the vault_analytics tool. Models follow project patterns.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class FolderStats(BaseModel):
    """Statistics for a folder in the vault.

    Attributes:
        path: Folder path relative to vault root
        note_count: Number of notes in this folder
        percentage: Percentage of total vault notes
    """

    path: str = Field(..., description="Folder path")
    note_count: int = Field(default=0, ge=0, description="Number of notes")
    percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Percent of total")


class VaultOverview(BaseModel):
    """Overview statistics for the entire vault.

    Provides a comprehensive snapshot of the vault's current state
    including note counts, organization metrics, and temporal boundaries.

    Attributes:
        total_notes: Total number of notes in the vault
        total_folders: Number of folders in the vault
        total_tags: Number of unique tags used
        notes_with_tags: Notes that have at least one tag
        notes_without_tags: Notes with no tags (orphans)
        notes_with_links: Notes containing wikilinks
        oldest_note: Tuple of (path, created_date) for oldest note
        newest_note: Tuple of (path, created_date) for newest note
        folders: List of folder statistics
    """

    total_notes: int = Field(default=0, ge=0, description="Total notes in vault")
    total_folders: int = Field(default=0, ge=0, description="Number of folders")
    total_tags: int = Field(default=0, ge=0, description="Unique tags count")
    notes_with_tags: int = Field(default=0, ge=0, description="Tagged notes")
    notes_without_tags: int = Field(default=0, ge=0, description="Untagged notes")
    notes_with_links: int = Field(default=0, ge=0, description="Notes with wikilinks")
    oldest_note: tuple[str, datetime] | None = Field(default=None, description="Oldest note")
    newest_note: tuple[str, datetime] | None = Field(default=None, description="Newest note")
    folders: list[FolderStats] = Field(default_factory=list, description="Folder breakdown")


class DayActivity(BaseModel):
    """Activity for a single day.

    Tracks note creation and modification counts for a specific date.

    Attributes:
        date: Date in YYYY-MM-DD format
        notes_created: Number of notes created on this day
        notes_modified: Number of notes modified on this day
    """

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    notes_created: int = Field(default=0, ge=0, description="Notes created")
    notes_modified: int = Field(default=0, ge=0, description="Notes modified")


class ActivityTrends(BaseModel):
    """Activity trends over a time period.

    Aggregates activity statistics for a specified period with
    daily breakdown for detailed analysis.

    Attributes:
        period: Human-readable period description (e.g., "Past Week")
        total_created: Total notes created in period
        total_modified: Total notes modified in period
        most_active_day: Date with highest activity
        daily_average: Average notes per day
        daily_breakdown: Daily activity details
    """

    period: str = Field(default="", description="Time period description")
    total_created: int = Field(default=0, ge=0, description="Notes created in period")
    total_modified: int = Field(default=0, ge=0, description="Notes modified in period")
    most_active_day: str | None = Field(default=None, description="Most active date")
    daily_average: float = Field(default=0.0, ge=0.0, description="Average per day")
    daily_breakdown: list[DayActivity] = Field(default_factory=list, description="Daily stats")


class TagStats(BaseModel):
    """Statistics for a single tag.

    Attributes:
        name: Tag name without # prefix
        count: Number of notes using this tag
        percentage: Percentage of tagged notes using this tag
    """

    name: str = Field(..., description="Tag name without #")
    count: int = Field(default=0, ge=0, description="Usage count")
    percentage: float = Field(default=0.0, ge=0.0, description="Percentage of tagged notes")


class TagDistribution(BaseModel):
    """Distribution of tags across the vault.

    Provides insights into tag usage patterns and identifies
    organizational gaps like orphan notes.

    Attributes:
        total_tags: Number of unique tags in vault
        total_tagged_notes: Notes with at least one tag
        orphan_notes: Notes without any tags
        top_tags: Most frequently used tags
        single_use_tags: Tags used only once
    """

    total_tags: int = Field(default=0, ge=0, description="Unique tags count")
    total_tagged_notes: int = Field(default=0, ge=0, description="Tagged notes count")
    orphan_notes: int = Field(default=0, ge=0, description="Untagged notes count")
    top_tags: list[TagStats] = Field(default_factory=list, description="Most used tags")
    single_use_tags: int = Field(default=0, ge=0, description="Single-use tag count")


class RecentActivity(BaseModel):
    """Recent vault activity.

    Shows recently created and modified notes along with
    the most active areas of the vault.

    Attributes:
        period_days: Number of days in the activity window
        recently_created: List of (path, datetime) for new notes
        recently_modified: List of (path, datetime) for modified notes
        most_active_folders: Folders with most activity
    """

    period_days: int = Field(default=7, ge=1, description="Activity window in days")
    recently_created: list[tuple[str, datetime]] = Field(
        default_factory=list, description="Recently created notes"
    )
    recently_modified: list[tuple[str, datetime]] = Field(
        default_factory=list, description="Recently modified notes"
    )
    most_active_folders: list[str] = Field(default_factory=list, description="Most active folders")


class VaultInsights(BaseModel):
    """AI-generated insights about the vault.

    Provides algorithmic analysis of vault patterns with
    highlights and actionable suggestions.

    Attributes:
        summary: Brief overall summary
        highlights: Notable observations
        suggestions: Actionable improvement suggestions
        metrics: Key metrics dictionary
    """

    summary: str = Field(default="", description="Overall summary")
    highlights: list[str] = Field(default_factory=list, description="Notable observations")
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")
    metrics: dict[str, int | float | str] = Field(default_factory=dict, description="Key metrics")
