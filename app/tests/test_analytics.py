"""Tests for analytics feature.

This module contains comprehensive tests for the vault_analytics tool
and its helper functions. Tests use pytest fixtures with temporary
directories to ensure isolation.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.analytics.models import (
    ActivityTrends,
    DayActivity,
    FolderStats,
    RecentActivity,
    TagDistribution,
    TagStats,
    VaultInsights,
    VaultOverview,
)
from app.analytics.tools import (
    ensure_utc,
    format_activity,
    format_insights,
    format_overview,
    format_tag_distribution,
    format_trends,
    get_folder_from_path,
    parse_period,
    vault_analytics,
)
from app.dependencies import ChatDependencies, VaultClient

# =============================================================================
# Get Folder From Path Tests
# =============================================================================


class TestGetFolderFromPath:
    """Tests for folder extraction helper function."""

    def test_nested_path(self) -> None:
        """Test extraction from nested path."""
        assert get_folder_from_path("Projects/API/design.md") == "Projects/API"

    def test_single_folder(self) -> None:
        """Test extraction from single folder path."""
        assert get_folder_from_path("Projects/note.md") == "Projects"

    def test_root_level(self) -> None:
        """Test extraction from root-level file."""
        assert get_folder_from_path("note.md") == "(root)"

    def test_deep_nesting(self) -> None:
        """Test extraction from deeply nested path."""
        assert get_folder_from_path("a/b/c/d/file.md") == "a/b/c/d"


# =============================================================================
# Parse Period Tests
# =============================================================================


class TestParsePeriod:
    """Tests for period parsing helper function."""

    def test_day_period(self) -> None:
        """Test day period returns 1 day difference."""
        start, end = parse_period("day")
        diff = (end - start).days
        assert diff == 1

    def test_week_period(self) -> None:
        """Test week period returns 7 days difference."""
        start, end = parse_period("week")
        diff = (end - start).days
        assert diff == 7

    def test_month_period(self) -> None:
        """Test month period returns 30 days difference."""
        start, end = parse_period("month")
        diff = (end - start).days
        assert diff == 30

    def test_year_period(self) -> None:
        """Test year period returns 365 days difference."""
        start, end = parse_period("year")
        diff = (end - start).days
        assert diff == 365

    def test_invalid_defaults_to_week(self) -> None:
        """Test invalid period defaults to week (7 days)."""
        start, end = parse_period("invalid")
        diff = (end - start).days
        assert diff == 7


# =============================================================================
# Ensure UTC Tests
# =============================================================================


class TestEnsureUtc:
    """Tests for UTC timezone helper function."""

    def test_none_input(self) -> None:
        """Test None input returns None."""
        assert ensure_utc(None) is None

    def test_naive_datetime(self) -> None:
        """Test naive datetime gets UTC timezone added."""
        naive = datetime(2025, 1, 1, 12, 0, 0)
        result = ensure_utc(naive)
        assert result is not None
        assert result.tzinfo is not None
        assert result.tzinfo == UTC

    def test_aware_datetime(self) -> None:
        """Test aware datetime preserves timezone."""
        aware = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = ensure_utc(aware)
        assert result is not None
        assert result.tzinfo == UTC
        assert result == aware


# =============================================================================
# Analytics Models Tests
# =============================================================================


class TestAnalyticsModels:
    """Tests for analytics Pydantic models."""

    def test_folder_stats_defaults(self) -> None:
        """Test FolderStats model defaults."""
        stats = FolderStats(path="Projects")
        assert stats.path == "Projects"
        assert stats.note_count == 0
        assert stats.percentage == 0.0

    def test_folder_stats_validation(self) -> None:
        """Test FolderStats percentage validation."""
        with pytest.raises(ValueError):
            FolderStats(path="test", percentage=101.0)

    def test_vault_overview_defaults(self) -> None:
        """Test VaultOverview model defaults."""
        overview = VaultOverview()
        assert overview.total_notes == 0
        assert overview.total_folders == 0
        assert overview.total_tags == 0
        assert overview.notes_with_tags == 0
        assert overview.notes_without_tags == 0
        assert overview.notes_with_links == 0
        assert overview.oldest_note is None
        assert overview.newest_note is None
        assert overview.folders == []

    def test_day_activity_creation(self) -> None:
        """Test DayActivity model creation."""
        activity = DayActivity(date="2025-01-15", notes_created=5, notes_modified=3)
        assert activity.date == "2025-01-15"
        assert activity.notes_created == 5
        assert activity.notes_modified == 3

    def test_tag_stats_validation(self) -> None:
        """Test TagStats count validation (must be non-negative)."""
        with pytest.raises(ValueError):
            TagStats(name="test", count=-1)

    def test_vault_insights_defaults(self) -> None:
        """Test VaultInsights model defaults."""
        insights = VaultInsights()
        assert insights.summary == ""
        assert insights.highlights == []
        assert insights.suggestions == []
        assert insights.metrics == {}

    def test_activity_trends_defaults(self) -> None:
        """Test ActivityTrends model defaults."""
        trends = ActivityTrends()
        assert trends.period == ""
        assert trends.total_created == 0
        assert trends.total_modified == 0
        assert trends.most_active_day is None
        assert trends.daily_average == 0.0
        assert trends.daily_breakdown == []

    def test_tag_distribution_defaults(self) -> None:
        """Test TagDistribution model defaults."""
        dist = TagDistribution()
        assert dist.total_tags == 0
        assert dist.total_tagged_notes == 0
        assert dist.orphan_notes == 0
        assert dist.top_tags == []
        assert dist.single_use_tags == 0

    def test_recent_activity_defaults(self) -> None:
        """Test RecentActivity model defaults."""
        activity = RecentActivity()
        assert activity.period_days == 7
        assert activity.recently_created == []
        assert activity.recently_modified == []
        assert activity.most_active_folders == []


# =============================================================================
# Formatting Tests
# =============================================================================


class TestFormatting:
    """Tests for formatting helper functions."""

    def test_format_overview(self) -> None:
        """Test overview formatting contains key sections."""
        overview = VaultOverview(
            total_notes=100,
            total_folders=10,
            total_tags=25,
            notes_with_tags=80,
            notes_without_tags=20,
            notes_with_links=60,
            folders=[FolderStats(path="Projects", note_count=30, percentage=30.0)],
        )
        result = format_overview(overview)

        assert "📊 Vault Overview" in result
        assert "100" in result
        assert "10" in result
        assert "Projects" in result

    def test_format_overview_with_dates(self) -> None:
        """Test overview formatting with oldest/newest notes."""
        now = datetime.now(UTC)
        overview = VaultOverview(
            total_notes=10,
            oldest_note=("old.md", now - timedelta(days=30)),
            newest_note=("new.md", now),
        )
        result = format_overview(overview)

        assert "Oldest Note" in result
        assert "Newest Note" in result
        assert "old.md" in result
        assert "new.md" in result

    def test_format_trends(self) -> None:
        """Test trends formatting contains period and stats."""
        trends = ActivityTrends(
            period="Past Week",
            total_created=10,
            total_modified=15,
            most_active_day="2025-01-15",
            daily_average=3.5,
            daily_breakdown=[
                DayActivity(date="2025-01-15", notes_created=5, notes_modified=3),
            ],
        )
        result = format_trends(trends)

        assert "📈 Activity Trends" in result
        assert "Past Week" in result
        assert "10" in result
        assert "Most Active Day" in result

    def test_format_tag_distribution(self) -> None:
        """Test tag distribution formatting contains tag list."""
        dist = TagDistribution(
            total_tags=10,
            total_tagged_notes=50,
            orphan_notes=5,
            top_tags=[
                TagStats(name="project", count=20, percentage=40.0),
                TagStats(name="api", count=10, percentage=20.0),
            ],
            single_use_tags=3,
        )
        result = format_tag_distribution(dist)

        assert "🏷️ Tag Distribution" in result
        assert "#project" in result
        assert "#api" in result
        assert "Orphan Notes" in result

    def test_format_activity(self) -> None:
        """Test activity formatting contains recent items."""
        now = datetime.now(UTC)
        activity = RecentActivity(
            period_days=7,
            recently_created=[("new.md", now)],
            recently_modified=[("modified.md", now)],
            most_active_folders=["Projects"],
        )
        result = format_activity(activity)

        assert "🕐 Recent Activity" in result
        assert "new.md" in result
        assert "modified.md" in result
        assert "Projects" in result

    def test_format_activity_empty(self) -> None:
        """Test activity formatting with no activity."""
        activity = RecentActivity(period_days=7)
        result = format_activity(activity)

        assert "No activity" in result

    def test_format_insights(self) -> None:
        """Test insights formatting contains summary and highlights."""
        insights = VaultInsights(
            summary="Your vault is well organized.",
            highlights=["Great tag coverage", "Well connected"],
            suggestions=["Add more links"],
        )
        result = format_insights(insights)

        assert "💡 Vault Insights" in result
        assert "well organized" in result
        assert "Great tag coverage" in result
        assert "Add more links" in result


# =============================================================================
# Vault Analytics Tool Tests (Integration)
# =============================================================================


class TestVaultAnalytics:
    """Integration tests for the vault_analytics tool function."""

    @pytest.fixture
    def sample_vault(self, tmp_path: Path) -> Path:
        """Create vault with sample notes for testing."""
        now = datetime.now(UTC)
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)

        # Welcome.md - root, month-old, one tag, has wikilink
        (tmp_path / "Welcome.md").write_text(f"""---
created: {month_ago.isoformat()}
modified: {week_ago.isoformat()}
tags:
  - welcome
---
# Welcome
See [[Projects/API]] for more.""")

        # Projects/API.md - week-old, two tags, has wikilinks
        (tmp_path / "Projects").mkdir()
        (tmp_path / "Projects" / "API.md").write_text(f"""---
created: {week_ago.isoformat()}
modified: {now.isoformat()}
tags:
  - project
  - api
---
# API Design
Links to [[Architecture]].""")

        # Projects/Architecture.md - week-old, two tags
        (tmp_path / "Projects" / "Architecture.md").write_text(f"""---
created: {week_ago.isoformat()}
modified: {week_ago.isoformat()}
tags:
  - project
  - architecture
---
# Architecture""")

        # Meetings/Standup.md - today, one tag
        (tmp_path / "Meetings").mkdir()
        (tmp_path / "Meetings" / "Standup.md").write_text(f"""---
created: {now.isoformat()}
modified: {now.isoformat()}
tags:
  - meeting
---
# Daily Standup""")

        # Ideas.md - no frontmatter (orphan)
        (tmp_path / "Ideas.md").write_text("# Ideas\nNo frontmatter.")

        return tmp_path

    @pytest.fixture
    def mock_ctx(self, sample_vault: Path) -> Mock:
        """Create mock context with vault client."""
        ctx = Mock()
        ctx.deps = ChatDependencies(
            vault=VaultClient(vault_path=sample_vault),
            trace_id="test-trace-analytics",
        )
        return ctx

    @pytest.mark.asyncio
    async def test_overview_operation(self, mock_ctx: Mock) -> None:
        """Test overview operation returns statistics."""
        result = await vault_analytics(mock_ctx, operation="overview")

        assert "📊 Vault Overview" in result
        assert "5" in result  # 5 total notes
        assert "Total Notes" in result

    @pytest.mark.asyncio
    async def test_overview_folder_stats(self, mock_ctx: Mock) -> None:
        """Test overview shows folder breakdown."""
        result = await vault_analytics(mock_ctx, operation="overview")

        assert "Projects" in result
        assert "Meetings" in result

    @pytest.mark.asyncio
    async def test_trends_week(self, mock_ctx: Mock) -> None:
        """Test trends operation with week period."""
        result = await vault_analytics(mock_ctx, operation="trends", period="week")

        assert "📈 Activity Trends" in result
        assert "Past Week" in result

    @pytest.mark.asyncio
    async def test_trends_month(self, mock_ctx: Mock) -> None:
        """Test trends operation with month period."""
        result = await vault_analytics(mock_ctx, operation="trends", period="month")

        assert "📈 Activity Trends" in result
        assert "Past Month" in result

    @pytest.mark.asyncio
    async def test_tag_distribution(self, mock_ctx: Mock) -> None:
        """Test tag distribution shows top tags."""
        result = await vault_analytics(mock_ctx, operation="tag_distribution")

        assert "🏷️ Tag Distribution" in result
        assert "#project" in result  # Most used tag (2 notes)

    @pytest.mark.asyncio
    async def test_tag_distribution_limit(self, mock_ctx: Mock) -> None:
        """Test tag distribution respects limit parameter."""
        result = await vault_analytics(mock_ctx, operation="tag_distribution", limit=2)

        assert "🏷️ Tag Distribution" in result

    @pytest.mark.asyncio
    async def test_activity_operation(self, mock_ctx: Mock) -> None:
        """Test activity operation shows recent changes."""
        result = await vault_analytics(mock_ctx, operation="activity", days=7)

        assert "🕐 Recent Activity" in result

    @pytest.mark.asyncio
    async def test_activity_custom_days(self, mock_ctx: Mock) -> None:
        """Test activity operation respects days parameter."""
        result = await vault_analytics(mock_ctx, operation="activity", days=30)

        assert "past 30 days" in result

    @pytest.mark.asyncio
    async def test_insights_operation(self, mock_ctx: Mock) -> None:
        """Test insights operation returns summary."""
        result = await vault_analytics(mock_ctx, operation="insights")

        assert "💡 Vault Insights" in result
        assert "5 notes" in result  # Summary mentions note count

    @pytest.mark.asyncio
    async def test_unknown_operation(self, mock_ctx: Mock) -> None:
        """Test unknown operation returns error message."""
        result = await vault_analytics(mock_ctx, operation="invalid")

        assert "Unknown operation" in result
        assert "Valid:" in result

    @pytest.mark.asyncio
    async def test_empty_vault(self, tmp_path: Path) -> None:
        """Test analytics handles empty vault gracefully."""
        ctx = Mock()
        ctx.deps = ChatDependencies(
            vault=VaultClient(vault_path=tmp_path),
            trace_id="test-empty",
        )

        result = await vault_analytics(ctx, operation="overview")

        assert "📊 Vault Overview" in result
        assert "0" in result  # Zero notes
