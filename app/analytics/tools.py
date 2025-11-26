"""Vault analytics tool for the chat agent.

This module implements the vault_analytics tool which provides
comprehensive analytics and insights about the Obsidian vault
through the PydanticAI agent.

The tool follows a consolidated multi-operation pattern where a single
tool handles multiple analytics operations via an `operation` parameter.

Example usage by the agent:
    vault_analytics(operation="overview")
    vault_analytics(operation="trends", period="month")
    vault_analytics(operation="tag_distribution", limit=20)
    vault_analytics(operation="activity", days=14)
    vault_analytics(operation="insights")
"""

from collections import Counter
from datetime import UTC, datetime, timedelta

from pydantic_ai import RunContext

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
from app.dependencies import ChatDependencies, logger
from app.notes.tools import parse_note
from app.search.tools import extract_wikilinks

# =============================================================================
# Helper Functions
# =============================================================================


def get_folder_from_path(path: str) -> str:
    """Extract folder from file path.

    Args:
        path: File path (e.g., "Projects/API/design.md")

    Returns:
        Folder path, or "(root)" for root-level files

    Examples:
        >>> get_folder_from_path("Projects/API/design.md")
        'Projects/API'
        >>> get_folder_from_path("Projects/note.md")
        'Projects'
        >>> get_folder_from_path("note.md")
        '(root)'
    """
    if "/" not in path:
        return "(root)"
    return path.rsplit("/", 1)[0]


def parse_period(period: str) -> tuple[datetime, datetime]:
    """Convert period string to date range.

    Args:
        period: One of 'day', 'week', 'month', 'year'

    Returns:
        Tuple of (start_date, end_date) with UTC timezone

    Examples:
        >>> start, end = parse_period("week")
        >>> (end - start).days
        7
    """
    now = datetime.now(UTC)
    end = now

    if period == "day":
        start = now - timedelta(days=1)
    elif period == "week":
        start = now - timedelta(days=7)
    elif period == "month":
        start = now - timedelta(days=30)
    elif period == "year":
        start = now - timedelta(days=365)
    else:
        # Default to week for invalid input
        start = now - timedelta(days=7)

    return start, end


def ensure_utc(dt: datetime | None) -> datetime | None:
    """Make datetime timezone-aware (UTC).

    Args:
        dt: Datetime object (may be naive or aware)

    Returns:
        UTC-aware datetime, or None if input is None

    Examples:
        >>> ensure_utc(None) is None
        True
        >>> ensure_utc(datetime(2025, 1, 1)).tzinfo is not None
        True
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _percent(part: int, total: int) -> str:
    """Calculate percentage string.

    Args:
        part: Numerator
        total: Denominator

    Returns:
        Percentage string (e.g., "42.5%")
    """
    if total == 0:
        return "0%"
    return f"{(part / total) * 100:.1f}%"


# =============================================================================
# Formatting Functions
# =============================================================================


def format_overview(overview: VaultOverview) -> str:
    """Format vault overview for display.

    Args:
        overview: VaultOverview model with statistics

    Returns:
        Formatted markdown string
    """
    lines = [
        "**📊 Vault Overview**",
        "",
        f"**Total Notes:** {overview.total_notes}",
        f"**Total Folders:** {overview.total_folders}",
        f"**Unique Tags:** {overview.total_tags}",
        "",
        "**Organization:**",
        f"- Tagged: {overview.notes_with_tags} "
        f"({_percent(overview.notes_with_tags, overview.total_notes)})",
        f"- Untagged: {overview.notes_without_tags} "
        f"({_percent(overview.notes_without_tags, overview.total_notes)})",
        f"- With links: {overview.notes_with_links} "
        f"({_percent(overview.notes_with_links, overview.total_notes)})",
        "",
    ]

    if overview.oldest_note:
        oldest_date = overview.oldest_note[1].strftime("%Y-%m-%d")
        lines.append(f"**Oldest Note:** {overview.oldest_note[0]} ({oldest_date})")
    if overview.newest_note:
        newest_date = overview.newest_note[1].strftime("%Y-%m-%d")
        lines.append(f"**Newest Note:** {overview.newest_note[0]} ({newest_date})")

    if overview.folders:
        lines.extend(["", "**Top Folders:**"])
        for folder in overview.folders[:5]:
            lines.append(f"- {folder.path}: {folder.note_count} notes ({folder.percentage:.1f}%)")

    return "\n".join(lines)


def format_trends(trends: ActivityTrends) -> str:
    """Format activity trends for display.

    Args:
        trends: ActivityTrends model with statistics

    Returns:
        Formatted markdown string
    """
    lines = [
        f"**📈 Activity Trends** ({trends.period})",
        "",
        f"**Notes Created:** {trends.total_created}",
        f"**Notes Modified:** {trends.total_modified}",
        f"**Daily Average:** {trends.daily_average:.1f} notes/day",
    ]

    if trends.most_active_day:
        lines.append(f"**Most Active Day:** {trends.most_active_day}")

    # Show last 7 days breakdown
    if trends.daily_breakdown:
        lines.extend(["", "**Recent Activity:**"])
        for day in trends.daily_breakdown[:7]:
            created = day.notes_created
            modified = day.notes_modified
            lines.append(f"- {day.date}: +{created} created, ~{modified} modified")

    return "\n".join(lines)


def format_tag_distribution(dist: TagDistribution) -> str:
    """Format tag distribution for display.

    Args:
        dist: TagDistribution model with statistics

    Returns:
        Formatted markdown string
    """
    lines = [
        "**🏷️ Tag Distribution**",
        "",
        f"**Unique Tags:** {dist.total_tags}",
        f"**Tagged Notes:** {dist.total_tagged_notes}",
        f"**Orphan Notes:** {dist.orphan_notes} (no tags)",
        f"**Single-Use Tags:** {dist.single_use_tags}",
        "",
        "**Top Tags:**",
    ]

    for tag in dist.top_tags:
        lines.append(f"- **#{tag.name}**: {tag.count} notes ({tag.percentage:.1f}%)")

    return "\n".join(lines)


def format_activity(activity: RecentActivity) -> str:
    """Format recent activity for display.

    Args:
        activity: RecentActivity model with statistics

    Returns:
        Formatted markdown string
    """
    lines = [
        f"**🕐 Recent Activity** (past {activity.period_days} days)",
        "",
    ]

    if activity.recently_created:
        lines.append("**Recently Created:**")
        for path, dt in activity.recently_created[:5]:
            lines.append(f"- {path} ({dt.strftime('%Y-%m-%d')})")
        lines.append("")

    if activity.recently_modified:
        lines.append("**Recently Modified:**")
        for path, dt in activity.recently_modified[:5]:
            lines.append(f"- {path} ({dt.strftime('%Y-%m-%d')})")
        lines.append("")

    if activity.most_active_folders:
        lines.append("**Most Active Folders:**")
        for folder in activity.most_active_folders[:5]:
            lines.append(f"- {folder}")

    if not activity.recently_created and not activity.recently_modified:
        lines.append("No activity in this period.")

    return "\n".join(lines)


def format_insights(insights: VaultInsights) -> str:
    """Format vault insights for display.

    Args:
        insights: VaultInsights model with analysis

    Returns:
        Formatted markdown string
    """
    lines = [
        "**💡 Vault Insights**",
        "",
        insights.summary,
        "",
    ]

    if insights.highlights:
        lines.append("**Highlights:**")
        for highlight in insights.highlights:
            lines.append(f"- {highlight}")
        lines.append("")

    if insights.suggestions:
        lines.append("**Suggestions:**")
        for suggestion in insights.suggestions:
            lines.append(f"- {suggestion}")

    return "\n".join(lines)


# =============================================================================
# Operation Handlers
# =============================================================================


async def _get_overview(ctx: RunContext[ChatDependencies]) -> VaultOverview:
    """Get comprehensive vault overview.

    Args:
        ctx: Context with vault access

    Returns:
        VaultOverview with all statistics
    """
    files = await ctx.deps.vault.list_files()

    total_notes = 0
    folders: dict[str, int] = {}
    all_tags: set[str] = set()
    notes_with_tags = 0
    notes_without_tags = 0
    notes_with_links = 0
    oldest: tuple[str, datetime] | None = None
    newest: tuple[str, datetime] | None = None

    for file_path in files:
        try:
            content = await ctx.deps.vault.read_file(file_path)
            parsed = parse_note(content)
            total_notes += 1

            # Track folders
            folder = get_folder_from_path(file_path)
            folders[folder] = folders.get(folder, 0) + 1

            # Track tags
            if parsed.frontmatter and parsed.frontmatter.tags:
                notes_with_tags += 1
                for tag in parsed.frontmatter.tags:
                    all_tags.add(tag.lower())
            else:
                notes_without_tags += 1

            # Track links
            links = extract_wikilinks(content)
            if links:
                notes_with_links += 1

            # Track dates
            if parsed.frontmatter and parsed.frontmatter.created:
                created = ensure_utc(parsed.frontmatter.created)
                if created:
                    if oldest is None or created < oldest[1]:
                        oldest = (file_path, created)
                    if newest is None or created > newest[1]:
                        newest = (file_path, created)

        except Exception:
            continue

    # Build folder stats
    folder_stats = [
        FolderStats(
            path=folder,
            note_count=count,
            percentage=(count / total_notes * 100) if total_notes > 0 else 0,
        )
        for folder, count in sorted(folders.items(), key=lambda x: x[1], reverse=True)
    ]

    return VaultOverview(
        total_notes=total_notes,
        total_folders=len(folders),
        total_tags=len(all_tags),
        notes_with_tags=notes_with_tags,
        notes_without_tags=notes_without_tags,
        notes_with_links=notes_with_links,
        oldest_note=oldest,
        newest_note=newest,
        folders=folder_stats,
    )


async def _get_trends(
    ctx: RunContext[ChatDependencies],
    period: str,
) -> ActivityTrends:
    """Get activity trends for a time period.

    Args:
        ctx: Context with vault access
        period: One of 'day', 'week', 'month', 'year'

    Returns:
        ActivityTrends with period statistics
    """
    start, end = parse_period(period)
    files = await ctx.deps.vault.list_files()

    # Track daily activity
    created_by_day: Counter[str] = Counter()
    modified_by_day: Counter[str] = Counter()

    for file_path in files:
        try:
            content = await ctx.deps.vault.read_file(file_path)
            parsed = parse_note(content)

            if not parsed.frontmatter:
                continue

            # Check created date
            if parsed.frontmatter.created:
                created = ensure_utc(parsed.frontmatter.created)
                if created and start <= created <= end:
                    day_str = created.strftime("%Y-%m-%d")
                    created_by_day[day_str] += 1

            # Check modified date
            if parsed.frontmatter.modified:
                modified = ensure_utc(parsed.frontmatter.modified)
                if modified and start <= modified <= end:
                    day_str = modified.strftime("%Y-%m-%d")
                    modified_by_day[day_str] += 1

        except Exception:
            continue

    # Build daily breakdown (most recent first)
    all_days = sorted(set(created_by_day.keys()) | set(modified_by_day.keys()), reverse=True)
    daily_breakdown = [
        DayActivity(
            date=day,
            notes_created=created_by_day[day],
            notes_modified=modified_by_day[day],
        )
        for day in all_days
    ]

    # Calculate totals
    total_created = sum(created_by_day.values())
    total_modified = sum(modified_by_day.values())

    # Find most active day (by total activity)
    activity_by_day = Counter()
    for day in all_days:
        activity_by_day[day] = created_by_day[day] + modified_by_day[day]
    most_active = activity_by_day.most_common(1)
    most_active_day = most_active[0][0] if most_active else None

    # Calculate daily average
    period_days = (end - start).days or 1
    daily_average = (total_created + total_modified) / period_days

    period_names = {
        "day": "Past Day",
        "week": "Past Week",
        "month": "Past Month",
        "year": "Past Year",
    }

    return ActivityTrends(
        period=period_names.get(period, "Custom Period"),
        total_created=total_created,
        total_modified=total_modified,
        most_active_day=most_active_day,
        daily_average=daily_average,
        daily_breakdown=daily_breakdown,
    )


async def _get_tag_distribution(
    ctx: RunContext[ChatDependencies],
    limit: int,
    include_orphans: bool,
) -> TagDistribution:
    """Get tag usage distribution.

    Args:
        ctx: Context with vault access
        limit: Maximum tags to return
        include_orphans: Include orphan note count

    Returns:
        TagDistribution with tag statistics
    """
    files = await ctx.deps.vault.list_files()

    tag_counts: Counter[str] = Counter()
    tagged_notes = 0
    orphan_notes = 0

    for file_path in files:
        try:
            content = await ctx.deps.vault.read_file(file_path)
            parsed = parse_note(content)

            if parsed.frontmatter and parsed.frontmatter.tags:
                tagged_notes += 1
                for tag in parsed.frontmatter.tags:
                    tag_counts[tag.lower()] += 1
            else:
                orphan_notes += 1

        except Exception:
            continue

    # Calculate single-use tags
    single_use = sum(1 for count in tag_counts.values() if count == 1)

    # Build top tags
    top_tags = [
        TagStats(
            name=tag,
            count=count,
            percentage=(count / tagged_notes * 100) if tagged_notes > 0 else 0,
        )
        for tag, count in tag_counts.most_common(limit)
    ]

    return TagDistribution(
        total_tags=len(tag_counts),
        total_tagged_notes=tagged_notes,
        orphan_notes=orphan_notes if include_orphans else 0,
        top_tags=top_tags,
        single_use_tags=single_use,
    )


async def _get_activity(
    ctx: RunContext[ChatDependencies],
    days: int,
) -> RecentActivity:
    """Get recent vault activity.

    Args:
        ctx: Context with vault access
        days: Number of days to look back

    Returns:
        RecentActivity with recent notes
    """
    cutoff = datetime.now(UTC) - timedelta(days=days)
    files = await ctx.deps.vault.list_files()

    recently_created: list[tuple[str, datetime]] = []
    recently_modified: list[tuple[str, datetime]] = []
    folder_activity: Counter[str] = Counter()

    for file_path in files:
        try:
            content = await ctx.deps.vault.read_file(file_path)
            parsed = parse_note(content)

            if not parsed.frontmatter:
                continue

            folder = get_folder_from_path(file_path)

            # Check created
            if parsed.frontmatter.created:
                created = ensure_utc(parsed.frontmatter.created)
                if created and created >= cutoff:
                    recently_created.append((file_path, created))
                    folder_activity[folder] += 1

            # Check modified
            if parsed.frontmatter.modified:
                modified = ensure_utc(parsed.frontmatter.modified)
                if modified and modified >= cutoff:
                    recently_modified.append((file_path, modified))
                    folder_activity[folder] += 1

        except Exception:
            continue

    # Sort by date (most recent first)
    recently_created.sort(key=lambda x: x[1], reverse=True)
    recently_modified.sort(key=lambda x: x[1], reverse=True)

    # Get most active folders
    most_active_folders = [folder for folder, _ in folder_activity.most_common(5)]

    return RecentActivity(
        period_days=days,
        recently_created=recently_created,
        recently_modified=recently_modified,
        most_active_folders=most_active_folders,
    )


async def _get_insights(
    ctx: RunContext[ChatDependencies],
    focus: str | None,
) -> VaultInsights:
    """Generate insights about the vault.

    Args:
        ctx: Context with vault access
        focus: Optional focus area (not currently used)

    Returns:
        VaultInsights with analysis
    """
    # Gather data
    overview = await _get_overview(ctx)
    trends = await _get_trends(ctx, "month")
    tag_dist = await _get_tag_distribution(ctx, limit=10, include_orphans=True)

    highlights: list[str] = []
    suggestions: list[str] = []

    # Generate summary
    notes = overview.total_notes
    folders = overview.total_folders
    summary = f"Your vault contains {notes} notes across {folders} folders."

    # Calculate metrics
    if overview.total_notes > 0:
        tag_rate = overview.notes_with_tags / overview.total_notes * 100
        link_rate = overview.notes_with_links / overview.total_notes * 100
    else:
        tag_rate = 0
        link_rate = 0

    # Tag-based insights
    if tag_rate >= 80:
        highlights.append(f"Excellent tagging: {tag_rate:.0f}% of notes are tagged")
    elif tag_rate >= 50:
        highlights.append(f"Good tagging coverage: {tag_rate:.0f}% of notes tagged")
    else:
        highlights.append(f"Low tagging: only {tag_rate:.0f}% of notes are tagged")
        suggestions.append("Consider using tags to organize your notes better")

    # Orphan insights
    orphans = tag_dist.orphan_notes
    if orphans > 0:
        if orphans <= 5:
            highlights.append(f"Nearly complete: only {orphans} untagged notes")
        else:
            suggestions.append(f"Tag your {orphans} orphan notes to improve organization")

    # Link-based insights
    if link_rate >= 50:
        highlights.append(f"Well-connected vault: {link_rate:.0f}% of notes have links")
    elif link_rate < 20:
        suggestions.append("Add more [[wikilinks]] to connect related notes")

    # Activity insights
    if trends.total_created > 0:
        highlights.append(f"Active vault: {trends.total_created} notes created this month")
    else:
        suggestions.append("Your vault hasn't had new notes recently - consider adding more")

    # Single-use tag insights
    if tag_dist.single_use_tags > tag_dist.total_tags * 0.5:
        suggestions.append(
            f"Many single-use tags ({tag_dist.single_use_tags}) - consider consolidating"
        )

    metrics = {
        "total_notes": overview.total_notes,
        "tag_rate": f"{tag_rate:.1f}%",
        "link_rate": f"{link_rate:.1f}%",
        "orphan_notes": tag_dist.orphan_notes,
        "monthly_created": trends.total_created,
    }

    return VaultInsights(
        summary=summary,
        highlights=highlights,
        suggestions=suggestions,
        metrics=metrics,
    )


# =============================================================================
# Main Tool Function
# =============================================================================


async def vault_analytics(
    ctx: RunContext[ChatDependencies],
    operation: str,
    period: str = "week",
    days: int = 7,
    limit: int = 10,
    include_orphans: bool = True,
    focus: str | None = None,
) -> str:
    """Analyze the Obsidian vault to provide statistics, trends, and insights.

    This is the main tool function that the LLM calls to analyze the vault.
    It supports five operation types: overview, trends, tag_distribution,
    activity, and insights.

    Args:
        ctx: Context with vault access and trace_id for logging
        operation: One of 'overview', 'trends', 'tag_distribution', 'activity', 'insights'
        period: Time period for trends - 'day', 'week', 'month', 'year' (default: week)
        days: Number of days for activity operation (default: 7)
        limit: Maximum items for tag_distribution (default: 10)
        include_orphans: Include untagged note count (default: True)
        focus: Optional focus area for insights

    Returns:
        Formatted analytics results or error message. Returns user-friendly
        error messages instead of raising exceptions.

    Examples:
        overview:          vault_analytics(ctx, "overview")
        trends:            vault_analytics(ctx, "trends", period="month")
        tag_distribution:  vault_analytics(ctx, "tag_distribution", limit=20)
        activity:          vault_analytics(ctx, "activity", days=14)
        insights:          vault_analytics(ctx, "insights")
    """
    logger.info(
        "vault_analytics_called",
        extra={
            "operation": operation,
            "period": period,
            "days": days,
            "trace_id": ctx.deps.trace_id,
        },
    )

    try:
        if operation == "overview":
            result = await _get_overview(ctx)
            return format_overview(result)

        elif operation == "trends":
            result = await _get_trends(ctx, period)
            return format_trends(result)

        elif operation == "tag_distribution":
            result = await _get_tag_distribution(ctx, limit, include_orphans)
            return format_tag_distribution(result)

        elif operation == "activity":
            result = await _get_activity(ctx, days)
            return format_activity(result)

        elif operation == "insights":
            result = await _get_insights(ctx, focus)
            return format_insights(result)

        else:
            return (
                f"Unknown operation: {operation}. "
                f"Valid: overview, trends, tag_distribution, activity, insights"
            )

    except Exception as e:
        logger.error(
            "vault_analytics_failed",
            extra={
                "operation": operation,
                "error": str(e),
                "trace_id": ctx.deps.trace_id,
            },
            exc_info=True,
        )
        return f"Error performing {operation}: {str(e)}"
