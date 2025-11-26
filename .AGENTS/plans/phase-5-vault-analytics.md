# Feature: vault_analytics Tool (Phase 5)

## Feature Description

Implement the `vault_analytics` tool to provide comprehensive analytics and insights about the user's Obsidian vault. This tool enables users to understand their note-taking patterns, track activity over time, analyze tag usage, and receive AI-generated narrative insights.

The tool follows the established multi-operation pattern, supporting five operations: `overview`, `trends`, `tag_distribution`, `activity`, and `insights`. Results are returned in a dual format combining narrative summaries with structured data that the LLM can present conversationally.

## User Story

As an Obsidian user, I want to ask about my vault's statistics, trends, and patterns so that I can understand my knowledge management habits and optimize my workflow.

## Problem Statement

Users have no visibility into their vault's health, activity patterns, or organizational effectiveness. They cannot easily answer:
- How many notes do I have? How are they distributed by folder?
- What are my most/least used tags?
- Am I creating notes consistently? What are my peak productivity times?
- Are there orphan notes without tags or connections?

## Solution Statement

Create a `vault_analytics` tool that scans the vault to compute statistics, identify patterns, and generate both structured data and natural language insights. The tool integrates with the existing chat agent for natural language interaction.

## Feature Metadata

- **Feature Type**: New Capability
- **Estimated Complexity**: Medium
- **Primary Systems Affected**: app/analytics/, app/chat/agent.py
- **Dependencies**: datetime, collections (Counter), existing parse_note/extract_title helpers

---

## CONTEXT REFERENCES

### Relevant Codebase Files - READ BEFORE IMPLEMENTING

| File | Lines | Purpose |
|------|-------|---------|
| `app/tags/tools.py` | 398-482 | Most recent tool, shows operation dispatch pattern |
| `app/search/tools.py` | 716-826 | Complex multi-operation tool with many parameters |
| `app/notes/tools.py` | 165-244 | Error handling and logging patterns |
| `app/search/models.py` | 12-53 | Result model structure with defaults |
| `app/tags/models.py` | 6-59 | Models with Field validators and counts |
| `app/notes/models.py` | 9-55 | Datetime handling in Pydantic models |
| `app/chat/agent.py` | 10-67 | System prompt structure and tool registration |
| `app/tests/test_tags.py` | 212-469 | Comprehensive test class with fixtures |
| `app/tests/conftest.py` | 1-49 | Shared fixtures pattern |

### New Files to Create

- `app/analytics/__init__.py` - Module exports
- `app/analytics/models.py` - Pydantic models for analytics results
- `app/analytics/tools.py` - vault_analytics tool implementation
- `app/tests/test_analytics.py` - Comprehensive tests

### Patterns to Follow

**Error Handling Pattern (from tags/tools.py):**
```python
try:
    if operation == "overview":
        return await _get_overview(ctx)
    elif operation == "trends":
        result = await _get_trends(ctx, period)
    # ... more operations
    else:
        return f"Unknown operation: {operation}. Valid: overview, trends, tag_distribution, activity, insights"
except Exception as e:
    logger.error(
        "vault_analytics_failed",
        extra={"operation": operation, "error": str(e), "trace_id": ctx.deps.trace_id},
        exc_info=True,
    )
    return f"Error performing {operation}: {str(e)}"
```

**Logging Pattern (from notes/tools.py):**
```python
logger.info(
    "vault_analytics_called",
    extra={
        "operation": operation,
        "period": period,
        "days": days,
        "trace_id": ctx.deps.trace_id,
    },
)
```

**Model Pattern (from search/models.py):**
```python
class TagStats(BaseModel):
    """Statistics for a single tag."""
    name: str = Field(..., description="Tag name without #")
    count: int = Field(default=0, ge=0)
    percentage: float = Field(default=0.0, ge=0.0)
```

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation - Models & Module Structure
- Create `app/analytics/` directory
- Create Pydantic models for each operation's return data
- Create module `__init__.py` with exports

### Phase 2: Core Implementation - Tool & Helpers
- Implement helper functions for vault scanning and calculations
- Implement private operation handlers for each operation
- Implement main `vault_analytics` tool function
- Implement result formatting functions

### Phase 3: Integration - Agent
- Update `app/chat/agent.py` to import and register the tool
- Update system prompt with analytics usage guidelines

### Phase 4: Testing & Validation
- Create test fixtures with sample vault data
- Write unit tests for helper functions
- Write integration tests for each operation

---

## STEP-BY-STEP TASKS

### Task 1: CREATE `app/analytics/__init__.py`

**IMPLEMENT**: Module exports for tool function

```python
"""Analytics feature module."""
from .tools import vault_analytics

__all__ = ["vault_analytics"]
```

**VALIDATE**: `python -c "from app.analytics import vault_analytics"`

---

### Task 2: CREATE `app/analytics/models.py`

**IMPLEMENT**: Pydantic models for all analytics result types
**PATTERN**: Mirror `app/search/models.py` and `app/tags/models.py`

Create the following models with proper docstrings and Field definitions:

**FolderStats:**
```python
class FolderStats(BaseModel):
    """Statistics for a folder in the vault."""
    path: str = Field(..., description="Folder path")
    note_count: int = Field(default=0, ge=0, description="Number of notes")
    percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Percent of total")
```

**VaultOverview:**
```python
class VaultOverview(BaseModel):
    """Overview statistics for the entire vault."""
    total_notes: int = Field(default=0, ge=0)
    total_folders: int = Field(default=0, ge=0)
    total_tags: int = Field(default=0, ge=0)
    notes_with_tags: int = Field(default=0, ge=0)
    notes_without_tags: int = Field(default=0, ge=0)
    notes_with_links: int = Field(default=0, ge=0)
    oldest_note: tuple[str, datetime] | None = Field(default=None)
    newest_note: tuple[str, datetime] | None = Field(default=None)
    folders: list[FolderStats] = Field(default_factory=list)
```

**DayActivity:**
```python
class DayActivity(BaseModel):
    """Activity for a single day."""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    notes_created: int = Field(default=0, ge=0)
    notes_modified: int = Field(default=0, ge=0)
```

**ActivityTrends:**
```python
class ActivityTrends(BaseModel):
    """Activity trends over a time period."""
    period: str = Field(default="", description="Time period description")
    total_created: int = Field(default=0, ge=0)
    total_modified: int = Field(default=0, ge=0)
    most_active_day: str | None = Field(default=None)
    daily_average: float = Field(default=0.0, ge=0.0)
    daily_breakdown: list[DayActivity] = Field(default_factory=list)
```

**TagStats:**
```python
class TagStats(BaseModel):
    """Statistics for a single tag."""
    name: str = Field(..., description="Tag name without #")
    count: int = Field(default=0, ge=0)
    percentage: float = Field(default=0.0, ge=0.0)
```

**TagDistribution:**
```python
class TagDistribution(BaseModel):
    """Distribution of tags across the vault."""
    total_tags: int = Field(default=0, ge=0)
    total_tagged_notes: int = Field(default=0, ge=0)
    orphan_notes: int = Field(default=0, ge=0)
    top_tags: list[TagStats] = Field(default_factory=list)
    single_use_tags: int = Field(default=0, ge=0)
```

**RecentActivity:**
```python
class RecentActivity(BaseModel):
    """Recent vault activity."""
    period_days: int = Field(default=7, ge=1)
    recently_created: list[tuple[str, datetime]] = Field(default_factory=list)
    recently_modified: list[tuple[str, datetime]] = Field(default_factory=list)
    most_active_folders: list[str] = Field(default_factory=list)
```

**VaultInsights:**
```python
class VaultInsights(BaseModel):
    """AI-generated insights about the vault."""
    summary: str = Field(default="")
    highlights: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    metrics: dict[str, int | float | str] = Field(default_factory=dict)
```

**VALIDATE**: `python -c "from app.analytics.models import VaultOverview, ActivityTrends, TagDistribution, RecentActivity, VaultInsights"`

---

### Task 3: CREATE `app/analytics/tools.py`

**IMPLEMENT**: Main vault_analytics tool and all helper functions
**PATTERN**: Mirror `app/tags/tools.py` structure

**Required Imports:**
```python
from collections import Counter
from datetime import UTC, datetime, timedelta

from pydantic_ai import RunContext

from app.analytics.models import (
    ActivityTrends, DayActivity, FolderStats, RecentActivity,
    TagDistribution, TagStats, VaultInsights, VaultOverview,
)
from app.dependencies import ChatDependencies, logger
from app.notes.tools import parse_note
from app.search.tools import extract_title, extract_wikilinks
```

**Helper Functions to Implement:**

1. `get_folder_from_path(path: str) -> str`
   - Extract folder from path, return "(root)" for root-level files
   - Example: "Projects/API.md" → "Projects"

2. `parse_period(period: str) -> tuple[datetime, datetime]`
   - Convert period string to date range
   - Support: "day", "week", "month", "year"
   - Default to week for invalid input

3. `ensure_utc(dt: datetime | None) -> datetime | None`
   - Make datetime timezone-aware (UTC)
   - Handle None gracefully

4. `_percent(part: int, total: int) -> str`
   - Calculate percentage string
   - Handle division by zero

**Formatting Functions:**

1. `format_overview(overview: VaultOverview) -> str`
   - Format with emoji headers: "**📊 Vault Overview**"
   - Show totals, organization stats, oldest/newest notes
   - Show top 5 folders

2. `format_trends(trends: ActivityTrends) -> str`
   - Format with: "**📈 Activity Trends** ({period})"
   - Show created/modified counts, daily average, most active day
   - Show last 7 days breakdown

3. `format_tag_distribution(dist: TagDistribution) -> str`
   - Format with: "**🏷️ Tag Distribution**"
   - Show totals, orphan count, single-use count
   - Show top tags with counts and percentages

4. `format_activity(activity: RecentActivity) -> str`
   - Format with: "**🕐 Recent Activity**"
   - Show recently created (max 5)
   - Show recently modified (max 5)
   - Show most active folders

5. `format_insights(insights: VaultInsights) -> str`
   - Format with: "**💡 Vault Insights**"
   - Show summary, highlights (bullets), suggestions (bullets)

**Operation Handlers:**

1. `_get_overview(ctx) -> VaultOverview`
   - Scan all files, count folders, tags, links
   - Track oldest/newest by frontmatter.created
   - Build folder statistics

2. `_get_trends(ctx, period) -> ActivityTrends`
   - Parse period to date range
   - Group notes by creation/modification date
   - Calculate daily breakdown and averages

3. `_get_tag_distribution(ctx, limit, include_orphans) -> TagDistribution`
   - Count tag occurrences across vault
   - Calculate single-use tag count
   - Build top tags list

4. `_get_activity(ctx, days) -> RecentActivity`
   - Find notes created/modified within days
   - Sort by date descending
   - Track most active folders

5. `_get_insights(ctx, focus) -> VaultInsights`
   - Call _get_overview, _get_trends, _get_tag_distribution
   - Compute highlights based on tag rate, link rate
   - Generate suggestions for improvement

**Main Tool Function Signature:**
```python
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

    Args:
        ctx: Context with vault access and trace_id
        operation: 'overview', 'trends', 'tag_distribution', 'activity', 'insights'
        period: Time period for trends - 'day', 'week', 'month', 'year'
        days: Number of days for activity operation (default 7)
        limit: Maximum items for tag_distribution (default 10)
        include_orphans: Include untagged note count (default True)
        focus: Optional focus area for insights

    Returns:
        Formatted analytics results or error message.

    Examples:
        vault_analytics(ctx, "overview")
        vault_analytics(ctx, "trends", period="month")
        vault_analytics(ctx, "tag_distribution", limit=20)
        vault_analytics(ctx, "activity", days=14)
        vault_analytics(ctx, "insights")
    """
```

**VALIDATE**: `uv run ruff check app/analytics/tools.py`

---

### Task 4: UPDATE `app/chat/agent.py`

**IMPLEMENT**: Import and register vault_analytics tool, update system prompt

**Add import (after line 8):**
```python
from app.analytics import vault_analytics
```

**Update tools list (line 64):**
```python
tools=[note_operations, vault_search, tag_management, vault_analytics],
```

**Add to SYSTEM_PROMPT (after tag_management section, before closing quotes):**
```
You have access to the vault_analytics tool which allows you to:
- overview: Get comprehensive vault statistics (notes, folders, tags, links)
- trends: Analyze activity over time (day/week/month/year periods)
- tag_distribution: See tag usage patterns and orphan notes
- activity: View recent vault activity (created/modified notes)
- insights: Get AI-generated analysis with suggestions

Analytics guidelines:
- For "how many notes" or "vault stats" → use overview operation
- For "trends" or "activity over time" → use trends with appropriate period
- For "tag usage" or "most used tags" → use tag_distribution
- For "recent changes" or "what did I create" → use activity with days parameter
- For general vault health → use insights operation
```

**VALIDATE**: `uv run python -c "from app.chat.agent import chat_agent; print([t.name for t in chat_agent._function_tools.values()])"`

---

### Task 5: CREATE `app/tests/test_analytics.py`

**IMPLEMENT**: Comprehensive tests for all analytics operations
**PATTERN**: Mirror `app/tests/test_tags.py` structure

**Test Structure:**

```python
"""Tests for analytics feature."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.dependencies import ChatDependencies, VaultClient
from app.analytics.models import (...)
from app.analytics.tools import (...)
```

**Test Classes:**

1. **TestGetFolderFromPath**
   - `test_nested_path`: "Projects/API/design.md" → "Projects/API"
   - `test_single_folder`: "Projects/note.md" → "Projects"
   - `test_root_level`: "note.md" → "(root)"

2. **TestParsePeriod**
   - `test_day_period`: 1 day difference
   - `test_week_period`: 7 days difference
   - `test_month_period`: 30 days difference
   - `test_year_period`: 365 days difference
   - `test_invalid_defaults_to_week`: unknown period → 7 days

3. **TestEnsureUtc**
   - `test_none_input`: None → None
   - `test_naive_datetime`: adds UTC timezone
   - `test_aware_datetime`: preserves existing timezone

4. **TestAnalyticsModels**
   - `test_folder_stats_defaults`: verify defaults
   - `test_folder_stats_validation`: percentage > 100 fails
   - `test_vault_overview_defaults`: verify all defaults
   - `test_day_activity_creation`: verify field assignment
   - `test_tag_stats_validation`: negative count fails
   - `test_vault_insights_defaults`: verify empty collections

5. **TestFormatting**
   - `test_format_overview`: contains key sections
   - `test_format_trends`: contains period and stats
   - `test_format_tag_distribution`: contains tag list
   - `test_format_activity`: contains recent items
   - `test_format_insights`: contains summary and highlights

6. **TestVaultAnalytics** (Integration)

   **Sample Vault Fixture:**
   ```python
   @pytest.fixture
   def sample_vault(self, tmp_path: Path) -> Path:
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
   ```

   **Integration Tests:**
   - `test_overview_operation`: Returns statistics, shows 5 notes
   - `test_overview_folder_stats`: Shows Projects, Meetings folders
   - `test_trends_week`: Returns 7 days data
   - `test_trends_month`: Returns 30 days data
   - `test_tag_distribution`: Shows #project as top tag
   - `test_tag_distribution_limit`: Respects limit param
   - `test_activity_operation`: Shows recent changes
   - `test_activity_custom_days`: Respects days param
   - `test_insights_operation`: Returns summary
   - `test_unknown_operation`: Returns error message
   - `test_empty_vault`: Handles gracefully with zeros

**VALIDATE**: `uv run pytest app/tests/test_analytics.py -v`

---

### Task 6: Run Full Validation

```bash
# Run all tests
uv run pytest app/tests/ -v --tb=short

# Lint check
uv run ruff check app/

# Format check
uv run ruff format app/ --check
```

---

## TESTING STRATEGY

### Unit Tests
- Helper functions: `get_folder_from_path`, `parse_period`, `ensure_utc`
- Formatting functions with various data states (full, partial, empty)
- Model validation bounds (negative counts, percentage > 100)

### Integration Tests
- Each operation with realistic vault data
- Empty vault handling
- Notes without frontmatter (should be excluded from date analytics)

### Edge Cases
- Empty vault (no files) - should return zeros
- Notes without frontmatter - excluded from date-based analytics
- Timezone handling - naive vs aware datetimes
- Very large results - verify limits are respected

---

## VALIDATION COMMANDS

```bash
# Level 1: Syntax & Style
uv run ruff check app/
uv run ruff format app/ --check

# Level 2: Tests
uv run pytest app/tests/ -v
uv run pytest app/tests/test_analytics.py -v
uv run pytest app/tests/ --cov=app --cov-report=term-missing

# Level 3: Integration
uv run python -c "from app.analytics import vault_analytics; print('OK')"
uv run python -c "from app.chat.agent import chat_agent; assert 'vault_analytics' in [t.name for t in chat_agent._function_tools.values()]; print('Tool registered')"

# Level 4: Manual (with server running on port 8000)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "messages": [{"role": "user", "content": "Show me my vault overview"}]}'

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "messages": [{"role": "user", "content": "What are my note trends this month?"}]}'
```

---

## ACCEPTANCE CRITERIA

- [ ] All 5 operations implemented (overview, trends, tag_distribution, activity, insights)
- [ ] All validation commands pass with zero errors
- [ ] Unit test coverage > 80%
- [ ] Code follows project conventions (logging, error handling, docstrings)
- [ ] System prompt updated with analytics guidelines
- [ ] Tool registered with chat agent
- [ ] No regressions in existing functionality

---

## COMPLETION CHECKLIST

- [ ] Task 1: `app/analytics/__init__.py` created
- [ ] Task 2: `app/analytics/models.py` with all 8 model classes
- [ ] Task 3: `app/analytics/tools.py` with tool, 5 handlers, 5 formatters, 4 helpers
- [ ] Task 4: `app/chat/agent.py` updated (import, tools list, system prompt)
- [ ] Task 5: `app/tests/test_analytics.py` with 6 test classes
- [ ] Task 6: All tests pass, linting clean

---

## NOTES

### Design Decisions

1. **Dual output format**: Returns formatted strings (not raw models) so LLM can present naturally to users. Formatting includes markdown and emoji headers for readability.

2. **No router needed**: Unlike other features, analytics doesn't need direct API endpoints - it's only accessed through the chat agent tool.

3. **Reusing existing helpers**: Import `parse_note`, `extract_title`, and `extract_wikilinks` from existing modules rather than duplicating code.

4. **Algorithmic insights**: The `insights` operation computes insights algorithmically based on vault data rather than calling the LLM. The LLM adds its own interpretation when presenting results to users.

5. **Timezone handling**: All datetime comparisons use UTC to avoid timezone issues. The `ensure_utc` helper handles both naive and aware datetimes gracefully.

### Trade-offs

- **Performance**: Full vault scans on each analytics request. For very large vaults (1000+ notes), consider adding caching or background indexing in future phases.

- **Trends accuracy**: Date-based analytics rely on frontmatter dates which may be missing or inaccurate. Notes without frontmatter are excluded from date-based operations.

- **Insight quality**: Algorithmic insights are limited compared to true AI analysis. The LLM enhances these when presenting to users.

### Future Enhancements

- Add caching for vault metadata to improve performance
- Implement incremental updates instead of full scans
- Support custom date ranges beyond preset periods
- Add graph/connection analytics (most connected notes, isolated clusters)
- Add word count and reading time statistics
