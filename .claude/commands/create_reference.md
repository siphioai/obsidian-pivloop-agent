# Create On-Demand Reference Guide

Help me create an on-demand reference guide for a specific task type or technology. This guide will be stored in a `reference/` folder and loaded only when working on this specific type of task.

## What I Need

I want to create a reference guide for: **[DESCRIBE TASK TYPE OR TECHNOLOGY]**

Research best practices and conventions from this resource: **[PASTE LINK HERE OR "none" to use only Archon RAG]**

## Instructions for Creating the Reference Guide

Create a concise, actionable reference guide following this structure:

### Required Sections:

1. **Title and Purpose**
   - Clear title describing the task type
   - 1-2 sentence explanation of when to use this guide

2. **Overall Pattern/Structure**
   - High-level overview of the approach or pattern
   - Visual representation if applicable (ASCII diagram, folder structure, etc.)
   - 2-3 sentences explaining the pattern

3. **Step-by-Step Instructions**
   - Break down the task into 3-6 clear steps
   - Each step should have:
     - Clear heading describing what to do
     - Code example showing how to do it
     - 3-5 key rules or requirements for that step
   - Use actual code, not placeholders

4. **Quick Checklist**
   - Bulleted markdown checklist summarizing all steps
   - Use checkbox format: `- [ ] Item`
   - Include critical validation or testing steps

### Critical Requirements:

- **Length: 50-200 lines MAXIMUM** - Must be concise and scannable
- **Code-heavy, explanation-light** - Show more than tell
- **No generic advice** - Specific to this task type and codebase
- **Real examples** - Based on best practices from the provided resource
- **Actionable** - A developer should be able to follow it step-by-step

## Process to Follow:

1. **Query Archon MCP for technology-specific documentation:**
   - Identify all technologies mentioned in the task description (e.g., React, FastAPI, Pydantic, Obsidian API, etc.)
   - For each technology, query Archon's knowledge base:
     - First: `rag_get_available_sources()` to see what documentation is available
     - Then: `rag_search_knowledge_base(query="<technology> <task-relevant-keywords>", source_id="<matching-source-id>", match_count=5)`
     - Also: `rag_search_code_examples(query="<technology> <specific-pattern>", source_id="<matching-source-id>", match_count=3)`
   - Extract up-to-date best practices, patterns, and conventions from the results
   - Use short, focused queries (2-5 keywords) for better RAG results

2. **Research the provided link thoroughly:**
   - Extract key patterns and best practices
   - Identify common steps or structure
   - Note specific conventions or anti-patterns
   - Look for code examples to adapt
   - Cross-reference with Archon documentation to ensure accuracy

3. **Analyze my existing codebase (if applicable):**
   - Check if similar patterns already exist
   - Identify naming conventions to match
   - Find existing examples to reference
   - Ensure consistency with global rules (CLAUDE.md)

4. **Create the guide following the structure above:**
   - Start with the overall pattern
   - Break into clear, numbered steps
   - Include code examples for each step (incorporating patterns from Archon docs)
   - End with a quick checklist

5. **Keep it focused:**
   - This guide is for ONE specific task type only
   - Don't include general principles (those belong in CLAUDE.md)
   - Don't duplicate information from global rules
   - Focus on the step-by-step "how" not the "why"

## Output Format:

Save the guide as `reference/{task_type}_guide.md` with:
- Clear section headers (## Step 1: ...)
- Code blocks with proper syntax highlighting
- Minimal explanatory text (let code speak)
- Practical checklist at the end

## Example Task Types:

- Building API endpoints
- Creating React components
- Adding database models
- Writing integration tests
- Implementing authentication
- Creating CLI commands
- Building custom tools for agents
- Setting up background jobs

Start by researching the provided link now and analyzing the codebase for existing patterns. Then create the focused, actionable reference guide.
