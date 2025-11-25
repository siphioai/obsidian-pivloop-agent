# Pydantic AI Agent Tools Reference

**Purpose**: Use this guide when building tools for Pydantic AI agents to enable function calling capabilities.

## Overall Pattern

Pydantic AI tools allow agents to perform actions and retrieve information during execution. There are three main registration methods:

```
Agent Creation → Tool Registration → Model Execution → Tool Calls → Final Response
     ↓               ↓                     ↓              ↓            ↓
  Define deps   @agent.tool or      LLM requests    Execute &     Return result
  & context    tools=[] param       tool usage      return data    to user
```

**Key Concepts**:
- Tools need `RunContext` to access dependencies (database, API clients, etc.)
- Tools without context use `@agent.tool_plain` decorator
- Docstrings auto-generate parameter descriptions for the model
- Tools return JSON-serializable data

## Step 1: Define Agent with Dependencies

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from httpx import AsyncClient

@dataclass
class Deps:
    client: AsyncClient
    db: DatabaseConnection

agent = Agent(
    'openai:gpt-4',
    deps_type=Deps,
    retries=2,
    system_prompt='You are a helpful assistant with access to tools.'
)
```

**Rules**:
- Use `deps_type` to specify dependency type (can be any Python type)
- Dependencies passed at runtime via `agent.run_sync('prompt', deps=deps_instance)`
- Use dataclass or Pydantic models for structured dependencies
- Set `retries` for automatic retry on tool failures

## Step 2: Register Context-Aware Tools

```python
from pydantic import BaseModel

class LatLng(BaseModel):
    lat: float
    lng: float

@agent.tool
async def get_coordinates(ctx: RunContext[Deps], location: str) -> LatLng:
    """Get latitude and longitude for a location.

    Args:
        ctx: The context containing dependencies
        location: A location description like "London" or "New York"
    """
    response = await ctx.deps.client.get(
        'https://api.geocoding.service/search',
        params={'q': location}
    )
    response.raise_for_status()
    return LatLng.model_validate_json(response.content)
```

**Rules**:
- First parameter MUST be `ctx: RunContext[YourDepsType]`
- Use google, numpy, or sphinx docstring format for parameter descriptions
- Descriptions extracted from docstrings become part of tool schema
- Return Pydantic models for type-safe, validated responses
- Can be sync or async functions

## Step 3: Register Plain Tools (No Context)

```python
import random

@agent.tool_plain
def roll_dice(sides: int = 6) -> int:
    """Roll a die with the specified number of sides.

    Args:
        sides: Number of sides on the die (default: 6)
    """
    return random.randint(1, sides)
```

**Rules**:
- Use when tool doesn't need access to agent context/dependencies
- No `RunContext` parameter required
- Still requires docstrings for parameter descriptions
- Simpler, more portable tool definitions

## Step 4: Alternative - Register via Constructor

```python
def fetch_data(ctx: RunContext[Deps], query: str) -> dict:
    """Fetch data from external API."""
    return ctx.deps.client.get(f'/api/data?q={query}').json()

def calculate(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

# Method 1: Simple list
agent = Agent(
    'openai:gpt-4',
    deps_type=Deps,
    tools=[fetch_data, calculate]  # Auto-detects RunContext usage
)

# Method 2: Explicit Tool instances for fine control
from pydantic_ai import Tool

agent = Agent(
    'openai:gpt-4',
    deps_type=Deps,
    tools=[
        Tool(fetch_data, takes_ctx=True, name='data_fetcher'),
        Tool(calculate, takes_ctx=False, retries=3)
    ]
)
```

**Rules**:
- Pass functions directly or wrap in `Tool()` for customization
- `Tool()` allows custom `name`, `description`, `retries`, `prepare` callback
- Function signature inspection determines if tool takes context
- Useful for reusable tools across multiple agents

## Step 5: Define Tool Schema with Type Hints

```python
from typing import Literal

class SearchParams(BaseModel):
    """Parameters for searching the database."""
    query: str
    category: Literal['users', 'posts', 'comments']
    limit: int = 10

@agent.tool
async def search_db(ctx: RunContext[Deps], params: SearchParams) -> list[dict]:
    """Search the database with specified parameters.

    Args:
        ctx: Context with database connection
        params: Search parameters including query, category, and limit
    """
    results = await ctx.deps.db.search(
        params.query,
        category=params.category,
        limit=params.limit
    )
    return results
```

**Rules**:
- Use Pydantic models for complex parameter objects
- Single object parameters flatten into tool schema automatically
- Leverage type hints: `Literal`, `Union`, `Optional` for precise schemas
- Model docstring becomes tool description
- Field descriptions from model add to parameter schema

## Step 6: Run Agent with Tools

```python
async def main():
    async with AsyncClient() as client:
        deps = Deps(client=client, db=get_db_connection())

        result = await agent.run(
            'What is the weather in London?',
            deps=deps
        )

        print(result.output)
        # Access tool calls made during execution
        for message in result.all_messages():
            print(message)
```

**Rules**:
- Always provide `deps` if agent has `deps_type` defined
- Use `run()` for async, `run_sync()` for synchronous execution
- Access conversation history via `result.all_messages()`
- Tool calls appear as `ToolCallPart` and `ToolReturnPart` in messages
- Multiple tool calls can happen in sequence automatically

## Quick Checklist

- [ ] Define agent with appropriate model and `deps_type`
- [ ] Create dependency container (dataclass/model) with API clients, DB connections
- [ ] Register tools using `@agent.tool` (with context) or `@agent.tool_plain` (without)
- [ ] Write comprehensive docstrings with Args sections for all parameters
- [ ] Use Pydantic models for return types and complex parameters
- [ ] Use type hints (`int`, `str`, `Literal`, etc.) for clear schemas
- [ ] Test tool execution: `agent.run_sync('test prompt', deps=test_deps)`
- [ ] Verify tool calls in messages: `result.all_messages()`
- [ ] Handle errors with try/except in tool functions
- [ ] Set appropriate `retries` for flaky external calls

## Common Patterns

**Multi-step tool chain**:
```python
# Agent automatically calls tools in sequence
# 1. get_coordinates("London") → {lat: 51.5, lng: -0.1}
# 2. get_weather(51.5, -0.1) → {temp: 15, desc: "Cloudy"}
# 3. Final response: "It's 15°C and cloudy in London"
```

**Tool with validation**:
```python
@agent.tool
async def get_user(ctx: RunContext[Deps], user_id: int) -> dict | None:
    """Fetch user by ID. Returns None if not found."""
    if user_id < 1:
        raise ValueError("user_id must be positive")
    return await ctx.deps.db.get_user(user_id)
```

**Conditional tool registration** (advanced):
```python
from pydantic_ai.tools import ToolDefinition

async def prep_admin_tool(
    ctx: RunContext[Deps],
    tool_def: ToolDefinition
) -> ToolDefinition | None:
    # Only register if user is admin
    if ctx.deps.user.is_admin:
        return tool_def
    return None

agent = Agent(
    'openai:gpt-4',
    tools=[Tool(admin_function, prepare=prep_admin_tool)]
)
```
