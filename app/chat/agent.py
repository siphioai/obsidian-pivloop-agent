"""PydanticAI agent definition for chat feature."""

from dataclasses import dataclass

from pydantic_ai import Agent

from app.dependencies import VaultClient


@dataclass
class ChatDependencies:
    """Dependencies injected into tools via RunContext."""

    vault: VaultClient
    trace_id: str


SYSTEM_PROMPT = """You are a helpful AI assistant integrated with Obsidian.
You help users manage their knowledge base through conversation.
Currently you can chat; vault tools will be added in future phases."""

chat_agent = Agent(
    "anthropic:claude-haiku-4-5",
    deps_type=ChatDependencies,
    retries=2,
    system_prompt=SYSTEM_PROMPT,
)
