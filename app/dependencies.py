"""Shared dependencies: VaultClient and structured logger."""

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import get_settings


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        data: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        if extra := getattr(record, "extra", {}):
            data.update(extra)
        return json.dumps(data)


def setup_logging() -> logging.Logger:
    """Configure and return the application logger."""
    settings = get_settings()
    logger = logging.getLogger("pivloop_agent")
    logger.setLevel(getattr(logging, settings.log_level.upper()))
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger


logger = setup_logging()


class VaultError(Exception):
    """Base exception for vault operations."""

    pass


class VaultNotFoundError(VaultError):
    """Raised when a file is not found in the vault."""

    pass


class VaultSecurityError(VaultError):
    """Raised when a security violation is detected."""

    pass


@dataclass
class VaultClient:
    """Client for interacting with Obsidian vault."""

    vault_path: Path

    def _validate_path(self, relative_path: str) -> Path:
        """Validate and resolve a path within the vault.

        Args:
            relative_path: Relative path within the vault

        Returns:
            Resolved absolute path

        Raises:
            VaultSecurityError: If path traversal is detected
        """
        full_path = (self.vault_path / relative_path).resolve()
        if not full_path.is_relative_to(self.vault_path.resolve()):
            raise VaultSecurityError(f"Path traversal detected: {relative_path}")
        return full_path

    async def read_file(self, path: str) -> str:
        """Read a file from the vault.

        Args:
            path: Relative path to file

        Returns:
            File content as string

        Raises:
            VaultNotFoundError: If file does not exist
        """
        full_path = self._validate_path(path)
        if not full_path.exists():
            raise VaultNotFoundError(f"File not found: {path}")
        return full_path.read_text(encoding="utf-8")

    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the vault.

        Args:
            path: Relative path to file
            content: Content to write
        """
        full_path = self._validate_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    async def delete_file(self, path: str) -> None:
        """Delete a file from the vault.

        Args:
            path: Relative path to file

        Raises:
            VaultNotFoundError: If file does not exist
            VaultSecurityError: If path traversal detected
        """
        full_path = self._validate_path(path)
        if not full_path.exists():
            raise VaultNotFoundError(f"File not found: {path}")
        full_path.unlink()

    async def file_exists(self, path: str) -> bool:
        """Check if a file exists in the vault.

        Args:
            path: Relative path to file

        Returns:
            True if file exists, False otherwise
        """
        try:
            full_path = self._validate_path(path)
            return full_path.exists()
        except VaultSecurityError:
            return False

    async def list_files(self, folder: str = "", pattern: str = "*.md") -> list[str]:
        """List files in the vault matching a pattern.

        Args:
            folder: Folder to search in (empty for root)
            pattern: Glob pattern for files

        Returns:
            List of relative file paths
        """
        base = self._validate_path(folder) if folder else self.vault_path
        return sorted(
            str(f.relative_to(self.vault_path)) for f in base.rglob(pattern) if f.is_file()
        )

    async def get_stats(self) -> dict[str, str | int]:
        """Get vault statistics.

        Returns:
            Dictionary with vault path and note count
        """
        return {"vault_path": str(self.vault_path), "note_count": len(await self.list_files())}


async def get_vault_client() -> AsyncIterator[VaultClient]:
    """FastAPI dependency provider for VaultClient."""
    yield VaultClient(vault_path=get_settings().vault_path)


@dataclass
class ChatDependencies:
    """Dependencies injected into agent tools via RunContext.

    This is defined here to avoid circular imports between chat.agent
    and notes.tools modules.
    """

    vault: VaultClient
    trace_id: str
