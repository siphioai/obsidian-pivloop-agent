"""Shared pytest fixtures."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load environment variables before importing app modules
load_dotenv()

# Set test defaults if not provided
if not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"
if not os.environ.get("VAULT_PATH"):
    os.environ["VAULT_PATH"] = "/tmp/test-vault"

from fastapi.testclient import TestClient  # noqa: E402

from app.chat.agent import ChatDependencies  # noqa: E402
from app.dependencies import VaultClient  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture
def mock_vault_path(tmp_path: Path) -> Path:
    """Create a temporary vault directory with a test file."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "test.md").write_text("# Test")
    return vault


@pytest.fixture
def mock_vault_client(mock_vault_path: Path) -> VaultClient:
    """Create a VaultClient with temporary vault path."""
    return VaultClient(vault_path=mock_vault_path)


@pytest.fixture
def chat_deps(mock_vault_client: VaultClient) -> ChatDependencies:
    """Create ChatDependencies with mock vault client."""
    return ChatDependencies(vault=mock_vault_client, trace_id="test-123")


@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI test client."""
    return TestClient(app)
