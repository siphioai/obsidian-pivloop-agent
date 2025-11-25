"""Tests for chat feature."""

import pytest
from fastapi.testclient import TestClient

from app.chat.models import ChatCompletionRequest, ChatMessage


class TestModels:
    """Tests for chat models."""

    def test_chat_message(self) -> None:
        """Test creating a valid chat message."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_empty_content_fails(self) -> None:
        """Test that empty content raises validation error."""
        with pytest.raises(ValueError):
            ChatMessage(role="user", content="")

    def test_chat_completion_request(self) -> None:
        """Test creating a valid chat completion request."""
        request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hello")])
        assert len(request.messages) == 1
        assert request.model == "pivloop-agent"
        assert request.max_tokens == 2048


class TestEndpoints:
    """Tests for API endpoints."""

    def test_health(self, client: TestClient) -> None:
        """Test health endpoint returns healthy status."""
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_root(self, client: TestClient) -> None:
        """Test root endpoint returns API info."""
        r = client.get("/")
        assert r.status_code == 200
        assert "PivLoop" in r.json()["name"]

    def test_chat_requires_messages(self, client: TestClient) -> None:
        """Test that chat endpoint requires messages field."""
        r = client.post("/v1/chat/completions", json={"model": "test"})
        assert r.status_code == 422
