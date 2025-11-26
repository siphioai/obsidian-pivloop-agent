"""Tests for chat feature."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.chat.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ContentPart,
    DeltaContent,
    ErrorDetail,
    ErrorResponse,
    StreamChoice,
)


class TestChatMessage:
    """Tests for ChatMessage model."""

    def test_string_content(self) -> None:
        """Test message with string content."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.get_text_content() == "Hello"

    def test_content_parts_list(self) -> None:
        """Test message with content parts list."""
        parts = [
            ContentPart(type="text", text="Hello"),
            ContentPart(type="text", text="World"),
        ]
        msg = ChatMessage(role="user", content=parts)
        assert msg.get_text_content() == "Hello\nWorld"

    def test_content_parts_filters_non_text(self) -> None:
        """Test that non-text content parts are filtered."""
        parts = [
            ContentPart(type="text", text="Hello"),
            ContentPart(type="image_url", text=None),
        ]
        msg = ChatMessage(role="user", content=parts)
        assert msg.get_text_content() == "Hello"

    def test_empty_content(self) -> None:
        """Test message with empty string content."""
        msg = ChatMessage(role="user", content="")
        assert msg.get_text_content() == ""

    def test_empty_parts_list(self) -> None:
        """Test message with empty content parts."""
        msg = ChatMessage(role="user", content=[])
        assert msg.get_text_content() == ""


class TestChatCompletionRequest:
    """Tests for ChatCompletionRequest model."""

    def test_defaults(self) -> None:
        """Test default values."""
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")])
        assert req.model == "pivloop-agent"
        assert req.max_tokens == 2048
        assert req.stream is False
        assert req.temperature == 0.7

    def test_max_tokens_validation(self) -> None:
        """Test max_tokens validation rejects invalid values."""
        with pytest.raises(ValueError):
            ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")], max_tokens=0)

    def test_max_tokens_upper_bound(self) -> None:
        """Test max_tokens validation rejects values over limit."""
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Hi")], max_tokens=20000
            )

    def test_temperature_validation_lower(self) -> None:
        """Test temperature validation rejects negative values."""
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Hi")], temperature=-0.1
            )

    def test_temperature_validation_upper(self) -> None:
        """Test temperature validation rejects values over 2."""
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Hi")], temperature=3.0
            )


class TestStreamingModels:
    """Tests for streaming-related models."""

    def test_delta_content_role(self) -> None:
        """Test DeltaContent with role."""
        delta = DeltaContent(role="assistant")
        assert delta.role == "assistant"
        assert delta.content is None

    def test_delta_content_content(self) -> None:
        """Test DeltaContent with content."""
        delta = DeltaContent(content="Hi")
        assert delta.content == "Hi"
        assert delta.role is None

    def test_delta_content_empty(self) -> None:
        """Test empty DeltaContent."""
        delta = DeltaContent()
        assert delta.role is None
        assert delta.content is None

    def test_stream_choice(self) -> None:
        """Test StreamChoice model."""
        choice = StreamChoice(delta=DeltaContent(content="test"))
        assert choice.delta.content == "test"
        assert choice.finish_reason is None
        assert choice.index == 0

    def test_stream_choice_with_finish_reason(self) -> None:
        """Test StreamChoice with finish_reason."""
        choice = StreamChoice(delta=DeltaContent(), finish_reason="stop")
        assert choice.finish_reason == "stop"

    def test_chunk_serialization(self) -> None:
        """Test ChatCompletionChunk serializes correctly."""
        chunk = ChatCompletionChunk(
            id="test",
            created=0,
            model="m",
            choices=[StreamChoice(delta=DeltaContent(content="Hi"))],
        )
        json_str = chunk.model_dump_json()
        assert "chat.completion.chunk" in json_str
        assert "test" in json_str


class TestErrorModels:
    """Tests for error-related models."""

    def test_error_detail_defaults(self) -> None:
        """Test ErrorDetail default values."""
        err = ErrorDetail(message="Test")
        assert err.type == "server_error"
        assert err.code is None

    def test_error_detail_custom_type(self) -> None:
        """Test ErrorDetail with custom type."""
        err = ErrorDetail(message="Bad request", type="invalid_request_error")
        assert err.type == "invalid_request_error"

    def test_error_detail_with_code(self) -> None:
        """Test ErrorDetail with code."""
        err = ErrorDetail(message="Test", code="test_code")
        assert err.code == "test_code"

    def test_error_response(self) -> None:
        """Test ErrorResponse wraps ErrorDetail."""
        resp = ErrorResponse(
            error=ErrorDetail(message="Test", type="invalid_request_error", code="test_code")
        )
        assert resp.error.code == "test_code"
        assert resp.error.message == "Test"

    def test_error_response_serialization(self) -> None:
        """Test ErrorResponse serializes correctly."""
        resp = ErrorResponse(error=ErrorDetail(message="Error occurred"))
        json_str = resp.model_dump_json()
        assert "Error occurred" in json_str
        assert "server_error" in json_str


class TestResponseModels:
    """Tests for response models."""

    def test_choice_defaults(self) -> None:
        """Test Choice default values."""
        choice = Choice(message=ChatMessage(role="assistant", content="Hi"))
        assert choice.index == 0
        assert choice.finish_reason == "stop"

    def test_chat_completion_response(self) -> None:
        """Test ChatCompletionResponse structure."""
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="test-model",
            choices=[Choice(message=ChatMessage(role="assistant", content="Hello!"))],
        )
        assert response.object == "chat.completion"
        assert response.id == "chatcmpl-123"
        assert len(response.choices) == 1


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

    def test_chat_requires_user_message(self, client: TestClient) -> None:
        """Test that chat endpoint requires at least one user message."""
        r = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "system", "content": "Hi"}]},
        )
        assert r.status_code == 400
        assert r.json()["detail"]["error"]["code"] == "missing_user_message"

    def test_streaming_no_user_message(self, client: TestClient) -> None:
        """Test streaming returns error when no user message."""
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "system", "content": "Hi"}],
                "stream": True,
            },
        )
        assert r.status_code == 200
        assert "invalid_request_error" in r.text


class TestChatIntegration:
    """Integration tests for chat functionality."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        with patch("app.chat.router.chat_agent") as mock:
            mock_result = Mock()
            mock_result.output = "Test response"
            mock.run = AsyncMock(return_value=mock_result)
            yield mock

    def test_successful_completion(self, client: TestClient, mock_agent) -> None:
        """Test successful non-streaming chat completion."""
        r = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["choices"][0]["message"]["content"] == "Test response"
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_response_has_id(self, client: TestClient, mock_agent) -> None:
        """Test response includes chatcmpl ID."""
        r = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.json()["id"].startswith("chatcmpl-")

    def test_response_has_model(self, client: TestClient, mock_agent) -> None:
        """Test response includes model name."""
        r = client.post(
            "/v1/chat/completions",
            json={"model": "my-model", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.json()["model"] == "my-model"

    def test_response_object_type(self, client: TestClient, mock_agent) -> None:
        """Test response object type is chat.completion."""
        r = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.json()["object"] == "chat.completion"

    def test_content_parts_supported(self, client: TestClient, mock_agent) -> None:
        """Test that content parts array format is supported."""
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Hello"}],
                    }
                ],
            },
        )
        assert r.status_code == 200

    def test_agent_error_returns_500(self, client: TestClient) -> None:
        """Test that agent errors return 500 with proper error format."""
        with patch("app.chat.router.chat_agent") as mock:
            mock.run = AsyncMock(side_effect=Exception("Test error"))
            r = client.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
            )
            assert r.status_code == 500
            assert r.json()["detail"]["error"]["code"] == "agent_error"
