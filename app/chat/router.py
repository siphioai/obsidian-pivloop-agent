"""FastAPI router for /v1/chat/completions endpoint."""

import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic_ai import AgentRunResultEvent, PartDeltaEvent

from app.chat.agent import chat_agent
from app.chat.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    DeltaContent,
    ErrorDetail,
    ErrorResponse,
    StreamChoice,
)
from app.dependencies import ChatDependencies, VaultClient, get_vault_client, logger

router = APIRouter(prefix="/v1", tags=["chat"])


async def process_chat_completion(
    request: ChatCompletionRequest, deps: ChatDependencies
) -> ChatCompletionResponse:
    """Process non-streaming chat completion.

    Args:
        request: The chat completion request
        deps: Dependencies for the chat agent

    Returns:
        OpenAI-compatible chat completion response

    Raises:
        HTTPException: If no user message or agent error occurs
    """
    logger.info(
        "processing_chat",
        extra={"trace_id": deps.trace_id, "messages": len(request.messages)},
    )

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="No user message provided.",
                    type="invalid_request_error",
                    code="missing_user_message",
                )
            ).model_dump(),
        )

    try:
        user_content = user_messages[-1].get_text_content()
        result = await chat_agent.run(user_prompt=user_content, deps=deps)
        return ChatCompletionResponse(
            id=f"chatcmpl-{deps.trace_id}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    message=ChatMessage(role="assistant", content=result.output),
                    finish_reason="stop",
                )
            ],
        )
    except Exception as e:
        logger.error(
            "chat_failed",
            extra={"trace_id": deps.trace_id, "error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error=ErrorDetail(message=f"Agent error: {e!s}", code="agent_error")
            ).model_dump(),
        )


async def stream_chat_completion(
    request: ChatCompletionRequest, deps: ChatDependencies
) -> AsyncGenerator[str, None]:
    """Stream chat completion using PydanticAI's run_stream_events().

    Args:
        request: The chat completion request
        deps: Dependencies for the chat agent

    Yields:
        Server-sent events in OpenAI streaming format
    """
    logger.info(
        "processing_chat_stream",
        extra={"trace_id": deps.trace_id, "messages": len(request.messages)},
    )

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        error = ErrorResponse(
            error=ErrorDetail(
                message="No user message provided.",
                type="invalid_request_error",
                code="missing_user_message",
            )
        )
        yield f"data: {error.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        return

    try:
        user_content = user_messages[-1].get_text_content()
        chunk_id = f"chatcmpl-{deps.trace_id}"
        created = int(time.time())

        # Send initial role chunk
        role_chunk = ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=request.model,
            choices=[StreamChoice(delta=DeltaContent(role="assistant"))],
        )
        yield f"data: {role_chunk.model_dump_json()}\n\n"

        # Stream content deltas
        async for event in chat_agent.run_stream_events(user_prompt=user_content, deps=deps):
            if isinstance(event, PartDeltaEvent):
                if hasattr(event.delta, "content_delta") and event.delta.content_delta:
                    content_chunk = ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=request.model,
                        choices=[
                            StreamChoice(delta=DeltaContent(content=event.delta.content_delta))
                        ],
                    )
                    yield f"data: {content_chunk.model_dump_json()}\n\n"
            elif isinstance(event, AgentRunResultEvent):
                final_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=request.model,
                    choices=[StreamChoice(delta=DeltaContent(), finish_reason="stop")],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(
            "chat_stream_failed",
            extra={"trace_id": deps.trace_id, "error": str(e)},
            exc_info=True,
        )
        error = ErrorResponse(
            error=ErrorDetail(message=f"Streaming error: {e!s}", code="streaming_error")
        )
        yield f"data: {error.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    req: Request,
    vault: VaultClient = Depends(get_vault_client),
) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-compatible chat completion endpoint.

    Args:
        request: Chat completion request body
        req: FastAPI request object
        vault: VaultClient dependency

    Returns:
        OpenAI-compatible chat completion response (or streaming response)
    """
    trace_id = req.headers.get("X-Trace-Id", str(uuid.uuid4()))
    deps = ChatDependencies(vault=vault, trace_id=trace_id)

    # Log whether streaming is requested
    logger.info(
        "chat_request_received",
        extra={"trace_id": trace_id, "stream": request.stream, "model": request.model},
    )

    if request.stream:
        return StreamingResponse(
            stream_chat_completion(request, deps),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    return await process_chat_completion(request, deps)
