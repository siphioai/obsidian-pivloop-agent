"""FastAPI router for /v1/chat/completions endpoint."""

import json
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.chat.agent import chat_agent
from app.chat.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
)
from app.dependencies import ChatDependencies, VaultClient, get_vault_client, logger

router = APIRouter(prefix="/v1", tags=["chat"])


async def process_chat_completion(
    request: ChatCompletionRequest, deps: ChatDependencies
) -> ChatCompletionResponse:
    """Process a chat completion request using the agent.

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
            detail={"error": {"message": "No user message", "type": "invalid_request"}},
        )

    try:
        # Extract text content (handles both string and array formats)
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
            detail={"error": {"message": str(e), "type": "server_error"}},
        )


async def stream_chat_completion(
    request: ChatCompletionRequest, deps: ChatDependencies
) -> AsyncGenerator[str, None]:
    """Stream chat completion response in SSE format."""
    logger.info(
        "processing_chat_stream",
        extra={"trace_id": deps.trace_id, "messages": len(request.messages)},
    )

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        error_data = {"error": {"message": "No user message", "type": "invalid_request"}}
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"
        return

    try:
        user_content = user_messages[-1].get_text_content()
        result = await chat_agent.run(user_prompt=user_content, deps=deps)

        # Send the response as a single chunk (simulated streaming)
        chunk = {
            "id": f"chatcmpl-{deps.trace_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": result.output},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        # Send final chunk with finish_reason
        final_chunk = {
            "id": f"chatcmpl-{deps.trace_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(
            "chat_stream_failed",
            extra={"trace_id": deps.trace_id, "error": str(e)},
            exc_info=True,
        )
        error_data = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"
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
