"""FastAPI streaming LLM gateway: proxies SSE from a provider and redacts PII in flight."""

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from aigw.config import LLMGatewaySettings
from aigw.llm_gateway.sse import SSEParser, SSEProtocolError, encode_event
from aigw.llm_gateway.transform import ChunkTransformer

logger = logging.getLogger(__name__)

STREAM_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1, max_length=200)
    messages: list[dict[str, Any]] = Field(min_length=1, max_length=1000)
    stream: bool | None = None


def gateway_error(status: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"type": error_type, "message": message}},
        status_code=status,
    )


async def redacted_stream(upstream: httpx.Response, max_event_chars: int) -> AsyncIterator[bytes]:
    transformer = ChunkTransformer()
    parser = SSEParser(max_buffered_chars=max_event_chars)
    try:
        async for chunk in upstream.aiter_bytes():
            for event in parser.feed(chunk):
                for frame in transformer.process_event(event):
                    yield frame
            if transformer.done:
                break
        for frame in transformer.finish():
            yield frame
    except (httpx.HTTPError, SSEProtocolError) as exc:
        logger.warning("Upstream stream failed: %s", type(exc).__name__)
        for frame in transformer.finish():
            yield frame
        error = {"error": {"type": "upstream_error", "message": "Upstream stream interrupted"}}
        yield encode_event(json.dumps(error), event="error")
    finally:
        # Runs on normal end, upstream failure, and client disconnect (cancellation).
        with anyio.CancelScope(shield=True):
            await upstream.aclose()
        if transformer.malformed_events:
            logger.warning("Dropped %d malformed SSE events", transformer.malformed_events)


def create_app(
    settings: LLMGatewaySettings,
    client: httpx.AsyncClient | None = None,
    provider_api_key: str | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if client is not None:
            yield
            return
        timeout = httpx.Timeout(
            settings.provider_timeout_seconds, connect=settings.connect_timeout_seconds
        )
        app.state.client = httpx.AsyncClient(timeout=timeout)
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(title="Streaming LLM Gateway", lifespan=lifespan)
    if client is not None:
        app.state.client = client

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(body: ChatRequest, request: Request):
        http: httpx.AsyncClient = request.app.state.client
        payload = body.model_dump()
        payload["stream"] = True  # the gateway always streams; non-streaming is not supported
        headers = {"accept": "text/event-stream", "accept-encoding": "identity"}
        if provider_api_key:
            headers["authorization"] = f"Bearer {provider_api_key}"
        try:
            upstream = await http.send(
                http.build_request("POST", settings.provider_url, json=payload, headers=headers),
                stream=True,
            )
        except httpx.TimeoutException:
            return gateway_error(504, "upstream_timeout", "Upstream provider timed out")
        except httpx.HTTPError:
            logger.warning("Upstream connection failed")
            return gateway_error(502, "upstream_error", "Upstream provider unavailable")

        content_type = upstream.headers.get("content-type", "")
        if upstream.status_code != 200 or "text/event-stream" not in content_type:
            status = upstream.status_code
            await upstream.aclose()
            if status == 429:
                return gateway_error(429, "upstream_rate_limited", "Upstream provider rate limited")
            logger.warning("Unexpected upstream response: status=%s", status)
            return gateway_error(502, "upstream_error", "Upstream provider error")

        return StreamingResponse(
            redacted_stream(upstream, settings.max_event_chars),
            media_type="text/event-stream",
            headers=STREAM_HEADERS,
        )

    return app


def build_app_from_env() -> FastAPI:
    return create_app(
        LLMGatewaySettings.from_env(), provider_api_key=os.environ.get("AIGW_LLM_PROVIDER_API_KEY")
    )
