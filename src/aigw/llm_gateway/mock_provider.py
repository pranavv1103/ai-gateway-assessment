"""Mock streaming LLM provider (OpenAI-style SSE) for local runs and tests.

It echoes the last user message as ``"You said: <message>"``. Optional ``mock`` request fields:
``delta_size`` (characters per delta), ``delay_ms`` (pause between deltas),
``fail_after`` (abort the stream after N deltas), ``status`` (respond with this HTTP status).
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def _frame(payload: dict[str, Any] | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {data}\n\n".encode()


def create_mock_llm() -> FastAPI:
    app = FastAPI(title="Mock LLM Provider", docs_url=None, redoc_url=None)
    app.state.open_streams = 0
    app.state.closed_streams = 0
    app.state.request_count = 0

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request):
        body = await request.json()
        app.state.request_count += 1
        options = body.get("mock", {})
        if "status" in options:
            return JSONResponse(
                {"error": {"message": "mock failure"}}, status_code=options["status"]
            )
        messages = body.get("messages", [])
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        text = f"You said: {last_user}"
        size = max(1, int(options.get("delta_size", 4)))
        delay = float(options.get("delay_ms", 0)) / 1000
        fail_after = options.get("fail_after")
        envelope = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": body.get("model", "mock-model"),
        }

        async def generate() -> AsyncIterator[bytes]:
            app.state.open_streams += 1
            try:
                role = {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                yield _frame({**envelope, "choices": [role]})
                for count, start in enumerate(range(0, len(text), size)):
                    if fail_after is not None and count >= fail_after:
                        raise RuntimeError("mock upstream failure")
                    piece = {"index": 0, "delta": {"content": text[start : start + size]}}
                    yield _frame({**envelope, "choices": [{**piece, "finish_reason": None}]})
                    if delay:
                        await asyncio.sleep(delay)
                stop = {"index": 0, "delta": {}, "finish_reason": "stop"}
                yield _frame({**envelope, "choices": [stop]})
                yield _frame("[DONE]")
            finally:
                app.state.open_streams -= 1
                app.state.closed_streams += 1

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app


app = create_mock_llm()
