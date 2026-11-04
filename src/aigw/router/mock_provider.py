"""Deterministic mock completion provider; behavior is controlled per request via ``mock``."""

import asyncio

from fastapi import FastAPI, Request


def create_mock_provider(name: str = "mock") -> FastAPI:
    app = FastAPI(title=f"Mock completion provider ({name})")
    app.state.request_count = 0

    @app.post("/v1/completions")
    async def completions(request: Request):
        app.state.request_count += 1
        body = await request.json()
        mock = (body.get("mock") or {}).get(name) or {}
        if mock.get("delay_ms"):
            await asyncio.sleep(mock["delay_ms"] / 1000)
        if mock.get("status", 200) != 200:
            from fastapi.responses import JSONResponse

            return JSONResponse({"error": "mock failure"}, status_code=mock["status"])
        prompt = str(body.get("prompt", ""))
        text = f"[{name}] echo: {prompt}"
        return {
            "text": text,
            "usage": {
                "prompt_tokens": max(1, len(prompt) // 4),
                "completion_tokens": max(1, len(text) // 4),
            },
        }

    return app


primary_app = create_mock_provider("primary")
secondary_app = create_mock_provider("secondary")
