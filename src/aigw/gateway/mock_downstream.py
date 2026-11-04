"""Mock downstream MCP server over HTTP JSON-RPC (for local runs and tests)."""

from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

TOOLS = [
    {"name": "get_status", "description": "Service status", "inputSchema": {"type": "object"}},
    {"name": "echo", "description": "Echo text", "inputSchema": {"type": "object"}},
    {
        "name": "admin_reset_key",
        "description": "Rotate an API key (admin only)",
        "inputSchema": {"type": "object"},
    },
]


def _handle(message: dict[str, Any]) -> dict[str, Any] | None:
    request_id = message.get("id")
    method = message.get("method")
    if "id" not in message:
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name not in {tool["name"] for tool in TOOLS}:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32602, "message": f"Unknown tool: {name}"},
            }
        text = f"{name} executed with {arguments}"
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"content": [{"type": "text", "text": text}], "isError": False},
        }
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": "Method not found"},
    }


def create_mock_downstream() -> FastAPI:
    app = FastAPI(title="Mock MCP Server", docs_url=None, redoc_url=None)
    app.state.received = []

    @app.post("/mcp")
    async def mcp(request: Request) -> Response:
        payload = await request.json()
        app.state.received.append(payload)
        if isinstance(payload, list):
            responses = [r for r in (_handle(m) for m in payload) if r is not None]
            return JSONResponse(responses) if responses else Response(status_code=202)
        response = _handle(payload)
        return JSONResponse(response) if response is not None else Response(status_code=202)

    return app


app = create_mock_downstream()
