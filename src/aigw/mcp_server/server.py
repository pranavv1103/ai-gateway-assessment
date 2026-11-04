"""MCP server definition: tool listing and call dispatch with protocol-correct errors.

Error semantics:
* Unknown tool / invalid arguments -> JSON-RPC error -32602 (Invalid params).
* Unexpected internal failure     -> JSON-RPC error -32603 (Internal error), sanitized.
* Business-logic failure          -> successful JSON-RPC response with ``isError: true``.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mcp import types
from mcp.server.lowlevel import Server
from mcp.shared.exceptions import McpError
from pydantic import BaseModel, ValidationError

from aigw.mcp_server import tools
from aigw.mcp_server.models import (
    GetCustomerRecordInput,
    TriggerRefundInput,
    tool_input_schema,
)

logger = logging.getLogger(__name__)

SERVER_NAME = "customer-support-mcp"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    model: type[BaseModel]
    handler: Callable[[Any], dict[str, Any]]


TOOL_SPECS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            "get_customer_record",
            "Look up a customer record by customer_id (format CUST-XXXXX).",
            GetCustomerRecordInput,
            tools.get_customer_record,
        ),
        ToolSpec(
            "trigger_refund",
            "Queue a refund for a customer. amount must be positive; reason needs 10+ characters.",
            TriggerRefundInput,
            tools.trigger_refund,
        ),
    )
}


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(p) for p in err["loc"]) or "arguments"
        parts.append(f"{location}: {err['msg']}")
    return "Invalid arguments: " + "; ".join(parts)


def _invalid_params(message: str) -> McpError:
    return McpError(types.ErrorData(code=types.INVALID_PARAMS, message=message))


def execute_tool(name: str, arguments: Any) -> types.CallToolResult:
    """Validate arguments and run a tool. Raises McpError for protocol-level failures."""
    spec = TOOL_SPECS.get(name)
    if spec is None:
        raise _invalid_params(f"Unknown tool: {name}")
    if not isinstance(arguments, dict):
        raise _invalid_params("Invalid arguments: arguments must be an object")
    try:
        parsed = spec.model.model_validate(arguments)
    except ValidationError as exc:
        raise _invalid_params(_format_validation_error(exc)) from None
    try:
        payload = spec.handler(parsed)
    except tools.ToolExecutionError as exc:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=str(exc))], isError=True
        )
    except Exception:
        logger.exception("Unexpected failure in tool %s", name)
        raise McpError(
            types.ErrorData(code=types.INTERNAL_ERROR, message="Internal error")
        ) from None
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload,
        isError=False,
    )


def build_server() -> Server:
    server: Server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=spec.name,
                description=spec.description,
                inputSchema=tool_input_schema(spec.model),
            )
            for spec in TOOL_SPECS.values()
        ]

    async def call_tool(req: types.CallToolRequest) -> types.ServerResult:
        # Registered directly (not via @call_tool) so validation failures surface as
        # JSON-RPC errors instead of being folded into isError results by the SDK.
        return types.ServerResult(execute_tool(req.params.name, req.params.arguments))

    server.request_handlers[types.CallToolRequest] = call_tool
    return server
