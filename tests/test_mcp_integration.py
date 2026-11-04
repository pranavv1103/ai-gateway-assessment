"""In-process integration tests using the official MCP client against the real server."""

import json

import pytest
from mcp import types
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_connected_server_and_client_session

from aigw.mcp_server.server import build_server
from aigw.mcp_server.tools import NOT_FOUND_CUSTOMER_ID

REASON = "Item arrived damaged in transit"


@pytest.fixture
def connect():
    """Factory so the anyio task group opens and closes inside the test's own task."""
    return lambda: create_connected_server_and_client_session(build_server())


async def test_lists_both_tools_with_strict_schemas(connect):
    async with connect() as client:
        result = await client.list_tools()
        tools = {tool.name: tool for tool in result.tools}
        assert set(tools) == {"get_customer_record", "trigger_refund"}
        for tool in tools.values():
            assert tool.inputSchema["additionalProperties"] is False
        assert tools["get_customer_record"].inputSchema["required"] == ["customer_id"]
        assert tools["trigger_refund"].inputSchema["properties"]["amount"]["type"] == "number"
        assert set(tools["trigger_refund"].inputSchema["required"]) == {
            "customer_id",
            "amount",
            "reason",
        }


async def test_get_customer_record_is_deterministic(connect):
    async with connect() as client:
        first = await client.call_tool("get_customer_record", {"customer_id": "CUST-12345"})
        second = await client.call_tool("get_customer_record", {"customer_id": "CUST-12345"})
        assert first.isError is False
        assert first.structuredContent == second.structuredContent
        assert first.structuredContent["customer_id"] == "CUST-12345"
        assert json.loads(first.content[0].text) == first.structuredContent


async def test_different_customers_differ(connect):
    async with connect() as client:
        a = await client.call_tool("get_customer_record", {"customer_id": "CUST-00001"})
        b = await client.call_tool("get_customer_record", {"customer_id": "CUST-00002"})
        assert a.structuredContent != b.structuredContent


async def test_refund_success_is_deterministic(connect):
    async with connect() as client:
        args = {"customer_id": "CUST-12345", "amount": 19.99, "reason": REASON}
        first = await client.call_tool("trigger_refund", args)
        second = await client.call_tool("trigger_refund", args)
        assert first.isError is False
        assert first.structuredContent == second.structuredContent
        assert first.structuredContent["amount"] == "19.99"
        assert first.structuredContent["refund_id"].startswith("RFD-")


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("get_customer_record", {"customer_id": "bad"}),
        ("get_customer_record", {}),
        ("get_customer_record", {"customer_id": 12345}),
        ("get_customer_record", {"customer_id": "CUST-12345", "extra": "x"}),
        ("trigger_refund", {"customer_id": "CUST-12345", "amount": 0, "reason": REASON}),
        ("trigger_refund", {"customer_id": "CUST-12345", "amount": -5, "reason": REASON}),
        ("trigger_refund", {"customer_id": "CUST-12345", "amount": "5", "reason": REASON}),
        ("trigger_refund", {"customer_id": "CUST-12345", "amount": 5, "reason": "too short"}),
        ("trigger_refund", {"customer_id": "CUST-1", "amount": 5, "reason": REASON}),
        ("trigger_refund", {"customer_id": "CUST-12345", "amount": 5}),
        ("no_such_tool", {}),
    ],
)
async def test_invalid_input_is_jsonrpc_invalid_params(connect, name, arguments):
    async with connect() as client:
        with pytest.raises(McpError) as exc_info:
            await client.call_tool(name, arguments)
        assert exc_info.value.error.code == types.INVALID_PARAMS


async def test_validation_error_does_not_echo_input(connect):
    async with connect() as client:
        secret = "CUST-SECRET-VALUE"
        with pytest.raises(McpError) as exc_info:
            await client.call_tool("get_customer_record", {"customer_id": secret})
        assert secret not in exc_info.value.error.message


async def test_business_failure_is_tool_error_not_protocol_error(connect):
    async with connect() as client:
        result = await client.call_tool(
            "get_customer_record", {"customer_id": NOT_FOUND_CUSTOMER_ID}
        )
        assert result.isError is True
        assert "not found" in result.content[0].text


async def test_refund_over_lifetime_spend_is_tool_error(connect):
    async with connect() as client:
        result = await client.call_tool(
            "trigger_refund", {"customer_id": "CUST-00001", "amount": 99999, "reason": REASON}
        )
        assert result.isError is True
        assert "lifetime spend" in result.content[0].text


async def test_unexpected_handler_failure_is_internal_error_and_sanitized(connect, monkeypatch):
    async with connect() as client:
        from aigw.mcp_server import server as server_module

        def boom(_args):
            raise RuntimeError("db password=hunter2")

        spec = server_module.TOOL_SPECS["get_customer_record"]
        monkeypatch.setitem(
            server_module.TOOL_SPECS,
            "get_customer_record",
            type(spec)(spec.name, spec.description, spec.model, boom),
        )
        with pytest.raises(McpError) as exc_info:
            await client.call_tool("get_customer_record", {"customer_id": "CUST-12345"})
        assert exc_info.value.error.code == types.INTERNAL_ERROR
        assert "hunter2" not in exc_info.value.error.message
