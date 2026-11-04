"""Subprocess tests: real stdio transport, raw JSON-RPC, and stdout isolation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "0"},
    },
}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}


class StdioSession:
    def __init__(self, extra_env: dict[str, str] | None = None):
        env = {**os.environ, "AIGW_LOG_LEVEL": "DEBUG", **(extra_env or {})}
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "aigw.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=ROOT,
            env=env,
        )
        self.raw_stdout_lines: list[str] = []

    def send(self, message: dict | str) -> None:
        line = message if isinstance(message, str) else json.dumps(message)
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def receive(self) -> dict:
        line = self.proc.stdout.readline()
        assert line, "server closed stdout unexpectedly"
        self.raw_stdout_lines.append(line)
        return json.loads(line)

    def request(self, message: dict) -> dict:
        self.send(message)
        return self.receive()

    def handshake(self) -> dict:
        result = self.request(INIT)
        self.send(INITIALIZED)
        return result

    def close(self) -> str:
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.raw_stdout_lines.extend(self.proc.stdout.read().splitlines(keepends=True))
        stderr = self.proc.stderr.read()
        self.proc.stdout.close()
        self.proc.stderr.close()
        return stderr


@pytest.fixture
def session():
    s = StdioSession()
    yield s
    if s.proc.poll() is None:
        s.close()


def call(id_, name, arguments):
    return {
        "jsonrpc": "2.0",
        "id": id_,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def test_handshake_and_tool_listing(session):
    init = session.handshake()
    assert init["jsonrpc"] == "2.0" and init["id"] == 1
    assert init["result"]["serverInfo"]["name"] == "customer-support-mcp"
    listing = session.request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert {t["name"] for t in listing["result"]["tools"]} == {
        "get_customer_record",
        "trigger_refund",
    }


def test_valid_call_over_stdio(session):
    session.handshake()
    response = session.request(call(7, "get_customer_record", {"customer_id": "CUST-12345"}))
    assert response["id"] == 7
    assert response["result"]["isError"] is False
    assert response["result"]["structuredContent"]["customer_id"] == "CUST-12345"


@pytest.mark.parametrize(
    "arguments",
    [
        {"customer_id": "nope"},
        {},
        {"customer_id": "CUST-12345", "x": 1},
    ],
)
def test_invalid_call_returns_jsonrpc_error_32602(session, arguments):
    session.handshake()
    response = session.request(call("req-1", "get_customer_record", arguments))
    assert response["id"] == "req-1"
    assert response["error"]["code"] == -32602
    assert "result" not in response


def test_refund_nan_and_infinity_literals_rejected(session):
    session.handshake()
    for literal in ("NaN", "Infinity", "-Infinity"):
        session.send(
            '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"trigger_refund",'
            '"arguments":{"customer_id":"CUST-12345","amount":'
            + literal
            + ',"reason":"valid reason here"}}}'
        )
        response = session.receive()
        assert response["error"]["code"] in (-32602, -32700)


def test_unknown_method_is_method_not_found(session):
    session.handshake()
    response = session.request({"jsonrpc": "2.0", "id": 3, "method": "does/not/exist"})
    assert response["error"]["code"] == -32601


def test_stdout_contains_only_jsonrpc_and_logs_go_to_stderr(session):
    session.handshake()
    session.request(call(2, "get_customer_record", {"customer_id": "CUST-00042"}))
    session.request(call(3, "trigger_refund", {"customer_id": "bad", "amount": -1, "reason": "x"}))
    session.send("this is not json")
    session.send(json.dumps({"jsonrpc": "2.0", "id": 4, "method": "ping"}))
    # The SDK may emit a notifications/message log for the unparsable line; skip such notifications.
    ping = session.receive()
    while "id" not in ping:
        assert ping["method"] == "notifications/message"
        ping = session.receive()
    assert ping["id"] == 4
    stderr = session.close()

    assert session.raw_stdout_lines
    for line in session.raw_stdout_lines:
        message = json.loads(line)
        assert message["jsonrpc"] == "2.0"
        assert line.endswith("\n") and line.count("\n") == 1
    assert "MCP server ready on stdio" in stderr
    assert "CallToolRequest" in stderr


NOISY_LAUNCHER = """
import dataclasses
from aigw.mcp_server import server
from aigw.mcp_server.__main__ import main

spec = server.TOOL_SPECS["get_customer_record"]

def noisy(args):
    print("STRAY PRINT FROM TOOL")
    return spec.handler(args)

server.TOOL_SPECS["get_customer_record"] = dataclasses.replace(spec, handler=noisy)
main()
"""


def test_stray_print_in_tool_code_cannot_pollute_stdout(tmp_path):
    launcher = tmp_path / "noisy_launcher.py"
    launcher.write_text(NOISY_LAUNCHER)
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, str(launcher)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
    )
    messages = [INIT, INITIALIZED, call(2, "get_customer_record", {"customer_id": "CUST-12345"})]
    stdin_payload = "".join(json.dumps(m) + "\n" for m in messages)
    proc.stdin.write(stdin_payload)
    proc.stdin.flush()
    out_lines = [proc.stdout.readline(), proc.stdout.readline()]
    rest, stderr = proc.communicate(timeout=10)
    out_lines += rest.splitlines(keepends=True)
    parsed = [json.loads(line) for line in out_lines if line.strip()]
    assert [m["id"] for m in parsed] == [1, 2]
    assert parsed[1]["result"]["isError"] is False
    assert "STRAY PRINT FROM TOOL" not in "".join(out_lines)
    assert "STRAY PRINT FROM TOOL" in stderr


async def test_official_client_over_stdio_subprocess():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-m", "aigw.mcp_server"], cwd=ROOT)
    async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
        await client.initialize()
        result = await client.call_tool("get_customer_record", {"customer_id": "CUST-77777"})
        assert result.isError is False
        assert result.structuredContent["customer_id"] == "CUST-77777"
