import json

import httpx
import pytest

from aigw.config import GatewaySettings, parse_token_roles
from aigw.gateway.app import create_app
from aigw.gateway.auth import TokenRoleResolver, extract_bearer_token
from aigw.gateway.mock_downstream import create_mock_downstream
from aigw.gateway.policy import is_admin_tool

ADMIN = "tok-admin-0123456789"
VIEWER = "tok-viewer-0123456789"
DOWNSTREAM = "http://downstream.internal.example:9001/mcp"


class Downstream:
    """Counts every request that reaches the downstream server."""

    def __init__(self, handler=None):
        self.requests: list[httpx.Request] = []
        self._handler = handler
        self.mock_app = create_mock_downstream()

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._handler:
            return self._handler(request)
        transport = httpx.ASGITransport(app=self.mock_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
            return await client.post("/mcp", content=request.content, headers=request.headers)

    @property
    def bodies(self):
        return [json.loads(r.content) for r in self.requests]


def make_gateway(downstream: Downstream, **overrides):
    settings = GatewaySettings(
        downstream_url=DOWNSTREAM,
        token_roles={ADMIN: "admin", VIEWER: "viewer"},
        **overrides,
    )
    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(downstream))
    app = create_app(settings, upstream_client)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")


@pytest.fixture
def downstream():
    return Downstream()


@pytest.fixture
async def gateway(downstream):
    async with make_gateway(downstream) as client:
        yield client


def rpc(method, params=None, id_=1):
    message = {"jsonrpc": "2.0", "method": method}
    if id_ is not None:
        message["id"] = id_
    if params is not None:
        message["params"] = params
    return message


def auth(token):
    return {"Authorization": f"Bearer {token}"}


# ---------- unit: auth and policy ----------


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Bearer abc123", "abc123"),
        ("bearer abc123", "abc123"),
        ("BEARER abc123", "abc123"),
        (None, None),
        ("", None),
        ("Bearer", None),
        ("Bearer ", None),
        ("Basic abc123", None),
        ("abc123", None),
        ("Bearer a b", None),
        ("Bearer " + "x" * 513, None),
        ("Bearer tok\x00en", None),
    ],
)
def test_extract_bearer_token(header, expected):
    assert extract_bearer_token(header) == expected


def test_resolver_returns_server_side_role_only():
    resolver = TokenRoleResolver({"a": "admin", "b": "viewer"})
    assert resolver.resolve("a") == "admin"
    assert resolver.resolve("b") == "viewer"
    assert resolver.resolve("c") is None
    assert resolver.resolve("A") is None


def test_token_config_rejects_invalid_roles():
    with pytest.raises(ValueError):
        parse_token_roles('{"t": "superuser"}')
    with pytest.raises(ValueError):
        parse_token_roles("not json")
    with pytest.raises(ValueError):
        parse_token_roles("{}")
    assert parse_token_roles('{"t": "admin"}') == {"t": "admin"}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("admin_reset_key", True),
        ("ADMIN_reset_key", True),
        (" admin_reset", True),
        ("\tadmin_x", True),
        ("ａｄｍｉｎ_x", True),  # full-width look-alike
        ("admin", False),
        ("administrator", False),
        ("get_admin_info", False),
        ("echo", False),
        ("", False),
    ],
)
def test_is_admin_tool(name, expected):
    assert is_admin_tool(name) is expected


# ---------- authentication ----------


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": "Basic abc"},
        {"Authorization": "Bearer"},
    ],
)
async def test_unauthenticated_requests_are_rejected_without_upstream_call(
    gateway, downstream, headers
):
    response = await gateway.post("/mcp", json=rpc("tools/list"), headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == -32002
    assert downstream.requests == []


async def test_error_responses_never_contain_credentials_or_internal_urls(gateway):
    response = await gateway.post(
        "/mcp", json=rpc("tools/list"), headers=auth("wrong-secret-token")
    )
    assert "wrong-secret-token" not in response.text
    assert "downstream" not in response.text.lower()


async def test_auth_checked_before_body_parsing(gateway, downstream):
    response = await gateway.post("/mcp", content=b"{not json", headers={})
    assert response.status_code == 401
    assert downstream.requests == []


# ---------- tools/list ----------


@pytest.mark.parametrize("token", [ADMIN, VIEWER])
async def test_tools_list_forwarded_transparently(gateway, downstream, token):
    response = await gateway.post("/mcp", json=rpc("tools/list", id_="abc"), headers=auth(token))
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "abc"
    assert {t["name"] for t in body["result"]["tools"]} == {"get_status", "echo", "admin_reset_key"}
    assert downstream.bodies == [rpc("tools/list", id_="abc")]


async def test_authorization_header_is_not_forwarded_downstream(gateway, downstream):
    await gateway.post("/mcp", json=rpc("tools/list"), headers=auth(VIEWER))
    assert "authorization" not in downstream.requests[0].headers


# ---------- tools/call authorization ----------


async def test_viewer_can_call_non_admin_tool(gateway, downstream):
    response = await gateway.post(
        "/mcp",
        json=rpc("tools/call", {"name": "echo", "arguments": {"x": 1}}, 5),
        headers=auth(VIEWER),
    )
    assert response.json()["id"] == 5
    assert response.json()["result"]["isError"] is False
    assert len(downstream.requests) == 1


async def test_admin_can_call_admin_tool(gateway, downstream):
    response = await gateway.post(
        "/mcp", json=rpc("tools/call", {"name": "admin_reset_key"}, 9), headers=auth(ADMIN)
    )
    assert response.json()["result"]["isError"] is False
    assert len(downstream.requests) == 1


@pytest.mark.parametrize("id_", [7, "req-7", 0, None])
async def test_viewer_blocked_from_admin_tool_with_id_preserved(gateway, downstream, id_):
    request = {
        "jsonrpc": "2.0",
        "id": id_,
        "method": "tools/call",
        "params": {"name": "admin_reset_key"},
    }
    response = await gateway.post("/mcp", json=request, headers=auth(VIEWER))
    assert response.status_code == 200
    assert response.json() == {
        "jsonrpc": "2.0",
        "id": id_,
        "error": {"code": -32001, "message": "Unauthorized Tool Call"},
    }
    assert downstream.requests == []


@pytest.mark.parametrize(
    "name", ["ADMIN_reset_key", " admin_reset_key", "Admin_Reset", "ａｄｍｉｎ_reset_key"]
)
async def test_admin_prefix_bypass_attempts_blocked(gateway, downstream, name):
    response = await gateway.post(
        "/mcp", json=rpc("tools/call", {"name": name}), headers=auth(VIEWER)
    )
    assert response.json()["error"]["code"] == -32001
    assert downstream.requests == []


async def test_duplicate_key_smuggling_cannot_bypass_policy(gateway, downstream):
    raw = (
        '{"jsonrpc":"2.0","id":1,"method":"tools/call",'
        '"params":{"name":"echo"},"params":{"name":"admin_reset_key"}}'
    )
    response = await gateway.post(
        "/mcp", content=raw, headers={**auth(VIEWER), "content-type": "application/json"}
    )
    assert response.json()["error"]["code"] == -32001
    assert downstream.requests == []


async def test_blocked_notification_is_dropped_silently(gateway, downstream):
    request = rpc("tools/call", {"name": "admin_reset_key"}, id_=None)
    response = await gateway.post("/mcp", json=request, headers=auth(VIEWER))
    assert response.status_code == 202
    assert response.content == b""
    assert downstream.requests == []


async def test_allowed_notification_forwarded_without_response(gateway, downstream):
    request = rpc("notifications/initialized", id_=None)
    response = await gateway.post("/mcp", json=request, headers=auth(VIEWER))
    assert response.status_code == 202
    assert downstream.bodies == [request]


@pytest.mark.parametrize(
    "params", [None, {}, {"name": 5}, {"name": None}, {"name": ["admin_x"]}, []]
)
async def test_malformed_tools_call_params_rejected_locally(gateway, downstream, params):
    response = await gateway.post("/mcp", json=rpc("tools/call", params, 3), headers=auth(ADMIN))
    body = response.json()
    assert body["id"] == 3
    assert body["error"]["code"] == -32602
    assert downstream.requests == []


# ---------- malformed payloads ----------


async def test_invalid_json_returns_parse_error(gateway, downstream):
    response = await gateway.post("/mcp", content=b"{oops", headers=auth(VIEWER))
    assert response.status_code == 400
    assert response.json() == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32700, "message": "Parse error"},
    }
    assert downstream.requests == []


@pytest.mark.parametrize(
    "payload",
    [
        "a string",
        42,
        None,
        True,
        {},
        {"jsonrpc": "1.0", "id": 1, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 1},
        {"jsonrpc": "2.0", "id": 1, "method": 5},
        {"jsonrpc": "2.0", "id": 1, "method": ""},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": "str"},
        {"jsonrpc": "2.0", "id": {"a": 1}, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": True, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 1.5, "method": "tools/list"},
    ],
)
async def test_invalid_requests_rejected(gateway, downstream, payload):
    response = await gateway.post("/mcp", content=json.dumps(payload), headers=auth(VIEWER))
    assert response.status_code in (200, 400)
    assert response.json()["error"]["code"] == -32600
    assert downstream.requests == []


async def test_invalid_request_preserves_valid_id(gateway):
    response = await gateway.post(
        "/mcp", json={"jsonrpc": "2.0", "id": "keep", "method": 5}, headers=auth(VIEWER)
    )
    assert response.json()["id"] == "keep"


async def test_oversized_body_rejected(downstream):
    async with make_gateway(downstream, max_body_bytes=200) as client:
        big = rpc("tools/call", {"name": "echo", "arguments": {"x": "y" * 500}})
        response = await client.post("/mcp", json=big, headers=auth(ADMIN))
    assert response.status_code == 413
    assert downstream.requests == []


async def test_get_method_not_allowed(gateway):
    response = await gateway.get("/mcp", headers=auth(ADMIN))
    assert response.status_code == 405


# ---------- batches ----------


async def test_batch_mixed_authorization(gateway, downstream):
    batch = [
        rpc("tools/list", id_=1),
        rpc("tools/call", {"name": "admin_reset_key"}, id_=2),
        rpc("tools/call", {"name": "echo"}, id_=3),
        rpc("notifications/initialized", id_=None),
        rpc("tools/call", {"name": "admin_reset_key"}, id_=None),
        {"jsonrpc": "2.0", "id": 6},
    ]
    response = await gateway.post("/mcp", json=batch, headers=auth(VIEWER))
    assert response.status_code == 200
    results = {r["id"]: r for r in response.json()}
    assert set(results) == {1, 2, 3, 6}
    assert results[2]["error"]["code"] == -32001
    assert results[6]["error"]["code"] == -32600
    assert "result" in results[1] and "result" in results[3]
    forwarded_methods = [b["method"] for b in downstream.bodies]
    assert sorted(forwarded_methods) == ["notifications/initialized", "tools/call", "tools/list"]
    assert all(b.get("params", {}).get("name") != "admin_reset_key" for b in downstream.bodies)


async def test_batch_of_only_blocked_calls_never_reaches_downstream(gateway, downstream):
    batch = [rpc("tools/call", {"name": "admin_reset_key"}, id_=i) for i in range(5)]
    response = await gateway.post("/mcp", json=batch, headers=auth(VIEWER))
    assert [r["error"]["code"] for r in response.json()] == [-32001] * 5
    assert downstream.requests == []


async def test_batch_all_notifications_returns_202(gateway):
    batch = [rpc("notifications/initialized", id_=None)] * 2
    response = await gateway.post("/mcp", json=batch, headers=auth(VIEWER))
    assert response.status_code == 202


async def test_empty_batch_is_invalid(gateway):
    response = await gateway.post("/mcp", json=[], headers=auth(VIEWER))
    assert response.json()["error"]["code"] == -32600


async def test_batch_with_non_object_members(gateway, downstream):
    response = await gateway.post("/mcp", json=[1, "x", None], headers=auth(VIEWER))
    assert [r["error"]["code"] for r in response.json()] == [-32600] * 3
    assert downstream.requests == []


async def test_oversized_batch_rejected(downstream):
    async with make_gateway(downstream, max_batch_size=2) as client:
        response = await client.post(
            "/mcp", json=[rpc("ping", id_=i) for i in range(3)], headers=auth(ADMIN)
        )
    assert response.status_code == 400
    assert downstream.requests == []


# ---------- downstream failures ----------


async def test_downstream_connection_failure_is_sanitized():
    def refuse(request):
        raise httpx.ConnectError("connect failed to downstream.internal.example:9001 secret-token")

    downstream = Downstream(refuse)
    async with make_gateway(downstream) as client:
        response = await client.post("/mcp", json=rpc("tools/list", id_=11), headers=auth(VIEWER))
    assert response.status_code == 502
    assert response.json() == {
        "jsonrpc": "2.0",
        "id": 11,
        "error": {"code": -32003, "message": "Downstream service unavailable"},
    }
    assert "internal" not in response.text and "secret" not in response.text


async def test_downstream_timeout_is_sanitized():
    def slow(request):
        raise httpx.ReadTimeout("timed out")

    async with make_gateway(Downstream(slow)) as client:
        response = await client.post("/mcp", json=rpc("tools/list"), headers=auth(VIEWER))
    assert response.status_code == 502
    assert response.json()["error"]["code"] == -32003


@pytest.mark.parametrize(
    "downstream_response",
    [
        httpx.Response(500, text="Traceback (most recent call last): File /srv/app.py"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json=["array"]),
        httpx.Response(200, json={"jsonrpc": "2.0", "id": 1}),
        httpx.Response(200, content=b""),
    ],
)
async def test_bad_downstream_responses_are_sanitized(downstream_response):
    async with make_gateway(Downstream(lambda r: downstream_response)) as client:
        response = await client.post("/mcp", json=rpc("tools/list"), headers=auth(VIEWER))
    assert response.status_code == 502
    assert response.json()["error"]["code"] == -32003
    assert "Traceback" not in response.text and "/srv" not in response.text


async def test_downstream_jsonrpc_error_is_passed_through():
    payload = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32602, "message": "Unknown tool: x"}}
    async with make_gateway(Downstream(lambda r: httpx.Response(200, json=payload))) as client:
        response = await client.post(
            "/mcp", json=rpc("tools/call", {"name": "x"}), headers=auth(VIEWER)
        )
    assert response.status_code == 200
    assert response.json() == payload


async def test_session_headers_forwarded_both_ways():
    def handler(request):
        assert request.headers["mcp-session-id"] == "sess-1"
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": 1, "result": {}},
            headers={"mcp-session-id": "sess-1", "set-cookie": "internal=1"},
        )

    async with make_gateway(Downstream(handler)) as client:
        response = await client.post(
            "/mcp", json=rpc("ping"), headers={**auth(VIEWER), "mcp-session-id": "sess-1"}
        )
    assert response.headers["mcp-session-id"] == "sess-1"
    assert "set-cookie" not in response.headers


async def test_concurrent_mixed_requests_keep_authorization_isolated(downstream):
    import asyncio

    async with make_gateway(downstream) as client:
        calls = []
        for i in range(20):
            token = ADMIN if i % 2 == 0 else VIEWER
            calls.append(
                client.post(
                    "/mcp",
                    json=rpc("tools/call", {"name": "admin_reset_key"}, id_=i),
                    headers=auth(token),
                )
            )
        responses = await asyncio.gather(*calls)
    for i, response in enumerate(responses):
        body = response.json()
        assert body["id"] == i
        assert ("error" in body) == (i % 2 == 1)
    assert len(downstream.requests) == 10


async def test_healthz_needs_no_auth(gateway):
    assert (await gateway.get("/healthz")).json() == {"status": "ok"}
