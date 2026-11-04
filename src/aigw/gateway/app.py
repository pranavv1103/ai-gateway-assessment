"""FastAPI MCP security gateway: authenticates callers and filters tools/call by role."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from aigw.config import GatewaySettings
from aigw.gateway import jsonrpc
from aigw.gateway.auth import TokenRoleResolver, extract_bearer_token
from aigw.gateway.policy import authorize, is_admin_tool

logger = logging.getLogger(__name__)

FORWARDED_REQUEST_HEADERS = ("mcp-session-id", "mcp-protocol-version")
FORWARDED_RESPONSE_HEADERS = ("mcp-session-id",)


class BodyTooLargeError(Exception):
    pass


async def read_limited_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise BodyTooLargeError
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise BodyTooLargeError
        chunks.append(chunk)
    return b"".join(chunks)


def create_app(
    settings: GatewaySettings, downstream_client: httpx.AsyncClient | None = None
) -> FastAPI:
    resolver = TokenRoleResolver(settings.token_roles)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if downstream_client is not None:
            yield
            return
        app.state.client = httpx.AsyncClient(timeout=settings.downstream_timeout_seconds)
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(title="MCP Security Gateway", lifespan=lifespan, docs_url=None, redoc_url=None)
    if downstream_client is not None:
        app.state.client = downstream_client

    def rpc_error_response(
        status: int, request_id: jsonrpc.RequestId, code: int, message: str, headers=None
    ) -> JSONResponse:
        return JSONResponse(
            jsonrpc.error_response(request_id, code, message), status_code=status, headers=headers
        )

    async def forward(
        client: httpx.AsyncClient, raw: dict[str, Any], headers: dict[str, str]
    ) -> tuple[httpx.Response | None, Any]:
        """Send one message downstream; returns (response, parsed_json_or_None)."""
        try:
            response = await client.post(
                settings.downstream_url,
                content=json.dumps(raw).encode("utf-8"),
                headers={
                    "content-type": "application/json",
                    "accept": "application/json",
                    **headers,
                },
            )
        except (TimeoutError, httpx.HTTPError):
            logger.warning("Downstream request failed")
            return None, None
        if not response.content:
            return response, None
        try:
            return response, response.json()
        except ValueError:
            return response, None

    async def process(
        raw: Any, role: str, client: httpx.AsyncClient, headers: dict[str, str]
    ) -> tuple[dict[str, Any] | None, httpx.Response | None]:
        """Returns (json-rpc response or None for notifications, downstream response if any)."""
        try:
            message = jsonrpc.parse_message(raw)
        except jsonrpc.InvalidMessage as exc:
            return jsonrpc.error_response(
                exc.request_id, jsonrpc.INVALID_REQUEST, exc.message
            ), None

        decision = authorize(message, role)
        if not decision.allowed:
            tool = message.params.get("name") if isinstance(message.params, dict) else None
            logger.warning(
                "Blocked tools/call role=%s tool=%r admin_tool=%s",
                role,
                tool if isinstance(tool, str) else None,
                isinstance(tool, str) and is_admin_tool(tool),
            )
            if message.is_notification:
                return None, None
            if decision.invalid_params:
                return jsonrpc.error_response(
                    message.request_id, jsonrpc.INVALID_PARAMS, "Invalid params"
                ), None
            return jsonrpc.error_response(
                message.request_id, jsonrpc.UNAUTHORIZED_TOOL_CALL, "Unauthorized Tool Call"
            ), None

        response, body = await forward(client, message.raw, headers)
        if message.is_notification:
            return None, response
        unavailable = jsonrpc.error_response(
            message.request_id, jsonrpc.DOWNSTREAM_UNAVAILABLE, "Downstream service unavailable"
        )
        if response is None or not isinstance(body, dict) or body.get("jsonrpc") != "2.0":
            return unavailable, response
        if "result" not in body and "error" not in body:
            return unavailable, response
        return body, response

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/mcp")
    async def mcp_endpoint(request: Request) -> Response:
        token = extract_bearer_token(request.headers.get("authorization"))
        role = resolver.resolve(token) if token else None
        if role is None:
            return rpc_error_response(
                401,
                None,
                jsonrpc.UNAUTHENTICATED,
                "Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            body = await read_limited_body(request, settings.max_body_bytes)
        except BodyTooLargeError:
            return rpc_error_response(413, None, jsonrpc.INVALID_REQUEST, "Request too large")
        try:
            payload = json.loads(body)
        except (ValueError, RecursionError):
            return rpc_error_response(400, None, jsonrpc.PARSE_ERROR, "Parse error")

        client: httpx.AsyncClient = request.app.state.client
        headers = {
            name: value
            for name in FORWARDED_REQUEST_HEADERS
            if (value := request.headers.get(name)) is not None
        }

        if isinstance(payload, list):
            if not payload:
                return rpc_error_response(400, None, jsonrpc.INVALID_REQUEST, "Invalid Request")
            if len(payload) > settings.max_batch_size:
                return rpc_error_response(400, None, jsonrpc.INVALID_REQUEST, "Batch too large")
            results = await asyncio.gather(
                *(process(item, role, client, headers) for item in payload)
            )
            responses = [response for response, _ in results if response is not None]
            if not responses:
                return Response(status_code=202)
            return JSONResponse(responses)

        response, downstream = await process(payload, role, client, headers)
        if response is None:
            return Response(status_code=202)
        extra_headers = {}
        if downstream is not None:
            extra_headers = {
                name: downstream.headers[name]
                for name in FORWARDED_RESPONSE_HEADERS
                if name in downstream.headers
            }
        status = 200
        if response.get("error", {}).get("code") == jsonrpc.DOWNSTREAM_UNAVAILABLE:
            status = 502
        elif response.get("error", {}).get("code") == jsonrpc.INVALID_REQUEST:
            status = 400
        return JSONResponse(response, status_code=status, headers=extra_headers)

    return app


def build_app_from_env() -> FastAPI:
    return create_app(GatewaySettings.from_env())
