"""JSON-RPC 2.0 wire-format helpers."""

from dataclasses import dataclass
from typing import Any

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
# Implementation-defined server errors (-32000..-32099)
UNAUTHORIZED_TOOL_CALL = -32001
UNAUTHENTICATED = -32002
DOWNSTREAM_UNAVAILABLE = -32003

RequestId = str | int | None


def error_response(request_id: RequestId, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


@dataclass(frozen=True)
class ParsedMessage:
    raw: dict[str, Any]
    method: str
    is_notification: bool
    request_id: RequestId
    params: Any


class InvalidMessage(Exception):
    def __init__(self, request_id: RequestId, message: str = "Invalid Request"):
        super().__init__(message)
        self.request_id = request_id
        self.message = message


def _valid_id(value: Any) -> bool:
    return value is None or (isinstance(value, str | int) and not isinstance(value, bool))


def parse_message(obj: Any) -> ParsedMessage:
    """Validate one JSON-RPC request/notification object."""
    if not isinstance(obj, dict):
        raise InvalidMessage(None)
    has_id = "id" in obj
    request_id = obj.get("id") if _valid_id(obj.get("id")) else None
    if obj.get("jsonrpc") != "2.0" or not isinstance(obj.get("method"), str) or not obj["method"]:
        raise InvalidMessage(request_id)
    if has_id and not _valid_id(obj["id"]):
        raise InvalidMessage(None)
    params = obj.get("params")
    if "params" in obj and not isinstance(params, dict | list):
        raise InvalidMessage(request_id)
    return ParsedMessage(
        raw=obj,
        method=obj["method"],
        is_notification=not has_id,
        request_id=request_id,
        params=params,
    )
