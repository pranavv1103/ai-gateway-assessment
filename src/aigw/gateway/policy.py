"""Method-level authorization policy for MCP JSON-RPC messages."""

import unicodedata
from dataclasses import dataclass

from aigw.gateway.jsonrpc import ParsedMessage

ADMIN_TOOL_PREFIX = "admin_"


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""
    invalid_params: bool = False


def is_admin_tool(tool_name: str) -> bool:
    """Case- and Unicode-normalized so look-alikes such as 'ADMIN_x' or ' admin_x' are caught."""
    normalized = unicodedata.normalize("NFKC", tool_name).strip().casefold()
    return normalized.startswith(ADMIN_TOOL_PREFIX)


def authorize(message: ParsedMessage, role: str) -> Decision:
    if message.method != "tools/call":
        return Decision(True)
    params = message.params
    if not isinstance(params, dict) or not isinstance(params.get("name"), str):
        return Decision(False, "params.name must be a string", invalid_params=True)
    if is_admin_tool(params["name"]) and role != "admin":
        return Decision(False, "unauthorized")
    return Decision(True)
