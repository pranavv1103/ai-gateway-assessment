"""Transport-level guard that gives unknown JSON-RPC methods the spec's -32601 code.

The SDK validates requests against a closed union and reports *any* validation failure as
-32602, so an unknown method would otherwise be indistinguishable from bad parameters.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal, get_args

import anyio
from mcp import types
from mcp.shared.message import SessionMessage


def known_request_methods() -> frozenset[str]:
    methods: set[str] = set()
    for request_type in get_args(types.ClientRequestType):
        annotation = request_type.model_fields["method"].annotation
        if getattr(annotation, "__origin__", None) is Literal:
            methods.update(get_args(annotation))
    return frozenset(methods)


KNOWN_METHODS = known_request_methods()


@asynccontextmanager
async def method_not_found_guard(read_stream, write_stream) -> AsyncIterator:
    """Yield a read stream that answers unknown request methods with -32601 itself."""
    forward_send, forward_recv = anyio.create_memory_object_stream[SessionMessage | Exception](0)

    async def pump() -> None:
        async with forward_send:
            async for item in read_stream:
                root = item.message.root if isinstance(item, SessionMessage) else None
                if isinstance(root, types.JSONRPCRequest) and root.method not in KNOWN_METHODS:
                    error = types.JSONRPCError(
                        jsonrpc="2.0",
                        id=root.id,
                        error=types.ErrorData(
                            code=types.METHOD_NOT_FOUND, message="Method not found"
                        ),
                    )
                    await write_stream.send(SessionMessage(types.JSONRPCMessage(error)))
                else:
                    await forward_send.send(item)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(pump)
        try:
            yield forward_recv
        finally:
            task_group.cancel_scope.cancel()
