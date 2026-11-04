"""Entry point: ``python -m aigw.mcp_server`` runs the MCP server over stdio."""

import asyncio
import contextlib
import io
import logging
import sys

import anyio
from mcp.server.stdio import stdio_server

from aigw.logging_utils import configure_logging
from aigw.mcp_server.server import build_server
from aigw.mcp_server.transport import method_not_found_guard

logger = logging.getLogger(__name__)


async def run() -> None:
    # Claim the real stdout for the transport, then point sys.stdout at stderr so any stray
    # print() in application or dependency code cannot corrupt the JSON-RPC stream.
    protocol_out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n")
    sys.stdout = sys.stderr
    server = build_server()
    async with (
        stdio_server(stdout=anyio.wrap_file(protocol_out)) as (read_stream, write_stream),
        method_not_found_guard(read_stream, write_stream) as guarded_read,
    ):
        logger.info("MCP server ready on stdio")
        await server.run(guarded_read, write_stream, server.create_initialization_options())


def main() -> None:
    configure_logging()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run())


if __name__ == "__main__":
    main()
