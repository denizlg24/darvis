import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent

from mcp_server.tools import TOOLS, TOOL_HANDLERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_server() -> Server:
    server = Server("darvis-mcp-server")

    @server.list_tools()
    async def list_tools():
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        handler = TOOL_HANDLERS.get(name)

        if handler is None:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        try:
            logger.info(f"Executing tool: {name} with args: {arguments}")
            result = await handler(arguments)
            logger.info(f"Tool {name} completed")
            return [TextContent(type="text", text=result)]
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    return server


async def run_server():
    server = create_server()
    logger.info("Starting DARVIS MCP server...")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )
