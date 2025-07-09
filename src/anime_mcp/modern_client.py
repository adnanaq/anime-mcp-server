"""
Modern FastMCP client adapter using native FastMCP patterns.

Replaces langchain-mcp with direct FastMCP Client usage for better integration.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastmcp import Client

logger = logging.getLogger(__name__)


class ModernMCPClient:
    """Modern MCP client using native FastMCP patterns."""

    def __init__(
        self,
        transport: str = "stdio",
        server_command: str = "python",
        server_args: Optional[List[str]] = None,
        host: str = "localhost",
        port: int = 8001,
        url: Optional[str] = None,
    ):
        """Initialize modern MCP client.

        Args:
            transport: Transport protocol ("stdio", "http", "sse", "streamable")
            server_command: Command to start MCP server (for stdio)
            server_args: Arguments for server command (for stdio)
            host: Server host (for HTTP-based transports)
            port: Server port (for HTTP-based transports)
            url: Direct URL (for HTTP-based transports)
        """
        self.transport = transport
        self.server_command = server_command
        self.server_args = server_args or ["-m", "src.anime_mcp.server"]
        self.host = host
        self.port = port
        self.url = url
        self.client: Optional[Client] = None
        self._tools_cache: Optional[Dict[str, Any]] = None

    async def connect(self) -> None:
        """Connect to MCP server using FastMCP Client."""
        if self.transport == "stdio":
            logger.info(
                f"Connecting to core MCP server via stdio: {self.server_command} {' '.join(self.server_args)}"
            )
        else:
            connection_target = self.url or f"{self.host}:{self.port}"
            logger.info(
                f"Connecting to MCP server via {self.transport}: {connection_target}"
            )

        try:
            # Create FastMCP client - FastMCP automatically infers transport type
            if self.transport == "stdio":
                # FastMCP can infer Python stdio transport from .py file path
                script_path = (
                    "run_core_server.py"  # Use core server that doesn't need LangGraph
                )
                self.client = Client(
                    script_path
                )  # FastMCP auto-infers PythonStdioTransport

            elif self.transport in ["http", "sse", "streamable"]:
                # For HTTP-based transports, FastMCP infers from URL
                server_url = self.url or f"http://{self.host}:{self.port}"
                self.client = Client(server_url)  # FastMCP auto-infers SSETransport

            else:
                raise ValueError(f"Unsupported transport: {self.transport}")

            logger.info(f"FastMCP client created for {self.transport} transport")

        except Exception as e:
            logger.error(f"Failed to create FastMCP client for {self.transport}: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        try:
            if self.client:
                # FastMCP Client handles cleanup automatically
                self.client = None
                logger.info("MCP client disconnected")
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")
        finally:
            self._tools_cache = None

    async def get_tools(self) -> Dict[str, Any]:
        """Get all available tools from MCP server.

        Returns:
            Dictionary mapping tool names to callable wrapper functions

        Raises:
            RuntimeError: If client is not connected
        """
        if not self.client:
            raise RuntimeError("Client not connected. Call connect() first.")

        # Return cached tools if available
        if self._tools_cache is not None:
            logger.debug("Returning cached tools")
            return self._tools_cache

        logger.info("Discovering tools via FastMCP client")

        try:
            async with self.client as session:
                # Use list_tools() method as per FastMCP Client API
                tools_list = await session.list_tools()

                # Create callable wrapper functions for each tool
                tools_dict = {}
                for tool in tools_list:
                    tools_dict[tool.name] = self._create_tool_wrapper(tool)

                logger.info(
                    f"Discovered {len(tools_dict)} tools: {list(tools_dict.keys())}"
                )

                # Cache the tools
                self._tools_cache = tools_dict
                return tools_dict

        except Exception as e:
            logger.error(f"Tool discovery failed: {e}")
            raise

    def _create_tool_wrapper(self, tool):
        """Create a callable wrapper function for an MCP tool.

        Args:
            tool: MCP Tool object

        Returns:
            Callable wrapper function that uses FastMCP Client call_tool API
        """

        async def tool_wrapper(**kwargs):
            """Wrapper function that calls the MCP tool via FastMCP Client."""
            if not self.client:
                raise RuntimeError("Client not connected")

            # MCP tools expect parameters wrapped in an "input" object
            # This fixes the "Input validation error: 'input' is a required property" error
            tool_arguments = {"input": kwargs}

            # Use FastMCP Client call_tool API with proper argument format
            async with self.client as session:
                result = await session.call_tool(tool.name, tool_arguments)

            # Extract text content from FastMCP result
            if isinstance(result, list) and len(result) > 0:
                # Handle TextContent objects from FastMCP
                if hasattr(result[0], "text"):
                    return result[0].text
                elif hasattr(result[0], "content"):
                    return result[0].content
                else:
                    return result[0]
            return result

        # Add metadata for debugging
        tool_wrapper.__name__ = tool.name
        tool_wrapper.__doc__ = (
            tool.description
            if hasattr(tool, "description")
            else f"MCP tool: {tool.name}"
        )

        return tool_wrapper

    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call a specific tool with arguments.

        Args:
            tool_name: Name of tool to call
            **kwargs: Tool arguments

        Returns:
            Tool execution result
        """
        if not self.client:
            raise RuntimeError("Client not connected. Call connect() first.")

        logger.info(f"Calling tool: {tool_name} with args: {kwargs}")

        try:
            async with self.client as session:
                # MCP tools expect parameters wrapped in an "input" object
                tool_arguments = {"input": kwargs}
                result = await session.call_tool(tool_name, tool_arguments)
                logger.debug(f"Tool {tool_name} executed successfully")
                return result

        except Exception as e:
            logger.error(f"Tool {tool_name} execution failed: {e}")
            raise

    async def refresh_tools(self) -> None:
        """Refresh tool cache by re-discovering tools."""
        if not self.client:
            raise RuntimeError("Client not connected. Call connect() first.")

        logger.info("Refreshing tool cache")
        self._tools_cache = None
        await self.get_tools()

    @asynccontextmanager
    async def session(self):
        """Context manager for automatic connection/disconnection.

        Usage:
            async with client.session():
                tools = await client.get_tools()
        """
        await self.connect()
        try:
            # Use FastMCP client context manager for the session
            async with self.client as mcp_session:
                yield mcp_session
        finally:
            await self.disconnect()

    def is_connected(self) -> bool:
        """Check if client is connected.

        Returns:
            True if connected, False otherwise
        """
        return self.client is not None

    def get_tool_names(self) -> List[str]:
        """Get list of tool names from cache.

        Returns:
            List of tool names if tools are cached, empty list otherwise
        """
        if self._tools_cache:
            return list(self._tools_cache.keys())
        return []


def create_modern_mcp_client(
    transport: str = "stdio",
    host: str = "localhost",
    port: int = 8001,
    url: Optional[str] = None,
) -> ModernMCPClient:
    """Create modern MCP client with configuration.

    Args:
        transport: Transport protocol ("stdio", "http", "sse", "streamable")
        host: Server host for HTTP-based transports
        port: Server port for HTTP-based transports
        url: Direct URL for HTTP-based transports

    Returns:
        Configured modern MCP client
    """
    return ModernMCPClient(
        transport=transport,
        server_command="python",
        server_args=["-m", "src.anime_mcp.server"],  # Use core server for stability
        host=host,
        port=port,
        url=url,
    )


# Global client instance for persistent connection
_global_client: Optional[ModernMCPClient] = None


async def get_global_mcp_client() -> ModernMCPClient:
    """Get or create global MCP client instance.

    Returns:
        Connected MCP client instance

    Raises:
        Exception: If connection fails
    """
    global _global_client

    if _global_client is None:
        _global_client = create_modern_mcp_client()

    if not _global_client.is_connected():
        await _global_client.connect()

    return _global_client


async def disconnect_global_client() -> None:
    """Disconnect the global client if connected.

    This should be called during application shutdown.
    """
    global _global_client

    if _global_client and _global_client.is_connected():
        await _global_client.disconnect()
        _global_client = None


async def get_all_mcp_tools() -> Dict[str, Any]:
    """Get all MCP tools using modern FastMCP client.

    Provides backward compatibility with existing interface.

    Returns:
        Dictionary mapping tool names to tool objects

    Raises:
        Exception: If tool discovery fails
    """
    client = await get_global_mcp_client()
    return await client.get_tools()
