"""FastMCP client adapter for automatic tool discovery.

This module provides tool extraction using the langchain-mcp MCPToolkit for automatic tool discovery.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools.base import BaseTool
from langchain_mcp import MCPToolkit

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class FastMCPClientAdapter:
    """Adapter for FastMCP client with automatic tool discovery.

    This class tool extraction with standards-compliant
    MCP protocol tool discovery using MCPToolkit from langchain-mcp.
    """

    def __init__(self, server_config: Dict[str, Any]):
        """Initialize FastMCP client adapter.

        Args:
            server_config: Configuration dict for the MCP server

        Raises:
            ValueError: If configuration is invalid
        """
        self._validate_config(server_config)
        self.server_config = server_config
        self.toolkit: Optional[MCPToolkit] = None
        self.session: Optional[ClientSession] = None
        self._stdio_context: Optional[Any] = None
        self._tools_cache: Optional[List[BaseTool]] = None

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate MCP server configuration.

        Args:
            config: Configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        if not config:
            raise ValueError("Configuration cannot be empty")

        required_fields = ["command", "args"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in server config")

        if not config["command"]:
            raise ValueError("Empty command in server config")

    async def connect(self) -> None:
        """Connect to MCP server and initialize toolkit.

        Raises:
            Exception: If connection fails
        """
        logger.info(f"Connecting to MCP server: {self.server_config['command']}")

        try:
            # Create server parameters for stdio connection
            server_params = StdioServerParameters(
                command=self.server_config["command"], args=self.server_config["args"]
            )

            # Use stdio_client context manager properly
            self._stdio_context = stdio_client(server_params)
            read_stream, write_stream = await self._stdio_context.__aenter__()

            # Create and initialize client session
            self.session = ClientSession(read_stream, write_stream)
            await self.session.__aenter__()
            await self.session.initialize()

            # Create and initialize toolkit
            self.toolkit = MCPToolkit(session=self.session)
            await self.toolkit.initialize()

            logger.info("FastMCP client connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect FastMCP client: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from MCP server and cleanup resources."""
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)

            if hasattr(self, "_stdio_context") and self._stdio_context:
                await self._stdio_context.__aexit__(None, None, None)

            logger.info("FastMCP client disconnected")
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")
        finally:
            self.session = None
            self.toolkit = None
            self._tools_cache = None
            if hasattr(self, "_stdio_context"):
                self._stdio_context = None

    async def get_tools(self) -> List[BaseTool]:
        """Get all available tools via automatic discovery.

        Returns:
            List of LangChain BaseTool instances

        Raises:
            RuntimeError: If client is not connected
            Exception: If tool discovery fails
        """
        if not self.toolkit:
            raise RuntimeError("Client not connected. Call connect() first.")

        # Return cached tools if available
        if self._tools_cache is not None:
            logger.debug("Returning cached tools")
            return self._tools_cache

        logger.info("Discovering tools via FastMCP toolkit")

        try:
            # Use automatic tool discovery
            discovered_tools = self.toolkit.get_tools()

            # Check if it's a coroutine and await it
            if hasattr(discovered_tools, "__await__"):
                discovered_tools = await discovered_tools

            logger.info(
                f"Discovered {len(discovered_tools)} tools: {[tool.name for tool in discovered_tools]}"
            )

            # Cache the tools
            self._tools_cache = discovered_tools

            return discovered_tools

        except Exception as e:
            logger.error(f"Tool discovery failed: {e}")
            raise

    async def get_tools_dict(self) -> Dict[str, Callable]:
        """Get tools as dictionary for backward compatibility.

        Returns:
            Dictionary mapping tool names to callable functions
        """
        tools = await self.get_tools()

        # Convert BaseTool instances to callables for backward compatibility
        tools_dict = {}
        for tool in tools:
            tools_dict[tool.name] = tool

        return tools_dict

    async def refresh_tools(self) -> None:
        """Refresh tool cache by re-discovering tools.

        Raises:
            RuntimeError: If client is not connected
        """
        if not self.toolkit:
            raise RuntimeError("Client not connected. Call connect() first.")

        logger.info("Refreshing tool cache")
        self._tools_cache = None
        await self.get_tools()

    @asynccontextmanager
    async def connected(self):
        """Context manager for automatic connection/disconnection.

        Usage:
            async with adapter.connected():
                tools = await adapter.get_tools()
        """
        try:
            await self.connect()
            yield self
        finally:
            await self.disconnect()

    def get_tool_names(self) -> List[str]:
        """Get list of tool names from cache.

        Returns:
            List of tool names if tools are cached, empty list otherwise
        """
        if self._tools_cache:
            return [tool.name for tool in self._tools_cache]
        return []

    def is_connected(self) -> bool:
        """Check if client is connected.

        Returns:
            True if connected, False otherwise
        """
        return self.toolkit is not None


# Factory function for creating adapter with default configuration
def create_fastmcp_adapter() -> FastMCPClientAdapter:
    """Create FastMCP client adapter with default configuration.

    Returns:
        Configured FastMCP client adapter
    """
    config = {"command": "python", "args": ["-m", "src.mcp.server"]}

    return FastMCPClientAdapter(config)


# Global adapter instance for persistent connection
_global_adapter: Optional[FastMCPClientAdapter] = None


async def _ensure_global_connection() -> FastMCPClientAdapter:
    """Ensure global adapter is connected and return it.
    
    Returns:
        Connected FastMCP client adapter
        
    Raises:
        Exception: If connection fails
    """
    global _global_adapter
    
    if _global_adapter is None:
        _global_adapter = create_fastmcp_adapter()
        
    if not _global_adapter.is_connected():
        await _global_adapter.connect()
        
    return _global_adapter


async def disconnect_global_adapter() -> None:
    """Disconnect the global adapter if connected.
    
    This should be called during application shutdown.
    """
    global _global_adapter
    
    if _global_adapter and _global_adapter.is_connected():
        await _global_adapter.disconnect()
        _global_adapter = None


# Replacement function for get_all_mcp_tools()
async def get_all_mcp_tools() -> Dict[str, Callable]:
    """Get all MCP tools using automatic discovery with persistent connection.

    This function provides backward compatibility with the existing
    get_all_mcp_tools() interface while using modern FastMCP client.
    
    The connection is maintained persistently to ensure tools remain functional.

    Returns:
        Dictionary mapping tool names to callable functions

    Raises:
        Exception: If tool discovery fails
    """
    adapter = await _ensure_global_connection()
    tools = await adapter.get_tools_dict()
    return tools
