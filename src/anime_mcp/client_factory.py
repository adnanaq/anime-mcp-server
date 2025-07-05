"""
Modern MCP client factory with full protocol support.

Provides clean, modern interface for all MCP transport protocols.
"""

import logging
from typing import Optional

from .modern_client import ModernMCPClient

logger = logging.getLogger(__name__)


def create_mcp_client(
    protocol: str = "stdio",
    host: str = "localhost",
    port: int = 8001,
    url: Optional[str] = None,
    server_module: str = "src.mcp.modern_server"
) -> ModernMCPClient:
    """Create modern MCP client with support for all protocols.

    Args:
        protocol: Transport protocol ("stdio", "http", "sse", "streamable")
        host: Server host for HTTP-based transports
        port: Server port for HTTP-based transports
        url: Direct URL for HTTP-based transports (overrides host:port)
        server_module: Server module to connect to

    Returns:
        Configured modern MCP client

    Examples:
        # Stdio transport (default)
        client = create_mcp_client()

        # HTTP transport
        client = create_mcp_client(protocol="http", port=8001)

        # SSE transport with custom URL
        client = create_mcp_client(protocol="sse", url="http://remote-server:8001")
    """
    logger.info(f"Creating modern MCP client: protocol={protocol}")

    return ModernMCPClient(
        transport=protocol,
        server_command="python",
        server_args=["run_modern_server.py"],  # Use direct script entry point
        host=host,
        port=port,
        url=url
    )


# Protocol-specific convenience functions
def create_stdio_client() -> ModernMCPClient:
    """Create MCP client with stdio transport."""
    return create_mcp_client(protocol="stdio")


def create_http_client(
    host: str = "localhost", 
    port: int = 8001,
    url: Optional[str] = None
) -> ModernMCPClient:
    """Create MCP client with HTTP transport."""
    return create_mcp_client(protocol="http", host=host, port=port, url=url)


def create_sse_client(
    host: str = "localhost", 
    port: int = 8001,
    url: Optional[str] = None
) -> ModernMCPClient:
    """Create MCP client with Server-Sent Events transport."""
    return create_mcp_client(protocol="sse", host=host, port=port, url=url)


def create_streamable_client(
    host: str = "localhost", 
    port: int = 8001,
    url: Optional[str] = None
) -> ModernMCPClient:
    """Create MCP client with streamable HTTP transport."""
    return create_mcp_client(protocol="streamable", host=host, port=port, url=url)