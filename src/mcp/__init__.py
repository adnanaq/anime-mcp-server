"""
MCP (Model Context Protocol) module for Anime Search Server

Provides MCP server implementation and tools for AI assistant integration
with anime database search and recommendation capabilities.
"""

from .server import initialize_mcp_server, main, mcp

__all__ = [
    "mcp",
    "initialize_mcp_server",
    "main",
]
