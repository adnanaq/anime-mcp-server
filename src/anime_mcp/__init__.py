"""
MCP (Model Context Protocol) module for Anime Search Server

Provides MCP server implementation and tools for AI assistant integration
with anime database search and recommendation capabilities.
"""

from .modern_server import main, mcp

__all__ = [
    "mcp",
    "main",
]
