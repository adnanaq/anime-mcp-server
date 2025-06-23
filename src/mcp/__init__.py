"""
MCP (Model Context Protocol) module for Anime Search Server

Provides MCP server implementation and tools for AI assistant integration
with anime database search and recommendation capabilities.
"""

from .server import mcp, initialize_mcp_server, main
from .tools import (
    validate_search_query, validate_anime_id, validate_limit,
    format_anime_result, format_search_results
)

__all__ = ["mcp", "initialize_mcp_server", "main", "validate_search_query", "validate_anime_id", "validate_limit", "format_anime_result", "format_search_results"]