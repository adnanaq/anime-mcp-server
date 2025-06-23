#!/usr/bin/env python3
"""
MCP Server Verification Script

This script verifies that the FastMCP anime server is working correctly
by connecting as an MCP client and testing the available tools.

Usage:
    python scripts/verify_mcp_server.py

Requirements:
    - MCP server dependencies installed
    - Qdrant running (if testing with real data)
"""
import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_mcp_tools():
    """Test MCP server tools."""
    print("🧪 Testing FastMCP Anime Server...")

    # Server parameters
    server_params = StdioServerParameters(
        command=sys.executable, args=["-m", "src.mcp.server"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the client
            await session.initialize()
            print("✅ MCP session initialized")

            # List available tools
            tools = await session.list_tools()
            print(f"📋 Available tools: {[tool.name for tool in tools.tools]}")

            # Test search_anime tool
            search_result = await session.call_tool(
                "search_anime", arguments={"query": "dragon ball", "limit": 2}
            )
            print(f"🔍 Search result: {search_result.content[0].text}")

            # Test get_anime_stats tool
            stats_result = await session.call_tool("get_anime_stats", arguments={})
            print(f"📊 Stats result: {stats_result.content[0].text}")

            # List available resources
            resources = await session.list_resources()
            print(f"📁 Available resources: {[res.uri for res in resources.resources]}")

            print("✅ All MCP tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_mcp_tools())
