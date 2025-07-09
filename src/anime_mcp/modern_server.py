"""
Modern Anime MCP Server with LangGraph Multi-Agent Workflows.

Replaces universal tool approach with platform-specific tools and intelligent workflows.
Built using modern LangGraph swarm architecture with conversation persistence.
"""

import argparse
import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from mcp.server.fastmcp import Context

from ..config import get_settings
from ..langgraph.anime_swarm import AnimeDiscoverySwarm

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize FastMCP server
mcp = FastMCP(
    name="Anime Discovery Server",
    instructions=(
        "Advanced anime discovery with multi-agent workflows and platform-specific tools. "
        "Provides intelligent search across 6 platforms with AI-powered recommendations, "
        "real-time scheduling, streaming availability, and semantic similarity matching."
    )
)

# Global workflow instance 
anime_swarm: Optional[AnimeDiscoverySwarm] = None


# High-Level Workflow Tools
@mcp.tool(
    name="discover_anime",
    description="Intelligent anime discovery using multi-agent workflow with cross-platform search",
    annotations={
        "title": "Anime Discovery Workflow",
        "readOnlyHint": True,
        "idempotentHint": False,  # Results may vary based on real-time data
    }
)
async def discover_anime(
    query: str,
    user_preferences: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Comprehensive anime discovery using intelligent multi-agent workflow.
    
    Routes queries to optimal platform-specific tools and enriches results with
    cross-platform data, scheduling information, and streaming availability.
    
    Args:
        query: Natural language anime search query
        user_preferences: Optional user preferences and context
        session_id: Session ID for conversation continuity
        
    Returns:
        Comprehensive workflow result with anime recommendations and metadata
    """
    if not anime_swarm:
        raise RuntimeError("Anime discovery workflow not initialized")
    
    if ctx:
        await ctx.info(f"Starting intelligent anime discovery for: '{query}'")
    
    logger.info(f"Workflow discovery request: '{query}'")
    
    try:
        # Execute multi-agent workflow
        result = await anime_swarm.discover_anime(
            query=query,
            user_context=user_preferences,
            session_id=session_id
        )
        
        if ctx:
            total_results = result.get("total_results", 0)
            agents_used = result.get("agents_used", [])
            await ctx.info(f"Discovery completed: {total_results} results using {len(agents_used)} agents")
        
        logger.info(f"Workflow completed: {result.get('total_results', 0)} results")
        return result
        
    except Exception as e:
        error_msg = f"Anime discovery workflow failed: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        logger.error(f"Workflow failed: {e}")
        raise RuntimeError(error_msg)


@mcp.tool(
    name="get_currently_airing_anime",
    description="Get currently airing anime with real-time broadcast schedules",
    annotations={
        "title": "Currently Airing Anime",
        "readOnlyHint": True,
        "idempotentHint": False,  # Time-sensitive data
    }
)
async def get_currently_airing_anime(
    day_filter: Optional[str] = None,
    timezone: str = "JST",
    streaming_platforms: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Get currently airing anime with broadcast schedules using specialized workflow.
    
    Args:
        day_filter: Filter by specific day of week
        timezone: Timezone for broadcast times (default: JST)
        streaming_platforms: Filter by streaming platforms
        session_id: Session ID for conversation continuity
        
    Returns:
        Currently airing anime with detailed broadcast information
    """
    if not anime_swarm:
        raise RuntimeError("Anime discovery workflow not initialized")
    
    if ctx:
        await ctx.info("Fetching currently airing anime with broadcast schedules")
    
    try:
        filters = {
            "day_filter": day_filter,
            "timezone": timezone,
            "streaming_platforms": streaming_platforms or []
        }
        
        result = await anime_swarm.get_currently_airing(
            filters=filters,
            session_id=session_id
        )
        
        if ctx:
            await ctx.info(f"Found {result.get('total_results', 0)} currently airing anime")
        
        return result
        
    except Exception as e:
        error_msg = f"Currently airing anime query failed: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)


@mcp.tool(
    name="find_similar_anime_workflow",
    description="Find anime similar to reference using AI-powered similarity analysis",
    annotations={
        "title": "Similar Anime Discovery",
        "readOnlyHint": True,
        "idempotentHint": True,
    }
)
async def find_similar_anime_workflow(
    reference_anime: str,
    similarity_mode: str = "hybrid",
    session_id: Optional[str] = None,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Find anime similar to reference using semantic similarity and cross-platform data.
    
    Args:
        reference_anime: Reference anime title or ID
        similarity_mode: "content", "visual", or "hybrid" similarity matching
        session_id: Session ID for conversation continuity
        
    Returns:
        Similar anime with similarity scores and detailed analysis
    """
    if not anime_swarm:
        raise RuntimeError("Anime discovery workflow not initialized")
    
    if ctx:
        await ctx.info(f"Finding anime similar to: '{reference_anime}' ({similarity_mode} mode)")
    
    try:
        result = await anime_swarm.find_similar_anime(
            reference_anime=reference_anime,
            similarity_mode=similarity_mode,
            session_id=session_id
        )
        
        if ctx:
            await ctx.info(f"Found {result.get('total_results', 0)} similar anime")
        
        return result
        
    except Exception as e:
        error_msg = f"Similar anime search failed: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)


@mcp.tool(
    name="search_by_streaming_platform",
    description="Search anime available on specific streaming platforms",
    annotations={
        "title": "Streaming Platform Search",
        "readOnlyHint": True,
        "idempotentHint": True,
    }
)
async def search_by_streaming_platform(
    platforms: List[str],
    content_filters: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Search anime available on specific streaming platforms with intelligent routing.
    
    Args:
        platforms: List of streaming platforms (e.g., ["Netflix", "Crunchyroll"])
        content_filters: Optional content filters (genre, year, rating, etc.)
        session_id: Session ID for conversation continuity
        
    Returns:
        Anime available on specified platforms with streaming details
    """
    if not anime_swarm:
        raise RuntimeError("Anime discovery workflow not initialized")
    
    if ctx:
        platforms_str = ", ".join(platforms)
        await ctx.info(f"Searching anime on platforms: {platforms_str}")
    
    try:
        result = await anime_swarm.search_by_streaming_platform(
            platforms=platforms,
            additional_filters=content_filters,
            session_id=session_id
        )
        
        if ctx:
            await ctx.info(f"Found {result.get('total_results', 0)} anime on specified platforms")
        
        return result
        
    except Exception as e:
        error_msg = f"Streaming platform search failed: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)


# Import and register all tiered and platform tools
def register_tiered_tools():
    """Register all tiered MCP tools."""
    try:
        from ..anime_mcp.tools import (
            register_basic_tools,
            register_standard_tools,
            register_detailed_tools,
            register_comprehensive_tools,
        )
        
        # Register each tier
        tool_count = 0
        
        # Tier 1: Basic tools (8 fields, 80% coverage)
        register_basic_tools(mcp)
        tool_count += 4  # search, get, similar, seasonal
        logger.info("✅ Registered Tier 1 (Basic) tools")
        
        # Tier 2: Standard tools (15 fields, 95% coverage)  
        register_standard_tools(mcp)
        tool_count += 5  # search, get, similar, seasonal, genre_search
        logger.info("✅ Registered Tier 2 (Standard) tools")
        
        # Tier 3: Detailed tools (25 fields, 99% coverage)
        register_detailed_tools(mcp)
        tool_count += 5  # search, get, similar, seasonal, analysis
        logger.info("✅ Registered Tier 3 (Detailed) tools")
        
        # Tier 4: Comprehensive tools (40+ fields, 100% coverage)
        register_comprehensive_tools(mcp)
        tool_count += 4  # search, get, similar, analytics
        logger.info("✅ Registered Tier 4 (Comprehensive) tools")
        
        logger.info(f"🎯 Registered {tool_count} tiered tools across 4 complexity levels")
        return tool_count
        
    except ImportError as e:
        logger.warning(f"Could not import tiered tools: {e}")
        return 0


def register_platform_tools():
    """Register legacy platform-specific MCP tools (if available)."""
    # Note: Platform tools temporarily disabled due to missing register functions
    # Will be re-enabled after platform tool modernization
    logger.info("⚠️  Platform tools temporarily disabled - using tiered tools instead")
    return 0


# MCP Resources
@mcp.resource("anime://server/capabilities")
async def server_capabilities() -> str:
    """Server capabilities and available tools."""
    capabilities = {
        "workflow_tools": [
            "discover_anime",
            "get_currently_airing_anime", 
            "find_similar_anime_workflow",
            "search_by_streaming_platform"
        ],
        "platform_coverage": {
            "myanimelist": "Community data, ratings, content filtering",
            "anilist": "GraphQL API with 70+ parameters, international content",
            "jikan": "MAL unofficial API, no key required, rich metadata",
            "animeschedule": "Broadcast schedules, streaming platforms, temporal data",
            "kitsu": "JSON:API, streaming platform specialization",
            "semantic_search": "AI-powered similarity, vector database"
        },
        "features": [
            "Multi-agent workflow orchestration",
            "4-tier progressive complexity architecture",
            "Cross-platform data enrichment", 
            "Real-time broadcast schedules",
            "Streaming platform integration",
            "AI-powered semantic search",
            "Conversation memory persistence",
            "Intelligent query routing"
        ],
        "tiered_architecture": {
            "tier_1_basic": "8 fields, 80% coverage, fastest performance",
            "tier_2_standard": "15 fields, 95% coverage, enhanced filtering",
            "tier_3_detailed": "25 fields, 99% coverage, cross-platform data",
            "tier_4_comprehensive": "40+ fields, 100% coverage, complete analytics"
        },
        "data_sources": 6,
        "total_tools": "20+ (4 workflow + 18 tiered)",
        "anime_database_size": "38,000+ entries",
        "workflow_architecture": "LangGraph multi-agent swarm"
    }
    
    return f"Server Capabilities: {capabilities}"


@mcp.resource("anime://platforms/status")
async def platforms_status() -> str:
    """Status of all anime platform integrations."""
    status = {
        "mal": {"status": "available" if hasattr(settings, 'mal_api_key') else "requires_api_key"},
        "anilist": {"status": "available" if hasattr(settings, 'anilist_token') else "requires_token"},
        "jikan": {"status": "available", "note": "No API key required"},
        "animeschedule": {"status": "available", "note": "No API key required"},
        "kitsu": {"status": "available", "note": "No API key required"},
        "semantic_search": {"status": "available", "note": "Vector database required"},
        "langgraph_workflows": {"status": "available" if anime_swarm else "initializing"}
    }
    
    return f"Platform Status: {status}"


@mcp.resource("anime://workflow/architecture")
async def workflow_architecture() -> str:
    """LangGraph workflow architecture information."""
    architecture = {
        "pattern": "Multi-agent swarm with handoff tools",
        "agents": {
            "SearchAgent": "Platform-specific anime search with intelligent routing",
            "ScheduleAgent": "Broadcast schedules and streaming platform enrichment"
        },
        "memory": {
            "short_term": "InMemorySaver for conversation continuity",
            "long_term": "InMemoryStore for user preferences"
        },
        "features": [
            "Intent analysis and query routing",
            "Cross-platform tool chaining", 
            "Intelligent agent handoffs",
            "Memory persistence across sessions"
        ],
        "tools_per_agent": {
            "SearchAgent": 20,
            "ScheduleAgent": 6,
            "Total": 26
        }
    }
    
    return f"Workflow Architecture: {architecture}"


async def initialize_server():
    """Initialize the anime discovery server with workflows."""
    global anime_swarm
    
    logger.info("🚀 Initializing Modern Anime Discovery Server")
    
    try:
        # Register tiered tools (modern architecture)
        tiered_count = register_tiered_tools()
        logger.info(f"🏗️  Registered {tiered_count} tiered tools")
        
        # Register platform-specific tools (legacy)
        platform_count = register_platform_tools()
        logger.info(f"🔧 Registered {platform_count} platform-specific tools")
        
        # Initialize multi-agent workflow system
        logger.info("📡 Initializing LangGraph multi-agent workflow system")
        anime_swarm = AnimeDiscoverySwarm()
        logger.info("✅ Multi-agent workflow system initialized with manual swarm creation")
        
        # Log final capabilities
        total_tools = 4 + tiered_count + platform_count  # 4 workflow + tiered + platform
        logger.info(f"🎯 Tool Architecture: 4-tier progressive complexity")
        logger.info(f"🤖 Agents: SearchAgent, ScheduleAgent")
        logger.info(f"🔧 Total tools: {total_tools} (4 workflow + {tiered_count} tiered + {platform_count} platform)")
        logger.info(f"📊 Coverage: 80%→95%→99%→100% across 4 tiers")
        
        logger.info("✅ Modern anime discovery server initialized")
        
    except Exception as e:
        logger.error(f"❌ Server initialization failed: {e}")
        raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Modern Anime Discovery MCP Server")
    parser.add_argument("--mode", choices=["stdio", "sse"], default="stdio",
                      help="MCP transport mode (default: stdio)")
    parser.add_argument("--host", default="localhost",
                      help="Host for SSE mode (default: localhost)")
    parser.add_argument("--port", type=int, default=8001,
                      help="Port for SSE mode (default: 8001)")
    parser.add_argument("--verbose", "-v", action="store_true",
                      help="Enable verbose logging")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    if args.debug:
        level = "DEBUG"
    elif args.verbose:
        level = "INFO"
    else:
        level = "WARNING"
        
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("🌟 Starting Modern Anime Discovery MCP Server")
    logger.info(f"📡 Transport mode: {args.mode}")
    if args.mode == "sse":
        logger.info(f"🌐 HTTP server: {args.host}:{args.port}")
    
    async def init():
        """Initialize server in async context."""
        try:
            await initialize_server()
            logger.info("✅ Server initialization completed")
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}", exc_info=True)
            raise
    
    try:
        # Initialize server
        asyncio.run(init())
        
        # Run server with appropriate transport
        if args.mode == "stdio":
            logger.info("🔌 Starting stdio transport")
            mcp.run(transport="stdio")
        else:
            logger.info(f"🌐 Starting SSE transport on {args.host}:{args.port}")
            mcp.run(transport="sse", host=args.host, port=args.port)
            
    except KeyboardInterrupt:
        logger.info("🛑 Server shutdown requested")
    except Exception as e:
        logger.error(f"❌ Server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()