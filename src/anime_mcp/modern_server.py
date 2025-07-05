"""
Modern MCP Server for Anime Search and Discovery.

Implements modern MCP patterns with separation of concerns, 
proper error handling, and standardized tool naming.
"""

import argparse
import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from mcp.server.fastmcp import Context

from ..config import get_settings
from ..vector.qdrant_client import QdrantClient
from .handlers.anime_handler import AnimeHandler
from .schemas import (
    AnimeDetailsInput,
    ImageSearchInput,
    MultimodalSearchInput,
    SearchAnimeInput,
    SimilarAnimeInput,
    VisualSimilarInput,
)

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize FastMCP server with proper metadata
mcp = FastMCP(
    name="Anime Search Server",
    instructions=(
        "Comprehensive anime search and discovery server providing semantic search, "
        "visual similarity matching, and detailed anime metadata. Supports text queries, "
        "image-based search, and multimodal discovery with advanced filtering options."
    ),
    dependencies=[
        "qdrant-client>=1.11.0",
        "fastembed>=0.3.0", 
        "pillow>=10.0.0",
        "torch>=2.0.0",
        "transformers>=4.30.0"
    ]
)

# Global handler - initialized in initialize_mcp_server()
anime_handler: Optional[AnimeHandler] = None


# Modern MCP Tools with Handler Pattern
@mcp.tool(
    name="anime_search",
    description="Search anime database with semantic similarity and filtering",
    tags={"anime", "search", "discovery"}
)
async def anime_search(
    search_params: SearchAnimeInput,
    ctx: Context,
) -> List[Dict[str, Any]]:
    """Search for anime using semantic search with comprehensive validation.
    
    Modern MCP implementation using handler pattern for business logic separation.
    """
    if not anime_handler:
        await ctx.error("Server not properly initialized")
        raise RuntimeError("Anime handler not initialized")
    
    return await anime_handler.search_anime(search_params, ctx)


@mcp.tool(
    name="anime_details",
    description="Get detailed information about a specific anime by ID",
    tags={"anime", "metadata"}
)
async def anime_details(
    details_params: AnimeDetailsInput,
    ctx: Context
) -> Dict[str, Any]:
    """Get detailed anime information including synopsis, tags, studios, platform IDs."""
    if not anime_handler:
        await ctx.error("Server not properly initialized")
        raise RuntimeError("Anime handler not initialized")
    
    return await anime_handler.get_anime_details(details_params.anime_id, ctx)


@mcp.tool(
    name="anime_similar",
    description="Find anime similar to a given anime by content and themes",
    tags={"anime", "recommendation", "similarity"}
)
async def anime_similar(
    similar_params: SimilarAnimeInput,
    ctx: Context
) -> List[Dict[str, Any]]:
    """Find anime similar to a reference anime based on content similarity."""
    if not anime_handler:
        await ctx.error("Server not properly initialized")
        raise RuntimeError("Anime handler not initialized")
    
    return await anime_handler.find_similar_anime(
        similar_params.anime_id, similar_params.limit, ctx
    )


@mcp.tool(
    name="anime_stats",
    description="Get statistics about the anime database",
    tags={"anime", "database", "stats"}
)
async def anime_stats(ctx: Context) -> Dict[str, Any]:
    """Get database statistics including total entries, health status, and configuration."""
    if not anime_handler:
        await ctx.error("Server not properly initialized")
        raise RuntimeError("Anime handler not initialized")
    
    return await anime_handler.get_database_stats(ctx)


@mcp.tool(
    name="anime_image_search",
    description="Search for anime using visual similarity from uploaded image",
    tags={"anime", "vision", "image", "similarity"}
)
async def anime_image_search(
    image_params: ImageSearchInput,
    ctx: Context
) -> List[Dict[str, Any]]:
    """Find anime with poster images visually similar to the provided image."""
    if not anime_handler:
        await ctx.error("Server not properly initialized")
        raise RuntimeError("Anime handler not initialized")
    
    return await anime_handler.search_by_image(
        image_params.image_data, image_params.limit, ctx
    )


@mcp.tool(
    name="anime_visual_similar",
    description="Find anime with similar visual style to a reference anime",
    tags={"anime", "vision", "similarity"}
)
async def anime_visual_similar(
    visual_params: VisualSimilarInput,
    ctx: Context
) -> List[Dict[str, Any]]:
    """Find anime with similar visual style using poster image comparison."""
    if not anime_handler:
        await ctx.error("Server not properly initialized")
        raise RuntimeError("Anime handler not initialized")
    
    client = anime_handler.verify_client("anime_visual_similar")
    anime_handler.check_multi_vector_support("anime_visual_similar")
    
    await ctx.info(f"Visual similarity search for anime: {visual_params.anime_id}")
    
    try:
        results = await client.find_visually_similar_anime(
            anime_id=visual_params.anime_id, limit=visual_params.limit
        )
        await ctx.info(f"Found {len(results)} visually similar anime")
        return results
    except Exception as e:
        await anime_handler.handle_error(e, "anime_visual_similar", ctx)


@mcp.tool(
    name="anime_multimodal_search",
    description="Search anime using both text query and image similarity",
    tags={"anime", "multimodal", "search", "vision"}
)
async def anime_multimodal_search(
    multimodal_params: MultimodalSearchInput,
    ctx: Context
) -> List[Dict[str, Any]]:
    """Combine semantic text search with visual image search for enhanced discovery."""
    if not anime_handler:
        await ctx.error("Server not properly initialized")
        raise RuntimeError("Anime handler not initialized")
    
    return await anime_handler.search_multimodal(
        query=multimodal_params.query,
        image_data=multimodal_params.image_data,
        limit=multimodal_params.limit,
        text_weight=multimodal_params.text_weight,
        ctx=ctx,
    )


# MCP Prompts for Anime Discovery
@mcp.prompt()
def anime_discovery_prompt(
    genre: str = "any",
    mood: str = "any", 
    content_type: str = "any"
) -> str:
    """Create an optimized prompt for anime discovery based on preferences."""
    prompt_parts = ["Find anime recommendations"]
    
    if genre != "any":
        prompt_parts.append(f"in the {genre} genre")
    
    if mood != "any":
        prompt_parts.append(f"with a {mood} mood/tone")
        
    if content_type != "any":
        prompt_parts.append(f"specifically {content_type}")
    
    prompt_parts.append("that would be engaging and well-suited for the specified preferences")
    
    return " ".join(prompt_parts) + "."


@mcp.prompt()
def anime_comparison_prompt(anime_title: str) -> str:
    """Create a prompt for finding anime similar to a specific title."""
    return (
        f"Find anime similar to '{anime_title}' in terms of themes, genre, "
        f"art style, or storytelling approach. Include brief explanations of "
        f"why each recommendation is similar."
    )


@mcp.prompt()
def seasonal_anime_prompt(year: int, season: str) -> str:
    """Create a prompt for discovering seasonal anime."""
    return (
        f"Find the best anime from {season.title()} {year}. Include both "
        f"popular titles and hidden gems from that season, with brief "
        f"descriptions of what makes each worth watching."
    )


# MCP Resources
@mcp.resource("anime://database/stats")
async def database_stats() -> str:
    """Provides current anime database statistics and health information."""
    if not anime_handler:
        return "Database handler not initialized"

    try:
        stats = await anime_handler.get_database_stats()
        return f"Anime Database Stats: {stats}"
    except Exception as e:
        return f"Error getting stats: {str(e)}"


@mcp.resource("anime://database/schema")
async def database_schema() -> str:
    """Provides the anime database schema and field definitions."""
    schema = {
        "fields": {
            "anime_id": {"type": "string", "description": "Unique anime identifier"},
            "title": {"type": "string", "description": "Anime title"},
            "synopsis": {"type": "string", "description": "Anime synopsis/description"},
            "type": {"type": "string", "description": "Anime type (TV, Movie, OVA, etc.)"},
            "episodes": {"type": "integer", "description": "Number of episodes"},
            "year": {"type": "integer", "description": "Release year"},
            "season": {"type": "string", "description": "Release season"},
            "tags": {"type": "array", "description": "Genre tags"},
            "studios": {"type": "array", "description": "Animation studios"},
            "picture": {"type": "string", "description": "Cover image URL"},
            "data_quality_score": {"type": "float", "description": "Data completeness score (0-1)"},
        },
        "platform_ids": {
            "myanimelist_id": {"type": "integer", "platform": "MyAnimeList"},
            "anilist_id": {"type": "integer", "platform": "AniList"},
            "kitsu_id": {"type": "integer", "platform": "Kitsu"},
            "anidb_id": {"type": "integer", "platform": "AniDB"},
        },
        "vector_info": {
            "text_embedding_model": settings.fastembed_model,
            "text_vector_size": settings.qdrant_vector_size,
            "image_embedding_model": getattr(settings, "clip_model", "ViT-B/32"),
            "image_vector_size": getattr(settings, "image_vector_size", 512),
            "distance_metric": settings.qdrant_distance_metric,
            "multi_vector_enabled": getattr(settings, "enable_multi_vector", False),
        },
    }
    return f"Anime Database Schema: {schema}"


async def initialize_mcp_server():
    """Initialize the modern MCP server with handler pattern."""
    global anime_handler

    logger.info("Initializing Modern Anime Search MCP Server")

    # Initialize Qdrant client
    qdrant_client = QdrantClient(settings=settings)
    
    # Verify connection
    if not await qdrant_client.health_check():
        logger.error("Qdrant connection failed - MCP server may have limited functionality")
        raise RuntimeError("Cannot initialize MCP server without database connection")

    # Initialize anime handler with business logic
    anime_handler = AnimeHandler(qdrant_client, settings)
    
    logger.info("Modern MCP server initialized successfully")


def parse_arguments():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(
        description="Modern Anime Search MCP Server with handler pattern"
    )
    parser.add_argument(
        "--mode",
        choices=["stdio", "http", "sse", "streamable"],
        default=settings.server_mode,
        help="MCP server transport mode (default: from config)",
    )
    parser.add_argument(
        "--host",
        default=settings.mcp_host,
        help="Server host for HTTP modes (default: from config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.mcp_port,
        help="Server port for HTTP modes (default: from config)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    return parser.parse_args()


def main():
    """Main entry point for running the modern MCP server."""
    args = parse_arguments()

    # Configure logging
    log_level = "DEBUG" if args.verbose else settings.log_level
    logging.basicConfig(level=getattr(logging, log_level), format=settings.log_format)

    # Log server configuration
    logger.info(f"Starting Modern Anime Search MCP Server")
    logger.info(f"Transport mode: {args.mode}")
    if args.mode in ["http", "sse", "streamable"]:
        logger.info(f"HTTP server: {args.host}:{args.port}")

    async def init_and_run():
        """Initialize server in async context."""
        try:
            await initialize_mcp_server()
            logger.info("Modern MCP server initialized successfully")
        except Exception as e:
            logger.error(f"MCP server initialization error: {e}", exc_info=True)
            raise

    try:
        # Initialize server
        asyncio.run(init_and_run())

        # Run FastMCP server with appropriate transport
        if args.mode == "stdio":
            logger.info("Starting stdio transport (local mode)")
            mcp.run(transport="stdio")
        elif args.mode == "http":
            logger.info(f"Starting HTTP transport on {args.host}:{args.port}")
            mcp.run(transport="sse", host=args.host, port=args.port)
        elif args.mode == "sse":
            logger.info(f"Starting SSE transport on {args.host}:{args.port}")
            mcp.run(transport="sse", host=args.host, port=args.port)
        elif args.mode == "streamable":
            logger.info(f"Starting Streamable HTTP transport on {args.host}:{args.port}")
            mcp.run(transport="streamable", host=args.host, port=args.port)

    except KeyboardInterrupt:
        logger.info("Modern MCP server shutdown requested")
    except Exception as e:
        logger.error(f"Modern MCP server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()