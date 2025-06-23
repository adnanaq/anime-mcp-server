"""
FastMCP Server for Anime Search and Discovery

Simple, clean MCP server implementation using FastMCP library.
Provides anime search tools to AI assistants via Model Context Protocol.
"""

import asyncio
import logging
import argparse
from typing import List, Dict, Any, Optional

from fastmcp import FastMCP

from ..config import get_settings
from ..vector.qdrant_client import QdrantClient

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize FastMCP server
mcp = FastMCP("Anime Search Server")

# Global Qdrant client - initialized in main()
qdrant_client: Optional[QdrantClient] = None


@mcp.tool
async def search_anime(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search for anime using semantic search with natural language queries.
    
    Args:
        query: Natural language search query (e.g., "romantic comedy school anime")
        limit: Maximum number of results to return (default: 10, max: 50)
    
    Returns:
        List of anime matching the search query with metadata
    """
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")
    
    # Validate limit
    limit = min(max(1, limit), 50)
    
    logger.info(f"MCP search request: '{query}' (limit: {limit})")
    
    try:
        results = await qdrant_client.search(query=query, limit=limit)
        logger.info(f"Found {len(results)} results for query: '{query}'")
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise RuntimeError(f"Search failed: {str(e)}")


@mcp.tool
async def get_anime_details(anime_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific anime by ID.
    
    Args:
        anime_id: Unique anime identifier
    
    Returns:
        Detailed anime information including synopsis, tags, studios, platform IDs
    """
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")
    
    logger.info(f"MCP details request for anime: {anime_id}")
    
    try:
        anime = await qdrant_client.get_by_id(anime_id)
        if not anime:
            raise ValueError(f"Anime not found: {anime_id}")
        
        logger.info(f"Retrieved details for anime: {anime.get('title', 'Unknown')}")
        return anime
    except Exception as e:
        logger.error(f"Failed to get anime details: {e}")
        raise RuntimeError(f"Failed to get anime details: {str(e)}")


@mcp.tool
async def find_similar_anime(anime_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Find anime similar to a given anime by ID.
    
    Args:
        anime_id: Reference anime ID to find similar anime for
        limit: Maximum number of similar anime to return (default: 10, max: 20)
    
    Returns:
        List of similar anime with similarity scores
    """
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")
    
    # Validate limit
    limit = min(max(1, limit), 20)
    
    logger.info(f"MCP similarity request for anime: {anime_id} (limit: {limit})")
    
    try:
        results = await qdrant_client.find_similar(anime_id=anime_id, limit=limit)
        logger.info(f"Found {len(results)} similar anime for: {anime_id}")
        return results
    except Exception as e:
        logger.error(f"Similar anime search failed: {e}")
        raise RuntimeError(f"Similar anime search failed: {str(e)}")


@mcp.tool
async def get_anime_stats() -> Dict[str, Any]:
    """Get statistics about the anime database.
    
    Returns:
        Database statistics including total entries, health status, and configuration
    """
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")
    
    logger.info("MCP stats request")
    
    try:
        stats = await qdrant_client.get_stats()
        health = await qdrant_client.health_check()
        
        result = {
            **stats,
            "health_status": "healthy" if health else "unhealthy",
            "server_info": {
                "qdrant_url": settings.qdrant_url,
                "collection_name": settings.qdrant_collection_name,
                "vector_model": settings.fastembed_model
            }
        }
        
        logger.info(f"Database stats: {stats.get('total_documents', 0)} anime entries")
        return result
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise RuntimeError(f"Failed to get database stats: {str(e)}")


@mcp.tool
async def recommend_anime(
    genres: Optional[str] = None,
    year: Optional[int] = None,
    anime_type: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Get anime recommendations based on preferences.
    
    Args:
        genres: Preferred genres (comma-separated, e.g., "Action,Comedy")
        year: Preferred release year (e.g., 2020)
        anime_type: Preferred type (TV, Movie, OVA, ONA, Special)
        limit: Maximum number of recommendations (default: 10, max: 25)
    
    Returns:
        List of recommended anime matching preferences
    """
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")
    
    # Validate limit
    limit = min(max(1, limit), 25)
    
    # Build search query from preferences
    query_parts = []
    if genres:
        genre_list = [g.strip() for g in genres.split(",")]
        query_parts.extend(genre_list)
    if year:
        query_parts.append(str(year))
    if anime_type:
        query_parts.append(anime_type)
    
    query = " ".join(query_parts) if query_parts else "popular anime"
    
    logger.info(f"MCP recommendation request: genres={genres}, year={year}, type={anime_type}")
    
    try:
        results = await qdrant_client.search(query=query, limit=limit)
        
        # Filter results by preferences if specified
        filtered_results = []
        for anime in results:
            # Check year filter
            if year and anime.get("year") != year:
                continue
            
            # Check type filter
            if anime_type and anime.get("type", "").lower() != anime_type.lower():
                continue
            
            # Check genre filter
            if genres:
                anime_tags = [tag.lower() for tag in anime.get("tags", [])]
                requested_genres = [g.strip().lower() for g in genres.split(",")]
                if not any(genre in anime_tags for genre in requested_genres):
                    continue
            
            filtered_results.append(anime)
            
            if len(filtered_results) >= limit:
                break
        
        logger.info(f"Generated {len(filtered_results)} recommendations")
        return filtered_results
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise RuntimeError(f"Recommendation failed: {str(e)}")


# Phase 4: Multi-Modal Image Search Tools
@mcp.tool
async def search_anime_by_image(image_data: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search for anime using visual similarity from an uploaded image.
    
    Find anime with poster images visually similar to the provided image.
    Requires multi-vector support to be enabled in server configuration.
    
    Args:
        image_data: Base64 encoded image data (supports JPG, PNG, WebP formats)
        limit: Maximum number of results to return (default: 10, max: 30)
    
    Returns:
        List of anime with visual similarity scores and metadata
    """
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")
    
    # Check if multi-vector support is enabled
    if not getattr(qdrant_client, '_supports_multi_vector', False):
        raise RuntimeError("Multi-vector image search not enabled. Enable in server configuration.")
    
    # Validate limit
    limit = min(max(1, limit), 30)
    
    logger.info(f"MCP image search request (limit: {limit})")
    
    try:
        results = await qdrant_client.search_by_image(image_data=image_data, limit=limit)
        logger.info(f"Found {len(results)} visually similar anime")
        return results
    except Exception as e:
        logger.error(f"Image search failed: {e}")
        raise RuntimeError(f"Image search failed: {str(e)}")


@mcp.tool
async def find_visually_similar_anime(anime_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Find anime with similar visual style to a reference anime.
    
    Uses the poster image of the reference anime to find visually similar anime.
    Requires multi-vector support to be enabled in server configuration.
    
    Args:
        anime_id: Reference anime ID to find visually similar anime for
        limit: Maximum number of similar anime to return (default: 10, max: 20)
    
    Returns:
        List of visually similar anime with similarity scores
    """
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")
    
    # Check if multi-vector support is enabled
    if not getattr(qdrant_client, '_supports_multi_vector', False):
        raise RuntimeError("Multi-vector visual similarity not enabled. Enable in server configuration.")
    
    # Validate limit
    limit = min(max(1, limit), 20)
    
    logger.info(f"MCP visual similarity request for anime: {anime_id} (limit: {limit})")
    
    try:
        results = await qdrant_client.find_visually_similar_anime(anime_id=anime_id, limit=limit)
        logger.info(f"Found {len(results)} visually similar anime for: {anime_id}")
        return results
    except Exception as e:
        logger.error(f"Visual similarity search failed: {e}")
        raise RuntimeError(f"Visual similarity search failed: {str(e)}")


@mcp.tool
async def search_multimodal_anime(
    query: str, 
    image_data: Optional[str] = None, 
    limit: int = 10,
    text_weight: float = 0.7
) -> List[Dict[str, Any]]:
    """Search for anime using both text query and image similarity.
    
    Combines semantic text search with visual image search for enhanced discovery.
    The text_weight parameter controls the balance between text and image matching.
    
    Args:
        query: Text search query (e.g., "mecha robots fighting")
        image_data: Optional base64 encoded image for visual similarity
        limit: Maximum number of results to return (default: 10, max: 25)
        text_weight: Weight for text similarity (0.0-1.0), default 0.7 means 70% text, 30% image
    
    Returns:
        List of anime with combined similarity scores (text + image)
    """
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")
    
    # Validate parameters
    limit = min(max(1, limit), 25)
    text_weight = max(0.0, min(1.0, text_weight))
    
    logger.info(f"MCP multimodal search: '{query}' with image={image_data is not None} (limit: {limit}, text_weight: {text_weight})")
    
    try:
        # Check if multi-vector support is available for enhanced search
        if getattr(qdrant_client, '_supports_multi_vector', False) and image_data:
            results = await qdrant_client.search_multimodal(
                query=query, 
                image_data=image_data, 
                limit=limit, 
                text_weight=text_weight
            )
        else:
            # Fallback to text-only search if multi-vector not available
            if image_data:
                logger.warning("Multi-vector not enabled, falling back to text-only search")
            results = await qdrant_client.search(query=query, limit=limit)
        
        logger.info(f"Found {len(results)} multimodal search results")
        return results
    except Exception as e:
        logger.error(f"Multimodal search failed: {e}")
        raise RuntimeError(f"Multimodal search failed: {str(e)}")


@mcp.resource("anime://database/stats")
async def database_stats() -> str:
    """Provides current anime database statistics and health information."""
    if not qdrant_client:
        return "Database client not initialized"
    
    try:
        stats = await get_anime_stats()
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
            "data_quality_score": {"type": "float", "description": "Data completeness score (0-1)"}
        },
        "platform_ids": {
            "myanimelist_id": {"type": "integer", "platform": "MyAnimeList"},
            "anilist_id": {"type": "integer", "platform": "AniList"},
            "kitsu_id": {"type": "integer", "platform": "Kitsu"},
            "anidb_id": {"type": "integer", "platform": "AniDB"}
        },
        "vector_info": {
            "text_embedding_model": settings.fastembed_model,
            "text_vector_size": settings.qdrant_vector_size,
            "image_embedding_model": getattr(settings, 'clip_model', 'ViT-B/32'),
            "image_vector_size": getattr(settings, 'image_vector_size', 512),
            "distance_metric": settings.qdrant_distance_metric,
            "multi_vector_enabled": getattr(settings, 'enable_multi_vector', False)
        }
    }
    return f"Anime Database Schema: {schema}"


async def initialize_mcp_server():
    """Initialize the MCP server with Qdrant client."""
    global qdrant_client
    
    logger.info("Initializing Anime Search MCP Server")
    
    # Initialize Qdrant client
    qdrant_client = QdrantClient(settings=settings)
    
    # Verify connection
    if not await qdrant_client.health_check():
        logger.error("Qdrant connection failed - MCP server may have limited functionality")
        raise RuntimeError("Cannot initialize MCP server without database connection")
    
    logger.info("Qdrant connection verified - MCP server ready")


def parse_arguments():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(
        description="Anime Search MCP Server with dual protocol support"
    )
    parser.add_argument(
        "--mode", 
        choices=["stdio", "http", "sse", "streamable"],
        default=settings.server_mode,
        help="MCP server transport mode (default: from config)"
    )
    parser.add_argument(
        "--host",
        default=settings.mcp_host,
        help="Server host for HTTP modes (default: from config)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.mcp_port,
        help="Server port for HTTP modes (default: from config)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    return parser.parse_args()


def main():
    """Main entry point for running the MCP server."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Override log level if verbose
    log_level = "DEBUG" if args.verbose else settings.log_level
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=settings.log_format
    )
    
    # Log server configuration
    logger.info(f"🚀 Starting Anime Search MCP Server")
    logger.info(f"📡 Transport mode: {args.mode}")
    if args.mode in ["http", "sse", "streamable"]:
        logger.info(f"🌐 HTTP server: {args.host}:{args.port}")
    logger.info(f"📊 Database: {settings.qdrant_collection_name} ({settings.qdrant_url})")
    
    async def init_and_run():
        """Initialize server in async context."""
        try:
            await initialize_mcp_server()
            logger.info("✅ MCP server initialized successfully")
        except Exception as e:
            logger.error(f"❌ MCP server initialization error: {e}", exc_info=True)
            raise
    
    try:
        # Initialize server synchronously
        asyncio.run(init_and_run())
        
        # Run FastMCP server with appropriate transport
        if args.mode == "stdio":
            logger.info("🔌 Starting stdio transport (local mode)")
            mcp.run(transport="stdio")
        elif args.mode == "http":
            logger.info(f"🌐 Starting HTTP transport on {args.host}:{args.port}")
            mcp.run(transport="sse", host=args.host, port=args.port)
        elif args.mode == "sse":
            logger.info(f"🌐 Starting SSE transport on {args.host}:{args.port}")
            mcp.run(transport="sse", host=args.host, port=args.port)
        elif args.mode == "streamable":
            logger.info(f"🌐 Starting Streamable HTTP transport on {args.host}:{args.port}")
            mcp.run(transport="streamable", host=args.host, port=args.port)
        
    except KeyboardInterrupt:
        logger.info("🛑 MCP server shutdown requested")
    except Exception as e:
        logger.error(f"❌ MCP server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()