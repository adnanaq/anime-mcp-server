"""
FastMCP Server for Anime Search and Discovery

Simple, clean MCP server implementation using FastMCP library.
Provides anime search tools to AI assistants via Model Context Protocol.
"""

import argparse
import asyncio
import logging
from typing import Any, Dict, List, Optional, Literal

from fastmcp import FastMCP
from mcp.server.fastmcp import Context
from pydantic import BaseModel, Field

from ..config import get_settings
from ..vector.qdrant_client import QdrantClient

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize FastMCP server with proper MCP metadata
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

# Global Qdrant client - initialized in main()
qdrant_client: Optional[QdrantClient] = None


# Pydantic Schema Models for Input Validation
class SearchAnimeInput(BaseModel):
    """Input schema for anime search with comprehensive validation."""
    
    query: str = Field(
        description="Natural language search query (e.g., 'romantic comedy school anime')",
        min_length=1,
        max_length=500
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of results to return"
    )
    genres: Optional[List[str]] = Field(
        None,
        description="List of anime genres to filter by (e.g., ['Action', 'Comedy'])"
    )
    year_range: Optional[List[int]] = Field(
        None,
        description="Year range as [start_year, end_year] (e.g., [2020, 2023])",
        min_length=2,
        max_length=2
    )
    anime_types: Optional[List[str]] = Field(
        None,
        description="List of anime types (e.g., ['TV', 'Movie', 'OVA'])"
    )
    studios: Optional[List[str]] = Field(
        None,
        description="List of animation studios (e.g., ['Mappa', 'Studio Ghibli'])"
    )
    exclusions: Optional[List[str]] = Field(
        None,
        description="List of genres/themes to exclude (e.g., ['Horror', 'Ecchi'])"
    )
    mood_keywords: Optional[List[str]] = Field(
        None,
        description="List of mood descriptors (e.g., ['dark', 'serious', 'funny'])"
    )


class AnimeDetailsInput(BaseModel):
    """Input schema for getting anime details."""
    
    anime_id: str = Field(
        description="Unique anime identifier",
        min_length=1
    )


class SimilarAnimeInput(BaseModel):
    """Input schema for finding similar anime."""
    
    anime_id: str = Field(
        description="Reference anime ID to find similar anime for",
        min_length=1
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum number of similar anime to return"
    )


class ImageSearchInput(BaseModel):
    """Input schema for image-based anime search."""
    
    image_data: str = Field(
        description="Base64 encoded image data (supports JPG, PNG, WebP formats)",
        min_length=1
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Maximum number of results to return"
    )


class MultimodalSearchInput(BaseModel):
    """Input schema for multimodal anime search."""
    
    query: str = Field(
        description="Text search query (e.g., 'mecha robots fighting')",
        min_length=1,
        max_length=500
    )
    image_data: Optional[str] = Field(
        None,
        description="Optional base64 encoded image for visual similarity"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=25,
        description="Maximum number of results to return"
    )
    text_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for text similarity (0.0-1.0), default 0.7 means 70% text, 30% image"
    )


def _build_search_filters(
    genres: Optional[List[str]] = None,
    year_range: Optional[List[int]] = None,
    anime_types: Optional[List[str]] = None,
    studios: Optional[List[str]] = None,
    exclusions: Optional[List[str]] = None,
    mood_keywords: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Build Qdrant-compatible filters from SearchIntent parameters.

    Args:
        genres: List of anime genres to include
        year_range: Year range as [start_year, end_year]
        anime_types: List of anime types (TV, Movie, etc.)
        studios: List of animation studios
        exclusions: List of genres/themes to exclude
        mood_keywords: List of mood descriptors

    Returns:
        Dictionary of filters compatible with QdrantClient.search() or None if no filters
    """
    filters = {}

    # Genre filters (include)
    if genres and any(genres):
        filters["tags"] = {"any": genres}

    # Year range filter
    if year_range and len(year_range) == 2:
        start_year, end_year = year_range
        if start_year and end_year:
            filters["year"] = {"gte": start_year, "lte": end_year}
        elif start_year:
            filters["year"] = {"gte": start_year}
        elif end_year:
            filters["year"] = {"lte": end_year}

    # Anime type filters
    if anime_types and any(anime_types):
        filters["type"] = {"any": anime_types}

    # Studio filters
    if studios and any(studios):
        filters["studios"] = {"any": studios}

    # Exclusion filters (handled by adding mood keywords to include and exclude opposites)
    if exclusions and any(exclusions):
        # For exclusions, we'll add them as negative filters
        # This will be handled in the QdrantClient._build_filter method
        filters["exclude_tags"] = exclusions

    # Mood keyword filters (treated as additional genre/tag filters)
    if mood_keywords and any(mood_keywords):
        # Combine with existing tag filters
        existing_tags = filters.get("tags", {}).get("any", [])
        if isinstance(existing_tags, list):
            combined_tags = existing_tags + mood_keywords
        else:
            combined_tags = mood_keywords
        filters["tags"] = {"any": combined_tags}

    # Return None if no meaningful filters were added
    return filters if filters else None


@mcp.tool()
async def search_anime_advanced(
    search_params: SearchAnimeInput,
    ctx: Optional[Context] = None,
) -> List[Dict[str, Any]]:
    """Search for anime using semantic search with comprehensive validation.
    
    Modern MCP implementation using Pydantic schema for input validation.
    """
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")

    # Extract validated parameters
    query = search_params.query
    limit = search_params.limit
    
    # CRITICAL DEBUG: Parameter validation for advanced search
    logger.error(f"🔍 MCP ADVANCED SEARCH DEBUG:")
    logger.error(f"  - search_params object: {search_params}")
    logger.error(f"  - Raw query: {query!r}")
    logger.error(f"  - Query type: {type(query)}")
    logger.error(f"  - Query length: {len(query) if query else 'None'}")
    logger.error(f"  - Limit: {limit!r}")
    
    # Fix empty query issue for advanced search too
    if not query or not query.strip():
        logger.error("  - FIXING: Empty query in advanced search, using default")
        query = "anime"  # Default fallback query
    
    # Build filters from validated parameters
    filters = _build_search_filters(
        genres=search_params.genres,
        year_range=search_params.year_range,
        anime_types=search_params.anime_types,
        studios=search_params.studios,
        exclusions=search_params.exclusions,
        mood_keywords=search_params.mood_keywords,
    )

    # Use MCP Context for enhanced logging if available
    if ctx:
        await ctx.info(f"Starting anime search for: '{query}' (limit: {limit})")
        if filters:
            await ctx.debug(f"Applied search filters: {filters}")
    
    logger.info(f"Advanced search request: '{query}' (limit: {limit})")
    if filters:
        logger.debug(f"Applied filters: {filters}")

    try:
        # Use enhanced search with filters if available
        results = await qdrant_client.search(query=query, limit=limit, filters=filters)
        
        if ctx:
            await ctx.info(f"Search completed: found {len(results)} results")
            
        logger.info(f"Found {len(results)} results for query: '{query}'")
        return results
    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        if ctx:
            await ctx.error(f"Anime search error: {error_msg}")
            
        logger.error(f"Search failed: {e}")
        logger.error(f"Failed with query: {query!r}, limit: {limit!r}")
        raise RuntimeError(error_msg)


# Keep the original function for backward compatibility
@mcp.tool()
async def search_anime(
    query: str,
    limit: int = 10,
    genres: Optional[List[str]] = None,
    year_range: Optional[List[int]] = None,
    anime_types: Optional[List[str]] = None,
    studios: Optional[List[str]] = None,
    exclusions: Optional[List[str]] = None,
    mood_keywords: Optional[List[str]] = None,
    ctx: Optional[Context] = None,
) -> List[Dict[str, Any]]:
    """Search for anime using semantic search with natural language queries.

    Args:
        query: Natural language search query (e.g., "romantic comedy school anime")
        limit: Maximum number of results to return (default: 10, max: 50)
        genres: List of anime genres to filter by (e.g., ["Action", "Comedy"])
        year_range: Year range as [start_year, end_year] (e.g., [2020, 2023])
        anime_types: List of anime types (e.g., ["TV", "Movie", "OVA"])
        studios: List of animation studios (e.g., ["Mappa", "Studio Ghibli"])
        exclusions: List of genres/themes to exclude (e.g., ["Horror", "Ecchi"])
        mood_keywords: List of mood descriptors (e.g., ["dark", "serious", "funny"])

    Returns:
        List of anime matching the search query with metadata
    """
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")

    # Validate limit
    limit = min(max(1, limit), 50)

    # Build filters from SearchIntent-style parameters
    filters = _build_search_filters(
        genres=genres,
        year_range=year_range,
        anime_types=anime_types,
        studios=studios,
        exclusions=exclusions,
        mood_keywords=mood_keywords,
    )

    # Use MCP Context for enhanced logging if available
    if ctx:
        await ctx.info(f"Starting anime search for: '{query}' (limit: {limit})")
        if filters:
            await ctx.debug(f"Applied search filters: {filters}")
    
    # CRITICAL DEBUG: Parameter validation and debugging
    logger.error(f"🔍 MCP SEARCH DEBUG - Parameter Analysis:")
    logger.error(f"  - Raw query: {query!r}")
    logger.error(f"  - Query type: {type(query)}")
    logger.error(f"  - Query length: {len(query) if query else 'None'}")
    logger.error(f"  - Query bool: {bool(query)}")
    logger.error(f"  - Limit: {limit!r}")
    logger.error(f"  - Filters: {filters!r}")
    logger.error(f"  - Process ID: {__import__('os').getpid()}")
    
    # Fix empty query issue
    if not query or not query.strip():
        logger.error("  - FIXING: Empty query detected, using default")
        query = "anime"  # Default fallback query
    
    logger.info(f"Search request: '{query}' (limit: {limit})")
    if filters:
        logger.debug(f"Applied filters: {filters}")

    try:
        # Use enhanced search with filters if available
        results = await qdrant_client.search(query=query, limit=limit, filters=filters)
        
        if ctx:
            await ctx.info(f"Search completed: found {len(results)} results")
            
        logger.info(f"Found {len(results)} results for query: '{query}'")
        return results
    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        if ctx:
            await ctx.error(f"Anime search error: {error_msg}")
            
        logger.error(f"Search failed: {e}")
        logger.error(f"Failed with query: {query!r}, limit: {limit!r}")
        raise RuntimeError(error_msg)


@mcp.tool()
async def get_anime_details(anime_id: str, ctx: Optional[Context] = None) -> Dict[str, Any]:
    """Get detailed information about a specific anime by ID.

    Args:
        anime_id: Unique anime identifier

    Returns:
        Detailed anime information including synopsis, tags, studios, platform IDs
    """
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")

    if ctx:
        await ctx.info(f"Fetching details for anime ID: {anime_id}")
        
    logger.info(f"MCP details request for anime: {anime_id}")

    try:
        anime = await qdrant_client.get_by_id(anime_id)
        if not anime:
            error_msg = f"Anime not found: {anime_id}"
            if ctx:
                await ctx.warning(error_msg)
            raise ValueError(error_msg)

        title = anime.get('title', 'Unknown')
        if ctx:
            await ctx.info(f"Successfully retrieved details for: {title}")
            
        logger.info(f"Retrieved details for anime: {title}")
        return anime
    except Exception as e:
        error_msg = f"Failed to get anime details: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        logger.error(f"Failed to get anime details: {e}")
        raise RuntimeError(error_msg)


@mcp.tool()
async def find_similar_anime(anime_id: str, limit: int = 10, ctx: Optional[Context] = None) -> List[Dict[str, Any]]:
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


@mcp.tool()
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
                "vector_model": settings.fastembed_model,
            },
        }

        logger.info(f"Database stats: {stats.get('total_documents', 0)} anime entries")
        return result
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise RuntimeError(f"Failed to get database stats: {str(e)}")


# Phase 4: Multi-Modal Image Search Tools
@mcp.tool()
async def search_anime_by_image(
    image_data: str, limit: int = 10
) -> List[Dict[str, Any]]:
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

    # Multi-vector support is always enabled

    # Validate limit
    limit = min(max(1, limit), 30)

    logger.info(f"MCP image search request (limit: {limit})")

    try:
        results = await qdrant_client.search_by_image(
            image_data=image_data, limit=limit
        )
        logger.info(f"Found {len(results)} visually similar anime")
        return results
    except Exception as e:
        logger.error(f"Image search failed: {e}")
        raise RuntimeError(f"Image search failed: {str(e)}")


@mcp.tool()
async def find_visually_similar_anime(
    anime_id: str, limit: int = 10
) -> List[Dict[str, Any]]:
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

    # Multi-vector support is always enabled

    # Validate limit
    limit = min(max(1, limit), 20)

    logger.info(f"MCP visual similarity request for anime: {anime_id} (limit: {limit})")

    try:
        results = await qdrant_client.find_visually_similar_anime(
            anime_id=anime_id, limit=limit
        )
        logger.info(f"Found {len(results)} visually similar anime for: {anime_id}")
        return results
    except Exception as e:
        logger.error(f"Visual similarity search failed: {e}")
        raise RuntimeError(f"Visual similarity search failed: {str(e)}")


@mcp.tool()
async def search_multimodal_anime(
    query: str,
    image_data: Optional[str] = None,
    limit: int = 10,
    text_weight: float = 0.7,
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

    logger.info(
        f"MCP multimodal search: '{query}' with image={image_data is not None} (limit: {limit}, text_weight: {text_weight})"
    )

    try:
        # Multi-vector support is always available for enhanced search
        if image_data:
            results = await qdrant_client.search_multimodal(
                query=query, image_data=image_data, limit=limit, text_weight=text_weight
            )
        else:
            # Fallback to text-only search if multi-vector not available
            if image_data:
                logger.warning(
                    "Multi-vector not enabled, falling back to text-only search"
                )
            results = await qdrant_client.search(query=query, limit=limit)

        logger.info(f"Found {len(results)} multimodal search results")
        return results
    except Exception as e:
        logger.error(f"Multimodal search failed: {e}")
        raise RuntimeError(f"Multimodal search failed: {str(e)}")


# MCP Prompts for Anime Discovery
@mcp.prompt()
def anime_discovery_prompt(
    genre: str = "any",
    mood: str = "any", 
    content_type: str = "any"
) -> str:
    """Create an optimized prompt for anime discovery based on preferences.
    
    Args:
        genre: Preferred anime genre (e.g., "action", "romance", "slice of life")
        mood: Desired mood (e.g., "lighthearted", "dark", "emotional", "exciting")
        content_type: Type of content (e.g., "TV series", "movie", "OVA")
    
    Returns:
        Optimized search prompt for anime discovery
    """
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
    """Create a prompt for finding anime similar to a specific title.
    
    Args:
        anime_title: Name of the reference anime
        
    Returns:
        Prompt for finding similar anime
    """
    return (
        f"Find anime similar to '{anime_title}' in terms of themes, genre, "
        f"art style, or storytelling approach. Include brief explanations of "
        f"why each recommendation is similar."
    )


@mcp.prompt()
def anime_analysis_prompt(anime_title: str, analysis_type: str = "overview") -> str:
    """Create a prompt for analyzing anime characteristics.
    
    Args:
        anime_title: Name of the anime to analyze
        analysis_type: Type of analysis ("overview", "themes", "characters", "production")
        
    Returns:
        Prompt for anime analysis
    """
    analysis_prompts = {
        "overview": f"Provide a comprehensive overview of '{anime_title}' including plot, characters, and overall appeal.",
        "themes": f"Analyze the major themes and deeper meanings in '{anime_title}'.",
        "characters": f"Analyze the main characters in '{anime_title}' and their development throughout the series.",
        "production": f"Discuss the production quality, animation style, and technical aspects of '{anime_title}'."
    }
    
    return analysis_prompts.get(
        analysis_type, 
        f"Analyze '{anime_title}' focusing on {analysis_type}"
    )


@mcp.prompt()
def seasonal_anime_prompt(year: int, season: str) -> str:
    """Create a prompt for discovering seasonal anime.
    
    Args:
        year: Year to search (e.g., 2024)
        season: Season to search ("spring", "summer", "fall", "winter")
        
    Returns:
        Prompt for seasonal anime discovery
    """
    return (
        f"Find the best anime from {season.title()} {year}. Include both "
        f"popular titles and hidden gems from that season, with brief "
        f"descriptions of what makes each worth watching."
    )


@mcp.resource("anime://database/stats")
async def database_stats() -> str:
    """Provides current anime database statistics and health information."""
    if not qdrant_client:
        return "Database client not initialized"

    try:
        stats = await get_anime_stats.fn()
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
            "type": {
                "type": "string",
                "description": "Anime type (TV, Movie, OVA, etc.)",
            },
            "episodes": {"type": "integer", "description": "Number of episodes"},
            "year": {"type": "integer", "description": "Release year"},
            "season": {"type": "string", "description": "Release season"},
            "tags": {"type": "array", "description": "Genre tags"},
            "studios": {"type": "array", "description": "Animation studios"},
            "picture": {"type": "string", "description": "Cover image URL"},
            "data_quality_score": {
                "type": "float",
                "description": "Data completeness score (0-1)",
            },
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
            "multi_vector_enabled": True,  # Always enabled
        },
    }
    return f"Anime Database Schema: {schema}"


def initialize_qdrant_client():
    """Initialize the Qdrant client synchronously if not already initialized."""
    global qdrant_client

    if qdrant_client is None:
        logger.info("Initializing MCP tools Qdrant client")
        qdrant_client = QdrantClient(settings=settings)
        logger.info("MCP tools Qdrant client initialized")


# Import and mount ALL platform-specific tool servers  
# Using correct FastMCP mounting syntax (server first, then optional prefix)
from .tools import (
    jikan_tools, mal_tools, anilist_tools, schedule_tools, 
    kitsu_tools, semantic_tools, enrichment_tools
)

# Mount ALL platform tool servers with the main MCP server
# This enables intelligent routing to external platforms
mcp.mount(jikan_tools.mcp)           # Jikan/MAL tools (3 tools)
mcp.mount(mal_tools.mcp)             # Official MAL tools (3 tools)  
mcp.mount(anilist_tools.mcp)         # AniList GraphQL tools (2 tools)
mcp.mount(schedule_tools.mcp)        # AnimeSchedule tools (3 tools)
mcp.mount(kitsu_tools.mcp)           # Kitsu JSON:API tools (3 tools) 
mcp.mount(semantic_tools.mcp)        # Semantic search tools (3 tools)
mcp.mount(enrichment_tools.mcp)      # Cross-platform enrichment (5 tools)


async def initialize_mcp_server():
    """Initialize the MCP server with Qdrant client."""
    global qdrant_client

    logger.info("Initializing Anime Search MCP Server")

    # DEBUG: Log process and configuration details
    logger.error(f"🔍 MCP SERVER INIT DEBUG:")
    logger.error(f"  - Process ID: {__import__('os').getpid()}")
    logger.error(f"  - Settings: {settings}")
    logger.error(f"  - Qdrant URL: {settings.qdrant_url}")
    logger.error(f"  - Collection: {settings.qdrant_collection_name}")

    # Initialize Qdrant client
    qdrant_client = QdrantClient(settings=settings)
    
    # DEBUG: Log client details after initialization
    logger.error(f"  - QdrantClient created: {qdrant_client}")
    logger.error(f"  - Client ID: {id(qdrant_client)}")

    # Verify connection
    if not await qdrant_client.health_check():
        logger.error(
            "Qdrant connection failed - MCP server may have limited functionality"
        )
        raise RuntimeError("Cannot initialize MCP server without database connection")

    # DEBUG: Test search capability during initialization
    try:
        test_results = await qdrant_client.search(query="test", limit=1)
        logger.error(f"  - Test search results: {len(test_results)} items")
        if test_results:
            logger.error(f"  - Sample result: {test_results[0].get('title', 'No title')}")
        
        # CRITICAL TEST: Compare embeddings between processes
        test_embedding = qdrant_client._create_embedding("science fiction")
        logger.error(f"  - 'science fiction' embedding (first 5 dims): {test_embedding[:5] if test_embedding else None}")
        
        # Test the exact failing query
        scifi_results = await qdrant_client.search(query="science fiction", limit=1)
        logger.error(f"  - 'science fiction' search results: {len(scifi_results)} items")
        
    except Exception as test_error:
        logger.error(f"  - Test search failed: {test_error}")

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
    """Main entry point for running the MCP server."""
    # Parse command line arguments
    args = parse_arguments()

    # Override log level if verbose
    log_level = "DEBUG" if args.verbose else settings.log_level

    # Configure logging
    logging.basicConfig(level=getattr(logging, log_level), format=settings.log_format)

    # Log server configuration
    logger.info(f"🚀 Starting Anime Search MCP Server")
    logger.info(f"📡 Transport mode: {args.mode}")
    if args.mode in ["http", "sse", "streamable"]:
        logger.info(f"🌐 HTTP server: {args.host}:{args.port}")
    logger.info(
        f"📊 Database: {settings.qdrant_collection_name} ({settings.qdrant_url})"
    )

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
            logger.info(
                f"🌐 Starting Streamable HTTP transport on {args.host}:{args.port}"
            )
            mcp.run(transport="streamable", host=args.host, port=args.port)

    except KeyboardInterrupt:
        logger.info("🛑 MCP server shutdown requested")
    except Exception as e:
        logger.error(f"❌ MCP server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
