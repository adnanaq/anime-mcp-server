"""
MyAnimeList (MAL) specific MCP tools.

Focused tools for MAL platform with clear parameter schemas and capabilities.
MAL specializes in community data, ratings, and content filtering.
"""

from typing import Dict, List, Literal, Optional, Any
from datetime import datetime
from fastmcp import FastMCP
from mcp.server.fastmcp import Context

from ...integrations.clients.mal_client import MALClient
from ...integrations.mappers.mal_mapper import MALMapper
from ...models.universal_anime import UniversalSearchParams
from ...config import get_settings

# Initialize components
settings = get_settings()
mal_client = MALClient(
    client_id=settings.mal_client_id,
    client_secret=settings.mal_client_secret
) if hasattr(settings, 'mal_client_id') and settings.mal_client_id else None
mal_mapper = MALMapper()

# Create FastMCP instance for tools
mcp = FastMCP("MAL Tools")


async def _search_anime_mal_impl(
    query: str,
    limit: int = 20,
    offset: int = 0,
    fields: Optional[List[str]] = None,
    ctx: Optional[Context] = None
) -> List[Dict[str, Any]]:
    """
    Search anime on MyAnimeList with response field selection.
    
    MAL API v2 Reality:
    - Only supports query (q), limit, offset as search parameters
    - All other data (rating, nsfw, source, broadcast, etc.) are response fields
    - Use fields parameter to control what data is returned
    
    Args:
        query: Search query for anime titles
        limit: Maximum number of results (default: 20, max: 100)
        offset: Pagination offset (default: 0)
        fields: List of response fields to return (optional)
        
    Returns:
        List of anime with MAL-specific data based on requested fields
    """
    if not mal_client:
        if ctx:
            await ctx.error("MAL client not configured - missing API key")
        raise RuntimeError("MAL client not available")
    
    if ctx:
        await ctx.info(f"Searching MAL for '{query}' with {limit} results")
    
    try:
        # Create UniversalSearchParams for mapping
        universal_params = UniversalSearchParams(
            query=query,
            limit=min(limit, 100),
            offset=offset
        )
        
        # Build MAL-specific parameters (only limit/offset)
        mal_specific = {
            "limit": min(limit, 100),
            "offset": offset
        }
        
        # Use mapper to convert parameters
        mal_params = mal_mapper.to_mal_search_params(universal_params, mal_specific)
        
        # Add fields parameter - use provided fields or default comprehensive set from MAL mapper
        if fields:
            mal_params["fields"] = ",".join(fields)
        else:
            # All available MAL API fields from mal_mapper.py (lines 78-156)
            default_fields = [
                # Core response fields (lines 78-121)
                "id", "title", "status", "media_type", "num_episodes", "mean", 
                "genres", "start_date", "end_date", "synopsis", "popularity", 
                "rank", "source", "rating", "studios",
                # MAL-specific response fields (lines 124-155)  
                "alternative_titles", "my_list_status", "num_list_users", 
                "num_scoring_users", "nsfw", "average_episode_duration", 
                "start_season", "broadcast", "main_picture", "created_at", 
                "updated_at"
            ]
            mal_params["fields"] = ",".join(default_fields)
            
        # Execute search and return raw MAL results directly (like Jikan)
        raw_results = await mal_client.search_anime(**mal_params)
        
        # Add source attribution to each result
        for result in raw_results:
            if isinstance(result, dict):
                result["source_platform"] = "mal"
        
        if ctx:
            await ctx.info(f"Found {len(raw_results)} anime via MAL")
            
        return raw_results
        
    except Exception as e:
        error_msg = f"MAL search failed: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)


# Export the implementation function for testing
search_anime_mal = _search_anime_mal_impl


# MCP tool wrapper
@mcp.tool(
    name="search_anime_mal",
    description="Search anime on MyAnimeList with response field selection",
    annotations={
        "title": "MAL Anime Search",
        "readOnlyHint": True,
        "idempotentHint": True,
    }
)
async def search_anime_mal_mcp(
    query: str,
    limit: int = 20,
    offset: int = 0,
    fields: Optional[List[str]] = None,
    ctx: Optional[Context] = None
) -> List[Dict[str, Any]]:
    """MCP wrapper for search_anime_mal."""
    return await _search_anime_mal_impl(query, limit, offset, fields, ctx)


@mcp.tool(
    name="get_anime_mal",
    description="Get detailed anime information from MyAnimeList by ID",
    annotations={
        "title": "MAL Anime Details",
        "readOnlyHint": True,
        "idempotentHint": True,
    }
)
async def get_anime_mal(
    mal_id: int,
    include_statistics: bool = True,
    include_related: bool = True,
    ctx: Optional[Context] = None
) -> Optional[Dict[str, Any]]:
    """
    Get detailed anime information from MyAnimeList by MAL ID.
    
    Provides comprehensive MAL data including:
    - Community statistics and engagement metrics
    - Related anime (sequels, prequels, adaptations)
    - Detailed broadcast and production information
    - User list statistics and scoring data
    
    Args:
        mal_id: MyAnimeList anime ID
        include_statistics: Include community statistics (default: True)
        include_related: Include related anime information (default: True)
        
    Returns:
        Detailed anime information with MAL-specific data, or None if not found
    """
    if not mal_client:
        if ctx:
            await ctx.error("MAL client not configured - missing API key")
        raise RuntimeError("MAL client not available")
    
    if ctx:
        await ctx.info(f"Fetching MAL anime details for ID: {mal_id}")
    
    try:
        # Build request fields
        fields = [
            "id", "title", "main_picture", "alternative_titles",
            "start_date", "end_date", "synopsis", "mean", "rank", 
            "popularity", "num_list_users", "num_scoring_users",
            "nsfw", "created_at", "updated_at", "media_type",
            "status", "genres", "my_list_status", "num_episodes",
            "start_season", "broadcast", "source", "average_episode_duration",
            "rating", "pictures", "background", "related_anime",
            "related_manga", "recommendations", "studios"
        ]
        
        if include_statistics:
            fields.extend(["statistics"])
        if include_related:
            fields.extend(["related_anime", "related_manga"])
            
        # Execute request
        raw_result = await mal_client.get_anime_by_id(mal_id, fields=fields)
        
        if not raw_result:
            if ctx:
                await ctx.info(f"Anime with MAL ID {mal_id} not found")
            return None
        
        # Return raw MAL API response with source attribution
        result = {
            **raw_result,  # Include all raw MAL data
            "source_platform": "myanimelist",
            "mal_id": mal_id,
            "fetched_at": datetime.now().isoformat(),
            
            # Optional statistics
            "statistics": raw_result.get("statistics", {}) if include_statistics else {},
            
            # Optional related content
            "related_anime": raw_result.get("related_anime", []) if include_related else [],
            "related_manga": raw_result.get("related_manga", []) if include_related else [],
        }
        
        if ctx:
            await ctx.info(f"Retrieved detailed MAL data for '{result['title']}'")
            
        return result
        
    except Exception as e:
        error_msg = f"Failed to get MAL anime {mal_id}: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)


@mcp.tool(
    name="get_mal_seasonal_anime",
    description="Get seasonal anime from MyAnimeList with filtering",
    annotations={
        "title": "MAL Seasonal Anime",
        "readOnlyHint": True,
        "idempotentHint": True,
    }
)
async def get_mal_seasonal_anime(
    year: int,
    season: Literal["winter", "spring", "summer", "fall"],
    sort: Literal["anime_score", "anime_num_list_users"] = "anime_score",
    limit: int = 50,
    ctx: Optional[Context] = None
) -> List[Dict[str, Any]]:
    """
    Get seasonal anime from MyAnimeList with sorting options.
    
    Args:
        year: Year for the season
        season: Season (winter, spring, summer, fall)
        sort: Sort criteria (anime_score or anime_num_list_users)
        limit: Maximum results (default: 50, max: 500)
        
    Returns:
        List of seasonal anime with MAL community data
    """
    if not mal_client:
        if ctx:
            await ctx.error("MAL client not configured - missing API key")
        raise RuntimeError("MAL client not available")
    
    if ctx:
        await ctx.info(f"Fetching MAL seasonal anime for {season.title()} {year}")
    
    try:
        params = {
            "year": year,
            "season": season,
            "sort": sort,
            "limit": min(limit, 500)
        }
        
        raw_results = await mal_client.get_seasonal_anime(params)
        
        results = []
        for raw_result in raw_results:
            try:
                universal_anime = mal_mapper.to_universal_anime(raw_result)
                
                result = {
                    "id": universal_anime.id,
                    "title": universal_anime.title,
                    "type": universal_anime.type_format,
                    "episodes": universal_anime.episodes,
                    "score": universal_anime.score,
                    "year": universal_anime.year,
                    "status": universal_anime.status,
                    "genres": universal_anime.genres or [],
                    "studios": universal_anime.studios or [],
                    "synopsis": universal_anime.description,
                    "image_url": universal_anime.image_url,
                    
                    # MAL seasonal data
                    "mal_id": raw_result.get("id"),
                    "mal_score": raw_result.get("mean"),
                    "mal_popularity": raw_result.get("popularity"),
                    "mal_num_list_users": raw_result.get("num_list_users"),
                    "season": season,
                    "season_year": year,
                    
                    "source_platform": "mal",
                    "data_quality_score": universal_anime.data_quality_score
                }
                
                results.append(result)
                
            except Exception as e:
                if ctx:
                    await ctx.error(f"Failed to process seasonal result: {str(e)}")
                continue
        
        if ctx:
            await ctx.info(f"Found {len(results)} seasonal anime")
            
        return results
        
    except Exception as e:
        error_msg = f"MAL seasonal search failed: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)