"""
MyAnimeList (MAL) specific MCP tools.

Focused tools for MAL platform with clear parameter schemas and capabilities.
MAL specializes in community data, ratings, and content filtering.
"""

from typing import Dict, List, Literal, Optional, Any, Union
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

# Shared MAL API field definitions
MAL_DEFAULT_FIELDS = [
    # Core response fields
    "id", "title", "main_picture", "alternative_titles",
    "start_date", "end_date", "synopsis", "mean", "rank", 
    "popularity", "num_list_users", "num_scoring_users",
    "nsfw", "created_at", "updated_at", "media_type",
    "status", "genres", "my_list_status", "num_episodes",
    "start_season", "broadcast", "source", "average_episode_duration",
    "rating", "studios"
]


async def _search_anime_mal_impl(
    query: str,
    limit: int = 20,
    offset: int = 0,
    fields: Optional[Union[str, List[str]]] = None,
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
        fields: List of response fields to return (optional). Can be a comma-separated string or list of strings.
        
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
            # Handle both string and list inputs for flexibility with type safety
            if isinstance(fields, str):
                if fields.strip():  # Only process non-empty strings
                    mal_params["fields"] = fields
                # Empty string falls through to use default fields
            elif isinstance(fields, list):
                # Type-safe check that all elements are strings
                if all(isinstance(field, str) for field in fields):
                    mal_params["fields"] = ",".join(fields)
                else:
                    raise TypeError(f"All fields must be strings, got: {[type(f).__name__ for f in fields]}")
            else:
                raise TypeError(f"Fields must be string or list of strings, got: {type(fields).__name__}")
        
        # Use default fields if no fields specified or empty string
        if not fields or (isinstance(fields, str) and not fields.strip()):
            mal_params["fields"] = ",".join(MAL_DEFAULT_FIELDS)
            
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
    fields: Optional[Union[str, List[str]]] = None,
    ctx: Optional[Context] = None
) -> List[Dict[str, Any]]:
    """MCP wrapper for search_anime_mal."""
    return await _search_anime_mal_impl(query, limit, offset, fields, ctx)


async def _get_anime_mal_impl(
    mal_id: int,
    fields: Optional[Union[str, List[str]]] = None,
    ctx: Optional[Context] = None
) -> Optional[Dict[str, Any]]:
    """
    Get detailed anime information from MyAnimeList by MAL ID.
    
    MAL API v2 Reality:
    - Takes anime ID as path parameter
    - Uses fields parameter to control what data is returned
    - All data (statistics, related content, etc.) are response fields
    
    Args:
        mal_id: MyAnimeList anime ID
        fields: List of response fields to return (optional). Can be a comma-separated string or list of strings.
        
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
        # Handle fields parameter with type safety
        if fields:
            if isinstance(fields, str):
                if fields.strip():  # Only process non-empty strings
                    request_fields = fields.split(",")
                else:
                    request_fields = None  # Use default
            elif isinstance(fields, list):
                # Type-safe check that all elements are strings
                if all(isinstance(field, str) for field in fields):
                    request_fields = fields
                else:
                    raise TypeError(f"All fields must be strings, got: {[type(f).__name__ for f in fields]}")
            else:
                raise TypeError(f"Fields must be string or list of strings, got: {type(fields).__name__}")
        else:
            request_fields = None
        
        # Use default comprehensive fields if no fields specified or empty string
        if not request_fields:
            # All available MAL API fields for anime details
            request_fields = [
                # Core anime fields
                "id", "title", "main_picture", "alternative_titles",
                "start_date", "end_date", "synopsis", "mean", "rank", 
                "popularity", "num_list_users", "num_scoring_users",
                "nsfw", "created_at", "updated_at", "media_type",
                "status", "genres", "my_list_status", "num_episodes",
                "start_season", "broadcast", "source", "average_episode_duration",
                "rating", "pictures", "background", "studios",
                
                # Optional detailed fields
                "statistics", "related_anime", "related_manga", 
                "recommendations"
            ]
            
        # Execute request
        raw_result = await mal_client.get_anime_by_id(mal_id, fields=request_fields)
        
        if not raw_result:
            if ctx:
                await ctx.info(f"Anime with MAL ID {mal_id} not found")
            return None
        
        # Return raw MAL API response with source attribution
        result = {
            **raw_result,  # Include all raw MAL data
            "source_platform": "mal",
            "mal_id": mal_id,
            "fetched_at": datetime.now().isoformat()
        }
        
        if ctx:
            await ctx.info(f"Retrieved detailed MAL data for '{result.get('title', 'Unknown')}'")
            
        return result
        
    except Exception as e:
        error_msg = f"Failed to get MAL anime {mal_id}: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)


# MCP tool wrapper
@mcp.tool(
    name="get_anime_by_id_mal",
    description="Get detailed anime information from MyAnimeList by ID with response field selection",
    annotations={
        "title": "MAL Anime Details by ID",
        "readOnlyHint": True,
        "idempotentHint": True,
    }
)
async def get_anime_by_id_mal_mcp(
    mal_id: int,
    fields: Optional[Union[str, List[str]]] = None,
    ctx: Optional[Context] = None
) -> Optional[Dict[str, Any]]:
    """MCP wrapper for get_anime_by_id_mal."""
    return await _get_anime_mal_impl(mal_id, fields, ctx)


async def _get_seasonal_anime_mal_impl(
    year: int,
    season: Literal["winter", "spring", "summer", "fall"],
    sort: Literal["anime_score", "anime_num_list_users"] = "anime_score",
    limit: int = 50,
    offset: int = 0,
    fields: Optional[Union[str, List[str]]] = None,
    ctx: Optional[Context] = None
) -> List[Dict[str, Any]]:
    """
    Get seasonal anime from MyAnimeList with sorting and pagination.
    
    MAL API v2 Reality:
    - Takes year and season as path parameters
    - Uses sort, limit, offset, and fields as query parameters
    - Returns raw MAL data with node structure
    
    Args:
        year: Year for the season
        season: Season (winter, spring, summer, fall)
        sort: Sort criteria (anime_score or anime_num_list_users)
        limit: Maximum results (default: 50, max: 500)
        offset: Pagination offset (default: 0)
        fields: List of response fields to return (optional). Can be a comma-separated string or list of strings.
        
    Returns:
        List of seasonal anime with raw MAL data
    """
    if not mal_client:
        if ctx:
            await ctx.error("MAL client not configured - missing API key")
        raise RuntimeError("MAL client not available")
    
    if ctx:
        await ctx.info(f"Fetching MAL seasonal anime for {season.title()} {year} (limit: {limit}, offset: {offset})")
    
    try:
        # Build MAL API parameters
        params = {
            "year": year,
            "season": season,
            "sort": sort,
            "limit": min(limit, 500),
            "offset": offset
        }
        
        # Handle fields parameter with type safety
        if fields:
            if isinstance(fields, str):
                if fields.strip():  # Only process non-empty strings
                    params["fields"] = fields
                # Empty string falls through to use default fields
            elif isinstance(fields, list):
                # Type-safe check that all elements are strings
                if all(isinstance(field, str) for field in fields):
                    params["fields"] = ",".join(fields)
                else:
                    raise TypeError(f"All fields must be strings, got: {[type(f).__name__ for f in fields]}")
            else:
                raise TypeError(f"Fields must be string or list of strings, got: {type(fields).__name__}")
        
        # Use default fields if no fields specified or empty string
        if not fields or (isinstance(fields, str) and not fields.strip()):
            params["fields"] = ",".join(MAL_DEFAULT_FIELDS)
        
        # Execute request
        raw_results = await mal_client.get_seasonal_anime(
            year=year,
            season=season,
            sort=sort,
            limit=params["limit"],
            offset=params["offset"],
            fields=params.get("fields")
        )
        
        # Add source attribution to each result
        results = []
        for raw_result in raw_results:
            if isinstance(raw_result, dict):
                result = {
                    **raw_result,  # Include all raw MAL data
                    "source_platform": "mal",
                    "season": season,
                    "season_year": year,
                    "fetched_at": datetime.now().isoformat()
                }
                results.append(result)
        
        if ctx:
            await ctx.info(f"Found {len(results)} seasonal anime from MAL")
            
        return results
        
    except Exception as e:
        error_msg = f"MAL seasonal search failed: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)


# MCP tool wrapper
@mcp.tool(
    name="get_seasonal_anime_mal",
    description="Get seasonal anime from MyAnimeList with sorting, pagination, and field selection",
    annotations={
        "title": "MAL Seasonal Anime",
        "readOnlyHint": True,
        "idempotentHint": True,
    }
)
async def get_seasonal_anime_mal_mcp(
    year: int,
    season: Literal["winter", "spring", "summer", "fall"],
    sort: Literal["anime_score", "anime_num_list_users"] = "anime_score",
    limit: int = 50,
    offset: int = 0,
    fields: Optional[Union[str, List[str]]] = None,
    ctx: Optional[Context] = None
) -> List[Dict[str, Any]]:
    """MCP wrapper for get_seasonal_anime_mal."""
    return await _get_seasonal_anime_mal_impl(year, season, sort, limit, offset, fields, ctx)