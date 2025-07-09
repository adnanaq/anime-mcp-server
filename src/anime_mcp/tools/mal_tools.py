"""
MyAnimeList (MAL) specific MCP tools.

Focused tools for MAL platform with clear parameter schemas and capabilities.
MAL specializes in community data, ratings, and content filtering.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from fastmcp import FastMCP

from mcp.server.fastmcp import Context

from ...config import get_settings
from ...integrations.clients.mal_client import MALClient

# Initialize components
settings = get_settings()
mal_client = (
    MALClient(
        client_id=settings.mal_client_id, client_secret=settings.mal_client_secret
    )
    if hasattr(settings, "mal_client_id") and settings.mal_client_id
    else None
)
# Removed MALMapper - using direct API calls instead

# Create FastMCP instance for tools
mcp = FastMCP("MAL Tools")

# Shared MAL API field definitions for list operations (search, seasonal)
MAL_DEFAULT_FIELDS = [
    # Core field parameters
    "id",
    "title",
    "main_picture",
    "alternative_titles",
    "start_date",
    "end_date",
    "synopsis",
    "mean",
    "rank",
    "popularity",
    "num_list_users",
    "num_scoring_users",
    "nsfw",
    "created_at",
    "updated_at",
    "media_type",
    "status",
    "genres",
    "my_list_status",
    "num_episodes",
    "start_season",
    "broadcast",
    "source",
    "average_episode_duration",
    "rating",
    "studios",
]

# Additional fields only available for individual anime details (get_anime_by_id)
MAL_DETAIL_ONLY_FIELDS = [
    "pictures",  # Array of Picture objects - cannot be used in lists
    "background",  # Background text - cannot be used in lists
    "related_anime",  # Array of RelatedAnimeEdge - cannot be used in lists
    "related_manga",  # Array of RelatedMangaEdge - cannot be used in lists
    "recommendations",  # Array of recommendations - cannot be used in lists
    "statistics",  # Statistics object - cannot be used in lists
]

# Complete field set for anime details (list fields + detail-only fields)
MAL_DETAIL_FIELDS = MAL_DEFAULT_FIELDS + MAL_DETAIL_ONLY_FIELDS


async def _validate_mal_client(ctx: Optional[Context] = None) -> None:
    """Validate MAL client is available and configured."""
    if not mal_client:
        if ctx:
            await ctx.error("MAL client not configured - missing API key")
        raise RuntimeError("MAL client not available")


def _process_fields_parameter(
    fields: Optional[Union[str, List[str]]], default_fields: List[str]
) -> str:
    """Process and validate fields parameter, returning comma-separated string.

    Args:
        fields: Fields parameter (string, list, or None)
        default_fields: Default fields to use if none specified

    Returns:
        Comma-separated string of fields

    Raises:
        TypeError: If fields parameter has invalid type or content
    """
    # Validate fields parameter type safety
    if fields is not None:
        if isinstance(fields, str):
            # String fields are valid
            pass
        elif isinstance(fields, list):
            # Type-safe check that all elements are strings
            if not all(isinstance(field, str) for field in fields):
                raise TypeError(
                    f"All fields must be strings, got: {[type(f).__name__ for f in fields]}"
                )
        else:
            raise TypeError(
                f"Fields must be string or list of strings, got: {type(fields).__name__}"
            )

    # Process fields into comma-separated string
    if fields:
        if isinstance(fields, str):
            if fields.strip():  # Only process non-empty strings
                return fields
            # Empty string falls through to use default fields
        elif isinstance(fields, list):
            return ",".join(fields)

    # Use default fields if no fields specified or empty string
    return ",".join(default_fields)


async def _handle_mal_error(
    exception: Exception, error_message: str, ctx: Optional[Context] = None
) -> None:
    """Handle MAL API errors with consistent logging and exception raising."""
    if ctx:
        await ctx.error(error_message)
    raise RuntimeError(error_message)


async def _search_anime_mal_impl(
    query: str,
    limit: int = 20,
    offset: int = 0,
    fields: Optional[Union[str, List[str]]] = None,
    ctx: Optional[Context] = None,
) -> List[Dict[str, Any]]:
    """
    Search anime on MyAnimeList with response field selection.

    MAL API v2 Reality:
    - Only supports query (q), limit, offset as search parameters
    - All other data (rating, nsfw, source, broadcast, etc.) are field parameters
    - Use fields parameter to control what data is returned

    Args:
        query: Search query for anime titles
        limit: Maximum number of results (default: 20, max: 100)
        offset: Pagination offset (default: 0)
        fields: List of field parameters to return (optional). Can be a comma-separated string or list of strings.

    Returns:
        List of anime with MAL-specific data based on requested fields
    """
    await _validate_mal_client(ctx)

    if ctx:
        await ctx.info(f"Searching MAL for '{query}' with {limit} results")

    try:
        # Build MAL API parameters directly (simplified - MAL API has limited parameters)
        mal_params = {"q": query, "limit": min(limit, 100), "offset": offset}

        # Process fields parameter
        mal_params["fields"] = _process_fields_parameter(fields, MAL_DEFAULT_FIELDS)

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
        await _handle_mal_error(e, f"MAL search failed: {str(e)}", ctx)


# MCP tool wrapper
@mcp.tool(
    name="search_anime_mal",
    description="Search anime on MyAnimeList with response field selection",
    annotations={
        "title": "MAL Anime Search",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def search_anime_mal_mcp(
    query: str,
    limit: int = 20,
    offset: int = 0,
    fields: Optional[Union[str, List[str]]] = None,
    ctx: Optional[Context] = None,
) -> List[Dict[str, Any]]:
    """MCP wrapper for search_anime_mal."""
    return await _search_anime_mal_impl(query, limit, offset, fields, ctx)


async def _get_anime_mal_impl(
    mal_id: int,
    fields: Optional[Union[str, List[str]]] = None,
    ctx: Optional[Context] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get detailed anime information from MyAnimeList by MAL ID.

    MAL API v2 Reality:
    - Takes anime ID as path parameter
    - Uses fields parameter to control what data is returned
    - All data (statistics, related content, etc.) are field parameters

    Args:
        mal_id: MyAnimeList anime ID
        fields: List of field parameters to return (optional). Can be a comma-separated string or list of strings.

    Returns:
        Detailed anime information with MAL-specific data, or None if not found
    """
    await _validate_mal_client(ctx)

    if ctx:
        await ctx.info(f"Fetching MAL anime details for ID: {mal_id}")

    try:
        # Process fields parameter
        mal_params = {"fields": _process_fields_parameter(fields, MAL_DETAIL_FIELDS)}

        # Execute request
        raw_result = await mal_client.get_anime_by_id(
            mal_id, fields=mal_params["fields"]
        )

        if not raw_result:
            if ctx:
                await ctx.info(f"Anime with MAL ID {mal_id} not found")
            return None

        # Return raw MAL API response with source attribution
        result = {
            **raw_result,  # Include all raw MAL data
            "source_platform": "mal",
            "mal_id": mal_id,
            "fetched_at": datetime.now().isoformat(),
        }

        if ctx:
            await ctx.info(
                f"Retrieved detailed MAL data for '{result.get('title', 'Unknown')}'"
            )

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
    },
)
async def get_anime_by_id_mal_mcp(
    mal_id: int,
    fields: Optional[Union[str, List[str]]] = None,
    ctx: Optional[Context] = None,
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
    ctx: Optional[Context] = None,
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
        fields: List of field parameters to return (optional). Can be a comma-separated string or list of strings.

    Returns:
        List of seasonal anime with raw MAL data
    """
    await _validate_mal_client(ctx)

    if ctx:
        await ctx.info(
            f"Fetching MAL seasonal anime for {season.title()} {year} (limit: {limit}, offset: {offset})"
        )

    try:
        # Build MAL API parameters
        mal_params = {
            "year": year,
            "season": season,
            "sort": sort,
            "limit": min(limit, 500),
            "offset": offset,
        }

        # Process fields parameter
        mal_params["fields"] = _process_fields_parameter(fields, MAL_DEFAULT_FIELDS)

        # Execute request
        raw_results = await mal_client.get_seasonal_anime(
            year=year,
            season=season,
            sort=sort,
            limit=mal_params["limit"],
            offset=mal_params["offset"],
            fields=mal_params.get("fields"),
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
                    "fetched_at": datetime.now().isoformat(),
                }
                results.append(result)

        if ctx:
            await ctx.info(f"Found {len(results)} seasonal anime from MAL")

        return results

    except Exception as e:
        await _handle_mal_error(e, f"MAL seasonal search failed: {str(e)}", ctx)


# MCP tool wrapper
@mcp.tool(
    name="get_seasonal_anime_mal",
    description="Get seasonal anime from MyAnimeList with sorting, pagination, and field selection",
    annotations={
        "title": "MAL Seasonal Anime",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def get_seasonal_anime_mal_mcp(
    year: int,
    season: Literal["winter", "spring", "summer", "fall"],
    sort: Literal["anime_score", "anime_num_list_users"] = "anime_score",
    limit: int = 50,
    offset: int = 0,
    fields: Optional[Union[str, List[str]]] = None,
    ctx: Optional[Context] = None,
) -> List[Dict[str, Any]]:
    """MCP wrapper for get_seasonal_anime_mal."""
    return await _get_seasonal_anime_mal_impl(
        year, season, sort, limit, offset, fields, ctx
    )
