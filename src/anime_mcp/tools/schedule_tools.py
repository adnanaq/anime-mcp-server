"""
AnimeSchedule specific MCP tools.

Specialized tools for AnimeSchedule API with comprehensive scheduling and streaming data.
AnimeSchedule excels at broadcast times, streaming platforms, and exclude filtering.
"""

from typing import Any, Dict, List, Literal, Optional

from fastmcp import FastMCP

from mcp.server.fastmcp import Context

from ...config import get_settings
from ...integrations.clients.animeschedule_client import AnimeScheduleClient
from ...models.structured_responses import (
    AnimeType,
    BasicAnimeResult,
)

# Removed UniversalAnime import - using structured responses now

# Initialize components
settings = get_settings()
animeschedule_client = AnimeScheduleClient()

# Create FastMCP instance for tools
mcp = FastMCP("AnimeSchedule Tools")


@mcp.tool(
    name="search_anime_schedule",
    description="Search anime with comprehensive scheduling and streaming platform filtering via AnimeSchedule",
    annotations={
        "title": "AnimeSchedule Anime Search",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def search_anime_schedule(
    # Basic search
    query: Optional[str] = None,
    # Scheduling filters
    streams: Optional[List[str]] = None,
    streams_exclude: Optional[List[str]] = None,
    # Media type filtering
    media_types: Optional[
        List[
            Literal[
                "TV",
                "Movie",
                "OVA",
                "ONA",
                "Special",
                "Music",
                "TV Short",
                "TV (Chinese)",
                "ONA (Chinese)",
            ]
        ]
    ] = None,
    media_types_exclude: Optional[List[str]] = None,
    # Status filtering
    airing_statuses: Optional[List[Literal["finished", "ongoing", "upcoming"]]] = None,
    airing_statuses_exclude: Optional[List[str]] = None,
    # Source filtering
    sources: Optional[
        List[
            Literal[
                "Manga",
                "Light Novel",
                "Web Manga",
                "Web Novel",
                "Novel",
                "Original",
                "Video Game",
                "Visual Novel",
                "4-koma Manga",
                "Book",
                "Music",
                "Game",
                "Other",
            ]
        ]
    ] = None,
    sources_exclude: Optional[List[str]] = None,
    # Studio filtering
    studios: Optional[List[str]] = None,
    studios_exclude: Optional[List[str]] = None,
    # Genre filtering
    genres: Optional[List[str]] = None,
    genres_exclude: Optional[List[str]] = None,
    # Temporal filtering
    years: Optional[List[int]] = None,
    years_exclude: Optional[List[int]] = None,
    seasons: Optional[List[Literal["Winter", "Spring", "Summer", "Fall"]]] = None,
    seasons_exclude: Optional[List[str]] = None,
    # External ID filtering
    mal_ids: Optional[List[int]] = None,
    anilist_ids: Optional[List[int]] = None,
    anidb_ids: Optional[List[int]] = None,
    # Episode and duration filtering
    episodes_min: Optional[int] = None,
    episodes_max: Optional[int] = None,
    duration_min: Optional[int] = None,
    duration_max: Optional[int] = None,
    # Sorting and pagination
    sort_by: Optional[
        Literal["popularity", "score", "alphabetic", "releaseDate"]
    ] = "popularity",
    limit: int = 20,
    offset: int = 0,
    ctx: Optional[Context] = None,
) -> List[Dict[str, Any]]:
    """
    Search anime using AnimeSchedule with comprehensive filtering and exclude options.

    AnimeSchedule Specializations:
    - Comprehensive streaming platform data (25+ platforms)
    - Broadcast scheduling and timing information
    - Advanced exclude filtering for all parameters
    - Multi-platform ID support (MAL, AniList, AniDB)
    - Detailed airing status tracking
    - Studio and production information

    Args:
        query: Search query for anime titles
        streams: Include specific streaming platforms (Crunchyroll, Netflix, etc.)
        streams_exclude: Exclude specific streaming platforms
        media_types: Include specific media types (TV, Movie, OVA, etc.)
        media_types_exclude: Exclude specific media types
        airing_statuses: Include specific airing statuses
        airing_statuses_exclude: Exclude specific airing statuses
        sources: Include specific source materials
        sources_exclude: Exclude specific source materials
        studios: Include specific animation studios
        studios_exclude: Exclude specific studios
        genres: Include specific genres
        genres_exclude: Exclude specific genres
        years/years_exclude: Year filtering with exclusion
        seasons/seasons_exclude: Season filtering with exclusion
        mal_ids/anilist_ids/anidb_ids: Filter by external platform IDs
        episodes_min/max: Episode count range filtering
        duration_min/max: Episode duration range (minutes)
        sort_by: Sort criteria (popularity, score, alphabetic, releaseDate)
        limit: Maximum results (default: 20, max: 100)
        offset: Pagination offset

    Returns:
        List of anime with comprehensive scheduling and streaming data
    """
    if ctx:
        await ctx.info(f"Searching AnimeSchedule with scheduling filters")

    try:
        # Build AnimeSchedule-specific parameters
        as_params = {
            "query": query,
            "limit": min(limit, 100),
            "offset": offset,
            "sort": sort_by,
        }

        if streams:
            as_params["streams"] = streams
        if streams_exclude:
            as_params["streams_exclude"] = streams_exclude
        if media_types:
            as_params["mt"] = media_types  # AnimeSchedule uses 'mt' parameter
        if media_types_exclude:
            as_params["media_types_exclude"] = media_types_exclude
        if airing_statuses:
            as_params["st"] = airing_statuses  # AnimeSchedule uses 'st' parameter
        if airing_statuses_exclude:
            as_params["airing_statuses_exclude"] = airing_statuses_exclude
        if sources:
            as_params["sources"] = sources
        if sources_exclude:
            as_params["sources_exclude"] = sources_exclude
        if studios:
            as_params["studios"] = studios
        if studios_exclude:
            as_params["studios_exclude"] = studios_exclude
        if genres:
            as_params["genres"] = genres
        if genres_exclude:
            as_params["genres_exclude"] = genres_exclude
        if years:
            as_params["years"] = years
        if years_exclude:
            as_params["years_exclude"] = years_exclude
        if seasons:
            as_params["seasons"] = seasons
        if seasons_exclude:
            as_params["seasons_exclude"] = seasons_exclude
        if mal_ids:
            as_params["mal_ids"] = mal_ids
        if anilist_ids:
            as_params["anilist_ids"] = anilist_ids
        if anidb_ids:
            as_params["anidb_ids"] = anidb_ids
        if episodes_min is not None:
            as_params["episodes_min"] = episodes_min
        if episodes_max is not None:
            as_params["episodes_max"] = episodes_max
        if duration_min is not None:
            as_params["duration_min"] = duration_min
        if duration_max is not None:
            as_params["duration_max"] = duration_max

        # Execute search
        raw_results = await animeschedule_client.search_anime(as_params)

        # Convert to standardized format
        results = []
        for raw_result in raw_results:
            try:
                # Convert to BasicAnimeResult for consistent data handling
                basic_anime = BasicAnimeResult(
                    id=str(raw_result.get("id")),
                    title=raw_result.get("title", ""),
                    type=(
                        AnimeType(raw_result.get("media_type", "tv"))
                        if raw_result.get("media_type")
                        else None
                    ),
                    score=raw_result.get("score"),
                    year=raw_result.get("year"),
                    genres=raw_result.get("genres", []),
                    synopsis=raw_result.get("synopsis"),
                    image_url=raw_result.get("image_url"),
                )

                # Build comprehensive result with scheduling data
                result = {
                    "id": basic_anime.id,
                    "title": basic_anime.title,
                    "type": basic_anime.type.value if basic_anime.type else None,
                    "score": basic_anime.score,
                    "year": basic_anime.year,
                    "season": raw_result.get("season"),
                    "genres": basic_anime.genres or [],
                    "synopsis": basic_anime.synopsis,
                    "image_url": basic_anime.image_url,
                    # AnimeSchedule-specific scheduling data
                    "animeschedule_id": raw_result.get("id"),
                    "broadcast_time": raw_result.get("broadcast_time"),
                    "broadcast_day": raw_result.get("broadcast_day"),
                    "broadcast_timezone": raw_result.get("broadcast_timezone"),
                    "next_episode_date": raw_result.get("next_episode_date"),
                    "next_episode_number": raw_result.get("next_episode_number"),
                    "episodes_aired": raw_result.get("episodes_aired"),
                    "episode_duration": raw_result.get("episode_duration"),
                    # Comprehensive streaming platform data
                    "streaming_platforms": raw_result.get("streams", []),
                    "streaming_regions": raw_result.get("stream_regions", {}),
                    "streaming_urls": raw_result.get("stream_urls", {}),
                    # Production information
                    "source_material": raw_result.get("source"),
                    "production_status": raw_result.get("production_status"),
                    "premiere_date": raw_result.get("premiere_date"),
                    "finale_date": raw_result.get("finale_date"),
                    # External platform IDs
                    "mal_id": raw_result.get("mal_id"),
                    "anilist_id": raw_result.get("anilist_id"),
                    "anidb_id": raw_result.get("anidb_id"),
                    "kitsu_id": raw_result.get("kitsu_id"),
                    # Additional metadata
                    "popularity_rank": raw_result.get("popularity_rank"),
                    "score_rank": raw_result.get("score_rank"),
                    "licensors": raw_result.get("licensors", []),
                    "producers": raw_result.get("producers", []),
                    # Source attribution
                    "source_platform": "animeschedule",
                    "data_quality_score": raw_result.get("data_quality_score", 0.8),
                    "last_updated": raw_result.get("updated_at"),
                }

                results.append(result)

            except Exception as e:
                if ctx:
                    await ctx.error(f"Failed to process AnimeSchedule result: {str(e)}")
                continue

        if ctx:
            await ctx.info(f"Found {len(results)} anime with scheduling data")

        return results

    except Exception as e:
        error_msg = f"AnimeSchedule search failed: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)


@mcp.tool(
    name="get_schedule_data",
    description="Get detailed broadcast schedule and streaming information for anime by external ID",
    annotations={
        "title": "Anime Schedule Details",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def get_schedule_data(
    mal_id: Optional[int] = None,
    anilist_id: Optional[int] = None,
    anidb_id: Optional[int] = None,
    animeschedule_id: Optional[int] = None,
    include_episode_history: bool = False,
    include_streaming_details: bool = True,
    ctx: Optional[Context] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get detailed broadcast schedule and streaming information using external platform IDs.

    This tool specializes in enriching anime data with scheduling information
    from AnimeSchedule using IDs from other platforms (MAL, AniList, AniDB).

    Args:
        mal_id: MyAnimeList ID for lookup
        anilist_id: AniList ID for lookup
        anidb_id: AniDB ID for lookup
        animeschedule_id: Direct AnimeSchedule ID
        include_episode_history: Include episode release history (default: False)
        include_streaming_details: Include detailed streaming platform info (default: True)

    Returns:
        Comprehensive scheduling and streaming data, or None if not found
    """
    if not any([mal_id, anilist_id, anidb_id, animeschedule_id]):
        if ctx:
            await ctx.error("At least one ID parameter is required")
        raise ValueError("At least one ID parameter is required")

    if ctx:
        await ctx.info(f"Fetching schedule data using provided IDs")

    try:
        # Build lookup parameters
        lookup_params = {}
        if mal_id:
            lookup_params["mal_id"] = mal_id
        if anilist_id:
            lookup_params["anilist_id"] = anilist_id
        if anidb_id:
            lookup_params["anidb_id"] = anidb_id
        if animeschedule_id:
            lookup_params["id"] = animeschedule_id

        # Add optional data flags
        if include_episode_history:
            lookup_params["include_episodes"] = True
        if include_streaming_details:
            lookup_params["include_streaming"] = True

        # Execute lookup
        raw_result = await animeschedule_client.get_anime_by_id(lookup_params)

        if not raw_result:
            if ctx:
                await ctx.info("No scheduling data found for provided IDs")
            return None

        # Convert base data
        basic_anime = BasicAnimeResult(
            id=str(raw_result.get("id")),
            title=raw_result.get("title", ""),
            type=(
                AnimeType(raw_result.get("media_type", "tv"))
                if raw_result.get("media_type")
                else None
            ),
            score=raw_result.get("score"),
            year=raw_result.get("year"),
            genres=raw_result.get("genres", []),
            synopsis=raw_result.get("synopsis"),
            image_url=raw_result.get("image_url"),
        )

        # Build comprehensive scheduling response
        result = {
            "id": basic_anime.id,
            "title": basic_anime.title,
            "type": basic_anime.type.value if basic_anime.type else None,
            "score": basic_anime.score,
            "year": basic_anime.year,
            # Core scheduling data
            "broadcast_schedule": {
                "day": raw_result.get("broadcast_day"),
                "time": raw_result.get("broadcast_time"),
                "timezone": raw_result.get("broadcast_timezone", "JST"),
                "next_episode": {
                    "number": raw_result.get("next_episode_number"),
                    "date": raw_result.get("next_episode_date"),
                    "countdown_seconds": raw_result.get("episode_countdown"),
                },
            },
            # Episode information
            "episode_info": {
                "total_episodes": raw_result.get("episodes", raw_result.get("episode_count")),
                "episodes_aired": raw_result.get("episodes_aired"),
                "episode_duration": raw_result.get("episode_duration"),
                "average_duration": raw_result.get("average_episode_duration"),
            },
            # Airing period
            "airing_period": {
                "start_date": raw_result.get("start_date", raw_result.get("aired_from")),
                "end_date": raw_result.get("end_date", raw_result.get("aired_to")),
                "premiere_date": raw_result.get("premiere_date"),
                "finale_date": raw_result.get("finale_date"),
                "season": raw_result.get("season"),
                "year": raw_result.get("year"),
            },
            # Comprehensive streaming data
            "streaming_platforms": {
                "available_on": raw_result.get("streams", []),
                "regional_availability": raw_result.get("stream_regions", {}),
                "streaming_urls": (
                    raw_result.get("stream_urls", {})
                    if include_streaming_details
                    else {}
                ),
                "subscription_required": raw_result.get("subscription_platforms", []),
                "free_platforms": raw_result.get("free_platforms", []),
            },
            # Production and licensing
            "production": {
                "studios": raw_result.get("studios", []),
                "producers": raw_result.get("producers", []),
                "licensors": raw_result.get("licensors", []),
                "source_material": raw_result.get("source"),
                "production_status": raw_result.get("production_status"),
            },
            # External platform IDs
            "external_ids": {
                "animeschedule_id": raw_result.get("id"),
                "mal_id": raw_result.get("mal_id"),
                "anilist_id": raw_result.get("anilist_id"),
                "anidb_id": raw_result.get("anidb_id"),
                "kitsu_id": raw_result.get("kitsu_id"),
                "tmdb_id": raw_result.get("tmdb_id"),
            },
            # Episode history (if requested)
            "episode_history": (
                raw_result.get("episodes", []) if include_episode_history else []
            ),
            # Popularity and ranking
            "rankings": {
                "popularity_rank": raw_result.get("popularity_rank"),
                "score_rank": raw_result.get("score_rank"),
                "trending_rank": raw_result.get("trending_rank"),
            },
            # Metadata
            "source_platform": "animeschedule",
            "data_quality_score": raw_result.get("data_quality_score", 0.8),
            "last_updated": raw_result.get("updated_at"),
            "data_freshness": raw_result.get("data_freshness", "unknown"),
        }

        if ctx:
            await ctx.info(
                f"Retrieved comprehensive schedule data for '{result['title']}'"
            )

        return result

    except Exception as e:
        error_msg = f"Failed to get schedule data: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)


@mcp.tool(
    name="get_currently_airing",
    description="Get currently airing anime with real-time broadcast information",
    annotations={
        "title": "Currently Airing Anime",
        "readOnlyHint": True,
        "idempotentHint": False,  # Time-sensitive data
    },
)
async def get_currently_airing(
    day_filter: Optional[
        Literal[
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
        ]
    ] = None,
    timezone: str = "JST",
    include_upcoming_episodes: bool = True,
    streaming_platforms: Optional[List[str]] = None,
    limit: int = 50,
    ctx: Optional[Context] = None,
) -> List[Dict[str, Any]]:
    """
    Get currently airing anime with real-time broadcast schedules.

    Specializes in time-sensitive scheduling data with broadcast times,
    upcoming episodes, and streaming platform availability.

    Args:
        day_filter: Filter by specific broadcast day
        timezone: Timezone for broadcast times (default: JST)
        include_upcoming_episodes: Include next episode information
        streaming_platforms: Filter by streaming platforms
        limit: Maximum results (default: 50, max: 100)

    Returns:
        List of currently airing anime with broadcast schedules
    """
    if ctx:
        await ctx.info(f"Fetching currently airing anime")

    try:
        # Build parameters for currently airing anime
        params = {
            "airing_status": "ongoing",
            "limit": min(limit, 100),
            "sort": "next_episode_date",
            "timezone": timezone,
        }

        # Add filters
        if day_filter:
            params["broadcast_day"] = day_filter
        if streaming_platforms:
            params["streams"] = streaming_platforms
        if include_upcoming_episodes:
            params["include_next_episode"] = True

        # Execute request
        raw_results = await animeschedule_client.get_currently_airing(params)

        # Process results
        results = []
        for raw_result in raw_results:
            try:
                basic_anime = BasicAnimeResult(
                    id=str(raw_result.get("id")),
                    title=raw_result.get("title", ""),
                    type=(
                        AnimeType(raw_result.get("media_type", "tv"))
                        if raw_result.get("media_type")
                        else None
                    ),
                    score=raw_result.get("score"),
                    year=raw_result.get("year"),
                    genres=raw_result.get("genres", []),
                    synopsis=raw_result.get("synopsis"),
                    image_url=raw_result.get("image_url"),
                )

                result = {
                    "id": basic_anime.id,
                    "title": basic_anime.title,
                    "type": basic_anime.type.value if basic_anime.type else None,
                    "year": basic_anime.year,
                    "episodes_aired": raw_result.get("episodes_aired"),
                    "image_url": basic_anime.image_url,
                    # Real-time broadcast data
                    "broadcast_info": {
                        "day": raw_result.get("broadcast_day"),
                        "time": raw_result.get("broadcast_time"),
                        "timezone": timezone,
                        "next_episode": {
                            "number": raw_result.get("next_episode_number"),
                            "date": raw_result.get("next_episode_date"),
                            "countdown_hours": raw_result.get(
                                "hours_until_next_episode"
                            ),
                        },
                    },
                    # Streaming availability
                    "streaming_info": {
                        "platforms": raw_result.get("streams", []),
                        "regions": raw_result.get("stream_regions", {}),
                        "premium_required": raw_result.get("premium_required", False),
                    },
                    # Additional data
                    "score": raw_result.get("score"),
                    "popularity_rank": raw_result.get("popularity_rank"),
                    "genres": raw_result.get("genres", []),
                    "studios": raw_result.get("studios", []),
                    # External IDs for cross-platform lookup
                    "mal_id": raw_result.get("mal_id"),
                    "anilist_id": raw_result.get("anilist_id"),
                    "source_platform": "animeschedule",
                    "last_updated": raw_result.get("updated_at"),
                }

                results.append(result)

            except Exception as e:
                if ctx:
                    await ctx.error(f"Failed to process airing anime: {str(e)}")
                continue

        if ctx:
            await ctx.info(f"Found {len(results)} currently airing anime")

        return results

    except Exception as e:
        error_msg = f"Failed to get currently airing anime: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)
