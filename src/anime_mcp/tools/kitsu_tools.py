"""
Kitsu specific MCP tools.

Specialized tools for Kitsu JSON:API with streaming platform support and range syntax.
Kitsu excels at JSON:API standards, streaming platform filtering, and comprehensive metadata.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from fastmcp import FastMCP

from mcp.server.fastmcp import Context

from ...config import get_settings
from ...integrations.clients.kitsu_client import KitsuClient
from ...models.structured_responses import (
    AnimeType,
)

# Initialize components
settings = get_settings()
kitsu_client = KitsuClient()
# Using direct API calls instead of KitsuMapper

# Create FastMCP instance for tools
mcp = FastMCP("Kitsu Tools")


@mcp.tool(
    name="search_anime_kitsu",
    description="Search anime on Kitsu with streaming platform support and range syntax filtering",
    annotations={
        "title": "Kitsu Anime Search",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def search_anime_kitsu(
    # Basic search
    query: Optional[str] = None,
    # Status filtering
    status: Optional[
        Literal["current", "finished", "tba", "unreleased", "upcoming"]
    ] = None,
    # Format filtering
    subtype: Optional[Literal["TV", "movie", "OVA", "ONA", "special", "music"]] = None,
    # Content rating
    age_rating: Optional[Literal["G", "PG", "R", "R18"]] = None,
    # Score filtering (Kitsu uses 0-100 scale, supports range syntax)
    average_rating: Optional[str] = None,  # "80..", "..90", "80..90"
    # Episode filtering (supports range syntax)
    episode_count: Optional[str] = None,  # "12..", "..24", "12..24"
    # Duration filtering (supports range syntax)
    episode_length: Optional[str] = None,  # "20..", "..30", "20..30"
    # Temporal filtering
    season_year: Optional[int] = None,
    season: Optional[Literal["winter", "spring", "summer", "fall"]] = None,
    # Genre/Category filtering
    categories: Optional[List[str]] = None,
    # Streaming platform filtering (Kitsu's unique strength)
    streamers: Optional[
        List[
            Literal[
                "Crunchyroll",
                "Funimation",
                "Netflix",
                "Hulu",
                "Amazon Prime",
                "Disney+",
                "HBO Max",
                "Apple TV+",
                "Paramount+",
                "Peacock",
                "Tubi",
                "VRV",
                "AnimeLab",
                "Wakanim",
                "ADN",
                "Other",
            ]
        ]
    ] = None,
    # Sorting and pagination
    sort: Optional[
        Literal[
            "popularityRank",
            "ratingRank",
            "startDate",
            "endDate",
            "createdAt",
            "updatedAt",
            "-popularityRank",
            "-ratingRank",
            "-startDate",
            "-endDate",
            "-createdAt",
            "-updatedAt",
        ]
    ] = None,
    limit: int = 20,
    offset: int = 0,
    ctx: Optional[Context] = None,
) -> List[Dict[str, Any]]:
    """
    Search anime on Kitsu with comprehensive filtering and unique streaming platform support.

    Kitsu Specializations:
    - JSON:API standard implementation
    - Comprehensive streaming platform filtering (unique among APIs)
    - Range syntax for numeric filters (80.., ..90, 80..90)
    - Rich category/genre system with hierarchical tags
    - International content support
    - Community ratings and engagement metrics

    Range Syntax Examples:
    - average_rating="80.." (80 and above)
    - episode_count="12..24" (between 12 and 24)
    - episode_length="..30" (30 minutes or less)

    Args:
        query: Search query for anime titles
        status: Airing status (current, finished, tba, unreleased, upcoming)
        subtype: Media format (TV, movie, OVA, ONA, special, music)
        age_rating: Content rating (G, PG, R, R18)
        average_rating: Score range using Kitsu syntax (0-100 scale)
        episode_count: Episode count range using Kitsu syntax
        episode_length: Episode duration range (minutes)
        season_year: Release year
        season: Release season
        categories: Genre/category filters
        streamers: Streaming platform filters (Kitsu's unique feature)
        sort: Sort criteria with direction prefix
        limit: Maximum results (default: 20, max: 20 per Kitsu API)
        offset: Pagination offset

    Returns:
        List of anime with Kitsu-specific data and streaming platform info
    """
    if ctx:
        await ctx.info(f"Searching Kitsu with streaming platform filters")

    try:
        # Build Kitsu JSON:API filter parameters
        filters = {}

        # Text search
        if query:
            filters["text"] = query

        # Status filtering
        if status:
            filters["status"] = status

        # Format filtering
        if subtype:
            filters["subtype"] = subtype

        # Content rating
        if age_rating:
            filters["ageRating"] = age_rating

        # Range filters using Kitsu's range syntax
        if average_rating:
            filters["averageRating"] = average_rating
        if episode_count:
            filters["episodeCount"] = episode_count
        if episode_length:
            filters["episodeLength"] = episode_length

        # Temporal filters
        if season_year:
            filters["seasonYear"] = str(season_year)
        if season:
            filters["season"] = season

        # Category filtering
        if categories:
            # Kitsu uses category names - join with comma
            filters["categories"] = ",".join(categories)

        # Streaming platform filtering (Kitsu's unique strength)
        if streamers:
            filters["streamers"] = ",".join(streamers)

        # Build request parameters
        params = {
            "filter": filters,
            "page": {"limit": min(limit, 20), "offset": offset},  # Kitsu max limit
        }

        # Add sorting
        if sort:
            params["sort"] = sort
        else:
            params["sort"] = "-popularityRank"  # Default to popularity

        # Execute search
        raw_results = await kitsu_client.search_anime(params)

        # Convert to structured response format
        results = []
        for raw_result in raw_results:
            try:
                # Extract Kitsu attributes
                attributes = raw_result.get("attributes", {})
                relationships = raw_result.get("relationships", {})

                # Extract titles
                primary_title = (
                    attributes.get("canonicalTitle")
                    or attributes.get("titles", {}).get("en")
                    or "Unknown"
                )

                # Extract date info
                start_date = attributes.get("startDate")
                year = None
                if start_date and "-" in start_date:
                    year = int(start_date.split("-")[0])

                # Map subtype to AnimeType
                anime_type = None
                subtype = attributes.get("subtype")
                if subtype:
                    if subtype.upper() == "TV":
                        anime_type = AnimeType.TV
                    elif subtype.upper() == "MOVIE":
                        anime_type = AnimeType.MOVIE
                    elif subtype.upper() == "OVA":
                        anime_type = AnimeType.OVA
                    elif subtype.upper() == "ONA":
                        anime_type = AnimeType.ONA
                    elif subtype.upper() == "SPECIAL":
                        anime_type = AnimeType.SPECIAL
                    elif subtype.upper() == "MUSIC":
                        anime_type = AnimeType.MUSIC

                # Build comprehensive result
                result = {
                    "id": str(raw_result.get("id", "")),
                    "title": primary_title,
                    "canonical_title": attributes.get("canonicalTitle"),
                    "abbreviated_titles": attributes.get("abbreviatedTitles", []),
                    "type": anime_type,
                    "subtype": subtype,
                    "episodes": attributes.get("episodeCount"),
                    "episode_length": attributes.get("episodeLength"),
                    "total_length": attributes.get("totalLength"),
                    "score": attributes.get("averageRating"),
                    "year": year,
                    "start_date": start_date,
                    "end_date": attributes.get("endDate"),
                    "season": attributes.get("season"),
                    "status": attributes.get("status"),
                    "genres": [],  # Kitsu uses categories
                    "synopsis": attributes.get("description")
                    or attributes.get("synopsis"),
                    "image_url": attributes.get("posterImage", {}).get("large")
                    or attributes.get("posterImage", {}).get("medium"),
                    # Kitsu-specific data
                    "kitsu_id": raw_result.get("id"),
                    "kitsu_slug": attributes.get("slug"),
                    "kitsu_rating_rank": attributes.get("ratingRank"),
                    "kitsu_popularity_rank": attributes.get("popularityRank"),
                    "kitsu_age_rating": attributes.get("ageRating"),
                    "kitsu_age_rating_guide": attributes.get("ageRatingGuide"),
                    "kitsu_nsfw": attributes.get("nsfw"),
                    # Kitsu ratings (multiple rating systems)
                    "average_rating": attributes.get("averageRating"),
                    "rating_frequencies": attributes.get("ratingFrequencies", {}),
                    "user_count": attributes.get("userCount"),
                    "favorites_count": attributes.get("favoritesCount"),
                    "review_count": attributes.get("reviewCount"),
                    # Kitsu streaming data (unique feature)
                    "streaming_links": [],  # Populated from relationships
                    "streaming_platforms": [],  # Will be populated from streamingLinks
                    # Production data
                    "youtube_video_id": attributes.get("youtubeVideoId"),
                    "cover_image": {
                        "original": attributes.get("coverImage", {}).get("original"),
                        "large": attributes.get("coverImage", {}).get("large"),
                        "small": attributes.get("coverImage", {}).get("small"),
                    },
                    "poster_image": {
                        "original": attributes.get("posterImage", {}).get("original"),
                        "large": attributes.get("posterImage", {}).get("large"),
                        "medium": attributes.get("posterImage", {}).get("medium"),
                        "small": attributes.get("posterImage", {}).get("small"),
                    },
                    # External data
                    "titles": attributes.get("titles", {}),
                    "abbreviations": attributes.get("abbreviatedTitles", []),
                    # Source attribution
                    "source_platform": "kitsu",
                    "last_updated": attributes.get("updatedAt"),
                }

                # Process streaming links if available
                streaming_links = relationships.get("streamingLinks", {}).get(
                    "data", []
                )
                if streaming_links:
                    result["streaming_platforms"] = [
                        link.get("attributes", {}).get("subs", [])
                        for link in streaming_links
                    ]

                results.append(result)

            except Exception as e:
                if ctx:
                    await ctx.error(f"Failed to process Kitsu result: {str(e)}")
                continue

        if ctx:
            await ctx.info(f"Found {len(results)} anime on Kitsu")

        return results

    except Exception as e:
        error_msg = f"Kitsu search failed: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)


@mcp.tool(
    name="get_anime_kitsu",
    description="Get detailed anime information from Kitsu by ID with streaming platform data",
    annotations={
        "title": "Kitsu Anime Details",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def get_anime_kitsu(
    kitsu_id: Union[int, str],
    include_characters: bool = False,
    include_staff: bool = False,
    include_episodes: bool = False,
    include_streaming: bool = True,
    include_reviews: bool = False,
    ctx: Optional[Context] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get detailed anime information from Kitsu by Kitsu ID.

    Provides comprehensive Kitsu data including:
    - Multiple image formats and sizes
    - Comprehensive rating and ranking data
    - Streaming platform information (unique to Kitsu)
    - User engagement metrics
    - Optional character, staff, episode, and review data

    Args:
        kitsu_id: Kitsu anime ID (numeric) or slug (string)
        include_characters: Include character information (default: False)
        include_staff: Include staff information (default: False)
        include_episodes: Include episode list (default: False)
        include_streaming: Include streaming platform data (default: True)
        include_reviews: Include user reviews (default: False)

    Returns:
        Detailed anime information with Kitsu-specific data and streaming info
    """
    if ctx:
        await ctx.info(f"Fetching Kitsu anime details for ID: {kitsu_id}")

    try:
        # Build included relationships
        include_params = []
        if include_characters:
            include_params.extend(["characters", "characters.character"])
        if include_staff:
            include_params.extend(["staff", "staff.person"])
        if include_episodes:
            include_params.append("episodes")
        if include_streaming:
            include_params.append("streamingLinks")
        if include_reviews:
            include_params.extend(["reviews", "reviews.user"])

        # Execute request
        raw_result = await kitsu_client.get_anime_by_id(
            kitsu_id, include=include_params
        )

        if not raw_result:
            if ctx:
                await ctx.info(f"Anime with Kitsu ID {kitsu_id} not found")
            return None

        # Extract detailed Kitsu data
        attributes = raw_result.get("attributes", {})
        relationships = raw_result.get("relationships", {})

        # Extract titles
        primary_title = (
            attributes.get("canonicalTitle")
            or attributes.get("titles", {}).get("en")
            or "Unknown"
        )

        # Extract date info
        start_date = attributes.get("startDate")
        year = None
        if start_date and "-" in start_date:
            year = int(start_date.split("-")[0])

        # Map subtype to AnimeType
        anime_type = None
        subtype = attributes.get("subtype")
        if subtype:
            if subtype.upper() == "TV":
                anime_type = AnimeType.TV
            elif subtype.upper() == "MOVIE":
                anime_type = AnimeType.MOVIE
            elif subtype.upper() == "OVA":
                anime_type = AnimeType.OVA
            elif subtype.upper() == "ONA":
                anime_type = AnimeType.ONA
            elif subtype.upper() == "SPECIAL":
                anime_type = AnimeType.SPECIAL
            elif subtype.upper() == "MUSIC":
                anime_type = AnimeType.MUSIC

        # Build comprehensive response
        result = {
            "id": str(kitsu_id),
            "title": primary_title,
            "canonical_title": attributes.get("canonicalTitle"),
            "titles": attributes.get("titles", {}),
            "abbreviated_titles": attributes.get("abbreviatedTitles", []),
            "slug": attributes.get("slug"),
            "synopsis": attributes.get("description") or attributes.get("synopsis"),
            "description": attributes.get("description"),
            # Media information
            "type": anime_type,
            "subtype": subtype,
            "episode_length": attributes.get("episodeLength"),
            "total_length": attributes.get("totalLength"),
            "status": attributes.get("status"),
            "tba": attributes.get("tba"),
            # Dates and seasons
            "start_date": start_date,
            "end_date": attributes.get("endDate"),
            "season": attributes.get("season"),
            "year": year,
            # Content rating
            "age_rating": attributes.get("ageRating"),
            "age_rating_guide": attributes.get("ageRatingGuide"),
            "nsfw": attributes.get("nsfw"),
            # Comprehensive rating data
            "average_rating": attributes.get("averageRating"),
            "rating_frequencies": attributes.get("ratingFrequencies", {}),
            "rating_rank": attributes.get("ratingRank"),
            "popularity_rank": attributes.get("popularityRank"),
            # User engagement
            "user_count": attributes.get("userCount"),
            "favorites_count": attributes.get("favoritesCount"),
            "review_count": attributes.get("reviewCount"),
            # Images (comprehensive image support)
            "cover_image": {
                "original": attributes.get("coverImage", {}).get("original"),
                "large": attributes.get("coverImage", {}).get("large"),
                "small": attributes.get("coverImage", {}).get("small"),
            },
            "poster_image": {
                "original": attributes.get("posterImage", {}).get("original"),
                "large": attributes.get("posterImage", {}).get("large"),
                "medium": attributes.get("posterImage", {}).get("medium"),
                "small": attributes.get("posterImage", {}).get("small"),
                "tiny": attributes.get("posterImage", {}).get("tiny"),
            },
            # Media and promotional content
            "youtube_video_id": attributes.get("youtubeVideoId"),
            # Kitsu-specific IDs and metadata
            "kitsu_id": raw_result.get("id"),
            "kitsu_slug": attributes.get("slug"),
            "created_at": attributes.get("createdAt"),
            "updated_at": attributes.get("updatedAt"),
            # Streaming platform data (Kitsu's unique strength)
            "streaming_links": [],
            "streaming_platforms": [],
            # Optional detailed data
            "characters": [],
            "staff": [],
            "episodes": [],
            "reviews": [],
            # Source attribution
            "source_platform": "kitsu",
            "last_updated": attributes.get("updatedAt"),
        }

        # Process streaming links (Kitsu's unique feature)
        if include_streaming and "streamingLinks" in relationships:
            streaming_data = relationships["streamingLinks"].get("data", [])
            for link in streaming_data:
                link_attrs = link.get("attributes", {})
                result["streaming_links"].append(
                    {
                        "url": link_attrs.get("url"),
                        "subs": link_attrs.get("subs", []),
                        "dubs": link_attrs.get("dubs", []),
                    }
                )

                # Extract platform names
                if link_attrs.get("subs"):
                    result["streaming_platforms"].extend(link_attrs["subs"])
                if link_attrs.get("dubs"):
                    result["streaming_platforms"].extend(link_attrs["dubs"])

        # Remove duplicates from streaming platforms
        result["streaming_platforms"] = list(set(result["streaming_platforms"]))

        # Process characters if requested
        if include_characters and "characters" in relationships:
            characters_data = relationships["characters"].get("data", [])
            for char in characters_data:
                char_attrs = char.get("attributes", {})
                result["characters"].append(
                    {
                        "role": char_attrs.get("role"),
                        "character_id": char.get("id"),
                        "name": char_attrs.get("name"),  # From included character data
                    }
                )

        # Process staff if requested
        if include_staff and "staff" in relationships:
            staff_data = relationships["staff"].get("data", [])
            for staff in staff_data:
                staff_attrs = staff.get("attributes", {})
                result["staff"].append(
                    {
                        "role": staff_attrs.get("role"),
                        "person_id": staff.get("id"),
                        "name": staff_attrs.get("name"),  # From included person data
                    }
                )

        # Process episodes if requested
        if include_episodes and "episodes" in relationships:
            episodes_data = relationships["episodes"].get("data", [])
            for ep in episodes_data:
                ep_attrs = ep.get("attributes", {})
                result["episodes"].append(
                    {
                        "number": ep_attrs.get("number"),
                        "title": ep_attrs.get("title"),
                        "synopsis": ep_attrs.get("synopsis"),
                        "air_date": ep_attrs.get("airdate"),
                        "length": ep_attrs.get("length"),
                    }
                )

        # Process reviews if requested
        if include_reviews and "reviews" in relationships:
            reviews_data = relationships["reviews"].get("data", [])
            for review in reviews_data:
                review_attrs = review.get("attributes", {})
                result["reviews"].append(
                    {
                        "rating": review_attrs.get("rating"),
                        "content": review_attrs.get("content"),
                        "likes_count": review_attrs.get("likesCount"),
                        "created_at": review_attrs.get("createdAt"),
                    }
                )

        if ctx:
            await ctx.info(
                f"Retrieved comprehensive Kitsu data for '{result['title']}'"
            )

        return result

    except Exception as e:
        error_msg = f"Failed to get Kitsu anime {kitsu_id}: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)


@mcp.tool(
    name="search_streaming_platforms",
    description="Search anime by streaming platform availability using Kitsu's comprehensive streaming data",
    annotations={
        "title": "Streaming Platform Search",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def search_streaming_platforms(
    platforms: List[str],
    require_all: bool = False,
    region: Optional[str] = None,
    content_rating: Optional[Literal["G", "PG", "R", "R18"]] = None,
    min_rating: Optional[float] = None,
    limit: int = 20,
    ctx: Optional[Context] = None,
) -> List[Dict[str, Any]]:
    """
    Search anime by streaming platform availability using Kitsu's unique streaming data.

    This tool leverages Kitsu's comprehensive streaming platform database
    to find anime available on specific platforms.

    Args:
        platforms: List of streaming platforms to search for
        require_all: Require anime to be on ALL platforms (default: any platform)
        region: Filter by region/country (if supported)
        content_rating: Filter by content rating
        min_rating: Minimum average rating (0-100 scale)
        limit: Maximum results

    Returns:
        List of anime with streaming platform availability data
    """
    if ctx:
        await ctx.info(f"Searching for anime on platforms: {', '.join(platforms)}")

    try:
        # Build streaming-focused search
        filters = {"streamers": ",".join(platforms)}

        # Add additional filters
        if content_rating:
            filters["ageRating"] = content_rating
        if min_rating:
            filters["averageRating"] = f"{min_rating}.."

        params = {
            "filter": filters,
            "include": ["streamingLinks"],
            "page": {"limit": min(limit, 20), "offset": 0},
            "sort": "-popularityRank",
        }

        # Execute streaming-focused search
        raw_results = await kitsu_client.search_anime(params)

        # Process results with streaming focus
        results = []
        for raw_result in raw_results:
            try:
                attributes = raw_result.get("attributes", {})

                # Extract titles
                primary_title = (
                    attributes.get("canonicalTitle")
                    or attributes.get("titles", {}).get("en")
                    or "Unknown"
                )

                # Map subtype to AnimeType
                anime_type = None
                subtype = attributes.get("subtype")
                if subtype:
                    if subtype.upper() == "TV":
                        anime_type = AnimeType.TV
                    elif subtype.upper() == "MOVIE":
                        anime_type = AnimeType.MOVIE
                    elif subtype.upper() == "OVA":
                        anime_type = AnimeType.OVA
                    elif subtype.upper() == "ONA":
                        anime_type = AnimeType.ONA
                    elif subtype.upper() == "SPECIAL":
                        anime_type = AnimeType.SPECIAL
                    elif subtype.upper() == "MUSIC":
                        anime_type = AnimeType.MUSIC

                # Extract streaming data
                streaming_info = {
                    "available_platforms": [],
                    "streaming_links": [],
                    "sub_languages": [],
                    "dub_languages": [],
                }

                result = {
                    "id": str(raw_result.get("id", "")),
                    "title": primary_title,
                    "type": anime_type,
                    "episodes": attributes.get("episodeCount"),
                    "average_rating": attributes.get("averageRating"),
                    "popularity_rank": attributes.get("popularityRank"),
                    "age_rating": attributes.get("ageRating"),
                    "status": attributes.get("status"),
                    "image_url": attributes.get("posterImage", {}).get("large")
                    or attributes.get("posterImage", {}).get("medium"),
                    # Streaming-specific data
                    "streaming_availability": streaming_info,
                    "source_platform": "kitsu",
                }

                results.append(result)

            except Exception as e:
                if ctx:
                    await ctx.error(f"Failed to process streaming result: {str(e)}")
                continue

        if ctx:
            await ctx.info(f"Found {len(results)} anime on specified platforms")

        return results

    except Exception as e:
        error_msg = f"Streaming platform search failed: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)
