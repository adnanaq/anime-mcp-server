"""
AniList specific MCP tools.

Comprehensive tools for AniList GraphQL API with 70+ parameters.
AniList specializes in international content, advanced filtering, and modern features.
"""

from typing import Any, Dict, List, Literal, Optional

from fastmcp import FastMCP

from mcp.server.fastmcp import Context

from ...config import get_settings
from ...integrations.clients.anilist_client import AniListClient
from ...models.structured_responses import (
    AnimeType,
)

# Initialize components
settings = get_settings()
anilist_client = AniListClient()
# Using direct API calls instead of AniListMapper

# Create FastMCP instance for tools
mcp = FastMCP("AniList Tools")


@mcp.tool(
    name="search_anime_anilist",
    description="Search anime on AniList with comprehensive GraphQL filtering (70+ parameters)",
    annotations={
        "title": "AniList Anime Search",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def search_anime_anilist(
    # Basic search
    query: Optional[str] = None,
    # Content filtering
    is_adult: Optional[bool] = None,
    country_of_origin: Optional[Literal["JP", "KR", "CN", "TW", "IN", "TH"]] = None,
    # Episode filtering
    episodes: Optional[int] = None,
    episodes_greater: Optional[int] = None,
    episodes_lesser: Optional[int] = None,
    # Duration filtering (minutes)
    duration: Optional[int] = None,
    duration_greater: Optional[int] = None,
    duration_lesser: Optional[int] = None,
    # Score filtering
    average_score: Optional[int] = None,
    average_score_greater: Optional[int] = None,
    average_score_lesser: Optional[int] = None,
    # Popularity filtering
    popularity: Optional[int] = None,
    popularity_greater: Optional[int] = None,
    popularity_lesser: Optional[int] = None,
    # Format filtering
    format: Optional[
        Literal["TV", "TV_SHORT", "MOVIE", "SPECIAL", "OVA", "ONA", "MUSIC"]
    ] = None,
    format_in: Optional[List[str]] = None,
    format_not_in: Optional[List[str]] = None,
    # Status filtering
    status: Optional[
        Literal["FINISHED", "RELEASING", "NOT_YET_RELEASED", "CANCELLED", "HIATUS"]
    ] = None,
    status_in: Optional[List[str]] = None,
    status_not_in: Optional[List[str]] = None,
    # Date filtering
    start_date_greater: Optional[str] = None,
    start_date_lesser: Optional[str] = None,
    end_date_greater: Optional[str] = None,
    end_date_lesser: Optional[str] = None,
    # Season filtering
    season: Optional[Literal["WINTER", "SPRING", "SUMMER", "FALL"]] = None,
    season_year: Optional[int] = None,
    # Source filtering
    source: Optional[
        Literal[
            "ORIGINAL",
            "MANGA",
            "LIGHT_NOVEL",
            "VISUAL_NOVEL",
            "VIDEO_GAME",
            "OTHER",
            "NOVEL",
            "DOUJINSHI",
            "ANIME",
            "WEB_NOVEL",
            "LIVE_ACTION",
            "GAME",
            "COMIC",
            "MULTIMEDIA_PROJECT",
            "PICTURE_BOOK",
        ]
    ] = None,
    source_in: Optional[List[str]] = None,
    # Genre filtering
    genre_in: Optional[List[str]] = None,
    genre_not_in: Optional[List[str]] = None,
    # Tag filtering
    tag_in: Optional[List[str]] = None,
    tag_not_in: Optional[List[str]] = None,
    tag_category_in: Optional[List[str]] = None,
    tag_category_not_in: Optional[List[str]] = None,
    minimum_tag_rank: Optional[int] = None,
    # Studio filtering
    studio_in: Optional[List[str]] = None,
    # Licensing
    is_licensed: Optional[bool] = None,
    licensed_by_in: Optional[List[str]] = None,
    # ID filtering
    id_in: Optional[List[int]] = None,
    id_not_in: Optional[List[int]] = None,
    id_mal_in: Optional[List[int]] = None,
    id_mal_not_in: Optional[List[int]] = None,
    # Sorting and pagination
    sort: Optional[
        List[
            Literal[
                "ID",
                "ID_DESC",
                "TITLE_ROMAJI",
                "TITLE_ROMAJI_DESC",
                "TITLE_ENGLISH",
                "TITLE_ENGLISH_DESC",
                "TITLE_NATIVE",
                "TITLE_NATIVE_DESC",
                "TYPE",
                "TYPE_DESC",
                "FORMAT",
                "FORMAT_DESC",
                "START_DATE",
                "START_DATE_DESC",
                "END_DATE",
                "END_DATE_DESC",
                "SCORE",
                "SCORE_DESC",
                "POPULARITY",
                "POPULARITY_DESC",
                "TRENDING",
                "TRENDING_DESC",
                "EPISODES",
                "EPISODES_DESC",
                "DURATION",
                "DURATION_DESC",
                "STATUS",
                "STATUS_DESC",
                "CHAPTERS",
                "CHAPTERS_DESC",
                "VOLUMES",
                "VOLUMES_DESC",
                "UPDATED_AT",
                "UPDATED_AT_DESC",
                "SEARCH_MATCH",
                "FAVOURITES",
                "FAVOURITES_DESC",
            ]
        ]
    ] = None,
    # Pagination
    page: int = 1,
    per_page: int = 20,
    ctx: Optional[Context] = None,
) -> List[Dict[str, Any]]:
    """
    Search anime on AniList with comprehensive GraphQL filtering capabilities.

    AniList Specializations:
    - International content with country filtering
    - Comprehensive adult content filtering
    - Advanced range-based filtering (episodes, duration, scores)
    - Extensive genre and tag system with categories
    - Studio and licensing information
    - Modern GraphQL API with flexible sorting

    This tool supports 70+ parameters for precise anime discovery.

    Args:
        query: Search query for anime titles
        is_adult: Filter adult content (true/false/null for all)
        country_of_origin: Filter by country (JP, KR, CN, TW, IN, TH)
        episodes_greater/lesser: Episode count range filtering
        duration_greater/lesser: Episode duration range (minutes)
        average_score_greater/lesser: Score range filtering (0-100)
        popularity_greater/lesser: Popularity range filtering
        format/format_in/format_not_in: Format filtering (TV, Movie, OVA, etc.)
        status/status_in/status_not_in: Airing status filtering
        start_date_greater/lesser: Date range filtering (YYYY-MM-DD)
        season/season_year: Seasonal filtering
        source/source_in: Source material filtering
        genre_in/genre_not_in: Genre filtering
        tag_in/tag_not_in: Tag filtering with category support
        sort: Sorting options (multiple criteria supported)
        page/per_page: Pagination (max 50 per page)

    Returns:
        List of anime with AniList-specific data and comprehensive metadata
    """
    if ctx:
        await ctx.info(f"Searching AniList with comprehensive filtering")

    try:
        # Build AniList GraphQL parameters directly
        anilist_params = {}

        # Basic search parameters
        if query:
            anilist_params["search"] = query

        # Pagination
        anilist_params["page"] = page
        anilist_params["perPage"] = min(per_page, 50)

        # Map all AniList-specific parameters directly
        if is_adult is not None:
            anilist_params["isAdult"] = is_adult
        if country_of_origin:
            anilist_params["countryOfOrigin"] = country_of_origin
        if episodes is not None:
            anilist_params["episodes"] = episodes
        if episodes_greater is not None:
            anilist_params["episodes_greater"] = episodes_greater
        if episodes_lesser is not None:
            anilist_params["episodes_lesser"] = episodes_lesser
        if duration is not None:
            anilist_params["duration"] = duration
        if duration_greater is not None:
            anilist_params["duration_greater"] = duration_greater
        if duration_lesser is not None:
            anilist_params["duration_lesser"] = duration_lesser
        if average_score is not None:
            anilist_params["averageScore"] = average_score
        if average_score_greater is not None:
            anilist_params["averageScore_greater"] = average_score_greater
        if average_score_lesser is not None:
            anilist_params["averageScore_lesser"] = average_score_lesser
        if popularity is not None:
            anilist_params["popularity"] = popularity
        if popularity_greater is not None:
            anilist_params["popularity_greater"] = popularity_greater
        if popularity_lesser is not None:
            anilist_params["popularity_lesser"] = popularity_lesser
        if format:
            anilist_params["format"] = format
        if format_in:
            anilist_params["format_in"] = format_in
        if format_not_in:
            anilist_params["format_not_in"] = format_not_in
        if status:
            anilist_params["status"] = status
        if status_in:
            anilist_params["status_in"] = status_in
        if status_not_in:
            anilist_params["status_not_in"] = status_not_in
        if start_date_greater:
            anilist_params["startDate_greater"] = start_date_greater
        if start_date_lesser:
            anilist_params["startDate_lesser"] = start_date_lesser
        if end_date_greater:
            anilist_params["endDate_greater"] = end_date_greater
        if end_date_lesser:
            anilist_params["endDate_lesser"] = end_date_lesser
        if season:
            anilist_params["season"] = season
        if season_year:
            anilist_params["seasonYear"] = season_year
        if source:
            anilist_params["source"] = source
        if source_in:
            anilist_params["source_in"] = source_in
        if genre_in:
            anilist_params["genre_in"] = genre_in
        if genre_not_in:
            anilist_params["genre_not_in"] = genre_not_in
        if tag_in:
            anilist_params["tag_in"] = tag_in
        if tag_not_in:
            anilist_params["tag_not_in"] = tag_not_in
        if tag_category_in:
            anilist_params["tagCategory_in"] = tag_category_in
        if tag_category_not_in:
            anilist_params["tagCategory_not_in"] = tag_category_not_in
        if minimum_tag_rank is not None:
            anilist_params["minimumTagRank"] = minimum_tag_rank
        if studio_in:
            anilist_params["studios_in"] = studio_in
        if is_licensed is not None:
            anilist_params["isLicensed"] = is_licensed
        if licensed_by_in:
            anilist_params["licensedBy_in"] = licensed_by_in
        if id_in:
            anilist_params["id_in"] = id_in
        if id_not_in:
            anilist_params["id_not_in"] = id_not_in
        if id_mal_in:
            anilist_params["idMal_in"] = id_mal_in
        if id_mal_not_in:
            anilist_params["idMal_not_in"] = id_mal_not_in
        if sort:
            anilist_params["sort"] = sort

        # Execute GraphQL query
        raw_results = await anilist_client.search_anime(anilist_params)

        # Convert to structured response format
        results = []
        for raw_result in raw_results:
            try:
                # Extract titles
                title_data = raw_result.get("title", {})
                primary_title = (
                    title_data.get("romaji")
                    or title_data.get("english")
                    or title_data.get("native")
                    or "Unknown"
                )

                # Extract date info
                start_date = raw_result.get("startDate", {})
                year = None
                if start_date and start_date.get("year"):
                    year = start_date.get("year")

                # Map format to AnimeType
                anime_type = None
                format_val = raw_result.get("format")
                if format_val:
                    if format_val == "TV":
                        anime_type = AnimeType.TV
                    elif format_val == "MOVIE":
                        anime_type = AnimeType.MOVIE
                    elif format_val == "OVA":
                        anime_type = AnimeType.OVA
                    elif format_val == "ONA":
                        anime_type = AnimeType.ONA
                    elif format_val == "SPECIAL":
                        anime_type = AnimeType.SPECIAL
                    elif format_val == "MUSIC":
                        anime_type = AnimeType.MUSIC

                # Build comprehensive result
                result = {
                    "id": str(raw_result.get("id", "")),
                    "title": primary_title,
                    "title_romaji": title_data.get("romaji"),
                    "title_english": title_data.get("english"),
                    "title_native": title_data.get("native"),
                    "type": anime_type,
                    "format": format_val,
                    "episodes": raw_result.get("episodes"),
                    "duration": raw_result.get("duration"),
                    "score": raw_result.get("averageScore"),
                    "year": year,
                    "start_date": (
                        f"{start_date.get('year', '')}-{start_date.get('month', ''):02d}-{start_date.get('day', ''):02d}"
                        if start_date.get("year")
                        else None
                    ),
                    "season": raw_result.get("season"),
                    "season_year": raw_result.get("seasonYear"),
                    "status": raw_result.get("status"),
                    "genres": raw_result.get("genres", []),
                    "tags": [tag.get("name") for tag in raw_result.get("tags", [])],
                    "studios": [
                        studio.get("name")
                        for studio in raw_result.get("studios", {}).get("nodes", [])
                    ],
                    "synopsis": raw_result.get("description"),
                    "image_url": raw_result.get("coverImage", {}).get("large"),
                    # AniList-specific data
                    "anilist_id": raw_result.get("id"),
                    "anilist_score": raw_result.get("averageScore"),
                    "anilist_popularity": raw_result.get("popularity"),
                    "anilist_trending": raw_result.get("trending"),
                    "anilist_favourites": raw_result.get("favourites"),
                    "anilist_is_adult": raw_result.get("isAdult"),
                    "anilist_country_of_origin": raw_result.get("countryOfOrigin"),
                    "anilist_source": raw_result.get("source"),
                    "anilist_hashtag": raw_result.get("hashtag"),
                    "anilist_is_licensed": raw_result.get("isLicensed"),
                    "anilist_updated_at": raw_result.get("updatedAt"),
                    # External IDs
                    "mal_id": raw_result.get("idMal"),
                    "external_links": raw_result.get("externalLinks", []),
                    # Source attribution
                    "source_platform": "anilist",
                }

                results.append(result)

            except Exception as e:
                if ctx:
                    await ctx.error(f"Failed to process AniList result: {str(e)}")
                continue

        if ctx:
            await ctx.info(f"Found {len(results)} anime on AniList")

        return results

    except Exception as e:
        error_msg = f"AniList search failed: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)


@mcp.tool(
    name="get_anime_anilist",
    description="Get detailed anime information from AniList by ID with comprehensive data",
    annotations={
        "title": "AniList Anime Details",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def get_anime_anilist(
    anilist_id: int,
    include_relations: bool = True,
    include_characters: bool = False,
    include_staff: bool = False,
    include_reviews: bool = False,
    ctx: Optional[Context] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get detailed anime information from AniList by AniList ID.

    Provides comprehensive AniList data including:
    - Multiple title formats (romaji, english, native)
    - Detailed tag system with categories and rankings
    - International data (country, licensing info)
    - Comprehensive external links and IDs
    - Optional relations, characters, staff, and reviews

    Args:
        anilist_id: AniList anime ID
        include_relations: Include related anime/manga (default: True)
        include_characters: Include character information (default: False)
        include_staff: Include staff information (default: False)
        include_reviews: Include user reviews (default: False)

    Returns:
        Detailed anime information with comprehensive AniList data
    """
    if ctx:
        await ctx.info(f"Fetching AniList anime details for ID: {anilist_id}")

    try:
        # Build GraphQL query with optional fields
        fields = [
            "id",
            "idMal",
            "title",
            "format",
            "status",
            "description",
            "startDate",
            "endDate",
            "season",
            "seasonYear",
            "episodes",
            "duration",
            "source",
            "hashtag",
            "updatedAt",
            "coverImage",
            "bannerImage",
            "genres",
            "synonyms",
            "averageScore",
            "meanScore",
            "popularity",
            "trending",
            "favourites",
            "tags",
            "isAdult",
            "countryOfOrigin",
            "isLicensed",
            "studios",
            "externalLinks",
        ]

        if include_relations:
            fields.append("relations")
        if include_characters:
            fields.append("characters")
        if include_staff:
            fields.append("staff")
        if include_reviews:
            fields.append("reviews")

        # Execute request
        raw_result = await anilist_client.get_anime_by_id(anilist_id, fields=fields)

        if not raw_result:
            if ctx:
                await ctx.info(f"Anime with AniList ID {anilist_id} not found")
            return None

        # Extract titles
        title_data = raw_result.get("title", {})
        primary_title = (
            title_data.get("romaji")
            or title_data.get("english")
            or title_data.get("native")
            or "Unknown"
        )

        # Extract date info
        start_date = raw_result.get("startDate", {})
        end_date = raw_result.get("endDate", {})
        year = None
        if start_date and start_date.get("year"):
            year = start_date.get("year")

        # Map format to AnimeType
        anime_type = None
        format_val = raw_result.get("format")
        if format_val:
            if format_val == "TV":
                anime_type = AnimeType.TV
            elif format_val == "MOVIE":
                anime_type = AnimeType.MOVIE
            elif format_val == "OVA":
                anime_type = AnimeType.OVA
            elif format_val == "ONA":
                anime_type = AnimeType.ONA
            elif format_val == "SPECIAL":
                anime_type = AnimeType.SPECIAL
            elif format_val == "MUSIC":
                anime_type = AnimeType.MUSIC

        # Build comprehensive response
        result = {
            "id": str(anilist_id),
            "title": primary_title,
            "title_romaji": title_data.get("romaji"),
            "title_english": title_data.get("english"),
            "title_native": title_data.get("native"),
            "synonyms": raw_result.get("synonyms", []),
            "type": anime_type,
            "format": format_val,
            "episodes": raw_result.get("episodes"),
            "duration": raw_result.get("duration"),
            "score": raw_result.get("averageScore"),
            "mean_score": raw_result.get("meanScore"),
            "year": year,
            "start_date": (
                f"{start_date.get('year', '')}-{start_date.get('month', ''):02d}-{start_date.get('day', ''):02d}"
                if start_date.get("year")
                else None
            ),
            "end_date": (
                f"{end_date.get('year', '')}-{end_date.get('month', ''):02d}-{end_date.get('day', ''):02d}"
                if end_date.get("year")
                else None
            ),
            "season": raw_result.get("season"),
            "season_year": raw_result.get("seasonYear"),
            "status": raw_result.get("status"),
            "genres": raw_result.get("genres", []),
            "studios": [
                studio.get("name")
                for studio in raw_result.get("studios", {}).get("nodes", [])
            ],
            "synopsis": raw_result.get("description"),
            "image_url": raw_result.get("coverImage", {}).get("large"),
            "banner_image": raw_result.get("bannerImage"),
            # Comprehensive tag data
            "tags": [
                {
                    "name": tag.get("name"),
                    "description": tag.get("description"),
                    "category": tag.get("category"),
                    "rank": tag.get("rank"),
                    "is_general_spoiler": tag.get("isGeneralSpoiler"),
                    "is_media_spoiler": tag.get("isMediaSpoiler"),
                    "is_adult": tag.get("isAdult"),
                }
                for tag in raw_result.get("tags", [])
            ],
            # AniList-specific detailed data
            "anilist_id": anilist_id,
            "anilist_score": raw_result.get("averageScore"),
            "anilist_mean_score": raw_result.get("meanScore"),
            "anilist_popularity": raw_result.get("popularity"),
            "anilist_trending": raw_result.get("trending"),
            "anilist_favourites": raw_result.get("favourites"),
            "anilist_is_adult": raw_result.get("isAdult"),
            "anilist_country_of_origin": raw_result.get("countryOfOrigin"),
            "anilist_source": raw_result.get("source"),
            "anilist_hashtag": raw_result.get("hashtag"),
            "anilist_is_licensed": raw_result.get("isLicensed"),
            "anilist_updated_at": raw_result.get("updatedAt"),
            # External data
            "mal_id": raw_result.get("idMal"),
            "external_links": [
                {
                    "site": link.get("site"),
                    "url": link.get("url"),
                    "type": link.get("type"),
                }
                for link in raw_result.get("externalLinks", [])
            ],
            # Optional detailed data
            "relations": raw_result.get("relations", {}) if include_relations else {},
            "characters": (
                raw_result.get("characters", {}) if include_characters else {}
            ),
            "staff": raw_result.get("staff", {}) if include_staff else {},
            "reviews": raw_result.get("reviews", {}) if include_reviews else {},
            # Source attribution
            "source_platform": "anilist",
            "last_updated": raw_result.get("updatedAt"),
        }

        if ctx:
            await ctx.info(
                f"Retrieved comprehensive AniList data for '{result['title']}'"
            )

        return result

    except Exception as e:
        error_msg = f"Failed to get AniList anime {anilist_id}: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)
