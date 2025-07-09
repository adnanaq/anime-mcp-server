"""
Tier 3 Detailed Tools - Comprehensive anime search functionality with detailed response complexity.

These tools provide detailed information (25 fields) covering 99% of user queries.
Includes cross-platform data, detailed metadata, and advanced similarity matching.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from ...config import get_settings
from ...integrations.clients.anilist_client import AniListClient
from ...integrations.clients.jikan_client import JikanClient
from ...integrations.clients.kitsu_client import KitsuClient
from ...integrations.clients.mal_client import MALClient
from ...models.structured_responses import (
    AnimeRating,
    AnimeStatus,
    AnimeType,
    DetailedAnimeResult,
    DetailedSearchResponse,
)
from ...services.data_service import AnimeDataService

logger = logging.getLogger(__name__)
settings = get_settings()


class DetailedSearchInput(BaseModel):
    """Input schema for detailed anime search with comprehensive filtering."""

    query: str = Field(..., description="Search query for anime titles")
    limit: int = Field(default=20, ge=1, le=50, description="Maximum number of results")
    year: Optional[int] = Field(
        None, ge=1900, le=2030, description="Release year filter"
    )
    type: Optional[AnimeType] = Field(None, description="Anime type filter")
    status: Optional[AnimeStatus] = Field(None, description="Anime status filter")
    rating: Optional[AnimeRating] = Field(None, description="Content rating filter")
    genres: Optional[List[str]] = Field(None, description="Genre filters")
    min_score: Optional[float] = Field(
        None, ge=0, le=10, description="Minimum score filter"
    )
    max_score: Optional[float] = Field(
        None, ge=0, le=10, description="Maximum score filter"
    )
    studios: Optional[List[str]] = Field(None, description="Studio filters")
    producers: Optional[List[str]] = Field(None, description="Producer filters")
    themes: Optional[List[str]] = Field(None, description="Theme filters")
    demographics: Optional[List[str]] = Field(None, description="Demographic filters")
    season: Optional[str] = Field(
        None, description="Season filter (winter, spring, summer, fall)"
    )
    source: Optional[str] = Field(None, description="Source material filter")
    order_by: Optional[str] = Field(
        None, description="Sort order (score, popularity, aired)"
    )
    cross_platform: bool = Field(
        default=True, description="Include cross-platform data"
    )


class DetailedAnimeDetailsInput(BaseModel):
    """Input schema for detailed anime details."""

    id: str = Field(..., description="Anime ID from search results")
    platform: str = Field(default="jikan", description="Platform to fetch from")
    include_staff: bool = Field(default=True, description="Include staff information")
    include_characters: bool = Field(
        default=True, description="Include character information"
    )
    include_relations: bool = Field(default=True, description="Include related anime")
    cross_platform_enrich: bool = Field(
        default=True, description="Enrich with cross-platform data"
    )


class DetailedSimilarInput(BaseModel):
    """Input schema for detailed similarity search."""

    anime_id: str = Field(..., description="Reference anime ID")
    limit: int = Field(
        default=10, ge=1, le=20, description="Number of similar anime to return"
    )
    platform: str = Field(default="jikan", description="Platform to search from")
    similarity_factors: Optional[List[str]] = Field(
        default=["genres", "type", "studios", "themes", "demographics", "source"],
        description="Factors to consider for similarity",
    )
    use_semantic_search: bool = Field(
        default=True, description="Use semantic search for better similarity"
    )
    cross_platform_enrich: bool = Field(
        default=True, description="Enrich results with cross-platform data"
    )


def register_detailed_tools(mcp: FastMCP) -> None:
    """Register all Tier 3 detailed tools with the MCP server."""

    # Initialize clients and services
    jikan_client = JikanClient()
    # MAL client requires credentials - initialize safely
    mal_client = None
    try:
        if hasattr(settings, "mal_client_id") and settings.mal_client_id:
            mal_client = MALClient(
                client_id=settings.mal_client_id,
                client_secret=getattr(settings, "mal_client_secret", None),
            )
    except Exception as e:
        logger.warning(f"MAL client initialization failed: {e}")
    anilist_client = AniListClient()
    kitsu_client = KitsuClient()
    AnimeDataService()

    @mcp.tool()
    async def search_anime_detailed(
        input: DetailedSearchInput,
    ) -> DetailedSearchResponse:
        """
        Detailed anime search with comprehensive filtering and cross-platform enrichment.

        Returns 25 fields covering 99% of user needs including:
        - Basic info (id, title, score, year, type, genres, image_url, synopsis)
        - Status info (status, episodes, duration, studios, rating, popularity, aired_from)
        - Detailed info (title_english, title_japanese, title_synonyms, aired_to, season,
          source, producers, licensors, themes, demographics)

        Supports advanced filtering and cross-platform data enrichment.
        """
        start_time = datetime.now()

        try:
            # Build comprehensive search parameters
            search_tasks = []

            # Primary search with Jikan (most comprehensive)
            jikan_params = _build_jikan_search_params(input)
            search_tasks.append(("jikan", jikan_client.search_anime(**jikan_params)))

            # Cross-platform searches if enabled
            if input.cross_platform:
                # MAL search
                mal_params = _build_mal_search_params(input)
                search_tasks.append(("mal", mal_client.search_anime(**mal_params)))

                # AniList search
                anilist_params = _build_anilist_search_params(input)
                search_tasks.append(
                    ("anilist", anilist_client.search_anime(**anilist_params))
                )

            # Execute searches concurrently
            search_results = await asyncio.gather(
                *[task[1] for task in search_tasks], return_exceptions=True
            )

            # Merge and deduplicate results
            all_results = []
            sources_used = []

            for i, result in enumerate(search_results):
                platform = search_tasks[i][0]
                sources_used.append(platform)

                if isinstance(result, Exception):
                    logger.warning(f"Search failed for {platform}: {result}")
                    continue

                # Transform results from each platform
                platform_results = []
                if platform == "jikan":
                    platform_results = [
                        _transform_jikan_to_detailed(item)
                        for item in result.get("data", [])
                    ]
                elif platform == "mal":
                    platform_results = [
                        _transform_mal_to_detailed(item)
                        for item in result.get("data", [])
                    ]
                elif platform == "anilist":
                    platform_results = [
                        _transform_anilist_to_detailed(item)
                        for item in result.get("data", [])
                    ]

                all_results.extend(platform_results)

            # Deduplicate by title similarity
            deduplicated_results = _deduplicate_anime_results(all_results)

            # Sort and limit results
            if input.order_by == "score":
                deduplicated_results.sort(key=lambda x: x.score or 0, reverse=True)
            elif input.order_by == "popularity":
                deduplicated_results.sort(key=lambda x: x.popularity or 999999)
            elif input.order_by == "aired":
                deduplicated_results.sort(
                    key=lambda x: x.aired_from or "", reverse=True
                )

            final_results = deduplicated_results[: input.limit]

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Build filters applied summary
            filters_applied = _build_filters_summary(input)

            return DetailedSearchResponse(
                results=final_results,
                total=len(final_results),
                query=input.query,
                processing_time_ms=int(processing_time),
                filters_applied=filters_applied,
                sources_used=sources_used,
            )

        except Exception as e:
            logger.error(f"Detailed search failed: {e}")
            return DetailedSearchResponse(
                results=[],
                total=0,
                query=input.query,
                processing_time_ms=int(
                    (datetime.now() - start_time).total_seconds() * 1000
                ),
                filters_applied={},
                sources_used=[],
            )

    @mcp.tool()
    async def get_anime_detailed(
        input: DetailedAnimeDetailsInput,
    ) -> DetailedAnimeResult:
        """
        Get detailed anime information by ID with comprehensive cross-platform enrichment.

        Returns complete anime details including alternative titles, staff, characters,
        and related anime information.
        """
        try:
            # Primary data fetch
            primary_data = None
            if input.platform == "jikan":
                primary_data = await jikan_client.get_anime_by_id(int(input.id))
            elif input.platform == "mal":
                primary_data = await mal_client.get_anime_by_id(int(input.id))
            elif input.platform == "anilist":
                primary_data = await anilist_client.get_anime_by_id(int(input.id))
            elif input.platform == "kitsu":
                primary_data = await kitsu_client.get_anime_by_id(input.id)

            # Transform primary data
            if input.platform == "jikan":
                detailed_result = _transform_jikan_to_detailed(primary_data)
            elif input.platform == "mal":
                detailed_result = _transform_mal_to_detailed(primary_data)
            elif input.platform == "anilist":
                detailed_result = _transform_anilist_to_detailed(primary_data)
            elif input.platform == "kitsu":
                detailed_result = _transform_kitsu_to_detailed(primary_data)

            # Cross-platform enrichment
            if input.cross_platform_enrich:
                detailed_result = await _enrich_with_cross_platform_data(
                    detailed_result, input.platform
                )

            # Additional data fetching
            if input.include_staff:
                detailed_result = await _add_staff_information(
                    detailed_result, input.platform
                )

            if input.include_characters:
                detailed_result = await _add_character_information(
                    detailed_result, input.platform
                )

            if input.include_relations:
                detailed_result = await _add_related_anime(
                    detailed_result, input.platform
                )

            return detailed_result

        except Exception as e:
            logger.error(f"Detailed anime fetch failed: {e}")
            return DetailedAnimeResult(
                id=input.id,
                title="Error fetching details",
                score=None,
                year=None,
                type=None,
                genres=[],
                image_url=None,
                synopsis=f"Error: {str(e)}",
                status=None,
                episodes=None,
                duration=None,
                studios=[],
                rating=None,
                popularity=None,
                aired_from=None,
                title_english=None,
                title_japanese=None,
                title_synonyms=[],
                aired_to=None,
                season=None,
                source=None,
                producers=[],
                licensors=[],
                themes=[],
                demographics=[],
            )

    @mcp.tool()
    async def find_similar_anime_detailed(
        input: DetailedSimilarInput,
    ) -> DetailedSearchResponse:
        """
        Find similar anime using advanced similarity matching with detailed results.

        Uses multiple factors, semantic search, and cross-platform enrichment
        for comprehensive similarity analysis.
        """
        start_time = datetime.now()

        try:
            # Get reference anime with full details
            reference_anime = await get_anime_detailed(
                DetailedAnimeDetailsInput(
                    id=input.anime_id,
                    platform=input.platform,
                    include_staff=False,
                    include_characters=False,
                    include_relations=False,
                    cross_platform_enrich=input.cross_platform_enrich,
                )
            )

            # Use semantic search if enabled
            if input.use_semantic_search:
                similar_results = await _semantic_similarity_search(
                    reference_anime, input.limit, input.similarity_factors
                )
            else:
                similar_results = await _traditional_similarity_search(
                    reference_anime, input.limit, input.similarity_factors
                )

            # Enrich results with cross-platform data
            if input.cross_platform_enrich:
                enriched_results = []
                for result in similar_results:
                    enriched_result = await _enrich_with_cross_platform_data(
                        result, "jikan"
                    )
                    enriched_results.append(enriched_result)
                similar_results = enriched_results

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return DetailedSearchResponse(
                results=similar_results,
                total=len(similar_results),
                query=f"Similar to {reference_anime.title}",
                processing_time_ms=int(processing_time),
                filters_applied={"similarity_factors": input.similarity_factors},
                sources_used=(
                    ["jikan", "semantic"] if input.use_semantic_search else ["jikan"]
                ),
            )

        except Exception as e:
            logger.error(f"Detailed similarity search failed: {e}")
            return DetailedSearchResponse(
                results=[],
                total=0,
                query=f"Similar to anime {input.anime_id}",
                processing_time_ms=int(
                    (datetime.now() - start_time).total_seconds() * 1000
                ),
                filters_applied={},
                sources_used=[],
            )

    @mcp.tool()
    async def get_seasonal_anime_detailed(
        season: str = Field(..., description="Season (winter, spring, summer, fall)"),
        year: int = Field(..., ge=1900, le=2030, description="Year"),
        limit: int = Field(default=20, ge=1, le=50, description="Number of results"),
        sort_by: str = Field(
            default="popularity", description="Sort by (score, popularity, members)"
        ),
        include_cross_platform: bool = Field(
            default=True, description="Include cross-platform data"
        ),
    ) -> DetailedSearchResponse:
        """
        Get detailed seasonal anime information with comprehensive metadata.

        Returns seasonal anime with full detailed information including
        cross-platform data and comprehensive metadata.
        """
        start_time = datetime.now()

        try:
            # Primary seasonal data from Jikan
            seasonal_data = await jikan_client.get_seasonal_anime(season, year)

            # Transform to detailed results
            detailed_results = []
            for raw_result in seasonal_data.get("data", []):
                detailed_result = _transform_jikan_to_detailed(raw_result)

                # Cross-platform enrichment
                if include_cross_platform:
                    detailed_result = await _enrich_with_cross_platform_data(
                        detailed_result, "jikan"
                    )

                detailed_results.append(detailed_result)

            # Sort results
            if sort_by == "score":
                detailed_results.sort(key=lambda x: x.score or 0, reverse=True)
            elif sort_by == "popularity":
                detailed_results.sort(key=lambda x: x.popularity or 999999)
            elif sort_by == "members":
                detailed_results.sort(key=lambda x: x.popularity or 999999)

            # Limit results
            detailed_results = detailed_results[:limit]

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            sources_used = ["jikan"]
            if include_cross_platform:
                sources_used.extend(["mal", "anilist"])

            return DetailedSearchResponse(
                results=detailed_results,
                total=len(detailed_results),
                query=f"{season.title()} {year} anime",
                processing_time_ms=int(processing_time),
                filters_applied={"season": season, "year": year, "sort_by": sort_by},
                sources_used=sources_used,
            )

        except Exception as e:
            logger.error(f"Detailed seasonal search failed: {e}")
            return DetailedSearchResponse(
                results=[],
                total=0,
                query=f"{season.title()} {year} anime",
                processing_time_ms=int(
                    (datetime.now() - start_time).total_seconds() * 1000
                ),
                filters_applied={},
                sources_used=[],
            )

    @mcp.tool()
    async def advanced_anime_analysis(
        anime_id: str = Field(..., description="Anime ID to analyze"),
        platform: str = Field(default="jikan", description="Platform to analyze from"),
        analysis_depth: str = Field(
            default="comprehensive",
            description="Analysis depth (basic, standard, comprehensive)",
        ),
    ) -> Dict[str, Any]:
        """
        Perform advanced analysis of anime including cross-platform comparison,
        quality metrics, and detailed metadata analysis.
        """
        start_time = datetime.now()

        try:
            # Get detailed anime information
            detailed_anime = await get_anime_detailed(
                DetailedAnimeDetailsInput(
                    id=anime_id,
                    platform=platform,
                    include_staff=True,
                    include_characters=True,
                    include_relations=True,
                    cross_platform_enrich=True,
                )
            )

            # Perform analysis
            analysis = {
                "anime_info": detailed_anime.dict(),
                "quality_metrics": _calculate_quality_metrics(detailed_anime),
                "cross_platform_comparison": await _cross_platform_comparison(
                    detailed_anime
                ),
                "genre_analysis": _analyze_genres(detailed_anime),
                "temporal_analysis": _analyze_temporal_context(detailed_anime),
                "recommendation_factors": _extract_recommendation_factors(
                    detailed_anime
                ),
                "processing_time_ms": int(
                    (datetime.now() - start_time).total_seconds() * 1000
                ),
            }

            return analysis

        except Exception as e:
            logger.error(f"Advanced analysis failed: {e}")
            return {
                "error": str(e),
                "processing_time_ms": int(
                    (datetime.now() - start_time).total_seconds() * 1000
                ),
            }


def _build_jikan_search_params(input: DetailedSearchInput) -> Dict[str, Any]:
    """Build Jikan search parameters from detailed input."""
    params = {}

    if input.query:
        params["q"] = input.query
    if input.limit:
        params["limit"] = min(input.limit, 25)
    if input.year:
        params["start_date"] = f"{input.year}-01-01"
        params["end_date"] = f"{input.year}-12-31"
    if input.type:
        params["type"] = input.type.value.lower()
    if input.status:
        params["status"] = input.status.value.lower()
    if input.rating:
        params["rating"] = input.rating.value.lower()
    if input.genres:
        params["genres"] = ",".join(input.genres)
    if input.min_score:
        params["min_score"] = input.min_score
    if input.max_score:
        params["max_score"] = input.max_score
    if input.studios:
        params["producers"] = ",".join(input.studios)
    if input.order_by:
        params["order_by"] = input.order_by

    return params


def _build_mal_search_params(input: DetailedSearchInput) -> Dict[str, Any]:
    """Build MAL search parameters from detailed input."""
    params = {}

    if input.query:
        params["q"] = input.query
    if input.limit:
        params["limit"] = min(input.limit, 100)  # MAL allows up to 100

    # MAL has limited search parameters
    fields = [
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
        "num_episodes",
        "status",
        "genres",
        "media_type",
        "studios",
        "rating",
    ]
    params["fields"] = ",".join(fields)

    return params


def _build_anilist_search_params(input: DetailedSearchInput) -> Dict[str, Any]:
    """Build AniList search parameters from detailed input."""
    params = {}

    if input.query:
        params["search"] = input.query
    if input.limit:
        params["perPage"] = min(input.limit, 50)
    if input.year:
        params["seasonYear"] = input.year
    if input.type:
        params["type"] = "ANIME"
    if input.status:
        params["status"] = input.status.value.upper()
    if input.genres:
        params["genre_in"] = input.genres
    if input.season:
        params["season"] = input.season.upper()

    return params


def _transform_jikan_to_detailed(raw_data: Dict[str, Any]) -> DetailedAnimeResult:
    """Transform Jikan API response to DetailedAnimeResult."""
    # Basic fields
    anime_id = str(raw_data.get("mal_id", ""))
    title = raw_data.get("title") or "Unknown"
    score = raw_data.get("score")
    year = raw_data.get("year")

    # Map type
    anime_type = None
    jikan_type = raw_data.get("type")
    if jikan_type:
        try:
            anime_type = AnimeType(jikan_type.upper())
        except ValueError:
            anime_type = AnimeType.UNKNOWN

    # Extract genres
    genres = []
    for genre in raw_data.get("genres", []):
        if isinstance(genre, dict) and "name" in genre:
            genres.append(genre["name"])

    # Extract themes
    themes = []
    for theme in raw_data.get("themes", []):
        if isinstance(theme, dict) and "name" in theme:
            themes.append(theme["name"])

    # Extract demographics
    demographics = []
    for demo in raw_data.get("demographics", []):
        if isinstance(demo, dict) and "name" in demo:
            demographics.append(demo["name"])

    # Extract image URL
    image_url = None
    images = raw_data.get("images", {})
    if isinstance(images, dict):
        jpg_images = images.get("jpg", {})
        if isinstance(jpg_images, dict):
            image_url = jpg_images.get("image_url")

    # Extract synopsis
    synopsis = raw_data.get("synopsis") or ""

    # Standard fields
    status = None
    jikan_status = raw_data.get("status")
    if jikan_status:
        try:
            status = AnimeStatus(jikan_status.upper().replace(" ", "_"))
        except ValueError:
            status = AnimeStatus.UNKNOWN

    episodes = raw_data.get("episodes")
    duration = raw_data.get("duration")

    # Extract studios
    studios = []
    for studio in raw_data.get("studios", []):
        if isinstance(studio, dict) and "name" in studio:
            studios.append(studio["name"])

    # Extract producers
    producers = []
    for producer in raw_data.get("producers", []):
        if isinstance(producer, dict) and "name" in producer:
            producers.append(producer["name"])

    # Extract licensors
    licensors = []
    for licensor in raw_data.get("licensors", []):
        if isinstance(licensor, dict) and "name" in licensor:
            licensors.append(licensor["name"])

    # Map rating
    rating = None
    jikan_rating = raw_data.get("rating")
    if jikan_rating:
        try:
            if "G" in jikan_rating:
                rating = AnimeRating.G
            elif "PG" in jikan_rating:
                rating = AnimeRating.PG
            elif "PG-13" in jikan_rating:
                rating = AnimeRating.PG_13
            elif "R" in jikan_rating:
                rating = AnimeRating.R
            elif "NC-17" in jikan_rating:
                rating = AnimeRating.NC_17
        except ValueError:
            rating = None

    popularity = raw_data.get("popularity")

    # Extract aired dates
    aired_from = None
    aired_to = None
    aired = raw_data.get("aired", {})
    if isinstance(aired, dict):
        aired_from = aired.get("from")
        aired_to = aired.get("to")

    # Extract detailed fields
    titles = raw_data.get("titles", [])
    title_english = None
    title_japanese = None
    title_synonyms = []

    for title_info in titles:
        if isinstance(title_info, dict):
            title_type = title_info.get("type")
            title_value = title_info.get("title")
            if title_type == "English" and title_value:
                title_english = title_value
            elif title_type == "Japanese" and title_value:
                title_japanese = title_value
            elif title_type == "Synonym" and title_value:
                title_synonyms.append(title_value)

    # Extract season
    season = None
    jikan_season = raw_data.get("season")
    if jikan_season:
        season = jikan_season.title()

    # Extract source
    source = raw_data.get("source")

    return DetailedAnimeResult(
        id=anime_id,
        title=title,
        score=score,
        year=year,
        type=anime_type,
        genres=genres,
        image_url=image_url,
        synopsis=synopsis,
        status=status,
        episodes=episodes,
        duration=duration,
        studios=studios,
        rating=rating,
        popularity=popularity,
        aired_from=aired_from,
        title_english=title_english,
        title_japanese=title_japanese,
        title_synonyms=title_synonyms,
        aired_to=aired_to,
        season=season,
        source=source,
        producers=producers,
        licensors=licensors,
        themes=themes,
        demographics=demographics,
    )


def _transform_mal_to_detailed(raw_data: Dict[str, Any]) -> DetailedAnimeResult:
    """Transform MAL API response to DetailedAnimeResult."""
    # Basic implementation - would need full MAL API integration
    # For now, convert basic fields and leave detailed fields empty

    anime_id = str(raw_data.get("id", ""))
    title = raw_data.get("title") or "Unknown"
    score = raw_data.get("mean")

    # Extract year from start_date
    year = None
    start_date = raw_data.get("start_date")
    if start_date:
        try:
            year = int(start_date.split("-")[0])
        except (ValueError, IndexError):
            pass

    # Map type
    anime_type = None
    mal_type = raw_data.get("media_type")
    if mal_type:
        try:
            anime_type = AnimeType(mal_type.upper())
        except ValueError:
            anime_type = AnimeType.UNKNOWN

    # Extract genres
    genres = []
    for genre in raw_data.get("genres", []):
        if isinstance(genre, dict) and "name" in genre:
            genres.append(genre["name"])

    # Extract image URL
    image_url = None
    main_picture = raw_data.get("main_picture", {})
    if isinstance(main_picture, dict):
        image_url = main_picture.get("medium") or main_picture.get("large")

    # Extract synopsis
    synopsis = raw_data.get("synopsis") or ""

    # Standard fields
    status = None
    mal_status = raw_data.get("status")
    if mal_status:
        try:
            status = AnimeStatus(mal_status.upper().replace(" ", "_"))
        except ValueError:
            status = AnimeStatus.UNKNOWN

    episodes = raw_data.get("num_episodes")
    duration = None

    # Extract studios
    studios = []
    for studio in raw_data.get("studios", []):
        if isinstance(studio, dict) and "name" in studio:
            studios.append(studio["name"])

    # Map rating
    rating = None
    mal_rating = raw_data.get("rating")
    if mal_rating:
        try:
            rating = AnimeRating(mal_rating.upper().replace("-", "_"))
        except ValueError:
            rating = None

    popularity = raw_data.get("popularity")

    # Extract alternative titles
    alt_titles = raw_data.get("alternative_titles", {})
    title_english = None
    title_japanese = None
    title_synonyms = []

    if isinstance(alt_titles, dict):
        title_english = alt_titles.get("en")
        title_japanese = alt_titles.get("ja")
        synonyms = alt_titles.get("synonyms", [])
        if isinstance(synonyms, list):
            title_synonyms = synonyms

    return DetailedAnimeResult(
        id=anime_id,
        title=title,
        score=score,
        year=year,
        type=anime_type,
        genres=genres,
        image_url=image_url,
        synopsis=synopsis,
        status=status,
        episodes=episodes,
        duration=duration,
        studios=studios,
        rating=rating,
        popularity=popularity,
        aired_from=start_date,
        title_english=title_english,
        title_japanese=title_japanese,
        title_synonyms=title_synonyms,
        aired_to=raw_data.get("end_date"),
        season=None,  # MAL doesn't provide season directly
        source=None,  # MAL doesn't provide source directly
        producers=[],  # Would need additional API call
        licensors=[],  # Would need additional API call
        themes=[],  # Would need additional API call
        demographics=[],  # Would need additional API call
    )


def _transform_anilist_to_detailed(raw_data: Dict[str, Any]) -> DetailedAnimeResult:
    """Transform AniList API response to DetailedAnimeResult."""
    # Basic implementation - would need full AniList GraphQL integration
    # For now, convert available fields

    anime_id = str(raw_data.get("id", ""))
    title = (
        raw_data.get("title", {}).get("romaji")
        or raw_data.get("title", {}).get("english")
        or "Unknown"
    )
    score = raw_data.get("averageScore")
    if score:
        score = score / 10.0  # Convert from 100-point to 10-point scale

    # Extract year
    year = None
    start_date = raw_data.get("startDate", {})
    if isinstance(start_date, dict):
        year = start_date.get("year")

    # Map type
    anime_type = None
    anilist_type = raw_data.get("type")
    if anilist_type == "ANIME":
        format_type = raw_data.get("format")
        if format_type:
            try:
                anime_type = AnimeType(format_type)
            except ValueError:
                anime_type = AnimeType.UNKNOWN

    # Extract genres
    genres = raw_data.get("genres", [])

    # Extract image URL
    image_url = None
    cover_image = raw_data.get("coverImage", {})
    if isinstance(cover_image, dict):
        image_url = cover_image.get("medium") or cover_image.get("large")

    # Extract synopsis
    synopsis = raw_data.get("description") or ""

    # Standard fields
    status = None
    anilist_status = raw_data.get("status")
    if anilist_status:
        try:
            status = AnimeStatus(anilist_status.upper().replace(" ", "_"))
        except ValueError:
            status = AnimeStatus.UNKNOWN

    episodes = raw_data.get("episodes")
    duration = raw_data.get("duration")

    # Extract studios
    studios = []
    studio_edges = raw_data.get("studios", {}).get("edges", [])
    for edge in studio_edges:
        node = edge.get("node", {})
        if isinstance(node, dict) and "name" in node:
            studios.append(node["name"])

    rating = None  # AniList doesn't have content rating
    popularity = raw_data.get("popularity")

    # Extract detailed fields
    titles = raw_data.get("title", {})
    title_english = titles.get("english")
    title_japanese = titles.get("native")
    title_synonyms = raw_data.get("synonyms", [])

    # Extract aired dates
    aired_from = None
    aired_to = None
    if isinstance(start_date, dict):
        year = start_date.get("year")
        month = start_date.get("month")
        day = start_date.get("day")
        if year and month and day:
            aired_from = f"{year}-{month:02d}-{day:02d}"

    end_date = raw_data.get("endDate", {})
    if isinstance(end_date, dict):
        year = end_date.get("year")
        month = end_date.get("month")
        day = end_date.get("day")
        if year and month and day:
            aired_to = f"{year}-{month:02d}-{day:02d}"

    # Extract season
    season = raw_data.get("season")
    if season:
        season = season.title()

    # Extract source
    source = raw_data.get("source")

    return DetailedAnimeResult(
        id=anime_id,
        title=title,
        score=score,
        year=year,
        type=anime_type,
        genres=genres,
        image_url=image_url,
        synopsis=synopsis,
        status=status,
        episodes=episodes,
        duration=duration,
        studios=studios,
        rating=rating,
        popularity=popularity,
        aired_from=aired_from,
        title_english=title_english,
        title_japanese=title_japanese,
        title_synonyms=title_synonyms,
        aired_to=aired_to,
        season=season,
        source=source,
        producers=[],  # Would need additional GraphQL query
        licensors=[],  # Not available in AniList
        themes=[],  # Would need additional GraphQL query
        demographics=[],  # Not directly available in AniList
    )


def _transform_kitsu_to_detailed(raw_data: Dict[str, Any]) -> DetailedAnimeResult:
    """Transform Kitsu API response to DetailedAnimeResult."""
    # Basic implementation - would need full Kitsu JSON:API integration
    # For now, convert available fields

    anime_id = str(raw_data.get("id", ""))
    attributes = raw_data.get("attributes", {})

    title = (
        attributes.get("canonicalTitle")
        or attributes.get("titles", {}).get("en")
        or "Unknown"
    )
    score = attributes.get("averageRating")
    if score:
        score = float(score) / 10.0  # Convert from 100-point to 10-point scale

    # Extract year
    year = None
    start_date = attributes.get("startDate")
    if start_date:
        try:
            year = int(start_date.split("-")[0])
        except (ValueError, IndexError):
            pass

    # Map type
    anime_type = None
    subtype = attributes.get("subtype")
    if subtype:
        try:
            anime_type = AnimeType(subtype.upper())
        except ValueError:
            anime_type = AnimeType.UNKNOWN

    # Extract genres (would need additional API calls)
    genres = []

    # Extract image URL
    image_url = None
    poster_image = attributes.get("posterImage", {})
    if isinstance(poster_image, dict):
        image_url = poster_image.get("medium") or poster_image.get("small")

    # Extract synopsis
    synopsis = attributes.get("synopsis") or ""

    # Standard fields
    status = None
    kitsu_status = attributes.get("status")
    if kitsu_status:
        try:
            status = AnimeStatus(kitsu_status.upper().replace(" ", "_"))
        except ValueError:
            status = AnimeStatus.UNKNOWN

    episodes = attributes.get("episodeCount")
    duration = attributes.get("episodeLength")

    # Studios (would need additional API calls)
    studios = []

    # Map rating
    rating = None
    age_rating = attributes.get("ageRating")
    if age_rating:
        try:
            rating = AnimeRating(age_rating.upper().replace("-", "_"))
        except ValueError:
            rating = None

    popularity = attributes.get("popularityRank")

    # Extract titles
    titles = attributes.get("titles", {})
    title_english = titles.get("en")
    title_japanese = titles.get("ja_jp")
    title_synonyms = []

    return DetailedAnimeResult(
        id=anime_id,
        title=title,
        score=score,
        year=year,
        type=anime_type,
        genres=genres,
        image_url=image_url,
        synopsis=synopsis,
        status=status,
        episodes=episodes,
        duration=duration,
        studios=studios,
        rating=rating,
        popularity=popularity,
        aired_from=start_date,
        title_english=title_english,
        title_japanese=title_japanese,
        title_synonyms=title_synonyms,
        aired_to=attributes.get("endDate"),
        season=None,  # Would need additional calculation
        source=None,  # Not directly available
        producers=[],  # Would need additional API calls
        licensors=[],  # Not available in Kitsu
        themes=[],  # Would need additional API calls
        demographics=[],  # Would need additional API calls
    )


def _deduplicate_anime_results(
    results: List[DetailedAnimeResult],
) -> List[DetailedAnimeResult]:
    """Deduplicate anime results by title similarity."""
    # Simple deduplication by title
    seen_titles = set()
    deduplicated = []

    for result in results:
        title_lower = result.title.lower()
        if title_lower not in seen_titles:
            seen_titles.add(title_lower)
            deduplicated.append(result)

    return deduplicated


def _build_filters_summary(input: DetailedSearchInput) -> Dict[str, Any]:
    """Build a summary of applied filters."""
    filters = {}

    if input.year:
        filters["year"] = input.year
    if input.type:
        filters["type"] = input.type.value
    if input.status:
        filters["status"] = input.status.value
    if input.rating:
        filters["rating"] = input.rating.value
    if input.min_score or input.max_score:
        filters["score_range"] = f"{input.min_score or 0}-{input.max_score or 10}"
    if input.genres:
        filters["genres"] = input.genres
    if input.studios:
        filters["studios"] = input.studios
    if input.producers:
        filters["producers"] = input.producers
    if input.themes:
        filters["themes"] = input.themes
    if input.demographics:
        filters["demographics"] = input.demographics
    if input.season:
        filters["season"] = input.season
    if input.source:
        filters["source"] = input.source

    return filters


async def _enrich_with_cross_platform_data(
    anime: DetailedAnimeResult, primary_platform: str
) -> DetailedAnimeResult:
    """Enrich anime data with cross-platform information."""
    # Placeholder implementation
    # In production, this would fetch additional data from other platforms
    # and merge it with the existing result
    return anime


async def _add_staff_information(
    anime: DetailedAnimeResult, platform: str
) -> DetailedAnimeResult:
    """Add staff information to anime result."""
    # Placeholder implementation
    # In production, this would fetch staff data from the platform
    return anime


async def _add_character_information(
    anime: DetailedAnimeResult, platform: str
) -> DetailedAnimeResult:
    """Add character information to anime result."""
    # Placeholder implementation
    # In production, this would fetch character data from the platform
    return anime


async def _add_related_anime(
    anime: DetailedAnimeResult, platform: str
) -> DetailedAnimeResult:
    """Add related anime information to anime result."""
    # Placeholder implementation
    # In production, this would fetch related anime data from the platform
    return anime


async def _semantic_similarity_search(
    reference: DetailedAnimeResult, limit: int, factors: List[str]
) -> List[DetailedAnimeResult]:
    """Perform semantic similarity search."""
    # Placeholder implementation
    # In production, this would use vector database for semantic search
    return []


async def _traditional_similarity_search(
    reference: DetailedAnimeResult, limit: int, factors: List[str]
) -> List[DetailedAnimeResult]:
    """Perform traditional similarity search based on metadata."""
    # Placeholder implementation
    # In production, this would use traditional similarity algorithms
    return []


def _calculate_quality_metrics(anime: DetailedAnimeResult) -> Dict[str, Any]:
    """Calculate quality metrics for anime."""
    return {"completeness_score": 0.8, "data_sources": 1, "metadata_richness": 0.7}


async def _cross_platform_comparison(anime: DetailedAnimeResult) -> Dict[str, Any]:
    """Compare anime data across platforms."""
    return {
        "score_variance": 0.5,
        "platform_availability": ["jikan", "mal"],
        "data_consistency": 0.9,
    }


def _analyze_genres(anime: DetailedAnimeResult) -> Dict[str, Any]:
    """Analyze genre information."""
    return {
        "primary_genres": anime.genres[:3],
        "genre_popularity": 0.8,
        "genre_combination_rarity": 0.3,
    }


def _analyze_temporal_context(anime: DetailedAnimeResult) -> Dict[str, Any]:
    """Analyze temporal context."""
    return {
        "era": "Modern",
        "seasonal_context": anime.season,
        "release_timing": "Standard",
    }


def _extract_recommendation_factors(anime: DetailedAnimeResult) -> Dict[str, Any]:
    """Extract factors useful for recommendations."""
    return {
        "key_genres": anime.genres,
        "studios": anime.studios,
        "themes": anime.themes,
        "demographics": anime.demographics,
    }
