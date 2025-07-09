"""
Tier 2 Standard Tools - Enhanced anime search functionality with moderate response complexity.

These tools provide commonly needed information (15 fields) covering 95% of user queries.
Includes advanced filtering, status information, and episode details.
"""

from typing import List, Optional, Dict, Any, Union
import asyncio
import logging
from datetime import datetime

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from ..handlers.anime_handler import AnimeHandler
from ...models.structured_responses import (
    StandardAnimeResult,
    StandardSearchResponse,
    AnimeType,
    AnimeStatus,
    AnimeRating
)
from ...integrations.clients.jikan_client import JikanClient
from ...integrations.clients.mal_client import MALClient
from ...integrations.clients.anilist_client import AniListClient
from ...integrations.clients.kitsu_client import KitsuClient
from ...config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class StandardSearchInput(BaseModel):
    """Input schema for standard anime search with enhanced filtering."""
    query: str = Field(..., description="Search query for anime titles")
    limit: int = Field(default=20, ge=1, le=50, description="Maximum number of results")
    year: Optional[int] = Field(None, ge=1900, le=2030, description="Release year filter")
    type: Optional[AnimeType] = Field(None, description="Anime type filter")
    status: Optional[AnimeStatus] = Field(None, description="Anime status filter")
    rating: Optional[AnimeRating] = Field(None, description="Content rating filter")
    genres: Optional[List[str]] = Field(None, description="Genre filters")
    min_score: Optional[float] = Field(None, ge=0, le=10, description="Minimum score filter")
    max_score: Optional[float] = Field(None, ge=0, le=10, description="Maximum score filter")
    studios: Optional[List[str]] = Field(None, description="Studio filters")
    order_by: Optional[str] = Field(None, description="Sort order (score, popularity, aired)")


class StandardAnimeDetailsInput(BaseModel):
    """Input schema for standard anime details."""
    id: str = Field(..., description="Anime ID from search results")
    platform: str = Field(default="jikan", description="Platform to fetch from")
    include_staff: bool = Field(default=False, description="Include staff information")
    include_characters: bool = Field(default=False, description="Include character information")


class StandardSimilarInput(BaseModel):
    """Input schema for standard similarity search."""
    anime_id: str = Field(..., description="Reference anime ID")
    limit: int = Field(default=10, ge=1, le=20, description="Number of similar anime to return")
    platform: str = Field(default="jikan", description="Platform to search from")
    similarity_factors: Optional[List[str]] = Field(
        default=["genres", "type", "studios"], 
        description="Factors to consider for similarity"
    )


def register_standard_tools(mcp: FastMCP) -> None:
    """Register all Tier 2 standard tools with the MCP server."""
    
    # Initialize clients
    jikan_client = JikanClient()
    mal_client = MALClient()
    anilist_client = AniListClient()
    kitsu_client = KitsuClient()
    
    @mcp.tool()
    async def search_anime_standard(input: StandardSearchInput) -> StandardSearchResponse:
        """
        Standard anime search with enhanced filtering and moderate detail.
        
        Returns 15 fields covering 95% of user needs including:
        - Basic info (id, title, score, year, type, genres, image_url, synopsis)
        - Status info (status, episodes, duration, studios, rating, popularity, aired_from)
        
        Supports advanced filtering and sorting options.
        """
        start_time = datetime.now()
        
        try:
            # Build enhanced search parameters
            jikan_params = {}
            if input.query:
                jikan_params["q"] = input.query
            if input.limit:
                jikan_params["limit"] = min(input.limit, 25)
            if input.year:
                jikan_params["start_date"] = f"{input.year}-01-01"
                jikan_params["end_date"] = f"{input.year}-12-31"
            if input.type:
                jikan_params["type"] = input.type.value.lower()
            if input.status:
                jikan_params["status"] = input.status.value.lower()
            if input.rating:
                jikan_params["rating"] = input.rating.value.lower()
            if input.min_score:
                jikan_params["min_score"] = input.min_score
            if input.max_score:
                jikan_params["max_score"] = input.max_score
            if input.order_by:
                jikan_params["order_by"] = input.order_by
            
            # Handle genres (convert to genre IDs for Jikan)
            if input.genres:
                # For simplicity, use genre names directly (Jikan supports this)
                jikan_params["genres"] = ",".join(input.genres)
            
            # Handle studios
            if input.studios:
                # For simplicity, use studio names directly
                jikan_params["producers"] = ",".join(input.studios)
            
            # Execute search
            raw_results = await jikan_client.search_anime(**jikan_params)
            
            # Transform to StandardAnimeResult
            results = []
            for raw_result in raw_results.get("data", []):
                standard_result = _transform_jikan_to_standard(raw_result)
                results.append(standard_result)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Build filters applied summary
            filters_applied = {}
            if input.year:
                filters_applied["year"] = input.year
            if input.type:
                filters_applied["type"] = input.type.value
            if input.status:
                filters_applied["status"] = input.status.value
            if input.rating:
                filters_applied["rating"] = input.rating.value
            if input.min_score or input.max_score:
                filters_applied["score_range"] = f"{input.min_score or 0}-{input.max_score or 10}"
            if input.genres:
                filters_applied["genres"] = input.genres
            if input.studios:
                filters_applied["studios"] = input.studios
            
            return StandardSearchResponse(
                results=results,
                total=len(results),
                query=input.query,
                processing_time_ms=int(processing_time),
                filters_applied=filters_applied
            )
            
        except Exception as e:
            logger.error(f"Standard search failed: {e}")
            return StandardSearchResponse(
                results=[],
                total=0,
                query=input.query,
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                filters_applied={}
            )
    
    @mcp.tool()
    async def get_anime_standard(input: StandardAnimeDetailsInput) -> StandardAnimeResult:
        """
        Get standard anime details by ID with enhanced information.
        
        Returns comprehensive information including episode counts, duration,
        studios, ratings, and popularity metrics.
        """
        try:
            if input.platform == "jikan":
                raw_data = await jikan_client.get_anime_by_id(int(input.id))
            elif input.platform == "mal":
                raw_data = await mal_client.get_anime_by_id(int(input.id))
            elif input.platform == "anilist":
                raw_data = await anilist_client.get_anime_by_id(int(input.id))
            elif input.platform == "kitsu":
                raw_data = await kitsu_client.get_anime_by_id(input.id)
            else:
                raise ValueError(f"Unsupported platform: {input.platform}")
            
            # Transform to StandardAnimeResult based on platform
            if input.platform == "jikan":
                return _transform_jikan_to_standard(raw_data)
            elif input.platform == "mal":
                return _transform_mal_to_standard(raw_data)
            elif input.platform == "anilist":
                return _transform_anilist_to_standard(raw_data)
            elif input.platform == "kitsu":
                return _transform_kitsu_to_standard(raw_data)
                
        except Exception as e:
            logger.error(f"Standard details fetch failed: {e}")
            return StandardAnimeResult(
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
                aired_from=None
            )
    
    @mcp.tool()
    async def find_similar_anime_standard(input: StandardSimilarInput) -> StandardSearchResponse:
        """
        Find similar anime using enhanced similarity matching.
        
        Uses multiple factors (genres, type, studios, ratings) for better
        similarity matching with standard response detail.
        """
        start_time = datetime.now()
        
        try:
            # Get reference anime details first
            reference_anime = await get_anime_standard(StandardAnimeDetailsInput(
                id=input.anime_id,
                platform=input.platform
            ))
            
            # Build similarity search parameters
            similar_params = {
                "limit": input.limit * 2,  # Get more results for better filtering
            }
            
            # Add similarity factors
            if "genres" in input.similarity_factors and reference_anime.genres:
                similar_params["genres"] = ",".join(reference_anime.genres[:5])
            if "type" in input.similarity_factors and reference_anime.type:
                similar_params["type"] = reference_anime.type.value.lower()
            if "studios" in input.similarity_factors and reference_anime.studios:
                similar_params["producers"] = ",".join(reference_anime.studios[:3])
            
            # Score range for similar quality
            if reference_anime.score:
                similar_params["min_score"] = max(0, reference_anime.score - 2)
                similar_params["max_score"] = min(10, reference_anime.score + 2)
            
            # Search for similar anime
            raw_results = await jikan_client.search_anime(**similar_params)
            
            # Transform and rank results
            results = []
            for raw_result in raw_results.get("data", []):
                # Skip the reference anime itself
                if str(raw_result.get("mal_id")) == input.anime_id:
                    continue
                
                standard_result = _transform_jikan_to_standard(raw_result)
                
                # Calculate similarity score
                similarity_score = _calculate_similarity_score(
                    reference_anime, standard_result, input.similarity_factors
                )
                
                # Add similarity score as a custom field
                standard_result.synopsis = f"[Similarity: {similarity_score:.2f}] {standard_result.synopsis}"
                
                results.append((similarity_score, standard_result))
            
            # Sort by similarity score and take top results
            results.sort(key=lambda x: x[0], reverse=True)
            final_results = [result[1] for result in results[:input.limit]]
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return StandardSearchResponse(
                results=final_results,
                total=len(final_results),
                query=f"Similar to {reference_anime.title}",
                processing_time_ms=int(processing_time),
                filters_applied={"similarity_factors": input.similarity_factors}
            )
            
        except Exception as e:
            logger.error(f"Standard similarity search failed: {e}")
            return StandardSearchResponse(
                results=[],
                total=0,
                query=f"Similar to anime {input.anime_id}",
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                filters_applied={}
            )
    
    @mcp.tool()
    async def get_seasonal_anime_standard(
        season: str = Field(..., description="Season (winter, spring, summer, fall)"),
        year: int = Field(..., ge=1900, le=2030, description="Year"),
        limit: int = Field(default=20, ge=1, le=50, description="Number of results"),
        sort_by: str = Field(default="popularity", description="Sort by (score, popularity, members)")
    ) -> StandardSearchResponse:
        """
        Get standard seasonal anime information with enhanced details.
        
        Returns current/upcoming seasonal anime with status, episode info,
        and studio details.
        """
        start_time = datetime.now()
        
        try:
            # Use Jikan for seasonal data
            raw_results = await jikan_client.get_seasonal_anime(season, year)
            
            # Transform to standard results
            results = []
            for raw_result in raw_results.get("data", []):
                standard_result = _transform_jikan_to_standard(raw_result)
                results.append(standard_result)
            
            # Sort results
            if sort_by == "score":
                results.sort(key=lambda x: x.score or 0, reverse=True)
            elif sort_by == "popularity":
                results.sort(key=lambda x: x.popularity or 999999)
            elif sort_by == "members":
                # For simplicity, use popularity as proxy
                results.sort(key=lambda x: x.popularity or 999999)
            
            # Limit results
            results = results[:limit]
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return StandardSearchResponse(
                results=results,
                total=len(results),
                query=f"{season.title()} {year} anime",
                processing_time_ms=int(processing_time),
                filters_applied={"season": season, "year": year, "sort_by": sort_by}
            )
            
        except Exception as e:
            logger.error(f"Standard seasonal search failed: {e}")
            return StandardSearchResponse(
                results=[],
                total=0,
                query=f"{season.title()} {year} anime",
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                filters_applied={}
            )
    
    @mcp.tool()
    async def search_by_genre_standard(
        genres: List[str] = Field(..., description="List of genres to search for"),
        genre_mode: str = Field(default="any", description="Match mode: 'any' or 'all'"),
        limit: int = Field(default=20, ge=1, le=50, description="Number of results"),
        min_score: Optional[float] = Field(None, ge=0, le=10, description="Minimum score filter")
    ) -> StandardSearchResponse:
        """
        Search anime by genres with standard response detail.
        
        Supports both 'any' and 'all' genre matching modes.
        """
        start_time = datetime.now()
        
        try:
            # Build genre search parameters
            search_params = {
                "limit": limit,
                "genres": ",".join(genres),
                "order_by": "score",
                "sort": "desc"
            }
            
            if min_score:
                search_params["min_score"] = min_score
            
            # Execute search
            raw_results = await jikan_client.search_anime(**search_params)
            
            # Transform results
            results = []
            for raw_result in raw_results.get("data", []):
                standard_result = _transform_jikan_to_standard(raw_result)
                
                # Filter by genre mode
                if genre_mode == "all":
                    # Check if all requested genres are present
                    result_genres = [g.lower() for g in standard_result.genres]
                    if not all(genre.lower() in result_genres for genre in genres):
                        continue
                
                results.append(standard_result)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return StandardSearchResponse(
                results=results,
                total=len(results),
                query=f"Anime with genres: {', '.join(genres)}",
                processing_time_ms=int(processing_time),
                filters_applied={"genres": genres, "genre_mode": genre_mode}
            )
            
        except Exception as e:
            logger.error(f"Standard genre search failed: {e}")
            return StandardSearchResponse(
                results=[],
                total=0,
                query=f"Anime with genres: {', '.join(genres)}",
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                filters_applied={}
            )


def _transform_jikan_to_standard(raw_data: Dict[str, Any]) -> StandardAnimeResult:
    """Transform Jikan API response to StandardAnimeResult."""
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
    # Map status
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
    
    # Map rating
    rating = None
    jikan_rating = raw_data.get("rating")
    if jikan_rating:
        try:
            # Convert Jikan rating to our enum
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
    
    # Extract aired from
    aired_from = None
    aired = raw_data.get("aired", {})
    if isinstance(aired, dict):
        aired_from = aired.get("from")
    
    return StandardAnimeResult(
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
        aired_from=aired_from
    )


def _transform_mal_to_standard(raw_data: Dict[str, Any]) -> StandardAnimeResult:
    """Transform MAL API response to StandardAnimeResult."""
    # Basic fields
    anime_id = str(raw_data.get("id", ""))
    title = raw_data.get("title") or "Unknown"
    score = raw_data.get("mean")
    year = None
    
    # Extract year from start_date
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
    # Map status
    status = None
    mal_status = raw_data.get("status")
    if mal_status:
        try:
            status = AnimeStatus(mal_status.upper().replace(" ", "_"))
        except ValueError:
            status = AnimeStatus.UNKNOWN
    
    episodes = raw_data.get("num_episodes")
    duration = None  # MAL doesn't provide duration in standard format
    
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
    aired_from = start_date
    
    return StandardAnimeResult(
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
        aired_from=aired_from
    )


def _transform_anilist_to_standard(raw_data: Dict[str, Any]) -> StandardAnimeResult:
    """Transform AniList API response to StandardAnimeResult."""
    # Basic fields
    anime_id = str(raw_data.get("id", ""))
    title = raw_data.get("title", {}).get("romaji") or raw_data.get("title", {}).get("english") or "Unknown"
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
    # Map status
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
    
    # Rating (AniList doesn't have content rating)
    rating = None
    
    popularity = raw_data.get("popularity")
    
    # Extract aired from
    aired_from = None
    if isinstance(start_date, dict):
        year = start_date.get("year")
        month = start_date.get("month")
        day = start_date.get("day")
        if year and month and day:
            aired_from = f"{year}-{month:02d}-{day:02d}"
    
    return StandardAnimeResult(
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
        aired_from=aired_from
    )


def _transform_kitsu_to_standard(raw_data: Dict[str, Any]) -> StandardAnimeResult:
    """Transform Kitsu API response to StandardAnimeResult."""
    # Basic fields
    anime_id = str(raw_data.get("id", ""))
    
    # Extract attributes
    attributes = raw_data.get("attributes", {})
    title = attributes.get("canonicalTitle") or attributes.get("titles", {}).get("en") or "Unknown"
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
    
    # Extract genres (simplified for standard tier)
    genres = []
    
    # Extract image URL
    image_url = None
    poster_image = attributes.get("posterImage", {})
    if isinstance(poster_image, dict):
        image_url = poster_image.get("medium") or poster_image.get("small")
    
    # Extract synopsis
    synopsis = attributes.get("synopsis") or ""
    
    # Standard fields
    # Map status
    status = None
    kitsu_status = attributes.get("status")
    if kitsu_status:
        try:
            status = AnimeStatus(kitsu_status.upper().replace(" ", "_"))
        except ValueError:
            status = AnimeStatus.UNKNOWN
    
    episodes = attributes.get("episodeCount")
    duration = attributes.get("episodeLength")
    
    # Studios (simplified)
    studios = []
    
    # Rating (Kitsu uses age rating)
    rating = None
    age_rating = attributes.get("ageRating")
    if age_rating:
        try:
            rating = AnimeRating(age_rating.upper().replace("-", "_"))
        except ValueError:
            rating = None
    
    popularity = attributes.get("popularityRank")
    aired_from = start_date
    
    return StandardAnimeResult(
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
        aired_from=aired_from
    )


def _calculate_similarity_score(
    reference: StandardAnimeResult,
    candidate: StandardAnimeResult,
    factors: List[str]
) -> float:
    """Calculate similarity score between two anime based on specified factors."""
    total_score = 0.0
    total_weight = 0.0
    
    # Genre similarity
    if "genres" in factors and reference.genres and candidate.genres:
        ref_genres = set(g.lower() for g in reference.genres)
        cand_genres = set(g.lower() for g in candidate.genres)
        if ref_genres and cand_genres:
            intersection = len(ref_genres & cand_genres)
            union = len(ref_genres | cand_genres)
            genre_score = intersection / union if union > 0 else 0
            total_score += genre_score * 0.4  # 40% weight for genres
            total_weight += 0.4
    
    # Type similarity
    if "type" in factors and reference.type and candidate.type:
        type_score = 1.0 if reference.type == candidate.type else 0.0
        total_score += type_score * 0.2  # 20% weight for type
        total_weight += 0.2
    
    # Studio similarity
    if "studios" in factors and reference.studios and candidate.studios:
        ref_studios = set(s.lower() for s in reference.studios)
        cand_studios = set(s.lower() for s in candidate.studios)
        if ref_studios and cand_studios:
            intersection = len(ref_studios & cand_studios)
            union = len(ref_studios | cand_studios)
            studio_score = intersection / union if union > 0 else 0
            total_score += studio_score * 0.3  # 30% weight for studios
            total_weight += 0.3
    
    # Score similarity
    if reference.score and candidate.score:
        score_diff = abs(reference.score - candidate.score)
        score_similarity = max(0, 1 - (score_diff / 10))  # Normalize to 0-1
        total_score += score_similarity * 0.1  # 10% weight for score
        total_weight += 0.1
    
    return total_score / total_weight if total_weight > 0 else 0.0