"""
Tier 1 Basic Tools - Essential anime search functionality with minimal response complexity.

These tools provide the most commonly needed information (8 fields) covering 80% of user queries.
Optimized for speed and simplicity while maintaining full functionality.
"""

from typing import List, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from ..handlers.anime_handler import AnimeHandler
from ...models.structured_responses import (
    BasicAnimeResult,
    BasicSearchResponse,
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


class BasicSearchInput(BaseModel):
    """Input schema for basic anime search."""
    query: str = Field(..., description="Search query for anime titles")
    limit: int = Field(default=20, ge=1, le=50, description="Maximum number of results")
    year: Optional[int] = Field(None, ge=1900, le=2030, description="Release year filter")
    type: Optional[AnimeType] = Field(None, description="Anime type filter")
    status: Optional[AnimeStatus] = Field(None, description="Anime status filter")


class BasicAnimeDetailsInput(BaseModel):
    """Input schema for basic anime details."""
    id: str = Field(..., description="Anime ID from search results")
    platform: str = Field(default="jikan", description="Platform to fetch from (jikan, mal, anilist, kitsu)")


class BasicSimilarInput(BaseModel):
    """Input schema for basic similarity search."""
    anime_id: str = Field(..., description="Reference anime ID")
    limit: int = Field(default=10, ge=1, le=20, description="Number of similar anime to return")
    platform: str = Field(default="jikan", description="Platform to search from")


def register_basic_tools(mcp: FastMCP) -> None:
    """Register all Tier 1 basic tools with the MCP server."""
    from ...config import get_settings
    
    # Initialize clients
    settings = get_settings()
    jikan_client = JikanClient()
    
    # MAL client requires credentials - initialize safely
    mal_client = None
    try:
        if hasattr(settings, 'mal_client_id') and settings.mal_client_id:
            mal_client = MALClient(
                client_id=settings.mal_client_id,
                client_secret=getattr(settings, 'mal_client_secret', None)
            )
    except Exception as e:
        logger.warning(f"MAL client initialization failed: {e}")
    
    anilist_client = AniListClient()
    kitsu_client = KitsuClient()
    
    @mcp.tool()
    async def search_anime_basic(input: BasicSearchInput) -> BasicSearchResponse:
        """
        Basic anime search with essential information only.
        
        Returns the 8 most important fields covering 80% of user needs:
        - id, title, score, year, type, genres, image_url, synopsis
        
        Optimized for speed and simplicity.
        """
        start_time = datetime.now()
        
        try:
            # Build Jikan search parameters
            jikan_params = {}
            if input.query:
                jikan_params["q"] = input.query
            if input.limit:
                jikan_params["limit"] = min(input.limit, 25)  # Jikan limit
            if input.year:
                jikan_params["start_date"] = f"{input.year}-01-01"
                jikan_params["end_date"] = f"{input.year}-12-31"
            if input.type:
                jikan_params["type"] = input.type.value.lower()
            if input.status:
                jikan_params["status"] = input.status.value.lower()
            
            # Execute search
            raw_results = await jikan_client.search_anime(**jikan_params)
            
            # Transform to BasicAnimeResult
            results = []
            for raw_result in raw_results.get("data", []):
                # Extract basic information
                anime_id = str(raw_result.get("mal_id", ""))
                title = raw_result.get("title") or "Unknown"
                score = raw_result.get("score")
                year = raw_result.get("year")
                
                # Map type
                anime_type = None
                jikan_type = raw_result.get("type")
                if jikan_type:
                    try:
                        anime_type = AnimeType(jikan_type.upper())
                    except ValueError:
                        anime_type = AnimeType.UNKNOWN
                
                # Extract genres
                genres = []
                for genre in raw_result.get("genres", []):
                    if isinstance(genre, dict) and "name" in genre:
                        genres.append(genre["name"])
                
                # Extract image URL
                image_url = None
                images = raw_result.get("images", {})
                if isinstance(images, dict):
                    jpg_images = images.get("jpg", {})
                    if isinstance(jpg_images, dict):
                        image_url = jpg_images.get("image_url")
                
                # Extract synopsis
                synopsis = raw_result.get("synopsis") or ""
                if len(synopsis) > 200:  # Truncate for basic tier
                    synopsis = synopsis[:197] + "..."
                
                basic_result = BasicAnimeResult(
                    id=anime_id,
                    title=title,
                    score=score,
                    year=year,
                    type=anime_type,
                    genres=genres,
                    image_url=image_url,
                    synopsis=synopsis
                )
                results.append(basic_result)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return BasicSearchResponse(
                results=results,
                total=len(results),
                query=input.query,
                processing_time_ms=int(processing_time)
            )
            
        except Exception as e:
            logger.error(f"Basic search failed: {e}")
            return BasicSearchResponse(
                results=[],
                total=0,
                query=input.query,
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
    
    @mcp.tool()
    async def get_anime_basic(input: BasicAnimeDetailsInput) -> BasicAnimeResult:
        """
        Get basic anime details by ID.
        
        Returns essential information optimized for quick lookup.
        """
        try:
            if input.platform == "jikan":
                raw_data = await jikan_client.get_anime_by_id(int(input.id))
            elif input.platform == "mal":
                if not mal_client:
                    raise ValueError("MAL client not available - missing credentials")
                raw_data = await mal_client.get_anime_by_id(int(input.id))
            elif input.platform == "anilist":
                raw_data = await anilist_client.get_anime_by_id(int(input.id))
            elif input.platform == "kitsu":
                raw_data = await kitsu_client.get_anime_by_id(input.id)
            else:
                raise ValueError(f"Unsupported platform: {input.platform}")
            
            # Transform to BasicAnimeResult based on platform
            if input.platform == "jikan":
                return _transform_jikan_to_basic(raw_data)
            elif input.platform == "mal":
                return _transform_mal_to_basic(raw_data)
            elif input.platform == "anilist":
                return _transform_anilist_to_basic(raw_data)
            elif input.platform == "kitsu":
                return _transform_kitsu_to_basic(raw_data)
                
        except Exception as e:
            logger.error(f"Basic details fetch failed: {e}")
            return BasicAnimeResult(
                id=input.id,
                title="Error fetching details",
                score=None,
                year=None,
                type=None,
                genres=[],
                image_url=None,
                synopsis=f"Error: {str(e)}"
            )
    
    @mcp.tool()
    async def find_similar_anime_basic(input: BasicSimilarInput) -> BasicSearchResponse:
        """
        Find similar anime using basic similarity matching.
        
        Returns anime similar to the reference with basic information only.
        """
        start_time = datetime.now()
        
        try:
            # Get reference anime details first
            reference_anime = await get_anime_basic(BasicAnimeDetailsInput(
                id=input.anime_id,
                platform=input.platform
            ))
            
            # Use genres and type for similarity search
            similar_params = {
                "limit": input.limit,
                "type": reference_anime.type.value.lower() if reference_anime.type else None,
                "genres": reference_anime.genres[:3] if reference_anime.genres else None  # Top 3 genres
            }
            
            # Search for similar anime
            if input.platform == "jikan":
                raw_results = await jikan_client.search_anime(**{k: v for k, v in similar_params.items() if v})
            else:
                # Fallback to Jikan for similarity search
                raw_results = await jikan_client.search_anime(**{k: v for k, v in similar_params.items() if v})
            
            # Transform results
            results = []
            for raw_result in raw_results.get("data", []):
                # Skip the reference anime itself
                if str(raw_result.get("mal_id")) == input.anime_id:
                    continue
                    
                basic_result = _transform_jikan_to_basic(raw_result)
                results.append(basic_result)
                
                if len(results) >= input.limit:
                    break
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return BasicSearchResponse(
                results=results,
                total=len(results),
                query=f"Similar to {reference_anime.title}",
                processing_time_ms=int(processing_time)
            )
            
        except Exception as e:
            logger.error(f"Basic similarity search failed: {e}")
            return BasicSearchResponse(
                results=[],
                total=0,
                query=f"Similar to anime {input.anime_id}",
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
    
    @mcp.tool()
    async def get_seasonal_anime_basic(
        season: str = Field(..., description="Season (winter, spring, summer, fall)"),
        year: int = Field(..., ge=1900, le=2030, description="Year"),
        limit: int = Field(default=20, ge=1, le=50, description="Number of results")
    ) -> BasicSearchResponse:
        """
        Get basic seasonal anime information.
        
        Returns current/upcoming seasonal anime with essential details only.
        """
        start_time = datetime.now()
        
        try:
            # Use Jikan for seasonal data
            raw_results = await jikan_client.get_seasonal_anime(season, year)
            
            # Transform to basic results
            results = []
            for raw_result in raw_results.get("data", []):
                basic_result = _transform_jikan_to_basic(raw_result)
                results.append(basic_result)
                
                if len(results) >= limit:
                    break
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return BasicSearchResponse(
                results=results,
                total=len(results),
                query=f"{season.title()} {year} anime",
                processing_time_ms=int(processing_time)
            )
            
        except Exception as e:
            logger.error(f"Basic seasonal search failed: {e}")
            return BasicSearchResponse(
                results=[],
                total=0,
                query=f"{season.title()} {year} anime",
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )


def _transform_jikan_to_basic(raw_data: Dict[str, Any]) -> BasicAnimeResult:
    """Transform Jikan API response to BasicAnimeResult."""
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
    if len(synopsis) > 200:  # Truncate for basic tier
        synopsis = synopsis[:197] + "..."
    
    return BasicAnimeResult(
        id=anime_id,
        title=title,
        score=score,
        year=year,
        type=anime_type,
        genres=genres,
        image_url=image_url,
        synopsis=synopsis
    )


def _transform_mal_to_basic(raw_data: Dict[str, Any]) -> BasicAnimeResult:
    """Transform MAL API response to BasicAnimeResult."""
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
    if len(synopsis) > 200:  # Truncate for basic tier
        synopsis = synopsis[:197] + "..."
    
    return BasicAnimeResult(
        id=anime_id,
        title=title,
        score=score,
        year=year,
        type=anime_type,
        genres=genres,
        image_url=image_url,
        synopsis=synopsis
    )


def _transform_anilist_to_basic(raw_data: Dict[str, Any]) -> BasicAnimeResult:
    """Transform AniList API response to BasicAnimeResult."""
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
    if len(synopsis) > 200:  # Truncate for basic tier
        synopsis = synopsis[:197] + "..."
    
    return BasicAnimeResult(
        id=anime_id,
        title=title,
        score=score,
        year=year,
        type=anime_type,
        genres=genres,
        image_url=image_url,
        synopsis=synopsis
    )


def _transform_kitsu_to_basic(raw_data: Dict[str, Any]) -> BasicAnimeResult:
    """Transform Kitsu API response to BasicAnimeResult."""
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
    
    # Extract genres (from categories relationship)
    genres = []
    # Note: Kitsu genres require additional API calls, skip for basic tier
    
    # Extract image URL
    image_url = None
    poster_image = attributes.get("posterImage", {})
    if isinstance(poster_image, dict):
        image_url = poster_image.get("medium") or poster_image.get("small")
    
    # Extract synopsis
    synopsis = attributes.get("synopsis") or ""
    if len(synopsis) > 200:  # Truncate for basic tier
        synopsis = synopsis[:197] + "..."
    
    return BasicAnimeResult(
        id=anime_id,
        title=title,
        score=score,
        year=year,
        type=anime_type,
        genres=genres,
        image_url=image_url,
        synopsis=synopsis
    )