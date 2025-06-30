"""AnimeCountdown API endpoints for external service integration."""

import logging
from typing import List, Optional
from enum import Enum

from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field

from ...services.external.animecountdown_service import AnimeCountdownService

logger = logging.getLogger(__name__)

# Initialize service
animecountdown_service = AnimeCountdownService()

router = APIRouter()


class TimePeriod(str, Enum):
    """Valid time periods for popular anime."""
    ALL_TIME = "all_time"
    THIS_YEAR = "this_year"
    THIS_MONTH = "this_month"
    THIS_WEEK = "this_week"


class AnimeCountdownSearchResponse(BaseModel):
    """Response model for anime search results."""
    results: List[dict] = Field(..., description="List of anime search results")
    total: int = Field(..., description="Total number of results")
    query: str = Field(..., description="Search query used")


class AnimeCountdownListResponse(BaseModel):
    """Response model for anime list results."""
    results: List[dict] = Field(..., description="List of anime results")
    total: int = Field(..., description="Total number of results")
    limit: int = Field(..., description="Limit applied to results")


@router.get(
    "/external/animecountdown/search",
    response_model=AnimeCountdownSearchResponse,
    summary="Search anime on AnimeCountdown",
    description="Search for anime using AnimeCountdown's database with countdown information."
)
async def search_anime(
    q: str = Query(..., description="Search query", min_length=1, max_length=200)
):
    """Search for anime on AnimeCountdown."""
    try:
        logger.info("AnimeCountdown search request: query='%s'", q)
        results = await animecountdown_service.search_anime(q)
        
        return AnimeCountdownSearchResponse(
            results=results,
            total=len(results),
            query=q
        )
        
    except ValueError as e:
        logger.warning("AnimeCountdown search validation error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("AnimeCountdown search failed: %s", e)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
    "/external/animecountdown/anime/{anime_id}",
    summary="Get anime details from AnimeCountdown",
    description="Get detailed information about a specific anime from AnimeCountdown."
)
async def get_anime_details(
    anime_id: str = Path(..., description="AnimeCountdown anime ID or slug", min_length=1)
):
    """Get anime details by ID."""
    try:
        logger.info("AnimeCountdown anime details request: anime_id='%s'", anime_id)
        result = await animecountdown_service.get_anime_details(anime_id)
        
        if result is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Anime with ID '{anime_id}' not found on AnimeCountdown"
            )
        
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("AnimeCountdown anime details validation error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("AnimeCountdown anime details failed: %s", e)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
    "/external/animecountdown/currently-airing",
    response_model=AnimeCountdownListResponse,
    summary="Get currently airing anime",
    description="Get currently airing anime with countdown information from AnimeCountdown."
)
async def get_currently_airing(
    limit: int = Query(25, description="Maximum number of results (1-100)", ge=1, le=100)
):
    """Get currently airing anime with countdown information."""
    try:
        logger.info("AnimeCountdown currently airing request: limit=%d", limit)
        results = await animecountdown_service.get_currently_airing(limit)
        
        return AnimeCountdownListResponse(
            results=results,
            total=len(results),
            limit=limit
        )
        
    except ValueError as e:
        logger.warning("AnimeCountdown currently airing validation error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("AnimeCountdown currently airing failed: %s", e)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
    "/external/animecountdown/upcoming",
    response_model=AnimeCountdownListResponse,
    summary="Get upcoming anime",
    description="Get upcoming anime with countdown information from AnimeCountdown."
)
async def get_upcoming_anime(
    limit: int = Query(20, description="Maximum number of results (1-100)", ge=1, le=100)
):
    """Get upcoming anime with countdown information."""
    try:
        logger.info("AnimeCountdown upcoming anime request: limit=%d", limit)
        results = await animecountdown_service.get_upcoming_anime(limit)
        
        return AnimeCountdownListResponse(
            results=results,
            total=len(results),
            limit=limit
        )
        
    except ValueError as e:
        logger.warning("AnimeCountdown upcoming anime validation error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("AnimeCountdown upcoming anime failed: %s", e)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
    "/external/animecountdown/popular",
    response_model=AnimeCountdownListResponse,
    summary="Get popular anime",
    description="Get popular anime from AnimeCountdown based on views and popularity metrics."
)
async def get_popular_anime(
    time_period: TimePeriod = Query(
        TimePeriod.ALL_TIME, 
        description="Time period for popularity ranking"
    ),
    limit: int = Query(25, description="Maximum number of results (1-100)", ge=1, le=100)
):
    """Get popular anime from AnimeCountdown."""
    try:
        logger.info("AnimeCountdown popular anime request: time_period='%s', limit=%d", time_period, limit)
        results = await animecountdown_service.get_popular_anime(time_period.value, limit)
        
        return AnimeCountdownListResponse(
            results=results,
            total=len(results),
            limit=limit
        )
        
    except ValueError as e:
        logger.warning("AnimeCountdown popular anime validation error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("AnimeCountdown popular anime failed: %s", e)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
    "/external/animecountdown/countdown/{anime_id}",
    summary="Get anime countdown information",
    description="Get specific countdown information for an anime from AnimeCountdown."
)
async def get_anime_countdown(
    anime_id: str = Path(..., description="AnimeCountdown anime ID or slug", min_length=1)
):
    """Get specific countdown information for an anime."""
    try:
        logger.info("AnimeCountdown countdown request: anime_id='%s'", anime_id)
        result = await animecountdown_service.get_anime_countdown(anime_id)
        
        if result is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Countdown information for anime '{anime_id}' not found on AnimeCountdown"
            )
        
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("AnimeCountdown countdown validation error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("AnimeCountdown countdown failed: %s", e)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
    "/external/animecountdown/health",
    summary="AnimeCountdown service health check",
    description="Check the health status of the AnimeCountdown service integration."
)
async def health_check():
    """Check AnimeCountdown service health."""
    try:
        logger.info("AnimeCountdown health check request")
        result = await animecountdown_service.health_check()
        
        if result.get("status") == "healthy":
            return result
        else:
            raise HTTPException(status_code=503, detail=result)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("AnimeCountdown health check failed: %s", e)
        raise HTTPException(status_code=500, detail={"error": str(e)})