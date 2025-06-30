"""MAL API endpoints for external anime data."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import Field, field_validator

from ...services.external.mal_service import MALService
from ...config import get_settings

logger = logging.getLogger(__name__)

# Create router for MAL endpoints
router = APIRouter(prefix="/external/mal", tags=["External APIs", "MAL"])

# Get settings for MAL configuration
settings = get_settings()

# Initialize MAL service (singleton pattern)
_mal_service = MALService(
    client_id=settings.mal_client_id,
    client_secret=settings.mal_client_secret
)


@router.get("/search")
async def search_anime(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    status: Optional[str] = Query(None, pattern="^(airing|complete|upcoming)$", description="Anime status filter"),
    genres: Optional[str] = Query(None, description="Comma-separated genre IDs (e.g., '1,2,3')")
) -> Dict[str, Any]:
    """Search for anime on MAL/Jikan.
    
    Args:
        q: Search query string
        limit: Maximum number of results (1-50)
        status: Anime status filter (airing, complete, upcoming)
        genres: Comma-separated genre IDs
        
    Returns:
        Search results with metadata
        
    Raises:
        HTTPException: If search service fails
    """
    try:
        # Parse genres if provided
        genre_list = None
        if genres:
            try:
                genre_list = [int(g.strip()) for g in genres.split(",") if g.strip()]
            except ValueError:
                raise HTTPException(status_code=422, detail="Invalid genre format. Use comma-separated integers.")
        
        logger.info("MAL search: query='%s', limit=%d, status=%s, genres=%s", q, limit, status, genre_list)
        
        results = await _mal_service.search_anime(
            query=q,
            limit=limit,
            status=status,
            genres=genre_list
        )
        
        return {
            "source": "mal",
            "query": q,
            "limit": limit,
            "status": status,
            "genres": genre_list,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error("MAL search failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"MAL search service unavailable: {str(e)}"
        )


@router.get("/anime/{anime_id}")
async def get_anime_details(
    anime_id: int = Path(..., ge=1, description="MAL anime ID")
) -> Dict[str, Any]:
    """Get detailed anime information by MAL ID.
    
    Args:
        anime_id: MAL anime ID
        
    Returns:
        Detailed anime information
        
    Raises:
        HTTPException: If anime not found or service fails
    """
    try:
        logger.info("MAL anime details: anime_id=%d", anime_id)
        
        anime_data = await _mal_service.get_anime_details(anime_id)
        
        if not anime_data:
            raise HTTPException(
                status_code=404,
                detail=f"Anime with ID {anime_id} not found on MAL"
            )
        
        return {
            "source": "mal",
            "anime_id": anime_id,
            "data": anime_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("MAL anime details failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"MAL service unavailable: {str(e)}"
        )


@router.get("/seasonal/{year}/{season}")
async def get_seasonal_anime(
    year: int = Path(..., ge=1990, le=2030, description="Year (1990-2030)"),
    season: str = Path(..., pattern="^(winter|spring|summer|fall)$", description="Season (winter, spring, summer, fall)")
) -> Dict[str, Any]:
    """Get seasonal anime for a specific year and season.
    
    Args:
        year: Year (1990-2030)
        season: Season (winter, spring, summer, fall)
        
    Returns:
        Seasonal anime list
        
    Raises:
        HTTPException: If service fails
    """
    try:
        logger.info("MAL seasonal: year=%d, season=%s", year, season)
        
        results = await _mal_service.get_seasonal_anime(year, season)
        
        return {
            "source": "mal",
            "year": year,
            "season": season,
            "results": results,
            "total_results": len(results)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("MAL seasonal anime failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"MAL service unavailable: {str(e)}"
        )


@router.get("/current-season")
async def get_current_season_anime() -> Dict[str, Any]:
    """Get current season anime.
    
    Returns:
        Current season anime list
        
    Raises:
        HTTPException: If service fails
    """
    try:
        logger.info("MAL current season anime")
        
        results = await _mal_service.get_current_season()
        
        return {
            "source": "mal",
            "type": "current_season",
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error("MAL current season failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"MAL service unavailable: {str(e)}"
        )


@router.get("/anime/{anime_id}/statistics")
async def get_anime_statistics(
    anime_id: int = Path(..., ge=1, description="MAL anime ID")
) -> Dict[str, Any]:
    """Get anime statistics (watching, completed, etc.).
    
    Args:
        anime_id: MAL anime ID
        
    Returns:
        Anime statistics
        
    Raises:
        HTTPException: If anime not found or service fails
    """
    try:
        logger.info("MAL anime statistics: anime_id=%d", anime_id)
        
        stats = await _mal_service.get_anime_statistics(anime_id)
        
        if not stats:
            raise HTTPException(
                status_code=404,
                detail=f"Statistics for anime ID {anime_id} not found on MAL"
            )
        
        return {
            "source": "mal",
            "anime_id": anime_id,
            "statistics": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("MAL anime statistics failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"MAL service unavailable: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check MAL service health.
    
    Returns:
        Service health status
    """
    try:
        logger.info("MAL health check")
        return await _mal_service.health_check()
        
    except Exception as e:
        logger.error("MAL health check failed: %s", e)
        return {
            "service": "mal",
            "status": "unhealthy",
            "error": str(e),
            "circuit_breaker_open": True
        }