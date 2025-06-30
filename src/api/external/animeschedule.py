"""AnimeSchedule.net API endpoints for external service integration."""

import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query, Path

from ...services.external.animeschedule_service import AnimeScheduleService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/external/animeschedule", tags=["External APIs", "AnimeSchedule"])

# Global service instance
_animeschedule_service = AnimeScheduleService()


@router.get("/today")
async def get_today_timetable(
    timezone: Optional[str] = Query(None, description="Timezone for schedule (e.g., 'Asia/Tokyo')"),
    region: Optional[str] = Query(None, description="Region code (e.g., 'JP', 'US')")
) -> Dict[str, Any]:
    """Get today's anime timetable from AnimeSchedule.net.
    
    Args:
        timezone: Timezone for schedule calculation
        region: Region code for localized results
        
    Returns:
        Today's anime timetable data
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        logger.info("AnimeSchedule today timetable: timezone=%s, region=%s", timezone, region)
        
        timetable_data = await _animeschedule_service.get_today_timetable(
            timezone=timezone, region=region
        )
        
        return {
            "source": "animeschedule",
            "type": "today_timetable",
            "timezone": timezone,
            "region": region,
            "data": timetable_data
        }
        
    except Exception as e:
        logger.error("AnimeSchedule today timetable failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AnimeSchedule today timetable service unavailable: {str(e)}"
        )


@router.get("/timetable/{date}")
async def get_timetable_by_date(
    date: str = Path(..., description="Date in YYYY-MM-DD format")
) -> Dict[str, Any]:
    """Get anime timetable for a specific date from AnimeSchedule.net.
    
    Args:
        date: Date in YYYY-MM-DD format
        
    Returns:
        Anime timetable data for the specified date
        
    Raises:
        HTTPException: If date format is invalid or service is unavailable
    """
    try:
        logger.info("AnimeSchedule timetable by date: date=%s", date)
        
        timetable_data = await _animeschedule_service.get_timetable_by_date(date)
        
        return {
            "source": "animeschedule",
            "type": "date_timetable",
            "date": date,
            "data": timetable_data
        }
        
    except ValueError as e:
        logger.warning("AnimeSchedule invalid date format: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("AnimeSchedule timetable by date failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AnimeSchedule timetable service unavailable: {str(e)}"
        )


@router.get("/search")
async def search_anime(
    q: str = Query(..., description="Search query string")
) -> Dict[str, Any]:
    """Search for anime on AnimeSchedule.net.
    
    Args:
        q: Search query string
        
    Returns:
        Anime search results
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        logger.info("AnimeSchedule search: query='%s'", q)
        
        search_results = await _animeschedule_service.search_anime(q)
        
        return {
            "source": "animeschedule",
            "query": q,
            "results": search_results,
            "total_results": len(search_results)
        }
        
    except Exception as e:
        logger.error("AnimeSchedule search failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AnimeSchedule search service unavailable: {str(e)}"
        )


@router.get("/seasonal/{season}/{year}")
async def get_seasonal_anime(
    season: str = Path(..., description="Season (winter, spring, summer, fall)"),
    year: int = Path(..., description="Year (2000-2025)", ge=2000, le=2025)
) -> Dict[str, Any]:
    """Get seasonal anime list from AnimeSchedule.net.
    
    Args:
        season: Season name (winter, spring, summer, fall)
        year: Year (must be between 2000 and 2025)
        
    Returns:
        Seasonal anime data
        
    Raises:
        HTTPException: If season/year is invalid or service is unavailable
    """
    try:
        logger.info("AnimeSchedule seasonal: season=%s, year=%d", season, year)
        
        seasonal_data = await _animeschedule_service.get_seasonal_anime(season, year)
        
        return {
            "source": "animeschedule",
            "season": season,
            "year": year,
            "data": seasonal_data
        }
        
    except ValueError as e:
        logger.warning("AnimeSchedule invalid season/year: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("AnimeSchedule seasonal failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AnimeSchedule seasonal service unavailable: {str(e)}"
        )


@router.get("/anime/{anime_id}/schedule")
async def get_anime_schedule(
    anime_id: int = Path(..., description="AnimeSchedule anime ID", gt=0)
) -> Dict[str, Any]:
    """Get anime schedule information by ID from AnimeSchedule.net.
    
    Args:
        anime_id: AnimeSchedule anime ID
        
    Returns:
        Anime schedule information
        
    Raises:
        HTTPException: If anime not found or service is unavailable
    """
    try:
        logger.info("AnimeSchedule anime schedule: anime_id=%d", anime_id)
        
        schedule_data = await _animeschedule_service.get_anime_schedule(anime_id)
        
        if schedule_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Anime schedule for ID {anime_id} not found on AnimeSchedule.net"
            )
        
        return {
            "source": "animeschedule",
            "anime_id": anime_id,
            "schedule": schedule_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("AnimeSchedule anime schedule failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AnimeSchedule service unavailable: {str(e)}"
        )


@router.get("/platforms")
async def get_streaming_platforms() -> Dict[str, Any]:
    """Get available streaming platforms from AnimeSchedule.net.
    
    Returns:
        List of streaming platforms
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        logger.info("AnimeSchedule streaming platforms")
        
        platforms_data = await _animeschedule_service.get_streaming_platforms()
        
        return {
            "source": "animeschedule",
            "platforms": platforms_data,
            "total_platforms": len(platforms_data)
        }
        
    except Exception as e:
        logger.error("AnimeSchedule platforms failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AnimeSchedule platforms service unavailable: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check AnimeSchedule.net service health status.
    
    Returns:
        Service health information
    """
    try:
        logger.debug("AnimeSchedule health check")
        health_data = await _animeschedule_service.health_check()
        return health_data
        
    except Exception as e:
        logger.error("AnimeSchedule health check failed: %s", e)
        return {
            "service": "animeschedule",
            "status": "unhealthy",
            "error": str(e),
            "circuit_breaker_open": True
        }