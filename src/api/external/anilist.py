"""AniList API integration endpoints."""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

from ...services.external.anilist_service import AniListService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/external/anilist", tags=["External APIs", "AniList"])

# Initialize service
_anilist_service = AniListService()


@router.get("/search")
async def search_anime(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Number of results to return"),
    page: int = Query(1, ge=1, description="Page number")
) -> Dict[str, Any]:
    """Search for anime on AniList.
    
    Args:
        q: Search query string
        limit: Maximum number of results (1-50)
        page: Page number for pagination
        
    Returns:
        Search results from AniList
        
    Raises:
        HTTPException: If search fails or service is unavailable
    """
    try:
        results = await _anilist_service.search_anime(
            query=q,
            limit=limit,
            page=page
        )
        
        return {
            "source": "anilist",
            "query": q,
            "page": page,
            "limit": limit,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error("AniList search failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AniList search service unavailable: {str(e)}"
        )


@router.get("/anime/{anime_id}")
async def get_anime_details(anime_id: int) -> Dict[str, Any]:
    """Get detailed anime information by ID."""
    try:
        result = await _anilist_service.get_anime_details(anime_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Anime with ID {anime_id} not found on AniList"
            )
        
        return {
            "source": "anilist",
            "anime_id": anime_id,
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("AniList anime details failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AniList service unavailable: {str(e)}"
        )


@router.get("/trending")
async def get_trending_anime(
    limit: int = Query(20, ge=1, le=50),
    page: int = Query(1, ge=1)
) -> Dict[str, Any]:
    """Get trending anime from AniList."""
    try:
        results = await _anilist_service.get_trending_anime(limit=limit, page=page)
        
        return {
            "source": "anilist",
            "type": "trending",
            "page": page,
            "limit": limit,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error("AniList trending failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AniList trending service unavailable: {str(e)}"
        )


@router.get("/upcoming")
async def get_upcoming_anime(
    limit: int = Query(20, ge=1, le=50),
    page: int = Query(1, ge=1)
) -> Dict[str, Any]:
    """Get upcoming anime from AniList."""
    try:
        results = await _anilist_service.get_upcoming_anime(limit=limit, page=page)
        
        return {
            "source": "anilist",
            "type": "upcoming",
            "page": page,
            "limit": limit,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error("AniList upcoming failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AniList upcoming service unavailable: {str(e)}"
        )


@router.get("/popular")
async def get_popular_anime(
    limit: int = Query(20, ge=1, le=50),
    page: int = Query(1, ge=1)
) -> Dict[str, Any]:
    """Get popular anime from AniList."""
    try:
        results = await _anilist_service.get_popular_anime(limit=limit, page=page)
        
        return {
            "source": "anilist",
            "type": "popular",
            "page": page,
            "limit": limit,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error("AniList popular failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AniList popular service unavailable: {str(e)}"
        )


@router.get("/staff/{staff_id}")
async def get_staff_details(staff_id: int) -> Dict[str, Any]:
    """Get staff member details from AniList."""
    try:
        result = await _anilist_service.get_staff_details(staff_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Staff with ID {staff_id} not found on AniList"
            )
        
        return {
            "source": "anilist",
            "staff_id": staff_id,
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("AniList staff details failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AniList service unavailable: {str(e)}"
        )


@router.get("/studio/{studio_id}")
async def get_studio_details(studio_id: int) -> Dict[str, Any]:
    """Get studio details from AniList."""
    try:
        result = await _anilist_service.get_studio_details(studio_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Studio with ID {studio_id} not found on AniList"
            )
        
        return {
            "source": "anilist",
            "studio_id": studio_id,
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("AniList studio details failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AniList service unavailable: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check AniList service health."""
    return await _anilist_service.health_check()