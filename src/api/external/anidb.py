"""AniDB API endpoints for external service integration."""

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Query, Path

from ...services.external.anidb_service import AniDBService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/external/anidb", tags=["External APIs", "AniDB"])

# Global service instance
_anidb_service = AniDBService()


@router.get("/search")
async def search_anime(
    q: str = Query(..., description="Search query string"),
    limit: int = Query(20, description="Maximum number of results", ge=1, le=50)
) -> Dict[str, Any]:
    """Search for anime on AniDB.
    
    Args:
        q: Search query string
        limit: Maximum number of results (1-50)
        
    Returns:
        Anime search results
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        logger.info("AniDB search: query='%s', limit=%d", q, limit)
        
        search_results = await _anidb_service.search_anime(q)
        
        return {
            "source": "anidb",
            "query": q,
            "limit": limit,
            "results": search_results,
            "total_results": len(search_results)
        }
        
    except ValueError as e:
        logger.warning("AniDB search invalid input: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("AniDB search failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AniDB search service unavailable: {str(e)}"
        )


@router.get("/anime/{anime_id}")
async def get_anime_details(
    anime_id: int = Path(..., description="AniDB anime ID", gt=0)
) -> Dict[str, Any]:
    """Get anime details by ID from AniDB.
    
    Args:
        anime_id: AniDB anime ID
        
    Returns:
        Anime details
        
    Raises:
        HTTPException: If anime not found or service is unavailable
    """
    try:
        logger.info("AniDB anime details: anime_id=%d", anime_id)
        
        anime_data = await _anidb_service.get_anime_details(anime_id)
        
        if anime_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Anime with ID {anime_id} not found on AniDB"
            )
        
        return {
            "source": "anidb",
            "anime_id": anime_id,
            "data": anime_data
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("AniDB anime details invalid input: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("AniDB anime details failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AniDB service unavailable: {str(e)}"
        )


@router.get("/anime/{anime_id}/characters")
async def get_anime_characters(
    anime_id: int = Path(..., description="AniDB anime ID", gt=0)
) -> Dict[str, Any]:
    """Get anime characters by anime ID from AniDB.
    
    Args:
        anime_id: AniDB anime ID
        
    Returns:
        Anime characters data
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        logger.info("AniDB anime characters: anime_id=%d", anime_id)
        
        characters_data = await _anidb_service.get_anime_characters(anime_id)
        
        return {
            "source": "anidb",
            "anime_id": anime_id,
            "characters": characters_data,
            "total_characters": len(characters_data)
        }
        
    except ValueError as e:
        logger.warning("AniDB anime characters invalid input: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("AniDB anime characters failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AniDB characters service unavailable: {str(e)}"
        )


@router.get("/anime/{anime_id}/episodes")
async def get_anime_episodes(
    anime_id: int = Path(..., description="AniDB anime ID", gt=0)
) -> Dict[str, Any]:
    """Get anime episodes by anime ID from AniDB.
    
    Args:
        anime_id: AniDB anime ID
        
    Returns:
        Anime episodes data
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        logger.info("AniDB anime episodes: anime_id=%d", anime_id)
        
        episodes_data = await _anidb_service.get_anime_episodes(anime_id)
        
        return {
            "source": "anidb",
            "anime_id": anime_id,
            "episodes": episodes_data,
            "total_episodes": len(episodes_data)
        }
        
    except ValueError as e:
        logger.warning("AniDB anime episodes invalid input: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("AniDB anime episodes failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AniDB episodes service unavailable: {str(e)}"
        )


@router.get("/anime/{anime_id}/similar")
async def get_similar_anime(
    anime_id: int = Path(..., description="AniDB anime ID", gt=0),
    limit: int = Query(10, description="Maximum number of recommendations", ge=1, le=50)
) -> Dict[str, Any]:
    """Get similar anime recommendations from AniDB.
    
    Args:
        anime_id: AniDB anime ID
        limit: Maximum number of recommendations (1-50)
        
    Returns:
        Similar anime recommendations
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        logger.info("AniDB similar anime: anime_id=%d, limit=%d", anime_id, limit)
        
        similar_anime = await _anidb_service.get_similar_anime(anime_id, limit)
        
        return {
            "source": "anidb",
            "anime_id": anime_id,
            "limit": limit,
            "similar_anime": similar_anime,
            "total_results": len(similar_anime)
        }
        
    except ValueError as e:
        logger.warning("AniDB similar anime invalid input: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("AniDB similar anime failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AniDB similar anime service unavailable: {str(e)}"
        )


@router.get("/random")
async def get_random_anime() -> Dict[str, Any]:
    """Get a random anime from AniDB.
    
    Returns:
        Random anime data
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        logger.info("AniDB random anime")
        
        random_anime = await _anidb_service.get_random_anime()
        
        return {
            "source": "anidb",
            "type": "random",
            "anime": random_anime
        }
        
    except Exception as e:
        logger.error("AniDB random anime failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AniDB random anime service unavailable: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check AniDB service health status.
    
    Returns:
        Service health information
    """
    try:
        logger.debug("AniDB health check")
        health_data = await _anidb_service.health_check()
        return health_data
        
    except Exception as e:
        logger.error("AniDB health check failed: %s", e)
        return {
            "service": "anidb",
            "status": "unhealthy",
            "error": str(e),
            "circuit_breaker_open": True
        }