"""Anime-Planet API endpoints for external service integration."""

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Query, Path

from ...services.external.animeplanet_service import AnimePlanetService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/external/animeplanet", tags=["External APIs", "Anime-Planet"])

# Global service instance
_animeplanet_service = AnimePlanetService()


@router.get("/search")
async def search_anime(
    q: str = Query(..., description="Search query string"),
    limit: int = Query(20, description="Maximum number of results", ge=1, le=50)
) -> Dict[str, Any]:
    """Search for anime on Anime-Planet.
    
    Args:
        q: Search query string
        limit: Maximum number of results (1-50)
        
    Returns:
        Anime search results
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        logger.info("Anime-Planet search: query='%s', limit=%d", q, limit)
        
        search_results = await _animeplanet_service.search_anime(q)
        
        return {
            "source": "animeplanet",
            "query": q,
            "limit": limit,
            "results": search_results,
            "total_results": len(search_results)
        }
        
    except ValueError as e:
        logger.warning("Anime-Planet search invalid input: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Anime-Planet search failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"Anime-Planet search service unavailable: {str(e)}"
        )


@router.get("/anime/{anime_id}")
async def get_anime_details(
    anime_id: str = Path(..., description="Anime-Planet anime ID/slug")
) -> Dict[str, Any]:
    """Get anime details by ID from Anime-Planet.
    
    Args:
        anime_id: Anime-Planet anime ID/slug
        
    Returns:
        Anime details
        
    Raises:
        HTTPException: If anime not found or service is unavailable
    """
    try:
        logger.info("Anime-Planet anime details: anime_id='%s'", anime_id)
        
        anime_data = await _animeplanet_service.get_anime_details(anime_id)
        
        if anime_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Anime with ID {anime_id} not found on Anime-Planet"
            )
        
        return {
            "source": "animeplanet",
            "anime_id": anime_id,
            "data": anime_data
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Anime-Planet anime details invalid input: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Anime-Planet anime details failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"Anime-Planet service unavailable: {str(e)}"
        )


@router.get("/anime/{anime_id}/characters")
async def get_anime_characters(
    anime_id: str = Path(..., description="Anime-Planet anime ID/slug")
) -> Dict[str, Any]:
    """Get anime characters by anime ID from Anime-Planet.
    
    Args:
        anime_id: Anime-Planet anime ID/slug
        
    Returns:
        Anime characters data
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        logger.info("Anime-Planet anime characters: anime_id='%s'", anime_id)
        
        characters_data = await _animeplanet_service.get_anime_characters(anime_id)
        
        return {
            "source": "animeplanet",
            "anime_id": anime_id,
            "characters": characters_data,
            "total_characters": len(characters_data)
        }
        
    except ValueError as e:
        logger.warning("Anime-Planet anime characters invalid input: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Anime-Planet anime characters failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"Anime-Planet characters service unavailable: {str(e)}"
        )


@router.get("/anime/{anime_id}/recommendations")
async def get_anime_recommendations(
    anime_id: str = Path(..., description="Anime-Planet anime ID/slug"),
    limit: int = Query(10, description="Maximum number of recommendations", ge=1, le=50)
) -> Dict[str, Any]:
    """Get anime recommendations from Anime-Planet.
    
    Args:
        anime_id: Anime-Planet anime ID/slug
        limit: Maximum number of recommendations (1-50)
        
    Returns:
        Anime recommendations
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        logger.info("Anime-Planet recommendations: anime_id='%s', limit=%d", anime_id, limit)
        
        recommendations = await _animeplanet_service.get_anime_recommendations(anime_id, limit)
        
        return {
            "source": "animeplanet",
            "anime_id": anime_id,
            "limit": limit,
            "recommendations": recommendations,
            "total_results": len(recommendations)
        }
        
    except ValueError as e:
        logger.warning("Anime-Planet recommendations invalid input: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Anime-Planet recommendations failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"Anime-Planet recommendations service unavailable: {str(e)}"
        )


@router.get("/top")
async def get_top_anime(
    category: str = Query("top-anime", description="Category (top-anime, most-watched, highest-rated, etc.)"),
    limit: int = Query(25, description="Maximum number of results", ge=1, le=100)
) -> Dict[str, Any]:
    """Get top anime from Anime-Planet.
    
    Args:
        category: Category type (top-anime, most-watched, highest-rated, etc.)
        limit: Maximum number of results (1-100)
        
    Returns:
        Top anime data
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        logger.info("Anime-Planet top anime: category='%s', limit=%d", category, limit)
        
        top_anime = await _animeplanet_service.get_top_anime(category, limit)
        
        return {
            "source": "animeplanet",
            "category": category,
            "limit": limit,
            "anime": top_anime,
            "total_results": len(top_anime)
        }
        
    except ValueError as e:
        logger.warning("Anime-Planet top anime invalid input: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Anime-Planet top anime failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"Anime-Planet top anime service unavailable: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check Anime-Planet service health status.
    
    Returns:
        Service health information
    """
    try:
        logger.debug("Anime-Planet health check")
        health_data = await _animeplanet_service.health_check()
        return health_data
        
    except Exception as e:
        logger.error("Anime-Planet health check failed: %s", e)
        return {
            "service": "animeplanet",
            "status": "unhealthy",
            "error": str(e),
            "circuit_breaker_open": True
        }