"""Kitsu API endpoints for external anime data."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Path

from ...services.external.kitsu_service import KitsuService

logger = logging.getLogger(__name__)

# Create router for Kitsu endpoints
router = APIRouter(prefix="/external/kitsu", tags=["External APIs", "Kitsu"])

# Initialize Kitsu service (singleton pattern)
_kitsu_service = KitsuService()


@router.get("/search")
async def search_anime(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results")
) -> Dict[str, Any]:
    """Search for anime on Kitsu.
    
    Args:
        q: Search query string
        limit: Maximum number of results (1-50)
        
    Returns:
        Search results with metadata
        
    Raises:
        HTTPException: If search service fails
    """
    try:
        logger.info("Kitsu search: query='%s', limit=%d", q, limit)
        
        results = await _kitsu_service.search_anime(
            query=q,
            limit=limit
        )
        
        return {
            "source": "kitsu",
            "query": q,
            "limit": limit,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error("Kitsu search failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"Kitsu search service unavailable: {str(e)}"
        )


@router.get("/anime/{anime_id}")
async def get_anime_details(
    anime_id: int = Path(..., ge=1, description="Kitsu anime ID")
) -> Dict[str, Any]:
    """Get detailed anime information by Kitsu ID.
    
    Args:
        anime_id: Kitsu anime ID
        
    Returns:
        Detailed anime information
        
    Raises:
        HTTPException: If anime not found or service fails
    """
    try:
        logger.info("Kitsu anime details: anime_id=%d", anime_id)
        
        anime_data = await _kitsu_service.get_anime_details(anime_id)
        
        if not anime_data:
            raise HTTPException(
                status_code=404,
                detail=f"Anime with ID {anime_id} not found on Kitsu"
            )
        
        return {
            "source": "kitsu",
            "anime_id": anime_id,
            "data": anime_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Kitsu anime details failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"Kitsu service unavailable: {str(e)}"
        )


@router.get("/trending")
async def get_trending_anime(
    limit: int = Query(20, ge=1, le=50, description="Maximum number of results")
) -> Dict[str, Any]:
    """Get trending anime from Kitsu.
    
    Args:
        limit: Maximum number of results (1-50)
        
    Returns:
        Trending anime list
        
    Raises:
        HTTPException: If service fails
    """
    try:
        logger.info("Kitsu trending: limit=%d", limit)
        
        results = await _kitsu_service.get_trending_anime(limit)
        
        return {
            "source": "kitsu",
            "type": "trending",
            "limit": limit,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error("Kitsu trending anime failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"Kitsu service unavailable: {str(e)}"
        )


@router.get("/anime/{anime_id}/episodes")
async def get_anime_episodes(
    anime_id: int = Path(..., ge=1, description="Kitsu anime ID")
) -> Dict[str, Any]:
    """Get anime episodes list.
    
    Args:
        anime_id: Kitsu anime ID
        
    Returns:
        Episodes list
        
    Raises:
        HTTPException: If service fails
    """
    try:
        logger.info("Kitsu anime episodes: anime_id=%d", anime_id)
        
        episodes = await _kitsu_service.get_anime_episodes(anime_id)
        
        return {
            "source": "kitsu",
            "anime_id": anime_id,
            "episodes": episodes,
            "total_episodes": len(episodes)
        }
        
    except Exception as e:
        logger.error("Kitsu anime episodes failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"Kitsu service unavailable: {str(e)}"
        )


@router.get("/anime/{anime_id}/streaming")
async def get_streaming_links(
    anime_id: int = Path(..., ge=1, description="Kitsu anime ID")
) -> Dict[str, Any]:
    """Get streaming links for anime.
    
    Args:
        anime_id: Kitsu anime ID
        
    Returns:
        Streaming links list
        
    Raises:
        HTTPException: If service fails
    """
    try:
        logger.info("Kitsu streaming links: anime_id=%d", anime_id)
        
        streaming_links = await _kitsu_service.get_streaming_links(anime_id)
        
        return {
            "source": "kitsu",
            "anime_id": anime_id,
            "streaming_links": streaming_links,
            "total_links": len(streaming_links)
        }
        
    except Exception as e:
        logger.error("Kitsu streaming links failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"Kitsu service unavailable: {str(e)}"
        )


@router.get("/anime/{anime_id}/characters")
async def get_anime_characters(
    anime_id: int = Path(..., ge=1, description="Kitsu anime ID")
) -> Dict[str, Any]:
    """Get anime characters list.
    
    Args:
        anime_id: Kitsu anime ID
        
    Returns:
        Characters list
        
    Raises:
        HTTPException: If service fails
    """
    try:
        logger.info("Kitsu anime characters: anime_id=%d", anime_id)
        
        characters = await _kitsu_service.get_anime_characters(anime_id)
        
        return {
            "source": "kitsu",
            "anime_id": anime_id,
            "characters": characters,
            "total_characters": len(characters)
        }
        
    except Exception as e:
        logger.error("Kitsu anime characters failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"Kitsu service unavailable: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check Kitsu service health.
    
    Returns:
        Service health status
    """
    try:
        logger.info("Kitsu health check")
        return await _kitsu_service.health_check()
        
    except Exception as e:
        logger.error("Kitsu health check failed: %s", e)
        return {
            "service": "kitsu",
            "status": "unhealthy",
            "error": str(e),
            "circuit_breaker_open": True
        }