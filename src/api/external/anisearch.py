"""AniSearch API endpoints for external service integration."""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Path, Query

from ...services.external.anisearch_service import AniSearchService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/external/anisearch", tags=["External APIs", "AniSearch"])

# Global service instance
_anisearch_service = AniSearchService()


@router.get("/search")
async def search_anime(
    q: str = Query(..., description="Search query string"),
    limit: int = Query(20, description="Maximum number of results", ge=1, le=100),
) -> Dict[str, Any]:
    """Search for anime on AniSearch.

    Args:
        q: Search query string
        limit: Maximum number of results (1-100)

    Returns:
        Anime search results

    Raises:
        HTTPException: If service is unavailable
    """
    try:
        logger.info("AniSearch search: query='%s', limit=%d", q, limit)

        search_results = await _anisearch_service.search_anime(q)

        return {
            "source": "anisearch",
            "query": q,
            "limit": limit,
            "results": search_results,
            "total_results": len(search_results),
        }

    except ValueError as e:
        logger.warning("AniSearch search invalid input: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("AniSearch search failed: %s", e)
        raise HTTPException(
            status_code=503, detail=f"AniSearch search service unavailable: {str(e)}"
        )


@router.get("/anime/{anime_id}")
async def get_anime_details(
    anime_id: str = Path(..., description="AniSearch anime ID/slug")
) -> Dict[str, Any]:
    """Get anime details by ID from AniSearch.

    Args:
        anime_id: AniSearch anime ID/slug

    Returns:
        Anime details

    Raises:
        HTTPException: If anime not found or service is unavailable
    """
    try:
        logger.info("AniSearch anime details: anime_id='%s'", anime_id)

        anime_data = await _anisearch_service.get_anime_details(anime_id)

        if anime_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Anime with ID {anime_id} not found on AniSearch",
            )

        return {"source": "anisearch", "anime_id": anime_id, "data": anime_data}

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("AniSearch anime details invalid input: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("AniSearch anime details failed: %s", e)
        raise HTTPException(
            status_code=503, detail=f"AniSearch service unavailable: {str(e)}"
        )


@router.get("/anime/{anime_id}/characters")
async def get_anime_characters(
    anime_id: str = Path(..., description="AniSearch anime ID/slug")
) -> Dict[str, Any]:
    """Get anime characters by anime ID from AniSearch.

    Args:
        anime_id: AniSearch anime ID/slug

    Returns:
        Anime characters data

    Raises:
        HTTPException: If service is unavailable
    """
    try:
        logger.info("AniSearch anime characters: anime_id='%s'", anime_id)

        characters_data = await _anisearch_service.get_anime_characters(anime_id)

        return {
            "source": "anisearch",
            "anime_id": anime_id,
            "characters": characters_data,
            "total_characters": len(characters_data),
        }

    except ValueError as e:
        logger.warning("AniSearch anime characters invalid input: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("AniSearch anime characters failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AniSearch characters service unavailable: {str(e)}",
        )


@router.get("/anime/{anime_id}/recommendations")
async def get_anime_recommendations(
    anime_id: str = Path(..., description="AniSearch anime ID/slug"),
    limit: int = Query(
        10, description="Maximum number of recommendations", ge=1, le=50
    ),
) -> Dict[str, Any]:
    """Get anime recommendations from AniSearch.

    Args:
        anime_id: AniSearch anime ID/slug
        limit: Maximum number of recommendations (1-50)

    Returns:
        Anime recommendations

    Raises:
        HTTPException: If service is unavailable
    """
    try:
        logger.info(
            "AniSearch recommendations: anime_id='%s', limit=%d", anime_id, limit
        )

        recommendations = await _anisearch_service.get_anime_recommendations(
            anime_id, limit
        )

        return {
            "source": "anisearch",
            "anime_id": anime_id,
            "limit": limit,
            "recommendations": recommendations,
            "total_results": len(recommendations),
        }

    except ValueError as e:
        logger.warning("AniSearch recommendations invalid input: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("AniSearch recommendations failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AniSearch recommendations service unavailable: {str(e)}",
        )


@router.get("/top")
async def get_top_anime(
    category: str = Query(
        "highest_rated",
        description="Category (highest_rated, most_popular, newest, etc.)",
    ),
    limit: int = Query(25, description="Maximum number of results", ge=1, le=100),
) -> Dict[str, Any]:
    """Get top anime from AniSearch.

    Args:
        category: Category type (highest_rated, most_popular, newest, etc.)
        limit: Maximum number of results (1-100)

    Returns:
        Top anime data

    Raises:
        HTTPException: If service is unavailable
    """
    try:
        logger.info("AniSearch top anime: category='%s', limit=%d", category, limit)

        top_anime = await _anisearch_service.get_top_anime(category, limit)

        return {
            "source": "anisearch",
            "category": category,
            "limit": limit,
            "anime": top_anime,
            "total_results": len(top_anime),
        }

    except ValueError as e:
        logger.warning("AniSearch top anime invalid input: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("AniSearch top anime failed: %s", e)
        raise HTTPException(
            status_code=503, detail=f"AniSearch top anime service unavailable: {str(e)}"
        )


@router.get("/seasonal")
async def get_seasonal_anime(
    year: Optional[int] = Query(None, description="Year (defaults to current year)"),
    season: Optional[str] = Query(
        None, description="Season (spring, summer, fall, winter)"
    ),
    limit: int = Query(25, description="Maximum number of results", ge=1, le=100),
) -> Dict[str, Any]:
    """Get seasonal anime from AniSearch.

    Args:
        year: Year (defaults to current year)
        season: Season (spring, summer, fall, winter) (defaults to current season)
        limit: Maximum number of results (1-100)

    Returns:
        Seasonal anime data

    Raises:
        HTTPException: If service is unavailable
    """
    try:
        logger.info(
            "AniSearch seasonal anime: year=%s, season='%s', limit=%d",
            year,
            season,
            limit,
        )

        seasonal_anime = await _anisearch_service.get_seasonal_anime(
            year, season, limit
        )

        return {
            "source": "anisearch",
            "year": year,
            "season": season,
            "limit": limit,
            "anime": seasonal_anime,
            "total_results": len(seasonal_anime),
        }

    except ValueError as e:
        logger.warning("AniSearch seasonal anime invalid input: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("AniSearch seasonal anime failed: %s", e)
        raise HTTPException(
            status_code=503, detail=f"AniSearch seasonal service unavailable: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check AniSearch service health status.

    Returns:
        Service health information
    """
    try:
        logger.debug("AniSearch health check")
        health_data = await _anisearch_service.health_check()
        return health_data

    except Exception as e:
        logger.error("AniSearch health check failed: %s", e)
        return {
            "service": "anisearch",
            "status": "unhealthy",
            "error": str(e),
            "circuit_breaker_open": True,
        }
