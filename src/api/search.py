# src/api/search.py - Search API Endpoints
import base64
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from ..models.anime import SearchRequest, SearchResponse, SearchResult

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/semantic", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """Perform semantic search on anime database"""
    try:
        # Import qdrant_client from main app context
        from ..main import qdrant_client

        if not qdrant_client:
            raise HTTPException(status_code=503, detail="Vector database not available")

        # Perform search
        raw_results = await qdrant_client.search(
            query=request.query,
            limit=request.limit,
        )

        # Convert to SearchResult format
        results = []
        for hit in raw_results:
            result = SearchResult(
                anime_id=hit.get("anime_id", ""),
                title=hit.get("title", ""),
                synopsis=hit.get("synopsis"),
                type=hit.get("type", ""),
                episodes=hit.get("episodes", 0),
                tags=hit.get("tags", []),
                studios=hit.get("studios", []),
                picture=hit.get("picture"),
                relevance_score=hit.get("_score", 0.0),
                anime_score=(
                    hit.get("score", {}).get("median") if hit.get("score") else None
                ),
                year=hit.get("year"),
                season=hit.get("season"),
                # Platform IDs
                myanimelist_id=hit.get("myanimelist_id"),
                anilist_id=hit.get("anilist_id"),
                kitsu_id=hit.get("kitsu_id"),
                anidb_id=hit.get("anidb_id"),
                anisearch_id=hit.get("anisearch_id"),
                simkl_id=hit.get("simkl_id"),
                livechart_id=hit.get("livechart_id"),
                animenewsnetwork_id=hit.get("animenewsnetwork_id"),
                animeplanet_id=hit.get("animeplanet_id"),
                notify_id=hit.get("notify_id"),
                animecountdown_id=hit.get("animecountdown_id"),
            )
            results.append(result)

        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            processing_time_ms=(
                raw_results[0].get("processing_time_ms", 0) if raw_results else 0
            ),
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/")
async def search_anime(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
) -> SearchResponse:
    """Simple GET search endpoint"""
    request = SearchRequest(query=q, limit=limit)
    return await semantic_search(request)


@router.get("/similar/{anime_id}")
async def get_similar_anime(
    anime_id: str,
    limit: int = Query(10, ge=1, le=50, description="Number of similar anime"),
) -> Dict[str, Any]:
    """Get similar anime based on vector similarity"""
    try:
        from ..main import qdrant_client

        if not qdrant_client:
            raise HTTPException(status_code=503, detail="Vector database not available")

        similar_anime = await qdrant_client.get_similar_anime(anime_id, limit)

        return {
            "anime_id": anime_id,
            "similar_anime": similar_anime,
            "count": len(similar_anime),
        }

    except Exception as e:
        logger.error(f"Similar anime search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Similar search failed: {str(e)}")


# Phase 4: Image Search Endpoints
@router.post("/by-image")
async def search_anime_by_image(
    image: UploadFile = File(..., description="Image file (JPG, PNG, WebP)"),
    limit: int = Form(10, ge=1, le=30, description="Number of results"),
) -> Dict[str, Any]:
    """Search for anime using image similarity.

    Upload an image to find anime with visually similar poster images.
    Supports JPG, PNG, and WebP formats.
    """
    try:
        from ..main import qdrant_client

        if not qdrant_client:
            raise HTTPException(status_code=503, detail="Vector database not available")

        # Check if multi-vector support is enabled
        if not getattr(qdrant_client, "_supports_multi_vector", False):
            raise HTTPException(
                status_code=501,
                detail="Image search not enabled. Enable multi-vector support in server configuration.",
            )

        # Validate file type
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="File must be an image (JPG, PNG, WebP)"
            )

        # Read and encode image
        image_data = await image.read()
        image_b64 = base64.b64encode(image_data).decode("utf-8")

        # Perform image search
        results = await qdrant_client.search_by_image(image_data=image_b64, limit=limit)

        return {
            "search_type": "image_similarity",
            "uploaded_file": image.filename,
            "file_size_bytes": len(image_data),
            "results": results,
            "total_results": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")


@router.post("/by-image-base64")
async def search_anime_by_image_base64(
    image_data: str = Form(..., description="Base64 encoded image data"),
    limit: int = Form(10, ge=1, le=30, description="Number of results"),
) -> Dict[str, Any]:
    """Search for anime using base64 image data.

    Alternative endpoint for applications that prefer base64 encoding.
    """
    try:
        from ..main import qdrant_client

        if not qdrant_client:
            raise HTTPException(status_code=503, detail="Vector database not available")

        # Check if multi-vector support is enabled
        if not getattr(qdrant_client, "_supports_multi_vector", False):
            raise HTTPException(
                status_code=501,
                detail="Image search not enabled. Enable multi-vector support in server configuration.",
            )

        # Validate base64 data
        if not image_data or not image_data.strip():
            raise HTTPException(status_code=400, detail="Image data cannot be empty")

        # Perform image search
        results = await qdrant_client.search_by_image(
            image_data=image_data, limit=limit
        )

        return {
            "search_type": "image_similarity_base64",
            "data_length": len(image_data),
            "results": results,
            "total_results": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Base64 image search failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Base64 image search failed: {str(e)}"
        )


@router.get("/visually-similar/{anime_id}")
async def find_visually_similar_anime(
    anime_id: str,
    limit: int = Query(10, ge=1, le=20, description="Number of similar anime"),
) -> Dict[str, Any]:
    """Find anime with similar visual style to a reference anime.

    Uses the poster image of the reference anime to find visually similar anime.
    """
    try:
        from ..main import qdrant_client

        if not qdrant_client:
            raise HTTPException(status_code=503, detail="Vector database not available")

        # Check if multi-vector support is enabled
        if not getattr(qdrant_client, "_supports_multi_vector", False):
            raise HTTPException(
                status_code=501,
                detail="Visual similarity search not enabled. Enable multi-vector support in server configuration.",
            )

        # Perform visual similarity search
        results = await qdrant_client.find_visually_similar_anime(
            anime_id=anime_id, limit=limit
        )

        return {
            "search_type": "visual_similarity",
            "reference_anime_id": anime_id,
            "similar_anime": results,
            "total_results": len(results),
        }

    except Exception as e:
        logger.error(f"Visual similarity search failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Visual similarity search failed: {str(e)}"
        )


@router.post("/multimodal")
async def search_multimodal_anime(
    query: str = Form(..., description="Text search query"),
    image: Optional[UploadFile] = File(
        None, description="Optional image file for visual similarity"
    ),
    limit: int = Form(10, ge=1, le=25, description="Number of results"),
    text_weight: float = Form(
        0.7,
        ge=0.0,
        le=1.0,
        description="Weight for text vs image similarity (0.7 = 70% text, 30% image)",
    ),
) -> Dict[str, Any]:
    """Search for anime using both text query and image similarity.

    Combines semantic text search with visual image search for enhanced discovery.
    The text_weight parameter controls the balance between text and image matching.
    """
    try:
        from ..main import qdrant_client

        if not qdrant_client:
            raise HTTPException(status_code=503, detail="Vector database not available")

        # Prepare image data if provided
        image_data = None
        file_info = {}

        if image:
            # Validate file type
            if not image.content_type or not image.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400, detail="File must be an image (JPG, PNG, WebP)"
                )

            # Read and encode image
            image_bytes = await image.read()
            image_data = base64.b64encode(image_bytes).decode("utf-8")
            file_info = {
                "filename": image.filename,
                "file_size_bytes": len(image_bytes),
                "content_type": image.content_type,
            }

        # Perform multimodal search
        results = await qdrant_client.search_multimodal(
            query=query, image_data=image_data, limit=limit, text_weight=text_weight
        )

        return {
            "search_type": "multimodal",
            "text_query": query,
            "has_image": image_data is not None,
            "text_weight": text_weight,
            "image_weight": 1.0 - text_weight,
            "file_info": file_info,
            "results": results,
            "total_results": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multimodal search failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Multimodal search failed: {str(e)}"
        )
