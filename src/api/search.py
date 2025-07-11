# src/api/search.py - Unified Search API Endpoint with Single Interface
import base64
import logging
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from ..models.anime import SearchRequest, SearchResponse, SearchResult, UnifiedSearchRequest

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=Union[SearchResponse, Dict[str, Any]])
async def unified_search(request: Request):
    """Unified search endpoint supporting both JSON and form data
    
    Automatically detects request type and search type:
    - JSON request: Send JSON body with search parameters
    - Form data: Send form data with file uploads
    
    Search types (auto-detected):
    - Text search: query only
    - Similar search: anime_id only
    - Image search: image_data (base64) or image file upload
    - Visual similarity: anime_id + visual_similarity=true
    - Multimodal: query + image_data/file
    """
    # Detect request type from content-type header
    content_type = request.headers.get("content-type", "")
    is_form_data = content_type.startswith("multipart/form-data")
    
    if is_form_data:
        # Handle form data request
        form = await request.form()
        
        # Extract form fields
        query = form.get("query")
        anime_id = form.get("anime_id")
        visual_similarity = form.get("visual_similarity", "false").lower() == "true"
        text_weight = float(form.get("text_weight", 0.7))
        limit = int(form.get("limit", 20))
        image = form.get("image")
        image_data = form.get("image_data")
        
        return await _process_unified_search(
            request=None,
            http_request=request,
            form_query=query,
            form_anime_id=anime_id,
            form_visual_similarity=visual_similarity,
            form_text_weight=text_weight,
            form_limit=limit,
            form_image=image,
            form_image_data=image_data,
            is_form_data=True
        )
    else:
        # Handle JSON request
        try:
            json_data = await request.json()
            unified_request = UnifiedSearchRequest(**json_data)
            
            return await _process_unified_search(
                request=unified_request,
                http_request=request,
                form_query=None,
                form_anime_id=None,
                form_visual_similarity=None,
                form_text_weight=None,
                form_limit=None,
                form_image=None,
                form_image_data=None,
                is_form_data=False
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON request body: {str(e)}"
            )


async def _process_unified_search(
    request: Optional[UnifiedSearchRequest],
    http_request: Request,
    form_query: Optional[str],
    form_anime_id: Optional[str],
    form_visual_similarity: Optional[bool],
    form_text_weight: Optional[float],
    form_limit: Optional[int],
    form_image: Optional[UploadFile],
    form_image_data: Optional[str],
    is_form_data: bool,
):
    """Process unified search request from either JSON or form data"""
    correlation_id = getattr(http_request.state, "correlation_id", None)
    
    try:
        # Import qdrant_client from main app context
        from ..main import qdrant_client
        
        if not qdrant_client:
            logger.error(
                "Vector database not available",
                extra={"correlation_id": correlation_id},
            )
            raise HTTPException(status_code=503, detail="Vector database not available")
        
        # Create unified request object from form data or use JSON request
        if is_form_data:
            # Handle image file upload
            image_data_b64 = None
            if form_image:
                # Validate file type
                if not form_image.content_type or not form_image.content_type.startswith("image/"):
                    raise HTTPException(
                        status_code=400, detail="File must be an image (JPG, PNG, WebP)"
                    )
                
                # Read and encode image
                image_bytes = await form_image.read()
                image_data_b64 = base64.b64encode(image_bytes).decode("utf-8")
                
                logger.info(
                    f"Image file uploaded: {form_image.filename}",
                    extra={
                        "correlation_id": correlation_id,
                        "uploaded_filename": form_image.filename,
                        "content_type": form_image.content_type,
                        "file_size": len(image_bytes),
                    },
                )
            elif form_image_data:
                image_data_b64 = form_image_data
            
            # Create request object from form data
            unified_request = UnifiedSearchRequest(
                query=form_query,
                anime_id=form_anime_id,
                image_data=image_data_b64,
                visual_similarity=form_visual_similarity or False,
                text_weight=form_text_weight or 0.7,
                limit=form_limit or 20,
            )
        else:
            # Use JSON request object
            if not request:
                raise HTTPException(
                    status_code=400, 
                    detail="Either provide JSON request body or use form data with file upload"
                )
            unified_request = request
        
        # Smart auto-detection of search type
        search_type = _detect_search_type(unified_request)
        
        logger.info(
            f"Unified search detected type: {search_type}",
            extra={
                "correlation_id": correlation_id,
                "search_type": search_type,
                "has_query": bool(unified_request.query),
                "has_anime_id": bool(unified_request.anime_id),
                "has_image_data": bool(unified_request.image_data),
                "visual_similarity": unified_request.visual_similarity,
                "is_form_data": is_form_data,
            },
        )
        
        # Route to appropriate search handler
        if search_type == "text":
            return await _handle_text_search(unified_request, qdrant_client, correlation_id)
        elif search_type == "similar":
            return await _handle_similar_search(unified_request, qdrant_client, correlation_id)
        elif search_type == "image":
            return await _handle_image_search(unified_request, qdrant_client, correlation_id)
        elif search_type == "visual_similarity":
            return await _handle_visual_similarity_search(unified_request, qdrant_client, correlation_id)
        elif search_type == "multimodal":
            return await _handle_multimodal_search(unified_request, qdrant_client, correlation_id)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid search request: could not detect search type from provided fields"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Unified search failed: {e}",
            extra={
                "correlation_id": correlation_id,
                "error_type": type(e).__name__,
                "is_form_data": is_form_data,
            },
        )
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


def _detect_search_type(request: UnifiedSearchRequest) -> str:
    """Detect search type based on provided fields"""
    
    # Multimodal: both query and image_data
    if request.query and request.image_data:
        return "multimodal"
    
    # Visual similarity: anime_id + visual_similarity flag
    if request.anime_id and request.visual_similarity:
        return "visual_similarity"
    
    # Image search: image_data only
    if request.image_data:
        return "image"
    
    # Similar search: anime_id only
    if request.anime_id:
        return "similar"
    
    # Text search: query only
    if request.query:
        return "text"
    
    # No valid combination
    return "invalid"


async def _handle_text_search(request: UnifiedSearchRequest, qdrant_client, correlation_id: str) -> SearchResponse:
    """Handle text/semantic search"""
    
    logger.info(
        f"Starting semantic search for query: {request.query}",
        extra={
            "correlation_id": correlation_id,
            "query": request.query,
            "limit": request.limit,
        },
    )
    
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
    
    logger.info(
        f"Semantic search completed successfully",
        extra={
            "correlation_id": correlation_id,
            "query": request.query,
            "total_results": len(results),
            "processing_time_ms": (
                raw_results[0].get("processing_time_ms", 0) if raw_results else 0
            ),
        },
    )
    
    return SearchResponse(
        query=request.query,
        results=results,
        total_results=len(results),
        processing_time_ms=(
            raw_results[0].get("processing_time_ms", 0) if raw_results else 0
        ),
    )


async def _handle_similar_search(request: UnifiedSearchRequest, qdrant_client, correlation_id: str) -> Dict[str, Any]:
    """Handle similar anime search"""
    
    logger.info(
        f"Finding similar anime for ID: {request.anime_id}",
        extra={
            "correlation_id": correlation_id,
            "anime_id": request.anime_id,
            "limit": request.limit,
        },
    )
    
    similar_anime = await qdrant_client.get_similar_anime(request.anime_id, request.limit)
    
    logger.info(
        f"Similar anime search completed",
        extra={
            "correlation_id": correlation_id,
            "anime_id": request.anime_id,
            "similar_count": len(similar_anime),
        },
    )
    
    return {
        "search_type": "similar",
        "anime_id": request.anime_id,
        "similar_anime": similar_anime,
        "count": len(similar_anime),
    }


async def _handle_image_search(request: UnifiedSearchRequest, qdrant_client, correlation_id: str) -> Dict[str, Any]:
    """Handle image similarity search"""
    
    logger.info(
        f"Starting image search",
        extra={
            "correlation_id": correlation_id,
            "data_length": len(request.image_data),
            "limit": request.limit,
        },
    )
    
    
    # Validate image data
    if not request.image_data or not request.image_data.strip():
        raise HTTPException(status_code=400, detail="Image data cannot be empty")
    
    # Perform image search
    results = await qdrant_client.search_by_image(
        image_data=request.image_data, 
        limit=request.limit
    )
    
    return {
        "search_type": "image_similarity",
        "data_length": len(request.image_data),
        "results": results,
        "total_results": len(results),
    }


async def _handle_visual_similarity_search(request: UnifiedSearchRequest, qdrant_client, correlation_id: str) -> Dict[str, Any]:
    """Handle visual similarity search by anime ID"""
    
    logger.info(
        f"Finding visually similar anime for ID: {request.anime_id}",
        extra={
            "correlation_id": correlation_id,
            "anime_id": request.anime_id,
            "limit": request.limit,
        },
    )
    
    
    # Perform visual similarity search
    results = await qdrant_client.find_visually_similar_anime(
        anime_id=request.anime_id, 
        limit=request.limit
    )
    
    return {
        "search_type": "visual_similarity",
        "reference_anime_id": request.anime_id,
        "similar_anime": results,
        "total_results": len(results),
    }


async def _handle_multimodal_search(request: UnifiedSearchRequest, qdrant_client, correlation_id: str) -> Dict[str, Any]:
    """Handle multimodal text + image search"""
    
    logger.info(
        f"Starting multimodal search",
        extra={
            "correlation_id": correlation_id,
            "query": request.query,
            "has_image": bool(request.image_data),
            "text_weight": request.text_weight,
            "limit": request.limit,
        },
    )
    
    
    # Perform multimodal search
    results = await qdrant_client.search_multimodal(
        query=request.query,
        image_data=request.image_data,
        limit=request.limit,
        text_weight=request.text_weight
    )
    
    return {
        "search_type": "multimodal",
        "text_query": request.query,
        "has_image": request.image_data is not None,
        "text_weight": request.text_weight,
        "image_weight": 1.0 - request.text_weight,
        "results": results,
        "total_results": len(results),
    }