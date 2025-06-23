# src/api/search.py - Search API Endpoints
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
import logging
from ..models.anime import SearchRequest, SearchResponse, SearchResult
from ..vector.qdrant_client import QdrantClient

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
                score=hit.get("_score", 0.0),
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
                animecountdown_id=hit.get("animecountdown_id")
            )
            results.append(result)
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            processing_time_ms=raw_results[0].get("processing_time_ms", 0) if raw_results else 0
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/")
async def search_anime(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Number of results")
) -> SearchResponse:
    """Simple GET search endpoint"""
    request = SearchRequest(query=q, limit=limit)
    return await semantic_search(request)

@router.get("/similar/{anime_id}")
async def get_similar_anime(
    anime_id: str,
    limit: int = Query(10, ge=1, le=50, description="Number of similar anime")
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
            "count": len(similar_anime)
        }
        
    except Exception as e:
        logger.error(f"Similar anime search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Similar search failed: {str(e)}")