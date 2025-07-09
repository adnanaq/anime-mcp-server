"""
Semantic search MCP tools for vector database operations.

Specialized tools for semantic anime search using Qdrant vector database with CLIP and text embeddings.
These tools provide AI-powered similarity search capabilities beyond traditional keyword matching.
"""

from typing import Any, Dict, List, Literal, Optional

from fastmcp import FastMCP

from mcp.server.fastmcp import Context

from ...config import get_settings
from ...vector.qdrant_client import QdrantClient

# Initialize components
settings = get_settings()
qdrant_client = None

# Create FastMCP instance for tools
mcp = FastMCP("Semantic Search Tools")


def get_qdrant_client():
    """Get or create the global qdrant client instance."""
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = QdrantClient()
    return qdrant_client


@mcp.tool(
    name="anime_semantic_search",
    description="Perform semantic search across anime database using natural language queries",
    annotations={
        "title": "Semantic Anime Search",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def anime_semantic_search(
    query: str,
    search_mode: Literal["text", "visual", "combined"] = "combined",
    similarity_threshold: float = 0.7,
    limit: int = 20,
    include_scores: bool = True,
    boost_popular: bool = False,
    ctx: Optional[Context] = None,
) -> List[Dict[str, Any]]:
    """
    Perform semantic search using natural language queries against anime database.

    Uses FastEmbed text embeddings and optional CLIP visual embeddings for similarity matching.
    This enables finding anime based on descriptions, themes, and visual characteristics.

    Args:
        query: Natural language search query (e.g., "dark fantasy with strong protagonist")
        search_mode: Search approach ("text", "visual", "combined")
        similarity_threshold: Minimum similarity score (0.0-1.0, default: 0.7)
        limit: Maximum results (default: 20, max: 100)
        include_scores: Include similarity scores in results
        boost_popular: Boost results by popularity score

    Returns:
        List of anime with semantic similarity scores and comprehensive metadata
    """
    if ctx:
        await ctx.info(
            f"Performing semantic search for: '{query}' ({search_mode} mode)"
        )

    try:
        # Build search parameters
        search_params = {
            "query": query,
            "limit": min(limit, 100),
            "score_threshold": similarity_threshold,
            "with_payload": True,
            "with_vectors": False,
        }

        # Configure search mode
        if search_mode == "text":
            search_params["vector_name"] = "text_vector"
        elif search_mode == "visual":
            search_params["vector_name"] = "image_vector"
        else:  # combined
            search_params["vector_name"] = ["text_vector", "image_vector"]

        # Add popularity boosting if requested
        if boost_popular:
            search_params["boost_factor"] = {"field": "popularity_score", "factor": 1.2}

        # Execute semantic search
        client = get_qdrant_client()
        raw_results = await client.semantic_search(search_params)

        # Process results
        results = []
        for raw_result in raw_results:
            try:
                payload = raw_result.get("payload", {})
                score = raw_result.get("score", 0.0)

                # Build semantic search result
                result = {
                    "id": payload.get("id"),
                    "title": payload.get("title"),
                    "type": payload.get("type"),
                    "episodes": payload.get("episodes"),
                    "score": payload.get("mal_score") or payload.get("anilist_score"),
                    "year": payload.get("year"),
                    "status": payload.get("status"),
                    "genres": payload.get("genres", []),
                    "studios": payload.get("studios", []),
                    "synopsis": payload.get("synopsis"),
                    "image_url": payload.get("image_url"),
                    # Semantic search specific data
                    "similarity_score": score if include_scores else None,
                    "search_mode": search_mode,
                    "matched_on": payload.get("embedding_source", "text+visual"),
                    # Quality and metadata
                    "data_quality_score": payload.get("data_quality_score"),
                    "popularity_rank": payload.get("popularity_rank"),
                    "source_platforms": payload.get("source_platforms", []),
                    # Source attribution
                    "source_platform": "semantic_search",
                    "search_query": query,
                    "search_timestamp": payload.get("indexed_at"),
                }

                results.append(result)

            except Exception as e:
                if ctx:
                    await ctx.error(f"Failed to process semantic result: {str(e)}")
                continue

        if ctx:
            await ctx.info(f"Found {len(results)} semantically similar anime")

        return results

    except Exception as e:
        error_msg = f"Semantic search failed: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)


@mcp.tool(
    name="anime_similar",
    description="Find anime similar to a specific anime using vector similarity",
    annotations={
        "title": "Similar Anime Finder",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def anime_similar(
    anime_id: str,
    similarity_mode: Literal["content", "visual", "hybrid"] = "hybrid",
    exclude_same_series: bool = True,
    min_similarity: float = 0.6,
    limit: int = 15,
    ctx: Optional[Context] = None,
) -> List[Dict[str, Any]]:
    """
    Find anime similar to a specific anime using vector similarity matching.

    Uses the target anime's embeddings to find other anime with similar content,
    visual style, or combined characteristics in the vector space.

    Args:
        anime_id: ID of the reference anime to find similarities for
        similarity_mode: Type of similarity ("content", "visual", "hybrid")
        exclude_same_series: Exclude sequels/prequels of the same series
        min_similarity: Minimum similarity threshold (0.0-1.0)
        limit: Maximum similar anime to return (default: 15, max: 50)

    Returns:
        List of similar anime with similarity scores and relationship metadata
    """
    if ctx:
        await ctx.info(f"Finding anime similar to ID: {anime_id}")

    try:
        # Get reference anime vector
        client = get_qdrant_client()
        reference_anime = await client.get_anime_by_id(anime_id)
        if not reference_anime:
            if ctx:
                await ctx.error(f"Reference anime {anime_id} not found")
            raise ValueError(f"Anime {anime_id} not found in vector database")

        # Extract reference metadata for filtering
        ref_payload = reference_anime.get("payload", {})
        ref_title = ref_payload.get("title", "")
        ref_series_id = ref_payload.get("series_id")

        # Build similarity search parameters
        search_params = {
            "reference_id": anime_id,
            "limit": min(limit + 10, 60),  # Get extra for filtering
            "score_threshold": min_similarity,
            "exclude_ids": [anime_id],  # Exclude self
        }

        # Configure similarity mode
        if similarity_mode == "content":
            search_params["vector_name"] = "text_vector"
        elif similarity_mode == "visual":
            search_params["vector_name"] = "image_vector"
        else:  # hybrid
            search_params["vector_name"] = ["text_vector", "image_vector"]
            search_params["fusion_method"] = "weighted_average"
            search_params["weights"] = {"text_vector": 0.7, "image_vector": 0.3}

        # Execute similarity search
        raw_results = await client.find_similar(search_params)

        # Filter and process results
        results = []
        processed_count = 0

        for raw_result in raw_results:
            if processed_count >= limit:
                break

            try:
                payload = raw_result.get("payload", {})
                score = raw_result.get("score", 0.0)

                # Apply same-series filtering
                if exclude_same_series:
                    result_series_id = payload.get("series_id")
                    result_title = payload.get("title", "")

                    # Skip if same series or very similar title
                    if (ref_series_id and result_series_id == ref_series_id) or (
                        ref_title
                        and result_title
                        and any(
                            word in result_title.lower()
                            for word in ref_title.lower().split()[:2]
                        )
                    ):
                        continue

                # Build similarity result
                result = {
                    "id": payload.get("id"),
                    "title": payload.get("title"),
                    "type": payload.get("type"),
                    "episodes": payload.get("episodes"),
                    "score": payload.get("mal_score") or payload.get("anilist_score"),
                    "year": payload.get("year"),
                    "status": payload.get("status"),
                    "genres": payload.get("genres", []),
                    "studios": payload.get("studios", []),
                    "synopsis": payload.get("synopsis"),
                    "image_url": payload.get("image_url"),
                    # Similarity-specific data
                    "similarity_score": score,
                    "similarity_mode": similarity_mode,
                    "reference_anime_id": anime_id,
                    "reference_anime_title": ref_title,
                    # Similarity analysis
                    "shared_genres": list(
                        set(payload.get("genres", []) or [])
                        & set(ref_payload.get("genres", []) or [])
                    ),
                    "shared_studios": list(
                        set(payload.get("studios", []) or [])
                        & set(ref_payload.get("studios", []) or [])
                    ),
                    "year_difference": abs(
                        (payload.get("year") or 0) - (ref_payload.get("year") or 0)
                    ),
                    # Quality metadata
                    "data_quality_score": payload.get("data_quality_score"),
                    "popularity_rank": payload.get("popularity_rank"),
                    "source_platforms": payload.get("source_platforms", []),
                    # Source attribution
                    "source_platform": "similarity_search",
                    "search_timestamp": payload.get("indexed_at"),
                }

                results.append(result)
                processed_count += 1

            except Exception as e:
                if ctx:
                    await ctx.error(f"Failed to process similarity result: {str(e)}")
                continue

        if ctx:
            await ctx.info(f"Found {len(results)} similar anime to '{ref_title}'")

        return results

    except Exception as e:
        error_msg = f"Similar anime search failed: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)


@mcp.tool(
    name="anime_vector_stats",
    description="Get vector database statistics and collection information",
    annotations={
        "title": "Vector Database Stats",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def anime_vector_stats(
    include_schema: bool = False,
    include_sample: bool = False,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """
    Get comprehensive statistics about the anime vector database.

    Provides information about collection size, vector dimensions,
    indexing status, and optional schema/sample data.

    Args:
        include_schema: Include vector collection schema details
        include_sample: Include sample vector data for inspection

    Returns:
        Comprehensive vector database statistics and metadata
    """
    if ctx:
        await ctx.info("Retrieving vector database statistics")

    try:
        # Get collection info
        client = get_qdrant_client()
        collection_info = await client.get_collection_info()

        # Build comprehensive stats
        stats = {
            "collection_name": settings.qdrant_collection_name,
            "total_vectors": collection_info.get("vectors_count", 0),
            "indexed_vectors": collection_info.get("indexed_vectors_count", 0),
            "vector_size": collection_info.get("config", {})
            .get("params", {})
            .get("vectors", {}),
            "distance_metric": collection_info.get("config", {})
            .get("params", {})
            .get("distance"),
            "indexing_threshold": collection_info.get("config", {})
            .get("params", {})
            .get("indexing_threshold"),
            # Multi-vector information
            "vector_types": {
                "text_embeddings": {
                    "dimension": 384,  # FastEmbed BAAI/bge-small-en-v1.5
                    "model": "BAAI/bge-small-en-v1.5",
                },
                "image_embeddings": {
                    "dimension": 512,  # CLIP ViT-B/32
                    "model": "ViT-B/32",
                },
            },
            # Status information
            "status": collection_info.get("status"),
            "optimizer_status": collection_info.get("optimizer_status", {}),
            "indexing_status": (
                "indexed"
                if collection_info.get("indexed_vectors_count", 0) > 0
                else "not_indexed"
            ),
            # Memory and performance
            "memory_usage_mb": collection_info.get("memory_usage_bytes", 0)
            / (1024 * 1024),
            "disk_usage_mb": collection_info.get("disk_usage_bytes", 0) / (1024 * 1024),
        }

        # Add schema information if requested
        if include_schema:
            stats["schema"] = {
                "payload_schema": {
                    "id": "string",
                    "title": "string",
                    "synopsis": "string",
                    "genres": "array[string]",
                    "studios": "array[string]",
                    "year": "integer",
                    "score": "float",
                    "popularity_rank": "integer",
                    "data_quality_score": "float",
                    "source_platforms": "array[string]",
                    "embedding_source": "string",
                    "indexed_at": "timestamp",
                },
                "vector_schema": {
                    "text_vector": "float[384] - FastEmbed text embeddings",
                    "image_vector": "float[512] - CLIP visual embeddings",
                },
            }

        # Add sample data if requested
        if include_sample and stats["total_vectors"] > 0:
            try:
                sample_results = await client.get_random_samples(limit=3)
                stats["sample_entries"] = [
                    {
                        "id": sample.get("id"),
                        "title": sample.get("payload", {}).get("title"),
                        "genres": sample.get("payload", {}).get("genres", [])[:3],
                        "vector_sizes": {
                            name: len(vectors) if vectors else 0
                            for name, vectors in sample.get("vector", {}).items()
                        },
                    }
                    for sample in sample_results
                ]
            except Exception as e:
                stats["sample_entries"] = f"Could not retrieve samples: {str(e)}"

        if ctx:
            await ctx.info(f"Retrieved stats for {stats['total_vectors']} vectors")

        return stats

    except Exception as e:
        error_msg = f"Failed to get vector stats: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)
