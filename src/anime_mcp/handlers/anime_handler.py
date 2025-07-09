"""
Anime-specific handler implementing core business logic.

Separates MCP concerns from business operations following modern patterns.
"""

import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import Context

from ...config import Settings
from ...exceptions import AnimeNotFoundError
from ...vector.qdrant_client import QdrantClient
from ..schemas import SearchAnimeInput
from .base_handler import BaseAnimeHandler

logger = logging.getLogger(__name__)


class AnimeHandler(BaseAnimeHandler):
    """Handler for anime-specific operations with business logic separation."""

    def __init__(self, qdrant_client: Optional[QdrantClient], settings: Settings):
        """Initialize anime handler.

        Args:
            qdrant_client: Qdrant vector database client
            settings: Application settings
        """
        super().__init__(qdrant_client, settings)
        # Removed ServiceManager - using direct tool calls instead

    async def search_anime(
        self, params: SearchAnimeInput, ctx: Optional[Context] = None
    ) -> List[Dict[str, Any]]:
        """Search for anime using semantic search with filters.

        Args:
            params: Validated search parameters
            ctx: Optional MCP context for logging

        Returns:
            List of anime matching the search criteria

        Raises:
            ClientNotInitializedError: If Qdrant client not available
            DatabaseOperationError: If search operation fails
        """
        client = self.verify_client("search_anime")

        # Validate and prepare parameters
        limit = self.validate_limit(params.limit, 50)
        query = params.query.strip() or "anime"  # Default fallback

        await self.log_operation_start(
            "anime_search", f"'{query}' (limit: {limit})", ctx
        )

        try:
            # Build filters from search parameters
            filters = self._build_search_filters(params)

            # Execute search with enhanced parameters
            results = await client.search(query=query, limit=limit, filters=filters)

            await self.log_operation_success("anime_search", len(results), ctx)
            return results

        except Exception as e:
            await self.handle_error(e, "anime_search", ctx)

    async def get_anime_details(
        self, anime_id: str, ctx: Optional[Context] = None
    ) -> Dict[str, Any]:
        """Get detailed information about a specific anime.

        Args:
            anime_id: Unique anime identifier
            ctx: Optional MCP context for logging

        Returns:
            Detailed anime information

        Raises:
            AnimeNotFoundError: If anime not found
            DatabaseOperationError: If retrieval fails
        """
        client = self.verify_client("get_anime_details")

        await self.log_operation_start("anime_details", f"ID: {anime_id}", ctx)

        try:
            anime = await client.get_by_id(anime_id)
            if not anime:
                raise AnimeNotFoundError("get_anime_details", anime_id)

            title = anime.get("title", "Unknown")
            await self.log_operation_success("anime_details", 1, ctx)

            if ctx:
                await ctx.info(f"Retrieved details for: {title}")

            return anime

        except AnimeNotFoundError:
            if ctx:
                await ctx.warning(f"Anime not found: {anime_id}")
            raise
        except Exception as e:
            await self.handle_error(e, "get_anime_details", ctx)

    async def find_similar_anime(
        self, anime_id: str, limit: int = 10, ctx: Optional[Context] = None
    ) -> List[Dict[str, Any]]:
        """Find anime similar to a given anime.

        Args:
            anime_id: Reference anime ID
            limit: Maximum number of similar anime to return
            ctx: Optional MCP context for logging

        Returns:
            List of similar anime with similarity scores
        """
        client = self.verify_client("find_similar_anime")

        # Validate limit
        limit = self.validate_limit(limit, 20)

        await self.log_operation_start(
            "similar_anime", f"ID: {anime_id} (limit: {limit})", ctx
        )

        try:
            results = await client.find_similar(anime_id=anime_id, limit=limit)
            await self.log_operation_success("similar_anime", len(results), ctx)
            return results

        except Exception as e:
            await self.handle_error(e, "find_similar_anime", ctx)

    async def get_database_stats(self, ctx: Optional[Context] = None) -> Dict[str, Any]:
        """Get statistics about the anime database.

        Args:
            ctx: Optional MCP context for logging

        Returns:
            Database statistics and health information
        """
        client = self.verify_client("get_database_stats")

        await self.log_operation_start("database_stats", "retrieving stats", ctx)

        try:
            stats = await client.get_stats()
            health = await client.health_check()

            result = {
                **stats,
                "health_status": "healthy" if health else "unhealthy",
                "server_info": {
                    "qdrant_url": self.settings.qdrant_url,
                    "collection_name": self.settings.qdrant_collection_name,
                    "vector_model": self.settings.fastembed_model,
                },
            }

            await self.log_operation_success("database_stats", 1, ctx)
            return result

        except Exception as e:
            await self.handle_error(e, "get_database_stats", ctx)

    async def search_by_image(
        self, image_data: str, limit: int = 10, ctx: Optional[Context] = None
    ) -> List[Dict[str, Any]]:
        """Search for anime using visual similarity.

        Args:
            image_data: Base64 encoded image data
            limit: Maximum number of results to return
            ctx: Optional MCP context for logging

        Returns:
            List of visually similar anime
        """
        client = self.verify_client("search_by_image")
        self.check_multi_vector_support("search_by_image")

        # Validate limit
        limit = self.validate_limit(limit, 30)

        await self.log_operation_start("image_search", f"limit: {limit}", ctx)

        try:
            results = await client.search_by_image(image_data=image_data, limit=limit)
            await self.log_operation_success("image_search", len(results), ctx)
            return results

        except Exception as e:
            await self.handle_error(e, "search_by_image", ctx)

    async def search_multimodal(
        self,
        query: str,
        image_data: Optional[str] = None,
        limit: int = 10,
        text_weight: float = 0.7,
        ctx: Optional[Context] = None,
    ) -> List[Dict[str, Any]]:
        """Search using both text and image.

        Args:
            query: Text search query
            image_data: Optional base64 image data
            limit: Maximum number of results
            text_weight: Weight for text vs image similarity
            ctx: Optional MCP context for logging

        Returns:
            List of anime with combined similarity scores
        """
        client = self.verify_client("search_multimodal")

        # Validate parameters
        limit = self.validate_limit(limit, 25)
        text_weight = max(0.0, min(1.0, text_weight))

        await self.log_operation_start(
            "multimodal_search",
            f"'{query}' with image={image_data is not None} (weight: {text_weight})",
            ctx,
        )

        try:
            # Check if enhanced multimodal search is available
            if getattr(client, "_supports_multi_vector", False) and image_data:
                results = await client.search_multimodal(
                    query=query,
                    image_data=image_data,
                    limit=limit,
                    text_weight=text_weight,
                )
            else:
                # Fallback to text-only search
                if image_data and ctx:
                    await ctx.warning(
                        "Multi-vector not enabled, falling back to text-only search"
                    )
                results = await client.search(query=query, limit=limit)

            await self.log_operation_success("multimodal_search", len(results), ctx)
            return results

        except Exception as e:
            await self.handle_error(e, "search_multimodal", ctx)

    # Removed Universal parameter methods - replaced with direct tool calls
    # Modern approach: Tools call specific platform APIs directly without Universal parameter abstraction
    # This eliminates the 444-parameter Universal system and provides better LLM experience

    def _build_search_filters(
        self, params: SearchAnimeInput
    ) -> Optional[Dict[str, Any]]:
        """Build Qdrant-compatible filters from search parameters.

        Args:
            params: Search parameters with filters

        Returns:
            Filter dictionary for Qdrant or None
        """
        filters = {}

        # Genre filters
        if params.genres and any(params.genres):
            filters["tags"] = {"any": params.genres}

        # Year range filter
        if params.year_range and len(params.year_range) == 2:
            start_year, end_year = params.year_range
            if start_year and end_year:
                filters["year"] = {"gte": start_year, "lte": end_year}
            elif start_year:
                filters["year"] = {"gte": start_year}
            elif end_year:
                filters["year"] = {"lte": end_year}

        # Anime type filters
        if params.anime_types and any(params.anime_types):
            filters["type"] = {"any": params.anime_types}

        # Studio filters
        if params.studios and any(params.studios):
            filters["studios"] = {"any": params.studios}

        # Exclusion filters
        if params.exclusions and any(params.exclusions):
            filters["exclude_tags"] = params.exclusions

        # Mood keyword filters (combine with existing tags)
        if params.mood_keywords and any(params.mood_keywords):
            existing_tags = filters.get("tags", {}).get("any", [])
            if isinstance(existing_tags, list):
                combined_tags = existing_tags + params.mood_keywords
            else:
                combined_tags = params.mood_keywords
            filters["tags"] = {"any": combined_tags}

        return filters if filters else None
