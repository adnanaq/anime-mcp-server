"""AniDB service integration following modular pattern."""

import logging
from typing import Any, Dict, List, Optional

from ...config import get_settings
from ...integrations.clients.anidb_client import AniDBClient
from ...integrations.error_handling import ErrorContext
from .base_service import BaseExternalService

logger = logging.getLogger(__name__)


class AniDBService(BaseExternalService):
    """AniDB service wrapper for anime database operations."""

    def __init__(self):
        """Initialize AniDB service with shared dependencies."""
        super().__init__(service_name="anidb")

        # Get configuration
        settings = get_settings()

        # Initialize AniDB client
        self.client = AniDBClient(
            client_name=settings.anidb_client or "anime-mcp-server",
            client_version=settings.anidb_clientver or "1.0.0",
            circuit_breaker=self.circuit_breaker,
            cache_manager=self.cache_manager,
            error_handler=ErrorContext(
                user_message="AniDB service error",
                debug_info="AniDB.net API integration error",
            ),
        )

    async def search_anime(self, query: str) -> List[Dict[str, Any]]:
        """Search for anime on AniDB.

        Args:
            query: Search query string

        Returns:
            List of anime search results

        Raises:
            ValueError: If query is empty
            Exception: If search fails
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        try:
            logger.info("AniDB search: query='%s'", query)
            # AniDB search returns single result, wrap in list for consistency
            result = await self.client.search_anime_by_name(query.strip())
            return [result] if result else []
        except Exception as e:
            logger.error("AniDB search failed: %s", e)
            raise

    async def get_anime_details(self, anime_id: int) -> Optional[Dict[str, Any]]:
        """Get anime details by ID.

        Args:
            anime_id: AniDB anime ID

        Returns:
            Anime details or None if not found

        Raises:
            ValueError: If anime ID is invalid
            Exception: If request fails
        """
        if anime_id <= 0:
            raise ValueError("Anime ID must be positive")

        try:
            logger.info("AniDB anime details: anime_id=%d", anime_id)
            return await self.client.get_anime_by_id(anime_id)
        except Exception as e:
            logger.error("AniDB anime details failed: %s", e)
            raise

    async def get_anime_characters(self, anime_id: int) -> List[Dict[str, Any]]:
        """Get anime characters by anime ID.

        Args:
            anime_id: AniDB anime ID

        Returns:
            List of anime characters

        Raises:
            ValueError: If anime ID is invalid
            Exception: If request fails
        """
        if anime_id <= 0:
            raise ValueError("Anime ID must be positive")

        try:
            logger.info("AniDB anime characters: anime_id=%d", anime_id)
            return await self.client.get_anime_characters(anime_id)
        except Exception as e:
            logger.error("AniDB anime characters failed: %s", e)
            raise

    async def get_anime_episodes(self, anime_id: int) -> List[Dict[str, Any]]:
        """Get anime episodes by anime ID.

        Args:
            anime_id: AniDB anime ID

        Returns:
            List of anime episodes

        Raises:
            ValueError: If anime ID is invalid
            Exception: If request fails
        """
        if anime_id <= 0:
            raise ValueError("Anime ID must be positive")

        try:
            logger.info("AniDB anime episodes: anime_id=%d", anime_id)
            # AniDB doesn't have a direct episodes endpoint, return empty list
            # Episodes would need to be extracted from anime details
            return []
        except Exception as e:
            logger.error("AniDB anime episodes failed: %s", e)
            raise

    async def get_similar_anime(
        self, anime_id: int, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get similar anime recommendations.

        Args:
            anime_id: AniDB anime ID
            limit: Maximum number of recommendations (1-50)

        Returns:
            List of similar anime

        Raises:
            ValueError: If parameters are invalid
            Exception: If request fails
        """
        if anime_id <= 0:
            raise ValueError("Anime ID must be positive")

        if limit < 1 or limit > 50:
            raise ValueError("Limit must be between 1 and 50")

        try:
            logger.info("AniDB similar anime: anime_id=%d, limit=%d", anime_id, limit)
            # AniDB doesn't have a direct similar anime endpoint
            # Return empty list for now
            return []
        except Exception as e:
            logger.error("AniDB similar anime failed: %s", e)
            raise

    async def get_random_anime(self) -> Dict[str, Any]:
        """Get a random anime from AniDB.

        Returns:
            Random anime data

        Raises:
            Exception: If request fails
        """
        try:
            logger.info("AniDB random anime")
            # AniDB doesn't have a direct random endpoint
            # Could implement by getting a random ID and fetching details
            import random

            random_id = random.randint(1, 1000)  # Basic implementation
            result = await self.client.get_anime_by_id(random_id)
            return result or {"error": "No random anime found"}
        except Exception as e:
            logger.error("AniDB random anime failed: %s", e)
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check service health status.

        Returns:
            Health status information
        """
        try:
            # Simple health check - try to get anime with ID 1 (should exist)
            await self.client.get_anime_by_id(1)

            return {
                "service": self.service_name,
                "status": "healthy",
                "circuit_breaker_open": self.circuit_breaker.is_open(),
                "response_time": "50ms",  # Placeholder
            }

        except Exception as e:
            logger.warning("AniDB health check failed: %s", e)
            return {
                "service": self.service_name,
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_open": self.circuit_breaker.is_open(),
            }
