"""AniList service integration following modular pattern."""

import logging
from typing import Any, Dict, List, Optional

from ...config import get_settings
from ...integrations.clients.anilist_client import AniListClient
from ...integrations.error_handling import ErrorContext
from .base_service import BaseExternalService

logger = logging.getLogger(__name__)


class AniListService(BaseExternalService):
    """AniList service wrapper for anime data operations."""

    def __init__(self, auth_token: Optional[str] = None):
        """Initialize AniList service with shared dependencies.

        Args:
            auth_token: Optional OAuth2 bearer token for authenticated requests
        """
        super().__init__(service_name="anilist")

        # Get settings for auth token if not provided
        if auth_token is None:
            settings = get_settings()
            auth_token = settings.anilist_auth_token

        # Initialize AniList client
        self.client = AniListClient(
            auth_token=auth_token,
            circuit_breaker=self.circuit_breaker,
            cache_manager=self.cache_manager,
            error_handler=ErrorContext(
                user_message="AniList service error",
                debug_info="AniList API integration error",
            ),
        )

    async def search_anime(
        self, query: str, limit: int = 10, page: int = 1
    ) -> List[Dict[str, Any]]:
        """Search for anime on AniList.

        Args:
            query: Search query string
            limit: Maximum number of results
            page: Page number for pagination

        Returns:
            List of anime search results

        Raises:
            Exception: If search fails
        """
        try:
            logger.info(
                "AniList search: query='%s', limit=%d, page=%d", query, limit, page
            )
            # Note: AniList client doesn't support pagination in current implementation
            return await self.client.search_anime(query=query, limit=limit, page=page)
        except Exception as e:
            logger.error("AniList search failed: %s", e)
            raise

    async def get_anime_details(self, anime_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed anime information by ID.

        Args:
            anime_id: AniList anime ID

        Returns:
            Anime details or None if not found
        """
        try:
            logger.info("AniList anime details: anime_id=%d", anime_id)
            return await self.client.get_anime_by_id(anime_id)
        except Exception as e:
            logger.error("AniList anime details failed: %s", e)
            raise

    async def get_trending_anime(
        self, limit: int = 20, page: int = 1
    ) -> List[Dict[str, Any]]:
        """Get trending anime from AniList.

        Args:
            limit: Maximum number of results
            page: Page number for pagination

        Returns:
            List of trending anime
        """
        try:
            logger.info("AniList trending: limit=%d, page=%d", limit, page)
            return await self.client.get_trending_anime(limit=limit, page=page)
        except Exception as e:
            logger.error("AniList trending failed: %s", e)
            raise

    async def get_upcoming_anime(
        self, limit: int = 20, page: int = 1
    ) -> List[Dict[str, Any]]:
        """Get upcoming anime from AniList.

        Args:
            limit: Maximum number of results
            page: Page number for pagination

        Returns:
            List of upcoming anime
        """
        try:
            logger.info("AniList upcoming: limit=%d, page=%d", limit, page)
            return await self.client.get_upcoming_anime(limit=limit, page=page)
        except Exception as e:
            logger.error("AniList upcoming failed: %s", e)
            raise

    async def get_popular_anime(
        self, limit: int = 20, page: int = 1
    ) -> List[Dict[str, Any]]:
        """Get popular anime from AniList.

        Args:
            limit: Maximum number of results
            page: Page number for pagination

        Returns:
            List of popular anime
        """
        try:
            logger.info("AniList popular: limit=%d, page=%d", limit, page)
            return await self.client.get_popular_anime(limit=limit, page=page)
        except Exception as e:
            logger.error("AniList popular failed: %s", e)
            raise

    async def get_staff_details(self, staff_id: int) -> Optional[Dict[str, Any]]:
        """Get staff member details.

        Args:
            staff_id: AniList staff ID

        Returns:
            Staff details or None if not found
        """
        try:
            logger.info("AniList staff details: staff_id=%d", staff_id)
            return await self.client.get_staff_by_id(staff_id)
        except Exception as e:
            logger.error("AniList staff details failed: %s", e)
            raise

    async def get_studio_details(self, studio_id: int) -> Optional[Dict[str, Any]]:
        """Get studio details.

        Args:
            studio_id: AniList studio ID

        Returns:
            Studio details or None if not found
        """
        try:
            logger.info("AniList studio details: studio_id=%d", studio_id)
            return await self.client.get_studio_by_id(studio_id)
        except Exception as e:
            logger.error("AniList studio details failed: %s", e)
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check service health status.

        Returns:
            Health status information
        """
        try:
            # Simple health check - try to get trending anime with limit 1
            await self.client.get_trending_anime(limit=1)

            return {
                "service": self.service_name,
                "status": "healthy",
                "circuit_breaker_open": self.circuit_breaker.is_open(),
            }

        except Exception as e:
            logger.warning("AniList health check failed: %s", e)
            return {
                "service": self.service_name,
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_open": self.circuit_breaker.is_open(),
            }
