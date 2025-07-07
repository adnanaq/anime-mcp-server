"""MAL service integration following modular pattern."""

import logging
from typing import Any, Dict, List, Optional

from ...integrations.clients.mal_client import MALClient
from ...integrations.error_handling import ErrorContext
from .base_service import BaseExternalService

logger = logging.getLogger(__name__)


class MALService(BaseExternalService):
    """MAL service wrapper for official MAL API v2 operations."""

    def __init__(
        self, client_id: str, client_secret: Optional[str] = None
    ):
        """Initialize MAL service with shared dependencies.

        Args:
            client_id: MAL OAuth2 client ID (required)
            client_secret: MAL OAuth2 client secret (optional)
        """
        super().__init__(service_name="mal")

        # Initialize MAL API v2 client
        self.client = MALClient(
            client_id=client_id,
            client_secret=client_secret,
            circuit_breaker=self.circuit_breaker,
            cache_manager=self.cache_manager,
            error_handler=ErrorContext(
                user_message="MAL service error",
                debug_info="MAL API v2 integration error",
            ),
        )

    async def search_anime(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        fields: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for anime on MAL API v2 with limited parameters.

        Note: MAL API v2 has very limited search parameters compared to Jikan.
        Most filtering is done on the response, not as query parameters.

        Args:
            query: Search query string (required)
            limit: Maximum number of results (1-100, default 10)
            offset: Offset for pagination (default 0)
            fields: Comma-separated list of response fields to include
            correlation_id: Correlation ID for request tracing

        Returns:
            List of anime search results

        Raises:
            ValueError: If parameter validation fails
            Exception: If search fails
        """
        if not query.strip():
            raise ValueError("Query parameter is required for MAL API v2 search")

        try:
            logger.info(
                "MAL search: query='%s', limit=%d, correlation_id=%s", 
                query, limit, correlation_id
            )
            
            return await self.client.search_anime(
                q=query,
                limit=limit,
                offset=offset,
                fields=fields,
                correlation_id=correlation_id,
            )
        except Exception as e:
            logger.error("MAL search failed: %s", e)
            raise

    async def get_anime_details(
        self, 
        anime_id: int, 
        fields: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get detailed anime information by ID.

        Args:
            anime_id: MAL anime ID
            fields: Comma-separated list of response fields to include
            correlation_id: Optional correlation ID for request tracing

        Returns:
            Anime details or None if not found
        """
        try:
            logger.info("MAL anime details: anime_id=%d, correlation_id=%s", anime_id, correlation_id)
            return await self.client.get_anime_by_id(
                anime_id=anime_id,
                fields=fields,
                correlation_id=correlation_id
            )
        except Exception as e:
            logger.error("MAL anime details failed: %s, correlation_id=%s", e, correlation_id)
            raise

    async def get_user_anime_list(
        self,
        username: str,
        status: Optional[str] = None,
        sort: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        fields: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get user's anime list from MAL.

        Args:
            username: MAL username
            status: Filter by status (watching, completed, on_hold, dropped, plan_to_watch)
            sort: Sort order (list_score, list_updated_at, anime_title, anime_start_date)
            limit: Maximum number of results (1-1000, default 100)
            offset: Offset for pagination
            fields: Comma-separated list of response fields
            correlation_id: Optional correlation ID for request tracing

        Returns:
            List of user's anime list entries
        """
        try:
            logger.info("MAL user anime list: username=%s, correlation_id=%s", username, correlation_id)
            return await self.client.get_user_anime_list(
                username=username,
                status=status,
                sort=sort,
                limit=limit,
                offset=offset,
                fields=fields,
                correlation_id=correlation_id,
            )
        except Exception as e:
            logger.error("MAL user anime list failed: %s, correlation_id=%s", e, correlation_id)
            raise

    async def get_anime_ranking(
        self,
        ranking_type: str = "all",
        limit: int = 100,
        offset: int = 0,
        fields: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get anime ranking from MAL.

        Args:
            ranking_type: Type of ranking (all, airing, upcoming, tv, ova, movie, special, bypopularity, favorite)
            limit: Maximum number of results (1-500, default 100)
            offset: Offset for pagination
            fields: Comma-separated list of response fields
            correlation_id: Optional correlation ID for request tracing

        Returns:
            List of ranked anime
        """
        try:
            logger.info("MAL anime ranking: type=%s, correlation_id=%s", ranking_type, correlation_id)
            return await self.client.get_anime_ranking(
                ranking_type=ranking_type,
                limit=limit,
                offset=offset,
                fields=fields,
            )
        except Exception as e:
            logger.error("MAL anime ranking failed: %s, correlation_id=%s", e, correlation_id)
            raise

    async def get_seasonal_anime(
        self,
        year: int,
        season: str,
        sort: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        fields: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get seasonal anime from MAL.

        Args:
            year: Year (e.g., 2024)
            season: Season (winter, spring, summer, fall)
            sort: Sort order (anime_score, anime_num_list_users)
            limit: Maximum number of results (1-500, default 100)
            offset: Offset for pagination
            fields: Comma-separated list of response fields
            correlation_id: Optional correlation ID for request tracing

        Returns:
            List of seasonal anime

        Raises:
            ValueError: If season is invalid
        """
        valid_seasons = ["winter", "spring", "summer", "fall"]
        if season.lower() not in valid_seasons:
            raise ValueError(
                f"Invalid season '{season}'. Must be one of: {valid_seasons}"
            )

        try:
            logger.info("MAL seasonal: year=%d, season=%s, correlation_id=%s", year, season, correlation_id)
            return await self.client.get_seasonal_anime(
                year=year,
                season=season.lower(),
                sort=sort,
                limit=limit,
                offset=offset,
                fields=fields,
                correlation_id=correlation_id,
            )
        except Exception as e:
            logger.error("MAL seasonal failed: %s, correlation_id=%s", e, correlation_id)
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check service health status without making unnecessary API calls.

        Returns:
            Health status information
        """
        try:
            # Check if client is properly initialized and circuit breaker is not open
            if not self.client:
                return {
                    "service": self.service_name,
                    "status": "unhealthy",
                    "error": "Client not initialized",
                    "circuit_breaker_open": self.circuit_breaker.is_open(),
                }
            
            # Check if required configuration is available
            if not hasattr(self.client, 'client_id') or not self.client.client_id:
                return {
                    "service": self.service_name,
                    "status": "unhealthy",
                    "error": "MAL client_id not configured",
                    "circuit_breaker_open": self.circuit_breaker.is_open(),
                }
            
            # If circuit breaker is open, service is degraded
            if self.circuit_breaker.is_open():
                return {
                    "service": self.service_name,
                    "status": "degraded",
                    "error": "Circuit breaker is open",
                    "circuit_breaker_open": True,
                }

            return {
                "service": self.service_name,
                "status": "healthy",
                "circuit_breaker_open": False,
                "base_url": getattr(self.client, 'base_url', 'unknown'),
                "auth_configured": bool(getattr(self.client, 'client_id', None)),
            }

        except Exception as e:
            logger.warning("MAL health check failed: %s", e)
            return {
                "service": self.service_name,
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_open": self.circuit_breaker.is_open(),
            }
