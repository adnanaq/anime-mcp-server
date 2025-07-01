"""Base service class for external anime API integrations."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ...integrations.cache_manager import CollaborativeCacheSystem
from ...integrations.error_handling import CircuitBreaker

logger = logging.getLogger(__name__)


class BaseExternalService(ABC):
    """Base class for external anime service integrations."""

    def __init__(self, service_name: str):
        """Initialize base service with shared dependencies.

        Args:
            service_name: Name of the external service (e.g., 'anilist', 'mal')
        """
        self.service_name = service_name

        # Initialize shared dependencies
        self.cache_manager = CollaborativeCacheSystem()
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=300)

        logger.info("Initialized %s service", service_name)

    @abstractmethod
    async def search_anime(
        self, query: str, limit: int = 10, **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for anime in the external service.

        Args:
            query: Search query string
            limit: Maximum number of results
            **kwargs: Additional service-specific parameters

        Returns:
            List of anime search results
        """

    @abstractmethod
    async def get_anime_details(self, anime_id: Any) -> Optional[Dict[str, Any]]:
        """Get detailed anime information by ID.

        Args:
            anime_id: Service-specific anime identifier

        Returns:
            Anime details or None if not found
        """

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check service health status.

        Returns:
            Health status information
        """

    async def close(self):
        """Close service connections and clean up resources."""
        try:
            if hasattr(self, "client") and hasattr(self.client, "close"):
                await self.client.close()

            if hasattr(self.cache_manager, "close"):
                await self.cache_manager.close()

            logger.info("Closed %s service", self.service_name)

        except Exception as e:
            logger.warning("Error closing %s service: %s", self.service_name, e)

    def is_healthy(self) -> bool:
        """Check if service is currently healthy.

        Returns:
            True if service is healthy, False otherwise
        """
        return not self.circuit_breaker.is_open()

    def get_service_info(self) -> Dict[str, Any]:
        """Get basic service information.

        Returns:
            Service information dictionary
        """
        return {
            "name": self.service_name,
            "healthy": self.is_healthy(),
            "circuit_breaker_open": self.circuit_breaker.is_open(),
        }
