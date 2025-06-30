"""MAL/Jikan service integration following modular pattern."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...integrations.clients.mal_client import MALClient
from ...integrations.error_handling import ErrorContext
from .base_service import BaseExternalService

logger = logging.getLogger(__name__)


class MALService(BaseExternalService):
    """MAL/Jikan service wrapper for anime data operations."""
    
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        """Initialize MAL service with shared dependencies.
        
        Args:
            client_id: Optional MAL OAuth2 client ID
            client_secret: Optional MAL OAuth2 client secret
        """
        super().__init__(service_name="mal")
        
        # Initialize MAL/Jikan client
        self.client = MALClient(
            client_id=client_id,
            client_secret=client_secret,
            circuit_breaker=self.circuit_breaker,
            cache_manager=self.cache_manager,
            error_handler=ErrorContext(
                user_message="MAL service error",
                debug_info="MAL/Jikan API integration error"
            )
        )
    
    async def search_anime(
        self, 
        query: str, 
        limit: int = 10,
        status: Optional[str] = None,
        genres: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Search for anime on MAL/Jikan.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            status: Anime status filter (airing, complete, upcoming)
            genres: List of genre IDs to filter by
            
        Returns:
            List of anime search results
            
        Raises:
            Exception: If search fails
        """
        try:
            logger.info("MAL search: query='%s', limit=%d, status=%s", query, limit, status)
            return await self.client.search_anime(
                query=query, 
                limit=limit, 
                status=status, 
                genres=genres
            )
        except Exception as e:
            logger.error("MAL search failed: %s", e)
            raise
    
    async def get_anime_details(self, anime_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed anime information by ID.
        
        Args:
            anime_id: MAL anime ID
            
        Returns:
            Anime details or None if not found
        """
        try:
            logger.info("MAL anime details: anime_id=%d", anime_id)
            return await self.client.get_anime_by_id(anime_id)
        except Exception as e:
            logger.error("MAL anime details failed: %s", e)
            raise
    
    async def get_seasonal_anime(self, year: int, season: str) -> List[Dict[str, Any]]:
        """Get seasonal anime from MAL/Jikan.
        
        Args:
            year: Year (e.g., 2024)
            season: Season (winter, spring, summer, fall)
            
        Returns:
            List of seasonal anime
            
        Raises:
            ValueError: If season is invalid
        """
        valid_seasons = ["winter", "spring", "summer", "fall"]
        if season.lower() not in valid_seasons:
            raise ValueError(f"Invalid season '{season}'. Must be one of: {valid_seasons}")
        
        try:
            logger.info("MAL seasonal: year=%d, season=%s", year, season)
            return await self.client.get_seasonal_anime(year, season.lower())
        except Exception as e:
            logger.error("MAL seasonal failed: %s", e)
            raise
    
    async def get_current_season(self) -> List[Dict[str, Any]]:
        """Get current season anime.
        
        Returns:
            List of current season anime
        """
        # Determine current season
        now = datetime.now()
        month = now.month
        year = now.year
        
        if month in [12, 1, 2]:
            season = "winter"
        elif month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        else:  # month in [9, 10, 11]
            season = "fall"
        
        try:
            logger.info("MAL current season: year=%d, season=%s", year, season)
            return await self.client.get_seasonal_anime(year, season)
        except Exception as e:
            logger.error("MAL current season failed: %s", e)
            raise
    
    async def get_anime_statistics(self, anime_id: int) -> Dict[str, Any]:
        """Get anime statistics (watching, completed, etc.).
        
        Args:
            anime_id: MAL anime ID
            
        Returns:
            Statistics data
        """
        try:
            logger.info("MAL anime statistics: anime_id=%d", anime_id)
            return await self.client.get_anime_statistics(anime_id)
        except Exception as e:
            logger.error("MAL anime statistics failed: %s", e)
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health status.
        
        Returns:
            Health status information
        """
        try:
            # Simple health check - try to search with limit 1
            await self.client.search_anime(query="test", limit=1)
            
            return {
                "service": self.service_name,
                "status": "healthy",
                "circuit_breaker_open": self.circuit_breaker.is_open()
            }
            
        except Exception as e:
            logger.warning("MAL health check failed: %s", e)
            return {
                "service": self.service_name,
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_open": self.circuit_breaker.is_open()
            }