"""AnimeCountdown service integration following modular pattern."""

import logging
from typing import Any, Dict, List, Optional

from ...integrations.scrapers.extractors.animecountdown import AnimeCountdownScraper
from ...integrations.error_handling import ErrorContext
from .base_service import BaseExternalService

logger = logging.getLogger(__name__)


class AnimeCountdownService(BaseExternalService):
    """AnimeCountdown service wrapper for anime database operations."""
    
    def __init__(self):
        """Initialize AnimeCountdown service with shared dependencies."""
        super().__init__(service_name="animecountdown")
        
        # Initialize AnimeCountdown scraper
        self.scraper = AnimeCountdownScraper(
            circuit_breaker=self.circuit_breaker,
            cache_manager=self.cache_manager,
            error_handler=ErrorContext(
                user_message="AnimeCountdown service error",
                debug_info="AnimeCountdown.com scraping error"
            )
        )
    
    async def search_anime(self, query: str) -> List[Dict[str, Any]]:
        """Search for anime on AnimeCountdown.
        
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
            logger.info("AnimeCountdown search: query='%s'", query)
            return await self.scraper.search_anime(query.strip())
        except Exception as e:
            logger.error("AnimeCountdown search failed: %s", e)
            raise
    
    async def get_anime_details(self, anime_id: str) -> Optional[Dict[str, Any]]:
        """Get anime details by ID.
        
        Args:
            anime_id: AnimeCountdown anime ID/slug
            
        Returns:
            Anime details or None if not found
            
        Raises:
            ValueError: If anime ID is empty
            Exception: If request fails
        """
        if not anime_id or not anime_id.strip():
            raise ValueError("Anime ID cannot be empty")
        
        try:
            logger.info("AnimeCountdown anime details: anime_id='%s'", anime_id)
            return await self.scraper.get_anime_by_slug(anime_id.strip())
        except Exception as e:
            logger.error("AnimeCountdown anime details failed: %s", e)
            raise
    
    async def get_currently_airing(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Get currently airing anime with countdown information.
        
        Args:
            limit: Maximum number of results (1-100)
            
        Returns:
            List of currently airing anime with countdown data
            
        Raises:
            ValueError: If limit is invalid
            Exception: If request fails
        """
        if limit < 1 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")
        
        try:
            logger.info("AnimeCountdown currently airing: limit=%d", limit)
            return await self.scraper.get_currently_airing(limit)
        except Exception as e:
            logger.error("AnimeCountdown currently airing failed: %s", e)
            raise
    
    async def get_upcoming_anime(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get upcoming anime with countdown information.
        
        Args:
            limit: Maximum number of results (1-100)
            
        Returns:
            List of upcoming anime with countdown data
            
        Raises:
            ValueError: If limit is invalid
            Exception: If request fails
        """
        if limit < 1 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")
        
        try:
            logger.info("AnimeCountdown upcoming anime: limit=%d", limit)
            return await self.scraper.get_upcoming_anime(limit)
        except Exception as e:
            logger.error("AnimeCountdown upcoming anime failed: %s", e)
            raise
    
    async def get_popular_anime(self, time_period: str = "all_time", limit: int = 25) -> List[Dict[str, Any]]:
        """Get popular anime from AnimeCountdown.
        
        Args:
            time_period: Time period (all_time, this_year, this_month, this_week)
            limit: Maximum number of results (1-100)
            
        Returns:
            List of popular anime
            
        Raises:
            ValueError: If parameters are invalid
            Exception: If request fails
        """
        valid_periods = ["all_time", "this_year", "this_month", "this_week"]
        if time_period not in valid_periods:
            raise ValueError(f"Invalid time period '{time_period}'. Must be one of: {valid_periods}")
        
        if limit < 1 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")
        
        try:
            logger.info("AnimeCountdown popular anime: time_period='%s', limit=%d", time_period, limit)
            return await self.scraper.get_popular_anime(time_period, limit)
        except Exception as e:
            logger.error("AnimeCountdown popular anime failed: %s", e)
            raise
    
    async def get_anime_countdown(self, anime_id: str) -> Optional[Dict[str, Any]]:
        """Get specific countdown information for an anime.
        
        Args:
            anime_id: AnimeCountdown anime ID/slug
            
        Returns:
            Countdown information or None if not found
            
        Raises:
            ValueError: If anime ID is empty
            Exception: If request fails
        """
        if not anime_id or not anime_id.strip():
            raise ValueError("Anime ID cannot be empty")
        
        try:
            logger.info("AnimeCountdown countdown info: anime_id='%s'", anime_id)
            return await self.scraper.get_anime_countdown(anime_id.strip())
        except Exception as e:
            logger.error("AnimeCountdown countdown info failed: %s", e)
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health status.
        
        Returns:
            Health status information
        """
        try:
            # Simple health check - verify scraper is initialized
            if self.scraper and hasattr(self.scraper, 'base_url'):
                return {
                    "service": self.service_name,
                    "status": "healthy",
                    "circuit_breaker_open": self.circuit_breaker.is_open(),
                    "last_check": "success"
                }
            else:
                raise Exception("Scraper not properly initialized")
            
        except Exception as e:
            logger.warning("AnimeCountdown health check failed: %s", e)
            return {
                "service": self.service_name,
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_open": self.circuit_breaker.is_open()
            }