"""Anime-Planet service integration following modular pattern."""

import logging
from typing import Any, Dict, List, Optional

from ...integrations.scrapers.extractors.anime_planet import AnimePlanetScraper
from ...integrations.error_handling import ErrorContext
from .base_service import BaseExternalService

logger = logging.getLogger(__name__)


class AnimePlanetService(BaseExternalService):
    """Anime-Planet service wrapper for anime database operations."""
    
    def __init__(self):
        """Initialize Anime-Planet service with shared dependencies."""
        super().__init__(service_name="animeplanet")
        
        # Initialize Anime-Planet scraper
        self.scraper = AnimePlanetScraper(
            circuit_breaker=self.circuit_breaker,
            cache_manager=self.cache_manager,
            error_handler=ErrorContext(
                user_message="Anime-Planet service error",
                debug_info="Anime-Planet.com scraping error"
            )
        )
    
    async def search_anime(self, query: str) -> List[Dict[str, Any]]:
        """Search for anime on Anime-Planet.
        
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
            logger.info("Anime-Planet search: query='%s'", query)
            return await self.scraper.search_anime(query.strip())
        except Exception as e:
            logger.error("Anime-Planet search failed: %s", e)
            raise
    
    async def get_anime_details(self, anime_id: str) -> Optional[Dict[str, Any]]:
        """Get anime details by ID.
        
        Args:
            anime_id: Anime-Planet anime ID/slug
            
        Returns:
            Anime details or None if not found
            
        Raises:
            ValueError: If anime ID is empty
            Exception: If request fails
        """
        if not anime_id or not anime_id.strip():
            raise ValueError("Anime ID cannot be empty")
        
        try:
            logger.info("Anime-Planet anime details: anime_id='%s'", anime_id)
            return await self.scraper.get_anime_by_slug(anime_id.strip())
        except Exception as e:
            logger.error("Anime-Planet anime details failed: %s", e)
            raise
    
    async def get_anime_characters(self, anime_id: str) -> List[Dict[str, Any]]:
        """Get anime characters by anime ID.
        
        Args:
            anime_id: Anime-Planet anime ID/slug
            
        Returns:
            List of anime characters
            
        Raises:
            ValueError: If anime ID is empty
            Exception: If request fails
        """
        if not anime_id or not anime_id.strip():
            raise ValueError("Anime ID cannot be empty")
        
        try:
            logger.info("Anime-Planet anime characters: anime_id='%s'", anime_id)
            # Anime-Planet scraper doesn't implement character extraction yet
            # Return empty list for now
            return []
        except Exception as e:
            logger.error("Anime-Planet anime characters failed: %s", e)
            raise
    
    async def get_anime_recommendations(self, anime_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get anime recommendations.
        
        Args:
            anime_id: Anime-Planet anime ID/slug
            limit: Maximum number of recommendations (1-50)
            
        Returns:
            List of recommended anime
            
        Raises:
            ValueError: If parameters are invalid
            Exception: If request fails
        """
        if not anime_id or not anime_id.strip():
            raise ValueError("Anime ID cannot be empty")
        
        if limit < 1 or limit > 50:
            raise ValueError("Limit must be between 1 and 50")
        
        try:
            logger.info("Anime-Planet recommendations: anime_id='%s', limit=%d", anime_id, limit)
            # Anime-Planet scraper doesn't implement recommendations yet
            # Return empty list for now  
            return []
        except Exception as e:
            logger.error("Anime-Planet recommendations failed: %s", e)
            raise
    
    async def get_top_anime(self, category: str = "top-anime", limit: int = 25) -> List[Dict[str, Any]]:
        """Get top anime by category.
        
        Args:
            category: Category type (top-anime, most-watched, highest-rated, etc.)
            limit: Maximum number of results (1-100)
            
        Returns:
            List of top anime
            
        Raises:
            ValueError: If parameters are invalid
            Exception: If request fails
        """
        valid_categories = [
            "top-anime", "most-watched", "highest-rated", 
            "most-popular", "newest", "recently-updated"
        ]
        
        if category not in valid_categories:
            raise ValueError(f"Invalid category '{category}'. Must be one of: {valid_categories}")
        
        if limit < 1 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")
        
        try:
            logger.info("Anime-Planet top anime: category='%s', limit=%d", category, limit)
            # Anime-Planet scraper doesn't implement top anime lists yet
            # Return empty list for now
            return []
        except Exception as e:
            logger.error("Anime-Planet top anime failed: %s", e)
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
            logger.warning("Anime-Planet health check failed: %s", e)
            return {
                "service": self.service_name,
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_open": self.circuit_breaker.is_open()
            }