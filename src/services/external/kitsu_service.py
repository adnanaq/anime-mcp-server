"""Kitsu service integration following modular pattern."""

import logging
from typing import Any, Dict, List, Optional

from ...integrations.clients.kitsu_client import KitsuClient
from ...integrations.error_handling import ErrorContext
from .base_service import BaseExternalService

logger = logging.getLogger(__name__)


class KitsuService(BaseExternalService):
    """Kitsu service wrapper for anime data operations."""
    
    def __init__(self):
        """Initialize Kitsu service with shared dependencies."""
        super().__init__(service_name="kitsu")
        
        # Initialize Kitsu client
        self.client = KitsuClient(
            circuit_breaker=self.circuit_breaker,
            cache_manager=self.cache_manager,
            error_handler=ErrorContext(
                user_message="Kitsu service error",
                debug_info="Kitsu API integration error"
            )
        )
    
    async def search_anime(
        self, 
        query: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for anime on Kitsu.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of anime search results
            
        Raises:
            Exception: If search fails
        """
        try:
            logger.info("Kitsu search: query='%s', limit=%d", query, limit)
            return await self.client.search_anime(query=query, limit=limit)
        except Exception as e:
            logger.error("Kitsu search failed: %s", e)
            raise
    
    async def get_anime_details(self, anime_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed anime information by ID.
        
        Args:
            anime_id: Kitsu anime ID
            
        Returns:
            Anime details or None if not found
        """
        try:
            logger.info("Kitsu anime details: anime_id=%d", anime_id)
            return await self.client.get_anime_by_id(anime_id)
        except Exception as e:
            logger.error("Kitsu anime details failed: %s", e)
            raise
    
    async def get_trending_anime(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get trending anime from Kitsu.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of trending anime
        """
        try:
            logger.info("Kitsu trending: limit=%d", limit)
            return await self.client.get_trending_anime(limit=limit)
        except Exception as e:
            logger.error("Kitsu trending failed: %s", e)
            raise
    
    async def get_anime_episodes(self, anime_id: int) -> List[Dict[str, Any]]:
        """Get anime episodes list.
        
        Args:
            anime_id: Kitsu anime ID
            
        Returns:
            List of episodes
        """
        try:
            logger.info("Kitsu anime episodes: anime_id=%d", anime_id)
            return await self.client.get_anime_episodes(anime_id)
        except Exception as e:
            logger.error("Kitsu anime episodes failed: %s", e)
            raise
    
    async def get_streaming_links(self, anime_id: int) -> List[Dict[str, Any]]:
        """Get streaming links for anime.
        
        Args:
            anime_id: Kitsu anime ID
            
        Returns:
            List of streaming links
        """
        try:
            logger.info("Kitsu streaming links: anime_id=%d", anime_id)
            return await self.client.get_streaming_links(anime_id)
        except Exception as e:
            logger.error("Kitsu streaming links failed: %s", e)
            raise
    
    async def get_anime_characters(self, anime_id: int) -> List[Dict[str, Any]]:
        """Get anime characters list.
        
        Args:
            anime_id: Kitsu anime ID
            
        Returns:
            List of characters
        """
        try:
            logger.info("Kitsu anime characters: anime_id=%d", anime_id)
            return await self.client.get_anime_characters(anime_id)
        except Exception as e:
            logger.error("Kitsu anime characters failed: %s", e)
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
                "circuit_breaker_open": self.circuit_breaker.is_open()
            }
            
        except Exception as e:
            logger.warning("Kitsu health check failed: %s", e)
            return {
                "service": self.service_name,
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_open": self.circuit_breaker.is_open()
            }