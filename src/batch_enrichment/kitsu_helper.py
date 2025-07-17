#!/usr/bin/env python3
"""
Kitsu Helper for AI Enrichment Integration

Simple data fetcher for Kitsu API without modifying existing kitsu_client.py
"""

import logging
from typing import Dict, Any, Optional, List
import aiohttp

logger = logging.getLogger(__name__)


class KitsuEnrichmentHelper:
    """Simple helper for Kitsu data fetching in AI enrichment pipeline."""
    
    def __init__(self):
        """Initialize Kitsu enrichment helper."""
        self.base_url = "https://kitsu.io/api/edge"
        
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make request to Kitsu API."""
        headers = {
            "Accept": "application/vnd.api+json",
            "Content-Type": "application/vnd.api+json",
        }
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"Kitsu API error: HTTP {response.status}")
                        return {}
        except Exception as e:
            logger.error(f"Kitsu API request failed: {e}")
            return {}
    
    async def get_anime_by_id(self, anime_id: int) -> Optional[Dict[str, Any]]:
        """Get anime by Kitsu ID."""
        try:
            response = await self._make_request(f"/anime/{anime_id}")
            return response.get("data") if response else None
        except Exception as e:
            logger.error(f"Kitsu get_anime_by_id failed for ID {anime_id}: {e}")
            return None
    
    async def get_anime_episodes(self, anime_id: int) -> List[Dict[str, Any]]:
        """Get anime episodes by Kitsu ID."""
        try:
            response = await self._make_request(f"/anime/{anime_id}/episodes")
            return response.get("data", []) if response else []
        except Exception as e:
            logger.error(f"Kitsu get_anime_episodes failed for ID {anime_id}: {e}")
            return []
    
    async def get_anime_categories(self, anime_id: int) -> List[Dict[str, Any]]:
        """Get anime categories by Kitsu ID."""
        try:
            response = await self._make_request(f"/anime/{anime_id}/categories")
            return response.get("data", []) if response else []
        except Exception as e:
            logger.error(f"Kitsu get_anime_categories failed for ID {anime_id}: {e}")
            return []
    
    async def fetch_all_data(self, anime_id: int) -> Dict[str, Any]:
        """Fetch all Kitsu data for an anime ID."""
        try:
            # Fetch all data concurrently
            import asyncio
            
            results = await asyncio.gather(
                self.get_anime_by_id(anime_id),
                self.get_anime_episodes(anime_id),
                self.get_anime_categories(anime_id),
                return_exceptions=True
            )
            
            # Handle exceptions and assign results
            anime_data, episodes_data, categories_data = results
            
            return {
                "anime": anime_data if not isinstance(anime_data, Exception) else None,
                "episodes": episodes_data if not isinstance(episodes_data, Exception) else [],
                "categories": categories_data if not isinstance(categories_data, Exception) else []
            }
        except Exception as e:
            logger.error(f"Kitsu fetch_all_data failed for ID {anime_id}: {e}")
            return {
                "anime": None,
                "episodes": [],
                "categories": []
            }