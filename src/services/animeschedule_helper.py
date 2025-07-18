#!/usr/bin/env python3
"""
AnimSchedule Helper for AI Enrichment Integration

Provides smart title matching and data extraction from AnimSchedule API
without modifying the existing animeschedule_client.py used by other services.
"""

import logging
from typing import Dict, Any, Optional, List
import aiohttp
import re

logger = logging.getLogger(__name__)


class AnimScheduleEnrichmentHelper:
    """Helper for AnimSchedule integration in AI enrichment pipeline."""
    
    def __init__(self):
        """Initialize AnimSchedule enrichment helper."""
        self.base_url = "https://animeschedule.net/api/v3"
        
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make request to AnimSchedule API."""
        headers = {
            "Accept": "application/json",
            "User-Agent": "AnimeMCP/1.0",
        }
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"AnimSchedule API error: HTTP {response.status}")
                        return {}
        except Exception as e:
            logger.error(f"AnimSchedule API request failed: {e}")
            return {}
    
    async def search_anime(self, query: str) -> List[Dict[str, Any]]:
        """Search anime using the correct AnimSchedule endpoint."""
        try:
            response = await self._make_request("/anime", {"q": query})
            return response.get("anime", [])
        except Exception as e:
            logger.error(f"AnimSchedule search failed for '{query}': {e}")
            return []
    
    async def get_anime_detail(self, route: str) -> Optional[Dict[str, Any]]:
        """Get detailed anime data by route/slug."""
        try:
            response = await self._make_request(f"/anime/{route}")
            return response if response else None
        except Exception as e:
            logger.error(f"AnimSchedule detail fetch failed for route '{route}': {e}")
            return None
    
    def _get_search_candidates(self, anime_data: Dict[str, Any]) -> List[str]:
        """Generate search term candidates from anime data."""
        candidates = []
        
        # Primary title
        candidates.append(anime_data['title'])
        
        # English synonyms (prioritize common English variants)
        synonyms = anime_data.get('synonyms', [])
        english_synonyms = []
        
        for synonym in synonyms:
            # Look for English-like titles (basic heuristic)
            if self._is_likely_english(synonym):
                english_synonyms.append(synonym)
        
        # Add top 2 English synonyms
        candidates.extend(english_synonyms[:2])
        
        # Base title (remove season/part info for fallback)
        base_title = self._get_base_title(anime_data['title'])
        if base_title != anime_data['title']:
            candidates.append(base_title)
        
        return candidates
    
    def _is_likely_english(self, text: str) -> bool:
        """Simple heuristic to identify English titles."""
        if not text:
            return False
        
        # Count ASCII letters vs non-ASCII characters
        ascii_chars = sum(1 for c in text if ord(c) < 128 and c.isalpha())
        total_chars = sum(1 for c in text if c.isalpha())
        
        if total_chars == 0:
            return False
        
        # Consider it English if >80% of letters are ASCII
        return (ascii_chars / total_chars) > 0.8
    
    def _get_base_title(self, title: str) -> str:
        """Extract base title by removing season/part information."""
        # Remove common season patterns
        patterns = [
            r'\s+2nd\s+Season\s*$',
            r'\s+Season\s+\d+\s*$', 
            r'\s+Part\s+\d+\s*$',
            r'\s+\d+期\s*$',  # Japanese season notation
            r':\s+.*$',  # Remove subtitle after colon
        ]
        
        base_title = title
        for pattern in patterns:
            base_title = re.sub(pattern, '', base_title, flags=re.IGNORECASE)
        
        return base_title.strip()
    
    def _is_good_match(self, anime_data: Dict[str, Any], search_result: Dict[str, Any]) -> bool:
        """Simple matching logic using year, episodes, and type."""
        
        # Year matching (most reliable)
        anime_year = anime_data.get('animeSeason', {}).get('year')
        search_year = search_result.get('year')
        
        if anime_year and search_year:
            if anime_year != search_year:
                return False
        
        # Episode count matching (when available)
        anime_episodes = anime_data.get('episodes')
        search_episodes = search_result.get('episodes')
        
        if anime_episodes and search_episodes:
            if anime_episodes != search_episodes:
                return False
        
        # Type matching (basic check)
        anime_type = anime_data.get('type', '').upper()
        search_media_types = search_result.get('mediaTypes', [])
        
        if anime_type and search_media_types:
            search_type_names = [mt.get('name', '').upper() for mt in search_media_types]
            if anime_type not in search_type_names:
                # Allow some flexibility (TV vs Television, etc.)
                if not (anime_type == 'TV' and any('TV' in t for t in search_type_names)):
                    return False
        
        return True
    
    async def find_anime_match(self, anime_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find matching AnimSchedule anime data using smart search strategy.
        
        Args:
            anime_data: Anime data from anime-offline-database
            
        Returns:
            Complete AnimSchedule anime data or None if no match found
        """
        search_candidates = self._get_search_candidates(anime_data)
        
        logger.info(f"Searching AnimSchedule for '{anime_data['title']}' with {len(search_candidates)} candidates")
        
        for i, candidate in enumerate(search_candidates):
            logger.debug(f"Trying search candidate {i+1}: '{candidate}'")
            
            search_results = await self.search_anime(candidate)
            
            if not search_results:
                continue
            
            # Check each result for a good match
            for result in search_results:
                if self._is_good_match(anime_data, result):
                    logger.info(f"Found match for '{anime_data['title']}' -> '{result['title']}' (route: {result['route']})")
                    
                    # Get detailed data
                    detailed_data = await self.get_anime_detail(result['route'])
                    if detailed_data:
                        return detailed_data
                    else:
                        logger.warning(f"Failed to get detailed data for route '{result['route']}'")
        
        logger.info(f"No AnimSchedule match found for '{anime_data['title']}'")
        return None