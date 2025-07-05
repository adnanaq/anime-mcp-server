"""Jikan (MAL Unofficial API) REST client implementation."""

from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from .base_client import BaseClient


class JikanClient(BaseClient):
    """Jikan (MAL Unofficial API) client."""

    def __init__(self, **kwargs):
        """Initialize Jikan client."""
        super().__init__(service_name="jikan", **kwargs)
        self.base_url = "https://api.jikan.moe/v4"

    async def _make_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30
    ) -> Dict[str, Any]:
        """Make request to Jikan API."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "AnimeMCP/1.0",
        }

        url = f"{self.base_url}{endpoint}"

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 404:
                    error_data = await response.json()
                    error_msg = error_data.get("message", "Anime not found")
                    raise Exception(f"Jikan API error: {error_msg}")

                if response.status == 429:
                    # Jikan has rate limiting
                    raise Exception("Jikan API rate limit exceeded. Please try again later.")

                if response.status >= 500:
                    error_data = await response.json()
                    error_msg = error_data.get("message", "Server error")
                    raise Exception(f"Jikan API server error: {error_msg}")

                if response.status != 200:
                    try:
                        error_data = await response.json()
                        error_msg = error_data.get("message", "Unknown error")
                        raise Exception(f"Jikan API error: {error_msg}")
                    except:
                        raise Exception(f"Jikan API HTTP {response.status} error")

                return await response.json()

    async def search_anime(self, query: str, **params) -> List[Dict[str, Any]]:
        """Search anime using Jikan API."""
        search_params = {"q": query, **params}
        
        # Clean up None values
        search_params = {k: v for k, v in search_params.items() if v is not None}
        
        response = await self._make_request("/anime", search_params)
        return response.get("data", [])

    async def get_anime(self, mal_id: int) -> Dict[str, Any]:
        """Get specific anime by MAL ID."""
        response = await self._make_request(f"/anime/{mal_id}")
        return response.get("data", {})

    async def get_anime_full(self, mal_id: int) -> Dict[str, Any]:
        """Get full anime data with statistics, relations, etc."""
        response = await self._make_request(f"/anime/{mal_id}/full")
        return response.get("data", {})

    async def get_seasonal_anime(self, year: int, season: str, **params) -> List[Dict[str, Any]]:
        """Get seasonal anime."""
        search_params = {**params}
        search_params = {k: v for k, v in search_params.items() if v is not None}
        
        response = await self._make_request(f"/seasons/{year}/{season}", search_params)
        return response.get("data", [])

    async def get_top_anime(self, **params) -> List[Dict[str, Any]]:
        """Get top anime."""
        search_params = {**params}
        search_params = {k: v for k, v in search_params.items() if v is not None}
        
        response = await self._make_request("/top/anime", search_params)
        return response.get("data", [])

    async def get_anime_genres(self) -> List[Dict[str, Any]]:
        """Get anime genres."""
        response = await self._make_request("/genres/anime")
        return response.get("data", [])

    async def get_anime_recommendations(self, mal_id: int) -> List[Dict[str, Any]]:
        """Get anime recommendations."""
        response = await self._make_request(f"/anime/{mal_id}/recommendations")
        return response.get("data", [])

    async def get_anime_characters(self, mal_id: int) -> List[Dict[str, Any]]:
        """Get anime characters."""
        response = await self._make_request(f"/anime/{mal_id}/characters")
        return response.get("data", [])

    async def get_anime_staff(self, mal_id: int) -> List[Dict[str, Any]]:
        """Get anime staff."""
        response = await self._make_request(f"/anime/{mal_id}/staff")
        return response.get("data", [])

    async def get_anime_statistics(self, mal_id: int) -> Dict[str, Any]:
        """Get anime statistics."""
        response = await self._make_request(f"/anime/{mal_id}/statistics")
        return response.get("data", {})