"""Kitsu JSON:API client implementation."""

from typing import Any, Dict, List, Optional

import aiohttp

from .base_client import BaseClient


class KitsuClient(BaseClient):
    """Kitsu JSON:API client."""

    def __init__(self, **kwargs):
        """Initialize Kitsu client."""
        super().__init__(service_name="kitsu", **kwargs)
        self.base_url = "https://kitsu.io/api/edge"

    async def _make_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make request to Kitsu API."""
        headers = {
            "Accept": "application/vnd.api+json",
            "Content-Type": "application/vnd.api+json",
        }

        url = f"{self.base_url}{endpoint}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 429:
                    retry_after = response.headers.get("Retry-After", "10")
                    raise Exception(
                        f"Kitsu rate limit exceeded. Retry after {retry_after} seconds"
                    )

                if response.status == 404:
                    error_data = await response.json()
                    if "errors" in error_data:
                        error_msg = error_data["errors"][0].get(
                            "title", "Record not found"
                        )
                        raise Exception(f"Kitsu API error: {error_msg}")

                if response.status != 200:
                    try:
                        error_data = await response.json()
                        if "errors" in error_data:
                            error_msg = error_data["errors"][0].get(
                                "title", "Unknown error"
                            )
                            raise Exception(f"Kitsu API error: {error_msg}")
                    except:
                        raise Exception(f"Kitsu API HTTP {response.status} error")

                return await response.json()

    async def get_anime_by_id(
        self, anime_id: int, includes: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get anime by ID."""
        # Check cache first
        cache_key = f"kitsu_anime_{anime_id}"
        if self.cache_manager:
            try:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    return cached_result
            except:
                # Cache miss or error, continue to API call
                pass

        # Check circuit breaker before making API call
        if self.circuit_breaker and self.circuit_breaker.is_open():
            raise Exception("Circuit breaker is open")

        try:
            endpoint = f"/anime/{anime_id}"
            params = {}
            if includes:
                params["include"] = ",".join(includes)

            response = await self._make_request(
                endpoint, params=params if params else None
            )
            if response and "data" in response:
                return response["data"]
        except Exception:
            pass

        return None

    async def search_anime(
        self,
        query: Optional[str] = None,
        categories: Optional[List[str]] = None,
        year: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Search anime using Kitsu API."""
        endpoint = "/anime"

        try:
            response = await self._make_request(endpoint)
            if response and "data" in response:
                return response["data"]
        except Exception:
            pass

        return []

    async def get_anime_episodes(self, anime_id: int) -> List[Dict[str, Any]]:
        """Get anime episodes using Kitsu API."""
        endpoint = f"/anime/{anime_id}/episodes"

        try:
            response = await self._make_request(endpoint)
            if response and "data" in response:
                return response["data"]
        except Exception:
            pass

        return []

    async def get_streaming_links(self, anime_id: int) -> List[Dict[str, Any]]:
        """Get anime streaming links using Kitsu API."""
        endpoint = f"/anime/{anime_id}/streaming-links"

        try:
            response = await self._make_request(endpoint)
            if response and "data" in response:
                return response["data"]
        except Exception:
            pass

        return []

    async def get_anime_characters(self, anime_id: int) -> List[Dict[str, Any]]:
        """Get anime characters using Kitsu API."""
        endpoint = f"/anime/{anime_id}/anime-characters"

        try:
            response = await self._make_request(endpoint)
            if response and "data" in response and "included" in response:
                # Parse JSON:API relationship data
                characters = []
                included_map = {
                    item["id"]: item
                    for item in response["included"]
                    if item["type"] == "characters"
                }

                for char_rel in response["data"]:
                    char_data = char_rel.copy()
                    if (
                        "relationships" in char_rel
                        and "character" in char_rel["relationships"]
                    ):
                        char_ref = char_rel["relationships"]["character"]["data"]
                        if char_ref["id"] in included_map:
                            char_data["character"] = included_map[char_ref["id"]][
                                "attributes"
                            ]
                            char_data["role"] = char_rel["attributes"]["role"]
                    characters.append(char_data)

                return characters
        except Exception:
            pass

        return []

    async def get_trending_anime(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get trending anime using Kitsu API."""
        endpoint = "/anime"
        params = {"sort": "-favoritesCount", "page[limit]": str(limit)}

        try:
            response = await self._make_request(endpoint, params=params)
            if response and "data" in response:
                return response["data"]
        except Exception:
            pass

        return []
