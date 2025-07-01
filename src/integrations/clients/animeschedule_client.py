"""AnimeSchedule.net REST client implementation."""

from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from .base_client import BaseClient


class AnimeScheduleClient(BaseClient):
    """AnimeSchedule.net API client."""

    def __init__(self, **kwargs):
        """Initialize AnimeSchedule client."""
        super().__init__(service_name="animeschedule", **kwargs)
        self.base_url = "https://animeschedule.net/api/v3"

    async def _make_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30
    ) -> Dict[str, Any]:
        """Make request to AnimeSchedule.net API."""
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
                    error_msg = error_data.get("error", "Anime not found")
                    raise Exception(f"AnimeSchedule API error: {error_msg}")

                if response.status >= 500:
                    error_data = await response.json()
                    error_msg = error_data.get("error", "Server error")
                    raise Exception(f"AnimeSchedule API server error: {error_msg}")

                if response.status != 200:
                    try:
                        error_data = await response.json()
                        error_msg = error_data.get("error", "Unknown error")
                        raise Exception(f"AnimeSchedule API error: {error_msg}")
                    except:
                        raise Exception(
                            f"AnimeSchedule API HTTP {response.status} error"
                        )

                return await response.json()

    async def get_today_timetable(
        self, timezone: Optional[str] = None, region: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get today's anime timetable."""
        # Check cache first
        cache_key = "animeschedule_timetable_today"
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
            endpoint = "/timetables"
            params = {}
            if timezone:
                params["timezone"] = timezone
            if region:
                params["region"] = region

            response = await self._make_request(
                endpoint, params=params if params else None
            )
            return response
        except Exception:
            return {"data": [], "meta": {}}

    async def get_timetable_by_date(self, date: str) -> Dict[str, Any]:
        """Get anime timetable for specific date."""
        try:
            validated_date = self._validate_date_format(date)
            endpoint = f"/timetables/{validated_date}"
            response = await self._make_request(endpoint)
            return response
        except Exception:
            return {"data": [], "meta": {}}

    async def search_anime(self, query: str) -> List[Dict[str, Any]]:
        """Search anime using AnimeSchedule.net API."""
        try:
            endpoint = "/anime/search"
            params = {"query": query}
            response = await self._make_request(endpoint, params=params)
            return response.get("data", [])
        except Exception:
            return []

    async def get_seasonal_anime(self, season: str, year: int) -> Dict[str, Any]:
        """Get seasonal anime using AnimeSchedule.net API."""
        try:
            endpoint = f"/seasons/{year}/{season}"
            response = await self._make_request(endpoint)
            return response
        except Exception:
            return {"data": [], "meta": {}}

    async def get_anime_schedule_by_id(self, anime_id: int) -> Optional[Dict[str, Any]]:
        """Get anime schedule by ID."""
        try:
            endpoint = f"/anime/{anime_id}"
            response = await self._make_request(endpoint)
            return response.get("data")
        except Exception:
            return None

    async def get_streaming_platforms(self) -> List[Dict[str, Any]]:
        """Get available streaming platforms."""
        try:
            endpoint = "/platforms"
            response = await self._make_request(endpoint)
            return response.get("data", [])
        except Exception:
            return []

    async def check_api_compatibility(self) -> Dict[str, Any]:
        """Check API version compatibility."""
        try:
            endpoint = "/version"
            response = await self._make_request(endpoint)
            return response
        except Exception:
            return {"compatible": False, "version": "unknown"}

    def _validate_date_format(self, date: str) -> str:
        """Validate date format (YYYY-MM-DD)."""
        try:
            datetime.strptime(date, "%Y-%m-%d")
            return date
        except ValueError:
            raise ValueError("Invalid date format. Expected YYYY-MM-DD")
