"""AnimeSchedule.net service integration following modular pattern."""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...integrations.clients.animeschedule_client import AnimeScheduleClient
from ...integrations.error_handling import ErrorContext
from .base_service import BaseExternalService

logger = logging.getLogger(__name__)


class AnimeScheduleService(BaseExternalService):
    """AnimeSchedule.net service wrapper for anime schedule operations."""

    def __init__(self):
        """Initialize AnimeSchedule service with shared dependencies."""
        super().__init__(service_name="animeschedule")

        # Initialize AnimeSchedule client
        self.client = AnimeScheduleClient(
            circuit_breaker=self.circuit_breaker,
            cache_manager=self.cache_manager,
            error_handler=ErrorContext(
                user_message="AnimeSchedule service error",
                debug_info="AnimeSchedule.net API integration error",
            ),
        )

    async def get_today_timetable(
        self, timezone: Optional[str] = None, region: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get today's anime timetable.

        Args:
            timezone: Timezone for schedule (e.g., 'Asia/Tokyo')
            region: Region code (e.g., 'JP', 'US')

        Returns:
            Today's anime timetable

        Raises:
            Exception: If request fails
        """
        try:
            logger.info(
                "AnimeSchedule today timetable: timezone=%s, region=%s",
                timezone,
                region,
            )
            return await self.client.get_today_timetable(
                timezone=timezone, region=region
            )
        except Exception as e:
            logger.error("AnimeSchedule today timetable failed: %s", e)
            raise

    async def get_timetable_by_date(self, date: str) -> Dict[str, Any]:
        """Get anime timetable for specific date.

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            Anime timetable for the date

        Raises:
            ValueError: If date format is invalid
            Exception: If request fails
        """
        # Validate date format
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            raise ValueError(f"Invalid date format '{date}'. Use YYYY-MM-DD format.")

        try:
            # Additional validation - check if date is parseable
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date '{date}'. Use valid YYYY-MM-DD format.")

        try:
            logger.info("AnimeSchedule timetable by date: date=%s", date)
            return await self.client.get_timetable_by_date(date)
        except Exception as e:
            logger.error("AnimeSchedule timetable by date failed: %s", e)
            raise

    async def search_anime(self, query: str) -> List[Dict[str, Any]]:
        """Search for anime on AnimeSchedule.

        Args:
            query: Search query string

        Returns:
            List of anime search results

        Raises:
            Exception: If search fails
        """
        try:
            logger.info("AnimeSchedule search: query='%s'", query)
            return await self.client.search_anime(query)
        except Exception as e:
            logger.error("AnimeSchedule search failed: %s", e)
            raise

    async def get_seasonal_anime(self, season: str, year: int) -> Dict[str, Any]:
        """Get seasonal anime list.

        Args:
            season: Season (winter, spring, summer, fall)
            year: Year (must be between 2000 and current year + 1)

        Returns:
            Seasonal anime data

        Raises:
            ValueError: If season or year is invalid
            Exception: If request fails
        """
        # Validate season
        valid_seasons = ["winter", "spring", "summer", "fall"]
        if season.lower() not in valid_seasons:
            raise ValueError(
                f"Invalid season '{season}'. Must be one of: {valid_seasons}"
            )

        # Validate year
        current_year = datetime.now().year
        if year < 2000 or year > current_year + 1:
            raise ValueError(
                f"Invalid year {year}. Must be between 2000 and {current_year + 1}"
            )

        try:
            logger.info("AnimeSchedule seasonal: season=%s, year=%d", season, year)
            return await self.client.get_seasonal_anime(season.lower(), year)
        except Exception as e:
            logger.error("AnimeSchedule seasonal failed: %s", e)
            raise

    async def get_anime_details(self, anime_id: int) -> Optional[Dict[str, Any]]:
        """Get anime details by ID (alias for get_anime_schedule).

        Args:
            anime_id: AnimeSchedule anime ID

        Returns:
            Anime details/schedule or None if not found
        """
        return await self.get_anime_schedule(anime_id)

    async def get_anime_schedule(self, anime_id: int) -> Optional[Dict[str, Any]]:
        """Get anime schedule by ID.

        Args:
            anime_id: AnimeSchedule anime ID

        Returns:
            Anime schedule or None if not found
        """
        try:
            logger.info("AnimeSchedule anime schedule: anime_id=%d", anime_id)
            return await self.client.get_anime_schedule_by_id(anime_id)
        except Exception as e:
            logger.error("AnimeSchedule anime schedule failed: %s", e)
            raise

    async def get_streaming_platforms(self) -> List[Dict[str, Any]]:
        """Get available streaming platforms.

        Returns:
            List of streaming platforms
        """
        try:
            logger.info("AnimeSchedule streaming platforms")
            return await self.client.get_streaming_platforms()
        except Exception as e:
            logger.error("AnimeSchedule streaming platforms failed: %s", e)
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check service health status.

        Returns:
            Health status information
        """
        try:
            # Simple health check - check API compatibility
            await self.client.check_api_compatibility()

            return {
                "service": self.service_name,
                "status": "healthy",
                "circuit_breaker_open": self.circuit_breaker.is_open(),
            }

        except Exception as e:
            logger.warning("AnimeSchedule health check failed: %s", e)
            return {
                "service": self.service_name,
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_open": self.circuit_breaker.is_open(),
            }
