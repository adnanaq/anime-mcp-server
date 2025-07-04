"""AniSearch service integration following modular pattern."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...integrations.error_handling import ErrorContext
from ...integrations.scrapers.extractors.anisearch_scraper import AniSearchScraper
from .base_service import BaseExternalService

logger = logging.getLogger(__name__)


class AniSearchService(BaseExternalService):
    """AniSearch service wrapper for anime database operations."""

    def __init__(self):
        """Initialize AniSearch service with shared dependencies."""
        super().__init__(service_name="anisearch")

        # Initialize AniSearch scraper
        self.scraper = AniSearchScraper(
            circuit_breaker=self.circuit_breaker,
            cache_manager=self.cache_manager,
            error_handler=ErrorContext(
                user_message="AniSearch service error",
                debug_info="AniSearch.com scraping error",
            ),
        )

    async def search_anime(self, query: str) -> List[Dict[str, Any]]:
        """Search for anime on AniSearch.

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
            logger.info("AniSearch search: query='%s'", query)
            return await self.scraper.search_anime(query.strip())
        except Exception as e:
            logger.error("AniSearch search failed: %s", e)
            raise

    async def get_anime_details(self, anime_id: str) -> Optional[Dict[str, Any]]:
        """Get anime details by ID.

        Args:
            anime_id: AniSearch anime ID/slug

        Returns:
            Anime details or None if not found

        Raises:
            ValueError: If anime ID is empty
            Exception: If request fails
        """
        if not anime_id or not anime_id.strip():
            raise ValueError("Anime ID cannot be empty")

        try:
            logger.info("AniSearch anime details: anime_id='%s'", anime_id)
            # Try to convert to int first for numeric IDs, otherwise use as slug
            try:
                numeric_id = int(anime_id.strip())
                return await self.scraper.get_anime_by_id(numeric_id)
            except ValueError:
                # Not a number, treat as slug
                return await self.scraper.get_anime_by_slug(anime_id.strip())
        except Exception as e:
            logger.error("AniSearch anime details failed: %s", e)
            raise

    async def get_anime_characters(self, anime_id: str) -> List[Dict[str, Any]]:
        """Get anime characters by anime ID.

        Args:
            anime_id: AniSearch anime ID/slug

        Returns:
            List of anime characters

        Raises:
            ValueError: If anime ID is empty
            Exception: If request fails
        """
        if not anime_id or not anime_id.strip():
            raise ValueError("Anime ID cannot be empty")

        try:
            logger.info("AniSearch anime characters: anime_id='%s'", anime_id)
            return await self.scraper.get_anime_characters(anime_id.strip())
        except Exception as e:
            logger.error("AniSearch anime characters failed: %s", e)
            raise

    async def get_anime_recommendations(
        self, anime_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get anime recommendations.

        Args:
            anime_id: AniSearch anime ID/slug
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
            logger.info(
                "AniSearch recommendations: anime_id='%s', limit=%d", anime_id, limit
            )
            return await self.scraper.get_anime_recommendations(anime_id.strip(), limit)
        except Exception as e:
            logger.error("AniSearch recommendations failed: %s", e)
            raise

    async def get_top_anime(
        self, category: str = "highest_rated", limit: int = 25
    ) -> List[Dict[str, Any]]:
        """Get top anime by category.

        Args:
            category: Category type (highest_rated, most_popular, newest, most_watched)
            limit: Maximum number of results (1-100)

        Returns:
            List of top anime

        Raises:
            ValueError: If parameters are invalid
            Exception: If request fails
        """
        valid_categories = [
            "highest_rated",
            "most_popular",
            "newest",
            "most_watched",
            "most_favorites",
            "recently_updated",
        ]

        if category not in valid_categories:
            raise ValueError(
                f"Invalid category '{category}'. Must be one of: {valid_categories}"
            )

        if limit < 1 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")

        try:
            logger.info("AniSearch top anime: category='%s', limit=%d", category, limit)
            return await self.scraper.get_top_anime(category, limit)
        except Exception as e:
            logger.error("AniSearch top anime failed: %s", e)
            raise

    async def get_seasonal_anime(
        self, year: Optional[int] = None, season: Optional[str] = None, limit: int = 25
    ) -> List[Dict[str, Any]]:
        """Get seasonal anime from AniSearch.

        Args:
            year: Year (defaults to current year)
            season: Season (spring, summer, fall, winter) (defaults to current season)
            limit: Maximum number of results (1-100)

        Returns:
            List of seasonal anime

        Raises:
            ValueError: If parameters are invalid
            Exception: If request fails
        """
        # Default to current year and season if not specified
        if year is None:
            year = datetime.now().year

        if season is None:
            # Determine current season based on month
            month = datetime.now().month
            if month in [3, 4, 5]:
                season = "spring"
            elif month in [6, 7, 8]:
                season = "summer"
            elif month in [9, 10, 11]:
                season = "fall"
            else:
                season = "winter"

        # Validate parameters
        current_year = datetime.now().year
        if year < 1900 or year > current_year + 2:
            raise ValueError(f"Year must be between 1900 and {current_year + 2}")

        valid_seasons = ["spring", "summer", "fall", "winter"]
        if season.lower() not in valid_seasons:
            raise ValueError(
                f"Invalid season '{season}'. Must be one of: {valid_seasons}"
            )

        if limit < 1 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")

        try:
            logger.info(
                "AniSearch seasonal anime: year=%d, season='%s', limit=%d",
                year,
                season,
                limit,
            )
            return await self.scraper.get_seasonal_anime(year, season.lower(), limit)
        except Exception as e:
            logger.error("AniSearch seasonal anime failed: %s", e)
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check service health status.

        Returns:
            Health status information
        """
        try:
            # Simple health check - verify scraper is initialized
            if self.scraper and hasattr(self.scraper, "base_url"):
                return {
                    "service": self.service_name,
                    "status": "healthy",
                    "circuit_breaker_open": self.circuit_breaker.is_open(),
                    "last_check": "success",
                }
            else:
                raise Exception("Scraper not properly initialized")

        except Exception as e:
            logger.warning("AniSearch health check failed: %s", e)
            return {
                "service": self.service_name,
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_open": self.circuit_breaker.is_open(),
            }

    async def get_popular_anime(
        self, time_period: str = "all_time", limit: int = 25
    ) -> List[Dict[str, Any]]:
        """Get popular anime from AniSearch - use search with popular terms."""
        try:
            logger.info(
                "AniSearch popular anime: time_period='%s', limit=%d",
                time_period,
                limit,
            )

            # Since AniSearch doesn't have explicit popular lists,
            # search for popular anime terms and return results
            popular_terms = ["popular", "top rated", "best anime", "trending"]
            all_results = []

            for term in popular_terms:
                if len(all_results) >= limit:
                    break
                results = await self.search_anime(term)
                all_results.extend(results)

            # Remove duplicates and limit
            seen_titles = set()
            unique_results = []
            for anime in all_results:
                title = anime.get("title", "").lower()
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_results.append(anime)
                if len(unique_results) >= limit:
                    break

            return unique_results[:limit]

        except Exception as e:
            logger.error("AniSearch popular anime failed: %s", e)
            raise
