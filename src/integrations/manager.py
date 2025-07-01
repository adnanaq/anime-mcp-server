"""Integration manager for coordinating API clients and scrapers."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from ..config import Settings, get_settings
from .cache_manager import CollaborativeCacheSystem
from .clients.anidb_client import AniDBClient

# API Clients
from .clients.anilist_client import AniListClient
from .clients.animeschedule_client import AnimeScheduleClient
from .clients.kitsu_client import KitsuClient
from .clients.mal_client import MyAnimeListClient
from .error_handling import CircuitBreaker, ErrorContext, LangGraphErrorHandler

# Scrapers
from .scrapers.extractors.anime_planet import AnimePlanetScraper
from .scrapers.extractors.animecountdown import AnimeCountdownScraper

# LiveChartScraper removed - unreliable schedule data
from .scrapers.extractors.anisearch import AniSearchScraper

logger = logging.getLogger(__name__)


class IntegrationsManager:
    """Manages all anime data integrations (API clients and scrapers)."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize integrations manager.

        Args:
            settings: Application settings. If None, uses default settings.
        """
        self.settings = settings or get_settings()

        # Initialize shared dependencies
        self.cache_manager = CollaborativeCacheSystem()
        self.error_handler = LangGraphErrorHandler()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5, reset_timeout=300, half_open_max_calls=3
        )

        # Initialize clients and scrapers
        self._init_api_clients()
        self._init_scrapers()

        logger.info(
            "IntegrationsManager initialized with %d API clients and %d scrapers",
            len(self.api_clients),
            len(self.scrapers),
        )

    def _init_api_clients(self):
        """Initialize all API clients."""
        shared_deps = {
            "circuit_breaker": self.circuit_breaker,
            "cache_manager": self.cache_manager,
            "error_handler": ErrorContext(self.error_handler),
        }

        self.api_clients = {
            "anilist": AniListClient(**shared_deps),
            "mal": MyAnimeListClient(**shared_deps),
            "kitsu": KitsuClient(**shared_deps),
            "animeschedule": AnimeScheduleClient(**shared_deps),
            "anidb": AniDBClient(**shared_deps),
        }

    def _init_scrapers(self):
        """Initialize all scrapers."""
        shared_deps = {
            "circuit_breaker": self.circuit_breaker,
            "cache_manager": self.cache_manager,
            "error_handler": ErrorContext(self.error_handler),
            "rate_limiter": None,  # Will be set per scraper if needed
        }

        self.scrapers = {
            "animeplanet": AnimePlanetScraper(**shared_deps),
            # 'livechart': LiveChartScraper(**shared_deps), # Removed - unreliable schedule data
            "anisearch": AniSearchScraper(**shared_deps),
            "animecountdown": AnimeCountdownScraper(**shared_deps),
        }

    async def search_anime(
        self,
        query: str,
        limit: int = 10,
        sources: Optional[List[str]] = None,
        prefer_apis: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search for anime across multiple sources.

        Args:
            query: Search query
            limit: Maximum results per source
            sources: Specific sources to search (if None, searches all)
            prefer_apis: Whether to prefer API clients over scrapers

        Returns:
            Dict mapping source names to result lists
        """
        results = {}
        sources_to_search = sources or self._get_available_sources(prefer_apis)

        # Search API clients
        api_tasks = []
        for source in sources_to_search:
            if source in self.api_clients:
                client = self.api_clients[source]
                if hasattr(client, "search_anime"):
                    task = self._search_with_client(client, source, query, limit)
                    api_tasks.append(task)

        # Search scrapers
        scraper_tasks = []
        for source in sources_to_search:
            if source in self.scrapers:
                scraper = self.scrapers[source]
                if hasattr(scraper, "search_anime"):
                    task = self._search_with_scraper(scraper, source, query, limit)
                    scraper_tasks.append(task)
                elif hasattr(scraper, "search_anime_countdowns"):
                    task = self._search_countdowns_with_scraper(
                        scraper, source, query, limit
                    )
                    scraper_tasks.append(task)

        # Execute searches concurrently
        all_tasks = api_tasks + scraper_tasks
        if all_tasks:
            search_results = await asyncio.gather(*all_tasks, return_exceptions=True)

            for result in search_results:
                if isinstance(result, dict):
                    results.update(result)
                elif isinstance(result, Exception):
                    logger.warning("Search failed for one source: %s", result)

        return results

    async def get_anime_details(
        self,
        anime_id: Union[int, str],
        source: str,
        fallback_sources: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get detailed anime information from a specific source with fallbacks.

        Args:
            anime_id: Anime identifier (ID or slug)
            source: Primary source to query
            fallback_sources: Sources to try if primary fails

        Returns:
            Anime details or None if not found
        """
        # Try primary source
        result = await self._get_anime_from_source(anime_id, source)
        if result:
            return result

        # Try fallback sources
        if fallback_sources:
            for fallback_source in fallback_sources:
                result = await self._get_anime_from_source(anime_id, fallback_source)
                if result:
                    logger.info(
                        "Found anime details using fallback source: %s", fallback_source
                    )
                    return result

        return None

    async def get_trending_anime(
        self, limit: int = 20, sources: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get trending anime from multiple sources.

        Args:
            limit: Maximum results per source
            sources: Specific sources to query

        Returns:
            Dict mapping source names to trending anime lists
        """
        results = {}
        sources_to_query = sources or ["anilist", "mal", "kitsu"]

        tasks = []
        for source in sources_to_query:
            if source in self.api_clients:
                client = self.api_clients[source]
                if hasattr(client, "get_trending_anime"):
                    task = self._get_trending_with_client(client, source, limit)
                    tasks.append(task)

        if tasks:
            trending_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in trending_results:
                if isinstance(result, dict):
                    results.update(result)
                elif isinstance(result, Exception):
                    logger.warning("Trending query failed for one source: %s", result)

        return results

    async def get_upcoming_anime(
        self, limit: int = 20, include_countdowns: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get upcoming anime from multiple sources.

        Args:
            limit: Maximum results per source
            include_countdowns: Whether to include countdown sources

        Returns:
            Dict mapping source names to upcoming anime lists
        """
        results = {}

        # Get from API clients
        api_tasks = []
        for source, client in self.api_clients.items():
            if hasattr(client, "get_upcoming_anime"):
                task = self._get_upcoming_with_client(client, source, limit)
                api_tasks.append(task)

        # Get from countdown scrapers
        countdown_tasks = []
        if include_countdowns:
            for source, scraper in self.scrapers.items():
                if hasattr(scraper, "get_upcoming_anime"):
                    task = self._get_upcoming_with_scraper(scraper, source, limit)
                    countdown_tasks.append(task)

        # Execute all tasks
        all_tasks = api_tasks + countdown_tasks
        if all_tasks:
            upcoming_results = await asyncio.gather(*all_tasks, return_exceptions=True)

            for result in upcoming_results:
                if isinstance(result, dict):
                    results.update(result)
                elif isinstance(result, Exception):
                    logger.warning("Upcoming query failed for one source: %s", result)

        return results

    async def _search_with_client(
        self, client: Any, source: str, query: str, limit: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search using an API client."""
        try:
            results = await client.search_anime(query, limit=limit)
            return {source: results}
        except Exception as e:
            logger.warning("Search failed for %s: %s", source, e)
            return {source: []}

    async def _search_with_scraper(
        self, scraper: Any, source: str, query: str, limit: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search using a scraper."""
        try:
            results = await scraper.search_anime(query, limit=limit)
            return {source: results}
        except Exception as e:
            logger.warning("Scraper search failed for %s: %s", source, e)
            return {source: []}

    async def _search_countdowns_with_scraper(
        self, scraper: Any, source: str, query: str, limit: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search countdowns using a scraper."""
        try:
            results = await scraper.search_anime_countdowns(query, limit=limit)
            return {source: results}
        except Exception as e:
            logger.warning("Countdown search failed for %s: %s", source, e)
            return {source: []}

    async def _get_anime_from_source(
        self, anime_id: Union[int, str], source: str
    ) -> Optional[Dict[str, Any]]:
        """Get anime details from a specific source."""
        try:
            # Try API client first
            if source in self.api_clients:
                client = self.api_clients[source]
                if hasattr(client, "get_anime_by_id") and isinstance(anime_id, int):
                    return await client.get_anime_by_id(anime_id)
                elif hasattr(client, "get_anime") and isinstance(anime_id, str):
                    return await client.get_anime(anime_id)

            # Try scraper
            if source in self.scrapers:
                scraper = self.scrapers[source]
                if hasattr(scraper, "get_anime_by_slug") and isinstance(anime_id, str):
                    return await scraper.get_anime_by_slug(anime_id)
                elif hasattr(scraper, "get_anime_by_id") and isinstance(anime_id, int):
                    return await scraper.get_anime_by_id(anime_id)
                elif hasattr(scraper, "get_anime_countdown_by_slug") and isinstance(
                    anime_id, str
                ):
                    return await scraper.get_anime_countdown_by_slug(anime_id)
                elif hasattr(scraper, "get_anime_countdown_by_id") and isinstance(
                    anime_id, int
                ):
                    return await scraper.get_anime_countdown_by_id(anime_id)

        except Exception as e:
            logger.warning("Failed to get anime from %s: %s", source, e)

        return None

    async def _get_trending_with_client(
        self, client: Any, source: str, limit: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get trending anime using an API client."""
        try:
            results = await client.get_trending_anime(limit=limit)
            return {source: results}
        except Exception as e:
            logger.warning("Trending query failed for %s: %s", source, e)
            return {source: []}

    async def _get_upcoming_with_client(
        self, client: Any, source: str, limit: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get upcoming anime using an API client."""
        try:
            results = await client.get_upcoming_anime(limit=limit)
            return {source: results}
        except Exception as e:
            logger.warning("Upcoming query failed for %s: %s", source, e)
            return {source: []}

    async def _get_upcoming_with_scraper(
        self, scraper: Any, source: str, limit: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get upcoming anime using a scraper."""
        try:
            results = await scraper.get_upcoming_anime(limit=limit)
            return {source: results}
        except Exception as e:
            logger.warning("Upcoming scraper query failed for %s: %s", source, e)
            return {source: []}

    def _get_available_sources(self, prefer_apis: bool = True) -> List[str]:
        """Get list of available sources, optionally preferring APIs."""
        sources = []

        if prefer_apis:
            sources.extend(self.api_clients.keys())
            sources.extend(self.scrapers.keys())
        else:
            sources.extend(self.scrapers.keys())
            sources.extend(self.api_clients.keys())

        return sources

    async def health_check(self) -> Dict[str, Dict[str, bool]]:
        """Check health of all integrations.

        Returns:
            Dict with health status for each integration
        """
        health_status = {"api_clients": {}, "scrapers": {}}

        # Check API clients
        for name, client in self.api_clients.items():
            try:
                # Simple health check - try to make a basic request
                if hasattr(client, "health_check"):
                    health_status["api_clients"][name] = await client.health_check()
                else:
                    # Fallback: check if circuit breaker is open
                    health_status["api_clients"][
                        name
                    ] = not self.circuit_breaker.is_open()
            except Exception:
                health_status["api_clients"][name] = False

        # Check scrapers
        for name, scraper in self.scrapers.items():
            try:
                # Simple health check for scrapers
                health_status["scrapers"][name] = not self.circuit_breaker.is_open()
            except Exception:
                health_status["scrapers"][name] = False

        return health_status

    async def close(self):
        """Close all connections and clean up resources."""
        # Close API clients
        for client in self.api_clients.values():
            if hasattr(client, "close"):
                await client.close()

        # Close scrapers
        for scraper in self.scrapers.values():
            if hasattr(scraper, "close"):
                await scraper.close()

        # Close shared resources
        if hasattr(self.cache_manager, "close"):
            await self.cache_manager.close()

        logger.info("IntegrationsManager closed successfully")
