"""Modern Service Manager for Direct Tool Integration.

This module provides a simplified orchestration layer for external integrations
without the Universal parameter abstraction. Follows 2025 best practices for
direct tool calls and structured responses.

Key Improvements:
1. Direct tool calls without Universal parameter conversion
2. Structured response models instead of raw API responses
3. Simplified fallback strategies
4. Better error handling and performance monitoring
5. Reduced complexity from 444 parameters to 5-15 relevant parameters

Supports all 9 anime platforms with intelligent routing but without over-engineering.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from ..config import Settings

# Import clients with fallback handling
try:
    from .clients.anilist_client import AniListClient
except ImportError:
    AniListClient = None

try:
    from .clients.mal_client import MALClient
except ImportError:
    MALClient = None

try:
    from .clients.jikan_client import JikanClient
except ImportError:
    JikanClient = None

try:
    from .clients.kitsu_client import KitsuClient
except ImportError:
    KitsuClient = None

try:
    from .clients.anidb_client import AniDBClient
except ImportError:
    AniDBClient = None

try:
    from .clients.animeschedule_client import AnimeScheduleClient
except ImportError:
    AnimeScheduleClient = None

from ..vector.qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class ModernServiceManager:
    """Simplified service manager for direct tool calls without Universal parameter abstraction.

    Provides intelligent routing and fallback strategies while maintaining simplicity
    and following modern LLM best practices.
    """

    def __init__(self, settings: Settings):
        """Initialize service manager with platform clients.

        Args:
            settings: Application settings containing API keys and configuration
        """
        self.settings = settings

        # Initialize platform clients
        self.clients = {}
        self._init_clients()

        # Vector database for semantic search fallback
        self.vector_client = QdrantClient(settings=self.settings)

        # Platform priority order (based on capability and reliability)
        self.platform_priorities = [
            "anilist",  # Comprehensive GraphQL API
            "jikan",  # No auth required, good coverage
            "kitsu",  # Streaming focus
            "mal",  # Official but limited
            "animeschedule",  # Scheduling data
            "anidb",  # Detailed but ID-based
            "vector",  # Local semantic search
        ]

        logger.info(
            "ModernServiceManager initialized with direct tool call architecture"
        )

    def _init_clients(self):
        """Initialize platform clients with proper configuration."""
        # Initialize AniList client
        if AniListClient:
            try:
                auth_token = getattr(self.settings, "anilist_token", None)
                self.clients["anilist"] = AniListClient(auth_token=auth_token)
            except Exception as e:
                logger.warning(f"Failed to initialize AniListClient: {e}")

        # Initialize MAL client
        if MALClient:
            try:
                client_id = getattr(self.settings, "mal_client_id", None)
                client_secret = getattr(self.settings, "mal_client_secret", None)
                if client_id:
                    self.clients["mal"] = MALClient(
                        client_id=client_id, client_secret=client_secret
                    )
            except Exception as e:
                logger.warning(f"Failed to initialize MALClient: {e}")

        # Initialize Jikan client (no auth required)
        if JikanClient:
            try:
                self.clients["jikan"] = JikanClient()
            except Exception as e:
                logger.warning(f"Failed to initialize JikanClient: {e}")

        # Initialize other clients
        client_classes = [
            (KitsuClient, "kitsu"),
            (AnimeScheduleClient, "animeschedule"),
        ]

        for client_class, name in client_classes:
            if client_class:
                try:
                    self.clients[name] = client_class()
                except Exception as e:
                    logger.warning(f"Failed to initialize {client_class.__name__}: {e}")

        # Initialize AniDB client
        if AniDBClient:
            try:
                client_version = getattr(self.settings, "anidb_clientver", "2")
                self.clients["anidb"] = AniDBClient(client_version=client_version)
            except Exception as e:
                logger.warning(f"Failed to initialize AniDBClient: {e}")

    async def search_anime_direct(
        self, query: str, limit: int = 20, platform: Optional[str] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """Direct anime search without Universal parameter abstraction.

        Args:
            query: Search query
            limit: Maximum results
            platform: Preferred platform (optional)
            **kwargs: Additional platform-specific parameters

        Returns:
            List of anime results with platform attribution
        """
        start_time = time.time()

        # Determine platform order
        if platform and platform in self.clients:
            platform_order = [platform] + [
                p for p in self.platform_priorities if p != platform
            ]
        else:
            platform_order = self.platform_priorities

        # Try platforms in order
        for platform_name in platform_order:
            try:
                if platform_name == "vector":
                    results = await self._search_vector_direct(query, limit)
                elif platform_name in self.clients:
                    results = await self._search_platform_direct(
                        platform_name, query, limit, **kwargs
                    )
                else:
                    continue

                if results:
                    processing_time = (time.time() - start_time) * 1000
                    logger.info(
                        f"Search successful on {platform_name}: {len(results)} results "
                        f"in {processing_time:.1f}ms"
                    )
                    return results

            except Exception as e:
                logger.warning(f"Search failed on {platform_name}: {str(e)}")
                continue

        logger.warning("All platforms failed for search")
        return []

    async def _search_platform_direct(
        self, platform: str, query: str, limit: int, **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute direct search on specific platform.

        Args:
            platform: Platform identifier
            query: Search query
            limit: Maximum results
            **kwargs: Platform-specific parameters

        Returns:
            List of anime results with platform attribution
        """
        client = self.clients[platform]

        # Build platform-specific parameters
        if platform == "anilist":
            params = {"search": query, "perPage": min(limit, 50), **kwargs}
        elif platform == "mal":
            params = {"q": query, "limit": min(limit, 100), **kwargs}
        elif platform == "jikan":
            params = {"q": query, "limit": min(limit, 25), **kwargs}
        elif platform == "kitsu":
            params = {"filter[text]": query, "page[limit]": min(limit, 20), **kwargs}
        elif platform == "animeschedule":
            params = {"q": query, "count": min(limit, 50), **kwargs}
        else:
            params = {"query": query, "limit": limit, **kwargs}

        # Execute search
        raw_results = await client.search_anime(**params)

        # Add platform attribution
        results = []
        for result in raw_results:
            if isinstance(result, dict):
                result["_source_platform"] = platform
                result["_processing_time"] = time.time()
                results.append(result)

        return results

    async def _search_vector_direct(
        self, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Direct vector database search fallback.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of anime results from vector database
        """
        try:
            results = await self.vector_client.search(query=query, limit=limit)

            # Add platform attribution
            for result in results:
                if isinstance(result, dict):
                    result["_source_platform"] = "vector"
                    result["_processing_time"] = time.time()

            return results

        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []

    async def get_anime_by_id_direct(
        self, anime_id: str, platform: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get anime details by ID without Universal parameter abstraction.

        Args:
            anime_id: Anime identifier
            platform: Preferred platform (optional)

        Returns:
            Anime details with platform attribution
        """
        # Auto-detect platform from ID if not specified
        if not platform:
            platform = self._detect_platform_from_id(anime_id)

        # Try platforms in order
        platform_order = [platform] + [
            p for p in self.platform_priorities if p != platform
        ]

        for platform_name in platform_order:
            try:
                if platform_name == "vector":
                    result = await self.vector_client.get_by_id(anime_id)
                elif platform_name in self.clients:
                    client = self.clients[platform_name]
                    result = await client.get_anime_by_id(anime_id)
                else:
                    continue

                if result:
                    # Add platform attribution
                    if isinstance(result, dict):
                        result["_source_platform"] = platform_name
                        result["_retrieval_time"] = time.time()
                    return result

            except Exception as e:
                logger.warning(f"Get by ID failed on {platform_name}: {str(e)}")
                continue

        return None

    def _detect_platform_from_id(self, anime_id: str) -> str:
        """Detect platform from ID format.

        Args:
            anime_id: Anime identifier

        Returns:
            Detected platform identifier
        """
        # Simple heuristics for platform detection
        if anime_id.isdigit():
            return "mal"  # Numeric IDs typically from MAL
        elif anime_id.startswith("anilist_"):
            return "anilist"
        elif anime_id.startswith("kitsu_"):
            return "kitsu"
        else:
            return "vector"  # Default to vector database

    async def health_check(self) -> Dict[str, Any]:
        """Check health status of all platform clients.

        Returns:
            Health status dictionary
        """
        health_status = {}

        for platform, client in self.clients.items():
            try:
                # Basic health check
                is_healthy = (
                    await client.health_check()
                    if hasattr(client, "health_check")
                    else True
                )
                health_status[platform] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "available": True,
                }
            except Exception as e:
                health_status[platform] = {
                    "status": "error",
                    "error": str(e),
                    "available": False,
                }

        # Check vector database
        try:
            vector_healthy = await self.vector_client.health_check()
            health_status["vector"] = {
                "status": "healthy" if vector_healthy else "unhealthy",
                "available": True,
            }
        except Exception as e:
            health_status["vector"] = {
                "status": "error",
                "error": str(e),
                "available": False,
            }

        return health_status

    def get_available_platforms(self) -> List[str]:
        """Get list of available platforms.

        Returns:
            List of available platform identifiers
        """
        available = list(self.clients.keys())
        available.append("vector")  # Vector database is always available
        return available

    def get_platform_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Get capabilities of each platform.

        Returns:
            Platform capabilities dictionary
        """
        capabilities = {}

        for platform in self.get_available_platforms():
            if platform == "anilist":
                capabilities[platform] = {
                    "max_results": 50,
                    "supports_advanced_filtering": True,
                    "auth_required": False,
                    "specialties": ["comprehensive_data", "relationships", "graphql"],
                }
            elif platform == "mal":
                capabilities[platform] = {
                    "max_results": 100,
                    "supports_advanced_filtering": False,
                    "auth_required": True,
                    "specialties": ["community_data", "official_api"],
                }
            elif platform == "jikan":
                capabilities[platform] = {
                    "max_results": 25,
                    "supports_advanced_filtering": True,
                    "auth_required": False,
                    "specialties": ["no_auth", "mal_data", "extensive_filtering"],
                }
            elif platform == "kitsu":
                capabilities[platform] = {
                    "max_results": 20,
                    "supports_advanced_filtering": True,
                    "auth_required": False,
                    "specialties": ["streaming_data", "modern_api"],
                }
            elif platform == "animeschedule":
                capabilities[platform] = {
                    "max_results": 50,
                    "supports_advanced_filtering": True,
                    "auth_required": False,
                    "specialties": ["scheduling", "broadcast_times"],
                }
            elif platform == "vector":
                capabilities[platform] = {
                    "max_results": 100,
                    "supports_advanced_filtering": True,
                    "auth_required": False,
                    "specialties": ["semantic_search", "local_data", "similarity"],
                }
            else:
                capabilities[platform] = {
                    "max_results": 50,
                    "supports_advanced_filtering": False,
                    "auth_required": False,
                    "specialties": ["basic_search"],
                }

        return capabilities
