"""Service Manager for Universal Anime Search Integration.

This module provides the central orchestration point for all external integrations,
implementing intelligent parameter routing and multi-source execution with fallback strategies.

Key Responsibilities:
1. Parameter extraction and routing via MapperRegistry
2. Intelligent platform selection based on query characteristics  
3. Multi-source execution with comprehensive fallback chains
4. Result harmonization and source attribution
5. Performance monitoring and optimization

Supports all 9 anime platforms with seamless fallback between sources.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

from ..exceptions import (
    AllSourcesFailedException,
    AnimeServiceException,
    PlatformNotAvailableException,
)
from ..models.universal_anime import UniversalAnime, UniversalSearchParams, UniversalSearchResponse, UniversalSearchResult
# Import cache and error handling with fallbacks
try:
    from .cache_manager import CommunityCache
except ImportError:
    CommunityCache = None

try:
    from .error_handling import CircuitBreakerManager
except ImportError:
    CircuitBreakerManager = None
from .mapper_registry import MapperRegistry
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


class ServiceManager:
    """Central orchestration point for all external anime data integrations.
    
    Implements intelligent query routing, parameter extraction, multi-source execution,
    and comprehensive fallback strategies across all 9 anime platforms.
    """
    
    def __init__(self, settings: Any):
        """Initialize service manager with all required components.
        
        Args:
            settings: Application settings containing API keys and configuration
        """
        self.settings = settings
        
        # Initialize core components (with fallbacks)
        self.cache = CommunityCache(settings) if CommunityCache else None
        self.circuit_breakers = CircuitBreakerManager() if CircuitBreakerManager else None
        
        # Initialize platform clients with proper parameters
        self.clients = {}
        self._init_clients()
        
    def _init_clients(self):
        """Initialize all API clients with proper parameters."""
        # Shared dependencies for clients that support them
        shared_deps = {}
        if self.circuit_breakers:
            shared_deps["circuit_breaker"] = self.circuit_breakers
        if self.cache:
            shared_deps["cache_manager"] = self.cache
            
        # Initialize AniList client
        if AniListClient:
            try:
                # AniList client expects auth_token and shared deps
                auth_token = getattr(self.settings, 'anilist_token', None)
                self.clients["anilist"] = AniListClient(
                    auth_token=auth_token,
                    **shared_deps
                )
            except Exception as e:
                logger.warning(f"Failed to initialize AniListClient: {e}")
                
        # Initialize MAL client  
        if MALClient:
            try:
                # MAL client expects client_id, client_secret
                client_id = getattr(self.settings, 'mal_client_id', None)
                client_secret = getattr(self.settings, 'mal_client_secret', None)
                self.clients["mal"] = MALClient(
                    client_id=client_id,
                    client_secret=client_secret,
                    **shared_deps
                )
            except Exception as e:
                logger.warning(f"Failed to initialize MALClient: {e}")
                
        # Initialize other clients with shared deps only (no individual params needed)
        client_classes = [
            (KitsuClient, "kitsu"),
            (AnimeScheduleClient, "animeschedule"),
        ]
        
        for client_class, name in client_classes:
            if client_class:
                try:
                    self.clients[name] = client_class(**shared_deps)
                except Exception as e:
                    logger.warning(f"Failed to initialize {client_class.__name__}: {e}")
        
        # Initialize AniDB client (needs special parameters)
        if AniDBClient:
            try:
                client_version = getattr(self.settings, 'anidb_clientver', '2')
                self.clients["anidb"] = AniDBClient(
                    client_version=client_version,
                    **shared_deps
                )
            except Exception as e:
                logger.warning(f"Failed to initialize AniDBClient: {e}")
        
        # Vector database for fallback
        self.vector_client = QdrantClient(settings=self.settings)
        
        # Platform priority order (highest to lowest capability and reliability)
        self.platform_priorities = [
            "anilist",      # 70+ parameters, best filtering, international content
            "animeschedule", # 25+ parameters, comprehensive scheduling, exclude options
            "kitsu",        # Streaming platform support, range syntax
            "jikan",        # No auth, broad compatibility, good coverage
            "mal",          # Official API but limited parameters
            "anidb",        # Detailed data but ID-based only
            "vector",       # Local fallback
        ]
        
        # Mapper method names for each platform
        self.mapper_methods = {
            "anilist": "to_anilist_search_params",
            "mal": "to_mal_search_params", 
            "jikan": "to_jikan_search_params",
            "kitsu": "to_kitsu_search_params",
            "anidb": "to_anidb_search_params",
            "animeschedule": "to_animeschedule_search_params",
            "animeplanet": "to_animeplanet_search_params",
            "anisearch": "to_anisearch_search_params"
        }
        
        logger.info("ServiceManager initialized with 5 platform clients + vector fallback")
    
    async def search_anime_universal(
        self, 
        params: UniversalSearchParams,
        correlation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Execute universal anime search with intelligent routing and fallback.
        
        Args:
            params: Universal search parameters
            correlation_id: Request correlation ID for tracing
            
        Returns:
            UniversalSearchResponse with results and metadata
            
        Raises:
            AllSourcesFailedException: If all sources fail
        """
        start_time = time.time()
        
        try:
            # Step 1: Extract universal vs platform-specific parameters
            universal_params, platform_specific = MapperRegistry.extract_platform_params(
                **params.dict(exclude_none=True)
            )
            
            logger.info(
                f"Parameter extraction complete - Universal: {len(universal_params)}, "
                f"Platform-specific: {list(platform_specific.keys())}"
            )
            
            # Step 2: Intelligent platform selection
            selected_platform = MapperRegistry.auto_select_platform(
                universal_params, platform_specific
            )
            
            logger.info(f"Auto-selected platform: {selected_platform}")
            
            # Step 3: Execute with fallback chain
            results = await self.execute_with_fallback(
                selected_platform, 
                universal_params, 
                platform_specific,
                correlation_id
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Return raw results directly for LLM processing
            logger.info(f"Returning {len(results)} raw results for LLM processing")
            return results
            
        except Exception as e:
            logger.error(f"Universal search failed: {str(e)}")
            raise AnimeServiceException(f"Universal search failed: {str(e)}") from e
    
    async def execute_with_fallback(
        self,
        primary_platform: str,
        universal_params: Dict[str, Any],
        platform_specific: Dict[str, Dict[str, Any]],
        correlation_id: Optional[str] = None
    ) -> List[UniversalSearchResult]:
        """Execute search with comprehensive fallback strategy.
        
        Args:
            primary_platform: Primary platform to try first
            universal_params: Universal search parameters
            platform_specific: Platform-specific parameters by platform
            correlation_id: Request correlation ID for tracing
            
        Returns:
            List of search results from successful platform
            
        Raises:
            AllSourcesFailedException: If all platforms fail
        """
        # Create platform priority list starting with selected platform
        platform_order = [primary_platform] + [
            p for p in self.platform_priorities if p != primary_platform
        ]
        
        failures = []
        
        for platform in platform_order:
            try:
                logger.info(f"Attempting search on platform: {platform}")
                
                # Check circuit breaker (if available)
                if self.circuit_breakers and not self.circuit_breakers.can_execute(platform):
                    logger.warning(f"Circuit breaker open for {platform}, skipping")
                    failures.append(f"{platform}: circuit breaker open")
                    continue
                
                # Execute platform-specific search
                results = await self._execute_platform_search(
                    platform, universal_params, platform_specific, correlation_id
                )
                
                if results:
                    logger.info(f"Search successful on {platform}, got {len(results)} results")
                    if self.circuit_breakers:
                        self.circuit_breakers.record_success(platform)
                    return results
                else:
                    logger.warning(f"Search on {platform} returned no results")
                    failures.append(f"{platform}: no results")
                    
            except Exception as e:
                logger.error(f"Search failed on {platform}: {str(e)}")
                if self.circuit_breakers:
                    self.circuit_breakers.record_failure(platform)
                failures.append(f"{platform}: {str(e)}")
                continue
        
        # All platforms failed
        failure_summary = "; ".join(failures)
        raise AllSourcesFailedException(
            f"All platforms failed. Failures: {failure_summary}"
        )
    
    async def _execute_platform_search(
        self,
        platform: str,
        universal_params: Dict[str, Any],
        platform_specific: Dict[str, Dict[str, Any]],
        correlation_id: Optional[str] = None
    ) -> List[UniversalSearchResult]:
        """Execute search on specific platform.
        
        Args:
            platform: Platform identifier
            universal_params: Universal search parameters
            platform_specific: Platform-specific parameters
            correlation_id: Request correlation ID
            
        Returns:
            List of search results in universal format
        """
        if platform == "vector":
            return await self._search_vector_database(universal_params)
        
        if platform not in self.clients:
            raise PlatformNotAvailableException(f"Platform {platform} not available")
        
        client = self.clients[platform]
        
        # Get mapper for parameter conversion
        mapper = MapperRegistry.get_mapper(platform)
        
        # Convert parameters to platform-specific format
        mapper_method = self.mapper_methods.get(platform)
        if not mapper_method:
            raise ValueError(f"No mapper method defined for platform: {platform}")
            
        platform_method = getattr(mapper, mapper_method)
        
        # Convert universal_params dict back to UniversalSearchParams for mapper
        universal_params_obj = UniversalSearchParams(**universal_params)
        
        # Check if mapper method accepts platform_specific parameter
        import inspect
        sig = inspect.signature(platform_method)
        
        if len(sig.parameters) >= 3:  # cls, universal_params, platform_specific
            platform_params = platform_method(
                universal_params_obj, 
                platform_specific.get(platform, {})
            )
        else:  # cls, universal_params only
            platform_params = platform_method(universal_params_obj)
        
        # Execute platform search with unpacked parameters
        raw_results = await client.search_anime(**platform_params)
        
        # Return raw results for LLM to process
        # TODO: Universal format conversion will be implemented later
        raw_results_with_metadata = []
        for raw_result in raw_results:
            # Add platform metadata to raw result
            result_with_meta = {
                **raw_result,
                "_source": platform,
                "_relevance_score": 1.0  # Basic score for now
            }
            raw_results_with_metadata.append(result_with_meta)
        
        return raw_results_with_metadata
    
    async def _search_vector_database(
        self, universal_params: Dict[str, Any]
    ) -> List[UniversalSearchResult]:
        """Fallback search using local vector database.
        
        Args:
            universal_params: Universal search parameters
            
        Returns:
            List of search results from vector database
        """
        try:
            query = universal_params.get("query", "")
            limit = universal_params.get("limit", 20)
            
            if not query:
                logger.warning("Vector fallback requires query parameter")
                return []
            
            # Use vector database for semantic search
            vector_results = await self.vector_client.search(
                query=query,
                limit=limit
            )
            
            # Convert to universal format
            universal_results = []
            for result in vector_results:
                # Create UniversalAnime from vector result
                universal_anime = UniversalAnime(
                    id=result.get("anime_id", ""),
                    title=result.get("title", ""),
                    type_format=result.get("type", "TV"),
                    status="FINISHED",  # Default for vector results
                    episodes=result.get("episodes"),
                    score=result.get("score"),
                    year=result.get("year"),
                    genres=result.get("tags", []),
                    studios=result.get("studios", []),
                    description=result.get("synopsis"),
                    image_url=result.get("picture")
                )
                
                universal_results.append(UniversalSearchResult(
                    anime=universal_anime,
                    relevance_score=result.get("score", 0.5),
                    source="vector",
                    enrichment_sources=[]
                ))
            
            return universal_results
            
        except Exception as e:
            logger.error(f"Vector database fallback failed: {str(e)}")
            return []
    
    def _calculate_relevance_score(
        self, anime: UniversalAnime, search_params: Dict[str, Any]
    ) -> float:
        """Calculate relevance score for search result.
        
        Args:
            anime: Universal anime object
            search_params: Search parameters used
            
        Returns:
            Relevance score between 0 and 1
        """
        score = 0.5  # Base score
        
        # Boost score based on data quality
        if anime.data_quality_score:
            score += anime.data_quality_score * 0.2
        
        # Boost score based on user rating
        if anime.score and anime.score > 7.0:
            score += 0.1
        
        # Boost score if genres match search
        search_genres = search_params.get("genres", [])
        if search_genres and anime.genres:
            genre_matches = len(set(search_genres) & set(anime.genres))
            score += (genre_matches / len(search_genres)) * 0.2
        
        return min(1.0, score)
    
    async def get_anime_by_id_universal(
        self, 
        anime_id: str, 
        preferred_source: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> Optional[UniversalAnime]:
        """Get anime details by ID with intelligent source routing.
        
        Args:
            anime_id: Anime identifier (universal or platform-specific)
            preferred_source: Preferred data source platform
            correlation_id: Request correlation ID
            
        Returns:
            UniversalAnime object or None if not found
        """
        # Determine platform from ID format if not specified
        if not preferred_source:
            preferred_source = self._detect_platform_from_id(anime_id)
        
        # Try preferred source first, then fallback
        platform_order = [preferred_source] + [
            p for p in self.platform_priorities if p != preferred_source
        ]
        
        for platform in platform_order:
            try:
                if platform == "vector":
                    result = await self.vector_client.get_by_id(anime_id)
                    if result:
                        return UniversalAnime(**result)
                    continue
                
                if platform not in self.clients:
                    continue
                
                client = self.clients[platform]
                mapper = MapperRegistry.get_mapper(platform)
                
                raw_result = await client.get_anime_by_id(anime_id)
                if raw_result:
                    return mapper.to_universal_anime(raw_result)
                    
            except Exception as e:
                logger.warning(f"Failed to get anime {anime_id} from {platform}: {str(e)}")
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
            return "mal"  # Numeric IDs are typically MAL
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
                # Basic health check (implementation depends on client)
                is_healthy = await client.health_check() if hasattr(client, 'health_check') else True
                health_status[platform] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "circuit_breaker": self.circuit_breakers.get_status(platform) if self.circuit_breakers else "not_available"
                }
            except Exception as e:
                health_status[platform] = {
                    "status": "error",
                    "error": str(e),
                    "circuit_breaker": self.circuit_breakers.get_status(platform) if self.circuit_breakers else "not_available"
                }
        
        # Check vector database
        try:
            vector_healthy = await self.vector_client.health_check()
            health_status["vector"] = {
                "status": "healthy" if vector_healthy else "unhealthy"
            }
        except Exception as e:
            health_status["vector"] = {
                "status": "error", 
                "error": str(e)
            }
        
        return health_status