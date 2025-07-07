"""Jikan REST client implementation."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from src.exceptions import APIError

from .base_client import BaseClient
from ..rate_limiting import JikanRateLimitAdapter
from ..rate_limiting.core import rate_limit_manager

logger = logging.getLogger(__name__)


class JikanClient(BaseClient):
    """Jikan REST API client for unofficial MAL data access."""

    def __init__(
        self,
        circuit_breaker=None,
        cache_manager=None,
        error_handler=None,
        execution_tracer=None,
    ):
        """Initialize Jikan client.

        Args:
            circuit_breaker: Circuit breaker instance
            cache_manager: Cache manager instance
            error_handler: Error handler instance
            execution_tracer: Execution tracer instance
        """
        super().__init__(
            "jikan",
            circuit_breaker,
            cache_manager,
            error_handler,
            execution_tracer,
        )
        self.base_url = "https://api.jikan.moe/v4"
        
        # Register platform-specific rate limiting adapter
        self._rate_limit_adapter = JikanRateLimitAdapter()
        rate_limit_manager.register_platform_adapter("jikan", self._rate_limit_adapter)

    async def _make_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Make request to Jikan API using rate-limited base client."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        url = f"{self.base_url}{endpoint}"

        return await self.make_request(
            url=url,
            method="GET",
            priority=1,
            endpoint=endpoint,
            headers=headers,
            params=params,
            **kwargs,
        )

    async def _make_request_with_retry(
        self, endpoint: str, max_retries: int = 3, **kwargs
    ) -> Dict[str, Any]:
        """Make Jikan request with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                return await self._make_request(endpoint, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                if "500" in str(e) or "502" in str(e) or "503" in str(e):
                    # Exponential backoff for server errors
                    await asyncio.sleep(2**attempt)
                    continue
                else:
                    # Don't retry for other errors
                    raise e

    async def get_anime_by_id(
        self,
        anime_id: int,
        *,
        correlation_id: Optional[str] = None,
        parent_correlation_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Get anime by ID with all available enhancements auto-enabled.

        This method automatically uses all available enhanced features:
        - Correlation logging (if correlation_logger available)
        - Execution tracing (if execution_tracer available)
        - Enhanced error contexts and graceful degradation
        - Dual API strategy (MAL → Jikan fallback)
        - Intelligent caching

        Args:
            anime_id: Anime ID to fetch
            correlation_id: Optional correlation ID for tracking
            parent_correlation_id: Optional parent correlation ID for chaining
            use_cache: Whether to use cache (default: True)

        Returns:
            Anime data or None if not found
        """
        # Auto-generate correlation ID if not provided
        if not correlation_id:
            import uuid
            correlation_id = f"jikan-anime-{uuid.uuid4().hex[:12]}"

        # Use correlation chaining if parent_correlation_id provided
        if parent_correlation_id:
            try:
                result = await self.make_request_with_correlation_chain(
                    url=f"{self.base_url}/anime/{anime_id}",
                    parent_correlation_id=parent_correlation_id,
                    method="GET",
                    endpoint=f"/anime/{anime_id}",
                )
                if result and "data" in result:
                    return result["data"]
                return None
            except Exception:
                # Fall through to regular handling if chaining fails
                pass

        # Start execution tracing if available
        trace_id = None
        if self.execution_tracer:
            trace_id = await self.execution_tracer.start_trace(
                operation="get_anime_by_id",
                context={
                    "anime_id": anime_id,
                    "service": "jikan",
                    "correlation_id": correlation_id,
                },
            )


        try:
            # Check cache first if enabled
            cache_key = f"jikan_anime_{anime_id}"
            if use_cache and self.cache_manager:
                try:
                    cached_result = await self.cache_manager.get(cache_key)
                    if cached_result:
                        logger.info(f"Cache hit for anime {anime_id}")
                        return cached_result
                except Exception as e:
                    logger.warning(f"Cache error for anime {anime_id}: {str(e)}")

            # Make Jikan API request
            if self.execution_tracer and trace_id:
                await self.execution_tracer.add_trace_step(
                    trace_id=trace_id,
                    step_name="jikan_api_request",
                    step_data={"endpoint": f"/anime/{anime_id}"},
                )

            endpoint = f"/anime/{anime_id}"
            response = await self._make_request(endpoint)

            if self.execution_tracer and trace_id:
                await self.execution_tracer.end_trace(
                    trace_id=trace_id,
                    status="success",
                    result={"source": "jikan", "anime_id": anime_id},
                )

            logger.info(f"Successfully retrieved anime {anime_id} from Jikan")

            if response and "data" in response:
                # Cache the result
                if use_cache and self.cache_manager:
                    try:
                        await self.cache_manager.set(cache_key, response["data"], ttl=3600)
                    except Exception as e:
                        logger.warning(f"Cache set error for anime {anime_id}: {str(e)}")
                
                return response["data"]

        except Exception as e:
            # Enhanced error handling
            if self.execution_tracer and trace_id:
                await self.execution_tracer.end_trace(
                    trace_id=trace_id, status="error", error=e
                )

            if correlation_id:
                await self.create_jikan_error_context(
                    error=e,
                    correlation_id=correlation_id,
                    endpoint=f"/anime/{anime_id}",
                    operation="get_anime_by_id",
                )
            else:
                logger.error(f"Jikan API failed for anime {anime_id}: {e}")

            # Try graceful degradation if available
            if correlation_id and hasattr(self, "handle_enhanced_graceful_degradation"):
                try:
                    async def primary_func():
                        raise e  # Re-raise the original error

                    degraded_result = await self.handle_enhanced_graceful_degradation(
                        primary_func=primary_func,
                        correlation_id=correlation_id,
                        context={
                            "operation": "get_anime_by_id",
                            "anime_id": anime_id,
                            "service": "jikan",
                        },
                        cache_key=cache_key,
                    )
                    return degraded_result.get("data")
                except Exception:
                    pass  # If degradation fails, continue to return None

        return None

    async def get_seasonal_anime(self, year: int, season: str) -> List[Dict[str, Any]]:
        """Get seasonal anime using Jikan API."""
        endpoint = f"/seasons/{year}/{season}"

        try:
            response = await self._make_request(endpoint)
            if response and "data" in response:
                logger.info(f"Retrieved {len(response['data'])} seasonal anime for {year} {season}")
                return response["data"]
        except Exception as e:
            logger.error(f"Jikan seasonal anime failed for {year} {season}: {e}")

        return []


    async def search_anime(
        self,
        q: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
        fields: Optional[str] = None,
        # Extended Jikan parameters
        genres: Optional[List[int]] = None,
        status: Optional[str] = None,
        # Enhanced Jikan parameters
        anime_type: Optional[str] = None,
        score: Optional[float] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        rating: Optional[str] = None,
        sfw: Optional[bool] = None,
        genres_exclude: Optional[List[int]] = None,
        order_by: Optional[str] = None,
        sort: Optional[str] = None,
        letter: Optional[str] = None,
        producers: Optional[List[int]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        page: Optional[int] = None,
        unapproved: Optional[bool] = None,
        *,
        correlation_id: Optional[str] = None,
        operation: str = "anime_search",
    ) -> List[Dict[str, Any]]:
        """Search anime with all Jikan parameters and enhancements auto-enabled.

        Args:
            q: Search query string
            limit: Maximum number of results (1-50)
            offset: Offset for pagination
            fields: Comma-separated list of response fields
            genres: List of genre IDs to include
            status: Anime status filter (airing, complete, upcoming)
            anime_type: Anime type filter (TV, Movie, OVA, ONA, Special)
            score: Exact score filter
            min_score: Minimum score filter (0.0-10.0)
            max_score: Maximum score filter (0.0-10.0)
            rating: Content rating (G, PG, PG-13, R, R+, Rx)
            sfw: Filter out adult content
            genres_exclude: List of genre IDs to exclude
            order_by: Order results by field
            sort: Sort direction (asc, desc)
            letter: Return entries starting with letter
            producers: List of producer/studio IDs
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            page: Page number for pagination
            unapproved: Include unapproved entries
            correlation_id: Optional correlation ID for tracking
            operation: Operation name for tracing

        Returns:
            List of anime results
        """
        # Auto-generate correlation ID if not provided
        if not correlation_id:
            import uuid
            correlation_id = f"jikan-search-{uuid.uuid4().hex[:12]}"

        # Start execution tracing if available
        trace_id = None
        if self.execution_tracer:
            trace_id = await self.execution_tracer.start_trace(
                operation=operation,
                context={
                    "query": q,
                    "service": "jikan",
                    "endpoint": "/anime",
                    "limit": limit,
                },
            )

        try:
            if self.execution_tracer and trace_id:
                await self.execution_tracer.add_trace_step(
                    trace_id=trace_id,
                    step_name="search_start",
                    step_data={"query": q, "genres": genres, "status": status},
                )

            # Build Jikan search parameters
            endpoint = "/anime"
            params = {}

            # Basic parameters
            if q:
                params["q"] = q
            if limit:
                params["limit"] = limit
            if page is not None:
                params["page"] = page
                
            # Jikan-specific parameters
            if genres:
                params["genres"] = ",".join(map(str, genres))
            if status:
                params["status"] = status
            
            # Enhanced Jikan parameters
            if anime_type:
                params["type"] = anime_type
            if score is not None:
                params["score"] = score
            if min_score is not None:
                params["min_score"] = min_score
            if max_score is not None:
                params["max_score"] = max_score
            if rating:
                params["rating"] = rating
            if sfw is not None:
                params["sfw"] = str(sfw).lower()
            if genres_exclude:
                params["genres_exclude"] = ",".join(map(str, genres_exclude))
            if order_by:
                params["order_by"] = order_by
            if sort:
                params["sort"] = sort
            if letter:
                params["letter"] = letter
            if producers:
                params["producers"] = ",".join(map(str, producers))
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date
            if page is not None:
                params["page"] = page
            if unapproved is not None:
                params["unapproved"] = str(unapproved).lower()

            try:
                response = await self._make_request(endpoint, params=params)
                if response and "data" in response:
                    result = response["data"]

                    if self.execution_tracer and trace_id:
                        await self.execution_tracer.end_trace(
                            trace_id=trace_id,
                            status="success",
                            result={"result_count": len(result), "query": q},
                        )

                    logger.info(f"Jikan search returned {len(result)} results for query: {q}")
                    return result
            except APIError as e:
                # Enhanced error handling for search
                if correlation_id:
                    await self.create_jikan_error_context(
                        error=e,
                        correlation_id=correlation_id,
                        endpoint=endpoint,
                        operation="search_anime",
                        query_params=params,
                    )
                else:
                    logger.error(f"Jikan search failed for query '{q}': {e}")
            except Exception as e:
                logger.error(f"Jikan search failed for query '{q}': {e}")

            return []

        except Exception as e:
            if self.execution_tracer and trace_id:
                await self.execution_tracer.end_trace(
                    trace_id=trace_id, status="error", error=e
                )
            raise


    async def create_jikan_error_context(
        self,
        error: Exception,
        correlation_id: str,
        endpoint: str,
        operation: str,
        query_params: Optional[Dict[str, Any]] = None,
    ) -> "ErrorContext":
        """Create enhanced ErrorContext for Jikan API errors.

        Args:
            error: Exception that occurred
            correlation_id: Correlation ID
            endpoint: Jikan API endpoint
            operation: Operation being performed
            query_params: Query parameters used

        Returns:
            Enhanced ErrorContext
        """
        from ..error_handling import ErrorSeverity

        # Determine severity and user message based on error
        if "429" in str(error) or "rate limit" in str(error):
            severity = ErrorSeverity.WARNING
            user_message = "Jikan API rate limit exceeded. Please wait before making more requests."
        elif "404" in str(error) or "not found" in str(error):
            severity = ErrorSeverity.INFO
            user_message = "The requested anime was not found."
        elif "500" in str(error) or "server" in str(error):
            severity = ErrorSeverity.ERROR
            user_message = (
                "Jikan API server error. The service may be temporarily unavailable."
            )
        else:
            severity = ErrorSeverity.ERROR
            user_message = "Jikan API request failed. Please try again."

        error_context = await self.create_error_context(
            error=error,
            correlation_id=correlation_id,
            user_message=user_message,
            trace_data={
                "api": "jikan",
                "endpoint": endpoint,
                "operation": operation,
                "query_params": query_params or {},
                "base_url": self.base_url,
            },
            severity=severity,
        )

        return error_context

    # PLATFORM-SPECIFIC RATE LIMITING OVERRIDES
    
    async def handle_rate_limit_response(self, response):
        """Handle rate limiting for Jikan API."""
        rate_info = self._rate_limit_adapter.extract_rate_limit_info(response)
        strategy = self._rate_limit_adapter.get_strategy()
        await strategy.handle_rate_limit_response(rate_info)
    
    async def monitor_rate_limits(self, response):
        """Monitor rate limits for Jikan API."""
        rate_info = self._rate_limit_adapter.extract_rate_limit_info(response)
        
        # Jikan-specific monitoring (no headers available)
        if rate_info.degraded:
            logger.warning("Jikan API rate limiting detected - no precise timing available")
    
    async def calculate_backoff_delay(self, response, attempt: int = 0):
        """Calculate backoff delay for Jikan API."""
        if not response:
            # Generic calculation when no response available
            import random
            base_delay = 2 ** attempt
            jitter = random.uniform(0.1, 0.3) * base_delay
            return base_delay + jitter
        
        rate_info = self._rate_limit_adapter.extract_rate_limit_info(response)
        strategy = self._rate_limit_adapter.get_strategy()
        return await strategy.calculate_backoff_delay(rate_info, attempt)


    async def get_anime_statistics(
        self,
        anime_id: int,
        *,
        correlation_id: Optional[str] = None,
        parent_correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get anime statistics with all available enhancements auto-enabled.

        Args:
            anime_id: Anime ID to get stats for
            correlation_id: Optional correlation ID for tracking
            parent_correlation_id: Optional parent correlation ID for chaining

        Returns:
            Anime statistics data
        """
        # Use correlation chaining if parent_correlation_id provided
        if parent_correlation_id:
            try:
                result = await self.make_request_with_correlation_chain(
                    url=f"{self.base_url}/anime/{anime_id}/statistics",
                    parent_correlation_id=parent_correlation_id,
                    method="GET",
                    endpoint=f"/anime/{anime_id}/statistics",
                )
                if result and "data" in result:
                    return result["data"]
                return {}
            except Exception:
                # Fall through to regular handling if chaining fails
                pass

        endpoint = f"/anime/{anime_id}/statistics"

        try:
            response = await self._make_request(endpoint)
            if response and "data" in response:
                return response["data"]
        except Exception as e:
            if correlation_id:
                await self.create_jikan_error_context(
                    error=e,
                    correlation_id=correlation_id,
                    endpoint=endpoint,
                    operation="get_anime_statistics",
                )
            else:
                logger.error(f"Jikan statistics failed for anime {anime_id}: {e}")

        return {}

    async def get_top_anime(
        self,
        anime_type: Optional[str] = None,
        filter_type: Optional[str] = None,
        rating: Optional[str] = None,
        page: Optional[int] = None,
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        """Get top anime rankings.

        Args:
            anime_type: Type filter (tv, movie, ova, special, ona, music)
            filter_type: Filter type (airing, upcoming, bypopularity, favorite)
            rating: Rating filter (g, pg, pg13, r17, r, rx)
            page: Page number
            limit: Results per page (max 25)

        Returns:
            List of top anime
        """
        endpoint = "/top/anime"
        params = {}

        if anime_type:
            params["type"] = anime_type
        if filter_type:
            params["filter"] = filter_type
        if rating:
            params["rating"] = rating
        if page is not None:
            params["page"] = page
        if limit:
            params["limit"] = min(limit, 25)  # Jikan max is 25

        try:
            response = await self._make_request(endpoint, params=params)
            if response and "data" in response:
                return response["data"]
        except Exception as e:
            logger.error(f"Jikan top anime failed: {e}")

        return []

    async def get_random_anime(self) -> Optional[Dict[str, Any]]:
        """Get random anime."""
        endpoint = "/random/anime"

        try:
            response = await self._make_request(endpoint)
            if response and "data" in response:
                return response["data"]
        except Exception as e:
            logger.error(f"Jikan random anime failed: {e}")

        return None

    async def get_anime_recommendations(self, anime_id: int) -> List[Dict[str, Any]]:
        """Get anime recommendations."""
        endpoint = f"/anime/{anime_id}/recommendations"

        try:
            response = await self._make_request(endpoint)
            if response and "data" in response:
                return response["data"]
        except Exception as e:
            logger.error(f"Jikan recommendations failed for anime {anime_id}: {e}")

        return []

    async def get_anime_characters(self, anime_id: int) -> List[Dict[str, Any]]:
        """Get anime characters."""
        endpoint = f"/anime/{anime_id}/characters"

        try:
            response = await self._make_request(endpoint)
            if response and "data" in response:
                return response["data"]
        except Exception as e:
            logger.error(f"Jikan characters failed for anime {anime_id}: {e}")

        return []

    async def get_anime_staff(self, anime_id: int) -> List[Dict[str, Any]]:
        """Get anime staff."""
        endpoint = f"/anime/{anime_id}/staff"

        try:
            response = await self._make_request(endpoint)
            if response and "data" in response:
                return response["data"]
        except Exception as e:
            logger.error(f"Jikan staff failed for anime {anime_id}: {e}")

        return []

    async def get_schedules(
        self,
        filter_day: Optional[str] = None,
        kids: Optional[bool] = None,
        sfw: Optional[bool] = None,
        unapproved: Optional[bool] = None,
        page: Optional[int] = None,
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        """Get broadcasting schedules."""
        endpoint = "/schedules"
        params = {}

        if filter_day:
            params["filter"] = filter_day
        if kids is not None:
            params["kids"] = str(kids).lower()
        if sfw is not None:
            params["sfw"] = str(sfw).lower()
        if unapproved is not None:
            params["unapproved"] = str(unapproved).lower()
        if page is not None:
            params["page"] = page
        if limit:
            params["limit"] = limit

        try:
            response = await self._make_request(endpoint, params=params)
            if response and "data" in response:
                return response["data"]
        except Exception as e:
            logger.error(f"Jikan schedules failed: {e}")

        return []

    async def get_genres(self, filter_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get available anime genres."""
        endpoint = "/genres/anime"
        params = {}

        if filter_name:
            params["filter"] = filter_name

        try:
            response = await self._make_request(endpoint, params=params)
            if response and "data" in response:
                return response["data"]
        except Exception as e:
            logger.error(f"Jikan genres failed: {e}")

        return []
