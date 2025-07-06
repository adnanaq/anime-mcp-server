"""MAL/Jikan REST client implementation."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from src.exceptions import APIError

from .base_client import BaseClient
from ..rate_limiting import MALRateLimitAdapter, JikanRateLimitAdapter
from ..rate_limiting.core import rate_limit_manager

logger = logging.getLogger(__name__)


class MALClient(BaseClient):
    """MAL/Jikan REST API client with dual API strategy."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        circuit_breaker=None,
        cache_manager=None,
        error_handler=None,
                execution_tracer=None,
    ):
        """Initialize MAL client.

        Args:
            client_id: MAL OAuth2 client ID
            client_secret: MAL OAuth2 client secret
            circuit_breaker: Circuit breaker instance
            cache_manager: Cache manager instance
            error_handler: Error handler instance
            execution_tracer: Execution tracer instance
        """
        super().__init__(
            "mal",
            circuit_breaker,
            cache_manager,
            error_handler,
            execution_tracer,
        )
        self.mal_base_url = "https://api.myanimelist.net/v2"
        self.jikan_base_url = "https://api.jikan.moe/v4"
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.refresh_token = None
        
        # Register platform-specific rate limiting adapters
        # Since this client handles both MAL and Jikan APIs, register both
        self._mal_rate_limit_adapter = MALRateLimitAdapter()
        self._jikan_rate_limit_adapter = JikanRateLimitAdapter()
        
        # Register with global rate limiter using dual service approach
        rate_limit_manager.register_platform_adapter("mal", self._mal_rate_limit_adapter)
        rate_limit_manager.register_platform_adapter("jikan", self._jikan_rate_limit_adapter)

    async def _make_mal_request(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make request to official MAL API using rate-limited base client."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        # Add X-MAL-CLIENT-ID header if client_id is available
        if self.client_id:
            headers["X-MAL-CLIENT-ID"] = self.client_id

        # Add Authorization header if access_token is available
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        elif not self.client_id:
            raise Exception("MAL API requires either client_id or access_token")

        url = f"{self.mal_base_url}{endpoint}"

        # Use inherited rate-limited make_request method
        return await self.make_request(
            url=url,
            method="GET",
            priority=1,
            endpoint=endpoint,
            headers=headers,
            **kwargs,
        )

    async def _make_jikan_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Make request to Jikan API using rate-limited base client."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        url = f"{self.jikan_base_url}{endpoint}"

        # Use inherited rate-limited make_request method
        return await self.make_request(
            url=url,
            method="GET",
            priority=1,
            endpoint=endpoint,
            headers=headers,
            params=params,
            **kwargs,
        )

    async def _make_jikan_request_with_retry(
        self, endpoint: str, max_retries: int = 3, **kwargs
    ) -> Dict[str, Any]:
        """Make Jikan request with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                return await self._make_jikan_request(endpoint, **kwargs)
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
        # Auto-generate correlation ID if not provided but we have correlation logger
        if not correlation_id:
            import uuid

            correlation_id = f"anime-{uuid.uuid4().hex[:12]}"

        # Use correlation chaining if parent_correlation_id provided
        if parent_correlation_id:
            try:
                result = await self.make_request_with_correlation_chain(
                    url=f"{self.jikan_base_url}/anime/{anime_id}",
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
                    "service": "mal",
                    "correlation_id": correlation_id,
                    "dual_api": True,
                },
            )


        try:
            # Check cache first if enabled
            cache_key = f"mal_anime_{anime_id}"
            if use_cache and self.cache_manager:
                try:
                    cached_result = await self.cache_manager.get(cache_key)
                    if cached_result:
                        if False:
                            await self.correlation_logger.log_with_correlation(
                                correlation_id=correlation_id,
                                level="info",
                                message=f"Cache hit for anime {anime_id}",
                                context={"service": "mal", "cache_key": cache_key},
                            )
                        return cached_result
                except Exception as e:
                    if False:
                        await self.correlation_logger.log_with_correlation(
                            correlation_id=correlation_id,
                            level="warning",
                            message=f"Cache error for anime {anime_id}: {str(e)}",
                            context={"service": "mal", "error": str(e)},
                        )

            # Try MAL API first if available
            if self.client_id:
                if self.execution_tracer and trace_id:
                    await self.execution_tracer.add_trace_step(
                        trace_id=trace_id,
                        step_name="mal_api_attempt",
                        step_data={"endpoint": f"/anime/{anime_id}", "api": "mal"},
                    )

                try:
                    endpoint = f"/anime/{anime_id}"
                    result = await self._make_mal_request(endpoint)

                    if self.execution_tracer and trace_id:
                        await self.execution_tracer.add_trace_step(
                            trace_id=trace_id,
                            step_name="mal_api_success",
                            step_data={"anime_id": anime_id},
                        )

                    if self.execution_tracer and trace_id:
                        await self.execution_tracer.end_trace(
                            trace_id=trace_id,
                            status="success",
                            result={"source": "mal", "anime_id": anime_id},
                        )

                    return result

                except APIError as e:
                    # Enhanced MAL error handling
                    if self.execution_tracer and trace_id:
                        await self.execution_tracer.add_trace_step(
                            trace_id=trace_id,
                            step_name="mal_api_failed",
                            step_data={"error": str(e), "fallback": "jikan"},
                        )

                    # Create enhanced error context if correlation_id available
                    if correlation_id:
                        await self._create_enhanced_error_context(
                            error=e,
                            correlation_id=correlation_id,
                            api="mal",
                            endpoint=endpoint,
                            operation="get_anime_by_id",
                        )
                    else:
                        # Fallback to basic error handling
                        logger.warning(
                            f"MAL API failed for anime {anime_id}, falling back to Jikan: {e}"
                        )

            # Fallback to Jikan API
            if self.execution_tracer and trace_id:
                await self.execution_tracer.add_trace_step(
                    trace_id=trace_id,
                    step_name="jikan_fallback",
                    step_data={"endpoint": f"/anime/{anime_id}", "api": "jikan"},
                )

            endpoint = f"/anime/{anime_id}"
            response = await self._make_jikan_request(endpoint)

            if self.execution_tracer and trace_id:
                await self.execution_tracer.end_trace(
                    trace_id=trace_id,
                    status="success",
                    result={"source": "jikan", "anime_id": anime_id},
                )

            if False:
                await self.correlation_logger.log_with_correlation(
                    correlation_id=correlation_id,
                    level="info",
                    message=f"Successfully retrieved anime {anime_id}",
                    context={
                        "service": "mal",
                        "anime_id": anime_id,
                        "source": "jikan",
                        "found": response is not None and "data" in response,
                    },
                )

            if response and "data" in response:
                return response["data"]

        except Exception as e:
            # Enhanced error handling
            if self.execution_tracer and trace_id:
                await self.execution_tracer.end_trace(
                    trace_id=trace_id, status="error", error=e
                )

            if correlation_id:
                # Use enhanced error context
                await self._create_enhanced_error_context(
                    error=e,
                    correlation_id=correlation_id,
                    api="jikan",
                    endpoint=f"/anime/{anime_id}",
                    operation="get_anime_by_id",
                )
            else:
                # Fallback to basic error logging
                logger.error(
                    f"Both MAL and Jikan APIs failed for anime {anime_id}: {e}"
                )

            # Try graceful degradation if enhanced error handling available
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
                            "service": "mal",
                            "api_strategy": "dual",
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
            response = await self._make_jikan_request(endpoint)
            if response and "data" in response:
                return response["data"]
        except Exception as e:
            logger.error(f"Jikan seasonal anime failed for {year} {season}: {e}")

        return []

    async def _handle_mal_error(
        self, status_code: int, response_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle MAL API specific error responses.

        Based on MAL API v2 documentation:
        - 400 Bad Request: Invalid Parameters
        - 401 Unauthorized: invalid_token
        - 403 Forbidden: DoS detected
        - 404 Not Found: Resource not found

        Args:
            status_code: HTTP status code
            response_data: MAL API error response data

        Returns:
            Structured error information for logging and handling
        """
        error_info = {
            "service": "mal",
            "status": status_code,
            "raw_response": response_data,
        }

        if status_code == 400:
            # MAL API 400: Invalid Parameters
            error_info.update(
                {
                    "error_type": "invalid_parameters",
                    "message": f"Invalid parameters provided to MAL API: {response_data.get('error_description', 'Bad request')}",
                    "recoverable": False,
                }
            )
        elif status_code == 401:
            # MAL API 401: Invalid or expired token
            error_info.update(
                {
                    "error_type": "authentication_error",
                    "message": f"MAL API authentication failed: {response_data.get('error_description', 'Invalid or expired token')}",
                    "recoverable": True,
                    "suggested_action": "refresh_token",
                }
            )
        elif status_code == 403:
            # MAL API 403: DoS detection
            error_info.update(
                {
                    "error_type": "rate_limit_exceeded",
                    "message": f"MAL API DoS protection activated: {response_data.get('error_description', 'Request blocked')}",
                    "recoverable": True,
                    "retry_after": 300,  # 5 minutes backoff for DoS detection
                    "suggested_action": "backoff_and_retry",
                }
            )
        elif status_code == 404:
            # MAL API 404: Resource not found
            error_info.update(
                {
                    "error_type": "not_found",
                    "message": f"MAL API resource not found: {response_data.get('error_description', 'Resource does not exist')}",
                    "recoverable": False,
                }
            )
        else:
            # Other MAL API errors
            error_info.update(
                {
                    "error_type": "api_error",
                    "message": f"MAL API error {status_code}: {response_data.get('error_description', 'Unknown error')}",
                    "recoverable": status_code >= 500,  # Server errors are recoverable
                }
            )

        # Log the structured error
        logger.warning(
            f"MAL API error: {error_info['error_type']} - {error_info['message']}"
        )

        return error_info

    async def _handle_jikan_error(
        self, status_code: int, response_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle Jikan API specific error responses.

        Based on Jikan API documentation:
        Structured error response with status, type, message, error, report_url

        Args:
            status_code: HTTP status code
            response_data: Jikan API error response data

        Returns:
            Structured error information for logging and handling
        """
        error_info = {
            "service": "jikan",
            "status": status_code,
            "raw_response": response_data,
        }

        # Extract Jikan-specific error fields
        jikan_type = response_data.get("type", "")
        jikan_message = response_data.get("message", "")
        jikan_error = response_data.get("error", "")
        report_url = response_data.get("report_url", "")

        # Add Jikan-specific fields
        error_info["original_type"] = jikan_type
        error_info["report_url"] = report_url

        if status_code == 429:
            # Jikan API 429: Rate limit exceeded
            error_info.update(
                {
                    "error_type": "rate_limit_exceeded",
                    "message": f"Jikan API rate limit exceeded: {jikan_error or jikan_message}",
                    "recoverable": True,
                    "retry_after": 60,  # 1 minute backoff for rate limiting
                    "suggested_action": "backoff_and_retry",
                }
            )
        elif status_code >= 500:
            # Jikan API 5xx: Server errors
            error_info.update(
                {
                    "error_type": "server_error",
                    "message": f"Jikan API server error: {jikan_error or jikan_message}",
                    "recoverable": True,
                    "retry_after": 30,  # 30 seconds backoff for server errors
                    "suggested_action": "retry_later",
                }
            )
        elif status_code == 404:
            # Jikan API 404: Resource not found
            error_info.update(
                {
                    "error_type": "not_found",
                    "message": f"Jikan API resource not found: {jikan_error or jikan_message}",
                    "recoverable": False,
                }
            )
        elif status_code == 400:
            # Jikan API 400: Bad request
            error_info.update(
                {
                    "error_type": "invalid_parameters",
                    "message": f"Jikan API bad request: {jikan_error or jikan_message}",
                    "recoverable": False,
                }
            )
        else:
            # Other Jikan API errors
            error_info.update(
                {
                    "error_type": "api_error",
                    "message": f"Jikan API error {status_code}: {jikan_error or jikan_message}",
                    "recoverable": status_code >= 500,
                }
            )

        # Log the structured error
        logger.warning(
            f"Jikan API error: {error_info['error_type']} - {error_info['message']}"
        )

        return error_info

    # Helper method for enhanced error context creation
    async def _create_enhanced_error_context(
        self,
        error: Exception,
        correlation_id: str,
        api: str,
        endpoint: str,
        operation: str,
        query_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create enhanced error context for both MAL and Jikan APIs.

        Args:
            error: Exception that occurred
            correlation_id: Correlation ID
            api: API name ("mal" or "jikan")
            endpoint: API endpoint
            operation: Operation being performed
            query_params: Query parameters used (for Jikan)
        """
        if api == "mal":
            await self.create_mal_error_context(
                error=error,
                correlation_id=correlation_id,
                endpoint=endpoint,
                operation=operation,
            )
        else:  # jikan
            await self.create_jikan_error_context(
                error=error,
                correlation_id=correlation_id,
                endpoint=endpoint,
                operation=operation,
                query_params=query_params,
            )

    async def search_anime(
        self,
        query: Optional[str] = None,
        genres: Optional[List[int]] = None,
        status: Optional[str] = None,
        limit: int = 10,
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
            query: Search query string
            genres: List of genre IDs to include
            status: Anime status filter (airing, complete, upcoming)
            limit: Maximum number of results (1-50)
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
        # Auto-generate correlation ID if not provided but we have correlation logger
        if not correlation_id:
            import uuid

            correlation_id = f"search-{uuid.uuid4().hex[:12]}"

        # Start execution tracing if available
        trace_id = None
        if self.execution_tracer:
            trace_id = await self.execution_tracer.start_trace(
                operation=operation,
                context={
                    "query": query,
                    "service": "mal",
                    "api": "jikan",
                    "endpoint": "/anime",
                    "limit": limit,
                },
            )

        try:
            if self.execution_tracer and trace_id:
                await self.execution_tracer.add_trace_step(
                    trace_id=trace_id,
                    step_name="search_start",
                    step_data={"query": query, "genres": genres, "status": status},
                )

            # Jikan search endpoint
            endpoint = "/anime"
            params = {}

            # Basic parameters
            if query:
                params["q"] = query
            if genres:
                params["genres"] = ",".join(map(str, genres))
            if status:
                params["status"] = status
            if limit:
                params["limit"] = limit
            
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
                response = await self._make_jikan_request(endpoint, params=params)
                if response and "data" in response:
                    result = response["data"]

                    if self.execution_tracer and trace_id:
                        await self.execution_tracer.end_trace(
                            trace_id=trace_id,
                            status="success",
                            result={"result_count": len(result), "query": query},
                        )

                    return result
            except APIError as e:
                # Enhanced error handling for search
                if correlation_id:
                    await self._create_enhanced_error_context(
                        error=e,
                        correlation_id=correlation_id,
                        api="jikan",
                        endpoint=endpoint,
                        operation="search_anime",
                        query_params=params,
                    )
                else:
                    # Fallback to basic error logging
                    logger.error(f"Jikan search failed for query '{query}': {e}")
            except Exception as e:
                logger.error(f"Jikan search failed for query '{query}': {e}")

            return []

        except Exception as e:
            if self.execution_tracer and trace_id:
                await self.execution_tracer.end_trace(
                    trace_id=trace_id, status="error", error=e
                )
            raise

    async def create_mal_error_context(
        self,
        error: Exception,
        correlation_id: str,
        endpoint: str,
        operation: str,
    ) -> "ErrorContext":
        """Create enhanced ErrorContext for MAL API errors.

        Args:
            error: Exception that occurred
            correlation_id: Correlation ID
            endpoint: MAL API endpoint
            operation: Operation being performed

        Returns:
            Enhanced ErrorContext
        """
        from ..error_handling import ErrorSeverity

        # Determine severity and user message based on error
        if "401" in str(error) or "invalid_token" in str(error):
            severity = ErrorSeverity.ERROR
            user_message = (
                "MAL authentication failed. Please refresh your access token."
            )
        elif "403" in str(error) or "forbidden" in str(error):
            severity = ErrorSeverity.WARNING
            user_message = "MAL API access denied. Please check your permissions or try again later."
        elif "429" in str(error) or "rate limit" in str(error):
            severity = ErrorSeverity.WARNING
            user_message = (
                "MAL API rate limit exceeded. Please wait before making more requests."
            )
        elif "500" in str(error) or "server" in str(error):
            severity = ErrorSeverity.ERROR
            user_message = (
                "MAL API server error. The service may be temporarily unavailable."
            )
        else:
            severity = ErrorSeverity.ERROR
            user_message = "MAL API request failed. Please try again."

        error_context = await self.create_error_context(
            error=error,
            correlation_id=correlation_id,
            user_message=user_message,
            trace_data={
                "api": "mal",
                "endpoint": endpoint,
                "operation": operation,
                "client_id": self.client_id is not None,
                "has_token": self.access_token is not None,
            },
            severity=severity,
        )

        return error_context

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
                "base_url": self.jikan_base_url,
            },
            severity=severity,
        )

        return error_context

    # PLATFORM-SPECIFIC RATE LIMITING OVERRIDES
    # Override BaseClient hooks to handle dual API strategy (MAL + Jikan)
    
    async def handle_rate_limit_response(self, response):
        """Handle rate limiting for MAL/Jikan based on request URL.
        
        This method intelligently routes to the appropriate platform strategy
        based on which API was called (MAL or Jikan).
        """
        # Determine which API was called based on response URL
        response_url = str(response.url) if hasattr(response, 'url') else ""
        
        if "myanimelist.net" in response_url:
            # Official MAL API
            rate_info = self._mal_rate_limit_adapter.extract_rate_limit_info(response)
            strategy = self._mal_rate_limit_adapter.get_strategy()
            await strategy.handle_rate_limit_response(rate_info)
        elif "jikan.moe" in response_url:
            # Jikan API
            rate_info = self._jikan_rate_limit_adapter.extract_rate_limit_info(response)
            strategy = self._jikan_rate_limit_adapter.get_strategy()
            await strategy.handle_rate_limit_response(rate_info)
        else:
            # Fallback to generic handling
            await self._generic_rate_limit_backoff(response)
    
    async def monitor_rate_limits(self, response):
        """Monitor rate limits for both MAL and Jikan APIs."""
        response_url = str(response.url) if hasattr(response, 'url') else ""
        
        if "myanimelist.net" in response_url:
            # MAL API monitoring
            rate_info = self._mal_rate_limit_adapter.extract_rate_limit_info(response)
            
            # MAL-specific monitoring (status code based since no headers)
            if response.status == 403:
                self.logger.warning(
                    "MAL API DoS protection activated - requests may be blocked"
                )
            elif rate_info.degraded:
                self.logger.warning(
                    f"MAL API rate limiting detected: Status {response.status}"
                )
                
        elif "jikan.moe" in response_url:
            # Jikan API monitoring  
            rate_info = self._jikan_rate_limit_adapter.extract_rate_limit_info(response)
            
            # Jikan-specific monitoring (no headers available)
            if rate_info.degraded:
                self.logger.warning(
                    "Jikan API rate limiting detected - no precise timing available"
                )
    
    async def calculate_backoff_delay(self, response, attempt: int = 0):
        """Calculate backoff delay based on the API being used."""
        if not response:
            # Generic calculation when no response available
            import random
            base_delay = 2 ** attempt
            jitter = random.uniform(0.1, 0.3) * base_delay
            return base_delay + jitter
        
        response_url = str(response.url) if hasattr(response, 'url') else ""
        
        if "myanimelist.net" in response_url:
            # MAL API backoff
            rate_info = self._mal_rate_limit_adapter.extract_rate_limit_info(response)
            strategy = self._mal_rate_limit_adapter.get_strategy()
            return await strategy.calculate_backoff_delay(rate_info, attempt)
        elif "jikan.moe" in response_url:
            # Jikan API backoff
            rate_info = self._jikan_rate_limit_adapter.extract_rate_limit_info(response)
            strategy = self._jikan_rate_limit_adapter.get_strategy()
            return await strategy.calculate_backoff_delay(rate_info, attempt)
        else:
            # Generic backoff
            import random
            base_delay = 2 ** attempt
            jitter = random.uniform(0.1, 0.3) * base_delay
            return base_delay + jitter

    async def refresh_access_token(
        self, *, correlation_id: Optional[str] = None
    ) -> None:
        """Refresh OAuth2 access token with optional correlation tracking.

        Args:
            correlation_id: Optional correlation ID for tracking
        """
        if not self.client_id or not self.client_secret or not self.refresh_token:
            raise Exception("OAuth2 credentials required for token refresh")

        # Auto-generate correlation ID if not provided but we have correlation logger
        if not correlation_id:
            import uuid

            correlation_id = f"refresh-{uuid.uuid4().hex[:12]}"

        if False:
            await self.correlation_logger.log_with_correlation(
                correlation_id=correlation_id,
                level="info",
                message="Starting OAuth2 token refresh",
                context={
                    "service": "mal",
                    "has_refresh_token": self.refresh_token is not None,
                },
            )

        trace_id = None
        if self.execution_tracer:
            trace_id = await self.execution_tracer.start_trace(
                operation="oauth_token_refresh",
                context={"service": "mal", "correlation_id": correlation_id},
            )

        try:
            data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
            }

            # Use base client's make_request to get error handling benefits
            response_data = await self.make_request(
                url="https://myanimelist.net/v1/oauth2/token",
                method="POST",
                priority=1,
                endpoint="/oauth2/token",
                data=data,
            )

            self.access_token = response_data["access_token"]
            self.refresh_token = response_data["refresh_token"]

            if False:
                await self.correlation_logger.log_with_correlation(
                    correlation_id=correlation_id,
                    level="info",
                    message="OAuth2 token refresh successful",
                    context={"service": "mal", "token_updated": True},
                )

            if self.execution_tracer and trace_id:
                await self.execution_tracer.end_trace(
                    trace_id=trace_id,
                    status="success",
                    result={"token_refreshed": True},
                )

        except Exception as e:
            if False:
                await self.correlation_logger.log_with_correlation(
                    correlation_id=correlation_id,
                    level="error",
                    message=f"OAuth2 token refresh failed: {str(e)}",
                    context={"service": "mal"},
                    error_details={
                        "exception_type": type(e).__name__,
                        "message": str(e),
                    },
                )

            if self.execution_tracer and trace_id:
                await self.execution_tracer.end_trace(
                    trace_id=trace_id, status="error", error=e
                )

            # Add specific context for token refresh failures
            raise Exception(f"Token refresh failed: {str(e)}")

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
                    url=f"{self.jikan_base_url}/anime/{anime_id}/statistics",
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
            response = await self._make_jikan_request(endpoint)
            if response and "data" in response:
                return response["data"]
        except Exception as e:
            if correlation_id:
                await self._create_enhanced_error_context(
                    error=e,
                    correlation_id=correlation_id,
                    api="jikan",
                    endpoint=endpoint,
                    operation="get_anime_statistics",
                )
            else:
                logger.error(f"Jikan statistics failed for anime {anime_id}: {e}")

        return {}
