"""Official MAL API v2 client implementation."""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from src.exceptions import APIError

from .base_client import BaseClient
from ..rate_limiting import MALRateLimitAdapter
from ..rate_limiting.core import rate_limit_manager

logger = logging.getLogger(__name__)


class MALClient(BaseClient):
    """Official MyAnimeList API v2 client with OAuth2 authentication."""

    def __init__(
        self,
        client_id: str,
        client_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        circuit_breaker=None,
        cache_manager=None,
        error_handler=None,
        execution_tracer=None,
    ):
        """Initialize MAL API v2 client.

        Args:
            client_id: MAL OAuth2 client ID (required)
            client_secret: MAL OAuth2 client secret (for token refresh)
            access_token: OAuth2 access token
            refresh_token: OAuth2 refresh token
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
        self.base_url = "https://api.myanimelist.net/v2"
        self.oauth_url = "https://myanimelist.net/v1/oauth2"
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.refresh_token = refresh_token
        
        if not client_id:
            raise ValueError("MAL API requires client_id")
        
        # Register platform-specific rate limiting adapter
        self._rate_limit_adapter = MALRateLimitAdapter()
        rate_limit_manager.register_platform_adapter("mal", self._rate_limit_adapter)

    async def _make_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Make request to official MAL API using rate-limited base client."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        # Add X-MAL-CLIENT-ID header (required for all requests)
        headers["X-MAL-CLIENT-ID"] = self.client_id

        # Add Authorization header if access_token is available
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        elif not self.client_id:
            raise APIError("MAL API requires either client_id or access_token")

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

    async def get_anime_by_id(
        self,
        anime_id: int,
        fields: Optional[str] = None,
        *,
        correlation_id: Optional[str] = None,
        parent_correlation_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Get anime by ID from official MAL API.

        Args:
            anime_id: MAL anime ID
            fields: Comma-separated list of fields to include in response
            correlation_id: Optional correlation ID for tracking
            parent_correlation_id: Optional parent correlation ID for chaining
            use_cache: Whether to use cache (default: True)

        Returns:
            Anime data or None if not found
        """
        # Auto-generate correlation ID if not provided
        if not correlation_id:
            import uuid
            correlation_id = f"mal-anime-{uuid.uuid4().hex[:12]}"

        # Use correlation chaining if parent_correlation_id provided
        if parent_correlation_id:
            try:
                result = await self.make_request_with_correlation_chain(
                    url=f"{self.base_url}/anime/{anime_id}",
                    parent_correlation_id=parent_correlation_id,
                    method="GET",
                    endpoint=f"/anime/{anime_id}",
                )
                return result
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
                    "fields": fields,
                },
            )

        try:
            # Check cache first if enabled
            cache_key = f"mal_anime_{anime_id}_{fields or 'default'}"
            if use_cache and self.cache_manager:
                try:
                    cached_result = await self.cache_manager.get(cache_key)
                    if cached_result:
                        logger.info(f"Cache hit for MAL anime {anime_id}")
                        return cached_result
                except Exception as e:
                    logger.warning(f"Cache error for MAL anime {anime_id}: {str(e)}")

            # Make MAL API request
            if self.execution_tracer and trace_id:
                await self.execution_tracer.add_trace_step(
                    trace_id=trace_id,
                    step_name="mal_api_request",
                    step_data={"endpoint": f"/anime/{anime_id}", "fields": fields},
                )

            endpoint = f"/anime/{anime_id}"
            params = {}
            
            if fields:
                params["fields"] = fields

            response = await self._make_request(endpoint, params=params)

            if self.execution_tracer and trace_id:
                await self.execution_tracer.end_trace(
                    trace_id=trace_id,
                    status="success",
                    result={"source": "mal", "anime_id": anime_id},
                )

            logger.info(f"Successfully retrieved anime {anime_id} from MAL API")

            # Cache the result
            if use_cache and self.cache_manager and response:
                try:
                    await self.cache_manager.set(cache_key, response, ttl=3600)
                except Exception as e:
                    logger.warning(f"Cache set error for MAL anime {anime_id}: {str(e)}")
            
            return response

        except Exception as e:
            # Enhanced error handling
            if self.execution_tracer and trace_id:
                await self.execution_tracer.end_trace(
                    trace_id=trace_id, status="error", error=e
                )

            if correlation_id:
                await self.create_mal_error_context(
                    error=e,
                    correlation_id=correlation_id,
                    endpoint=f"/anime/{anime_id}",
                    operation="get_anime_by_id",
                )
            else:
                logger.error(f"MAL API failed for anime {anime_id}: {e}")

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
                            "service": "mal",
                        },
                        cache_key=cache_key,
                    )
                    return degraded_result.get("data")
                except Exception:
                    pass  # If degradation fails, continue to return None

        return None

    async def search_anime(
        self,
        q: str,
        limit: int = 10,
        offset: int = 0,
        fields: Optional[str] = None,
        *,
        correlation_id: Optional[str] = None,
        operation: str = "anime_search",
    ) -> List[Dict[str, Any]]:
        """Search anime using official MAL API.

        Note: MAL API v2 has limited search capabilities compared to Jikan.
        No filtering by genre, status, type, etc. Only basic text search.

        Args:
            q: Search query string (required)
            limit: Maximum number of results (1-100, default 10)
            offset: Offset for pagination (default 0)
            fields: Comma-separated list of field parameters
            correlation_id: Optional correlation ID for tracking
            operation: Operation name for tracing

        Returns:
            List of anime results
        """
        if not q:
            raise ValueError("Search query 'q' is required for MAL API")

        # Auto-generate correlation ID if not provided
        if not correlation_id:
            import uuid
            correlation_id = f"mal-search-{uuid.uuid4().hex[:12]}"

        # Start execution tracing if available
        trace_id = None
        if self.execution_tracer:
            trace_id = await self.execution_tracer.start_trace(
                operation=operation,
                context={
                    "query": q,
                    "service": "mal",
                    "endpoint": "/anime",
                    "limit": limit,
                },
            )

        try:
            if self.execution_tracer and trace_id:
                await self.execution_tracer.add_trace_step(
                    trace_id=trace_id,
                    step_name="search_start",
                    step_data={"query": q, "limit": limit, "offset": offset},
                )

            # Build MAL search parameters
            endpoint = "/anime"
            params = {
                "q": q,
                "limit": min(limit, 100),  # MAL max is 100
                "offset": offset,
            }

            if fields:
                params["fields"] = fields

            try:
                response = await self._make_request(endpoint, params=params)
                
                # MAL API returns {"data": [...], "paging": {...}}
                if response and "data" in response:
                    result = response["data"]

                    if self.execution_tracer and trace_id:
                        await self.execution_tracer.end_trace(
                            trace_id=trace_id,
                            status="success",
                            result={"result_count": len(result), "query": q},
                        )

                    logger.info(f"MAL search returned {len(result)} results for query: {q}")
                    return result
                else:
                    logger.warning(f"MAL search returned unexpected response format: {response}")
                    return []
                    
            except APIError as e:
                # Enhanced error handling for search
                if correlation_id:
                    await self.create_mal_error_context(
                        error=e,
                        correlation_id=correlation_id,
                        endpoint=endpoint,
                        operation="search_anime",
                        query_params=params,
                    )
                else:
                    logger.error(f"MAL search failed for query '{q}': {e}")
            except Exception as e:
                logger.error(f"MAL search failed for query '{q}': {e}")

            return []

        except Exception as e:
            if self.execution_tracer and trace_id:
                await self.execution_tracer.end_trace(
                    trace_id=trace_id, status="error", error=e
                )
            raise

    async def get_user_anime_list(
        self,
        username: str,
        status: Optional[str] = None,
        sort: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        fields: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get user's anime list (requires user permission or public list).

        Args:
            username: MAL username
            status: Filter by status (watching, completed, on_hold, dropped, plan_to_watch)
            sort: Sort order (list_score, list_updated_at, anime_title, anime_start_date, anime_id)
            limit: Maximum results (1-1000)
            offset: Pagination offset
            fields: Comma-separated field parameters

        Returns:
            List of user's anime entries
        """
        endpoint = f"/users/{username}/animelist"
        params = {
            "limit": min(limit, 1000),  # MAL max is 1000
            "offset": offset,
        }

        if status:
            params["status"] = status
        if sort:
            params["sort"] = sort
        if fields:
            params["fields"] = fields

        try:
            response = await self._make_request(endpoint, params=params)
            if response and "data" in response:
                return response["data"]
        except Exception as e:
            logger.error(f"MAL user anime list failed for {username}: {e}")

        return []

    async def get_anime_ranking(
        self,
        ranking_type: str,
        limit: int = 10,
        offset: int = 0,
        fields: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get anime ranking from MAL.

        Args:
            ranking_type: Type of ranking (all, airing, upcoming, tv, ova, movie, special, bypopularity, favorite)
            limit: Maximum results (1-500)
            offset: Pagination offset
            fields: Comma-separated field parameters

        Returns:
            List of ranked anime
        """
        endpoint = "/anime/ranking"
        params = {
            "ranking_type": ranking_type,
            "limit": min(limit, 500),  # MAL max is 500
            "offset": offset,
        }

        if fields:
            params["fields"] = fields

        try:
            response = await self._make_request(endpoint, params=params)
            if response and "data" in response:
                return response["data"]
        except Exception as e:
            logger.error(f"MAL ranking failed for type {ranking_type}: {e}")

        return []

    async def get_seasonal_anime(
        self,
        year: int,
        season: str,
        sort: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        fields: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get seasonal anime from MAL.

        Args:
            year: Year (1917-current)
            season: Season (winter, spring, summer, fall)
            sort: Sort order (anime_score, anime_num_list_users)
            limit: Maximum results (1-500)
            offset: Pagination offset
            fields: Comma-separated field parameters

        Returns:
            List of seasonal anime
        """
        endpoint = f"/anime/season/{year}/{season}"
        params = {
            "limit": min(limit, 500),  # MAL max is 500
            "offset": offset,
        }

        if sort:
            params["sort"] = sort
        if fields:
            params["fields"] = fields

        try:
            response = await self._make_request(endpoint, params=params)
            if response and "data" in response:
                logger.info(f"Retrieved {len(response['data'])} seasonal anime for {year} {season}")
                return response["data"]
        except Exception as e:
            logger.error(f"MAL seasonal anime failed for {year} {season}: {e}")

        return []

    async def refresh_access_token(
        self, *, correlation_id: Optional[str] = None
    ) -> Dict[str, str]:
        """Refresh OAuth2 access token.

        Args:
            correlation_id: Optional correlation ID for tracking

        Returns:
            Token response with access_token and refresh_token

        Raises:
            ValueError: If required OAuth2 credentials are missing
            APIError: If token refresh fails
        """
        if not self.client_id or not self.client_secret or not self.refresh_token:
            raise ValueError("OAuth2 credentials (client_id, client_secret, refresh_token) required for token refresh")

        # Auto-generate correlation ID if not provided
        if not correlation_id:
            import uuid
            correlation_id = f"mal-refresh-{uuid.uuid4().hex[:12]}"

        logger.info("Starting MAL OAuth2 token refresh")

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

            # Use base client's make_request for consistency
            response_data = await self.make_request(
                url=f"{self.oauth_url}/token",
                method="POST",
                priority=1,
                endpoint="/oauth2/token",
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            # Update stored tokens
            self.access_token = response_data["access_token"]
            self.refresh_token = response_data["refresh_token"]

            logger.info("MAL OAuth2 token refresh successful")

            if self.execution_tracer and trace_id:
                await self.execution_tracer.end_trace(
                    trace_id=trace_id,
                    status="success",
                    result={"token_refreshed": True},
                )

            return {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "token_type": response_data.get("token_type", "Bearer"),
                "expires_in": response_data.get("expires_in"),
            }

        except Exception as e:
            logger.error(f"MAL OAuth2 token refresh failed: {str(e)}")

            if self.execution_tracer and trace_id:
                await self.execution_tracer.end_trace(
                    trace_id=trace_id, status="error", error=e
                )

            raise APIError(f"Token refresh failed: {str(e)}")

    async def create_mal_error_context(
        self,
        error: Exception,
        correlation_id: str,
        endpoint: str,
        operation: str,
        query_params: Optional[Dict[str, Any]] = None,
    ) -> "ErrorContext":
        """Create enhanced ErrorContext for MAL API errors.

        Args:
            error: Exception that occurred
            correlation_id: Correlation ID
            endpoint: MAL API endpoint
            operation: Operation being performed
            query_params: Query parameters used

        Returns:
            Enhanced ErrorContext
        """
        from ..error_handling import ErrorSeverity

        # Determine severity and user message based on error
        if "401" in str(error) or "invalid_token" in str(error):
            severity = ErrorSeverity.ERROR
            user_message = "MAL authentication failed. Please refresh your access token."
        elif "403" in str(error) or "forbidden" in str(error):
            severity = ErrorSeverity.WARNING
            user_message = "MAL API access denied. Please check your permissions or try again later."
        elif "429" in str(error) or "rate limit" in str(error):
            severity = ErrorSeverity.WARNING
            user_message = "MAL API rate limit exceeded. Please wait before making more requests."
        elif "500" in str(error) or "server" in str(error):
            severity = ErrorSeverity.ERROR
            user_message = "MAL API server error. The service may be temporarily unavailable."
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
                "query_params": query_params or {},
                "client_id": self.client_id,
                "has_token": self.access_token is not None,
                "base_url": self.base_url,
            },
            severity=severity,
        )

        return error_context

    # PLATFORM-SPECIFIC RATE LIMITING OVERRIDES
    
    async def handle_rate_limit_response(self, response):
        """Handle rate limiting for MAL API."""
        rate_info = self._rate_limit_adapter.extract_rate_limit_info(response)
        strategy = self._rate_limit_adapter.get_strategy()
        await strategy.handle_rate_limit_response(rate_info)
    
    async def monitor_rate_limits(self, response):
        """Monitor rate limits for MAL API."""
        rate_info = self._rate_limit_adapter.extract_rate_limit_info(response)
        
        # MAL-specific monitoring (status code based since no headers)
        if response.status == 403:
            logger.warning("MAL API DoS protection activated - requests may be blocked")
        elif rate_info.degraded:
            logger.warning(f"MAL API rate limiting detected: Status {response.status}")
    
    async def calculate_backoff_delay(self, response, attempt: int = 0):
        """Calculate backoff delay for MAL API."""
        if not response:
            # Generic calculation when no response available
            import random
            base_delay = 2 ** attempt
            jitter = random.uniform(0.1, 0.3) * base_delay
            return base_delay + jitter
        
        rate_info = self._rate_limit_adapter.extract_rate_limit_info(response)
        strategy = self._rate_limit_adapter.get_strategy()
        return await strategy.calculate_backoff_delay(rate_info, attempt)

    @staticmethod
    def generate_oauth_url(
        client_id: str,
        redirect_uri: str,
        state: Optional[str] = None,
        code_challenge: Optional[str] = None,
    ) -> str:
        """Generate OAuth2 authorization URL for user consent.

        Args:
            client_id: MAL OAuth2 client ID
            redirect_uri: Redirect URI registered with MAL
            state: Optional state parameter for CSRF protection
            code_challenge: Optional PKCE code challenge

        Returns:
            OAuth2 authorization URL
        """
        params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
        }

        if state:
            params["state"] = state
        if code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "plain"

        return f"https://myanimelist.net/v1/oauth2/authorize?{urlencode(params)}"

    async def exchange_code_for_token(
        self,
        code: str,
        redirect_uri: str,
        code_verifier: Optional[str] = None,
    ) -> Dict[str, str]:
        """Exchange authorization code for access token.

        Args:
            code: Authorization code from OAuth2 callback
            redirect_uri: Redirect URI used in authorization
            code_verifier: Optional PKCE code verifier

        Returns:
            Token response with access_token and refresh_token
        """
        if not self.client_id or not self.client_secret:
            raise ValueError("OAuth2 credentials (client_id, client_secret) required")

        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        }

        if code_verifier:
            data["code_verifier"] = code_verifier

        try:
            response_data = await self.make_request(
                url=f"{self.oauth_url}/token",
                method="POST",
                priority=1,
                endpoint="/oauth2/token",
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            # Store tokens
            self.access_token = response_data["access_token"]
            self.refresh_token = response_data["refresh_token"]

            logger.info("MAL OAuth2 token exchange successful")

            return {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "token_type": response_data.get("token_type", "Bearer"),
                "expires_in": response_data.get("expires_in"),
            }

        except Exception as e:
            logger.error(f"MAL OAuth2 token exchange failed: {str(e)}")
            raise APIError(f"Token exchange failed: {str(e)}")