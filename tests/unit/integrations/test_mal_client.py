"""Tests for unified MAL/Jikan REST client API."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.exceptions import APIError
from src.integrations.clients.mal_client import MALClient
from src.integrations.error_handling import (
    CorrelationLogger,
    ExecutionTracer,
)


class TestMALClientUnifiedAPI:
    """Test MAL client unified API with auto-enabled enhancements."""

    @pytest.fixture
    def mal_client(self):
        """Create basic MAL client for testing."""
        with patch("src.integrations.clients.base_client.rate_limit_manager"):
            return MALClient()

    @pytest.fixture
    def enhanced_mal_client(self):
        """Create MAL client with enhanced error handling components."""
        correlation_logger = CorrelationLogger()
        execution_tracer = ExecutionTracer()

        with patch("src.integrations.clients.base_client.rate_limit_manager"):
            return MALClient(
                client_id="test_client_id",
                client_secret="test_client_secret",
                correlation_logger=correlation_logger,
                execution_tracer=execution_tracer,
            )

    def test_mal_client_initialization(self, mal_client):
        """Test basic MAL client initializes correctly."""
        assert mal_client.service_name == "mal"
        assert mal_client.mal_base_url == "https://api.myanimelist.net/v2"
        assert mal_client.jikan_base_url == "https://api.jikan.moe/v4"
        assert mal_client.client_id is None
        assert mal_client.access_token is None

    @pytest.mark.asyncio
    async def test_get_anime_by_id_basic_usage(self, mal_client):
        """Test get_anime_by_id works without any enhancements."""
        sample_response = {
            "data": {"mal_id": 21, "title": "One Piece", "synopsis": "Test synopsis"}
        }

        with patch.object(
            mal_client, "_make_jikan_request", return_value=sample_response
        ) as mock_jikan:
            result = await mal_client.get_anime_by_id(21)

            assert result == sample_response["data"]
            mock_jikan.assert_called_once_with("/anime/21")

    @pytest.mark.asyncio
    async def test_get_anime_by_id_with_correlation_auto_enabled(
        self, enhanced_mal_client
    ):
        """Test get_anime_by_id auto-enables correlation logging when available."""
        correlation_id = "test-123"
        sample_response = {"data": {"mal_id": 21, "title": "One Piece"}}

        enhanced_mal_client.correlation_logger.log_with_correlation = AsyncMock()

        with patch.object(
            enhanced_mal_client, "_make_jikan_request", return_value=sample_response
        ) as mock_jikan:
            result = await enhanced_mal_client.get_anime_by_id(
                21, correlation_id=correlation_id
            )

            assert result == sample_response["data"]

            # Verify correlation logging was auto-enabled
            enhanced_mal_client.correlation_logger.log_with_correlation.assert_called()
            log_calls = (
                enhanced_mal_client.correlation_logger.log_with_correlation.call_args_list
            )
            assert any(correlation_id in str(call) for call in log_calls)

    @pytest.mark.asyncio
    async def test_get_anime_by_id_auto_generates_correlation_id(
        self, enhanced_mal_client
    ):
        """Test get_anime_by_id auto-generates correlation ID when none provided."""
        sample_response = {"data": {"mal_id": 21, "title": "One Piece"}}

        enhanced_mal_client.correlation_logger.log_with_correlation = AsyncMock()

        with patch.object(
            enhanced_mal_client, "_make_jikan_request", return_value=sample_response
        ):
            result = await enhanced_mal_client.get_anime_by_id(21)

            assert result == sample_response["data"]

            # Should auto-generate correlation ID and use it
            enhanced_mal_client.correlation_logger.log_with_correlation.assert_called()
            log_calls = (
                enhanced_mal_client.correlation_logger.log_with_correlation.call_args_list
            )
            # Check that some correlation ID was used (auto-generated)
            assert len(log_calls) > 0

    @pytest.mark.asyncio
    async def test_search_anime_with_tracing_auto_enabled(self, enhanced_mal_client):
        """Test search_anime auto-enables tracing when available."""
        sample_response = {
            "data": [
                {"mal_id": 1, "title": "Cowboy Bebop"},
                {"mal_id": 2, "title": "Trigun"},
            ]
        }

        enhanced_mal_client.execution_tracer.start_trace = AsyncMock(
            return_value="trace-123"
        )
        enhanced_mal_client.execution_tracer.add_trace_step = AsyncMock()
        enhanced_mal_client.execution_tracer.end_trace = AsyncMock()

        with patch.object(
            enhanced_mal_client, "_make_jikan_request", return_value=sample_response
        ):
            result = await enhanced_mal_client.search_anime(query="cowboy", limit=2)

            assert len(result) == 2
            assert result[0]["title"] == "Cowboy Bebop"

            # Verify tracing was auto-enabled
            enhanced_mal_client.execution_tracer.start_trace.assert_called_once()
            enhanced_mal_client.execution_tracer.end_trace.assert_called_once()

    @pytest.mark.asyncio
    async def test_parent_correlation_id_chaining(self, enhanced_mal_client):
        """Test parent_correlation_id enables request chaining."""
        parent_id = "parent-123"
        sample_response = {"data": {"mal_id": 1, "title": "Test Anime"}}

        enhanced_mal_client.correlation_logger.log_with_correlation = AsyncMock()

        with patch.object(
            enhanced_mal_client,
            "make_request_with_correlation_chain",
            return_value=sample_response,
        ) as mock_chain:
            result = await enhanced_mal_client.get_anime_by_id(
                1, parent_correlation_id=parent_id
            )

            assert result == sample_response["data"]
            mock_chain.assert_called_once()
            call_args = mock_chain.call_args
            assert call_args.kwargs["parent_correlation_id"] == parent_id

    @pytest.mark.asyncio
    async def test_cache_integration_seamless(self, enhanced_mal_client):
        """Test cache integration works seamlessly."""
        cached_data = {"mal_id": 5, "title": "Cached Anime"}

        # Mock cache hit
        mock_cache = AsyncMock()
        mock_cache.get.return_value = cached_data
        enhanced_mal_client.cache_manager = mock_cache
        enhanced_mal_client.correlation_logger.log_with_correlation = AsyncMock()

        result = await enhanced_mal_client.get_anime_by_id(
            5, correlation_id="cache-test"
        )

        assert result == cached_data
        mock_cache.get.assert_called_once_with("mal_anime_5")

    @pytest.mark.asyncio
    async def test_enhanced_error_handling_automatic(self, enhanced_mal_client):
        """Test enhanced error handling is automatic with correlation_id."""
        enhanced_mal_client.correlation_logger.log_with_correlation = AsyncMock()
        enhanced_mal_client.execution_tracer.start_trace = AsyncMock(
            return_value="trace-error"
        )
        enhanced_mal_client.execution_tracer.end_trace = AsyncMock()

        with patch.object(
            enhanced_mal_client,
            "_make_jikan_request",
            side_effect=APIError("API failed"),
        ):
            with patch.object(
                enhanced_mal_client, "_create_enhanced_error_context"
            ) as mock_error:
                result = await enhanced_mal_client.get_anime_by_id(
                    1, correlation_id="error-test"
                )

                assert result is None  # Should return None on error
                mock_error.assert_called()  # Enhanced error context should be created

    @pytest.mark.asyncio
    async def test_graceful_degradation_automatic(self, enhanced_mal_client):
        """Test graceful degradation kicks in automatically."""
        enhanced_mal_client.correlation_logger.log_with_correlation = AsyncMock()

        # Mock both APIs to fail
        with patch.object(
            enhanced_mal_client,
            "_make_jikan_request",
            side_effect=APIError("API failed"),
        ):
            with patch.object(
                enhanced_mal_client, "handle_enhanced_graceful_degradation"
            ) as mock_degradation:
                mock_degradation.return_value = {"data": {"degraded": True}}

                result = await enhanced_mal_client.get_anime_by_id(
                    1, correlation_id="degradation-test"
                )

                # Should attempt graceful degradation
                assert "degraded" in str(result) or result is None

    @pytest.mark.asyncio
    async def test_refresh_access_token_enhanced(self, enhanced_mal_client):
        """Test refresh_access_token with optional correlation tracking."""
        enhanced_mal_client.refresh_token = "old_token"
        enhanced_mal_client.correlation_logger.log_with_correlation = AsyncMock()
        enhanced_mal_client.execution_tracer.start_trace = AsyncMock(
            return_value="refresh-trace"
        )
        enhanced_mal_client.execution_tracer.end_trace = AsyncMock()

        token_response = {
            "access_token": "new_token",
            "refresh_token": "new_refresh_token",
        }

        with patch.object(
            enhanced_mal_client, "make_request", return_value=token_response
        ):
            await enhanced_mal_client.refresh_access_token(
                correlation_id="refresh-test"
            )

            assert enhanced_mal_client.access_token == "new_token"

            # Verify enhanced features were used
            enhanced_mal_client.correlation_logger.log_with_correlation.assert_called()
            enhanced_mal_client.execution_tracer.start_trace.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_anime_statistics_enhanced(self, enhanced_mal_client):
        """Test get_anime_statistics with enhancements."""
        stats_response = {"data": {"watching": 100, "completed": 500}}

        with patch.object(
            enhanced_mal_client, "_make_jikan_request", return_value=stats_response
        ):
            result = await enhanced_mal_client.get_anime_statistics(
                1, correlation_id="stats-test"
            )

            assert result == stats_response["data"]

    @pytest.mark.asyncio
    async def test_backward_compatibility(self, mal_client):
        """Test all methods work without enhanced components (backward compatibility)."""
        sample_response = {"data": {"mal_id": 21, "title": "One Piece"}}

        with patch.object(
            mal_client, "_make_jikan_request", return_value=sample_response
        ):
            # All methods should work without enhanced components
            result = await mal_client.get_anime_by_id(21)
            assert result == sample_response["data"]

            search_result = await mal_client.search_anime("test")
            assert search_result == sample_response["data"]

            stats_result = await mal_client.get_anime_statistics(21)
            assert stats_result == sample_response["data"]

    @pytest.mark.asyncio
    async def test_dual_api_strategy_still_works(self, enhanced_mal_client):
        """Test dual API strategy (MAL -> Jikan fallback) still works."""
        jikan_response = {"data": {"mal_id": 1, "title": "Fallback Anime"}}

        # Mock MAL API failure and Jikan success
        with patch.object(
            enhanced_mal_client, "_make_mal_request", side_effect=APIError("MAL failed")
        ):
            with patch.object(
                enhanced_mal_client, "_make_jikan_request", return_value=jikan_response
            ):
                result = await enhanced_mal_client.get_anime_by_id(
                    1, correlation_id="dual-test"
                )

                assert result == jikan_response["data"]

    @pytest.mark.asyncio
    async def test_no_correlation_fallback_behavior(self, enhanced_mal_client):
        """Test behavior when no correlation_id provided but enhanced components available."""
        sample_response = {"data": {"mal_id": 21, "title": "One Piece"}}

        enhanced_mal_client.correlation_logger.log_with_correlation = AsyncMock()

        with patch.object(
            enhanced_mal_client, "_make_jikan_request", return_value=sample_response
        ):
            # Should auto-generate correlation ID when logger available
            result = await enhanced_mal_client.get_anime_by_id(21)

            assert result == sample_response["data"]
            # Should have called correlation logger with auto-generated ID
            enhanced_mal_client.correlation_logger.log_with_correlation.assert_called()

    @pytest.mark.asyncio
    async def test_mal_api_authentication_headers(self, enhanced_mal_client):
        """Test MAL API request with authentication headers."""
        enhanced_mal_client.access_token = "test_token"
        sample_response = {"mal_id": 1, "title": "Test Anime"}

        with patch.object(
            enhanced_mal_client, "make_request", return_value=sample_response
        ) as mock_request:
            result = await enhanced_mal_client._make_mal_request("/anime/1")

            assert result == sample_response
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            headers = call_args.kwargs["headers"]
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer test_token"

    @pytest.mark.asyncio
    async def test_mal_api_client_id_only(self, enhanced_mal_client):
        """Test MAL API request with client_id only (no access token)."""
        enhanced_mal_client.access_token = None  # Clear access token
        sample_response = {"mal_id": 1, "title": "Test Anime"}

        with patch.object(
            enhanced_mal_client, "make_request", return_value=sample_response
        ) as mock_request:
            result = await enhanced_mal_client._make_mal_request("/anime/1")

            assert result == sample_response
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            headers = call_args.kwargs["headers"]
            assert "X-MAL-CLIENT-ID" in headers
            assert headers["X-MAL-CLIENT-ID"] == "test_client_id"

    @pytest.mark.asyncio
    async def test_mal_api_no_credentials(self):
        """Test MAL API request fails without credentials."""
        with patch("src.integrations.clients.base_client.rate_limit_manager"):
            client = MALClient()  # No credentials

        with pytest.raises(Exception, match="MAL API requires either client_id or access_token"):
            await client._make_mal_request("/anime/1")

    @pytest.mark.asyncio
    async def test_jikan_request_with_params(self, mal_client):
        """Test Jikan API request with query parameters."""
        sample_response = {"data": [{"mal_id": 1, "title": "Test"}]}
        params = {"q": "cowboy", "limit": 5}

        with patch.object(
            mal_client, "make_request", return_value=sample_response
        ) as mock_request:
            result = await mal_client._make_jikan_request("/anime", params=params)

            assert result == sample_response
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args.kwargs["params"] == params

    @pytest.mark.asyncio
    async def test_jikan_request_with_retry_success(self, mal_client):
        """Test Jikan request with retry mechanism - eventual success."""
        sample_response = {"data": [{"mal_id": 1, "title": "Test"}]}

        # Mock: First call fails with 500, second succeeds
        with patch.object(
            mal_client, "_make_jikan_request", side_effect=[Exception("500 Server Error"), sample_response]
        ):
            result = await mal_client._make_jikan_request_with_retry("/anime/1", max_retries=3)

            assert result == sample_response

    @pytest.mark.asyncio
    async def test_jikan_request_with_retry_max_retries(self, mal_client):
        """Test Jikan request with retry mechanism - max retries exceeded."""
        # Mock: All calls fail with 500 error
        with patch.object(
            mal_client, "_make_jikan_request", side_effect=Exception("500 Server Error")
        ):
            with pytest.raises(Exception, match="500 Server Error"):
                await mal_client._make_jikan_request_with_retry("/anime/1", max_retries=2)

    @pytest.mark.asyncio
    async def test_jikan_request_with_retry_non_server_error(self, mal_client):
        """Test Jikan request with retry - non-server error doesn't retry."""
        # Mock: First call fails with 404 (should not retry)
        with patch.object(
            mal_client, "_make_jikan_request", side_effect=Exception("404 Not Found")
        ):
            with pytest.raises(Exception, match="404 Not Found"):
                await mal_client._make_jikan_request_with_retry("/anime/1", max_retries=3)

    @pytest.mark.asyncio
    async def test_cache_error_handling(self, enhanced_mal_client):
        """Test cache error handling doesn't break the flow."""
        sample_response = {"data": {"mal_id": 21, "title": "One Piece"}}
        enhanced_mal_client.correlation_logger.log_with_correlation = AsyncMock()

        # Mock cache manager to throw error
        mock_cache = AsyncMock()
        mock_cache.get.side_effect = Exception("Cache connection failed")
        enhanced_mal_client.cache_manager = mock_cache

        with patch.object(
            enhanced_mal_client, "_make_jikan_request", return_value=sample_response
        ):
            result = await enhanced_mal_client.get_anime_by_id(
                21, correlation_id="cache-error-test"
            )

            assert result == sample_response["data"]
            # Should log cache error
            enhanced_mal_client.correlation_logger.log_with_correlation.assert_called()

    @pytest.mark.asyncio
    async def test_search_anime_api_error_with_correlation(self, enhanced_mal_client):
        """Test search_anime handles APIError with correlation logging."""
        enhanced_mal_client.correlation_logger.log_with_correlation = AsyncMock()
        enhanced_mal_client.execution_tracer.start_trace = AsyncMock(return_value="trace-123")
        enhanced_mal_client.execution_tracer.end_trace = AsyncMock()

        with patch.object(
            enhanced_mal_client, "_make_jikan_request", side_effect=APIError("Search failed")
        ):
            with patch.object(
                enhanced_mal_client, "_create_enhanced_error_context"
            ) as mock_error:
                result = await enhanced_mal_client.search_anime(
                    query="test", correlation_id="search-error"
                )

                assert result == []
                mock_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_anime_api_error_without_correlation(self, mal_client):
        """Test search_anime handles APIError without correlation logging."""
        with patch.object(
            mal_client, "_make_jikan_request", side_effect=APIError("Search failed")
        ):
            result = await mal_client.search_anime(query="test")
            assert result == []

    @pytest.mark.asyncio
    async def test_search_anime_general_exception(self, mal_client):
        """Test search_anime handles general exceptions."""
        with patch.object(
            mal_client, "_make_jikan_request", side_effect=Exception("Unexpected error")
        ):
            result = await mal_client.search_anime(query="test")
            assert result == []

    @pytest.mark.asyncio
    async def test_search_anime_with_all_params(self, mal_client):
        """Test search_anime with all possible parameters."""
        sample_response = {"data": [{"mal_id": 1, "title": "Test"}]}

        with patch.object(
            mal_client, "_make_jikan_request", return_value=sample_response
        ) as mock_request:
            result = await mal_client.search_anime(
                query="cowboy", genres=[1, 2], status="completed", limit=25
            )

            assert result == sample_response["data"]
            call_args = mock_request.call_args
            params = call_args.kwargs["params"]
            assert params["q"] == "cowboy"
            assert params["genres"] == "1,2"
            assert params["status"] == "completed"
            assert params["limit"] == 25

    @pytest.mark.asyncio
    async def test_search_anime_outer_exception(self, enhanced_mal_client):
        """Test search_anime handles outer exception and traces it."""
        enhanced_mal_client.execution_tracer.start_trace = AsyncMock(return_value="trace-123")
        enhanced_mal_client.execution_tracer.end_trace = AsyncMock()

        # Mock the execution tracer start_trace to raise an exception
        enhanced_mal_client.execution_tracer.start_trace.side_effect = RuntimeError("Outer error")

        with pytest.raises(RuntimeError, match="Outer error"):
            await enhanced_mal_client.search_anime(query="test")

        # start_trace should have been called before the exception
        enhanced_mal_client.execution_tracer.start_trace.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_seasonal_anime_success(self, mal_client):
        """Test get_seasonal_anime successful response."""
        sample_response = {
            "data": [
                {"mal_id": 1, "title": "Spring Anime 1"},
                {"mal_id": 2, "title": "Spring Anime 2"},
            ]
        }

        with patch.object(
            mal_client, "_make_jikan_request", return_value=sample_response
        ):
            result = await mal_client.get_seasonal_anime(2023, "spring")

            assert len(result) == 2
            assert result[0]["title"] == "Spring Anime 1"

    @pytest.mark.asyncio
    async def test_get_seasonal_anime_error(self, mal_client):
        """Test get_seasonal_anime handles errors gracefully."""
        with patch.object(
            mal_client, "_make_jikan_request", side_effect=Exception("Seasonal API failed")
        ):
            result = await mal_client.get_seasonal_anime(2023, "spring")
            assert result == []

    @pytest.mark.asyncio
    async def test_handle_mal_error_400(self, mal_client):
        """Test _handle_mal_error for 400 Bad Request."""
        response_data = {"error_description": "Invalid parameters"}
        error_info = await mal_client._handle_mal_error(400, response_data)

        assert error_info["service"] == "mal"
        assert error_info["status"] == 400
        assert error_info["error_type"] == "invalid_parameters"
        assert error_info["recoverable"] == False
        assert "Invalid parameters" in error_info["message"]

    @pytest.mark.asyncio
    async def test_handle_mal_error_401(self, mal_client):
        """Test _handle_mal_error for 401 Unauthorized."""
        response_data = {"error_description": "Invalid token"}
        error_info = await mal_client._handle_mal_error(401, response_data)

        assert error_info["error_type"] == "authentication_error"
        assert error_info["recoverable"] == True
        assert error_info["suggested_action"] == "refresh_token"

    @pytest.mark.asyncio
    async def test_handle_mal_error_403(self, mal_client):
        """Test _handle_mal_error for 403 Forbidden (DoS)."""
        response_data = {"error_description": "DoS detected"}
        error_info = await mal_client._handle_mal_error(403, response_data)

        assert error_info["error_type"] == "rate_limit_exceeded"
        assert error_info["retry_after"] == 300
        assert error_info["suggested_action"] == "backoff_and_retry"

    @pytest.mark.asyncio
    async def test_handle_mal_error_404(self, mal_client):
        """Test _handle_mal_error for 404 Not Found."""
        response_data = {"error_description": "Resource not found"}
        error_info = await mal_client._handle_mal_error(404, response_data)

        assert error_info["error_type"] == "not_found"
        assert error_info["recoverable"] == False

    @pytest.mark.asyncio
    async def test_handle_mal_error_500(self, mal_client):
        """Test _handle_mal_error for 500 Server Error."""
        response_data = {"error_description": "Internal server error"}
        error_info = await mal_client._handle_mal_error(500, response_data)

        assert error_info["error_type"] == "api_error"
        assert error_info["recoverable"] == True  # Server errors are recoverable

    @pytest.mark.asyncio
    async def test_handle_mal_error_other(self, mal_client):
        """Test _handle_mal_error for other status codes."""
        response_data = {"error_description": "Unknown error"}
        error_info = await mal_client._handle_mal_error(422, response_data)

        assert error_info["error_type"] == "api_error"
        assert error_info["recoverable"] == False  # Client errors are not recoverable

    @pytest.mark.asyncio
    async def test_handle_jikan_error_429(self, mal_client):
        """Test _handle_jikan_error for 429 Rate Limit."""
        response_data = {
            "type": "RateLimitException",
            "message": "Too many requests",
            "error": "Rate limit exceeded",
            "report_url": "https://github.com/jikan-me/jikan/issues"
        }
        error_info = await mal_client._handle_jikan_error(429, response_data)

        assert error_info["service"] == "jikan"
        assert error_info["error_type"] == "rate_limit_exceeded"
        assert error_info["retry_after"] == 60
        assert error_info["report_url"] == "https://github.com/jikan-me/jikan/issues"

    @pytest.mark.asyncio
    async def test_handle_jikan_error_500(self, mal_client):
        """Test _handle_jikan_error for 500 Server Error."""
        response_data = {
            "type": "ServerException",
            "message": "Server error",
            "error": "Internal server error"
        }
        error_info = await mal_client._handle_jikan_error(500, response_data)

        assert error_info["error_type"] == "server_error"
        assert error_info["recoverable"] == True
        assert error_info["retry_after"] == 30

    @pytest.mark.asyncio
    async def test_handle_jikan_error_404(self, mal_client):
        """Test _handle_jikan_error for 404 Not Found."""
        response_data = {
            "type": "NotFoundException",
            "message": "Not found",
            "error": "Resource not found"
        }
        error_info = await mal_client._handle_jikan_error(404, response_data)

        assert error_info["error_type"] == "not_found"
        assert error_info["recoverable"] == False

    @pytest.mark.asyncio
    async def test_handle_jikan_error_400(self, mal_client):
        """Test _handle_jikan_error for 400 Bad Request."""
        response_data = {
            "type": "BadRequestException",
            "message": "Bad request",
            "error": "Invalid parameters"
        }
        error_info = await mal_client._handle_jikan_error(400, response_data)

        assert error_info["error_type"] == "invalid_parameters"
        assert error_info["recoverable"] == False

    @pytest.mark.asyncio
    async def test_handle_jikan_error_other(self, mal_client):
        """Test _handle_jikan_error for other status codes."""
        response_data = {
            "type": "UnknownException",
            "message": "Unknown error",
            "error": "Something went wrong"
        }
        error_info = await mal_client._handle_jikan_error(418, response_data)

        assert error_info["error_type"] == "api_error"
        assert error_info["recoverable"] == False

    @pytest.mark.asyncio
    async def test_create_mal_error_context_401(self, enhanced_mal_client):
        """Test create_mal_error_context for 401 authentication error."""
        error = Exception("401 Unauthorized - invalid_token")
        
        with patch.object(
            enhanced_mal_client, "create_error_context", return_value=AsyncMock()
        ) as mock_create:
            await enhanced_mal_client.create_mal_error_context(
                error=error,
                correlation_id="test-401",
                endpoint="/anime/1",
                operation="get_anime"
            )
            
            mock_create.assert_called_once()
            call_args = mock_create.call_args.kwargs
            assert "MAL authentication failed" in call_args["user_message"]

    @pytest.mark.asyncio
    async def test_create_mal_error_context_403(self, enhanced_mal_client):
        """Test create_mal_error_context for 403 forbidden error."""
        error = Exception("403 Forbidden")
        
        with patch.object(
            enhanced_mal_client, "create_error_context", return_value=AsyncMock()
        ) as mock_create:
            await enhanced_mal_client.create_mal_error_context(
                error=error,
                correlation_id="test-403",
                endpoint="/anime/1",
                operation="get_anime"
            )
            
            mock_create.assert_called_once()
            call_args = mock_create.call_args.kwargs
            assert "access denied" in call_args["user_message"]

    @pytest.mark.asyncio
    async def test_create_mal_error_context_429(self, enhanced_mal_client):
        """Test create_mal_error_context for 429 rate limit error."""
        error = Exception("429 rate limit exceeded")
        
        with patch.object(
            enhanced_mal_client, "create_error_context", return_value=AsyncMock()
        ) as mock_create:
            await enhanced_mal_client.create_mal_error_context(
                error=error,
                correlation_id="test-429",
                endpoint="/anime/1",
                operation="get_anime"
            )
            
            mock_create.assert_called_once()
            call_args = mock_create.call_args.kwargs
            assert "rate limit exceeded" in call_args["user_message"]

    @pytest.mark.asyncio
    async def test_create_mal_error_context_500(self, enhanced_mal_client):
        """Test create_mal_error_context for 500 server error."""
        error = Exception("500 Internal Server Error")
        
        with patch.object(
            enhanced_mal_client, "create_error_context", return_value=AsyncMock()
        ) as mock_create:
            await enhanced_mal_client.create_mal_error_context(
                error=error,
                correlation_id="test-500",
                endpoint="/anime/1",
                operation="get_anime"
            )
            
            mock_create.assert_called_once()
            call_args = mock_create.call_args.kwargs
            assert "server error" in call_args["user_message"]

    @pytest.mark.asyncio
    async def test_create_jikan_error_context_429(self, enhanced_mal_client):
        """Test create_jikan_error_context for 429 rate limit error."""
        error = Exception("429 rate limit")
        
        with patch.object(
            enhanced_mal_client, "create_error_context", return_value=AsyncMock()
        ) as mock_create:
            await enhanced_mal_client.create_jikan_error_context(
                error=error,
                correlation_id="test-jikan-429",
                endpoint="/anime/1",
                operation="get_anime",
                query_params={"q": "test"}
            )
            
            mock_create.assert_called_once()
            call_args = mock_create.call_args.kwargs
            assert "rate limit exceeded" in call_args["user_message"]

    @pytest.mark.asyncio
    async def test_create_jikan_error_context_404(self, enhanced_mal_client):
        """Test create_jikan_error_context for 404 not found error."""
        error = Exception("404 not found")
        
        with patch.object(
            enhanced_mal_client, "create_error_context", return_value=AsyncMock()
        ) as mock_create:
            await enhanced_mal_client.create_jikan_error_context(
                error=error,
                correlation_id="test-jikan-404",
                endpoint="/anime/1",
                operation="get_anime"
            )
            
            mock_create.assert_called_once()
            call_args = mock_create.call_args.kwargs
            assert "not found" in call_args["user_message"]

    @pytest.mark.asyncio
    async def test_create_jikan_error_context_500(self, enhanced_mal_client):
        """Test create_jikan_error_context for 500 server error."""
        error = Exception("500 server error")
        
        with patch.object(
            enhanced_mal_client, "create_error_context", return_value=AsyncMock()
        ) as mock_create:
            await enhanced_mal_client.create_jikan_error_context(
                error=error,
                correlation_id="test-jikan-500",
                endpoint="/anime/1",
                operation="get_anime"
            )
            
            mock_create.assert_called_once()
            call_args = mock_create.call_args.kwargs
            assert "server error" in call_args["user_message"]

    @pytest.mark.asyncio
    async def test_refresh_access_token_missing_credentials(self):
        """Test refresh_access_token fails without required credentials."""
        with patch("src.integrations.clients.base_client.rate_limit_manager"):
            client = MALClient()  # No credentials
        
        with pytest.raises(Exception, match="OAuth2 credentials required"):
            await client.refresh_access_token()

    @pytest.mark.asyncio
    async def test_refresh_access_token_auto_correlation_id(self, enhanced_mal_client):
        """Test refresh_access_token auto-generates correlation ID."""
        enhanced_mal_client.refresh_token = "old_token"
        enhanced_mal_client.correlation_logger.log_with_correlation = AsyncMock()
        enhanced_mal_client.execution_tracer.start_trace = AsyncMock(return_value="trace-123")
        enhanced_mal_client.execution_tracer.end_trace = AsyncMock()

        token_response = {
            "access_token": "new_token",
            "refresh_token": "new_refresh_token",
        }

        with patch.object(
            enhanced_mal_client, "make_request", return_value=token_response
        ):
            # Call without correlation_id - should auto-generate
            await enhanced_mal_client.refresh_access_token()

            assert enhanced_mal_client.access_token == "new_token"
            enhanced_mal_client.correlation_logger.log_with_correlation.assert_called()

    @pytest.mark.asyncio
    async def test_refresh_access_token_failure(self, enhanced_mal_client):
        """Test refresh_access_token handles failure properly."""
        enhanced_mal_client.refresh_token = "old_token"
        enhanced_mal_client.correlation_logger.log_with_correlation = AsyncMock()
        enhanced_mal_client.execution_tracer.start_trace = AsyncMock(return_value="trace-123")
        enhanced_mal_client.execution_tracer.end_trace = AsyncMock()

        with patch.object(
            enhanced_mal_client, "make_request", side_effect=Exception("Token refresh failed")
        ):
            with pytest.raises(Exception, match="Token refresh failed"):
                await enhanced_mal_client.refresh_access_token(correlation_id="refresh-error")

            # Should log error and trace failure
            enhanced_mal_client.correlation_logger.log_with_correlation.assert_called()
            enhanced_mal_client.execution_tracer.end_trace.assert_called_once()
            call_args = enhanced_mal_client.execution_tracer.end_trace.call_args
            assert call_args.kwargs["status"] == "error"

    @pytest.mark.asyncio
    async def test_get_anime_statistics_correlation_chain_success(self, enhanced_mal_client):
        """Test get_anime_statistics with correlation chaining success."""
        parent_id = "parent-stats-123"
        stats_response = {"data": {"watching": 100, "completed": 500}}

        with patch.object(
            enhanced_mal_client,
            "make_request_with_correlation_chain",
            return_value=stats_response,
        ) as mock_chain:
            result = await enhanced_mal_client.get_anime_statistics(
                1, parent_correlation_id=parent_id
            )

            assert result == stats_response["data"]
            mock_chain.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_anime_statistics_correlation_chain_failure(self, enhanced_mal_client):
        """Test get_anime_statistics falls back when correlation chaining fails."""
        parent_id = "parent-stats-fail"
        stats_response = {"data": {"watching": 100, "completed": 500}}

        # Mock correlation chain to fail, regular request to succeed
        with patch.object(
            enhanced_mal_client,
            "make_request_with_correlation_chain",
            side_effect=Exception("Chain failed"),
        ):
            with patch.object(
                enhanced_mal_client, "_make_jikan_request", return_value=stats_response
            ):
                result = await enhanced_mal_client.get_anime_statistics(
                    1, parent_correlation_id=parent_id
                )

                assert result == stats_response["data"]

    @pytest.mark.asyncio
    async def test_get_anime_statistics_error_with_correlation(self, enhanced_mal_client):
        """Test get_anime_statistics handles errors with correlation."""
        with patch.object(
            enhanced_mal_client, "_make_jikan_request", side_effect=Exception("Stats failed")
        ):
            with patch.object(
                enhanced_mal_client, "_create_enhanced_error_context"
            ) as mock_error:
                result = await enhanced_mal_client.get_anime_statistics(
                    1, correlation_id="stats-error"
                )

                assert result == {}
                mock_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_anime_statistics_error_without_correlation(self, mal_client):
        """Test get_anime_statistics handles errors without correlation."""
        with patch.object(
            mal_client, "_make_jikan_request", side_effect=Exception("Stats failed")
        ):
            result = await mal_client.get_anime_statistics(1)
            assert result == {}

    @pytest.mark.asyncio
    async def test_get_anime_by_id_correlation_chain_fallback(self, enhanced_mal_client):
        """Test get_anime_by_id correlation chain fallback when exception occurs."""
        parent_id = "parent-fail-123"
        sample_response = {"data": {"mal_id": 1, "title": "Fallback Anime"}}
        
        # Mock correlation chain to fail, then regular flow to succeed
        with patch.object(
            enhanced_mal_client,
            "make_request_with_correlation_chain", 
            side_effect=Exception("Chain failed")
        ):
            with patch.object(
                enhanced_mal_client, "_make_jikan_request", return_value=sample_response
            ):
                result = await enhanced_mal_client.get_anime_by_id(
                    1, parent_correlation_id=parent_id
                )
                
                assert result == sample_response["data"]

    @pytest.mark.asyncio
    async def test_get_anime_by_id_no_data_in_response(self, mal_client):
        """Test get_anime_by_id when response has no 'data' field."""
        # Response without 'data' field
        sample_response = {"error": "Some error occurred"}
        
        with patch.object(
            mal_client, "_make_jikan_request", return_value=sample_response
        ):
            result = await mal_client.get_anime_by_id(1)
            assert result is None

    @pytest.mark.asyncio
    async def test_get_anime_by_id_mal_api_success_with_tracing(self, enhanced_mal_client):
        """Test get_anime_by_id MAL API success with execution tracing."""
        enhanced_mal_client.access_token = "test_token"
        sample_response = {"mal_id": 1, "title": "Test Anime"}
        
        enhanced_mal_client.execution_tracer.start_trace = AsyncMock(return_value="trace-123")
        enhanced_mal_client.execution_tracer.add_trace_step = AsyncMock()
        enhanced_mal_client.execution_tracer.end_trace = AsyncMock()
        
        with patch.object(
            enhanced_mal_client, "_make_mal_request", return_value=sample_response
        ):
            result = await enhanced_mal_client.get_anime_by_id(
                1, correlation_id="mal-success-test"
            )
            
            assert result == sample_response
            # Should have traced MAL API success steps
            assert enhanced_mal_client.execution_tracer.add_trace_step.call_count >= 2
            enhanced_mal_client.execution_tracer.end_trace.assert_called_once()

    @pytest.mark.asyncio 
    async def test_get_anime_by_id_basic_error_handling_without_correlation(self, enhanced_mal_client):
        """Test get_anime_by_id basic error handling when no correlation_id provided."""
        # Clear correlation_logger to test basic error path
        enhanced_mal_client.correlation_logger = None
        
        with patch.object(
            enhanced_mal_client, "_make_jikan_request", side_effect=APIError("API failed")
        ):
            result = await enhanced_mal_client.get_anime_by_id(1)
            assert result is None

    @pytest.mark.asyncio
    async def test_get_anime_by_id_graceful_degradation_failure(self, enhanced_mal_client):
        """Test get_anime_by_id when graceful degradation also fails."""
        enhanced_mal_client.correlation_logger.log_with_correlation = AsyncMock()
        
        # Mock both APIs to fail
        with patch.object(
            enhanced_mal_client, "_make_jikan_request", side_effect=APIError("API failed")
        ):
            # Mock graceful degradation to also fail
            with patch.object(
                enhanced_mal_client, "handle_enhanced_graceful_degradation",
                side_effect=Exception("Degradation failed")
            ):
                result = await enhanced_mal_client.get_anime_by_id(
                    1, correlation_id="degradation-fail-test"
                )
                
                assert result is None

    @pytest.mark.asyncio
    async def test_search_anime_no_response_data(self, mal_client):
        """Test search_anime when response has no 'data' field."""
        # Response without 'data' field
        sample_response = {"error": "Search failed"}
        
        with patch.object(
            mal_client, "_make_jikan_request", return_value=sample_response
        ):
            result = await mal_client.search_anime(query="test")
            assert result == []

    @pytest.mark.asyncio
    async def test_search_anime_exception_in_try_block(self, enhanced_mal_client):
        """Test search_anime exception in the main try block."""
        enhanced_mal_client.execution_tracer.start_trace = AsyncMock(return_value="trace-123")
        enhanced_mal_client.execution_tracer.add_trace_step = AsyncMock()
        enhanced_mal_client.execution_tracer.end_trace = AsyncMock()
        
        # Mock add_trace_step to raise an exception after start_trace succeeds
        enhanced_mal_client.execution_tracer.add_trace_step.side_effect = Exception("Trace step failed")
        
        with pytest.raises(Exception, match="Trace step failed"):
            await enhanced_mal_client.search_anime(query="test")
            
        # Should have traced the error
        enhanced_mal_client.execution_tracer.end_trace.assert_called_once()
        call_args = enhanced_mal_client.execution_tracer.end_trace.call_args
        assert call_args.kwargs["status"] == "error"

    @pytest.mark.asyncio
    async def test_get_anime_statistics_no_data_response(self, mal_client):
        """Test get_anime_statistics when response has no 'data' field."""
        # Response without 'data' field
        sample_response = {"error": "Stats not available"}
        
        with patch.object(
            mal_client, "_make_jikan_request", return_value=sample_response
        ):
            result = await mal_client.get_anime_statistics(1)
            assert result == {}

    @pytest.mark.asyncio
    async def test_get_anime_by_id_correlation_chain_no_data(self, enhanced_mal_client):
        """Test get_anime_by_id correlation chain when response has no data field."""
        parent_id = "parent-no-data-123"
        # Response without data field
        response_without_data = {"error": "No data available"}
        
        with patch.object(
            enhanced_mal_client,
            "make_request_with_correlation_chain",
            return_value=response_without_data,
        ) as mock_chain:
            result = await enhanced_mal_client.get_anime_by_id(
                1, parent_correlation_id=parent_id
            )
            
            # Should return None when correlation chain response has no data
            assert result is None
            mock_chain.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_anime_statistics_correlation_chain_no_data(self, enhanced_mal_client):
        """Test get_anime_statistics correlation chain when response has no data field."""
        parent_id = "parent-stats-no-data-123"
        # Response without data field
        response_without_data = {"error": "Stats not available"}
        
        with patch.object(
            enhanced_mal_client,
            "make_request_with_correlation_chain",
            return_value=response_without_data,
        ) as mock_chain:
            result = await enhanced_mal_client.get_anime_statistics(
                1, parent_correlation_id=parent_id
            )
            
            # Should return {} when correlation chain response has no data
            assert result == {}
            mock_chain.assert_called_once()
