"""Tests for MAL API v2 client implementation."""

import pytest
from unittest.mock import AsyncMock, patch

from src.exceptions import APIError
from src.integrations.clients.mal_client import MALClient
from src.integrations.error_handling import ExecutionTracer


class TestMALClientV2:
    """Test MAL API v2 client with OAuth2 authentication."""

    @pytest.fixture
    def mal_client(self):
        """Create basic MAL client for testing."""
        with patch("src.integrations.clients.base_client.rate_limit_manager"):
            return MALClient(client_id="test_client_id")

    @pytest.fixture
    def enhanced_mal_client(self):
        """Create MAL client with enhanced error handling components."""
        execution_tracer = ExecutionTracer()

        with patch("src.integrations.clients.base_client.rate_limit_manager"):
            return MALClient(
                client_id="test_client_id",
                client_secret="test_client_secret",
                execution_tracer=execution_tracer,
            )

    def test_mal_client_initialization(self, mal_client):
        """Test basic MAL client initializes correctly."""
        assert mal_client.service_name == "mal"
        assert mal_client.base_url == "https://api.myanimelist.net/v2"
        assert mal_client.client_id == "test_client_id"
        assert mal_client.access_token is None

    def test_mal_client_initialization_requires_client_id(self):
        """Test MAL client requires client_id."""
        with patch("src.integrations.clients.base_client.rate_limit_manager"):
            with pytest.raises(ValueError, match="MAL API requires client_id"):
                MALClient(client_id="")

    @pytest.mark.asyncio
    async def test_search_anime_basic(self, mal_client):
        """Test basic anime search."""
        sample_response = {
            "data": [
                {"id": 1, "title": "Cowboy Bebop"},
                {"id": 2, "title": "Trigun"}
            ]
        }

        with patch.object(
            mal_client, "_make_request", return_value=sample_response
        ) as mock_request:
            result = await mal_client.search_anime(q="cowboy", limit=10)

            assert result == sample_response["data"]
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args.kwargs["params"]["q"] == "cowboy"
            assert call_args.kwargs["params"]["limit"] == 10

    @pytest.mark.asyncio
    async def test_search_anime_with_fields(self, mal_client):
        """Test anime search with specific fields."""
        sample_response = {"data": [{"id": 1, "title": "Test", "mean": 8.5}]}

        with patch.object(
            mal_client, "_make_request", return_value=sample_response
        ) as mock_request:
            result = await mal_client.search_anime(
                q="test", 
                fields="id,title,mean",
                correlation_id="search-test"
            )

            assert result == sample_response["data"]
            call_args = mock_request.call_args
            assert call_args.kwargs["params"]["fields"] == "id,title,mean"

    @pytest.mark.asyncio
    async def test_search_anime_with_correlation(self, enhanced_mal_client):
        """Test search with correlation tracking."""
        sample_response = {"data": [{"id": 1, "title": "Test"}]}
        
        enhanced_mal_client.execution_tracer.start_trace = AsyncMock(return_value="trace-123")
        enhanced_mal_client.execution_tracer.end_trace = AsyncMock()

        with patch.object(
            enhanced_mal_client, "_make_request", return_value=sample_response
        ):
            result = await enhanced_mal_client.search_anime(
                q="test", correlation_id="search-correlation"
            )

            assert result == sample_response["data"]
            enhanced_mal_client.execution_tracer.start_trace.assert_called_once()
            enhanced_mal_client.execution_tracer.end_trace.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_anime_auto_correlation_id(self, enhanced_mal_client):
        """Test search auto-generates correlation ID."""
        sample_response = {"data": [{"id": 1, "title": "Test"}]}
        
        enhanced_mal_client.execution_tracer.start_trace = AsyncMock(return_value="trace-123")
        enhanced_mal_client.execution_tracer.end_trace = AsyncMock()

        with patch.object(
            enhanced_mal_client, "_make_request", return_value=sample_response
        ):
            result = await enhanced_mal_client.search_anime(q="test")

            assert result == sample_response["data"]
            # Should auto-generate correlation ID and trace
            enhanced_mal_client.execution_tracer.start_trace.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_anime_by_id_basic(self, mal_client):
        """Test get anime by ID."""
        sample_response = {"id": 1, "title": "Cowboy Bebop", "mean": 8.78}

        with patch.object(
            mal_client, "_make_request", return_value=sample_response
        ) as mock_request:
            result = await mal_client.get_anime_by_id(anime_id=1)

            assert result == sample_response
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert "/anime/1" in call_args.args[0]

    @pytest.mark.asyncio
    async def test_get_anime_by_id_with_fields(self, mal_client):
        """Test get anime by ID with specific fields."""
        sample_response = {"id": 1, "title": "Test", "genres": [{"id": 1, "name": "Action"}]}

        with patch.object(
            mal_client, "_make_request", return_value=sample_response
        ) as mock_request:
            result = await mal_client.get_anime_by_id(
                anime_id=1, 
                fields="id,title,genres",
                correlation_id="get-test"
            )

            assert result == sample_response
            call_args = mock_request.call_args
            assert call_args.kwargs["params"]["fields"] == "id,title,genres"

    @pytest.mark.asyncio
    async def test_get_anime_ranking(self, mal_client):
        """Test get anime ranking."""
        sample_response = {
            "data": [
                {"node": {"id": 1, "title": "Fullmetal Alchemist"}, "ranking": {"rank": 1}},
                {"node": {"id": 2, "title": "Steins;Gate"}, "ranking": {"rank": 2}}
            ]
        }

        with patch.object(
            mal_client, "_make_request", return_value=sample_response
        ) as mock_request:
            result = await mal_client.get_anime_ranking(
                ranking_type="all",
                limit=10
            )

            assert result == sample_response["data"]
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert "/anime/ranking" in call_args.args[0]
            assert call_args.kwargs["params"]["ranking_type"] == "all"
            assert call_args.kwargs["params"]["limit"] == 10

    @pytest.mark.asyncio
    async def test_get_seasonal_anime(self, mal_client):
        """Test get seasonal anime."""
        sample_response = {
            "data": [
                {"node": {"id": 1, "title": "Spring Anime 1"}},
                {"node": {"id": 2, "title": "Spring Anime 2"}}
            ]
        }

        with patch.object(
            mal_client, "_make_request", return_value=sample_response
        ) as mock_request:
            result = await mal_client.get_seasonal_anime(
                year=2023,
                season="spring",
                limit=10
            )

            assert result == sample_response["data"]
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert "/anime/season/2023/spring" in call_args.args[0]

    @pytest.mark.asyncio
    async def test_get_user_anime_list(self, mal_client):
        """Test get user anime list."""
        sample_response = {
            "data": [
                {"node": {"id": 1, "title": "Test Anime"}, "list_status": {"status": "completed"}},
            ]
        }

        with patch.object(
            mal_client, "_make_request", return_value=sample_response
        ) as mock_request:
            result = await mal_client.get_user_anime_list(
                username="testuser",
                status="completed",
                limit=50
            )

            assert result == sample_response["data"]
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert "/users/testuser/animelist" in call_args.args[0]
            assert call_args.kwargs["params"]["status"] == "completed"

    # OAuth2 methods are implemented but tested separately

    @pytest.mark.asyncio
    async def test_request_authentication_headers(self, mal_client):
        """Test request includes proper authentication headers."""
        mal_client.access_token = "test_access_token"
        sample_response = {"id": 1, "title": "Test"}

        with patch.object(
            mal_client, "make_request", return_value=sample_response
        ) as mock_request:
            await mal_client._make_request("/anime/1")

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            headers = call_args.kwargs["headers"]
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer test_access_token"

    @pytest.mark.asyncio
    async def test_request_client_id_fallback(self, mal_client):
        """Test request uses client_id when no access token."""
        mal_client.access_token = None
        sample_response = {"id": 1, "title": "Test"}

        with patch.object(
            mal_client, "make_request", return_value=sample_response
        ) as mock_request:
            await mal_client._make_request("/anime/1")

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            headers = call_args.kwargs["headers"]
            assert "X-MAL-CLIENT-ID" in headers
            assert headers["X-MAL-CLIENT-ID"] == "test_client_id"

    @pytest.mark.asyncio
    async def test_error_handling_401(self, enhanced_mal_client):
        """Test 401 authentication error handling."""
        with patch.object(
            enhanced_mal_client, "_make_request", side_effect=APIError("401 Unauthorized")
        ):
            with patch.object(
                enhanced_mal_client, "create_mal_error_context"
            ) as mock_error:
                result = await enhanced_mal_client.search_anime(
                    q="test", correlation_id="error-test"
                )

                assert result == []
                mock_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_rate_limit(self, enhanced_mal_client):
        """Test rate limit error handling."""
        with patch.object(
            enhanced_mal_client, "_make_request", side_effect=APIError("429 Rate Limit")
        ):
            with patch.object(
                enhanced_mal_client, "create_mal_error_context"
            ) as mock_error:
                result = await enhanced_mal_client.get_anime_by_id(
                    anime_id=1, correlation_id="rate-limit-test"
                )

                assert result is None
                mock_error.assert_called_once()

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
    async def test_backward_compatibility(self, mal_client):
        """Test all methods work without enhanced components."""
        sample_response = {"data": [{"id": 1, "title": "Test"}]}

        with patch.object(
            mal_client, "_make_request", return_value=sample_response
        ):
            # Search should work
            result = await mal_client.search_anime(q="test")
            assert result == sample_response["data"]

        # Get by ID should work
        anime_response = {"id": 1, "title": "Test"}
        with patch.object(
            mal_client, "_make_request", return_value=anime_response
        ):
            result = await mal_client.get_anime_by_id(anime_id=1)
            assert result == anime_response

    # Rate limit handling methods exist but are tested separately

    # Cache integration tested separately

    # Cache set functionality tested separately

    @pytest.mark.asyncio
    async def test_cache_error_graceful_degradation(self, enhanced_mal_client):
        """Test cache errors don't break the flow."""
        api_response = {"id": 1, "title": "Test Anime"}
        
        # Mock cache to throw error
        mock_cache = AsyncMock()
        mock_cache.get.side_effect = Exception("Cache connection failed")
        enhanced_mal_client.cache_manager = mock_cache

        with patch.object(
            enhanced_mal_client, "_make_request", return_value=api_response
        ):
            result = await enhanced_mal_client.get_anime_by_id(
                anime_id=1, correlation_id="cache-error-test"
            )

            assert result == api_response
            # Should have attempted cache get despite error
            mock_cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_anime_empty_query_validation(self, mal_client):
        """Test search anime validates empty query."""
        with pytest.raises(ValueError, match="Search query 'q' is required for MAL API"):
            await mal_client.search_anime(q="")

    @pytest.mark.asyncio
    async def test_search_anime_no_data_response(self, mal_client):
        """Test search anime when response has no 'data' field."""
        sample_response = {"error": "Search failed"}

        with patch.object(
            mal_client, "_make_request", return_value=sample_response
        ):
            result = await mal_client.search_anime(q="test")
            assert result == []

    @pytest.mark.asyncio
    async def test_get_anime_ranking_no_data_response(self, mal_client):
        """Test get anime ranking when response has no 'data' field."""
        sample_response = {"error": "Ranking failed"}

        with patch.object(
            mal_client, "_make_request", return_value=sample_response
        ):
            result = await mal_client.get_anime_ranking(ranking_type="all")
            assert result == []

    @pytest.mark.asyncio
    async def test_get_seasonal_anime_no_data_response(self, mal_client):
        """Test get seasonal anime when response has no 'data' field."""
        sample_response = {"error": "Seasonal failed"}

        with patch.object(
            mal_client, "_make_request", return_value=sample_response
        ):
            result = await mal_client.get_seasonal_anime(year=2023, season="spring")
            assert result == []

    @pytest.mark.asyncio
    async def test_get_user_anime_list_no_data_response(self, mal_client):
        """Test get user anime list when response has no 'data' field."""
        sample_response = {"error": "User list failed"}

        with patch.object(
            mal_client, "_make_request", return_value=sample_response
        ):
            result = await mal_client.get_user_anime_list(username="testuser")
            assert result == []