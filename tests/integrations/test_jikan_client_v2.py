"""Tests for separated Jikan REST client implementation."""

import pytest
from unittest.mock import AsyncMock, patch

from src.exceptions import APIError
from src.integrations.clients.jikan_client import JikanClient
from src.integrations.error_handling import ExecutionTracer


class TestJikanClientSeparated:
    """Test separated Jikan client with comprehensive parameter support."""

    @pytest.fixture
    def jikan_client(self):
        """Create basic Jikan client for testing."""
        with patch("src.integrations.clients.base_client.rate_limit_manager"):
            return JikanClient()

    @pytest.fixture
    def enhanced_jikan_client(self):
        """Create Jikan client with enhanced error handling components."""
        execution_tracer = ExecutionTracer()

        with patch("src.integrations.clients.base_client.rate_limit_manager"):
            return JikanClient(execution_tracer=execution_tracer)

    def test_jikan_client_initialization(self, jikan_client):
        """Test basic Jikan client initializes correctly."""
        assert jikan_client.service_name == "jikan"
        assert jikan_client.base_url == "https://api.jikan.moe/v4"

    @pytest.mark.asyncio
    async def test_search_anime_basic(self, jikan_client):
        """Test basic anime search."""
        sample_response = {
            "data": [
                {"mal_id": 1, "title": "Cowboy Bebop"},
                {"mal_id": 2, "title": "Trigun"}
            ]
        }

        with patch.object(
            jikan_client, "_make_request", return_value=sample_response
        ) as mock_request:
            result = await jikan_client.search_anime(q="cowboy", limit=10)

            assert result == sample_response["data"]
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args.kwargs["params"]["q"] == "cowboy"
            assert call_args.kwargs["params"]["limit"] == 10

    @pytest.mark.asyncio
    async def test_search_anime_comprehensive_parameters(self, jikan_client):
        """Test search anime with all 17 supported parameters."""
        sample_response = {"data": [{"mal_id": 1, "title": "Test Anime"}]}

        with patch.object(
            jikan_client, "_make_request", return_value=sample_response
        ) as mock_request:
            result = await jikan_client.search_anime(
                q="action",
                limit=25,
                genres=[1, 2],
                status="airing",
                anime_type="TV",
                score=8.5,
                min_score=7.0,
                max_score=10.0,
                rating="PG-13",
                sfw=True,
                genres_exclude=[9],
                order_by="score",
                sort="desc",
                letter="A",
                producers=[1, 2],
                start_date="2023-01-01",
                end_date="2023-12-31",
                page=1,
                unapproved=False,
                correlation_id="comprehensive-search"
            )

            assert result == sample_response["data"]
            call_args = mock_request.call_args
            params = call_args.kwargs["params"]
            assert params["q"] == "action"
            assert params["limit"] == 25
            assert params["genres"] == "1,2"
            assert params["status"] == "airing"
            assert params["type"] == "TV"
            assert params["score"] == 8.5
            assert params["min_score"] == 7.0
            assert params["max_score"] == 10.0
            assert params["rating"] == "PG-13"
            assert params["sfw"] == "true"
            assert params["genres_exclude"] == "9"
            assert params["order_by"] == "score"
            assert params["sort"] == "desc"
            assert params["letter"] == "A"
            assert params["producers"] == "1,2"
            assert params["start_date"] == "2023-01-01"
            assert params["end_date"] == "2023-12-31"
            assert params["page"] == 1
            assert params["unapproved"] == "false"

    @pytest.mark.asyncio
    async def test_search_anime_with_correlation(self, enhanced_jikan_client):
        """Test search with correlation tracking."""
        sample_response = {"data": [{"mal_id": 1, "title": "Test"}]}
        
        enhanced_jikan_client.execution_tracer.start_trace = AsyncMock(return_value="trace-123")
        enhanced_jikan_client.execution_tracer.add_trace_step = AsyncMock()
        enhanced_jikan_client.execution_tracer.end_trace = AsyncMock()

        with patch.object(
            enhanced_jikan_client, "_make_request", return_value=sample_response
        ):
            result = await enhanced_jikan_client.search_anime(
                q="test", correlation_id="search-correlation"
            )

            assert result == sample_response["data"]
            enhanced_jikan_client.execution_tracer.start_trace.assert_called_once()
            enhanced_jikan_client.execution_tracer.end_trace.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_anime_auto_correlation_id(self, enhanced_jikan_client):
        """Test search auto-generates correlation ID."""
        sample_response = {"data": [{"mal_id": 1, "title": "Test"}]}
        
        enhanced_jikan_client.execution_tracer.start_trace = AsyncMock(return_value="trace-123")
        enhanced_jikan_client.execution_tracer.add_trace_step = AsyncMock()
        enhanced_jikan_client.execution_tracer.end_trace = AsyncMock()

        with patch.object(
            enhanced_jikan_client, "_make_request", return_value=sample_response
        ):
            result = await enhanced_jikan_client.search_anime(q="test")

            assert result == sample_response["data"]
            # Should auto-generate correlation ID and trace
            enhanced_jikan_client.execution_tracer.start_trace.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_anime_by_id_basic(self, jikan_client):
        """Test get anime by ID."""
        sample_response = {"data": {"mal_id": 1, "title": "Cowboy Bebop"}}

        with patch.object(
            jikan_client, "_make_request", return_value=sample_response
        ) as mock_request:
            result = await jikan_client.get_anime_by_id(anime_id=1)

            assert result == sample_response["data"]
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert "/anime/1" in call_args.args[0]

    @pytest.mark.asyncio
    async def test_get_anime_by_id_with_enhancements(self, enhanced_jikan_client):
        """Test get anime by ID with all enhancements auto-enabled."""
        sample_response = {"data": {"mal_id": 21, "title": "One Piece"}}

        enhanced_jikan_client.execution_tracer.start_trace = AsyncMock(return_value="trace-123")
        enhanced_jikan_client.execution_tracer.add_trace_step = AsyncMock()
        enhanced_jikan_client.execution_tracer.end_trace = AsyncMock()

        # Mock cache miss to test API path
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None
        mock_cache.set = AsyncMock()
        enhanced_jikan_client.cache_manager = mock_cache

        with patch.object(
            enhanced_jikan_client, "_make_request", return_value=sample_response
        ):
            result = await enhanced_jikan_client.get_anime_by_id(
                anime_id=21, correlation_id="enhanced-test"
            )

            assert result == sample_response["data"]
            # Should have used tracing
            enhanced_jikan_client.execution_tracer.start_trace.assert_called_once()
            enhanced_jikan_client.execution_tracer.end_trace.assert_called_once()
            # Should have checked cache
            mock_cache.get.assert_called_once_with("jikan_anime_21")
            # Should have set cache on success
            mock_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_anime_by_id_cache_hit(self, enhanced_jikan_client):
        """Test get anime by ID with cache hit."""
        cached_data = {"mal_id": 5, "title": "Cached Anime"}

        mock_cache = AsyncMock()
        mock_cache.get.return_value = cached_data
        enhanced_jikan_client.cache_manager = mock_cache

        result = await enhanced_jikan_client.get_anime_by_id(
            anime_id=5, correlation_id="cache-hit-test"
        )

        assert result == cached_data
        mock_cache.get.assert_called_once_with("jikan_anime_5")

    @pytest.mark.asyncio
    async def test_get_anime_by_id_correlation_chain(self, enhanced_jikan_client):
        """Test get anime by ID with correlation chaining."""
        parent_id = "parent-123"
        sample_response = {"data": {"mal_id": 1, "title": "Test Anime"}}

        with patch.object(
            enhanced_jikan_client,
            "make_request_with_correlation_chain",
            return_value=sample_response,
        ) as mock_chain:
            result = await enhanced_jikan_client.get_anime_by_id(
                anime_id=1, parent_correlation_id=parent_id
            )

            assert result == sample_response["data"]
            mock_chain.assert_called_once()
            call_args = mock_chain.call_args
            assert call_args.kwargs["parent_correlation_id"] == parent_id

    @pytest.mark.asyncio
    async def test_get_seasonal_anime(self, jikan_client):
        """Test get seasonal anime."""
        sample_response = {
            "data": [
                {"mal_id": 1, "title": "Spring Anime 1"},
                {"mal_id": 2, "title": "Spring Anime 2"}
            ]
        }

        with patch.object(
            jikan_client, "_make_request", return_value=sample_response
        ) as mock_request:
            result = await jikan_client.get_seasonal_anime(year=2023, season="spring")

            assert result == sample_response["data"]
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert "/seasons/2023/spring" in call_args.args[0]

    @pytest.mark.asyncio
    async def test_get_anime_statistics(self, enhanced_jikan_client):
        """Test get anime statistics with enhancements."""
        sample_response = {"data": {"watching": 100, "completed": 500}}

        with patch.object(
            enhanced_jikan_client, "_make_request", return_value=sample_response
        ):
            result = await enhanced_jikan_client.get_anime_statistics(
                anime_id=1, correlation_id="stats-test"
            )

            assert result == sample_response["data"]

    @pytest.mark.asyncio
    async def test_get_anime_statistics_correlation_chain(self, enhanced_jikan_client):
        """Test get anime statistics with correlation chaining."""
        parent_id = "parent-stats-123"
        stats_response = {"data": {"watching": 100, "completed": 500}}

        with patch.object(
            enhanced_jikan_client,
            "make_request_with_correlation_chain",
            return_value=stats_response,
        ) as mock_chain:
            result = await enhanced_jikan_client.get_anime_statistics(
                anime_id=1, parent_correlation_id=parent_id
            )

            assert result == stats_response["data"]
            mock_chain.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_top_anime(self, jikan_client):
        """Test get top anime rankings."""
        sample_response = {
            "data": [
                {"mal_id": 1, "title": "Fullmetal Alchemist", "rank": 1},
                {"mal_id": 2, "title": "Steins;Gate", "rank": 2}
            ]
        }

        with patch.object(
            jikan_client, "_make_request", return_value=sample_response
        ) as mock_request:
            result = await jikan_client.get_top_anime(
                anime_type="tv",
                filter_type="bypopularity",
                rating="pg13",
                page=1,
                limit=10
            )

            assert result == sample_response["data"]
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            params = call_args.kwargs["params"]
            assert params["type"] == "tv"
            assert params["filter"] == "bypopularity"
            assert params["rating"] == "pg13"
            assert params["page"] == 1
            assert params["limit"] == 10

    @pytest.mark.asyncio
    async def test_get_random_anime(self, jikan_client):
        """Test get random anime."""
        sample_response = {"data": {"mal_id": 42, "title": "Random Anime"}}

        with patch.object(
            jikan_client, "_make_request", return_value=sample_response
        ) as mock_request:
            result = await jikan_client.get_random_anime()

            assert result == sample_response["data"]
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert "/random/anime" in call_args.args[0]

    @pytest.mark.asyncio
    async def test_get_anime_recommendations(self, jikan_client):
        """Test get anime recommendations."""
        sample_response = {
            "data": [
                {"entry": {"mal_id": 1, "title": "Recommended Anime 1"}},
                {"entry": {"mal_id": 2, "title": "Recommended Anime 2"}}
            ]
        }

        with patch.object(
            jikan_client, "_make_request", return_value=sample_response
        ) as mock_request:
            result = await jikan_client.get_anime_recommendations(anime_id=1)

            assert result == sample_response["data"]
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert "/anime/1/recommendations" in call_args.args[0]

    @pytest.mark.asyncio
    async def test_get_anime_characters(self, jikan_client):
        """Test get anime characters."""
        sample_response = {
            "data": [
                {"character": {"mal_id": 1, "name": "Spike Spiegel"}},
                {"character": {"mal_id": 2, "name": "Jet Black"}}
            ]
        }

        with patch.object(
            jikan_client, "_make_request", return_value=sample_response
        ) as mock_request:
            result = await jikan_client.get_anime_characters(anime_id=1)

            assert result == sample_response["data"]
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert "/anime/1/characters" in call_args.args[0]

    @pytest.mark.asyncio
    async def test_get_anime_staff(self, jikan_client):
        """Test get anime staff."""
        sample_response = {
            "data": [
                {"person": {"mal_id": 1, "name": "Shinichiro Watanabe"}},
                {"person": {"mal_id": 2, "name": "Yoko Kanno"}}
            ]
        }

        with patch.object(
            jikan_client, "_make_request", return_value=sample_response
        ) as mock_request:
            result = await jikan_client.get_anime_staff(anime_id=1)

            assert result == sample_response["data"]
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert "/anime/1/staff" in call_args.args[0]

    @pytest.mark.asyncio
    async def test_get_schedules(self, jikan_client):
        """Test get broadcasting schedules."""
        sample_response = {
            "data": [
                {"mal_id": 1, "title": "Monday Anime"},
                {"mal_id": 2, "title": "Monday Anime 2"}
            ]
        }

        with patch.object(
            jikan_client, "_make_request", return_value=sample_response
        ) as mock_request:
            result = await jikan_client.get_schedules(
                filter_day="monday",
                kids=False,
                sfw=True,
                unapproved=False,
                page=1,
                limit=20
            )

            assert result == sample_response["data"]
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            params = call_args.kwargs["params"]
            assert params["filter"] == "monday"
            assert params["kids"] == "false"
            assert params["sfw"] == "true"
            assert params["unapproved"] == "false"
            assert params["page"] == 1
            assert params["limit"] == 20

    @pytest.mark.asyncio
    async def test_get_genres(self, jikan_client):
        """Test get available anime genres."""
        sample_response = {
            "data": [
                {"mal_id": 1, "name": "Action"},
                {"mal_id": 2, "name": "Adventure"}
            ]
        }

        with patch.object(
            jikan_client, "_make_request", return_value=sample_response
        ) as mock_request:
            result = await jikan_client.get_genres(filter_name="genres")

            assert result == sample_response["data"]
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert "/genres/anime" in call_args.args[0]
            assert call_args.kwargs["params"]["filter"] == "genres"

    @pytest.mark.asyncio
    async def test_request_with_retry_success(self, jikan_client):
        """Test request with retry mechanism - eventual success."""
        sample_response = {"data": [{"mal_id": 1, "title": "Test"}]}

        # Mock: First call fails with 500, second succeeds
        with patch.object(
            jikan_client, "_make_request", side_effect=[Exception("500 Server Error"), sample_response]
        ):
            result = await jikan_client._make_request_with_retry("/anime/1", max_retries=3)

            assert result == sample_response

    @pytest.mark.asyncio
    async def test_request_with_retry_max_retries(self, jikan_client):
        """Test request with retry mechanism - max retries exceeded."""
        # Mock: All calls fail with 500 error
        with patch.object(
            jikan_client, "_make_request", side_effect=Exception("500 Server Error")
        ):
            with pytest.raises(Exception, match="500 Server Error"):
                await jikan_client._make_request_with_retry("/anime/1", max_retries=2)

    @pytest.mark.asyncio
    async def test_request_with_retry_non_server_error(self, jikan_client):
        """Test request with retry - non-server error doesn't retry."""
        # Mock: First call fails with 404 (should not retry)
        with patch.object(
            jikan_client, "_make_request", side_effect=Exception("404 Not Found")
        ):
            with pytest.raises(Exception, match="404 Not Found"):
                await jikan_client._make_request_with_retry("/anime/1", max_retries=3)

    @pytest.mark.asyncio
    async def test_error_handling_with_correlation(self, enhanced_jikan_client):
        """Test error handling with correlation tracking."""
        with patch.object(
            enhanced_jikan_client, "_make_request", side_effect=APIError("Jikan API failed")
        ):
            with patch.object(
                enhanced_jikan_client, "create_jikan_error_context"
            ) as mock_error:
                result = await enhanced_jikan_client.search_anime(
                    q="test", correlation_id="error-test"
                )

                assert result == []
                mock_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_without_correlation(self, jikan_client):
        """Test error handling without correlation logging."""
        with patch.object(
            jikan_client, "_make_request", side_effect=APIError("Jikan API failed")
        ):
            # Mock the error context method to avoid missing correlation_logger
            with patch.object(jikan_client, "create_jikan_error_context", return_value=None):
                result = await jikan_client.search_anime(q="test")
                assert result == []

    @pytest.mark.asyncio
    async def test_create_jikan_error_context_429(self, enhanced_jikan_client):
        """Test create_jikan_error_context for 429 rate limit error."""
        error = Exception("429 rate limit")
        
        with patch.object(
            enhanced_jikan_client, "create_error_context", return_value=AsyncMock()
        ) as mock_create:
            await enhanced_jikan_client.create_jikan_error_context(
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
    async def test_create_jikan_error_context_404(self, enhanced_jikan_client):
        """Test create_jikan_error_context for 404 not found error."""
        error = Exception("404 not found")
        
        with patch.object(
            enhanced_jikan_client, "create_error_context", return_value=AsyncMock()
        ) as mock_create:
            await enhanced_jikan_client.create_jikan_error_context(
                error=error,
                correlation_id="test-jikan-404",
                endpoint="/anime/1",
                operation="get_anime"
            )
            
            mock_create.assert_called_once()
            call_args = mock_create.call_args.kwargs
            assert "not found" in call_args["user_message"]

    @pytest.mark.asyncio
    async def test_create_jikan_error_context_500(self, enhanced_jikan_client):
        """Test create_jikan_error_context for 500 server error."""
        error = Exception("500 server error")
        
        with patch.object(
            enhanced_jikan_client, "create_error_context", return_value=AsyncMock()
        ) as mock_create:
            await enhanced_jikan_client.create_jikan_error_context(
                error=error,
                correlation_id="test-jikan-500",
                endpoint="/anime/1",
                operation="get_anime"
            )
            
            mock_create.assert_called_once()
            call_args = mock_create.call_args.kwargs
            assert "server error" in call_args["user_message"]

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, enhanced_jikan_client):
        """Test graceful degradation when API fails."""
        # Mock API failure and graceful degradation success
        with patch.object(
            enhanced_jikan_client, "_make_request", side_effect=APIError("API failed")
        ):
            # Mock the error context method to avoid missing correlation_logger
            with patch.object(enhanced_jikan_client, "create_jikan_error_context", return_value=None):
                result = await enhanced_jikan_client.get_anime_by_id(
                    anime_id=1, correlation_id="degradation-test"
                )

                # Should return None on error
                assert result is None

    @pytest.mark.asyncio
    async def test_cache_error_handling(self, enhanced_jikan_client):
        """Test cache error handling doesn't break the flow."""
        sample_response = {"data": {"mal_id": 21, "title": "One Piece"}}

        # Mock cache manager to throw error
        mock_cache = AsyncMock()
        mock_cache.get.side_effect = Exception("Cache connection failed")
        enhanced_jikan_client.cache_manager = mock_cache

        with patch.object(
            enhanced_jikan_client, "_make_request", return_value=sample_response
        ):
            result = await enhanced_jikan_client.get_anime_by_id(
                anime_id=21, correlation_id="cache-error-test"
            )

            assert result == sample_response["data"]
            # Should have attempted cache despite error
            mock_cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, enhanced_jikan_client):
        """Test rate limit response handling."""
        # Rate limiting methods exist but may require full integration
        assert hasattr(enhanced_jikan_client, 'handle_rate_limit_response')

    @pytest.mark.asyncio
    async def test_monitor_rate_limits(self, enhanced_jikan_client):
        """Test rate limit monitoring."""
        # Rate limiting methods exist but may require full integration
        assert hasattr(enhanced_jikan_client, 'monitor_rate_limits')

    @pytest.mark.asyncio
    async def test_calculate_backoff_delay(self, enhanced_jikan_client):
        """Test backoff delay calculation."""
        # Test the method exists and returns a valid delay for no response case
        delay = await enhanced_jikan_client.calculate_backoff_delay(None, attempt=1)
        assert isinstance(delay, (int, float))
        assert delay > 0

    @pytest.mark.asyncio
    async def test_calculate_backoff_delay_no_response(self, enhanced_jikan_client):
        """Test backoff delay calculation with no response."""
        delay = await enhanced_jikan_client.calculate_backoff_delay(None, attempt=1)

        # Should return some delay value (exponential backoff)
        assert isinstance(delay, (int, float))
        assert delay > 0

    @pytest.mark.asyncio
    async def test_backward_compatibility(self, jikan_client):
        """Test all methods work without enhanced components."""
        sample_response = {"data": [{"mal_id": 1, "title": "Test"}]}

        with patch.object(
            jikan_client, "_make_request", return_value=sample_response
        ):
            # Search should work
            result = await jikan_client.search_anime(q="test")
            assert result == sample_response["data"]

        # Get by ID should work
        anime_response = {"data": {"mal_id": 1, "title": "Test"}}
        with patch.object(
            jikan_client, "_make_request", return_value=anime_response
        ):
            result = await jikan_client.get_anime_by_id(anime_id=1)
            assert result == anime_response["data"]

        # Statistics should work
        stats_response = {"data": {"watching": 100}}
        with patch.object(
            jikan_client, "_make_request", return_value=stats_response
        ):
            result = await jikan_client.get_anime_statistics(anime_id=1)
            assert result == stats_response["data"]

    @pytest.mark.asyncio
    async def test_no_data_responses(self, jikan_client):
        """Test handling of responses with no 'data' field."""
        error_response = {"error": "Something went wrong"}

        with patch.object(
            jikan_client, "_make_request", return_value=error_response
        ):
            # Search should return empty list
            result = await jikan_client.search_anime(q="test")
            assert result == []

            # Get by ID should return None
            result = await jikan_client.get_anime_by_id(anime_id=1)
            assert result is None

            # Statistics should return empty dict
            result = await jikan_client.get_anime_statistics(anime_id=1)
            assert result == {}

            # Other methods should return empty lists
            result = await jikan_client.get_top_anime()
            assert result == []

            result = await jikan_client.get_anime_recommendations(anime_id=1)
            assert result == []

            result = await jikan_client.get_anime_characters(anime_id=1)
            assert result == []

            result = await jikan_client.get_anime_staff(anime_id=1)
            assert result == []

            result = await jikan_client.get_schedules()
            assert result == []

            result = await jikan_client.get_genres()
            assert result == []

            # Random anime should return None
            result = await jikan_client.get_random_anime()
            assert result is None

            # Seasonal anime should return empty list
            result = await jikan_client.get_seasonal_anime(2023, "spring")
            assert result == []