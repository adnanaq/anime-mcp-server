"""Tests for MAL service implementation using MAL API v2."""

import pytest
from unittest.mock import AsyncMock, patch

from src.services.external.mal_service import MALService


class TestMALServiceV2:
    """Test MAL service with proper MAL API v2 client integration."""

    @pytest.fixture
    def mal_service(self):
        """Create MAL service for testing."""
        return MALService(client_id="test_client_id", client_secret="test_client_secret")

    @pytest.fixture
    def mal_service_minimal(self):
        """Create MAL service with minimal configuration."""
        return MALService(client_id="test_client_id")

    def test_mal_service_initialization(self, mal_service):
        """Test MAL service initializes correctly."""
        assert mal_service.service_name == "mal"
        assert mal_service.client is not None

    def test_mal_service_requires_client_id(self):
        """Test MAL service requires client_id."""
        with pytest.raises(TypeError):
            MALService()  # Should fail without client_id

    @pytest.mark.asyncio
    async def test_search_anime_basic(self, mal_service):
        """Test basic anime search."""
        sample_response = [
            {"id": 1, "title": "Cowboy Bebop"},
            {"id": 2, "title": "Trigun"}
        ]

        with patch.object(
            mal_service.client, "search_anime", return_value=sample_response
        ) as mock_search:
            result = await mal_service.search_anime(
                query="cowboy",
                limit=10,
                correlation_id="search-test"
            )

            assert result == sample_response
            mock_search.assert_called_once_with(
                q="cowboy",
                limit=10,
                offset=0,
                fields=None,
                correlation_id="search-test"
            )

    @pytest.mark.asyncio
    async def test_search_anime_with_fields(self, mal_service):
        """Test anime search with specific fields."""
        sample_response = [{"id": 1, "title": "Test", "mean": 8.5}]

        with patch.object(
            mal_service.client, "search_anime", return_value=sample_response
        ) as mock_search:
            result = await mal_service.search_anime(
                query="test",
                limit=5,
                offset=10,
                fields="id,title,mean",
                correlation_id="search-fields-test"
            )

            assert result == sample_response
            mock_search.assert_called_once_with(
                q="test",
                limit=5,
                offset=10,
                fields="id,title,mean",
                correlation_id="search-fields-test"
            )

    @pytest.mark.asyncio
    async def test_search_anime_empty_query_validation(self, mal_service):
        """Test search anime validates empty query."""
        with pytest.raises(ValueError, match="Query parameter is required"):
            await mal_service.search_anime(query="")

    @pytest.mark.asyncio
    async def test_get_anime_details_basic(self, mal_service):
        """Test get anime details."""
        sample_response = {"id": 1, "title": "Cowboy Bebop", "mean": 8.78}

        with patch.object(
            mal_service.client, "get_anime_by_id", return_value=sample_response
        ) as mock_get:
            result = await mal_service.get_anime_details(
                anime_id=1,
                correlation_id="details-test"
            )

            assert result == sample_response
            mock_get.assert_called_once_with(
                anime_id=1,
                fields=None,
                correlation_id="details-test"
            )

    @pytest.mark.asyncio
    async def test_get_anime_details_with_fields(self, mal_service):
        """Test get anime details with specific fields."""
        sample_response = {"id": 1, "title": "Test", "genres": [{"id": 1, "name": "Action"}]}

        with patch.object(
            mal_service.client, "get_anime_by_id", return_value=sample_response
        ) as mock_get:
            result = await mal_service.get_anime_details(
                anime_id=1,
                fields="id,title,genres",
                correlation_id="details-fields-test"
            )

            assert result == sample_response
            mock_get.assert_called_once_with(
                anime_id=1,
                fields="id,title,genres",
                correlation_id="details-fields-test"
            )

    @pytest.mark.asyncio
    async def test_get_user_anime_list(self, mal_service):
        """Test get user anime list."""
        sample_response = [
            {"node": {"id": 1, "title": "Test Anime"}, "list_status": {"status": "completed"}},
        ]

        with patch.object(
            mal_service.client, "get_user_anime_list", return_value=sample_response
        ) as mock_get:
            result = await mal_service.get_user_anime_list(
                username="testuser",
                status="completed",
                limit=50,
                correlation_id="userlist-test"
            )

            assert result == sample_response
            mock_get.assert_called_once_with(
                username="testuser",
                status="completed",
                sort=None,
                limit=50,
                offset=0,
                fields=None,
                correlation_id="userlist-test"
            )

    @pytest.mark.asyncio
    async def test_get_anime_ranking(self, mal_service):
        """Test get anime ranking."""
        sample_response = [
            {"node": {"id": 1, "title": "Fullmetal Alchemist"}, "ranking": {"rank": 1}},
            {"node": {"id": 2, "title": "Steins;Gate"}, "ranking": {"rank": 2}}
        ]

        with patch.object(
            mal_service.client, "get_anime_ranking", return_value=sample_response
        ) as mock_get:
            result = await mal_service.get_anime_ranking(
                ranking_type="all",
                limit=10,
                correlation_id="ranking-test"
            )

            assert result == sample_response
            mock_get.assert_called_once_with(
                ranking_type="all",
                limit=10,
                offset=0,
                fields=None,
                correlation_id="ranking-test"
            )

    @pytest.mark.asyncio
    async def test_get_seasonal_anime(self, mal_service):
        """Test get seasonal anime."""
        sample_response = [
            {"node": {"id": 1, "title": "Spring Anime 1"}},
            {"node": {"id": 2, "title": "Spring Anime 2"}}
        ]

        with patch.object(
            mal_service.client, "get_seasonal_anime", return_value=sample_response
        ) as mock_get:
            result = await mal_service.get_seasonal_anime(
                year=2023,
                season="spring",
                limit=10,
                correlation_id="seasonal-test"
            )

            assert result == sample_response
            mock_get.assert_called_once_with(
                year=2023,
                season="spring",
                sort=None,
                limit=10,
                offset=0,
                fields=None,
                correlation_id="seasonal-test"
            )

    @pytest.mark.asyncio
    async def test_get_seasonal_anime_invalid_season(self, mal_service):
        """Test get seasonal anime with invalid season."""
        with pytest.raises(ValueError, match="Invalid season 'invalid'"):
            await mal_service.get_seasonal_anime(
                year=2023,
                season="invalid",
                correlation_id="invalid-season-test"
            )

    @pytest.mark.asyncio
    async def test_health_check_success(self, mal_service):
        """Test health check when service is healthy."""
        with patch.object(
            mal_service.client, "search_anime", return_value=[]
        ):
            result = await mal_service.health_check()

            assert result["service"] == "mal"
            assert result["status"] == "healthy"
            assert "circuit_breaker_open" in result

    @pytest.mark.asyncio
    async def test_health_check_failure(self, mal_service):
        """Test health check when service is unhealthy."""
        with patch.object(
            mal_service.client, "search_anime", side_effect=Exception("API failed")
        ):
            result = await mal_service.health_check()

            assert result["service"] == "mal"
            assert result["status"] == "unhealthy"
            assert "error" in result
            assert "circuit_breaker_open" in result

    @pytest.mark.asyncio
    async def test_search_anime_error_handling(self, mal_service):
        """Test search anime error handling."""
        with patch.object(
            mal_service.client, "search_anime", side_effect=Exception("Search failed")
        ):
            with pytest.raises(Exception, match="Search failed"):
                await mal_service.search_anime(
                    query="test",
                    correlation_id="error-test"
                )

    @pytest.mark.asyncio
    async def test_get_anime_details_error_handling(self, mal_service):
        """Test get anime details error handling."""
        with patch.object(
            mal_service.client, "get_anime_by_id", side_effect=Exception("Details failed")
        ):
            with pytest.raises(Exception, match="Details failed"):
                await mal_service.get_anime_details(
                    anime_id=1,
                    correlation_id="error-test"
                )

    @pytest.mark.asyncio
    async def test_get_user_anime_list_error_handling(self, mal_service):
        """Test get user anime list error handling."""
        with patch.object(
            mal_service.client, "get_user_anime_list", side_effect=Exception("User list failed")
        ):
            with pytest.raises(Exception, match="User list failed"):
                await mal_service.get_user_anime_list(
                    username="testuser",
                    correlation_id="error-test"
                )

    @pytest.mark.asyncio
    async def test_get_anime_ranking_error_handling(self, mal_service):
        """Test get anime ranking error handling."""
        with patch.object(
            mal_service.client, "get_anime_ranking", side_effect=Exception("Ranking failed")
        ):
            with pytest.raises(Exception, match="Ranking failed"):
                await mal_service.get_anime_ranking(
                    ranking_type="all",
                    correlation_id="error-test"
                )

    @pytest.mark.asyncio
    async def test_get_seasonal_anime_error_handling(self, mal_service):
        """Test get seasonal anime error handling."""
        with patch.object(
            mal_service.client, "get_seasonal_anime", side_effect=Exception("Seasonal failed")
        ):
            with pytest.raises(Exception, match="Seasonal failed"):
                await mal_service.get_seasonal_anime(
                    year=2023,
                    season="spring",
                    correlation_id="error-test"
                )

    @pytest.mark.asyncio
    async def test_service_integration_with_base_components(self, mal_service):
        """Test service integrates properly with base service components."""
        # Test that service has inherited base service functionality
        assert hasattr(mal_service, "circuit_breaker")
        assert hasattr(mal_service, "cache_manager")
        assert mal_service.service_name == "mal"

    @pytest.mark.asyncio
    async def test_correlation_id_propagation(self, mal_service):
        """Test correlation ID is properly propagated to client."""
        correlation_id = "test-correlation-123"
        
        with patch.object(
            mal_service.client, "search_anime", return_value=[]
        ) as mock_search:
            await mal_service.search_anime(
                query="test",
                correlation_id=correlation_id
            )

            mock_search.assert_called_once()
            call_args = mock_search.call_args
            assert call_args.kwargs["correlation_id"] == correlation_id

    @pytest.mark.asyncio
    async def test_logging_integration(self, mal_service):
        """Test service logs operations properly."""
        with patch.object(
            mal_service.client, "search_anime", return_value=[]
        ):
            with patch("src.services.external.mal_service.logger") as mock_logger:
                await mal_service.search_anime(
                    query="test",
                    correlation_id="log-test"
                )

                # Should log the search operation
                mock_logger.info.assert_called()
                call_args = mock_logger.info.call_args[0]
                assert "MAL search" in call_args[0]
                assert "test" in str(call_args)

    @pytest.mark.asyncio
    async def test_multiple_operations_work_together(self, mal_service):
        """Test multiple service operations work together."""
        search_response = [{"id": 1, "title": "Test Anime"}]
        details_response = {"id": 1, "title": "Test Anime", "mean": 8.5}
        ranking_response = [{"node": {"id": 1, "title": "Test Anime"}, "ranking": {"rank": 1}}]

        with patch.object(mal_service.client, "search_anime", return_value=search_response):
            with patch.object(mal_service.client, "get_anime_by_id", return_value=details_response):
                with patch.object(mal_service.client, "get_anime_ranking", return_value=ranking_response):
                    # Perform multiple operations
                    search_result = await mal_service.search_anime(query="test")
                    details_result = await mal_service.get_anime_details(anime_id=1)
                    ranking_result = await mal_service.get_anime_ranking()

                    assert search_result == search_response
                    assert details_result == details_response
                    assert ranking_result == ranking_response