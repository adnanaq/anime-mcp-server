"""Tests for Jikan service implementation."""

import pytest
from unittest.mock import AsyncMock, patch, Mock

from src.services.external.jikan_service import JikanService


class TestJikanService:
    """Test Jikan service with pure Jikan API implementation."""

    @pytest.fixture
    def jikan_service(self):
        """Create Jikan service for testing."""
        return JikanService()

    def test_jikan_service_initialization(self, jikan_service):
        """Test Jikan service initializes correctly."""
        assert jikan_service.service_name == "jikan"
        assert jikan_service.client is not None

    @pytest.mark.asyncio
    async def test_search_anime_basic(self, jikan_service):
        """Test basic anime search."""
        sample_response = [
            {"mal_id": 1, "title": "Cowboy Bebop"},
            {"mal_id": 2, "title": "Trigun"}
        ]

        with patch.object(
            jikan_service.client, "search_anime", return_value=sample_response
        ) as mock_search:
            result = await jikan_service.search_anime(
                query="cowboy",
                limit=10,
                correlation_id="search-test"
            )

            assert result == sample_response
            mock_search.assert_called_once_with(
                q="cowboy",
                limit=10,
                status=None,
                genres=None,
                anime_type=None,
                score=None,
                min_score=None,
                max_score=None,
                rating=None,
                sfw=None,
                genres_exclude=None,
                order_by=None,
                sort=None,
                letter=None,
                producers=None,
                start_date=None,
                end_date=None,
                page=None,
                unapproved=None,
                correlation_id="search-test"
            )

    @pytest.mark.asyncio
    async def test_search_anime_comprehensive(self, jikan_service):
        """Test anime search with all Jikan parameters."""
        sample_response = [{"mal_id": 1, "title": "Test Anime"}]

        with patch.object(
            jikan_service.client, "search_anime", return_value=sample_response
        ) as mock_search:
            result = await jikan_service.search_anime(
                query="action",
                limit=25,
                status="airing",
                genres=[1, 2],
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

            assert result == sample_response
            mock_search.assert_called_once()
            call_args = mock_search.call_args
            assert call_args.kwargs["q"] == "action"
            assert call_args.kwargs["limit"] == 25
            assert call_args.kwargs["status"] == "airing"
            assert call_args.kwargs["genres"] == [1, 2]
            assert call_args.kwargs["anime_type"] == "TV"
            assert call_args.kwargs["correlation_id"] == "comprehensive-search"

    @pytest.mark.asyncio
    async def test_search_anime_parameter_validation_invalid_type(self, jikan_service):
        """Test search anime parameter validation for invalid anime type."""
        with pytest.raises(ValueError, match="Invalid anime type 'Invalid'"):
            await jikan_service.search_anime(
                query="test",
                anime_type="Invalid"
            )

    @pytest.mark.asyncio
    async def test_search_anime_parameter_validation_score_range(self, jikan_service):
        """Test search anime parameter validation for invalid score range."""
        with pytest.raises(ValueError, match="min_score must be less than or equal to max_score"):
            await jikan_service.search_anime(
                query="test",
                min_score=8.0,
                max_score=7.0
            )

    @pytest.mark.asyncio
    async def test_search_anime_parameter_validation_invalid_rating(self, jikan_service):
        """Test search anime parameter validation for invalid rating."""
        with pytest.raises(ValueError, match="Invalid rating 'Invalid'"):
            await jikan_service.search_anime(
                query="test",
                rating="Invalid"
            )

    @pytest.mark.asyncio
    async def test_search_anime_parameter_validation_invalid_order_by(self, jikan_service):
        """Test search anime parameter validation for invalid order_by."""
        with pytest.raises(ValueError, match="Invalid order_by 'invalid'"):
            await jikan_service.search_anime(
                query="test",
                order_by="invalid"
            )

    @pytest.mark.asyncio
    async def test_search_anime_parameter_validation_invalid_sort(self, jikan_service):
        """Test search anime parameter validation for invalid sort."""
        with pytest.raises(ValueError, match="Invalid sort 'invalid'"):
            await jikan_service.search_anime(
                query="test",
                sort="invalid"
            )

    @pytest.mark.asyncio
    async def test_search_anime_parameter_validation_invalid_date(self, jikan_service):
        """Test search anime parameter validation for invalid date format."""
        with pytest.raises(ValueError, match="Invalid start_date format"):
            await jikan_service.search_anime(
                query="test",
                start_date="invalid-date"
            )

    @pytest.mark.asyncio
    async def test_get_anime_details_basic(self, jikan_service):
        """Test get anime details."""
        sample_response = {"mal_id": 1, "title": "Cowboy Bebop"}

        with patch.object(
            jikan_service.client, "get_anime_by_id", return_value=sample_response
        ) as mock_get:
            result = await jikan_service.get_anime_details(
                anime_id=1,
                correlation_id="details-test"
            )

            assert result == sample_response
            mock_get.assert_called_once_with(
                1,
                correlation_id="details-test"
            )

    @pytest.mark.asyncio
    async def test_get_seasonal_anime(self, jikan_service):
        """Test get seasonal anime."""
        sample_response = [
            {"mal_id": 1, "title": "Spring Anime 1"},
            {"mal_id": 2, "title": "Spring Anime 2"}
        ]

        with patch.object(
            jikan_service.client, "get_seasonal_anime", return_value=sample_response
        ) as mock_get:
            result = await jikan_service.get_seasonal_anime(
                year=2023,
                season="spring",
                correlation_id="seasonal-test"
            )

            assert result == sample_response
            mock_get.assert_called_once_with(2023, "spring")

    @pytest.mark.asyncio
    async def test_get_seasonal_anime_invalid_season(self, jikan_service):
        """Test get seasonal anime with invalid season."""
        with pytest.raises(ValueError, match="Invalid season 'invalid'"):
            await jikan_service.get_seasonal_anime(
                year=2023,
                season="invalid",
                correlation_id="invalid-season-test"
            )

    @pytest.mark.asyncio
    async def test_get_current_season(self, jikan_service):
        """Test get current season anime."""
        sample_response = [{"mal_id": 1, "title": "Current Season Anime"}]

        with patch.object(
            jikan_service.client, "get_seasonal_anime", return_value=sample_response
        ) as mock_get:
            # Mock current date to be in spring (April)
            mock_date = Mock()
            mock_date.month = 4
            mock_date.year = 2023
            with patch("src.services.external.jikan_service.datetime") as mock_datetime:
                mock_datetime.now.return_value = mock_date
                
                result = await jikan_service.get_current_season(
                    correlation_id="current-season-test"
                )

                assert result == sample_response
                mock_get.assert_called_once_with(2023, "spring")

    @pytest.mark.asyncio
    async def test_get_anime_statistics(self, jikan_service):
        """Test get anime statistics."""
        sample_response = {"watching": 100, "completed": 500}

        with patch.object(
            jikan_service.client, "get_anime_statistics", return_value=sample_response
        ) as mock_get:
            result = await jikan_service.get_anime_statistics(
                anime_id=1,
                correlation_id="stats-test"
            )

            assert result == sample_response
            mock_get.assert_called_once_with(
                1,
                correlation_id="stats-test"
            )

    @pytest.mark.asyncio
    async def test_get_top_anime(self, jikan_service):
        """Test get top anime."""
        sample_response = [
            {"mal_id": 1, "title": "Top Anime 1", "rank": 1},
            {"mal_id": 2, "title": "Top Anime 2", "rank": 2}
        ]

        with patch.object(
            jikan_service.client, "get_top_anime", return_value=sample_response
        ) as mock_get:
            result = await jikan_service.get_top_anime(
                anime_type="tv",
                filter_type="bypopularity",
                rating="pg13",
                page=1,
                limit=10,
                correlation_id="top-anime-test"
            )

            assert result == sample_response
            mock_get.assert_called_once_with(
                anime_type="tv",
                filter_type="bypopularity",
                rating="pg13",
                page=1,
                limit=10
            )

    @pytest.mark.asyncio
    async def test_get_random_anime(self, jikan_service):
        """Test get random anime."""
        sample_response = {"mal_id": 42, "title": "Random Anime"}

        with patch.object(
            jikan_service.client, "get_random_anime", return_value=sample_response
        ) as mock_get:
            result = await jikan_service.get_random_anime(
                correlation_id="random-test"
            )

            assert result == sample_response
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_anime_recommendations(self, jikan_service):
        """Test get anime recommendations."""
        sample_response = [
            {"entry": {"mal_id": 1, "title": "Recommended Anime 1"}},
            {"entry": {"mal_id": 2, "title": "Recommended Anime 2"}}
        ]

        with patch.object(
            jikan_service.client, "get_anime_recommendations", return_value=sample_response
        ) as mock_get:
            result = await jikan_service.get_anime_recommendations(
                anime_id=1,
                correlation_id="recommendations-test"
            )

            assert result == sample_response
            mock_get.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_get_anime_characters(self, jikan_service):
        """Test get anime characters."""
        sample_response = [
            {"character": {"mal_id": 1, "name": "Spike Spiegel"}},
            {"character": {"mal_id": 2, "name": "Jet Black"}}
        ]

        with patch.object(
            jikan_service.client, "get_anime_characters", return_value=sample_response
        ) as mock_get:
            result = await jikan_service.get_anime_characters(
                anime_id=1,
                correlation_id="characters-test"
            )

            assert result == sample_response
            mock_get.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_get_anime_staff(self, jikan_service):
        """Test get anime staff."""
        sample_response = [
            {"person": {"mal_id": 1, "name": "Shinichiro Watanabe"}},
            {"person": {"mal_id": 2, "name": "Yoko Kanno"}}
        ]

        with patch.object(
            jikan_service.client, "get_anime_staff", return_value=sample_response
        ) as mock_get:
            result = await jikan_service.get_anime_staff(
                anime_id=1,
                correlation_id="staff-test"
            )

            assert result == sample_response
            mock_get.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_get_schedules(self, jikan_service):
        """Test get broadcasting schedules."""
        sample_response = [
            {"mal_id": 1, "title": "Monday Anime"},
            {"mal_id": 2, "title": "Monday Anime 2"}
        ]

        with patch.object(
            jikan_service.client, "get_schedules", return_value=sample_response
        ) as mock_get:
            result = await jikan_service.get_schedules(
                filter_day="monday",
                kids=False,
                sfw=True,
                unapproved=False,
                page=1,
                limit=20,
                correlation_id="schedules-test"
            )

            assert result == sample_response
            mock_get.assert_called_once_with(
                filter_day="monday",
                kids=False,
                sfw=True,
                unapproved=False,
                page=1,
                limit=20
            )

    @pytest.mark.asyncio
    async def test_get_genres(self, jikan_service):
        """Test get available genres."""
        sample_response = [
            {"mal_id": 1, "name": "Action"},
            {"mal_id": 2, "name": "Adventure"}
        ]

        with patch.object(
            jikan_service.client, "get_genres", return_value=sample_response
        ) as mock_get:
            result = await jikan_service.get_genres(
                filter_name="genres",
                correlation_id="genres-test"
            )

            assert result == sample_response
            mock_get.assert_called_once_with(filter_name="genres")

    @pytest.mark.asyncio
    async def test_health_check_success(self, jikan_service):
        """Test health check when service is healthy."""
        with patch.object(
            jikan_service.client, "search_anime", return_value=[]
        ):
            result = await jikan_service.health_check()

            assert result["service"] == "jikan"
            assert result["status"] == "healthy"
            assert "circuit_breaker_open" in result

    @pytest.mark.asyncio
    async def test_health_check_failure(self, jikan_service):
        """Test health check when service is unhealthy."""
        with patch.object(
            jikan_service.client, "search_anime", side_effect=Exception("API failed")
        ):
            result = await jikan_service.health_check()

            assert result["service"] == "jikan"
            assert result["status"] == "unhealthy"
            assert "error" in result
            assert "circuit_breaker_open" in result

    @pytest.mark.asyncio
    async def test_error_handling_propagation(self, jikan_service):
        """Test that service properly propagates client errors."""
        with patch.object(
            jikan_service.client, "search_anime", side_effect=Exception("Search failed")
        ):
            with pytest.raises(Exception, match="Search failed"):
                await jikan_service.search_anime(
                    query="test",
                    correlation_id="error-test"
                )

    @pytest.mark.asyncio
    async def test_service_integration_with_base_components(self, jikan_service):
        """Test service integrates properly with base service components."""
        # Test that service has inherited base service functionality
        assert hasattr(jikan_service, "circuit_breaker")
        assert hasattr(jikan_service, "cache_manager")
        assert jikan_service.service_name == "jikan"

    @pytest.mark.asyncio
    async def test_correlation_id_propagation(self, jikan_service):
        """Test correlation ID is properly propagated to client."""
        correlation_id = "test-correlation-123"
        
        with patch.object(
            jikan_service.client, "search_anime", return_value=[]
        ) as mock_search:
            await jikan_service.search_anime(
                query="test",
                correlation_id=correlation_id
            )

            mock_search.assert_called_once()
            call_args = mock_search.call_args
            assert call_args.kwargs["correlation_id"] == correlation_id

    @pytest.mark.asyncio
    async def test_logging_integration(self, jikan_service):
        """Test service logs operations properly."""
        with patch.object(
            jikan_service.client, "search_anime", return_value=[]
        ):
            with patch("src.services.external.jikan_service.logger") as mock_logger:
                await jikan_service.search_anime(
                    query="test",
                    correlation_id="log-test"
                )

                # Should log the search operation
                mock_logger.info.assert_called()
                call_args = mock_logger.info.call_args[0]
                assert "Jikan search" in call_args[0]
                assert "test" in str(call_args)

    @pytest.mark.asyncio
    async def test_seasonal_determination_logic(self, jikan_service):
        """Test seasonal determination logic for all seasons."""
        
        with patch.object(
            jikan_service.client, "get_seasonal_anime", return_value=[]
        ) as mock_get:
            # Test Winter (December)
            mock_winter = Mock()
            mock_winter.month = 12
            mock_winter.year = 2023
            with patch("src.services.external.jikan_service.datetime") as mock_datetime:
                mock_datetime.now.return_value = mock_winter
                await jikan_service.get_current_season()
                mock_get.assert_called_with(2023, "winter")

            # Test Spring (March)
            mock_spring = Mock()
            mock_spring.month = 3
            mock_spring.year = 2023
            with patch("src.services.external.jikan_service.datetime") as mock_datetime:
                mock_datetime.now.return_value = mock_spring
                await jikan_service.get_current_season()
                mock_get.assert_called_with(2023, "spring")

            # Test Summer (June)
            mock_summer = Mock()
            mock_summer.month = 6
            mock_summer.year = 2023
            with patch("src.services.external.jikan_service.datetime") as mock_datetime:
                mock_datetime.now.return_value = mock_summer
                await jikan_service.get_current_season()
                mock_get.assert_called_with(2023, "summer")

            # Test Fall (September)
            mock_fall = Mock()
            mock_fall.month = 9
            mock_fall.year = 2023
            with patch("src.services.external.jikan_service.datetime") as mock_datetime:
                mock_datetime.now.return_value = mock_fall
                await jikan_service.get_current_season()
                mock_get.assert_called_with(2023, "fall")

    @pytest.mark.asyncio
    async def test_comprehensive_workflow(self, jikan_service):
        """Test multiple service operations work together."""
        search_response = [{"mal_id": 1, "title": "Test Anime"}]
        details_response = {"mal_id": 1, "title": "Test Anime", "score": 8.5}
        stats_response = {"watching": 100, "completed": 500}

        with patch.object(jikan_service.client, "search_anime", return_value=search_response):
            with patch.object(jikan_service.client, "get_anime_by_id", return_value=details_response):
                with patch.object(jikan_service.client, "get_anime_statistics", return_value=stats_response):
                    # Perform workflow: search -> get details -> get stats
                    search_result = await jikan_service.search_anime(query="test")
                    details_result = await jikan_service.get_anime_details(anime_id=1)
                    stats_result = await jikan_service.get_anime_statistics(anime_id=1)

                    assert search_result == search_response
                    assert details_result == details_response
                    assert stats_result == stats_response