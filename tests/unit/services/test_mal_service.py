"""Tests for MAL service integration."""

from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

import pytest

from src.services.external.mal_service import MALService


class TestMALService:
    """Test cases for MAL service."""

    @pytest.fixture
    def mal_service(self):
        """Create MAL service for testing."""
        with (
            patch("src.services.external.mal_service.MALClient") as mock_client_class,
            patch("src.services.external.base_service.CircuitBreaker") as mock_cb_class,
        ):

            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_circuit_breaker = Mock()
            mock_circuit_breaker.is_open = Mock(return_value=False)
            mock_cb_class.return_value = mock_circuit_breaker

            service = MALService()
            service.client = mock_client
            return service

    @pytest.fixture
    def sample_anime_data(self):
        """Sample anime data for testing."""
        return {
            "mal_id": 21,
            "title": "One Piece",
            "title_english": "One Piece",
            "title_japanese": "ワンピース",
            "synopsis": "Gol D. Roger was known as the Pirate King...",
            "episodes": 1000,
            "status": "Currently Airing",
            "genres": [{"name": "Action"}, {"name": "Adventure"}],
            "score": 8.7,
            "scored_by": 500000,
            "images": {"jpg": {"image_url": "https://example.com/image.jpg"}},
        }

    @pytest.mark.asyncio
    async def test_search_anime_success(self, mal_service, sample_anime_data):
        """Test successful anime search."""
        # Setup
        query = "one piece"
        limit = 10
        expected_results = [sample_anime_data]

        mal_service.client.search_anime.return_value = expected_results

        # Execute
        results = await mal_service.search_anime(query, limit)

        # Verify
        assert results == expected_results
        # Verify the client was called (parameters will include all with defaults)
        mal_service.client.search_anime.assert_called_once()
        call_args = mal_service.client.search_anime.call_args
        assert call_args.kwargs["query"] == query
        assert call_args.kwargs["limit"] == limit

    @pytest.mark.asyncio
    async def test_search_anime_with_filters(self, mal_service, sample_anime_data):
        """Test anime search with additional filters."""
        # Setup
        query = "action"
        limit = 20
        status = "airing"
        genres = [1, 2]  # Action, Adventure
        expected_results = [sample_anime_data]

        mal_service.client.search_anime.return_value = expected_results

        # Execute
        results = await mal_service.search_anime(query, limit, status, genres)

        # Verify
        assert results == expected_results
        # Verify the client was called with our specific parameters
        mal_service.client.search_anime.assert_called_once()
        call_args = mal_service.client.search_anime.call_args
        assert call_args.kwargs["query"] == query
        assert call_args.kwargs["limit"] == limit
        assert call_args.kwargs["status"] == status
        assert call_args.kwargs["genres"] == genres

    @pytest.mark.asyncio
    async def test_search_anime_failure(self, mal_service):
        """Test anime search failure handling."""
        # Setup
        mal_service.client.search_anime.side_effect = Exception("API Error")

        # Execute & Verify
        with pytest.raises(Exception, match="API Error"):
            await mal_service.search_anime("test", 10)

    @pytest.mark.asyncio
    async def test_get_anime_details_success(self, mal_service, sample_anime_data):
        """Test successful anime details retrieval."""
        # Setup
        anime_id = 21
        mal_service.client.get_anime_by_id.return_value = sample_anime_data

        # Execute
        result = await mal_service.get_anime_details(anime_id)

        # Verify
        assert result == sample_anime_data
        mal_service.client.get_anime_by_id.assert_called_once_with(anime_id)

    @pytest.mark.asyncio
    async def test_get_anime_details_not_found(self, mal_service):
        """Test anime details when not found."""
        # Setup
        mal_service.client.get_anime_by_id.return_value = None

        # Execute
        result = await mal_service.get_anime_details(999)

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_get_seasonal_anime_success(self, mal_service, sample_anime_data):
        """Test successful seasonal anime retrieval."""
        # Setup
        year = 2024
        season = "winter"
        expected_results = [sample_anime_data]

        mal_service.client.get_seasonal_anime.return_value = expected_results

        # Execute
        results = await mal_service.get_seasonal_anime(year, season)

        # Verify
        assert results == expected_results
        mal_service.client.get_seasonal_anime.assert_called_once_with(year, season)

    @pytest.mark.asyncio
    async def test_get_seasonal_anime_invalid_season(self, mal_service):
        """Test seasonal anime with invalid season."""
        # Setup
        year = 2024
        season = "invalid"

        # Execute & Verify
        with pytest.raises(ValueError, match="Invalid season"):
            await mal_service.get_seasonal_anime(year, season)

    @pytest.mark.asyncio
    async def test_get_anime_statistics_success(self, mal_service):
        """Test successful anime statistics retrieval."""
        # Setup
        anime_id = 21
        stats_data = {
            "watching": 100000,
            "completed": 200000,
            "on_hold": 5000,
            "dropped": 3000,
            "plan_to_watch": 50000,
        }

        mal_service.client.get_anime_statistics.return_value = stats_data

        # Execute
        result = await mal_service.get_anime_statistics(anime_id)

        # Verify
        assert result == stats_data
        mal_service.client.get_anime_statistics.assert_called_once_with(anime_id)

    @pytest.mark.asyncio
    async def test_get_current_season_success(self, mal_service, sample_anime_data):
        """Test successful current season anime retrieval."""
        # Setup
        expected_results = [sample_anime_data]

        mal_service.client.get_seasonal_anime.return_value = expected_results

        # Execute
        results = await mal_service.get_current_season()

        # Verify
        assert results == expected_results
        # Should call with current year and season
        mal_service.client.get_seasonal_anime.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mal_service):
        """Test health check when service is healthy."""
        # Setup
        mal_service.client.search_anime.return_value = [{"test": "data"}]
        mal_service.circuit_breaker.is_open.return_value = False

        # Execute
        result = await mal_service.health_check()

        # Verify
        assert result["service"] == "mal"
        assert result["status"] == "healthy"
        assert result["circuit_breaker_open"] == False

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, mal_service):
        """Test health check when service is unhealthy."""
        # Setup
        mal_service.client.search_anime.side_effect = Exception("Service down")
        mal_service.circuit_breaker.is_open.return_value = True

        # Execute
        result = await mal_service.health_check()

        # Verify
        assert result["service"] == "mal"
        assert result["status"] == "unhealthy"
        assert "error" in result
        assert result["circuit_breaker_open"] == True

    def test_service_initialization(self, mal_service):
        """Test service initialization."""
        assert mal_service.service_name == "mal"
        assert mal_service.cache_manager is not None
        assert mal_service.circuit_breaker is not None
        assert mal_service.client is not None

    def test_is_healthy(self, mal_service):
        """Test is_healthy method."""
        # Setup - healthy
        mal_service.circuit_breaker.is_open.return_value = False

        # Execute & Verify
        assert mal_service.is_healthy() == True

        # Setup - unhealthy
        mal_service.circuit_breaker.is_open.return_value = True

        # Execute & Verify
        assert mal_service.is_healthy() == False

    def test_get_service_info(self, mal_service):
        """Test get_service_info method."""
        # Setup
        mal_service.circuit_breaker.is_open.return_value = False

        # Execute
        info = mal_service.get_service_info()

        # Verify
        assert info["name"] == "mal"
        assert info["healthy"] == True
        assert info["circuit_breaker_open"] == False


class TestMALServiceFullParameters:
    """Test cases for MAL service with full Jikan parameter support."""

    @pytest.fixture
    def mal_service(self):
        """Create MAL service for testing."""
        with (
            patch("src.services.external.mal_service.MALClient") as mock_client_class,
            patch("src.services.external.base_service.CircuitBreaker") as mock_cb_class,
        ):

            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_circuit_breaker = Mock()
            mock_circuit_breaker.is_open = Mock(return_value=False)
            mock_cb_class.return_value = mock_circuit_breaker

            service = MALService()
            service.client = mock_client
            return service

    @pytest.fixture
    def full_sample_data(self):
        """Full sample anime data with complete Jikan response structure."""
        return {
            "mal_id": 1535,
            "title": "Death Note",
            "title_english": "Death Note",
            "title_japanese": "デスノート",
            "synopsis": "Brutal murders, petty thefts, and senseless violence...",
            "type": "TV",
            "episodes": 37,
            "status": "Finished Airing",
            "score": 8.62,
            "scored_by": 2889017,
            "rank": 91,
            "popularity": 2,
            "rating": "R - 17+ (violence & profanity)",
            "year": 2006,
            "season": "fall",
            "aired": {
                "from": "2006-10-04T00:00:00+00:00",
                "to": "2007-06-27T00:00:00+00:00"
            },
            "genres": [
                {"mal_id": 37, "name": "Supernatural"},
                {"mal_id": 41, "name": "Suspense"}
            ],
            "themes": [
                {"mal_id": 40, "name": "Psychological"}
            ],
            "studios": [
                {"mal_id": 11, "name": "Madhouse"}
            ],
            "producers": [
                {"mal_id": 29, "name": "VAP"}
            ]
        }

    @pytest.mark.asyncio
    async def test_search_anime_with_current_parameters(self, mal_service, full_sample_data):
        """Test search_anime with current basic parameters."""
        # Setup
        query = "death note"
        limit = 5
        status = "complete"
        genres = [40, 41]  # Psychological, Suspense
        expected_results = [full_sample_data]

        mal_service.client.search_anime.return_value = expected_results

        # Execute
        results = await mal_service.search_anime(
            query=query, limit=limit, status=status, genres=genres
        )

        # Verify results
        assert results == expected_results
        
        # Verify the client was called with our parameters (others will be None)
        mal_service.client.search_anime.assert_called_once()
        call_args = mal_service.client.search_anime.call_args
        assert call_args.kwargs["query"] == query
        assert call_args.kwargs["limit"] == limit
        assert call_args.kwargs["status"] == status
        assert call_args.kwargs["genres"] == genres

    @pytest.mark.asyncio
    async def test_search_anime_with_type_filter(self, mal_service, full_sample_data):
        """Test search_anime with anime type filter."""
        # Setup
        query = "psychological"
        limit = 10
        anime_type = "TV"
        expected_results = [full_sample_data]

        mal_service.client.search_anime.return_value = expected_results

        # Execute - now this should work
        results = await mal_service.search_anime(
            query=query, limit=limit, anime_type=anime_type
        )
        
        # Verify results
        assert results == expected_results
        
        # Verify the client was called with our anime_type parameter
        mal_service.client.search_anime.assert_called_once()
        call_args = mal_service.client.search_anime.call_args
        assert call_args.kwargs["query"] == query
        assert call_args.kwargs["anime_type"] == anime_type

    @pytest.mark.asyncio
    async def test_search_anime_with_score_filters(self, mal_service, full_sample_data):
        """Test search_anime with score range filters."""
        # Setup
        query = "thriller"
        min_score = 8.0
        max_score = 9.5
        expected_results = [full_sample_data]

        mal_service.client.search_anime.return_value = expected_results

        # Execute - now this should work
        results = await mal_service.search_anime(
            query=query, min_score=min_score, max_score=max_score
        )
        
        # Verify results
        assert results == expected_results
        
        # Verify the client was called with our score parameters
        mal_service.client.search_anime.assert_called_once()
        call_args = mal_service.client.search_anime.call_args
        assert call_args.kwargs["min_score"] == min_score
        assert call_args.kwargs["max_score"] == max_score

    @pytest.mark.asyncio
    async def test_search_anime_with_date_range(self, mal_service, full_sample_data):
        """Test search_anime with start/end date filters."""
        # Setup
        query = "mecha"
        start_date = "2010"
        end_date = "2020"
        expected_results = [full_sample_data]

        mal_service.client.search_anime.return_value = expected_results

        # Execute - now this should work
        results = await mal_service.search_anime(
            query=query, start_date=start_date, end_date=end_date
        )
        
        # Verify results
        assert results == expected_results
        
        # Verify the client was called with our date parameters
        mal_service.client.search_anime.assert_called_once()
        call_args = mal_service.client.search_anime.call_args
        assert call_args.kwargs["start_date"] == start_date
        assert call_args.kwargs["end_date"] == end_date

    @pytest.mark.asyncio
    async def test_search_anime_with_genre_exclusions(self, mal_service, full_sample_data):
        """Test search_anime with genre exclusion filters."""
        # Setup
        query = "action"
        genres = [1, 2]  # Include Action, Adventure
        genres_exclude = [14, 26]  # Exclude Horror, Ecchi
        expected_results = [full_sample_data]

        mal_service.client.search_anime.return_value = expected_results

        # Execute - now this should work
        results = await mal_service.search_anime(
            query=query, genres=genres, genres_exclude=genres_exclude
        )
        
        # Verify results
        assert results == expected_results
        
        # Verify the client was called with our genre parameters
        mal_service.client.search_anime.assert_called_once()
        call_args = mal_service.client.search_anime.call_args
        assert call_args.kwargs["genres"] == genres
        assert call_args.kwargs["genres_exclude"] == genres_exclude

    @pytest.mark.asyncio
    async def test_search_anime_with_producer_filter(self, mal_service, full_sample_data):
        """Test search_anime with producer/studio filters."""
        # Setup
        query = "psychological"
        producers = [11]  # Madhouse studio
        expected_results = [full_sample_data]

        mal_service.client.search_anime.return_value = expected_results

        # Execute - now this should work
        results = await mal_service.search_anime(
            query=query, producers=producers
        )
        
        # Verify results
        assert results == expected_results
        mal_service.client.search_anime.assert_called_once()
        call_args = mal_service.client.search_anime.call_args
        assert call_args.kwargs["producers"] == producers

    @pytest.mark.asyncio
    async def test_search_anime_with_rating_filter(self, mal_service, full_sample_data):
        """Test search_anime with content rating filter."""
        # Setup
        query = "family friendly"
        rating = "PG-13"
        sfw = True
        expected_results = [full_sample_data]

        mal_service.client.search_anime.return_value = expected_results

        # Execute - now this should work
        results = await mal_service.search_anime(
            query=query, rating=rating, sfw=sfw
        )
        
        # Verify results
        assert results == expected_results
        mal_service.client.search_anime.assert_called_once()
        call_args = mal_service.client.search_anime.call_args
        assert call_args.kwargs["rating"] == rating
        assert call_args.kwargs["sfw"] == sfw

    @pytest.mark.asyncio
    async def test_search_anime_with_sorting(self, mal_service, full_sample_data):
        """Test search_anime with ordering and sorting."""
        # Setup
        query = "action"
        order_by = "score"
        sort = "desc"
        expected_results = [full_sample_data]

        mal_service.client.search_anime.return_value = expected_results

        # Execute - now this should work
        results = await mal_service.search_anime(
            query=query, order_by=order_by, sort=sort
        )
        
        # Verify results
        assert results == expected_results
        mal_service.client.search_anime.assert_called_once()
        call_args = mal_service.client.search_anime.call_args
        assert call_args.kwargs["order_by"] == order_by
        assert call_args.kwargs["sort"] == sort

    @pytest.mark.asyncio
    async def test_search_anime_with_pagination(self, mal_service, full_sample_data):
        """Test search_anime with pagination support."""
        # Setup
        query = "romance"
        page = 2
        limit = 25
        expected_results = [full_sample_data]

        mal_service.client.search_anime.return_value = expected_results

        # Execute - now this should work
        results = await mal_service.search_anime(
            query=query, page=page, limit=limit
        )
        
        # Verify results
        assert results == expected_results
        mal_service.client.search_anime.assert_called_once()
        call_args = mal_service.client.search_anime.call_args
        assert call_args.kwargs["page"] == page
        assert call_args.kwargs["limit"] == limit

    @pytest.mark.asyncio
    async def test_search_anime_with_letter_filter(self, mal_service, full_sample_data):
        """Test search_anime with starting letter filter."""
        # Setup
        letter = "D"
        expected_results = [full_sample_data]

        mal_service.client.search_anime.return_value = expected_results

        # Execute - now this should work
        results = await mal_service.search_anime(letter=letter)
        
        # Verify results
        assert results == expected_results
        mal_service.client.search_anime.assert_called_once()
        call_args = mal_service.client.search_anime.call_args
        assert call_args.kwargs["letter"] == letter

    @pytest.mark.asyncio
    async def test_search_anime_with_complex_combination(self, mal_service, full_sample_data):
        """Test search_anime with complex parameter combination."""
        # Setup - represents real-world complex query
        query = "psychological thriller"
        anime_type = "TV"
        min_score = 8.0
        genres = [40, 41]  # Psychological, Suspense
        genres_exclude = [1, 14]  # Exclude Action, Horror
        producers = [11]  # Madhouse
        start_date = "2005"
        end_date = "2015"
        rating = "R"
        order_by = "score"
        sort = "desc"
        limit = 10
        expected_results = [full_sample_data]

        mal_service.client.search_anime.return_value = expected_results

        # Execute - now this should work with all parameters
        results = await mal_service.search_anime(
            query=query,
            anime_type=anime_type,
            min_score=min_score,
            genres=genres,
            genres_exclude=genres_exclude,
            producers=producers,
            start_date=start_date,
            end_date=end_date,
            rating=rating,
            order_by=order_by,
            sort=sort,
            limit=limit
        )
        
        # Verify results
        assert results == expected_results
        
        # Verify key parameters were passed to the client
        mal_service.client.search_anime.assert_called_once()
        call_args = mal_service.client.search_anime.call_args
        assert call_args.kwargs["query"] == query
        assert call_args.kwargs["anime_type"] == anime_type
        assert call_args.kwargs["min_score"] == min_score
        assert call_args.kwargs["genres"] == genres
        assert call_args.kwargs["genres_exclude"] == genres_exclude
        assert call_args.kwargs["producers"] == producers
        assert call_args.kwargs["rating"] == rating
        assert call_args.kwargs["order_by"] == order_by
        assert call_args.kwargs["sort"] == sort

    @pytest.mark.asyncio
    async def test_search_anime_parameter_validation(self, mal_service):
        """Test search_anime parameter validation."""
        # This test will FAIL initially (TDD)
        
        # Test invalid anime_type
        with pytest.raises(ValueError, match="Invalid anime type"):
            await mal_service.search_anime(anime_type="INVALID")
        
        # Test invalid score range
        with pytest.raises(ValueError, match="min_score must be less than or equal to max_score"):
            await mal_service.search_anime(min_score=9.0, max_score=8.0)
        
        # Test invalid rating
        with pytest.raises(ValueError, match="Invalid rating"):
            await mal_service.search_anime(rating="INVALID")
        
        # Test invalid order_by
        with pytest.raises(ValueError, match="Invalid order_by"):
            await mal_service.search_anime(order_by="INVALID")
        
        # Test invalid sort
        with pytest.raises(ValueError, match="Invalid sort"):
            await mal_service.search_anime(sort="INVALID")

    @pytest.mark.asyncio
    async def test_search_anime_with_correlation_id(self, mal_service, full_sample_data):
        """Test search_anime with correlation ID support."""
        # Setup
        query = "test"
        correlation_id = "test-correlation-123"
        expected_results = [full_sample_data]

        mal_service.client.search_anime.return_value = expected_results

        # Execute - now this should work
        results = await mal_service.search_anime(
            query=query, correlation_id=correlation_id
        )
        
        # Verify results
        assert results == expected_results
        
        # Verify the client was called with our correlation_id
        mal_service.client.search_anime.assert_called_once()
        call_args = mal_service.client.search_anime.call_args
        assert call_args.kwargs["correlation_id"] == correlation_id
