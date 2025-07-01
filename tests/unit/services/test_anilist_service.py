"""Tests for AniList service integration."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.services.external.anilist_service import AniListService


class TestAniListService:
    """Test cases for AniList service."""

    @pytest.fixture
    def anilist_service(self):
        """Create AniList service for testing."""
        with (
            patch(
                "src.services.external.anilist_service.AniListClient"
            ) as mock_client_class,
            patch("src.services.external.base_service.CircuitBreaker") as mock_cb_class,
        ):

            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_circuit_breaker = Mock()
            mock_circuit_breaker.is_open = Mock(return_value=False)
            mock_cb_class.return_value = mock_circuit_breaker

            service = AniListService()
            service.client = mock_client
            return service

    @pytest.fixture
    def sample_anime_data(self):
        """Sample anime data for testing."""
        return {
            "id": 21,
            "title": {
                "romaji": "One Piece",
                "english": "One Piece",
                "native": "ワンピース",
            },
            "description": "Gol D. Roger was known as the Pirate King...",
            "episodes": 1000,
            "status": "RELEASING",
            "genres": ["Action", "Adventure", "Comedy"],
            "averageScore": 87,
        }

    @pytest.mark.asyncio
    async def test_search_anime_success(self, anilist_service, sample_anime_data):
        """Test successful anime search."""
        # Setup
        query = "one piece"
        limit = 10
        page = 1
        expected_results = [sample_anime_data]

        anilist_service.client.search_anime.return_value = expected_results

        # Execute
        results = await anilist_service.search_anime(query, limit, page)

        # Verify
        assert results == expected_results
        anilist_service.client.search_anime.assert_called_once_with(
            query=query, limit=limit, page=page
        )

    @pytest.mark.asyncio
    async def test_search_anime_failure(self, anilist_service):
        """Test anime search failure handling."""
        # Setup
        anilist_service.client.search_anime.side_effect = Exception("API Error")

        # Execute & Verify
        with pytest.raises(Exception, match="API Error"):
            await anilist_service.search_anime("test", 10, 1)

    @pytest.mark.asyncio
    async def test_get_anime_details_success(self, anilist_service, sample_anime_data):
        """Test successful anime details retrieval."""
        # Setup
        anime_id = 21
        anilist_service.client.get_anime_by_id.return_value = sample_anime_data

        # Execute
        result = await anilist_service.get_anime_details(anime_id)

        # Verify
        assert result == sample_anime_data
        anilist_service.client.get_anime_by_id.assert_called_once_with(anime_id)

    @pytest.mark.asyncio
    async def test_get_anime_details_not_found(self, anilist_service):
        """Test anime details when not found."""
        # Setup
        anilist_service.client.get_anime_by_id.return_value = None

        # Execute
        result = await anilist_service.get_anime_details(999)

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_get_trending_anime_success(self, anilist_service, sample_anime_data):
        """Test successful trending anime retrieval."""
        # Setup
        limit = 20
        page = 1
        expected_results = [sample_anime_data]

        anilist_service.client.get_trending_anime.return_value = expected_results

        # Execute
        results = await anilist_service.get_trending_anime(limit, page)

        # Verify
        assert results == expected_results
        anilist_service.client.get_trending_anime.assert_called_once_with(
            limit=limit, page=page
        )

    @pytest.mark.asyncio
    async def test_get_upcoming_anime_success(self, anilist_service, sample_anime_data):
        """Test successful upcoming anime retrieval."""
        # Setup
        limit = 20
        page = 1
        expected_results = [sample_anime_data]

        anilist_service.client.get_upcoming_anime.return_value = expected_results

        # Execute
        results = await anilist_service.get_upcoming_anime(limit, page)

        # Verify
        assert results == expected_results
        anilist_service.client.get_upcoming_anime.assert_called_once_with(
            limit=limit, page=page
        )

    @pytest.mark.asyncio
    async def test_get_popular_anime_success(self, anilist_service, sample_anime_data):
        """Test successful popular anime retrieval."""
        # Setup
        limit = 20
        page = 1
        expected_results = [sample_anime_data]

        anilist_service.client.get_popular_anime.return_value = expected_results

        # Execute
        results = await anilist_service.get_popular_anime(limit, page)

        # Verify
        assert results == expected_results
        anilist_service.client.get_popular_anime.assert_called_once_with(
            limit=limit, page=page
        )

    @pytest.mark.asyncio
    async def test_get_staff_details_success(self, anilist_service):
        """Test successful staff details retrieval."""
        # Setup
        staff_id = 123
        staff_data = {
            "id": 123,
            "name": {"first": "Eiichiro", "last": "Oda"},
            "description": "Manga artist and creator of One Piece",
        }

        anilist_service.client.get_staff_by_id.return_value = staff_data

        # Execute
        result = await anilist_service.get_staff_details(staff_id)

        # Verify
        assert result == staff_data
        anilist_service.client.get_staff_by_id.assert_called_once_with(staff_id)

    @pytest.mark.asyncio
    async def test_get_studio_details_success(self, anilist_service):
        """Test successful studio details retrieval."""
        # Setup
        studio_id = 456
        studio_data = {"id": 456, "name": "Studio Pierrot", "isAnimationStudio": True}

        anilist_service.client.get_studio_by_id.return_value = studio_data

        # Execute
        result = await anilist_service.get_studio_details(studio_id)

        # Verify
        assert result == studio_data
        anilist_service.client.get_studio_by_id.assert_called_once_with(studio_id)

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, anilist_service):
        """Test health check when service is healthy."""
        # Setup
        anilist_service.client.get_trending_anime.return_value = [{"test": "data"}]
        anilist_service.circuit_breaker.is_open.return_value = False

        # Execute
        result = await anilist_service.health_check()

        # Verify
        assert result["service"] == "anilist"
        assert result["status"] == "healthy"
        assert result["circuit_breaker_open"] == False

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, anilist_service):
        """Test health check when service is unhealthy."""
        # Setup
        anilist_service.client.get_trending_anime.side_effect = Exception(
            "Service down"
        )
        anilist_service.circuit_breaker.is_open.return_value = True

        # Execute
        result = await anilist_service.health_check()

        # Verify
        assert result["service"] == "anilist"
        assert result["status"] == "unhealthy"
        assert "error" in result
        assert result["circuit_breaker_open"] == True

    def test_service_initialization(self, anilist_service):
        """Test service initialization."""
        assert anilist_service.service_name == "anilist"
        assert anilist_service.cache_manager is not None
        assert anilist_service.circuit_breaker is not None
        assert anilist_service.client is not None

    def test_is_healthy(self, anilist_service):
        """Test is_healthy method."""
        # Setup - healthy
        anilist_service.circuit_breaker.is_open.return_value = False

        # Execute & Verify
        assert anilist_service.is_healthy() == True

        # Setup - unhealthy
        anilist_service.circuit_breaker.is_open.return_value = True

        # Execute & Verify
        assert anilist_service.is_healthy() == False

    def test_get_service_info(self, anilist_service):
        """Test get_service_info method."""
        # Setup
        anilist_service.circuit_breaker.is_open.return_value = False

        # Execute
        info = anilist_service.get_service_info()

        # Verify
        assert info["name"] == "anilist"
        assert info["healthy"] == True
        assert info["circuit_breaker_open"] == False
