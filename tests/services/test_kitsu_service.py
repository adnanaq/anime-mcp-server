"""Tests for Kitsu service integration."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.services.external.kitsu_service import KitsuService


class TestKitsuService:
    """Test cases for Kitsu service."""

    @pytest.fixture
    def kitsu_service(self):
        """Create Kitsu service for testing."""
        with (
            patch(
                "src.services.external.kitsu_service.KitsuClient"
            ) as mock_client_class,
            patch("src.services.external.base_service.CircuitBreaker") as mock_cb_class,
        ):

            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_circuit_breaker = Mock()
            mock_circuit_breaker.is_open = Mock(return_value=False)
            mock_cb_class.return_value = mock_circuit_breaker

            service = KitsuService()
            service.client = mock_client
            return service

    @pytest.fixture
    def sample_anime_data(self):
        """Sample anime data for testing."""
        return {
            "id": "1",
            "type": "anime",
            "attributes": {
                "slug": "cowboy-bebop",
                "synopsis": "In the year 2071, humanity has colonized several of the planets...",
                "coverImageTopOffset": 400,
                "titles": {
                    "en": "Cowboy Bebop",
                    "en_jp": "Cowboy Bebop",
                    "ja_jp": "カウボーイビバップ",
                },
                "canonicalTitle": "Cowboy Bebop",
                "abbreviatedTitles": ["COWBOY BEBOP"],
                "averageRating": "82.95",
                "ratingFrequencies": {},
                "userCount": 89569,
                "favoritesCount": 4906,
                "startDate": "1998-04-03",
                "endDate": "1999-04-24",
                "nextRelease": None,
                "popularityRank": 138,
                "ratingRank": 28,
                "ageRating": "R",
                "subtype": "TV",
                "status": "finished",
                "tba": None,
                "posterImage": {
                    "tiny": "https://media.kitsu.io/anime/poster_images/1/tiny.jpg",
                    "small": "https://media.kitsu.io/anime/poster_images/1/small.jpg",
                    "medium": "https://media.kitsu.io/anime/poster_images/1/medium.jpg",
                    "large": "https://media.kitsu.io/anime/poster_images/1/large.jpg",
                    "original": "https://media.kitsu.io/anime/poster_images/1/original.jpg",
                },
                "coverImage": {
                    "tiny": "https://media.kitsu.io/anime/cover_images/1/tiny.jpg",
                    "small": "https://media.kitsu.io/anime/cover_images/1/small.jpg",
                    "large": "https://media.kitsu.io/anime/cover_images/1/large.jpg",
                    "original": "https://media.kitsu.io/anime/cover_images/1/original.jpg",
                },
                "episodeCount": 26,
                "episodeLength": 24,
                "totalLength": 624,
                "youtubeVideoId": "QCaEJZqLeTU",
            },
        }

    @pytest.mark.asyncio
    async def test_search_anime_success(self, kitsu_service, sample_anime_data):
        """Test successful anime search."""
        # Setup
        query = "cowboy bebop"
        limit = 10
        expected_results = [sample_anime_data]

        kitsu_service.client.search_anime.return_value = expected_results

        # Execute
        results = await kitsu_service.search_anime(query, limit)

        # Verify
        assert results == expected_results
        kitsu_service.client.search_anime.assert_called_once_with(
            query=query, limit=limit
        )

    @pytest.mark.asyncio
    async def test_search_anime_failure(self, kitsu_service):
        """Test anime search failure handling."""
        # Setup
        kitsu_service.client.search_anime.side_effect = Exception("API Error")

        # Execute & Verify
        with pytest.raises(Exception, match="API Error"):
            await kitsu_service.search_anime("test", 10)

    @pytest.mark.asyncio
    async def test_get_anime_details_success(self, kitsu_service, sample_anime_data):
        """Test successful anime details retrieval."""
        # Setup
        anime_id = 1
        kitsu_service.client.get_anime_by_id.return_value = sample_anime_data

        # Execute
        result = await kitsu_service.get_anime_details(anime_id)

        # Verify
        assert result == sample_anime_data
        kitsu_service.client.get_anime_by_id.assert_called_once_with(anime_id)

    @pytest.mark.asyncio
    async def test_get_anime_details_not_found(self, kitsu_service):
        """Test anime details when not found."""
        # Setup
        kitsu_service.client.get_anime_by_id.return_value = None

        # Execute
        result = await kitsu_service.get_anime_details(999)

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_get_trending_anime_success(self, kitsu_service, sample_anime_data):
        """Test successful trending anime retrieval."""
        # Setup
        limit = 20
        expected_results = [sample_anime_data]

        kitsu_service.client.get_trending_anime.return_value = expected_results

        # Execute
        results = await kitsu_service.get_trending_anime(limit)

        # Verify
        assert results == expected_results
        kitsu_service.client.get_trending_anime.assert_called_once_with(limit=limit)

    @pytest.mark.asyncio
    async def test_get_anime_episodes_success(self, kitsu_service):
        """Test successful anime episodes retrieval."""
        # Setup
        anime_id = 1
        episodes_data = [
            {
                "id": "1",
                "type": "episodes",
                "attributes": {
                    "titles": {"en": "Asteroid Blues"},
                    "canonicalTitle": "Asteroid Blues",
                    "seasonNumber": 1,
                    "number": 1,
                    "synopsis": "Spike and Jet head to Tijuana...",
                    "airdate": "1998-04-03T00:00:00.000Z",
                    "length": 24,
                },
            }
        ]

        kitsu_service.client.get_anime_episodes.return_value = episodes_data

        # Execute
        result = await kitsu_service.get_anime_episodes(anime_id)

        # Verify
        assert result == episodes_data
        kitsu_service.client.get_anime_episodes.assert_called_once_with(anime_id)

    @pytest.mark.asyncio
    async def test_get_streaming_links_success(self, kitsu_service):
        """Test successful streaming links retrieval."""
        # Setup
        anime_id = 1
        streaming_data = [
            {
                "id": "1",
                "type": "streaming-links",
                "attributes": {
                    "url": "https://www.funimation.com/shows/cowboy-bebop/",
                    "subs": ["en"],
                    "dubs": ["en"],
                },
                "relationships": {
                    "streamer": {"data": {"id": "1", "type": "streamers"}}
                },
            }
        ]

        kitsu_service.client.get_streaming_links.return_value = streaming_data

        # Execute
        result = await kitsu_service.get_streaming_links(anime_id)

        # Verify
        assert result == streaming_data
        kitsu_service.client.get_streaming_links.assert_called_once_with(anime_id)

    @pytest.mark.asyncio
    async def test_get_anime_characters_success(self, kitsu_service):
        """Test successful anime characters retrieval."""
        # Setup
        anime_id = 1
        characters_data = [
            {
                "id": "1",
                "type": "anime-characters",
                "attributes": {"role": "main"},
                "relationships": {
                    "character": {"data": {"id": "1", "type": "characters"}}
                },
            }
        ]

        kitsu_service.client.get_anime_characters.return_value = characters_data

        # Execute
        result = await kitsu_service.get_anime_characters(anime_id)

        # Verify
        assert result == characters_data
        kitsu_service.client.get_anime_characters.assert_called_once_with(anime_id)

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, kitsu_service):
        """Test health check when service is healthy."""
        # Setup
        kitsu_service.client.get_trending_anime.return_value = [{"test": "data"}]
        kitsu_service.circuit_breaker.is_open.return_value = False

        # Execute
        result = await kitsu_service.health_check()

        # Verify
        assert result["service"] == "kitsu"
        assert result["status"] == "healthy"
        assert result["circuit_breaker_open"] == False

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, kitsu_service):
        """Test health check when service is unhealthy."""
        # Setup
        kitsu_service.client.get_trending_anime.side_effect = Exception("Service down")
        kitsu_service.circuit_breaker.is_open.return_value = True

        # Execute
        result = await kitsu_service.health_check()

        # Verify
        assert result["service"] == "kitsu"
        assert result["status"] == "unhealthy"
        assert "error" in result
        assert result["circuit_breaker_open"] == True

    def test_service_initialization(self, kitsu_service):
        """Test service initialization."""
        assert kitsu_service.service_name == "kitsu"
        assert kitsu_service.cache_manager is not None
        assert kitsu_service.circuit_breaker is not None
        assert kitsu_service.client is not None

    def test_is_healthy(self, kitsu_service):
        """Test is_healthy method."""
        # Setup - healthy
        kitsu_service.circuit_breaker.is_open.return_value = False

        # Execute & Verify
        assert kitsu_service.is_healthy() == True

        # Setup - unhealthy
        kitsu_service.circuit_breaker.is_open.return_value = True

        # Execute & Verify
        assert kitsu_service.is_healthy() == False

    def test_get_service_info(self, kitsu_service):
        """Test get_service_info method."""
        # Setup
        kitsu_service.circuit_breaker.is_open.return_value = False

        # Execute
        info = kitsu_service.get_service_info()

        # Verify
        assert info["name"] == "kitsu"
        assert info["healthy"] == True
        assert info["circuit_breaker_open"] == False
