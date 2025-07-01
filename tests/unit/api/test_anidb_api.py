"""Tests for AniDB API endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.external.anidb import router


class TestAniDBAP:
    """Test cases for AniDB API endpoints."""

    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_anime_data(self):
        """Sample anime data for testing."""
        return {
            "id": 30,
            "title": "Neon Genesis Evangelion",
            "type": "TV Series",
            "episodes": 26,
            "start_date": "1995-10-04",
            "end_date": "1996-03-27",
            "rating": "9.01",
            "votes": 15234,
            "synopsis": "At the age of 14 Shinji Ikari is summoned by his father...",
            "genres": ["Mecha", "Psychological", "Drama"],
            "studios": ["Gainax", "Tatsunoko"],
            "year": 1995,
        }

    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing."""
        return [
            {
                "id": 30,
                "title": "Neon Genesis Evangelion",
                "type": "TV Series",
                "episodes": 26,
                "year": 1995,
                "rating": "9.01",
            },
            {
                "id": 32,
                "title": "Neon Genesis Evangelion: Death & Rebirth",
                "type": "Movie",
                "episodes": 1,
                "year": 1997,
                "rating": "7.25",
            },
        ]

    @pytest.fixture
    def sample_character_data(self):
        """Sample character data for testing."""
        return [
            {
                "id": 23,
                "name": "Shinji Ikari",
                "gender": "male",
                "description": "Third Child and pilot of Evangelion Unit-01",
            },
            {
                "id": 24,
                "name": "Rei Ayanami",
                "gender": "female",
                "description": "First Child and pilot of Evangelion Unit-00",
            },
        ]

    @pytest.fixture
    def sample_episode_data(self):
        """Sample episode data for testing."""
        return [
            {
                "episode_number": 1,
                "title": "Angel Attack",
                "air_date": "1995-10-04",
                "rating": "8.5",
            },
            {
                "episode_number": 2,
                "title": "The Beast",
                "air_date": "1995-10-11",
                "rating": "8.2",
            },
        ]

    @patch("src.api.external.anidb._anidb_service")
    def test_search_anime_success(self, mock_service, client, sample_search_results):
        """Test successful anime search endpoint."""
        # Setup
        mock_service.search_anime = AsyncMock(return_value=sample_search_results)

        # Execute
        response = client.get("/external/anidb/search?q=evangelion&limit=10")

        # Verify
        assert response.status_code == 200
        data = response.json()

        assert data["source"] == "anidb"
        assert data["query"] == "evangelion"
        assert data["limit"] == 10
        assert len(data["results"]) == 2
        assert data["results"] == sample_search_results
        assert data["total_results"] == 2

        mock_service.search_anime.assert_called_once_with("evangelion")

    @patch("src.api.external.anidb._anidb_service")
    def test_search_anime_default_params(self, mock_service, client):
        """Test search endpoint with default parameters."""
        # Setup
        mock_service.search_anime = AsyncMock(return_value=[])

        # Execute
        response = client.get("/external/anidb/search?q=test")

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 20  # Default limit
        mock_service.search_anime.assert_called_once_with("test")

    @patch("src.api.external.anidb._anidb_service")
    def test_search_anime_service_error(self, mock_service, client):
        """Test search endpoint when service fails."""
        # Setup
        mock_service.search_anime = AsyncMock(side_effect=Exception("Service error"))

        # Execute
        response = client.get("/external/anidb/search?q=test")

        # Verify
        assert response.status_code == 503
        assert "AniDB search service unavailable" in response.json()["detail"]

    @patch("src.api.external.anidb._anidb_service")
    def test_get_anime_details_success(self, mock_service, client, sample_anime_data):
        """Test successful anime details endpoint."""
        # Setup
        anime_id = 30
        mock_service.get_anime_details = AsyncMock(return_value=sample_anime_data)

        # Execute
        response = client.get(f"/external/anidb/anime/{anime_id}")

        # Verify
        assert response.status_code == 200
        data = response.json()

        assert data["source"] == "anidb"
        assert data["anime_id"] == anime_id
        assert data["data"] == sample_anime_data

        mock_service.get_anime_details.assert_called_once_with(anime_id)

    @patch("src.api.external.anidb._anidb_service")
    def test_get_anime_details_not_found(self, mock_service, client):
        """Test anime details endpoint when anime not found."""
        # Setup
        anime_id = 999
        mock_service.get_anime_details = AsyncMock(return_value=None)

        # Execute
        response = client.get(f"/external/anidb/anime/{anime_id}")

        # Verify
        assert response.status_code == 404
        assert f"Anime with ID {anime_id} not found" in response.json()["detail"]

    @patch("src.api.external.anidb._anidb_service")
    def test_get_anime_details_service_error(self, mock_service, client):
        """Test anime details endpoint when service fails."""
        # Setup
        mock_service.get_anime_details = AsyncMock(
            side_effect=Exception("Service error")
        )

        # Execute
        response = client.get("/external/anidb/anime/1")

        # Verify
        assert response.status_code == 503
        assert "AniDB service unavailable" in response.json()["detail"]

    @patch("src.api.external.anidb._anidb_service")
    def test_get_anime_characters_success(
        self, mock_service, client, sample_character_data
    ):
        """Test successful anime characters endpoint."""
        # Setup
        anime_id = 30
        mock_service.get_anime_characters = AsyncMock(
            return_value=sample_character_data
        )

        # Execute
        response = client.get(f"/external/anidb/anime/{anime_id}/characters")

        # Verify
        assert response.status_code == 200
        data = response.json()

        assert data["source"] == "anidb"
        assert data["anime_id"] == anime_id
        assert data["characters"] == sample_character_data
        assert data["total_characters"] == 2

        mock_service.get_anime_characters.assert_called_once_with(anime_id)

    @patch("src.api.external.anidb._anidb_service")
    def test_get_anime_episodes_success(
        self, mock_service, client, sample_episode_data
    ):
        """Test successful anime episodes endpoint."""
        # Setup
        anime_id = 30
        mock_service.get_anime_episodes = AsyncMock(return_value=sample_episode_data)

        # Execute
        response = client.get(f"/external/anidb/anime/{anime_id}/episodes")

        # Verify
        assert response.status_code == 200
        data = response.json()

        assert data["source"] == "anidb"
        assert data["anime_id"] == anime_id
        assert data["episodes"] == sample_episode_data
        assert data["total_episodes"] == 2

        mock_service.get_anime_episodes.assert_called_once_with(anime_id)

    @patch("src.api.external.anidb._anidb_service")
    def test_get_similar_anime_success(
        self, mock_service, client, sample_search_results
    ):
        """Test successful similar anime endpoint."""
        # Setup
        anime_id = 30
        limit = 5
        mock_service.get_similar_anime = AsyncMock(return_value=sample_search_results)

        # Execute
        response = client.get(f"/external/anidb/anime/{anime_id}/similar?limit={limit}")

        # Verify
        assert response.status_code == 200
        data = response.json()

        assert data["source"] == "anidb"
        assert data["anime_id"] == anime_id
        assert data["limit"] == limit
        assert data["similar_anime"] == sample_search_results
        assert data["total_results"] == 2

        mock_service.get_similar_anime.assert_called_once_with(anime_id, limit)

    @patch("src.api.external.anidb._anidb_service")
    def test_get_similar_anime_default_limit(
        self, mock_service, client, sample_search_results
    ):
        """Test similar anime endpoint with default limit."""
        # Setup
        anime_id = 30
        mock_service.get_similar_anime = AsyncMock(return_value=sample_search_results)

        # Execute
        response = client.get(f"/external/anidb/anime/{anime_id}/similar")

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10  # Default limit

        mock_service.get_similar_anime.assert_called_once_with(anime_id, 10)

    @patch("src.api.external.anidb._anidb_service")
    def test_get_random_anime_success(self, mock_service, client, sample_anime_data):
        """Test successful random anime endpoint."""
        # Setup
        mock_service.get_random_anime = AsyncMock(return_value=sample_anime_data)

        # Execute
        response = client.get("/external/anidb/random")

        # Verify
        assert response.status_code == 200
        data = response.json()

        assert data["source"] == "anidb"
        assert data["type"] == "random"
        assert data["anime"] == sample_anime_data

        mock_service.get_random_anime.assert_called_once()

    @patch("src.api.external.anidb._anidb_service")
    def test_get_random_anime_service_error(self, mock_service, client):
        """Test random anime endpoint when service fails."""
        # Setup
        mock_service.get_random_anime = AsyncMock(
            side_effect=Exception("Service error")
        )

        # Execute
        response = client.get("/external/anidb/random")

        # Verify
        assert response.status_code == 503
        assert "AniDB random anime service unavailable" in response.json()["detail"]

    @patch("src.api.external.anidb._anidb_service")
    def test_health_check_healthy(self, mock_service, client):
        """Test health check endpoint when service is healthy."""
        # Setup
        health_data = {
            "service": "anidb",
            "status": "healthy",
            "circuit_breaker_open": False,
            "response_time": "50ms",
        }
        mock_service.health_check = AsyncMock(return_value=health_data)

        # Execute
        response = client.get("/external/anidb/health")

        # Verify
        assert response.status_code == 200
        assert response.json() == health_data

        mock_service.health_check.assert_called_once()

    @patch("src.api.external.anidb._anidb_service")
    def test_health_check_unhealthy(self, mock_service, client):
        """Test health check endpoint when service is unhealthy."""
        # Setup
        health_data = {
            "service": "anidb",
            "status": "unhealthy",
            "error": "Connection timeout",
            "circuit_breaker_open": True,
        }
        mock_service.health_check = AsyncMock(return_value=health_data)

        # Execute
        response = client.get("/external/anidb/health")

        # Verify
        assert response.status_code == 200
        assert response.json() == health_data

    def test_search_anime_validation_errors(self, client):
        """Test search endpoint parameter validation."""
        # Test missing query parameter
        response = client.get("/external/anidb/search")
        assert response.status_code == 422

        # Test invalid limit (too high)
        response = client.get("/external/anidb/search?q=test&limit=100")
        assert response.status_code == 422

        # Test invalid limit (too low)
        response = client.get("/external/anidb/search?q=test&limit=0")
        assert response.status_code == 422

    def test_anime_id_validation_errors(self, client):
        """Test anime ID parameter validation."""
        # Test invalid anime ID (negative)
        response = client.get("/external/anidb/anime/-1")
        assert response.status_code == 422

        # Test invalid anime ID (zero)
        response = client.get("/external/anidb/anime/0")
        assert response.status_code == 422

    def test_similar_anime_validation_errors(self, client):
        """Test similar anime endpoint parameter validation."""
        # Test invalid limit (too high)
        response = client.get("/external/anidb/anime/1/similar?limit=51")
        assert response.status_code == 422

        # Test invalid limit (too low)
        response = client.get("/external/anidb/anime/1/similar?limit=0")
        assert response.status_code == 422
