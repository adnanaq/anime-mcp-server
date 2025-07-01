"""Tests for AniSearch API endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.external.anisearch import router


class TestAniSearchAPI:
    """Test cases for AniSearch API endpoints."""

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
            "id": "cowboy-bebop",
            "title": "Cowboy Bebop",
            "url": "https://anisearch.com/anime/25",
            "type": "TV Series",
            "episodes": 26,
            "year": 1998,
            "season": "Spring",
            "studio": "Sunrise",
            "rating": 8.9,
            "status": "finished",
            "synopsis": "In the year 2071, humanity has colonized several of the planets...",
            "genres": ["Action", "Space Western", "Drama", "Sci-Fi"],
            "start_date": "1998-04-03",
            "end_date": "1999-04-24",
        }

    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing."""
        return [
            {
                "id": "cowboy-bebop",
                "title": "Cowboy Bebop",
                "url": "https://anisearch.com/anime/25",
                "type": "TV Series",
                "year": 1998,
                "rating": 8.9,
                "status": "finished",
            },
            {
                "id": "space-dandy",
                "title": "Space Dandy",
                "url": "https://anisearch.com/anime/9735",
                "type": "TV Series",
                "year": 2014,
                "rating": 7.5,
                "status": "finished",
            },
        ]

    @pytest.fixture
    def sample_character_data(self):
        """Sample character data for testing."""
        return [
            {
                "id": "spike-spiegel",
                "name": "Spike Spiegel",
                "url": "https://anisearch.com/character/157",
                "description": "A bounty hunter traveling on the spaceship Bebop",
                "age": 27,
                "height": "185 cm",
            },
            {
                "id": "faye-valentine",
                "name": "Faye Valentine",
                "url": "https://anisearch.com/character/158",
                "description": "A bounty hunter and former con artist",
                "age": 77,
                "height": "168 cm",
            },
        ]

    @pytest.fixture
    def sample_recommendations_data(self):
        """Sample recommendations data for testing."""
        return [
            {
                "id": "trigun",
                "title": "Trigun",
                "url": "https://anisearch.com/anime/53",
                "similarity_score": 0.85,
                "rating": 8.3,
            },
            {
                "id": "samurai-champloo",
                "title": "Samurai Champloo",
                "url": "https://anisearch.com/anime/631",
                "similarity_score": 0.80,
                "rating": 8.5,
            },
        ]

    @pytest.fixture
    def sample_top_anime_data(self):
        """Sample top anime data for testing."""
        return [
            {
                "rank": 1,
                "id": "fullmetal-alchemist-brotherhood",
                "title": "Fullmetal Alchemist: Brotherhood",
                "url": "https://anisearch.com/anime/5391",
                "rating": 9.4,
                "votes": 87000,
                "year": 2009,
            },
            {
                "rank": 2,
                "id": "spirited-away",
                "title": "Spirited Away",
                "url": "https://anisearch.com/anime/424",
                "rating": 9.3,
                "votes": 65000,
                "year": 2001,
            },
        ]

    @pytest.fixture
    def sample_seasonal_data(self):
        """Sample seasonal anime data for testing."""
        return [
            {
                "id": "attack-on-titan-s4",
                "title": "Attack on Titan: The Final Season",
                "url": "https://anisearch.com/anime/14977",
                "type": "TV Series",
                "year": 2020,
                "season": "Fall",
                "studio": "Wit Studio",
                "rating": 9.0,
                "status": "finished",
            },
            {
                "id": "jujutsu-kaisen",
                "title": "Jujutsu Kaisen",
                "url": "https://anisearch.com/anime/14403",
                "type": "TV Series",
                "year": 2020,
                "season": "Fall",
                "studio": "MAPPA",
                "rating": 8.9,
                "status": "finished",
            },
        ]

    @patch("src.api.external.anisearch._anisearch_service")
    def test_search_anime_success(self, mock_service, client, sample_search_results):
        """Test successful anime search endpoint."""
        # Setup
        mock_service.search_anime = AsyncMock(return_value=sample_search_results)

        # Execute
        response = client.get("/external/anisearch/search?q=cowboy bebop&limit=10")

        # Verify
        assert response.status_code == 200
        data = response.json()

        assert data["source"] == "anisearch"
        assert data["query"] == "cowboy bebop"
        assert data["limit"] == 10
        assert len(data["results"]) == 2
        assert data["results"] == sample_search_results
        assert data["total_results"] == 2

        mock_service.search_anime.assert_called_once_with("cowboy bebop")

    @patch("src.api.external.anisearch._anisearch_service")
    def test_search_anime_default_params(self, mock_service, client):
        """Test search endpoint with default parameters."""
        # Setup
        mock_service.search_anime = AsyncMock(return_value=[])

        # Execute
        response = client.get("/external/anisearch/search?q=test")

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 20  # Default limit
        mock_service.search_anime.assert_called_once_with("test")

    @patch("src.api.external.anisearch._anisearch_service")
    def test_search_anime_service_error(self, mock_service, client):
        """Test search endpoint when service fails."""
        # Setup
        mock_service.search_anime = AsyncMock(side_effect=Exception("Scraping error"))

        # Execute
        response = client.get("/external/anisearch/search?q=test")

        # Verify
        assert response.status_code == 503
        assert "AniSearch search service unavailable" in response.json()["detail"]

    @patch("src.api.external.anisearch._anisearch_service")
    def test_get_anime_details_success(self, mock_service, client, sample_anime_data):
        """Test successful anime details endpoint."""
        # Setup
        anime_id = "cowboy-bebop"
        mock_service.get_anime_details = AsyncMock(return_value=sample_anime_data)

        # Execute
        response = client.get(f"/external/anisearch/anime/{anime_id}")

        # Verify
        assert response.status_code == 200
        data = response.json()

        assert data["source"] == "anisearch"
        assert data["anime_id"] == anime_id
        assert data["data"] == sample_anime_data

        mock_service.get_anime_details.assert_called_once_with(anime_id)

    @patch("src.api.external.anisearch._anisearch_service")
    def test_get_anime_details_not_found(self, mock_service, client):
        """Test anime details endpoint when anime not found."""
        # Setup
        anime_id = "nonexistent-anime"
        mock_service.get_anime_details = AsyncMock(return_value=None)

        # Execute
        response = client.get(f"/external/anisearch/anime/{anime_id}")

        # Verify
        assert response.status_code == 404
        assert f"Anime with ID {anime_id} not found" in response.json()["detail"]

    @patch("src.api.external.anisearch._anisearch_service")
    def test_get_anime_details_service_error(self, mock_service, client):
        """Test anime details endpoint when service fails."""
        # Setup
        mock_service.get_anime_details = AsyncMock(
            side_effect=Exception("Scraping error")
        )

        # Execute
        response = client.get("/external/anisearch/anime/test")

        # Verify
        assert response.status_code == 503
        assert "AniSearch service unavailable" in response.json()["detail"]

    @patch("src.api.external.anisearch._anisearch_service")
    def test_get_anime_characters_success(
        self, mock_service, client, sample_character_data
    ):
        """Test successful anime characters endpoint."""
        # Setup
        anime_id = "cowboy-bebop"
        mock_service.get_anime_characters = AsyncMock(
            return_value=sample_character_data
        )

        # Execute
        response = client.get(f"/external/anisearch/anime/{anime_id}/characters")

        # Verify
        assert response.status_code == 200
        data = response.json()

        assert data["source"] == "anisearch"
        assert data["anime_id"] == anime_id
        assert data["characters"] == sample_character_data
        assert data["total_characters"] == 2

        mock_service.get_anime_characters.assert_called_once_with(anime_id)

    @patch("src.api.external.anisearch._anisearch_service")
    def test_get_anime_characters_service_error(self, mock_service, client):
        """Test anime characters endpoint when service fails."""
        # Setup
        mock_service.get_anime_characters = AsyncMock(
            side_effect=Exception("Scraping error")
        )

        # Execute
        response = client.get("/external/anisearch/anime/test/characters")

        # Verify
        assert response.status_code == 503
        assert "AniSearch characters service unavailable" in response.json()["detail"]

    @patch("src.api.external.anisearch._anisearch_service")
    def test_get_anime_recommendations_success(
        self, mock_service, client, sample_recommendations_data
    ):
        """Test successful anime recommendations endpoint."""
        # Setup
        anime_id = "cowboy-bebop"
        limit = 5
        mock_service.get_anime_recommendations = AsyncMock(
            return_value=sample_recommendations_data
        )

        # Execute
        response = client.get(
            f"/external/anisearch/anime/{anime_id}/recommendations?limit={limit}"
        )

        # Verify
        assert response.status_code == 200
        data = response.json()

        assert data["source"] == "anisearch"
        assert data["anime_id"] == anime_id
        assert data["limit"] == limit
        assert data["recommendations"] == sample_recommendations_data
        assert data["total_results"] == 2

        mock_service.get_anime_recommendations.assert_called_once_with(anime_id, limit)

    @patch("src.api.external.anisearch._anisearch_service")
    def test_get_anime_recommendations_default_limit(
        self, mock_service, client, sample_recommendations_data
    ):
        """Test recommendations endpoint with default limit."""
        # Setup
        anime_id = "cowboy-bebop"
        mock_service.get_anime_recommendations = AsyncMock(
            return_value=sample_recommendations_data
        )

        # Execute
        response = client.get(f"/external/anisearch/anime/{anime_id}/recommendations")

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10  # Default limit

        mock_service.get_anime_recommendations.assert_called_once_with(anime_id, 10)

    @patch("src.api.external.anisearch._anisearch_service")
    def test_get_top_anime_success(self, mock_service, client, sample_top_anime_data):
        """Test successful top anime endpoint."""
        # Setup
        category = "highest_rated"
        limit = 20
        mock_service.get_top_anime = AsyncMock(return_value=sample_top_anime_data)

        # Execute
        response = client.get(
            f"/external/anisearch/top?category={category}&limit={limit}"
        )

        # Verify
        assert response.status_code == 200
        data = response.json()

        assert data["source"] == "anisearch"
        assert data["category"] == category
        assert data["limit"] == limit
        assert data["anime"] == sample_top_anime_data
        assert data["total_results"] == 2

        mock_service.get_top_anime.assert_called_once_with(category, limit)

    @patch("src.api.external.anisearch._anisearch_service")
    def test_get_top_anime_default_params(
        self, mock_service, client, sample_top_anime_data
    ):
        """Test top anime endpoint with default parameters."""
        # Setup
        mock_service.get_top_anime = AsyncMock(return_value=sample_top_anime_data)

        # Execute
        response = client.get("/external/anisearch/top")

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "highest_rated"  # Default category
        assert data["limit"] == 25  # Default limit

        mock_service.get_top_anime.assert_called_once_with("highest_rated", 25)

    @patch("src.api.external.anisearch._anisearch_service")
    def test_get_top_anime_service_error(self, mock_service, client):
        """Test top anime endpoint when service fails."""
        # Setup
        mock_service.get_top_anime = AsyncMock(side_effect=Exception("Scraping error"))

        # Execute
        response = client.get("/external/anisearch/top")

        # Verify
        assert response.status_code == 503
        assert "AniSearch top anime service unavailable" in response.json()["detail"]

    @patch("src.api.external.anisearch._anisearch_service")
    def test_get_seasonal_anime_success(
        self, mock_service, client, sample_seasonal_data
    ):
        """Test successful seasonal anime endpoint."""
        # Setup
        year = 2020
        season = "fall"
        limit = 20
        mock_service.get_seasonal_anime = AsyncMock(return_value=sample_seasonal_data)

        # Execute
        response = client.get(
            f"/external/anisearch/seasonal?year={year}&season={season}&limit={limit}"
        )

        # Verify
        assert response.status_code == 200
        data = response.json()

        assert data["source"] == "anisearch"
        assert data["year"] == year
        assert data["season"] == season
        assert data["limit"] == limit
        assert data["anime"] == sample_seasonal_data
        assert data["total_results"] == 2

        mock_service.get_seasonal_anime.assert_called_once_with(year, season, limit)

    @patch("src.api.external.anisearch._anisearch_service")
    def test_get_seasonal_anime_default_params(
        self, mock_service, client, sample_seasonal_data
    ):
        """Test seasonal anime endpoint with default parameters."""
        # Setup
        mock_service.get_seasonal_anime = AsyncMock(return_value=sample_seasonal_data)

        # Execute
        response = client.get("/external/anisearch/seasonal")

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 25  # Default limit
        # Should use current year and season
        mock_service.get_seasonal_anime.assert_called_once_with(None, None, 25)

    @patch("src.api.external.anisearch._anisearch_service")
    def test_get_seasonal_anime_service_error(self, mock_service, client):
        """Test seasonal anime endpoint when service fails."""
        # Setup
        mock_service.get_seasonal_anime = AsyncMock(
            side_effect=Exception("Scraping error")
        )

        # Execute
        response = client.get("/external/anisearch/seasonal")

        # Verify
        assert response.status_code == 503
        assert "AniSearch seasonal service unavailable" in response.json()["detail"]

    @patch("src.api.external.anisearch._anisearch_service")
    def test_health_check_healthy(self, mock_service, client):
        """Test health check endpoint when service is healthy."""
        # Setup
        health_data = {
            "service": "anisearch",
            "status": "healthy",
            "circuit_breaker_open": False,
            "last_check": "success",
        }
        mock_service.health_check = AsyncMock(return_value=health_data)

        # Execute
        response = client.get("/external/anisearch/health")

        # Verify
        assert response.status_code == 200
        assert response.json() == health_data

        mock_service.health_check.assert_called_once()

    @patch("src.api.external.anisearch._anisearch_service")
    def test_health_check_unhealthy(self, mock_service, client):
        """Test health check endpoint when service is unhealthy."""
        # Setup
        health_data = {
            "service": "anisearch",
            "status": "unhealthy",
            "error": "Scraping failed",
            "circuit_breaker_open": True,
        }
        mock_service.health_check = AsyncMock(return_value=health_data)

        # Execute
        response = client.get("/external/anisearch/health")

        # Verify
        assert response.status_code == 200
        assert response.json() == health_data

    def test_search_anime_validation_errors(self, client):
        """Test search endpoint parameter validation."""
        # Test missing query parameter
        response = client.get("/external/anisearch/search")
        assert response.status_code == 422

        # Test invalid limit (too high)
        response = client.get("/external/anisearch/search?q=test&limit=101")
        assert response.status_code == 422

        # Test invalid limit (too low)
        response = client.get("/external/anisearch/search?q=test&limit=0")
        assert response.status_code == 422

    def test_top_anime_validation_errors(self, client):
        """Test top anime endpoint parameter validation."""
        # Test invalid limit (too high)
        response = client.get("/external/anisearch/top?limit=101")
        assert response.status_code == 422

        # Test invalid limit (too low)
        response = client.get("/external/anisearch/top?limit=0")
        assert response.status_code == 422

    def test_recommendations_validation_errors(self, client):
        """Test recommendations endpoint parameter validation."""
        # Test invalid limit (too high)
        response = client.get("/external/anisearch/anime/test/recommendations?limit=51")
        assert response.status_code == 422

        # Test invalid limit (too low)
        response = client.get("/external/anisearch/anime/test/recommendations?limit=0")
        assert response.status_code == 422

    def test_seasonal_anime_validation_errors(self, client):
        """Test seasonal anime endpoint parameter validation."""
        # Test invalid limit (too high)
        response = client.get("/external/anisearch/seasonal?limit=101")
        assert response.status_code == 422

        # Test invalid limit (too low)
        response = client.get("/external/anisearch/seasonal?limit=0")
        assert response.status_code == 422
