"""Tests for Anime-Planet API endpoints."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from typing import Dict, Any

from src.api.external.animeplanet import router


class TestAnimePlanetAPI:
    """Test cases for Anime-Planet API endpoints."""
    
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
            "url": "https://anime-planet.com/anime/cowboy-bebop",
            "type": "TV",
            "episodes": 26,
            "year": 1998,
            "season": "Spring",
            "studio": "Sunrise",
            "rating": 4.4,
            "rank": 15,
            "synopsis": "In the year 2071, humanity has colonized several of the planets...",
            "genres": ["Action", "Space Western", "Drama"],
            "tags": ["Bounty Hunters", "Space", "Adult Cast"],
            "status": "completed"
        }
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing."""
        return [
            {
                "id": "cowboy-bebop",
                "title": "Cowboy Bebop",
                "url": "https://anime-planet.com/anime/cowboy-bebop",
                "type": "TV",
                "year": 1998,
                "rating": 4.4
            },
            {
                "id": "space-dandy",
                "title": "Space Dandy",
                "url": "https://anime-planet.com/anime/space-dandy",
                "type": "TV",
                "year": 2014,
                "rating": 4.0
            }
        ]
    
    @pytest.fixture
    def sample_character_data(self):
        """Sample character data for testing."""
        return [
            {
                "id": "spike-spiegel",
                "name": "Spike Spiegel",
                "url": "https://anime-planet.com/characters/spike-spiegel",
                "description": "A bounty hunter traveling on the spaceship Bebop"
            },
            {
                "id": "faye-valentine",
                "name": "Faye Valentine",
                "url": "https://anime-planet.com/characters/faye-valentine",
                "description": "A bounty hunter and former con artist"
            }
        ]
    
    @pytest.fixture
    def sample_recommendations_data(self):
        """Sample recommendations data for testing."""
        return [
            {
                "id": "trigun",
                "title": "Trigun",
                "url": "https://anime-planet.com/anime/trigun",
                "similarity_score": 0.85
            },
            {
                "id": "samurai-champloo",
                "title": "Samurai Champloo",
                "url": "https://anime-planet.com/anime/samurai-champloo",
                "similarity_score": 0.80
            }
        ]
    
    @pytest.fixture
    def sample_top_anime_data(self):
        """Sample top anime data for testing."""
        return [
            {
                "rank": 1,
                "id": "fullmetal-alchemist-brotherhood",
                "title": "Fullmetal Alchemist: Brotherhood",
                "url": "https://anime-planet.com/anime/fullmetal-alchemist-brotherhood",
                "rating": 4.6
            },
            {
                "rank": 2,
                "id": "hunter-x-hunter-2011",
                "title": "Hunter x Hunter (2011)",
                "url": "https://anime-planet.com/anime/hunter-x-hunter-2011",
                "rating": 4.5
            }
        ]
    
    @patch('src.api.external.animeplanet._animeplanet_service')
    def test_search_anime_success(self, mock_service, client, sample_search_results):
        """Test successful anime search endpoint."""
        # Setup
        mock_service.search_anime = AsyncMock(return_value=sample_search_results)
        
        # Execute
        response = client.get("/external/animeplanet/search?q=cowboy bebop&limit=10")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "animeplanet"
        assert data["query"] == "cowboy bebop"
        assert data["limit"] == 10
        assert len(data["results"]) == 2
        assert data["results"] == sample_search_results
        assert data["total_results"] == 2
        
        mock_service.search_anime.assert_called_once_with("cowboy bebop")
    
    @patch('src.api.external.animeplanet._animeplanet_service')
    def test_search_anime_default_params(self, mock_service, client):
        """Test search endpoint with default parameters."""
        # Setup
        mock_service.search_anime = AsyncMock(return_value=[])
        
        # Execute
        response = client.get("/external/animeplanet/search?q=test")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 20  # Default limit
        mock_service.search_anime.assert_called_once_with("test")
    
    @patch('src.api.external.animeplanet._animeplanet_service')
    def test_search_anime_service_error(self, mock_service, client):
        """Test search endpoint when service fails."""
        # Setup
        mock_service.search_anime = AsyncMock(side_effect=Exception("Scraping error"))
        
        # Execute
        response = client.get("/external/animeplanet/search?q=test")
        
        # Verify
        assert response.status_code == 503
        assert "Anime-Planet search service unavailable" in response.json()["detail"]
    
    @patch('src.api.external.animeplanet._animeplanet_service')
    def test_get_anime_details_success(self, mock_service, client, sample_anime_data):
        """Test successful anime details endpoint."""
        # Setup
        anime_id = "cowboy-bebop"
        mock_service.get_anime_details = AsyncMock(return_value=sample_anime_data)
        
        # Execute
        response = client.get(f"/external/animeplanet/anime/{anime_id}")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "animeplanet"
        assert data["anime_id"] == anime_id
        assert data["data"] == sample_anime_data
        
        mock_service.get_anime_details.assert_called_once_with(anime_id)
    
    @patch('src.api.external.animeplanet._animeplanet_service')
    def test_get_anime_details_not_found(self, mock_service, client):
        """Test anime details endpoint when anime not found."""
        # Setup
        anime_id = "nonexistent-anime"
        mock_service.get_anime_details = AsyncMock(return_value=None)
        
        # Execute
        response = client.get(f"/external/animeplanet/anime/{anime_id}")
        
        # Verify
        assert response.status_code == 404
        assert f"Anime with ID {anime_id} not found" in response.json()["detail"]
    
    @patch('src.api.external.animeplanet._animeplanet_service')
    def test_get_anime_details_service_error(self, mock_service, client):
        """Test anime details endpoint when service fails."""
        # Setup
        mock_service.get_anime_details = AsyncMock(side_effect=Exception("Scraping error"))
        
        # Execute
        response = client.get("/external/animeplanet/anime/test")
        
        # Verify
        assert response.status_code == 503
        assert "Anime-Planet service unavailable" in response.json()["detail"]
    
    @patch('src.api.external.animeplanet._animeplanet_service')
    def test_get_anime_characters_success(self, mock_service, client, sample_character_data):
        """Test successful anime characters endpoint."""
        # Setup
        anime_id = "cowboy-bebop"
        mock_service.get_anime_characters = AsyncMock(return_value=sample_character_data)
        
        # Execute
        response = client.get(f"/external/animeplanet/anime/{anime_id}/characters")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "animeplanet"
        assert data["anime_id"] == anime_id
        assert data["characters"] == sample_character_data
        assert data["total_characters"] == 2
        
        mock_service.get_anime_characters.assert_called_once_with(anime_id)
    
    @patch('src.api.external.animeplanet._animeplanet_service')
    def test_get_anime_recommendations_success(self, mock_service, client, sample_recommendations_data):
        """Test successful anime recommendations endpoint."""
        # Setup
        anime_id = "cowboy-bebop"
        limit = 5
        mock_service.get_anime_recommendations = AsyncMock(return_value=sample_recommendations_data)
        
        # Execute
        response = client.get(f"/external/animeplanet/anime/{anime_id}/recommendations?limit={limit}")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "animeplanet"
        assert data["anime_id"] == anime_id
        assert data["limit"] == limit
        assert data["recommendations"] == sample_recommendations_data
        assert data["total_results"] == 2
        
        mock_service.get_anime_recommendations.assert_called_once_with(anime_id, limit)
    
    @patch('src.api.external.animeplanet._animeplanet_service')
    def test_get_anime_recommendations_default_limit(self, mock_service, client, sample_recommendations_data):
        """Test recommendations endpoint with default limit."""
        # Setup
        anime_id = "cowboy-bebop"
        mock_service.get_anime_recommendations = AsyncMock(return_value=sample_recommendations_data)
        
        # Execute
        response = client.get(f"/external/animeplanet/anime/{anime_id}/recommendations")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10  # Default limit
        
        mock_service.get_anime_recommendations.assert_called_once_with(anime_id, 10)
    
    @patch('src.api.external.animeplanet._animeplanet_service')
    def test_get_top_anime_success(self, mock_service, client, sample_top_anime_data):
        """Test successful top anime endpoint."""
        # Setup
        category = "top-anime"
        limit = 20
        mock_service.get_top_anime = AsyncMock(return_value=sample_top_anime_data)
        
        # Execute
        response = client.get(f"/external/animeplanet/top?category={category}&limit={limit}")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "animeplanet"
        assert data["category"] == category
        assert data["limit"] == limit
        assert data["anime"] == sample_top_anime_data
        assert data["total_results"] == 2
        
        mock_service.get_top_anime.assert_called_once_with(category, limit)
    
    @patch('src.api.external.animeplanet._animeplanet_service')
    def test_get_top_anime_default_params(self, mock_service, client, sample_top_anime_data):
        """Test top anime endpoint with default parameters."""
        # Setup
        mock_service.get_top_anime = AsyncMock(return_value=sample_top_anime_data)
        
        # Execute
        response = client.get("/external/animeplanet/top")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "top-anime"  # Default category
        assert data["limit"] == 25  # Default limit
        
        mock_service.get_top_anime.assert_called_once_with("top-anime", 25)
    
    @patch('src.api.external.animeplanet._animeplanet_service')
    def test_get_top_anime_service_error(self, mock_service, client):
        """Test top anime endpoint when service fails."""
        # Setup
        mock_service.get_top_anime = AsyncMock(side_effect=Exception("Scraping error"))
        
        # Execute
        response = client.get("/external/animeplanet/top")
        
        # Verify
        assert response.status_code == 503
        assert "Anime-Planet top anime service unavailable" in response.json()["detail"]
    
    @patch('src.api.external.animeplanet._animeplanet_service')
    def test_health_check_healthy(self, mock_service, client):
        """Test health check endpoint when service is healthy."""
        # Setup
        health_data = {
            "service": "animeplanet",
            "status": "healthy",
            "circuit_breaker_open": False,
            "last_check": "success"
        }
        mock_service.health_check = AsyncMock(return_value=health_data)
        
        # Execute
        response = client.get("/external/animeplanet/health")
        
        # Verify
        assert response.status_code == 200
        assert response.json() == health_data
        
        mock_service.health_check.assert_called_once()
    
    @patch('src.api.external.animeplanet._animeplanet_service')
    def test_health_check_unhealthy(self, mock_service, client):
        """Test health check endpoint when service is unhealthy."""
        # Setup
        health_data = {
            "service": "animeplanet",
            "status": "unhealthy",
            "error": "Scraping failed",
            "circuit_breaker_open": True
        }
        mock_service.health_check = AsyncMock(return_value=health_data)
        
        # Execute
        response = client.get("/external/animeplanet/health")
        
        # Verify
        assert response.status_code == 200
        assert response.json() == health_data
    
    def test_search_anime_validation_errors(self, client):
        """Test search endpoint parameter validation."""
        # Test missing query parameter
        response = client.get("/external/animeplanet/search")
        assert response.status_code == 422
        
        # Test invalid limit (too high)
        response = client.get("/external/animeplanet/search?q=test&limit=100")
        assert response.status_code == 422
        
        # Test invalid limit (too low)
        response = client.get("/external/animeplanet/search?q=test&limit=0")
        assert response.status_code == 422
    
    def test_top_anime_validation_errors(self, client):
        """Test top anime endpoint parameter validation."""
        # Test invalid limit (too high)
        response = client.get("/external/animeplanet/top?limit=101")
        assert response.status_code == 422
        
        # Test invalid limit (too low)
        response = client.get("/external/animeplanet/top?limit=0")
        assert response.status_code == 422
    
    def test_recommendations_validation_errors(self, client):
        """Test recommendations endpoint parameter validation."""
        # Test invalid limit (too high)
        response = client.get("/external/animeplanet/anime/test/recommendations?limit=51")
        assert response.status_code == 422
        
        # Test invalid limit (too low)
        response = client.get("/external/animeplanet/anime/test/recommendations?limit=0")
        assert response.status_code == 422