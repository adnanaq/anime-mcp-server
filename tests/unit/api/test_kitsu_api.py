"""Tests for Kitsu API endpoints."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from typing import Dict, Any

from src.api.external.kitsu import router


class TestKitsuAPI:
    """Test cases for Kitsu API endpoints."""
    
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
            "id": "1",
            "type": "anime",
            "attributes": {
                "slug": "cowboy-bebop",
                "synopsis": "In the year 2071, humanity has colonized several of the planets...",
                "titles": {
                    "en": "Cowboy Bebop",
                    "en_jp": "Cowboy Bebop",
                    "ja_jp": "カウボーイビバップ"
                },
                "canonicalTitle": "Cowboy Bebop",
                "averageRating": "82.95",
                "userCount": 89569,
                "favoritesCount": 4906,
                "startDate": "1998-04-03",
                "endDate": "1999-04-24",
                "popularityRank": 138,
                "ratingRank": 28,
                "subtype": "TV",
                "status": "finished",
                "episodeCount": 26,
                "episodeLength": 24
            }
        }
    
    @patch('src.api.external.kitsu._kitsu_service')
    def test_search_anime_success(self, mock_service, client, sample_anime_data):
        """Test successful anime search endpoint."""
        # Setup
        mock_service.search_anime = AsyncMock(return_value=[sample_anime_data])
        
        # Execute
        response = client.get("/external/kitsu/search?q=cowboy bebop&limit=10")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "kitsu"
        assert data["query"] == "cowboy bebop"
        assert data["limit"] == 10
        assert len(data["results"]) == 1
        assert data["results"][0] == sample_anime_data
        assert data["total_results"] == 1
        
        mock_service.search_anime.assert_called_once_with(
            query="cowboy bebop", limit=10
        )
    
    @patch('src.api.external.kitsu._kitsu_service')
    def test_search_anime_default_params(self, mock_service, client):
        """Test search endpoint with default parameters."""
        # Setup
        mock_service.search_anime = AsyncMock(return_value=[])
        
        # Execute
        response = client.get("/external/kitsu/search?q=test")
        
        # Verify
        assert response.status_code == 200
        mock_service.search_anime.assert_called_once_with(
            query="test", limit=10
        )
    
    @patch('src.api.external.kitsu._kitsu_service')
    def test_search_anime_service_error(self, mock_service, client):
        """Test search endpoint when service fails."""
        # Setup
        mock_service.search_anime = AsyncMock(side_effect=Exception("Service error"))
        
        # Execute
        response = client.get("/external/kitsu/search?q=test")
        
        # Verify
        assert response.status_code == 503
        assert "Kitsu search service unavailable" in response.json()["detail"]
    
    @patch('src.api.external.kitsu._kitsu_service')
    def test_get_anime_details_success(self, mock_service, client, sample_anime_data):
        """Test successful anime details endpoint."""
        # Setup
        anime_id = 1
        mock_service.get_anime_details = AsyncMock(return_value=sample_anime_data)
        
        # Execute
        response = client.get(f"/external/kitsu/anime/{anime_id}")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "kitsu"
        assert data["anime_id"] == anime_id
        assert data["data"] == sample_anime_data
        
        mock_service.get_anime_details.assert_called_once_with(anime_id)
    
    @patch('src.api.external.kitsu._kitsu_service')
    def test_get_anime_details_not_found(self, mock_service, client):
        """Test anime details endpoint when anime not found."""
        # Setup
        anime_id = 999
        mock_service.get_anime_details = AsyncMock(return_value=None)
        
        # Execute
        response = client.get(f"/external/kitsu/anime/{anime_id}")
        
        # Verify
        assert response.status_code == 404
        assert f"Anime with ID {anime_id} not found" in response.json()["detail"]
    
    @patch('src.api.external.kitsu._kitsu_service')
    def test_get_anime_details_service_error(self, mock_service, client):
        """Test anime details endpoint when service fails."""
        # Setup
        mock_service.get_anime_details = AsyncMock(side_effect=Exception("Service error"))
        
        # Execute
        response = client.get("/external/kitsu/anime/1")
        
        # Verify
        assert response.status_code == 503
        assert "Kitsu service unavailable" in response.json()["detail"]
    
    @patch('src.api.external.kitsu._kitsu_service')
    def test_get_trending_anime_success(self, mock_service, client, sample_anime_data):
        """Test successful trending anime endpoint."""
        # Setup
        mock_service.get_trending_anime = AsyncMock(return_value=[sample_anime_data])
        
        # Execute
        response = client.get("/external/kitsu/trending?limit=15")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "kitsu"
        assert data["type"] == "trending"
        assert data["limit"] == 15
        assert len(data["results"]) == 1
        assert data["total_results"] == 1
        
        mock_service.get_trending_anime.assert_called_once_with(15)
    
    @patch('src.api.external.kitsu._kitsu_service')
    def test_get_trending_anime_default_params(self, mock_service, client):
        """Test trending endpoint with default parameters."""
        # Setup
        mock_service.get_trending_anime = AsyncMock(return_value=[])
        
        # Execute
        response = client.get("/external/kitsu/trending")
        
        # Verify
        assert response.status_code == 200
        mock_service.get_trending_anime.assert_called_once_with(20)
    
    @patch('src.api.external.kitsu._kitsu_service')
    def test_get_anime_episodes_success(self, mock_service, client):
        """Test successful anime episodes endpoint."""
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
                    "synopsis": "Spike and Jet head to Tijuana..."
                }
            }
        ]
        mock_service.get_anime_episodes = AsyncMock(return_value=episodes_data)
        
        # Execute
        response = client.get(f"/external/kitsu/anime/{anime_id}/episodes")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "kitsu"
        assert data["anime_id"] == anime_id
        assert data["episodes"] == episodes_data
        assert data["total_episodes"] == 1
        
        mock_service.get_anime_episodes.assert_called_once_with(anime_id)
    
    @patch('src.api.external.kitsu._kitsu_service')
    def test_get_streaming_links_success(self, mock_service, client):
        """Test successful streaming links endpoint."""
        # Setup
        anime_id = 1
        streaming_data = [
            {
                "id": "1",
                "type": "streaming-links",
                "attributes": {
                    "url": "https://www.funimation.com/shows/cowboy-bebop/",
                    "subs": ["en"],
                    "dubs": ["en"]
                }
            }
        ]
        mock_service.get_streaming_links = AsyncMock(return_value=streaming_data)
        
        # Execute
        response = client.get(f"/external/kitsu/anime/{anime_id}/streaming")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "kitsu"
        assert data["anime_id"] == anime_id
        assert data["streaming_links"] == streaming_data
        assert data["total_links"] == 1
        
        mock_service.get_streaming_links.assert_called_once_with(anime_id)
    
    @patch('src.api.external.kitsu._kitsu_service')
    def test_get_anime_characters_success(self, mock_service, client):
        """Test successful anime characters endpoint."""
        # Setup
        anime_id = 1
        characters_data = [
            {
                "id": "1",
                "type": "anime-characters",
                "attributes": {
                    "role": "main"
                }
            }
        ]
        mock_service.get_anime_characters = AsyncMock(return_value=characters_data)
        
        # Execute
        response = client.get(f"/external/kitsu/anime/{anime_id}/characters")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "kitsu"
        assert data["anime_id"] == anime_id
        assert data["characters"] == characters_data
        assert data["total_characters"] == 1
        
        mock_service.get_anime_characters.assert_called_once_with(anime_id)
    
    @patch('src.api.external.kitsu._kitsu_service')
    def test_health_check_healthy(self, mock_service, client):
        """Test health check endpoint when service is healthy."""
        # Setup
        health_data = {
            "service": "kitsu",
            "status": "healthy",
            "circuit_breaker_open": False
        }
        mock_service.health_check = AsyncMock(return_value=health_data)
        
        # Execute
        response = client.get("/external/kitsu/health")
        
        # Verify
        assert response.status_code == 200
        assert response.json() == health_data
        
        mock_service.health_check.assert_called_once()
    
    @patch('src.api.external.kitsu._kitsu_service')
    def test_health_check_unhealthy(self, mock_service, client):
        """Test health check endpoint when service is unhealthy."""
        # Setup
        health_data = {
            "service": "kitsu",
            "status": "unhealthy",
            "error": "Service down",
            "circuit_breaker_open": True
        }
        mock_service.health_check = AsyncMock(return_value=health_data)
        
        # Execute
        response = client.get("/external/kitsu/health")
        
        # Verify
        assert response.status_code == 200
        assert response.json() == health_data
    
    def test_search_anime_validation_errors(self, client):
        """Test search endpoint parameter validation."""
        # Test missing query parameter
        response = client.get("/external/kitsu/search")
        assert response.status_code == 422
        
        # Test invalid limit (too high)
        response = client.get("/external/kitsu/search?q=test&limit=100")
        assert response.status_code == 422
        
        # Test invalid limit (too low)
        response = client.get("/external/kitsu/search?q=test&limit=0")
        assert response.status_code == 422
    
    def test_trending_anime_validation_errors(self, client):
        """Test trending endpoint parameter validation."""
        # Test invalid limit (too high)
        response = client.get("/external/kitsu/trending?limit=100")
        assert response.status_code == 422
        
        # Test invalid limit (too low)
        response = client.get("/external/kitsu/trending?limit=0")
        assert response.status_code == 422