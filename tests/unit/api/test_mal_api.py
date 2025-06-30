"""Tests for MAL API endpoints."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from typing import Dict, Any

from src.api.external.mal import router


class TestMALAPI:
    """Test cases for MAL API endpoints."""
    
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
            "mal_id": 21,
            "title": "One Piece",
            "title_english": "One Piece",
            "title_japanese": "ワンピース",
            "synopsis": "Gol D. Roger was known as the Pirate King...",
            "episodes": 1000,
            "status": "Currently Airing",
            "genres": [{"name": "Action"}, {"name": "Adventure"}],
            "score": 8.7
        }
    
    @patch('src.api.external.mal._mal_service')
    def test_search_anime_success(self, mock_service, client, sample_anime_data):
        """Test successful anime search endpoint."""
        # Setup
        mock_service.search_anime = AsyncMock(return_value=[sample_anime_data])
        
        # Execute
        response = client.get("/external/mal/search?q=one piece&limit=10")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "mal"
        assert data["query"] == "one piece"
        assert data["limit"] == 10
        assert len(data["results"]) == 1
        assert data["results"][0] == sample_anime_data
        assert data["total_results"] == 1
        
        mock_service.search_anime.assert_called_once_with(
            query="one piece", limit=10, status=None, genres=None
        )
    
    @patch('src.api.external.mal._mal_service')
    def test_search_anime_with_filters(self, mock_service, client, sample_anime_data):
        """Test search endpoint with filters."""
        # Setup
        mock_service.search_anime = AsyncMock(return_value=[sample_anime_data])
        
        # Execute
        response = client.get("/external/mal/search?q=action&limit=20&status=airing&genres=1,2")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "mal"
        assert data["query"] == "action"
        assert data["limit"] == 20
        assert data["status"] == "airing"
        assert data["genres"] == [1, 2]
        
        mock_service.search_anime.assert_called_once_with(
            query="action", limit=20, status="airing", genres=[1, 2]
        )
    
    @patch('src.api.external.mal._mal_service')
    def test_search_anime_default_params(self, mock_service, client):
        """Test search endpoint with default parameters."""
        # Setup
        mock_service.search_anime = AsyncMock(return_value=[])
        
        # Execute
        response = client.get("/external/mal/search?q=test")
        
        # Verify
        assert response.status_code == 200
        mock_service.search_anime.assert_called_once_with(
            query="test", limit=10, status=None, genres=None
        )
    
    @patch('src.api.external.mal._mal_service')
    def test_search_anime_service_error(self, mock_service, client):
        """Test search endpoint when service fails."""
        # Setup
        mock_service.search_anime = AsyncMock(side_effect=Exception("Service error"))
        
        # Execute
        response = client.get("/external/mal/search?q=test")
        
        # Verify
        assert response.status_code == 503
        assert "MAL search service unavailable" in response.json()["detail"]
    
    @patch('src.api.external.mal._mal_service')
    def test_get_anime_details_success(self, mock_service, client, sample_anime_data):
        """Test successful anime details endpoint."""
        # Setup
        anime_id = 21
        mock_service.get_anime_details = AsyncMock(return_value=sample_anime_data)
        
        # Execute
        response = client.get(f"/external/mal/anime/{anime_id}")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "mal"
        assert data["anime_id"] == anime_id
        assert data["data"] == sample_anime_data
        
        mock_service.get_anime_details.assert_called_once_with(anime_id)
    
    @patch('src.api.external.mal._mal_service')
    def test_get_anime_details_not_found(self, mock_service, client):
        """Test anime details endpoint when anime not found."""
        # Setup
        anime_id = 999
        mock_service.get_anime_details = AsyncMock(return_value=None)
        
        # Execute
        response = client.get(f"/external/mal/anime/{anime_id}")
        
        # Verify
        assert response.status_code == 404
        assert f"Anime with ID {anime_id} not found" in response.json()["detail"]
    
    @patch('src.api.external.mal._mal_service')
    def test_get_anime_details_service_error(self, mock_service, client):
        """Test anime details endpoint when service fails."""
        # Setup
        mock_service.get_anime_details = AsyncMock(side_effect=Exception("Service error"))
        
        # Execute
        response = client.get("/external/mal/anime/21")
        
        # Verify
        assert response.status_code == 503
        assert "MAL service unavailable" in response.json()["detail"]
    
    @patch('src.api.external.mal._mal_service')
    def test_get_seasonal_anime_success(self, mock_service, client, sample_anime_data):
        """Test successful seasonal anime endpoint."""
        # Setup
        mock_service.get_seasonal_anime = AsyncMock(return_value=[sample_anime_data])
        
        # Execute
        response = client.get("/external/mal/seasonal/2024/winter")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "mal"
        assert data["year"] == 2024
        assert data["season"] == "winter"
        assert len(data["results"]) == 1
        assert data["total_results"] == 1
        
        mock_service.get_seasonal_anime.assert_called_once_with(2024, "winter")
    
    @patch('src.api.external.mal._mal_service')
    def test_get_current_season_success(self, mock_service, client, sample_anime_data):
        """Test successful current season endpoint."""
        # Setup
        mock_service.get_current_season = AsyncMock(return_value=[sample_anime_data])
        
        # Execute
        response = client.get("/external/mal/current-season")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "mal"
        assert data["type"] == "current_season"
        assert len(data["results"]) == 1
        
        mock_service.get_current_season.assert_called_once()
    
    @patch('src.api.external.mal._mal_service')
    def test_get_anime_statistics_success(self, mock_service, client):
        """Test successful anime statistics endpoint."""
        # Setup
        anime_id = 21
        stats_data = {
            "watching": 100000,
            "completed": 200000,
            "on_hold": 5000,
            "dropped": 3000,
            "plan_to_watch": 50000
        }
        mock_service.get_anime_statistics = AsyncMock(return_value=stats_data)
        
        # Execute
        response = client.get(f"/external/mal/anime/{anime_id}/statistics")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "mal"
        assert data["anime_id"] == anime_id
        assert data["statistics"] == stats_data
        
        mock_service.get_anime_statistics.assert_called_once_with(anime_id)
    
    @patch('src.api.external.mal._mal_service')
    def test_get_anime_statistics_not_found(self, mock_service, client):
        """Test anime statistics endpoint when anime not found."""
        # Setup
        anime_id = 999
        mock_service.get_anime_statistics = AsyncMock(return_value={})
        
        # Execute
        response = client.get(f"/external/mal/anime/{anime_id}/statistics")
        
        # Verify
        assert response.status_code == 404
        assert f"Statistics for anime ID {anime_id} not found" in response.json()["detail"]
    
    @patch('src.api.external.mal._mal_service')
    def test_health_check_healthy(self, mock_service, client):
        """Test health check endpoint when service is healthy."""
        # Setup
        health_data = {
            "service": "mal",
            "status": "healthy",
            "circuit_breaker_open": False
        }
        mock_service.health_check = AsyncMock(return_value=health_data)
        
        # Execute
        response = client.get("/external/mal/health")
        
        # Verify
        assert response.status_code == 200
        assert response.json() == health_data
        
        mock_service.health_check.assert_called_once()
    
    @patch('src.api.external.mal._mal_service')
    def test_health_check_unhealthy(self, mock_service, client):
        """Test health check endpoint when service is unhealthy."""
        # Setup
        health_data = {
            "service": "mal",
            "status": "unhealthy",
            "error": "Service down",
            "circuit_breaker_open": True
        }
        mock_service.health_check = AsyncMock(return_value=health_data)
        
        # Execute
        response = client.get("/external/mal/health")
        
        # Verify
        assert response.status_code == 200
        assert response.json() == health_data
    
    def test_search_anime_validation_errors(self, client):
        """Test search endpoint parameter validation."""
        # Test missing query parameter
        response = client.get("/external/mal/search")
        assert response.status_code == 422
        
        # Test invalid limit (too high)
        response = client.get("/external/mal/search?q=test&limit=100")
        assert response.status_code == 422
        
        # Test invalid status
        response = client.get("/external/mal/search?q=test&status=invalid")
        assert response.status_code == 422
    
    def test_seasonal_anime_validation_errors(self, client):
        """Test seasonal endpoint parameter validation."""
        # Test invalid year
        response = client.get("/external/mal/seasonal/1900/winter")
        assert response.status_code == 422
        
        # Test invalid season
        response = client.get("/external/mal/seasonal/2024/invalid")
        assert response.status_code == 422