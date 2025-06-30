"""Tests for AniList API endpoints."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from typing import Dict, Any

from src.api.external.anilist import router


class TestAniListAPI:
    """Test cases for AniList API endpoints."""
    
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
            "id": 21,
            "title": {
                "romaji": "One Piece",
                "english": "One Piece"
            },
            "description": "Gol D. Roger was known as the Pirate King...",
            "episodes": 1000,
            "status": "RELEASING",
            "genres": ["Action", "Adventure"],
            "averageScore": 87
        }
    
    @patch('src.api.external.anilist._anilist_service')
    def test_search_anime_success(self, mock_service, client, sample_anime_data):
        """Test successful anime search endpoint."""
        # Setup
        mock_service.search_anime = AsyncMock(return_value=[sample_anime_data])
        
        # Execute
        response = client.get("/external/anilist/search?q=one piece&limit=10&page=1")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "anilist"
        assert data["query"] == "one piece"
        assert data["page"] == 1
        assert data["limit"] == 10
        assert len(data["results"]) == 1
        assert data["results"][0] == sample_anime_data
        assert data["total_results"] == 1
        
        mock_service.search_anime.assert_called_once_with(
            query="one piece", limit=10, page=1
        )
    
    @patch('src.api.external.anilist._anilist_service')
    def test_search_anime_default_params(self, mock_service, client):
        """Test search endpoint with default parameters."""
        # Setup
        mock_service.search_anime = AsyncMock(return_value=[])
        
        # Execute
        response = client.get("/external/anilist/search?q=test")
        
        # Verify
        assert response.status_code == 200
        mock_service.search_anime.assert_called_once_with(
            query="test", limit=10, page=1
        )
    
    @patch('src.api.external.anilist._anilist_service')
    def test_search_anime_service_error(self, mock_service, client):
        """Test search endpoint when service fails."""
        # Setup
        mock_service.search_anime.side_effect = Exception("Service error")
        
        # Execute
        response = client.get("/external/anilist/search?q=test")
        
        # Verify
        assert response.status_code == 503
        assert "AniList search service unavailable" in response.json()["detail"]
    
    @patch('src.api.external.anilist._anilist_service')
    def test_get_anime_details_success(self, mock_service, client, sample_anime_data):
        """Test successful anime details endpoint."""
        # Setup
        anime_id = 21
        mock_service.get_anime_details = AsyncMock(return_value=sample_anime_data)
        
        # Execute
        response = client.get(f"/external/anilist/anime/{anime_id}")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "anilist"
        assert data["anime_id"] == anime_id
        assert data["data"] == sample_anime_data
        
        mock_service.get_anime_details.assert_called_once_with(anime_id)
    
    @patch('src.api.external.anilist._anilist_service')
    def test_get_anime_details_not_found(self, mock_service, client):
        """Test anime details endpoint when anime not found."""
        # Setup
        anime_id = 999
        mock_service.get_anime_details = AsyncMock(return_value=None)
        
        # Execute
        response = client.get(f"/external/anilist/anime/{anime_id}")
        
        # Verify
        assert response.status_code == 404
        assert f"Anime with ID {anime_id} not found" in response.json()["detail"]
    
    @patch('src.api.external.anilist._anilist_service')
    def test_get_anime_details_service_error(self, mock_service, client):
        """Test anime details endpoint when service fails."""
        # Setup
        mock_service.get_anime_details = AsyncMock(side_effect=Exception("Service error"))
        
        # Execute
        response = client.get("/external/anilist/anime/21")
        
        # Verify
        assert response.status_code == 503
        assert "AniList service unavailable" in response.json()["detail"]
    
    @patch('src.api.external.anilist._anilist_service')
    def test_get_trending_anime_success(self, mock_service, client, sample_anime_data):
        """Test successful trending anime endpoint."""
        # Setup
        mock_service.get_trending_anime = AsyncMock(return_value=[sample_anime_data])
        
        # Execute
        response = client.get("/external/anilist/trending?limit=20&page=1")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "anilist"
        assert data["type"] == "trending"
        assert data["page"] == 1
        assert data["limit"] == 20
        assert len(data["results"]) == 1
        assert data["total_results"] == 1
        
        mock_service.get_trending_anime.assert_called_once_with(limit=20, page=1)
    
    @patch('src.api.external.anilist._anilist_service')
    def test_get_trending_anime_default_params(self, mock_service, client):
        """Test trending endpoint with default parameters."""
        # Setup
        mock_service.get_trending_anime = AsyncMock(return_value=[])
        
        # Execute
        response = client.get("/external/anilist/trending")
        
        # Verify
        assert response.status_code == 200
        mock_service.get_trending_anime.assert_called_once_with(limit=20, page=1)
    
    @patch('src.api.external.anilist._anilist_service')
    def test_get_upcoming_anime_success(self, mock_service, client, sample_anime_data):
        """Test successful upcoming anime endpoint."""
        # Setup
        mock_service.get_upcoming_anime = AsyncMock(return_value=[sample_anime_data])
        
        # Execute
        response = client.get("/external/anilist/upcoming?limit=15&page=2")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "anilist"
        assert data["type"] == "upcoming"
        assert data["page"] == 2
        assert data["limit"] == 15
        
        mock_service.get_upcoming_anime.assert_called_once_with(limit=15, page=2)
    
    @patch('src.api.external.anilist._anilist_service')
    def test_get_popular_anime_success(self, mock_service, client, sample_anime_data):
        """Test successful popular anime endpoint."""
        # Setup
        mock_service.get_popular_anime = AsyncMock(return_value=[sample_anime_data])
        
        # Execute
        response = client.get("/external/anilist/popular")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "anilist"
        assert data["type"] == "popular"
        
        mock_service.get_popular_anime.assert_called_once_with(limit=20, page=1)
    
    @patch('src.api.external.anilist._anilist_service')
    def test_get_staff_details_success(self, mock_service, client):
        """Test successful staff details endpoint."""
        # Setup
        staff_id = 123
        staff_data = {
            "id": 123,
            "name": {"first": "Eiichiro", "last": "Oda"}
        }
        mock_service.get_staff_details = AsyncMock(return_value=staff_data)
        
        # Execute
        response = client.get(f"/external/anilist/staff/{staff_id}")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "anilist"
        assert data["staff_id"] == staff_id
        assert data["data"] == staff_data
        
        mock_service.get_staff_details.assert_called_once_with(staff_id)
    
    @patch('src.api.external.anilist._anilist_service')
    def test_get_staff_details_not_found(self, mock_service, client):
        """Test staff details endpoint when staff not found."""
        # Setup
        staff_id = 999
        mock_service.get_staff_details = AsyncMock(return_value=None)
        
        # Execute
        response = client.get(f"/external/anilist/staff/{staff_id}")
        
        # Verify
        assert response.status_code == 404
        assert f"Staff with ID {staff_id} not found" in response.json()["detail"]
    
    @patch('src.api.external.anilist._anilist_service')
    def test_get_studio_details_success(self, mock_service, client):
        """Test successful studio details endpoint."""
        # Setup
        studio_id = 456
        studio_data = {
            "id": 456,
            "name": "Studio Pierrot"
        }
        mock_service.get_studio_details = AsyncMock(return_value=studio_data)
        
        # Execute
        response = client.get(f"/external/anilist/studio/{studio_id}")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "anilist"
        assert data["studio_id"] == studio_id
        assert data["data"] == studio_data
        
        mock_service.get_studio_details.assert_called_once_with(studio_id)
    
    @patch('src.api.external.anilist._anilist_service')
    def test_get_studio_details_not_found(self, mock_service, client):
        """Test studio details endpoint when studio not found."""
        # Setup
        studio_id = 999
        mock_service.get_studio_details = AsyncMock(return_value=None)
        
        # Execute
        response = client.get(f"/external/anilist/studio/{studio_id}")
        
        # Verify
        assert response.status_code == 404
        assert f"Studio with ID {studio_id} not found" in response.json()["detail"]
    
    @patch('src.api.external.anilist._anilist_service')
    def test_health_check_healthy(self, mock_service, client):
        """Test health check endpoint when service is healthy."""
        # Setup
        health_data = {
            "service": "anilist",
            "status": "healthy",
            "circuit_breaker_open": False
        }
        mock_service.health_check = AsyncMock(return_value=health_data)
        
        # Execute
        response = client.get("/external/anilist/health")
        
        # Verify
        assert response.status_code == 200
        assert response.json() == health_data
        
        mock_service.health_check.assert_called_once()
    
    @patch('src.api.external.anilist._anilist_service')
    def test_health_check_unhealthy(self, mock_service, client):
        """Test health check endpoint when service is unhealthy."""
        # Setup
        health_data = {
            "service": "anilist",
            "status": "unhealthy",
            "error": "Service down",
            "circuit_breaker_open": True
        }
        mock_service.health_check = AsyncMock(return_value=health_data)
        
        # Execute
        response = client.get("/external/anilist/health")
        
        # Verify
        assert response.status_code == 200
        assert response.json() == health_data
    
    def test_search_anime_validation_errors(self, client):
        """Test search endpoint parameter validation."""
        # Test missing query parameter
        response = client.get("/external/anilist/search")
        assert response.status_code == 422
        
        # Test invalid limit (too high)
        response = client.get("/external/anilist/search?q=test&limit=100")
        assert response.status_code == 422
        
        # Test invalid page (too low)
        response = client.get("/external/anilist/search?q=test&page=0")
        assert response.status_code == 422
    
    def test_trending_anime_validation_errors(self, client):
        """Test trending endpoint parameter validation."""
        # Test invalid limit (too high)
        response = client.get("/external/anilist/trending?limit=100")
        assert response.status_code == 422
        
        # Test invalid page (too low)
        response = client.get("/external/anilist/trending?page=0")
        assert response.status_code == 422