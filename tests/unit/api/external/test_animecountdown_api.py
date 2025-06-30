"""Tests for AnimeCountdown API endpoints."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from typing import Dict, Any, List

from src.main import app

client = TestClient(app)


class TestAnimeCountdownAPI:
    """Test cases for AnimeCountdown API endpoints."""
    
    @pytest.fixture
    def sample_anime_data(self):
        """Sample anime data for testing."""
        return {
            "id": "cowboy-bebop",
            "title": "Cowboy Bebop",
            "url": "https://animecountdown.com/cowboy-bebop",
            "type": "TV",
            "episodes": 26,
            "year": 1998,
            "season": "Spring",
            "studio": "Sunrise",
            "status": "finished",
            "synopsis": "In the year 2071, humanity has colonized several of the planets...",
            "genres": ["Action", "Space Western", "Drama"],
            "tags": ["Bounty Hunters", "Space", "Adult Cast"],
            "alternative_titles": [
                "Space Warriors",
                "カウボーイビバップ"
            ],
            "image_url": "https://animecountdown.com/images/cowboy-bebop.jpg",
            "air_date": "1998-04-03",
            "end_date": "1999-04-24",
            "countdown_info": {
                "next_episode": None,
                "time_until_next": None,
                "is_airing": False
            }
        }
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing."""
        return [
            {
                "id": "cowboy-bebop",
                "title": "Cowboy Bebop",
                "url": "https://animecountdown.com/cowboy-bebop",
                "type": "TV",
                "year": 1998,
                "image_url": "https://animecountdown.com/images/cowboy-bebop.jpg",
                "status": "finished"
            },
            {
                "id": "space-dandy",
                "title": "Space Dandy",
                "url": "https://animecountdown.com/space-dandy",
                "type": "TV",
                "year": 2014,
                "image_url": "https://animecountdown.com/images/space-dandy.jpg",
                "status": "finished"
            }
        ]
    
    @pytest.fixture
    def sample_airing_data(self):
        """Sample currently airing anime data."""
        return [
            {
                "id": "demon-slayer-s3",
                "title": "Demon Slayer: Swordsmith Village Arc",
                "url": "https://animecountdown.com/demon-slayer-s3",
                "type": "TV",
                "year": 2023,
                "season": "Spring",
                "studio": "Ufotable",
                "status": "airing",
                "air_date": "2023-04-09",
                "countdown_info": {
                    "next_episode": 8,
                    "time_until_next": "2 days, 5 hours",
                    "is_airing": True,
                    "next_air_time": "2023-05-28T15:15:00Z"
                }
            }
        ]
    
    @pytest.fixture
    def sample_upcoming_data(self):
        """Sample upcoming anime data."""
        return [
            {
                "id": "chainsaw-man-s2",
                "title": "Chainsaw Man Season 2",
                "url": "https://animecountdown.com/chainsaw-man-s2",
                "type": "TV",
                "year": 2024,
                "season": "Fall",
                "studio": "MAPPA",
                "status": "upcoming",
                "countdown_info": {
                    "premiere_date": "2024-10-15",
                    "time_until_premiere": "120 days",
                    "is_airing": False
                }
            }
        ]
    
    @pytest.fixture
    def sample_popular_data(self):
        """Sample popular anime data."""
        return [
            {
                "id": "your-name",
                "title": "Your Name",
                "url": "https://animecountdown.com/your-name",
                "type": "Movie",
                "year": 2016,
                "studio": "CoMix Wave Films",
                "status": "finished",
                "popularity_rank": 1,
                "countdown_views": 1250000
            }
        ]
    
    @pytest.fixture
    def sample_countdown_data(self):
        """Sample countdown data."""
        return {
            "anime_id": "demon-slayer-s3",
            "title": "Demon Slayer: Swordsmith Village Arc",
            "next_episode": 8,
            "time_until_next": "2 days, 5 hours, 23 minutes",
            "next_air_time": "2023-05-28T15:15:00Z",
            "is_airing": True,
            "total_episodes": 11,
            "remaining_episodes": 4
        }
    
    @patch('src.api.external.animecountdown.animecountdown_service')
    def test_search_anime_success(self, mock_service, sample_search_results):
        """Test successful anime search."""
        # Setup
        mock_service.search_anime = AsyncMock(return_value=sample_search_results)
        
        # Execute
        response = client.get("/api/external/animecountdown/search?q=cowboy%20bebop")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == sample_search_results
        assert data["total"] == len(sample_search_results)
        mock_service.search_anime.assert_called_once_with("cowboy bebop")
    
    @patch('src.api.external.animecountdown.animecountdown_service')
    def test_search_anime_empty_query(self, mock_service):
        """Test anime search with empty query."""
        # Execute
        response = client.get("/api/external/animecountdown/search")
        
        # Verify
        assert response.status_code == 422  # Validation error
    
    @patch('src.api.external.animecountdown.animecountdown_service')
    def test_search_anime_service_error(self, mock_service):
        """Test anime search service error handling."""
        # Setup
        mock_service.search_anime = AsyncMock(side_effect=Exception("Service error"))
        
        # Execute
        response = client.get("/api/external/animecountdown/search?q=test")
        
        # Verify
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "error" in data["detail"]
        assert "Service error" in data["detail"]["error"]
    
    @patch('src.api.external.animecountdown.animecountdown_service')
    def test_get_anime_details_success(self, mock_service, sample_anime_data):
        """Test successful anime details retrieval."""
        # Setup
        mock_service.get_anime_details = AsyncMock(return_value=sample_anime_data)
        
        # Execute
        response = client.get("/api/external/animecountdown/anime/cowboy-bebop")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data == sample_anime_data
        mock_service.get_anime_details.assert_called_once_with("cowboy-bebop")
    
    @patch('src.api.external.animecountdown.animecountdown_service')
    def test_get_anime_details_not_found(self, mock_service):
        """Test anime details when not found."""
        # Setup
        mock_service.get_anime_details = AsyncMock(return_value=None)
        
        # Execute
        response = client.get("/api/external/animecountdown/anime/nonexistent")
        
        # Verify
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    @patch('src.api.external.animecountdown.animecountdown_service')
    def test_get_anime_details_service_error(self, mock_service):
        """Test anime details service error handling."""
        # Setup
        mock_service.get_anime_details = AsyncMock(side_effect=Exception("Service error"))
        
        # Execute
        response = client.get("/api/external/animecountdown/anime/test-id")
        
        # Verify
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "error" in data["detail"]
        assert "Service error" in data["detail"]["error"]
    
    @patch('src.api.external.animecountdown.animecountdown_service')
    def test_get_currently_airing_success(self, mock_service, sample_airing_data):
        """Test successful currently airing anime retrieval."""
        # Setup
        mock_service.get_currently_airing = AsyncMock(return_value=sample_airing_data)
        
        # Execute
        response = client.get("/api/external/animecountdown/currently-airing?limit=10")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == sample_airing_data
        assert data["total"] == len(sample_airing_data)
        mock_service.get_currently_airing.assert_called_once_with(10)
    
    @patch('src.api.external.animecountdown.animecountdown_service')
    def test_get_currently_airing_default_limit(self, mock_service, sample_airing_data):
        """Test currently airing with default limit."""
        # Setup
        mock_service.get_currently_airing = AsyncMock(return_value=sample_airing_data)
        
        # Execute
        response = client.get("/api/external/animecountdown/currently-airing")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == sample_airing_data
        mock_service.get_currently_airing.assert_called_once_with(25)  # Default limit
    
    @patch('src.api.external.animecountdown.animecountdown_service')
    def test_get_currently_airing_invalid_limit(self, mock_service):
        """Test currently airing with invalid limit."""
        # Execute
        response = client.get("/api/external/animecountdown/currently-airing?limit=0")
        
        # Verify
        assert response.status_code == 422  # Validation error
    
    @patch('src.api.external.animecountdown.animecountdown_service')
    def test_get_upcoming_anime_success(self, mock_service, sample_upcoming_data):
        """Test successful upcoming anime retrieval."""
        # Setup
        mock_service.get_upcoming_anime = AsyncMock(return_value=sample_upcoming_data)
        
        # Execute
        response = client.get("/api/external/animecountdown/upcoming?limit=15")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == sample_upcoming_data
        assert data["total"] == len(sample_upcoming_data)
        mock_service.get_upcoming_anime.assert_called_once_with(15)
    
    @patch('src.api.external.animecountdown.animecountdown_service')
    def test_get_upcoming_anime_default_limit(self, mock_service, sample_upcoming_data):
        """Test upcoming anime with default limit."""
        # Setup
        mock_service.get_upcoming_anime = AsyncMock(return_value=sample_upcoming_data)
        
        # Execute
        response = client.get("/api/external/animecountdown/upcoming")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == sample_upcoming_data
        mock_service.get_upcoming_anime.assert_called_once_with(20)  # Default limit
    
    @patch('src.api.external.animecountdown.animecountdown_service')
    def test_get_popular_anime_success(self, mock_service, sample_popular_data):
        """Test successful popular anime retrieval."""
        # Setup
        mock_service.get_popular_anime = AsyncMock(return_value=sample_popular_data)
        
        # Execute
        response = client.get("/api/external/animecountdown/popular?time_period=this_month&limit=10")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == sample_popular_data
        assert data["total"] == len(sample_popular_data)
        mock_service.get_popular_anime.assert_called_once_with("this_month", 10)
    
    @patch('src.api.external.animecountdown.animecountdown_service')
    def test_get_popular_anime_default_params(self, mock_service, sample_popular_data):
        """Test popular anime with default parameters."""
        # Setup
        mock_service.get_popular_anime = AsyncMock(return_value=sample_popular_data)
        
        # Execute
        response = client.get("/api/external/animecountdown/popular")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == sample_popular_data
        mock_service.get_popular_anime.assert_called_once_with("all_time", 25)
    
    @patch('src.api.external.animecountdown.animecountdown_service')
    def test_get_popular_anime_invalid_time_period(self, mock_service):
        """Test popular anime with invalid time period."""
        # Execute
        response = client.get("/api/external/animecountdown/popular?time_period=invalid")
        
        # Verify
        assert response.status_code == 422  # Validation error
    
    @patch('src.api.external.animecountdown.animecountdown_service')
    def test_get_anime_countdown_success(self, mock_service, sample_countdown_data):
        """Test successful anime countdown retrieval."""
        # Setup
        mock_service.get_anime_countdown = AsyncMock(return_value=sample_countdown_data)
        
        # Execute
        response = client.get("/api/external/animecountdown/countdown/demon-slayer-s3")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data == sample_countdown_data
        mock_service.get_anime_countdown.assert_called_once_with("demon-slayer-s3")
    
    @patch('src.api.external.animecountdown.animecountdown_service')
    def test_get_anime_countdown_not_found(self, mock_service):
        """Test anime countdown when not found."""
        # Setup
        mock_service.get_anime_countdown = AsyncMock(return_value=None)
        
        # Execute
        response = client.get("/api/external/animecountdown/countdown/nonexistent")
        
        # Verify
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    @patch('src.api.external.animecountdown.animecountdown_service')
    def test_health_check_success(self, mock_service):
        """Test successful health check."""
        # Setup
        mock_service.health_check = AsyncMock(return_value={
            "service": "animecountdown",
            "status": "healthy",
            "circuit_breaker_open": False,
            "last_check": "success"
        })
        
        # Execute
        response = client.get("/api/external/animecountdown/health")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "animecountdown"
        assert data["status"] == "healthy"
        mock_service.health_check.assert_called_once()
    
    @patch('src.api.external.animecountdown.animecountdown_service')
    def test_health_check_unhealthy(self, mock_service):
        """Test health check when service is unhealthy."""
        # Setup
        mock_service.health_check = AsyncMock(return_value={
            "service": "animecountdown",
            "status": "unhealthy",
            "error": "Connection failed",
            "circuit_breaker_open": True
        })
        
        # Execute
        response = client.get("/api/external/animecountdown/health")
        
        # Verify
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data
        assert data["detail"]["service"] == "animecountdown"
        assert data["detail"]["status"] == "unhealthy"
        assert "error" in data["detail"]