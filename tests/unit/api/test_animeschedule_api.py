"""Tests for AnimeSchedule.net API endpoints."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from typing import Dict, Any

from src.api.external.animeschedule import router


class TestAnimeScheduleAPI:
    """Test cases for AnimeSchedule API endpoints."""
    
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
    def sample_timetable_data(self):
        """Sample timetable data for testing."""
        return {
            "date": "2024-01-15",
            "timezone": "Asia/Tokyo",
            "entries": [
                {
                    "id": 123,
                    "title": "Attack on Titan Final Season",
                    "time": "23:00",
                    "episode": 12,
                    "broadcast_network": "NHK",
                    "streaming_platforms": ["Crunchyroll", "Funimation"],
                    "duration": 24,
                    "thumbnail": "https://example.com/thumb.jpg"
                }
            ]
        }
    
    @pytest.fixture
    def sample_anime_data(self):
        """Sample anime search data for testing."""
        return [
            {
                "id": 789,
                "title": "One Piece",
                "status": "airing",
                "start_date": "1999-10-20",
                "end_date": None,
                "episode_count": None,
                "genres": ["Action", "Adventure"],
                "studio": "Toei Animation",
                "rating": "PG-13"
            }
        ]
    
    @pytest.fixture
    def sample_seasonal_data(self):
        """Sample seasonal anime data for testing."""
        return {
            "season": "winter",
            "year": 2024,
            "anime_list": [
                {
                    "id": 111,
                    "title": "Frieren: Beyond Journey's End",
                    "status": "airing",
                    "episodes": 28,
                    "studio": "Madhouse",
                    "genres": ["Adventure", "Drama", "Fantasy"]
                }
            ]
        }
    
    @pytest.fixture
    def sample_schedule_data(self):
        """Sample anime schedule data for testing."""
        return {
            "id": 123,
            "title": "Attack on Titan Final Season",
            "next_episode": {
                "episode": 13,
                "air_date": "2024-01-22",
                "air_time": "23:00"
            },
            "broadcast_info": {
                "network": "NHK",
                "timezone": "Asia/Tokyo"
            }
        }
    
    @pytest.fixture
    def sample_platforms_data(self):
        """Sample streaming platforms data for testing."""
        return [
            {
                "id": 1,
                "name": "Crunchyroll",
                "url": "https://crunchyroll.com",
                "regions": ["US", "CA", "UK"]
            },
            {
                "id": 2,
                "name": "Funimation",
                "url": "https://funimation.com",
                "regions": ["US", "CA"]
            }
        ]
    
    @patch('src.api.external.animeschedule._animeschedule_service')
    def test_get_today_timetable_success(self, mock_service, client, sample_timetable_data):
        """Test successful today timetable endpoint."""
        # Setup
        mock_service.get_today_timetable = AsyncMock(return_value=sample_timetable_data)
        
        # Execute
        response = client.get("/external/animeschedule/today?timezone=Asia/Tokyo&region=JP")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "animeschedule"
        assert data["type"] == "today_timetable"
        assert data["timezone"] == "Asia/Tokyo"
        assert data["region"] == "JP"
        assert data["data"] == sample_timetable_data
        
        mock_service.get_today_timetable.assert_called_once_with(
            timezone="Asia/Tokyo", region="JP"
        )
    
    @patch('src.api.external.animeschedule._animeschedule_service')
    def test_get_today_timetable_default_params(self, mock_service, client, sample_timetable_data):
        """Test today timetable endpoint with default parameters."""
        # Setup
        mock_service.get_today_timetable = AsyncMock(return_value=sample_timetable_data)
        
        # Execute
        response = client.get("/external/animeschedule/today")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["timezone"] is None
        assert data["region"] is None
        
        mock_service.get_today_timetable.assert_called_once_with(
            timezone=None, region=None
        )
    
    @patch('src.api.external.animeschedule._animeschedule_service')
    def test_get_today_timetable_service_error(self, mock_service, client):
        """Test today timetable endpoint when service fails."""
        # Setup
        mock_service.get_today_timetable = AsyncMock(side_effect=Exception("Service error"))
        
        # Execute
        response = client.get("/external/animeschedule/today")
        
        # Verify
        assert response.status_code == 503
        assert "AnimeSchedule today timetable service unavailable" in response.json()["detail"]
    
    @patch('src.api.external.animeschedule._animeschedule_service')
    def test_get_timetable_by_date_success(self, mock_service, client, sample_timetable_data):
        """Test successful timetable by date endpoint."""
        # Setup
        date = "2024-01-15"
        mock_service.get_timetable_by_date = AsyncMock(return_value=sample_timetable_data)
        
        # Execute
        response = client.get(f"/external/animeschedule/timetable/{date}")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "animeschedule"
        assert data["type"] == "date_timetable"
        assert data["date"] == date
        assert data["data"] == sample_timetable_data
        
        mock_service.get_timetable_by_date.assert_called_once_with(date)
    
    @patch('src.api.external.animeschedule._animeschedule_service')
    def test_get_timetable_by_date_invalid_format(self, mock_service, client):
        """Test timetable by date endpoint with invalid date format."""
        # Setup
        invalid_date = "2024-01-15"  # Valid format but will trigger ValueError from service
        mock_service.get_timetable_by_date = AsyncMock(
            side_effect=ValueError("Invalid date format")
        )
        
        # Execute
        response = client.get(f"/external/animeschedule/timetable/{invalid_date}")
        
        # Verify
        assert response.status_code == 400
        assert "Invalid date format" in response.json()["detail"]
    
    @patch('src.api.external.animeschedule._animeschedule_service')
    def test_search_anime_success(self, mock_service, client, sample_anime_data):
        """Test successful anime search endpoint."""
        # Setup
        mock_service.search_anime = AsyncMock(return_value=sample_anime_data)
        
        # Execute
        response = client.get("/external/animeschedule/search?q=one piece")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "animeschedule"
        assert data["query"] == "one piece"
        assert data["results"] == sample_anime_data
        assert data["total_results"] == 1
        
        mock_service.search_anime.assert_called_once_with("one piece")
    
    @patch('src.api.external.animeschedule._animeschedule_service')
    def test_search_anime_service_error(self, mock_service, client):
        """Test search endpoint when service fails."""
        # Setup
        mock_service.search_anime = AsyncMock(side_effect=Exception("Service error"))
        
        # Execute
        response = client.get("/external/animeschedule/search?q=test")
        
        # Verify
        assert response.status_code == 503
        assert "AnimeSchedule search service unavailable" in response.json()["detail"]
    
    @patch('src.api.external.animeschedule._animeschedule_service')
    def test_get_seasonal_anime_success(self, mock_service, client, sample_seasonal_data):
        """Test successful seasonal anime endpoint."""
        # Setup
        season = "winter"
        year = 2024
        mock_service.get_seasonal_anime = AsyncMock(return_value=sample_seasonal_data)
        
        # Execute
        response = client.get(f"/external/animeschedule/seasonal/{season}/{year}")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "animeschedule"
        assert data["season"] == season
        assert data["year"] == year
        assert data["data"] == sample_seasonal_data
        
        mock_service.get_seasonal_anime.assert_called_once_with(season, year)
    
    @patch('src.api.external.animeschedule._animeschedule_service')
    def test_get_seasonal_anime_invalid_season(self, mock_service, client):
        """Test seasonal anime endpoint with invalid season."""
        # Setup
        invalid_season = "invalid"
        year = 2024
        mock_service.get_seasonal_anime = AsyncMock(
            side_effect=ValueError("Invalid season")
        )
        
        # Execute
        response = client.get(f"/external/animeschedule/seasonal/{invalid_season}/{year}")
        
        # Verify
        assert response.status_code == 400
        assert "Invalid season" in response.json()["detail"]
    
    @patch('src.api.external.animeschedule._animeschedule_service')
    def test_get_seasonal_anime_invalid_year(self, mock_service, client):
        """Test seasonal anime endpoint with invalid year."""
        # Setup - FastAPI validates year range, so test with service ValueError
        season = "winter"
        year = 2001  # Valid for FastAPI but service will reject
        mock_service.get_seasonal_anime = AsyncMock(
            side_effect=ValueError("Invalid year")
        )
        
        # Execute
        response = client.get(f"/external/animeschedule/seasonal/{season}/{year}")
        
        # Verify
        assert response.status_code == 400
        assert "Invalid year" in response.json()["detail"]
    
    @patch('src.api.external.animeschedule._animeschedule_service')
    def test_get_anime_schedule_success(self, mock_service, client, sample_schedule_data):
        """Test successful anime schedule endpoint."""
        # Setup
        anime_id = 123
        mock_service.get_anime_schedule = AsyncMock(return_value=sample_schedule_data)
        
        # Execute
        response = client.get(f"/external/animeschedule/anime/{anime_id}/schedule")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "animeschedule"
        assert data["anime_id"] == anime_id
        assert data["schedule"] == sample_schedule_data
        
        mock_service.get_anime_schedule.assert_called_once_with(anime_id)
    
    @patch('src.api.external.animeschedule._animeschedule_service')
    def test_get_anime_schedule_not_found(self, mock_service, client):
        """Test anime schedule endpoint when anime not found."""
        # Setup
        anime_id = 999
        mock_service.get_anime_schedule = AsyncMock(return_value=None)
        
        # Execute
        response = client.get(f"/external/animeschedule/anime/{anime_id}/schedule")
        
        # Verify
        assert response.status_code == 404
        assert f"Anime schedule for ID {anime_id} not found" in response.json()["detail"]
    
    @patch('src.api.external.animeschedule._animeschedule_service')
    def test_get_anime_schedule_service_error(self, mock_service, client):
        """Test anime schedule endpoint when service fails."""
        # Setup
        mock_service.get_anime_schedule = AsyncMock(side_effect=Exception("Service error"))
        
        # Execute
        response = client.get("/external/animeschedule/anime/1/schedule")
        
        # Verify
        assert response.status_code == 503
        assert "AnimeSchedule service unavailable" in response.json()["detail"]
    
    @patch('src.api.external.animeschedule._animeschedule_service')
    def test_get_streaming_platforms_success(self, mock_service, client, sample_platforms_data):
        """Test successful streaming platforms endpoint."""
        # Setup
        mock_service.get_streaming_platforms = AsyncMock(return_value=sample_platforms_data)
        
        # Execute
        response = client.get("/external/animeschedule/platforms")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "animeschedule"
        assert data["platforms"] == sample_platforms_data
        assert data["total_platforms"] == 2
        
        mock_service.get_streaming_platforms.assert_called_once()
    
    @patch('src.api.external.animeschedule._animeschedule_service')
    def test_get_streaming_platforms_service_error(self, mock_service, client):
        """Test streaming platforms endpoint when service fails."""
        # Setup
        mock_service.get_streaming_platforms = AsyncMock(side_effect=Exception("Service error"))
        
        # Execute
        response = client.get("/external/animeschedule/platforms")
        
        # Verify
        assert response.status_code == 503
        assert "AnimeSchedule platforms service unavailable" in response.json()["detail"]
    
    @patch('src.api.external.animeschedule._animeschedule_service')
    def test_health_check_healthy(self, mock_service, client):
        """Test health check endpoint when service is healthy."""
        # Setup
        health_data = {
            "service": "animeschedule",
            "status": "healthy",
            "circuit_breaker_open": False
        }
        mock_service.health_check = AsyncMock(return_value=health_data)
        
        # Execute
        response = client.get("/external/animeschedule/health")
        
        # Verify
        assert response.status_code == 200
        assert response.json() == health_data
        
        mock_service.health_check.assert_called_once()
    
    @patch('src.api.external.animeschedule._animeschedule_service')
    def test_health_check_unhealthy(self, mock_service, client):
        """Test health check endpoint when service is unhealthy."""
        # Setup
        health_data = {
            "service": "animeschedule",
            "status": "unhealthy",
            "error": "Service down",
            "circuit_breaker_open": True
        }
        mock_service.health_check = AsyncMock(return_value=health_data)
        
        # Execute
        response = client.get("/external/animeschedule/health")
        
        # Verify
        assert response.status_code == 200
        assert response.json() == health_data
    
    def test_search_anime_validation_errors(self, client):
        """Test search endpoint parameter validation."""
        # Test missing query parameter
        response = client.get("/external/animeschedule/search")
        assert response.status_code == 422
    
    def test_anime_id_validation_errors(self, client):
        """Test anime ID parameter validation."""
        # Test invalid anime ID (negative)
        response = client.get("/external/animeschedule/anime/-1/schedule")
        assert response.status_code == 422
        
        # Test invalid anime ID (zero)
        response = client.get("/external/animeschedule/anime/0/schedule")
        assert response.status_code == 422
    
    def test_year_validation_errors(self, client):
        """Test year parameter validation."""
        # Test invalid year (too low)
        response = client.get("/external/animeschedule/seasonal/winter/1999")
        assert response.status_code == 422
        
        # Test invalid year (too high)
        response = client.get("/external/animeschedule/seasonal/winter/2026")
        assert response.status_code == 422