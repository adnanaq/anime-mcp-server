"""Tests for AnimeSchedule.net service integration."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any, List
from datetime import datetime

from src.services.external.animeschedule_service import AnimeScheduleService


class TestAnimeScheduleService:
    """Test cases for AnimeSchedule service."""
    
    @pytest.fixture
    def animeschedule_service(self):
        """Create AnimeSchedule service for testing."""
        with patch('src.services.external.animeschedule_service.AnimeScheduleClient') as mock_client_class, \
             patch('src.services.external.base_service.CircuitBreaker') as mock_cb_class:
            
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            mock_circuit_breaker = Mock()
            mock_circuit_breaker.is_open = Mock(return_value=False)
            mock_cb_class.return_value = mock_circuit_breaker
            
            service = AnimeScheduleService()
            service.client = mock_client
            return service
    
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
                },
                {
                    "id": 456,
                    "title": "Demon Slayer",
                    "time": "23:30",
                    "episode": 8,
                    "broadcast_network": "Fuji TV",
                    "streaming_platforms": ["Crunchyroll"],
                    "duration": 24,
                    "thumbnail": "https://example.com/thumb2.jpg"
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
    
    @pytest.mark.asyncio
    async def test_get_today_timetable_success(self, animeschedule_service, sample_timetable_data):
        """Test successful today timetable retrieval."""
        # Setup
        timezone = "Asia/Tokyo"
        region = "JP"
        
        animeschedule_service.client.get_today_timetable.return_value = sample_timetable_data
        
        # Execute
        result = await animeschedule_service.get_today_timetable(timezone, region)
        
        # Verify
        assert result == sample_timetable_data
        animeschedule_service.client.get_today_timetable.assert_called_once_with(
            timezone=timezone, region=region
        )
    
    @pytest.mark.asyncio
    async def test_get_today_timetable_default_params(self, animeschedule_service, sample_timetable_data):
        """Test today timetable with default parameters."""
        # Setup
        animeschedule_service.client.get_today_timetable.return_value = sample_timetable_data
        
        # Execute
        result = await animeschedule_service.get_today_timetable()
        
        # Verify
        assert result == sample_timetable_data
        animeschedule_service.client.get_today_timetable.assert_called_once_with(
            timezone=None, region=None
        )
    
    @pytest.mark.asyncio
    async def test_get_timetable_by_date_success(self, animeschedule_service, sample_timetable_data):
        """Test successful timetable by date retrieval."""
        # Setup
        date = "2024-01-15"
        
        animeschedule_service.client.get_timetable_by_date.return_value = sample_timetable_data
        
        # Execute
        result = await animeschedule_service.get_timetable_by_date(date)
        
        # Verify
        assert result == sample_timetable_data
        animeschedule_service.client.get_timetable_by_date.assert_called_once_with(date)
    
    @pytest.mark.asyncio
    async def test_get_timetable_by_date_invalid_format(self, animeschedule_service):
        """Test timetable by date with invalid date format."""
        # Setup
        invalid_date = "2024/01/15"  # Wrong format
        
        # Execute & Verify
        with pytest.raises(ValueError, match="Invalid date format"):
            await animeschedule_service.get_timetable_by_date(invalid_date)
    
    @pytest.mark.asyncio
    async def test_search_anime_success(self, animeschedule_service, sample_anime_data):
        """Test successful anime search."""
        # Setup
        query = "one piece"
        
        animeschedule_service.client.search_anime.return_value = sample_anime_data
        
        # Execute
        result = await animeschedule_service.search_anime(query)
        
        # Verify
        assert result == sample_anime_data
        animeschedule_service.client.search_anime.assert_called_once_with(query)
    
    @pytest.mark.asyncio
    async def test_search_anime_failure(self, animeschedule_service):
        """Test anime search failure handling."""
        # Setup
        animeschedule_service.client.search_anime.side_effect = Exception("API Error")
        
        # Execute & Verify
        with pytest.raises(Exception, match="API Error"):
            await animeschedule_service.search_anime("test")
    
    @pytest.mark.asyncio
    async def test_get_seasonal_anime_success(self, animeschedule_service, sample_seasonal_data):
        """Test successful seasonal anime retrieval."""
        # Setup
        season = "winter"
        year = 2024
        
        animeschedule_service.client.get_seasonal_anime.return_value = sample_seasonal_data
        
        # Execute
        result = await animeschedule_service.get_seasonal_anime(season, year)
        
        # Verify
        assert result == sample_seasonal_data
        animeschedule_service.client.get_seasonal_anime.assert_called_once_with(season, year)
    
    @pytest.mark.asyncio
    async def test_get_seasonal_anime_invalid_season(self, animeschedule_service):
        """Test seasonal anime with invalid season."""
        # Setup
        invalid_season = "invalid"
        year = 2024
        
        # Execute & Verify
        with pytest.raises(ValueError, match="Invalid season"):
            await animeschedule_service.get_seasonal_anime(invalid_season, year)
    
    @pytest.mark.asyncio
    async def test_get_seasonal_anime_invalid_year(self, animeschedule_service):
        """Test seasonal anime with invalid year."""
        # Setup
        season = "winter"
        invalid_year = 1990  # Too old
        
        # Execute & Verify
        with pytest.raises(ValueError, match="Invalid year"):
            await animeschedule_service.get_seasonal_anime(season, invalid_year)
    
    @pytest.mark.asyncio
    async def test_get_anime_schedule_success(self, animeschedule_service):
        """Test successful anime schedule by ID retrieval."""
        # Setup
        anime_id = 123
        schedule_data = {
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
        
        animeschedule_service.client.get_anime_schedule_by_id.return_value = schedule_data
        
        # Execute
        result = await animeschedule_service.get_anime_schedule(anime_id)
        
        # Verify
        assert result == schedule_data
        animeschedule_service.client.get_anime_schedule_by_id.assert_called_once_with(anime_id)
    
    @pytest.mark.asyncio
    async def test_get_anime_schedule_not_found(self, animeschedule_service):
        """Test anime schedule when not found."""
        # Setup
        animeschedule_service.client.get_anime_schedule_by_id.return_value = None
        
        # Execute
        result = await animeschedule_service.get_anime_schedule(999)
        
        # Verify
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_streaming_platforms_success(self, animeschedule_service):
        """Test successful streaming platforms retrieval."""
        # Setup
        platforms_data = [
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
        
        animeschedule_service.client.get_streaming_platforms.return_value = platforms_data
        
        # Execute
        result = await animeschedule_service.get_streaming_platforms()
        
        # Verify
        assert result == platforms_data
        animeschedule_service.client.get_streaming_platforms.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, animeschedule_service):
        """Test health check when service is healthy."""
        # Setup
        compatibility_data = {"status": "ok", "version": "v3"}
        animeschedule_service.client.check_api_compatibility.return_value = compatibility_data
        animeschedule_service.circuit_breaker.is_open.return_value = False
        
        # Execute
        result = await animeschedule_service.health_check()
        
        # Verify
        assert result["service"] == "animeschedule"
        assert result["status"] == "healthy"
        assert result["circuit_breaker_open"] == False
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, animeschedule_service):
        """Test health check when service is unhealthy."""
        # Setup
        animeschedule_service.client.check_api_compatibility.side_effect = Exception("Service down")
        animeschedule_service.circuit_breaker.is_open.return_value = True
        
        # Execute
        result = await animeschedule_service.health_check()
        
        # Verify
        assert result["service"] == "animeschedule"
        assert result["status"] == "unhealthy"
        assert "error" in result
        assert result["circuit_breaker_open"] == True
    
    def test_service_initialization(self, animeschedule_service):
        """Test service initialization."""
        assert animeschedule_service.service_name == "animeschedule"
        assert animeschedule_service.cache_manager is not None
        assert animeschedule_service.circuit_breaker is not None
        assert animeschedule_service.client is not None
    
    def test_is_healthy(self, animeschedule_service):
        """Test is_healthy method."""
        # Setup - healthy
        animeschedule_service.circuit_breaker.is_open.return_value = False
        
        # Execute & Verify
        assert animeschedule_service.is_healthy() == True
        
        # Setup - unhealthy
        animeschedule_service.circuit_breaker.is_open.return_value = True
        
        # Execute & Verify
        assert animeschedule_service.is_healthy() == False
    
    def test_get_service_info(self, animeschedule_service):
        """Test get_service_info method."""
        # Setup
        animeschedule_service.circuit_breaker.is_open.return_value = False
        
        # Execute
        info = animeschedule_service.get_service_info()
        
        # Verify
        assert info["name"] == "animeschedule"
        assert info["healthy"] == True
        assert info["circuit_breaker_open"] == False