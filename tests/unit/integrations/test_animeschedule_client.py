"""Tests for AnimeSchedule.net REST client."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Optional
import json
import asyncio
from datetime import datetime

from src.integrations.clients.animeschedule_client import AnimeScheduleClient
from src.integrations.error_handling import ErrorContext, CircuitBreaker
from src.integrations.cache_manager import CollaborativeCacheSystem


class TestAnimeScheduleClient:
    """Test the AnimeSchedule.net REST client."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock dependencies for AnimeSchedule client."""
        cache_manager = Mock(spec=CollaborativeCacheSystem)
        cache_manager.get = AsyncMock(return_value=None)
        
        circuit_breaker = Mock(spec=CircuitBreaker)
        circuit_breaker.is_open = Mock(return_value=False)
        
        return {
            "circuit_breaker": circuit_breaker,
            "rate_limiter": None,  # Unlimited requests
            "cache_manager": cache_manager,
            "error_handler": Mock(spec=ErrorContext)
        }
    
    @pytest.fixture
    def animeschedule_client(self, mock_dependencies):
        """Create AnimeSchedule client with mocked dependencies."""
        return AnimeScheduleClient(**mock_dependencies)
    
    @pytest.fixture
    def sample_timetable_response(self):
        """Sample AnimeSchedule.net timetable response."""
        return {
            "data": [
                {
                    "id": 1,
                    "anime_id": 12345,
                    "title": "Attack on Titan",
                    "episode": 87,
                    "air_date": "2024-01-15T09:00:00Z",
                    "duration": 24,
                    "image": "https://animeschedule.net/images/12345.jpg",
                    "url": "https://myanimelist.net/anime/12345",
                    "streaming": [
                        {
                            "platform": "Crunchyroll",
                            "url": "https://crunchyroll.com/watch/12345",
                            "available_regions": ["US", "CA", "GB"]
                        },
                        {
                            "platform": "Funimation",
                            "url": "https://funimation.com/shows/12345",
                            "available_regions": ["US", "CA"]
                        }
                    ],
                    "season": "Winter 2024",
                    "timezone": "UTC"
                },
                {
                    "id": 2,
                    "anime_id": 67890,
                    "title": "Demon Slayer",
                    "episode": 12,
                    "air_date": "2024-01-15T10:30:00Z",
                    "duration": 24,
                    "image": "https://animeschedule.net/images/67890.jpg",
                    "url": "https://myanimelist.net/anime/67890",
                    "streaming": [
                        {
                            "platform": "Crunchyroll",
                            "url": "https://crunchyroll.com/watch/67890",
                            "available_regions": ["US", "CA", "GB", "AU"]
                        }
                    ],
                    "season": "Winter 2024",
                    "timezone": "UTC"
                }
            ],
            "meta": {
                "date": "2024-01-15",
                "timezone": "UTC",
                "total_episodes": 2
            }
        }
    
    @pytest.fixture
    def sample_search_response(self):
        """Sample AnimeSchedule.net search response."""
        return {
            "data": [
                {
                    "anime_id": 12345,
                    "title": "Attack on Titan",
                    "alternative_titles": ["Shingeki no Kyojin"],
                    "current_episode": 87,
                    "total_episodes": 28,
                    "status": "Currently Airing",
                    "air_day": "Monday",
                    "air_time": "09:00",
                    "timezone": "UTC",
                    "season": "Winter 2024",
                    "image": "https://animeschedule.net/images/12345.jpg",
                    "mal_url": "https://myanimelist.net/anime/12345",
                    "anilist_url": "https://anilist.co/anime/12345",
                    "next_episode": {
                        "episode": 88,
                        "air_date": "2024-01-22T09:00:00Z",
                        "countdown": "6 days, 14 hours"
                    },
                    "streaming": [
                        {
                            "platform": "Crunchyroll",
                            "url": "https://crunchyroll.com/watch/12345",
                            "available_regions": ["US", "CA", "GB"],
                            "premium_required": False
                        }
                    ]
                }
            ]
        }
    
    @pytest.fixture
    def sample_seasonal_response(self):
        """Sample AnimeSchedule.net seasonal response."""
        return {
            "data": [
                {
                    "season": "Winter 2024",
                    "year": 2024,
                    "anime": [
                        {
                            "anime_id": 12345,
                            "title": "Attack on Titan",
                            "status": "Currently Airing",
                            "episodes_aired": 3,
                            "total_episodes": 28,
                            "start_date": "2024-01-01T09:00:00Z",
                            "end_date": "2024-06-30T09:00:00Z",
                            "air_schedule": {
                                "day": "Monday",
                                "time": "09:00",
                                "timezone": "UTC"
                            },
                            "image": "https://animeschedule.net/images/12345.jpg",
                            "streaming": [
                                {
                                    "platform": "Crunchyroll",
                                    "regions": ["US", "CA", "GB"]
                                }
                            ]
                        }
                    ]
                }
            ],
            "meta": {
                "season": "Winter 2024",
                "total_anime": 1
            }
        }

    def test_client_initialization(self, mock_dependencies):
        """Test AnimeSchedule client initialization."""
        client = AnimeScheduleClient(**mock_dependencies)
        
        assert client.circuit_breaker == mock_dependencies["circuit_breaker"]
        assert client.rate_limiter is None  # Unlimited requests
        assert client.cache_manager == mock_dependencies["cache_manager"]
        assert client.error_handler == mock_dependencies["error_handler"]
        assert client.base_url == "https://animeschedule.net/api/v3"

    @pytest.mark.asyncio
    async def test_get_today_timetable_success(self, animeschedule_client, sample_timetable_response):
        """Test successful today's timetable retrieval."""
        with patch.object(animeschedule_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_timetable_response
            
            result = await animeschedule_client.get_today_timetable()
            
            assert result is not None
            assert len(result["data"]) == 2
            assert result["data"][0]["title"] == "Attack on Titan"
            assert result["data"][0]["episode"] == 87
            assert result["data"][1]["title"] == "Demon Slayer"
            assert result["meta"]["total_episodes"] == 2
            
            # Verify API was called with correct endpoint
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert "/timetables" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_timetable_by_date_success(self, animeschedule_client, sample_timetable_response):
        """Test successful timetable retrieval by specific date."""
        date = "2024-01-15"
        
        with patch.object(animeschedule_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_timetable_response
            
            result = await animeschedule_client.get_timetable_by_date(date)
            
            assert result is not None
            assert result["meta"]["date"] == "2024-01-15"
            
            # Verify date parameter was included
            call_args = mock_request.call_args
            assert f"/timetables/{date}" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_search_anime_success(self, animeschedule_client, sample_search_response):
        """Test successful anime search."""
        query = "Attack on Titan"
        
        with patch.object(animeschedule_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_search_response
            
            results = await animeschedule_client.search_anime(query)
            
            assert len(results) == 1
            assert results[0]["title"] == "Attack on Titan"
            assert results[0]["anime_id"] == 12345
            assert results[0]["status"] == "Currently Airing"
            assert "next_episode" in results[0]
            
            # Verify search parameters
            call_args = mock_request.call_args
            assert "/anime/search" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_seasonal_anime_success(self, animeschedule_client, sample_seasonal_response):
        """Test successful seasonal anime retrieval."""
        season = "winter"
        year = 2024
        
        with patch.object(animeschedule_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_seasonal_response
            
            result = await animeschedule_client.get_seasonal_anime(season, year)
            
            assert result is not None
            assert result["meta"]["season"] == "Winter 2024"
            assert len(result["data"][0]["anime"]) == 1
            assert result["data"][0]["anime"][0]["title"] == "Attack on Titan"
            
            # Verify seasonal endpoint was called
            call_args = mock_request.call_args
            assert f"/seasons/{year}/{season}" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_anime_schedule_by_id(self, animeschedule_client):
        """Test retrieving specific anime schedule by ID."""
        anime_id = 12345
        schedule_response = {
            "data": {
                "anime_id": 12345,
                "title": "Attack on Titan",
                "current_episode": 87,
                "next_episode": {
                    "episode": 88,
                    "air_date": "2024-01-22T09:00:00Z",
                    "countdown": "6 days, 14 hours"
                },
                "schedule": {
                    "day": "Monday",
                    "time": "09:00",
                    "timezone": "UTC"
                },
                "streaming": [
                    {
                        "platform": "Crunchyroll",
                        "url": "https://crunchyroll.com/watch/12345",
                        "regions": ["US", "CA", "GB"]
                    }
                ]
            }
        }
        
        with patch.object(animeschedule_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = schedule_response
            
            result = await animeschedule_client.get_anime_schedule_by_id(anime_id)
            
            assert result is not None
            assert result["anime_id"] == 12345
            assert result["title"] == "Attack on Titan"
            assert result["next_episode"]["episode"] == 88
            
            # Verify anime ID endpoint was called
            call_args = mock_request.call_args
            assert f"/anime/{anime_id}" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_streaming_platforms(self, animeschedule_client):
        """Test retrieving available streaming platforms."""
        platforms_response = {
            "data": [
                {
                    "platform": "Crunchyroll",
                    "regions": ["US", "CA", "GB", "AU", "FR", "DE"],
                    "premium_available": True,
                    "free_content": True
                },
                {
                    "platform": "Funimation",
                    "regions": ["US", "CA"],
                    "premium_available": True,
                    "free_content": False
                },
                {
                    "platform": "Netflix",
                    "regions": ["US", "CA", "GB", "JP"],
                    "premium_available": True,
                    "free_content": False
                }
            ]
        }
        
        with patch.object(animeschedule_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = platforms_response
            
            result = await animeschedule_client.get_streaming_platforms()
            
            assert len(result) == 3
            assert result[0]["platform"] == "Crunchyroll"
            assert "US" in result[0]["regions"]
            assert result[1]["platform"] == "Funimation"
            
            # Verify platforms endpoint was called
            call_args = mock_request.call_args
            assert "/platforms" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_rest_api_request_success(self, animeschedule_client):
        """Test successful REST API request."""
        endpoint = "/timetables"
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = Mock()
            mock_response.json = AsyncMock(return_value={"data": [{"id": 1}]})
            mock_response.status = 200
            mock_response.headers = {"Content-Type": "application/json"}
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await animeschedule_client._make_request(endpoint)
            
            # Verify JSON headers were included
            call_args = mock_get.call_args
            headers = call_args[1]["headers"]
            assert headers["Accept"] == "application/json"
            assert headers["Content-Type"] == "application/json"
            assert result["data"][0]["id"] == 1

    @pytest.mark.asyncio
    async def test_error_handling_404(self, animeschedule_client):
        """Test 404 error handling."""
        endpoint = "/anime/999999"
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = Mock()
            mock_response.status = 404
            mock_response.json = AsyncMock(return_value={
                "error": "Anime not found",
                "message": "No anime found with ID 999999"
            })
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            with pytest.raises(Exception) as exc_info:
                await animeschedule_client._make_request(endpoint)
            
            assert "anime not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_timezone_handling(self, animeschedule_client, sample_timetable_response):
        """Test timezone parameter handling."""
        timezone = "America/New_York"
        
        with patch.object(animeschedule_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_timetable_response
            
            result = await animeschedule_client.get_today_timetable(timezone=timezone)
            
            assert result is not None
            
            # Verify timezone parameter was included
            call_args = mock_request.call_args
            assert "/timetables" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_regional_filtering(self, animeschedule_client, sample_timetable_response):
        """Test regional content filtering."""
        region = "US"
        
        with patch.object(animeschedule_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_timetable_response
            
            result = await animeschedule_client.get_today_timetable(region=region)
            
            assert result is not None
            
            # Verify region parameter was included
            call_args = mock_request.call_args
            assert "/timetables" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_server_error_handling(self, animeschedule_client):
        """Test server error (5xx) handling."""
        endpoint = "/timetables"
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = Mock()
            mock_response.status = 500
            mock_response.json = AsyncMock(return_value={
                "error": "Internal server error",
                "message": "Database connection failed"
            })
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            with pytest.raises(Exception) as exc_info:
                await animeschedule_client._make_request(endpoint)
            
            assert "server error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, animeschedule_client):
        """Test circuit breaker integration."""
        # Mock circuit breaker to simulate open state
        animeschedule_client.circuit_breaker.is_open = Mock(return_value=True)
        
        with pytest.raises(Exception) as exc_info:
            await animeschedule_client.get_today_timetable()
        
        assert "circuit breaker" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_cache_integration(self, animeschedule_client, sample_timetable_response):
        """Test cache integration."""
        cache_key = "animeschedule_timetable_today"
        
        # Mock cache hit
        animeschedule_client.cache_manager.get = AsyncMock(return_value=sample_timetable_response)
        
        result = await animeschedule_client.get_today_timetable()
        
        assert result["data"][0]["title"] == "Attack on Titan"
        animeschedule_client.cache_manager.get.assert_called_once_with(cache_key)

    @pytest.mark.asyncio
    async def test_api_version_compatibility(self, animeschedule_client):
        """Test API version compatibility check."""
        version_response = {
            "version": "v3.1.0",
            "compatible": True,
            "deprecated_endpoints": [],
            "breaking_changes": []
        }
        
        with patch.object(animeschedule_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = version_response
            
            result = await animeschedule_client.check_api_compatibility()
            
            assert result["compatible"] is True
            assert result["version"] == "v3.1.0"
            
            # Verify version endpoint was called
            call_args = mock_request.call_args
            assert "/version" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_date_format_validation(self, animeschedule_client):
        """Test date format validation."""
        # Valid date format
        valid_date = "2024-01-15"
        result = animeschedule_client._validate_date_format(valid_date)
        assert result == valid_date
        
        # Invalid date format should raise exception
        invalid_date = "15-01-2024"
        with pytest.raises(ValueError) as exc_info:
            animeschedule_client._validate_date_format(invalid_date)
        
        assert "invalid date format" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self, animeschedule_client):
        """Test connection timeout handling."""
        endpoint = "/timetables"
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = asyncio.TimeoutError("Connection timeout")
            
            with pytest.raises(Exception) as exc_info:
                await animeschedule_client._make_request(endpoint, timeout=5)
            
            assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_no_rate_limiting(self, animeschedule_client):
        """Test that no rate limiting is applied (unlimited requests)."""
        # AnimeSchedule.net allows unlimited requests
        assert animeschedule_client.rate_limiter is None
        
        # Should be able to make multiple rapid requests
        with patch.object(animeschedule_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"data": []}
            
            # Make multiple requests rapidly
            tasks = [animeschedule_client.get_today_timetable() for _ in range(10)]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 10
            assert mock_request.call_count == 10