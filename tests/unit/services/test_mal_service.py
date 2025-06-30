"""Tests for MAL service integration."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any, List

from src.services.external.mal_service import MALService


class TestMALService:
    """Test cases for MAL service."""
    
    @pytest.fixture
    def mal_service(self):
        """Create MAL service for testing."""
        with patch('src.services.external.mal_service.MALClient') as mock_client_class, \
             patch('src.services.external.base_service.CircuitBreaker') as mock_cb_class:
            
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            mock_circuit_breaker = Mock()
            mock_circuit_breaker.is_open = Mock(return_value=False)
            mock_cb_class.return_value = mock_circuit_breaker
            
            service = MALService()
            service.client = mock_client
            return service
    
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
            "score": 8.7,
            "scored_by": 500000,
            "images": {
                "jpg": {
                    "image_url": "https://example.com/image.jpg"
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_search_anime_success(self, mal_service, sample_anime_data):
        """Test successful anime search."""
        # Setup
        query = "one piece"
        limit = 10
        expected_results = [sample_anime_data]
        
        mal_service.client.search_anime.return_value = expected_results
        
        # Execute
        results = await mal_service.search_anime(query, limit)
        
        # Verify
        assert results == expected_results
        mal_service.client.search_anime.assert_called_once_with(
            query=query, limit=limit, status=None, genres=None
        )
    
    @pytest.mark.asyncio
    async def test_search_anime_with_filters(self, mal_service, sample_anime_data):
        """Test anime search with additional filters."""
        # Setup
        query = "action"
        limit = 20
        status = "airing"
        genres = [1, 2]  # Action, Adventure
        expected_results = [sample_anime_data]
        
        mal_service.client.search_anime.return_value = expected_results
        
        # Execute
        results = await mal_service.search_anime(query, limit, status, genres)
        
        # Verify
        assert results == expected_results
        mal_service.client.search_anime.assert_called_once_with(
            query=query, limit=limit, status=status, genres=genres
        )
    
    @pytest.mark.asyncio
    async def test_search_anime_failure(self, mal_service):
        """Test anime search failure handling."""
        # Setup
        mal_service.client.search_anime.side_effect = Exception("API Error")
        
        # Execute & Verify
        with pytest.raises(Exception, match="API Error"):
            await mal_service.search_anime("test", 10)
    
    @pytest.mark.asyncio
    async def test_get_anime_details_success(self, mal_service, sample_anime_data):
        """Test successful anime details retrieval."""
        # Setup
        anime_id = 21
        mal_service.client.get_anime_by_id.return_value = sample_anime_data
        
        # Execute
        result = await mal_service.get_anime_details(anime_id)
        
        # Verify
        assert result == sample_anime_data
        mal_service.client.get_anime_by_id.assert_called_once_with(anime_id)
    
    @pytest.mark.asyncio
    async def test_get_anime_details_not_found(self, mal_service):
        """Test anime details when not found."""
        # Setup
        mal_service.client.get_anime_by_id.return_value = None
        
        # Execute
        result = await mal_service.get_anime_details(999)
        
        # Verify
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_seasonal_anime_success(self, mal_service, sample_anime_data):
        """Test successful seasonal anime retrieval."""
        # Setup
        year = 2024
        season = "winter"
        expected_results = [sample_anime_data]
        
        mal_service.client.get_seasonal_anime.return_value = expected_results
        
        # Execute
        results = await mal_service.get_seasonal_anime(year, season)
        
        # Verify
        assert results == expected_results
        mal_service.client.get_seasonal_anime.assert_called_once_with(year, season)
    
    @pytest.mark.asyncio
    async def test_get_seasonal_anime_invalid_season(self, mal_service):
        """Test seasonal anime with invalid season."""
        # Setup
        year = 2024
        season = "invalid"
        
        # Execute & Verify
        with pytest.raises(ValueError, match="Invalid season"):
            await mal_service.get_seasonal_anime(year, season)
    
    @pytest.mark.asyncio
    async def test_get_anime_statistics_success(self, mal_service):
        """Test successful anime statistics retrieval."""
        # Setup
        anime_id = 21
        stats_data = {
            "watching": 100000,
            "completed": 200000,
            "on_hold": 5000,
            "dropped": 3000,
            "plan_to_watch": 50000
        }
        
        mal_service.client.get_anime_statistics.return_value = stats_data
        
        # Execute
        result = await mal_service.get_anime_statistics(anime_id)
        
        # Verify
        assert result == stats_data
        mal_service.client.get_anime_statistics.assert_called_once_with(anime_id)
    
    @pytest.mark.asyncio
    async def test_get_current_season_success(self, mal_service, sample_anime_data):
        """Test successful current season anime retrieval."""
        # Setup
        expected_results = [sample_anime_data]
        
        mal_service.client.get_seasonal_anime.return_value = expected_results
        
        # Execute
        results = await mal_service.get_current_season()
        
        # Verify
        assert results == expected_results
        # Should call with current year and season
        mal_service.client.get_seasonal_anime.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mal_service):
        """Test health check when service is healthy."""
        # Setup
        mal_service.client.search_anime.return_value = [{"test": "data"}]
        mal_service.circuit_breaker.is_open.return_value = False
        
        # Execute
        result = await mal_service.health_check()
        
        # Verify
        assert result["service"] == "mal"
        assert result["status"] == "healthy"
        assert result["circuit_breaker_open"] == False
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, mal_service):
        """Test health check when service is unhealthy."""
        # Setup
        mal_service.client.search_anime.side_effect = Exception("Service down")
        mal_service.circuit_breaker.is_open.return_value = True
        
        # Execute
        result = await mal_service.health_check()
        
        # Verify
        assert result["service"] == "mal"
        assert result["status"] == "unhealthy"
        assert "error" in result
        assert result["circuit_breaker_open"] == True
    
    def test_service_initialization(self, mal_service):
        """Test service initialization."""
        assert mal_service.service_name == "mal"
        assert mal_service.cache_manager is not None
        assert mal_service.circuit_breaker is not None
        assert mal_service.client is not None
    
    def test_is_healthy(self, mal_service):
        """Test is_healthy method."""
        # Setup - healthy
        mal_service.circuit_breaker.is_open.return_value = False
        
        # Execute & Verify
        assert mal_service.is_healthy() == True
        
        # Setup - unhealthy
        mal_service.circuit_breaker.is_open.return_value = True
        
        # Execute & Verify
        assert mal_service.is_healthy() == False
    
    def test_get_service_info(self, mal_service):
        """Test get_service_info method."""
        # Setup
        mal_service.circuit_breaker.is_open.return_value = False
        
        # Execute
        info = mal_service.get_service_info()
        
        # Verify
        assert info["name"] == "mal"
        assert info["healthy"] == True
        assert info["circuit_breaker_open"] == False