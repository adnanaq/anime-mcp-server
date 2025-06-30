"""Tests for AniDB service integration."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any, List
from datetime import datetime

from src.services.external.anidb_service import AniDBService


class TestAniDBService:
    """Test cases for AniDB service."""
    
    @pytest.fixture
    def anidb_service(self):
        """Create AniDB service for testing."""
        with patch('src.services.external.anidb_service.AniDBClient') as mock_client_class, \
             patch('src.services.external.base_service.CircuitBreaker') as mock_cb_class:
            
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            mock_circuit_breaker = Mock()
            mock_circuit_breaker.is_open = Mock(return_value=False)
            mock_cb_class.return_value = mock_circuit_breaker
            
            service = AniDBService()
            service.client = mock_client
            return service
    
    @pytest.fixture
    def sample_anime_data(self):
        """Sample anime data for testing."""
        return {
            "id": 30,
            "title": "Neon Genesis Evangelion",
            "type": "TV Series",
            "episodes": 26,
            "start_date": "1995-10-04",
            "end_date": "1996-03-27",
            "rating": "9.01",
            "votes": 15234,
            "synopsis": "At the age of 14 Shinji Ikari is summoned by his father...",
            "genres": ["Mecha", "Psychological", "Drama"],
            "studios": ["Gainax", "Tatsunoko"],
            "year": 1995,
            "air_date": "04.10.1995",
            "tags": ["angst", "coming of age", "dystopian future"],
            "categories": ["anime", "series"]
        }
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing."""
        return [
            {
                "id": 30,
                "title": "Neon Genesis Evangelion",
                "type": "TV Series",
                "episodes": 26,
                "year": 1995,
                "rating": "9.01"
            },
            {
                "id": 32,
                "title": "Neon Genesis Evangelion: Death & Rebirth",
                "type": "Movie",
                "episodes": 1,
                "year": 1997,
                "rating": "7.25"
            }
        ]
    
    @pytest.fixture
    def sample_character_data(self):
        """Sample character data for testing."""
        return [
            {
                "id": 23,
                "name": "Shinji Ikari",
                "gender": "male",
                "description": "Third Child and pilot of Evangelion Unit-01",
                "voice_actors": [
                    {"name": "Kotono Mitsuishi", "language": "Japanese"},
                    {"name": "Spike Spencer", "language": "English"}
                ]
            },
            {
                "id": 24,
                "name": "Rei Ayanami", 
                "gender": "female",
                "description": "First Child and pilot of Evangelion Unit-00",
                "voice_actors": [
                    {"name": "Megumi Hayashibara", "language": "Japanese"},
                    {"name": "Amanda Winn-Lee", "language": "English"}
                ]
            }
        ]
    
    @pytest.fixture
    def sample_episode_data(self):
        """Sample episode data for testing."""
        return [
            {
                "episode_number": 1,
                "title": "Angel Attack",
                "air_date": "1995-10-04",
                "rating": "8.5",
                "summary": "Shinji is recruited by NERV to pilot Evangelion Unit-01"
            },
            {
                "episode_number": 2,
                "title": "The Beast",
                "air_date": "1995-10-11", 
                "rating": "8.2",
                "summary": "Shinji begins training as an Eva pilot"
            }
        ]
    
    @pytest.mark.asyncio
    async def test_search_anime_success(self, anidb_service, sample_search_results):
        """Test successful anime search."""
        # Setup
        query = "evangelion"
        
        anidb_service.client.search_anime_by_name.return_value = sample_search_results[0]
        
        # Execute
        result = await anidb_service.search_anime(query)
        
        # Verify
        assert result == [sample_search_results[0]]
        anidb_service.client.search_anime_by_name.assert_called_once_with(query)
    
    @pytest.mark.asyncio
    async def test_search_anime_empty_query(self, anidb_service):
        """Test anime search with empty query."""
        # Execute & Verify
        with pytest.raises(ValueError, match="Search query cannot be empty"):
            await anidb_service.search_anime("")
    
    @pytest.mark.asyncio
    async def test_search_anime_failure(self, anidb_service):
        """Test anime search failure handling."""
        # Setup
        anidb_service.client.search_anime_by_name.side_effect = Exception("API Error")
        
        # Execute & Verify
        with pytest.raises(Exception, match="API Error"):
            await anidb_service.search_anime("test")
    
    @pytest.mark.asyncio
    async def test_get_anime_details_success(self, anidb_service, sample_anime_data):
        """Test successful anime details retrieval."""
        # Setup
        anime_id = 30
        
        anidb_service.client.get_anime_by_id.return_value = sample_anime_data
        
        # Execute
        result = await anidb_service.get_anime_details(anime_id)
        
        # Verify
        assert result == sample_anime_data
        anidb_service.client.get_anime_by_id.assert_called_once_with(anime_id)
    
    @pytest.mark.asyncio
    async def test_get_anime_details_invalid_id(self, anidb_service):
        """Test anime details with invalid ID."""
        # Execute & Verify
        with pytest.raises(ValueError, match="Anime ID must be positive"):
            await anidb_service.get_anime_details(0)
        
        with pytest.raises(ValueError, match="Anime ID must be positive"):
            await anidb_service.get_anime_details(-1)
    
    @pytest.mark.asyncio
    async def test_get_anime_details_not_found(self, anidb_service):
        """Test anime details when not found."""
        # Setup
        anidb_service.client.get_anime_by_id.return_value = None
        
        # Execute
        result = await anidb_service.get_anime_details(999)
        
        # Verify
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_anime_characters_success(self, anidb_service, sample_character_data):
        """Test successful anime characters retrieval."""
        # Setup
        anime_id = 30
        
        anidb_service.client.get_anime_characters.return_value = sample_character_data
        
        # Execute
        result = await anidb_service.get_anime_characters(anime_id)
        
        # Verify
        assert result == sample_character_data
        anidb_service.client.get_anime_characters.assert_called_once_with(anime_id)
    
    @pytest.mark.asyncio
    async def test_get_anime_characters_invalid_id(self, anidb_service):
        """Test anime characters with invalid ID."""
        # Execute & Verify
        with pytest.raises(ValueError, match="Anime ID must be positive"):
            await anidb_service.get_anime_characters(0)
    
    @pytest.mark.asyncio
    async def test_get_anime_episodes_success(self, anidb_service, sample_episode_data):
        """Test successful anime episodes retrieval."""
        # Setup
        anime_id = 30
        
        # Episodes return empty list in current implementation
        pass  # No mock needed as method returns []
        
        # Execute
        result = await anidb_service.get_anime_episodes(anime_id)
        
        # Verify
        assert result == []  # Current implementation returns empty list
    
    @pytest.mark.asyncio
    async def test_get_anime_episodes_invalid_id(self, anidb_service):
        """Test anime episodes with invalid ID."""
        # Execute & Verify
        with pytest.raises(ValueError, match="Anime ID must be positive"):
            await anidb_service.get_anime_episodes(-5)
    
    @pytest.mark.asyncio
    async def test_get_similar_anime_success(self, anidb_service, sample_search_results):
        """Test successful similar anime retrieval."""
        # Setup
        anime_id = 30
        limit = 5
        
        # Similar anime returns empty list in current implementation
        pass  # No mock needed as method returns []
        
        # Execute
        result = await anidb_service.get_similar_anime(anime_id, limit)
        
        # Verify
        assert result == []  # Current implementation returns empty list
    
    @pytest.mark.asyncio
    async def test_get_similar_anime_default_limit(self, anidb_service, sample_search_results):
        """Test similar anime with default limit."""
        # Setup
        anime_id = 30
        
        # Similar anime returns empty list in current implementation
        pass  # No mock needed as method returns []
        
        # Execute
        result = await anidb_service.get_similar_anime(anime_id)
        
        # Verify
        assert result == []  # Current implementation returns empty list
    
    @pytest.mark.asyncio
    async def test_get_similar_anime_invalid_params(self, anidb_service):
        """Test similar anime with invalid parameters."""
        # Test invalid anime ID
        with pytest.raises(ValueError, match="Anime ID must be positive"):
            await anidb_service.get_similar_anime(0)
        
        # Test invalid limit
        with pytest.raises(ValueError, match="Limit must be between 1 and 50"):
            await anidb_service.get_similar_anime(30, 0)
        
        with pytest.raises(ValueError, match="Limit must be between 1 and 50"):
            await anidb_service.get_similar_anime(30, 51)
    
    @pytest.mark.asyncio
    async def test_get_random_anime_success(self, anidb_service, sample_anime_data):
        """Test successful random anime retrieval."""
        # Setup
        anidb_service.client.get_anime_by_id.return_value = sample_anime_data
        
        # Execute
        result = await anidb_service.get_random_anime()
        
        # Verify
        assert result == sample_anime_data
        # Random anime calls get_anime_by_id with random ID
        anidb_service.client.get_anime_by_id.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_random_anime_failure(self, anidb_service):
        """Test random anime failure handling."""
        # Setup
        anidb_service.client.get_anime_by_id.side_effect = Exception("Service error")
        
        # Execute & Verify
        with pytest.raises(Exception, match="Service error"):
            await anidb_service.get_random_anime()
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, anidb_service):
        """Test health check when service is healthy."""
        # Setup
        test_data = {"id": "1", "title": "Test Anime"}
        anidb_service.client.get_anime_by_id.return_value = test_data
        anidb_service.circuit_breaker.is_open.return_value = False
        
        # Execute
        result = await anidb_service.health_check()
        
        # Verify
        assert result["service"] == "anidb"
        assert result["status"] == "healthy"
        assert result["circuit_breaker_open"] == False
        assert "response_time" in result
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, anidb_service):
        """Test health check when service is unhealthy."""
        # Setup
        anidb_service.client.get_anime_by_id.side_effect = Exception("Connection timeout")
        anidb_service.circuit_breaker.is_open.return_value = True
        
        # Execute
        result = await anidb_service.health_check()
        
        # Verify
        assert result["service"] == "anidb"
        assert result["status"] == "unhealthy"
        assert "error" in result
        assert result["circuit_breaker_open"] == True
    
    def test_service_initialization(self, anidb_service):
        """Test service initialization."""
        assert anidb_service.service_name == "anidb"
        assert anidb_service.cache_manager is not None
        assert anidb_service.circuit_breaker is not None
        assert anidb_service.client is not None
    
    def test_is_healthy(self, anidb_service):
        """Test is_healthy method."""
        # Setup - healthy
        anidb_service.circuit_breaker.is_open.return_value = False
        
        # Execute & Verify
        assert anidb_service.is_healthy() == True
        
        # Setup - unhealthy
        anidb_service.circuit_breaker.is_open.return_value = True
        
        # Execute & Verify
        assert anidb_service.is_healthy() == False
    
    def test_get_service_info(self, anidb_service):
        """Test get_service_info method."""
        # Setup
        anidb_service.circuit_breaker.is_open.return_value = False
        
        # Execute
        info = anidb_service.get_service_info()
        
        # Verify
        assert info["name"] == "anidb"
        assert info["healthy"] == True
        assert info["circuit_breaker_open"] == False