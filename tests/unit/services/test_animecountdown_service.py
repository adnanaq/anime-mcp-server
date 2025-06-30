"""Tests for AnimeCountdown service integration."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any, List
from datetime import datetime

from src.services.external.animecountdown_service import AnimeCountdownService


class TestAnimeCountdownService:
    """Test cases for AnimeCountdown service."""
    
    @pytest.fixture
    def animecountdown_service(self):
        """Create AnimeCountdown service for testing."""
        with patch('src.services.external.animecountdown_service.AnimeCountdownScraper') as mock_scraper_class, \
             patch('src.services.external.base_service.CircuitBreaker') as mock_cb_class:
            
            mock_scraper = AsyncMock()
            mock_scraper_class.return_value = mock_scraper
            
            mock_circuit_breaker = Mock()
            mock_circuit_breaker.is_open = Mock(return_value=False)
            mock_cb_class.return_value = mock_circuit_breaker
            
            service = AnimeCountdownService()
            service.scraper = mock_scraper
            return service
    
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
        """Sample currently airing anime data for testing."""
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
            },
            {
                "id": "jujutsu-kaisen-s2",
                "title": "Jujutsu Kaisen Season 2",
                "url": "https://animecountdown.com/jujutsu-kaisen-s2",
                "type": "TV",
                "year": 2023,
                "season": "Summer",
                "studio": "MAPPA",
                "status": "airing",
                "air_date": "2023-07-06",
                "countdown_info": {
                    "next_episode": 12,
                    "time_until_next": "5 days, 2 hours",
                    "is_airing": True,
                    "next_air_time": "2023-06-01T16:30:00Z"
                }
            }
        ]
    
    @pytest.fixture
    def sample_upcoming_data(self):
        """Sample upcoming anime data for testing."""
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
            },
            {
                "id": "attack-on-titan-movie",
                "title": "Attack on Titan: The Last Attack",
                "url": "https://animecountdown.com/aot-movie",
                "type": "Movie",
                "year": 2024,
                "studio": "Wit Studio",
                "status": "upcoming",
                "countdown_info": {
                    "premiere_date": "2024-11-08",
                    "time_until_premiere": "140 days",
                    "is_airing": False
                }
            }
        ]
    
    @pytest.fixture
    def sample_popular_data(self):
        """Sample popular anime data for testing."""
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
            },
            {
                "id": "spirited-away",
                "title": "Spirited Away",
                "url": "https://animecountdown.com/spirited-away",
                "type": "Movie",
                "year": 2001,
                "studio": "Studio Ghibli",
                "status": "finished",
                "popularity_rank": 2,
                "countdown_views": 980000
            }
        ]
    
    @pytest.fixture
    def sample_countdown_data(self):
        """Sample countdown data for testing."""
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
    
    @pytest.mark.asyncio
    async def test_search_anime_success(self, animecountdown_service, sample_search_results):
        """Test successful anime search."""
        # Setup
        query = "cowboy bebop"
        
        # Mock the scraper to have search_anime method
        animecountdown_service.scraper.search_anime = AsyncMock(return_value=sample_search_results)
        
        # Execute
        result = await animecountdown_service.search_anime(query)
        
        # Verify
        assert result == sample_search_results
        animecountdown_service.scraper.search_anime.assert_called_once_with(query)
    
    @pytest.mark.asyncio
    async def test_search_anime_empty_query(self, animecountdown_service):
        """Test anime search with empty query."""
        # Execute & Verify
        with pytest.raises(ValueError, match="Search query cannot be empty"):
            await animecountdown_service.search_anime("")
    
    @pytest.mark.asyncio
    async def test_search_anime_failure(self, animecountdown_service):
        """Test anime search failure handling."""
        # Setup
        animecountdown_service.scraper.search_anime.side_effect = Exception("Scraping error")
        
        # Execute & Verify
        with pytest.raises(Exception, match="Scraping error"):
            await animecountdown_service.search_anime("test")
    
    @pytest.mark.asyncio
    async def test_get_anime_details_success(self, animecountdown_service, sample_anime_data):
        """Test successful anime details retrieval."""
        # Setup
        anime_id = "cowboy-bebop"
        
        animecountdown_service.scraper.get_anime_by_slug.return_value = sample_anime_data
        
        # Execute
        result = await animecountdown_service.get_anime_details(anime_id)
        
        # Verify
        assert result == sample_anime_data
        animecountdown_service.scraper.get_anime_by_slug.assert_called_once_with(anime_id)
    
    @pytest.mark.asyncio
    async def test_get_anime_details_empty_id(self, animecountdown_service):
        """Test anime details with empty ID."""
        # Execute & Verify
        with pytest.raises(ValueError, match="Anime ID cannot be empty"):
            await animecountdown_service.get_anime_details("")
    
    @pytest.mark.asyncio
    async def test_get_anime_details_not_found(self, animecountdown_service):
        """Test anime details when not found."""
        # Setup
        animecountdown_service.scraper.get_anime_by_slug.return_value = None
        
        # Execute
        result = await animecountdown_service.get_anime_details("nonexistent-anime")
        
        # Verify
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_currently_airing_success(self, animecountdown_service, sample_airing_data):
        """Test successful currently airing anime retrieval."""
        # Setup
        limit = 15
        
        animecountdown_service.scraper.get_currently_airing.return_value = sample_airing_data
        
        # Execute
        result = await animecountdown_service.get_currently_airing(limit)
        
        # Verify
        assert result == sample_airing_data
        animecountdown_service.scraper.get_currently_airing.assert_called_once_with(limit)
    
    @pytest.mark.asyncio
    async def test_get_currently_airing_default_limit(self, animecountdown_service, sample_airing_data):
        """Test currently airing with default limit."""
        # Setup
        animecountdown_service.scraper.get_currently_airing.return_value = sample_airing_data
        
        # Execute
        result = await animecountdown_service.get_currently_airing()
        
        # Verify
        assert result == sample_airing_data
        animecountdown_service.scraper.get_currently_airing.assert_called_once_with(25)  # Default limit
    
    @pytest.mark.asyncio
    async def test_get_currently_airing_invalid_limit(self, animecountdown_service):
        """Test currently airing with invalid limit."""
        # Execute & Verify
        with pytest.raises(ValueError, match="Limit must be between 1 and 100"):
            await animecountdown_service.get_currently_airing(0)
        
        with pytest.raises(ValueError, match="Limit must be between 1 and 100"):
            await animecountdown_service.get_currently_airing(101)
    
    @pytest.mark.asyncio
    async def test_get_upcoming_anime_success(self, animecountdown_service, sample_upcoming_data):
        """Test successful upcoming anime retrieval."""
        # Setup
        limit = 10
        
        animecountdown_service.scraper.get_upcoming_anime.return_value = sample_upcoming_data
        
        # Execute
        result = await animecountdown_service.get_upcoming_anime(limit)
        
        # Verify
        assert result == sample_upcoming_data
        animecountdown_service.scraper.get_upcoming_anime.assert_called_once_with(limit)
    
    @pytest.mark.asyncio
    async def test_get_upcoming_anime_default_limit(self, animecountdown_service, sample_upcoming_data):
        """Test upcoming anime with default limit."""
        # Setup
        animecountdown_service.scraper.get_upcoming_anime.return_value = sample_upcoming_data
        
        # Execute
        result = await animecountdown_service.get_upcoming_anime()
        
        # Verify
        assert result == sample_upcoming_data
        animecountdown_service.scraper.get_upcoming_anime.assert_called_once_with(20)  # Default limit
    
    @pytest.mark.asyncio
    async def test_get_upcoming_anime_invalid_limit(self, animecountdown_service):
        """Test upcoming anime with invalid limit."""
        # Execute & Verify
        with pytest.raises(ValueError, match="Limit must be between 1 and 100"):
            await animecountdown_service.get_upcoming_anime(0)
        
        with pytest.raises(ValueError, match="Limit must be between 1 and 100"):
            await animecountdown_service.get_upcoming_anime(101)
    
    @pytest.mark.asyncio
    async def test_get_popular_anime_success(self, animecountdown_service, sample_popular_data):
        """Test successful popular anime retrieval."""
        # Setup
        time_period = "this_month"
        limit = 15
        
        animecountdown_service.scraper.get_popular_anime.return_value = sample_popular_data
        
        # Execute
        result = await animecountdown_service.get_popular_anime(time_period, limit)
        
        # Verify
        assert result == sample_popular_data
        animecountdown_service.scraper.get_popular_anime.assert_called_once_with(time_period, limit)
    
    @pytest.mark.asyncio
    async def test_get_popular_anime_default_params(self, animecountdown_service, sample_popular_data):
        """Test popular anime with default parameters."""
        # Setup
        animecountdown_service.scraper.get_popular_anime.return_value = sample_popular_data
        
        # Execute
        result = await animecountdown_service.get_popular_anime()
        
        # Verify
        assert result == sample_popular_data
        animecountdown_service.scraper.get_popular_anime.assert_called_once_with("all_time", 25)
    
    @pytest.mark.asyncio
    async def test_get_popular_anime_invalid_params(self, animecountdown_service):
        """Test popular anime with invalid parameters."""
        # Test invalid time period
        with pytest.raises(ValueError, match="Invalid time period"):
            await animecountdown_service.get_popular_anime("invalid_period")
        
        # Test invalid limit
        with pytest.raises(ValueError, match="Limit must be between 1 and 100"):
            await animecountdown_service.get_popular_anime("all_time", 0)
        
        with pytest.raises(ValueError, match="Limit must be between 1 and 100"):
            await animecountdown_service.get_popular_anime("all_time", 101)
    
    @pytest.mark.asyncio
    async def test_get_anime_countdown_success(self, animecountdown_service, sample_countdown_data):
        """Test successful anime countdown retrieval."""
        # Setup
        anime_id = "demon-slayer-s3"
        
        animecountdown_service.scraper.get_anime_countdown.return_value = sample_countdown_data
        
        # Execute
        result = await animecountdown_service.get_anime_countdown(anime_id)
        
        # Verify
        assert result == sample_countdown_data
        animecountdown_service.scraper.get_anime_countdown.assert_called_once_with(anime_id)
    
    @pytest.mark.asyncio
    async def test_get_anime_countdown_empty_id(self, animecountdown_service):
        """Test anime countdown with empty ID."""
        # Execute & Verify
        with pytest.raises(ValueError, match="Anime ID cannot be empty"):
            await animecountdown_service.get_anime_countdown("")
    
    @pytest.mark.asyncio
    async def test_get_anime_countdown_not_found(self, animecountdown_service):
        """Test anime countdown when not found."""
        # Setup
        animecountdown_service.scraper.get_anime_countdown.return_value = None
        
        # Execute
        result = await animecountdown_service.get_anime_countdown("nonexistent-anime")
        
        # Verify
        assert result is None
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, animecountdown_service):
        """Test health check when service is healthy."""
        # Setup
        animecountdown_service.scraper.base_url = "https://animecountdown.com"
        animecountdown_service.circuit_breaker.is_open.return_value = False
        
        # Execute
        result = await animecountdown_service.health_check()
        
        # Verify
        assert result["service"] == "animecountdown"
        assert result["status"] == "healthy"
        assert result["circuit_breaker_open"] == False
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, animecountdown_service):
        """Test health check when service is unhealthy."""
        # Setup
        animecountdown_service.scraper = None
        animecountdown_service.circuit_breaker.is_open.return_value = True
        
        # Execute
        result = await animecountdown_service.health_check()
        
        # Verify
        assert result["service"] == "animecountdown"
        assert result["status"] == "unhealthy"
        assert "error" in result
        assert result["circuit_breaker_open"] == True
    
    def test_service_initialization(self, animecountdown_service):
        """Test service initialization."""
        assert animecountdown_service.service_name == "animecountdown"
        assert animecountdown_service.cache_manager is not None
        assert animecountdown_service.circuit_breaker is not None
        assert animecountdown_service.scraper is not None
    
    def test_is_healthy(self, animecountdown_service):
        """Test is_healthy method."""
        # Setup - healthy
        animecountdown_service.circuit_breaker.is_open.return_value = False
        
        # Execute & Verify
        assert animecountdown_service.is_healthy() == True
        
        # Setup - unhealthy
        animecountdown_service.circuit_breaker.is_open.return_value = True
        
        # Execute & Verify
        assert animecountdown_service.is_healthy() == False
    
    def test_get_service_info(self, animecountdown_service):
        """Test get_service_info method."""
        # Setup
        animecountdown_service.circuit_breaker.is_open.return_value = False
        
        # Execute
        info = animecountdown_service.get_service_info()
        
        # Verify
        assert info["name"] == "animecountdown"
        assert info["healthy"] == True
        assert info["circuit_breaker_open"] == False
    
    @pytest.mark.asyncio 
    async def test_scraper_has_required_methods(self, animecountdown_service):
        """Test that scraper has all required methods for service compatibility."""
        scraper = animecountdown_service.scraper
        
        # Generic methods that all scrapers should have
        required_methods = [
            'search_anime',
            'get_anime_by_slug', 
            'get_anime_countdown',
            'get_currently_airing',
            'get_upcoming_anime',
            'get_popular_anime'
        ]
        
        for method_name in required_methods:
            assert hasattr(scraper, method_name), f"Scraper missing required method: {method_name}"
            
        # Specialized AnimeCountdown methods
        specialized_methods = [
            'search_anime_countdowns',
            'get_anime_countdown_by_slug',
            'get_episode_countdowns'
        ]
        
        for method_name in specialized_methods:
            assert hasattr(scraper, method_name), f"Scraper missing specialized method: {method_name}"