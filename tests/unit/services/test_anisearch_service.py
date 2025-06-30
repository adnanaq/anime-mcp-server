"""Tests for AniSearch service integration."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any, List
from datetime import datetime

from src.services.external.anisearch_service import AniSearchService


class TestAniSearchService:
    """Test cases for AniSearch service."""
    
    @pytest.fixture
    def anisearch_service(self):
        """Create AniSearch service for testing."""
        with patch('src.services.external.anisearch_service.AniSearchScraper') as mock_scraper_class, \
             patch('src.services.external.base_service.CircuitBreaker') as mock_cb_class:
            
            mock_scraper = AsyncMock()
            mock_scraper_class.return_value = mock_scraper
            
            mock_circuit_breaker = Mock()
            mock_circuit_breaker.is_open = Mock(return_value=False)
            mock_cb_class.return_value = mock_circuit_breaker
            
            service = AniSearchService()
            service.scraper = mock_scraper
            return service
    
    @pytest.fixture
    def sample_anime_data(self):
        """Sample anime data for testing."""
        return {
            "id": "cowboy-bebop",
            "title": "Cowboy Bebop",
            "url": "https://anisearch.com/anime/25",
            "type": "TV Series",
            "episodes": 26,
            "year": 1998,
            "season": "Spring",
            "studio": "Sunrise",
            "rating": 8.9,
            "status": "finished",
            "synopsis": "In the year 2071, humanity has colonized several of the planets...",
            "genres": ["Action", "Space Western", "Drama", "Sci-Fi"],
            "tags": ["Bounty Hunters", "Space", "Adult Cast", "Jazz"],
            "alternative_titles": [
                "Space Warriors",
                "カウボーイビバップ"
            ],
            "image_url": "https://anisearch.com/images/anime/cowboy-bebop.jpg",
            "start_date": "1998-04-03",
            "end_date": "1999-04-24"
        }
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing."""
        return [
            {
                "id": "cowboy-bebop",
                "title": "Cowboy Bebop",
                "url": "https://anisearch.com/anime/25",
                "type": "TV Series",
                "year": 1998,
                "rating": 8.9,
                "image_url": "https://anisearch.com/images/anime/cowboy-bebop.jpg",
                "status": "finished"
            },
            {
                "id": "space-dandy",
                "title": "Space Dandy",
                "url": "https://anisearch.com/anime/9735",
                "type": "TV Series",
                "year": 2014,
                "rating": 7.5,
                "image_url": "https://anisearch.com/images/anime/space-dandy.jpg",
                "status": "finished"
            }
        ]
    
    @pytest.fixture
    def sample_character_data(self):
        """Sample character data for testing."""
        return [
            {
                "id": "spike-spiegel",
                "name": "Spike Spiegel",
                "url": "https://anisearch.com/character/157",
                "image_url": "https://anisearch.com/images/character/spike-spiegel.jpg",
                "description": "A bounty hunter traveling on the spaceship Bebop",
                "voice_actors": [
                    {"name": "Koichi Yamadera", "language": "Japanese"},
                    {"name": "Steve Blum", "language": "English"}
                ],
                "age": 27,
                "height": "185 cm",
                "weight": "70 kg"
            },
            {
                "id": "faye-valentine",
                "name": "Faye Valentine",
                "url": "https://anisearch.com/character/158",
                "image_url": "https://anisearch.com/images/character/faye-valentine.jpg",
                "description": "A bounty hunter and former con artist",
                "voice_actors": [
                    {"name": "Megumi Hayashibara", "language": "Japanese"},
                    {"name": "Wendee Lee", "language": "English"}
                ],
                "age": 77,
                "height": "168 cm",
                "weight": "49 kg"
            }
        ]
    
    @pytest.fixture
    def sample_top_anime_data(self):
        """Sample top anime data for testing."""
        return [
            {
                "rank": 1,
                "id": "fullmetal-alchemist-brotherhood",
                "title": "Fullmetal Alchemist: Brotherhood",
                "url": "https://anisearch.com/anime/5391",
                "rating": 9.4,
                "votes": 87000,
                "year": 2009,
                "type": "TV Series"
            },
            {
                "rank": 2,
                "id": "spirited-away",
                "title": "Spirited Away",
                "url": "https://anisearch.com/anime/424",
                "rating": 9.3,
                "votes": 65000,
                "year": 2001,
                "type": "Movie"
            }
        ]
    
    @pytest.fixture
    def sample_seasonal_data(self):
        """Sample seasonal anime data for testing."""
        return [
            {
                "id": "attack-on-titan-s4",
                "title": "Attack on Titan: The Final Season",
                "url": "https://anisearch.com/anime/14977",
                "type": "TV Series",
                "year": 2020,
                "season": "Fall",
                "studio": "Wit Studio",
                "rating": 9.0,
                "status": "finished",
                "start_date": "2020-12-07"
            },
            {
                "id": "jujutsu-kaisen",
                "title": "Jujutsu Kaisen",
                "url": "https://anisearch.com/anime/14403",
                "type": "TV Series",
                "year": 2020,
                "season": "Fall",
                "studio": "MAPPA",
                "rating": 8.9,
                "status": "finished",
                "start_date": "2020-10-03"
            }
        ]
    
    @pytest.fixture
    def sample_recommendations_data(self):
        """Sample recommendations data for testing."""
        return [
            {
                "id": "trigun",
                "title": "Trigun",
                "url": "https://anisearch.com/anime/53",
                "similarity_score": 0.85,
                "reason": "Similar space western themes and adult protagonists",
                "rating": 8.3
            },
            {
                "id": "samurai-champloo",
                "title": "Samurai Champloo",
                "url": "https://anisearch.com/anime/631",
                "similarity_score": 0.80,
                "reason": "Same director and similar episodic structure",
                "rating": 8.5
            }
        ]
    
    @pytest.mark.asyncio
    async def test_search_anime_success(self, anisearch_service, sample_search_results):
        """Test successful anime search."""
        # Setup
        query = "cowboy bebop"
        
        anisearch_service.scraper.search_anime.return_value = sample_search_results
        
        # Execute
        result = await anisearch_service.search_anime(query)
        
        # Verify
        assert result == sample_search_results
        anisearch_service.scraper.search_anime.assert_called_once_with(query)
    
    @pytest.mark.asyncio
    async def test_search_anime_empty_query(self, anisearch_service):
        """Test anime search with empty query."""
        # Execute & Verify
        with pytest.raises(ValueError, match="Search query cannot be empty"):
            await anisearch_service.search_anime("")
    
    @pytest.mark.asyncio
    async def test_search_anime_failure(self, anisearch_service):
        """Test anime search failure handling."""
        # Setup
        anisearch_service.scraper.search_anime.side_effect = Exception("Scraping error")
        
        # Execute & Verify
        with pytest.raises(Exception, match="Scraping error"):
            await anisearch_service.search_anime("test")
    
    @pytest.mark.asyncio
    async def test_get_anime_details_success(self, anisearch_service, sample_anime_data):
        """Test successful anime details retrieval."""
        # Setup
        anime_id = "cowboy-bebop"
        
        anisearch_service.scraper.get_anime_by_slug.return_value = sample_anime_data
        
        # Execute
        result = await anisearch_service.get_anime_details(anime_id)
        
        # Verify
        assert result == sample_anime_data
        anisearch_service.scraper.get_anime_by_slug.assert_called_once_with(anime_id)
    
    @pytest.mark.asyncio
    async def test_get_anime_details_empty_id(self, anisearch_service):
        """Test anime details with empty ID."""
        # Execute & Verify
        with pytest.raises(ValueError, match="Anime ID cannot be empty"):
            await anisearch_service.get_anime_details("")
    
    @pytest.mark.asyncio
    async def test_get_anime_details_not_found(self, anisearch_service):
        """Test anime details when not found."""
        # Setup
        anisearch_service.scraper.get_anime_by_slug.return_value = None
        
        # Execute
        result = await anisearch_service.get_anime_details("nonexistent-anime")
        
        # Verify
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_anime_characters_success(self, anisearch_service, sample_character_data):
        """Test successful anime characters retrieval."""
        # Setup
        anime_id = "cowboy-bebop"
        
        anisearch_service.scraper.get_anime_characters.return_value = sample_character_data
        
        # Execute
        result = await anisearch_service.get_anime_characters(anime_id)
        
        # Verify
        assert result == sample_character_data
        anisearch_service.scraper.get_anime_characters.assert_called_once_with(anime_id)
    
    @pytest.mark.asyncio
    async def test_get_anime_characters_empty_id(self, anisearch_service):
        """Test anime characters with empty ID."""
        # Execute & Verify
        with pytest.raises(ValueError, match="Anime ID cannot be empty"):
            await anisearch_service.get_anime_characters("")
    
    @pytest.mark.asyncio
    async def test_get_anime_recommendations_success(self, anisearch_service, sample_recommendations_data):
        """Test successful anime recommendations retrieval."""
        # Setup
        anime_id = "cowboy-bebop"
        limit = 5
        
        anisearch_service.scraper.get_anime_recommendations.return_value = sample_recommendations_data
        
        # Execute
        result = await anisearch_service.get_anime_recommendations(anime_id, limit)
        
        # Verify
        assert result == sample_recommendations_data
        anisearch_service.scraper.get_anime_recommendations.assert_called_once_with(anime_id, limit)
    
    @pytest.mark.asyncio
    async def test_get_anime_recommendations_default_limit(self, anisearch_service, sample_recommendations_data):
        """Test anime recommendations with default limit."""
        # Setup
        anime_id = "cowboy-bebop"
        
        anisearch_service.scraper.get_anime_recommendations.return_value = sample_recommendations_data
        
        # Execute
        result = await anisearch_service.get_anime_recommendations(anime_id)
        
        # Verify
        assert result == sample_recommendations_data
        anisearch_service.scraper.get_anime_recommendations.assert_called_once_with(anime_id, 10)
    
    @pytest.mark.asyncio
    async def test_get_anime_recommendations_invalid_params(self, anisearch_service):
        """Test anime recommendations with invalid parameters."""
        # Test empty anime ID
        with pytest.raises(ValueError, match="Anime ID cannot be empty"):
            await anisearch_service.get_anime_recommendations("")
        
        # Test invalid limit
        with pytest.raises(ValueError, match="Limit must be between 1 and 50"):
            await anisearch_service.get_anime_recommendations("test", 0)
        
        with pytest.raises(ValueError, match="Limit must be between 1 and 50"):
            await anisearch_service.get_anime_recommendations("test", 51)
    
    @pytest.mark.asyncio
    async def test_get_top_anime_success(self, anisearch_service, sample_top_anime_data):
        """Test successful top anime retrieval."""
        # Setup
        category = "highest_rated"
        limit = 20
        
        anisearch_service.scraper.get_top_anime.return_value = sample_top_anime_data
        
        # Execute
        result = await anisearch_service.get_top_anime(category, limit)
        
        # Verify
        assert result == sample_top_anime_data
        anisearch_service.scraper.get_top_anime.assert_called_once_with(category, limit)
    
    @pytest.mark.asyncio
    async def test_get_top_anime_default_params(self, anisearch_service, sample_top_anime_data):
        """Test top anime with default parameters."""
        # Setup
        anisearch_service.scraper.get_top_anime.return_value = sample_top_anime_data
        
        # Execute
        result = await anisearch_service.get_top_anime()
        
        # Verify
        assert result == sample_top_anime_data
        anisearch_service.scraper.get_top_anime.assert_called_once_with("highest_rated", 25)
    
    @pytest.mark.asyncio
    async def test_get_top_anime_invalid_params(self, anisearch_service):
        """Test top anime with invalid parameters."""
        # Test invalid category
        with pytest.raises(ValueError, match="Invalid category"):
            await anisearch_service.get_top_anime("invalid_category")
        
        # Test invalid limit
        with pytest.raises(ValueError, match="Limit must be between 1 and 100"):
            await anisearch_service.get_top_anime("highest_rated", 0)
        
        with pytest.raises(ValueError, match="Limit must be between 1 and 100"):
            await anisearch_service.get_top_anime("highest_rated", 101)
    
    @pytest.mark.asyncio
    async def test_get_seasonal_anime_success(self, anisearch_service, sample_seasonal_data):
        """Test successful seasonal anime retrieval."""
        # Setup
        year = 2020
        season = "fall"
        limit = 20
        
        anisearch_service.scraper.get_seasonal_anime.return_value = sample_seasonal_data
        
        # Execute
        result = await anisearch_service.get_seasonal_anime(year, season, limit)
        
        # Verify
        assert result == sample_seasonal_data
        anisearch_service.scraper.get_seasonal_anime.assert_called_once_with(year, season, limit)
    
    @pytest.mark.asyncio
    async def test_get_seasonal_anime_default_params(self, anisearch_service, sample_seasonal_data):
        """Test seasonal anime with default parameters."""
        # Setup
        current_year = datetime.now().year
        anisearch_service.scraper.get_seasonal_anime.return_value = sample_seasonal_data
        
        # Execute
        result = await anisearch_service.get_seasonal_anime()
        
        # Verify
        assert result == sample_seasonal_data
        # Should use current year and current season
        anisearch_service.scraper.get_seasonal_anime.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_seasonal_anime_invalid_params(self, anisearch_service):
        """Test seasonal anime with invalid parameters."""
        # Test invalid year
        with pytest.raises(ValueError, match="Year must be between 1900 and"):
            await anisearch_service.get_seasonal_anime(1800)
        
        # Test invalid season
        with pytest.raises(ValueError, match="Invalid season"):
            await anisearch_service.get_seasonal_anime(2023, "invalid_season")
        
        # Test invalid limit
        with pytest.raises(ValueError, match="Limit must be between 1 and 100"):
            await anisearch_service.get_seasonal_anime(2023, "spring", 0)
        
        with pytest.raises(ValueError, match="Limit must be between 1 and 100"):
            await anisearch_service.get_seasonal_anime(2023, "spring", 101)
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, anisearch_service):
        """Test health check when service is healthy."""
        # Setup
        anisearch_service.scraper.base_url = "https://anisearch.com"
        anisearch_service.circuit_breaker.is_open.return_value = False
        
        # Execute
        result = await anisearch_service.health_check()
        
        # Verify
        assert result["service"] == "anisearch"
        assert result["status"] == "healthy"
        assert result["circuit_breaker_open"] == False
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, anisearch_service):
        """Test health check when service is unhealthy."""
        # Setup
        anisearch_service.scraper = None
        anisearch_service.circuit_breaker.is_open.return_value = True
        
        # Execute
        result = await anisearch_service.health_check()
        
        # Verify
        assert result["service"] == "anisearch"
        assert result["status"] == "unhealthy"
        assert "error" in result
        assert result["circuit_breaker_open"] == True
    
    def test_service_initialization(self, anisearch_service):
        """Test service initialization."""
        assert anisearch_service.service_name == "anisearch"
        assert anisearch_service.cache_manager is not None
        assert anisearch_service.circuit_breaker is not None
        assert anisearch_service.scraper is not None
    
    def test_is_healthy(self, anisearch_service):
        """Test is_healthy method."""
        # Setup - healthy
        anisearch_service.circuit_breaker.is_open.return_value = False
        
        # Execute & Verify
        assert anisearch_service.is_healthy() == True
        
        # Setup - unhealthy
        anisearch_service.circuit_breaker.is_open.return_value = True
        
        # Execute & Verify
        assert anisearch_service.is_healthy() == False
    
    def test_get_service_info(self, anisearch_service):
        """Test get_service_info method."""
        # Setup
        anisearch_service.circuit_breaker.is_open.return_value = False
        
        # Execute
        info = anisearch_service.get_service_info()
        
        # Verify
        assert info["name"] == "anisearch"
        assert info["healthy"] == True
        assert info["circuit_breaker_open"] == False