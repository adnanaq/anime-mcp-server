"""Simple unit tests for Jikan MCP tools - direct function testing."""

import pytest
from unittest.mock import AsyncMock, patch
from typing import Dict, Any, List

from src.models.structured_responses import BasicAnimeResult, AnimeType, AnimeStatus


class TestJikanToolsSimple:
    """Test suite for Jikan MCP tools using direct function calls."""

    @pytest.fixture
    def mock_jikan_client(self):
        """Create mock Jikan client."""
        client = AsyncMock()
        
        # Mock search_anime response
        client.search_anime.return_value = [
            {
                "mal_id": 16498,
                "title": "Shingeki no Kyojin",
                "type": "TV",
                "score": 8.54,
                "year": 2013,
                "genres": [{"name": "Action"}, {"name": "Drama"}],
                "synopsis": "Titans attack",
                "images": {"jpg": {"image_url": "https://example.com/aot.jpg"}}
            }
        ]
        
        # Mock get_anime_by_id response
        client.get_anime_by_id.return_value = {
            "mal_id": 16498,
            "title": "Shingeki no Kyojin",
            "type": "TV",
            "score": 8.54,
            "year": 2013,
            "genres": [{"name": "Action"}, {"name": "Drama"}],
            "synopsis": "Titans attack detailed",
            "images": {"jpg": {"image_url": "https://example.com/aot.jpg"}}
        }
        
        return client

    @pytest.mark.asyncio
    async def test_search_anime_jikan_basic(self, mock_jikan_client):
        """Test basic Jikan anime search functionality."""
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_jikan_client):
            from src.anime_mcp.tools.jikan_tools import search_anime_jikan
            
            # Test the search function (using .fn to access the underlying function)
            result = await search_anime_jikan.fn(
                query="attack on titan", 
                limit=10
            )
            
            # Verify the call was made correctly
            mock_jikan_client.search_anime.assert_called_once()
            call_kwargs = mock_jikan_client.search_anime.call_args[1]
            assert call_kwargs["q"] == "attack on titan"
            assert call_kwargs["limit"] == 10
            
            # Verify the response structure
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], BasicAnimeResult)
            assert result[0].title == "Shingeki no Kyojin"
            assert result[0].type == AnimeType.TV
            assert result[0].score == 8.54

    @pytest.mark.asyncio
    async def test_get_anime_jikan_basic(self, mock_jikan_client):
        """Test basic Jikan anime details functionality."""
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_jikan_client):
            from src.anime_mcp.tools.jikan_tools import get_anime_jikan
            
            # Test the get function
            result = await get_anime_jikan.fn(mal_id=16498)
            
            # Verify the call was made correctly
            mock_jikan_client.get_anime_by_id.assert_called_once_with(16498)
            
            # Verify the response structure
            assert isinstance(result, dict)
            assert result["title"] == "Shingeki no Kyojin"
            assert result["type"] == AnimeType.TV
            assert result["score"] == 8.54

    @pytest.mark.asyncio
    async def test_search_anime_jikan_with_filters(self, mock_jikan_client):
        """Test Jikan anime search with filters."""
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_jikan_client):
            from src.anime_mcp.tools.jikan_tools import search_anime_jikan
            
            # Test with filters
            result = await search_anime_jikan.fn(
                query="action",
                limit=5,
                type="tv",
                status="complete",
                min_score=8.0
            )
            
            # Verify the call was made with correct parameters
            mock_jikan_client.search_anime.assert_called_once()
            call_kwargs = mock_jikan_client.search_anime.call_args[1]
            assert call_kwargs["q"] == "action"
            assert call_kwargs["limit"] == 5
            assert call_kwargs["type"] == "tv"
            assert call_kwargs["status"] == "complete"
            assert call_kwargs["min_score"] == 8.0

    @pytest.mark.asyncio
    async def test_search_anime_jikan_error_handling(self, mock_jikan_client):
        """Test error handling in Jikan search."""
        mock_jikan_client.search_anime.side_effect = Exception("API Error")
        
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_jikan_client):
            from src.anime_mcp.tools.jikan_tools import search_anime_jikan
            
            # Test that errors are properly handled
            with pytest.raises(RuntimeError, match="Jikan search failed"):
                await search_anime_jikan.fn(query="test", limit=10)

    @pytest.mark.asyncio
    async def test_search_anime_jikan_empty_response(self, mock_jikan_client):
        """Test handling of empty search results."""
        mock_jikan_client.search_anime.return_value = []
        
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_jikan_client):
            from src.anime_mcp.tools.jikan_tools import search_anime_jikan
            
            # Test with empty results
            result = await search_anime_jikan.fn(query="nonexistent", limit=10)
            
            # Verify empty response handling
            assert isinstance(result, list)
            assert len(result) == 0

    def test_jikan_client_initialization(self):
        """Test that Jikan client is properly initialized."""
        from src.anime_mcp.tools.jikan_tools import jikan_client
        assert jikan_client is not None
        
    def test_mcp_tools_exist(self):
        """Test that MCP tools are defined."""
        from src.anime_mcp.tools.jikan_tools import mcp
        assert mcp is not None
        # Just check that the mcp server instance exists - tool registration is handled internally

    @pytest.mark.asyncio
    async def test_structured_response_format(self, mock_jikan_client):
        """Test that responses are properly structured."""
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_jikan_client):
            from src.anime_mcp.tools.jikan_tools import search_anime_jikan
            
            # Test search response
            result = await search_anime_jikan.fn(query="test", limit=10)
            
            # Verify BasicAnimeResult structure
            assert len(result) == 1
            anime_result = result[0]
            assert isinstance(anime_result, BasicAnimeResult)
            
            # Check required fields
            assert anime_result.id == "16498"
            assert anime_result.title == "Shingeki no Kyojin"
            assert anime_result.score == 8.54
            assert anime_result.year == 2013
            assert anime_result.type == AnimeType.TV
            assert "Action" in anime_result.genres
            assert anime_result.image_url == "https://example.com/aot.jpg"

    @pytest.mark.asyncio
    async def test_type_conversion(self, mock_jikan_client):
        """Test that anime types are properly converted."""
        test_cases = [
            ("TV", AnimeType.TV),
            ("Movie", AnimeType.MOVIE),
            ("OVA", AnimeType.OVA),
            ("Special", AnimeType.SPECIAL),
        ]
        
        for jikan_type, expected_type in test_cases:
            mock_jikan_client.search_anime.return_value = [
                {
                    "mal_id": 1,
                    "title": "Test",
                    "type": jikan_type,
                    "score": 8.0,
                    "year": 2023,
                    "genres": [],
                    "synopsis": "Test",
                    "images": {"jpg": {"image_url": "test.jpg"}}
                }
            ]
            
            with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_jikan_client):
                from src.anime_mcp.tools.jikan_tools import search_anime_jikan
                
                result = await search_anime_jikan.fn(query="test", limit=1)
                assert result[0].type == expected_type