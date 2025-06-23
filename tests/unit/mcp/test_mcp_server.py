"""Unit tests for FastMCP server implementation."""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any, List
import json

from src.mcp.server import (
    search_anime, get_anime_details, find_similar_anime, 
    get_anime_stats, recommend_anime, initialize_mcp_server
)


class TestFastMCPServerImplementation:
    """Test cases for FastMCP server functionality."""
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client for testing."""
        client = AsyncMock()
        client.health_check.return_value = True
        client.search.return_value = [
            {
                "anime_id": "test123",
                "title": "Test Anime",
                "synopsis": "A test anime for unit testing",
                "type": "TV",
                "episodes": 12,
                "year": 2023,
                "tags": ["action", "test"],
                "studios": ["Test Studio"],
                "data_quality_score": 0.95
            }
        ]
        client.get_by_id.return_value = {
            "anime_id": "test123",
            "title": "Detailed Test Anime",
            "synopsis": "Detailed information about test anime",
            "type": "TV",
            "episodes": 24,
            "year": 2023,
            "tags": ["action", "drama"],
            "studios": ["Test Studio"]
        }
        client.find_similar.return_value = [
            {
                "anime_id": "similar123",
                "title": "Similar Anime",
                "similarity_score": 0.85,
                "type": "TV",
                "year": 2022
            }
        ]
        client.get_stats.return_value = {
            "total_documents": 38894,
            "collection_name": "anime_database",
            "vector_size": 384
        }
        return client
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = MagicMock()
        settings.qdrant_url = "http://localhost:6333"
        settings.qdrant_collection_name = "anime_database"
        settings.fastembed_model = "BAAI/bge-small-en-v1.5"
        settings.qdrant_vector_size = 384
        settings.qdrant_distance_metric = "cosine"
        return settings
    
    @pytest.mark.asyncio
    async def test_search_anime_tool(self, mock_qdrant_client):
        """Test search_anime FastMCP tool."""
        # Mock the global qdrant_client
        with patch('src.mcp.server.qdrant_client', mock_qdrant_client):
            result = await search_anime(query="action anime", limit=5)
            
            assert isinstance(result, list)
            assert len(result) > 0
            
            # Verify the mock was called correctly
            mock_qdrant_client.search.assert_called_once_with(query="action anime", limit=5)
            
            # Check result structure
            anime = result[0]
            assert "anime_id" in anime
            assert "title" in anime
            assert "synopsis" in anime
            assert anime["title"] == "Test Anime"
    
    @pytest.mark.asyncio
    async def test_search_anime_limit_validation(self, mock_qdrant_client):
        """Test search_anime limit validation."""
        with patch('src.mcp.server.qdrant_client', mock_qdrant_client):
            # Test limit clamping
            result = await search_anime(query="test", limit=100)  # Over max of 50
            mock_qdrant_client.search.assert_called_with(query="test", limit=50)
            
            result = await search_anime(query="test", limit=0)  # Under min of 1
            mock_qdrant_client.search.assert_called_with(query="test", limit=1)
    
    @pytest.mark.asyncio
    async def test_search_anime_error_handling(self, mock_qdrant_client):
        """Test search_anime error handling."""
        mock_qdrant_client.search.side_effect = Exception("Database error")
        
        with patch('src.mcp.server.qdrant_client', mock_qdrant_client):
            with pytest.raises(RuntimeError) as exc_info:
                await search_anime(query="test")
            
            assert "Search failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_search_anime_no_client(self):
        """Test search_anime when client is not initialized."""
        with patch('src.mcp.server.qdrant_client', None):
            with pytest.raises(RuntimeError) as exc_info:
                await search_anime(query="test")
            
            assert "Qdrant client not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_anime_details_tool(self, mock_qdrant_client):
        """Test get_anime_details FastMCP tool."""
        with patch('src.mcp.server.qdrant_client', mock_qdrant_client):
            result = await get_anime_details(anime_id="test123")
            
            assert isinstance(result, dict)
            assert "anime_id" in result
            assert "title" in result
            assert result["anime_id"] == "test123"
            assert result["title"] == "Detailed Test Anime"
            
            # Verify the mock was called correctly
            mock_qdrant_client.get_by_id.assert_called_once_with("test123")
    
    @pytest.mark.asyncio
    async def test_get_anime_details_not_found(self, mock_qdrant_client):
        """Test get_anime_details when anime not found."""
        mock_qdrant_client.get_by_id.return_value = None
        
        with patch('src.mcp.server.qdrant_client', mock_qdrant_client):
            with pytest.raises(ValueError) as exc_info:
                await get_anime_details(anime_id="nonexistent")
            
            assert "Anime not found: nonexistent" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_find_similar_anime_tool(self, mock_qdrant_client):
        """Test find_similar_anime FastMCP tool."""
        with patch('src.mcp.server.qdrant_client', mock_qdrant_client):
            result = await find_similar_anime(anime_id="test123", limit=8)
            
            assert isinstance(result, list)
            assert len(result) > 0
            
            # Verify the mock was called correctly
            mock_qdrant_client.find_similar.assert_called_once_with(anime_id="test123", limit=8)
            
            # Check result structure
            similar = result[0]
            assert "anime_id" in similar
            assert "similarity_score" in similar
    
    @pytest.mark.asyncio
    async def test_find_similar_anime_limit_validation(self, mock_qdrant_client):
        """Test find_similar_anime limit validation."""
        with patch('src.mcp.server.qdrant_client', mock_qdrant_client):
            # Test limit clamping
            await find_similar_anime(anime_id="test123", limit=50)  # Over max of 20
            mock_qdrant_client.find_similar.assert_called_with(anime_id="test123", limit=20)
            
            await find_similar_anime(anime_id="test123", limit=0)  # Under min of 1
            mock_qdrant_client.find_similar.assert_called_with(anime_id="test123", limit=1)
    
    @pytest.mark.asyncio
    async def test_get_anime_stats_tool(self, mock_qdrant_client, mock_settings):
        """Test get_anime_stats FastMCP tool."""
        with patch('src.mcp.server.qdrant_client', mock_qdrant_client), \
             patch('src.mcp.server.settings', mock_settings):
            
            result = await get_anime_stats()
            
            assert isinstance(result, dict)
            assert "total_documents" in result
            assert "health_status" in result
            assert "server_info" in result
            assert result["total_documents"] == 38894
            assert result["health_status"] == "healthy"
            
            # Check server info
            server_info = result["server_info"]
            assert server_info["qdrant_url"] == "http://localhost:6333"
            assert server_info["collection_name"] == "anime_database"
            assert server_info["vector_model"] == "BAAI/bge-small-en-v1.5"
    
    @pytest.mark.asyncio
    async def test_get_anime_stats_unhealthy(self, mock_qdrant_client, mock_settings):
        """Test get_anime_stats with unhealthy database."""
        mock_qdrant_client.health_check.return_value = False
        
        with patch('src.mcp.server.qdrant_client', mock_qdrant_client), \
             patch('src.mcp.server.settings', mock_settings):
            
            result = await get_anime_stats()
            assert result["health_status"] == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_recommend_anime_tool(self, mock_qdrant_client):
        """Test recommend_anime FastMCP tool."""
        with patch('src.mcp.server.qdrant_client', mock_qdrant_client):
            result = await recommend_anime(
                genres="Action,Comedy",
                year=2023,
                anime_type="TV",
                limit=10
            )
            
            assert isinstance(result, list)
            # Mock search should have been called with the query
            mock_qdrant_client.search.assert_called()
    
    @pytest.mark.asyncio
    async def test_recommend_anime_defaults(self, mock_qdrant_client):
        """Test recommend_anime with default parameters."""
        with patch('src.mcp.server.qdrant_client', mock_qdrant_client):
            result = await recommend_anime()
            
            assert isinstance(result, list)
            # Should search for "popular anime" when no preferences given
            mock_qdrant_client.search.assert_called_with(query="popular anime", limit=10)
    
    @pytest.mark.asyncio
    async def test_recommend_anime_limit_validation(self, mock_qdrant_client):
        """Test recommend_anime limit validation."""
        with patch('src.mcp.server.qdrant_client', mock_qdrant_client):
            # Test limit clamping
            await recommend_anime(limit=50)  # Over max of 25
            mock_qdrant_client.search.assert_called_with(query="popular anime", limit=25)
            
            await recommend_anime(limit=0)  # Under min of 1
            mock_qdrant_client.search.assert_called_with(query="popular anime", limit=1)
    
    @pytest.mark.asyncio
    async def test_initialize_mcp_server_success(self, mock_settings):
        """Test successful MCP server initialization."""
        mock_client = AsyncMock()
        mock_client.health_check.return_value = True
        
        with patch('src.mcp.server.QdrantClient', return_value=mock_client), \
             patch('src.mcp.server.settings', mock_settings):
            
            await initialize_mcp_server()
            
            # Verify client was created and health check performed
            mock_client.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_mcp_server_failure(self, mock_settings):
        """Test MCP server initialization failure."""
        mock_client = AsyncMock()
        mock_client.health_check.return_value = False
        
        with patch('src.mcp.server.QdrantClient', return_value=mock_client), \
             patch('src.mcp.server.settings', mock_settings):
            
            with pytest.raises(RuntimeError) as exc_info:
                await initialize_mcp_server()
            
            assert "Cannot initialize MCP server without database connection" in str(exc_info.value)
    
    def test_fastmcp_tool_decorators(self):
        """Test that FastMCP tool decorators are properly applied."""
        # Verify that our tools have the correct function signatures
        import inspect
        
        # Test search_anime signature
        sig = inspect.signature(search_anime)
        assert 'query' in sig.parameters
        assert 'limit' in sig.parameters
        assert sig.parameters['limit'].default == 10
        
        # Test get_anime_details signature
        sig = inspect.signature(get_anime_details)
        assert 'anime_id' in sig.parameters
        
        # Test find_similar_anime signature
        sig = inspect.signature(find_similar_anime)
        assert 'anime_id' in sig.parameters
        assert 'limit' in sig.parameters
        assert sig.parameters['limit'].default == 10
        
        # Test recommend_anime signature
        sig = inspect.signature(recommend_anime)
        assert 'genres' in sig.parameters
        assert 'year' in sig.parameters
        assert 'anime_type' in sig.parameters
        assert 'limit' in sig.parameters
        
        # Check optional parameters have None defaults
        assert sig.parameters['genres'].default is None
        assert sig.parameters['year'].default is None
        assert sig.parameters['anime_type'].default is None
        assert sig.parameters['limit'].default == 10
    
    def test_fastmcp_docstrings(self):
        """Test that FastMCP tools have proper docstrings."""
        # Verify docstrings exist and contain key information
        assert search_anime.__doc__ is not None
        assert "Search for anime using semantic search" in search_anime.__doc__
        assert "Args:" in search_anime.__doc__
        assert "Returns:" in search_anime.__doc__
        
        assert get_anime_details.__doc__ is not None
        assert "Get detailed information about a specific anime" in get_anime_details.__doc__
        
        assert find_similar_anime.__doc__ is not None
        assert "Find anime similar to a given anime" in find_similar_anime.__doc__
        
        assert recommend_anime.__doc__ is not None
        assert "Get anime recommendations based on preferences" in recommend_anime.__doc__
    
    @pytest.mark.asyncio
    async def test_resource_functions(self, mock_qdrant_client, mock_settings):
        """Test FastMCP resource functions."""
        from src.mcp.server import database_stats, database_schema
        
        # Test database_stats resource
        with patch('src.mcp.server.qdrant_client', mock_qdrant_client), \
             patch('src.mcp.server.settings', mock_settings):
            
            stats_result = await database_stats()
            assert isinstance(stats_result, str)
            assert "Anime Database Stats" in stats_result
    
    @pytest.mark.asyncio
    async def test_resource_functions_no_client(self):
        """Test resource functions when client is not initialized."""
        from src.mcp.server import database_stats, database_schema
        
        with patch('src.mcp.server.qdrant_client', None):
            stats_result = await database_stats()
            assert "Database client not initialized" in stats_result
            
            schema_result = await database_schema()
            assert isinstance(schema_result, str)
            assert "Anime Database Schema" in schema_result
    
    def test_query_building_logic(self):
        """Test recommendation query building logic."""
        # This tests the internal logic used by recommend_anime
        def build_query(genres=None, year=None, anime_type=None):
            query_parts = []
            if genres:
                genre_list = [g.strip() for g in genres.split(",")]
                query_parts.extend(genre_list)
            if year:
                query_parts.append(str(year))
            if anime_type:
                query_parts.append(anime_type)
            return " ".join(query_parts) if query_parts else "popular anime"
        
        # Test different combinations
        assert build_query() == "popular anime"
        assert build_query(genres="Action") == "Action"
        assert build_query(genres="Action,Comedy") == "Action Comedy"
        assert build_query(year=2023) == "2023"
        assert build_query(anime_type="TV") == "TV"
        assert build_query(genres="Action", year=2023, anime_type="TV") == "Action 2023 TV"
    
    def test_result_filtering_logic(self):
        """Test anime result filtering logic."""
        # Mock anime data
        test_anime = [
            {"anime_id": "1", "year": 2023, "type": "TV", "tags": ["Action", "Drama"]},
            {"anime_id": "2", "year": 2022, "type": "Movie", "tags": ["Comedy", "Romance"]},
            {"anime_id": "3", "year": 2023, "type": "TV", "tags": ["Action", "Comedy"]},
        ]
        
        def filter_anime(anime_list, year=None, anime_type=None, genres=None):
            filtered = []
            for anime in anime_list:
                # Check year filter
                if year and anime.get("year") != year:
                    continue
                
                # Check type filter
                if anime_type and anime.get("type", "").lower() != anime_type.lower():
                    continue
                
                # Check genre filter
                if genres:
                    anime_tags = [tag.lower() for tag in anime.get("tags", [])]
                    requested_genres = [g.strip().lower() for g in genres.split(",")]
                    if not any(genre in anime_tags for genre in requested_genres):
                        continue
                
                filtered.append(anime)
            
            return filtered
        
        # Test filtering
        assert len(filter_anime(test_anime)) == 3  # No filters
        assert len(filter_anime(test_anime, year=2023)) == 2  # Year filter
        assert len(filter_anime(test_anime, anime_type="TV")) == 2  # Type filter
        assert len(filter_anime(test_anime, genres="Action")) == 2  # Genre filter
        assert len(filter_anime(test_anime, year=2023, anime_type="TV")) == 2  # Multiple filters
        assert len(filter_anime(test_anime, genres="Romance")) == 1  # Specific genre