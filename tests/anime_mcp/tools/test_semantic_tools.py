"""Unit tests for semantic search MCP tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.anime_mcp.tools.semantic_tools import (
    anime_semantic_search,
    anime_similar,
    qdrant_client
)


class TestSemanticTools:
    """Test suite for semantic search MCP tools."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client."""
        client = AsyncMock()
        
        # Mock search results
        client.search.return_value = [
            {
                "anime_id": "test_anime_1",
                "title": "Attack on Titan",
                "synopsis": "Humanity fights against titans",
                "type": "TV",
                "episodes": 25,
                "year": 2013,
                "tags": ["Action", "Drama", "Military"],
                "studios": ["Wit Studio"],
                "data_quality_score": 0.95,
                "_score": 0.87
            },
            {
                "anime_id": "test_anime_2", 
                "title": "Demon Slayer",
                "synopsis": "Boy becomes demon slayer",
                "type": "TV",
                "episodes": 26,
                "year": 2019,
                "tags": ["Action", "Supernatural"],
                "studios": ["Ufotable"],
                "data_quality_score": 0.92,
                "_score": 0.82
            }
        ]
        
        # Mock similarity search results
        client.find_similar.return_value = [
            {
                "anime_id": "similar_anime_1",
                "title": "Hunter x Hunter",
                "synopsis": "Young boy searches for his father",
                "type": "TV",
                "episodes": 148,
                "year": 2011,
                "tags": ["Adventure", "Action"],
                "similarity_score": 0.89,
                "data_quality_score": 0.93
            },
            {
                "anime_id": "similar_anime_2",
                "title": "Fullmetal Alchemist",
                "synopsis": "Brothers seek philosopher's stone",
                "type": "TV", 
                "episodes": 64,
                "year": 2009,
                "tags": ["Adventure", "Drama"],
                "similarity_score": 0.85,
                "data_quality_score": 0.96
            }
        ]
        
        return client

    @pytest.fixture
    def mock_context(self):
        """Create mock MCP context."""
        context = AsyncMock()
        context.info = AsyncMock()
        context.error = AsyncMock()
        return context

    @pytest.mark.asyncio
    async def test_anime_semantic_search_success(self, mock_qdrant_client, mock_context):
        """Test successful semantic search."""
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', mock_qdrant_client):
            
            result = await anime_semantic_search(
                query="action anime with military themes",
                limit=10,
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == 2
            
            anime = result[0]
            assert anime["anime_id"] == "test_anime_1"
            assert anime["title"] == "Attack on Titan"
            assert anime["semantic_score"] == 0.87
            assert anime["data_quality_score"] == 0.95
            assert "tags" in anime
            assert "studios" in anime
            
            # Verify client was called correctly
            mock_qdrant_client.search.assert_called_once_with(
                query="action anime with military themes",
                limit=10,
                filters=None
            )
            
            # Verify context calls
            mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_anime_semantic_search_with_filters(self, mock_qdrant_client, mock_context):
        """Test semantic search with filters."""
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', mock_qdrant_client):
            
            filters = {
                "year": {"gte": 2010, "lte": 2020},
                "type": "TV",
                "tags": {"any": ["Action", "Drama"]}
            }
            
            result = await anime_semantic_search(
                query="modern action anime",
                limit=5,
                filters=filters,
                ctx=mock_context
            )
            
            # Verify client was called with filters
            mock_qdrant_client.search.assert_called_once_with(
                query="modern action anime",
                limit=5,
                filters=filters
            )

    @pytest.mark.asyncio
    async def test_anime_semantic_search_limit_validation(self, mock_qdrant_client):
        """Test semantic search limit validation."""
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', mock_qdrant_client):
            
            # Test limit over maximum (should clamp to 100)
            await anime_semantic_search(query="test", limit=200)
            
            call_args = mock_qdrant_client.search.call_args
            assert call_args[1]["limit"] == 100
            
            # Test limit under minimum (should clamp to 1)
            await anime_semantic_search(query="test", limit=0)
            
            call_args = mock_qdrant_client.search.call_args
            assert call_args[1]["limit"] == 1

    @pytest.mark.asyncio
    async def test_anime_semantic_search_no_client(self, mock_context):
        """Test semantic search when client not available."""
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', None):
            
            with pytest.raises(RuntimeError, match="Qdrant client not initialized"):
                await anime_semantic_search(query="test", ctx=mock_context)
            
            mock_context.error.assert_called_with("Qdrant client not initialized")

    @pytest.mark.asyncio
    async def test_anime_semantic_search_client_error(self, mock_qdrant_client, mock_context):
        """Test semantic search when client raises error."""
        mock_qdrant_client.search.side_effect = Exception("Vector search failed")
        
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', mock_qdrant_client):
            
            with pytest.raises(RuntimeError, match="Semantic search failed: Vector search failed"):
                await anime_semantic_search(query="test", ctx=mock_context)

    @pytest.mark.asyncio
    async def test_anime_semantic_search_empty_results(self, mock_qdrant_client, mock_context):
        """Test semantic search with empty results."""
        mock_qdrant_client.search.return_value = []
        
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', mock_qdrant_client):
            
            result = await anime_semantic_search(query="nonexistent anime", ctx=mock_context)
            
            assert result == []
            mock_context.info.assert_called_with("Found 0 anime in semantic search")

    @pytest.mark.asyncio
    async def test_anime_similar_success(self, mock_qdrant_client, mock_context):
        """Test successful similarity search."""
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', mock_qdrant_client):
            
            result = await anime_similar(
                reference_anime_id="attack_on_titan_123",
                limit=5,
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == 2
            
            similar = result[0]
            assert similar["anime_id"] == "similar_anime_1"
            assert similar["title"] == "Hunter x Hunter"
            assert similar["similarity_score"] == 0.89
            assert similar["data_quality_score"] == 0.93
            
            # Verify client was called correctly
            mock_qdrant_client.find_similar.assert_called_once_with(
                anime_id="attack_on_titan_123",
                limit=5
            )

    @pytest.mark.asyncio
    async def test_anime_similar_limit_validation(self, mock_qdrant_client):
        """Test similarity search limit validation."""
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', mock_qdrant_client):
            
            # Test limit over maximum (should clamp to 50)
            await anime_similar(reference_anime_id="test", limit=100)
            
            call_args = mock_qdrant_client.find_similar.call_args
            assert call_args[1]["limit"] == 50
            
            # Test limit under minimum (should clamp to 1)
            await anime_similar(reference_anime_id="test", limit=0)
            
            call_args = mock_qdrant_client.find_similar.call_args
            assert call_args[1]["limit"] == 1

    @pytest.mark.asyncio
    async def test_anime_similar_no_client(self, mock_context):
        """Test similarity search when client not available."""
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', None):
            
            with pytest.raises(RuntimeError, match="Qdrant client not initialized"):
                await anime_similar(reference_anime_id="test", ctx=mock_context)

    @pytest.mark.asyncio
    async def test_anime_similar_client_error(self, mock_qdrant_client, mock_context):
        """Test similarity search when client raises error."""
        mock_qdrant_client.find_similar.side_effect = Exception("Similarity computation failed")
        
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', mock_qdrant_client):
            
            with pytest.raises(RuntimeError, match="Similarity search failed: Similarity computation failed"):
                await anime_similar(reference_anime_id="test", ctx=mock_context)

    @pytest.mark.asyncio
    async def test_anime_similar_empty_results(self, mock_qdrant_client, mock_context):
        """Test similarity search with empty results."""
        mock_qdrant_client.find_similar.return_value = []
        
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', mock_qdrant_client):
            
            result = await anime_similar(reference_anime_id="unknown_anime", ctx=mock_context)
            
            assert result == []
            mock_context.info.assert_called_with("Found 0 similar anime")


class TestSemanticToolsAdvanced:
    """Advanced tests for semantic search tools."""

    @pytest.mark.asyncio
    async def test_semantic_search_score_processing(self, mock_context):
        """Test semantic search score processing and ranking."""
        mock_client = AsyncMock()
        
        # Mock results with different scores
        mock_client.search.return_value = [
            {
                "anime_id": "high_score",
                "title": "High Score Anime",
                "_score": 0.95,
                "data_quality_score": 0.9
            },
            {
                "anime_id": "medium_score",
                "title": "Medium Score Anime", 
                "_score": 0.78,
                "data_quality_score": 0.8
            },
            {
                "anime_id": "low_score",
                "title": "Low Score Anime",
                "_score": 0.65,
                "data_quality_score": 0.7
            }
        ]
        
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', mock_client):
            
            result = await anime_semantic_search(query="test", ctx=mock_context)
            
            # Verify results are in score order
            assert len(result) == 3
            assert result[0]["semantic_score"] == 0.95
            assert result[1]["semantic_score"] == 0.78
            assert result[2]["semantic_score"] == 0.65
            
            # Verify semantic scores are properly extracted
            for anime in result:
                assert "semantic_score" in anime
                assert anime["semantic_score"] == anime.get("_score", 0)

    @pytest.mark.asyncio
    async def test_similarity_search_score_processing(self, mock_context):
        """Test similarity search score processing."""
        mock_client = AsyncMock()
        
        # Mock similarity results with different scores
        mock_client.find_similar.return_value = [
            {
                "anime_id": "very_similar",
                "title": "Very Similar Anime",
                "similarity_score": 0.92,
                "data_quality_score": 0.85
            },
            {
                "anime_id": "somewhat_similar", 
                "title": "Somewhat Similar Anime",
                "similarity_score": 0.76,
                "data_quality_score": 0.88
            }
        ]
        
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', mock_client):
            
            result = await anime_similar(reference_anime_id="reference", ctx=mock_context)
            
            # Verify similarity scores are preserved
            assert len(result) == 2
            assert result[0]["similarity_score"] == 0.92
            assert result[1]["similarity_score"] == 0.76

    @pytest.mark.asyncio
    async def test_semantic_search_complex_filters(self, mock_context):
        """Test semantic search with complex filter structures."""
        mock_client = AsyncMock()
        mock_client.search.return_value = []
        
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', mock_client):
            
            # Test complex nested filters
            complex_filters = {
                "must": [
                    {"year": {"gte": 2010}},
                    {"type": {"match": {"value": "TV"}}}
                ],
                "should": [
                    {"tags": {"any": ["Action", "Adventure"]}},
                    {"studios": {"any": ["Mappa", "Wit Studio"]}}
                ],
                "must_not": [
                    {"tags": {"any": ["Ecchi", "Horror"]}}
                ]
            }
            
            await anime_semantic_search(
                query="complex filter test",
                filters=complex_filters,
                ctx=mock_context
            )
            
            # Verify complex filters are passed through
            call_args = mock_client.search.call_args
            assert call_args[1]["filters"] == complex_filters

    @pytest.mark.asyncio
    async def test_semantic_tools_data_quality_handling(self, mock_context):
        """Test handling of data quality scores in semantic tools."""
        mock_client = AsyncMock()
        
        # Mock results with various data quality scores
        mock_client.search.return_value = [
            {
                "anime_id": "high_quality",
                "title": "High Quality Data",
                "_score": 0.85,
                "data_quality_score": 0.95
            },
            {
                "anime_id": "medium_quality",
                "title": "Medium Quality Data",
                "_score": 0.88,
                "data_quality_score": 0.75
            },
            {
                "anime_id": "missing_quality",
                "title": "Missing Quality Score",
                "_score": 0.82
                # No data_quality_score field
            }
        ]
        
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', mock_client):
            
            result = await anime_semantic_search(query="quality test", ctx=mock_context)
            
            # Verify data quality scores are handled
            assert result[0]["data_quality_score"] == 0.95
            assert result[1]["data_quality_score"] == 0.75
            assert result[2]["data_quality_score"] == 0.0  # Default for missing

    @pytest.mark.asyncio
    async def test_semantic_tools_missing_fields_handling(self, mock_context):
        """Test handling of missing optional fields."""
        mock_client = AsyncMock()
        
        # Mock result with minimal data
        mock_client.search.return_value = [
            {
                "anime_id": "minimal_data",
                "title": "Minimal Data Anime",
                "_score": 0.8
                # Missing: synopsis, tags, studios, etc.
            }
        ]
        
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', mock_client):
            
            result = await anime_semantic_search(query="minimal test", ctx=mock_context)
            
            anime = result[0]
            # Verify missing fields are handled gracefully
            assert anime["synopsis"] == ""  # Default empty string
            assert anime["tags"] == []  # Default empty list
            assert anime["studios"] == []  # Default empty list
            assert anime["type"] == ""  # Default empty string
            assert anime["episodes"] == 0  # Default zero
            assert anime["year"] == 0  # Default zero

    @pytest.mark.asyncio
    async def test_semantic_tools_without_context(self):
        """Test semantic tools without context (no logging)."""
        mock_client = AsyncMock()
        mock_client.search.return_value = []
        mock_client.find_similar.return_value = []
        
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', mock_client):
            
            # Test search without context
            result = await anime_semantic_search(query="test")  # No ctx parameter
            assert result == []
            
            # Test similarity without context
            result = await anime_similar(reference_anime_id="test")  # No ctx parameter
            assert result == []

    @pytest.mark.asyncio
    async def test_semantic_tools_boundary_values(self):
        """Test semantic tools with boundary values."""
        mock_client = AsyncMock()
        mock_client.search.return_value = []
        mock_client.find_similar.return_value = []
        
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', mock_client):
            
            # Test minimum valid limit
            await anime_semantic_search(query="test", limit=1)
            call_args = mock_client.search.call_args
            assert call_args[1]["limit"] == 1
            
            # Test maximum valid limit
            await anime_semantic_search(query="test", limit=100)
            call_args = mock_client.search.call_args
            assert call_args[1]["limit"] == 100
            
            # Test similarity minimum
            await anime_similar(reference_anime_id="test", limit=1)
            call_args = mock_client.find_similar.call_args
            assert call_args[1]["limit"] == 1
            
            # Test similarity maximum
            await anime_similar(reference_anime_id="test", limit=50)
            call_args = mock_client.find_similar.call_args
            assert call_args[1]["limit"] == 50


class TestSemanticToolsIntegration:
    """Integration tests for semantic tools."""

    @pytest.mark.asyncio
    async def test_semantic_tools_annotation_verification(self):
        """Verify that semantic tools have correct MCP annotations."""
        from src.anime_mcp.tools.semantic_tools import mcp
        
        # Get registered tools
        tools = mcp._tools
        
        # Verify anime_semantic_search annotations
        search_tool = tools.get("anime_semantic_search")
        assert search_tool is not None
        assert search_tool.annotations["title"] == "Semantic Anime Search"
        assert search_tool.annotations["readOnlyHint"] is True
        assert search_tool.annotations["idempotentHint"] is True
        
        # Verify anime_similar annotations
        similar_tool = tools.get("anime_similar")
        assert similar_tool is not None
        assert similar_tool.annotations["title"] == "Find Similar Anime"
        assert similar_tool.annotations["readOnlyHint"] is True

    @pytest.mark.asyncio
    async def test_semantic_to_similarity_workflow(self, mock_context):
        """Test workflow from semantic search to similarity search."""
        mock_client = AsyncMock()
        
        # Setup semantic search result
        mock_client.search.return_value = [
            {
                "anime_id": "found_anime_123",
                "title": "Found Anime",
                "_score": 0.9,
                "data_quality_score": 0.85
            }
        ]
        
        # Setup similarity search result
        mock_client.find_similar.return_value = [
            {
                "anime_id": "similar_anime_456",
                "title": "Similar Anime",
                "similarity_score": 0.88,
                "data_quality_score": 0.9
            }
        ]
        
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', mock_client):
            
            # Step 1: Semantic search
            search_results = await anime_semantic_search(
                query="action adventure anime", 
                ctx=mock_context
            )
            assert len(search_results) == 1
            reference_id = search_results[0]["anime_id"]
            
            # Step 2: Find similar anime
            similar_results = await anime_similar(
                reference_anime_id=reference_id, 
                ctx=mock_context
            )
            assert len(similar_results) == 1
            assert similar_results[0]["anime_id"] == "similar_anime_456"

    @pytest.mark.asyncio
    async def test_semantic_tools_comprehensive_error_scenarios(self, mock_context):
        """Test comprehensive error scenarios for semantic tools."""
        mock_client = AsyncMock()
        
        error_scenarios = [
            ("Connection timeout", "Database connection timeout"),
            ("Vector computation", "Vector similarity computation failed"),
            ("Index corruption", "Search index corrupted"),
            ("Memory error", "Insufficient memory for vector operations")
        ]
        
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', mock_client):
            
            for error_type, error_msg in error_scenarios:
                # Test semantic search errors
                mock_client.search.side_effect = Exception(error_msg)
                
                with pytest.raises(RuntimeError, match=f"Semantic search failed: {error_msg}"):
                    await anime_semantic_search(query="test", ctx=mock_context)
                
                mock_context.error.assert_called_with(f"Semantic search failed: {error_msg}")
                
                # Test similarity search errors
                mock_client.find_similar.side_effect = Exception(error_msg)
                
                with pytest.raises(RuntimeError, match=f"Similarity search failed: {error_msg}"):
                    await anime_similar(reference_anime_id="test", ctx=mock_context)
                
                mock_context.error.assert_called_with(f"Similarity search failed: {error_msg}")
                mock_context.reset_mock()

    @pytest.mark.asyncio
    async def test_semantic_tools_performance_scenarios(self, mock_context):
        """Test semantic tools with performance-related scenarios."""
        mock_client = AsyncMock()
        
        # Mock large result set
        large_result_set = []
        for i in range(100):
            large_result_set.append({
                "anime_id": f"anime_{i}",
                "title": f"Anime {i}",
                "_score": 0.9 - (i * 0.01),
                "data_quality_score": 0.8 + (i % 20) * 0.01
            })
        
        mock_client.search.return_value = large_result_set
        
        with patch('src.anime_mcp.tools.semantic_tools.qdrant_client', mock_client):
            
            # Test handling large result sets
            result = await anime_semantic_search(
                query="popular anime", 
                limit=100, 
                ctx=mock_context
            )
            
            assert len(result) == 100
            # Verify scores are in descending order
            for i in range(1, len(result)):
                assert result[i-1]["semantic_score"] >= result[i]["semantic_score"]