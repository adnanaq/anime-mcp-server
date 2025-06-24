"""Tests for MCP tool adapter layer."""

from unittest.mock import AsyncMock, patch

import pytest

from src.langgraph.adapters import (
    AnimeDetailsAdapter,
    AnimeSearchAdapter,
    ImageSearchAdapter,
    MCPToolAdapter,
    MultimodalSearchAdapter,
    RecommendationAdapter,
    SimilarityAdapter,
    StatsAdapter,
    VisualSimilarityAdapter,
)


class TestMCPToolAdapter:
    """Test base MCP tool adapter."""

    def test_create_adapter(self):
        """Test creating a basic adapter."""
        mock_tool_func = AsyncMock(return_value={"result": "test"})
        adapter = MCPToolAdapter(
            tool_name="test_tool",
            tool_function=mock_tool_func,
            description="Test tool adapter",
        )

        assert adapter.tool_name == "test_tool"
        assert adapter.description == "Test tool adapter"
        assert adapter.tool_function == mock_tool_func

    @pytest.mark.asyncio
    async def test_invoke_adapter(self):
        """Test invoking an adapter."""
        mock_tool_func = AsyncMock(return_value=[{"title": "Test Anime", "score": 0.9}])
        adapter = MCPToolAdapter(
            tool_name="test_search",
            tool_function=mock_tool_func,
            description="Test search",
        )

        result = await adapter.invoke({"query": "test", "limit": 5})

        mock_tool_func.assert_called_once_with(query="test", limit=5)
        assert result == [{"title": "Test Anime", "score": 0.9}]

    @pytest.mark.asyncio
    async def test_adapter_error_handling(self):
        """Test adapter error handling."""
        mock_tool_func = AsyncMock(side_effect=Exception("Tool error"))
        adapter = MCPToolAdapter(
            tool_name="error_tool",
            tool_function=mock_tool_func,
            description="Error tool",
        )

        with pytest.raises(Exception, match="Tool error"):
            await adapter.invoke({"query": "test"})


class TestAnimeSearchAdapter:
    """Test anime search adapter."""

    @pytest.mark.asyncio
    async def test_search_adapter(self):
        """Test anime search adapter."""
        mock_search_func = AsyncMock(
            return_value=[
                {"title": "Naruto", "anime_id": "abc123", "score": 0.95},
                {"title": "One Piece", "anime_id": "def456", "score": 0.90},
            ]
        )

        adapter = AnimeSearchAdapter(mock_search_func)
        result = await adapter.invoke({"query": "shounen anime", "limit": 2})

        mock_search_func.assert_called_once_with(query="shounen anime", limit=2)
        assert len(result) == 2
        assert result[0]["title"] == "Naruto"
        assert result[1]["title"] == "One Piece"

    def test_search_adapter_properties(self):
        """Test search adapter properties."""
        mock_func = AsyncMock()
        adapter = AnimeSearchAdapter(mock_func)

        assert adapter.tool_name == "search_anime"
        assert "semantic search" in adapter.description.lower()


class TestAnimeDetailsAdapter:
    """Test anime details adapter."""

    @pytest.mark.asyncio
    async def test_details_adapter(self):
        """Test anime details adapter."""
        mock_details_func = AsyncMock(
            return_value={
                "anime_id": "test123",
                "title": "Attack on Titan",
                "synopsis": "Epic story",
                "episodes": 25,
                "year": 2013,
            }
        )

        adapter = AnimeDetailsAdapter(mock_details_func)
        result = await adapter.invoke({"anime_id": "test123"})

        mock_details_func.assert_called_once_with(anime_id="test123")
        assert result["title"] == "Attack on Titan"
        assert result["episodes"] == 25

    def test_details_adapter_properties(self):
        """Test details adapter properties."""
        mock_func = AsyncMock()
        adapter = AnimeDetailsAdapter(mock_func)

        assert adapter.tool_name == "get_anime_details"
        assert "detailed information" in adapter.description.lower()


class TestSimilarityAdapter:
    """Test similarity search adapter."""

    @pytest.mark.asyncio
    async def test_similarity_adapter(self):
        """Test similarity adapter."""
        mock_similarity_func = AsyncMock(
            return_value=[
                {"title": "Similar Anime 1", "score": 0.88},
                {"title": "Similar Anime 2", "score": 0.85},
            ]
        )

        adapter = SimilarityAdapter(mock_similarity_func)
        result = await adapter.invoke({"anime_id": "source123", "limit": 2})

        mock_similarity_func.assert_called_once_with(anime_id="source123", limit=2)
        assert len(result) == 2
        assert result[0]["title"] == "Similar Anime 1"


class TestImageSearchAdapter:
    """Test image search adapter."""

    @pytest.mark.asyncio
    async def test_image_search_adapter(self):
        """Test image search adapter."""
        mock_image_func = AsyncMock(
            return_value=[
                {"title": "Visually Similar 1", "score": 0.82},
                {"title": "Visually Similar 2", "score": 0.78},
            ]
        )

        adapter = ImageSearchAdapter(mock_image_func)
        result = await adapter.invoke({"image_data": "base64_data", "limit": 2})

        mock_image_func.assert_called_once_with(image_data="base64_data", limit=2)
        assert len(result) == 2
        assert result[0]["title"] == "Visually Similar 1"

    def test_image_search_adapter_properties(self):
        """Test image search adapter properties."""
        mock_func = AsyncMock()
        adapter = ImageSearchAdapter(mock_func)

        assert adapter.tool_name == "search_anime_by_image"
        assert "image" in adapter.description.lower()


class TestMultimodalSearchAdapter:
    """Test multimodal search adapter."""

    @pytest.mark.asyncio
    async def test_multimodal_adapter(self):
        """Test multimodal search adapter."""
        mock_multimodal_func = AsyncMock(
            return_value=[
                {"title": "Multimodal Result 1", "score": 0.91},
                {"title": "Multimodal Result 2", "score": 0.87},
            ]
        )

        adapter = MultimodalSearchAdapter(mock_multimodal_func)
        result = await adapter.invoke(
            {
                "query": "mecha anime",
                "image_data": "base64_image",
                "text_weight": 0.7,
                "limit": 2,
            }
        )

        mock_multimodal_func.assert_called_once_with(
            query="mecha anime", image_data="base64_image", text_weight=0.7, limit=2
        )
        assert len(result) == 2
        assert result[0]["title"] == "Multimodal Result 1"


class TestStatsAdapter:
    """Test stats adapter."""

    @pytest.mark.asyncio
    async def test_stats_adapter(self):
        """Test stats adapter."""
        mock_stats_func = AsyncMock(
            return_value={
                "total_documents": 38894,
                "vector_size": 384,
                "status": "green",
            }
        )

        adapter = StatsAdapter(mock_stats_func)
        result = await adapter.invoke({})

        mock_stats_func.assert_called_once_with()
        assert result["total_documents"] == 38894
        assert result["status"] == "green"


class TestRecommendationAdapter:
    """Test recommendation adapter."""

    @pytest.mark.asyncio
    async def test_recommendation_adapter(self):
        """Test recommendation adapter."""
        mock_rec_func = AsyncMock(
            return_value=[
                {"title": "Recommended 1", "score": 0.88},
                {"title": "Recommended 2", "score": 0.85},
            ]
        )

        adapter = RecommendationAdapter(mock_rec_func)
        result = await adapter.invoke(
            {"genres": ["action", "adventure"], "year": 2020, "limit": 2}
        )

        mock_rec_func.assert_called_once_with(
            genres=["action", "adventure"], year=2020, limit=2
        )
        assert len(result) == 2
        assert result[0]["title"] == "Recommended 1"


class TestAdapterIntegration:
    """Test adapter integration scenarios."""

    @pytest.mark.asyncio
    async def test_adapter_chaining(self):
        """Test chaining multiple adapters."""
        # Mock search that returns anime IDs
        search_func = AsyncMock(
            return_value=[
                {"anime_id": "anime1", "title": "Search Result 1", "score": 0.9}
            ]
        )

        # Mock similarity that uses the anime ID
        similarity_func = AsyncMock(
            return_value=[
                {"anime_id": "anime2", "title": "Similar Result 1", "score": 0.8}
            ]
        )

        search_adapter = AnimeSearchAdapter(search_func)
        similarity_adapter = SimilarityAdapter(similarity_func)

        # Step 1: Search for anime
        search_results = await search_adapter.invoke({"query": "action", "limit": 1})

        # Step 2: Find similar anime using first result
        anime_id = search_results[0]["anime_id"]
        similar_results = await similarity_adapter.invoke(
            {"anime_id": anime_id, "limit": 1}
        )

        assert search_results[0]["title"] == "Search Result 1"
        assert similar_results[0]["title"] == "Similar Result 1"
        similarity_func.assert_called_once_with(anime_id="anime1", limit=1)

    @pytest.mark.asyncio
    async def test_adapter_error_propagation(self):
        """Test error propagation through adapters."""
        error_func = AsyncMock(side_effect=ValueError("Invalid anime ID"))
        adapter = AnimeDetailsAdapter(error_func)

        with pytest.raises(ValueError, match="Invalid anime ID"):
            await adapter.invoke({"anime_id": "invalid"})

    def test_all_adapters_have_unique_names(self):
        """Test that all adapters have unique tool names."""
        mock_func = AsyncMock()
        adapters = [
            AnimeSearchAdapter(mock_func),
            AnimeDetailsAdapter(mock_func),
            SimilarityAdapter(mock_func),
            StatsAdapter(mock_func),
            RecommendationAdapter(mock_func),
            ImageSearchAdapter(mock_func),
            VisualSimilarityAdapter(mock_func),
            MultimodalSearchAdapter(mock_func),
        ]

        tool_names = [adapter.tool_name for adapter in adapters]
        assert len(tool_names) == len(set(tool_names)), "Tool names must be unique"

        expected_names = {
            "search_anime",
            "get_anime_details",
            "find_similar_anime",
            "get_anime_stats",
            "recommend_anime",
            "search_anime_by_image",
            "find_visually_similar_anime",
            "search_multimodal_anime",
        }
        assert set(tool_names) == expected_names

    @patch("src.mcp.server.QdrantClient")
    def test_mcp_client_initialization_through_tools(self, mock_qdrant_client):
        """Test that MCP client gets initialized when tools are loaded."""
        import src.mcp.server as server_module
        from src.mcp.tools import get_all_mcp_tools

        # Reset the client to None to test initialization
        server_module.qdrant_client = None

        # Call get_all_mcp_tools which should initialize client
        tools = get_all_mcp_tools()

        # Should return all 8 tools
        assert len(tools) == 8
        expected_tools = [
            "search_anime",
            "get_anime_details",
            "find_similar_anime",
            "get_anime_stats",
            "recommend_anime",
            "search_anime_by_image",
            "find_visually_similar_anime",
            "search_multimodal_anime",
        ]
        for tool_name in expected_tools:
            assert tool_name in tools
            assert callable(tools[tool_name])

        # After calling get_all_mcp_tools, QdrantClient should have been instantiated
        mock_qdrant_client.assert_called_once()

        # The module-level client should now be set
        assert server_module.qdrant_client is not None
