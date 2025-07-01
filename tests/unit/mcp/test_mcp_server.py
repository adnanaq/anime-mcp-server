"""Unit tests for FastMCP server implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mcp.server import (
    find_similar_anime,
    get_anime_details,
    get_anime_stats,
    initialize_mcp_server,
    search_anime,
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
                "data_quality_score": 0.95,
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
            "studios": ["Test Studio"],
        }
        client.find_similar.return_value = [
            {
                "anime_id": "similar123",
                "title": "Similar Anime",
                "similarity_score": 0.85,
                "type": "TV",
                "year": 2022,
            }
        ]
        client.get_stats.return_value = {
            "total_documents": 38894,
            "collection_name": "anime_database",
            "vector_size": 384,
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
        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            # search_anime is a FunctionTool, so we need to call its underlying function
            result = await search_anime.fn(query="action anime", limit=5)

            assert isinstance(result, list)
            assert len(result) > 0

            # Verify the mock was called correctly
            mock_qdrant_client.search.assert_called_once_with(
                query="action anime", limit=5, filters=None
            )

            # Check result structure
            anime = result[0]
            assert "anime_id" in anime
            assert "title" in anime
            assert "synopsis" in anime
            assert anime["title"] == "Test Anime"

    @pytest.mark.asyncio
    async def test_search_anime_limit_validation(self, mock_qdrant_client):
        """Test search_anime limit validation."""
        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            # Test limit clamping
            result = await search_anime.fn(query="test", limit=100)  # Over max of 50
            mock_qdrant_client.search.assert_called_with(
                query="test", limit=50, filters=None
            )

            result = await search_anime.fn(query="test", limit=0)  # Under min of 1
            mock_qdrant_client.search.assert_called_with(
                query="test", limit=1, filters=None
            )

    @pytest.mark.asyncio
    async def test_search_anime_error_handling(self, mock_qdrant_client):
        """Test search_anime error handling."""
        mock_qdrant_client.search.side_effect = Exception("Database error")

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            with pytest.raises(RuntimeError) as exc_info:
                await search_anime.fn(query="test")

            assert "Search failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_anime_no_client(self):
        """Test search_anime when client is not initialized."""
        with patch("src.mcp.server.qdrant_client", None):
            with pytest.raises(RuntimeError) as exc_info:
                await search_anime.fn(query="test")

            assert "Qdrant client not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_anime_details_tool(self, mock_qdrant_client):
        """Test get_anime_details FastMCP tool."""
        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            result = await get_anime_details.fn(anime_id="test123")

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

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            with pytest.raises(RuntimeError) as exc_info:
                await get_anime_details.fn(anime_id="nonexistent")

            assert "Failed to get anime details: Anime not found: nonexistent" in str(
                exc_info.value
            )

    @pytest.mark.asyncio
    async def test_find_similar_anime_tool(self, mock_qdrant_client):
        """Test find_similar_anime FastMCP tool."""
        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            result = await find_similar_anime.fn(anime_id="test123", limit=8)

            assert isinstance(result, list)
            assert len(result) > 0

            # Verify the mock was called correctly
            mock_qdrant_client.find_similar.assert_called_once_with(
                anime_id="test123", limit=8
            )

            # Check result structure
            similar = result[0]
            assert "anime_id" in similar
            assert "similarity_score" in similar

    @pytest.mark.asyncio
    async def test_find_similar_anime_limit_validation(self, mock_qdrant_client):
        """Test find_similar_anime limit validation."""
        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            # Test limit clamping
            await find_similar_anime.fn(anime_id="test123", limit=50)  # Over max of 20
            mock_qdrant_client.find_similar.assert_called_with(
                anime_id="test123", limit=20
            )

            await find_similar_anime.fn(anime_id="test123", limit=0)  # Under min of 1
            mock_qdrant_client.find_similar.assert_called_with(
                anime_id="test123", limit=1
            )

    @pytest.mark.asyncio
    async def test_get_anime_stats_tool(self, mock_qdrant_client, mock_settings):
        """Test get_anime_stats FastMCP tool."""
        with (
            patch("src.mcp.server.qdrant_client", mock_qdrant_client),
            patch("src.mcp.server.settings", mock_settings),
        ):

            result = await get_anime_stats.fn()

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

        with (
            patch("src.mcp.server.qdrant_client", mock_qdrant_client),
            patch("src.mcp.server.settings", mock_settings),
        ):

            result = await get_anime_stats.fn()
            assert result["health_status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_initialize_mcp_server_success(self, mock_settings):
        """Test successful MCP server initialization."""
        mock_client = AsyncMock()
        mock_client.health_check.return_value = True

        with (
            patch("src.mcp.server.QdrantClient", return_value=mock_client),
            patch("src.mcp.server.settings", mock_settings),
        ):

            await initialize_mcp_server()

            # Verify client was created and health check performed
            mock_client.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_mcp_server_failure(self, mock_settings):
        """Test MCP server initialization failure."""
        mock_client = AsyncMock()
        mock_client.health_check.return_value = False

        with (
            patch("src.mcp.server.QdrantClient", return_value=mock_client),
            patch("src.mcp.server.settings", mock_settings),
        ):

            with pytest.raises(RuntimeError) as exc_info:
                await initialize_mcp_server()

            assert "Cannot initialize MCP server without database connection" in str(
                exc_info.value
            )

    def test_fastmcp_tool_decorators(self):
        """Test that FastMCP tool decorators are properly applied."""
        # Verify that our tools have the correct function signatures
        import inspect

        # Test search_anime signature
        sig = inspect.signature(search_anime.fn)
        assert "query" in sig.parameters
        assert "limit" in sig.parameters
        assert sig.parameters["limit"].default == 10

        # Test get_anime_details signature
        sig = inspect.signature(get_anime_details.fn)
        assert "anime_id" in sig.parameters

        # Test find_similar_anime signature
        sig = inspect.signature(find_similar_anime.fn)
        assert "anime_id" in sig.parameters
        assert "limit" in sig.parameters
        assert sig.parameters["limit"].default == 10

        # Test search_anime signature (now handles recommendations too)
        sig = inspect.signature(search_anime.fn)
        assert "query" in sig.parameters
        assert "limit" in sig.parameters
        assert "genres" in sig.parameters  # SearchIntent parameters
        assert "year_range" in sig.parameters
        assert "anime_types" in sig.parameters

        # Check optional parameters have correct defaults
        assert sig.parameters["limit"].default == 10
        assert sig.parameters["genres"].default is None
        assert sig.parameters["year_range"].default is None
        assert sig.parameters["anime_types"].default is None

    def test_fastmcp_docstrings(self):
        """Test that FastMCP tools have proper docstrings."""
        # Verify docstrings exist and contain key information via fn attribute
        assert search_anime.fn.__doc__ is not None
        assert "Search for anime using semantic search" in search_anime.fn.__doc__
        assert "Args:" in search_anime.fn.__doc__
        assert "Returns:" in search_anime.fn.__doc__

        assert get_anime_details.fn.__doc__ is not None
        assert (
            "Get detailed information about a specific anime"
            in get_anime_details.fn.__doc__
        )

        assert find_similar_anime.fn.__doc__ is not None
        assert "Find anime similar to a given anime" in find_similar_anime.fn.__doc__

        # Note: recommend_anime functionality now handled by search_anime

    @pytest.mark.asyncio
    async def test_resource_functions(self, mock_qdrant_client, mock_settings):
        """Test FastMCP resource functions."""
        from src.mcp.server import database_stats

        # Test database_stats resource
        with (
            patch("src.mcp.server.qdrant_client", mock_qdrant_client),
            patch("src.mcp.server.settings", mock_settings),
        ):

            stats_result = await database_stats.fn()
            assert isinstance(stats_result, str)
            assert "Anime Database Stats" in stats_result

    @pytest.mark.asyncio
    async def test_resource_functions_no_client(self):
        """Test resource functions when client is not initialized."""
        from src.mcp.server import database_schema, database_stats

        with patch("src.mcp.server.qdrant_client", None):
            stats_result = await database_stats.fn()
            assert "Database client not initialized" in stats_result

            schema_result = await database_schema.fn()
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
        assert (
            build_query(genres="Action", year=2023, anime_type="TV") == "Action 2023 TV"
        )

    def test_result_filtering_logic(self):
        """Test anime result filtering logic."""
        # Mock anime data
        test_anime = [
            {"anime_id": "1", "year": 2023, "type": "TV", "tags": ["Action", "Drama"]},
            {
                "anime_id": "2",
                "year": 2022,
                "type": "Movie",
                "tags": ["Comedy", "Romance"],
            },
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
        assert (
            len(filter_anime(test_anime, year=2023, anime_type="TV")) == 2
        )  # Multiple filters
        assert len(filter_anime(test_anime, genres="Romance")) == 1  # Specific genre


# ============================================================================
# IMAGE SEARCH TOOLS TESTS (PHASE 4)
# ============================================================================

from src.mcp.server import (
    find_visually_similar_anime,
    search_anime_by_image,
    search_multimodal_anime,
)


class TestMCPImageTools:
    """Test suite for MCP image search tools."""

    @pytest.fixture(autouse=True)
    def setup_mock_client(self):
        """Setup mock Qdrant client for testing."""
        import src.mcp.server

        # Save original client
        self.original_client = src.mcp.server.qdrant_client

        # Create mock client
        mock_client = MagicMock()
        mock_client._supports_multi_vector = True

        # Mock search methods
        mock_client.search_by_image = AsyncMock(
            return_value=[
                {
                    "anime_id": "test1",
                    "title": "Visual Match 1",
                    "visual_similarity_score": 0.95,
                },
                {
                    "anime_id": "test2",
                    "title": "Visual Match 2",
                    "visual_similarity_score": 0.88,
                },
            ]
        )

        mock_client.find_visually_similar_anime = AsyncMock(
            return_value=[
                {
                    "anime_id": "similar1",
                    "title": "Similar Anime 1",
                    "visual_similarity_score": 0.92,
                },
                {
                    "anime_id": "similar2",
                    "title": "Similar Anime 2",
                    "visual_similarity_score": 0.85,
                },
            ]
        )

        mock_client.search_multimodal = AsyncMock(
            return_value=[
                {
                    "anime_id": "multi1",
                    "title": "Multimodal Result 1",
                    "multimodal_score": 0.89,
                    "text_score": 0.85,
                    "image_score": 0.93,
                }
            ]
        )

        mock_client.search = AsyncMock(
            return_value=[
                {"anime_id": "text1", "title": "Text Result 1", "_score": 0.87}
            ]
        )

        # Replace global client
        src.mcp.server.qdrant_client = mock_client

        yield mock_client

        # Restore original client
        src.mcp.server.qdrant_client = self.original_client

    @pytest.mark.asyncio
    async def test_search_anime_by_image_success(self, setup_mock_client):
        """Test successful image search."""
        mock_client = setup_mock_client

        # Test data
        image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

        # Call the tool
        results = await search_anime_by_image.fn(image_data, limit=5)

        # Verify results
        assert len(results) == 2
        assert results[0]["title"] == "Visual Match 1"
        assert results[0]["visual_similarity_score"] == 0.95

        # Verify client was called correctly
        mock_client.search_by_image.assert_called_once_with(
            image_data=image_data, limit=5
        )

    @pytest.mark.asyncio
    async def test_search_anime_by_image_no_multi_vector(self, setup_mock_client):
        """Test image search when multi-vector is not enabled."""
        mock_client = setup_mock_client
        mock_client._supports_multi_vector = False

        image_data = "test_image_data"

        # Should raise runtime error
        with pytest.raises(RuntimeError, match="Multi-vector image search not enabled"):
            await search_anime_by_image.fn(image_data)

    @pytest.mark.asyncio
    async def test_search_anime_by_image_no_client(self):
        """Test image search when client is not initialized."""
        import src.mcp.server

        original_client = src.mcp.server.qdrant_client
        src.mcp.server.qdrant_client = None

        try:
            with pytest.raises(RuntimeError, match="Qdrant client not initialized"):
                await search_anime_by_image.fn("test_data")
        finally:
            src.mcp.server.qdrant_client = original_client

    @pytest.mark.asyncio
    async def test_search_anime_by_image_limit_validation(self, setup_mock_client):
        """Test limit validation for image search."""
        mock_client = setup_mock_client

        # Test limit clamping
        await search_anime_by_image.fn("test_data", limit=50)  # Should be clamped to 30
        mock_client.search_by_image.assert_called_with(image_data="test_data", limit=30)

        await search_anime_by_image.fn("test_data", limit=0)  # Should be clamped to 1
        mock_client.search_by_image.assert_called_with(image_data="test_data", limit=1)

    @pytest.mark.asyncio
    async def test_find_visually_similar_anime_success(self, setup_mock_client):
        """Test successful visual similarity search."""
        mock_client = setup_mock_client

        # Call the tool
        results = await find_visually_similar_anime.fn("reference_anime_123", limit=10)

        # Verify results
        assert len(results) == 2
        assert results[0]["title"] == "Similar Anime 1"
        assert results[0]["visual_similarity_score"] == 0.92

        # Verify client was called correctly
        mock_client.find_visually_similar_anime.assert_called_once_with(
            anime_id="reference_anime_123", limit=10
        )

    @pytest.mark.asyncio
    async def test_find_visually_similar_anime_no_multi_vector(self, setup_mock_client):
        """Test visual similarity when multi-vector is not enabled."""
        mock_client = setup_mock_client
        mock_client._supports_multi_vector = False

        # Should raise runtime error
        with pytest.raises(
            RuntimeError, match="Multi-vector visual similarity not enabled"
        ):
            await find_visually_similar_anime.fn("test_anime_id")

    @pytest.mark.asyncio
    async def test_search_multimodal_anime_with_image(self, setup_mock_client):
        """Test multimodal search with both text and image."""
        mock_client = setup_mock_client

        query = "mecha robots"
        image_data = "base64_image_data"
        text_weight = 0.6

        # Call the tool
        results = await search_multimodal_anime.fn(
            query=query, image_data=image_data, limit=15, text_weight=text_weight
        )

        # Verify results
        assert len(results) == 1
        assert results[0]["title"] == "Multimodal Result 1"
        assert results[0]["multimodal_score"] == 0.89

        # Verify client was called correctly
        mock_client.search_multimodal.assert_called_once_with(
            query=query, image_data=image_data, limit=15, text_weight=text_weight
        )

    @pytest.mark.asyncio
    async def test_search_multimodal_anime_text_only(self, setup_mock_client):
        """Test multimodal search with text only."""
        mock_client = setup_mock_client

        query = "romantic comedy"

        # Call the tool without image
        results = await search_multimodal_anime.fn(query=query, limit=10)

        # Should fall back to regular search when no image provided
        mock_client.search.assert_called_once_with(query=query, limit=10)

    @pytest.mark.asyncio
    async def test_search_multimodal_anime_no_multi_vector(self, setup_mock_client):
        """Test multimodal search fallback when multi-vector is not enabled."""
        mock_client = setup_mock_client
        mock_client._supports_multi_vector = False

        query = "action anime"
        image_data = "test_image"

        # Call the tool
        results = await search_multimodal_anime.fn(query=query, image_data=image_data)

        # Should fall back to text search
        mock_client.search.assert_called_once_with(query=query, limit=10)
        assert len(results) == 1
        assert results[0]["title"] == "Text Result 1"

    @pytest.mark.asyncio
    async def test_image_search_error_handling(self, setup_mock_client):
        """Test error handling in image search tools."""
        mock_client = setup_mock_client

        # Mock client method to raise exception
        mock_client.search_by_image.side_effect = Exception("Database error")

        # Should raise RuntimeError with appropriate message
        with pytest.raises(RuntimeError, match="Image search failed: Database error"):
            await search_anime_by_image.fn("test_image")

    @pytest.mark.asyncio
    async def test_visual_similarity_error_handling(self, setup_mock_client):
        """Test error handling in visual similarity search."""
        mock_client = setup_mock_client

        # Mock client method to raise exception
        mock_client.find_visually_similar_anime.side_effect = Exception(
            "Similarity error"
        )

        # Should raise RuntimeError with appropriate message
        with pytest.raises(
            RuntimeError, match="Visual similarity search failed: Similarity error"
        ):
            await find_visually_similar_anime.fn("test_anime_id")

    @pytest.mark.asyncio
    async def test_multimodal_search_error_handling(self, setup_mock_client):
        """Test error handling in multimodal search."""
        mock_client = setup_mock_client

        # Mock client method to raise exception
        mock_client.search_multimodal.side_effect = Exception("Multimodal error")

        # Should raise RuntimeError with appropriate message
        with pytest.raises(
            RuntimeError, match="Multimodal search failed: Multimodal error"
        ):
            await search_multimodal_anime.fn("test query", "test_image")


class TestMCPImageToolsIntegration:
    """Integration tests for MCP image tools."""

    def test_image_tools_parameter_types(self):
        """Test that image tools have correct parameter types."""
        import inspect

        # Test search_anime_by_image signature
        sig = inspect.signature(search_anime_by_image.fn)
        assert sig.parameters["image_data"].annotation == str
        assert sig.parameters["limit"].annotation == int
        assert sig.parameters["limit"].default == 10

        # Test find_visually_similar_anime signature
        sig = inspect.signature(find_visually_similar_anime.fn)
        assert sig.parameters["anime_id"].annotation == str
        assert sig.parameters["limit"].annotation == int
        assert sig.parameters["limit"].default == 10

        # Test search_multimodal_anime signature
        sig = inspect.signature(search_multimodal_anime.fn)
        assert sig.parameters["query"].annotation == str
        assert "Optional" in str(sig.parameters["image_data"].annotation)
        assert sig.parameters["limit"].annotation == int
        assert sig.parameters["text_weight"].annotation == float
        assert sig.parameters["text_weight"].default == 0.7


# ============================================================================
# COMPREHENSIVE COVERAGE TESTS FOR 100% COVERAGE
# ============================================================================


class TestMCPServerCompleteCoverage:
    """Test suite to achieve 100% coverage of MCP server functionality."""

    @pytest.mark.asyncio
    async def test_initialize_qdrant_client_function(self):
        """Test initialize_qdrant_client function."""
        import src.mcp.server
        from src.mcp.server import initialize_qdrant_client

        # Save original client
        original_client = src.mcp.server.qdrant_client

        try:
            # Reset client to None
            src.mcp.server.qdrant_client = None

            with patch("src.mcp.server.QdrantClient") as mock_qdrant_class:
                mock_client = Mock()
                mock_qdrant_class.return_value = mock_client

                # Call function
                initialize_qdrant_client()

                # Verify client was created
                assert src.mcp.server.qdrant_client == mock_client
                mock_qdrant_class.assert_called_once()

        finally:
            # Restore original client
            src.mcp.server.qdrant_client = original_client

    @pytest.mark.asyncio
    async def test_initialize_qdrant_client_already_initialized(self):
        """Test initialize_qdrant_client when client already exists."""
        import src.mcp.server
        from src.mcp.server import initialize_qdrant_client

        # Save original client
        original_client = src.mcp.server.qdrant_client

        try:
            # Set existing client
            existing_client = Mock()
            src.mcp.server.qdrant_client = existing_client

            with patch("src.mcp.server.QdrantClient") as mock_qdrant_class:
                # Call function
                initialize_qdrant_client()

                # Verify existing client wasn't replaced
                assert src.mcp.server.qdrant_client == existing_client
                mock_qdrant_class.assert_not_called()

        finally:
            # Restore original client
            src.mcp.server.qdrant_client = original_client

    def test_parse_arguments_default(self):
        """Test parse_arguments with default values."""
        from src.mcp.server import parse_arguments

        with patch("sys.argv", ["mcp_server.py"]):
            args = parse_arguments()

            # Check default values are set from settings
            assert hasattr(args, "mode")
            assert hasattr(args, "host")
            assert hasattr(args, "port")
            assert hasattr(args, "verbose")
            assert args.verbose is False

    def test_parse_arguments_custom(self):
        """Test parse_arguments with custom values."""
        from src.mcp.server import parse_arguments

        with patch(
            "sys.argv",
            [
                "mcp_server.py",
                "--mode",
                "http",
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
                "--verbose",
            ],
        ):
            args = parse_arguments()

            assert args.mode == "http"
            assert args.host == "127.0.0.1"
            assert args.port == 8080
            assert args.verbose is True

    def test_build_search_filters_empty(self):
        """Test _build_search_filters with empty inputs."""
        from src.mcp.server import _build_search_filters

        # Test with all None values
        result = _build_search_filters()
        assert result is None

        # Test with empty lists
        result = _build_search_filters(
            genres=[],
            year_range=[],
            anime_types=[],
            studios=[],
            exclusions=[],
            mood_keywords=[],
        )
        assert result is None

    def test_build_search_filters_single_year(self):
        """Test _build_search_filters with single year values."""
        from src.mcp.server import _build_search_filters

        # Test with start year only
        result = _build_search_filters(year_range=[2020, None])
        expected = {"year": {"gte": 2020}}
        assert result == expected

        # Test with end year only
        result = _build_search_filters(year_range=[None, 2023])
        expected = {"year": {"lte": 2023}}
        assert result == expected

    def test_build_search_filters_exclusions(self):
        """Test _build_search_filters with exclusions."""
        from src.mcp.server import _build_search_filters

        result = _build_search_filters(exclusions=["Horror", "Ecchi"])
        expected = {"exclude_tags": ["Horror", "Ecchi"]}
        assert result == expected

    def test_build_search_filters_mood_keywords_combined(self):
        """Test _build_search_filters combining mood keywords with existing tags."""
        from src.mcp.server import _build_search_filters

        result = _build_search_filters(
            genres=["Action", "Comedy"], mood_keywords=["dark", "serious"]
        )
        expected = {"tags": {"any": ["Action", "Comedy", "dark", "serious"]}}
        assert result == expected

    @pytest.mark.asyncio
    async def test_search_anime_limit_edge_cases(self, mock_qdrant_client):
        """Test search_anime limit validation edge cases."""
        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            # Test limit over maximum (should clamp to 50)
            await search_anime.fn(query="test", limit=100)
            mock_qdrant_client.search.assert_called_with(
                query="test", limit=50, filters=None
            )

            # Test limit under minimum (should clamp to 1)
            await search_anime.fn(query="test", limit=-5)
            mock_qdrant_client.search.assert_called_with(
                query="test", limit=1, filters=None
            )

    @pytest.mark.asyncio
    async def test_find_similar_anime_limit_edge_cases(self, mock_qdrant_client):
        """Test find_similar_anime limit validation edge cases."""
        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            # Test limit over maximum (should clamp to 20)
            await find_similar_anime.fn(anime_id="test", limit=100)
            mock_qdrant_client.find_similar.assert_called_with(
                anime_id="test", limit=20
            )

            # Test limit under minimum (should clamp to 1)
            await find_similar_anime.fn(anime_id="test", limit=-5)
            mock_qdrant_client.find_similar.assert_called_with(anime_id="test", limit=1)

    @pytest.mark.asyncio
    async def test_search_anime_by_image_limit_edge_cases(self, mock_qdrant_client):
        """Test search_anime_by_image limit validation edge cases."""
        mock_qdrant_client._supports_multi_vector = True

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            # Test limit over maximum (should clamp to 30)
            await search_anime_by_image.fn(image_data="test", limit=100)
            mock_qdrant_client.search_by_image.assert_called_with(
                image_data="test", limit=30
            )

            # Test limit under minimum (should clamp to 1)
            await search_anime_by_image.fn(image_data="test", limit=-5)
            mock_qdrant_client.search_by_image.assert_called_with(
                image_data="test", limit=1
            )

    @pytest.mark.asyncio
    async def test_find_visually_similar_anime_limit_edge_cases(
        self, mock_qdrant_client
    ):
        """Test find_visually_similar_anime limit validation edge cases."""
        mock_qdrant_client._supports_multi_vector = True

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            # Test limit over maximum (should clamp to 20)
            await find_visually_similar_anime.fn(anime_id="test", limit=100)
            mock_qdrant_client.find_visually_similar_anime.assert_called_with(
                anime_id="test", limit=20
            )

            # Test limit under minimum (should clamp to 1)
            await find_visually_similar_anime.fn(anime_id="test", limit=-5)
            mock_qdrant_client.find_visually_similar_anime.assert_called_with(
                anime_id="test", limit=1
            )

    @pytest.mark.asyncio
    async def test_search_multimodal_anime_limit_and_weight_validation(
        self, mock_qdrant_client
    ):
        """Test search_multimodal_anime parameter validation."""
        mock_qdrant_client._supports_multi_vector = True

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            # Test limit over maximum (should clamp to 25)
            await search_multimodal_anime.fn(
                query="test", image_data="test", limit=100, text_weight=0.8
            )
            mock_qdrant_client.search_multimodal.assert_called_with(
                query="test", image_data="test", limit=25, text_weight=0.8
            )

            # Test limit under minimum (should clamp to 1)
            await search_multimodal_anime.fn(
                query="test", image_data="test", limit=-5, text_weight=0.3
            )
            mock_qdrant_client.search_multimodal.assert_called_with(
                query="test", image_data="test", limit=1, text_weight=0.3
            )

            # Test text_weight over maximum (should clamp to 1.0)
            await search_multimodal_anime.fn(
                query="test", image_data="test", limit=10, text_weight=1.5
            )
            mock_qdrant_client.search_multimodal.assert_called_with(
                query="test", image_data="test", limit=10, text_weight=1.0
            )

            # Test text_weight under minimum (should clamp to 0.0)
            await search_multimodal_anime.fn(
                query="test", image_data="test", limit=10, text_weight=-0.5
            )
            mock_qdrant_client.search_multimodal.assert_called_with(
                query="test", image_data="test", limit=10, text_weight=0.0
            )

    @pytest.mark.asyncio
    async def test_search_multimodal_anime_text_only_fallback(self, mock_qdrant_client):
        """Test search_multimodal_anime fallback to text-only search."""
        mock_qdrant_client._supports_multi_vector = True

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            # Test with no image data - should use regular search
            await search_multimodal_anime.fn(query="test", limit=10)
            mock_qdrant_client.search.assert_called_with(query="test", limit=10)

    @pytest.mark.asyncio
    async def test_search_multimodal_anime_multi_vector_disabled_fallback(
        self, mock_qdrant_client
    ):
        """Test search_multimodal_anime fallback when multi-vector disabled."""
        mock_qdrant_client._supports_multi_vector = False

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            # Test with image data but multi-vector disabled - should fallback to text search
            await search_multimodal_anime.fn(
                query="test", image_data="test_image", limit=10
            )
            mock_qdrant_client.search.assert_called_with(query="test", limit=10)

    @pytest.mark.asyncio
    async def test_get_anime_details_not_found_error(self, mock_qdrant_client):
        """Test get_anime_details when anime not found."""
        mock_qdrant_client.get_by_id.return_value = None

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            with pytest.raises(RuntimeError) as exc_info:
                await get_anime_details.fn(anime_id="nonexistent")

            assert "Anime not found: nonexistent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_database_stats_resource_error_handling(self):
        """Test database_stats resource error handling."""
        from src.mcp.server import database_stats

        with patch("src.mcp.server.qdrant_client", None):
            result = await database_stats.fn()
            assert "Database client not initialized" in result

    @pytest.mark.asyncio
    async def test_database_stats_resource_exception_handling(self, mock_qdrant_client):
        """Test database_stats resource exception handling."""
        from src.mcp.server import database_stats

        # Mock get_anime_stats to raise an exception
        with patch("src.mcp.server.get_anime_stats") as mock_get_stats:
            mock_get_stats.fn = AsyncMock(side_effect=Exception("Stats error"))

            with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
                result = await database_stats.fn()
                assert "Error getting stats: Stats error" in result

    @pytest.mark.asyncio
    async def test_database_schema_resource(self):
        """Test database_schema resource function."""
        from src.mcp.server import database_schema

        result = await database_schema.fn()

        assert isinstance(result, str)
        assert "Anime Database Schema" in result
        assert "fields" in result
        assert "anime_id" in result
        assert "title" in result

    @pytest.mark.asyncio
    async def test_search_anime_with_complex_filters(self, mock_qdrant_client):
        """Test search_anime with all filter types."""
        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            await search_anime.fn(
                query="test",
                limit=10,
                genres=["Action", "Comedy"],
                year_range=[2020, 2023],
                anime_types=["TV", "Movie"],
                studios=["Mappa", "Wit Studio"],
                exclusions=["Horror", "Ecchi"],
                mood_keywords=["dark", "serious"],
            )

            # Should be called with complex filters
            mock_qdrant_client.search.assert_called_once()
            call_args = mock_qdrant_client.search.call_args
            assert call_args[1]["query"] == "test"
            assert call_args[1]["limit"] == 10
            assert call_args[1]["filters"] is not None

    @pytest.mark.asyncio
    async def test_main_function_components(self):
        """Test main function components."""
        from src.mcp.server import main

        # Test that main function can be called and handles arguments
        with patch("src.mcp.server.parse_arguments") as mock_parse:
            with patch("src.mcp.server.logging.basicConfig") as mock_logging:
                with patch("asyncio.run") as mock_asyncio_run:
                    with patch("src.mcp.server.mcp") as mock_mcp:

                        # Mock arguments
                        mock_args = Mock()
                        mock_args.verbose = False
                        mock_args.mode = "stdio"
                        mock_parse.return_value = mock_args

                        # Mock settings
                        with patch("src.mcp.server.settings") as mock_settings:
                            mock_settings.log_level = "INFO"
                            mock_settings.log_format = "%(message)s"
                            mock_settings.qdrant_collection_name = "test"
                            mock_settings.qdrant_url = "http://localhost:6333"

                            try:
                                main()
                            except SystemExit:
                                pass  # Expected from mcp.run()

                            # Verify function calls
                            mock_parse.assert_called_once()
                            mock_logging.assert_called_once()
                            mock_asyncio_run.assert_called_once()

    def test_main_function_http_mode(self):
        """Test main function with HTTP mode."""
        from src.mcp.server import main

        with patch("src.mcp.server.parse_arguments") as mock_parse:
            with patch("src.mcp.server.logging.basicConfig"):
                with patch("asyncio.run"):
                    with patch("src.mcp.server.mcp") as mock_mcp:

                        # Mock arguments for HTTP mode
                        mock_args = Mock()
                        mock_args.verbose = True
                        mock_args.mode = "http"
                        mock_args.host = "localhost"
                        mock_args.port = 8080
                        mock_parse.return_value = mock_args

                        # Mock settings
                        with patch("src.mcp.server.settings") as mock_settings:
                            mock_settings.log_level = "INFO"
                            mock_settings.log_format = "%(message)s"
                            mock_settings.qdrant_collection_name = "test"
                            mock_settings.qdrant_url = "http://localhost:6333"

                            try:
                                main()
                            except SystemExit:
                                pass  # Expected from mcp.run()

                            # Verify mcp.run was called with SSE transport (http mode uses SSE)
                            mock_mcp.run.assert_called_with(
                                transport="sse", host="localhost", port=8080
                            )

    def test_main_function_sse_mode(self):
        """Test main function with SSE mode."""
        from src.mcp.server import main

        with patch("src.mcp.server.parse_arguments") as mock_parse:
            with patch("src.mcp.server.logging.basicConfig"):
                with patch("asyncio.run"):
                    with patch("src.mcp.server.mcp") as mock_mcp:

                        # Mock arguments for SSE mode
                        mock_args = Mock()
                        mock_args.verbose = False
                        mock_args.mode = "sse"
                        mock_args.host = "127.0.0.1"
                        mock_args.port = 3001
                        mock_parse.return_value = mock_args

                        # Mock settings
                        with patch("src.mcp.server.settings") as mock_settings:
                            mock_settings.log_level = "DEBUG"
                            mock_settings.log_format = "%(message)s"
                            mock_settings.qdrant_collection_name = "test"
                            mock_settings.qdrant_url = "http://localhost:6333"

                            try:
                                main()
                            except SystemExit:
                                pass  # Expected from mcp.run()

                            # Verify mcp.run was called with SSE transport
                            mock_mcp.run.assert_called_with(
                                transport="sse", host="127.0.0.1", port=3001
                            )

    def test_main_function_streamable_mode(self):
        """Test main function with streamable mode."""
        from src.mcp.server import main

        with patch("src.mcp.server.parse_arguments") as mock_parse:
            with patch("src.mcp.server.logging.basicConfig"):
                with patch("asyncio.run"):
                    with patch("src.mcp.server.mcp") as mock_mcp:

                        # Mock arguments for streamable mode
                        mock_args = Mock()
                        mock_args.verbose = False
                        mock_args.mode = "streamable"
                        mock_args.host = "0.0.0.0"
                        mock_args.port = 8000
                        mock_parse.return_value = mock_args

                        # Mock settings
                        with patch("src.mcp.server.settings") as mock_settings:
                            mock_settings.log_level = "INFO"
                            mock_settings.log_format = "%(message)s"
                            mock_settings.qdrant_collection_name = "test"
                            mock_settings.qdrant_url = "http://localhost:6333"

                            try:
                                main()
                            except SystemExit:
                                pass  # Expected from mcp.run()

                            # Verify mcp.run was called with streamable transport
                            mock_mcp.run.assert_called_with(
                                transport="streamable", host="0.0.0.0", port=8000
                            )

    def test_main_function_keyboard_interrupt(self):
        """Test main function handles KeyboardInterrupt."""
        from src.mcp.server import main

        with patch("src.mcp.server.parse_arguments") as mock_parse:
            with patch("src.mcp.server.logging.basicConfig"):
                with patch("asyncio.run"):
                    with patch("src.mcp.server.mcp") as mock_mcp:

                        # Mock KeyboardInterrupt
                        mock_mcp.run.side_effect = KeyboardInterrupt()

                        mock_args = Mock()
                        mock_args.verbose = False
                        mock_args.mode = "stdio"
                        mock_parse.return_value = mock_args

                        with patch("src.mcp.server.settings") as mock_settings:
                            mock_settings.log_level = "INFO"
                            mock_settings.log_format = "%(message)s"
                            mock_settings.qdrant_collection_name = "test"
                            mock_settings.qdrant_url = "http://localhost:6333"

                            # Should handle KeyboardInterrupt gracefully
                            main()

    def test_main_function_exception_handling(self):
        """Test main function handles general exceptions."""
        from src.mcp.server import main

        with patch("src.mcp.server.parse_arguments") as mock_parse:
            with patch("src.mcp.server.logging.basicConfig"):
                with patch("asyncio.run"):
                    with patch("src.mcp.server.mcp") as mock_mcp:

                        # Mock general exception
                        mock_mcp.run.side_effect = Exception("Server error")

                        mock_args = Mock()
                        mock_args.verbose = False
                        mock_args.mode = "stdio"
                        mock_parse.return_value = mock_args

                        with patch("src.mcp.server.settings") as mock_settings:
                            mock_settings.log_level = "INFO"
                            mock_settings.log_format = "%(message)s"
                            mock_settings.qdrant_collection_name = "test"
                            mock_settings.qdrant_url = "http://localhost:6333"

                            # Should raise the exception
                            with pytest.raises(Exception, match="Server error"):
                                main()

    @pytest.mark.asyncio
    async def test_init_and_run_function_success(self):
        """Test the nested init_and_run function success path."""
        from src.mcp.server import main

        with patch("src.mcp.server.initialize_mcp_server") as mock_init:
            mock_init.return_value = None  # Successful initialization

            # Extract and test the init_and_run function
            with patch("src.mcp.server.parse_arguments") as mock_parse:
                with patch("src.mcp.server.logging.basicConfig"):
                    with patch("asyncio.run") as mock_asyncio_run:
                        with patch("src.mcp.server.mcp"):

                            mock_args = Mock()
                            mock_args.verbose = False
                            mock_args.mode = "stdio"
                            mock_parse.return_value = mock_args

                            with patch("src.mcp.server.settings") as mock_settings:
                                mock_settings.log_level = "INFO"
                                mock_settings.log_format = "%(message)s"
                                mock_settings.qdrant_collection_name = "test"
                                mock_settings.qdrant_url = "http://localhost:6333"

                                try:
                                    main()
                                except SystemExit:
                                    pass

                                # Verify asyncio.run was called (which runs init_and_run)
                                mock_asyncio_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_and_run_function_failure(self):
        """Test the nested init_and_run function failure path."""
        from src.mcp.server import main

        with patch("src.mcp.server.initialize_mcp_server") as mock_init:
            mock_init.side_effect = Exception("Initialization failed")

            with patch("src.mcp.server.parse_arguments") as mock_parse:
                with patch("src.mcp.server.logging.basicConfig"):
                    with patch("asyncio.run") as mock_asyncio_run:
                        with patch("src.mcp.server.mcp"):

                            mock_args = Mock()
                            mock_args.verbose = False
                            mock_args.mode = "stdio"
                            mock_parse.return_value = mock_args

                            with patch("src.mcp.server.settings") as mock_settings:
                                mock_settings.log_level = "INFO"
                                mock_settings.log_format = "%(message)s"
                                mock_settings.qdrant_collection_name = "test"
                                mock_settings.qdrant_url = "http://localhost:6333"

                                # Should raise exception from init_and_run
                                with pytest.raises(Exception):
                                    main()


class TestEnhancedSearchAnime:
    """Test enhanced search_anime tool with SearchIntent parameters."""

    @pytest.fixture
    def mock_qdrant_client_enhanced(self):
        """Mock Qdrant client for enhanced search testing."""
        mock_client = MagicMock()
        mock_anime_results = [
            {
                "anime_id": "test_1",
                "title": "Attack on Titan",
                "type": "TV",
                "year": 2013,
                "tags": ["Action", "Drama", "Fantasy"],
                "studios": ["Mappa", "TOEI Animation"],
                "synopsis": "Humanity fights giant titans",
                "_score": 0.95,
            },
            {
                "anime_id": "test_2",
                "title": "Demon Slayer",
                "type": "TV",
                "year": 2019,
                "tags": ["Action", "Supernatural"],
                "studios": ["Ufotable"],
                "synopsis": "Boy becomes demon slayer",
                "_score": 0.90,
            },
        ]
        mock_client.search = AsyncMock(return_value=mock_anime_results)
        return mock_client

    @pytest.mark.asyncio
    async def test_search_anime_backward_compatibility(
        self, mock_qdrant_client_enhanced
    ):
        """Test that existing search_anime functionality remains intact."""
        from src.mcp.server import search_anime

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client_enhanced):
            result = await search_anime.fn(query="action anime", limit=10)

            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["title"] == "Attack on Titan"

            mock_qdrant_client_enhanced.search.assert_called_once_with(
                query="action anime",
                limit=10,
                filters=None,
            )

    @pytest.mark.asyncio
    async def test_search_anime_with_genres_filter(self, mock_qdrant_client_enhanced):
        """Test search_anime with genres parameter."""
        from src.mcp.server import search_anime

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client_enhanced):
            result = await search_anime.fn(
                query="anime", limit=5, genres=["Action", "Drama"]
            )

            assert isinstance(result, list)
            mock_qdrant_client_enhanced.search.assert_called_once()

            call_args = mock_qdrant_client_enhanced.search.call_args
            assert call_args[1]["query"] == "anime"
            assert call_args[1]["limit"] == 5
            assert "filters" in call_args[1]

    @pytest.mark.asyncio
    async def test_search_anime_with_year_range(self, mock_qdrant_client_enhanced):
        """Test search_anime with year_range parameter."""
        from src.mcp.server import search_anime

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client_enhanced):
            result = await search_anime.fn(
                query="mecha anime", limit=10, year_range=[2020, 2023]
            )

            assert isinstance(result, list)
            call_args = mock_qdrant_client_enhanced.search.call_args
            assert "filters" in call_args[1]

    @pytest.mark.asyncio
    async def test_search_anime_with_anime_types(self, mock_qdrant_client_enhanced):
        """Test search_anime with anime_types parameter."""
        from src.mcp.server import search_anime

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client_enhanced):
            result = await search_anime.fn(query="anime", anime_types=["TV", "Movie"])

            assert isinstance(result, list)
            call_args = mock_qdrant_client_enhanced.search.call_args
            assert "filters" in call_args[1]

    @pytest.mark.asyncio
    async def test_search_anime_with_studios(self, mock_qdrant_client_enhanced):
        """Test search_anime with studios parameter."""
        from src.mcp.server import search_anime

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client_enhanced):
            result = await search_anime.fn(
                query="anime", studios=["Mappa", "Studio Ghibli"]
            )

            assert isinstance(result, list)
            call_args = mock_qdrant_client_enhanced.search.call_args
            assert "filters" in call_args[1]

    @pytest.mark.asyncio
    async def test_search_anime_with_exclusions(self, mock_qdrant_client_enhanced):
        """Test search_anime with exclusions parameter."""
        from src.mcp.server import search_anime

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client_enhanced):
            result = await search_anime.fn(
                query="anime", exclusions=["Horror", "Ecchi"]
            )

            assert isinstance(result, list)
            call_args = mock_qdrant_client_enhanced.search.call_args
            assert "filters" in call_args[1]

    @pytest.mark.asyncio
    async def test_search_anime_with_mood_keywords(self, mock_qdrant_client_enhanced):
        """Test search_anime with mood_keywords parameter."""
        from src.mcp.server import search_anime

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client_enhanced):
            result = await search_anime.fn(
                query="anime", mood_keywords=["dark", "philosophical"]
            )

            assert isinstance(result, list)
            call_args = mock_qdrant_client_enhanced.search.call_args
            assert "filters" in call_args[1]

    @pytest.mark.asyncio
    async def test_search_anime_with_all_parameters(self, mock_qdrant_client_enhanced):
        """Test search_anime with all SearchIntent parameters."""
        from src.mcp.server import search_anime

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client_enhanced):
            result = await search_anime.fn(
                query="mecha anime",
                limit=5,
                genres=["Action", "Mecha"],
                year_range=[2015, 2023],
                anime_types=["TV"],
                studios=["Sunrise"],
                exclusions=["Ecchi"],
                mood_keywords=["epic", "dramatic"],
            )

            assert isinstance(result, list)
            call_args = mock_qdrant_client_enhanced.search.call_args
            assert call_args[1]["query"] == "mecha anime"
            assert call_args[1]["limit"] == 5
            assert "filters" in call_args[1]


class TestMCPServerMissingCoverage:
    """Test specific missing coverage lines to reach 100%."""

    @pytest.mark.asyncio
    async def test_search_anime_mood_keywords_no_genres(self):
        """Test search_anime mood keywords when no existing tags - covers line 86."""
        from src.mcp.server import _build_search_filters

        # Test the specific case where combined_tags = mood_keywords (line 86)
        result = _build_search_filters(mood_keywords=["dark", "serious"])
        expected = {"tags": {"any": ["dark", "serious"]}}
        assert result == expected

    @pytest.mark.asyncio
    async def test_all_tools_runtime_error_not_initialized(self):
        """Test RuntimeError for all tools when client not initialized - covers lines 161, 189, 213, 297, 343."""
        from src.mcp.server import (
            find_similar_anime,
            get_anime_details,
            get_anime_stats,
            search_anime,
        )

        with patch("src.mcp.server.qdrant_client", None):
            # Test search_anime - line 161
            with pytest.raises(RuntimeError, match="Qdrant client not initialized"):
                await search_anime.fn(query="test")

            # Test get_anime_details - line 189
            with pytest.raises(RuntimeError, match="Qdrant client not initialized"):
                await get_anime_details.fn(anime_id="test")

            # Test find_similar_anime - line 213
            with pytest.raises(RuntimeError, match="Qdrant client not initialized"):
                await find_similar_anime.fn(anime_id="test")

            # Test get_anime_stats - line 297
            with pytest.raises(RuntimeError, match="Qdrant client not initialized"):
                await get_anime_stats.fn()

    @pytest.mark.asyncio
    async def test_find_similar_anime_exception_handling(self):
        """Test find_similar_anime exception handling - covers lines 200-202."""
        from src.mcp.server import find_similar_anime

        mock_client = AsyncMock()
        mock_client.find_similar.side_effect = Exception("Similarity search failed")

        with patch("src.mcp.server.qdrant_client", mock_client):
            with pytest.raises(RuntimeError) as exc_info:
                await find_similar_anime.fn(anime_id="test123")

            assert "Similarity search failed: Similarity search failed" in str(
                exc_info.value
            )

    @pytest.mark.asyncio
    async def test_get_anime_stats_exception_handling(self):
        """Test get_anime_stats exception handling - covers lines 233-235."""
        from src.mcp.server import get_anime_stats

        mock_client = AsyncMock()
        mock_client.get_stats.side_effect = Exception("Stats retrieval failed")

        with patch("src.mcp.server.qdrant_client", mock_client):
            with patch("src.mcp.server.settings") as mock_settings:
                mock_settings.qdrant_url = "http://localhost:6333"
                mock_settings.qdrant_collection_name = "anime_database"
                mock_settings.fastembed_model = "BAAI/bge-small-en-v1.5"

                with pytest.raises(RuntimeError) as exc_info:
                    await get_anime_stats.fn()

                assert "Failed to get stats: Stats retrieval failed" in str(
                    exc_info.value
                )

    @pytest.mark.asyncio
    async def test_image_search_tools_runtime_errors(self):
        """Test RuntimeError for image search tools when client not initialized - covers line 343."""
        from src.mcp.server import find_visually_similar_anime, search_anime_by_image

        with patch("src.mcp.server.qdrant_client", None):
            # Test search_anime_by_image
            with pytest.raises(RuntimeError, match="Qdrant client not initialized"):
                await search_anime_by_image.fn(image_data="test")

            # Test find_visually_similar_anime
            with pytest.raises(RuntimeError, match="Qdrant client not initialized"):
                await find_visually_similar_anime.fn(anime_id="test")
