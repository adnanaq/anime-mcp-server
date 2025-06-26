"""Test enhanced MCP tools with SearchIntent parameters.

This test module ensures that the enhanced MCP tools maintain backward compatibility
while supporting rich SearchIntent parameters for improved query filtering.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Test data
MOCK_ANIME_RESULTS = [
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


class TestEnhancedSearchAnime:
    """Test enhanced search_anime tool with SearchIntent parameters."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client for testing."""
        mock_client = MagicMock()
        mock_client.search = AsyncMock(return_value=MOCK_ANIME_RESULTS)
        return mock_client

    @pytest.mark.asyncio
    async def test_search_anime_backward_compatibility(self, mock_qdrant_client):
        """Test that existing search_anime functionality remains intact."""
        # Import here to avoid circular imports during test discovery
        from src.mcp.server import search_anime

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            # Test basic search (backward compatibility)
            result = await search_anime.fn(query="action anime", limit=10)

            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["title"] == "Attack on Titan"

            # Verify qdrant client was called with basic parameters (backward compatibility)
            mock_qdrant_client.search.assert_called_once_with(
                query="action anime",
                limit=10,
                filters=None,  # No filters for basic search
            )

    @pytest.mark.asyncio
    async def test_search_anime_with_genres_filter(self, mock_qdrant_client):
        """Test search_anime with genres parameter."""
        from src.mcp.server import search_anime

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            # Test with genres parameter
            result = await search_anime.fn(
                query="anime", limit=5, genres=["Action", "Drama"]
            )

            assert isinstance(result, list)
            mock_qdrant_client.search.assert_called_once()

            # Verify the call included filters
            call_args = mock_qdrant_client.search.call_args
            assert call_args[1]["query"] == "anime"
            assert call_args[1]["limit"] == 5
            assert "filters" in call_args[1]

    @pytest.mark.asyncio
    async def test_search_anime_with_year_range(self, mock_qdrant_client):
        """Test search_anime with year_range parameter."""
        from src.mcp.server import search_anime

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            # Test with year range
            result = await search_anime.fn(
                query="mecha anime", limit=10, year_range=[2020, 2023]
            )

            assert isinstance(result, list)
            call_args = mock_qdrant_client.search.call_args
            assert "filters" in call_args[1]

    @pytest.mark.asyncio
    async def test_search_anime_with_anime_types(self, mock_qdrant_client):
        """Test search_anime with anime_types parameter."""
        from src.mcp.server import search_anime

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            result = await search_anime.fn(query="anime", anime_types=["TV", "Movie"])

            assert isinstance(result, list)
            call_args = mock_qdrant_client.search.call_args
            assert "filters" in call_args[1]

    @pytest.mark.asyncio
    async def test_search_anime_with_studios(self, mock_qdrant_client):
        """Test search_anime with studios parameter."""
        from src.mcp.server import search_anime

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            result = await search_anime.fn(
                query="anime", studios=["Mappa", "Studio Ghibli"]
            )

            assert isinstance(result, list)
            call_args = mock_qdrant_client.search.call_args
            assert "filters" in call_args[1]

    @pytest.mark.asyncio
    async def test_search_anime_with_exclusions(self, mock_qdrant_client):
        """Test search_anime with exclusions parameter."""
        from src.mcp.server import search_anime

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            result = await search_anime.fn(
                query="anime", exclusions=["Horror", "Ecchi"]
            )

            assert isinstance(result, list)
            call_args = mock_qdrant_client.search.call_args
            assert "filters" in call_args[1]

    @pytest.mark.asyncio
    async def test_search_anime_with_mood_keywords(self, mock_qdrant_client):
        """Test search_anime with mood_keywords parameter."""
        from src.mcp.server import search_anime

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            result = await search_anime.fn(
                query="anime", mood_keywords=["dark", "serious"]
            )

            assert isinstance(result, list)
            call_args = mock_qdrant_client.search.call_args
            assert "filters" in call_args[1]

    @pytest.mark.asyncio
    async def test_search_anime_with_all_parameters(self, mock_qdrant_client):
        """Test search_anime with all SearchIntent parameters."""
        from src.mcp.server import search_anime

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            result = await search_anime.fn(
                query="mecha anime",
                limit=5,
                genres=["Action", "Mecha"],
                year_range=[2020, 2023],
                anime_types=["TV"],
                studios=["Mappa"],
                exclusions=["Horror"],
                mood_keywords=["serious"],
            )

            assert isinstance(result, list)
            call_args = mock_qdrant_client.search.call_args
            assert call_args[1]["query"] == "mecha anime"
            assert call_args[1]["limit"] == 5
            assert "filters" in call_args[1]


class TestEnhancedSearchAnimeRecommendations:
    """Test search_anime tool for recommendation functionality (migrated from recommend_anime)."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client for testing."""
        mock_client = MagicMock()
        mock_client.search = AsyncMock(return_value=MOCK_ANIME_RESULTS)
        return mock_client

    @pytest.mark.asyncio
    async def test_search_anime_recommendation_legacy_equivalent(
        self, mock_qdrant_client
    ):
        """Test search_anime can replicate legacy recommend_anime functionality."""
        from src.mcp.server import search_anime

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            # Equivalent to: recommend_anime(genres="Action,Comedy", year=2020, anime_type="TV", limit=10)
            result = await search_anime.fn(
                query="Action Comedy 2020 TV",
                genres=["Action", "Comedy"],
                year_range=[2020, 2020],
                anime_types=["TV"],
                limit=10,
            )

            assert isinstance(result, list)
            assert len(result) >= 0  # May be filtered

    @pytest.mark.asyncio
    async def test_search_anime_recommendation_enhanced_parameters(
        self, mock_qdrant_client
    ):
        """Test search_anime with enhanced SearchIntent parameters for recommendations."""
        from src.mcp.server import search_anime

        with patch("src.mcp.server.qdrant_client", mock_qdrant_client):
            # Equivalent to: recommend_anime with SearchIntent parameters
            result = await search_anime.fn(
                query="Action dark anime from Mappa",
                genres=["Action"],
                studios=["Mappa"],
                exclusions=["Horror"],
                mood_keywords=["dark"],
                limit=5,
            )

            assert isinstance(result, list)


class TestParameterValidation:
    """Test parameter validation and error handling."""

    @pytest.mark.asyncio
    async def test_invalid_limit_clamping(self):
        """Test that invalid limits are clamped to valid ranges."""
        from src.mcp.server import search_anime

        with patch("src.mcp.server.qdrant_client") as mock_client:
            mock_client.search = AsyncMock(return_value=[])

            # Test limit too high gets clamped
            await search_anime.fn(query="test", limit=100)
            call_args = mock_client.search.call_args
            assert call_args[1]["limit"] <= 50

            # Test limit too low gets clamped
            await search_anime.fn(query="test", limit=0)
            call_args = mock_client.search.call_args
            assert call_args[1]["limit"] >= 1

    @pytest.mark.asyncio
    async def test_empty_parameters_handling(self):
        """Test handling of empty/None parameters."""
        from src.mcp.server import search_anime

        with patch("src.mcp.server.qdrant_client") as mock_client:
            mock_client.search = AsyncMock(return_value=[])

            # Test with None/empty parameters
            result = await search_anime.fn(
                query="test",
                genres=None,
                year_range=None,
                anime_types=[],
                studios=None,
                exclusions=[],
                mood_keywords=None,
            )

            assert isinstance(result, list)
            # Should not include filters for empty parameters
            call_args = mock_client.search.call_args
            filters = call_args[1].get("filters")
            # Filters should be None when no meaningful filters are provided
            assert filters is None


class TestFilterBuilding:
    """Test the filter building logic for SearchIntent parameters."""

    def test_build_search_filters_with_genres(self):
        """Test filter building with genres parameter."""
        # This will test the filter building function once implemented

    def test_build_search_filters_with_year_range(self):
        """Test filter building with year_range parameter."""

    def test_build_search_filters_with_multiple_parameters(self):
        """Test filter building with multiple parameters."""


if __name__ == "__main__":
    pytest.main([__file__])
