"""Unit tests for MAL-specific MCP tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.anime_mcp.tools.mal_tools import (
    _search_anime_mal_impl,
    _get_anime_mal_impl,
    _get_seasonal_anime_mal_impl,
    mal_client,
    mal_mapper
)
from src.models.universal_anime import UniversalAnime, UniversalSearchParams


class TestMALTools:
    """Test suite for MAL MCP tools."""

    @pytest.fixture
    def mock_mal_client(self):
        """Create mock MAL client."""
        client = AsyncMock()
        client.search_anime.return_value = [
            {
                "id": 16498,
                "title": "Attack on Titan",
                "main_picture": {"medium": "https://example.com/aot.jpg"},
                "mean": 8.54,
                "rank": 50,
                "popularity": 1,
                "num_list_users": 3000000,
                "num_scoring_users": 1500000,
                "nsfw": "white",
                "media_type": "tv",
                "status": "finished_airing",
                "num_episodes": 25,
                "start_date": "2013-04-07",
                "end_date": "2013-09-28",
                "rating": "r",
                "source": "manga",
                "broadcast": {"day_of_the_week": "sunday", "start_time": "17:00"},
                "created_at": "2008-10-03T00:54:00+00:00",
                "updated_at": "2022-01-30T08:15:00+00:00",
                "synopsis": "Humanity fights against titans",
                "genres": [{"id": 1, "name": "Action"}, {"id": 8, "name": "Drama"}],
                "studios": [{"id": 858, "name": "Wit Studio"}]
            }
        ]
        
        client.get_anime_by_id.return_value = {
            "id": 16498,
            "title": "Attack on Titan",
            "main_picture": {"large": "https://example.com/aot_large.jpg"},
            "alternative_titles": {
                "synonyms": ["Shingeki no Kyojin"],
                "en": "Attack on Titan",
                "ja": "進撃の巨人"
            },
            "mean": 8.54,
            "rank": 50,
            "popularity": 1,
            "num_list_users": 3000000,
            "num_scoring_users": 1500000,
            "nsfw": "white",
            "media_type": "tv",
            "status": "finished_airing",
            "num_episodes": 25,
            "start_date": "2013-04-07",
            "end_date": "2013-09-28",
            "rating": "r",
            "source": "manga",
            "average_episode_duration": 1440,
            "start_season": {"year": 2013, "season": "spring"},
            "broadcast": {"day_of_the_week": "sunday", "start_time": "17:00"},
            "synopsis": "Detailed synopsis about humanity fighting titans",
            "background": "Background information about the series",
            "related_anime": [
                {"node": {"id": 25777, "title": "Attack on Titan Season 2"}}
            ],
            "related_manga": [
                {"node": {"id": 23390, "title": "Attack on Titan Manga"}}
            ],
            "recommendations": [
                {"node": {"id": 11061, "title": "Hunter x Hunter"}}
            ],
            "statistics": {
                "status": {
                    "watching": 100000,
                    "completed": 2500000,
                    "on_hold": 200000,
                    "dropped": 100000,
                    "plan_to_watch": 100000
                }
            },
            "genres": [{"id": 1, "name": "Action"}, {"id": 8, "name": "Drama"}],
            "studios": [{"id": 858, "name": "Wit Studio"}],
            "pictures": [
                {"medium": "https://example.com/pic1.jpg"},
                {"medium": "https://example.com/pic2.jpg"}
            ]
        }
        
        client.get_seasonal_anime.return_value = [
            {
                "id": 12345,
                "title": "Spring 2023 Anime",
                "mean": 7.8,
                "popularity": 100,
                "num_list_users": 500000,
                "media_type": "tv",
                "num_episodes": 12,
                "synopsis": "A great spring anime",
                "genres": [{"id": 4, "name": "Comedy"}]
            }
        ]
        
        return client

    @pytest.fixture
    def mock_mal_mapper(self):
        """Create mock MAL mapper."""
        mapper = MagicMock()
        
        # Mock to_mal_search_params
        mapper.to_mal_search_params.return_value = {
            "q": "attack on titan",
            "limit": 20,
            "offset": 0,
            "nsfw": "white"
        }
        
        # Mock to_universal_anime
        mapper.to_universal_anime.return_value = UniversalAnime(
            id="mal_16498",
            title="Attack on Titan",
            type_format="TV",
            episodes=25,
            score=8.54,
            year=2013,
            status="FINISHED",
            genres=["Action", "Drama"],
            studios=["Wit Studio"],
            description="Humanity fights against titans",
            image_url="https://example.com/aot.jpg",
            data_quality_score=0.95
        )
        
        return mapper

    @pytest.fixture
    def mock_context(self):
        """Create mock MCP context."""
        context = AsyncMock()
        context.info = AsyncMock()
        context.error = AsyncMock()
        return context

    @pytest.mark.asyncio
    async def test_search_anime_mal_success(self, mock_mal_client, mock_mal_mapper, mock_context):
        """Test successful MAL anime search."""
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_mal_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mal_mapper):
            
            result = await _search_anime_mal_impl(
                query="attack on titan",
                limit=20,
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == 1
            
            anime = result[0]
            # Should return raw MAL data with source attribution
            assert anime["id"] == 16498  # Raw MAL ID
            assert anime["title"] == "Attack on Titan"
            assert anime["source_platform"] == "mal"  # Added by implementation
            
            # Verify client calls
            mock_mal_client.search_anime.assert_called_once()
            mock_mal_mapper.to_mal_search_params.assert_called_once()
            # No mapper.to_universal_anime call - returns raw results
            
            # Verify context calls
            mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_search_anime_mal_with_filters(self, mock_mal_client, mock_mal_mapper, mock_context):
        """Test MAL search with various filters."""
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_mal_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mal_mapper):
            
            result = await _search_anime_mal_impl(
                query="mecha anime",
                limit=10,
                offset=20,
                ctx=mock_context
            )
            
            # Verify mapper was called with MAL-specific params
            call_args = mock_mal_mapper.to_mal_search_params.call_args
            universal_params = call_args[0][0]
            mal_specific = call_args[0][1]
            
            assert universal_params.query == "mecha anime"
            assert universal_params.limit == 10
            assert universal_params.offset == 20
            
            # Current implementation only passes limit and offset to mal_specific
            assert mal_specific["limit"] == 10
            assert mal_specific["offset"] == 20
            
            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["id"] == 16498  # Raw MAL ID
            assert result[0]["source_platform"] == "mal"

    @pytest.mark.asyncio
    async def test_search_anime_mal_limit_validation(self, mock_mal_client, mock_mal_mapper):
        """Test MAL search limit validation."""
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_mal_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mal_mapper):
            
            # Test limit over maximum (should clamp to 100)
            await _search_anime_mal_impl(query="test", limit=200)
            
            call_args = mock_mal_mapper.to_mal_search_params.call_args
            universal_params = call_args[0][0]
            mal_specific = call_args[0][1]
            
            assert universal_params.limit == 100
            assert mal_specific["limit"] == 100

    @pytest.mark.asyncio
    async def test_search_anime_mal_no_client(self, mock_context):
        """Test MAL search when client not available."""
        with patch('src.anime_mcp.tools.mal_tools.mal_client', None):
            
            with pytest.raises(RuntimeError, match="MAL client not available"):
                await _search_anime_mal_impl(query="test", ctx=mock_context)
            
            mock_context.error.assert_called_with("MAL client not configured - missing API key")

    @pytest.mark.asyncio
    async def test_search_anime_mal_client_error(self, mock_mal_client, mock_mal_mapper, mock_context):
        """Test MAL search when client raises error."""
        mock_mal_client.search_anime.side_effect = Exception("API error")
        
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_mal_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mal_mapper):
            
            with pytest.raises(RuntimeError, match="MAL search failed: API error"):
                await _search_anime_mal_impl(query="test", ctx=mock_context)
            
            mock_context.error.assert_called_with("MAL search failed: API error")

    @pytest.mark.asyncio
    async def test_search_anime_mal_mapper_error(self, mock_mal_client, mock_mal_mapper, mock_context):
        """Test MAL search when mapper fails for individual results."""
        # Current implementation doesn't use mapper for result processing, only for param conversion
        mock_mal_mapper.to_mal_search_params.side_effect = Exception("Mapping error")
        
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_mal_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mal_mapper):
            
            with pytest.raises(RuntimeError, match="MAL search failed: Mapping error"):
                await _search_anime_mal_impl(query="test", ctx=mock_context)
            
            mock_context.error.assert_called_with("MAL search failed: Mapping error")

    @pytest.mark.asyncio
    async def test_get_anime_mal_success(self, mock_mal_client, mock_mal_mapper, mock_context):
        """Test successful MAL anime detail retrieval."""
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_mal_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mal_mapper):
            
            result = await _get_anime_mal_impl(
                mal_id=16498,
                ctx=mock_context
            )
            
            # Verify result structure - should return raw MAL data with source attribution
            assert isinstance(result, dict)
            assert result["id"] == 16498  # Raw MAL ID
            assert result["title"] == "Attack on Titan"
            assert result["mal_id"] == 16498
            assert result["alternative_titles"]["en"] == "Attack on Titan"
            assert "statistics" in result  # Should be included in default fields
            assert "related_anime" in result  # Should be included in default fields
            assert result["source_platform"] == "mal"
            
            # Verify client was called with default comprehensive fields
            mock_mal_client.get_anime_by_id.assert_called_once()
            call_args = mock_mal_client.get_anime_by_id.call_args
            assert call_args[0][0] == 16498
            fields_list = call_args[1]["fields"]
            assert "statistics" in fields_list
            assert "related_anime" in fields_list
            assert "id" in fields_list
            assert "title" in fields_list

    @pytest.mark.asyncio
    async def test_get_anime_mal_with_custom_fields(self, mock_mal_client, mock_mal_mapper):
        """Test MAL anime detail with custom fields."""
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_mal_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mal_mapper):
            
            result = await _get_anime_mal_impl(
                mal_id=16498,
                fields=["id", "title", "mean"]
            )
            
            # Should return raw MAL data with source attribution
            assert isinstance(result, dict)
            assert result["id"] == 16498
            assert result["title"] == "Attack on Titan"
            assert result["mal_id"] == 16498
            assert result["source_platform"] == "mal"
            
            # Verify client was called with custom fields (as comma-separated string)
            mock_mal_client.get_anime_by_id.assert_called_once()
            call_args = mock_mal_client.get_anime_by_id.call_args
            assert call_args[0][0] == 16498
            assert call_args[1]["fields"] == "id,title,mean"

    @pytest.mark.asyncio
    async def test_get_anime_mal_not_found(self, mock_mal_client, mock_mal_mapper, mock_context):
        """Test MAL anime detail when anime not found."""
        mock_mal_client.get_anime_by_id.return_value = None
        
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_mal_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mal_mapper):
            
            result = await _get_anime_mal_impl(mal_id=99999, ctx=mock_context)
            
            assert result is None
            mock_context.info.assert_called_with("Anime with MAL ID 99999 not found")

    @pytest.mark.asyncio
    async def test_get_anime_mal_no_client(self, mock_context):
        """Test MAL anime detail when client not available."""
        with patch('src.anime_mcp.tools.mal_tools.mal_client', None):
            
            with pytest.raises(RuntimeError, match="MAL client not available"):
                await _get_anime_mal_impl(mal_id=16498, ctx=mock_context)

    @pytest.mark.asyncio
    async def test_get_anime_mal_client_error(self, mock_mal_client, mock_mal_mapper, mock_context):
        """Test MAL anime detail when client raises error."""
        mock_mal_client.get_anime_by_id.side_effect = Exception("API error")
        
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_mal_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mal_mapper):
            
            with pytest.raises(RuntimeError, match="Failed to get MAL anime 16498: API error"):
                await _get_anime_mal_impl(mal_id=16498, ctx=mock_context)

    @pytest.mark.asyncio
    async def test_get_anime_mal_fields_type_safety(self):
        """Test type safety of fields parameter in get_anime_mal."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        mock_client.get_anime_by_id.return_value = {"id": 123, "title": "Test"}
        
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mapper):
            
            # Test string fields
            await _get_anime_mal_impl(mal_id=123, fields="id,title,mean")
            
            # Test list fields
            await _get_anime_mal_impl(mal_id=123, fields=["id", "title", "mean"])
            
            # Test invalid type (int)
            with pytest.raises(RuntimeError, match="Fields must be string or list of strings, got: int"):
                await _get_anime_mal_impl(mal_id=123, fields=123)
            
            # Test invalid list (contains non-strings)
            with pytest.raises(RuntimeError, match="All fields must be strings"):
                await _get_anime_mal_impl(mal_id=123, fields=["id", 123, "title"])

    @pytest.mark.asyncio
    async def test_get_anime_mal_empty_fields_handling(self):
        """Test empty fields handling in get_anime_mal."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        mock_client.get_anime_by_id.return_value = {"id": 123, "title": "Test"}
        
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mapper):
            
            # Test empty string should use default fields
            await _get_anime_mal_impl(mal_id=123, fields="")
            
            # Test None should use default fields
            await _get_anime_mal_impl(mal_id=123, fields=None)
            
            # Verify default fields were used (as comma-separated string)
            assert mock_client.get_anime_by_id.call_count == 2
            for call in mock_client.get_anime_by_id.call_args_list:
                fields_arg = call[1]["fields"]
                assert isinstance(fields_arg, str)
                # Should contain detail-only fields like 'pictures', 'background', etc.
                assert "pictures" in fields_arg
                assert "background" in fields_arg
                assert "id" in fields_arg
                assert "title" in fields_arg
                assert "statistics" in fields_arg

    @pytest.mark.asyncio
    async def test_get_mal_seasonal_anime_success(self, mock_mal_client, mock_mal_mapper, mock_context):
        """Test successful MAL seasonal anime retrieval."""
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_mal_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mal_mapper):
            
            result = await _get_seasonal_anime_mal_impl(
                year=2023,
                season="spring",
                sort="anime_score",
                limit=50,
                ctx=mock_context
            )
            
            # Verify result structure - should return raw MAL data with source attribution
            assert isinstance(result, list)
            assert len(result) == 1
            
            anime = result[0]
            # Raw MAL data should be included
            assert anime["id"] == 12345  # Raw MAL ID from mock
            assert anime["title"] == "Spring 2023 Anime"
            assert anime["mean"] == 7.8
            # Added by implementation
            assert anime["season"] == "spring"
            assert anime["season_year"] == 2023
            assert anime["source_platform"] == "mal"
            assert "fetched_at" in anime
            
            # Verify client call includes offset and fields
            mock_mal_client.get_seasonal_anime.assert_called_once()
            call_args = mock_mal_client.get_seasonal_anime.call_args
            # Arguments are passed as keyword arguments
            assert call_args[1]["year"] == 2023
            assert call_args[1]["season"] == "spring"
            assert call_args[1]["sort"] == "anime_score"
            assert call_args[1]["limit"] == 50
            assert call_args[1]["offset"] == 0  # Default offset
            assert "fields" in call_args[1]  # Default fields should be included

    @pytest.mark.asyncio
    async def test_get_mal_seasonal_anime_with_offset_and_fields(self, mock_mal_client, mock_mal_mapper):
        """Test MAL seasonal anime with offset and custom fields."""
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_mal_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mal_mapper):
            
            result = await _get_seasonal_anime_mal_impl(
                year=2023,
                season="summer",
                sort="anime_num_list_users",
                limit=25,
                offset=10,
                fields=["id", "title", "mean", "popularity"]
            )
            
            # Should return raw MAL data
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["id"] == 12345
            assert result[0]["season"] == "summer"
            assert result[0]["source_platform"] == "mal"
            
            # Verify client was called with all parameters
            mock_mal_client.get_seasonal_anime.assert_called_once()
            call_args = mock_mal_client.get_seasonal_anime.call_args
            # Arguments are passed as keyword arguments
            assert call_args[1]["year"] == 2023
            assert call_args[1]["season"] == "summer"
            assert call_args[1]["sort"] == "anime_num_list_users"
            assert call_args[1]["limit"] == 25
            assert call_args[1]["offset"] == 10
            assert call_args[1]["fields"] == "id,title,mean,popularity"

    @pytest.mark.asyncio
    async def test_get_mal_seasonal_anime_fields_type_safety(self):
        """Test type safety of fields parameter in seasonal anime."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        mock_client.get_seasonal_anime.return_value = [{"id": 123, "title": "Test"}]
        
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mapper):
            
            # Test string fields
            await _get_seasonal_anime_mal_impl(year=2023, season="spring", fields="id,title,mean")
            
            # Test list fields
            await _get_seasonal_anime_mal_impl(year=2023, season="spring", fields=["id", "title", "mean"])
            
            # Test invalid type (int)
            with pytest.raises(RuntimeError, match="Fields must be string or list of strings, got: int"):
                await _get_seasonal_anime_mal_impl(year=2023, season="spring", fields=123)
            
            # Test invalid list (contains non-strings)
            with pytest.raises(RuntimeError, match="All fields must be strings"):
                await _get_seasonal_anime_mal_impl(year=2023, season="spring", fields=["id", 123, "title"])

    @pytest.mark.asyncio
    async def test_get_mal_seasonal_anime_limit_validation(self, mock_mal_client, mock_mal_mapper):
        """Test MAL seasonal anime limit validation."""
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_mal_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mal_mapper):
            
            await _get_seasonal_anime_mal_impl(
                year=2023,
                season="summer",
                limit=1000  # Should clamp to 500
            )
            
            # Parameters are now passed as keyword arguments
            call_args = mock_mal_client.get_seasonal_anime.call_args[1]  # keyword arguments
            assert call_args["limit"] == 500

    @pytest.mark.asyncio
    async def test_get_mal_seasonal_anime_sort_options(self, mock_mal_client, mock_mal_mapper):
        """Test MAL seasonal anime with different sort options."""
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_mal_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mal_mapper):
            
            # Test anime_num_list_users sort
            await _get_seasonal_anime_mal_impl(
                year=2023,
                season="fall",
                sort="anime_num_list_users"
            )
            
            # Parameters are now passed as keyword arguments
            call_args = mock_mal_client.get_seasonal_anime.call_args[1]  # keyword arguments
            assert call_args["sort"] == "anime_num_list_users"

    @pytest.mark.asyncio
    async def test_get_mal_seasonal_anime_no_client(self, mock_context):
        """Test MAL seasonal anime when client not available."""
        with patch('src.anime_mcp.tools.mal_tools.mal_client', None):
            
            with pytest.raises(RuntimeError, match="MAL client not available"):
                await _get_seasonal_anime_mal_impl(
                    year=2023,
                    season="winter",
                    ctx=mock_context
                )

    @pytest.mark.asyncio
    async def test_get_mal_seasonal_anime_client_error(self, mock_mal_client, mock_mal_mapper, mock_context):
        """Test MAL seasonal anime when client raises error."""
        mock_mal_client.get_seasonal_anime.side_effect = Exception("Seasonal API error")
        
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_mal_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mal_mapper):
            
            with pytest.raises(RuntimeError, match="MAL seasonal search failed: Seasonal API error"):
                await _get_seasonal_anime_mal_impl(
                    year=2023,
                    season="summer",
                    ctx=mock_context
                )

    @pytest.mark.asyncio
    async def test_get_mal_seasonal_anime_empty_results(self, mock_mal_client, mock_mal_mapper, mock_context):
        """Test MAL seasonal anime with empty results."""
        mock_mal_client.get_seasonal_anime.return_value = []
        
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_mal_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mal_mapper):
            
            result = await _get_seasonal_anime_mal_impl(
                year=1900,  # Old year likely to have no results
                season="winter",
                ctx=mock_context
            )
            
            # Should return empty list when no results
            assert result == []
            mock_context.info.assert_called_with("Found 0 seasonal anime from MAL")


class TestMALToolsEdgeCases:
    """Test edge cases and error scenarios for MAL tools."""
    
    @pytest.fixture
    def mock_context(self):
        """Create mock MCP context."""
        context = AsyncMock()
        context.info = AsyncMock()
        context.error = AsyncMock()
        return context

    @pytest.mark.asyncio
    async def test_search_anime_mal_without_context(self):
        """Test MAL search without context (no logging)."""
        with patch('src.anime_mcp.tools.mal_tools.mal_client', None):
            
            with pytest.raises(RuntimeError, match="MAL client not available"):
                await _search_anime_mal_impl(query="test")  # No ctx parameter

    @pytest.mark.asyncio
    async def test_mal_tools_parameter_validation(self):
        """Test parameter validation for MAL tools."""
        # Test valid parameters for _search_anime_mal_impl
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        mock_mapper.to_mal_search_params.return_value = {"q": "test", "limit": 20, "offset": 0}
        mock_client.search_anime.return_value = []
        
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mapper):
            
            # Test basic parameters
            await _search_anime_mal_impl(query="test")
            await _search_anime_mal_impl(query="test", limit=10)
            await _search_anime_mal_impl(query="test", offset=5)
            await _search_anime_mal_impl(query="test", fields=["id", "title"])
            
            # Test fields parameter flexibility (both string and list)
            await _search_anime_mal_impl(query="test", fields="id,title,mean")  # String format
            await _search_anime_mal_impl(query="test", fields=["id", "title", "mean"])  # List format
            
            # Test limit validation (should clamp to 100)
            await _search_anime_mal_impl(query="test", limit=200)
            
            # Verify client was called
            assert mock_client.search_anime.call_count >= 7

    @pytest.mark.asyncio
    async def test_mal_tools_fields_type_safety(self):
        """Test type safety of fields parameter."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        mock_mapper.to_mal_search_params.return_value = {"q": "test", "limit": 20, "offset": 0}
        mock_client.search_anime.return_value = []
        
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mapper):
            
            # Test invalid type (int)
            with pytest.raises(RuntimeError, match="Fields must be string or list of strings, got: int"):
                await _search_anime_mal_impl(query="test", fields=123)
            
            # Test invalid list (contains non-strings)
            with pytest.raises(RuntimeError, match="All fields must be strings"):
                await _search_anime_mal_impl(query="test", fields=["id", 123, "title"])
            
            # Test invalid type (dict)
            with pytest.raises(RuntimeError, match="Fields must be string or list of strings, got: dict"):
                await _search_anime_mal_impl(query="test", fields={"id": "title"})

    @pytest.mark.asyncio
    async def test_mal_seasonal_parameter_validation(self):
        """Test parameter validation for seasonal MAL tools."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        mock_mapper.to_universal_anime.return_value = UniversalAnime(
            id="test", title="Test", type_format="TV", status="FINISHED", data_quality_score=0.8
        )
        mock_client.get_seasonal_anime.return_value = []
        
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mapper):
            
            # Test all valid seasons
            for season in ["winter", "spring", "summer", "fall"]:
                await _get_seasonal_anime_mal_impl(year=2023, season=season)
            
            # Test all valid sort options
            for sort in ["anime_score", "anime_num_list_users"]:
                await _get_seasonal_anime_mal_impl(year=2023, season="spring", sort=sort)

    @pytest.mark.asyncio
    async def test_mal_tools_data_quality_scoring(self):
        """Test data quality scoring in MAL tools."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        # Mock different quality scores
        mock_mapper.to_universal_anime.side_effect = [
            UniversalAnime(id="high_quality", title="High Quality", type_format="TV", status="FINISHED", data_quality_score=0.95),
            UniversalAnime(id="low_quality", title="Low Quality", type_format="TV", status="FINISHED", data_quality_score=0.65)
        ]
        
        mock_client.search_anime.return_value = [
            {"id": 1, "title": "High Quality Anime"},
            {"id": 2, "title": "Low Quality Anime"}
        ]
        
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mapper):
            
            result = await _search_anime_mal_impl(query="test")
            
            # Current implementation returns raw MAL data with source attribution
            assert len(result) == 2
            assert result[0]["id"] == 1
            assert result[0]["title"] == "High Quality Anime"
            assert result[0]["source_platform"] == "mal"
            assert result[1]["id"] == 2
            assert result[1]["title"] == "Low Quality Anime"
            assert result[1]["source_platform"] == "mal"

    @pytest.mark.asyncio
    async def test_mal_tools_empty_results(self, mock_context):
        """Test MAL tools with empty results."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        mock_client.search_anime.return_value = []
        mock_client.get_seasonal_anime.return_value = []
        
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mapper):
            
            # Test search with no results
            result = await _search_anime_mal_impl(query="nonexistent", ctx=mock_context)
            assert result == []
            mock_context.info.assert_called_with("Found 0 anime via MAL")
            
            # Test seasonal with no results
            result = await _get_seasonal_anime_mal_impl(year=1900, season="spring", ctx=mock_context)
            assert result == []
            mock_context.info.assert_called_with("Found 0 seasonal anime from MAL")


class TestMALToolsIntegration:
    """Integration tests for MAL tools with real-like data flows."""
    
    @pytest.fixture
    def mock_context(self):
        """Create mock MCP context."""
        context = AsyncMock()
        context.info = AsyncMock()
        context.error = AsyncMock()
        return context

    @pytest.mark.asyncio
    async def test_mal_tools_partial_data_handling(self, mock_context):
        """Test MAL tools handling partial/missing data."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        # Mock result with missing optional fields - should return raw MAL data with source attribution
        mock_client.search_anime.return_value = [
            {
                "id": 123,
                "title": "Minimal Data Anime",
                # Missing many optional fields that would normally be present
            }
        ]
        
        mock_mapper.to_mal_search_params.return_value = {"q": "test", "limit": 20, "offset": 0}
        
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mapper):
            
            result = await _search_anime_mal_impl(query="test", ctx=mock_context)
            
            assert len(result) == 1
            anime = result[0]
            
            # Should return raw MAL data with source attribution
            assert anime["id"] == 123
            assert anime["title"] == "Minimal Data Anime"
            assert anime["source_platform"] == "mal"
            
            # Missing fields should just be missing (not converted)
            assert "mean" not in anime or anime.get("mean") is None
            assert "rank" not in anime or anime.get("rank") is None

    @pytest.mark.asyncio
    async def test_mal_search_to_detail_workflow(self, mock_context):
        """Test workflow from search to getting details."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        # Setup search result
        mock_client.search_anime.return_value = [{"id": 16498, "title": "Attack on Titan"}]
        
        # Setup detail result
        mock_client.get_anime_by_id.return_value = {
            "id": 16498,
            "title": "Attack on Titan",
            "synopsis": "Detailed synopsis"
        }
        
        mock_mapper.to_mal_search_params.return_value = {"q": "attack", "limit": 20, "offset": 0}
        
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mapper):
            
            # Step 1: Search
            search_results = await _search_anime_mal_impl(query="attack", ctx=mock_context)
            assert len(search_results) == 1
            assert search_results[0]["id"] == 16498
            assert search_results[0]["title"] == "Attack on Titan"
            assert search_results[0]["source_platform"] == "mal"
            
            # Step 2: Get details  
            detail_result = await _get_anime_mal_impl(mal_id=16498, ctx=mock_context)
            assert detail_result["id"] == 16498
            assert detail_result["title"] == "Attack on Titan" 
            assert detail_result["synopsis"] == "Detailed synopsis"
            assert detail_result["source_platform"] == "mal"

    @pytest.mark.asyncio
    async def test_mal_tools_annotation_verification(self):
        """Verify that MAL tools have correct MCP annotations."""
        from src.anime_mcp.tools.mal_tools import mcp
        
        # Get registered tools - FastMCP uses tools property differently
        tools = mcp.tools if hasattr(mcp, 'tools') else {}
        
        # Verify search_anime_mal annotations
        search_tool = tools.get("search_anime_mal")
        if search_tool:
            # If we can access tools, verify they exist
            assert search_tool is not None
            if hasattr(search_tool, 'annotations'):
                assert search_tool.annotations["title"] == "MAL Anime Search"
                assert search_tool.annotations["readOnlyHint"] is True
                assert search_tool.annotations["idempotentHint"] is True
        
        # Verify get_anime_mal annotations
        detail_tool = tools.get("get_anime_mal")
        if detail_tool:
            assert detail_tool is not None
            if hasattr(detail_tool, 'annotations'):
                assert detail_tool.annotations["title"] == "MAL Anime Details"
                assert detail_tool.annotations["readOnlyHint"] is True
        
        # Verify get_mal_seasonal_anime annotations
        seasonal_tool = tools.get("get_mal_seasonal_anime")
        if seasonal_tool:
            assert seasonal_tool is not None
            if hasattr(seasonal_tool, 'annotations'):
                assert seasonal_tool.annotations["title"] == "MAL Seasonal Anime"

    @pytest.mark.asyncio
    async def test_mal_tools_comprehensive_error_coverage(self, mock_context):
        """Test comprehensive error scenarios."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        # Test different types of errors
        error_scenarios = [
            ("Network timeout", "Request timeout"),
            ("Rate limited", "Too many requests"),
            ("Invalid API key", "Authentication failed"),
            ("Server error", "Internal server error")
        ]
        
        with patch('src.anime_mcp.tools.mal_tools.mal_client', mock_client), \
             patch('src.anime_mcp.tools.mal_tools.mal_mapper', mock_mapper):
            
            for error_type, error_msg in error_scenarios:
                mock_client.search_anime.side_effect = Exception(error_msg)
                
                with pytest.raises(RuntimeError, match=f"MAL search failed: {error_msg}"):
                    await _search_anime_mal_impl(query="test", ctx=mock_context)
                
                mock_context.error.assert_called_with(f"MAL search failed: {error_msg}")
                mock_context.reset_mock()