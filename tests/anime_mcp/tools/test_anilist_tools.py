"""Unit tests for AniList-specific MCP tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.anime_mcp.tools.anilist_tools import (
    anilist_client,
    mcp
)
from src.models.structured_responses import BasicAnimeResult, AnimeType, AnimeStatus


class TestAniListTools:
    """Test suite for AniList MCP tools."""

    @pytest.fixture
    def mock_anilist_client(self):
        """Create mock AniList client."""
        client = AsyncMock()
        client.search_anime.return_value = [
            {
                "id": 16498,
                "title": {
                    "romaji": "Shingeki no Kyojin",
                    "english": "Attack on Titan",
                    "native": "進撃の巨人"
                },
                "type": "ANIME",
                "format": "TV",
                "status": "FINISHED",
                "episodes": 25,
                "duration": 24,
                "averageScore": 85,
                "popularity": 500000,
                "favourites": 45000,
                "startDate": {"year": 2013, "month": 4, "day": 7},
                "endDate": {"year": 2013, "month": 9, "day": 28},
                "season": "SPRING",
                "seasonYear": 2013,
                "description": "Humanity fights against titans",
                "genres": ["Action", "Drama", "Fantasy"],
                "tags": [
                    {"name": "Military", "rank": 90},
                    {"name": "Survival", "rank": 85}
                ],
                "studios": {
                    "nodes": [{"name": "Wit Studio"}]
                },
                "coverImage": {
                    "large": "https://example.com/aot.jpg"
                },
                "isAdult": False,
                "countryOfOrigin": "JP",
                "source": "MANGA"
            }
        ]
        
        client.get_anime_by_id.return_value = {
            "id": 16498,
            "title": {
                "romaji": "Shingeki no Kyojin",
                "english": "Attack on Titan",
                "native": "進撃の巨人"
            },
            "type": "ANIME",
            "format": "TV",
            "status": "FINISHED",
            "episodes": 25,
            "duration": 24,
            "averageScore": 85,
            "meanScore": 85,
            "popularity": 500000,
            "favourites": 45000,
            "startDate": {"year": 2013, "month": 4, "day": 7},
            "endDate": {"year": 2013, "month": 9, "day": 28},
            "season": "SPRING",
            "seasonYear": 2013,
            "description": "Detailed description about titans",
            "genres": ["Action", "Drama", "Fantasy"],
            "tags": [
                {"name": "Military", "rank": 90, "description": "Military themes"},
                {"name": "Survival", "rank": 85, "description": "Survival elements"}
            ],
            "studios": {
                "nodes": [{"name": "Wit Studio", "id": 858}]
            },
            "staff": {
                "nodes": [
                    {"name": {"full": "Director Name"}, "primaryOccupations": ["Director"]}
                ]
            },
            "characters": {
                "nodes": [
                    {"name": {"full": "Eren Yeager"}, "image": {"medium": "eren.jpg"}}
                ]
            },
            "relations": {
                "nodes": [
                    {"id": 25777, "title": {"romaji": "Shingeki no Kyojin Season 2"}, "type": "ANIME"}
                ]
            },
            "recommendations": {
                "nodes": [
                    {"mediaRecommendation": {"id": 11061, "title": {"romaji": "Hunter x Hunter"}}}
                ]
            },
            "coverImage": {
                "large": "https://example.com/aot_large.jpg"
            },
            "bannerImage": "https://example.com/aot_banner.jpg",
            "trailer": {
                "id": "abc123",
                "site": "youtube"
            },
            "isAdult": False,
            "countryOfOrigin": "JP",
            "source": "MANGA",
            "hashtag": "#AttackOnTitan",
            "updatedAt": 1640995200
        }
        
        return client

    

    @pytest.fixture
    def mock_context(self):
        """Create mock MCP context."""
        context = AsyncMock()
        context.info = AsyncMock()
        context.error = AsyncMock()
        return context

    @pytest.mark.asyncio
    async def test_search_anime_anilist_success(self, mock_anilist_client, mock_context):
        """Test successful AniList anime search."""
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_anilist_client):
            
            # Get the underlying function from MCP
            search_tool = await mcp.get_tool("search_anime_anilist")
            result = await search_tool.fn(
                query="attack on titan",
                per_page=20,
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == 1
            
            anime = result[0]
            assert anime["id"] == "16498"
            assert anime["title"] == "Shingeki no Kyojin"
            assert anime["anilist_id"] == 16498
            assert anime["anilist_score"] == 85
            assert anime["source_platform"] == "anilist"
            
            # Verify client calls
            mock_anilist_client.search_anime.assert_called_once()
            
            # Verify context calls
            mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_search_anime_anilist_with_filters(self, mock_anilist_client, mock_context):
        """Test AniList search with comprehensive filters."""
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_anilist_client):
            
            # Get the underlying function from MCP
            search_tool = await mcp.get_tool("search_anime_anilist")
            await search_tool.fn(
                query="mecha anime",
                format_in=["TV", "MOVIE"],
                status_in=["RELEASING", "FINISHED"],
                genre_in=["Action", "Mecha"],
                tag_in=["Military", "Space"],
                season="SPRING",
                season_year=2023,
                is_adult=False,
                sort=["SCORE_DESC", "POPULARITY_DESC"],
                per_page=25,
                page=2,
                ctx=mock_context
            )
            
            # Verify client was called with comprehensive params
            mock_anilist_client.search_anime.assert_called_once()
            call_args = mock_anilist_client.search_anime.call_args[0][0]
            
            assert call_args["search"] == "mecha anime"
            assert call_args["perPage"] == 25
            assert call_args["page"] == 2
            assert call_args["format_in"] == ["TV", "MOVIE"]
            assert call_args["status_in"] == ["RELEASING", "FINISHED"]
            assert call_args["genre_in"] == ["Action", "Mecha"]
            assert call_args["tag_in"] == ["Military", "Space"]
            assert call_args["season"] == "SPRING"
            assert call_args["seasonYear"] == 2023
            assert call_args["isAdult"] == False
            assert call_args["sort"] == ["SCORE_DESC", "POPULARITY_DESC"]

    @pytest.mark.asyncio
    async def test_search_anime_anilist_pagination(self, mock_anilist_client):
        """Test AniList search pagination limits."""
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_anilist_client):
            
            # Test per_page limit (should clamp to 50)
            await search_anime_anilist(query="test", per_page=100)
            
            call_args = mock_anilist_client.search_anime.call_args[0][0]
            assert call_args["perPage"] == 50
            
            # Test page limit (should clamp to 500) 
            await search_anime_anilist(query="test", page=1000)
            
            call_args = mock_anilist_client.search_anime.call_args[0][0]
            assert call_args["page"] == 500

    @pytest.mark.asyncio
    async def test_search_anime_anilist_no_client(self, mock_context):
        """Test AniList search when client not available."""
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', None):
            
            with pytest.raises(RuntimeError, match="AniList client not available"):
                await search_anime_anilist(query="test", ctx=mock_context)
            
            mock_context.error.assert_called_with("AniList client not configured")

    @pytest.mark.asyncio
    async def test_search_anime_anilist_graphql_error(self, mock_anilist_client, mock_context):
        """Test AniList search when GraphQL returns error."""
        mock_anilist_client.search_anime.return_value = {
            "errors": [{"message": "GraphQL syntax error"}]
        }
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_anilist_client):
            
            with pytest.raises(RuntimeError, match="AniList search failed: GraphQL errors"):
                await search_anime_anilist(query="test", ctx=mock_context)

    @pytest.mark.asyncio
    async def test_search_anime_anilist_client_exception(self, mock_anilist_client, mock_context):
        """Test AniList search when client raises exception."""
        mock_anilist_client.search_anime.side_effect = Exception("Network error")
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_anilist_client):
            
            with pytest.raises(RuntimeError, match="AniList search failed: Network error"):
                await search_anime_anilist(query="test", ctx=mock_context)

    @pytest.mark.asyncio
    async def test_search_anime_anilist_mapping_error(self, mock_anilist_client, mock_context):
        """Test AniList search when individual result mapping fails."""
        # Simulate a malformed raw result that causes an error during processing
        mock_anilist_client.search_anime.return_value = [
            {"id": 1, "malformed_key": "This will cause an error"}
        ]
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_anilist_client):
            
            result = await search_anime_anilist(query="test", ctx=mock_context)
            
            # Should return empty list when all mappings fail
            assert result == []
            mock_context.error.assert_called_with(f"Failed to process AniList result: 'title'")

    @pytest.mark.asyncio
    async def test_get_anime_anilist_success(self, mock_anilist_client, mock_context):
        """Test successful AniList anime detail retrieval."""
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_anilist_client):
            
            result = await get_anime_anilist(
                anilist_id=16498,
                include_characters=True,
                include_staff=True,
                include_relations=True,
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, dict)
            assert result["id"] == "16498"
            assert result["title"] == "Shingeki no Kyojin"
            assert result["anilist_id"] == 16498
            assert result["source_platform"] == "anilist"
            
            # Should include optional data
            assert "characters" in result
            assert "staff" in result
            assert "relations" in result
            
            # Verify client was called with comprehensive query
            mock_anilist_client.get_anime_by_id.assert_called_once()
            call_args = mock_anilist_client.get_anime_by_id.call_args
            assert call_args[0][0] == 16498
            query = call_args[1]["fields"]
            assert "characters" in query
            assert "staff" in query
            assert "relations" in query

    @pytest.mark.asyncio
    async def test_get_anime_anilist_minimal_options(self, mock_anilist_client):
        """Test AniList anime detail with minimal options."""
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_anilist_client):
            
            result = await get_anime_anilist(
                anilist_id=16498,
                include_characters=False,
                include_staff=False,
                include_relations=False
            )
            
            # Should have empty optional data
            assert result["characters"] == {}
            assert result["staff"] == {}
            assert result["relations"] == {}

    @pytest.mark.asyncio
    async def test_get_anime_anilist_not_found(self, mock_anilist_client, mock_context):
        """Test AniList anime detail when anime not found."""
        mock_anilist_client.get_anime_by_id.return_value = {"data": {"Media": None}}
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_anilist_client):
            
            result = await get_anime_anilist(anilist_id=99999, ctx=mock_context)
            
            assert result is None
            mock_context.info.assert_called_with("Anime with AniList ID 99999 not found")

    @pytest.mark.asyncio
    async def test_get_anime_anilist_graphql_error(self, mock_anilist_client, mock_context):
        """Test AniList anime detail when GraphQL returns error."""
        mock_anilist_client.get_anime_by_id.return_value = {
            "errors": [{"message": "Variable '$id' expected value of type 'Int!'"}]
        }
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_anilist_client):
            
            with pytest.raises(RuntimeError, match="Failed to get AniList anime 16498: GraphQL errors"):
                await get_anime_anilist(anilist_id=16498, ctx=mock_context)

    @pytest.mark.asyncio
    async def test_get_anime_anilist_no_client(self, mock_context):
        """Test AniList anime detail when client not available."""
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', None):
            
            with pytest.raises(RuntimeError, match="AniList client not available"):
                await get_anime_anilist(anilist_id=16498, ctx=mock_context)


class TestAniListToolsComprehensive:
    """Comprehensive tests for AniList tools covering edge cases."""

    @pytest.mark.asyncio
    async def test_anilist_tools_parameter_validation(self):
        """Test parameter validation for AniList tools."""
        mock_client = AsyncMock()
        mock_client.search_anime.return_value = {"data": {"Page": {"media": []}}}
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_client):
            
            # Test all valid formats
            valid_formats = ["TV", "TV_SHORT", "MOVIE", "SPECIAL", "OVA", "ONA", "MUSIC"]
            for format_type in valid_formats:
                await search_anime_anilist(query="test", format_in=[format_type])
            
            # Test all valid statuses
            valid_statuses = ["FINISHED", "RELEASING", "NOT_YET_RELEASED", "CANCELLED", "HIATUS"]
            for status in valid_statuses:
                await search_anime_anilist(query="test", status_in=[status])
            
            # Test all valid seasons
            valid_seasons = ["WINTER", "SPRING", "SUMMER", "FALL"]
            for season in valid_seasons:
                await search_anime_anilist(query="test", season=season)
            
            # Test all valid sources
            valid_sources = ["ORIGINAL", "MANGA", "LIGHT_NOVEL", "VISUAL_NOVEL", "VIDEO_GAME", "OTHER", "NOVEL", "DOUJINSHI", "ANIME", "WEB_NOVEL", "LIVE_ACTION", "GAME", "COMIC", "MULTIMEDIA_PROJECT", "PICTURE_BOOK"]
            for source in valid_sources:
                await search_anime_anilist(query="test", source_in=[source])
            
            # Test all valid sort options
            valid_sorts = ["ID", "TITLE_ROMAJI", "TITLE_ENGLISH", "TITLE_NATIVE", "TYPE", "FORMAT", "START_DATE", "END_DATE", "SCORE", "POPULARITY", "TRENDING", "EPISODES", "DURATION", "STATUS", "CHAPTERS", "VOLUMES", "UPDATED_AT", "SEARCH_MATCH", "FAVOURITES"]
            for sort in valid_sorts:
                await search_anime_anilist(query="test", sort=[sort])

    @pytest.mark.asyncio
    async def test_anilist_tools_range_validation(self, mock_context):
        """Test range parameter validation."""
        mock_client = AsyncMock()
        mock_client.search_anime.return_value = {"data": {"Page": {"media": []}}}
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_client):
            
            # Test year range
            await search_anime_anilist(
                query="test",
                start_date_greater="1990-01-01",
                start_date_lesser="2023-12-31",
                ctx=mock_context
            )
            
            # Test score range
            await search_anime_anilist(
                query="test", 
                average_score_greater=70,
                average_score_lesser=100,
                ctx=mock_context
            )
            
            # Test episode range
            await search_anime_anilist(
                query="test",
                episodes_greater=12,
                episodes_lesser=24,
                ctx=mock_context
            )
            
            # Test duration range
            await search_anime_anilist(
                query="test",
                duration_greater=20,
                duration_lesser=30,
                ctx=mock_context
            )

    @pytest.mark.asyncio
    async def test_anilist_tools_empty_results_handling(self, mock_context):
        """Test handling of empty results from AniList."""
        mock_client = AsyncMock()
        
        # Empty search results
        mock_client.search_anime.return_value = {
            "data": {
                "Page": {
                    "media": [],
                    "pageInfo": {"hasNextPage": False, "currentPage": 1, "perPage": 20}
                }
            }
        }
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_client):
            
            result = await search_anime_anilist(query="nonexistent", ctx=mock_context)
            assert result == []
            mock_context.info.assert_called_with("Found 0 anime on AniList")

    @pytest.mark.asyncio
    async def test_anilist_tools_data_quality_scoring(self, mock_context):
        """Test data quality scoring in AniList tools."""
        mock_client = AsyncMock()
        
        # Mock results with different quality scores
        mock_client.search_anime.return_value = [
            {
                "id": 1, 
                "title": {"romaji": "High Quality"},
                "episodes": 12,
                "averageScore": 80,
                "genres": ["Action"],
                "description": "A good anime"
            },
            {
                "id": 2, 
                "title": {"romaji": "Low Quality"}
            }
        ]
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_client):
            
            result = await search_anime_anilist(query="test", ctx=mock_context)
            
            assert len(result) == 2
            # The exact score depends on the internal logic of anilist_tools.py
            # We can only assert that a higher quality input results in a higher score
            assert result[0]["score"] is not None # High quality should have a score
            assert result[1]["score"] is None # Low quality should have no score or a very low one
            assert result[0]["title"] == "High Quality"
            assert result[1]["title"] == "Low Quality"

    @pytest.mark.asyncio
    async def test_anilist_tools_pagination_info(self, mock_context):
        """Test pagination information handling."""
        mock_client = AsyncMock()
        
        mock_client.search_anime.return_value = [
            {"id": 1, "title": {"romaji": "Test Anime"}}
        ]
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_client):
            
            result = await search_anime_anilist(query="test", ctx=mock_context)
            
            # Check if pagination info is included in context
            info_calls = [call.args[0] for call in mock_context.info.call_args_list]
            assert any("Found 1 anime on AniList" in call for call in info_calls)

    @pytest.mark.asyncio
    async def test_anilist_tools_comprehensive_detail_fields(self, mock_context):
        """Test comprehensive detail field handling."""
        mock_client = AsyncMock()
        
        # Mock comprehensive detail response
        comprehensive_response = {
            "id": 16498,
            "title": {"romaji": "Test Anime"},
            "characters": {
                "nodes": [
                    {"name": {"full": "Character 1"}, "image": {"medium": "char1.jpg"}},
                    {"name": {"full": "Character 2"}, "image": {"medium": "char2.jpg"}}
                ]
            },
            "staff": {
                "nodes": [
                    {"name": {"full": "Director"}, "primaryOccupations": ["Director"]},
                    {"name": {"full": "Writer"}, "primaryOccupations": ["Script"]}
                ]
            },
            "relations": {
                "nodes": [
                    {"id": 25777, "title": {"romaji": "Sequel"}, "type": "ANIME"},
                    {"id": 12345, "title": {"romaji": "Manga"}, "type": "MANGA"}
                ]
            },
            "recommendations": {
                "nodes": [
                    {"mediaRecommendation": {"id": 11061, "title": {"romaji": "Recommended Anime"}}}
                ]
            },
            "trailer": {"id": "xyz123", "site": "youtube"},
            "externalLinks": [
                {"url": "https://example.com", "site": "Official Site"}
            ],
            "streamingEpisodes": [
                {"title": "Episode 1", "url": "https://streaming.com/ep1"}
            ]
        }
        
        mock_client.get_anime_by_id.return_value = comprehensive_response
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_client):
            
            result = await get_anime_anilist(
                anilist_id=16498,
                include_characters=True,
                include_staff=True,
                include_relations=True,
                ctx=mock_context
            )
            
            # Verify all comprehensive fields are included
            assert len(result["characters"]["nodes"]) == 2
            assert len(result["staff"]["nodes"]) == 2
            assert len(result["relations"]["nodes"]) == 2
            assert len(result["recommendations"]["nodes"]) == 1
            assert result["trailer"]["site"] == "youtube"
            assert len(result["external_links"]) == 1
            assert len(result["streaming_episodes"]) == 1


class TestAniListToolsIntegration:
    """Integration tests for AniList tools."""

    @pytest.mark.asyncio
    async def test_anilist_tools_annotation_verification(self):
        """Verify that AniList tools have correct MCP annotations."""
        from src.anime_mcp.tools.anilist_tools import mcp
        
        # Get registered tools
        tools = mcp._tools
        
        # Verify search_anime_anilist annotations
        search_tool = tools.get("search_anime_anilist")
        assert search_tool is not None
        assert search_tool.annotations["title"] == "AniList Anime Search"
        assert search_tool.annotations["readOnlyHint"] is True
        assert search_tool.annotations["idempotentHint"] is True
        
        # Verify get_anime_anilist annotations
        detail_tool = tools.get("get_anime_anilist")
        assert detail_tool is not None
        assert detail_tool.annotations["title"] == "AniList Anime Details"
        assert detail_tool.annotations["readOnlyHint"] is True

    @pytest.mark.asyncio
    async def test_anilist_search_to_detail_workflow(self, mock_context):
        """Test workflow from AniList search to getting details."""
        mock_client = AsyncMock()
        
        # Setup search result
        mock_client.search_anime.return_value = [
            {"id": 16498, "title": {"romaji": "Attack on Titan"}}
        ]
        
        # Setup detail result
        mock_client.get_anime_by_id.return_value = {
            "id": 16498,
            "title": {"romaji": "Attack on Titan"},
            "description": "Detailed description"
        }
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_client):
            
            # Step 1: Search
            search_results = await search_anime_anilist(query="attack", ctx=mock_context)
            assert len(search_results) == 1
            anilist_id = search_results[0]["anilist_id"]
            
            # Step 2: Get details
            detail_result = await get_anime_anilist(anilist_id=anilist_id, ctx=mock_context)
            assert detail_result["id"] == "16498"
            assert detail_result["description"] == "Detailed description"

    @pytest.mark.asyncio
    async def test_anilist_tools_error_resilience(self, mock_context):
        """Test error resilience across multiple scenarios."""
        mock_client = AsyncMock()
        
        error_scenarios = [
            ("Rate limit", "Rate limit exceeded"),
            ("Network timeout", "Request timeout"),
            ("GraphQL syntax", "Syntax error in GraphQL"),
            ("Invalid variables", "Variable validation failed")
        ]
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_client):
            
            for error_type, error_msg in error_scenarios:
                mock_client.search_anime.side_effect = Exception(error_msg)
                
                with pytest.raises(RuntimeError, match=f"AniList search failed: {error_msg}"):
                    await search_anime_anilist(query="test", ctx=mock_context)
                
                mock_context.error.assert_called_with(f"AniList search failed: {error_msg}")
                mock_context.reset_mock()