"""Unit tests for AniList-specific MCP tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.anime_mcp.tools.anilist_tools import (
    search_anime_anilist,
    get_anime_anilist,
    anilist_client,
    anilist_mapper
)
from src.models.universal_anime import UniversalAnime, UniversalSearchParams


class TestAniListTools:
    """Test suite for AniList MCP tools."""

    @pytest.fixture
    def mock_anilist_client(self):
        """Create mock AniList client."""
        client = AsyncMock()
        client.search_anime.return_value = {
            "data": {
                "Page": {
                    "media": [
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
                    ],
                    "pageInfo": {
                        "hasNextPage": False,
                        "currentPage": 1,
                        "perPage": 20
                    }
                }
            }
        }
        
        client.get_anime_by_id.return_value = {
            "data": {
                "Media": {
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
            }
        }
        
        return client

    @pytest.fixture
    def mock_anilist_mapper(self):
        """Create mock AniList mapper."""
        mapper = MagicMock()
        
        # Mock to_anilist_search_params
        mapper.to_anilist_search_params.return_value = {
            "search": "attack on titan",
            "perPage": 20,
            "page": 1,
            "format_in": ["TV"],
            "status_in": ["FINISHED"]
        }
        
        # Mock to_universal_anime
        mapper.to_universal_anime.return_value = UniversalAnime(
            id="anilist_16498",
            title="Attack on Titan",
            type_format="TV",
            episodes=25,
            score=8.5,
            year=2013,
            status="FINISHED",
            genres=["Action", "Drama", "Fantasy"],
            studios=["Wit Studio"],
            description="Humanity fights against titans",
            image_url="https://example.com/aot.jpg",
            data_quality_score=0.92
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
    async def test_search_anime_anilist_success(self, mock_anilist_client, mock_anilist_mapper, mock_context):
        """Test successful AniList anime search."""
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_anilist_client), \
             patch('src.anime_mcp.tools.anilist_tools.anilist_mapper', mock_anilist_mapper):
            
            result = await search_anime_anilist(
                query="attack on titan",
                per_page=20,
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == 1
            
            anime = result[0]
            assert anime["id"] == "anilist_16498"
            assert anime["title"] == "Attack on Titan"
            assert anime["anilist_id"] == 16498
            assert anime["anilist_score"] == 85
            assert anime["source_platform"] == "anilist"
            
            # Verify client calls
            mock_anilist_client.search_anime.assert_called_once()
            mock_anilist_mapper.to_anilist_search_params.assert_called_once()
            mock_anilist_mapper.to_universal_anime.assert_called_once()
            
            # Verify context calls
            mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_search_anime_anilist_with_filters(self, mock_anilist_client, mock_anilist_mapper, mock_context):
        """Test AniList search with comprehensive filters."""
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_anilist_client), \
             patch('src.anime_mcp.tools.anilist_tools.anilist_mapper', mock_anilist_mapper):
            
            result = await search_anime_anilist(
                query="mecha anime",
                formats=["TV", "MOVIE"],
                status=["RELEASING", "FINISHED"],
                genres=["Action", "Mecha"],
                tags=["Military", "Space"],
                year_range=[2020, 2023],
                score_range=[70, 100],
                episode_range=[12, 24],
                duration_range=[20, 30],
                studios=["Sunrise", "Mappa"],
                country_of_origin="JP",
                source=["MANGA", "ORIGINAL"],
                season="SPRING",
                season_year=2023,
                is_adult=False,
                on_list=True,
                sort=["SCORE_DESC", "POPULARITY_DESC"],
                per_page=25,
                page=2,
                ctx=mock_context
            )
            
            # Verify mapper was called with comprehensive params
            call_args = mock_anilist_mapper.to_anilist_search_params.call_args
            universal_params = call_args[0][0]
            anilist_specific = call_args[0][1]
            
            assert universal_params.query == "mecha anime"
            assert universal_params.limit == 25
            
            assert anilist_specific["formats"] == ["TV", "MOVIE"]
            assert anilist_specific["status"] == ["RELEASING", "FINISHED"]
            assert anilist_specific["genres"] == ["Action", "Mecha"]
            assert anilist_specific["season"] == "SPRING"
            assert anilist_specific["season_year"] == 2023
            assert anilist_specific["is_adult"] == False

    @pytest.mark.asyncio
    async def test_search_anime_anilist_pagination(self, mock_anilist_client, mock_anilist_mapper):
        """Test AniList search pagination limits."""
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_anilist_client), \
             patch('src.anime_mcp.tools.anilist_tools.anilist_mapper', mock_anilist_mapper):
            
            # Test per_page limit (should clamp to 50)
            await search_anime_anilist(query="test", per_page=100)
            
            call_args = mock_anilist_mapper.to_anilist_search_params.call_args
            universal_params = call_args[0][0]
            assert universal_params.limit == 50
            
            # Test page limit (should clamp to 500) 
            await search_anime_anilist(query="test", page=1000)
            
            call_args = mock_anilist_mapper.to_anilist_search_params.call_args
            anilist_specific = call_args[0][1]
            assert anilist_specific["page"] == 500

    @pytest.mark.asyncio
    async def test_search_anime_anilist_no_client(self, mock_context):
        """Test AniList search when client not available."""
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', None):
            
            with pytest.raises(RuntimeError, match="AniList client not available"):
                await search_anime_anilist(query="test", ctx=mock_context)
            
            mock_context.error.assert_called_with("AniList client not configured")

    @pytest.mark.asyncio
    async def test_search_anime_anilist_graphql_error(self, mock_anilist_client, mock_anilist_mapper, mock_context):
        """Test AniList search when GraphQL returns error."""
        mock_anilist_client.search_anime.return_value = {
            "errors": [{"message": "GraphQL syntax error"}]
        }
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_anilist_client), \
             patch('src.anime_mcp.tools.anilist_tools.anilist_mapper', mock_anilist_mapper):
            
            with pytest.raises(RuntimeError, match="AniList search failed: GraphQL errors"):
                await search_anime_anilist(query="test", ctx=mock_context)

    @pytest.mark.asyncio
    async def test_search_anime_anilist_client_exception(self, mock_anilist_client, mock_anilist_mapper, mock_context):
        """Test AniList search when client raises exception."""
        mock_anilist_client.search_anime.side_effect = Exception("Network error")
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_anilist_client), \
             patch('src.anime_mcp.tools.anilist_tools.anilist_mapper', mock_anilist_mapper):
            
            with pytest.raises(RuntimeError, match="AniList search failed: Network error"):
                await search_anime_anilist(query="test", ctx=mock_context)

    @pytest.mark.asyncio
    async def test_search_anime_anilist_mapping_error(self, mock_anilist_client, mock_anilist_mapper, mock_context):
        """Test AniList search when individual result mapping fails."""
        mock_anilist_mapper.to_universal_anime.side_effect = Exception("Mapping error")
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_anilist_client), \
             patch('src.anime_mcp.tools.anilist_tools.anilist_mapper', mock_anilist_mapper):
            
            result = await search_anime_anilist(query="test", ctx=mock_context)
            
            # Should return empty list when all mappings fail
            assert result == []
            mock_context.error.assert_called_with("Failed to process AniList result: Mapping error")

    @pytest.mark.asyncio
    async def test_get_anime_anilist_success(self, mock_anilist_client, mock_anilist_mapper, mock_context):
        """Test successful AniList anime detail retrieval."""
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_anilist_client), \
             patch('src.anime_mcp.tools.anilist_tools.anilist_mapper', mock_anilist_mapper):
            
            result = await get_anime_anilist(
                anilist_id=16498,
                include_characters=True,
                include_staff=True,
                include_relations=True,
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, dict)
            assert result["id"] == "anilist_16498"
            assert result["title"] == "Attack on Titan"
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
            query = call_args[0][1]
            assert "characters" in query
            assert "staff" in query
            assert "relations" in query

    @pytest.mark.asyncio
    async def test_get_anime_anilist_minimal_options(self, mock_anilist_client, mock_anilist_mapper):
        """Test AniList anime detail with minimal options."""
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_anilist_client), \
             patch('src.anime_mcp.tools.anilist_tools.anilist_mapper', mock_anilist_mapper):
            
            result = await get_anime_anilist(
                anilist_id=16498,
                include_characters=False,
                include_staff=False,
                include_relations=False
            )
            
            # Should have empty optional data
            assert result["characters"] == []
            assert result["staff"] == []
            assert result["relations"] == []

    @pytest.mark.asyncio
    async def test_get_anime_anilist_not_found(self, mock_anilist_client, mock_anilist_mapper, mock_context):
        """Test AniList anime detail when anime not found."""
        mock_anilist_client.get_anime_by_id.return_value = {"data": {"Media": None}}
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_anilist_client), \
             patch('src.anime_mcp.tools.anilist_tools.anilist_mapper', mock_anilist_mapper):
            
            result = await get_anime_anilist(anilist_id=99999, ctx=mock_context)
            
            assert result is None
            mock_context.info.assert_called_with("Anime with AniList ID 99999 not found")

    @pytest.mark.asyncio
    async def test_get_anime_anilist_graphql_error(self, mock_anilist_client, mock_anilist_mapper, mock_context):
        """Test AniList anime detail when GraphQL returns error."""
        mock_anilist_client.get_anime_by_id.return_value = {
            "errors": [{"message": "Variable '$id' expected value of type 'Int!'"}]
        }
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_anilist_client), \
             patch('src.anime_mcp.tools.anilist_tools.anilist_mapper', mock_anilist_mapper):
            
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
        mock_mapper = MagicMock()
        mock_mapper.to_anilist_search_params.return_value = {}
        mock_client.search_anime.return_value = {"data": {"Page": {"media": []}}}
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_client), \
             patch('src.anime_mcp.tools.anilist_tools.anilist_mapper', mock_mapper):
            
            # Test all valid formats
            valid_formats = ["TV", "TV_SHORT", "MOVIE", "SPECIAL", "OVA", "ONA", "MUSIC"]
            for format_type in valid_formats:
                await search_anime_anilist(query="test", formats=[format_type])
            
            # Test all valid statuses
            valid_statuses = ["FINISHED", "RELEASING", "NOT_YET_RELEASED", "CANCELLED", "HIATUS"]
            for status in valid_statuses:
                await search_anime_anilist(query="test", status=[status])
            
            # Test all valid seasons
            valid_seasons = ["WINTER", "SPRING", "SUMMER", "FALL"]
            for season in valid_seasons:
                await search_anime_anilist(query="test", season=season)
            
            # Test all valid sources
            valid_sources = ["ORIGINAL", "MANGA", "LIGHT_NOVEL", "VISUAL_NOVEL", "VIDEO_GAME", "OTHER", "NOVEL", "DOUJINSHI", "ANIME", "WEB_NOVEL", "LIVE_ACTION", "GAME", "COMIC", "MULTIMEDIA_PROJECT", "PICTURE_BOOK"]
            for source in valid_sources:
                await search_anime_anilist(query="test", source=[source])
            
            # Test all valid sort options
            valid_sorts = ["ID", "TITLE_ROMAJI", "TITLE_ENGLISH", "TITLE_NATIVE", "TYPE", "FORMAT", "START_DATE", "END_DATE", "SCORE", "POPULARITY", "TRENDING", "EPISODES", "DURATION", "STATUS", "CHAPTERS", "VOLUMES", "UPDATED_AT", "SEARCH_MATCH", "FAVOURITES"]
            for sort in valid_sorts:
                await search_anime_anilist(query="test", sort=[sort])

    @pytest.mark.asyncio
    async def test_anilist_tools_range_validation(self, mock_context):
        """Test range parameter validation."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        mock_mapper.to_anilist_search_params.return_value = {}
        mock_client.search_anime.return_value = {"data": {"Page": {"media": []}}}
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_client), \
             patch('src.anime_mcp.tools.anilist_tools.anilist_mapper', mock_mapper):
            
            # Test year range
            await search_anime_anilist(
                query="test",
                year_range=[1990, 2023],
                ctx=mock_context
            )
            
            # Test score range
            await search_anime_anilist(
                query="test", 
                score_range=[70, 100],
                ctx=mock_context
            )
            
            # Test episode range
            await search_anime_anilist(
                query="test",
                episode_range=[12, 24],
                ctx=mock_context
            )
            
            # Test duration range
            await search_anime_anilist(
                query="test",
                duration_range=[20, 30],
                ctx=mock_context
            )

    @pytest.mark.asyncio
    async def test_anilist_tools_empty_results_handling(self, mock_context):
        """Test handling of empty results from AniList."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        # Empty search results
        mock_client.search_anime.return_value = {
            "data": {
                "Page": {
                    "media": [],
                    "pageInfo": {"hasNextPage": False, "currentPage": 1, "perPage": 20}
                }
            }
        }
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_client), \
             patch('src.anime_mcp.tools.anilist_tools.anilist_mapper', mock_mapper):
            
            result = await search_anime_anilist(query="nonexistent", ctx=mock_context)
            assert result == []
            mock_context.info.assert_called_with("Found 0 anime on AniList")

    @pytest.mark.asyncio
    async def test_anilist_tools_data_quality_scoring(self, mock_context):
        """Test data quality scoring in AniList tools."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        # Mock results with different quality scores
        mock_client.search_anime.return_value = {
            "data": {
                "Page": {
                    "media": [
                        {"id": 1, "title": {"romaji": "High Quality"}},
                        {"id": 2, "title": {"romaji": "Low Quality"}}
                    ]
                }
            }
        }
        
        mock_mapper.to_universal_anime.side_effect = [
            UniversalAnime(id="anilist_1", title="High Quality", data_quality_score=0.95),
            UniversalAnime(id="anilist_2", title="Low Quality", data_quality_score=0.65)
        ]
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_client), \
             patch('src.anime_mcp.tools.anilist_tools.anilist_mapper', mock_mapper):
            
            result = await search_anime_anilist(query="test", ctx=mock_context)
            
            assert len(result) == 2
            assert result[0]["data_quality_score"] == 0.95
            assert result[1]["data_quality_score"] == 0.65

    @pytest.mark.asyncio
    async def test_anilist_tools_pagination_info(self, mock_context):
        """Test pagination information handling."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        mock_client.search_anime.return_value = {
            "data": {
                "Page": {
                    "media": [{"id": 1, "title": {"romaji": "Test Anime"}}],
                    "pageInfo": {
                        "hasNextPage": True,
                        "currentPage": 1,
                        "perPage": 20,
                        "total": 100,
                        "lastPage": 5
                    }
                }
            }
        }
        
        mock_mapper.to_universal_anime.return_value = UniversalAnime(
            id="anilist_1", title="Test Anime", data_quality_score=0.8
        )
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_client), \
             patch('src.anime_mcp.tools.anilist_tools.anilist_mapper', mock_mapper):
            
            result = await search_anime_anilist(query="test", ctx=mock_context)
            
            # Check if pagination info is included in context
            info_calls = [call.args[0] for call in mock_context.info.call_args_list]
            assert any("page 1" in call.lower() for call in info_calls)

    @pytest.mark.asyncio
    async def test_anilist_tools_comprehensive_detail_fields(self, mock_context):
        """Test comprehensive detail field handling."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        # Mock comprehensive detail response
        comprehensive_response = {
            "data": {
                "Media": {
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
            }
        }
        
        mock_client.get_anime_by_id.return_value = comprehensive_response
        mock_mapper.to_universal_anime.return_value = UniversalAnime(
            id="anilist_16498", title="Test Anime", data_quality_score=0.9
        )
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_client), \
             patch('src.anime_mcp.tools.anilist_tools.anilist_mapper', mock_mapper):
            
            result = await get_anime_anilist(
                anilist_id=16498,
                include_characters=True,
                include_staff=True,
                include_relations=True,
                ctx=mock_context
            )
            
            # Verify all comprehensive fields are included
            assert len(result["characters"]) == 2
            assert len(result["staff"]) == 2
            assert len(result["relations"]) == 2
            assert len(result["recommendations"]) == 1
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
        mock_mapper = MagicMock()
        
        # Setup search result
        mock_client.search_anime.return_value = {
            "data": {
                "Page": {
                    "media": [{"id": 16498, "title": {"romaji": "Attack on Titan"}}]
                }
            }
        }
        
        # Setup detail result
        mock_client.get_anime_by_id.return_value = {
            "data": {
                "Media": {
                    "id": 16498,
                    "title": {"romaji": "Attack on Titan"},
                    "description": "Detailed description"
                }
            }
        }
        
        search_universal = UniversalAnime(
            id="anilist_16498", title="Attack on Titan", data_quality_score=0.9
        )
        detail_universal = UniversalAnime(
            id="anilist_16498", title="Attack on Titan", 
            description="Detailed description", data_quality_score=0.95
        )
        
        mock_mapper.to_universal_anime.side_effect = [search_universal, detail_universal]
        mock_mapper.to_anilist_search_params.return_value = {"search": "attack"}
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_client), \
             patch('src.anime_mcp.tools.anilist_tools.anilist_mapper', mock_mapper):
            
            # Step 1: Search
            search_results = await search_anime_anilist(query="attack", ctx=mock_context)
            assert len(search_results) == 1
            anilist_id = search_results[0]["anilist_id"]
            
            # Step 2: Get details
            detail_result = await get_anime_anilist(anilist_id=anilist_id, ctx=mock_context)
            assert detail_result["id"] == "anilist_16498"
            assert detail_result["description"] == "Detailed description"

    @pytest.mark.asyncio
    async def test_anilist_tools_error_resilience(self, mock_context):
        """Test error resilience across multiple scenarios."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        error_scenarios = [
            ("Rate limit", "Rate limit exceeded"),
            ("Network timeout", "Request timeout"),
            ("GraphQL syntax", "Syntax error in GraphQL"),
            ("Invalid variables", "Variable validation failed")
        ]
        
        with patch('src.anime_mcp.tools.anilist_tools.anilist_client', mock_client), \
             patch('src.anime_mcp.tools.anilist_tools.anilist_mapper', mock_mapper):
            
            for error_type, error_msg in error_scenarios:
                mock_client.search_anime.side_effect = Exception(error_msg)
                
                with pytest.raises(RuntimeError, match=f"AniList search failed: {error_msg}"):
                    await search_anime_anilist(query="test", ctx=mock_context)
                
                mock_context.error.assert_called_with(f"AniList search failed: {error_msg}")
                mock_context.reset_mock()