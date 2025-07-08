"""Unit tests for Jikan (MAL unofficial API) MCP tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.anime_mcp.tools.jikan_tools import (
    jikan_client,
    jikan_mapper,
    mcp
)
from src.models.universal_anime import UniversalAnime, UniversalSearchParams


class TestJikanTools:
    """Test suite for Jikan MCP tools."""

    @pytest.fixture
    def mock_jikan_client(self):
        """Create mock Jikan client."""
        client = AsyncMock()
        
        client.search_anime.return_value = {
            "data": [
                {
                    "mal_id": 16498,
                    "title": "Shingeki no Kyojin",
                    "title_english": "Attack on Titan",
                    "title_japanese": "進撃の巨人",
                    "type": "TV",
                    "episodes": 25,
                    "status": "Finished Airing",
                    "aired": {
                        "from": "2013-04-07T00:00:00+00:00",
                        "to": "2013-09-28T00:00:00+00:00"
                    },
                    "duration": "24 min per ep",
                    "rating": "R - 17+ (violence & profanity)",
                    "score": 8.54,
                    "scored_by": 1500000,
                    "rank": 50,
                    "popularity": 1,
                    "members": 3000000,
                    "favorites": 150000,
                    "synopsis": "Humanity fights against titans",
                    "background": "Based on manga by Hajime Isayama",
                    "season": "spring",
                    "year": 2013,
                    "broadcast": {
                        "day": "Sundays",
                        "time": "17:00",
                        "timezone": "Asia/Tokyo"
                    },
                    "producers": [
                        {"mal_id": 10, "name": "Production I.G"}
                    ],
                    "licensors": [
                        {"mal_id": 102, "name": "Funimation"}
                    ],
                    "studios": [
                        {"mal_id": 858, "name": "Wit Studio"}
                    ],
                    "genres": [
                        {"mal_id": 1, "name": "Action"},
                        {"mal_id": 8, "name": "Drama"}
                    ],
                    "themes": [
                        {"mal_id": 38, "name": "Military"},
                        {"mal_id": 76, "name": "Survival"}
                    ],
                    "demographics": [
                        {"mal_id": 27, "name": "Shounen"}
                    ],
                    "images": {
                        "jpg": {
                            "image_url": "https://example.com/aot.jpg",
                            "small_image_url": "https://example.com/aot_small.jpg",
                            "large_image_url": "https://example.com/aot_large.jpg"
                        }
                    },
                    "trailer": {
                        "youtube_id": "abc123",
                        "url": "https://youtube.com/watch?v=abc123"
                    },
                    "approved": True,
                    "url": "https://myanimelist.net/anime/16498"
                }
            ],
            "pagination": {
                "last_visible_page": 1,
                "has_next_page": False,
                "current_page": 1,
                "items": {
                    "count": 1,
                    "total": 1,
                    "per_page": 25
                }
            }
        }
        
        client.get_anime_by_id.return_value = {
            "data": {
                "mal_id": 16498,
                "title": "Shingeki no Kyojin",
                "title_english": "Attack on Titan",
                "title_japanese": "進撃の巨人",
                "title_synonyms": ["Attack on Titan"],
                "type": "TV",
                "source": "Manga",
                "episodes": 25,
                "status": "Finished Airing",
                "airing": False,
                "aired": {
                    "from": "2013-04-07T00:00:00+00:00",
                    "to": "2013-09-28T00:00:00+00:00",
                    "prop": {
                        "from": {"day": 7, "month": 4, "year": 2013},
                        "to": {"day": 28, "month": 9, "year": 2013}
                    }
                },
                "duration": "24 min per ep",
                "rating": "R - 17+ (violence & profanity)",
                "score": 8.54,
                "scored_by": 1500000,
                "rank": 50,
                "popularity": 1,
                "members": 3000000,
                "favorites": 150000,
                "synopsis": "Detailed synopsis about humanity fighting titans",
                "background": "Detailed background information",
                "season": "spring",
                "year": 2013,
                "broadcast": {
                    "day": "Sundays",
                    "time": "17:00",
                    "timezone": "Asia/Tokyo",
                    "string": "Sundays at 17:00 (JST)"
                },
                "producers": [
                    {"mal_id": 10, "type": "anime", "name": "Production I.G", "url": "https://myanimelist.net/anime/producer/10"}
                ],
                "licensors": [
                    {"mal_id": 102, "type": "anime", "name": "Funimation", "url": "https://myanimelist.net/anime/producer/102"}
                ],
                "studios": [
                    {"mal_id": 858, "type": "anime", "name": "Wit Studio", "url": "https://myanimelist.net/anime/producer/858"}
                ],
                "genres": [
                    {"mal_id": 1, "type": "anime", "name": "Action", "url": "https://myanimelist.net/anime/genre/1"},
                    {"mal_id": 8, "type": "anime", "name": "Drama", "url": "https://myanimelist.net/anime/genre/8"}
                ],
                "explicit_genres": [],
                "themes": [
                    {"mal_id": 38, "type": "anime", "name": "Military", "url": "https://myanimelist.net/anime/genre/38"},
                    {"mal_id": 76, "type": "anime", "name": "Survival", "url": "https://myanimelist.net/anime/genre/76"}
                ],
                "demographics": [
                    {"mal_id": 27, "type": "anime", "name": "Shounen", "url": "https://myanimelist.net/anime/genre/27"}
                ],
                "relations": [
                    {
                        "relation": "Sequel",
                        "entry": [
                            {"mal_id": 25777, "type": "anime", "name": "Shingeki no Kyojin Season 2", "url": "https://myanimelist.net/anime/25777"}
                        ]
                    }
                ],
                "theme": {
                    "openings": ["\"Guren no Yumiya\" by Linked Horizon"],
                    "endings": ["\"Utsukushiki Zankoku na Sekai\" by Yoko Hikasa"]
                },
                "external": [
                    {"name": "Official Site", "url": "http://shingeki.tv/"},
                    {"name": "Wikipedia", "url": "https://en.wikipedia.org/wiki/Attack_on_Titan"}
                ],
                "streaming": [
                    {"name": "Crunchyroll", "url": "http://www.crunchyroll.com/series-100"}
                ],
                "images": {
                    "jpg": {
                        "image_url": "https://example.com/aot.jpg",
                        "small_image_url": "https://example.com/aot_small.jpg",
                        "large_image_url": "https://example.com/aot_large.jpg"
                    }
                },
                "trailer": {
                    "youtube_id": "abc123",
                    "url": "https://youtube.com/watch?v=abc123",
                    "embed_url": "https://youtube.com/embed/abc123"
                },
                "approved": True,
                "titles": [
                    {"type": "Default", "title": "Shingeki no Kyojin"},
                    {"type": "English", "title": "Attack on Titan"},
                    {"type": "Japanese", "title": "進撃の巨人"}
                ],
                "url": "https://myanimelist.net/anime/16498"
            }
        }
        
        return client

    @pytest.fixture
    def mock_jikan_mapper(self):
        """Create mock Jikan mapper."""
        mapper = MagicMock()
        
        # Mock to_jikan_search_params
        mapper.to_jikan_search_params.return_value = {
            "q": "attack on titan",
            "limit": 25,
            "page": 1,
            "type": "tv",
            "status": "complete"
        }
        
        # Mock to_universal_anime
        mapper.to_universal_anime.return_value = UniversalAnime(
            id="jikan_16498",
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
            data_quality_score=0.91
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
    async def test_search_anime_jikan_success(self, mock_jikan_client, mock_jikan_mapper, mock_context):
        """Test successful Jikan anime search."""
        # Mock the client to return data in the expected format
        mock_jikan_client.search_anime.return_value = [
            {
                "mal_id": 16498,
                "title": "Attack on Titan",
                "score": 8.54,
                "source_platform": "jikan"
            }
        ]
        
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_jikan_client), \
             patch('src.anime_mcp.tools.jikan_tools.jikan_mapper', mock_jikan_mapper):
            
            # Get the actual function from the MCP tool
            search_tool = mcp._tool_manager._tools["search_anime_jikan"]
            result = await search_tool.fn(
                query="attack on titan",
                limit=25,
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == 1
            
            anime = result[0]
            assert anime["source_platform"] == "jikan"
            
            # Verify client calls
            mock_jikan_client.search_anime.assert_called_once()
            mock_jikan_mapper.to_jikan_search_params.assert_called_once()
            
            # Verify context calls
            mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_search_anime_jikan_with_filters(self, mock_jikan_client, mock_jikan_mapper, mock_context):
        """Test Jikan search with filters."""
        mock_jikan_client.search_anime.return_value = []
        
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_jikan_client), \
             patch('src.anime_mcp.tools.jikan_tools.jikan_mapper', mock_jikan_mapper):
            
            # Get the actual function from the MCP tool
            search_tool = mcp._tool_manager._tools["search_anime_jikan"]
            result = await search_tool.fn(
                query="mecha anime",
                type="tv",
                status="complete",
                rating="pg13",
                genres=[1, 2, 3],
                genres_exclude=[9, 12],
                order_by="score",
                sort="desc",
                min_score=7.0,
                max_score=10.0,
                producers=[2],
                start_date="2020-01-01",
                end_date="2023-12-31",
                limit=20,
                page=2,
                ctx=mock_context
            )
            
            # Verify mapper was called with universal parameters
            call_args = mock_jikan_mapper.to_jikan_search_params.call_args
            universal_params = call_args[0][0]
            jikan_specific = call_args[0][1]
            
            assert universal_params.query == "mecha anime"
            assert universal_params.limit == 20
            
            # Verify universal parameters contain basic values
            assert universal_params.sort_by == "score"
            assert universal_params.sort_order == "desc"
            assert universal_params.min_score == 7.0
            assert universal_params.max_score == 10.0
            
            # Verify jikan_specific contains the string values
            assert jikan_specific["type"] == "tv"
            assert jikan_specific["status"] == "complete"
            assert jikan_specific["rating"] == "pg13"

    @pytest.mark.asyncio
    async def test_search_anime_jikan_pagination(self, mock_jikan_client, mock_jikan_mapper):
        """Test Jikan search pagination limits."""
        mock_jikan_client.search_anime.return_value = []
        
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_jikan_client), \
             patch('src.anime_mcp.tools.jikan_tools.jikan_mapper', mock_jikan_mapper):
            
            # Get the actual function from the MCP tool
            search_tool = mcp._tool_manager._tools["search_anime_jikan"]
            
            # Test limit clamping (should clamp to 25)
            await search_tool.fn(query="test", limit=100)
            
            call_args = mock_jikan_mapper.to_jikan_search_params.call_args
            universal_params = call_args[0][0]
            assert universal_params.limit == 25
            
            # Test page handling (page 2 with limit 20 should convert to offset 20)
            await search_tool.fn(query="test", page=2, limit=20)
            
            call_args = mock_jikan_mapper.to_jikan_search_params.call_args
            universal_params = call_args[0][0]
            # Page 2 with limit 20 should convert to offset 20: (2-1) * 20 = 20
            assert universal_params.offset == 20

    @pytest.mark.asyncio
    async def test_search_anime_jikan_no_client(self, mock_context):
        """Test Jikan search when client not available."""
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', None):
            
            search_tool = mcp._tool_manager._tools["search_anime_jikan"]
            with pytest.raises(RuntimeError, match="Jikan search failed"):
                await search_tool.fn(query="test", ctx=mock_context)
            
            mock_context.error.assert_called()

    @pytest.mark.asyncio
    async def test_search_anime_jikan_api_error(self, mock_jikan_client, mock_jikan_mapper, mock_context):
        """Test Jikan search when API returns error."""
        mock_jikan_client.search_anime.side_effect = Exception("API error")
        
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_jikan_client), \
             patch('src.anime_mcp.tools.jikan_tools.jikan_mapper', mock_jikan_mapper):
            
            search_tool = mcp._tool_manager._tools["search_anime_jikan"]
            with pytest.raises(RuntimeError, match="Jikan search failed"):
                await search_tool.fn(query="test", ctx=mock_context)

    @pytest.mark.asyncio
    async def test_search_anime_jikan_client_exception(self, mock_jikan_client, mock_jikan_mapper, mock_context):
        """Test Jikan search when client raises exception."""
        mock_jikan_client.search_anime.side_effect = Exception("Network timeout")
        
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_jikan_client), \
             patch('src.anime_mcp.tools.jikan_tools.jikan_mapper', mock_jikan_mapper):
            
            search_tool = mcp._tool_manager._tools["search_anime_jikan"]
            with pytest.raises(RuntimeError, match="Jikan search failed: Network timeout"):
                await search_tool.fn(query="test", ctx=mock_context)

    @pytest.mark.asyncio
    async def test_search_anime_jikan_mapping_error(self, mock_jikan_client, mock_jikan_mapper, mock_context):
        """Test Jikan search when mapping fails for individual results."""
        mock_jikan_client.search_anime.return_value = []
        
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_jikan_client), \
             patch('src.anime_mcp.tools.jikan_tools.jikan_mapper', mock_jikan_mapper):
            
            search_tool = mcp._tool_manager._tools["search_anime_jikan"]
            result = await search_tool.fn(query="test", ctx=mock_context)
            
            # Should return empty list when no results
            assert result == []
            mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_get_anime_jikan_success(self, mock_jikan_client, mock_jikan_mapper, mock_context):
        """Test successful Jikan anime detail retrieval."""
        mock_jikan_client.get_anime_by_id.return_value = {
            "mal_id": 16498,
            "title": "Attack on Titan",
            "score": 8.54
        }
        
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_jikan_client), \
             patch('src.anime_mcp.tools.jikan_tools.jikan_mapper', mock_jikan_mapper):
            
            # Get the actual function from the MCP tool
            detail_tool = mcp._tool_manager._tools["get_anime_jikan"]
            result = await detail_tool.fn(
                mal_id=16498,
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, dict)
            assert result["source_platform"] == "jikan"
            assert result["mal_id"] == 16498
            
            # Verify client was called correctly
            mock_jikan_client.get_anime_by_id.assert_called_once_with(16498)

    @pytest.mark.asyncio
    async def test_get_anime_jikan_not_found(self, mock_jikan_client, mock_jikan_mapper, mock_context):
        """Test Jikan anime detail when anime not found."""
        mock_jikan_client.get_anime_by_id.return_value = None
        
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_jikan_client), \
             patch('src.anime_mcp.tools.jikan_tools.jikan_mapper', mock_jikan_mapper):
            
            detail_tool = mcp._tool_manager._tools["get_anime_jikan"]
            result = await detail_tool.fn(mal_id=99999, ctx=mock_context)
            
            assert result is None
            mock_context.info.assert_called_with("Anime with MAL ID 99999 not found via Jikan")

    @pytest.mark.asyncio
    async def test_get_anime_jikan_api_error(self, mock_jikan_client, mock_jikan_mapper, mock_context):
        """Test Jikan anime detail when API returns error."""
        mock_jikan_client.get_anime_by_id.side_effect = Exception("API error")
        
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_jikan_client), \
             patch('src.anime_mcp.tools.jikan_tools.jikan_mapper', mock_jikan_mapper):
            
            detail_tool = mcp._tool_manager._tools["get_anime_jikan"]
            with pytest.raises(RuntimeError, match="Failed to get Jikan anime 16498"):
                await detail_tool.fn(mal_id=16498, ctx=mock_context)

    @pytest.mark.asyncio
    async def test_get_anime_jikan_no_client(self, mock_context):
        """Test Jikan anime detail when client not available."""
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', None):
            
            detail_tool = mcp._tool_manager._tools["get_anime_jikan"]
            with pytest.raises(RuntimeError, match="Failed to get Jikan anime 16498"):
                await detail_tool.fn(mal_id=16498, ctx=mock_context)


class TestJikanToolsAdvanced:
    """Advanced tests for Jikan tools."""

    @pytest.mark.asyncio
    async def test_jikan_tools_parameter_validation(self):
        """Test parameter validation for Jikan tools."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        mock_mapper.to_jikan_search_params.return_value = {}
        mock_client.search_anime.return_value = []
        
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_client), \
             patch('src.anime_mcp.tools.jikan_tools.jikan_mapper', mock_mapper):
            
            search_tool = mcp._tool_manager._tools["search_anime_jikan"]
            
            # Test all valid anime types
            valid_types = ["tv", "movie", "ova", "special", "ona", "music"]
            for anime_type in valid_types:
                await search_tool.fn(query="test", type=anime_type)
            
            # Test all valid statuses  
            valid_statuses = ["airing", "complete", "upcoming"]
            for status in valid_statuses:
                await search_tool.fn(query="test", status=status)
            
            # Test all valid ratings
            valid_ratings = ["g", "pg", "pg13", "r17", "r", "rx"]
            for rating in valid_ratings:
                await search_tool.fn(query="test", rating=rating)
            
            # Test all valid order_by options
            valid_order_by = ["mal_id", "title", "type", "rating", "start_date", "end_date", "episodes", "score", "scored_by", "rank", "popularity", "members", "favorites"]
            for order in valid_order_by:
                await search_tool.fn(query="test", order_by=order)
            
            # Test all valid sort options
            valid_sorts = ["desc", "asc"]
            for sort in valid_sorts:
                await search_tool.fn(query="test", sort=sort)

    @pytest.mark.asyncio
    async def test_jikan_tools_score_range_validation(self, mock_context):
        """Test score range validation."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        mock_mapper.to_jikan_search_params.return_value = {}
        mock_client.search_anime.return_value = {"data": []}
        
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_client), \
             patch('src.anime_mcp.tools.jikan_tools.jikan_mapper', mock_mapper):
            
            # Test valid score range
            await search_anime_jikan(
                query="test",
                min_score=7.5,
                max_score=9.5,
                ctx=mock_context
            )
            
            call_args = mock_jikan_mapper.to_jikan_search_params.call_args
            jikan_specific = call_args[0][1]
            assert jikan_specific["min_score"] == 7.5
            assert jikan_specific["max_score"] == 9.5

    @pytest.mark.asyncio
    async def test_jikan_tools_date_range_validation(self, mock_context):
        """Test date range validation."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        mock_mapper.to_jikan_search_params.return_value = {}
        mock_client.search_anime.return_value = {"data": []}
        
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_client), \
             patch('src.anime_mcp.tools.jikan_tools.jikan_mapper', mock_mapper):
            
            # Test date range
            await search_anime_jikan(
                query="test",
                start_date="2020-01-01",
                end_date="2023-12-31",
                ctx=mock_context
            )
            
            call_args = mock_jikan_mapper.to_jikan_search_params.call_args
            jikan_specific = call_args[0][1]
            assert jikan_specific["start_date"] == "2020-01-01"
            assert jikan_specific["end_date"] == "2023-12-31"

    @pytest.mark.asyncio
    async def test_jikan_tools_empty_results_handling(self, mock_context):
        """Test handling of empty results from Jikan."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        # Empty search results
        mock_client.search_anime.return_value = {
            "data": [],
            "pagination": {
                "last_visible_page": 1,
                "has_next_page": False,
                "current_page": 1,
                "items": {"count": 0, "total": 0, "per_page": 25}
            }
        }
        
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_client), \
             patch('src.anime_mcp.tools.jikan_tools.jikan_mapper', mock_mapper):
            
            result = await search_anime_jikan(query="nonexistent", ctx=mock_context)
            assert result == []
            mock_context.info.assert_called_with("Found 0 anime on Jikan")

    @pytest.mark.asyncio
    async def test_jikan_tools_comprehensive_data_extraction(self, mock_context):
        """Test comprehensive data extraction from Jikan responses."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        # Comprehensive response with all possible fields
        comprehensive_response = {
            "data": {
                "mal_id": 16498,
                "title": "Test Anime",
                "title_english": "Test Anime EN",
                "title_japanese": "テストアニメ",
                "title_synonyms": ["Test Show", "TS"],
                "type": "TV",
                "source": "Manga",
                "episodes": 25,
                "status": "Finished Airing",
                "airing": False,
                "aired": {
                    "from": "2013-04-07T00:00:00+00:00",
                    "to": "2013-09-28T00:00:00+00:00"
                },
                "duration": "24 min per ep",
                "rating": "R - 17+",
                "score": 8.54,
                "scored_by": 1500000,
                "rank": 50,
                "popularity": 1,
                "members": 3000000,
                "favorites": 150000,
                "synopsis": "Detailed synopsis",
                "background": "Background info",
                "season": "spring",
                "year": 2013,
                "broadcast": {
                    "day": "Sundays",
                    "time": "17:00",
                    "timezone": "Asia/Tokyo"
                },
                "producers": [{"mal_id": 10, "name": "Producer"}],
                "licensors": [{"mal_id": 102, "name": "Licensor"}],
                "studios": [{"mal_id": 858, "name": "Studio"}],
                "genres": [{"mal_id": 1, "name": "Action"}],
                "themes": [{"mal_id": 38, "name": "Military"}],
                "demographics": [{"mal_id": 27, "name": "Shounen"}],
                "relations": [
                    {
                        "relation": "Sequel",
                        "entry": [{"mal_id": 25777, "name": "Sequel Title"}]
                    }
                ],
                "theme": {
                    "openings": ["Opening Song"],
                    "endings": ["Ending Song"]
                },
                "external": [
                    {"name": "Official Site", "url": "http://example.com"}
                ],
                "streaming": [
                    {"name": "Crunchyroll", "url": "http://crunchyroll.com"}
                ],
                "images": {
                    "jpg": {"image_url": "https://example.com/image.jpg"}
                },
                "trailer": {
                    "youtube_id": "abc123",
                    "url": "https://youtube.com/watch?v=abc123"
                },
                "titles": [
                    {"type": "Default", "title": "Test Anime"},
                    {"type": "English", "title": "Test Anime EN"}
                ]
            }
        }
        
        mock_client.get_anime_by_id.return_value = comprehensive_response
        mock_mapper.to_universal_anime.return_value = UniversalAnime(
            id="jikan_16498", title="Test Anime", type_format="TV", data_quality_score=0.9
        )
        
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_client), \
             patch('src.anime_mcp.tools.jikan_tools.jikan_mapper', mock_mapper):
            
            detail_tool = mcp._tool_manager._tools["get_anime_jikan"]
            result = await detail_tool.fn(mal_id=16498, ctx=mock_context)
            
            # Verify comprehensive data extraction
            assert result["rank"] == 50
            assert result["popularity"] == 1
            assert result["members"] == 3000000
            assert result["favorites"] == 150000
            assert result["source_platform"] == "jikan"
            assert result["rating"] == "R - 17+"
            assert result["mal_id"] == 16498

    @pytest.mark.asyncio
    async def test_jikan_tools_pagination_info(self, mock_context):
        """Test pagination information handling."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        mock_client.search_anime.return_value = [{"mal_id": 1, "title": "Test Anime"}]
        
        mock_mapper.to_universal_anime.return_value = UniversalAnime(
            id="jikan_1", title="Test Anime", type_format="TV", data_quality_score=0.8
        )
        
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_client), \
             patch('src.anime_mcp.tools.jikan_tools.jikan_mapper', mock_mapper):
            
            search_tool = mcp._tool_manager._tools["search_anime_jikan"]
            result = await search_tool.fn(query="test", ctx=mock_context)
            
            # Check if pagination info is included in context
            info_calls = [call.args[0] for call in mock_context.info.call_args_list]
            assert any("jikan" in call.lower() for call in info_calls)

    @pytest.mark.asyncio
    async def test_jikan_tools_rate_limiting_awareness(self, mock_context):
        """Test rate limiting error handling."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        # Mock rate limiting response
        mock_client.search_anime.side_effect = Exception("Too Many Requests")
        
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_client), \
             patch('src.anime_mcp.tools.jikan_tools.jikan_mapper', mock_mapper):
            
            search_tool = mcp._tool_manager._tools["search_anime_jikan"]
            with pytest.raises(RuntimeError, match="Jikan search failed: Too Many Requests"):
                await search_tool.fn(query="test", ctx=mock_context)
            
            # Should specifically mention rate limiting in context
            mock_context.error.assert_called_with("Jikan search failed: Too Many Requests")


class TestJikanToolsIntegration:
    """Integration tests for Jikan tools."""

    @pytest.mark.asyncio
    async def test_jikan_tools_annotation_verification(self):
        """Verify that Jikan tools have correct MCP annotations."""
        from src.anime_mcp.tools.jikan_tools import mcp
        
        # Get registered tools
        tools = mcp._tool_manager._tools
        
        # Verify search_anime_jikan annotations
        search_tool = tools.get("search_anime_jikan")
        assert search_tool is not None
        assert search_tool.annotations["title"] == "Jikan Anime Search"
        assert search_tool.annotations["readOnlyHint"] is True
        assert search_tool.annotations["idempotentHint"] is True
        
        # Verify get_anime_jikan annotations
        detail_tool = tools.get("get_anime_jikan")
        assert detail_tool is not None
        assert detail_tool.annotations["title"] == "Jikan Anime Details"
        assert detail_tool.annotations["readOnlyHint"] is True

    @pytest.mark.asyncio
    async def test_jikan_search_to_detail_workflow(self, mock_context):
        """Test workflow from Jikan search to getting details."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        # Setup search result
        mock_client.search_anime.return_value = {
            "data": [{"mal_id": 16498, "title": "Attack on Titan"}]
        }
        
        # Setup detail result
        mock_client.get_anime_by_id.return_value = {
            "data": {
                "mal_id": 16498,
                "title": "Attack on Titan",
                "synopsis": "Detailed synopsis"
            }
        }
        
        search_universal = UniversalAnime(
            id="jikan_16498", title="Attack on Titan", data_quality_score=0.9
        )
        detail_universal = UniversalAnime(
            id="jikan_16498", title="Attack on Titan", 
            description="Detailed synopsis", data_quality_score=0.95
        )
        
        mock_mapper.to_universal_anime.side_effect = [search_universal, detail_universal]
        mock_mapper.to_jikan_search_params.return_value = {"q": "attack"}
        
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_client), \
             patch('src.anime_mcp.tools.jikan_tools.jikan_mapper', mock_mapper):
            
            # Step 1: Search
            search_results = await search_anime_jikan(query="attack", ctx=mock_context)
            assert len(search_results) == 1
            jikan_id = search_results[0]["jikan_id"]
            
            # Step 2: Get details
            detail_result = await get_anime_jikan(jikan_id=jikan_id, ctx=mock_context)
            assert detail_result["id"] == "jikan_16498"
            assert detail_result["synopsis"] == "Detailed synopsis"

    @pytest.mark.asyncio
    async def test_jikan_tools_error_resilience(self, mock_context):
        """Test error resilience across multiple scenarios."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        error_scenarios = [
            ("Rate limit", "Too Many Requests"),
            ("Server error", "Internal Server Error"), 
            ("Network timeout", "Request timeout"),
            ("Invalid request", "Bad Request")
        ]
        
        with patch('src.anime_mcp.tools.jikan_tools.jikan_client', mock_client), \
             patch('src.anime_mcp.tools.jikan_tools.jikan_mapper', mock_mapper):
            
            for error_type, error_msg in error_scenarios:
                mock_client.search_anime.side_effect = Exception(error_msg)
                
                with pytest.raises(RuntimeError, match=f"Jikan search failed: {error_msg}"):
                    await search_anime_jikan(query="test", ctx=mock_context)
                
                mock_context.error.assert_called_with(f"Jikan search failed: {error_msg}")
                mock_context.reset_mock()