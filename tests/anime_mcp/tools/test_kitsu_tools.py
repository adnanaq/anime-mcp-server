"""Unit tests for Kitsu-specific MCP tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.anime_mcp.tools.kitsu_tools import (
    search_anime_kitsu,
    get_anime_kitsu,
    search_streaming_platforms,
    kitsu_client,
    kitsu_mapper
)
from src.models.universal_anime import UniversalAnime, UniversalSearchParams


class TestKitsuTools:
    """Test suite for Kitsu MCP tools."""

    @pytest.fixture
    def mock_kitsu_client(self):
        """Create mock Kitsu client."""
        client = AsyncMock()
        
        client.search_anime.return_value = {
            "data": [
                {
                    "id": "1",
                    "type": "anime",
                    "attributes": {
                        "slug": "attack-on-titan",
                        "synopsis": "Humanity fights against titans",
                        "description": "Detailed description",
                        "titles": {
                            "en": "Attack on Titan",
                            "en_jp": "Shingeki no Kyojin",
                            "ja_jp": "進撃の巨人"
                        },
                        "canonicalTitle": "Attack on Titan",
                        "abbreviatedTitles": ["AoT", "SnK"],
                        "averageRating": "85.4",
                        "ratingFrequencies": {
                            "2": "100",
                            "3": "200",
                            "4": "500",
                            "5": "1000"
                        },
                        "userCount": 50000,
                        "favoritesCount": 15000,
                        "startDate": "2013-04-07",
                        "endDate": "2013-09-28",
                        "nextRelease": None,
                        "popularityRank": 1,
                        "ratingRank": 50,
                        "ageRating": "R",
                        "ageRatingGuide": "17+ (violence & profanity)",
                        "subtype": "TV",
                        "status": "finished",
                        "episodeCount": 25,
                        "episodeLength": 24,
                        "totalLength": 600,
                        "youtubeVideoId": "abc123",
                        "nsfw": False
                    },
                    "relationships": {
                        "genres": {
                            "data": [
                                {"id": "1", "type": "genres"},
                                {"id": "8", "type": "genres"}
                            ]
                        },
                        "categories": {
                            "data": [
                                {"id": "1", "type": "categories"},
                                {"id": "2", "type": "categories"}
                            ]
                        },
                        "productions": {
                            "data": [
                                {"id": "1", "type": "productions"}
                            ]
                        },
                        "streamingLinks": {
                            "data": [
                                {"id": "1", "type": "streamingLinks"},
                                {"id": "2", "type": "streamingLinks"}
                            ]
                        }
                    }
                }
            ],
            "included": [
                {
                    "id": "1",
                    "type": "genres",
                    "attributes": {"name": "Action", "slug": "action"}
                },
                {
                    "id": "8", 
                    "type": "genres",
                    "attributes": {"name": "Drama", "slug": "drama"}
                },
                {
                    "id": "1",
                    "type": "categories",
                    "attributes": {"title": "Military", "slug": "military"}
                },
                {
                    "id": "1",
                    "type": "productions",
                    "attributes": {"role": "studio"},
                    "relationships": {
                        "company": {
                            "data": {"id": "1", "type": "producers"}
                        }
                    }
                },
                {
                    "id": "1",
                    "type": "producers",
                    "attributes": {"name": "Wit Studio", "slug": "wit-studio"}
                },
                {
                    "id": "1",
                    "type": "streamingLinks",
                    "attributes": {
                        "url": "https://crunchyroll.com/attack-on-titan",
                        "subs": ["en", "es", "fr"],
                        "dubs": ["en"]
                    },
                    "relationships": {
                        "streamer": {
                            "data": {"id": "1", "type": "streamers"}
                        }
                    }
                },
                {
                    "id": "1",
                    "type": "streamers",
                    "attributes": {"name": "Crunchyroll", "siteName": "Crunchyroll"}
                }
            ],
            "meta": {
                "count": 1
            }
        }
        
        client.get_anime_by_id.return_value = {
            "data": {
                "id": "1",
                "type": "anime",
                "attributes": {
                    "slug": "attack-on-titan",
                    "synopsis": "Detailed synopsis about humanity fighting titans",
                    "description": "Comprehensive description",
                    "titles": {
                        "en": "Attack on Titan",
                        "en_jp": "Shingeki no Kyojin",
                        "ja_jp": "進撃の巨人"
                    },
                    "canonicalTitle": "Attack on Titan",
                    "abbreviatedTitles": ["AoT", "SnK"],
                    "averageRating": "85.4",
                    "ratingFrequencies": {
                        "2": "100", "3": "200", "4": "500", "5": "1000"
                    },
                    "userCount": 50000,
                    "favoritesCount": 15000,
                    "startDate": "2013-04-07",
                    "endDate": "2013-09-28",
                    "popularityRank": 1,
                    "ratingRank": 50,
                    "ageRating": "R",
                    "ageRatingGuide": "17+ (violence & profanity)",
                    "subtype": "TV",
                    "status": "finished",
                    "episodeCount": 25,
                    "episodeLength": 24,
                    "totalLength": 600,
                    "youtubeVideoId": "abc123",
                    "nsfw": False,
                    "coverImageTopOffset": 200
                },
                "relationships": {
                    "genres": {"data": [{"id": "1", "type": "genres"}]},
                    "categories": {"data": [{"id": "1", "type": "categories"}]},
                    "productions": {"data": [{"id": "1", "type": "productions"}]},
                    "streamingLinks": {"data": [{"id": "1", "type": "streamingLinks"}]},
                    "animeCharacters": {"data": [{"id": "1", "type": "animeCharacters"}]},
                    "animeStaff": {"data": [{"id": "1", "type": "animeStaff"}]}
                }
            },
            "included": [
                {
                    "id": "1",
                    "type": "genres", 
                    "attributes": {"name": "Action", "slug": "action"}
                },
                {
                    "id": "1",
                    "type": "streamingLinks",
                    "attributes": {
                        "url": "https://crunchyroll.com/attack-on-titan",
                        "subs": ["en", "es"],
                        "dubs": ["en"]
                    },
                    "relationships": {
                        "streamer": {"data": {"id": "1", "type": "streamers"}}
                    }
                },
                {
                    "id": "1",
                    "type": "streamers",
                    "attributes": {"name": "Crunchyroll", "siteName": "Crunchyroll"}
                }
            ]
        }
        
        client.search_streaming_platforms.return_value = {
            "data": [
                {
                    "id": "1",
                    "type": "streamers",
                    "attributes": {
                        "name": "Crunchyroll",
                        "siteName": "Crunchyroll",
                        "logo": {
                            "tiny": "https://example.com/cr_tiny.jpg",
                            "small": "https://example.com/cr_small.jpg"
                        }
                    }
                },
                {
                    "id": "2", 
                    "type": "streamers",
                    "attributes": {
                        "name": "Funimation",
                        "siteName": "Funimation",
                        "logo": {
                            "tiny": "https://example.com/funi_tiny.jpg",
                            "small": "https://example.com/funi_small.jpg"
                        }
                    }
                }
            ]
        }
        
        return client

    @pytest.fixture
    def mock_kitsu_mapper(self):
        """Create mock Kitsu mapper."""
        mapper = MagicMock()
        
        # Mock to_kitsu_search_params
        mapper.to_kitsu_search_params.return_value = {
            "filter[text]": "attack on titan",
            "page[limit]": 20,
            "page[offset]": 0,
            "filter[subtype]": "TV",
            "filter[status]": "finished"
        }
        
        # Mock to_universal_anime
        mapper.to_universal_anime.return_value = UniversalAnime(
            id="kitsu_1",
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
            data_quality_score=0.88
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
    async def test_search_anime_kitsu_success(self, mock_kitsu_client, mock_kitsu_mapper, mock_context):
        """Test successful Kitsu anime search."""
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', mock_kitsu_client), \
             patch('src.anime_mcp.tools.kitsu_tools.kitsu_mapper', mock_kitsu_mapper):
            
            result = await search_anime_kitsu(
                query="attack on titan",
                limit=20,
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == 1
            
            anime = result[0]
            assert anime["id"] == "kitsu_1"
            assert anime["title"] == "Attack on Titan"
            assert anime["kitsu_id"] == "1"
            assert anime["kitsu_slug"] == "attack-on-titan"
            assert anime["source_platform"] == "kitsu"
            
            # Verify streaming data is included
            assert "streaming_links" in anime
            assert len(anime["streaming_links"]) > 0
            
            # Verify client calls
            mock_kitsu_client.search_anime.assert_called_once()
            mock_kitsu_mapper.to_kitsu_search_params.assert_called_once()
            mock_kitsu_mapper.to_universal_anime.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_anime_kitsu_with_filters(self, mock_kitsu_client, mock_kitsu_mapper, mock_context):
        """Test Kitsu search with comprehensive filters."""
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', mock_kitsu_client), \
             patch('src.anime_mcp.tools.kitsu_tools.kitsu_mapper', mock_kitsu_mapper):
            
            result = await search_anime_kitsu(
                query="mecha anime",
                subtype="TV",
                status="current",
                age_rating="PG",
                season="spring",
                season_year=2023,
                streaming_platforms=["crunchyroll", "funimation"],
                include_nsfw=False,
                sort_by="popularityRank",
                limit=15,
                offset=30,
                ctx=mock_context
            )
            
            # Verify mapper was called with filters
            call_args = mock_kitsu_mapper.to_kitsu_search_params.call_args
            universal_params = call_args[0][0]
            kitsu_specific = call_args[0][1]
            
            assert universal_params.query == "mecha anime"
            assert universal_params.limit == 15
            assert universal_params.offset == 30
            
            assert kitsu_specific["subtype"] == "TV"
            assert kitsu_specific["status"] == "current"
            assert kitsu_specific["age_rating"] == "PG"
            assert kitsu_specific["season"] == "spring"
            assert kitsu_specific["season_year"] == 2023

    @pytest.mark.asyncio
    async def test_search_anime_kitsu_pagination(self, mock_kitsu_client, mock_kitsu_mapper):
        """Test Kitsu search pagination limits."""
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', mock_kitsu_client), \
             patch('src.anime_mcp.tools.kitsu_tools.kitsu_mapper', mock_kitsu_mapper):
            
            # Test limit clamping (should clamp to 20)
            await search_anime_kitsu(query="test", limit=50)
            
            call_args = mock_kitsu_mapper.to_kitsu_search_params.call_args
            universal_params = call_args[0][0]
            assert universal_params.limit == 20

    @pytest.mark.asyncio
    async def test_search_anime_kitsu_no_client(self, mock_context):
        """Test Kitsu search when client not available."""
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', None):
            
            with pytest.raises(RuntimeError, match="Kitsu client not available"):
                await search_anime_kitsu(query="test", ctx=mock_context)
            
            mock_context.error.assert_called_with("Kitsu client not configured")

    @pytest.mark.asyncio
    async def test_search_anime_kitsu_api_error(self, mock_kitsu_client, mock_kitsu_mapper, mock_context):
        """Test Kitsu search when API returns error."""
        mock_kitsu_client.search_anime.return_value = {
            "errors": [
                {
                    "title": "Bad Request",
                    "detail": "Invalid filter parameter",
                    "status": "400"
                }
            ]
        }
        
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', mock_kitsu_client), \
             patch('src.anime_mcp.tools.kitsu_tools.kitsu_mapper', mock_kitsu_mapper):
            
            with pytest.raises(RuntimeError, match="Kitsu search failed: API errors"):
                await search_anime_kitsu(query="test", ctx=mock_context)

    @pytest.mark.asyncio
    async def test_search_anime_kitsu_client_exception(self, mock_kitsu_client, mock_kitsu_mapper, mock_context):
        """Test Kitsu search when client raises exception."""
        mock_kitsu_client.search_anime.side_effect = Exception("Connection timeout")
        
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', mock_kitsu_client), \
             patch('src.anime_mcp.tools.kitsu_tools.kitsu_mapper', mock_kitsu_mapper):
            
            with pytest.raises(RuntimeError, match="Kitsu search failed: Connection timeout"):
                await search_anime_kitsu(query="test", ctx=mock_context)

    @pytest.mark.asyncio
    async def test_search_anime_kitsu_mapping_error(self, mock_kitsu_client, mock_kitsu_mapper, mock_context):
        """Test Kitsu search when mapping fails for individual results."""
        mock_kitsu_mapper.to_universal_anime.side_effect = Exception("Mapping error")
        
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', mock_kitsu_client), \
             patch('src.anime_mcp.tools.kitsu_tools.kitsu_mapper', mock_kitsu_mapper):
            
            result = await search_anime_kitsu(query="test", ctx=mock_context)
            
            # Should return empty list when all mappings fail
            assert result == []
            mock_context.error.assert_called_with("Failed to process Kitsu result: Mapping error")

    @pytest.mark.asyncio
    async def test_get_anime_kitsu_success(self, mock_kitsu_client, mock_kitsu_mapper, mock_context):
        """Test successful Kitsu anime detail retrieval."""
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', mock_kitsu_client), \
             patch('src.anime_mcp.tools.kitsu_tools.kitsu_mapper', mock_kitsu_mapper):
            
            result = await get_anime_kitsu(
                kitsu_id="1",
                include_characters=True,
                include_staff=True,
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, dict)
            assert result["id"] == "kitsu_1"
            assert result["title"] == "Attack on Titan"
            assert result["kitsu_id"] == "1"
            assert result["source_platform"] == "kitsu"
            
            # Should include comprehensive Kitsu data
            assert "kitsu_slug" in result
            assert "kitsu_rating_rank" in result
            assert "kitsu_popularity_rank" in result
            assert "kitsu_user_count" in result
            assert "kitsu_favorites_count" in result
            assert "streaming_links" in result
            
            # Verify client was called with includes
            mock_kitsu_client.get_anime_by_id.assert_called_once()
            call_args = mock_kitsu_client.get_anime_by_id.call_args
            assert "animeCharacters" in call_args[1]["include"]
            assert "animeStaff" in call_args[1]["include"]

    @pytest.mark.asyncio
    async def test_get_anime_kitsu_minimal_options(self, mock_kitsu_client, mock_kitsu_mapper):
        """Test Kitsu anime detail with minimal options."""
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', mock_kitsu_client), \
             patch('src.anime_mcp.tools.kitsu_tools.kitsu_mapper', mock_kitsu_mapper):
            
            result = await get_anime_kitsu(
                kitsu_id="1",
                include_characters=False,
                include_staff=False
            )
            
            # Should not include optional character/staff data
            call_args = mock_kitsu_client.get_anime_by_id.call_args
            include = call_args[1]["include"]
            assert "animeCharacters" not in include
            assert "animeStaff" not in include

    @pytest.mark.asyncio
    async def test_get_anime_kitsu_not_found(self, mock_kitsu_client, mock_kitsu_mapper, mock_context):
        """Test Kitsu anime detail when anime not found."""
        mock_kitsu_client.get_anime_by_id.return_value = {
            "errors": [
                {
                    "title": "Record not found",
                    "detail": "The record identified by 99999 could not be found.",
                    "status": "404"
                }
            ]
        }
        
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', mock_kitsu_client), \
             patch('src.anime_mcp.tools.kitsu_tools.kitsu_mapper', mock_kitsu_mapper):
            
            result = await get_anime_kitsu(kitsu_id="99999", ctx=mock_context)
            
            assert result is None
            mock_context.info.assert_called_with("Anime with Kitsu ID 99999 not found")

    @pytest.mark.asyncio
    async def test_get_anime_kitsu_no_client(self, mock_context):
        """Test Kitsu anime detail when client not available."""
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', None):
            
            with pytest.raises(RuntimeError, match="Kitsu client not available"):
                await get_anime_kitsu(kitsu_id="1", ctx=mock_context)

    @pytest.mark.asyncio
    async def test_search_streaming_platforms_success(self, mock_kitsu_client, mock_context):
        """Test successful streaming platforms search."""
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', mock_kitsu_client):
            
            result = await search_streaming_platforms(
                platform_name="crunchy",
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == 2
            
            platform = result[0]
            assert platform["id"] == "1"
            assert platform["name"] == "Crunchyroll"
            assert platform["site_name"] == "Crunchyroll"
            
            # Verify client was called correctly
            mock_kitsu_client.search_streaming_platforms.assert_called_once_with("crunchy")

    @pytest.mark.asyncio
    async def test_search_streaming_platforms_no_client(self, mock_context):
        """Test streaming platforms search when client not available."""
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', None):
            
            with pytest.raises(RuntimeError, match="Kitsu client not available"):
                await search_streaming_platforms(platform_name="test", ctx=mock_context)

    @pytest.mark.asyncio
    async def test_search_streaming_platforms_empty_results(self, mock_kitsu_client, mock_context):
        """Test streaming platforms search with empty results."""
        mock_kitsu_client.search_streaming_platforms.return_value = {"data": []}
        
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', mock_kitsu_client):
            
            result = await search_streaming_platforms(platform_name="nonexistent", ctx=mock_context)
            
            assert result == []
            mock_context.info.assert_called_with("Found 0 streaming platforms")


class TestKitsuToolsAdvanced:
    """Advanced tests for Kitsu tools."""

    @pytest.mark.asyncio
    async def test_kitsu_tools_parameter_validation(self):
        """Test parameter validation for Kitsu tools."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        mock_mapper.to_kitsu_search_params.return_value = {}
        mock_client.search_anime.return_value = {"data": []}
        
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', mock_client), \
             patch('src.anime_mcp.tools.kitsu_tools.kitsu_mapper', mock_mapper):
            
            # Test all valid subtypes
            valid_subtypes = ["TV", "movie", "OVA", "ONA", "special", "music"]
            for subtype in valid_subtypes:
                await search_anime_kitsu(query="test", subtype=subtype)
            
            # Test all valid statuses
            valid_statuses = ["current", "finished", "tba", "unreleased", "upcoming"]
            for status in valid_statuses:
                await search_anime_kitsu(query="test", status=status)
            
            # Test all valid age ratings
            valid_ratings = ["G", "PG", "R", "R18"]
            for rating in valid_ratings:
                await search_anime_kitsu(query="test", age_rating=rating)
            
            # Test all valid seasons
            valid_seasons = ["spring", "summer", "fall", "winter"]
            for season in valid_seasons:
                await search_anime_kitsu(query="test", season=season)
            
            # Test all valid sort options
            valid_sorts = ["popularityRank", "ratingRank", "-popularityRank", "-ratingRank", "startDate", "-startDate"]
            for sort in valid_sorts:
                await search_anime_kitsu(query="test", sort_by=sort)

    @pytest.mark.asyncio
    async def test_kitsu_tools_streaming_data_processing(self, mock_context):
        """Test streaming data processing and aggregation."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        # Mock response with comprehensive streaming data
        streaming_response = {
            "data": [{
                "id": "1",
                "type": "anime",
                "attributes": {"canonicalTitle": "Test Anime"},
                "relationships": {
                    "streamingLinks": {
                        "data": [
                            {"id": "1", "type": "streamingLinks"},
                            {"id": "2", "type": "streamingLinks"}
                        ]
                    }
                }
            }],
            "included": [
                {
                    "id": "1",
                    "type": "streamingLinks",
                    "attributes": {
                        "url": "https://crunchyroll.com/test",
                        "subs": ["en", "es", "fr"],
                        "dubs": ["en", "es"]
                    },
                    "relationships": {
                        "streamer": {"data": {"id": "1", "type": "streamers"}}
                    }
                },
                {
                    "id": "2", 
                    "type": "streamingLinks",
                    "attributes": {
                        "url": "https://funimation.com/test",
                        "subs": ["en"],
                        "dubs": ["en"]
                    },
                    "relationships": {
                        "streamer": {"data": {"id": "2", "type": "streamers"}}
                    }
                },
                {
                    "id": "1",
                    "type": "streamers",
                    "attributes": {"name": "Crunchyroll", "siteName": "Crunchyroll"}
                },
                {
                    "id": "2",
                    "type": "streamers", 
                    "attributes": {"name": "Funimation", "siteName": "Funimation"}
                }
            ]
        }
        
        mock_client.search_anime.return_value = streaming_response
        mock_mapper.to_universal_anime.return_value = UniversalAnime(
            id="kitsu_1", title="Test Anime", data_quality_score=0.85
        )
        
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', mock_client), \
             patch('src.anime_mcp.tools.kitsu_tools.kitsu_mapper', mock_mapper):
            
            result = await search_anime_kitsu(query="test", ctx=mock_context)
            
            anime = result[0]
            # Verify streaming links are properly aggregated
            assert len(anime["streaming_links"]) == 2
            
            # Check Crunchyroll link
            cr_link = next(link for link in anime["streaming_links"] if link["platform"] == "Crunchyroll")
            assert cr_link["url"] == "https://crunchyroll.com/test"
            assert "en" in cr_link["subtitles"]
            assert "en" in cr_link["dubs"]
            
            # Check Funimation link
            funi_link = next(link for link in anime["streaming_links"] if link["platform"] == "Funimation")
            assert funi_link["url"] == "https://funimation.com/test"

    @pytest.mark.asyncio
    async def test_kitsu_tools_data_quality_scoring(self, mock_context):
        """Test data quality scoring for Kitsu results."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        # Mock results with different quality scores
        mock_client.search_anime.return_value = {"data": [{"id": "1", "attributes": {}}]}
        
        mock_mapper.to_universal_anime.side_effect = [
            UniversalAnime(id="kitsu_1", title="High Quality", data_quality_score=0.92),
            UniversalAnime(id="kitsu_2", title="Low Quality", data_quality_score=0.65)
        ]
        
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', mock_client), \
             patch('src.anime_mcp.tools.kitsu_tools.kitsu_mapper', mock_mapper):
            
            # First call
            result = await search_anime_kitsu(query="test1", ctx=mock_context)
            assert result[0]["data_quality_score"] == 0.92
            
            # Second call 
            result = await search_anime_kitsu(query="test2", ctx=mock_context)
            assert result[0]["data_quality_score"] == 0.65

    @pytest.mark.asyncio
    async def test_kitsu_tools_comprehensive_include_handling(self, mock_context):
        """Test comprehensive include parameter handling."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        # Mock comprehensive response
        mock_client.get_anime_by_id.return_value = {
            "data": {
                "id": "1",
                "type": "anime",
                "attributes": {"canonicalTitle": "Test Anime"}
            }
        }
        
        mock_mapper.to_universal_anime.return_value = UniversalAnime(
            id="kitsu_1", title="Test Anime", data_quality_score=0.9
        )
        
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', mock_client), \
             patch('src.anime_mcp.tools.kitsu_tools.kitsu_mapper', mock_mapper):
            
            # Test with all includes
            await get_anime_kitsu(
                kitsu_id="1", 
                include_characters=True,
                include_staff=True,
                ctx=mock_context
            )
            
            call_args = mock_client.get_anime_by_id.call_args
            include = call_args[1]["include"]
            
            # Verify all expected includes are present
            expected_includes = [
                "genres", "categories", "productions", "productions.company",
                "streamingLinks", "streamingLinks.streamer",
                "animeCharacters", "animeStaff"
            ]
            
            for expected in expected_includes:
                assert expected in include

    @pytest.mark.asyncio
    async def test_kitsu_tools_error_response_parsing(self, mock_context):
        """Test parsing of different Kitsu error response formats."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        error_responses = [
            # Standard error format
            {
                "errors": [
                    {
                        "title": "Bad Request",
                        "detail": "Invalid parameter value",
                        "status": "400"
                    }
                ]
            },
            # Multiple errors
            {
                "errors": [
                    {"title": "Validation Error", "detail": "Field required"},
                    {"title": "Type Error", "detail": "Invalid type"}
                ]
            },
            # Error without details
            {
                "errors": [{"title": "Unknown Error"}]
            }
        ]
        
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', mock_client), \
             patch('src.anime_mcp.tools.kitsu_tools.kitsu_mapper', mock_mapper):
            
            for error_response in error_responses:
                mock_client.search_anime.return_value = error_response
                
                with pytest.raises(RuntimeError, match="Kitsu search failed: API errors"):
                    await search_anime_kitsu(query="test", ctx=mock_context)


class TestKitsuToolsIntegration:
    """Integration tests for Kitsu tools."""

    @pytest.mark.asyncio
    async def test_kitsu_tools_annotation_verification(self):
        """Verify that Kitsu tools have correct MCP annotations."""
        from src.anime_mcp.tools.kitsu_tools import mcp
        
        # Get registered tools
        tools = mcp._tools
        
        # Verify search_anime_kitsu annotations
        search_tool = tools.get("search_anime_kitsu")
        assert search_tool is not None
        assert search_tool.annotations["title"] == "Kitsu Anime Search"
        assert search_tool.annotations["readOnlyHint"] is True
        assert search_tool.annotations["idempotentHint"] is True
        
        # Verify get_anime_kitsu annotations
        detail_tool = tools.get("get_anime_kitsu")
        assert detail_tool is not None
        assert detail_tool.annotations["title"] == "Kitsu Anime Details"
        
        # Verify search_streaming_platforms annotations
        streaming_tool = tools.get("search_streaming_platforms")
        assert streaming_tool is not None
        assert streaming_tool.annotations["title"] == "Search Streaming Platforms"

    @pytest.mark.asyncio
    async def test_kitsu_search_to_streaming_workflow(self, mock_context):
        """Test workflow from anime search to streaming platform discovery."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        # Setup search result with streaming data
        mock_client.search_anime.return_value = {
            "data": [{
                "id": "1", 
                "type": "anime",
                "attributes": {"canonicalTitle": "Test Anime"},
                "relationships": {
                    "streamingLinks": {
                        "data": [{"id": "1", "type": "streamingLinks"}]
                    }
                }
            }],
            "included": [
                {
                    "id": "1",
                    "type": "streamingLinks",
                    "relationships": {
                        "streamer": {"data": {"id": "1", "type": "streamers"}}
                    }
                },
                {
                    "id": "1",
                    "type": "streamers",
                    "attributes": {"name": "Crunchyroll"}
                }
            ]
        }
        
        # Setup platform search result
        mock_client.search_streaming_platforms.return_value = {
            "data": [{
                "id": "1",
                "type": "streamers", 
                "attributes": {"name": "Crunchyroll", "siteName": "Crunchyroll"}
            }]
        }
        
        mock_mapper.to_universal_anime.return_value = UniversalAnime(
            id="kitsu_1", title="Test Anime", data_quality_score=0.9
        )
        
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', mock_client), \
             patch('src.anime_mcp.tools.kitsu_tools.kitsu_mapper', mock_mapper):
            
            # Step 1: Search anime
            anime_results = await search_anime_kitsu(query="test", ctx=mock_context)
            assert len(anime_results) == 1
            assert len(anime_results[0]["streaming_links"]) > 0
            
            # Step 2: Search for specific platform
            platform_results = await search_streaming_platforms(
                platform_name="crunchyroll", 
                ctx=mock_context
            )
            assert len(platform_results) == 1
            assert platform_results[0]["name"] == "Crunchyroll"

    @pytest.mark.asyncio
    async def test_kitsu_tools_comprehensive_error_scenarios(self, mock_context):
        """Test comprehensive error scenarios for Kitsu tools."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        error_scenarios = [
            ("Rate limit", "Too Many Requests"),
            ("Auth error", "Unauthorized access"),
            ("Server error", "Internal Server Error"),
            ("Timeout", "Request timeout")
        ]
        
        with patch('src.anime_mcp.tools.kitsu_tools.kitsu_client', mock_client), \
             patch('src.anime_mcp.tools.kitsu_tools.kitsu_mapper', mock_mapper):
            
            for error_type, error_msg in error_scenarios:
                mock_client.search_anime.side_effect = Exception(error_msg)
                
                with pytest.raises(RuntimeError, match=f"Kitsu search failed: {error_msg}"):
                    await search_anime_kitsu(query="test", ctx=mock_context)
                
                mock_context.error.assert_called_with(f"Kitsu search failed: {error_msg}")
                mock_context.reset_mock()