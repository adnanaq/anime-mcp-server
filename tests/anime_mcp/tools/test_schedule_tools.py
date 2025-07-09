"""Unit tests for AnimeSchedule-specific MCP tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.anime_mcp.tools.schedule_tools import (
    search_anime_schedule,
    get_schedule_data,
    get_currently_airing,
    animeschedule_client
)
from src.models.structured_responses import BasicAnimeResult, AnimeType, AnimeStatus


class TestScheduleTools:
    """Test suite for AnimeSchedule MCP tools."""

    @pytest.fixture
    def mock_schedule_client(self):
        """Create mock AnimeSchedule client."""
        client = AsyncMock()
        
        client.search_anime.return_value = [
            {
                "title": "Attack on Titan",
                "route": "attack-on-titan",
                "imageVersionRoute": "attack-on-titan",
                "episodeCount": 25,
                "episodeLength": 24,
                "season": "Spring",
                "year": 2013,
                "broadcastInterval": "Weekly",
                "broadcastDay": "Sunday",
                "broadcastTime": "17:00",
                "broadcastTimezone": "Asia/Tokyo", 
                "sources": [
                    "https://myanimelist.net/anime/16498",
                    "https://anilist.co/anime/16498"
                ],
                "synopsis": "Humanity fights against titans",
                "genres": ["Action", "Drama", "Military"],
                "studios": ["Wit Studio"],
                "streamingSites": [
                    {
                        "site": "Crunchyroll",
                        "url": "https://crunchyroll.com/attack-on-titan",
                        "regions": ["US", "CA", "UK"]
                    },
                    {
                        "site": "Funimation", 
                        "url": "https://funimation.com/shows/attack-on-titan",
                        "regions": ["US", "CA"]
                    }
                ],
                "status": "finished",
                "nextEpisodeDate": None,
                "hasEnded": True
            }
        ]
        
        client.get_schedule_data.return_value = {
            "currentTime": "2024-01-15T12:00:00Z",
            "currentWeek": [
                {
                    "title": "Current Anime 1",
                    "route": "current-anime-1",
                    "episodeNumber": 15,
                    "episodeLength": 24,
                    "broadcastDay": "Monday",
                    "broadcastTime": "09:00",
                    "broadcastTimezone": "Asia/Tokyo",
                    "nextEpisodeDate": "2024-01-22T00:00:00Z",
                    "sources": ["https://anilist.co/anime/12345"],
                    "streamingSites": [
                        {
                            "site": "Crunchyroll",
                            "url": "https://crunchyroll.com/current-anime-1",
                            "regions": ["US", "CA"]
                        }
                    ],
                    "status": "airing"
                }
            ],
            "nextWeek": [
                {
                    "title": "Upcoming Anime 1",
                    "route": "upcoming-anime-1", 
                    "episodeNumber": 1,
                    "episodeLength": 24,
                    "broadcastDay": "Saturday",
                    "broadcastTime": "23:00",
                    "broadcastTimezone": "Asia/Tokyo",
                    "nextEpisodeDate": "2024-01-27T14:00:00Z",
                    "sources": ["https://myanimelist.net/anime/54321"],
                    "streamingSites": [
                        {
                            "site": "Netflix",
                            "url": "https://netflix.com/upcoming-anime-1",
                            "regions": ["US", "JP"]
                        }
                    ],
                    "status": "upcoming"
                }
            ],
            "timezone": "UTC"
        }
        
        client.get_currently_airing.return_value = [
            {
                "title": "Airing Now 1",
                "route": "airing-now-1",
                "episodeNumber": 8,
                "episodeLength": 24,
                "broadcastDay": "Wednesday",
                "broadcastTime": "12:30",
                "broadcastTimezone": "Asia/Tokyo",
                "nextEpisodeDate": "2024-01-24T03:30:00Z",
                "streamingSites": [
                    {
                        "site": "Crunchyroll",
                        "url": "https://crunchyroll.com/airing-now-1",
                        "regions": ["US", "CA", "UK", "AU"]
                    }
                ],
                "status": "airing",
                "timeUntilNextEpisode": "6 days, 15 hours"
            },
            {
                "title": "Airing Now 2",
                "route": "airing-now-2",
                "episodeNumber": 12,
                "episodeLength": 24,
                "broadcastDay": "Friday", 
                "broadcastTime": "16:00",
                "broadcastTimezone": "Asia/Tokyo",
                "nextEpisodeDate": "2024-01-19T07:00:00Z",
                "streamingSites": [
                    {
                        "site": "Funimation",
                        "url": "https://funimation.com/airing-now-2",
                        "regions": ["US"]
                    }
                ],
                "status": "airing",
                "timeUntilNextEpisode": "2 days, 19 hours"
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
    async def test_search_anime_schedule_success(self, mock_schedule_client, mock_context):
        """Test successful AnimeSchedule anime search."""
        with patch('src.anime_mcp.tools.schedule_tools.animeschedule_client', mock_schedule_client):
            
            result = await search_anime_schedule(
                query="attack on titan",
                limit=20,
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == 1
            
            anime = result[0]
            assert anime["id"] == "attack-on-titan"
            assert anime["title"] == "Attack on Titan"
            assert anime["animeschedule_id"] == "attack-on-titan"
            assert anime["source_platform"] == "animeschedule"
            
            # Verify schedule-specific data
            assert "broadcast_day" in anime
            assert "broadcast_time" in anime
            assert "broadcast_timezone" in anime
            assert "streaming_platforms" in anime
            assert len(anime["streaming_platforms"]) > 0
            
            # Verify client calls
            mock_schedule_client.search_anime.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_anime_schedule_with_filters(self, mock_schedule_client, mock_context):
        """Test AnimeSchedule search with filters."""
        with patch('src.anime_mcp.tools.schedule_tools.animeschedule_client', mock_schedule_client):
            
            await search_anime_schedule(
                query="mecha anime",
                airing_statuses=["ongoing"],
                seasons=["winter"],
                years=[2024],
                ctx=mock_context
            )
            
            # Verify client was called with filters
            mock_schedule_client.search_anime.assert_called_once()
            call_args = mock_schedule_client.search_anime.call_args[0][0]
            
            assert call_args["query"] == "mecha anime"
            assert call_args["st"] == ["ongoing"]
            assert call_args["seasons"] == ["winter"]
            assert call_args["years"] == [2024]

    @pytest.mark.asyncio
    async def test_search_anime_schedule_no_client(self, mock_context):
        """Test AnimeSchedule search when client not available."""
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', None):
            
            with pytest.raises(RuntimeError, match="AnimeSchedule client not available"):
                await search_anime_schedule(query="test", ctx=mock_context)
            
            mock_context.error.assert_called_with("AnimeSchedule client not configured")

    @pytest.mark.asyncio
    async def test_search_anime_schedule_client_error(self, mock_schedule_client, mock_context):
        """Test AnimeSchedule search when client raises error."""
        mock_schedule_client.search_anime.side_effect = Exception("API unavailable")
        
        with patch('src.anime_mcp.tools.schedule_tools.animeschedule_client', mock_schedule_client):
            
            with pytest.raises(RuntimeError, match="AnimeSchedule search failed: API unavailable"):
                await search_anime_schedule(query="test", ctx=mock_context)

    @pytest.mark.asyncio
    async def test_search_anime_schedule_mapping_error(self, mock_schedule_client, mock_context):
        """Test AnimeSchedule search when mapping fails for individual results."""
        # Simulate a malformed raw result that causes an error during processing
        mock_schedule_client.search_anime.return_value = [
            {"id": "1", "malformed_key": "This will cause an error"}
        ]
        
        with patch('src.anime_mcp.tools.schedule_tools.animeschedule_client', mock_schedule_client):
            
            result = await search_anime_schedule(query="test", ctx=mock_context)
            
            # Should return empty list when all mappings fail
            assert result == []
            mock_context.error.assert_called_with(f"Failed to process AnimeSchedule result: 'title'")

    @pytest.mark.asyncio
    async def test_get_schedule_data_success(self, mock_schedule_client, mock_context):
        """Test successful schedule data retrieval."""
        with patch('src.anime_mcp.tools.schedule_tools.animeschedule_client', mock_schedule_client):
            
            result = await get_schedule_data(
                mal_id=12345, # Use an ID for lookup
                timezone="America/New_York",
                include_episode_history=True,
                include_streaming_details=True,
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, dict)
            assert "broadcast_schedule" in result
            assert "episode_info" in result
            assert "airing_period" in result
            assert "streaming_platforms" in result
            assert "production" in result
            assert "external_ids" in result
            assert "episode_history" in result
            assert "rankings" in result
            
            # Verify current week data
            assert result["title"] == "Current Anime 1"
            assert result["broadcast_schedule"]["day"] == "Monday"
            assert result["broadcast_schedule"]["time"] == "09:00"
            
            # Verify client was called correctly
            mock_schedule_client.get_anime_schedule_by_id.assert_called_once_with({
                "mal_id": 12345,
                "include_episodes": True,
                "include_streaming": True
            })

    @pytest.mark.asyncio
    async def test_get_schedule_data_current_week_only(self, mock_schedule_client, mock_context):
        """Test schedule data retrieval for current week only."""
        with patch('src.anime_mcp.tools.schedule_tools.animeschedule_client', mock_schedule_client):
            
            result = await get_schedule_data(
                mal_id=12345,
                include_episode_history=False,
                include_streaming_details=False,
                ctx=mock_context
            )
            
            # Should only include current week
            assert "episode_history" not in result
            assert "streaming_platforms" not in result
            
            mock_schedule_client.get_anime_schedule_by_id.assert_called_once_with({
                "mal_id": 12345,
                "include_episodes": False,
                "include_streaming": False
            })

    @pytest.mark.asyncio
    async def test_get_schedule_data_no_client(self, mock_context):
        """Test schedule data retrieval when client not available."""
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', None):
            
            with pytest.raises(RuntimeError, match="AnimeSchedule client not available"):
                await get_schedule_data(ctx=mock_context)

    @pytest.mark.asyncio
    async def test_get_schedule_data_client_error(self, mock_schedule_client, mock_context):
        """Test schedule data retrieval when client raises error."""
        mock_schedule_client.get_anime_schedule_by_id.side_effect = Exception("Schedule API error")
        
        with patch('src.anime_mcp.tools.schedule_tools.animeschedule_client', mock_schedule_client):
            
            with pytest.raises(RuntimeError, match="Failed to get schedule data: Schedule API error"):
                await get_schedule_data(mal_id=123, ctx=mock_context)

    @pytest.mark.asyncio
    async def test_get_currently_airing_success(self, mock_schedule_client, mock_context):
        """Test successful currently airing anime retrieval."""
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', mock_schedule_client):
            
            result = await get_currently_airing(
                timezone="Asia/Tokyo",
                limit=10,
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == 2
            
            # Verify first airing anime
            anime1 = result[0]
            assert anime1["title"] == "Airing Now 1"
            assert anime1["episode_number"] == 8
            assert anime1["status"] == "airing"
            assert anime1["time_until_next_episode"] == "6 days, 15 hours"
            assert len(anime1["streaming_sites"]) == 1
            
            # Verify second airing anime
            anime2 = result[1]
            assert anime2["title"] == "Airing Now 2"
            assert anime2["episode_number"] == 12
            assert anime2["time_until_next_episode"] == "2 days, 19 hours"
            
            # Verify client was called correctly
            mock_schedule_client.get_currently_airing.assert_called_once_with(
                timezone="Asia/Tokyo",
                limit=10
            )

    @pytest.mark.asyncio
    async def test_get_currently_airing_no_client(self, mock_context):
        """Test currently airing retrieval when client not available."""
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', None):
            
            with pytest.raises(RuntimeError, match="AnimeSchedule client not available"):
                await get_currently_airing(ctx=mock_context)

    @pytest.mark.asyncio
    async def test_get_currently_airing_empty_results(self, mock_schedule_client, mock_context):
        """Test currently airing retrieval with empty results."""
        mock_schedule_client.get_currently_airing.return_value = []
        
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', mock_schedule_client):
            
            result = await get_currently_airing(ctx=mock_context)
            
            assert result == []
            mock_context.info.assert_called_with("Found 0 currently airing anime")

    @pytest.mark.asyncio
    async def test_get_currently_airing_client_error(self, mock_schedule_client, mock_context):
        """Test currently airing retrieval when client raises error."""
        mock_schedule_client.get_currently_airing.side_effect = Exception("Airing data error")
        
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', mock_schedule_client):
            
            with pytest.raises(RuntimeError, match="Failed to get currently airing anime: Airing data error"):
                await get_currently_airing(ctx=mock_context)


class TestScheduleToolsAdvanced:
    """Advanced tests for AnimeSchedule tools."""

    @pytest.mark.asyncio
    async def test_schedule_tools_parameter_validation(self):
        """Test parameter validation for AnimeSchedule tools."""
        mock_client = AsyncMock()
        mock_client.search_anime.return_value = []
        
        with patch('src.anime_mcp.tools.schedule_tools.animeschedule_client', mock_client):
            
            # Test all valid statuses
            valid_statuses = ["finished", "ongoing", "upcoming"]
            for status in valid_statuses:
                await search_anime_schedule(query="test", airing_statuses=[status])
            
            # Test all valid seasons
            valid_seasons = ["Winter", "Spring", "Summer", "Fall"]
            for season in valid_seasons:
                await search_anime_schedule(query="test", seasons=[season])

    @pytest.mark.asyncio
    async def test_schedule_tools_timezone_handling(self, mock_context):
        """Test timezone handling in schedule tools."""
        mock_client = AsyncMock()
        mock_client.get_anime_schedule_by_id.return_value = {
            "id": 1,
            "title": "Test Anime",
            "media_type": "TV",
            "episodes": 12,
            "score": 7.5,
            "year": 2023,
            "airing_status": "finished",
            "genres": ["Action"],
            "studios": ["Studio A"],
            "synopsis": "Synopsis",
            "image_url": "url",
            "premiere_date": "2023-01-01",
            "finale_date": "2023-03-25",
            "broadcast_day": "Monday",
            "broadcast_time": "12:00",
            "broadcast_timezone": "Asia/Tokyo",
            "next_episode_date": "2023-01-08",
            "next_episode_number": 2,
            "episodes_aired": 1,
            "episode_duration": 24,
            "streams": [],
            "stream_regions": {},
            "stream_urls": {},
            "subscription_platforms": [],
            "free_platforms": [],
            "source": "original",
            "production_status": "finished",
            "mal_id": 123,
            "anilist_id": 456,
            "anidb_id": 789,
            "kitsu_id": 1011,
            "popularity_rank": 100,
            "score_rank": 50,
            "licensors": [],
            "producers": [],
            "updated_at": "2023-03-25T12:00:00Z"
        }
        mock_client.get_currently_airing.return_value = [
            {
                "id": 1,
                "title": "Test Airing Anime",
                "media_type": "TV",
                "episodes": 12,
                "episodes_aired": 1,
                "image_url": "url",
                "broadcast_day": "Monday",
                "broadcast_time": "12:00",
                "broadcast_timezone": "Asia/Tokyo",
                "next_episode_date": "2023-01-08",
                "next_episode_number": 2,
                "hours_until_next_episode": 168,
                "streams": [],
                "stream_regions": {},
                "premium_required": False,
                "score": 7.5,
                "popularity_rank": 100,
                "genres": ["Action"],
                "studios": ["Studio A"],
                "mal_id": 123,
                "anilist_id": 456,
                "updated_at": "2023-03-25T12:00:00Z"
            }
        ]
        
        with patch('src.anime_mcp.tools.schedule_tools.animeschedule_client', mock_client):
            
            # Test various timezones
            timezones = [
                "UTC",
                "America/New_York", 
                "America/Los_Angeles",
                "Europe/London",
                "Asia/Tokyo",
                "Australia/Sydney"
            ]
            
            for timezone in timezones:
                # Test schedule data
                await get_schedule_data(mal_id=123, ctx=mock_context)
                call_args = mock_client.get_anime_schedule_by_id.call_args
                assert call_args[0][0]["mal_id"] == 123
                
                # Test currently airing
                await get_currently_airing(timezone=timezone, ctx=mock_context)
                call_args = mock_client.get_currently_airing.call_args
                assert call_args[0][0]["timezone"] == timezone

    @pytest.mark.asyncio
    async def test_schedule_tools_streaming_data_aggregation(self, mock_context):
        """Test streaming data aggregation and processing."""
        mock_client = AsyncMock()
        
        # Mock response with comprehensive streaming data
        streaming_response = [
            {
                "title": "Multi-Platform Anime",
                "route": "multi-platform-anime",
                "media_type": "TV",
                "episodes": 12,
                "score": 8.0,
                "year": 2023,
                "airing_status": "airing",
                "genres": ["Action"],
                "studios": ["Studio A"],
                "synopsis": "Synopsis",
                "image_url": "url",
                "premiere_date": "2023-01-01",
                "finale_date": "2023-03-25",
                "broadcast_day": "Friday",
                "broadcast_time": "15:30",
                "broadcast_timezone": "Asia/Tokyo",
                "next_episode_date": "2023-01-08",
                "next_episode_number": 2,
                "episodes_aired": 1,
                "episode_duration": 24,
                "streams": [
                    "Crunchyroll",
                    "Funimation",
                    "Netflix",
                    "Hulu"
                ],
                "stream_regions": {
                    "Crunchyroll": ["US", "CA", "UK", "AU", "BR"],
                    "Funimation": ["US", "CA"],
                    "Netflix": ["US", "JP", "DE", "FR"],
                    "Hulu": ["US"]
                },
                "stream_urls": {
                    "Crunchyroll": "https://crunchyroll.com/multi-platform",
                    "Funimation": "https://funimation.com/multi-platform", 
                    "Netflix": "https://netflix.com/multi-platform",
                    "Hulu": "https://hulu.com/multi-platform"
                },
                "subscription_platforms": [],
                "free_platforms": [],
                "source": "original",
                "production_status": "finished",
                "mal_id": 123,
                "anilist_id": 456,
                "anidb_id": 789,
                "kitsu_id": 1011,
                "popularity_rank": 100,
                "score_rank": 50,
                "licensors": [],
                "producers": [],
                "updated_at": "2023-03-25T12:00:00Z"
            }
        ]
        
        mock_client.search_anime.return_value = streaming_response
        
        with patch('src.anime_mcp.tools.schedule_tools.animeschedule_client', mock_client):
            
            result = await search_anime_schedule(query="test", ctx=mock_context)
            
            anime = result[0]
            streaming_platforms = anime["streaming_platforms"]["available_on"]
            
            # Verify all platforms are included
            assert len(streaming_platforms) == 4
            platform_names = [site for site in streaming_platforms]
            assert "Crunchyroll" in platform_names
            assert "Funimation" in platform_names
            assert "Netflix" in platform_names
            assert "Hulu" in platform_names
            
            # Verify region data is preserved
            cr_regions = anime["streaming_platforms"]["regional_availability"]["Crunchyroll"]
            assert "US" in cr_regions
            assert "BR" in cr_regions  # Check international regions

    @pytest.mark.asyncio
    async def test_schedule_tools_broadcast_time_processing(self, mock_context):
        """Test broadcast time and scheduling information processing."""
        mock_client = AsyncMock()
        
        # Mock response with various broadcast times
        broadcast_response = [
            {
                "title": "Morning Show",
                "route": "morning-show",
                "media_type": "TV",
                "episodes": 12,
                "score": 7.0,
                "year": 2023,
                "airing_status": "airing",
                "genres": ["Slice of Life"],
                "studios": ["Studio B"],
                "synopsis": "Synopsis",
                "image_url": "url",
                "premiere_date": "2023-01-01",
                "finale_date": "2023-03-25",
                "broadcast_day": "Monday",
                "broadcast_time": "08:00",
                "broadcast_timezone": "Asia/Tokyo",
                "next_episode_date": "2024-01-22T23:00:00Z",
                "next_episode_number": 2,
                "episodes_aired": 1,
                "episode_duration": 24,
                "streams": [],
                "stream_regions": {},
                "stream_urls": {},
                "subscription_platforms": [],
                "free_platforms": [],
                "source": "original",
                "production_status": "finished",
                "mal_id": 123,
                "anilist_id": 456,
                "anidb_id": 789,
                "kitsu_id": 1011,
                "popularity_rank": 100,
                "score_rank": 50,
                "licensors": [],
                "producers": [],
                "updated_at": "2023-03-25T12:00:00Z"
            },
            {
                "title": "Late Night Show", 
                "route": "late-night-show",
                "media_type": "TV",
                "episodes": 12,
                "score": 7.0,
                "year": 2023,
                "airing_status": "airing",
                "genres": ["Slice of Life"],
                "studios": ["Studio B"],
                "synopsis": "Synopsis",
                "image_url": "url",
                "premiere_date": "2023-01-01",
                "finale_date": "2023-03-25",
                "broadcast_day": "Saturday",
                "broadcast_time": "23:30",
                "broadcast_timezone": "Asia/Tokyo",
                "next_episode_date": "2024-01-27T14:30:00Z",
                "next_episode_number": 2,
                "episodes_aired": 1,
                "episode_duration": 24,
                "streams": [],
                "stream_regions": {},
                "stream_urls": {},
                "subscription_platforms": [],
                "free_platforms": [],
                "source": "original",
                "production_status": "finished",
                "mal_id": 123,
                "anilist_id": 456,
                "anidb_id": 789,
                "kitsu_id": 1011,
                "popularity_rank": 100,
                "score_rank": 50,
                "licensors": [],
                "producers": [],
                "updated_at": "2023-03-25T12:00:00Z"
            }
        ]
        
        mock_client.search_anime.return_value = broadcast_response
        
        with patch('src.anime_mcp.tools.schedule_tools.animeschedule_client', mock_client):
            
            result = await search_anime_schedule(query="test", ctx=mock_context)
            
            # Verify broadcast time processing
            morning = result[0]
            assert morning["broadcast_time"] == "08:00"
            assert morning["broadcast_day"] == "Monday"
            assert morning["broadcast_timezone"] == "Asia/Tokyo"
            assert morning["next_episode_date"] == "2024-01-22T23:00:00Z"
            
            late_night = result[1] 
            assert late_night["broadcast_time"] == "23:30"
            assert late_night["broadcast_day"] == "Saturday"

    @pytest.mark.asyncio
    async def test_schedule_tools_data_quality_scoring(self, mock_context):
        """Test data quality scoring for AnimeSchedule results."""
        mock_client = AsyncMock()
        
        # Mock results with different completeness levels
        mock_client.search_anime.side_effect = [
            # High quality
            {
                "title": "Anime 0",
                "id": 0,
                "media_type": "TV",
                "episodes": 12,
                "score": 8.0,
                "year": 2023,
                "airing_status": "finished",
                "genres": ["Action"],
                "studios": ["Studio A"],
                "synopsis": "Synopsis",
                "image_url": "url",
                "premiere_date": "2023-01-01",
                "finale_date": "2023-03-25",
                "broadcast_day": "Monday",
                "broadcast_time": "12:00",
                "broadcast_timezone": "Asia/Tokyo",
                "next_episode_date": "2023-01-08",
                "next_episode_number": 2,
                "episodes_aired": 1,
                "episode_duration": 24,
                "streams": [],
                "stream_regions": {},
                "stream_urls": {},
                "subscription_platforms": [],
                "free_platforms": [],
                "source": "original",
                "production_status": "finished",
                "mal_id": 123,
                "anilist_id": 456,
                "anidb_id": 789,
                "kitsu_id": 1011,
                "popularity_rank": 100,
                "score_rank": 50,
                "licensors": [],
                "producers": [],
                "updated_at": "2023-03-25T12:00:00Z"
            },
            # Medium quality
            {
                "title": "Anime 1",
                "id": 1,
                "media_type": "TV",
                "episodes": 12,
                "score": 7.0,
                "year": 2023,
                "airing_status": "finished",
                "genres": ["Action"],
                "studios": ["Studio A"],
                "synopsis": "Synopsis",
                "image_url": "url",
                "premiere_date": "2023-01-01",
                "finale_date": "2023-03-25",
                "broadcast_day": "Monday",
                "broadcast_time": "12:00",
                "broadcast_timezone": "Asia/Tokyo",
                "next_episode_date": "2023-01-08",
                "next_episode_number": 2,
                "episodes_aired": 1,
                "episode_duration": 24,
                "streams": [],
                "stream_regions": {},
                "stream_urls": {},
                "subscription_platforms": [],
                "free_platforms": [],
                "source": "original",
                "production_status": "finished",
                "mal_id": 123,
                "anilist_id": 456,
                "anidb_id": 789,
                "kitsu_id": 1011,
                "popularity_rank": 100,
                "score_rank": 50,
                "licensors": [],
                "producers": [],
                "updated_at": "2023-03-25T12:00:00Z"
            },
            # Low quality
            {
                "title": "Anime 2",
                "id": 2,
                "media_type": "TV",
                "episodes": 12,
                "score": 6.0,
                "year": 2023,
                "airing_status": "finished",
                "genres": ["Action"],
                "studios": ["Studio A"],
                "synopsis": "Synopsis",
                "image_url": "url",
                "premiere_date": "2023-01-01",
                "finale_date": "2023-03-25",
                "broadcast_day": "Monday",
                "broadcast_time": "12:00",
                "broadcast_timezone": "Asia/Tokyo",
                "next_episode_date": "2023-01-08",
                "next_episode_number": 2,
                "episodes_aired": 1,
                "episode_duration": 24,
                "streams": [],
                "stream_regions": {},
                "stream_urls": {},
                "subscription_platforms": [],
                "free_platforms": [],
                "source": "original",
                "production_status": "finished",
                "mal_id": 123,
                "anilist_id": 456,
                "anidb_id": 789,
                "kitsu_id": 1011,
                "popularity_rank": 100,
                "score_rank": 50,
                "licensors": [],
                "producers": [],
                "updated_at": "2023-03-25T12:00:00Z"
            }
        ]
        
        with patch('src.anime_mcp.tools.schedule_tools.animeschedule_client', mock_client):
            
            # Test multiple calls to verify different quality scores
            for i, expected_score in enumerate([8.0, 7.0, 6.0]):
                result = await search_anime_schedule(query=f"test{i}", ctx=mock_context)
                assert result[0]["score"] == expected_score

    @pytest.mark.asyncio
    async def test_schedule_tools_empty_results_handling(self, mock_context):
        """Test handling of empty results from AnimeSchedule."""
        mock_client = AsyncMock()
        
        # Empty results for all endpoints
        mock_client.search_anime.return_value = []
        mock_client.get_anime_schedule_by_id.return_value = None
        mock_client.get_currently_airing.return_value = []
        
        with patch('src.anime_mcp.tools.schedule_tools.animeschedule_client', mock_client):
            
            # Test search
            result = await search_anime_schedule(query="nonexistent", ctx=mock_context)
            assert result == []
            mock_context.info.assert_called_with("Found 0 anime with scheduling data")
            
            # Test schedule data
            result = await get_schedule_data(mal_id=123, ctx=mock_context)
            assert result is None
            
            # Test currently airing
            result = await get_currently_airing(ctx=mock_context)
            assert result == []


class TestScheduleToolsIntegration:
    """Integration tests for AnimeSchedule tools."""

    @pytest.mark.asyncio
    async def test_schedule_tools_annotation_verification(self):
        """Verify that AnimeSchedule tools have correct MCP annotations."""
        from src.anime_mcp.tools.schedule_tools import mcp
        
        # Get registered tools
        tools = mcp._tools
        
        # Verify search_anime_schedule annotations
        search_tool = tools.get("search_anime_schedule")
        assert search_tool is not None
        assert search_tool.annotations["title"] == "AnimeSchedule Search"
        assert search_tool.annotations["readOnlyHint"] is True
        assert search_tool.annotations["idempotentHint"] is True
        
        # Verify get_schedule_data annotations
        schedule_tool = tools.get("get_schedule_data") 
        assert schedule_tool is not None
        assert schedule_tool.annotations["title"] == "Weekly Anime Schedule"
        
        # Verify get_currently_airing annotations
        airing_tool = tools.get("get_currently_airing")
        assert airing_tool is not None
        assert airing_tool.annotations["title"] == "Currently Airing Anime"

    @pytest.mark.asyncio
    async def test_schedule_tools_workflow_integration(self, mock_context):
        """Test integration workflow between different schedule tools."""
        mock_client = AsyncMock()
        
        # Setup search result
        mock_client.search_anime.return_value = [
            {
                "title": "Popular Airing Anime",
                "route": "popular-airing-anime",
                "status": "airing",
                "broadcastDay": "Wednesday",
                "media_type": "TV",
                "episodes": 12,
                "score": 8.0,
                "year": 2023,
                "genres": ["Action"],
                "studios": ["Studio A"],
                "synopsis": "Synopsis",
                "image_url": "url",
                "premiere_date": "2023-01-01",
                "finale_date": "2023-03-25",
                "broadcast_time": "12:00",
                "broadcast_timezone": "Asia/Tokyo",
                "next_episode_date": "2023-01-08",
                "next_episode_number": 2,
                "episodes_aired": 1,
                "episode_duration": 24,
                "streams": [],
                "stream_regions": {},
                "stream_urls": {},
                "subscription_platforms": [],
                "free_platforms": [],
                "source": "original",
                "production_status": "finished",
                "mal_id": 123,
                "anilist_id": 456,
                "anidb_id": 789,
                "kitsu_id": 1011,
                "popularity_rank": 100,
                "score_rank": 50,
                "licensors": [],
                "producers": [],
                "updated_at": "2023-03-25T12:00:00Z"
            }
        ]
        
        # Setup schedule data result
        mock_client.get_anime_schedule_by_id.return_value = {
            "title": "Popular Airing Anime",
            "id": 123,
            "media_type": "TV",
            "episodes": 12,
            "score": 8.0,
            "year": 2023,
            "airing_status": "airing",
            "genres": ["Action"],
            "studios": ["Studio A"],
            "synopsis": "Synopsis",
            "image_url": "url",
            "premiere_date": "2023-01-01",
            "finale_date": "2023-03-25",
            "broadcast_day": "Wednesday",
            "broadcast_time": "12:00",
            "broadcast_timezone": "Asia/Tokyo",
            "next_episode_date": "2023-01-08",
            "next_episode_number": 2,
            "episodes_aired": 1,
            "episode_duration": 24,
            "streams": [],
            "stream_regions": {},
            "stream_urls": {},
            "subscription_platforms": [],
            "free_platforms": [],
            "source": "original",
            "production_status": "finished",
            "mal_id": 123,
            "anilist_id": 456,
            "anidb_id": 789,
            "kitsu_id": 1011,
            "popularity_rank": 100,
            "score_rank": 50,
            "licensors": [],
            "producers": [],
            "updated_at": "2023-03-25T12:00:00Z"
        }
        
        # Setup currently airing result
        mock_client.get_currently_airing.return_value = [
            {
                "title": "Popular Airing Anime",
                "id": 123,
                "media_type": "TV",
                "episodes": 12,
                "episodes_aired": 1,
                "image_url": "url",
                "broadcast_day": "Wednesday",
                "broadcast_time": "12:00",
                "broadcast_timezone": "Asia/Tokyo",
                "next_episode_date": "2023-01-08",
                "next_episode_number": 2,
                "hours_until_next_episode": 48,
                "streams": [],
                "stream_regions": {},
                "premium_required": False,
                "score": 8.0,
                "popularity_rank": 100,
                "genres": ["Action"],
                "studios": ["Studio A"],
                "mal_id": 123,
                "anilist_id": 456,
                "updated_at": "2023-03-25T12:00:00Z"
            }
        ]
        
        with patch('src.anime_mcp.tools.schedule_tools.animeschedule_client', mock_client):
            
            # Step 1: Search for anime
            search_results = await search_anime_schedule(
                query="popular anime", 
                airing_statuses=["airing"],
                ctx=mock_context
            )
            assert len(search_results) == 1
            assert search_results[0]["title"] == "Popular Airing Anime"
            
            # Step 2: Get weekly schedule
            schedule_data = await get_schedule_data(mal_id=123, ctx=mock_context)
            assert schedule_data["title"] == "Popular Airing Anime"
            
            # Step 3: Get currently airing details
            airing_data = await get_currently_airing(ctx=mock_context)
            assert len(airing_data) == 1
            assert airing_data[0]["title"] == "Popular Airing Anime"
            assert airing_data[0]["broadcast_info"]["countdown_hours"] == 48

    @pytest.mark.asyncio
    async def test_schedule_tools_comprehensive_error_scenarios(self, mock_context):
        """Test comprehensive error scenarios for AnimeSchedule tools."""
        mock_client = AsyncMock()
        
        error_scenarios = [
            ("Service unavailable", "AnimeSchedule API temporarily unavailable"),
            ("Rate limit", "Too many requests to AnimeSchedule"),
            ("Parse error", "Failed to parse schedule data"),
            ("Network timeout", "Request to AnimeSchedule timed out")
        ]
        
        with patch('src.anime_mcp.tools.schedule_tools.animeschedule_client', mock_client):
            
            for error_type, error_msg in error_scenarios:
                # Test search errors
                mock_client.search_anime.side_effect = Exception(error_msg)
                
                with pytest.raises(RuntimeError, match=f"AnimeSchedule search failed: {error_msg}"):
                    await search_anime_schedule(query="test", ctx=mock_context)
                
                # Test schedule data errors
                mock_client.get_anime_schedule_by_id.side_effect = Exception(error_msg)
                
                with pytest.raises(RuntimeError, match=f"Failed to get schedule data: {error_msg}"):
                    await get_schedule_data(mal_id=123, ctx=mock_context)
                
                # Test currently airing errors
                mock_client.get_currently_airing.side_effect = Exception(error_msg)
                
                with pytest.raises(RuntimeError, match=f"Failed to get currently airing anime: {error_msg}"):
                    await get_currently_airing(ctx=mock_context)
                
                mock_context.reset_mock()