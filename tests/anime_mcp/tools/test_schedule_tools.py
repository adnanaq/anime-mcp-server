"""Unit tests for AnimeSchedule-specific MCP tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.anime_mcp.tools.schedule_tools import (
    search_anime_schedule,
    get_schedule_data,
    get_currently_airing,
    animeschedule_client,
    animeschedule_mapper
)
from src.models.universal_anime import UniversalAnime, UniversalSearchParams


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
    def mock_schedule_mapper(self):
        """Create mock AnimeSchedule mapper."""
        mapper = MagicMock()
        
        # Mock to_animeschedule_search_params
        mapper.to_animeschedule_search_params.return_value = {
            "query": "attack on titan",
            "limit": 20,
            "offset": 0,
            "status": "finished",
            "season": "spring",
            "year": 2013
        }
        
        # Mock to_universal_anime
        mapper.to_universal_anime.return_value = UniversalAnime(
            id="schedule_attack_on_titan",
            title="Attack on Titan",
            type_format="TV",
            episodes=25,
            score=0.0,  # AnimeSchedule doesn't have scores
            year=2013,
            status="FINISHED",
            genres=["Action", "Drama", "Military"],
            studios=["Wit Studio"],
            description="Humanity fights against titans",
            image_url="https://animeschedule.net/images/attack-on-titan.jpg",
            data_quality_score=0.82
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
    async def test_search_anime_schedule_success(self, mock_schedule_client, mock_schedule_mapper, mock_context):
        """Test successful AnimeSchedule anime search."""
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', mock_schedule_client), \
             patch('src.anime_mcp.tools.schedule_tools.schedule_mapper', mock_schedule_mapper):
            
            result = await search_anime_schedule(
                query="attack on titan",
                limit=20,
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == 1
            
            anime = result[0]
            assert anime["id"] == "schedule_attack_on_titan"
            assert anime["title"] == "Attack on Titan"
            assert anime["schedule_route"] == "attack-on-titan"
            assert anime["source_platform"] == "animeschedule"
            
            # Verify schedule-specific data
            assert "broadcast_day" in anime
            assert "broadcast_time" in anime
            assert "broadcast_timezone" in anime
            assert "streaming_sites" in anime
            assert len(anime["streaming_sites"]) > 0
            
            # Verify client calls
            mock_schedule_client.search_anime.assert_called_once()
            mock_schedule_mapper.to_animeschedule_search_params.assert_called_once()
            mock_schedule_mapper.to_universal_anime.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_anime_schedule_with_filters(self, mock_schedule_client, mock_schedule_mapper, mock_context):
        """Test AnimeSchedule search with filters."""
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', mock_schedule_client), \
             patch('src.anime_mcp.tools.schedule_tools.schedule_mapper', mock_schedule_mapper):
            
            result = await search_anime_schedule(
                query="mecha anime",
                status="airing",
                season="winter",
                year=2024,
                broadcast_day="saturday",
                timezone="America/New_York",
                streaming_platforms=["crunchyroll", "netflix"],
                limit=15,
                offset=30,
                ctx=mock_context
            )
            
            # Verify mapper was called with filters
            call_args = mock_schedule_mapper.to_animeschedule_search_params.call_args
            universal_params = call_args[0][0]
            schedule_specific = call_args[0][1]
            
            assert universal_params.query == "mecha anime"
            assert universal_params.limit == 15
            assert universal_params.offset == 30
            
            assert schedule_specific["status"] == "airing"
            assert schedule_specific["season"] == "winter"
            assert schedule_specific["year"] == 2024
            assert schedule_specific["broadcast_day"] == "saturday"
            assert schedule_specific["timezone"] == "America/New_York"

    @pytest.mark.asyncio
    async def test_search_anime_schedule_no_client(self, mock_context):
        """Test AnimeSchedule search when client not available."""
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', None):
            
            with pytest.raises(RuntimeError, match="AnimeSchedule client not available"):
                await search_anime_schedule(query="test", ctx=mock_context)
            
            mock_context.error.assert_called_with("AnimeSchedule client not configured")

    @pytest.mark.asyncio
    async def test_search_anime_schedule_client_error(self, mock_schedule_client, mock_schedule_mapper, mock_context):
        """Test AnimeSchedule search when client raises error."""
        mock_schedule_client.search_anime.side_effect = Exception("API unavailable")
        
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', mock_schedule_client), \
             patch('src.anime_mcp.tools.schedule_tools.schedule_mapper', mock_schedule_mapper):
            
            with pytest.raises(RuntimeError, match="AnimeSchedule search failed: API unavailable"):
                await search_anime_schedule(query="test", ctx=mock_context)

    @pytest.mark.asyncio
    async def test_search_anime_schedule_mapping_error(self, mock_schedule_client, mock_schedule_mapper, mock_context):
        """Test AnimeSchedule search when mapping fails for individual results."""
        mock_schedule_mapper.to_universal_anime.side_effect = Exception("Mapping error")
        
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', mock_schedule_client), \
             patch('src.anime_mcp.tools.schedule_tools.schedule_mapper', mock_schedule_mapper):
            
            result = await search_anime_schedule(query="test", ctx=mock_context)
            
            # Should return empty list when all mappings fail
            assert result == []
            mock_context.error.assert_called_with("Failed to process AnimeSchedule result: Mapping error")

    @pytest.mark.asyncio
    async def test_get_schedule_data_success(self, mock_schedule_client, mock_schedule_mapper, mock_context):
        """Test successful schedule data retrieval."""
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', mock_schedule_client), \
             patch('src.anime_mcp.tools.schedule_tools.schedule_mapper', mock_schedule_mapper):
            
            result = await get_schedule_data(
                timezone="America/New_York",
                include_next_week=True,
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, dict)
            assert "current_time" in result
            assert "current_week" in result
            assert "next_week" in result
            assert "timezone" in result
            
            # Verify current week data
            current_week = result["current_week"]
            assert len(current_week) == 1
            assert current_week[0]["title"] == "Current Anime 1"
            assert current_week[0]["status"] == "airing"
            
            # Verify next week data  
            next_week = result["next_week"]
            assert len(next_week) == 1
            assert next_week[0]["title"] == "Upcoming Anime 1"
            assert next_week[0]["status"] == "upcoming"
            
            # Verify client was called correctly
            mock_schedule_client.get_schedule_data.assert_called_once_with(
                timezone="America/New_York",
                include_next_week=True
            )

    @pytest.mark.asyncio
    async def test_get_schedule_data_current_week_only(self, mock_schedule_client, mock_schedule_mapper, mock_context):
        """Test schedule data retrieval for current week only."""
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', mock_schedule_client), \
             patch('src.anime_mcp.tools.schedule_tools.schedule_mapper', mock_schedule_mapper):
            
            result = await get_schedule_data(
                timezone="UTC",
                include_next_week=False,
                ctx=mock_context
            )
            
            # Should only include current week
            assert "current_week" in result
            assert "next_week" in result  # Still included but might be empty based on include_next_week=False
            
            mock_schedule_client.get_schedule_data.assert_called_once_with(
                timezone="UTC",
                include_next_week=False
            )

    @pytest.mark.asyncio
    async def test_get_schedule_data_no_client(self, mock_context):
        """Test schedule data retrieval when client not available."""
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', None):
            
            with pytest.raises(RuntimeError, match="AnimeSchedule client not available"):
                await get_schedule_data(ctx=mock_context)

    @pytest.mark.asyncio
    async def test_get_schedule_data_client_error(self, mock_schedule_client, mock_context):
        """Test schedule data retrieval when client raises error."""
        mock_schedule_client.get_schedule_data.side_effect = Exception("Schedule API error")
        
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', mock_schedule_client):
            
            with pytest.raises(RuntimeError, match="Failed to get schedule data: Schedule API error"):
                await get_schedule_data(ctx=mock_context)

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
        mock_mapper = MagicMock()
        mock_mapper.to_animeschedule_search_params.return_value = {}
        mock_client.search_anime.return_value = []
        
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', mock_client), \
             patch('src.anime_mcp.tools.schedule_tools.schedule_mapper', mock_mapper):
            
            # Test all valid statuses
            valid_statuses = ["airing", "finished", "upcoming", "cancelled", "hiatus"]
            for status in valid_statuses:
                await search_anime_schedule(query="test", status=status)
            
            # Test all valid seasons
            valid_seasons = ["spring", "summer", "fall", "winter"]
            for season in valid_seasons:
                await search_anime_schedule(query="test", season=season)
            
            # Test all valid broadcast days
            valid_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            for day in valid_days:
                await search_anime_schedule(query="test", broadcast_day=day)

    @pytest.mark.asyncio
    async def test_schedule_tools_timezone_handling(self, mock_context):
        """Test timezone handling in schedule tools."""
        mock_client = AsyncMock()
        mock_client.get_schedule_data.return_value = {
            "currentTime": "2024-01-15T12:00:00Z",
            "timezone": "America/New_York",
            "currentWeek": [],
            "nextWeek": []
        }
        mock_client.get_currently_airing.return_value = []
        
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', mock_client):
            
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
                await get_schedule_data(timezone=timezone, ctx=mock_context)
                call_args = mock_client.get_schedule_data.call_args
                assert call_args[1]["timezone"] == timezone
                
                # Test currently airing
                await get_currently_airing(timezone=timezone, ctx=mock_context)
                call_args = mock_client.get_currently_airing.call_args
                assert call_args[1]["timezone"] == timezone

    @pytest.mark.asyncio
    async def test_schedule_tools_streaming_data_aggregation(self, mock_context):
        """Test streaming data aggregation and processing."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        # Mock response with comprehensive streaming data
        streaming_response = [
            {
                "title": "Multi-Platform Anime",
                "route": "multi-platform-anime",
                "streamingSites": [
                    {
                        "site": "Crunchyroll",
                        "url": "https://crunchyroll.com/multi-platform",
                        "regions": ["US", "CA", "UK", "AU", "BR"]
                    },
                    {
                        "site": "Funimation",
                        "url": "https://funimation.com/multi-platform", 
                        "regions": ["US", "CA"]
                    },
                    {
                        "site": "Netflix",
                        "url": "https://netflix.com/multi-platform",
                        "regions": ["US", "JP", "DE", "FR"]
                    },
                    {
                        "site": "Hulu",
                        "url": "https://hulu.com/multi-platform",
                        "regions": ["US"]
                    }
                ],
                "broadcastDay": "Friday",
                "broadcastTime": "15:30",
                "status": "airing"
            }
        ]
        
        mock_client.search_anime.return_value = streaming_response
        mock_mapper.to_universal_anime.return_value = UniversalAnime(
            id="schedule_multi", title="Multi-Platform Anime", data_quality_score=0.9
        )
        
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', mock_client), \
             patch('src.anime_mcp.tools.schedule_tools.schedule_mapper', mock_mapper):
            
            result = await search_anime_schedule(query="test", ctx=mock_context)
            
            anime = result[0]
            streaming_sites = anime["streaming_sites"]
            
            # Verify all platforms are included
            assert len(streaming_sites) == 4
            platform_names = [site["site"] for site in streaming_sites]
            assert "Crunchyroll" in platform_names
            assert "Funimation" in platform_names
            assert "Netflix" in platform_names
            assert "Hulu" in platform_names
            
            # Verify region data is preserved
            cr_site = next(site for site in streaming_sites if site["site"] == "Crunchyroll")
            assert "US" in cr_site["regions"]
            assert "BR" in cr_site["regions"]  # Check international regions

    @pytest.mark.asyncio
    async def test_schedule_tools_broadcast_time_processing(self, mock_context):
        """Test broadcast time and scheduling information processing."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        # Mock response with various broadcast times
        broadcast_response = [
            {
                "title": "Morning Show",
                "route": "morning-show",
                "broadcastDay": "Monday",
                "broadcastTime": "08:00",
                "broadcastTimezone": "Asia/Tokyo",
                "nextEpisodeDate": "2024-01-22T23:00:00Z",
                "status": "airing"
            },
            {
                "title": "Late Night Show", 
                "route": "late-night-show",
                "broadcastDay": "Saturday",
                "broadcastTime": "23:30",
                "broadcastTimezone": "Asia/Tokyo",
                "nextEpisodeDate": "2024-01-27T14:30:00Z",
                "status": "airing"
            }
        ]
        
        mock_client.search_anime.return_value = broadcast_response
        mock_mapper.to_universal_anime.side_effect = [
            UniversalAnime(id="schedule_morning", title="Morning Show", data_quality_score=0.85),
            UniversalAnime(id="schedule_late", title="Late Night Show", data_quality_score=0.87)
        ]
        
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', mock_client), \
             patch('src.anime_mcp.tools.schedule_tools.schedule_mapper', mock_mapper):
            
            result = await search_anime_schedule(query="test", ctx=mock_context)
            
            # Verify broadcast time processing
            morning = result[0]
            assert morning["broadcast_day"] == "Monday"
            assert morning["broadcast_time"] == "08:00"
            assert morning["broadcast_timezone"] == "Asia/Tokyo"
            assert morning["next_episode_date"] == "2024-01-22T23:00:00Z"
            
            late_night = result[1] 
            assert late_night["broadcast_day"] == "Saturday"
            assert late_night["broadcast_time"] == "23:30"

    @pytest.mark.asyncio
    async def test_schedule_tools_data_quality_scoring(self, mock_context):
        """Test data quality scoring for AnimeSchedule results."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        # Mock results with different completeness levels
        mock_client.search_anime.return_value = [{"title": "Test Anime"}]
        
        # Different quality scores based on data completeness
        quality_scores = [0.95, 0.75, 0.60]  # High, medium, low quality
        mock_mapper.to_universal_anime.side_effect = [
            UniversalAnime(id=f"schedule_{i}", title=f"Anime {i}", data_quality_score=score)
            for i, score in enumerate(quality_scores)
        ]
        
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', mock_client), \
             patch('src.anime_mcp.tools.schedule_tools.schedule_mapper', mock_mapper):
            
            # Test multiple calls to verify different quality scores
            for i, expected_score in enumerate(quality_scores):
                result = await search_anime_schedule(query=f"test{i}", ctx=mock_context)
                assert result[0]["data_quality_score"] == expected_score

    @pytest.mark.asyncio
    async def test_schedule_tools_empty_results_handling(self, mock_context):
        """Test handling of empty results from AnimeSchedule."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        # Empty results for all endpoints
        mock_client.search_anime.return_value = []
        mock_client.get_schedule_data.return_value = {
            "currentTime": "2024-01-15T12:00:00Z",
            "currentWeek": [],
            "nextWeek": [],
            "timezone": "UTC"
        }
        mock_client.get_currently_airing.return_value = []
        
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', mock_client), \
             patch('src.anime_mcp.tools.schedule_tools.schedule_mapper', mock_mapper):
            
            # Test search
            result = await search_anime_schedule(query="nonexistent", ctx=mock_context)
            assert result == []
            mock_context.info.assert_called_with("Found 0 anime on AnimeSchedule")
            
            # Test schedule data
            result = await get_schedule_data(ctx=mock_context)
            assert len(result["current_week"]) == 0
            assert len(result["next_week"]) == 0
            
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
        mock_mapper = MagicMock()
        
        # Setup search result
        mock_client.search_anime.return_value = [
            {
                "title": "Popular Airing Anime",
                "route": "popular-airing-anime",
                "status": "airing",
                "broadcastDay": "Wednesday"
            }
        ]
        
        # Setup schedule data result
        mock_client.get_schedule_data.return_value = {
            "currentTime": "2024-01-15T12:00:00Z",
            "currentWeek": [
                {
                    "title": "Popular Airing Anime",
                    "broadcastDay": "Wednesday",
                    "status": "airing"
                }
            ],
            "nextWeek": [],
            "timezone": "UTC"
        }
        
        # Setup currently airing result
        mock_client.get_currently_airing.return_value = [
            {
                "title": "Popular Airing Anime",
                "status": "airing",
                "timeUntilNextEpisode": "2 days"
            }
        ]
        
        mock_mapper.to_universal_anime.return_value = UniversalAnime(
            id="schedule_popular", title="Popular Airing Anime", data_quality_score=0.9
        )
        
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', mock_client), \
             patch('src.anime_mcp.tools.schedule_tools.schedule_mapper', mock_mapper):
            
            # Step 1: Search for anime
            search_results = await search_anime_schedule(
                query="popular anime", 
                status="airing",
                ctx=mock_context
            )
            assert len(search_results) == 1
            assert search_results[0]["title"] == "Popular Airing Anime"
            
            # Step 2: Get weekly schedule
            schedule_data = await get_schedule_data(ctx=mock_context)
            assert len(schedule_data["current_week"]) == 1
            assert schedule_data["current_week"][0]["title"] == "Popular Airing Anime"
            
            # Step 3: Get currently airing details
            airing_data = await get_currently_airing(ctx=mock_context)
            assert len(airing_data) == 1
            assert airing_data[0]["title"] == "Popular Airing Anime"
            assert airing_data[0]["time_until_next_episode"] == "2 days"

    @pytest.mark.asyncio
    async def test_schedule_tools_comprehensive_error_scenarios(self, mock_context):
        """Test comprehensive error scenarios for AnimeSchedule tools."""
        mock_client = AsyncMock()
        mock_mapper = MagicMock()
        
        error_scenarios = [
            ("Service unavailable", "AnimeSchedule API temporarily unavailable"),
            ("Rate limit", "Too many requests to AnimeSchedule"),
            ("Parse error", "Failed to parse schedule data"),
            ("Network timeout", "Request to AnimeSchedule timed out")
        ]
        
        with patch('src.anime_mcp.tools.schedule_tools.schedule_client', mock_client), \
             patch('src.anime_mcp.tools.schedule_tools.schedule_mapper', mock_mapper):
            
            for error_type, error_msg in error_scenarios:
                # Test search errors
                mock_client.search_anime.side_effect = Exception(error_msg)
                
                with pytest.raises(RuntimeError, match=f"AnimeSchedule search failed: {error_msg}"):
                    await search_anime_schedule(query="test", ctx=mock_context)
                
                # Test schedule data errors
                mock_client.get_schedule_data.side_effect = Exception(error_msg)
                
                with pytest.raises(RuntimeError, match=f"Failed to get schedule data: {error_msg}"):
                    await get_schedule_data(ctx=mock_context)
                
                # Test currently airing errors
                mock_client.get_currently_airing.side_effect = Exception(error_msg)
                
                with pytest.raises(RuntimeError, match=f"Failed to get currently airing anime: {error_msg}"):
                    await get_currently_airing(ctx=mock_context)
                
                mock_context.reset_mock()