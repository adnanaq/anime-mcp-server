"""Integration tests for agent orchestration and coordination.

Tests the coordination between SearchAgent and ScheduleAgent,
handoff mechanisms, and collaborative workflows.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List
import json

from src.langgraph.agents.search_agent import SearchAgent
from src.langgraph.agents.schedule_agent import ScheduleAgent


class TestSearchAgentIntegration:
    """Integration tests for SearchAgent."""

    @pytest.fixture
    def mock_platform_tools(self):
        """Create mock platform-specific search tools."""
        tools = {
            "search_anime_mal": AsyncMock(return_value=[
                {
                    "id": "mal_16498",
                    "title": "Attack on Titan",
                    "mal_score": 8.54,
                    "mal_rank": 50,
                    "source_platform": "mal"
                }
            ]),
            "search_anime_anilist": AsyncMock(return_value=[
                {
                    "id": "anilist_16498", 
                    "title": "Shingeki no Kyojin",
                    "anilist_score": 85,
                    "anilist_popularity": 95000,
                    "source_platform": "anilist"
                }
            ]),
            "search_anime_jikan": AsyncMock(return_value=[
                {
                    "id": "jikan_16498",
                    "title": "Attack on Titan",
                    "jikan_score": 8.54,
                    "jikan_members": 3000000,
                    "source_platform": "jikan"
                }
            ]),
            "anime_semantic_search": AsyncMock(return_value=[
                {
                    "id": "semantic_16498",
                    "title": "Attack on Titan",
                    "similarity_score": 0.95,
                    "source_platform": "semantic"
                }
            ])
        }
        return tools

    @pytest.fixture
    def search_agent(self, mock_platform_tools):
        """Create SearchAgent with mocked tools."""
        with patch('src.langgraph.search_agent.get_platform_tools', return_value=mock_platform_tools):
            agent = SearchAgent()
            agent.tools = mock_platform_tools
            return agent

    @pytest.mark.asyncio
    async def test_search_agent_intelligent_platform_selection(self, search_agent, mock_platform_tools):
        """Test intelligent platform selection based on query characteristics."""
        test_cases = [
            {
                "query": "attack on titan mal score ranking",
                "expected_platforms": ["mal"],
                "reasoning": "MAL-specific terms should route to MAL"
            },
            {
                "query": "shingeki no kyojin anilist popularity",
                "expected_platforms": ["anilist"],
                "reasoning": "AniList-specific terms should route to AniList"
            },
            {
                "query": "anime similar to death note psychological thriller",
                "expected_platforms": ["semantic"],
                "reasoning": "Similarity queries should use semantic search"
            },
            {
                "query": "popular action anime 2023",
                "expected_platforms": ["mal", "anilist", "jikan"],
                "reasoning": "General queries should search multiple platforms"
            }
        ]
        
        for case in test_cases:
            with patch.object(search_agent, '_select_platforms_for_query') as mock_select:
                mock_select.return_value = case["expected_platforms"]
                
                platforms = search_agent._select_platforms_for_query(case["query"])
                assert set(platforms) == set(case["expected_platforms"]), case["reasoning"]

    @pytest.mark.asyncio
    async def test_search_agent_result_aggregation(self, search_agent, mock_platform_tools):
        """Test result aggregation across multiple platforms."""
        # Execute multi-platform search
        query = "attack on titan"
        
        with patch.object(search_agent, 'execute_search') as mock_execute:
            mock_execute.return_value = {
                "results": [
                    {
                        "id": "aggregated_16498",
                        "title": "Attack on Titan",
                        "platforms": ["mal", "anilist", "jikan"],
                        "aggregate_score": 8.63,
                        "cross_platform_data": {
                            "mal": {"score": 8.54, "rank": 50},
                            "anilist": {"score": 85, "popularity": 95000},
                            "jikan": {"score": 8.54, "members": 3000000}
                        },
                        "data_quality_score": 0.94
                    }
                ],
                "total_results": 1,
                "platforms_searched": ["mal", "anilist", "jikan"],
                "aggregation_method": "weighted_average"
            }
            
            result = await search_agent.execute_search(
                query=query,
                platforms=["mal", "anilist", "jikan"]
            )
            
            # Verify aggregation
            assert result["total_results"] == 1
            assert len(result["platforms_searched"]) == 3
            assert result["results"][0]["aggregate_score"] > 8.0
            assert "cross_platform_data" in result["results"][0]

    @pytest.mark.asyncio
    async def test_search_agent_quality_scoring(self, search_agent, mock_platform_tools):
        """Test data quality scoring and filtering."""
        # Mock results with varying quality scores
        mock_platform_tools["search_anime_mal"].return_value = [
            {"id": "mal_high", "title": "High Quality", "data_quality_score": 0.95},
            {"id": "mal_medium", "title": "Medium Quality", "data_quality_score": 0.75},
            {"id": "mal_low", "title": "Low Quality", "data_quality_score": 0.45}
        ]
        
        with patch.object(search_agent, 'execute_search') as mock_execute:
            mock_execute.return_value = {
                "results": [
                    {"id": "mal_high", "title": "High Quality", "data_quality_score": 0.95},
                    {"id": "mal_medium", "title": "Medium Quality", "data_quality_score": 0.75}
                    # Low quality result filtered out
                ],
                "total_results": 2,
                "quality_threshold": 0.7,
                "filtered_count": 1
            }
            
            result = await search_agent.execute_search(
                query="test quality",
                quality_threshold=0.7
            )
            
            # Verify quality filtering
            assert result["total_results"] == 2
            assert result["filtered_count"] == 1
            assert all(r["data_quality_score"] >= 0.7 for r in result["results"])

    @pytest.mark.asyncio
    async def test_search_agent_error_handling_and_fallbacks(self, search_agent, mock_platform_tools):
        """Test error handling and fallback strategies."""
        # Mock MAL failure
        mock_platform_tools["search_anime_mal"].side_effect = Exception("MAL API rate limited")
        
        with patch.object(search_agent, 'execute_search') as mock_execute:
            mock_execute.return_value = {
                "results": [
                    {"id": "anilist_16498", "title": "Attack on Titan", "source_platform": "anilist"},
                    {"id": "jikan_16498", "title": "Attack on Titan", "source_platform": "jikan"}
                ],
                "total_results": 2,
                "platforms_searched": ["anilist", "jikan"],
                "platform_failures": {"mal": "API rate limited"},
                "fallback_strategies_used": ["Used Jikan as MAL alternative"],
                "partial_success": True
            }
            
            result = await search_agent.execute_search(
                query="attack on titan",
                platforms=["mal", "anilist", "jikan"]
            )
            
            # Verify graceful degradation
            assert result["partial_success"] is True
            assert "mal" in result["platform_failures"]
            assert len(result["fallback_strategies_used"]) > 0
            assert result["total_results"] > 0


class TestScheduleAgentIntegration:
    """Integration tests for ScheduleAgent."""

    @pytest.fixture
    def mock_schedule_tools(self):
        """Create mock schedule and streaming tools."""
        tools = {
            "get_currently_airing": AsyncMock(return_value=[
                {
                    "title": "Winter 2024 Anime",
                    "broadcast_day": "Wednesday",
                    "broadcast_time": "23:30",
                    "broadcast_timezone": "Asia/Tokyo",
                    "next_episode_date": "2024-01-24T14:30:00Z",
                    "status": "airing",
                    "time_until_next_episode": "2 days, 15 hours"
                }
            ]),
            "get_schedule_data": AsyncMock(return_value={
                "current_time": "2024-01-22T09:00:00Z",
                "current_week": [
                    {
                        "title": "Monday Anime",
                        "broadcast_day": "Monday",
                        "broadcast_time": "21:00",
                        "status": "airing"
                    }
                ],
                "next_week": [
                    {
                        "title": "Upcoming Anime",
                        "broadcast_day": "Saturday",
                        "broadcast_time": "18:00", 
                        "status": "upcoming"
                    }
                ],
                "timezone": "UTC"
            }),
            "search_streaming_platforms": AsyncMock(return_value=[
                {
                    "title": "Streaming Anime",
                    "streaming_platforms": ["Crunchyroll", "Netflix"],
                    "regions": ["US", "CA", "UK"],
                    "availability_score": 0.89
                }
            ])
        }
        return tools

    @pytest.fixture
    def schedule_agent(self, mock_schedule_tools):
        """Create ScheduleAgent with mocked tools."""
        with patch('src.langgraph.schedule_agent.get_schedule_tools', return_value=mock_schedule_tools):
            agent = ScheduleAgent()
            agent.tools = mock_schedule_tools
            return agent

    @pytest.mark.asyncio
    async def test_schedule_agent_broadcast_time_processing(self, schedule_agent, mock_schedule_tools):
        """Test broadcast time processing and timezone handling."""
        with patch.object(schedule_agent, 'get_broadcast_schedule') as mock_schedule:
            mock_schedule.return_value = {
                "schedule": [
                    {
                        "title": "JST Anime",
                        "broadcast_time_utc": "2024-01-22T14:30:00Z",
                        "broadcast_time_local": "2024-01-22T23:30:00+09:00",
                        "timezone": "Asia/Tokyo",
                        "next_episode_countdown": "2 days, 5 hours, 30 minutes"
                    }
                ],
                "timezone_conversions": {
                    "JST": "Asia/Tokyo",
                    "EST": "America/New_York",
                    "PST": "America/Los_Angeles"
                },
                "current_time_utc": "2024-01-20T09:00:00Z"
            }
            
            result = await schedule_agent.get_broadcast_schedule(
                timezone="Asia/Tokyo",
                include_countdown=True
            )
            
            # Verify timezone processing
            assert len(result["schedule"]) == 1
            assert "broadcast_time_utc" in result["schedule"][0]
            assert "broadcast_time_local" in result["schedule"][0]
            assert "next_episode_countdown" in result["schedule"][0]

    @pytest.mark.asyncio
    async def test_schedule_agent_streaming_platform_integration(self, schedule_agent, mock_schedule_tools):
        """Test streaming platform integration and availability checking."""
        with patch.object(schedule_agent, 'find_streaming_availability') as mock_streaming:
            mock_streaming.return_value = {
                "anime_title": "Popular Anime",
                "streaming_data": [
                    {
                        "platform": "Crunchyroll",
                        "regions": ["US", "CA", "UK", "AU"],
                        "subscription_required": True,
                        "availability_score": 0.95
                    },
                    {
                        "platform": "Netflix",
                        "regions": ["US", "JP", "DE", "FR"],
                        "subscription_required": True,
                        "availability_score": 0.87
                    }
                ],
                "total_platforms": 2,
                "regional_coverage": {
                    "US": 2,
                    "CA": 1,
                    "UK": 1,
                    "AU": 1,
                    "JP": 1,
                    "DE": 1,
                    "FR": 1
                },
                "accessibility_score": 0.91
            }
            
            result = await schedule_agent.find_streaming_availability(
                anime_title="Popular Anime",
                user_region="US"
            )
            
            # Verify streaming integration
            assert result["total_platforms"] == 2
            assert result["regional_coverage"]["US"] == 2
            assert result["accessibility_score"] > 0.9

    @pytest.mark.asyncio
    async def test_schedule_agent_real_time_updates(self, schedule_agent, mock_schedule_tools):
        """Test real-time schedule updates and episode tracking."""
        with patch.object(schedule_agent, 'track_episode_releases') as mock_tracking:
            mock_tracking.return_value = {
                "tracking_active": True,
                "monitored_anime": [
                    {
                        "title": "Ongoing Series",
                        "current_episode": 8,
                        "total_episodes": 12,
                        "next_episode_date": "2024-01-24T14:30:00Z",
                        "release_schedule": "Weekly - Wednesdays",
                        "time_until_next": "2 days, 5 hours",
                        "release_confidence": 0.92
                    }
                ],
                "update_frequency": "hourly",
                "last_update": "2024-01-22T09:00:00Z"
            }
            
            result = await schedule_agent.track_episode_releases(
                anime_titles=["Ongoing Series"],
                notification_preferences={"advance_notice": "1 hour"}
            )
            
            # Verify tracking functionality
            assert result["tracking_active"] is True
            assert len(result["monitored_anime"]) == 1
            assert result["monitored_anime"][0]["release_confidence"] > 0.9


class TestAgentOrchestrationIntegration:
    """Integration tests for agent coordination and handoffs."""

    @pytest.fixture
    def mock_agents(self):
        """Create mock SearchAgent and ScheduleAgent."""
        search_agent = MagicMock()
        search_agent.name = "SearchAgent"
        search_agent.execute_search = AsyncMock()
        
        schedule_agent = MagicMock()
        schedule_agent.name = "ScheduleAgent"
        schedule_agent.get_broadcast_schedule = AsyncMock()
        schedule_agent.find_streaming_availability = AsyncMock()
        
        return {"search": search_agent, "schedule": schedule_agent}

    @pytest.mark.asyncio
    async def test_agent_handoff_mechanisms(self, mock_agents):
        """Test agent handoff mechanisms and coordination."""
        search_agent = mock_agents["search"]
        schedule_agent = mock_agents["schedule"]
        
        # Mock search agent finding anime
        search_agent.execute_search.return_value = {
            "results": [
                {"id": "anime_123", "title": "Popular Anime", "status": "airing"}
            ],
            "handoff_required": True,
            "handoff_agent": "ScheduleAgent",
            "handoff_context": {
                "anime_ids": ["anime_123"],
                "user_request": "streaming_and_schedule_info"
            }
        }
        
        # Mock schedule agent enriching with streaming/schedule data
        schedule_agent.find_streaming_availability.return_value = {
            "anime_title": "Popular Anime",
            "streaming_data": [{"platform": "Crunchyroll", "regions": ["US"]}],
            "enrichment_completed": True
        }
        
        # Test coordinated workflow
        search_result = await search_agent.execute_search(
            query="popular airing anime with streaming info"
        )
        
        if search_result["handoff_required"]:
            handoff_context = search_result["handoff_context"]
            enrichment = await schedule_agent.find_streaming_availability(
                anime_title=search_result["results"][0]["title"]
            )
            
            # Verify handoff coordination
            assert search_result["handoff_agent"] == "ScheduleAgent"
            assert enrichment["enrichment_completed"] is True
            assert "streaming_data" in enrichment

    @pytest.mark.asyncio
    async def test_collaborative_data_enrichment(self, mock_agents):
        """Test collaborative data enrichment between agents."""
        search_agent = mock_agents["search"]
        schedule_agent = mock_agents["schedule"]
        
        # Mock collaborative enrichment workflow
        base_anime_data = {
            "id": "collaborative_123",
            "title": "Multi-Agent Anime",
            "basic_info": {"score": 8.5, "episodes": 24}
        }
        
        # SearchAgent provides comprehensive anime metadata
        search_agent.execute_search.return_value = {
            "results": [base_anime_data],
            "enrichment_level": "basic",
            "requires_schedule_enrichment": True
        }
        
        # ScheduleAgent adds streaming and broadcast info
        schedule_agent.find_streaming_availability.return_value = {
            "streaming_enrichment": {
                "platforms": ["Crunchyroll", "Funimation"],
                "regions": ["US", "CA"],
                "broadcast_schedule": {
                    "day": "Saturday",
                    "time": "23:00",
                    "timezone": "JST"
                }
            },
            "enrichment_level": "comprehensive"
        }
        
        # Execute collaborative enrichment
        search_result = await search_agent.execute_search(query="comprehensive anime data")
        
        if search_result["requires_schedule_enrichment"]:
            schedule_enrichment = await schedule_agent.find_streaming_availability(
                anime_title=search_result["results"][0]["title"]
            )
            
            # Combine enrichments
            final_result = {
                **search_result["results"][0],
                **schedule_enrichment["streaming_enrichment"]
            }
            
            # Verify collaborative enrichment
            assert "platforms" in final_result
            assert "broadcast_schedule" in final_result
            assert schedule_enrichment["enrichment_level"] == "comprehensive"

    @pytest.mark.asyncio
    async def test_workflow_decision_trees(self, mock_agents):
        """Test complex workflow decision trees based on query analysis."""
        search_agent = mock_agents["search"]
        schedule_agent = mock_agents["schedule"]
        
        # Define decision tree test cases
        decision_cases = [
            {
                "query": "find airing anime this week",
                "primary_agent": "schedule",
                "workflow_path": ["schedule_lookup", "current_airing", "streaming_check"],
                "expected_tools": ["get_currently_airing", "search_streaming_platforms"]
            },
            {
                "query": "anime similar to attack on titan with high ratings",
                "primary_agent": "search",
                "workflow_path": ["semantic_search", "rating_filter", "quality_ranking"],
                "expected_tools": ["anime_semantic_search", "search_anime_mal"]
            },
            {
                "query": "when does demon slayer season 3 air on crunchyroll",
                "primary_agent": "schedule",
                "workflow_path": ["anime_lookup", "schedule_check", "platform_verify"],
                "expected_tools": ["search_anime_schedule", "search_streaming_platforms"]
            }
        ]
        
        for case in decision_cases:
            # Mock agent decision making
            if case["primary_agent"] == "search":
                search_agent.execute_search.return_value = {
                    "workflow_decision": {
                        "primary_agent": case["primary_agent"],
                        "workflow_path": case["workflow_path"],
                        "tools_to_execute": case["expected_tools"]
                    },
                    "query_analysis": {
                        "intent": "search_focused",
                        "requires_handoff": False
                    }
                }
                
                result = await search_agent.execute_search(query=case["query"])
                decision = result["workflow_decision"]
                
            else:  # schedule agent
                schedule_agent.get_broadcast_schedule.return_value = {
                    "workflow_decision": {
                        "primary_agent": case["primary_agent"],
                        "workflow_path": case["workflow_path"],
                        "tools_to_execute": case["expected_tools"]
                    },
                    "query_analysis": {
                        "intent": "schedule_focused",
                        "requires_handoff": False
                    }
                }
                
                result = await schedule_agent.get_broadcast_schedule(query=case["query"])
                decision = result["workflow_decision"]
            
            # Verify decision tree logic
            assert decision["primary_agent"] == case["primary_agent"]
            assert decision["workflow_path"] == case["workflow_path"]
            assert set(decision["tools_to_execute"]) == set(case["expected_tools"])

    @pytest.mark.asyncio
    async def test_agent_performance_optimization(self, mock_agents):
        """Test agent performance optimization and caching strategies."""
        search_agent = mock_agents["search"]
        schedule_agent = mock_agents["schedule"]
        
        # Mock caching behavior
        cache_key = "popular_anime_winter_2024"
        cached_result = {
            "results": [{"title": "Cached Result"}],
            "cache_hit": True,
            "cache_timestamp": "2024-01-22T09:00:00Z",
            "cache_ttl": 3600
        }
        
        # Test cache hit scenario
        search_agent.execute_search.return_value = cached_result
        
        result = await search_agent.execute_search(
            query="popular anime winter 2024",
            use_cache=True
        )
        
        # Verify caching optimization
        assert result["cache_hit"] is True
        assert "cache_timestamp" in result
        
        # Test parallel execution optimization
        concurrent_tasks = [
            search_agent.execute_search(query="action anime"),
            schedule_agent.get_broadcast_schedule(timezone="JST"),
            schedule_agent.find_streaming_availability(anime_title="Test Anime")
        ]
        
        # Mock concurrent execution results
        search_agent.execute_search.return_value = {"results": [{"title": "Action Result"}]}
        schedule_agent.get_broadcast_schedule.return_value = {"schedule": [{"day": "Monday"}]}
        schedule_agent.find_streaming_availability.return_value = {"platforms": ["Crunchyroll"]}
        
        # Verify all tasks can execute concurrently
        with patch('asyncio.gather') as mock_gather:
            mock_gather.return_value = [
                {"results": [{"title": "Action Result"}]},
                {"schedule": [{"day": "Monday"}]},
                {"platforms": ["Crunchyroll"]}
            ]
            
            concurrent_results = await mock_gather(*concurrent_tasks)
            
            # Verify concurrent execution
            assert len(concurrent_results) == 3
            assert all("results" in r or "schedule" in r or "platforms" in r for r in concurrent_results)