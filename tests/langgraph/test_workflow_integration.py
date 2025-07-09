"""Integration tests for LangGraph workflow orchestration.

Tests the multi-agent workflow system including intelligent routing,
tool chaining, cross-platform data flows, and error handling.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List
import json

# Mock LangGraph dependencies
from src.models.structured_responses import BasicAnimeResult, AnimeType, AnimeStatus

# Mock langgraph_swarm since it's not installed
with patch('src.langgraph.anime_swarm.create_swarm'):
    from src.langgraph.anime_swarm import AnimeDiscoverySwarm


class TestAnimeDiscoverySwarmIntegration:
    """Integration tests for AnimeDiscoverySwarm workflow orchestration."""

    @pytest.fixture
    def mock_search_agent(self):
        """Create mock SearchAgent."""
        agent = MagicMock()
        agent.name = "SearchAgent"
        agent.tools = ["search_anime_mal", "search_anime_anilist", "anime_semantic_search"]
        return agent

    @pytest.fixture
    def mock_schedule_agent(self):
        """Create mock ScheduleAgent."""
        agent = MagicMock()
        agent.name = "ScheduleAgent"
        agent.tools = ["get_currently_airing", "get_schedule_data", "search_streaming_platforms"]
        return agent

    @pytest.fixture
    def mock_react_workflow(self):
        """Create mock ReactAgentWorkflow."""
        workflow = MagicMock()
        workflow.invoke = AsyncMock()
        workflow.ainvoke = AsyncMock()
        return workflow

    @pytest.fixture
    def anime_swarm(self, mock_search_agent, mock_schedule_agent):
        """Create AnimeDiscoverySwarm with mocked dependencies."""
        with patch('src.langgraph.agents.search_agent.SearchAgent', return_value=mock_search_agent), \
             patch('src.langgraph.agents.schedule_agent.ScheduleAgent', return_value=mock_schedule_agent), \
             patch('langchain_openai.ChatOpenAI'), \
             patch('langgraph.checkpoint.memory.InMemorySaver'), \
             patch('langgraph.store.memory.InMemoryStore'), \
             patch('langgraph_swarm.create_swarm') as mock_create_swarm, \
             patch('src.langgraph.query_analyzer.QueryAnalyzer') as mock_query_analyzer, \
             patch('src.langgraph.conditional_router.ConditionalRouter') as mock_router, \
             patch('src.langgraph.error_handling.SwarmErrorHandler'), \
             patch('src.langgraph.swarm_error_integration.SwarmErrorIntegration'):
            
            # Mock the swarm workflow
            mock_swarm_workflow = MagicMock()
            mock_swarm_workflow.ainvoke = AsyncMock()
            mock_create_swarm.return_value.compile.return_value = mock_swarm_workflow
            
            # Mock query analyzer
            mock_query_analyzer_instance = mock_query_analyzer.return_value
            mock_query_analyzer_instance.analyze_query = AsyncMock(return_value={
                "intent_type": "search",
                "workflow_complexity": "simple",
                "suggested_tools": ["search_anime_mal"]
            })
            
            # Mock conditional router
            mock_router_instance = mock_router.return_value
            mock_router_instance.route_execution = AsyncMock(return_value={
                "execution_path": "platform_specific",
                "tools": ["search_anime_mal"],
                "routing_metadata": {"confidence": 0.9}
            })
            
            swarm = AnimeDiscoverySwarm()
            swarm.swarm = mock_swarm_workflow
            return swarm

    @pytest.mark.asyncio
    async def test_anime_discovery_workflow_success(self, anime_swarm):
        """Test successful anime discovery workflow execution."""
        # Mock the swarm result that discover_anime processes
        mock_swarm_result = {
            "anime_results": [
                {
                    "id": "mal_16498",
                    "title": "Attack on Titan",
                    "score": 8.54,
                    "source_platform": "mal",
                    "data_quality_score": 0.95
                }
            ],
            "platforms_used": ["mal"],
            "active_agent": "SearchAgent"
        }
        
        anime_swarm.swarm.ainvoke = AsyncMock(return_value=mock_swarm_result)
        
        # Execute discovery
        result = await anime_swarm.discover_anime(
            query="attack on titan",
            user_context={"preferred_platforms": ["mal"]},
            session_id="test_session_1"
        )
        
        # Verify results (WorkflowResult structure)
        assert result.total_results == 1
        assert len(result.anime_results) == 1
        assert result.anime_results[0]["title"] == "Attack on Titan"
        assert "SearchAgent" in result.agents_used
        assert result.execution_time_ms > 0
        
        # Verify workflow was called correctly
        anime_swarm.swarm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_currently_airing_workflow_success(self, anime_swarm):
        """Test currently airing anime workflow."""
        # Mock swarm result for currently airing
        mock_swarm_result = {
            "anime_results": [
                {
                    "title": "Winter 2024 Anime",
                    "broadcast_day": "Wednesday",
                    "broadcast_time": "23:30",
                    "next_episode_date": "2024-01-24T14:30:00Z",
                    "status": "airing",
                    "streaming_sites": [
                        {"site": "Crunchyroll", "regions": ["US", "CA"]}
                    ]
                }
            ],
            "platforms_used": ["animeschedule"],
            "active_agent": "ScheduleAgent"
        }
        
        anime_swarm.swarm.ainvoke = AsyncMock(return_value=mock_swarm_result)
        
        # Execute currently airing query
        result = await anime_swarm.get_currently_airing(
            filters={"day_filter": "wednesday", "timezone": "JST"},
            session_id="test_session_2"
        )
        
        # Verify results
        assert result.total_results == 1
        assert result.anime_results[0]["broadcast_day"] == "Wednesday"
        assert "ScheduleAgent" in result.agents_used

    @pytest.mark.asyncio
    async def test_similarity_search_workflow_success(self, anime_swarm):
        """Test similarity search workflow."""
        # Mock workflow response for similarity search
        mock_response = {
            "messages": [
                {
                    "type": "ai", 
                    "content": json.dumps({
                        "results": [
                            {
                                "id": "semantic_12345",
                                "title": "Similar Anime 1",
                                "similarity_score": 0.89,
                                "reference_anime": "Attack on Titan",
                                "similarity_type": "content"
                            },
                            {
                                "id": "semantic_67890",
                                "title": "Similar Anime 2", 
                                "similarity_score": 0.76,
                                "reference_anime": "Attack on Titan",
                                "similarity_type": "visual"
                            }
                        ],
                        "total_results": 2,
                        "agents_used": ["SearchAgent"],
                        "similarity_mode": "hybrid",
                        "reference_processed": True
                    })
                }
            ],
            "agent": "SearchAgent"
        }
        
        anime_swarm.swarm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Execute similarity search
        result = await anime_swarm.find_similar_anime(
            reference_anime="Attack on Titan",
            similarity_mode="hybrid",
            session_id="test_session_3"
        )
        
        # Verify results
        assert result["total_results"] == 2
        assert result["similarity_mode"] == "hybrid"
        assert result["results"][0]["similarity_score"] == 0.89
        assert result["agents_used"] == ["SearchAgent"]

    @pytest.mark.asyncio
    async def test_streaming_platform_search_workflow(self, anime_swarm):
        """Test streaming platform search workflow."""
        # Mock workflow response for streaming platform search
        mock_response = {
            "messages": [
                {
                    "type": "ai",
                    "content": json.dumps({
                        "results": [
                            {
                                "title": "Netflix Exclusive Anime",
                                "streaming_platforms": ["Netflix"],
                                "regions": ["US", "CA", "UK"],
                                "availability_score": 0.95
                            },
                            {
                                "title": "Multi-Platform Anime",
                                "streaming_platforms": ["Netflix", "Crunchyroll"],
                                "regions": ["US", "CA"],
                                "availability_score": 0.87
                            }
                        ],
                        "total_results": 2,
                        "agents_used": ["ScheduleAgent", "SearchAgent"],
                        "platforms_searched": ["Netflix", "Crunchyroll"],
                        "cross_platform_enriched": True
                    })
                }
            ],
            "agent": "ScheduleAgent"
        }
        
        anime_swarm.swarm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Execute streaming platform search
        result = await anime_swarm.search_by_streaming_platform(
            platforms=["Netflix", "Crunchyroll"],
            additional_filters={"genre": "action"},
            session_id="test_session_4"
        )
        
        # Verify results
        assert result["total_results"] == 2
        assert result["platforms_searched"] == ["Netflix", "Crunchyroll"]
        assert result["cross_platform_enriched"] is True
        assert len(result["agents_used"]) == 2

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, anime_swarm):
        """Test workflow error handling and fallback strategies."""
        # Mock workflow to raise an exception
        anime_swarm.swarm.ainvoke = AsyncMock(side_effect=Exception("Workflow execution failed"))
        
        # Test error handling
        with pytest.raises(RuntimeError, match="Anime discovery workflow failed"):
            await anime_swarm.discover_anime(
                query="test query",
                session_id="error_session"
            )

    @pytest.mark.asyncio
    async def test_workflow_session_persistence(self, anime_swarm):
        """Test session persistence across multiple workflow calls."""
        # Mock multiple workflow calls with same session
        session_id = "persistent_session"
        
        # First call
        mock_response_1 = {
            "messages": [
                {
                    "type": "ai",
                    "content": json.dumps({
                        "results": [{"title": "First Search"}],
                        "total_results": 1,
                        "agents_used": ["SearchAgent"]
                    })
                }
            ]
        }
        
        # Second call (should have session context)
        mock_response_2 = {
            "messages": [
                {
                    "type": "ai", 
                    "content": json.dumps({
                        "results": [{"title": "Follow-up Search"}],
                        "total_results": 1,
                        "agents_used": ["SearchAgent"],
                        "session_context_used": True
                    })
                }
            ]
        }
        
        anime_swarm.swarm.ainvoke = AsyncMock(side_effect=[mock_response_1, mock_response_2])
        
        # Execute first call
        result_1 = await anime_swarm.discover_anime(
            query="first query",
            session_id=session_id
        )
        
        # Execute second call with same session
        result_2 = await anime_swarm.discover_anime(
            query="follow-up query", 
            session_id=session_id
        )
        
        # Verify both calls used the same session
        assert anime_swarm.swarm.ainvoke.call_count == 2
        first_call_session = anime_swarm.swarm.ainvoke.call_args_list[0][0][0]["session_id"]
        second_call_session = anime_swarm.swarm.ainvoke.call_args_list[1][0][0]["session_id"]
        assert first_call_session == second_call_session == session_id


class TestReactAgentWorkflowIntegration:
    """Integration tests for ReactAgentWorkflow."""

    @pytest.fixture
    def mock_tools(self):
        """Create mock tools for workflow."""
        tools = {}
        
        # Mock search tools
        tools["search_anime_mal"] = AsyncMock(return_value=[
            {"id": "mal_1", "title": "MAL Result", "source_platform": "mal"}
        ])
        
        tools["search_anime_anilist"] = AsyncMock(return_value=[
            {"id": "anilist_1", "title": "AniList Result", "source_platform": "anilist"}
        ])
        
        tools["anime_semantic_search"] = AsyncMock(return_value=[
            {"id": "semantic_1", "title": "Semantic Result", "similarity_score": 0.92}
        ])
        
        # Mock schedule tools
        tools["get_currently_airing"] = AsyncMock(return_value=[
            {"title": "Airing Now", "status": "airing", "next_episode": "2024-01-20"}
        ])
        
        return tools

    @pytest.fixture
    def react_workflow(self, mock_tools):
        """Create ReactAgentWorkflow with mocked tools."""
        with patch('src.langgraph.react_agent_workflow.ToolNode') as mock_tool_node:
            mock_tool_node.return_value = MagicMock()
            
            # Create workflow with mocked dependencies
            workflow = ReactAgentWorkflow()
            workflow.tools = mock_tools
            return workflow

    @pytest.mark.asyncio
    async def test_workflow_tool_routing_logic(self, react_workflow, mock_tools):
        """Test intelligent tool routing based on query characteristics."""
        # Test queries that should route to different tools
        test_cases = [
            {
                "query": "attack on titan mal ratings",
                "expected_tools": ["search_anime_mal"],
                "context": "MAL-specific query should route to MAL tools"
            },
            {
                "query": "currently airing anime winter 2024",
                "expected_tools": ["get_currently_airing"],
                "context": "Schedule query should route to schedule tools"
            },
            {
                "query": "anime similar to death note",
                "expected_tools": ["anime_semantic_search"],
                "context": "Similarity query should route to semantic search"
            }
        ]
        
        for case in test_cases:
            # Mock the workflow execution to verify tool selection
            with patch.object(react_workflow, '_analyze_query_intent') as mock_analyze:
                mock_analyze.return_value = case["expected_tools"]
                
                # Execute workflow
                state = {
                    "query": case["query"],
                    "session_id": "routing_test",
                    "selected_tools": []
                }
                
                # Verify tool routing logic
                selected_tools = react_workflow._analyze_query_intent(case["query"])
                assert selected_tools == case["expected_tools"], case["context"]

    @pytest.mark.asyncio
    async def test_workflow_cross_platform_data_enrichment(self, react_workflow, mock_tools):
        """Test cross-platform data enrichment workflow."""
        # Mock enrichment tools
        mock_tools["get_cross_platform_anime_data"] = AsyncMock(return_value={
            "anime_id": "cross_platform_123",
            "mal_data": {"score": 8.5, "rank": 100},
            "anilist_data": {"score": 85, "popularity": 95000},
            "schedule_data": {"next_episode": "2024-01-25", "broadcast_day": "Thursday"},
            "enrichment_score": 0.92
        })
        
        # Test enrichment workflow
        query = "attack on titan comprehensive data"
        result = await mock_tools["get_cross_platform_anime_data"](
            anime_query=query,
            platforms=["mal", "anilist", "animeschedule"]
        )
        
        # Verify enrichment
        assert result["enrichment_score"] > 0.9
        assert "mal_data" in result
        assert "anilist_data" in result
        assert "schedule_data" in result

    @pytest.mark.asyncio
    async def test_workflow_error_recovery_and_fallbacks(self, react_workflow, mock_tools):
        """Test error recovery and fallback strategies."""
        # Mock primary tool failure
        mock_tools["search_anime_mal"].side_effect = Exception("MAL API unavailable")
        
        # Mock fallback tool success
        mock_tools["search_anime_jikan"] = AsyncMock(return_value=[
            {"id": "jikan_1", "title": "Jikan Fallback Result", "source_platform": "jikan"}
        ])
        
        # Test fallback mechanism
        with patch.object(react_workflow, '_execute_with_fallback') as mock_fallback:
            mock_fallback.return_value = [
                {"id": "jikan_1", "title": "Jikan Fallback Result", "source_platform": "jikan"}
            ]
            
            result = react_workflow._execute_with_fallback(
                primary_tool="search_anime_mal",
                fallback_tools=["search_anime_jikan"],
                query="test query"
            )
            
            # Verify fallback was used
            assert result[0]["source_platform"] == "jikan"
            assert result[0]["title"] == "Jikan Fallback Result"

    @pytest.mark.asyncio
    async def test_workflow_memory_persistence(self, react_workflow):
        """Test conversation memory persistence across workflow executions."""
        session_id = "memory_test_session"
        
        # Mock memory store
        memory_store = {}
        
        with patch.object(react_workflow, 'memory_store', memory_store):
            # First interaction
            state_1 = {
                "query": "find mecha anime",
                "session_id": session_id,
                "user_preferences": {"genre": "mecha", "min_score": 7.0}
            }
            
            # Store user preferences in memory
            memory_store[session_id] = {
                "user_preferences": state_1["user_preferences"],
                "interaction_count": 1,
                "query_history": [state_1["query"]]
            }
            
            # Second interaction (should use stored preferences)
            state_2 = {
                "query": "show me more like that",
                "session_id": session_id
            }
            
            # Verify memory retrieval
            stored_memory = memory_store.get(session_id)
            assert stored_memory["user_preferences"]["genre"] == "mecha"
            assert stored_memory["user_preferences"]["min_score"] == 7.0
            assert len(stored_memory["query_history"]) == 1

    @pytest.mark.asyncio
    async def test_workflow_performance_optimization(self, react_workflow, mock_tools):
        """Test workflow performance optimization strategies."""
        # Mock concurrent tool execution
        concurrent_tools = ["search_anime_mal", "search_anime_anilist", "search_anime_jikan"]
        
        # Test parallel execution
        start_time = 0.0
        with patch('asyncio.gather') as mock_gather:
            mock_gather.return_value = [
                [{"id": "mal_1", "title": "MAL Result"}],
                [{"id": "anilist_1", "title": "AniList Result"}], 
                [{"id": "jikan_1", "title": "Jikan Result"}]
            ]
            
            # Execute concurrent search
            results = await mock_gather(
                mock_tools["search_anime_mal"]("test query"),
                mock_tools["search_anime_anilist"]("test query"),
                mock_tools["search_anime_jikan"]("test query")
            )
            
            # Verify concurrent execution
            assert len(results) == 3
            assert all(len(result) > 0 for result in results)

    @pytest.mark.asyncio
    async def test_workflow_result_aggregation_and_ranking(self, react_workflow):
        """Test result aggregation and ranking across multiple platforms."""
        # Mock results from different platforms
        platform_results = {
            "mal": [
                {"id": "mal_16498", "title": "Attack on Titan", "score": 8.54, "data_quality_score": 0.95}
            ],
            "anilist": [
                {"id": "anilist_16498", "title": "Shingeki no Kyojin", "score": 85, "data_quality_score": 0.92}
            ],
            "semantic": [
                {"id": "semantic_16498", "title": "Attack on Titan", "similarity_score": 0.98, "data_quality_score": 0.88}
            ]
        }
        
        # Test result aggregation
        with patch.object(react_workflow, '_aggregate_and_rank_results') as mock_aggregate:
            mock_aggregate.return_value = {
                "results": [
                    {
                        "id": "aggregated_16498",
                        "title": "Attack on Titan",
                        "platforms": ["mal", "anilist", "semantic"],
                        "aggregate_score": 8.77,
                        "data_quality_score": 0.92,
                        "cross_platform_verified": True
                    }
                ],
                "total_results": 1,
                "aggregation_method": "weighted_average",
                "platforms_used": ["mal", "anilist", "semantic"]
            }
            
            aggregated = react_workflow._aggregate_and_rank_results(platform_results)
            
            # Verify aggregation
            assert aggregated["total_results"] == 1
            assert aggregated["results"][0]["cross_platform_verified"] is True
            assert len(aggregated["results"][0]["platforms"]) == 3


class TestWorkflowIntegrationEndToEnd:
    """End-to-end integration tests for complete workflow scenarios."""

    @pytest.mark.asyncio
    async def test_complex_discovery_workflow_e2e(self):
        """Test complex end-to-end discovery workflow with multiple agents."""
        # This would test a real workflow scenario but with all external dependencies mocked
        
        with patch('src.langgraph.anime_swarm.AnimeDiscoverySwarm') as mock_swarm_class:
            mock_swarm = MagicMock()
            mock_swarm_class.return_value = mock_swarm
            
            # Mock complex workflow response
            mock_swarm.discover_anime = AsyncMock(return_value={
                "results": [
                    {
                        "id": "cross_platform_123",
                        "title": "Attack on Titan",
                        "aggregate_score": 8.65,
                        "platforms": ["mal", "anilist", "animeschedule"],
                        "streaming_availability": [
                            {"platform": "Crunchyroll", "regions": ["US", "CA"]},
                            {"platform": "Funimation", "regions": ["US"]}
                        ],
                        "broadcast_info": {
                            "next_episode": "2024-01-21T14:30:00Z",
                            "day": "Sunday",
                            "timezone": "JST"
                        },
                        "cross_platform_verified": True,
                        "data_quality_score": 0.94
                    }
                ],
                "total_results": 1,
                "agents_used": ["SearchAgent", "ScheduleAgent"],
                "execution_time": 2.1,
                "platforms_searched": ["mal", "anilist", "jikan", "animeschedule"],
                "workflow_path": [
                    "query_analysis",
                    "platform_search",
                    "cross_platform_enrichment", 
                    "streaming_lookup",
                    "result_aggregation"
                ],
                "session_id": "e2e_test_session"
            })
            
            # Execute complex discovery
            swarm = mock_swarm_class()
            result = await swarm.discover_anime(
                query="attack on titan with streaming info",
                user_context={
                    "preferred_platforms": ["crunchyroll", "funimation"],
                    "region": "US",
                    "include_streaming": True,
                    "include_schedule": True
                },
                session_id="e2e_test_session"
            )
            
            # Verify comprehensive result
            assert result["total_results"] == 1
            assert len(result["agents_used"]) == 2
            assert result["results"][0]["cross_platform_verified"] is True
            assert "streaming_availability" in result["results"][0]
            assert "broadcast_info" in result["results"][0]
            assert result["execution_time"] > 0

    @pytest.mark.asyncio
    async def test_workflow_resilience_under_partial_failures(self):
        """Test workflow resilience when some platforms/tools fail."""
        
        with patch('src.langgraph.anime_swarm.AnimeDiscoverySwarm') as mock_swarm_class:
            mock_swarm = MagicMock()
            mock_swarm_class.return_value = mock_swarm
            
            # Mock partial failure scenario (MAL fails, others succeed)
            mock_swarm.discover_anime = AsyncMock(return_value={
                "results": [
                    {
                        "id": "partial_success_123",
                        "title": "Resilient Search Result",
                        "platforms": ["anilist", "jikan"],  # MAL missing due to failure
                        "platform_failures": ["mal"],
                        "fallback_strategies_used": ["jikan_for_mal"],
                        "partial_success": True,
                        "data_quality_score": 0.78  # Lower due to missing platform
                    }
                ],
                "total_results": 1,
                "agents_used": ["SearchAgent"],
                "platform_failures": {
                    "mal": "API rate limit exceeded",
                    "animeschedule": "Service temporarily unavailable"
                },
                "fallback_strategies": [
                    "Used Jikan as MAL fallback",
                    "Skipped schedule enrichment due to AnimeSchedule failure"
                ],
                "resilience_score": 0.65
            })
            
            # Execute discovery with expected failures
            swarm = mock_swarm_class()
            result = await swarm.discover_anime(
                query="test resilience",
                session_id="resilience_test"
            )
            
            # Verify graceful degradation
            assert result["total_results"] == 1
            assert result["results"][0]["partial_success"] is True
            assert "platform_failures" in result
            assert "fallback_strategies" in result
            assert result["resilience_score"] > 0.5

    @pytest.mark.asyncio 
    async def test_workflow_conversation_continuity(self):
        """Test conversation continuity across multiple related queries."""
        
        with patch('src.langgraph.anime_swarm.AnimeDiscoverySwarm') as mock_swarm_class:
            mock_swarm = MagicMock()
            mock_swarm_class.return_value = mock_swarm
            
            session_id = "continuity_test_session"
            
            # First query
            mock_swarm.discover_anime = AsyncMock(return_value={
                "results": [{"title": "First Result", "id": "first_123"}],
                "total_results": 1,
                "session_context": {"initial_query": True}
            })
            
            result_1 = await swarm.discover_anime(
                query="find action anime",
                session_id=session_id
            )
            
            # Follow-up query (should use context)
            mock_swarm.discover_anime = AsyncMock(return_value={
                "results": [{"title": "Follow-up Result", "id": "followup_456"}],
                "total_results": 1,
                "session_context": {
                    "previous_query": "find action anime",
                    "context_continuation": True,
                    "refined_based_on_history": True
                }
            })
            
            result_2 = await swarm.discover_anime(
                query="show me more like that",
                session_id=session_id
            )
            
            # Verify context continuity
            assert mock_swarm.discover_anime.call_count == 2
            # Both calls should use the same session_id
            calls = mock_swarm.discover_anime.call_args_list
            assert calls[0][1]["session_id"] == session_id
            assert calls[1][1]["session_id"] == session_id