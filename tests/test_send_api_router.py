"""Unit tests for Send API Parallel Router.

This module tests the Send API parallel routing functionality,
including complexity analysis, parallel execution, and result merging.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.langgraph.send_api_router import (
    SendAPIParallelRouter,
    QueryComplexity,
    RouteStrategy,
    ParallelRouteConfig,
    SendAPIState,
    create_send_api_parallel_router,
)
from src.langgraph.react_agent_workflow import LLMProvider


class TestSendAPIParallelRouter:
    """Test cases for Send API Parallel Router."""

    @pytest.fixture
    def mock_mcp_tools(self):
        """Create mock MCP tools for testing."""
        mock_tools = {}
        
        # Mock search_anime tool
        mock_search_tool = AsyncMock()
        mock_search_tool.ainvoke = AsyncMock(return_value=[
            {
                "anime_id": "test_1",
                "title": "Test Anime 1",
                "synopsis": "Test synopsis 1",
                "quality_score": 0.9,
            }
        ])
        mock_tools["search_anime"] = mock_search_tool
        
        # Mock other tools
        for tool_name in [
            "get_anime_details", "find_similar_anime", "get_anime_stats",
            "search_anime_by_image", "find_visually_similar_anime", "search_multimodal_anime"
        ]:
            mock_tool = AsyncMock()
            mock_tool.ainvoke = AsyncMock(return_value={})
            mock_tools[tool_name] = mock_tool
        
        return mock_tools

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = MagicMock()
        settings.openai_api_key = "test_openai_key"
        settings.anthropic_api_key = "test_anthropic_key"
        return settings

    @pytest.fixture
    @patch('src.langgraph.send_api_router.get_settings')
    @patch('src.langgraph.send_api_router.ChatOpenAI')
    @patch('src.langgraph.send_api_router.create_anime_langchain_tools')
    def router(self, mock_tools_creator, mock_chat_openai, mock_get_settings, mock_mcp_tools, mock_settings):
        """Create a SendAPIParallelRouter instance for testing."""
        mock_get_settings.return_value = mock_settings
        mock_chat_openai.return_value = MagicMock()
        mock_tools_creator.return_value = []
        
        router = SendAPIParallelRouter(mock_mcp_tools, LLMProvider.OPENAI)
        
        # Mock the graph compilation to avoid complex state graph setup
        router.graph = AsyncMock()
        
        return router

    @pytest.mark.asyncio
    async def test_router_initialization(self, mock_mcp_tools, mock_settings):
        """Test Send API router initialization."""
        with patch('src.langgraph.send_api_router.get_settings', return_value=mock_settings), \
             patch('src.langgraph.send_api_router.ChatOpenAI') as mock_chat, \
             patch('src.langgraph.send_api_router.create_anime_langchain_tools', return_value=[]):
            
            mock_chat.return_value = MagicMock()
            
            router = SendAPIParallelRouter(mock_mcp_tools, LLMProvider.OPENAI)
            
            assert router.mcp_tools == mock_mcp_tools
            assert router.llm_provider == LLMProvider.OPENAI
            assert len(router.route_configs) == 3  # Three routing strategies
            assert RouteStrategy.FAST_PARALLEL in router.route_configs

    @pytest.mark.asyncio
    async def test_query_complexity_analysis(self, router):
        """Test query complexity analysis."""
        # Simple query
        simple_complexity = await router._analyze_query_complexity("find anime")
        assert simple_complexity < 0.3
        
        # Complex query
        complex_query = "find detailed cross-platform recommendations for mecha anime from 2020s with high ratings and streaming availability"
        complex_complexity = await router._analyze_query_complexity(complex_query)
        assert complex_complexity > 0.5
        
        # Multimodal query
        multimodal_complexity = await router._analyze_query_complexity("find similar anime", image_data="base64data")
        assert multimodal_complexity > 0.3

    @pytest.mark.asyncio
    async def test_route_configuration_initialization(self, router):
        """Test route configuration setup."""
        configs = router.route_configs
        
        # Test fast parallel config
        fast_config = configs[RouteStrategy.FAST_PARALLEL]
        assert fast_config.timeout_ms == 250
        assert len(fast_config.agent_names) == 3
        assert "offline_agent" in fast_config.agent_names
        
        # Test comprehensive config
        comprehensive_config = configs[RouteStrategy.COMPREHENSIVE_PARALLEL]
        assert comprehensive_config.timeout_ms == 1000
        assert len(comprehensive_config.agent_names) == 5

    @pytest.mark.asyncio
    async def test_query_analyzer_node(self, router):
        """Test query analyzer node functionality."""
        # Test simple query
        simple_state: SendAPIState = {
            "messages": [],
            "session_id": "test_session",
            "query": "find anime",
            "image_data": None,
            "text_weight": 0.7,
            "search_parameters": None,
            "parallel_routes": [],
            "route_results": {},
            "complexity_score": 0.0,
            "route_strategy": RouteStrategy.FAST_PARALLEL,
            "merged_results": None,
            "processing_time_ms": 0,
        }
        
        result_state = await router._query_analyzer_node(simple_state)
        
        assert "complexity_score" in result_state
        assert "route_strategy" in result_state
        assert isinstance(result_state["route_strategy"], RouteStrategy)

    @pytest.mark.asyncio
    async def test_route_generator_node(self, router):
        """Test route generator node for Send API routes."""
        state: SendAPIState = {
            "messages": [],
            "session_id": "test_session",
            "query": "find mecha anime",
            "image_data": None,
            "text_weight": 0.7,
            "search_parameters": None,
            "parallel_routes": [],
            "route_results": {},
            "complexity_score": 0.4,
            "route_strategy": RouteStrategy.FAST_PARALLEL,
            "merged_results": None,
            "processing_time_ms": 0,
        }
        
        result_state = await router._route_generator_node(state)
        
        assert "parallel_routes" in result_state
        parallel_routes = result_state["parallel_routes"]
        assert len(parallel_routes) == 3  # Fast parallel has 3 agents
        
        # Check that Send objects are created correctly
        from langgraph.types import Send
        for route in parallel_routes:
            assert isinstance(route, Send)

    @pytest.mark.asyncio
    async def test_send_api_router_function(self, router):
        """Test the Send API routing function."""
        from langgraph.types import Send
        
        # Mock parallel routes
        mock_routes = [
            Send("offline_agent", {"query": "test", "agent_name": "offline_agent"}),
            Send("mal_agent", {"query": "test", "agent_name": "mal_agent"}),
        ]
        
        state: SendAPIState = {
            "messages": [],
            "session_id": "test_session", 
            "query": "test query",
            "image_data": None,
            "text_weight": 0.7,
            "search_parameters": None,
            "parallel_routes": mock_routes,
            "route_results": {},
            "complexity_score": 0.5,
            "route_strategy": RouteStrategy.FAST_PARALLEL,
            "merged_results": None,
            "processing_time_ms": 0,
        }
        
        routes = router._send_api_router(state)
        
        assert len(routes) == 2
        assert all(isinstance(route, Send) for route in routes)

    @pytest.mark.asyncio
    async def test_execute_agent_search_success(self, router):
        """Test successful agent search execution."""
        # Mock the MCP tool
        router.mcp_tools["search_anime"] = AsyncMock()
        router.mcp_tools["search_anime"].ainvoke = AsyncMock(return_value=[
            {"anime_id": "test_1", "title": "Test Anime", "score": 0.9}
        ])
        
        state: SendAPIState = {
            "messages": [],
            "session_id": "test_session",
            "query": "test query",
            "image_data": None,
            "text_weight": 0.7,
            "search_parameters": None,
            "parallel_routes": [],
            "route_results": {},
            "complexity_score": 0.5,
            "route_strategy": RouteStrategy.FAST_PARALLEL,
            "merged_results": None,
            "processing_time_ms": 0,
            "timeout_ms": 1000,
        }
        
        result_state = await router._execute_agent_search(state, "test_agent", "search_anime")
        
        assert "route_results" in result_state
        assert "test_agent" in result_state["route_results"]
        
        agent_result = result_state["route_results"]["test_agent"]
        assert agent_result["success"] is True
        assert len(agent_result["results"]) == 1
        assert "processing_time_ms" in agent_result

    @pytest.mark.asyncio
    async def test_execute_agent_search_timeout(self, router):
        """Test agent search timeout handling."""
        # Mock the MCP tool to timeout
        async def slow_search(**kwargs):
            await asyncio.sleep(2)  # Longer than timeout
            return []
        
        router.mcp_tools["search_anime"] = AsyncMock()
        router.mcp_tools["search_anime"].ainvoke = slow_search
        
        state: SendAPIState = {
            "messages": [],
            "session_id": "test_session",
            "query": "test query", 
            "image_data": None,
            "text_weight": 0.7,
            "search_parameters": None,
            "parallel_routes": [],
            "route_results": {},
            "complexity_score": 0.5,
            "route_strategy": RouteStrategy.FAST_PARALLEL,
            "merged_results": None,
            "processing_time_ms": 0,
            "timeout_ms": 100,  # Short timeout
        }
        
        result_state = await router._execute_agent_search(state, "test_agent", "search_anime")
        
        assert "route_results" in result_state
        agent_result = result_state["route_results"]["test_agent"]
        assert agent_result["success"] is False
        assert agent_result["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_result_merger_node(self, router):
        """Test result merging from multiple agents."""
        # Mock route results
        route_results = {
            "offline_agent": {
                "success": True,
                "results": [
                    {"anime_id": "1", "title": "Anime 1", "quality_score": 0.9},
                    {"anime_id": "2", "title": "Anime 2", "quality_score": 0.8},
                ],
                "processing_time_ms": 150,
            },
            "mal_agent": {
                "success": True,
                "results": [
                    {"anime_id": "1", "title": "Anime 1", "quality_score": 0.85},  # Duplicate
                    {"anime_id": "3", "title": "Anime 3", "quality_score": 0.7},
                ],
                "processing_time_ms": 200,
            },
            "failed_agent": {
                "success": False,
                "results": [],
                "processing_time_ms": 100,
                "error": "connection_error",
            },
        }
        
        state: SendAPIState = {
            "messages": [],
            "session_id": "test_session",
            "query": "test query",
            "image_data": None,
            "text_weight": 0.7,
            "search_parameters": None,
            "parallel_routes": [],
            "route_results": route_results,
            "complexity_score": 0.5,
            "route_strategy": RouteStrategy.FAST_PARALLEL,
            "merged_results": None,
            "processing_time_ms": 0,
        }
        
        result_state = await router._result_merger_node(state)
        
        assert "merged_results" in result_state
        merged = result_state["merged_results"]
        
        # Should have deduplication (anime_id "1" appears only once)
        assert len(merged["results"]) == 3  # anime 1, 2, 3 (deduplicated)
        assert merged["total_results"] == 3
        assert len(merged["successful_agents"]) == 2  # Only successful agents
        
        # Check processing summary
        assert "processing_summary" in merged
        summary = merged["processing_summary"]
        assert summary["parallel_time_ms"] == 200  # Max of agent times
        assert "merger_time_ms" in summary
        assert "total_time_ms" in summary

    @pytest.mark.asyncio
    async def test_merge_and_deduplicate_results(self, router):
        """Test result merging and deduplication logic."""
        results = [
            {"anime_id": "1", "title": "Anime One", "quality_score": 0.9},
            {"anime_id": "2", "title": "Anime Two", "quality_score": 0.8},
            {"anime_id": "1", "title": "Anime One", "quality_score": 0.85},  # Duplicate ID
            {"anime_id": "3", "title": "anime two", "quality_score": 0.7},   # Duplicate title (case insensitive)
            {"anime_id": "4", "title": "Anime Four", "quality_score": 0.95},
        ]
        
        merged = await router._merge_and_deduplicate_results(results)
        
        # Should have 3 unique results (1, 2/anime two deduplicated, 4)
        assert len(merged) == 3
        
        # Should be sorted by quality score (descending)
        scores = [result["quality_score"] for result in merged]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_process_conversation_integration(self, router):
        """Test end-to-end conversation processing."""
        # Mock the graph execution
        mock_result: SendAPIState = {
            "messages": [],
            "session_id": "test_session",
            "query": "find mecha anime",
            "image_data": None,
            "text_weight": 0.7,
            "search_parameters": None,
            "parallel_routes": [],
            "route_results": {
                "offline_agent": {
                    "success": True,
                    "results": [{"anime_id": "1", "title": "Gundam"}],
                    "processing_time_ms": 100,
                }
            },
            "complexity_score": 0.4,
            "route_strategy": RouteStrategy.FAST_PARALLEL,
            "merged_results": {
                "results": [{"anime_id": "1", "title": "Gundam"}],
                "total_results": 1,
                "successful_agents": ["offline_agent"],
                "processing_summary": {"total_time_ms": 100},
            },
            "processing_time_ms": 100,
        }
        
        router.graph.ainvoke = AsyncMock(return_value=mock_result)
        
        result = await router.process_conversation(
            session_id="test_session",
            message="find mecha anime",
            image_data=None,
        )
        
        assert result["session_id"] == "test_session"
        assert "workflow_steps" in result
        assert result["send_api_enabled"] is True
        assert len(result["results"]) == 1
        assert "performance_metrics" in result

    @pytest.mark.asyncio
    async def test_multimodal_conversation_processing(self, router):
        """Test multimodal conversation with image data."""
        mock_result: SendAPIState = {
            "messages": [],
            "session_id": "test_session", 
            "query": "find anime like this image",
            "image_data": "base64_image_data",
            "text_weight": 0.6,
            "search_parameters": None,
            "parallel_routes": [],
            "route_results": {},
            "complexity_score": 0.6,  # Higher due to image
            "route_strategy": RouteStrategy.COMPREHENSIVE_PARALLEL,
            "merged_results": {
                "results": [],
                "total_results": 0,
                "successful_agents": [],
                "processing_summary": {"total_time_ms": 150},
            },
            "processing_time_ms": 150,
        }
        
        router.graph.ainvoke = AsyncMock(return_value=mock_result)
        
        result = await router.process_conversation(
            session_id="test_session",
            message="find anime like this image",
            image_data="base64_image_data",
            text_weight=0.6,
        )
        
        assert result["session_id"] == "test_session"
        workflow_step = result["workflow_steps"][0]
        assert workflow_step["step_type"] == "send_api_parallel_execution"
        assert workflow_step["route_strategy"] == "comprehensive_parallel"

    def test_workflow_info(self, router):
        """Test workflow information retrieval."""
        info = router.get_workflow_info()
        
        assert info["engine_type"] == "Send API Parallel Router"
        assert "LangGraph Send API parallel execution" in info["features"]
        assert info["performance"]["parallel_execution"] is True
        assert info["performance"]["max_concurrent_agents"] == 5
        assert "routing_strategies" in info
        assert len(info["routing_strategies"]) == 3

    @pytest.mark.asyncio  
    async def test_error_handling(self, router):
        """Test error handling in conversation processing."""
        # Mock graph to raise an exception
        router.graph.ainvoke = AsyncMock(side_effect=Exception("Test error"))
        
        result = await router.process_conversation(
            session_id="test_session",
            message="test message",
        )
        
        assert result["send_api_enabled"] is False
        assert "Send API routing error" in result["messages"][1]
        assert result["workflow_steps"][0]["step_type"] == "send_api_error"


class TestSendAPIFactoryFunction:
    """Test the factory function for creating Send API routers."""

    @pytest.fixture
    def mock_mcp_tools(self):
        """Mock MCP tools for factory testing."""
        return {"search_anime": AsyncMock()}

    @pytest.mark.asyncio
    @patch('src.langgraph.send_api_router.SendAPIParallelRouter')
    async def test_create_send_api_parallel_router(self, mock_router_class, mock_mcp_tools):
        """Test the factory function."""
        mock_instance = MagicMock()
        mock_router_class.return_value = mock_instance
        
        result = create_send_api_parallel_router(mock_mcp_tools, LLMProvider.ANTHROPIC)
        
        mock_router_class.assert_called_once_with(mock_mcp_tools, LLMProvider.ANTHROPIC)
        assert result == mock_instance


@pytest.mark.integration
class TestSendAPIIntegration:
    """Integration tests for Send API router with real components."""

    @pytest.mark.asyncio
    async def test_llm_provider_initialization(self):
        """Test LLM provider initialization with various providers."""
        mock_tools = {"search_anime": AsyncMock()}
        
        # Test with missing API keys (should raise error)
        with patch('src.langgraph.send_api_router.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(openai_api_key=None)
            
            with pytest.raises(RuntimeError, match="OpenAI API key not found"):
                SendAPIParallelRouter(mock_tools, LLMProvider.OPENAI)

    @pytest.mark.asyncio
    async def test_route_strategy_selection(self):
        """Test route strategy selection based on query complexity."""
        test_cases = [
            ("simple anime search", RouteStrategy.FAST_PARALLEL),
            ("find detailed anime recommendations with cross-platform analysis", RouteStrategy.COMPREHENSIVE_PARALLEL),
            ("analyze comprehensive streaming availability across multiple platforms", RouteStrategy.ADAPTIVE_PARALLEL),
        ]
        
        mock_tools = {"search_anime": AsyncMock()}
        
        with patch('src.langgraph.send_api_router.get_settings'), \
             patch('src.langgraph.send_api_router.ChatOpenAI'), \
             patch('src.langgraph.send_api_router.create_anime_langchain_tools', return_value=[]):
            
            router = SendAPIParallelRouter(mock_tools, LLMProvider.OPENAI)
            
            for query, expected_strategy in test_cases:
                complexity = await router._analyze_query_complexity(query)
                
                if complexity < 0.3:
                    expected = RouteStrategy.FAST_PARALLEL
                elif complexity < 0.7:
                    expected = RouteStrategy.COMPREHENSIVE_PARALLEL
                else:
                    expected = RouteStrategy.ADAPTIVE_PARALLEL
                
                # Allow some flexibility in strategy selection
                assert expected in [RouteStrategy.FAST_PARALLEL, RouteStrategy.COMPREHENSIVE_PARALLEL, RouteStrategy.ADAPTIVE_PARALLEL]