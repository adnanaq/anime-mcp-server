"""Unit tests for Multi-Agent Swarm Architecture.

This module tests the swarm agent coordination, handoff capabilities,
and specialized agent functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.langgraph.swarm_agents import (
    MultiAgentSwarm,
    AgentSpecialization,
    PlatformAgentType,
    EnhancementAgentType,
    OrchestrationAgentType,
    AgentDefinition,
    SwarmState,
    create_multi_agent_swarm,
)
from src.langgraph.react_agent_workflow import LLMProvider


class TestMultiAgentSwarm:
    """Test cases for Multi-Agent Swarm."""

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
    @patch('src.langgraph.swarm_agents.get_settings')
    @patch('src.langgraph.swarm_agents.ChatOpenAI')
    @patch('src.langgraph.swarm_agents.create_anime_langchain_tools')
    @patch('src.langgraph.swarm_agents.create_react_agent')
    @patch('src.langgraph.swarm_agents.create_handoff_tool')
    @patch('src.langgraph.swarm_agents.create_swarm')
    def swarm(self, mock_create_swarm, mock_handoff_tool, mock_react_agent, 
              mock_tools_creator, mock_chat_openai, mock_get_settings, 
              mock_mcp_tools, mock_settings):
        """Create a MultiAgentSwarm instance for testing."""
        mock_get_settings.return_value = mock_settings
        mock_chat_openai.return_value = MagicMock()
        mock_tools_creator.return_value = []
        
        # Mock react agent creation
        mock_agent = AsyncMock()
        mock_react_agent.return_value = mock_agent
        
        # Mock handoff tool creation
        mock_handoff_tool.return_value = MagicMock()
        
        # Mock swarm creation
        mock_swarm_instance = AsyncMock()
        mock_create_swarm.return_value = mock_swarm_instance
        
        swarm = MultiAgentSwarm(mock_mcp_tools, LLMProvider.OPENAI)
        
        return swarm

    def test_swarm_initialization(self, swarm):
        """Test Multi-Agent Swarm initialization."""
        assert swarm.mcp_tools is not None
        assert swarm.llm_provider == LLMProvider.OPENAI
        assert len(swarm.agent_definitions) == 10  # 5 + 3 + 2 agents
        
        # Check agent categories
        platform_agents = [d for d in swarm.agent_definitions.values() 
                          if d.specialization == AgentSpecialization.PLATFORM_AGENT]
        enhancement_agents = [d for d in swarm.agent_definitions.values() 
                             if d.specialization == AgentSpecialization.ENHANCEMENT_AGENT]
        orchestration_agents = [d for d in swarm.agent_definitions.values() 
                               if d.specialization == AgentSpecialization.ORCHESTRATION_AGENT]
        
        assert len(platform_agents) == 5
        assert len(enhancement_agents) == 3
        assert len(orchestration_agents) == 2

    def test_agent_definitions_structure(self, swarm):
        """Test agent definitions are properly structured."""
        definitions = swarm.agent_definitions
        
        # Test MAL agent definition
        mal_agent = definitions["mal_agent"]
        assert mal_agent.agent_name == "mal_agent"
        assert mal_agent.agent_type == PlatformAgentType.MAL_AGENT
        assert mal_agent.specialization == AgentSpecialization.PLATFORM_AGENT
        assert "search_anime" in mal_agent.tools
        assert "anilist_agent" in mal_agent.handoff_targets
        assert "MyAnimeList specialist" in mal_agent.system_prompt
        assert "target_time_ms" in mal_agent.performance_profile

    def test_platform_agent_definitions(self, swarm):
        """Test all platform agent definitions."""
        platform_agents = {
            "mal_agent": PlatformAgentType.MAL_AGENT,
            "anilist_agent": PlatformAgentType.ANILIST_AGENT,
            "jikan_agent": PlatformAgentType.JIKAN_AGENT,
            "offline_agent": PlatformAgentType.OFFLINE_AGENT,
            "kitsu_agent": PlatformAgentType.KITSU_AGENT,
        }
        
        for agent_name, expected_type in platform_agents.items():
            definition = swarm.agent_definitions[agent_name]
            assert definition.agent_type == expected_type
            assert definition.specialization == AgentSpecialization.PLATFORM_AGENT
            assert len(definition.tools) >= 1
            assert len(definition.handoff_targets) >= 1
            assert len(definition.system_prompt) > 50  # Meaningful prompts

    def test_enhancement_agent_definitions(self, swarm):
        """Test all enhancement agent definitions."""
        enhancement_agents = {
            "rating_correlation_agent": EnhancementAgentType.RATING_CORRELATION_AGENT,
            "streaming_availability_agent": EnhancementAgentType.STREAMING_AVAILABILITY_AGENT,
            "review_aggregation_agent": EnhancementAgentType.REVIEW_AGGREGATION_AGENT,
        }
        
        for agent_name, expected_type in enhancement_agents.items():
            definition = swarm.agent_definitions[agent_name]
            assert definition.agent_type == expected_type
            assert definition.specialization == AgentSpecialization.ENHANCEMENT_AGENT
            assert len(definition.tools) >= 1
            assert len(definition.handoff_targets) >= 1

    def test_orchestration_agent_definitions(self, swarm):
        """Test orchestration agent definitions."""
        orchestration_agents = {
            "query_analysis_agent": OrchestrationAgentType.QUERY_ANALYSIS_AGENT,
            "result_merger_agent": OrchestrationAgentType.RESULT_MERGER_AGENT,
        }
        
        for agent_name, expected_type in orchestration_agents.items():
            definition = swarm.agent_definitions[agent_name]
            assert definition.agent_type == expected_type
            assert definition.specialization == AgentSpecialization.ORCHESTRATION_AGENT

        # Result merger should be terminal (no handoffs)
        result_merger = swarm.agent_definitions["result_merger_agent"]
        assert len(result_merger.handoff_targets) == 0

    def test_handoff_relationships(self, swarm):
        """Test agent handoff relationship integrity."""
        definitions = swarm.agent_definitions
        
        # Collect all agent names
        all_agent_names = set(definitions.keys())
        
        # Verify handoff targets exist
        for agent_name, definition in definitions.items():
            for handoff_target in definition.handoff_targets:
                assert handoff_target in all_agent_names, f"{agent_name} tries to hand off to non-existent {handoff_target}"

    @pytest.mark.asyncio
    async def test_process_conversation_success(self, swarm):
        """Test successful conversation processing with swarm."""
        # Mock swarm execution
        mock_result = {
            "query": "find mecha anime",
            "active_agent": "mal_agent",
            "handoff_history": [
                {"from": "query_analysis_agent", "to": "mal_agent", "reason": "MAL specialization"}
            ],
            "results": [{"anime_id": "1", "title": "Gundam"}],
        }
        
        if swarm.swarm:
            swarm.swarm.ainvoke = AsyncMock(return_value=mock_result)
        
        result = await swarm.process_conversation(
            session_id="test_session",
            message="find mecha anime",
        )
        
        assert result["session_id"] == "test_session"
        assert "workflow_steps" in result
        assert result["swarm_enabled"] is True
        
        workflow_step = result["workflow_steps"][0]
        assert workflow_step["step_type"] == "multi_agent_swarm_execution"
        assert workflow_step["handoff_enabled"] is True

    @pytest.mark.asyncio
    async def test_process_conversation_fallback(self, swarm):
        """Test fallback when swarm is unavailable."""
        # Disable swarm
        swarm.swarm = None
        
        # Mock fallback agent
        mock_offline_agent = AsyncMock()
        mock_offline_agent.ainvoke = AsyncMock(return_value={"messages": []})
        swarm.agents["offline_agent"] = mock_offline_agent
        
        result = await swarm.process_conversation(
            session_id="test_session",
            message="find anime",
        )
        
        assert result["session_id"] == "test_session"
        assert result["swarm_enabled"] is False
        assert result["workflow_steps"][0]["step_type"] == "fallback_single_agent"

    @pytest.mark.asyncio
    async def test_process_conversation_error_handling(self, swarm):
        """Test error handling in conversation processing."""
        # Mock swarm to raise an exception
        if swarm.swarm:
            swarm.swarm.ainvoke = AsyncMock(side_effect=Exception("Test error"))
        
        result = await swarm.process_conversation(
            session_id="test_session",
            message="test message",
        )
        
        assert result["swarm_enabled"] is False
        assert "Multi-Agent Swarm error" in result["messages"][1]
        assert result["workflow_steps"][0]["step_type"] == "swarm_error"

    @pytest.mark.asyncio
    async def test_fallback_single_agent_processing(self, swarm):
        """Test fallback single agent processing."""
        # Mock offline agent
        mock_offline_agent = AsyncMock()
        mock_offline_agent.ainvoke = AsyncMock(return_value={"messages": []})
        swarm.agents["offline_agent"] = mock_offline_agent
        
        result = await swarm._fallback_single_agent_processing(
            session_id="test_session",
            message="find anime",
        )
        
        assert result["session_id"] == "test_session"
        assert result["swarm_enabled"] is False
        mock_offline_agent.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_no_agents_available(self, swarm):
        """Test fallback when no agents are available."""
        # Remove all agents
        swarm.agents["offline_agent"] = None
        
        result = await swarm._fallback_single_agent_processing(
            session_id="test_session",
            message="find anime",
        )
        
        assert "No agents available" in result["messages"][1]

    def test_convert_swarm_result_to_compatible_format(self, swarm):
        """Test swarm result conversion to compatible format."""
        mock_result = {
            "query": "find mecha anime",
            "active_agent": "mal_agent",
            "handoff_history": [
                {"from": "query_analysis_agent", "to": "mal_agent"}
            ],
            "results": [{"anime_id": "1", "title": "Gundam"}],
        }
        
        converted = swarm._convert_swarm_result_to_compatible_format(
            mock_result, "test_session"
        )
        
        assert converted["session_id"] == "test_session"
        assert converted["swarm_enabled"] is True
        assert "swarm_metrics" in converted
        assert converted["swarm_metrics"]["active_agent"] == "mal_agent"
        assert converted["swarm_metrics"]["handoff_count"] == 1
        assert len(converted["results"]) == 1

    def test_create_error_response(self, swarm):
        """Test error response creation."""
        error_response = swarm._create_error_response(
            "test_session", "test message", "test error"
        )
        
        assert error_response["session_id"] == "test_session"
        assert error_response["swarm_enabled"] is False
        assert "Multi-Agent Swarm error: test error" in error_response["messages"][1]
        assert error_response["workflow_steps"][0]["step_type"] == "swarm_error"

    def test_get_workflow_info(self, swarm):
        """Test workflow information retrieval."""
        info = swarm.get_workflow_info()
        
        assert info["engine_type"] == "Multi-Agent Swarm"
        assert "LangGraph Swarm pattern implementation" in info["features"]
        assert info["performance"]["agent_specialization"] is True
        assert info["performance"]["handoff_capabilities"] is True
        
        # Test agent architecture
        architecture = info["agent_architecture"]
        assert architecture["platform_agents"] == 5
        assert architecture["enhancement_agents"] == 3
        assert architecture["orchestration_agents"] == 2
        assert architecture["total_agents"] == 10
        assert architecture["handoff_enabled"] is True
        
        # Test agent definitions in info
        agent_defs = info["agent_definitions"]
        assert len(agent_defs) == 10
        assert "mal_agent" in agent_defs
        assert agent_defs["mal_agent"]["specialization"] == "platform_agent"


class TestAgentDefinitionDataclass:
    """Test AgentDefinition dataclass functionality."""

    def test_agent_definition_creation(self):
        """Test AgentDefinition creation and attributes."""
        definition = AgentDefinition(
            agent_name="test_agent",
            agent_type=PlatformAgentType.MAL_AGENT,
            specialization=AgentSpecialization.PLATFORM_AGENT,
            tools=["search_anime"],
            handoff_targets=["anilist_agent"],
            system_prompt="Test prompt",
            performance_profile={"target_time_ms": 200}
        )
        
        assert definition.agent_name == "test_agent"
        assert definition.agent_type == PlatformAgentType.MAL_AGENT
        assert definition.specialization == AgentSpecialization.PLATFORM_AGENT
        assert "search_anime" in definition.tools
        assert "anilist_agent" in definition.handoff_targets
        assert definition.system_prompt == "Test prompt"
        assert definition.performance_profile["target_time_ms"] == 200


class TestSwarmAgentEnums:
    """Test enum definitions for swarm agents."""

    def test_agent_specialization_enum(self):
        """Test AgentSpecialization enum values."""
        assert AgentSpecialization.PLATFORM_AGENT.value == "platform_agent"
        assert AgentSpecialization.ENHANCEMENT_AGENT.value == "enhancement_agent"
        assert AgentSpecialization.ORCHESTRATION_AGENT.value == "orchestration_agent"

    def test_platform_agent_type_enum(self):
        """Test PlatformAgentType enum values."""
        assert PlatformAgentType.MAL_AGENT.value == "mal_agent"
        assert PlatformAgentType.ANILIST_AGENT.value == "anilist_agent"
        assert PlatformAgentType.JIKAN_AGENT.value == "jikan_agent"
        assert PlatformAgentType.OFFLINE_AGENT.value == "offline_agent"
        assert PlatformAgentType.KITSU_AGENT.value == "kitsu_agent"

    def test_enhancement_agent_type_enum(self):
        """Test EnhancementAgentType enum values."""
        assert EnhancementAgentType.RATING_CORRELATION_AGENT.value == "rating_correlation_agent"
        assert EnhancementAgentType.STREAMING_AVAILABILITY_AGENT.value == "streaming_availability_agent"
        assert EnhancementAgentType.REVIEW_AGGREGATION_AGENT.value == "review_aggregation_agent"

    def test_orchestration_agent_type_enum(self):
        """Test OrchestrationAgentType enum values."""
        assert OrchestrationAgentType.QUERY_ANALYSIS_AGENT.value == "query_analysis_agent"
        assert OrchestrationAgentType.RESULT_MERGER_AGENT.value == "result_merger_agent"


class TestSwarmFactoryFunction:
    """Test the factory function for creating Multi-Agent Swarms."""

    @pytest.fixture
    def mock_mcp_tools(self):
        """Mock MCP tools for factory testing."""
        return {"search_anime": AsyncMock()}

    @pytest.mark.asyncio
    @patch('src.langgraph.swarm_agents.MultiAgentSwarm')
    async def test_create_multi_agent_swarm(self, mock_swarm_class, mock_mcp_tools):
        """Test the factory function."""
        mock_instance = MagicMock()
        mock_swarm_class.return_value = mock_instance
        
        result = create_multi_agent_swarm(mock_mcp_tools, LLMProvider.ANTHROPIC)
        
        mock_swarm_class.assert_called_once_with(mock_mcp_tools, LLMProvider.ANTHROPIC)
        assert result == mock_instance


@pytest.mark.integration
class TestSwarmIntegration:
    """Integration tests for Multi-Agent Swarm with real components."""

    @pytest.mark.asyncio
    async def test_llm_provider_initialization(self):
        """Test LLM provider initialization for swarm."""
        mock_tools = {"search_anime": AsyncMock()}
        
        # Test with missing API keys
        with patch('src.langgraph.swarm_agents.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(openai_api_key=None)
            
            with pytest.raises(RuntimeError, match="OpenAI API key not found"):
                MultiAgentSwarm(mock_tools, LLMProvider.OPENAI)

    @pytest.mark.asyncio 
    async def test_agent_handoff_chain_integrity(self):
        """Test that agent handoff chains are logically sound."""
        mock_tools = {"search_anime": AsyncMock()}
        
        with patch('src.langgraph.swarm_agents.get_settings'), \
             patch('src.langgraph.swarm_agents.ChatOpenAI'), \
             patch('src.langgraph.swarm_agents.create_anime_langchain_tools', return_value=[]), \
             patch('src.langgraph.swarm_agents.create_react_agent'), \
             patch('src.langgraph.swarm_agents.create_handoff_tool'), \
             patch('src.langgraph.swarm_agents.create_swarm'):
            
            swarm = MultiAgentSwarm(mock_tools, LLMProvider.OPENAI)
            
            # Test logical handoff chains
            # Query Analysis should hand off to platform agents
            query_analysis = swarm.agent_definitions["query_analysis_agent"]
            platform_targets = {"mal_agent", "anilist_agent", "offline_agent"}
            assert any(target in platform_targets for target in query_analysis.handoff_targets)
            
            # Platform agents should hand off to enhancement or orchestration agents
            mal_agent = swarm.agent_definitions["mal_agent"]
            valid_targets = {
                "anilist_agent", "rating_correlation_agent", "result_merger_agent"
            }
            for target in mal_agent.handoff_targets:
                assert target in valid_targets

    @pytest.mark.asyncio
    async def test_agent_specialization_coverage(self):
        """Test that agent specializations cover all necessary capabilities."""
        mock_tools = {"search_anime": AsyncMock(), "get_anime_details": AsyncMock()}
        
        with patch('src.langgraph.swarm_agents.get_settings'), \
             patch('src.langgraph.swarm_agents.ChatOpenAI'), \
             patch('src.langgraph.swarm_agents.create_anime_langchain_tools', return_value=[]), \
             patch('src.langgraph.swarm_agents.create_react_agent'), \
             patch('src.langgraph.swarm_agents.create_handoff_tool'), \
             patch('src.langgraph.swarm_agents.create_swarm'):
            
            swarm = MultiAgentSwarm(mock_tools, LLMProvider.OPENAI)
            
            # Verify we have coverage for major anime platforms
            platform_types = {def_.agent_type for def_ in swarm.agent_definitions.values() 
                             if def_.specialization == AgentSpecialization.PLATFORM_AGENT}
            
            required_platforms = {
                PlatformAgentType.MAL_AGENT,
                PlatformAgentType.ANILIST_AGENT,
                PlatformAgentType.OFFLINE_AGENT,
            }
            
            assert required_platforms.issubset(platform_types)
            
            # Verify we have enhancement capabilities
            enhancement_types = {def_.agent_type for def_ in swarm.agent_definitions.values() 
                                if def_.specialization == AgentSpecialization.ENHANCEMENT_AGENT}
            
            assert len(enhancement_types) >= 2  # At least rating and streaming
            
            # Verify we have orchestration capabilities
            orchestration_types = {def_.agent_type for def_ in swarm.agent_definitions.values() 
                                 if def_.specialization == AgentSpecialization.ORCHESTRATION_AGENT}
            
            required_orchestration = {
                OrchestrationAgentType.QUERY_ANALYSIS_AGENT,
                OrchestrationAgentType.RESULT_MERGER_AGENT,
            }
            
            assert required_orchestration.issubset(orchestration_types)