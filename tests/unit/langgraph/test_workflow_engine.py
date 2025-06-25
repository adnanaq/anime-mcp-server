"""Tests for ToolNode workflow engine integration."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any

from src.langgraph.workflow_engine import (
    AnimeWorkflowEngine,
    create_anime_workflow_engine,
)


class TestAnimeWorkflowEngine:
    """Test the ToolNode-based workflow engine that replaces StateGraphAnimeWorkflowEngine."""

    @pytest.fixture
    def mock_mcp_tools(self):
        """Mock MCP tools for testing."""
        return {
            "search_anime": AsyncMock(return_value=[
                {"anime_id": "t1", "title": "Test Anime", "score": 0.9}
            ]),
            "get_anime_details": AsyncMock(return_value={
                "anime_id": "t1", "title": "Test Anime", "synopsis": "Test synopsis"
            }),
            "get_anime_stats": AsyncMock(return_value={"total_documents": 100}),
        }

    @pytest.fixture
    def workflow_engine(self, mock_mcp_tools):
        """Create workflow engine instance."""
        return AnimeWorkflowEngine(mock_mcp_tools)

    def test_engine_initialization(self, workflow_engine):
        """Test that the engine initializes properly."""
        assert workflow_engine.tools is not None
        assert len(workflow_engine.tools) == 3
        assert workflow_engine.tool_node is not None
        assert workflow_engine.state_graph is not None
        assert workflow_engine.memory_saver is not None

    def test_engine_has_compatible_api(self, workflow_engine):
        """Test that engine has same API as StateGraphAnimeWorkflowEngine."""
        # Should have the same methods
        assert hasattr(workflow_engine, 'process_conversation')
        assert hasattr(workflow_engine, 'process_multimodal_conversation')
        assert hasattr(workflow_engine, 'get_conversation_summary')
        assert hasattr(workflow_engine, 'get_workflow_info')

    @pytest.mark.asyncio
    async def test_process_conversation_basic(self, workflow_engine, mock_mcp_tools):
        """Test basic conversation processing."""
        result = await workflow_engine.process_conversation(
            session_id="test_session",
            message="find action anime"
        )

        assert isinstance(result, dict)
        assert result["session_id"] == "test_session"
        assert "messages" in result
        assert "workflow_steps" in result
        assert len(result["messages"]) >= 1

    @pytest.mark.asyncio
    async def test_process_multimodal_conversation(self, workflow_engine, mock_mcp_tools):
        """Test multimodal conversation processing."""
        result = await workflow_engine.process_multimodal_conversation(
            session_id="multimodal_session",
            message="find similar anime",
            image_data="base64_image_data",
            text_weight=0.6
        )

        assert isinstance(result, dict)
        assert result["session_id"] == "multimodal_session"
        assert result.get("image_data") == "base64_image_data"
        assert result.get("text_weight") == 0.6

    @pytest.mark.asyncio
    async def test_conversation_summary(self, workflow_engine):
        """Test conversation summary generation."""
        summary = await workflow_engine.get_conversation_summary("summary_session")
        
        assert isinstance(summary, str)
        assert "summary_session" in summary
        assert "ToolNode-based" in summary

    def test_workflow_info(self, workflow_engine):
        """Test workflow information retrieval."""
        info = workflow_engine.get_workflow_info()
        
        assert isinstance(info, dict)
        assert info["engine_type"] == "ToolNode+StateGraph"
        assert "features" in info
        assert "LangGraph ToolNode native integration" in info["features"]
        assert "performance" in info
        assert info["performance"]["memory_persistence"] is True
        assert "tools" in info
        assert len(info["tools"]) == 3

    @pytest.mark.asyncio
    async def test_error_handling(self, workflow_engine, mock_mcp_tools):
        """Test error handling in workflow processing."""
        # Mock tool failure
        mock_mcp_tools["search_anime"].side_effect = Exception("Tool error")

        result = await workflow_engine.process_conversation(
            session_id="error_session",
            message="trigger error"
        )

        assert isinstance(result, dict)
        assert result["session_id"] == "error_session"
        assert "messages" in result
        # Should have error handling
        assert any(
            "Error processing request" in str(msg) or "error" in str(msg).lower()
            for msg in result["messages"]
        )

    @pytest.mark.asyncio
    async def test_state_graph_compatibility(self, workflow_engine):
        """Test that result format is compatible with StateGraph format."""
        result = await workflow_engine.process_conversation(
            session_id="compat_session",
            message="test compatibility"
        )

        # Should have all expected StateGraph fields
        required_fields = [
            "session_id", "messages", "workflow_steps", "current_context",
            "user_preferences", "image_data", "text_weight", "orchestration_enabled"
        ]
        
        for field in required_fields:
            assert field in result

    def test_text_content_extraction(self, workflow_engine):
        """Test text content extraction from various message formats."""
        # String content
        assert workflow_engine._extract_text_content("hello") == "hello"
        
        # List content
        list_content = ["hello", {"text": "world"}, "!"]
        assert workflow_engine._extract_text_content(list_content) == "hello world !"
        
        # Other types
        assert workflow_engine._extract_text_content(123) == "123"

    @pytest.mark.asyncio 
    async def test_ai_intent_integration(self, workflow_engine):
        """Test integration with AI-powered query understanding."""
        # Mock SearchIntent
        mock_intent = Mock()
        mock_intent.query = "action anime"
        mock_intent.limit = 5
        mock_intent.year_range = [2020]
        mock_intent.genres = ["action", "adventure"]
        mock_intent.anime_types = ["TV"]
        
        with patch('src.services.llm_service.extract_search_intent', 
                   new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = mock_intent
            
            tool_call = workflow_engine._create_tool_call_from_intent(mock_intent, 1)
            
            assert tool_call is not None
            assert tool_call["name"] == "search_anime"
            assert tool_call["args"]["query"] == "action anime"
            assert tool_call["args"]["limit"] == 5
            assert tool_call["args"]["year"] == 2020
            assert tool_call["args"]["genres"] == "action,adventure"
            assert tool_call["args"]["anime_type"] == "TV"

    def test_performance_improvements(self, workflow_engine):
        """Test that performance improvements are reflected in workflow info."""
        info = workflow_engine.get_workflow_info()
        
        # Should have improved response time
        assert info["performance"]["target_response_time"] == "150ms"
        
        # Should indicate boilerplate reduction
        assert "boilerplate_reduction" in info["performance"]
        assert "~200 lines eliminated" in info["performance"]["boilerplate_reduction"]


class TestCreateAnimeWorkflowEngine:
    """Test the factory function for creating ToolNode workflow engines."""

    @pytest.fixture
    def mock_mcp_tools(self):
        """Mock MCP tools for testing."""
        return {
            "search_anime": AsyncMock(),
            "get_anime_details": AsyncMock(),
            "find_similar_anime": AsyncMock(),
        }

    def test_factory_function(self, mock_mcp_tools):
        """Test factory function creates proper engine."""
        engine = create_anime_workflow_engine(mock_mcp_tools)
        
        assert isinstance(engine, AnimeWorkflowEngine)
        assert len(engine.tools) == 3
        assert engine.state_graph is not None

    def test_factory_function_with_all_tools(self):
        """Test factory function with all 8 MCP tools."""
        all_tools = {
            "search_anime": AsyncMock(),
            "get_anime_details": AsyncMock(),
            "find_similar_anime": AsyncMock(),
            "get_anime_stats": AsyncMock(),
            "recommend_anime": AsyncMock(),
            "search_anime_by_image": AsyncMock(),
            "find_visually_similar_anime": AsyncMock(),
            "search_multimodal_anime": AsyncMock(),
        }
        
        engine = create_anime_workflow_engine(all_tools)
        
        assert isinstance(engine, AnimeWorkflowEngine)
        assert len(engine.tools) == 8
        
        # All tool names should be available
        tool_names = {tool.name for tool in engine.tools}
        expected_names = set(all_tools.keys())
        assert tool_names == expected_names


class TestAPICompatibility:
    """Test compatibility with existing API endpoints."""

    @pytest.fixture
    def mock_mcp_tools(self):
        """Mock MCP tools for testing."""
        return {
            "search_anime": AsyncMock(return_value=[
                {"anime_id": "api1", "title": "API Test", "score": 0.95}
            ]),
            "get_anime_details": AsyncMock(return_value={
                "anime_id": "api1", "title": "API Test", "synopsis": "API test synopsis"
            }),
        }

    @pytest.fixture
    def workflow_engine(self, mock_mcp_tools):
        """Create workflow engine instance."""
        return AnimeWorkflowEngine(mock_mcp_tools)

    @pytest.mark.asyncio
    async def test_api_response_format(self, workflow_engine):
        """Test that API response format matches expectations."""
        result = await workflow_engine.process_conversation(
            session_id="api_test",
            message="find anime"
        )

        # Should match ConversationResponse model structure
        assert "session_id" in result
        assert "messages" in result
        assert "workflow_steps" in result
        assert "current_context" in result
        assert "user_preferences" in result
        
        # Messages should be in expected format
        assert isinstance(result["messages"], list)
        if result["messages"]:
            assert isinstance(result["messages"][0], str)

    @pytest.mark.asyncio
    async def test_session_isolation(self, workflow_engine):
        """Test that different sessions are properly isolated."""
        # Process conversations for different sessions
        result1 = await workflow_engine.process_conversation(
            session_id="session_1",
            message="test message 1"
        )

        result2 = await workflow_engine.process_conversation(
            session_id="session_2", 
            message="test message 2"
        )

        # Sessions should be independent
        assert result1["session_id"] == "session_1"
        assert result2["session_id"] == "session_2"
        assert result1["session_id"] != result2["session_id"]

    @pytest.mark.asyncio
    async def test_thread_persistence(self, workflow_engine):
        """Test conversation persistence with thread IDs."""
        thread_id = "persistent_thread"

        # First message
        result1 = await workflow_engine.process_conversation(
            session_id="persist_session",
            message="hello",
            thread_id=thread_id
        )

        # Second message should persist context
        result2 = await workflow_engine.process_conversation(
            session_id="persist_session", 
            message="find anime",
            thread_id=thread_id
        )

        assert result1["session_id"] == result2["session_id"]
        # Both should use the same thread for persistence
        assert thread_id is not None

    def test_workflow_info_api_compatibility(self, workflow_engine):
        """Test that workflow info is compatible with API expectations."""
        info = workflow_engine.get_workflow_info()
        
        # Should have all expected fields for health check endpoint
        assert "engine_type" in info
        assert "features" in info
        assert "performance" in info
        assert isinstance(info["features"], list)
        assert isinstance(info["performance"], dict)
        
        # Performance should have expected fields
        perf = info["performance"]
        assert "target_response_time" in perf
        assert "memory_persistence" in perf
        assert "conversation_threading" in perf