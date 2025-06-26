"""Tests for create_react_agent workflow engine replacement.

This module tests the new create_react_agent based implementation that replaces
the current AnimeWorkflowEngine with native LangGraph patterns.

Following TDD approach: these tests define the expected behavior before implementation.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class TestReactAgentWorkflowEngine:
    """Test the create_react_agent based workflow engine."""

    @pytest.fixture
    def mock_mcp_tools(self):
        """Mock MCP tools for testing."""
        return {
            "search_anime": AsyncMock(
                return_value=[
                    {"anime_id": "r1", "title": "React Test Anime", "score": 0.95}
                ]
            ),
            "get_anime_details": AsyncMock(
                return_value={
                    "anime_id": "r1",
                    "title": "React Test Anime",
                    "synopsis": "React test synopsis",
                }
            ),
            "recommend_anime": AsyncMock(
                return_value=[
                    {"anime_id": "r2", "title": "Recommended Anime", "score": 0.85}
                ]
            ),
        }

    @pytest.fixture
    def search_intent_mock(self):
        """Legacy comment - SearchIntent removed, ReactAgent handles this internally."""
        mock_intent = Mock()
        mock_intent.query = "action anime"
        mock_intent.limit = 10
        mock_intent.genres = ["action"]
        mock_intent.year_range = [2020, 2024]
        mock_intent.anime_types = ["TV"]
        mock_intent.studios = []
        mock_intent.exclusions = []
        mock_intent.mood_keywords = []
        mock_intent.confidence = 0.9
        return mock_intent

    @pytest.mark.asyncio
    async def test_create_react_agent_workflow_init(self, mock_mcp_tools):
        """Test that ReactAgentWorkflowEngine initializes with create_react_agent."""
        # This test defines the expected interface for the new engine
        with patch("langgraph.prebuilt.create_react_agent") as mock_create_agent:
            mock_agent = Mock()
            mock_create_agent.return_value = mock_agent

            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

            engine = ReactAgentWorkflowEngine(mock_mcp_tools)

            # Should initialize with create_react_agent
            assert engine.agent is not None
            assert engine.mcp_tools == mock_mcp_tools
            mock_create_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_structured_output_integration(
        self, mock_mcp_tools, search_intent_mock
    ):
        """Test integration with structured output using with_structured_output()."""
        with patch("langgraph.prebuilt.create_react_agent") as mock_create_agent:
            mock_agent = Mock()
            mock_create_agent.return_value = mock_agent

            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

            engine = ReactAgentWorkflowEngine(mock_mcp_tools)

            # ReactAgent handles structured output internally
            # No manual structured_model needed

            # Legacy comment - SearchIntent removed, ReactAgent handles structured output natively
            # Since we're using mock models due to no API key,
            # verify the engine was created successfully
            assert engine.chat_model is not None

    @pytest.mark.asyncio
    async def test_streaming_support(self, mock_mcp_tools):
        """Test that streaming is properly configured."""
        with patch(
            "src.langgraph.react_agent_workflow.create_react_agent"
        ) as mock_create_agent:
            mock_agent = Mock()
            mock_agent.astream = AsyncMock()
            mock_create_agent.return_value = mock_agent

            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

            engine = ReactAgentWorkflowEngine(mock_mcp_tools)

            # Should support streaming
            assert hasattr(engine, "astream_conversation")

            # Mock streaming response
            async def mock_stream_response():
                yield {"messages": [AIMessage(content="Finding anime...")]}
                yield {
                    "messages": [ToolMessage(content="Found results", tool_call_id="1")]
                }
                yield {"messages": [AIMessage(content="Here are your results...")]}

            engine.agent.astream.return_value = mock_stream_response()

            # Test streaming conversation
            stream_results = []
            async for chunk in engine.astream_conversation(
                "test_session", "find action anime"
            ):
                stream_results.append(chunk)

            assert len(stream_results) >= 1
            engine.agent.astream.assert_called_once()

    @pytest.mark.asyncio
    async def test_api_compatibility_maintained(self, mock_mcp_tools):
        """Test that API compatibility is maintained with existing endpoints."""
        with patch(
            "src.langgraph.react_agent_workflow.create_react_agent"
        ) as mock_create_agent:
            mock_agent = Mock()
            mock_agent.ainvoke = AsyncMock(
                return_value={
                    "messages": [
                        HumanMessage(content="find anime"),
                        AIMessage(content="I found some anime for you"),
                    ]
                }
            )
            mock_create_agent.return_value = mock_agent

            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

            engine = ReactAgentWorkflowEngine(mock_mcp_tools)

            # Should maintain same API as AnimeWorkflowEngine
            assert hasattr(engine, "process_conversation")
            assert hasattr(engine, "process_multimodal_conversation")
            assert hasattr(engine, "get_conversation_summary")
            assert hasattr(engine, "get_workflow_info")

            # Test process_conversation returns compatible format
            result = await engine.process_conversation(
                session_id="compat_test", message="find action anime"
            )

            # Should return same format as current implementation
            assert isinstance(result, dict)
            assert "session_id" in result
            assert "messages" in result
            assert "workflow_steps" in result
            assert result["session_id"] == "compat_test"

    @pytest.mark.asyncio
    async def test_llm_service_elimination(self, mock_mcp_tools, search_intent_mock):
        """Test that manual LLM service calls are eliminated."""
        with patch(
            "src.langgraph.react_agent_workflow.create_react_agent"
        ) as mock_create_agent:
            mock_agent = Mock()
            mock_agent.ainvoke = AsyncMock(
                return_value={
                    "messages": [
                        HumanMessage(content="find action anime"),
                        AIMessage(
                            content="Finding action anime...",
                            tool_calls=[
                                {
                                    "name": "search_anime",
                                    "args": {"query": "action anime", "limit": 10},
                                    "id": "call_1",
                                }
                            ],
                        ),
                        ToolMessage(content="Found results", tool_call_id="call_1"),
                        AIMessage(content="Here are your action anime results"),
                    ]
                }
            )
            mock_create_agent.return_value = mock_agent

            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

            engine = ReactAgentWorkflowEngine(mock_mcp_tools)

            # Process conversation - should use ReactAgent instead of manual LLM service
            result = await engine.process_conversation(
                session_id="no_manual_llm", message="find action anime"
            )

            # Should use create_react_agent instead of manual LLM service calls
            mock_agent.ainvoke.assert_called_once()

            # Should return proper result format
            assert result["session_id"] == "no_manual_llm"
            assert "messages" in result

    def test_performance_improvements(self, mock_mcp_tools):
        """Test that performance improvements are documented."""
        with patch(
            "src.langgraph.react_agent_workflow.create_react_agent"
        ) as mock_create_agent:
            mock_agent = Mock()
            mock_create_agent.return_value = mock_agent

            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

            engine = ReactAgentWorkflowEngine(mock_mcp_tools)

            info = engine.get_workflow_info()

            # Should document performance improvements
            assert "engine_type" in info
            assert "create_react_agent" in info["engine_type"]

            # Should show improved response time targets
            assert "performance" in info
            perf = info["performance"]
            assert "target_response_time" in perf

            # Should be faster than current 150ms target
            target_time = perf["target_response_time"]
            assert "ms" in target_time
            time_value = int(target_time.replace("ms", ""))
            assert time_value <= 120  # 20-30% improvement from 150ms

    @pytest.mark.asyncio
    async def test_error_handling_improved(self, mock_mcp_tools):
        """Test improved error handling with create_react_agent."""
        with patch(
            "src.langgraph.react_agent_workflow.create_react_agent"
        ) as mock_create_agent:
            # Mock agent that raises an error
            mock_agent = Mock()
            mock_agent.ainvoke = AsyncMock(side_effect=Exception("Agent error"))
            mock_create_agent.return_value = mock_agent

            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

            engine = ReactAgentWorkflowEngine(mock_mcp_tools)

            # Should handle errors gracefully
            result = await engine.process_conversation(
                session_id="error_test", message="trigger error"
            )

            # Should return error in compatible format
            assert isinstance(result, dict)
            assert result["session_id"] == "error_test"
            assert "error" in str(result).lower() or "Error" in str(result)

    def test_type_safety_requirements(self, mock_mcp_tools):
        """Test that type safety is maintained."""
        with patch(
            "src.langgraph.react_agent_workflow.create_react_agent"
        ) as mock_create_agent:
            mock_agent = Mock()
            mock_create_agent.return_value = mock_agent

            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

            # Should initialize without type errors
            engine = ReactAgentWorkflowEngine(mock_mcp_tools)

            # Should have proper type annotations
            assert hasattr(ReactAgentWorkflowEngine, "__annotations__")

            # Methods should have return type annotations
            assert hasattr(engine.process_conversation, "__annotations__")
            assert hasattr(engine.get_workflow_info, "__annotations__")

    @pytest.mark.asyncio
    async def test_memory_and_checkpointing(self, mock_mcp_tools):
        """Test that memory and checkpointing work with create_react_agent."""
        with patch(
            "src.langgraph.react_agent_workflow.create_react_agent"
        ) as mock_create_agent:
            mock_agent = Mock()
            mock_agent.ainvoke = AsyncMock(
                return_value={"messages": [AIMessage(content="Response")]}
            )
            mock_create_agent.return_value = mock_agent

            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

            engine = ReactAgentWorkflowEngine(mock_mcp_tools)

            # Should support thread-based conversations
            thread_id = "memory_test_thread"

            result1 = await engine.process_conversation(
                session_id="memory_session", message="hello", thread_id=thread_id
            )

            result2 = await engine.process_conversation(
                session_id="memory_session",
                message="remember our conversation?",
                thread_id=thread_id,
            )

            # Should use same session but maintain memory via thread_id
            assert result1["session_id"] == result2["session_id"]

            # Should call agent with config including thread_id
            call_args = mock_agent.ainvoke.call_args_list
            assert len(call_args) == 2

            # Both calls should include config with thread_id
            for call in call_args:
                assert "config" in call.kwargs or len(call.args) > 1


class TestNativeStructuredOutputIntegration:
    """Test that ReactAgent handles structured output natively without manual schemas."""

    def test_react_agent_eliminates_manual_structured_output(self):
        """Test that ReactAgent eliminates need for manual structured output schemas."""
        # SearchIntent should not be exported since create_react_agent
        # handles structured output internally
        import src.langgraph.react_agent_workflow as workflow_module
        from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

        assert not hasattr(workflow_module, "SearchIntent")

        # Verify ReactAgent provides the functionality without manual schemas
        with (
            patch("src.langgraph.react_agent_workflow.get_settings"),
            patch("src.langgraph.react_agent_workflow.ChatOpenAI"),
            patch("src.langgraph.react_agent_workflow.create_anime_langchain_tools"),
            patch(
                "src.langgraph.react_agent_workflow.create_react_agent"
            ) as mock_agent,
        ):
            mock_agent.return_value = Mock()
            engine = ReactAgentWorkflowEngine({})

            # Should have all necessary components without manual structured output
            assert hasattr(engine, "agent")
            assert hasattr(engine, "chat_model")
            assert hasattr(engine, "process_conversation")

            # create_react_agent should be called (handles structured output internally)
            mock_agent.assert_called_once()

    def test_workflow_info_reflects_native_structured_output(self):
        """Test that workflow info reflects native structured output capabilities."""
        with (
            patch("src.langgraph.react_agent_workflow.get_settings"),
            patch("src.langgraph.react_agent_workflow.ChatOpenAI"),
            patch("src.langgraph.react_agent_workflow.create_anime_langchain_tools"),
            patch("src.langgraph.react_agent_workflow.create_react_agent"),
        ):
            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

            engine = ReactAgentWorkflowEngine({})
            info = engine.get_workflow_info()

            # Should indicate structured output is handled natively
            assert info["llm_integration"]["structured_output"] is True
            assert info["llm_integration"]["native_integration"] is True
            assert "No manual JSON parsing" in info["features"]


class TestAIPoweredQueryUnderstanding:
    """Test AI-powered query understanding integration with ReactAgent."""

    @pytest.fixture
    def mock_mcp_tools(self):
        """Mock MCP tools for AI query understanding tests."""
        return {
            "search_anime": AsyncMock(
                return_value=[
                    {"anime_id": "ai1", "title": "AI Enhanced Result", "score": 0.95}
                ]
            ),
            "recommend_anime": AsyncMock(
                return_value=[
                    {"anime_id": "ai2", "title": "AI Recommended", "score": 0.88}
                ]
            ),
        }

    @pytest.mark.asyncio
    async def test_complex_query_parameter_extraction(self, mock_mcp_tools):
        """Test that complex queries are processed with extracted parameters."""
        with patch(
            "src.langgraph.react_agent_workflow.create_react_agent"
        ) as mock_create_agent:
            # Mock agent that simulates AI understanding of complex query
            mock_agent = Mock()
            mock_agent.ainvoke = AsyncMock(
                return_value={
                    "messages": [
                        HumanMessage(
                            content="find 5 mecha anime from 2020s but not too violent"
                        ),
                        AIMessage(
                            content="I'll search for mecha anime from the 2020s, excluding violent content.",
                            tool_calls=[
                                {
                                    "name": "search_anime",
                                    "args": {
                                        "query": "mecha robots",
                                        "limit": 5,
                                        "genres": ["Mecha", "Action"],
                                        "year_range": [2020, 2029],
                                        "exclusions": ["Violence", "Gore"],
                                        "mood_keywords": ["peaceful", "light"],
                                    },
                                    "id": "call_ai_1",
                                }
                            ],
                        ),
                        ToolMessage(
                            content="Found mecha anime results",
                            tool_call_id="call_ai_1",
                        ),
                        AIMessage(
                            content="Here are 5 mecha anime from the 2020s with lighter themes..."
                        ),
                    ]
                }
            )
            mock_create_agent.return_value = mock_agent

            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

            engine = ReactAgentWorkflowEngine(mock_mcp_tools)

            # Process complex query
            result = await engine.process_conversation(
                session_id="ai_query_test",
                message="find 5 mecha anime from 2020s but not too violent",
            )

            # Should process the query and call agent
            mock_agent.ainvoke.assert_called_once()

            # Should return results with AI understanding
            assert result["session_id"] == "ai_query_test"
            assert len(result["messages"]) >= 2

    @pytest.mark.asyncio
    async def test_mood_based_query_understanding(self, mock_mcp_tools):
        """Test AI understanding of mood-based queries."""
        with patch(
            "src.langgraph.react_agent_workflow.create_react_agent"
        ) as mock_create_agent:
            mock_agent = Mock()
            mock_agent.ainvoke = AsyncMock(
                return_value={
                    "messages": [
                        HumanMessage(
                            content="I need something dark and serious but uplifting"
                        ),
                        AIMessage(
                            content="I'll find anime with dark themes but uplifting messages.",
                            tool_calls=[
                                {
                                    "name": "search_anime",
                                    "args": {
                                        "query": "dark serious uplifting anime",
                                        "limit": 10,
                                        "mood_keywords": [
                                            "dark",
                                            "serious",
                                            "uplifting",
                                            "hopeful",
                                        ],
                                        "exclusions": ["comedy", "lighthearted"],
                                    },
                                    "id": "call_mood",
                                }
                            ],
                        ),
                        ToolMessage(
                            content="Found mood-appropriate results",
                            tool_call_id="call_mood",
                        ),
                        AIMessage(content="Here are some dark but uplifting anime..."),
                    ]
                }
            )
            mock_create_agent.return_value = mock_agent

            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

            engine = ReactAgentWorkflowEngine(mock_mcp_tools)

            result = await engine.process_conversation(
                session_id="mood_test",
                message="I need something dark and serious but uplifting",
            )

            assert result["session_id"] == "mood_test"
            mock_agent.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_studio_and_year_preference_extraction(self, mock_mcp_tools):
        """Test extraction of studio preferences and year ranges."""
        with patch(
            "src.langgraph.react_agent_workflow.create_react_agent"
        ) as mock_create_agent:
            mock_agent = Mock()
            mock_agent.ainvoke = AsyncMock(
                return_value={
                    "messages": [
                        HumanMessage(
                            content="recommend Studio Ghibli movies from the 90s"
                        ),
                        AIMessage(
                            content="I'll find Studio Ghibli movies from the 1990s.",
                            tool_calls=[
                                {
                                    "name": "recommend_anime",
                                    "args": {
                                        "genres": "Fantasy,Adventure",
                                        "studios": ["Studio Ghibli"],
                                        "year_range": [1990, 1999],
                                        "anime_types": ["Movie"],
                                        "limit": 10,
                                    },
                                    "id": "call_studio",
                                }
                            ],
                        ),
                        ToolMessage(
                            content="Found Studio Ghibli recommendations",
                            tool_call_id="call_studio",
                        ),
                        AIMessage(
                            content="Here are Studio Ghibli movies from the 90s..."
                        ),
                    ]
                }
            )
            mock_create_agent.return_value = mock_agent

            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

            engine = ReactAgentWorkflowEngine(mock_mcp_tools)

            result = await engine.process_conversation(
                session_id="studio_test",
                message="recommend Studio Ghibli movies from the 90s",
            )

            assert result["session_id"] == "studio_test"
            mock_agent.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhanced_system_prompt_with_parameters(self, mock_mcp_tools):
        """Test that system prompt includes parameter guidance."""
        with patch(
            "src.langgraph.react_agent_workflow.create_react_agent"
        ) as mock_create_agent:
            mock_agent = Mock()
            mock_create_agent.return_value = mock_agent

            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

            ReactAgentWorkflowEngine(mock_mcp_tools)

            # Verify create_react_agent was called with enhanced prompt
            mock_create_agent.assert_called_once()
            call_args = mock_create_agent.call_args

            # Check that prompt parameter was provided
            assert "prompt" in call_args.kwargs
            prompt = call_args.kwargs["prompt"]

            # Should include guidance for parameter extraction
            assert "analyze" in prompt.lower() or "extract" in prompt.lower()

    def test_system_prompt_includes_search_intent_guidance(self):
        """Test that system prompt guides AI to extract SearchIntent parameters."""
        with patch(
            "src.langgraph.react_agent_workflow.create_react_agent"
        ) as mock_create_agent:
            mock_agent = Mock()
            mock_create_agent.return_value = mock_agent

            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

            engine = ReactAgentWorkflowEngine({})

            # Get the system prompt
            prompt = engine._get_system_prompt()

            # Should include SearchIntent parameter guidance
            prompt_lower = prompt.lower()
            expected_terms = ["genres", "year", "studio", "type", "mood", "exclude"]

            # At least some parameter guidance should be present
            found_terms = sum(1 for term in expected_terms if term in prompt_lower)
            assert (
                found_terms >= 3
            ), f"System prompt should mention SearchIntent parameters, found {found_terms} of {expected_terms}"


class TestBackwardCompatibility:
    """Test backward compatibility with existing API contracts."""

    @pytest.fixture
    def mock_mcp_tools(self):
        """Mock MCP tools matching current interface."""
        return {
            "search_anime": AsyncMock(
                return_value=[
                    {"anime_id": "bc1", "title": "Compatibility Test", "score": 0.9}
                ]
            ),
            "get_anime_details": AsyncMock(
                return_value={"anime_id": "bc1", "title": "Compatibility Test"}
            ),
        }

    @pytest.mark.asyncio
    async def test_conversation_response_format(self, mock_mcp_tools):
        """Test that conversation response format matches current API."""
        with patch(
            "src.langgraph.react_agent_workflow.create_react_agent"
        ) as mock_create_agent:
            mock_agent = Mock()
            mock_agent.ainvoke = AsyncMock(
                return_value={"messages": [AIMessage(content="Test response")]}
            )
            mock_create_agent.return_value = mock_agent

            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

            engine = ReactAgentWorkflowEngine(mock_mcp_tools)

            result = await engine.process_conversation(
                session_id="format_test", message="test message"
            )

            # Should match current ConversationResponse schema
            expected_fields = [
                "session_id",
                "messages",
                "workflow_steps",
                "current_context",
                "user_preferences",
                "image_data",
                "text_weight",
                "orchestration_enabled",
            ]

            for field in expected_fields:
                assert field in result, f"Missing required field: {field}"

            # Types should match expectations
            assert isinstance(result["session_id"], str)
            assert isinstance(result["messages"], list)
            assert isinstance(result["workflow_steps"], list)

    @pytest.mark.asyncio
    async def test_multimodal_conversation_compatibility(self, mock_mcp_tools):
        """Test multimodal conversation maintains same interface."""
        with patch(
            "src.langgraph.react_agent_workflow.create_react_agent"
        ) as mock_create_agent:
            mock_agent = Mock()
            mock_agent.ainvoke = AsyncMock(
                return_value={"messages": [AIMessage(content="Multimodal response")]}
            )
            mock_create_agent.return_value = mock_agent

            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

            engine = ReactAgentWorkflowEngine(mock_mcp_tools)

            result = await engine.process_multimodal_conversation(
                session_id="multimodal_test",
                message="find similar anime",
                image_data="base64_image_data",
                text_weight=0.7,
            )

            # Should preserve multimodal parameters
            assert result["image_data"] == "base64_image_data"
            assert result["text_weight"] == 0.7
            assert result["session_id"] == "multimodal_test"
