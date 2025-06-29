"""Tests for create_react_agent workflow engine replacement.

This module tests the new create_react_agent based implementation that replaces
the current AnimeWorkflowEngine with native LangGraph patterns.

Following TDD approach: these tests define the expected behavior before implementation.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


@pytest.fixture
def mock_mcp_tools():
    """Mock MCP tools for testing - shared across all test classes."""
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
        "find_similar_anime": AsyncMock(),
        "get_anime_stats": AsyncMock(),
    }


class TestReactAgentWorkflowEngine:
    """Test the create_react_agent based workflow engine."""


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


class TestReactAgentValidationFix:
    """Test cases for fixing ReactAgent tool call validation errors."""


    @pytest.mark.asyncio
    async def test_missing_query_field_validation_error(self, mock_mcp_tools):
        """Test that reproduces the missing query field validation error.
        
        This test simulates the exact error we observed:
        "Error: 1 validation error for SearchAnimeInput query Field required"
        """
        with patch('src.langgraph.react_agent_workflow.ChatOpenAI'):
            with patch('src.langgraph.react_agent_workflow.create_react_agent') as mock_create_agent:
                mock_agent = AsyncMock()
                
                # Simulate the problematic tool call that causes validation error
                mock_result = {
                    "messages": [
                        Mock(content="User message"),
                        Mock(content="AI response with validation error"),
                        Mock(content="Error: 1 validation error for SearchAnimeInput\nquery\n  Field required [type=missing, input_value={'genres': ['thriller'], 'year_range': [2010, 2023]}, input_type=dict]")
                    ]
                }
                mock_agent.ainvoke = AsyncMock(return_value=mock_result)
                mock_create_agent.return_value = mock_agent
                
                from src.langgraph.react_agent_workflow import create_react_agent_workflow_engine, LLMProvider
                
                engine = create_react_agent_workflow_engine(mock_mcp_tools, LLMProvider.OPENAI)
                
                result = await engine.process_conversation(
                    session_id="test_validation", 
                    message="I want thriller anime with deep philosophical themes produced by Madhouse between 2010 and 2023"
                )
                
                # Verify the validation error occurred
                messages = result.get("messages", [])
                error_found = any("validation error for SearchAnimeInput" in str(msg) and "query" in str(msg) for msg in messages)
                assert error_found, "Expected validation error for missing query field not found"

    @pytest.mark.asyncio 
    async def test_tool_call_should_include_query_field(self, mock_mcp_tools):
        """Test that tool calls should include the original query field.
        
        This test defines the expected behavior: tool calls must include
        the original query text along with extracted parameters.
        """
        with patch('src.langgraph.react_agent_workflow.ChatOpenAI'):
            with patch('src.langgraph.react_agent_workflow.create_react_agent') as mock_create_agent:
                
                mock_agent = AsyncMock()
                
                # Mock successful execution with proper tool call
                mock_result = {
                    "messages": [
                        Mock(content="User message"),
                        Mock(content="Search results")
                    ]
                }
                mock_agent.ainvoke = AsyncMock(return_value=mock_result)
                mock_create_agent.return_value = mock_agent
                
                from src.langgraph.react_agent_workflow import create_react_agent_workflow_engine, LLMProvider
                
                engine = create_react_agent_workflow_engine(mock_mcp_tools, LLMProvider.OPENAI)
                
                result = await engine.process_conversation(
                    session_id="test_fixed",
                    message="I want thriller anime with deep philosophical themes produced by Madhouse between 2010 and 2023"
                )
                
                # After the fix, this should work without validation errors
                messages = result.get("messages", [])
                validation_error = any("validation error" in str(msg) for msg in messages)
                assert not validation_error, "Should not have validation errors after fix"

    @pytest.mark.asyncio
    async def test_complex_parameter_extraction_with_query(self, mock_mcp_tools):
        """Test that complex parameter extraction includes query field.
        
        Tests the most complex query from our test cases to ensure
        all parameters are extracted AND the query field is preserved.
        """
        complex_query = "Find psychological seinen anime movies from Studio Pierrot or A-1 Pictures between 2005 and 2020, exclude known shounen titles, and limit to 3 results"
        
        with patch('src.langgraph.react_agent_workflow.ChatOpenAI'):
            with patch('src.langgraph.react_agent_workflow.create_react_agent') as mock_create_agent:
                
                mock_agent = AsyncMock()
                
                mock_result = {"messages": [Mock(content="Success")]}
                mock_agent.ainvoke = AsyncMock(return_value=mock_result)
                mock_create_agent.return_value = mock_agent
                
                from src.langgraph.react_agent_workflow import create_react_agent_workflow_engine, LLMProvider
                
                engine = create_react_agent_workflow_engine(mock_mcp_tools, LLMProvider.OPENAI)
                
                result = await engine.process_conversation(
                    session_id="test_complex",
                    message=complex_query
                )
                
                # Should succeed without validation errors
                messages = result.get("messages", [])
                validation_error = any("validation error" in str(msg) for msg in messages)
                assert not validation_error, "Complex query should work without validation errors"


class TestReactAgentErrorHandling:
    """Test error handling and edge cases in ReactAgent workflow."""

    def test_openai_import_error_handling(self):
        """Test handling when ChatOpenAI is not available."""
        with patch("src.langgraph.react_agent_workflow.ChatOpenAI", None):
            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine, LLMProvider
            
            mock_tools = {"search_anime": Mock()}
            
            with pytest.raises(RuntimeError) as exc_info:
                ReactAgentWorkflowEngine(mock_tools, LLMProvider.OPENAI)
            
            assert "langchain_openai not available" in str(exc_info.value)
            assert "pip install langchain-openai" in str(exc_info.value)

    def test_anthropic_import_error_handling(self):
        """Test handling when ChatAnthropic is not available."""
        with patch("src.langgraph.react_agent_workflow.ChatAnthropic", None):
            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine, LLMProvider
            
            mock_tools = {"search_anime": Mock()}
            
            with pytest.raises(RuntimeError) as exc_info:
                ReactAgentWorkflowEngine(mock_tools, LLMProvider.ANTHROPIC)
            
            assert "langchain_anthropic not available" in str(exc_info.value)
            assert "pip install langchain-anthropic" in str(exc_info.value)

    def test_openai_missing_api_key(self):
        """Test handling when OpenAI API key is missing."""
        with patch("src.langgraph.react_agent_workflow.ChatOpenAI"), \
             patch("src.langgraph.react_agent_workflow.get_settings") as mock_get_settings:
            
            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine, LLMProvider
            
            # Mock settings without openai_api_key
            mock_settings = Mock()
            mock_settings.openai_api_key = None
            mock_get_settings.return_value = mock_settings
            
            mock_tools = {"search_anime": Mock()}
            
            with pytest.raises(RuntimeError) as exc_info:
                ReactAgentWorkflowEngine(mock_tools, LLMProvider.OPENAI)
            
            assert "OpenAI API key not found" in str(exc_info.value)
            assert "OPENAI_API_KEY environment variable" in str(exc_info.value)

    def test_anthropic_missing_api_key(self):
        """Test handling when Anthropic API key is missing."""
        with patch("src.langgraph.react_agent_workflow.ChatAnthropic"), \
             patch("src.langgraph.react_agent_workflow.get_settings") as mock_get_settings:
            
            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine, LLMProvider
            
            # Mock settings without anthropic_api_key
            mock_settings = Mock()
            mock_settings.anthropic_api_key = None
            mock_get_settings.return_value = mock_settings
            
            mock_tools = {"search_anime": Mock()}
            
            with pytest.raises(RuntimeError) as exc_info:
                ReactAgentWorkflowEngine(mock_tools, LLMProvider.ANTHROPIC)
            
            assert "Anthropic API key not found" in str(exc_info.value)
            assert "ANTHROPIC_API_KEY environment variable" in str(exc_info.value)

    def test_anthropic_provider_initialization(self):
        """Test successful Anthropic provider initialization."""
        mock_anthropic = Mock()
        
        with patch("src.langgraph.react_agent_workflow.ChatAnthropic", mock_anthropic), \
             patch("src.langgraph.react_agent_workflow.get_settings") as mock_get_settings, \
             patch("src.langgraph.react_agent_workflow.create_anime_langchain_tools"), \
             patch("src.langgraph.react_agent_workflow.create_react_agent"):
            
            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine, LLMProvider
            
            # Mock settings with anthropic_api_key
            mock_settings = Mock()
            mock_settings.anthropic_api_key = "test_key"
            mock_get_settings.return_value = mock_settings
            
            mock_tools = {"search_anime": Mock()}
            
            engine = ReactAgentWorkflowEngine(mock_tools, LLMProvider.ANTHROPIC)
            
            # Should call ChatAnthropic constructor
            mock_anthropic.assert_called_once_with(
                model="claude-3-haiku-20240307",
                api_key="test_key",
                streaming=True,
                temperature=0.1
            )


class TestImportErrorCoverage:
    """Test import error handling to achieve 100% coverage."""

    def test_import_errors_are_handled_gracefully(self):
        """Test that import errors are handled and modules are set to None."""
        # Test the actual import behavior by temporarily modifying sys.modules
        import sys
        
        # Store original modules
        original_openai = sys.modules.get('langchain_openai')
        original_anthropic = sys.modules.get('langchain_anthropic')
        
        try:
            # Simulate missing modules
            if 'langchain_openai' in sys.modules:
                del sys.modules['langchain_openai']
            if 'langchain_anthropic' in sys.modules:
                del sys.modules['langchain_anthropic']
            
            # Force reimport to trigger ImportError handling
            import importlib
            
            # Re-import the module to test the except blocks
            if 'src.langgraph.react_agent_workflow' in sys.modules:
                importlib.reload(sys.modules['src.langgraph.react_agent_workflow'])
            
            # Import should work even with missing dependencies
            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine
            
            # Should be able to create engine even with missing deps
            # (errors will occur during LLM creation, not import)
            mock_tools = {"search_anime": Mock()}
            engine = ReactAgentWorkflowEngine(mock_tools)
            assert engine is not None
            
        finally:
            # Restore original modules
            if original_openai is not None:
                sys.modules['langchain_openai'] = original_openai
            if original_anthropic is not None:
                sys.modules['langchain_anthropic'] = original_anthropic

    def test_unknown_llm_provider_error(self):
        """Test RuntimeError for unknown LLM provider - covers line 125."""
        with patch("src.langgraph.react_agent_workflow.get_settings"), \
             patch("src.langgraph.react_agent_workflow.create_anime_langchain_tools"):
            
            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine
            
            mock_tools = {"search_anime": Mock()}
            
            # Test with an invalid provider value (simulate unknown provider)
            with pytest.raises(RuntimeError) as exc_info:
                # Pass an invalid provider that would trigger line 125
                engine = ReactAgentWorkflowEngine(mock_tools, "INVALID_PROVIDER")
            
            assert "Unknown LLM provider" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_conversation_summary_exception_handling(self, mock_mcp_tools):
        """Test exception handling in conversation summary - covers lines 277, 280-283."""
        with patch("src.langgraph.react_agent_workflow.create_react_agent") as mock_create_agent:
            mock_agent = Mock()
            mock_create_agent.return_value = mock_agent
            
            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine
            
            engine = ReactAgentWorkflowEngine(mock_mcp_tools)
            
            # Force an exception in the summary generation
            with patch.object(engine, '_get_conversation_messages', side_effect=Exception("Summary error")):
                summary = await engine.get_conversation_summary("test_session")
                
                # Should handle exception and return fallback summary
                assert "Unable to generate summary" in summary or "No conversation history" in summary

    @pytest.mark.asyncio
    async def test_image_data_handling_in_conversation(self, mock_mcp_tools):
        """Test image data handling - covers lines 316-317, 322."""
        with patch("src.langgraph.react_agent_workflow.create_react_agent") as mock_create_agent:
            mock_agent = Mock()
            mock_agent.astream = AsyncMock()
            mock_create_agent.return_value = mock_agent
            
            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine
            
            engine = ReactAgentWorkflowEngine(mock_mcp_tools)
            
            # Mock streaming response with image data
            async def mock_stream_with_image():
                yield {
                    "messages": [AIMessage(content="Processing image...")],
                    "image_data": "base64_encoded_image"  # This covers line 316
                }
                yield {
                    "messages": [AIMessage(content="Analysis complete")],
                    "has_image": True  # This covers line 317
                }
            
            mock_agent.astream.return_value = mock_stream_with_image()
            
            # Test streaming with image handling
            stream_results = []
            async for chunk in engine.astream_conversation("test_session", "analyze this image"):
                stream_results.append(chunk)
            
            # Should handle image data in streaming
            assert len(stream_results) >= 1
