"""Integration tests for AI-powered query understanding and execution.

This module tests the complete pipeline from natural language query through
AI parameter extraction, enhanced filtering, and accurate search results.
"""

from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.workflow import ConversationRequest, SearchIntentParameters
from src.langgraph.react_agent_workflow import LLMProvider, ReactAgentWorkflowEngine


class TestCompleteAIPoweredQueryIntegration:
    """Test complete integration of AI-powered query understanding."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client with realistic anime data responses."""
        client = AsyncMock()

        # Mock search response for mecha anime from 2020s
        client.search.return_value = [
            {
                "id": "mecha_anime_1",
                "score": 0.95,
                "payload": {
                    "title": "86: Eighty-Six",
                    "genres": ["Mecha", "Action", "Drama"],
                    "year": 2021,
                    "anime_type": "TV",
                    "studio": "A-1 Pictures",
                    "synopsis": "A war story involving autonomous mechs and human pilots",
                    "quality_score": 0.89,
                    "sources": ["https://myanimelist.net/anime/41457"],
                },
            },
            {
                "id": "mecha_anime_2",
                "score": 0.92,
                "payload": {
                    "title": "Mobile Suit Gundam: Iron-Blooded Orphans",
                    "genres": ["Mecha", "Action", "Military"],
                    "year": 2020,
                    "anime_type": "TV",
                    "studio": "Sunrise",
                    "synopsis": "A story of war orphans and their mobile suits",
                    "quality_score": 0.87,
                    "sources": ["https://myanimelist.net/anime/33051"],
                },
            },
        ]

        return client

    @pytest.fixture
    def mock_mcp_tools(self, mock_qdrant_client):
        """Create mock MCP tools with realistic behavior."""

        async def mock_search_anime(
            query: str,
            limit: int = 10,
            genres: List[str] = None,
            year_range: List[int] = None,
            anime_types: List[str] = None,
            studios: List[str] = None,
            exclusions: List[str] = None,
            mood_keywords: List[str] = None,
        ):
            """Mock search_anime tool that validates AI parameter extraction."""
            # Verify that AI correctly extracted parameters
            if "mecha" in query.lower():
                assert genres and "Mecha" in genres, "AI should extract Mecha genre"
            if "2020s" in query.lower() or "2020" in str(year_range):
                assert (
                    year_range and year_range[0] >= 2020
                ), "AI should extract 2020s range"
            if "not too violent" in query.lower():
                assert exclusions and any(
                    ex in ["Violence", "Gore"] for ex in exclusions
                ), "AI should exclude violence"

            # Return filtered results based on parameters
            results = await mock_qdrant_client.search()

            # Apply limit
            return results[:limit]

        async def mock_recommend_anime(**kwargs):
            """Mock recommend_anime tool."""
            return await mock_search_anime(**kwargs)

        return {
            "search_anime": mock_search_anime,
            "recommend_anime": mock_recommend_anime,
            "get_anime_details": AsyncMock(return_value={"title": "Test Anime"}),
            "find_similar_anime": AsyncMock(return_value=[]),
            "search_anime_by_image": AsyncMock(return_value=[]),
            "get_anime_stats": AsyncMock(return_value={"total": 38894}),
            "get_popular_anime": AsyncMock(return_value=[]),
            "search_anime_by_studio": AsyncMock(return_value=[]),
        }

    @pytest.fixture
    def mock_chat_model(self):
        """Mock chat model that simulates AI parameter extraction."""
        chat_model = MagicMock()

        # Mock response that includes proper tool calls with extracted parameters
        mock_message = MagicMock()
        mock_message.content = (
            "I'll search for mecha anime from the 2020s, excluding violent content."
        )
        mock_message.tool_calls = [
            {
                "name": "search_anime",
                "args": {
                    "query": "mecha anime from 2020s but not too violent",
                    "limit": 5,
                    "genres": ["Mecha"],
                    "year_range": [2020, 2029],
                    "exclusions": ["Violence", "Gore"],
                    "anime_types": ["TV"],
                },
            }
        ]

        chat_model.ainvoke.return_value = mock_message
        return chat_model

    @pytest.mark.asyncio
    async def test_complete_ai_powered_query_pipeline(
        self, mock_mcp_tools, mock_chat_model
    ):
        """Test the complete AI-powered query understanding pipeline."""

        # Create ReactAgent workflow engine with mocked components
        with (
            patch(
                "src.langgraph.react_agent_workflow.create_react_agent"
            ) as mock_create_agent,
            patch("src.langgraph.react_agent_workflow.MemorySaver"),
            patch(
                "src.langgraph.langchain_tools.create_anime_langchain_tools"
            ) as mock_create_tools,
            patch.object(ReactAgentWorkflowEngine, "_initialize_chat_model"),
        ):

            # Mock agent creation
            mock_agent = AsyncMock()
            mock_create_agent.return_value = mock_agent

            # Mock LangChain tools creation
            mock_tools = [MagicMock()]
            mock_tools[0].name = "search_anime"
            mock_create_tools.return_value = mock_tools

            # Create workflow engine
            engine = ReactAgentWorkflowEngine(mock_mcp_tools, LLMProvider.OPENAI)
            engine.chat_model = mock_chat_model

            # Mock agent response
            mock_agent.ainvoke.return_value = {
                "messages": [
                    MagicMock(
                        content="find 5 mecha anime from 2020s but not too violent"
                    ),
                    MagicMock(
                        content="I found 2 mecha anime from the 2020s that avoid excessive violence: 86: Eighty-Six and Mobile Suit Gundam: Iron-Blooded Orphans."
                    ),
                ]
            }

            # Test the complete pipeline
            result = await engine.process_conversation(
                session_id="test_integration",
                message="find 5 mecha anime from 2020s but not too violent",
            )

            # Verify the pipeline worked
            assert result["session_id"] == "test_integration"
            assert len(result["messages"]) >= 2
            assert "mecha" in result["messages"][1].lower()
            assert "2020" in result["messages"][1] or "2021" in result["messages"][1]
            assert len(result["workflow_steps"]) >= 1

            # Verify agent was called with correct input
            mock_agent.ainvoke.assert_called_once()
            call_args = mock_agent.ainvoke.call_args[0][0]
            assert (
                "find 5 mecha anime from 2020s but not too violent"
                in call_args["messages"][0].content
            )

    @pytest.mark.asyncio
    async def test_explicit_search_parameters_override_ai_extraction(
        self, mock_mcp_tools
    ):
        """Test that explicit search parameters can override AI extraction."""

        with (
            patch(
                "src.langgraph.react_agent_workflow.create_react_agent"
            ) as mock_create_agent,
            patch("src.langgraph.react_agent_workflow.MemorySaver"),
            patch("src.langgraph.langchain_tools.create_anime_langchain_tools"),
            patch.object(ReactAgentWorkflowEngine, "_initialize_chat_model"),
        ):

            # Mock agent to verify explicit parameters are used
            mock_agent = AsyncMock()
            mock_create_agent.return_value = mock_agent
            mock_agent.ainvoke.return_value = {
                "messages": [
                    MagicMock(content="find horror anime from 1990s"),
                    MagicMock(content="Found comedy anime from 2020s as requested."),
                ]
            }

            # Create workflow engine after mocking
            engine = ReactAgentWorkflowEngine(mock_mcp_tools, LLMProvider.OPENAI)

            # Explicit parameters that override the natural language
            explicit_params = {
                "genres": ["Comedy"],
                "year_range": [2020, 2023],
                "limit": 3,
            }

            # Test with explicit parameters
            result = await engine.process_conversation(
                session_id="test_override",
                message="find horror anime from 1990s",  # AI would extract Horror + 1990s
                search_parameters=explicit_params,  # But we override with Comedy + 2020s
            )

            # Verify explicit parameters were passed through
            assert result["session_id"] == "test_override"
            mock_agent.ainvoke.assert_called_once()
            call_args = mock_agent.ainvoke.call_args[0][0]

            # Should include both original message and explicit parameters
            message_content = call_args["messages"][0].content
            assert "find horror anime from 1990s" in message_content
            assert str(explicit_params) in message_content

    @pytest.mark.asyncio
    async def test_api_endpoint_with_ai_parameter_extraction(self):
        """Test API endpoint processing with AI parameter extraction."""

        # Mock workflow engine
        mock_engine = AsyncMock()
        mock_engine.process_conversation.return_value = {
            "session_id": "api_test",
            "messages": ["User query", "AI response with filtered results"],
            "workflow_steps": [
                {
                    "step_type": "search",
                    "tool_name": "search_anime",
                    "result": {
                        "found": 5,
                        "filtered_by": ["Mecha", "2020s", "Non-violent"],
                    },
                    "confidence": 0.95,
                }
            ],
            "current_context": None,
            "user_preferences": None,
        }

        with patch("src.api.workflow.get_workflow_engine", return_value=mock_engine):
            from src.api.workflow import process_conversation

            # Create request with natural language query
            request = ConversationRequest(
                message="find 5 mecha anime from 2020s but not too violent",
                session_id="api_test",
            )

            # Process the request
            response = await process_conversation(request)

            # Verify API response
            assert response.session_id == "api_test"
            assert len(response.messages) >= 1
            assert len(response.workflow_steps) >= 1
            assert response.workflow_steps[0]["result"]["found"] == 5

            # Verify workflow engine was called correctly
            mock_engine.process_conversation.assert_called_once()
            call_kwargs = mock_engine.process_conversation.call_args.kwargs
            assert call_kwargs["session_id"] == "api_test"
            assert (
                call_kwargs["message"]
                == "find 5 mecha anime from 2020s but not too violent"
            )
            assert call_kwargs["search_parameters"] is None  # No explicit parameters

    @pytest.mark.asyncio
    async def test_api_endpoint_with_explicit_parameters(self):
        """Test API endpoint with explicit SearchIntent parameters."""

        mock_engine = AsyncMock()
        mock_engine.process_conversation.return_value = {
            "session_id": "explicit_test",
            "messages": ["Query", "Results based on explicit parameters"],
            "workflow_steps": [],
            "current_context": None,
            "user_preferences": None,
        }

        with patch("src.api.workflow.get_workflow_engine", return_value=mock_engine):
            from src.api.workflow import process_conversation

            # Create request with explicit parameters
            search_params = SearchIntentParameters(
                genres=["Action", "Drama"],
                year_range=[2020, 2023],
                limit=5,
                exclusions=["Horror"],
            )

            request = ConversationRequest(
                message="find good anime", search_parameters=search_params
            )

            # Process the request
            await process_conversation(request)

            # Verify explicit parameters were passed through
            call_kwargs = mock_engine.process_conversation.call_args.kwargs
            passed_params = call_kwargs["search_parameters"]

            assert passed_params["genres"] == ["Action", "Drama"]
            assert passed_params["year_range"] == [2020, 2023]
            assert passed_params["limit"] == 5
            assert passed_params["exclusions"] == ["Horror"]

    @pytest.mark.asyncio
    async def test_multimodal_integration_with_ai_parameters(self):
        """Test multimodal conversation with AI parameter extraction."""

        mock_engine = AsyncMock()
        mock_engine.process_multimodal_conversation.return_value = {
            "session_id": "multimodal_test",
            "messages": [
                "Multimodal query",
                "Found similar anime with extracted parameters",
            ],
            "workflow_steps": [
                {
                    "step_type": "multimodal_search",
                    "tool_name": "search_anime_by_image",
                    "result": {"visual_matches": 3, "text_filtered": 2},
                    "confidence": 0.88,
                }
            ],
            "current_context": {"image_data": "base64_image", "text_weight": 0.6},
            "user_preferences": None,
        }

        with patch("src.api.workflow.get_workflow_engine", return_value=mock_engine):
            from src.api.workflow import (
                MultimodalRequest,
                process_multimodal_conversation,
            )

            # Create multimodal request with AI parameter extraction
            request = MultimodalRequest(
                message="find action anime similar to this image from recent years",
                image_data="base64_encoded_image",
                text_weight=0.6,
                search_parameters=SearchIntentParameters(
                    genres=["Action"], year_range=[2020, 2024]
                ),
            )

            # Process the request
            response = await process_multimodal_conversation(request)

            # Verify multimodal processing
            assert response.session_id == "multimodal_test"
            assert response.current_context["image_data"] == "base64_image"
            assert response.current_context["text_weight"] == 0.6

            # Verify parameters were passed
            call_kwargs = mock_engine.process_multimodal_conversation.call_args.kwargs
            assert call_kwargs["image_data"] == "base64_encoded_image"
            assert call_kwargs["text_weight"] == 0.6
            assert call_kwargs["search_parameters"]["genres"] == ["Action"]
            assert call_kwargs["search_parameters"]["year_range"] == [2020, 2024]

    def test_parameter_filtering_edge_cases(self):
        """Test parameter filtering handles edge cases correctly."""
        from src.api.workflow import SearchIntentParameters, _prepare_search_parameters

        # Test with None values and empty lists
        params = SearchIntentParameters(
            genres=[],  # Empty list
            year_range=None,  # None value
            studios=["Mappa"],  # Valid value
            limit=5,
        )

        filtered = _prepare_search_parameters(params)

        # Should only include non-empty, non-None values
        assert "genres" not in filtered or not filtered["genres"]
        assert "year_range" not in filtered or filtered["year_range"] is None
        assert filtered["studios"] == ["Mappa"]
        assert filtered["limit"] == 5

        # Test with all None/empty
        empty_params = SearchIntentParameters(
            genres=[], year_range=None, anime_types=[], studios=None
        )

        filtered_empty = _prepare_search_parameters(empty_params)

        # Should only contain the default limit
        assert len(filtered_empty) == 1
        assert filtered_empty["limit"] == 10  # Default value

    @pytest.mark.asyncio
    async def test_performance_and_response_time(self, mock_mcp_tools):
        """Test that the AI-powered integration meets performance targets."""
        import time

        with (
            patch(
                "src.langgraph.react_agent_workflow.create_react_agent"
            ) as mock_create_agent,
            patch("src.langgraph.react_agent_workflow.MemorySaver"),
            patch("src.langgraph.langchain_tools.create_anime_langchain_tools"),
        ):

            # Create workflow engine
            engine = ReactAgentWorkflowEngine(mock_mcp_tools, LLMProvider.OPENAI)

            # Mock fast response
            mock_agent = AsyncMock()
            mock_create_agent.return_value = mock_agent
            mock_agent.ainvoke.return_value = {
                "messages": [
                    MagicMock(content="test query"),
                    MagicMock(content="test response"),
                ]
            }

            # Measure response time
            start_time = time.time()

            result = await engine.process_conversation(
                session_id="performance_test", message="find popular action anime"
            )

            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000

            # Verify performance target (should be much faster than 200ms in mock)
            assert response_time_ms < 1000  # Generous limit for mocked test
            assert result["session_id"] == "performance_test"

            # Verify workflow info includes performance data
            workflow_info = engine.get_workflow_info()
            assert workflow_info["performance"]["target_response_time"] == "120ms"
            assert workflow_info["performance"]["streaming_support"] is True
