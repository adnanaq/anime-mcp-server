"""Tests for LLM integration in workflow engine."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.langgraph.adapters import MCPAdapterRegistry
from src.langgraph.models import (
    AnimeSearchContext,
    ConversationState,
    MessageType,
    WorkflowMessage,
)
from src.langgraph.workflow_engine import ConversationalAgent
from src.services.llm_service import SearchIntent


class TestLLMIntegration:
    """Test LLM integration in workflow engine."""

    @pytest.fixture
    def mock_adapter_registry(self):
        """Create mock adapter registry."""
        registry = Mock(spec=MCPAdapterRegistry)
        registry.invoke_tool = AsyncMock()
        return registry

    @pytest.fixture
    def agent(self, mock_adapter_registry):
        """Create conversational agent with mocked registry."""
        return ConversationalAgent(mock_adapter_registry)

    @pytest.fixture
    def conversation_state(self):
        """Create basic conversation state."""
        state = ConversationState(session_id="test_session")
        state.add_message(
            WorkflowMessage(
                message_type=MessageType.USER, content="find 5 mecha anime from 2020s"
            )
        )
        return state

    @pytest.mark.asyncio
    async def test_extract_search_context_with_llm(self, agent):
        """Test search context extraction using LLM service."""
        mock_intent = SearchIntent(
            query="mecha anime",
            limit=5,
            genres=["mecha", "action"],
            year_range=(2020, 2029),
            anime_types=["TV"],
            confidence=0.9,
        )

        with patch(
            "src.services.llm_service.extract_search_intent",
            new_callable=AsyncMock,
            return_value=mock_intent,
        ) as mock_extract:

            context = await agent._extract_search_context(
                "find 5 mecha anime from 2020s"
            )

            # Verify LLM service was called
            mock_extract.assert_called_once_with("find 5 mecha anime from 2020s")

            # Verify context was properly constructed
            assert context.query == "mecha anime"
            assert context.limit == 5
            assert context.filters["genres"] == ["mecha", "action"]
            assert context.filters["year_range"] == (2020, 2029)
            assert context.filters["type"] == "TV"

    @pytest.mark.asyncio
    async def test_extract_search_context_single_year(self, agent):
        """Test extraction with single year (not range)."""
        mock_intent = SearchIntent(
            query="anime from 2015",
            year_range=(2015, 2015),  # Same year for start and end
            confidence=0.8,
        )

        with patch(
            "src.services.llm_service.extract_search_intent",
            new_callable=AsyncMock,
            return_value=mock_intent,
        ):

            context = await agent._extract_search_context("anime from 2015")

            # Should use single year, not range
            assert context.filters["year"] == 2015
            assert "year_range" not in context.filters

    @pytest.mark.asyncio
    async def test_extract_search_context_multiple_types(self, agent):
        """Test extraction with multiple anime types."""
        mock_intent = SearchIntent(
            query="movies and OVAs", anime_types=["Movie", "OVA"], confidence=0.85
        )

        with patch(
            "src.services.llm_service.extract_search_intent",
            new_callable=AsyncMock,
            return_value=mock_intent,
        ):

            context = await agent._extract_search_context("movies and OVAs")

            # Should use plural types for multiple
            assert context.filters["types"] == ["Movie", "OVA"]
            assert "type" not in context.filters

    @pytest.mark.asyncio
    async def test_extract_search_context_with_exclusions(self, agent):
        """Test extraction with exclusions and mood keywords."""
        mock_intent = SearchIntent(
            query="comedy anime",
            genres=["comedy"],
            exclusions=["horror", "dark"],
            mood_keywords=["light", "fun"],
            studios=["Studio Ghibli"],
            confidence=0.9,
        )

        with patch(
            "src.services.llm_service.extract_search_intent",
            new_callable=AsyncMock,
            return_value=mock_intent,
        ):

            context = await agent._extract_search_context(
                "light comedy anime but not horror"
            )

            assert context.filters["genres"] == ["comedy"]
            assert context.filters["exclusions"] == ["horror", "dark"]
            assert context.filters["mood"] == ["light", "fun"]
            assert context.filters["studios"] == ["Studio Ghibli"]

    @pytest.mark.asyncio
    async def test_extract_search_context_fallback_on_error(self, agent):
        """Test fallback to basic extraction when LLM fails."""
        with patch(
            "src.services.llm_service.extract_search_intent",
            side_effect=Exception("LLM API Error"),
        ):

            context = await agent._extract_search_context("movie from 2020")

            # Should fall back to basic extraction
            assert context.query == "movie from 2020"
            assert context.filters["type"] == "Movie"
            assert context.filters["year"] == 2020

    @pytest.mark.asyncio
    async def test_understand_node_with_llm_integration(
        self, agent, conversation_state
    ):
        """Test the understand node with LLM integration."""
        mock_intent = SearchIntent(
            query="action adventure anime",
            limit=3,
            genres=["action", "adventure"],
            confidence=0.9,
        )

        with patch(
            "src.services.llm_service.extract_search_intent",
            new_callable=AsyncMock,
            return_value=mock_intent,
        ):

            # Test the understand node
            result_state = await agent._understand_node(conversation_state)

            # Verify context was updated
            assert result_state.current_context is not None
            assert result_state.current_context.query == "action adventure anime"
            assert result_state.current_context.limit == 3
            assert result_state.current_context.filters["genres"] == [
                "action",
                "adventure",
            ]

            # Verify workflow step was added
            assert len(result_state.workflow_steps) == 1
            step = result_state.workflow_steps[0]
            assert "action adventure anime" in step.reasoning

    @pytest.mark.asyncio
    async def test_understand_node_preserves_existing_context(self, agent):
        """Test that understand node preserves existing context data."""
        # Create state with existing context
        state = ConversationState(session_id="test")
        existing_context = AnimeSearchContext(
            image_data="base64_image_data",
            text_weight=0.8,
            search_history=[{"previous": "search"}],
        )
        state.update_context(existing_context)
        state.add_message(
            WorkflowMessage(message_type=MessageType.USER, content="find comedy anime")
        )

        mock_intent = SearchIntent(
            query="comedy anime", genres=["comedy"], confidence=0.85
        )

        with patch(
            "src.services.llm_service.extract_search_intent",
            new_callable=AsyncMock,
            return_value=mock_intent,
        ):

            result_state = await agent._understand_node(state)

            # Verify new context includes extracted data
            assert result_state.current_context.query == "comedy anime"
            assert result_state.current_context.filters["genres"] == ["comedy"]

            # Verify existing data was preserved
            assert result_state.current_context.image_data == "base64_image_data"
            assert result_state.current_context.text_weight == 0.8
            assert result_state.current_context.search_history == [
                {"previous": "search"}
            ]


class TestFallbackExtraction:
    """Test fallback extraction functionality."""

    @pytest.fixture
    def agent(self):
        """Create agent for fallback testing."""
        registry = Mock(spec=MCPAdapterRegistry)
        return ConversationalAgent(registry)

    def test_fallback_extract_search_context_year(self, agent):
        """Test fallback year extraction."""
        context = agent._fallback_extract_search_context("anime from 2020")

        assert context.query == "anime from 2020"
        assert context.filters["year"] == 2020

    def test_fallback_extract_search_context_type_movie(self, agent):
        """Test fallback type extraction for movies."""
        context = agent._fallback_extract_search_context("great animated movie")

        assert context.filters["type"] == "Movie"

    def test_fallback_extract_search_context_type_tv(self, agent):
        """Test fallback type extraction for TV series."""
        context = agent._fallback_extract_search_context("tv series recommendation")

        assert context.filters["type"] == "TV"

    def test_fallback_extract_search_context_type_ova(self, agent):
        """Test fallback type extraction for OVA."""
        context = agent._fallback_extract_search_context("good ova episodes")

        assert context.filters["type"] == "OVA"

    def test_fallback_extract_search_context_no_patterns(self, agent):
        """Test fallback with no matching patterns."""
        context = agent._fallback_extract_search_context("great anime")

        assert context.query == "great anime"
        assert len(context.filters) == 0


class TestLLMIntegrationEndToEnd:
    """End-to-end tests for LLM integration in workflow."""

    @pytest.mark.asyncio
    async def test_realistic_query_processing(self):
        """Test realistic query processing with LLM integration."""
        registry = Mock(spec=MCPAdapterRegistry)
        registry.invoke_tool = AsyncMock(
            return_value=[
                {"title": "Gundam Wing", "score": 0.9},
                {"title": "Evangelion", "score": 0.85},
            ]
        )

        agent = ConversationalAgent(registry)
        state = ConversationState(session_id="test")

        mock_intent = SearchIntent(
            query="mecha anime",
            limit=5,
            genres=["mecha", "action"],
            year_range=(1995, 2005),
            confidence=0.9,
        )

        with patch(
            "src.services.llm_service.extract_search_intent",
            new_callable=AsyncMock,
            return_value=mock_intent,
        ):

            # Process user message through workflow
            result_state = await agent.process_user_message(
                state, "find 5 mecha anime from late 90s to early 2000s"
            )

            # Verify LLM extracted parameters correctly
            assert result_state.current_context.query == "mecha anime"
            assert result_state.current_context.limit == 5
            assert result_state.current_context.filters["genres"] == ["mecha", "action"]
            assert result_state.current_context.filters["year_range"] == (1995, 2005)

            # Verify search was executed with correct parameters
            registry.invoke_tool.assert_called()
            search_call = registry.invoke_tool.call_args
            assert "mecha anime" in str(search_call)

    @pytest.mark.asyncio
    async def test_complex_query_with_multiple_constraints(self):
        """Test complex query with multiple constraints."""
        registry = Mock(spec=MCPAdapterRegistry)
        registry.invoke_tool = AsyncMock(return_value=[])
        agent = ConversationalAgent(registry)
        state = ConversationState(session_id="test")

        mock_intent = SearchIntent(
            query="romantic comedy anime",
            limit=3,
            genres=["romance", "comedy"],
            anime_types=["TV"],
            year_range=(2015, 2023),
            exclusions=["ecchi"],
            mood_keywords=["wholesome"],
            confidence=0.92,
        )

        with patch(
            "src.services.llm_service.extract_search_intent",
            new_callable=AsyncMock,
            return_value=mock_intent,
        ):

            result_state = await agent.process_user_message(
                state,
                "find 3 wholesome romantic comedy TV series from 2015-2023 but not ecchi",
            )

            context = result_state.current_context
            assert context.query == "romantic comedy anime"
            assert context.limit == 3
            assert context.filters["genres"] == ["romance", "comedy"]
            assert context.filters["type"] == "TV"
            assert context.filters["year_range"] == (2015, 2023)
            assert context.filters["exclusions"] == ["ecchi"]
            assert context.filters["mood"] == ["wholesome"]

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and graceful fallback."""
        registry = Mock(spec=MCPAdapterRegistry)
        registry.invoke_tool = AsyncMock(return_value=[])
        agent = ConversationalAgent(registry)
        state = ConversationState(session_id="test")

        # Simulate LLM service failure
        with patch(
            "src.services.llm_service.extract_search_intent",
            side_effect=Exception("Network error"),
        ):

            result_state = await agent.process_user_message(state, "movie from 2020")

            # Should fall back to basic extraction
            assert result_state.current_context.query == "movie from 2020"
            assert result_state.current_context.filters["type"] == "Movie"
            assert result_state.current_context.filters["year"] == 2020

            # Workflow should still complete successfully
            assert len(result_state.workflow_steps) > 0
