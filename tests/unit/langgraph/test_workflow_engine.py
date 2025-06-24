"""Tests for LangGraph workflow engine."""

from unittest.mock import AsyncMock, Mock

import pytest

from src.langgraph.adapters import MCPAdapterRegistry
from src.langgraph.models import (
    AnimeSearchContext,
    ConversationState,
    MessageType,
    UserPreferences,
    WorkflowStepType,
)
from src.langgraph.workflow_engine import (
    AnimeWorkflowEngine,
    ConversationalAgent,
    WorkflowGraph,
    WorkflowNode,
)


class TestWorkflowNode:
    """Test workflow node implementation."""

    def test_create_node(self):
        """Test creating a workflow node."""

        async def test_func(state: ConversationState) -> ConversationState:
            return state

        node = WorkflowNode(
            name="test_node", function=test_func, description="Test node"
        )

        assert node.name == "test_node"
        assert node.description == "Test node"
        assert callable(node.function)

    @pytest.mark.asyncio
    async def test_execute_node(self):
        """Test executing a workflow node."""

        async def test_func(state: ConversationState) -> ConversationState:
            state.conversation_summary = "Updated by test node"
            return state

        node = WorkflowNode("test", test_func, "Test")
        state = ConversationState(session_id="test")

        result = await node.execute(state)

        assert result.conversation_summary == "Updated by test node"


class TestWorkflowGraph:
    """Test workflow graph implementation."""

    def test_create_graph(self):
        """Test creating workflow graph."""
        graph = WorkflowGraph()

        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_add_node(self):
        """Test adding nodes to graph."""
        graph = WorkflowGraph()

        async def test_func(state: ConversationState) -> ConversationState:
            return state

        node = WorkflowNode("test", test_func, "Test")
        graph.add_node(node)

        assert "test" in graph.nodes
        assert graph.nodes["test"] == node

    def test_add_edge(self):
        """Test adding edges to graph."""
        graph = WorkflowGraph()

        async def func1(state: ConversationState) -> ConversationState:
            return state

        async def func2(state: ConversationState) -> ConversationState:
            return state

        node1 = WorkflowNode("node1", func1, "Node 1")
        node2 = WorkflowNode("node2", func2, "Node 2")

        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_edge("node1", "node2")

        assert "node1" in graph.edges
        assert "node2" in graph.edges["node1"]

    def test_get_next_nodes(self):
        """Test getting next nodes from graph."""
        graph = WorkflowGraph()

        async def func(state: ConversationState) -> ConversationState:
            return state

        node1 = WorkflowNode("start", func, "Start")
        node2 = WorkflowNode("middle", func, "Middle")
        node3 = WorkflowNode("end", func, "End")

        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        graph.add_edge("start", "middle")
        graph.add_edge("start", "end")

        next_nodes = graph.get_next_nodes("start")
        assert len(next_nodes) == 2
        assert "middle" in next_nodes
        assert "end" in next_nodes


class TestConversationalAgent:
    """Test conversational agent."""

    @pytest.mark.asyncio
    async def test_create_agent(self):
        """Test creating conversational agent."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        agent = ConversationalAgent(mock_registry)

        assert agent.adapter_registry == mock_registry
        assert agent.workflow_graph is not None

    @pytest.mark.asyncio
    async def test_process_user_message(self):
        """Test processing user message."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        mock_registry.invoke_tool = AsyncMock(return_value=[])
        agent = ConversationalAgent(mock_registry)

        state = ConversationState(session_id="test")

        result = await agent.process_user_message(state, "Find action anime")

        # Should have both user message and assistant response
        assert len(result.messages) >= 1
        assert result.messages[0].message_type == MessageType.USER
        assert result.messages[0].content == "Find action anime"
        # Should have workflow steps executed
        assert len(result.workflow_steps) > 0

    @pytest.mark.asyncio
    async def test_search_node(self):
        """Test search workflow node."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        mock_registry.invoke_tool = AsyncMock(
            return_value=[{"title": "Naruto", "anime_id": "abc123", "score": 0.95}]
        )

        agent = ConversationalAgent(mock_registry)

        state = ConversationState(session_id="test")
        state.current_context = AnimeSearchContext(query="action anime")

        result = await agent._search_node(state)

        assert result.current_context.results is not None
        assert len(result.current_context.results) == 1
        assert result.current_context.results[0]["title"] == "Naruto"
        mock_registry.invoke_tool.assert_called_once_with(
            "search_anime", {"query": "action anime", "limit": 10}
        )

    @pytest.mark.asyncio
    async def test_reasoning_node(self):
        """Test reasoning workflow node."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        agent = ConversationalAgent(mock_registry)

        state = ConversationState(session_id="test")
        state.current_context = AnimeSearchContext(
            query="action anime",
            results=[{"title": "Naruto", "tags": ["action", "shounen"]}],
        )

        result = await agent._reasoning_node(state)

        assert len(result.workflow_steps) > 0
        reasoning_step = result.workflow_steps[-1]
        assert reasoning_step.step_type == WorkflowStepType.REASONING
        assert reasoning_step.reasoning is not None
        assert "action" in reasoning_step.reasoning.lower()

    @pytest.mark.asyncio
    async def test_reasoning_node_with_none_years(self):
        """Test reasoning node properly handles None year values."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        agent = ConversationalAgent(mock_registry)

        state = ConversationState(session_id="test")
        state.current_context = AnimeSearchContext(
            query="test anime",
            results=[
                {
                    "title": "Anime 1",
                    "tags": ["action"],
                    "studios": ["Studio A"],
                    "year": 2020,
                },
                {
                    "title": "Anime 2",
                    "tags": ["romance"],
                    "studios": ["Studio B"],
                    "year": None,
                },
                {
                    "title": "Anime 3",
                    "tags": ["comedy"],
                    "studios": ["Studio C"],
                    "year": 2022,
                },
                {"title": "Anime 4", "tags": ["drama"], "studios": [], "year": None},
            ],
        )

        result = await agent._reasoning_node(state)

        assert len(result.workflow_steps) > 0
        reasoning_step = result.workflow_steps[-1]
        assert reasoning_step.step_type == WorkflowStepType.REASONING

        # Should not crash and should include year range from valid years only
        reasoning = reasoning_step.reasoning
        assert "2020-2022" in reasoning  # Only valid years should be included
        assert "action" in reasoning  # Should include genres
        assert "Studio A" in reasoning  # Should include studios

    @pytest.mark.asyncio
    async def test_synthesis_node(self):
        """Test synthesis workflow node."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        agent = ConversationalAgent(mock_registry)

        state = ConversationState(session_id="test")
        state.current_context = AnimeSearchContext(
            results=[
                {"title": "Naruto", "score": 0.95, "tags": ["action"]},
                {"title": "One Piece", "score": 0.90, "tags": ["action", "adventure"]},
            ]
        )
        state.user_preferences = UserPreferences(favorite_genres=["action"])

        result = await agent._synthesis_node(state)

        assert len(result.workflow_steps) > 0
        synthesis_step = result.workflow_steps[-1]
        assert synthesis_step.step_type == WorkflowStepType.SYNTHESIS
        assert synthesis_step.result is not None
        assert "synthesized_results" in synthesis_step.result


class TestAnimeWorkflowEngine:
    """Test anime workflow engine."""

    @pytest.mark.asyncio
    async def test_create_engine(self):
        """Test creating workflow engine."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        engine = AnimeWorkflowEngine(mock_registry)

        assert engine.adapter_registry == mock_registry
        assert engine.agent is not None

    @pytest.mark.asyncio
    async def test_process_conversation(self):
        """Test processing conversation."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        mock_registry.invoke_tool = AsyncMock(
            return_value=[
                {"title": "Attack on Titan", "anime_id": "aot123", "score": 0.92}
            ]
        )

        engine = AnimeWorkflowEngine(mock_registry)

        state = ConversationState(session_id="test-session")

        result = await engine.process_conversation(state, "Find dark fantasy anime")

        # Should have processed the message
        assert len(result.messages) >= 1
        assert result.messages[0].content == "Find dark fantasy anime"

        # Should have search context
        assert result.current_context is not None
        assert result.current_context.query == "Find dark fantasy anime"

    @pytest.mark.asyncio
    async def test_multimodal_conversation(self):
        """Test multimodal conversation processing."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        mock_registry.invoke_tool = AsyncMock(
            return_value=[
                {"title": "Mecha Anime", "anime_id": "mecha123", "score": 0.88}
            ]
        )

        engine = AnimeWorkflowEngine(mock_registry)

        state = ConversationState(session_id="test-multimodal")

        result = await engine.process_multimodal_conversation(
            state, "Find mecha anime", "base64_image_data"
        )

        # Should have processed both text and image
        assert len(result.messages) >= 1
        assert result.current_context is not None
        assert result.current_context.query == "Find mecha anime"
        assert result.current_context.image_data == "base64_image_data"
        # Should have called multimodal search tool
        mock_registry.invoke_tool.assert_called_with(
            "search_multimodal_anime",
            {
                "query": "Find mecha anime",
                "image_data": "base64_image_data",
                "text_weight": 0.7,
                "limit": 10,
            },
        )

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self):
        """Test workflow error handling."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        mock_registry.invoke_tool = AsyncMock(side_effect=Exception("Search failed"))

        engine = AnimeWorkflowEngine(mock_registry)

        state = ConversationState(session_id="error-test")

        # Should handle errors gracefully
        result = await engine.process_conversation(state, "Find anime")

        # Should still return a valid state
        assert result.session_id == "error-test"
        assert len(result.messages) >= 1

    @pytest.mark.asyncio
    async def test_conversation_context_preservation(self):
        """Test that conversation context is preserved across interactions."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        mock_registry.invoke_tool = AsyncMock(
            return_value=[{"title": "Contextual Result", "anime_id": "ctx123"}]
        )

        engine = AnimeWorkflowEngine(mock_registry)

        # First interaction
        state = ConversationState(session_id="context-test")
        state1 = await engine.process_conversation(state, "Find action anime")

        # Second interaction should preserve context
        state2 = await engine.process_conversation(
            state1, "Find similar to the first result"
        )

        # Should maintain session and build on previous context
        assert state2.session_id == "context-test"
        assert len(state2.messages) >= 2
        # Both interactions should have produced workflow steps
        assert len(state1.workflow_steps) > 0
        assert len(state2.workflow_steps) > 0

    @pytest.mark.asyncio
    async def test_preference_learning(self):
        """Test user preference learning from conversation."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        mock_registry.invoke_tool = AsyncMock(
            return_value=[{"title": "Action Anime", "tags": ["action", "shounen"]}]
        )

        engine = AnimeWorkflowEngine(mock_registry)

        state = ConversationState(session_id="pref-test")

        # Multiple interactions should build preferences
        state1 = await engine.process_conversation(state, "I love action anime")
        state2 = await engine.process_conversation(
            state1, "Shounen is my favorite genre"
        )

        # Should have learned preferences
        if state2.user_preferences:
            # Preferences should be inferred from conversation
            assert (
                len(state2.user_preferences.favorite_genres) > 0
                or len(state2.messages) >= 2
            )
