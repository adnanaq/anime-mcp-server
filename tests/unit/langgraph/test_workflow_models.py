"""Tests for LangGraph workflow state models."""

import pytest
from pydantic import ValidationError

from src.langgraph.models import (
    AnimeSearchContext,
    ConversationState,
    MessageType,
    UserPreferences,
    WorkflowMessage,
    WorkflowStep,
    WorkflowStepType,
)


class TestWorkflowMessage:
    """Test WorkflowMessage model."""

    def test_create_user_message(self):
        """Test creating a user message."""
        message = WorkflowMessage(
            message_type=MessageType.USER,
            content="Find me some action anime",
            timestamp=1234567890.0,
            metadata={"session_id": "test-123"},
        )

        assert message.message_type == MessageType.USER
        assert message.content == "Find me some action anime"
        assert message.timestamp == 1234567890.0
        assert message.metadata["session_id"] == "test-123"

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        message = WorkflowMessage(
            message_type=MessageType.ASSISTANT,
            content="I found 5 action anime for you",
            tool_call_id="search_anime_123",
            tool_results=[{"title": "Naruto", "score": 0.95}],
        )

        assert message.message_type == MessageType.ASSISTANT
        assert message.tool_call_id == "search_anime_123"
        assert len(message.tool_results) == 1
        assert message.tool_results[0]["title"] == "Naruto"

    def test_create_system_message(self):
        """Test creating a system message."""
        message = WorkflowMessage(
            message_type=MessageType.SYSTEM, content="Workflow initialized"
        )

        assert message.message_type == MessageType.SYSTEM
        assert message.content == "Workflow initialized"

    def test_message_validation_error(self):
        """Test validation errors for invalid message types."""
        with pytest.raises(ValidationError):
            WorkflowMessage(message_type="invalid_type", content="test")  # type: ignore


class TestAnimeSearchContext:
    """Test AnimeSearchContext model."""

    def test_create_basic_context(self):
        """Test creating basic search context."""
        context = AnimeSearchContext(
            query="action anime",
            filters={"type": "TV", "year": 2023},
            results=[{"title": "Attack on Titan", "score": 0.9}],
        )

        assert context.query == "action anime"
        assert context.filters["type"] == "TV"
        assert context.filters["year"] == 2023
        assert len(context.results) == 1
        assert context.results[0]["title"] == "Attack on Titan"

    def test_create_multimodal_context(self):
        """Test creating multimodal search context."""
        context = AnimeSearchContext(
            query="mecha anime",
            image_data="base64_image_data",
            text_weight=0.7,
            results=[],
        )

        assert context.query == "mecha anime"
        assert context.image_data == "base64_image_data"
        assert context.text_weight == 0.7
        assert context.results == []

    def test_empty_context(self):
        """Test creating empty context."""
        context = AnimeSearchContext()

        assert context.query is None
        assert context.image_data is None
        assert context.text_weight == 0.5  # default value
        assert context.filters == {}
        assert context.results == []


class TestUserPreferences:
    """Test UserPreferences model."""

    def test_create_preferences(self):
        """Test creating user preferences."""
        prefs = UserPreferences(
            favorite_genres=["action", "adventure", "mecha"],
            favorite_studios=["Studio Ghibli", "Madhouse"],
            preferred_year_range=(2010, 2023),
            preferred_episode_count=(12, 24),
            language_preference="japanese",
            content_rating="PG-13",
        )

        assert "action" in prefs.favorite_genres
        assert "Studio Ghibli" in prefs.favorite_studios
        assert prefs.preferred_year_range == (2010, 2023)
        assert prefs.preferred_episode_count == (12, 24)
        assert prefs.language_preference == "japanese"
        assert prefs.content_rating == "PG-13"

    def test_empty_preferences(self):
        """Test creating empty preferences."""
        prefs = UserPreferences()

        assert prefs.favorite_genres == []
        assert prefs.favorite_studios == []
        assert prefs.preferred_year_range is None
        assert prefs.preferred_episode_count is None
        assert prefs.language_preference == "any"
        assert prefs.content_rating == "any"


class TestWorkflowStep:
    """Test WorkflowStep model."""

    def test_create_search_step(self):
        """Test creating a search workflow step."""
        step = WorkflowStep(
            step_type=WorkflowStepType.SEARCH,
            tool_name="search_anime",
            parameters={"query": "dragon ball", "limit": 5},
            result={"results": [{"title": "Dragon Ball Z"}]},
            confidence=0.95,
            execution_time_ms=150.0,
        )

        assert step.step_type == WorkflowStepType.SEARCH
        assert step.tool_name == "search_anime"
        assert step.parameters["query"] == "dragon ball"
        assert step.result["results"][0]["title"] == "Dragon Ball Z"
        assert step.confidence == 0.95
        assert step.execution_time_ms == 150.0

    def test_create_reasoning_step(self):
        """Test creating a reasoning workflow step."""
        step = WorkflowStep(
            step_type=WorkflowStepType.REASONING,
            reasoning="User seems to prefer action anime based on previous searches",
            confidence=0.8,
        )

        assert step.step_type == WorkflowStepType.REASONING
        assert "action anime" in step.reasoning
        assert step.confidence == 0.8
        assert step.tool_name is None

    def test_create_synthesis_step(self):
        """Test creating a synthesis workflow step."""
        step = WorkflowStep(
            step_type=WorkflowStepType.SYNTHESIS,
            reasoning="Combining search results with user preferences",
            result={"synthesized_recommendations": []},
            confidence=0.9,
        )

        assert step.step_type == WorkflowStepType.SYNTHESIS
        assert "Combining search results" in step.reasoning
        assert "synthesized_recommendations" in step.result
        assert step.confidence == 0.9


class TestConversationState:
    """Test ConversationState model."""

    def test_create_conversation_state(self):
        """Test creating conversation state."""
        messages = [
            WorkflowMessage(message_type=MessageType.USER, content="Find action anime")
        ]

        context = AnimeSearchContext(
            query="action anime", results=[{"title": "Naruto"}]
        )

        preferences = UserPreferences(favorite_genres=["action"])

        steps = [
            WorkflowStep(step_type=WorkflowStepType.SEARCH, tool_name="search_anime")
        ]

        state = ConversationState(
            session_id="test-session-123",
            messages=messages,
            current_context=context,
            user_preferences=preferences,
            workflow_steps=steps,
            current_step_index=0,
            conversation_summary="User looking for action anime",
        )

        assert state.session_id == "test-session-123"
        assert len(state.messages) == 1
        assert state.current_context.query == "action anime"
        assert "action" in state.user_preferences.favorite_genres
        assert len(state.workflow_steps) == 1
        assert state.current_step_index == 0
        assert "action anime" in state.conversation_summary

    def test_empty_conversation_state(self):
        """Test creating empty conversation state."""
        state = ConversationState(session_id="empty-session")

        assert state.session_id == "empty-session"
        assert state.messages == []
        assert state.current_context is None
        assert state.user_preferences is None
        assert state.workflow_steps == []
        assert state.current_step_index == 0
        assert state.conversation_summary == ""

    def test_add_message_to_state(self):
        """Test adding messages to conversation state."""
        state = ConversationState(session_id="test")

        # Add user message
        user_msg = WorkflowMessage(
            message_type=MessageType.USER, content="Find romance anime"
        )
        state.messages.append(user_msg)

        # Add assistant message
        assistant_msg = WorkflowMessage(
            message_type=MessageType.ASSISTANT,
            content="I found 3 romance anime",
            tool_results=[{"title": "Your Name"}],
        )
        state.messages.append(assistant_msg)

        assert len(state.messages) == 2
        assert state.messages[0].message_type == MessageType.USER
        assert state.messages[1].message_type == MessageType.ASSISTANT
        assert state.messages[1].tool_results[0]["title"] == "Your Name"

    def test_update_workflow_step(self):
        """Test updating workflow steps."""
        state = ConversationState(session_id="test")

        # Add initial step
        step1 = WorkflowStep(
            step_type=WorkflowStepType.SEARCH, tool_name="search_anime"
        )
        state.workflow_steps.append(step1)

        # Add reasoning step
        step2 = WorkflowStep(
            step_type=WorkflowStepType.REASONING, reasoning="Analyzing search results"
        )
        state.workflow_steps.append(step2)

        # Update current step index
        state.current_step_index = 1

        assert len(state.workflow_steps) == 2
        assert state.current_step_index == 1
        assert state.workflow_steps[1].step_type == WorkflowStepType.REASONING
