"""LangGraph workflow state models with type safety."""
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import time


class MessageType(str, Enum):
    """Message types in workflow conversations."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class WorkflowStepType(str, Enum):
    """Types of workflow steps."""
    SEARCH = "search"
    REASONING = "reasoning"
    SYNTHESIS = "synthesis"
    RECOMMENDATION = "recommendation"
    VALIDATION = "validation"


class WorkflowMessage(BaseModel):
    """A message in the workflow conversation."""
    message_type: MessageType
    content: str
    timestamp: float = Field(default_factory=time.time)
    tool_call_id: Optional[str] = None
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnimeSearchContext(BaseModel):
    """Context for anime search operations."""
    query: Optional[str] = None
    image_data: Optional[str] = None
    text_weight: float = 0.5
    filters: Dict[str, Any] = Field(default_factory=dict)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    search_history: List[Dict[str, Any]] = Field(default_factory=list)


class UserPreferences(BaseModel):
    """User preferences for anime recommendations."""
    favorite_genres: List[str] = Field(default_factory=list)
    favorite_studios: List[str] = Field(default_factory=list)
    preferred_year_range: Optional[Tuple[int, int]] = None
    preferred_episode_count: Optional[Tuple[int, int]] = None
    language_preference: str = "any"
    content_rating: str = "any"
    excluded_genres: List[str] = Field(default_factory=list)
    viewing_history: List[Dict[str, Any]] = Field(default_factory=list)


class WorkflowStep(BaseModel):
    """A single step in the workflow execution."""
    step_type: WorkflowStepType
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class ConversationState(BaseModel):
    """Complete state of a conversation workflow."""
    session_id: str
    messages: List[WorkflowMessage] = Field(default_factory=list)
    current_context: Optional[AnimeSearchContext] = None
    user_preferences: Optional[UserPreferences] = None
    workflow_steps: List[WorkflowStep] = Field(default_factory=list)
    current_step_index: int = 0
    conversation_summary: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def add_message(self, message: WorkflowMessage) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = time.time()

    def add_workflow_step(self, step: WorkflowStep) -> None:
        """Add a workflow step."""
        self.workflow_steps.append(step)
        self.updated_at = time.time()

    def get_current_step(self) -> Optional[WorkflowStep]:
        """Get the current workflow step."""
        if 0 <= self.current_step_index < len(self.workflow_steps):
            return self.workflow_steps[self.current_step_index]
        return None

    def advance_step(self) -> bool:
        """Advance to the next workflow step."""
        if self.current_step_index < len(self.workflow_steps) - 1:
            self.current_step_index += 1
            self.updated_at = time.time()
            return True
        return False

    def update_context(self, context: AnimeSearchContext) -> None:
        """Update the current search context."""
        self.current_context = context
        self.updated_at = time.time()

    def update_preferences(self, preferences: UserPreferences) -> None:
        """Update user preferences."""
        self.user_preferences = preferences
        self.updated_at = time.time()