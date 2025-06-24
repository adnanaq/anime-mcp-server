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
    REFINEMENT = "refinement"
    ORCHESTRATION = "orchestration"
    ADAPTATION = "adaptation"


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


class QueryChain(BaseModel):
    """Represents a chain of related queries for complex discovery."""
    chain_id: str
    queries: List[str] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    results_mapping: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class RefinementCriteria(BaseModel):
    """Criteria for refining search results through multiple iterations."""
    min_confidence: float = 0.7
    max_iterations: int = 3
    target_result_count: int = 10
    focus_areas: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)
    quality_thresholds: Dict[str, float] = Field(default_factory=dict)


class OrchestrationPlan(BaseModel):
    """Plan for orchestrating multiple tools and operations."""
    plan_id: str
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    parallel_groups: List[List[str]] = Field(default_factory=list)
    estimated_duration_ms: Optional[int] = None
    priority_weights: Dict[str, float] = Field(default_factory=dict)


class ConversationFlow(BaseModel):
    """Enhanced conversation flow for multi-modal and complex interactions."""
    flow_id: str
    flow_type: str  # "discovery", "refinement", "exploration", "comparison"
    current_stage: str
    stages: List[Dict[str, Any]] = Field(default_factory=list)
    branch_conditions: Dict[str, Any] = Field(default_factory=dict)
    context_carryover: Dict[str, Any] = Field(default_factory=dict)
    user_feedback: List[Dict[str, Any]] = Field(default_factory=list)


class AdaptivePreferences(BaseModel):
    """Adaptive user preferences that learn from interactions."""
    base_preferences: UserPreferences
    learned_patterns: Dict[str, Any] = Field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_levels: Dict[str, float] = Field(default_factory=dict)
    adaptation_rate: float = 0.1
    last_update: float = Field(default_factory=time.time)
    
    def adapt_from_interaction(self, interaction: Dict[str, Any]) -> None:
        """Adapt preferences based on user interaction."""
        self.interaction_history.append(interaction)
        self.last_update = time.time()
        # Adaptation logic would be implemented in the workflow engine


class SmartOrchestrationState(ConversationState):
    """Extended conversation state for smart orchestration features."""
    query_chains: List[QueryChain] = Field(default_factory=list)
    refinement_criteria: Optional[RefinementCriteria] = None
    orchestration_plan: Optional[OrchestrationPlan] = None
    conversation_flow: Optional[ConversationFlow] = None
    adaptive_preferences: Optional[AdaptivePreferences] = None
    discovery_depth: int = 1
    max_discovery_depth: int = 5
    
    def create_query_chain(self, initial_query: str) -> QueryChain:
        """Create a new query chain for complex discovery."""
        chain = QueryChain(
            chain_id=f"chain_{len(self.query_chains) + 1}",
            queries=[initial_query]
        )
        self.query_chains.append(chain)
        return chain
    
    def add_to_chain(self, chain_id: str, query: str, relationship: Dict[str, Any]) -> bool:
        """Add a query to an existing chain."""
        for chain in self.query_chains:
            if chain.chain_id == chain_id:
                chain.queries.append(query)
                chain.relationships.append(relationship)
                return True
        return False