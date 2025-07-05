"""
LangGraph workflow state definitions and schemas.

Defines shared state structures for multi-agent anime discovery workflows
using the modern LangGraph swarm architecture with memory persistence.
"""

from typing import Annotated, List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class AnimeSwarmState(TypedDict):
    """
    Shared state for anime discovery multi-agent swarm.

    Based on LangGraph swarm architecture with conversation persistence
    and cross-agent task handoff capabilities.
    """

    # Conversation state (shared across all agents)
    messages: Annotated[List[AnyMessage], add_messages]

    # Active agent tracking
    active_agent: str

    # Task and context
    task_description: Optional[str]
    user_query: str
    query_intent: Dict[str, Any]

    # Search and discovery results
    anime_results: List[Dict[str, Any]]
    enrichment_data: Dict[str, Any]

    # Platform routing and preferences
    preferred_platforms: List[str]
    search_strategy: Literal["semantic", "platform_specific", "hybrid", "seasonal"]

    # Memory and context
    conversation_context: Dict[str, Any]
    user_preferences: Dict[str, Any]

    # Workflow control
    current_step: str
    completed_steps: List[str]
    next_actions: List[str]


class SearchAgentState(TypedDict):
    """State specific to the SearchAgent for anime discovery."""

    search_messages: Annotated[List[AnyMessage], add_messages]
    search_query: str
    search_results: List[Dict[str, Any]]
    search_metadata: Dict[str, Any]
    platforms_used: List[str]


class ScheduleAgentState(TypedDict):
    """State specific to the ScheduleAgent for broadcast/streaming enrichment."""

    schedule_messages: Annotated[List[AnyMessage], add_messages]
    anime_ids: List[str]
    schedule_data: Dict[str, Any]
    streaming_info: Dict[str, Any]
    broadcast_schedules: List[Dict[str, Any]]


class EnrichmentAgentState(TypedDict):
    """State specific to the EnrichmentAgent for cross-platform data combination."""

    enrichment_messages: Annotated[List[AnyMessage], add_messages]
    base_results: List[Dict[str, Any]]
    enriched_results: List[Dict[str, Any]]
    enrichment_sources: List[str]
    data_quality_scores: Dict[str, float]


class QueryIntent(TypedDict):
    """
    Structured query intent analysis for intelligent routing.

    Helps determine which agents and tools to use based on user query characteristics.
    """

    # Intent classification
    intent_type: Literal[
        "search",
        "discover",
        "recommend",
        "schedule",
        "streaming",
        "seasonal",
        "similar",
        "details",
    ]
    confidence: float

    # Content preferences
    needs_semantic_search: bool
    needs_streaming_info: bool
    needs_broadcast_schedule: bool
    needs_seasonal_data: bool
    needs_similarity_search: bool

    # Platform routing
    platform_hints: List[str]
    data_requirements: List[str]

    # Query parameters
    extracted_params: Dict[str, Any]
    temporal_context: Optional[Dict[str, Any]]

    # Workflow routing
    recommended_agents: List[str]
    suggested_tools: List[str]
    workflow_complexity: Literal["simple", "moderate", "complex"]


class WorkflowResult(TypedDict):
    """
    Final workflow result with comprehensive anime data and metadata.
    """

    # Core results
    anime_results: List[Dict[str, Any]]
    total_results: int

    # Query context
    original_query: str
    processed_query: str
    intent_analysis: QueryIntent

    # Workflow execution
    agents_used: List[str]
    tools_executed: List[str]
    execution_time_ms: int

    # Data sources and quality
    platforms_queried: List[str]
    data_quality_summary: Dict[str, Any]
    enrichment_applied: List[str]

    # User context
    user_preferences_applied: List[str]
    recommendation_factors: List[str]

    # Follow-up suggestions
    suggested_refinements: List[str]
    related_queries: List[str]

    # Metadata
    workflow_version: str
    timestamp: str
    session_id: Optional[str]
    routing_metadata: Optional[Dict[str, Any]]
