"""
Query intent analysis and routing logic for LangGraph workflows.

Analyzes user queries to determine optimal agent routing and tool selection
using LLM-powered intent classification and parameter extraction.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..config import get_settings
from ..services.llm_service import LLMService
from .workflow_state import QueryIntent

settings = get_settings()


class QueryAnalysisOutput(BaseModel):
    """Structured output for LLM-based query analysis."""

    intent_type: Literal[
        "search",
        "discover",
        "recommend",
        "schedule",
        "streaming",
        "seasonal",
        "similar",
        "details",
    ] = Field(description="Primary intent of the user query")

    confidence: float = Field(
        description="Confidence score 0-1 for intent classification"
    )

    needs_semantic_search: bool = Field(
        description="Query requires semantic/AI-powered search"
    )
    needs_streaming_info: bool = Field(
        description="User wants streaming platform information"
    )
    needs_broadcast_schedule: bool = Field(
        description="User needs broadcast times/schedules"
    )
    needs_seasonal_data: bool = Field(description="Query is about seasonal anime")
    needs_similarity_search: bool = Field(
        description="User wants anime similar to something"
    )

    platform_hints: List[str] = Field(
        description="Specific platforms mentioned or implied"
    )
    extracted_params: Dict[str, Any] = Field(description="Extracted search parameters")
    temporal_context: Optional[Dict[str, Any]] = Field(
        description="Time-related context"
    )

    recommended_agents: List[str] = Field(
        description="Agents that should handle this query"
    )
    workflow_complexity: Literal["simple", "moderate", "complex"] = Field(
        description="Complexity level for workflow planning"
    )


class QueryAnalyzer:
    """
    Intelligent query analysis for anime discovery workflows.

    Uses LLM-powered intent classification combined with pattern matching
    to determine optimal routing and tool selection.
    """

    def __init__(self):
        # Import here to avoid circular imports
        from .intelligent_router import IntelligentRouter

        self.router = IntelligentRouter()
        self.llm_service = LLMService()

    async def analyze_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze user query to determine intent and routing strategy using intelligent router.

        Args:
            query: User's anime search query
            context: Optional conversation/user context

        Returns:
            Dictionary with routing recommendations and extracted parameters
        """
        # Use intelligent router for comprehensive analysis
        routing_decision = await self.router.route_query(
            query=query,
            user_context=context,
            preferred_platforms=context.get("preferred_platforms") if context else None,
        )

        # Convert routing decision to query intent format compatible with existing workflow
        return {
            "intent_type": (
                routing_decision.primary_tools[0]
                if routing_decision.primary_tools
                else "search"
            ),
            "workflow_complexity": routing_decision.estimated_complexity,
            "needs_semantic_search": any(
                "semantic" in tool for tool in routing_decision.primary_tools
            ),
            "needs_similarity_search": any(
                "similar" in tool for tool in routing_decision.primary_tools
            ),
            "needs_seasonal_data": any(
                "seasonal" in tool for tool in routing_decision.primary_tools
            ),
            "needs_streaming_info": any(
                "streaming" in tool or "kitsu" in tool
                for tool in routing_decision.primary_tools
            ),
            "needs_broadcast_schedule": any(
                "schedule" in tool or "animeschedule" in tool
                for tool in routing_decision.primary_tools
            ),
            "recommended_agents": self._map_tools_to_agents(
                routing_decision.primary_tools
            ),
            "suggested_tools": routing_decision.primary_tools,
            "secondary_tools": routing_decision.secondary_tools,
            "execution_strategy": routing_decision.execution_strategy,
            "confidence": routing_decision.confidence,
            "reasoning": routing_decision.reasoning,
            "platform_priorities": routing_decision.platform_priorities,
            "enrichment_recommended": routing_decision.enrichment_recommended,
            "fallback_tools": routing_decision.fallback_tools,
            "data_requirements": self._extract_data_requirements(routing_decision),
            "platform_hints": list(routing_decision.platform_priorities.keys()),
            "temporal_context": None,  # Could be enhanced to extract from routing decision
            "extracted_params": {},  # Could be enhanced to extract from routing decision
        }

    def _map_tools_to_agents(self, tools: List[str]) -> List[str]:
        """Map selected tools to appropriate agents."""
        agents = []

        search_tools = [
            "search_anime_mal",
            "search_anime_anilist",
            "search_anime_jikan",
            "search_anime_kitsu",
            "anime_semantic_search",
        ]

        schedule_tools = [
            "search_anime_schedule",
            "get_schedule_data",
            "get_currently_airing",
        ]

        enrichment_tools = [
            "compare_anime_ratings_cross_platform",
            "get_cross_platform_anime_data",
            "correlate_anime_across_platforms",
            "get_streaming_availability_multi_platform",
        ]

        if any(tool in search_tools for tool in tools):
            agents.append("SearchAgent")

        if any(tool in schedule_tools for tool in tools):
            agents.append("ScheduleAgent")

        if any(tool in enrichment_tools for tool in tools):
            agents.append("SearchAgent")  # Enrichment tools are handled by SearchAgent

        return agents if agents else ["SearchAgent"]

    def _extract_data_requirements(self, routing_decision) -> List[str]:
        """Extract data requirements from routing decision."""
        requirements = []

        if any(
            "streaming" in tool or "kitsu" in tool
            for tool in routing_decision.primary_tools
        ):
            requirements.append("streaming_platforms")

        if any(
            "schedule" in tool or "animeschedule" in tool
            for tool in routing_decision.primary_tools
        ):
            requirements.append("broadcast_schedules")

        if any("seasonal" in tool for tool in routing_decision.primary_tools):
            requirements.append("seasonal_data")

        if any(
            "similar" in tool or "semantic" in tool
            for tool in routing_decision.primary_tools
        ):
            requirements.append("similarity_scores")

        if routing_decision.enrichment_recommended:
            requirements.append("cross_platform_enrichment")

        return requirements


    async def _llm_analyze_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        pattern_hints: Dict[str, Any],
    ) -> QueryAnalysisOutput:
        """Use LLM for deep query intent analysis."""

        system_prompt = """You are an anime discovery expert analyzing user queries to determine the best search strategy.

Analyze the user's query and classify their intent, then recommend the optimal agents and tools to use.

Intent Types:
- search: Basic anime search by title/keywords
- discover: Finding new anime based on preferences  
- recommend: Personalized recommendations
- schedule: Broadcast times and airing information
- streaming: Where to watch anime
- seasonal: Current/upcoming seasonal anime
- similar: Finding anime similar to a reference
- details: Getting detailed information about specific anime

Agent Types Available:
- SearchAgent: Platform-specific anime search (MAL, AniList, Jikan, Kitsu)
- ScheduleAgent: Broadcast schedules and streaming info (AnimeSchedule)
- EnrichmentAgent: Cross-platform data combination
- SemanticAgent: AI-powered similarity and discovery

Consider the pattern analysis hints and provide structured recommendations."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"""
User Query: "{query}"

Pattern Analysis Hints:
{pattern_hints}

Additional Context: {context or "None"}

Analyze this query and provide structured intent classification and routing recommendations.
"""
            ),
        ]

        try:
            result = await self.llm_service.get_structured_output(
                messages=messages, output_schema=QueryAnalysisOutput, model_type="fast"
            )
            return result
        except Exception:
            # Fallback to pattern-based analysis
            return self._fallback_analysis(query, pattern_hints)

    def _fallback_analysis(
        self, query: str, pattern_hints: Dict[str, Any]
    ) -> QueryAnalysisOutput:
        """Fallback analysis when LLM is unavailable."""

        # Determine intent based on patterns
        if pattern_hints["has_semantic_keywords"]:
            intent_type = "similar"
            agents = ["SemanticAgent"]
        elif pattern_hints["has_streaming_keywords"]:
            intent_type = "streaming"
            agents = ["ScheduleAgent", "SearchAgent"]
        elif pattern_hints["has_schedule_keywords"]:
            intent_type = "schedule"
            agents = ["ScheduleAgent"]
        elif pattern_hints["has_seasonal_keywords"]:
            intent_type = "seasonal"
            agents = ["SearchAgent"]
        else:
            intent_type = "search"
            agents = ["SearchAgent"]

        return QueryAnalysisOutput(
            intent_type=intent_type,
            confidence=0.7,
            needs_semantic_search=pattern_hints["has_semantic_keywords"],
            needs_streaming_info=pattern_hints["has_streaming_keywords"],
            needs_broadcast_schedule=pattern_hints["has_schedule_keywords"],
            needs_seasonal_data=pattern_hints["has_seasonal_keywords"],
            needs_similarity_search=pattern_hints["has_semantic_keywords"],
            platform_hints=pattern_hints["mentioned_platforms"],
            extracted_params={},
            temporal_context=pattern_hints["temporal_indicators"],
            recommended_agents=agents,
            workflow_complexity="simple",
        )

    def _build_query_intent(
        self,
        query: str,
        pattern_hints: Dict[str, Any],
        llm_analysis: QueryAnalysisOutput,
        context: Optional[Dict[str, Any]],
    ) -> QueryIntent:
        """Combine all analysis into final QueryIntent."""

        # Determine data requirements
        data_requirements = []
        if llm_analysis.needs_streaming_info:
            data_requirements.append("streaming_platforms")
        if llm_analysis.needs_broadcast_schedule:
            data_requirements.append("broadcast_schedules")
        if llm_analysis.needs_seasonal_data:
            data_requirements.append("seasonal_data")
        if llm_analysis.needs_similarity_search:
            data_requirements.append("similarity_scores")

        # Suggest tools based on requirements
        suggested_tools = []
        if llm_analysis.needs_semantic_search:
            suggested_tools.extend(["anime_semantic_search", "anime_similar"])
        if llm_analysis.needs_streaming_info:
            suggested_tools.extend(["search_anime_kitsu", "search_anime_schedule"])
        if llm_analysis.needs_broadcast_schedule:
            suggested_tools.extend(["get_currently_airing", "get_schedule_data"])
        if llm_analysis.needs_seasonal_data:
            suggested_tools.extend(["get_jikan_seasonal", "get_seasonal_anime_mal"])
        if not suggested_tools:  # Default search tools
            suggested_tools.extend(["search_anime_mal", "search_anime_anilist"])

        return QueryIntent(
            intent_type=llm_analysis.intent_type,
            confidence=llm_analysis.confidence,
            needs_semantic_search=llm_analysis.needs_semantic_search,
            needs_streaming_info=llm_analysis.needs_streaming_info,
            needs_broadcast_schedule=llm_analysis.needs_broadcast_schedule,
            needs_seasonal_data=llm_analysis.needs_seasonal_data,
            needs_similarity_search=llm_analysis.needs_similarity_search,
            platform_hints=llm_analysis.platform_hints,
            data_requirements=data_requirements,
            extracted_params=llm_analysis.extracted_params,
            temporal_context=llm_analysis.temporal_context,
            recommended_agents=llm_analysis.recommended_agents,
            suggested_tools=suggested_tools,
            workflow_complexity=llm_analysis.workflow_complexity,
        )
