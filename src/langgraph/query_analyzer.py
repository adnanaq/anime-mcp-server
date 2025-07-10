"""
Query intent analysis and routing logic for LangGraph workflows.

Analyzes user queries to determine optimal agent routing and tool selection
using LLM-powered intent classification and parameter extraction.
"""

from typing import Any, Dict, List, Optional

from ..config import get_settings

settings = get_settings()


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
