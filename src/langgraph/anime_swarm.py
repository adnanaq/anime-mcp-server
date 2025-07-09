"""
Main anime discovery swarm workflow using modern LangGraph architecture.

Implements multi-agent anime discovery system with intelligent routing,
memory persistence, and cross-platform data enrichment.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from langgraph_swarm import create_swarm

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from ..config import get_settings
from .agents.schedule_agent import ScheduleAgent
from .agents.search_agent import SearchAgent
from .conditional_router import ConditionalRouter, ExecutionContext
from .error_handling import SwarmErrorHandler
from .query_analyzer import QueryAnalyzer
from .swarm_error_integration import SwarmErrorIntegration
from .workflow_state import AnimeSwarmState, WorkflowResult

settings = get_settings()
logger = logging.getLogger(__name__)


class AnimeDiscoverySwarm:
    """
    Multi-agent anime discovery system using LangGraph swarm architecture.

    Features:
    - Intelligent query analysis and agent routing
    - Cross-platform anime search with specialized agents
    - Memory persistence for conversation continuity
    - Real-time broadcast schedules and streaming info
    - Semantic similarity and AI-powered discovery
    """

    def __init__(self):
        self.settings = settings
        self.query_analyzer = QueryAnalyzer()
        self.conditional_router = ConditionalRouter()

        # Initialize error handling
        self.error_handler = SwarmErrorHandler()
        self.error_integration = SwarmErrorIntegration(self.error_handler)

        # Initialize agents
        self.search_agent = SearchAgent()
        self.schedule_agent = ScheduleAgent()

        # Memory components for conversation persistence
        self.checkpointer = InMemorySaver()  # Short-term memory
        self.store = InMemoryStore()  # Long-term memory

        # Execution tracking
        self.execution_history = []

        # Create the swarm workflow
        self.swarm = self._create_swarm()

    def _create_swarm(self):
        """
        Create LangGraph swarm with specialized anime discovery agents.

        Uses create_swarm with the schema attribute bug fix applied.
        """
        # Get agent instances
        search_agent = self.search_agent.get_agent()
        schedule_agent = self.schedule_agent.get_agent()

        # Create swarm with intelligent routing
        workflow = create_swarm(
            agents=[search_agent, schedule_agent], default_active_agent="SearchAgent"
        )

        # Compile with memory components
        app = workflow.compile(checkpointer=self.checkpointer, store=self.store)

        return app

    async def discover_anime(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> WorkflowResult:
        """
        Main entry point for anime discovery using multi-agent workflow.

        Args:
            query: User's anime search query
            user_context: Optional user preferences and context
            session_id: Session ID for conversation persistence

        Returns:
            Comprehensive workflow result with anime recommendations
        """
        start_time = datetime.now()

        # Step 1: Analyze query intent with intelligent routing
        intent = await self.query_analyzer.analyze_query(query, user_context)

        # Step 2: Get platform status for conditional routing
        platform_status = self._get_platform_status()

        # Step 3: Create execution context for conditional routing
        execution_context = ExecutionContext(
            query=query,
            routing_decision=None,  # Will be set by conditional router
            user_context=user_context,
            platform_status=platform_status,
            previous_results=None,
            execution_history=self.execution_history,
            error_count=0,
            execution_time_ms=0,
        )

        # Step 4: Get conditional routing decision
        execution_config = await self.conditional_router.route_execution(
            execution_context
        )

        # Step 5: Build initial swarm state with routing
        initial_state = self._build_initial_state(
            query, intent, user_context, execution_config
        )

        # Step 6: Execute workflow with conditional routing
        config = self._build_workflow_config(session_id)

        # Create workflow state for swarm
        workflow_state = initial_state

        try:
            # Invoke the workflow with intelligent routing and error handling
            result = await self.error_integration.execute_with_error_handling(
                "swarm_workflow", self.swarm.ainvoke, workflow_state, config
            )

            # Step 7: Update execution history
            self._update_execution_history(
                query, execution_config, result, datetime.now() - start_time
            )

            # Update agent health on success
            active_agent = result.get("active_agent", "SearchAgent")
            self.error_integration.update_agent_health(active_agent, True)

            # Step 8: Build comprehensive result
            workflow_result = self._build_workflow_result(
                query=query,
                intent=intent,
                swarm_result=result,
                execution_time=datetime.now() - start_time,
                session_id=session_id,
                execution_config=execution_config,
            )

            return workflow_result

        except Exception as e:
            # Enhanced error handling with swarm fallbacks
            return await self.error_integration.handle_workflow_error(
                query,
                intent,
                e,
                execution_config,
                session_id,
                self.swarm.ainvoke,
                self._build_initial_state,
                self._build_workflow_config,
                self._build_workflow_result,
            )

    async def get_currently_airing(
        self, filters: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None
    ) -> WorkflowResult:
        """
        Specialized workflow for currently airing anime.

        Args:
            filters: Optional filters (day, timezone, platforms)
            session_id: Session ID for conversation persistence

        Returns:
            Currently airing anime with broadcast schedules
        """
        query = "What anime is currently airing?"
        if filters:
            query += f" Filters: {filters}"

        # Route directly to ScheduleAgent for airing info
        intent = await self.query_analyzer.analyze_query(query)
        intent["recommended_agents"] = ["ScheduleAgent"]

        return await self.discover_anime(query, filters, session_id)

    async def find_similar_anime(
        self,
        reference_anime: str,
        similarity_mode: str = "hybrid",
        session_id: Optional[str] = None,
    ) -> WorkflowResult:
        """
        Specialized workflow for finding similar anime.

        Args:
            reference_anime: Reference anime title or ID
            similarity_mode: "content", "visual", or "hybrid"
            session_id: Session ID for conversation persistence

        Returns:
            Similar anime with similarity scores and analysis
        """
        query = f"Find anime similar to {reference_anime}"
        context = {"similarity_mode": similarity_mode}

        # Route to SearchAgent with semantic focus
        return await self.discover_anime(query, context, session_id)

    async def search_by_streaming_platform(
        self,
        platforms: List[str],
        additional_filters: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> WorkflowResult:
        """
        Specialized workflow for streaming platform search.

        Args:
            platforms: List of streaming platforms
            additional_filters: Optional content filters
            session_id: Session ID for conversation persistence

        Returns:
            Anime available on specified platforms
        """
        platforms_str = ", ".join(platforms)
        query = f"Find anime available on {platforms_str}"

        context = {
            "streaming_platforms": platforms,
            "additional_filters": additional_filters or {},
        }

        # Route to ScheduleAgent for streaming focus
        return await self.discover_anime(query, context, session_id)

    def _build_initial_state(
        self,
        query: str,
        intent: Dict[str, Any],
        user_context: Optional[Dict[str, Any]],
        execution_config: Optional[Dict[str, Any]] = None,
    ) -> AnimeSwarmState:
        """Build initial state for swarm workflow."""
        # Determine search strategy from execution config or intent
        search_strategy = "platform_specific"
        if execution_config:
            if execution_config.get("execution_path") == "semantic_first":
                search_strategy = "semantic"
            elif execution_config.get("execution_path") == "enriched_search":
                search_strategy = "hybrid"
            elif execution_config.get("execution_path") in [
                "schedule_focused",
                "streaming_focused",
            ]:
                search_strategy = "seasonal"
        else:
            search_strategy = self._determine_search_strategy(intent)

        return AnimeSwarmState(
            messages=[],
            active_agent="SearchAgent",  # Default starting agent
            task_description=f"Help user discover anime: {query}",
            user_query=query,
            query_intent=intent,
            anime_results=[],
            enrichment_data={},
            preferred_platforms=(
                user_context.get("preferred_platforms", []) if user_context else []
            ),
            search_strategy=search_strategy,
            conversation_context=user_context or {},
            user_preferences=(
                user_context.get("preferences", {}) if user_context else {}
            ),
            current_step="initializing",
            completed_steps=[],
            next_actions=(
                execution_config.get("tools", [])
                if execution_config
                else intent.get("suggested_tools", [])
            ),
        )

    def _determine_search_strategy(self, intent: Dict[str, Any]) -> str:
        """Determine optimal search strategy based on intent."""
        if intent.get("needs_semantic_search") or intent.get("needs_similarity_search"):
            return "semantic"
        elif intent.get("needs_seasonal_data"):
            return "seasonal"
        elif intent.get("workflow_complexity") == "complex":
            return "hybrid"
        else:
            return "platform_specific"

    def _build_workflow_config(self, session_id: Optional[str]) -> Dict[str, Any]:
        """Build configuration for workflow execution."""
        config = {
            "configurable": {
                "thread_id": session_id or f"session_{datetime.now().timestamp()}"
            }
        }
        return config

    def _get_platform_status(self) -> Dict[str, bool]:
        """Get current platform availability status."""
        # In a real implementation, this would check actual platform health
        return {
            "mal": True,
            "anilist": True,
            "jikan": True,
            "kitsu": True,
            "animeschedule": True,
            "semantic": True,
        }

    def _update_execution_history(
        self,
        query: str,
        execution_config: Dict[str, Any],
        result: Dict[str, Any],
        execution_time,
    ):
        """Update execution history for learning and optimization."""
        execution_record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "execution_path": execution_config.get("execution_path"),
            "tools_used": execution_config.get("tools", []),
            "execution_time_ms": int(execution_time.total_seconds() * 1000),
            "success": bool(result.get("anime_results")),
            "result_count": len(result.get("anime_results", [])),
        }

        self.execution_history.append(execution_record)

        # Keep only last 100 executions
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]

    def _build_workflow_result(
        self,
        query: str,
        intent: Dict[str, Any],
        swarm_result: Dict[str, Any],
        execution_time,
        session_id: Optional[str],
        execution_config: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        """Build comprehensive workflow result."""
        # Include routing information in the result
        routing_metadata = {}
        if execution_config:
            routing_metadata = {
                "execution_path": execution_config.get("execution_path"),
                "routing_confidence": execution_config.get("routing_metadata", {}).get(
                    "original_routing_confidence"
                ),
                "platform_health_score": execution_config.get(
                    "routing_metadata", {}
                ).get("platform_health_score"),
                "complexity": execution_config.get("routing_metadata", {}).get(
                    "complexity"
                ),
            }

        return WorkflowResult(
            anime_results=swarm_result.get("anime_results", []),
            total_results=len(swarm_result.get("anime_results", [])),
            original_query=query,
            processed_query=query,  # Could be enhanced by query preprocessing
            intent_analysis=intent,
            agents_used=self._extract_agents_used(swarm_result, execution_config),
            tools_executed=self._extract_tools_executed(swarm_result, execution_config),
            execution_time_ms=int(execution_time.total_seconds() * 1000),
            platforms_queried=swarm_result.get("platforms_used", []),
            data_quality_summary=self._calculate_data_quality(swarm_result),
            enrichment_applied=self._extract_enrichment_applied(swarm_result),
            user_preferences_applied=[],
            recommendation_factors=[],
            suggested_refinements=self._generate_suggested_refinements(query, intent),
            related_queries=self._generate_related_queries(query, intent),
            workflow_version="2.0.0",  # Updated for intelligent routing
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            routing_metadata=routing_metadata,
        )

    def _build_error_result(
        self, query: str, intent: Dict[str, Any], error: str
    ) -> WorkflowResult:
        """Build simple error result when workflow fails."""
        return WorkflowResult(
            anime_results=[],
            total_results=0,
            original_query=query,
            processed_query=query,
            intent_analysis=intent,
            agents_used=[],
            tools_executed=[],
            execution_time_ms=0,
            platforms_queried=[],
            data_quality_summary={"error": error},
            enrichment_applied=[],
            user_preferences_applied=[],
            recommendation_factors=[],
            suggested_refinements=[
                "Try simplifying your search query",
                "Try again later",
            ],
            related_queries=[],
            workflow_version="2.0.0",
            timestamp=datetime.now().isoformat(),
            session_id=None,
        )

    def _extract_agents_used(
        self,
        swarm_result: Dict[str, Any],
        execution_config: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Extract which agents were used in the workflow."""
        agents = ["SearchAgent"]  # Default

        if execution_config:
            # Map execution path to agents
            path = execution_config.get("execution_path")
            if path in ["schedule_focused"]:
                agents.append("ScheduleAgent")
            elif path in ["enriched_search", "comparison_workflow"]:
                agents.extend(["SearchAgent", "EnrichmentAgent"])

        return list(set(agents))  # Remove duplicates

    def _extract_tools_executed(
        self,
        swarm_result: Dict[str, Any],
        execution_config: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Extract which tools were executed."""
        if execution_config:
            return execution_config.get("tools", [])
        return []  # Placeholder for swarm result parsing

    def _calculate_data_quality(self, swarm_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate data quality metrics."""
        return {"overall_quality": 0.85}  # Placeholder

    def _extract_enrichment_applied(self, swarm_result: Dict[str, Any]) -> List[str]:
        """Extract what enrichment was applied."""
        return []  # Placeholder

    def _generate_suggested_refinements(
        self, query: str, intent: Dict[str, Any]
    ) -> List[str]:
        """Generate suggested query refinements."""
        suggestions = []

        if intent.get("workflow_complexity") == "simple":
            suggestions.append("Add genre or year filters for more specific results")

        if not intent.get("needs_streaming_info"):
            suggestions.append("Ask about streaming availability")

        return suggestions

    def _generate_related_queries(
        self, query: str, intent: Dict[str, Any]
    ) -> List[str]:
        """Generate related query suggestions."""
        related = []

        if intent.get("intent_type") == "search":
            related.extend(
                [
                    "Find similar anime",
                    "Check streaming availability",
                    "Get broadcast schedule",
                ]
            )

        return related

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report for the swarm system."""
        return self.error_integration.get_health_report()
