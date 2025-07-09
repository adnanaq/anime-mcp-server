"""
Swarm error handling integration for anime discovery workflows.

Provides integration utilities between SwarmErrorHandler and AnimeDiscoverySwarm
to keep the main workflow file under 500 lines.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from .error_handling import SwarmErrorContext, SwarmErrorHandler
from .workflow_state import WorkflowResult

logger = logging.getLogger(__name__)


class SwarmErrorIntegration:
    """Integration layer for swarm error handling in anime discovery workflows."""

    def __init__(self, error_handler: SwarmErrorHandler):
        self.error_handler = error_handler

    async def handle_workflow_error(
        self,
        query: str,
        intent: Dict[str, Any],
        error: Exception,
        execution_config: Optional[Dict[str, Any]],
        session_id: Optional[str],
        swarm_invoke_func,
        build_initial_state_func,
        build_workflow_config_func,
        build_workflow_result_func,
    ) -> WorkflowResult:
        """
        Handle swarm workflow errors with intelligent fallbacks.

        Args:
            query: User's anime search query
            intent: Analyzed query intent
            error: The exception that occurred
            execution_config: Execution configuration
            session_id: Session ID for conversation persistence
            swarm_invoke_func: Function to invoke swarm workflow
            build_initial_state_func: Function to build initial state
            build_workflow_config_func: Function to build workflow config
            build_workflow_result_func: Function to build workflow result

        Returns:
            WorkflowResult with fallback data or error information
        """
        # Create swarm error context
        error_context = SwarmErrorContext.from_exception(
            exception=error,
            user_message="Unable to complete anime search due to system error",
            trace_data={
                "query": query,
                "intent": intent,
                "execution_config": execution_config,
                "session_id": session_id,
            },
            recovery_suggestions=[
                "Try simplifying your search query",
                "Try again in a few moments",
                "Use specific anime titles for better results",
            ],
        )

        # Determine available agents and current agent
        available_agents = ["SearchAgent", "ScheduleAgent"]
        current_agent = intent.get("recommended_agents", ["SearchAgent"])[0]

        # Update agent health
        self.error_handler.update_agent_health(current_agent, False)

        # Attempt swarm error recovery
        try:
            fallback_result = await self.error_handler.handle_swarm_error(
                error_context, current_agent, available_agents
            )

            if fallback_result.get("fallback_strategy") == "agent_handoff":
                # Attempt recovery with different agent
                target_agent = fallback_result.get("target_agent")
                if target_agent:
                    logger.info(
                        f"Attempting agent handoff from {current_agent} to {target_agent}"
                    )

                    # Update intent for fallback agent
                    fallback_intent = intent.copy()
                    fallback_intent["recommended_agents"] = [target_agent]

                    # Simplified execution for fallback
                    simplified_state = build_initial_state_func(
                        query, fallback_intent, None
                    )
                    config = build_workflow_config_func(session_id)

                    try:
                        fallback_swarm_result = await swarm_invoke_func(
                            simplified_state, config
                        )
                        self.error_handler.update_agent_health(target_agent, True)

                        logger.info(
                            f"Agent handoff successful, recovered with {target_agent}"
                        )

                        return build_workflow_result_func(
                            query=query,
                            intent=fallback_intent,
                            swarm_result=fallback_swarm_result,
                            execution_time=timedelta(seconds=0),
                            session_id=session_id,
                            execution_config={
                                "fallback_recovery": True,
                                "handoff_from": current_agent,
                            },
                        )
                    except Exception as fallback_error:
                        logger.warning(
                            f"Agent handoff to {target_agent} failed: {fallback_error}"
                        )
                        self.error_handler.update_agent_health(target_agent, False)
                        # Continue to graceful degradation

            elif fallback_result.get("fallback_strategy") == "tool_substitution":
                # Tool substitution fallback
                alternative_tools = fallback_result.get("alternative_tools", [])
                logger.info(
                    f"Tool substitution fallback suggested: {alternative_tools}"
                )

                # Return result indicating tool substitution should be attempted
                return self._build_tool_substitution_result(
                    query, intent, alternative_tools, error_context, session_id
                )

            elif fallback_result.get("fallback_strategy") == "workflow_simplification":
                # Workflow simplification fallback
                logger.info("Workflow simplification fallback suggested")

                # Return simplified workflow result
                return self._build_simplified_workflow_result(
                    query, intent, error_context, session_id
                )

        except Exception as recovery_error:
            logger.warning(f"Swarm error recovery failed: {recovery_error}")

        # Final fallback - return graceful error response
        return self._build_error_result(query, intent, str(error), error_context)

    def _build_tool_substitution_result(
        self,
        query: str,
        intent: Dict[str, Any],
        alternative_tools: list,
        error_context: SwarmErrorContext,
        session_id: Optional[str],
    ) -> WorkflowResult:
        """Build result for tool substitution fallback."""
        return WorkflowResult(
            anime_results=[],
            total_results=0,
            original_query=query,
            processed_query=query,
            intent_analysis=intent,
            agents_used=["FallbackAgent"],
            tools_executed=alternative_tools,
            execution_time_ms=0,
            platforms_queried=[],
            data_quality_summary={
                "fallback_strategy": "tool_substitution",
                "alternative_tools": alternative_tools,
                "error": error_context.debug_info,
                "correlation_id": error_context.correlation_id,
            },
            enrichment_applied=[],
            user_preferences_applied=[],
            recommendation_factors=[],
            suggested_refinements=error_context.recovery_suggestions,
            related_queries=[
                "Try a different search approach",
                "Use more specific terms",
            ],
            workflow_version="2.0.0-fallback",
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
        )

    def _build_simplified_workflow_result(
        self,
        query: str,
        intent: Dict[str, Any],
        error_context: SwarmErrorContext,
        session_id: Optional[str],
    ) -> WorkflowResult:
        """Build result for simplified workflow fallback."""
        return WorkflowResult(
            anime_results=[],
            total_results=0,
            original_query=query,
            processed_query=f"Simplified: {query}",
            intent_analysis=intent,
            agents_used=["SimpleSearchAgent"],
            tools_executed=["anime_semantic_search"],
            execution_time_ms=0,
            platforms_queried=["semantic"],
            data_quality_summary={
                "fallback_strategy": "workflow_simplification",
                "simplified": True,
                "error": error_context.debug_info,
                "correlation_id": error_context.correlation_id,
            },
            enrichment_applied=[],
            user_preferences_applied=[],
            recommendation_factors=["simplified_search"],
            suggested_refinements=error_context.recovery_suggestions,
            related_queries=["Try semantic search", "Use basic keyword search"],
            workflow_version="2.0.0-simplified",
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
        )

    def _build_error_result(
        self,
        query: str,
        intent: Dict[str, Any],
        error: str,
        error_context: SwarmErrorContext,
    ) -> WorkflowResult:
        """Build error result when all fallbacks fail."""
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
            data_quality_summary={
                "error": error,
                "error_severity": error_context.severity.value,
                "correlation_id": error_context.correlation_id,
                "recovery_suggestions": error_context.recovery_suggestions,
            },
            enrichment_applied=[],
            user_preferences_applied=[],
            recommendation_factors=[],
            suggested_refinements=error_context.recovery_suggestions,
            related_queries=[],
            workflow_version="2.0.0-error",
            timestamp=datetime.now().isoformat(),
            session_id=None,
        )

    async def execute_with_error_handling(
        self, operation_name: str, operation_func, *args, **kwargs
    ):
        """Execute operation with circuit breaker protection."""
        return await self.error_handler.execute_with_circuit_breaker(
            operation_name, operation_func, *args, **kwargs
        )

    def update_agent_health(self, agent_name: str, success: bool):
        """Update agent health based on operation results."""
        self.error_handler.update_agent_health(agent_name, success)

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        return {
            "circuit_breakers": self.error_handler.get_circuit_breaker_status(),
            "agents": self.error_handler.get_agent_health_report(),
        }
