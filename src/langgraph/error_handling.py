"""
LangGraph-specific error handling and fallback strategies for anime discovery workflows.

Integrates with existing comprehensive error handling infrastructure while adding
LangGraph swarm-specific patterns for agent handoffs and tool chain resilience.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ..integrations.error_handling import (
    CircuitBreaker,
)
from ..integrations.error_handling import ErrorContext as BaseErrorContext

logger = logging.getLogger(__name__)


class SwarmFallbackStrategy(str, Enum):
    """LangGraph swarm-specific fallback strategies."""

    AGENT_HANDOFF = "agent_handoff"  # Transfer to different agent
    TOOL_SUBSTITUTION = "tool_substitution"  # Use alternative tools
    WORKFLOW_SIMPLIFICATION = "workflow_simplification"  # Simplify workflow
    PARALLEL_EXECUTION = "parallel_execution"  # Execute multiple paths
    MEMORY_RECOVERY = "memory_recovery"  # Recover from conversation state
    CHECKPOINT_ROLLBACK = "checkpoint_rollback"  # Rollback to previous state


@dataclass
class SwarmErrorContext(BaseErrorContext):
    """Extended error context for LangGraph swarm operations."""

    agent_name: Optional[str] = None
    tool_name: Optional[str] = None
    workflow_step: Optional[str] = None
    available_agents: List[str] = None
    checkpoint_available: bool = False

    def __post_init__(self):
        if self.available_agents is None:
            self.available_agents = []


@dataclass
class SwarmFallbackConfig:
    """Configuration for LangGraph swarm fallback strategies."""

    strategy: SwarmFallbackStrategy
    target_agent: Optional[str] = None
    alternative_tools: List[str] = None
    fallback_agents: List[str] = None
    max_retries: int = 2
    preserve_context: bool = True

    def __post_init__(self):
        if self.alternative_tools is None:
            self.alternative_tools = []
        if self.fallback_agents is None:
            self.fallback_agents = []


class SwarmErrorHandler:
    """
    LangGraph swarm-specific error handler with agent handoff capabilities.

    Integrates with existing circuit breaker infrastructure while adding
    swarm-specific patterns like agent handoffs and workflow simplification.
    """

    def __init__(self):
        # Initialize circuit breakers for different components
        self.circuit_breakers = {
            "search_agent": CircuitBreaker(
                failure_threshold=3, api_name="search_agent"
            ),
            "schedule_agent": CircuitBreaker(
                failure_threshold=3, api_name="schedule_agent"
            ),
            "semantic_search": CircuitBreaker(
                failure_threshold=5, api_name="semantic_search"
            ),
            "mal_client": CircuitBreaker(failure_threshold=5, api_name="mal_client"),
            "anilist_client": CircuitBreaker(
                failure_threshold=5, api_name="anilist_client"
            ),
        }

        # Swarm-specific fallback configurations
        self.swarm_fallback_configs = self._initialize_swarm_fallbacks()

        # Agent health and capabilities
        self.agent_health = {
            "SearchAgent": {
                "available": True,
                "specializations": ["search", "semantic"],
            },
            "ScheduleAgent": {
                "available": True,
                "specializations": ["schedule", "streaming"],
            },
        }

    def _initialize_swarm_fallbacks(self) -> Dict[str, List[SwarmFallbackConfig]]:
        """Initialize swarm-specific fallback configurations."""
        return {
            "agent_failure": [
                SwarmFallbackConfig(
                    strategy=SwarmFallbackStrategy.AGENT_HANDOFF,
                    fallback_agents=["SearchAgent", "ScheduleAgent"],
                ),
                SwarmFallbackConfig(
                    strategy=SwarmFallbackStrategy.TOOL_SUBSTITUTION,
                    alternative_tools=["anime_semantic_search", "search_anime_jikan"],
                ),
            ],
            "tool_failure": [
                SwarmFallbackConfig(
                    strategy=SwarmFallbackStrategy.TOOL_SUBSTITUTION,
                    alternative_tools=["search_anime_anilist", "anime_semantic_search"],
                ),
                SwarmFallbackConfig(
                    strategy=SwarmFallbackStrategy.AGENT_HANDOFF,
                    target_agent="SearchAgent",
                ),
            ],
            "workflow_timeout": [
                SwarmFallbackConfig(
                    strategy=SwarmFallbackStrategy.WORKFLOW_SIMPLIFICATION
                ),
                SwarmFallbackConfig(
                    strategy=SwarmFallbackStrategy.PARALLEL_EXECUTION,
                    alternative_tools=["anime_semantic_search", "search_anime_jikan"],
                ),
            ],
            "memory_error": [
                SwarmFallbackConfig(strategy=SwarmFallbackStrategy.CHECKPOINT_ROLLBACK),
                SwarmFallbackConfig(
                    strategy=SwarmFallbackStrategy.WORKFLOW_SIMPLIFICATION
                ),
            ],
        }

    async def handle_swarm_error(
        self,
        error_context: SwarmErrorContext,
        current_agent: str,
        available_agents: List[str],
    ) -> Dict[str, Any]:
        """
        Handle error with swarm-specific fallback strategies.

        Args:
            error_context: Swarm-specific error context
            current_agent: Currently active agent
            available_agents: List of available agents for handoff

        Returns:
            Fallback result with agent handoff instructions or graceful degradation
        """
        logger.warning(
            f"Handling swarm error in {current_agent}: {error_context.debug_info}"
        )

        error_context.agent_name = current_agent
        error_context.available_agents = available_agents

        # Check circuit breaker for current agent
        agent_breaker = self.circuit_breakers.get(
            current_agent.lower().replace("agent", "_agent")
        )
        if agent_breaker and agent_breaker.is_open():
            logger.warning(
                f"Circuit breaker open for {current_agent}, forcing agent handoff"
            )
            return await self._execute_agent_handoff(error_context, available_agents)

        # Classify error for swarm handling
        error_category = self._classify_swarm_error(error_context.debug_info)

        # Get swarm fallback strategies
        fallback_configs = self.swarm_fallback_configs.get(
            error_category,
            [SwarmFallbackConfig(strategy=SwarmFallbackStrategy.AGENT_HANDOFF)],
        )

        # Try each fallback strategy
        for config in fallback_configs:
            try:
                result = await self._execute_swarm_fallback(
                    error_context, config, current_agent
                )
                if result:
                    logger.info(f"Swarm fallback successful using: {config.strategy}")
                    return result
            except Exception as fallback_error:
                logger.warning(
                    f"Swarm fallback {config.strategy} failed: {fallback_error}"
                )
                continue

        # All fallbacks failed - return graceful degradation
        return self._build_swarm_error_response(error_context)

    def _classify_swarm_error(self, debug_info: str) -> str:
        """Classify error for swarm-specific handling."""
        debug_lower = debug_info.lower()

        if any(term in debug_lower for term in ["agent", "handoff", "tool_node"]):
            return "agent_failure"
        elif any(term in debug_lower for term in ["tool", "function", "api"]):
            return "tool_failure"
        elif any(term in debug_lower for term in ["timeout", "deadline", "slow"]):
            return "workflow_timeout"
        elif any(term in debug_lower for term in ["memory", "checkpoint", "state"]):
            return "memory_error"
        else:
            return "general_failure"

    async def _execute_swarm_fallback(
        self,
        error_context: SwarmErrorContext,
        config: SwarmFallbackConfig,
        current_agent: str,
    ) -> Optional[Dict[str, Any]]:
        """Execute a specific swarm fallback strategy."""

        if config.strategy == SwarmFallbackStrategy.AGENT_HANDOFF:
            return await self._execute_agent_handoff(
                error_context, config.fallback_agents or error_context.available_agents
            )

        elif config.strategy == SwarmFallbackStrategy.TOOL_SUBSTITUTION:
            return self._suggest_tool_substitution(config.alternative_tools)

        elif config.strategy == SwarmFallbackStrategy.WORKFLOW_SIMPLIFICATION:
            return self._suggest_workflow_simplification(error_context)

        elif config.strategy == SwarmFallbackStrategy.PARALLEL_EXECUTION:
            return self._suggest_parallel_execution(config.alternative_tools)

        elif config.strategy == SwarmFallbackStrategy.CHECKPOINT_ROLLBACK:
            return self._suggest_checkpoint_rollback(error_context)

        return None

    async def _execute_agent_handoff(
        self, error_context: SwarmErrorContext, available_agents: List[str]
    ) -> Dict[str, Any]:
        """Execute agent handoff to a healthy agent."""
        # Filter out current agent and check health
        healthy_agents = []
        for agent in available_agents:
            if agent != error_context.agent_name and self._is_agent_healthy(agent):
                healthy_agents.append(agent)

        if not healthy_agents:
            return None

        # Choose best agent based on error context
        target_agent = self._select_best_agent(error_context, healthy_agents)

        return {
            "fallback_strategy": "agent_handoff",
            "target_agent": target_agent,
            "handoff_reason": f"Error in {error_context.agent_name}: {error_context.user_message}",
            "preserve_context": True,
            "error_recovery": True,
        }

    def _suggest_tool_substitution(
        self, alternative_tools: List[str]
    ) -> Dict[str, Any]:
        """Suggest alternative tools for the failed operation."""
        return {
            "fallback_strategy": "tool_substitution",
            "alternative_tools": alternative_tools,
            "execution_mode": "sequential",
            "preserve_query": True,
        }

    def _suggest_workflow_simplification(
        self, error_context: SwarmErrorContext
    ) -> Dict[str, Any]:
        """Suggest simplified workflow approach."""
        return {
            "fallback_strategy": "workflow_simplification",
            "simplified_tools": ["anime_semantic_search"],
            "reduce_complexity": True,
            "skip_enrichment": True,
        }

    def _suggest_parallel_execution(
        self, alternative_tools: List[str]
    ) -> Dict[str, Any]:
        """Suggest parallel execution of alternative tools."""
        return {
            "fallback_strategy": "parallel_execution",
            "parallel_tools": alternative_tools,
            "merge_results": True,
            "timeout_per_tool": 10.0,
        }

    def _suggest_checkpoint_rollback(
        self, error_context: SwarmErrorContext
    ) -> Optional[Dict[str, Any]]:
        """Suggest rollback to previous checkpoint if available."""
        if not error_context.checkpoint_available:
            return None

        return {
            "fallback_strategy": "checkpoint_rollback",
            "rollback_steps": 1,
            "preserve_user_query": True,
            "retry_with_fallback": True,
        }

    def _is_agent_healthy(self, agent_name: str) -> bool:
        """Check if an agent is healthy and available."""
        health = self.agent_health.get(agent_name, {})

        # Check circuit breaker
        breaker_key = agent_name.lower().replace("agent", "_agent")
        breaker = self.circuit_breakers.get(breaker_key)
        if breaker and breaker.is_open():
            return False

        return health.get("available", False)

    def _select_best_agent(
        self, error_context: SwarmErrorContext, healthy_agents: List[str]
    ) -> str:
        """Select the best agent for handoff based on error context and specializations."""
        # Simple selection based on agent specializations
        for agent in healthy_agents:
            specializations = self.agent_health.get(agent, {}).get(
                "specializations", []
            )

            # Match agent to error context
            if error_context.tool_name:
                if "search" in error_context.tool_name and "search" in specializations:
                    return agent
                elif (
                    "schedule" in error_context.tool_name
                    and "schedule" in specializations
                ):
                    return agent

        # Default to first healthy agent
        return healthy_agents[0]

    def _build_swarm_error_response(
        self, error_context: SwarmErrorContext
    ) -> Dict[str, Any]:
        """Build error response when all swarm fallbacks fail."""
        return {
            "anime_results": [],
            "total_results": 0,
            "error_message": error_context.user_message,
            "error_severity": error_context.severity.value,
            "failed_agent": error_context.agent_name,
            "recovery_suggestions": error_context.recovery_suggestions,
            "fallback_strategy": "graceful_degradation",
            "partial_success": False,
            "correlation_id": error_context.correlation_id,
        }

    async def execute_with_circuit_breaker(
        self, operation_name: str, operation_func: Callable, *args, **kwargs
    ) -> Any:
        """Execute operation with circuit breaker protection."""
        breaker = self.circuit_breakers.get(operation_name)
        if not breaker:
            # No circuit breaker for this operation, execute directly
            return await operation_func(*args, **kwargs)

        return await breaker.call_with_breaker(lambda: operation_func(*args, **kwargs))

    def update_agent_health(self, agent_name: str, success: bool):
        """Update agent health based on operation results."""
        if agent_name in self.agent_health:
            self.agent_health[agent_name]["available"] = success

        # Also update circuit breaker
        breaker_key = agent_name.lower().replace("agent", "_agent")
        if breaker_key in self.circuit_breakers:
            breaker = self.circuit_breakers[breaker_key]
            if success:
                breaker._reset()
            else:
                breaker._record_failure()

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        return {
            name: breaker.get_metrics()
            for name, breaker in self.circuit_breakers.items()
        }

    def get_agent_health_report(self) -> Dict[str, Any]:
        """Get comprehensive agent health report."""
        return {
            "agents": self.agent_health.copy(),
            "circuit_breakers": self.get_circuit_breaker_status(),
            "healthy_agents": [
                name
                for name, health in self.agent_health.items()
                if health.get("available", False)
            ],
        }
