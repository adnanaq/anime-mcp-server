"""
Conditional workflow routing based on query characteristics.

Implements dynamic workflow routing that adapts execution paths based on:
- Query complexity and intent analysis
- Real-time platform availability
- User preferences and context
- Tool execution results and feedback
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from ..config import get_settings
from .intelligent_router import RoutingDecision

logger = logging.getLogger(__name__)
settings = get_settings()


class RoutingCondition(str, Enum):
    """Types of routing conditions."""

    QUERY_COMPLEXITY = "query_complexity"
    PLATFORM_AVAILABILITY = "platform_availability"
    USER_PREFERENCE = "user_preference"
    TOOL_SUCCESS_RATE = "tool_success_rate"
    DATA_QUALITY = "data_quality"
    RESPONSE_TIME = "response_time"
    ERROR_THRESHOLD = "error_threshold"


class ExecutionPath(str, Enum):
    """Available execution paths."""

    SIMPLE_SEARCH = "simple_search"  # Direct platform search
    ENRICHED_SEARCH = "enriched_search"  # Multi-platform with enrichment
    FALLBACK_SEARCH = "fallback_search"  # Alternative tools on failure
    SEMANTIC_FIRST = "semantic_first"  # AI-powered search priority
    SCHEDULE_FOCUSED = "schedule_focused"  # Broadcast/timing priority
    STREAMING_FOCUSED = "streaming_focused"  # Availability priority
    COMPARISON_WORKFLOW = "comparison_workflow"  # Cross-platform comparison
    HYBRID_WORKFLOW = "hybrid_workflow"  # Adaptive multi-step


@dataclass
class RoutingRule:
    """Rule for conditional routing decisions."""

    condition: RoutingCondition
    threshold: Any  # Threshold value for condition
    target_path: ExecutionPath  # Path to route to if condition met
    priority: int  # Rule priority (higher = more important)
    description: str  # Human-readable description


@dataclass
class ExecutionContext:
    """Context for routing decisions."""

    query: str
    routing_decision: RoutingDecision
    user_context: Optional[Dict[str, Any]]
    platform_status: Dict[str, bool]  # Platform availability
    previous_results: Optional[Dict[str, Any]]
    execution_history: List[Dict[str, Any]]
    error_count: int
    execution_time_ms: int


class ConditionalRouter:
    """
    Conditional workflow router for adaptive execution paths.

    Dynamically selects execution paths based on real-time conditions,
    user preferences, and system state.
    """

    def __init__(self):
        # Initialize routing rules
        self.routing_rules = self._initialize_routing_rules()

        # Platform health tracking
        self.platform_health = {
            "mal": {"available": True, "success_rate": 0.95, "avg_response_ms": 800},
            "anilist": {
                "available": True,
                "success_rate": 0.98,
                "avg_response_ms": 600,
            },
            "jikan": {"available": True, "success_rate": 0.90, "avg_response_ms": 1000},
            "kitsu": {"available": True, "success_rate": 0.85, "avg_response_ms": 1200},
            "animeschedule": {
                "available": True,
                "success_rate": 0.92,
                "avg_response_ms": 500,
            },
            "semantic": {
                "available": True,
                "success_rate": 0.99,
                "avg_response_ms": 300,
            },
        }

        # Execution path definitions
        self.execution_paths = self._define_execution_paths()

    def _initialize_routing_rules(self) -> List[RoutingRule]:
        """Initialize conditional routing rules."""
        return [
            # Complexity-based routing
            RoutingRule(
                condition=RoutingCondition.QUERY_COMPLEXITY,
                threshold="complex",
                target_path=ExecutionPath.ENRICHED_SEARCH,
                priority=100,
                description="Use enriched search for complex queries",
            ),
            # Platform availability routing
            RoutingRule(
                condition=RoutingCondition.PLATFORM_AVAILABILITY,
                threshold=0.5,  # If less than 50% of platforms available
                target_path=ExecutionPath.FALLBACK_SEARCH,
                priority=90,
                description="Use fallback tools when platforms are unavailable",
            ),
            # User preference routing
            RoutingRule(
                condition=RoutingCondition.USER_PREFERENCE,
                threshold="semantic_priority",
                target_path=ExecutionPath.SEMANTIC_FIRST,
                priority=80,
                description="Prioritize AI search for users who prefer semantic results",
            ),
            # Tool success rate routing
            RoutingRule(
                condition=RoutingCondition.TOOL_SUCCESS_RATE,
                threshold=0.7,  # If primary tools have <70% success rate
                target_path=ExecutionPath.FALLBACK_SEARCH,
                priority=85,
                description="Switch to fallback tools for low success rates",
            ),
            # Response time routing
            RoutingRule(
                condition=RoutingCondition.RESPONSE_TIME,
                threshold=5000,  # 5 seconds
                target_path=ExecutionPath.SIMPLE_SEARCH,
                priority=70,
                description="Use simple search for faster response times",
            ),
            # Error threshold routing
            RoutingRule(
                condition=RoutingCondition.ERROR_THRESHOLD,
                threshold=3,  # 3 errors
                target_path=ExecutionPath.FALLBACK_SEARCH,
                priority=95,
                description="Switch to fallback after repeated errors",
            ),
        ]

    def _define_execution_paths(self) -> Dict[ExecutionPath, Dict[str, Any]]:
        """Define execution path configurations."""
        return {
            ExecutionPath.SIMPLE_SEARCH: {
                "tools": ["search_anime_anilist", "search_anime_mal"],
                "strategy": "parallel",
                "max_tools": 2,
                "timeout_ms": 3000,
                "description": "Fast, basic search using reliable platforms",
            },
            ExecutionPath.ENRICHED_SEARCH: {
                "tools": [
                    "search_anime_anilist",
                    "search_anime_mal",
                    "search_anime_jikan",
                    "get_cross_platform_anime_data",
                    "compare_anime_ratings_cross_platform",
                ],
                "strategy": "sequential",
                "max_tools": 5,
                "timeout_ms": 10000,
                "description": "Comprehensive search with cross-platform enrichment",
            },
            ExecutionPath.FALLBACK_SEARCH: {
                "tools": ["search_anime_jikan", "anime_semantic_search"],
                "strategy": "parallel",
                "max_tools": 2,
                "timeout_ms": 5000,
                "description": "Alternative tools when primary platforms fail",
            },
            ExecutionPath.SEMANTIC_FIRST: {
                "tools": [
                    "anime_semantic_search",
                    "search_anime_anilist",
                    "search_anime_mal",
                ],
                "strategy": "sequential",
                "max_tools": 3,
                "timeout_ms": 5000,
                "description": "AI-powered search with platform verification",
            },
            ExecutionPath.SCHEDULE_FOCUSED: {
                "tools": [
                    "search_anime_schedule",
                    "get_currently_airing",
                    "search_anime_anilist",
                ],
                "strategy": "parallel",
                "max_tools": 3,
                "timeout_ms": 4000,
                "description": "Broadcast and scheduling information priority",
            },
            ExecutionPath.STREAMING_FOCUSED: {
                "tools": [
                    "search_anime_kitsu",
                    "search_anime_schedule",
                    "get_streaming_availability_multi_platform",
                ],
                "strategy": "parallel",
                "max_tools": 3,
                "timeout_ms": 6000,
                "description": "Streaming availability and platform focus",
            },
            ExecutionPath.COMPARISON_WORKFLOW: {
                "tools": [
                    "compare_anime_ratings_cross_platform",
                    "correlate_anime_across_platforms",
                    "detect_platform_discrepancies",
                ],
                "strategy": "sequential",
                "max_tools": 3,
                "timeout_ms": 8000,
                "description": "Cross-platform comparison and analysis",
            },
            ExecutionPath.HYBRID_WORKFLOW: {
                "tools": [
                    "anime_semantic_search",
                    "search_anime_anilist",
                    "get_cross_platform_anime_data",
                ],
                "strategy": "adaptive",
                "max_tools": 4,
                "timeout_ms": 7000,
                "description": "Adaptive workflow based on intermediate results",
            },
        }

    async def route_execution(self, context: ExecutionContext) -> Dict[str, Any]:
        """
        Determine optimal execution path based on current context and conditions.

        Args:
            context: Current execution context with query, routing decision, and state

        Returns:
            Execution configuration with selected path and tools
        """
        logger.info(f"Conditional routing for query: '{context.query}'")

        # Evaluate all routing rules
        applicable_rules = self._evaluate_routing_rules(context)

        # Select execution path based on highest priority applicable rule
        selected_path = self._select_execution_path(context, applicable_rules)

        # Generate execution configuration
        execution_config = self._build_execution_config(selected_path, context)

        logger.info(f"Selected execution path: {selected_path.value}")

        return execution_config

    def _evaluate_routing_rules(self, context: ExecutionContext) -> List[RoutingRule]:
        """Evaluate routing rules against current context."""
        applicable_rules = []

        for rule in self.routing_rules:
            if self._check_rule_condition(rule, context):
                applicable_rules.append(rule)

        # Sort by priority (highest first)
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)

        return applicable_rules

    def _check_rule_condition(
        self, rule: RoutingRule, context: ExecutionContext
    ) -> bool:
        """Check if a routing rule condition is met."""
        try:
            if rule.condition == RoutingCondition.QUERY_COMPLEXITY:
                if context.routing_decision is None:
                    return False
                return context.routing_decision.estimated_complexity == rule.threshold

            elif rule.condition == RoutingCondition.PLATFORM_AVAILABILITY:
                available_platforms = sum(
                    1 for status in context.platform_status.values() if status
                )
                availability_ratio = available_platforms / len(context.platform_status)
                return availability_ratio < rule.threshold

            elif rule.condition == RoutingCondition.USER_PREFERENCE:
                user_prefs = (
                    context.user_context.get("preferences", {})
                    if context.user_context
                    else {}
                )
                return user_prefs.get("search_mode") == rule.threshold

            elif rule.condition == RoutingCondition.TOOL_SUCCESS_RATE:
                # Calculate average success rate of primary tools
                if context.routing_decision is None:
                    return False
                primary_tools = context.routing_decision.primary_tools
                success_rates = []
                for tool in primary_tools:
                    platform = self._extract_platform_from_tool(tool)
                    if platform in self.platform_health:
                        success_rates.append(
                            self.platform_health[platform]["success_rate"]
                        )

                if success_rates:
                    avg_success_rate = sum(success_rates) / len(success_rates)
                    return avg_success_rate < rule.threshold
                return False

            elif rule.condition == RoutingCondition.RESPONSE_TIME:
                return context.execution_time_ms > rule.threshold

            elif rule.condition == RoutingCondition.ERROR_THRESHOLD:
                return context.error_count >= rule.threshold

            return False

        except Exception as e:
            logger.warning(f"Error evaluating routing rule {rule.condition}: {e}")
            return False

    def _extract_platform_from_tool(self, tool_name: str) -> str:
        """Extract platform name from tool name."""
        if "mal" in tool_name:
            return "mal"
        elif "anilist" in tool_name:
            return "anilist"
        elif "jikan" in tool_name:
            return "jikan"
        elif "kitsu" in tool_name:
            return "kitsu"
        elif "schedule" in tool_name:
            return "animeschedule"
        elif "semantic" in tool_name:
            return "semantic"
        return "unknown"

    def _select_execution_path(
        self, context: ExecutionContext, applicable_rules: List[RoutingRule]
    ) -> ExecutionPath:
        """Select execution path based on applicable rules and context."""

        # If we have applicable rules, use the highest priority one
        if applicable_rules:
            return applicable_rules[0].target_path

        # Fallback based on routing decision characteristics
        routing_decision = context.routing_decision
        
        if routing_decision is None:
            logger.warning("No routing decision available, using simple search")
            return ExecutionPath.SIMPLE_SEARCH

        # Check for specific intent patterns
        if routing_decision.enrichment_recommended:
            return ExecutionPath.ENRICHED_SEARCH

        if any("compare" in tool for tool in routing_decision.primary_tools):
            return ExecutionPath.COMPARISON_WORKFLOW

        if any("semantic" in tool for tool in routing_decision.primary_tools):
            return ExecutionPath.SEMANTIC_FIRST

        if any("schedule" in tool for tool in routing_decision.primary_tools):
            return ExecutionPath.SCHEDULE_FOCUSED

        if any(
            "streaming" in tool or "kitsu" in tool
            for tool in routing_decision.primary_tools
        ):
            return ExecutionPath.STREAMING_FOCUSED

        # Default based on complexity
        if routing_decision.estimated_complexity == "complex":
            return ExecutionPath.HYBRID_WORKFLOW
        elif routing_decision.estimated_complexity == "medium":
            return ExecutionPath.ENRICHED_SEARCH
        else:
            return ExecutionPath.SIMPLE_SEARCH

    def _build_execution_config(
        self, selected_path: ExecutionPath, context: ExecutionContext
    ) -> Dict[str, Any]:
        """Build execution configuration for selected path."""
        path_config = self.execution_paths[selected_path]

        # Merge with routing decision tools if compatible
        if context.routing_decision is not None:
            routing_tools = context.routing_decision.primary_tools
            path_tools = path_config["tools"]
            # Use path tools but consider routing decision preferences
            final_tools = path_tools.copy()
        else:
            final_tools = path_config["tools"].copy()
            routing_tools = []

        # Add high-priority tools from routing decision if not in path
        for tool in routing_tools[:2]:  # Only add top 2 routing tools
            if tool not in final_tools and len(final_tools) < path_config["max_tools"]:
                final_tools.append(tool)

        # Build final configuration
        execution_config = {
            "execution_path": selected_path.value,
            "tools": final_tools[: path_config["max_tools"]],
            "execution_strategy": path_config["strategy"],
            "timeout_ms": path_config["timeout_ms"],
            "description": path_config["description"],
            "fallback_tools": context.routing_decision.fallback_tools if context.routing_decision else [],
            "max_retries": 2,
            "error_handling": "graceful_degradation",
            "routing_metadata": {
                "original_routing_confidence": context.routing_decision.confidence if context.routing_decision else 0.5,
                "applicable_rules": len(
                    [
                        r
                        for r in self.routing_rules
                        if self._check_rule_condition(r, context)
                    ]
                ),
                "platform_health_score": self._calculate_platform_health_score(),
                "complexity": context.routing_decision.estimated_complexity if context.routing_decision else "medium",
            },
        }

        return execution_config

    def _calculate_platform_health_score(self) -> float:
        """Calculate overall platform health score."""
        if not self.platform_health:
            return 0.0

        total_score = 0.0
        for platform_data in self.platform_health.values():
            availability = 1.0 if platform_data["available"] else 0.0
            success_rate = platform_data["success_rate"]
            response_score = max(
                0.0, 1.0 - (platform_data["avg_response_ms"] / 2000)
            )  # Normalize to 0-1

            platform_score = (
                (availability * 0.4) + (success_rate * 0.4) + (response_score * 0.2)
            )
            total_score += platform_score

        return total_score / len(self.platform_health)

    def update_platform_health(
        self, platform: str, success: bool, response_time_ms: int
    ):
        """Update platform health metrics based on execution results."""
        if platform in self.platform_health:
            health = self.platform_health[platform]

            # Update success rate with exponential moving average
            alpha = 0.1
            current_success = 1.0 if success else 0.0
            health["success_rate"] = (alpha * current_success) + (
                (1 - alpha) * health["success_rate"]
            )

            # Update average response time
            health["avg_response_ms"] = (alpha * response_time_ms) + (
                (1 - alpha) * health["avg_response_ms"]
            )

            # Update availability
            health["available"] = success or health["success_rate"] > 0.5

            logger.debug(
                f"Updated {platform} health: success_rate={health['success_rate']:.2f}, "
                f"avg_response_ms={health['avg_response_ms']:.0f}"
            )

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get current routing statistics and health metrics."""
        return {
            "platform_health": self.platform_health,
            "routing_rules_count": len(self.routing_rules),
            "execution_paths_count": len(self.execution_paths),
            "overall_health_score": self._calculate_platform_health_score(),
            "available_platforms": [
                platform
                for platform, health in self.platform_health.items()
                if health["available"]
            ],
        }
