"""Error handling infrastructure for anime MCP server.

Provides three-layer error context preservation, circuit breaker patterns,
and graceful degradation strategies for robust API integrations.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorContext(BaseModel):
    """Enhanced three-layer error context preservation.

    Provides structured error information at different levels:
    - user_message: Friendly, actionable message for end users
    - debug_info: Technical context for developers
    - trace_data: Complete execution path for debugging
    - correlation_id: Unique identifier for request tracing
    - severity: Error severity level
    - recovery_suggestions: Actionable recovery steps
    - breadcrumbs: Execution path breadcrumbs
    """

    user_message: str = Field(
        description="User-friendly error message that is actionable"
    )
    debug_info: str = Field(description="Technical error information for developers")
    trace_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Complete execution context and trace information",
    )
    correlation_id: str = Field(
        default_factory=lambda: f"err-{uuid.uuid4().hex[:12]}",
        description="Unique correlation ID for request tracing",
    )
    severity: ErrorSeverity = Field(
        default=ErrorSeverity.ERROR, description="Error severity level"
    )
    recovery_suggestions: List[str] = Field(
        default_factory=list, description="Actionable recovery steps for users"
    )
    breadcrumbs: List[Dict[str, Any]] = Field(
        default_factory=list, description="Execution path breadcrumbs for debugging"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error occurrence timestamp",
    )

    def add_breadcrumb(self, step: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Add execution breadcrumb.

        Args:
            step: Name of the execution step
            data: Additional data for the step
        """
        breadcrumb = {
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data or {},
        }
        self.breadcrumbs.append(breadcrumb)

    def is_recoverable(self) -> bool:
        """Check if error is recoverable based on recovery suggestions.

        Returns:
            True if recovery suggestions are provided, False otherwise
        """
        return len(self.recovery_suggestions) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert ErrorContext to dictionary.

        Returns:
            Dictionary representation of ErrorContext
        """
        return {
            "user_message": self.user_message,
            "debug_info": self.debug_info,
            "trace_data": self.trace_data,
            "correlation_id": self.correlation_id,
            "severity": self.severity.value,
            "recovery_suggestions": self.recovery_suggestions,
            "breadcrumbs": self.breadcrumbs,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        user_message: str,
        trace_data: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        recovery_suggestions: Optional[List[str]] = None,
    ) -> "ErrorContext":
        """Create ErrorContext from an exception.

        Args:
            exception: The original exception
            user_message: User-friendly message
            trace_data: Additional trace information
            correlation_id: Optional correlation ID
            severity: Error severity level
            recovery_suggestions: Optional recovery suggestions

        Returns:
            ErrorContext with populated debug info
        """
        debug_info = f"{type(exception).__name__}: {str(exception)}"

        kwargs = {
            "user_message": user_message,
            "debug_info": debug_info,
            "trace_data": trace_data or {},
            "severity": severity,
            "recovery_suggestions": recovery_suggestions or [],
        }

        if correlation_id:
            kwargs["correlation_id"] = correlation_id

        return cls(**kwargs)


class CircuitBreaker:
    """Enhanced circuit breaker pattern for API failure prevention.

    Prevents cascading failures by monitoring API call success rates
    and temporarily blocking requests when failure threshold is exceeded.
    Now includes per-API management and comprehensive metrics.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 300,
        api_name: str = "unknown",
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            api_name: Name of the API this breaker protects
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.api_name = api_name
        self.failure_count = 0
        self.state = "closed"  # closed, open, half_open
        self.last_failure_time = None

        # Enhanced metrics tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.blocked_requests = 0
        self.created_at = datetime.now(timezone.utc)

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == "open"

    async def call_with_breaker(self, func):
        """Execute function with circuit breaker protection.

        Args:
            func: Async function to execute

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        self.total_requests += 1

        # Check if we should transition to half-open
        if self.state == "open" and self._should_attempt_reset():
            self.state = "half_open"
            logger.info(
                f"Circuit breaker for {self.api_name} transitioning to half-open"
            )

        # Block calls if circuit is open
        if self.state == "open":
            self.blocked_requests += 1
            raise Exception("Circuit breaker is open - blocking request")

        try:
            result = await func()

            # Success - reset failure count and close circuit
            self.successful_requests += 1
            if self.state == "half_open":
                self._reset()
                logger.info(
                    f"Circuit breaker for {self.api_name} reset - returning to closed state"
                )

            return result

        except Exception as e:
            self._record_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if not self.last_failure_time:
            return False

        time_since_failure = datetime.now(timezone.utc) - self.last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout

    def _record_failure(self):
        """Record a failure and potentially open the circuit."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"Circuit breaker for {self.api_name} opened after {self.failure_count} failures"
            )

    def _reset(self):
        """Reset circuit breaker to healthy state."""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker metrics.

        Returns:
            Dictionary containing all circuit breaker metrics
        """
        now = datetime.now(timezone.utc)
        uptime_seconds = (now - self.created_at).total_seconds()

        # Calculate error rate
        error_rate = 0.0
        if self.total_requests > 0:
            failed_requests = (
                self.total_requests - self.successful_requests - self.blocked_requests
            )
            error_rate = failed_requests / self.total_requests

        return {
            "api_name": self.api_name,
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "blocked_requests": self.blocked_requests,
            "error_rate": error_rate,
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "recovery_timeout": self.recovery_timeout,
            "uptime_seconds": uptime_seconds,
            "created_at": self.created_at.isoformat(),
        }


class GracefulDegradation:
    """Enhanced 5-level graceful degradation strategies for service failures.

    Provides comprehensive fallback mechanisms when primary services are unavailable:
    Level 1: Primary cache
    Level 2: Secondary cache
    Level 3: Offline database
    Level 4: Minimal response
    Level 5: Error response
    """

    @staticmethod
    def get_degradation_strategies() -> List[Dict[str, Any]]:
        """Get the 5-level degradation strategy configuration.

        Returns:
            List of degradation strategies in order of preference
        """
        return [
            {
                "level": 1,
                "name": "primary_cache",
                "description": "Retrieve data from primary cache",
            },
            {
                "level": 2,
                "name": "secondary_cache",
                "description": "Retrieve data from secondary/backup cache",
            },
            {
                "level": 3,
                "name": "offline_database",
                "description": "Retrieve data from local offline database",
            },
            {
                "level": 4,
                "name": "minimal_response",
                "description": "Provide minimal functionality response",
            },
            {
                "level": 5,
                "name": "error_response",
                "description": "Structured error response with recovery suggestions",
            },
        ]

    @staticmethod
    async def execute_degradation_strategy(
        level: int, context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Execute a specific degradation strategy level.

        Args:
            level: Degradation level (1-5)
            context: Context data for strategy execution

        Returns:
            Strategy result or None if strategy fails

        Raises:
            ValueError: If level is invalid
        """
        if level < 1 or level > 5:
            raise ValueError(f"Invalid degradation level: {level}. Must be 1-5.")

        try:
            if level == 1:
                return await GracefulDegradation._execute_primary_cache_strategy(
                    context
                )
            elif level == 2:
                return await GracefulDegradation._execute_secondary_cache_strategy(
                    context
                )
            elif level == 3:
                return await GracefulDegradation._execute_offline_database_strategy(
                    context
                )
            elif level == 4:
                return await GracefulDegradation._execute_minimal_response_strategy(
                    context
                )
            elif level == 5:
                return await GracefulDegradation._execute_error_response_strategy(
                    context
                )
        except Exception as e:
            logger.warning(f"Degradation strategy level {level} failed: {e}")
            return None

    @staticmethod
    async def execute_degradation_cascade(context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute degradation cascade through all levels until success.

        Args:
            context: Context data for strategy execution

        Returns:
            First successful strategy result or final error response
        """
        for level in range(1, 6):
            result = await GracefulDegradation.execute_degradation_strategy(
                level, context
            )
            if result is not None:
                logger.info(f"Degradation cascade succeeded at level {level}")
                return result

        # This should never happen as level 5 always returns a result
        logger.error("Degradation cascade failed at all levels")
        return {
            "degradation_level": 5,
            "source": "error_response",
            "error": True,
            "error_message": "All degradation strategies failed",
        }

    @staticmethod
    async def _execute_primary_cache_strategy(
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Level 1: Primary cache strategy."""
        cache_manager = context.get("cache_manager")
        cache_key = context.get("cache_key")

        if cache_manager and cache_key and hasattr(cache_manager, "get"):
            cached_data = await cache_manager.get(cache_key)
            if cached_data:
                logger.info(f"Primary cache hit for {cache_key}")
                cached_data["degradation_level"] = 1
                cached_data["source"] = "primary_cache"
                return cached_data

        return None

    @staticmethod
    async def _execute_secondary_cache_strategy(
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Level 2: Secondary cache strategy."""
        secondary_cache = context.get("secondary_cache")
        cache_key = context.get("cache_key")

        if secondary_cache and cache_key and hasattr(secondary_cache, "get"):
            cached_data = await secondary_cache.get(cache_key)
            if cached_data:
                logger.info(f"Secondary cache hit for {cache_key}")
                cached_data["degradation_level"] = 2
                cached_data["source"] = "secondary_cache"
                return cached_data

        return None

    @staticmethod
    async def _execute_offline_database_strategy(
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Level 3: Offline database strategy."""
        anime_id = context.get("anime_id", "unknown")

        # This would integrate with the existing offline database
        result = {
            "anime_id": anime_id,
            "title": "Data temporarily unavailable",
            "synopsis": "Please try again later when services are restored.",
            "degradation_level": 3,
            "source": "offline_database",
            "degraded": True,
        }

        logger.info(f"Using offline database fallback for anime {anime_id}")
        return result

    @staticmethod
    async def _execute_minimal_response_strategy(
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Level 4: Minimal response strategy."""
        anime_id = context.get("anime_id", "unknown")

        result = {
            "anime_id": anime_id,
            "title": "Limited functionality available",
            "synopsis": "Basic service is operating with reduced features.",
            "degradation_level": 4,
            "source": "minimal_response",
            "limited_functionality": True,
        }

        # Include any context data that might be useful
        for key, value in context.items():
            if (
                key not in result
                and key != "cache_manager"
                and key != "secondary_cache"
            ):
                result[key] = value

        logger.info(f"Using minimal response for anime {anime_id}")
        return result

    @staticmethod
    async def _execute_error_response_strategy(
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Level 5: Error response strategy."""
        anime_id = context.get("anime_id", "unknown")
        error_message = context.get("error_message", "Service temporarily unavailable")

        result = {
            "anime_id": anime_id,
            "title": "Service unavailable",
            "synopsis": "All services are currently unavailable. Please try again later.",
            "degradation_level": 5,
            "source": "error_response",
            "error": True,
            "error_message": error_message,
            "recovery_suggestions": [
                "Try again in a few minutes",
                "Check your internet connection",
                "Contact support if the problem persists",
            ],
        }

        logger.warning(f"Using error response for anime {anime_id}: {error_message}")
        return result

    @staticmethod
    async def fallback_to_cache(
        cache_key: str, cache_manager
    ) -> Optional[Dict[str, Any]]:
        """Attempt to retrieve cached data as fallback.

        Args:
            cache_key: Key to look up in cache
            cache_manager: Cache management instance

        Returns:
            Cached data if available, None otherwise
        """
        try:
            if hasattr(cache_manager, "get"):
                cached_data = await cache_manager.get(cache_key)
                if cached_data:
                    logger.info(f"Using cached fallback for {cache_key}")
                    return cached_data
        except Exception as e:
            logger.warning(f"Cache fallback failed: {e}")

        return None

    @staticmethod
    async def fallback_to_offline_data(anime_id: str) -> Dict[str, Any]:
        """Fallback to offline anime database.

        Args:
            anime_id: ID of anime to retrieve

        Returns:
            Basic anime data from offline sources
        """
        # This would integrate with the existing offline database
        return {
            "anime_id": anime_id,
            "title": "Data temporarily unavailable",
            "synopsis": "Please try again later when services are restored.",
            "source": "offline_fallback",
            "degraded": True,
        }


class LangGraphErrorHandler:
    """LangGraph-specific error handling for workflow orchestration.

    Handles the 6 primary LangGraph error patterns:
    1. Node execution failures
    2. Workflow state corruption
    3. Tool invocation errors
    4. Memory management errors
    5. Agent recursion limits
    6. Workflow timeouts

    This class provides foundational infrastructure for future LangGraph integration.
    """

    def __init__(self):
        """Initialize LangGraph error handler."""
        self.error_patterns = self.get_langgraph_error_patterns()
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.correlation_tracker = {}  # Track workflow correlations
        self.error_history = {}  # Track error patterns per workflow
        self.circuit_breaker_thresholds = {
            "node_execution_failure": 3,
            "tool_invocation_error": 5,
            "workflow_timeout": 2,
            "memory_management_error": 2,
            "agent_recursion_limit": 1,
            "workflow_state_corruption": 1,
        }

    @staticmethod
    def get_langgraph_error_patterns() -> List[Dict[str, Any]]:
        """Get the 6 LangGraph-specific error patterns.

        Returns:
            List of error patterns with metadata
        """
        return [
            {
                "name": "node_execution_failure",
                "description": "Individual node fails during workflow execution",
                "severity": "high",
                "recoverable": True,
                "common_causes": [
                    "API timeouts",
                    "validation errors",
                    "resource unavailable",
                ],
            },
            {
                "name": "workflow_state_corruption",
                "description": "Workflow state becomes inconsistent or corrupted",
                "severity": "critical",
                "recoverable": True,
                "common_causes": [
                    "concurrent modifications",
                    "schema mismatch",
                    "serialization errors",
                ],
            },
            {
                "name": "tool_invocation_error",
                "description": "Error invoking external tools or functions",
                "severity": "medium",
                "recoverable": True,
                "common_causes": [
                    "tool not found",
                    "parameter mismatch",
                    "tool execution failure",
                ],
            },
            {
                "name": "memory_management_error",
                "description": "Memory limits exceeded or memory corruption",
                "severity": "high",
                "recoverable": True,
                "common_causes": [
                    "memory leaks",
                    "large datasets",
                    "inefficient algorithms",
                ],
            },
            {
                "name": "agent_recursion_limit",
                "description": "Agent exceeds maximum recursion depth",
                "severity": "medium",
                "recoverable": True,
                "common_causes": [
                    "infinite loops",
                    "circular dependencies",
                    "deep call stacks",
                ],
            },
            {
                "name": "workflow_timeout",
                "description": "Workflow execution exceeds time limits",
                "severity": "medium",
                "recoverable": True,
                "common_causes": [
                    "slow APIs",
                    "complex computations",
                    "resource contention",
                ],
            },
        ]

    def _initialize_recovery_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize recovery strategies for each error pattern."""
        return {
            "node_execution_failure": {
                "name": "retry_with_backoff",
                "max_retries": 3,
                "backoff_multiplier": 2.0,
                "fallback_action": "skip_node",
            },
            "workflow_state_corruption": {
                "name": "restore_state",
                "backup_retention": 5,
                "validation_enabled": True,
                "fallback_action": "restart_workflow",
            },
            "tool_invocation_error": {
                "name": "fallback_tool",
                "alternative_tools": True,
                "parameter_adaptation": True,
                "fallback_action": "manual_intervention",
            },
            "memory_management_error": {
                "name": "cleanup_memory",
                "strategy": "lru_eviction",
                "threshold_percentage": 80,
                "fallback_action": "reduce_complexity",
            },
            "agent_recursion_limit": {
                "name": "break_recursion",
                "strategy": "intermediate_result",
                "max_depth": 25,
                "fallback_action": "simplified_execution",
            },
            "workflow_timeout": {
                "name": "partial_completion",
                "preserve_partial_results": True,
                "extend_timeout": False,
                "fallback_action": "async_completion",
            },
        }

    async def handle_node_execution_failure(
        self, error_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle node execution failure with retry and fallback strategies."""
        result = {
            "error_type": "node_execution_failure",
            "node_name": error_context.get("node_name"),
            "workflow_id": error_context.get("workflow_id"),
            "recoverable": True,
            "suggested_action": "retry_with_backoff",
            "recovery_steps": [
                "Implement exponential backoff",
                "Check node dependencies",
                "Validate input parameters",
                "Consider alternative execution path",
            ],
            "max_retries": 3,
            "current_retry": error_context.get("retry_count", 0),
        }

        # Track the error for pattern analysis
        self.track_workflow_error(
            {
                "workflow_id": error_context.get("workflow_id"),
                "error_type": "node_execution_failure",
                "node_name": error_context.get("node_name"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return result

    async def handle_workflow_state_corruption(
        self, error_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle workflow state corruption with restoration strategies."""
        result = {
            "error_type": "workflow_state_corruption",
            "workflow_id": error_context.get("workflow_id"),
            "recoverable": True,
            "suggested_action": "restore_state",
            "state_backup_available": True,
            "corruption_source": error_context.get("corruption_source"),
            "recovery_steps": [
                "Validate current state schema",
                "Restore from last known good state",
                "Re-execute from checkpoint",
                "Update state validation rules",
            ],
        }

        self.track_workflow_error(
            {
                "workflow_id": error_context.get("workflow_id"),
                "error_type": "workflow_state_corruption",
                "corruption_source": error_context.get("corruption_source"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return result

    async def handle_tool_invocation_error(
        self, error_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle tool invocation errors with fallback mechanisms."""
        result = {
            "error_type": "tool_invocation_error",
            "tool_name": error_context.get("tool_name"),
            "workflow_id": error_context.get("workflow_id"),
            "recoverable": True,
            "suggested_action": "fallback_tool",
            "alternative_tools": [
                "search_anime_alternative",
                "cached_search_fallback",
                "manual_search_prompt",
            ],
            "recovery_steps": [
                "Validate tool parameters",
                "Check tool availability",
                "Use alternative tool",
                "Adapt parameters for fallback",
            ],
        }

        self.track_workflow_error(
            {
                "workflow_id": error_context.get("workflow_id"),
                "error_type": "tool_invocation_error",
                "tool_name": error_context.get("tool_name"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return result

    async def handle_memory_management_error(
        self, error_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle memory management errors with cleanup strategies."""
        result = {
            "error_type": "memory_management_error",
            "workflow_id": error_context.get("workflow_id"),
            "recoverable": True,
            "suggested_action": "cleanup_memory",
            "memory_cleanup_strategy": "lru_eviction",
            "memory_type": error_context.get("memory_type"),
            "current_size": error_context.get("memory_size"),
            "recovery_steps": [
                "Implement LRU eviction",
                "Compress memory objects",
                "Offload to persistent storage",
                "Reduce workflow complexity",
            ],
        }

        self.track_workflow_error(
            {
                "workflow_id": error_context.get("workflow_id"),
                "error_type": "memory_management_error",
                "memory_type": error_context.get("memory_type"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return result

    async def handle_agent_recursion_limit(
        self, error_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle agent recursion limit errors with recursion breaking."""
        result = {
            "error_type": "agent_recursion_limit",
            "workflow_id": error_context.get("workflow_id"),
            "recoverable": True,
            "suggested_action": "break_recursion",
            "recursion_break_strategy": "intermediate_result",
            "agent_name": error_context.get("agent_name"),
            "recursion_depth": error_context.get("recursion_depth"),
            "max_depth": error_context.get("max_recursion", 25),
            "recovery_steps": [
                "Return intermediate result",
                "Implement depth limiting",
                "Detect recursion patterns",
                "Use iterative approach",
            ],
        }

        self.track_workflow_error(
            {
                "workflow_id": error_context.get("workflow_id"),
                "error_type": "agent_recursion_limit",
                "agent_name": error_context.get("agent_name"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return result

    async def handle_workflow_timeout(
        self, error_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle workflow timeout errors with partial completion."""
        result = {
            "error_type": "workflow_timeout",
            "workflow_id": error_context.get("workflow_id"),
            "recoverable": True,
            "suggested_action": "partial_completion",
            "partial_results_available": True,
            "timeout_seconds": error_context.get("timeout_seconds"),
            "execution_time": error_context.get("execution_time"),
            "last_completed_node": error_context.get("last_completed_node"),
            "recovery_steps": [
                "Preserve partial results",
                "Continue execution asynchronously",
                "Optimize workflow performance",
                "Implement streaming results",
            ],
        }

        self.track_workflow_error(
            {
                "workflow_id": error_context.get("workflow_id"),
                "error_type": "workflow_timeout",
                "execution_time": error_context.get("execution_time"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return result

    def create_workflow_error_context(
        self, error: Exception, workflow_state: Dict[str, Any]
    ) -> ErrorContext:
        """Create workflow-specific error context with correlation tracking."""
        workflow_id = workflow_state.get("workflow_id", "unknown")
        correlation_id = f"{workflow_id}-{uuid.uuid4().hex[:8]}"

        # Create user-friendly message based on workflow context
        current_node = workflow_state.get("current_node", "unknown")
        user_message = f"Workflow execution failed at {current_node}. Please try again."

        # Extract relevant workflow information for debugging
        trace_data = {
            "workflow_id": workflow_id,
            "current_node": current_node,
            "execution_step": workflow_state.get("execution_step"),
            "user_query": workflow_state.get("user_query", ""),
            "workflow_type": workflow_state.get("workflow_type", "anime_search"),
        }

        return ErrorContext.from_exception(
            exception=error,
            user_message=user_message,
            trace_data=trace_data,
            correlation_id=correlation_id,
            severity=ErrorSeverity.ERROR,
            recovery_suggestions=[
                "Try the request again",
                "Simplify your search query",
                "Contact support if the problem persists",
            ],
        )

    def get_recovery_strategy(self, error_type: str) -> Dict[str, Any]:
        """Get recovery strategy for specific error type."""
        return self.recovery_strategies.get(
            error_type,
            {
                "name": "default_recovery",
                "max_retries": 1,
                "fallback_action": "manual_intervention",
            },
        )

    def track_workflow_error(self, error_info: Dict[str, Any]) -> None:
        """Track workflow error for pattern analysis and circuit breaker logic."""
        workflow_id = error_info.get("workflow_id", "unknown")

        if workflow_id not in self.error_history:
            self.error_history[workflow_id] = []

        # Add timestamp if not provided
        if "timestamp" not in error_info:
            error_info["timestamp"] = datetime.now(timezone.utc).isoformat()

        self.error_history[workflow_id].append(error_info)

        # Keep only recent errors (last 100 per workflow)
        if len(self.error_history[workflow_id]) > 100:
            self.error_history[workflow_id] = self.error_history[workflow_id][-100:]

    def get_workflow_error_history(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get error history for specific workflow."""
        return self.error_history.get(workflow_id, [])

    def get_workflow_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Analyze error patterns across all workflows."""
        patterns = {}

        for workflow_id, errors in self.error_history.items():
            for error in errors:
                error_type = error.get("error_type", "unknown")
                if error_type not in patterns:
                    patterns[error_type] = {"count": 0, "workflows": set()}

                patterns[error_type]["count"] += 1
                patterns[error_type]["workflows"].add(workflow_id)

        # Convert sets to lists for JSON serialization
        for pattern in patterns.values():
            pattern["workflows"] = list(pattern["workflows"])

        return patterns

    def should_trigger_circuit_breaker(self, workflow_id: str) -> bool:
        """Determine if circuit breaker should be triggered for workflow."""
        errors = self.get_workflow_error_history(workflow_id)

        # Count recent errors (last 5 minutes)
        recent_threshold = datetime.now(timezone.utc) - timedelta(minutes=5)
        recent_errors = []

        for error in errors:
            try:
                error_time = datetime.fromisoformat(error.get("timestamp", ""))
                if error_time.replace(tzinfo=timezone.utc) > recent_threshold:
                    recent_errors.append(error)
            except (ValueError, TypeError):
                continue

        # Check if any error type exceeds threshold
        error_counts = {}
        for error in recent_errors:
            error_type = error.get("error_type", "unknown")
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        for error_type, count in error_counts.items():
            threshold = self.circuit_breaker_thresholds.get(error_type, 5)
            if count >= threshold:
                return True

        return False


class TraceContext:
    """Context manager for execution tracing."""

    def __init__(
        self,
        tracer: "ExecutionTracer",
        trace_id: str,
        operation: str = None,
        context: Dict[str, Any] = None,
    ):
        """Initialize trace context.

        Args:
            tracer: ExecutionTracer instance
            trace_id: Unique trace identifier
            operation: Operation name (for context manager)
            context: Initial context data (for context manager)
        """
        self.tracer = tracer
        self.trace_id = trace_id
        self.operation = operation
        self.context = context

    async def add_step(
        self,
        step_name: str,
        step_data: Dict[str, Any] = None,
        step_status: str = "completed",
    ) -> None:
        """Add step to the trace."""
        await self.tracer.add_trace_step(
            self.trace_id, step_name, step_data or {}, step_status
        )

    async def __aenter__(self):
        """Enter async context manager."""
        # Start the trace when entering context
        if self.operation:
            await self.tracer.start_trace(self.operation, self.context, self.trace_id)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        if exc_type is not None:
            # Exception occurred, end trace with failure
            await self.tracer.end_trace(self.trace_id, status="failure", error=exc_val)
        else:
            # No exception, end trace with success
            await self.tracer.end_trace(self.trace_id, status="success")


class ExecutionTracer:
    """Comprehensive execution tracing for debugging and performance analysis.

    Provides detailed tracking of execution flows, performance metrics,
    and debugging information across complex operations. Essential for
    monitoring distributed systems and workflow orchestration.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize execution tracer.

        Args:
            config: Optional configuration for tracer behavior
        """
        self.traces = {}  # Completed traces
        self.active_traces = {}  # Currently running traces
        self.performance_metrics = {}  # Aggregated performance data

        # Default configuration
        default_config = {
            "max_trace_history": 1000,
            "performance_tracking": True,
            "detailed_logging": True,
            "auto_cleanup": True,
            "export_format": "json",
        }

        self.trace_configs = {**default_config, **(config or {})}
        self.max_trace_history = self.trace_configs["max_trace_history"]

    async def start_trace(
        self, operation: str, context: Dict[str, Any], trace_id: Optional[str] = None
    ) -> str:
        """Start a new execution trace.

        Args:
            operation: Name of the operation being traced
            context: Initial context data for the trace
            trace_id: Optional custom trace ID

        Returns:
            Unique trace identifier
        """
        if trace_id is None:
            trace_id = f"trace-{uuid.uuid4().hex[:12]}"

        start_time = datetime.now(timezone.utc)

        trace_data = {
            "trace_id": trace_id,
            "operation": operation,
            "context": context.copy(),
            "status": "active",
            "start_time": start_time,
            "steps": [],
            "performance_data": {"start_timestamp": start_time.isoformat()},
        }

        self.active_traces[trace_id] = trace_data

        logger.debug(f"Started trace {trace_id} for operation: {operation}")
        return trace_id

    async def add_trace_step(
        self,
        trace_id: str,
        step_name: str,
        step_data: Dict[str, Any] = None,
        step_status: str = "completed",
    ) -> None:
        """Add a step to an active trace.

        Args:
            trace_id: Trace identifier
            step_name: Name of the step
            step_data: Data associated with the step
            step_status: Status of the step (completed, failed, in_progress)
        """
        if trace_id not in self.active_traces:
            logger.warning(f"Attempted to add step to non-existent trace: {trace_id}")
            return

        step = {
            "step_name": step_name,
            "step_data": step_data or {},
            "step_status": step_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.active_traces[trace_id]["steps"].append(step)

        logger.debug(f"Added step '{step_name}' to trace {trace_id}")

    async def end_trace(
        self,
        trace_id: str,
        status: str = "success",
        result: Dict[str, Any] = None,
        error: Exception = None,
        performance_metrics: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """End an active trace.

        Args:
            trace_id: Trace identifier
            status: Final status (success, failure, timeout)
            result: Result data from the operation
            error: Exception if operation failed
            performance_metrics: Performance data

        Returns:
            Completed trace data
        """
        if trace_id not in self.active_traces:
            logger.warning(f"Attempted to end non-existent trace: {trace_id}")
            return {}

        trace = self.active_traces[trace_id]
        end_time = datetime.now(timezone.utc)

        # Calculate duration
        start_time = trace["start_time"]
        duration = (end_time - start_time).total_seconds() * 1000  # milliseconds

        # Update trace with completion data
        trace.update(
            {
                "status": status,
                "end_time": end_time,
                "total_duration_ms": duration,
                "result": result or {},
                "performance_metrics": performance_metrics or {},
            }
        )

        if error:
            trace["error"] = str(error)
            trace["error_type"] = type(error).__name__

        # Move from active to completed traces
        self.traces[trace_id] = trace
        del self.active_traces[trace_id]

        # Update performance metrics
        self._update_performance_metrics(trace)

        # Auto-cleanup if enabled
        if self.trace_configs.get("auto_cleanup", True):
            self._auto_cleanup_traces()

        logger.debug(
            f"Ended trace {trace_id} with status: {status}, duration: {duration:.2f}ms"
        )

        return trace

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get trace data by ID.

        Args:
            trace_id: Trace identifier

        Returns:
            Trace data or None if not found
        """
        # Check active traces first
        if trace_id in self.active_traces:
            return self.active_traces[trace_id]

        # Check completed traces
        return self.traces.get(trace_id)

    def get_active_traces(self) -> List[Dict[str, Any]]:
        """Get all active traces.

        Returns:
            List of active trace data
        """
        return list(self.active_traces.values())

    def get_traces_by_operation(self, operation: str) -> List[Dict[str, Any]]:
        """Get traces filtered by operation type.

        Args:
            operation: Operation name to filter by

        Returns:
            List of traces matching the operation
        """
        matching_traces = []

        # Check completed traces
        for trace in self.traces.values():
            if trace.get("operation") == operation:
                matching_traces.append(trace)

        # Check active traces
        for trace in self.active_traces.values():
            if trace.get("operation") == operation:
                matching_traces.append(trace)

        return matching_traces

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all traces.

        Returns:
            Performance metrics summary
        """
        if not self.traces:
            return {
                "total_traces": 0,
                "successful_traces": 0,
                "failed_traces": 0,
                "average_duration_ms": 0,
                "min_duration_ms": 0,
                "max_duration_ms": 0,
            }

        total_traces = len(self.traces)
        successful_traces = sum(
            1 for t in self.traces.values() if t.get("status") == "success"
        )
        failed_traces = total_traces - successful_traces

        durations = [t.get("total_duration_ms", 0) for t in self.traces.values()]

        return {
            "total_traces": total_traces,
            "successful_traces": successful_traces,
            "failed_traces": failed_traces,
            "average_duration_ms": sum(durations) / len(durations) if durations else 0,
            "min_duration_ms": min(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
        }

    def cleanup_old_traces(self) -> None:
        """Clean up old traces to maintain memory limits."""
        if len(self.traces) <= self.max_trace_history:
            return

        # Sort traces by end time and keep the most recent ones
        sorted_traces = sorted(
            self.traces.items(),
            key=lambda x: x[1].get(
                "end_time", datetime.min.replace(tzinfo=timezone.utc)
            ),
        )

        # Keep only the most recent traces
        traces_to_keep = sorted_traces[-self.max_trace_history :]
        self.traces = dict(traces_to_keep)

        logger.debug(f"Cleaned up old traces, keeping {len(self.traces)} most recent")

    def _update_performance_metrics(self, trace: Dict[str, Any]) -> None:
        """Update aggregated performance metrics."""
        operation = trace.get("operation", "unknown")
        duration = trace.get("total_duration_ms", 0)
        status = trace.get("status", "unknown")

        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_duration_ms": 0,
                "avg_duration_ms": 0,
                "min_duration_ms": float("inf"),
                "max_duration_ms": 0,
            }

        metrics = self.performance_metrics[operation]
        metrics["total_calls"] += 1

        if status == "success":
            metrics["successful_calls"] += 1
        else:
            metrics["failed_calls"] += 1

        metrics["total_duration_ms"] += duration
        metrics["avg_duration_ms"] = (
            metrics["total_duration_ms"] / metrics["total_calls"]
        )
        metrics["min_duration_ms"] = min(metrics["min_duration_ms"], duration)
        metrics["max_duration_ms"] = max(metrics["max_duration_ms"], duration)

    def _auto_cleanup_traces(self) -> None:
        """Automatically cleanup traces if needed."""
        if len(self.traces) > self.max_trace_history * 1.2:  # 20% buffer
            self.cleanup_old_traces()

    def trace(self, operation: str, context: Dict[str, Any] = None) -> TraceContext:
        """Create trace context manager.

        Args:
            operation: Operation name
            context: Initial context data

        Returns:
            TraceContext for use in async with statement
        """
        # Generate trace ID and create context
        trace_id = f"trace-{uuid.uuid4().hex[:12]}"

        # Create context that will start trace on __aenter__
        return TraceContext(self, trace_id, operation, context or {})

    def export_traces(self, operation_filter: Optional[str] = None) -> Dict[str, Any]:
        """Export traces for analysis or storage.

        Args:
            operation_filter: Optional operation filter

        Returns:
            Exported trace data with summary
        """
        traces_to_export = list(self.traces.values())

        if operation_filter:
            traces_to_export = [
                t for t in traces_to_export if t.get("operation") == operation_filter
            ]

        # Convert datetime objects to ISO strings for JSON serialization
        serializable_traces = []
        for trace in traces_to_export:
            serializable_trace = trace.copy()

            # Convert datetime objects to ISO strings
            if "start_time" in serializable_trace:
                serializable_trace["start_time"] = serializable_trace[
                    "start_time"
                ].isoformat()
            if "end_time" in serializable_trace:
                serializable_trace["end_time"] = serializable_trace[
                    "end_time"
                ].isoformat()

            serializable_traces.append(serializable_trace)

        summary = self.get_performance_summary()

        return {
            "traces": serializable_traces,
            "summary": summary,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_exported": len(serializable_traces),
            "operation_filter": operation_filter,
        }


class CorrelationContext:
    """Context manager for correlation logging."""

    def __init__(
        self, logger, correlation_id: str, operation: str, context: Dict[str, Any]
    ):
        """Initialize correlation context.

        Args:
            logger: CorrelationLogger instance
            correlation_id: Correlation ID for this context
            operation: Operation name
            context: Initial context data
        """
        self.logger = logger
        self.correlation_id = correlation_id
        self.operation = operation
        self.context = context
        self.start_time = None

    async def __aenter__(self):
        """Enter correlation context."""
        self.start_time = datetime.now(timezone.utc)

        # Log context entry
        await self.logger.log_with_correlation(
            self.correlation_id,
            "info",
            f"Started operation: {self.operation}",
            {**self.context, "context_event": "enter", "operation": self.operation},
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit correlation context."""
        end_time = datetime.now(timezone.utc)
        duration_ms = (
            (end_time - self.start_time).total_seconds() * 1000
            if self.start_time
            else 0
        )

        if exc_type is not None:
            # Log error exit
            await self.logger.log_with_correlation(
                self.correlation_id,
                "error",
                f"Operation {self.operation} failed: {str(exc_val)}",
                {
                    **self.context,
                    "context_event": "error_exit",
                    "operation": self.operation,
                    "error_type": exc_type.__name__,
                    "duration_ms": duration_ms,
                },
                error_details={"exception": str(exc_val), "type": exc_type.__name__},
            )
        else:
            # Log successful exit
            await self.logger.log_with_correlation(
                self.correlation_id,
                "info",
                f"Completed operation: {self.operation}",
                {
                    **self.context,
                    "context_event": "exit",
                    "operation": self.operation,
                    "duration_ms": duration_ms,
                },
            )

    async def log(
        self, level: str, message: str, context: Optional[Dict[str, Any]] = None
    ):
        """Log within this correlation context.

        Args:
            level: Log level
            message: Log message
            context: Additional context data
        """
        combined_context = {**self.context}
        if context:
            combined_context.update(context)

        await self.logger.log_with_correlation(
            self.correlation_id, level, message, combined_context
        )


class CorrelationLogger:
    """Structured correlation logging for request tracing and debugging.

    Provides comprehensive logging with correlation ID tracking, chain relationships,
    filtering capabilities, and performance metrics for deep system observability.
    """

    def __init__(self, max_logs_in_memory: int = 10000):
        """Initialize correlation logger.

        Args:
            max_logs_in_memory: Maximum number of logs to keep in memory
        """
        self.logs: List[Dict[str, Any]] = []
        self.correlation_chains: Dict[str, Dict[str, Any]] = {}
        self.log_filters: Dict[str, Dict[str, Any]] = {}
        self.correlation_metrics: Dict[str, Any] = {}
        self.max_logs_in_memory = max_logs_in_memory
        self._lock = asyncio.Lock()

    async def log_with_correlation(
        self,
        correlation_id: str,
        level: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        parent_correlation_id: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log with correlation ID tracking.

        Args:
            correlation_id: Unique correlation identifier
            level: Log level (debug, info, warning, error, critical)
            message: Log message
            context: Additional context data
            parent_correlation_id: Parent correlation ID for chain building
            error_details: Additional error details for error-level logs
        """
        async with self._lock:
            # Create log entry
            log_entry = {
                "correlation_id": correlation_id,
                "level": level,
                "message": message,
                "context": context or {},
                "timestamp": datetime.now(timezone.utc),
                "error_details": error_details or {},
            }

            # Add to logs list
            self.logs.append(log_entry)

            # Manage memory by removing old logs if needed
            if len(self.logs) > self.max_logs_in_memory:
                # Remove oldest 10% of logs
                remove_count = self.max_logs_in_memory // 10
                self.logs = self.logs[remove_count:]

            # Handle correlation chain
            if parent_correlation_id:
                self._build_correlation_chain(correlation_id, parent_correlation_id)
            elif correlation_id not in self.correlation_chains:
                # Initialize chain entry for new root correlation
                self.correlation_chains[correlation_id] = {
                    "correlation_id": correlation_id,
                    "parent": None,
                    "children": [],
                    "created_at": datetime.now(timezone.utc),
                }

    def _build_correlation_chain(self, child_id: str, parent_id: str) -> None:
        """Build correlation chain relationships.

        Args:
            child_id: Child correlation ID
            parent_id: Parent correlation ID
        """
        # Create child entry
        self.correlation_chains[child_id] = {
            "correlation_id": child_id,
            "parent": parent_id,
            "children": [],
            "created_at": datetime.now(timezone.utc),
        }

        # Update parent to include child
        if parent_id not in self.correlation_chains:
            self.correlation_chains[parent_id] = {
                "correlation_id": parent_id,
                "parent": None,
                "children": [],
                "created_at": datetime.now(timezone.utc),
            }

        if child_id not in self.correlation_chains[parent_id]["children"]:
            self.correlation_chains[parent_id]["children"].append(child_id)

    def get_correlation_chain(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        """Get correlation chain information for a given ID.

        Args:
            correlation_id: Correlation ID to look up

        Returns:
            Chain information or None if not found
        """
        return self.correlation_chains.get(correlation_id)

    def get_logs_by_correlation(
        self, correlation_id: str, include_chain: bool = False
    ) -> List[Dict[str, Any]]:
        """Get all logs for a specific correlation ID.

        Args:
            correlation_id: Correlation ID to filter by
            include_chain: Whether to include logs from child correlations

        Returns:
            List of matching log entries
        """
        correlation_ids = {correlation_id}

        if include_chain:
            # Add all child correlation IDs
            def add_children(corr_id: str):
                chain = self.correlation_chains.get(corr_id)
                if chain:
                    for child_id in chain["children"]:
                        correlation_ids.add(child_id)
                        add_children(child_id)  # Recursive for nested chains

            add_children(correlation_id)

        # Filter and sort logs
        matching_logs = [
            log for log in self.logs if log["correlation_id"] in correlation_ids
        ]

        # Sort by timestamp
        return sorted(matching_logs, key=lambda x: x["timestamp"])

    def add_log_filter(
        self,
        filter_name: str,
        level: Optional[str] = None,
        context_filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a log filter for selective log retrieval.

        Args:
            filter_name: Unique name for the filter
            level: Log level to filter by
            context_filters: Context field filters
        """
        self.log_filters[filter_name] = {
            "level": level,
            "context_filters": context_filters or {},
        }

    def get_filtered_logs(self, filter_name: str) -> List[Dict[str, Any]]:
        """Get logs matching a specific filter.

        Args:
            filter_name: Name of the filter to apply

        Returns:
            List of matching log entries
        """
        if filter_name not in self.log_filters:
            return []

        filter_config = self.log_filters[filter_name]
        filtered_logs = []

        for log in self.logs:
            # Check level filter
            if filter_config["level"] and log["level"] != filter_config["level"]:
                continue

            # Check context filters
            context_match = True
            for key, value in filter_config["context_filters"].items():
                if key not in log["context"] or log["context"][key] != value:
                    context_match = False
                    break

            if context_match:
                filtered_logs.append(log)

        return filtered_logs

    def get_correlation_metrics(self) -> Dict[str, Any]:
        """Get metrics about correlation logging activity.

        Returns:
            Dictionary containing various metrics
        """
        if not self.logs:
            return {
                "total_correlations": 0,
                "total_logs": 0,
                "log_levels": {},
                "average_logs_per_correlation": 0.0,
                "most_active_correlations": [],
            }

        # Count unique correlations
        correlation_ids = set(log["correlation_id"] for log in self.logs)

        # Count logs per level
        level_counts = {}
        for log in self.logs:
            level = log["level"]
            level_counts[level] = level_counts.get(level, 0) + 1

        # Count logs per correlation
        correlation_log_counts = {}
        for log in self.logs:
            corr_id = log["correlation_id"]
            correlation_log_counts[corr_id] = correlation_log_counts.get(corr_id, 0) + 1

        # Most active correlations
        most_active = sorted(
            correlation_log_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return {
            "total_correlations": len(correlation_ids),
            "total_logs": len(self.logs),
            "log_levels": level_counts,
            "average_logs_per_correlation": len(self.logs) / len(correlation_ids),
            "most_active_correlations": [
                {"correlation_id": corr_id, "log_count": count}
                for corr_id, count in most_active
            ],
        }

    def clear_old_logs(self, max_age_seconds: int = 3600) -> int:
        """Clear logs older than specified age.

        Args:
            max_age_seconds: Maximum age of logs to keep in seconds

        Returns:
            Number of logs cleared
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
        initial_count = len(self.logs)

        # Filter out old logs
        self.logs = [log for log in self.logs if log["timestamp"] > cutoff_time]

        return initial_count - len(self.logs)

    def export_correlation_logs(self, correlation_id: str) -> Dict[str, Any]:
        """Export all logs and chain information for a correlation.

        Args:
            correlation_id: Correlation ID to export

        Returns:
            Exported data with logs and chain information
        """
        logs = self.get_logs_by_correlation(correlation_id, include_chain=True)
        chain_info = self.get_correlation_chain(correlation_id)

        # Convert datetime objects to ISO strings for serialization
        serializable_logs = []
        for log in logs:
            serializable_log = log.copy()
            serializable_log["timestamp"] = log["timestamp"].isoformat()
            serializable_logs.append(serializable_log)

        return {
            "correlation_id": correlation_id,
            "logs": serializable_logs,
            "chain_info": chain_info,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_logs": len(logs),
        }

    def correlation_context(
        self,
        correlation_id: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> CorrelationContext:
        """Create correlation context manager for automatic logging.

        Args:
            correlation_id: Correlation ID for this context
            operation: Operation name
            context: Initial context data

        Returns:
            CorrelationContext manager
        """
        return CorrelationContext(self, correlation_id, operation, context or {})

    def get_performance_metrics(self, correlation_id: str) -> Dict[str, Any]:
        """Get performance metrics for a specific correlation.

        Args:
            correlation_id: Correlation ID to analyze

        Returns:
            Performance metrics dictionary
        """
        logs = self.get_logs_by_correlation(correlation_id)

        if not logs:
            return {
                "correlation_id": correlation_id,
                "total_logs": 0,
                "performance_data": None,
            }

        # Extract performance data from logs
        performance_logs = [log for log in logs if "performance" in log["context"]]

        return {
            "correlation_id": correlation_id,
            "total_logs": len(logs),
            "performance_logs": len(performance_logs),
            "first_log": logs[0]["timestamp"].isoformat(),
            "last_log": logs[-1]["timestamp"].isoformat(),
            "duration_seconds": (
                logs[-1]["timestamp"] - logs[0]["timestamp"]
            ).total_seconds(),
        }
