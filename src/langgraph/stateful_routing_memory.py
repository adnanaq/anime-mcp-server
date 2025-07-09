"""
Stateful Routing Memory and Context Learning for LangGraph workflows.

This module implements Task #89: Stateful Routing Memory and Context Learning.
Provides conversation context memory across user sessions, agent handoff sequence
learning and optimization, query pattern embedding and similarity matching, and
user preference learning for personalized routing.

Architecture:
- RoutingMemoryStore: Core memory management with vector-based pattern matching
- ConversationContextMemory: User session persistence and preference learning
- AgentHandoffOptimizer: Agent sequence learning and optimization patterns
- UserPreferenceLearner: Personalized routing based on historical interactions
- StatefulRoutingEngine: Main orchestrator that integrates with existing workflows

Integration Points:
- ReactAgentWorkflowEngine: Add STATEFUL execution mode
- IntelligentRouter: Enhance with historical pattern matching
- SwarmAgents: Dynamic handoff optimization based on success patterns
- MemorySaver: Leverage existing LangGraph memory with extended capabilities
"""

import hashlib
import logging
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

from langgraph.checkpoint.memory import MemorySaver

from ..config import get_settings
from .workflow_state import QueryIntent, WorkflowResult

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """Routing strategies for stateful routing decisions."""

    LEARNED_OPTIMAL = "learned_optimal"  # Use learned optimal agent sequences
    PREFERENCE_BASED = "preference_based"  # Route based on user preferences
    PATTERN_MATCHING = "pattern_matching"  # Match against historical query patterns
    ADAPTIVE_HYBRID = "adaptive_hybrid"  # Combine multiple strategies dynamically
    FALLBACK_STANDARD = "fallback_standard"  # Standard routing when learning fails


class MemoryScope(str, Enum):
    """Scope of memory persistence for different memory types."""

    SESSION = "session"  # Memory persists for current conversation session
    USER = "user"  # Memory persists across user sessions (if identified)
    GLOBAL = "global"  # Memory persists globally across all users
    TEMPORARY = "temporary"  # Memory expires after short time period


@dataclass
class QueryPattern:
    """Represents a learned query pattern with routing optimization data."""

    pattern_id: str
    query_embedding: Optional[List[float]] = None
    query_text: str = ""
    intent_type: str = ""

    # Routing performance data
    successful_agent_sequences: List[List[str]] = field(default_factory=list)
    average_response_time: float = 0.0
    success_rate: float = 0.0
    usage_count: int = 0

    # Context and preferences
    typical_parameters: Dict[str, Any] = field(default_factory=dict)
    user_satisfaction_score: float = 0.0

    # Temporal data
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryPattern":
        """Create from dictionary."""
        # Convert datetime strings back to datetime objects
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("last_used"), str):
            data["last_used"] = datetime.fromisoformat(data["last_used"])
        return cls(**data)


@dataclass
class AgentSequencePattern:
    """Learned patterns for optimal agent handoff sequences."""

    sequence_id: str
    agent_sequence: List[str]
    context_triggers: List[str]  # What contexts trigger this sequence

    # Performance metrics
    success_rate: float = 0.0
    average_completion_time: float = 0.0
    usage_count: int = 0

    # Quality metrics
    result_quality_score: float = 0.0
    user_satisfaction: float = 0.0

    # Context data
    typical_query_types: List[str] = field(default_factory=list)
    optimal_for_intents: List[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=datetime.now)
    last_successful: datetime = field(default_factory=datetime.now)


@dataclass
class UserPreferenceProfile:
    """User preference profile learned from interaction history."""

    user_id: str

    # Platform preferences
    preferred_platforms: List[str] = field(default_factory=list)
    avoided_platforms: List[str] = field(default_factory=list)

    # Content preferences
    preferred_genres: List[str] = field(default_factory=list)
    preferred_studios: List[str] = field(default_factory=list)
    preferred_years: Optional[Tuple[int, int]] = None

    # Interaction patterns
    typical_query_complexity: str = "moderate"  # simple, moderate, complex
    prefers_detailed_results: bool = True
    prefers_streaming_info: bool = False

    # Routing preferences
    optimal_agent_sequences: List[List[str]] = field(default_factory=list)
    response_time_sensitivity: float = 0.5  # 0=don't care, 1=very sensitive

    # Learning metadata
    interaction_count: int = 0
    confidence_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class ConversationContextMemory:
    """Manages conversation context memory across user sessions."""

    def __init__(self, max_sessions: int = 1000, session_ttl_hours: int = 24):
        self.max_sessions = max_sessions
        self.session_ttl = timedelta(hours=session_ttl_hours)

        # Session-based memory
        self.session_contexts: Dict[str, Dict[str, Any]] = {}
        self.session_timestamps: Dict[str, datetime] = {}

        # Cross-session user memory (if user identification available)
        self.user_profiles: Dict[str, UserPreferenceProfile] = {}

        logger.info(
            f"ConversationContextMemory initialized: max_sessions={max_sessions}, ttl={session_ttl_hours}h"
        )

    def store_session_context(self, session_id: str, context: Dict[str, Any]) -> None:
        """Store conversation context for a session."""
        self.session_contexts[session_id] = context
        self.session_timestamps[session_id] = datetime.now()

        # Clean up expired sessions and enforce session limits
        self._cleanup_expired_sessions()
        self._enforce_session_limits()
        logger.debug(f"Stored session context for {session_id}, size: {len(context)}")

    def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve conversation context for a session."""
        if session_id in self.session_contexts:
            # Update access time
            self.session_timestamps[session_id] = datetime.now()
            return self.session_contexts[session_id]
        return None

    def update_user_profile(
        self, user_id: str, interaction_data: Dict[str, Any]
    ) -> None:
        """Update user preference profile based on interaction."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserPreferenceProfile(user_id=user_id)

        profile = self.user_profiles[user_id]
        profile.interaction_count += 1
        profile.last_updated = datetime.now()

        # Update preferences based on interaction
        if "platforms_used" in interaction_data:
            for platform in interaction_data["platforms_used"]:
                if platform not in profile.preferred_platforms:
                    profile.preferred_platforms.append(platform)

        if "query_complexity" in interaction_data:
            profile.typical_query_complexity = interaction_data["query_complexity"]

        # Update confidence based on interaction count
        profile.confidence_score = min(1.0, profile.interaction_count / 50.0)

        logger.debug(
            f"Updated user profile for {user_id}, interactions: {profile.interaction_count}"
        )

    def get_user_profile(self, user_id: str) -> Optional[UserPreferenceProfile]:
        """Get user preference profile."""
        return self.user_profiles.get(user_id)

    def _cleanup_expired_sessions(self) -> None:
        """Clean up expired session contexts."""
        now = datetime.now()
        expired_sessions = [
            session_id
            for session_id, timestamp in self.session_timestamps.items()
            if now - timestamp > self.session_ttl
        ]

        for session_id in expired_sessions:
            self.session_contexts.pop(session_id, None)
            self.session_timestamps.pop(session_id, None)

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    def _enforce_session_limits(self) -> None:
        """Enforce maximum session limits by removing oldest sessions."""
        if len(self.session_contexts) <= self.max_sessions:
            return

        # Sort sessions by timestamp (oldest first)
        sessions_by_age = sorted(self.session_timestamps.items(), key=lambda x: x[1])

        # Remove oldest sessions to get back under limit
        sessions_to_remove = sessions_by_age[: len(sessions_by_age) - self.max_sessions]

        for session_id, _ in sessions_to_remove:
            self.session_contexts.pop(session_id, None)
            self.session_timestamps.pop(session_id, None)

        if sessions_to_remove:
            logger.info(f"Removed {len(sessions_to_remove)} sessions to enforce limits")


class AgentHandoffOptimizer:
    """Learns and optimizes agent handoff sequences based on success patterns."""

    def __init__(self, max_patterns: int = 500):
        self.max_patterns = max_patterns

        # Learned handoff patterns
        self.sequence_patterns: Dict[str, AgentSequencePattern] = {}

        # Agent performance tracking
        self.agent_success_rates: Dict[str, float] = defaultdict(float)
        self.agent_handoff_success: Dict[Tuple[str, str], float] = defaultdict(float)

        # Handoff sequence cache for performance
        self.sequence_cache: Dict[str, List[str]] = {}

        logger.info(f"AgentHandoffOptimizer initialized: max_patterns={max_patterns}")

    def learn_from_execution(
        self,
        agent_sequence: List[str],
        execution_result: WorkflowResult,
        context: Dict[str, Any],
    ) -> None:
        """Learn from a completed agent execution sequence."""
        if not agent_sequence:
            return

        sequence_key = "->".join(agent_sequence)

        # Create or update sequence pattern
        if sequence_key not in self.sequence_patterns:
            self.sequence_patterns[sequence_key] = AgentSequencePattern(
                sequence_id=sequence_key,
                agent_sequence=agent_sequence,
                context_triggers=[],
            )

        pattern = self.sequence_patterns[sequence_key]
        pattern.usage_count += 1
        pattern.last_successful = datetime.now()

        # Update performance metrics
        if execution_result.get("execution_time_ms"):
            pattern.average_completion_time = (
                pattern.average_completion_time * (pattern.usage_count - 1)
                + execution_result["execution_time_ms"]
            ) / pattern.usage_count

        # Update success rate based on result quality
        result_success = len(execution_result.get("anime_results", [])) > 0
        pattern.success_rate = (
            pattern.success_rate * (pattern.usage_count - 1)
            + (1.0 if result_success else 0.0)
        ) / pattern.usage_count

        # Learn context triggers
        if (
            context.get("intent_type")
            and context["intent_type"] not in pattern.optimal_for_intents
        ):
            pattern.optimal_for_intents.append(context["intent_type"])

        # Update individual agent success rates
        for agent in agent_sequence:
            self.agent_success_rates[agent] = (
                self.agent_success_rates[agent] * 0.9
            ) + (0.1 if result_success else 0.0)

        # Update handoff success rates
        for i in range(len(agent_sequence) - 1):
            handoff_key = (agent_sequence[i], agent_sequence[i + 1])
            self.agent_handoff_success[handoff_key] = (
                self.agent_handoff_success[handoff_key] * 0.9
            ) + (0.1 if result_success else 0.0)

        logger.debug(
            f"Learned from sequence: {sequence_key}, success: {result_success}"
        )

    def get_optimal_sequence(
        self, query_intent: str, context: Dict[str, Any], max_agents: int = 5
    ) -> Optional[List[str]]:
        """Get optimal agent sequence for given query intent and context."""
        cache_key = f"{query_intent}:{hash(str(sorted(context.items())))}"

        if cache_key in self.sequence_cache:
            return self.sequence_cache[cache_key]

        # Find patterns that match the intent
        matching_patterns = [
            pattern
            for pattern in self.sequence_patterns.values()
            if query_intent in pattern.optimal_for_intents
            and pattern.success_rate > 0.5
        ]

        if not matching_patterns:
            return None

        # Sort by success rate and recency
        matching_patterns.sort(
            key=lambda p: (p.success_rate, p.usage_count, -p.average_completion_time),
            reverse=True,
        )

        optimal_sequence = matching_patterns[0].agent_sequence[:max_agents]

        # Cache the result
        self.sequence_cache[cache_key] = optimal_sequence

        logger.debug(f"Found optimal sequence for {query_intent}: {optimal_sequence}")
        return optimal_sequence

    def get_best_handoff_target(
        self, current_agent: str, context: Dict[str, Any]
    ) -> Optional[str]:
        """Get the best next agent for handoff from current agent."""
        # Find the best handoff target based on success rates
        possible_handoffs = [
            (target, rate)
            for (source, target), rate in self.agent_handoff_success.items()
            if source == current_agent and rate > 0.3
        ]

        if not possible_handoffs:
            return None

        # Sort by success rate
        possible_handoffs.sort(key=lambda x: x[1], reverse=True)
        return possible_handoffs[0][0]


class RoutingMemoryStore:
    """Core memory store for routing patterns with vector-based pattern matching."""

    def __init__(self, max_patterns: int = 10000):
        self.max_patterns = max_patterns

        # Pattern storage
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.pattern_embeddings: Dict[str, List[float]] = {}

        # Pattern indexing for fast lookup
        self.intent_patterns: Dict[str, List[str]] = defaultdict(list)
        self.performance_index: List[Tuple[str, float]] = (
            []
        )  # (pattern_id, success_rate)

        logger.info(f"RoutingMemoryStore initialized: max_patterns={max_patterns}")

    def store_pattern(self, pattern: QueryPattern) -> None:
        """Store a query pattern in memory."""
        self.query_patterns[pattern.pattern_id] = pattern

        # Update intent index
        if pattern.intent_type:
            if pattern.pattern_id not in self.intent_patterns[pattern.intent_type]:
                self.intent_patterns[pattern.intent_type].append(pattern.pattern_id)

        # Update performance index
        self._update_performance_index(pattern.pattern_id, pattern.success_rate)

        # Clean up if we exceed max patterns
        if len(self.query_patterns) > self.max_patterns:
            self._cleanup_old_patterns()

        logger.debug(
            f"Stored pattern {pattern.pattern_id}, intent: {pattern.intent_type}"
        )

    def find_similar_patterns(
        self, query: str, intent_type: str, limit: int = 5
    ) -> List[QueryPattern]:
        """Find similar query patterns for routing optimization."""
        # Start with intent-based filtering
        candidate_pattern_ids = self.intent_patterns.get(intent_type, [])

        if not candidate_pattern_ids:
            return []

        # Simple text similarity for now (could be enhanced with embeddings)
        query_lower = query.lower()
        similarities = []

        for pattern_id in candidate_pattern_ids:
            pattern = self.query_patterns.get(pattern_id)
            if not pattern:
                continue

            # Calculate simple text similarity
            pattern_text = pattern.query_text.lower()
            common_words = set(query_lower.split()) & set(pattern_text.split())
            similarity = len(common_words) / max(
                len(query_lower.split()), len(pattern_text.split())
            )

            similarities.append((pattern, similarity))

        # Sort by similarity and success rate
        similarities.sort(key=lambda x: (x[1], x[0].success_rate), reverse=True)

        return [pattern for pattern, _ in similarities[:limit]]

    def get_best_patterns_for_intent(
        self, intent_type: str, limit: int = 3
    ) -> List[QueryPattern]:
        """Get the best performing patterns for a specific intent type."""
        candidate_pattern_ids = self.intent_patterns.get(intent_type, [])

        patterns = [
            self.query_patterns[pid]
            for pid in candidate_pattern_ids
            if pid in self.query_patterns
        ]

        # Sort by success rate and usage count
        patterns.sort(key=lambda p: (p.success_rate, p.usage_count), reverse=True)

        return patterns[:limit]

    def _update_performance_index(self, pattern_id: str, success_rate: float) -> None:
        """Update the performance index for fast pattern lookup."""
        # Remove existing entry
        self.performance_index = [
            (pid, sr) for pid, sr in self.performance_index if pid != pattern_id
        ]

        # Add new entry
        self.performance_index.append((pattern_id, success_rate))

        # Keep sorted by success rate
        self.performance_index.sort(key=lambda x: x[1], reverse=True)

    def _cleanup_old_patterns(self) -> None:
        """Clean up old patterns when memory limit is exceeded."""
        # Sort patterns by last_used and success_rate (worst first)
        patterns_by_age = list(self.query_patterns.values())
        patterns_by_age.sort(key=lambda p: (p.last_used, p.success_rate))

        # Remove enough patterns to get back to the limit
        current_count = len(patterns_by_age)
        target_count = self.max_patterns
        patterns_to_remove_count = max(
            current_count - target_count, current_count // 10
        )
        patterns_to_remove = patterns_by_age[:patterns_to_remove_count]

        for pattern in patterns_to_remove:
            self.query_patterns.pop(pattern.pattern_id, None)

            # Clean up indices
            if pattern.intent_type in self.intent_patterns:
                if pattern.pattern_id in self.intent_patterns[pattern.intent_type]:
                    self.intent_patterns[pattern.intent_type].remove(pattern.pattern_id)

        logger.info(f"Cleaned up {len(patterns_to_remove)} old patterns")


class StatefulRoutingEngine:
    """Main orchestrator for stateful routing memory and context learning."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()

        # Initialize memory components
        self.memory_store = RoutingMemoryStore()
        self.context_memory = ConversationContextMemory()
        self.handoff_optimizer = AgentHandoffOptimizer()

        # Integration with existing LangGraph memory
        self.langgraph_memory = MemorySaver()

        # Routing strategy configuration
        self.default_strategy = RoutingStrategy.ADAPTIVE_HYBRID
        self.fallback_strategy = RoutingStrategy.FALLBACK_STANDARD

        # Performance tracking
        self.routing_performance_history: deque = deque(maxlen=1000)

        logger.info("StatefulRoutingEngine initialized successfully")

    async def get_optimal_routing(
        self,
        query: str,
        intent: QueryIntent,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get optimal routing strategy based on learned patterns and context."""
        context = context or {}

        routing_decision = {
            "strategy": self.default_strategy,
            "agent_sequence": [],
            "confidence": 0.0,
            "reasoning": [],
            "fallback_strategy": self.fallback_strategy,
            "memory_used": False,
        }

        try:
            # Step 1: Look for similar query patterns
            similar_patterns = self.memory_store.find_similar_patterns(
                query, intent, limit=3
            )

            if similar_patterns:
                routing_decision["memory_used"] = True
                best_pattern = similar_patterns[0]

                if best_pattern.successful_agent_sequences:
                    routing_decision["agent_sequence"] = (
                        best_pattern.successful_agent_sequences[0]
                    )
                    routing_decision["confidence"] = best_pattern.success_rate
                    routing_decision["reasoning"].append(
                        f"Using pattern {best_pattern.pattern_id} (success: {best_pattern.success_rate:.2f})"
                    )

            # Step 2: Get optimal sequence from handoff optimizer
            if not routing_decision["agent_sequence"]:
                optimal_sequence = self.handoff_optimizer.get_optimal_sequence(
                    intent, context
                )
                if optimal_sequence:
                    routing_decision["agent_sequence"] = optimal_sequence
                    routing_decision["confidence"] = 0.7
                    routing_decision["reasoning"].append(
                        "Using learned optimal agent sequence"
                    )

            # Step 3: Apply user preferences if available
            if user_id:
                user_profile = self.context_memory.get_user_profile(user_id)
                if user_profile and user_profile.optimal_agent_sequences:
                    # Prefer user's optimal sequences
                    routing_decision["agent_sequence"] = (
                        user_profile.optimal_agent_sequences[0]
                    )
                    routing_decision["confidence"] = user_profile.confidence_score
                    routing_decision["reasoning"].append(
                        "Using user preference profile"
                    )

            # Step 4: Get session context
            if session_id:
                session_context = self.context_memory.get_session_context(session_id)
                if session_context:
                    # Apply session-specific routing adjustments
                    routing_decision["reasoning"].append("Applied session context")

            # Step 5: Set strategy based on confidence
            if routing_decision["confidence"] > 0.8:
                routing_decision["strategy"] = RoutingStrategy.LEARNED_OPTIMAL
            elif routing_decision["confidence"] > 0.6:
                routing_decision["strategy"] = RoutingStrategy.PATTERN_MATCHING
            elif user_id and self.context_memory.get_user_profile(user_id):
                routing_decision["strategy"] = RoutingStrategy.PREFERENCE_BASED
            else:
                routing_decision["strategy"] = RoutingStrategy.FALLBACK_STANDARD

        except Exception as e:
            logger.error(f"Error in get_optimal_routing: {e}")
            routing_decision["strategy"] = RoutingStrategy.FALLBACK_STANDARD
            routing_decision["reasoning"].append(f"Fallback due to error: {e}")

        return routing_decision

    async def learn_from_execution(
        self,
        query: str,
        intent: QueryIntent,
        agent_sequence: List[str],
        execution_result: WorkflowResult,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Learn from a completed execution to improve future routing."""
        try:
            # Create query pattern
            pattern_id = hashlib.md5(f"{query}:{intent}".encode()).hexdigest()[:12]

            pattern = QueryPattern(
                pattern_id=pattern_id,
                query_text=query,
                intent_type=intent,
                successful_agent_sequences=[agent_sequence] if agent_sequence else [],
                average_response_time=execution_result.get("execution_time_ms", 0),
                success_rate=1.0 if execution_result.get("anime_results") else 0.0,
                usage_count=1,
            )

            # Store the pattern
            self.memory_store.store_pattern(pattern)

            # Learn handoff optimization
            self.handoff_optimizer.learn_from_execution(
                agent_sequence, execution_result, {"intent_type": intent}
            )

            # Update user profile if available
            if user_id:
                interaction_data = {
                    "platforms_used": execution_result.get("platforms_queried", []),
                    "query_complexity": execution_result.get("intent_analysis", {}).get(
                        "workflow_complexity", "moderate"
                    ),
                    "successful_agents": agent_sequence,
                }
                self.context_memory.update_user_profile(user_id, interaction_data)

            # Update session context
            if session_id:
                session_context = {
                    "last_query": query,
                    "last_intent": intent,
                    "last_successful_agents": agent_sequence,
                    "last_result_count": len(execution_result.get("anime_results", [])),
                }
                self.context_memory.store_session_context(session_id, session_context)

            # Track performance
            self.routing_performance_history.append(
                {
                    "timestamp": datetime.now(),
                    "success": len(execution_result.get("anime_results", [])) > 0,
                    "response_time": execution_result.get("execution_time_ms", 0),
                    "agent_count": len(agent_sequence),
                }
            )

            logger.debug(
                f"Learned from execution: pattern={pattern_id}, success={pattern.success_rate}"
            )

        except Exception as e:
            logger.error(f"Error in learn_from_execution: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        recent_performance = list(self.routing_performance_history)[-100:]
        avg_success_rate = sum(1 for p in recent_performance if p["success"]) / max(
            len(recent_performance), 1
        )
        avg_response_time = sum(p["response_time"] for p in recent_performance) / max(
            len(recent_performance), 1
        )

        return {
            "query_patterns_stored": len(self.memory_store.query_patterns),
            "active_sessions": len(self.context_memory.session_contexts),
            "user_profiles": len(self.context_memory.user_profiles),
            "agent_sequences_learned": len(self.handoff_optimizer.sequence_patterns),
            "recent_success_rate": avg_success_rate,
            "average_response_time_ms": avg_response_time,
            "memory_efficiency": (
                "Good" if avg_success_rate > 0.8 else "Needs Improvement"
            ),
        }


# Global stateful routing engine instance
_stateful_routing_engine: Optional[StatefulRoutingEngine] = None


def get_stateful_routing_engine() -> StatefulRoutingEngine:
    """Get or create the global stateful routing engine instance."""
    global _stateful_routing_engine
    if _stateful_routing_engine is None:
        _stateful_routing_engine = StatefulRoutingEngine()
    return _stateful_routing_engine
