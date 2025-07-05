"""
Intelligent routing logic for query-based tool selection.

Analyzes user queries to determine optimal platform tools and routing strategies
for anime discovery workflows. Handles complex routing decisions based on:
- Query intent analysis (search, schedule, streaming, comparison)
- Platform specializations and capabilities
- Data requirements and enrichment needs
- User preferences and context
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from ..models.universal_anime import (
    UniversalSearchParams,
    AnimeStatus,
    AnimeFormat,
    AnimeSeason,
)

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Classification of user query intent."""

    SEARCH = "search"  # Basic anime search
    SIMILAR = "similar"  # Find similar anime
    SCHEDULE = "schedule"  # Broadcast/airing information
    STREAMING = "streaming"  # Streaming availability
    COMPARISON = "comparison"  # Cross-platform comparison
    ENRICHMENT = "enrichment"  # Multi-platform data gathering
    DISCOVERY = "discovery"  # General discovery/recommendation
    SEASONAL = "seasonal"  # Seasonal anime queries
    TRENDING = "trending"  # Popular/trending content


class PlatformSpecialization(str, Enum):
    """Platform specializations for routing decisions."""

    COMMUNITY_DATA = "community_data"  # MAL: ratings, lists, community metrics
    COMPREHENSIVE = "comprehensive"  # AniList: extensive filtering, GraphQL
    STREAMING_FOCUS = "streaming_focus"  # Kitsu: streaming platforms, availability
    SCHEDULE_DATA = "schedule_data"  # AnimeSchedule: broadcast times, airing
    NO_AUTH_REQUIRED = "no_auth"  # Jikan: unofficial MAL data, no API key
    SEMANTIC_SEARCH = "semantic_search"  # Vector DB: AI-powered similarity
    VISUAL_SIMILARITY = "visual_search"  # CLIP: image-based search


@dataclass
class RoutingDecision:
    """Result of intelligent routing analysis."""

    primary_tools: List[str]  # Main tools to execute
    secondary_tools: List[str]  # Optional enrichment tools
    execution_strategy: str  # parallel, sequential, conditional
    confidence: float  # Confidence in routing decision (0-1)
    reasoning: List[str]  # Human-readable reasoning
    estimated_complexity: str  # simple, medium, complex
    platform_priorities: Dict[str, float]  # Platform preference scores
    enrichment_recommended: bool  # Whether to use cross-platform enrichment
    fallback_tools: List[str]  # Fallback tools if primary fails


class IntelligentRouter:
    """
    Intelligent query router for optimal tool selection.

    Analyzes user queries and determines the best combination of platform tools
    to execute based on intent, complexity, and data requirements.
    """

    def __init__(self):
        # Platform capabilities mapping
        self.platform_capabilities = {
            "mal": {
                "specializations": [PlatformSpecialization.COMMUNITY_DATA],
                "strengths": ["ratings", "lists", "popularity", "community_metrics"],
                "auth_required": True,
                "response_speed": "medium",
                "data_richness": "high",
            },
            "anilist": {
                "specializations": [PlatformSpecialization.COMPREHENSIVE],
                "strengths": [
                    "advanced_filtering",
                    "metadata",
                    "relationships",
                    "graphql",
                ],
                "auth_required": False,
                "response_speed": "fast",
                "data_richness": "very_high",
            },
            "jikan": {
                "specializations": [PlatformSpecialization.NO_AUTH_REQUIRED],
                "strengths": ["mal_data", "no_auth", "comprehensive_metadata"],
                "auth_required": False,
                "response_speed": "medium",
                "data_richness": "high",
            },
            "animeschedule": {
                "specializations": [PlatformSpecialization.SCHEDULE_DATA],
                "strengths": ["broadcast_times", "airing_status", "temporal_data"],
                "auth_required": False,
                "response_speed": "fast",
                "data_richness": "medium",
            },
            "kitsu": {
                "specializations": [PlatformSpecialization.STREAMING_FOCUS],
                "strengths": ["streaming_platforms", "availability", "json_api"],
                "auth_required": False,
                "response_speed": "medium",
                "data_richness": "medium",
            },
            "semantic": {
                "specializations": [
                    PlatformSpecialization.SEMANTIC_SEARCH,
                    PlatformSpecialization.VISUAL_SIMILARITY,
                ],
                "strengths": ["ai_similarity", "semantic_search", "vector_matching"],
                "auth_required": False,
                "response_speed": "fast",
                "data_richness": "medium",
            },
        }

        # Query patterns for intent classification
        self.intent_patterns = {
            QueryIntent.SIMILAR: [
                r"similar to",
                r"like.*but",
                r"recommendations based on",
                r"if you liked",
                r"anime like",
                r"comparable to",
            ],
            QueryIntent.SCHEDULE: [
                r"airing",
                r"broadcast",
                r"when does.*air",
                r"schedule",
                r"currently airing",
                r"upcoming",
                r"next episode",
            ],
            QueryIntent.STREAMING: [
                r"watch on",
                r"streaming",
                r"available on",
                r"netflix",
                r"crunchyroll",
                r"funimation",
                r"where to watch",
            ],
            QueryIntent.COMPARISON: [
                r"compare.*across",
                r"ratings.*between",
                r"differences between",
                r"vs",
                r"versus",
                r"better than",
            ],
            QueryIntent.SEASONAL: [
                r"winter \d+",
                r"spring \d+",
                r"summer \d+",
                r"fall \d+",
                r"seasonal",
                r"this season",
                r"current season",
            ],
            QueryIntent.TRENDING: [
                r"popular",
                r"trending",
                r"most watched",
                r"top rated",
                r"best.*\d+",
                r"highest.*score",
            ],
        }

        # Tool routing rules
        self.routing_rules = {
            QueryIntent.SEARCH: {
                "primary": ["anilist", "mal", "jikan"],
                "strategy": "parallel",
                "complexity": "simple",
            },
            QueryIntent.SIMILAR: {
                "primary": ["semantic"],
                "secondary": ["anilist", "mal"],
                "strategy": "sequential",
                "complexity": "medium",
            },
            QueryIntent.SCHEDULE: {
                "primary": ["animeschedule"],
                "secondary": ["anilist"],
                "strategy": "parallel",
                "complexity": "simple",
            },
            QueryIntent.STREAMING: {
                "primary": ["kitsu", "animeschedule"],
                "strategy": "parallel",
                "complexity": "simple",
            },
            QueryIntent.COMPARISON: {
                "primary": ["compare_anime_ratings_cross_platform"],
                "secondary": ["correlate_anime_across_platforms"],
                "strategy": "sequential",
                "complexity": "complex",
            },
            QueryIntent.SEASONAL: {
                "primary": ["mal", "anilist"],
                "secondary": ["animeschedule"],
                "strategy": "parallel",
                "complexity": "medium",
            },
            QueryIntent.TRENDING: {
                "primary": ["mal", "anilist"],
                "strategy": "parallel",
                "complexity": "simple",
            },
        }

    async def route_query(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None,
        preferred_platforms: Optional[List[str]] = None,
    ) -> RoutingDecision:
        """
        Analyze query and determine optimal tool routing strategy.

        Args:
            query: User's anime search query
            user_context: Optional user preferences and context
            preferred_platforms: User's preferred platforms

        Returns:
            Routing decision with selected tools and execution strategy
        """
        logger.info(f"Routing query: '{query}'")

        # Step 1: Analyze query intent
        intent = self._classify_intent(query)

        # Step 2: Extract query features
        features = self._extract_query_features(query)

        # Step 3: Determine complexity
        complexity = self._assess_complexity(query, features, user_context)

        # Step 4: Select tools based on intent and features
        tool_selection = self._select_tools(
            intent, features, complexity, preferred_platforms
        )

        # Step 5: Determine execution strategy
        strategy = self._determine_execution_strategy(
            intent, complexity, tool_selection
        )

        # Step 6: Calculate platform priorities
        platform_priorities = self._calculate_platform_priorities(
            intent, features, preferred_platforms
        )

        # Step 7: Build routing decision
        decision = RoutingDecision(
            primary_tools=tool_selection["primary"],
            secondary_tools=tool_selection.get("secondary", []),
            execution_strategy=strategy,
            confidence=self._calculate_confidence(intent, features, complexity),
            reasoning=self._generate_reasoning(
                intent, features, tool_selection, strategy
            ),
            estimated_complexity=complexity,
            platform_priorities=platform_priorities,
            enrichment_recommended=complexity in ["medium", "complex"],
            fallback_tools=self._select_fallback_tools(tool_selection["primary"]),
        )

        logger.info(
            f"Routing decision: {len(decision.primary_tools)} primary tools, "
            f"{decision.execution_strategy} strategy, {decision.confidence:.2f} confidence"
        )

        return decision

    def _classify_intent(self, query: str) -> QueryIntent:
        """Classify the user's intent from the query."""
        query_lower = query.lower()

        # Score each intent based on pattern matches
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            if score > 0:
                intent_scores[intent] = score

        # Return highest scoring intent, or default to SEARCH
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]

        # Check for cross-platform keywords
        if any(word in query_lower for word in ["compare", "across", "between", "vs"]):
            return QueryIntent.COMPARISON

        # Check for similarity keywords
        if any(word in query_lower for word in ["like", "similar", "recommend"]):
            return QueryIntent.SIMILAR

        return QueryIntent.SEARCH

    def _extract_query_features(self, query: str) -> Dict[str, Any]:
        """Extract features from the query for routing decisions."""
        query_lower = query.lower()
        features = {
            "has_specific_title": False,
            "has_genre_filters": False,
            "has_temporal_filters": False,
            "has_platform_mention": False,
            "has_comparison_request": False,
            "complexity_indicators": [],
            "mentioned_platforms": [],
            "mentioned_genres": [],
            "temporal_references": [],
        }

        # Check for specific titles (quoted strings or proper nouns)
        if '"' in query or any(word.istitle() for word in query.split()):
            features["has_specific_title"] = True

        # Check for genre mentions
        common_genres = [
            "action",
            "adventure",
            "comedy",
            "drama",
            "fantasy",
            "horror",
            "mystery",
            "romance",
            "sci-fi",
            "slice of life",
            "sports",
            "thriller",
            "shounen",
            "seinen",
            "shoujo",
            "josei",
            "mecha",
            "isekai",
        ]
        mentioned_genres = [genre for genre in common_genres if genre in query_lower]
        if mentioned_genres:
            features["has_genre_filters"] = True
            features["mentioned_genres"] = mentioned_genres

        # Check for temporal references
        temporal_patterns = [
            r"\d{4}",
            r"winter|spring|summer|fall|autumn",
            r"this year|last year|next year",
            r"recently|latest|new",
            r"old|classic|retro",
            r"upcoming|future",
        ]
        temporal_refs = []
        for pattern in temporal_patterns:
            matches = re.findall(pattern, query_lower)
            temporal_refs.extend(matches)
        if temporal_refs:
            features["has_temporal_filters"] = True
            features["temporal_references"] = temporal_refs

        # Check for platform mentions
        platforms = ["mal", "anilist", "kitsu", "crunchyroll", "netflix", "funimation"]
        mentioned_platforms = [p for p in platforms if p in query_lower]
        if mentioned_platforms:
            features["has_platform_mention"] = True
            features["mentioned_platforms"] = mentioned_platforms

        # Check for comparison requests
        comparison_words = [
            "compare",
            "vs",
            "versus",
            "between",
            "difference",
            "better",
        ]
        if any(word in query_lower for word in comparison_words):
            features["has_comparison_request"] = True

        # Identify complexity indicators
        complexity_indicators = []
        if len(query.split()) > 10:
            complexity_indicators.append("long_query")
        if features["has_comparison_request"]:
            complexity_indicators.append("cross_platform_comparison")
        if len(mentioned_genres) > 2:
            complexity_indicators.append("multiple_genres")
        if features["has_temporal_filters"] and features["has_genre_filters"]:
            complexity_indicators.append("multiple_filter_types")

        features["complexity_indicators"] = complexity_indicators

        return features

    def _assess_complexity(
        self,
        query: str,
        features: Dict[str, Any],
        user_context: Optional[Dict[str, Any]],
    ) -> str:
        """Assess query complexity for routing decisions."""
        complexity_score = 0

        # Base complexity from query length
        if len(query.split()) > 15:
            complexity_score += 2
        elif len(query.split()) > 8:
            complexity_score += 1

        # Complexity from features
        complexity_score += len(features["complexity_indicators"])

        # Cross-platform requests add complexity
        if features["has_comparison_request"]:
            complexity_score += 2

        # Multiple filter types add complexity
        if features["has_genre_filters"] and features["has_temporal_filters"]:
            complexity_score += 1

        # User context complexity
        if user_context:
            if len(user_context.get("preferences", {})) > 3:
                complexity_score += 1
            if user_context.get("conversation_history"):
                complexity_score += 1

        # Map score to complexity level
        if complexity_score >= 4:
            return "complex"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "simple"

    def _select_tools(
        self,
        intent: QueryIntent,
        features: Dict[str, Any],
        complexity: str,
        preferred_platforms: Optional[List[str]],
    ) -> Dict[str, List[str]]:
        """Select optimal tools based on intent and features."""
        # Start with default routing rules
        base_rules = self.routing_rules.get(
            intent, self.routing_rules[QueryIntent.SEARCH]
        )
        selected_tools = {
            "primary": base_rules["primary"].copy(),
            "secondary": base_rules.get("secondary", []).copy(),
        }

        # Adjust based on features
        if features["has_platform_mention"]:
            # Prioritize mentioned platforms
            mentioned = features["mentioned_platforms"]
            platform_tools = []
            for platform in mentioned:
                if platform in ["mal", "anilist", "jikan", "kitsu", "animeschedule"]:
                    platform_tools.append(f"search_anime_{platform}")
            if platform_tools:
                selected_tools["primary"] = platform_tools

        if features["has_comparison_request"]:
            # Add cross-platform comparison tools
            if "compare_anime_ratings_cross_platform" not in selected_tools["primary"]:
                selected_tools["primary"].append("compare_anime_ratings_cross_platform")
            selected_tools["secondary"].append("correlate_anime_across_platforms")

        if complexity == "complex":
            # Add enrichment tools for complex queries
            selected_tools["secondary"].extend(
                ["get_cross_platform_anime_data", "detect_platform_discrepancies"]
            )

        # Apply user preferences
        if preferred_platforms:
            # Reorder primary tools based on preferences
            preferred_tools = []
            other_tools = []
            for tool in selected_tools["primary"]:
                if any(pref in tool for pref in preferred_platforms):
                    preferred_tools.append(tool)
                else:
                    other_tools.append(tool)
            selected_tools["primary"] = preferred_tools + other_tools

        return selected_tools

    def _determine_execution_strategy(
        self, intent: QueryIntent, complexity: str, tool_selection: Dict[str, List[str]]
    ) -> str:
        """Determine optimal execution strategy."""
        # Start with default strategy
        base_strategy = self.routing_rules.get(intent, {}).get("strategy", "parallel")

        # Adjust based on complexity and tools
        if complexity == "complex":
            return "sequential"  # Complex queries need careful orchestration

        if len(tool_selection["primary"]) > 3:
            return "parallel"  # Many tools benefit from parallel execution

        # Cross-platform tools often need sequential execution
        cross_platform_tools = [
            "compare_anime_ratings_cross_platform",
            "get_cross_platform_anime_data",
            "correlate_anime_across_platforms",
        ]
        if any(tool in tool_selection["primary"] for tool in cross_platform_tools):
            return "sequential"

        return base_strategy

    def _calculate_platform_priorities(
        self,
        intent: QueryIntent,
        features: Dict[str, Any],
        preferred_platforms: Optional[List[str]],
    ) -> Dict[str, float]:
        """Calculate priority scores for each platform."""
        priorities = {}

        # Base priorities by intent
        intent_priorities = {
            QueryIntent.SEARCH: {"anilist": 0.9, "mal": 0.8, "jikan": 0.7},
            QueryIntent.SIMILAR: {"semantic": 1.0, "anilist": 0.7, "mal": 0.6},
            QueryIntent.SCHEDULE: {"animeschedule": 1.0, "anilist": 0.5},
            QueryIntent.STREAMING: {"kitsu": 1.0, "animeschedule": 0.8},
            QueryIntent.COMPARISON: {"mal": 0.9, "anilist": 0.9, "kitsu": 0.7},
        }

        base_scores = intent_priorities.get(intent, {"anilist": 0.8, "mal": 0.7})
        priorities.update(base_scores)

        # Boost mentioned platforms
        if features["has_platform_mention"]:
            for platform in features["mentioned_platforms"]:
                if platform in priorities:
                    priorities[platform] = min(1.0, priorities[platform] + 0.3)

        # Apply user preferences
        if preferred_platforms:
            for platform in preferred_platforms:
                if platform in priorities:
                    priorities[platform] = min(1.0, priorities[platform] + 0.2)

        return priorities

    def _calculate_confidence(
        self, intent: QueryIntent, features: Dict[str, Any], complexity: str
    ) -> float:
        """Calculate confidence in routing decision."""
        confidence = 0.7  # Base confidence

        # Higher confidence for clear intent patterns
        if intent != QueryIntent.SEARCH:  # Non-default intent was detected
            confidence += 0.1

        # Platform mentions increase confidence
        if features["has_platform_mention"]:
            confidence += 0.1

        # Specific titles increase confidence
        if features["has_specific_title"]:
            confidence += 0.1

        # Complexity affects confidence
        if complexity == "simple":
            confidence += 0.05
        elif complexity == "complex":
            confidence -= 0.05

        return min(1.0, confidence)

    def _generate_reasoning(
        self,
        intent: QueryIntent,
        features: Dict[str, Any],
        tool_selection: Dict[str, List[str]],
        strategy: str,
    ) -> List[str]:
        """Generate human-readable reasoning for routing decision."""
        reasoning = []

        reasoning.append(f"Classified query intent as '{intent.value}'")

        if features["has_specific_title"]:
            reasoning.append("Detected specific anime title - using targeted search")

        if features["has_comparison_request"]:
            reasoning.append(
                "Cross-platform comparison requested - using enrichment tools"
            )

        if features["has_platform_mention"]:
            platforms = ", ".join(features["mentioned_platforms"])
            reasoning.append(f"Specific platforms mentioned: {platforms}")

        if len(tool_selection["primary"]) > 1:
            reasoning.append(
                f"Using {len(tool_selection['primary'])} primary tools for comprehensive coverage"
            )

        reasoning.append(f"Execution strategy: {strategy}")

        return reasoning

    def _select_fallback_tools(self, primary_tools: List[str]) -> List[str]:
        """Select fallback tools if primary tools fail."""
        fallback_mapping = {
            "search_anime_mal": ["search_anime_jikan", "search_anime_anilist"],
            "search_anime_anilist": ["search_anime_mal", "search_anime_jikan"],
            "search_anime_jikan": ["search_anime_mal", "search_anime_anilist"],
            "search_anime_kitsu": ["search_anime_anilist"],
            "search_anime_schedule": ["search_anime_anilist"],
            "anime_semantic_search": ["search_anime_anilist", "search_anime_mal"],
        }

        fallbacks = []
        for tool in primary_tools:
            if tool in fallback_mapping:
                fallbacks.extend(fallback_mapping[tool])

        # Remove duplicates and primary tools
        fallbacks = list(set(fallbacks) - set(primary_tools))
        return fallbacks[:3]  # Limit to 3 fallback tools
