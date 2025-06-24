"""LangGraph workflow engine for anime conversations."""

import logging
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional, Set

from .adapters import MCPAdapterRegistry
from .models import (
    AnimeSearchContext,
    ConversationState,
    MessageType,
    SmartOrchestrationState,
    UserPreferences,
    WorkflowMessage,
    WorkflowStep,
    WorkflowStepType,
)
from .smart_orchestration import SmartOrchestrationEngine

logger = logging.getLogger(__name__)


@dataclass
class WorkflowNode:
    """A node in the workflow graph."""

    name: str
    function: Callable[[ConversationState], Awaitable[ConversationState]]
    description: str

    async def execute(self, state: ConversationState) -> ConversationState:
        """Execute the node function."""
        return await self.function(state)


class WorkflowGraph:
    """Graph structure for workflow execution."""

    def __init__(self):
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: Dict[str, Set[str]] = {}

    def add_node(self, node: WorkflowNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.name] = node
        if node.name not in self.edges:
            self.edges[node.name] = set()

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add an edge between nodes."""
        if from_node not in self.edges:
            self.edges[from_node] = set()
        self.edges[from_node].add(to_node)

    def get_next_nodes(self, current_node: str) -> List[str]:
        """Get list of next nodes from current node."""
        return list(self.edges.get(current_node, set()))


class ConversationalAgent:
    """LangGraph-based conversational agent for anime discovery."""

    def __init__(self, adapter_registry: MCPAdapterRegistry):
        self.adapter_registry = adapter_registry
        self.workflow_graph = self._build_workflow_graph()

    def _build_workflow_graph(self) -> WorkflowGraph:
        """Build the workflow graph for anime conversations."""
        graph = WorkflowGraph()

        # Create workflow nodes
        nodes = [
            WorkflowNode("start", self._start_node, "Initialize conversation"),
            WorkflowNode("understand", self._understand_node, "Understand user intent"),
            WorkflowNode("search", self._search_node, "Execute anime search"),
            WorkflowNode("reasoning", self._reasoning_node, "Analyze search results"),
            WorkflowNode(
                "synthesis", self._synthesis_node, "Synthesize recommendations"
            ),
            WorkflowNode("response", self._response_node, "Generate response"),
        ]

        # Add nodes to graph
        for node in nodes:
            graph.add_node(node)

        # Define workflow edges
        graph.add_edge("start", "understand")
        graph.add_edge("understand", "search")
        graph.add_edge("search", "reasoning")
        graph.add_edge("reasoning", "synthesis")
        graph.add_edge("synthesis", "response")

        return graph

    async def process_user_message(
        self, state: ConversationState, message: str
    ) -> ConversationState:
        """Process a user message through the workflow."""
        # Add user message to conversation
        user_msg = WorkflowMessage(message_type=MessageType.USER, content=message)
        state.add_message(user_msg)

        # Execute workflow
        return await self._execute_workflow(state, "start")

    async def _execute_workflow(
        self, state: ConversationState, start_node: str
    ) -> ConversationState:
        """Execute the workflow starting from a specific node."""
        current_node = start_node

        while current_node and current_node in self.workflow_graph.nodes:
            try:
                # Execute current node
                node = self.workflow_graph.nodes[current_node]
                logger.debug(f"Executing node: {current_node}")
                state = await node.execute(state)

                # Determine next node (for now, just follow linear path)
                next_nodes = self.workflow_graph.get_next_nodes(current_node)
                current_node = next_nodes[0] if next_nodes else None

            except Exception as e:
                logger.error(f"Error executing node {current_node}: {e}")
                # Add error handling
                await self._handle_workflow_error(state, current_node, e)
                break

        return state

    async def _start_node(self, state: ConversationState) -> ConversationState:
        """Initialize conversation workflow."""
        step = WorkflowStep(
            step_type=WorkflowStepType.REASONING,
            reasoning="Starting conversation workflow",
            confidence=1.0,
        )
        state.add_workflow_step(step)
        return state

    async def _understand_node(self, state: ConversationState) -> ConversationState:
        """Understand user intent and extract search parameters."""
        if not state.messages:
            return state

        last_message = state.messages[-1]
        if last_message.message_type != MessageType.USER:
            return state

        content = last_message.content

        # Extract search intent and parameters
        new_context = await self._extract_search_context(content)

        # Merge with existing context if it exists (preserve image data, etc.)
        if state.current_context:
            # Preserve existing image data and other settings
            new_context.image_data = (
                state.current_context.image_data or new_context.image_data
            )
            new_context.text_weight = state.current_context.text_weight
            new_context.search_history = state.current_context.search_history

        state.update_context(new_context)

        # Extract or update user preferences
        preferences = self._extract_preferences(content, state.user_preferences)
        if preferences:
            state.update_preferences(preferences)

        step = WorkflowStep(
            step_type=WorkflowStepType.REASONING,
            reasoning=f"Understood user intent: query='{new_context.query}', "
            f"has_image={new_context.image_data is not None}",
            confidence=0.9,
        )
        state.add_workflow_step(step)

        return state

    async def _search_node(self, state: ConversationState) -> ConversationState:
        """Execute anime search based on context."""
        if not state.current_context or not state.current_context.query:
            return state

        context = state.current_context

        try:
            # Choose search strategy based on available data
            if context.image_data:
                # Multimodal search
                result = await self.adapter_registry.invoke_tool(
                    "search_multimodal_anime",
                    {
                        "query": context.query,
                        "image_data": context.image_data,
                        "text_weight": context.text_weight,
                        "limit": 10,
                    },
                )
            else:
                # Text-only search
                result = await self.adapter_registry.invoke_tool(
                    "search_anime", {"query": context.query, "limit": 10}
                )

            # Update context with results
            context.results = result if isinstance(result, list) else []

            step = WorkflowStep(
                step_type=WorkflowStepType.SEARCH,
                tool_name=(
                    "search_anime"
                    if not context.image_data
                    else "search_multimodal_anime"
                ),
                parameters={"query": context.query, "limit": 10},
                result={"count": len(context.results)},
                confidence=0.8,
            )
            state.add_workflow_step(step)

        except Exception as e:
            logger.error(f"Search failed: {e}")
            step = WorkflowStep(
                step_type=WorkflowStepType.SEARCH, error=str(e), confidence=0.0
            )
            state.add_workflow_step(step)

        return state

    async def _reasoning_node(self, state: ConversationState) -> ConversationState:
        """Analyze search results and user preferences."""
        if not state.current_context or not state.current_context.results:
            reasoning = "No search results available for analysis"
        else:
            results = state.current_context.results

            # Analyze result patterns
            genres = []
            studios = []
            years = []

            for result in results[:5]:  # Analyze top 5 results
                if "tags" in result:
                    genres.extend(result["tags"])
                if "studios" in result:
                    studios.extend(result["studios"])
                if "year" in result:
                    years.append(result["year"])

            # Generate reasoning about results
            reasoning_parts = []

            if genres:
                top_genres = list(set(genres))[:3]
                reasoning_parts.append(f"Common genres: {', '.join(top_genres)}")

            if studios:
                top_studios = list(set(studios))[:2]
                reasoning_parts.append(f"Studios: {', '.join(top_studios)}")

            if years:
                # Filter out None values
                valid_years = [y for y in years if y is not None]
                if valid_years:
                    year_range = (
                        f"{min(valid_years)}-{max(valid_years)}"
                        if len(set(valid_years)) > 1
                        else str(valid_years[0])
                    )
                    reasoning_parts.append(f"Year range: {year_range}")

            reasoning = (
                "Analysis: " + "; ".join(reasoning_parts)
                if reasoning_parts
                else "Diverse results found"
            )

        step = WorkflowStep(
            step_type=WorkflowStepType.REASONING, reasoning=reasoning, confidence=0.85
        )
        state.add_workflow_step(step)

        return state

    async def _synthesis_node(self, state: ConversationState) -> ConversationState:
        """Synthesize recommendations based on results and preferences."""
        if not state.current_context or not state.current_context.results:
            return state

        results = state.current_context.results
        preferences = state.user_preferences

        # Score and rank results based on preferences
        synthesized_results = []

        for result in results:
            score = result.get("score", 0.0)

            # Boost score based on user preferences
            if preferences:
                if preferences.favorite_genres:
                    result_genres = result.get("tags", [])
                    genre_match = len(
                        set(preferences.favorite_genres) & set(result_genres)
                    )
                    score += genre_match * 0.1

                if preferences.favorite_studios:
                    result_studios = result.get("studios", [])
                    studio_match = len(
                        set(preferences.favorite_studios) & set(result_studios)
                    )
                    score += studio_match * 0.15

            synthesized_result = result.copy()
            synthesized_result["final_score"] = score
            synthesized_results.append(synthesized_result)

        # Sort by final score
        synthesized_results.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        step = WorkflowStep(
            step_type=WorkflowStepType.SYNTHESIS,
            reasoning="Synthesized recommendations based on search results and user preferences",
            result={
                "synthesized_results": synthesized_results[:5],
                "total_processed": len(results),
            },
            confidence=0.9,
        )
        state.add_workflow_step(step)

        # Update context with synthesized results
        state.current_context.results = synthesized_results

        return state

    async def _response_node(self, state: ConversationState) -> ConversationState:
        """Generate final response to user."""
        if not state.current_context or not state.current_context.results:
            response_content = "I couldn't find any anime matching your criteria. Please try a different search."
        else:
            results = state.current_context.results[:3]  # Top 3 results

            response_parts = [
                f"I found {len(state.current_context.results)} anime for you. Here are the top recommendations:"
            ]

            for i, result in enumerate(results, 1):
                title = result.get("title", "Unknown")
                score = result.get("final_score", result.get("score", 0))
                response_parts.append(f"{i}. {title} (match: {score:.2f})")

            response_content = "\n".join(response_parts)

        # Add assistant message
        assistant_msg = WorkflowMessage(
            message_type=MessageType.ASSISTANT,
            content=response_content,
            tool_results=(
                state.current_context.results[:3] if state.current_context else []
            ),
        )
        state.add_message(assistant_msg)

        return state

    async def _extract_search_context(self, message: str) -> AnimeSearchContext:
        """Extract search context from user message using AI-powered understanding."""
        try:
            # Use LLM service for intelligent parameter extraction
            from ..services.llm_service import extract_search_intent

            intent = await extract_search_intent(message)

            # Convert SearchIntent to AnimeSearchContext
            filters = {}

            # Add year range to filters
            if intent.year_range and len(intent.year_range) >= 2:
                if intent.year_range[0] == intent.year_range[1]:
                    filters["year"] = intent.year_range[0]
                else:
                    filters["year_range"] = tuple(intent.year_range[:2])

            # Add anime types to filters
            if intent.anime_types:
                if len(intent.anime_types) == 1:
                    filters["type"] = intent.anime_types[0]
                else:
                    filters["types"] = intent.anime_types

            # Add genres to filters
            if intent.genres:
                filters["genres"] = intent.genres

            # Add studios to filters
            if intent.studios:
                filters["studios"] = intent.studios

            # Add exclusions to filters
            if intent.exclusions:
                filters["exclusions"] = intent.exclusions

            # Add mood keywords to filters
            if intent.mood_keywords:
                filters["mood"] = intent.mood_keywords

            return AnimeSearchContext(
                query=intent.query, filters=filters, limit=intent.limit
            )

        except Exception as e:
            logger.warning(f"LLM extraction failed, using fallback: {e}")
            # Fallback to basic extraction if LLM fails
            return self._fallback_extract_search_context(message)

    def _fallback_extract_search_context(self, message: str) -> AnimeSearchContext:
        """Fallback extraction using basic patterns."""
        query = message.lower()

        # Extract filters from common patterns
        filters = {}

        # Year patterns
        year_match = re.search(r"\b(19|20)\d{2}\b", message)
        if year_match:
            filters["year"] = int(year_match.group())

        # Type patterns
        if any(word in query for word in ["movie", "film"]):
            filters["type"] = "Movie"
        elif any(word in query for word in ["tv", "series"]):
            filters["type"] = "TV"
        elif "ova" in query:
            filters["type"] = "OVA"

        return AnimeSearchContext(query=message, filters=filters)

    def _extract_preferences(
        self, message: str, existing_prefs: Optional[UserPreferences]
    ) -> Optional[UserPreferences]:
        """Extract user preferences from message."""
        message_lower = message.lower()

        # Genre extraction
        genre_keywords = {
            "action": ["action", "fighting", "battle"],
            "romance": ["romance", "romantic", "love"],
            "comedy": ["comedy", "funny", "humor"],
            "drama": ["drama", "dramatic", "serious"],
            "fantasy": ["fantasy", "magic", "magical"],
            "sci-fi": ["sci-fi", "science fiction", "futuristic"],
            "mecha": ["mecha", "robot", "gundam"],
            "shounen": ["shounen", "shonen", "action-packed"],
            "shoujo": ["shoujo", "shojo"],
            "seinen": ["seinen"],
            "josei": ["josei"],
        }

        detected_genres = []
        for genre, keywords in genre_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_genres.append(genre)

        # Studio extraction
        studio_keywords = {
            "studio ghibli": ["ghibli", "miyazaki"],
            "madhouse": ["madhouse"],
            "bones": ["bones"],
            "wit studio": ["wit studio", "wit"],
            "mappa": ["mappa"],
            "toei animation": ["toei"],
            "pierrot": ["pierrot"],
        }

        detected_studios = []
        for studio, keywords in studio_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_studios.append(studio)

        # Only create preferences if we detected something
        if detected_genres or detected_studios:
            if existing_prefs:
                # Merge with existing preferences
                all_genres = list(set(existing_prefs.favorite_genres + detected_genres))
                all_studios = list(
                    set(existing_prefs.favorite_studios + detected_studios)
                )

                return UserPreferences(
                    favorite_genres=all_genres,
                    favorite_studios=all_studios,
                    preferred_year_range=existing_prefs.preferred_year_range,
                    preferred_episode_count=existing_prefs.preferred_episode_count,
                    language_preference=existing_prefs.language_preference,
                    content_rating=existing_prefs.content_rating,
                )
            else:
                return UserPreferences(
                    favorite_genres=detected_genres, favorite_studios=detected_studios
                )

        return None

    async def _handle_workflow_error(
        self, state: ConversationState, node_name: str, error: Exception
    ) -> None:
        """Handle workflow execution errors."""
        error_msg = WorkflowMessage(
            message_type=MessageType.SYSTEM,
            content=f"Error in workflow node '{node_name}': {str(error)}",
        )
        state.add_message(error_msg)

        error_step = WorkflowStep(
            step_type=WorkflowStepType.VALIDATION, error=str(error), confidence=0.0
        )
        state.add_workflow_step(error_step)


class AnimeWorkflowEngine:
    """Main workflow engine for anime conversations."""

    def __init__(self, adapter_registry: MCPAdapterRegistry):
        self.adapter_registry = adapter_registry
        self.agent = ConversationalAgent(adapter_registry)
        self.smart_orchestration = SmartOrchestrationEngine(adapter_registry)

    async def process_conversation(
        self, state: ConversationState, message: str
    ) -> ConversationState:
        """Process a conversation message with smart orchestration."""
        logger.info(f"Processing conversation for session {state.session_id}")

        try:
            # Check if we need to upgrade to smart orchestration
            if isinstance(state, SmartOrchestrationState):
                # Use smart orchestration for enhanced features
                result = await self.smart_orchestration.process_complex_conversation(
                    state, message
                )
                logger.info(
                    f"Smart orchestration processed, {len(result.workflow_steps)} steps executed"
                )
                return result
            else:
                # Use standard agent workflow
                result = await self.agent.process_user_message(state, message)
                logger.info(
                    f"Standard workflow processed, {len(result.workflow_steps)} steps executed"
                )
                return result
        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
            # Return state with error message
            error_msg = WorkflowMessage(
                message_type=MessageType.SYSTEM,
                content=f"Sorry, I encountered an error processing your request: {str(e)}",
            )
            state.add_message(error_msg)
            return state

    async def process_multimodal_conversation(
        self,
        state: ConversationState,
        message: str,
        image_data: str,
        text_weight: float = 0.7,
    ) -> ConversationState:
        """Process a multimodal conversation with text and image."""
        logger.info(
            f"Processing multimodal conversation for session {state.session_id}"
        )

        try:
            # Check if we can use smart orchestration for multimodal
            if isinstance(state, SmartOrchestrationState):
                # Use smart orchestration multimodal processing
                result = (
                    await self.smart_orchestration.process_multimodal_orchestration(
                        state, message, image_data, text_weight
                    )
                )
                logger.info(
                    f"Smart multimodal orchestration processed, {len(result.workflow_steps)} steps executed"
                )
                return result
            else:
                # Pre-populate context with image data for standard processing
                if not state.current_context:
                    state.current_context = AnimeSearchContext()

                state.current_context.image_data = image_data
                state.current_context.text_weight = text_weight

                return await self.process_conversation(state, message)
        except Exception as e:
            logger.error(f"Error processing multimodal conversation: {e}")
            # Return state with error message
            error_msg = WorkflowMessage(
                message_type=MessageType.SYSTEM,
                content=f"Sorry, I encountered an error processing your multimodal request: {str(e)}",
            )
            state.add_message(error_msg)
            return state

    async def get_conversation_summary(self, state: ConversationState) -> str:
        """Generate a summary of the conversation."""
        if not state.messages:
            return "No conversation history"

        user_messages = [
            msg.content
            for msg in state.messages
            if msg.message_type == MessageType.USER
        ]
        assistant_messages = [
            msg.content
            for msg in state.messages
            if msg.message_type == MessageType.ASSISTANT
        ]

        summary_parts = []

        if user_messages:
            summary_parts.append(f"User queries: {len(user_messages)}")
            summary_parts.append(f"Latest query: '{user_messages[-1][:50]}...'")

        if assistant_messages:
            summary_parts.append(f"Assistant responses: {len(assistant_messages)}")

        if state.user_preferences and state.user_preferences.favorite_genres:
            summary_parts.append(
                f"Preferred genres: {', '.join(state.user_preferences.favorite_genres[:3])}"
            )

        return "; ".join(summary_parts)
