"""Smart orchestration engine for Phase 6B advanced workflow features."""
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from uuid import uuid4

from .models import (
    SmartOrchestrationState,
    QueryChain,
    RefinementCriteria,
    OrchestrationPlan,
    ConversationFlow,
    AdaptivePreferences,
    WorkflowStep,
    WorkflowStepType,
    WorkflowMessage,
    MessageType,
    UserPreferences,
    AnimeSearchContext
)
from .adapters import MCPAdapterRegistry


class QueryChainOrchestrator:
    """Orchestrates complex query chains for multi-step discovery."""
    
    def __init__(self, adapter_registry: MCPAdapterRegistry):
        self.adapter_registry = adapter_registry
    
    async def execute_chain(self, chain: QueryChain, state: SmartOrchestrationState) -> Dict[str, Any]:
        """Execute a query chain with intelligent orchestration."""
        results = {}
        
        for i, query in enumerate(chain.queries):
            # Determine the best search strategy for this query
            strategy = self._determine_search_strategy(query, chain, i)
            
            # Execute with appropriate tools
            if strategy == "semantic":
                result = await self.adapter_registry.invoke_tool("search_anime", {"query": query, "limit": 10})
            elif strategy == "similarity" and i > 0:
                # Use results from previous query for similarity search
                prev_results = results.get(f"query_{i-1}", [])
                if prev_results:
                    result = await self.adapter_registry.invoke_tool(
                        "find_similar_anime", 
                        {"anime_id": prev_results[0].get("anime_id"), "limit": 10}
                    )
                else:
                    result = await self.adapter_registry.invoke_tool("search_anime", {"query": query, "limit": 10})
            else:
                result = await self.adapter_registry.invoke_tool("search_anime", {"query": query, "limit": 10})
            
            results[f"query_{i}"] = result
            chain.results_mapping[query] = result
            
            # Calculate confidence for this step
            confidence = self._calculate_query_confidence(result, query)
            chain.confidence_scores[query] = confidence
        
        return results
    
    def _determine_search_strategy(self, query: str, chain: QueryChain, index: int) -> str:
        """Determine the best search strategy for a query in the chain."""
        if index == 0:
            return "semantic"
        
        # Check if this query is asking for similar content
        similarity_keywords = ["similar", "like", "comparable", "related"]
        if any(keyword in query.lower() for keyword in similarity_keywords):
            return "similarity"
        
        return "semantic"
    
    def _calculate_query_confidence(self, results: List[Dict[str, Any]], query: str) -> float:
        """Calculate confidence score for query results."""
        if not results:
            return 0.0
        
        # Base confidence on average scores with result count bonus
        avg_score = sum(r.get("score", 0.0) for r in results) / len(results)
        result_bonus = min(len(results) / 10.0, 0.1)  # Up to 10% bonus for more results
        
        return min(avg_score + result_bonus, 1.0)


class ResultRefinementEngine:
    """Refines search results through iterative improvement."""
    
    def __init__(self, adapter_registry: MCPAdapterRegistry):
        self.adapter_registry = adapter_registry
    
    async def refine_results(
        self, 
        initial_results: List[Dict[str, Any]], 
        criteria: RefinementCriteria,
        context: AnimeSearchContext
    ) -> Tuple[List[Dict[str, Any]], List[WorkflowStep]]:
        """Refine results through multiple iterations."""
        current_results = initial_results
        refinement_steps = []
        
        for iteration in range(criteria.max_iterations):
            step_start = time.time()
            
            # Apply quality filters
            filtered_results = self._apply_quality_filters(current_results, criteria)
            
            # Check if we need more results
            if len(filtered_results) < criteria.target_result_count:
                # Expand search with related queries
                expanded_results = await self._expand_search(
                    filtered_results, 
                    context,
                    criteria.target_result_count - len(filtered_results)
                )
                filtered_results.extend(expanded_results)
            
            # Apply focus area filtering
            focused_results = self._apply_focus_filtering(filtered_results, criteria.focus_areas)
            
            # Remove excluded content
            final_results = self._apply_exclusions(focused_results, criteria.exclusion_criteria)
            
            execution_time = (time.time() - step_start) * 1000
            
            # Create refinement step
            step = WorkflowStep(
                step_type=WorkflowStepType.REFINEMENT,
                parameters={
                    "iteration": iteration + 1,
                    "initial_count": len(current_results),
                    "filtered_count": len(filtered_results),
                    "final_count": len(final_results)
                },
                result={"refined_results": final_results},
                confidence=self._calculate_refinement_confidence(final_results, criteria),
                execution_time_ms=execution_time,
                reasoning=f"Iteration {iteration + 1}: Applied quality filters, focus areas, and exclusions"
            )
            refinement_steps.append(step)
            
            # Check convergence
            avg_confidence = sum(r.get("score", 0.0) for r in final_results) / len(final_results) if final_results else 0.0
            if avg_confidence >= criteria.min_confidence and len(final_results) >= criteria.target_result_count:
                break
            
            current_results = final_results
        
        return final_results, refinement_steps
    
    def _apply_quality_filters(
        self, 
        results: List[Dict[str, Any]], 
        criteria: RefinementCriteria
    ) -> List[Dict[str, Any]]:
        """Apply quality threshold filters."""
        filtered = []
        for result in results:
            score = result.get("score", 0.0)
            if score >= criteria.min_confidence:
                # Check specific quality thresholds
                passes_quality = True
                for field, threshold in criteria.quality_thresholds.items():
                    if field in result and result[field] < threshold:
                        passes_quality = False
                        break
                
                if passes_quality:
                    filtered.append(result)
        
        return filtered
    
    def _apply_focus_filtering(
        self, 
        results: List[Dict[str, Any]], 
        focus_areas: List[str]
    ) -> List[Dict[str, Any]]:
        """Apply focus area filtering."""
        if not focus_areas:
            return results
        
        focused = []
        for result in results:
            # Check if result matches any focus areas
            tags = result.get("tags", [])
            studios = result.get("studios", [])
            title = result.get("title", "").lower()
            
            matches_focus = False
            for focus in focus_areas:
                focus_lower = focus.lower()
                if (focus_lower in tags or 
                    focus_lower in studios or 
                    focus_lower in title):
                    matches_focus = True
                    break
            
            if matches_focus:
                focused.append(result)
        
        return focused
    
    def _apply_exclusions(
        self, 
        results: List[Dict[str, Any]], 
        exclusions: List[str]
    ) -> List[Dict[str, Any]]:
        """Remove excluded content."""
        if not exclusions:
            return results
        
        filtered = []
        for result in results:
            tags = result.get("tags", [])
            studios = result.get("studios", [])
            title = result.get("title", "").lower()
            
            is_excluded = False
            for exclusion in exclusions:
                exclusion_lower = exclusion.lower()
                if (exclusion_lower in tags or 
                    exclusion_lower in studios or 
                    exclusion_lower in title):
                    is_excluded = True
                    break
            
            if not is_excluded:
                filtered.append(result)
        
        return filtered
    
    async def _expand_search(
        self, 
        current_results: List[Dict[str, Any]], 
        context: AnimeSearchContext,
        needed_count: int
    ) -> List[Dict[str, Any]]:
        """Expand search to find more relevant results."""
        if not current_results:
            return []
        
        # Use the best result to find similar anime
        best_result = max(current_results, key=lambda x: x.get("score", 0.0))
        anime_id = best_result.get("anime_id")
        
        if anime_id:
            similar_results = await self.adapter_registry.invoke_tool(
                "find_similar_anime", 
                {"anime_id": anime_id, "limit": needed_count}
            )
            return similar_results[:needed_count]
        
        return []
    
    def _calculate_refinement_confidence(
        self, 
        results: List[Dict[str, Any]], 
        criteria: RefinementCriteria
    ) -> float:
        """Calculate confidence in refinement results."""
        if not results:
            return 0.0
        
        avg_score = sum(r.get("score", 0.0) for r in results) / len(results)
        count_factor = min(len(results) / criteria.target_result_count, 1.0)
        
        return avg_score * count_factor


class ConversationFlowManager:
    """Manages enhanced conversation flows for complex interactions."""
    
    def __init__(self, adapter_registry: MCPAdapterRegistry):
        self.adapter_registry = adapter_registry
    
    def create_discovery_flow(self, initial_query: str) -> ConversationFlow:
        """Create a discovery conversation flow."""
        return ConversationFlow(
            flow_id=str(uuid4()),
            flow_type="discovery",
            current_stage="initial_search",
            stages=[
                {"name": "initial_search", "description": "Initial search based on user query"},
                {"name": "preference_extraction", "description": "Extract user preferences from context"},
                {"name": "result_refinement", "description": "Refine results based on preferences"},
                {"name": "similarity_exploration", "description": "Explore similar content"},
                {"name": "final_recommendation", "description": "Provide final recommendations"}
            ],
            branch_conditions={
                "initial_search_to_refinement": {"min_results": 3},
                "refinement_to_exploration": {"confidence_threshold": 0.7},
                "exploration_to_recommendation": {"sufficient_variety": True}
            }
        )
    
    def create_multimodal_flow(self, text_query: str, has_image: bool) -> ConversationFlow:
        """Create a multimodal conversation flow."""
        stages = [
            {"name": "multimodal_analysis", "description": "Analyze text and image inputs"},
            {"name": "cross_modal_search", "description": "Perform cross-modal search"},
            {"name": "result_synthesis", "description": "Synthesize text and visual results"}
        ]
        
        if has_image:
            stages.insert(1, {"name": "visual_similarity", "description": "Find visually similar content"})
        
        return ConversationFlow(
            flow_id=str(uuid4()),
            flow_type="multimodal",
            current_stage="multimodal_analysis",
            stages=stages,
            context_carryover={"text_query": text_query, "has_image": has_image}
        )
    
    async def execute_flow_stage(
        self, 
        flow: ConversationFlow, 
        state: SmartOrchestrationState
    ) -> Tuple[ConversationFlow, List[WorkflowStep]]:
        """Execute the current stage of a conversation flow."""
        stage_steps = []
        current_stage = flow.current_stage
        
        if current_stage == "initial_search":
            steps = await self._execute_initial_search(flow, state)
            stage_steps.extend(steps)
            
        elif current_stage == "preference_extraction":
            steps = await self._execute_preference_extraction(flow, state)
            stage_steps.extend(steps)
            
        elif current_stage == "result_refinement":
            steps = await self._execute_result_refinement(flow, state)
            stage_steps.extend(steps)
            
        elif current_stage == "multimodal_analysis":
            steps = await self._execute_multimodal_analysis(flow, state)
            stage_steps.extend(steps)
        
        # Advance to next stage if conditions are met
        next_stage = self._determine_next_stage(flow, state)
        if next_stage:
            flow.current_stage = next_stage
        
        return flow, stage_steps
    
    async def _execute_initial_search(
        self, 
        flow: ConversationFlow, 
        state: SmartOrchestrationState
    ) -> List[WorkflowStep]:
        """Execute initial search stage."""
        if not state.current_context or not state.current_context.query:
            return []
        
        step_start = time.time()
        results = await self.adapter_registry.invoke_tool(
            "search_anime", 
            {"query": state.current_context.query, "limit": 15}
        )
        
        execution_time = (time.time() - step_start) * 1000
        
        step = WorkflowStep(
            step_type=WorkflowStepType.SEARCH,
            tool_name="search_anime",
            parameters={"query": state.current_context.query, "limit": 15},
            result={"results": results},
            confidence=sum(r.get("score", 0.0) for r in results) / len(results) if results else 0.0,
            execution_time_ms=execution_time,
            reasoning="Initial search to establish baseline results"
        )
        
        # Update context with results
        if state.current_context:
            state.current_context.results = results
        
        return [step]
    
    async def _execute_preference_extraction(
        self, 
        flow: ConversationFlow, 
        state: SmartOrchestrationState
    ) -> List[WorkflowStep]:
        """Extract user preferences from conversation context."""
        step_start = time.time()
        
        # Extract preferences from messages and results
        extracted_prefs = self._extract_preferences_from_context(state)
        
        execution_time = (time.time() - step_start) * 1000
        
        step = WorkflowStep(
            step_type=WorkflowStepType.ADAPTATION,
            parameters={"extraction_method": "context_analysis"},
            result={"extracted_preferences": extracted_prefs},
            confidence=0.8,  # High confidence in preference extraction
            execution_time_ms=execution_time,
            reasoning="Extracted user preferences from conversation context and search results"
        )
        
        return [step]
    
    async def _execute_result_refinement(
        self, 
        flow: ConversationFlow, 
        state: SmartOrchestrationState
    ) -> List[WorkflowStep]:
        """Execute result refinement stage."""
        if not state.current_context or not state.current_context.results:
            return []
        
        # Create refinement criteria based on context
        criteria = RefinementCriteria(
            min_confidence=0.7,
            max_iterations=2,
            target_result_count=10
        )
        
        # Apply user preferences if available
        if state.user_preferences:
            criteria.focus_areas = state.user_preferences.favorite_genres
            criteria.exclusion_criteria = state.user_preferences.excluded_genres
        
        refinement_engine = ResultRefinementEngine(self.adapter_registry)
        refined_results, refinement_steps = await refinement_engine.refine_results(
            state.current_context.results,
            criteria,
            state.current_context
        )
        
        # Update context with refined results
        state.current_context.results = refined_results
        
        return refinement_steps
    
    async def _execute_multimodal_analysis(
        self, 
        flow: ConversationFlow, 
        state: SmartOrchestrationState
    ) -> List[WorkflowStep]:
        """Execute multimodal analysis stage."""
        if not state.current_context:
            return []
        
        step_start = time.time()
        
        # Perform multimodal search if both text and image are available
        if state.current_context.query and state.current_context.image_data:
            results = await self.adapter_registry.invoke_tool(
                "search_multimodal_anime",
                {
                    "query": state.current_context.query,
                    "image_data": state.current_context.image_data,
                    "text_weight": state.current_context.text_weight,
                    "limit": 10
                }
            )
        elif state.current_context.image_data:
            # Image-only search
            results = await self.adapter_registry.invoke_tool(
                "search_anime_by_image",
                {
                    "image_data": state.current_context.image_data,
                    "limit": 10
                }
            )
        else:
            # Text-only search
            results = await self.adapter_registry.invoke_tool(
                "search_anime",
                {
                    "query": state.current_context.query,
                    "limit": 10
                }
            )
        
        execution_time = (time.time() - step_start) * 1000
        
        step = WorkflowStep(
            step_type=WorkflowStepType.SEARCH,
            tool_name="multimodal_search",
            parameters={
                "has_text": bool(state.current_context.query),
                "has_image": bool(state.current_context.image_data),
                "text_weight": state.current_context.text_weight
            },
            result={"multimodal_results": results},
            confidence=sum(r.get("score", 0.0) for r in results) / len(results) if results else 0.0,
            execution_time_ms=execution_time,
            reasoning="Performed multimodal analysis combining text and visual inputs"
        )
        
        # Update context with multimodal results
        state.current_context.results = results
        
        return [step]
    
    def _extract_preferences_from_context(self, state: SmartOrchestrationState) -> Dict[str, Any]:
        """Extract user preferences from conversation context."""
        preferences = {
            "favorite_genres": [],
            "favorite_studios": [],
            "content_patterns": []
        }
        
        # Analyze messages for preference indicators
        for message in state.messages:
            if message.message_type == MessageType.USER:
                content = message.content.lower()
                
                # Extract genre preferences
                genre_keywords = ["action", "romance", "comedy", "drama", "thriller", "horror", "sci-fi", "fantasy"]
                for genre in genre_keywords:
                    if genre in content and "love" in content or "like" in content:
                        preferences["favorite_genres"].append(genre)
        
        # Analyze search results for patterns
        if state.current_context and state.current_context.results:
            # Find common genres in highly scored results
            high_scored = [r for r in state.current_context.results if r.get("score", 0) > 0.8]
            genre_counts = {}
            
            for result in high_scored:
                for tag in result.get("tags", []):
                    genre_counts[tag] = genre_counts.get(tag, 0) + 1
            
            # Add frequently occurring genres
            for genre, count in genre_counts.items():
                if count >= 2 and genre not in preferences["favorite_genres"]:
                    preferences["favorite_genres"].append(genre)
        
        return preferences
    
    def _determine_next_stage(self, flow: ConversationFlow, state: SmartOrchestrationState) -> Optional[str]:
        """Determine the next stage in the conversation flow."""
        current_stage = flow.current_stage
        current_index = next(
            (i for i, stage in enumerate(flow.stages) if stage["name"] == current_stage), 
            -1
        )
        
        if current_index == -1 or current_index >= len(flow.stages) - 1:
            return None
        
        # Check branch conditions
        next_stage_name = flow.stages[current_index + 1]["name"]
        branch_key = f"{current_stage}_to_{next_stage_name}"
        
        if branch_key in flow.branch_conditions:
            conditions = flow.branch_conditions[branch_key]
            
            # Check minimum results condition
            if "min_results" in conditions:
                if not state.current_context or len(state.current_context.results) < conditions["min_results"]:
                    return None
            
            # Check confidence threshold
            if "confidence_threshold" in conditions:
                if not state.current_context or not state.current_context.results:
                    return None
                avg_confidence = sum(r.get("score", 0.0) for r in state.current_context.results) / len(state.current_context.results)
                if avg_confidence < conditions["confidence_threshold"]:
                    return None
        
        return next_stage_name


class SmartOrchestrationEngine:
    """Main engine for Phase 6B smart orchestration features."""
    
    def __init__(self, adapter_registry: MCPAdapterRegistry):
        self.adapter_registry = adapter_registry
        self.query_orchestrator = QueryChainOrchestrator(adapter_registry)
        self.refinement_engine = ResultRefinementEngine(adapter_registry)
        self.flow_manager = ConversationFlowManager(adapter_registry)
    
    async def process_complex_conversation(
        self, 
        state: SmartOrchestrationState, 
        message: str
    ) -> SmartOrchestrationState:
        """Process a complex conversation with smart orchestration."""
        step_start = time.time()
        
        # Add user message
        user_message = WorkflowMessage(
            message_type=MessageType.USER,
            content=message
        )
        state.add_message(user_message)
        
        # Determine if this is a complex query requiring orchestration
        complexity_score = self._assess_query_complexity(message, state)
        
        if complexity_score > 0.7:  # High complexity - use orchestration
            orchestrated_state = await self._execute_orchestrated_workflow(state, message)
        else:  # Lower complexity - use standard flow
            orchestrated_state = await self._execute_standard_workflow(state, message)
        
        execution_time = (time.time() - step_start) * 1000
        
        # Add orchestration step
        orchestration_step = WorkflowStep(
            step_type=WorkflowStepType.ORCHESTRATION,
            parameters={"complexity_score": complexity_score, "message": message},
            result={"orchestrated": complexity_score > 0.7},
            confidence=0.9,
            execution_time_ms=execution_time,
            reasoning=f"Orchestrated workflow for {'complex' if complexity_score > 0.7 else 'standard'} query"
        )
        orchestrated_state.add_workflow_step(orchestration_step)
        
        return orchestrated_state
    
    async def process_multimodal_orchestration(
        self, 
        state: SmartOrchestrationState,
        message: str,
        image_data: str,
        text_weight: float = 0.7
    ) -> SmartOrchestrationState:
        """Process multimodal conversation with orchestration."""
        # Create multimodal context
        if not state.current_context:
            state.current_context = AnimeSearchContext()
        
        state.current_context.query = message
        state.current_context.image_data = image_data
        state.current_context.text_weight = text_weight
        
        # Create multimodal flow
        flow = self.flow_manager.create_multimodal_flow(message, True)
        state.conversation_flow = flow
        
        # Execute multimodal workflow
        while flow.current_stage:
            flow, stage_steps = await self.flow_manager.execute_flow_stage(flow, state)
            for step in stage_steps:
                state.add_workflow_step(step)
            
            # Break if no next stage
            if not flow.current_stage or flow.current_stage == stage_steps[-1].tool_name if stage_steps else False:
                break
        
        # Generate final response
        response_content = self._generate_multimodal_response(state)
        assistant_message = WorkflowMessage(
            message_type=MessageType.ASSISTANT,
            content=response_content
        )
        state.add_message(assistant_message)
        
        return state
    
    def _assess_query_complexity(self, message: str, state: SmartOrchestrationState) -> float:
        """Assess the complexity of a user query."""
        complexity_factors = 0.0
        
        # Check for multiple requirements
        if "and" in message.lower() or "but" in message.lower():
            complexity_factors += 0.3
        
        # Check for comparison requests
        comparison_words = ["compare", "versus", "vs", "difference", "similar", "like"]
        if any(word in message.lower() for word in comparison_words):
            complexity_factors += 0.3
        
        # Check for filtering/refinement requests
        filter_words = ["except", "not", "exclude", "only", "specifically"]
        if any(word in message.lower() for word in filter_words):
            complexity_factors += 0.2
        
        # Check conversation history depth
        if len(state.messages) > 3:
            complexity_factors += 0.2
        
        # Check for multi-step requirements
        step_words = ["first", "then", "after", "next", "finally"]
        if any(word in message.lower() for word in step_words):
            complexity_factors += 0.4
        
        return min(complexity_factors, 1.0)
    
    async def _execute_orchestrated_workflow(
        self, 
        state: SmartOrchestrationState, 
        message: str
    ) -> SmartOrchestrationState:
        """Execute an orchestrated workflow for complex queries."""
        # Create query chain for multi-step discovery
        chain = state.create_query_chain(message)
        
        # Break down complex query into sub-queries
        sub_queries = self._decompose_complex_query(message)
        for sub_query in sub_queries:
            state.add_to_chain(chain.chain_id, sub_query, {"type": "decomposition"})
        
        # Execute query chain
        chain_results = await self.query_orchestrator.execute_chain(chain, state)
        
        # Create discovery flow
        flow = self.flow_manager.create_discovery_flow(message)
        state.conversation_flow = flow
        
        # Set up context with initial results
        if not state.current_context:
            state.current_context = AnimeSearchContext()
        state.current_context.query = message
        
        # Combine results from chain
        all_results = []
        for query_results in chain_results.values():
            all_results.extend(query_results)
        
        state.current_context.results = all_results
        
        # Execute refinement
        if all_results:
            criteria = RefinementCriteria(
                min_confidence=0.75,
                max_iterations=3,
                target_result_count=12
            )
            
            refined_results, refinement_steps = await self.refinement_engine.refine_results(
                all_results, criteria, state.current_context
            )
            
            for step in refinement_steps:
                state.add_workflow_step(step)
            
            state.current_context.results = refined_results
        
        # Generate orchestrated response
        response_content = self._generate_orchestrated_response(state, chain)
        assistant_message = WorkflowMessage(
            message_type=MessageType.ASSISTANT,
            content=response_content
        )
        state.add_message(assistant_message)
        
        return state
    
    async def _execute_standard_workflow(
        self, 
        state: SmartOrchestrationState, 
        message: str
    ) -> SmartOrchestrationState:
        """Execute standard workflow for simpler queries."""
        # Set up context
        if not state.current_context:
            state.current_context = AnimeSearchContext()
        state.current_context.query = message
        
        # Perform search
        results = await self.adapter_registry.invoke_tool(
            "search_anime", 
            {"query": message, "limit": 10}
        )
        
        state.current_context.results = results
        
        # Create search step
        search_step = WorkflowStep(
            step_type=WorkflowStepType.SEARCH,
            tool_name="search_anime",
            parameters={"query": message, "limit": 10},
            result={"results": results},
            confidence=sum(r.get("score", 0.0) for r in results) / len(results) if results else 0.0,
            reasoning="Standard search for straightforward query"
        )
        state.add_workflow_step(search_step)
        
        # Generate standard response
        response_content = self._generate_standard_response(state)
        assistant_message = WorkflowMessage(
            message_type=MessageType.ASSISTANT,
            content=response_content
        )
        state.add_message(assistant_message)
        
        return state
    
    def _decompose_complex_query(self, message: str) -> List[str]:
        """Decompose a complex query into simpler sub-queries."""
        sub_queries = []
        
        # Split on common connectors
        if " and " in message.lower():
            parts = message.lower().split(" and ")
            for part in parts:
                if part.strip():
                    sub_queries.append(part.strip())
        elif " but " in message.lower():
            parts = message.lower().split(" but ")
            sub_queries.append(parts[0].strip())
            if len(parts) > 1:
                sub_queries.append(f"not {parts[1].strip()}")
        else:
            sub_queries.append(message)
        
        return sub_queries
    
    def _generate_orchestrated_response(
        self, 
        state: SmartOrchestrationState, 
        chain: QueryChain
    ) -> str:
        """Generate response for orchestrated workflow."""
        if not state.current_context or not state.current_context.results:
            return "I couldn't find relevant anime based on your complex query. Please try rephrasing your request."
        
        results = state.current_context.results[:5]  # Top 5 results
        
        response_parts = [
            f"I found {len(results)} anime that match your complex requirements:",
            ""
        ]
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "Unknown")
            score = result.get("score", 0.0)
            tags = result.get("tags", [])[:3]  # First 3 tags
            
            response_parts.append(
                f"{i}. **{title}** (Match: {score:.1%})"
            )
            if tags:
                response_parts.append(f"   - Genres: {', '.join(tags)}")
        
        response_parts.extend([
            "",
            f"This search involved {len(chain.queries)} query steps with intelligent orchestration.",
            f"Confidence scores: {', '.join(f'{q}: {s:.1%}' for q, s in chain.confidence_scores.items())}"
        ])
        
        return "\n".join(response_parts)
    
    def _generate_multimodal_response(self, state: SmartOrchestrationState) -> str:
        """Generate response for multimodal workflow."""
        if not state.current_context or not state.current_context.results:
            return "I couldn't find anime matching both your text and image criteria."
        
        results = state.current_context.results[:5]
        
        response_parts = [
            f"Based on your multimodal search (text + image), I found {len(results)} matching anime:",
            ""
        ]
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "Unknown")
            score = result.get("score", 0.0)
            
            response_parts.append(
                f"{i}. **{title}** (Similarity: {score:.1%})"
            )
        
        response_parts.extend([
            "",
            f"Search combined text query '{state.current_context.query}' with visual similarity analysis.",
            f"Text weight: {state.current_context.text_weight:.1%}, Image weight: {1-state.current_context.text_weight:.1%}"
        ])
        
        return "\n".join(response_parts)
    
    def _generate_standard_response(self, state: SmartOrchestrationState) -> str:
        """Generate response for standard workflow."""
        if not state.current_context or not state.current_context.results:
            return "I couldn't find anime matching your request. Please try a different search term."
        
        results = state.current_context.results[:5]
        
        response_parts = [
            f"I found {len(results)} anime for your search:",
            ""
        ]
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "Unknown")
            score = result.get("score", 0.0)
            
            response_parts.append(
                f"{i}. **{title}** (Relevance: {score:.1%})"
            )
        
        return "\n".join(response_parts)