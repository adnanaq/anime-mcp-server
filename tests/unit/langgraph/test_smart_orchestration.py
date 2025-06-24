"""Tests for Phase 6B smart orchestration features."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

from src.langgraph.smart_orchestration import (
    QueryChainOrchestrator,
    ResultRefinementEngine,
    ConversationFlowManager,
    SmartOrchestrationEngine
)
from src.langgraph.models import (
    SmartOrchestrationState,
    QueryChain,
    RefinementCriteria,
    ConversationFlow,
    AnimeSearchContext,
    UserPreferences,
    WorkflowStepType
)
from src.langgraph.adapters import MCPAdapterRegistry


class TestQueryChainOrchestrator:
    """Test query chain orchestration."""
    
    @pytest.mark.asyncio
    async def test_execute_chain_semantic_search(self):
        """Test executing a query chain with semantic search."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        mock_registry.invoke_tool = AsyncMock(return_value=[
            {"title": "Naruto", "anime_id": "naruto123", "score": 0.95}
        ])
        
        orchestrator = QueryChainOrchestrator(mock_registry)
        
        chain = QueryChain(
            chain_id="test_chain",
            queries=["action anime", "shounen series"]
        )
        
        state = SmartOrchestrationState(session_id="test")
        
        results = await orchestrator.execute_chain(chain, state)
        
        assert len(results) == 2
        assert "query_0" in results
        assert "query_1" in results
        assert len(chain.confidence_scores) == 2
        
        # Should have called search_anime for both queries
        assert mock_registry.invoke_tool.call_count == 2
    
    @pytest.mark.asyncio
    async def test_execute_chain_similarity_search(self):
        """Test chain execution with similarity search strategy."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        mock_registry.invoke_tool = AsyncMock(side_effect=[
            [{"title": "Naruto", "anime_id": "naruto123", "score": 0.95}],  # First search
            [{"title": "One Piece", "anime_id": "onepiece456", "score": 0.90}]   # Similarity search
        ])
        
        orchestrator = QueryChainOrchestrator(mock_registry)
        
        chain = QueryChain(
            chain_id="similarity_chain",
            queries=["action anime", "similar to the first result"]
        )
        
        state = SmartOrchestrationState(session_id="test")
        
        results = await orchestrator.execute_chain(chain, state)
        
        assert len(results) == 2
        
        # First call should be search_anime
        first_call = mock_registry.invoke_tool.call_args_list[0]
        assert first_call[0][0] == "search_anime"
        
        # Second call should be find_similar_anime
        second_call = mock_registry.invoke_tool.call_args_list[1]
        assert second_call[0][0] == "find_similar_anime"
        assert second_call[0][1]["anime_id"] == "naruto123"
    
    def test_determine_search_strategy(self):
        """Test search strategy determination."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        orchestrator = QueryChainOrchestrator(mock_registry)
        
        chain = QueryChain(chain_id="test")
        
        # First query should always be semantic
        strategy = orchestrator._determine_search_strategy("action anime", chain, 0)
        assert strategy == "semantic"
        
        # Query with similarity keywords should use similarity strategy
        strategy = orchestrator._determine_search_strategy("similar anime", chain, 1)
        assert strategy == "similarity"
        
        # Regular query at index > 0 should be semantic
        strategy = orchestrator._determine_search_strategy("romance anime", chain, 1)
        assert strategy == "semantic"
    
    def test_calculate_query_confidence(self):
        """Test query confidence calculation."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        orchestrator = QueryChainOrchestrator(mock_registry)
        
        # Test with good results
        results = [
            {"score": 0.9},
            {"score": 0.8},
            {"score": 0.85}
        ]
        confidence = orchestrator._calculate_query_confidence(results, "test query")
        # Should be avg_score (0.85) + bonus (0.03) = 0.88
        assert 0.85 <= confidence <= 0.95
        
        # Test with empty results
        confidence = orchestrator._calculate_query_confidence([], "test query")
        assert confidence == 0.0


class TestResultRefinementEngine:
    """Test result refinement engine."""
    
    @pytest.mark.asyncio
    async def test_refine_results_basic(self):
        """Test basic result refinement."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        mock_registry.invoke_tool = AsyncMock(return_value=[])  # No expansion needed
        
        engine = ResultRefinementEngine(mock_registry)
        
        initial_results = [
            {"title": "High Quality", "score": 0.9, "tags": ["action"]},
            {"title": "Low Quality", "score": 0.3, "tags": ["action"]},
            {"title": "Medium Quality", "score": 0.75, "tags": ["action"]}
        ]
        
        criteria = RefinementCriteria(
            min_confidence=0.7,
            max_iterations=1,
            target_result_count=5
        )
        
        context = AnimeSearchContext(query="action anime")
        
        refined_results, steps = await engine.refine_results(initial_results, criteria, context)
        
        # Should filter out low quality result
        assert len(refined_results) == 2
        assert all(r["score"] >= 0.7 for r in refined_results)
        assert len(steps) == 1
        assert steps[0].step_type == WorkflowStepType.REFINEMENT
    
    @pytest.mark.asyncio
    async def test_refine_results_with_expansion(self):
        """Test refinement with search expansion."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        mock_registry.invoke_tool = AsyncMock(return_value=[
            {"title": "Expanded Result", "score": 0.8, "tags": ["action"]}
        ])
        
        engine = ResultRefinementEngine(mock_registry)
        
        initial_results = [
            {"title": "High Quality", "score": 0.9, "tags": ["action"], "anime_id": "anime1"}
        ]
        
        criteria = RefinementCriteria(
            min_confidence=0.7,
            target_result_count=5  # Need more results
        )
        
        context = AnimeSearchContext(query="action anime")
        
        refined_results, steps = await engine.refine_results(initial_results, criteria, context)
        
        # Should have expanded to get more results
        assert len(refined_results) >= 1
        # Should have called find_similar_anime for expansion
        mock_registry.invoke_tool.assert_called_with(
            "find_similar_anime",
            {"anime_id": "anime1", "limit": 2}  # After multiple iterations, fewer needed
        )
    
    def test_apply_quality_filters(self):
        """Test quality filter application."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        engine = ResultRefinementEngine(mock_registry)
        
        results = [
            {"score": 0.9},
            {"score": 0.5},
            {"score": 0.8}
        ]
        
        criteria = RefinementCriteria(min_confidence=0.7)
        
        filtered = engine._apply_quality_filters(results, criteria)
        
        assert len(filtered) == 2
        assert all(r["score"] >= 0.7 for r in filtered)
    
    def test_apply_focus_filtering(self):
        """Test focus area filtering."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        engine = ResultRefinementEngine(mock_registry)
        
        results = [
            {"title": "Action Hero", "tags": ["action", "adventure"]},
            {"title": "Romance Story", "tags": ["romance", "drama"]},
            {"title": "Action Movie", "tags": ["action", "thriller"]}
        ]
        
        focus_areas = ["action"]
        
        focused = engine._apply_focus_filtering(results, focus_areas)
        
        assert len(focused) == 2
        assert all("action" in r["tags"] for r in focused)
    
    def test_apply_exclusions(self):
        """Test exclusion filtering."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        engine = ResultRefinementEngine(mock_registry)
        
        results = [
            {"title": "Good Anime", "tags": ["action"]},
            {"title": "Horror Anime", "tags": ["horror"]},
            {"title": "Another Good", "tags": ["adventure"]}
        ]
        
        exclusions = ["horror"]
        
        filtered = engine._apply_exclusions(results, exclusions)
        
        assert len(filtered) == 2
        assert all("horror" not in r["tags"] for r in filtered)


class TestConversationFlowManager:
    """Test conversation flow management."""
    
    def test_create_discovery_flow(self):
        """Test creating a discovery conversation flow."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        manager = ConversationFlowManager(mock_registry)
        
        flow = manager.create_discovery_flow("find action anime")
        
        assert flow.flow_type == "discovery"
        assert flow.current_stage == "initial_search"
        assert len(flow.stages) == 5
        assert "branch_conditions" in flow.__dict__
    
    def test_create_multimodal_flow(self):
        """Test creating a multimodal conversation flow."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        manager = ConversationFlowManager(mock_registry)
        
        # With image
        flow = manager.create_multimodal_flow("find mecha anime", True)
        
        assert flow.flow_type == "multimodal"
        assert flow.current_stage == "multimodal_analysis"
        assert len(flow.stages) == 4  # Includes visual_similarity stage
        assert flow.context_carryover["has_image"] is True
        
        # Without image
        flow_no_image = manager.create_multimodal_flow("find anime", False)
        assert len(flow_no_image.stages) == 3  # No visual_similarity stage
    
    @pytest.mark.asyncio
    async def test_execute_initial_search_stage(self):
        """Test executing initial search stage."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        mock_registry.invoke_tool = AsyncMock(return_value=[
            {"title": "Test Anime", "score": 0.9}
        ])
        
        manager = ConversationFlowManager(mock_registry)
        
        flow = ConversationFlow(
            flow_id="test",
            flow_type="discovery",
            current_stage="initial_search",
            stages=[]
        )
        
        state = SmartOrchestrationState(session_id="test")
        state.current_context = AnimeSearchContext(query="action anime")
        
        updated_flow, steps = await manager.execute_flow_stage(flow, state)
        
        assert len(steps) == 1
        assert steps[0].step_type == WorkflowStepType.SEARCH
        assert state.current_context.results is not None
        mock_registry.invoke_tool.assert_called_once_with(
            "search_anime",
            {"query": "action anime", "limit": 15}
        )
    
    @pytest.mark.asyncio
    async def test_execute_multimodal_analysis_stage(self):
        """Test executing multimodal analysis stage."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        mock_registry.invoke_tool = AsyncMock(return_value=[
            {"title": "Multimodal Result", "score": 0.85}
        ])
        
        manager = ConversationFlowManager(mock_registry)
        
        flow = ConversationFlow(
            flow_id="test",
            flow_type="multimodal",
            current_stage="multimodal_analysis",
            stages=[]
        )
        
        state = SmartOrchestrationState(session_id="test")
        state.current_context = AnimeSearchContext(
            query="mecha anime",
            image_data="base64_image_data",
            text_weight=0.7
        )
        
        updated_flow, steps = await manager.execute_flow_stage(flow, state)
        
        assert len(steps) == 1
        assert steps[0].step_type == WorkflowStepType.SEARCH
        assert steps[0].tool_name == "multimodal_search"
        
        # Should call multimodal search tool
        mock_registry.invoke_tool.assert_called_once_with(
            "search_multimodal_anime",
            {
                "query": "mecha anime",
                "image_data": "base64_image_data",
                "text_weight": 0.7,
                "limit": 10
            }
        )
    
    def test_extract_preferences_from_context(self):
        """Test preference extraction from conversation context."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        manager = ConversationFlowManager(mock_registry)
        
        state = SmartOrchestrationState(session_id="test")
        
        # Add messages with preference indicators
        from src.langgraph.models import WorkflowMessage, MessageType
        state.messages = [
            WorkflowMessage(message_type=MessageType.USER, content="I love action anime"),
            WorkflowMessage(message_type=MessageType.USER, content="Romance is also nice")
        ]
        
        # Add search results with patterns
        state.current_context = AnimeSearchContext()
        state.current_context.results = [
            {"tags": ["action", "adventure"], "score": 0.9},
            {"tags": ["action", "thriller"], "score": 0.85}
        ]
        
        preferences = manager._extract_preferences_from_context(state)
        
        assert "favorite_genres" in preferences
        # Should extract action from high-scored results
        assert "action" in preferences["favorite_genres"]


class TestSmartOrchestrationEngine:
    """Test smart orchestration engine."""
    
    @pytest.mark.asyncio
    async def test_process_simple_conversation(self):
        """Test processing a simple conversation."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        mock_registry.invoke_tool = AsyncMock(return_value=[
            {"title": "Simple Result", "score": 0.8}
        ])
        
        engine = SmartOrchestrationEngine(mock_registry)
        
        state = SmartOrchestrationState(session_id="test")
        
        result_state = await engine.process_complex_conversation(state, "find action anime")
        
        # Should use standard workflow for simple query
        assert len(result_state.messages) == 2  # User + Assistant
        assert len(result_state.workflow_steps) >= 2  # Search + Orchestration
        assert result_state.current_context is not None
        assert result_state.current_context.results is not None
    
    @pytest.mark.asyncio
    async def test_process_complex_conversation(self):
        """Test processing a complex conversation."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        mock_registry.invoke_tool = AsyncMock(return_value=[
            {"title": "Complex Result", "score": 0.9}
        ])
        
        engine = SmartOrchestrationEngine(mock_registry)
        
        state = SmartOrchestrationState(session_id="test")
        
        # Complex query with multiple requirements
        complex_query = "find action anime but not horror and similar to naruto"
        
        result_state = await engine.process_complex_conversation(state, complex_query)
        
        # Should use orchestrated workflow
        assert len(result_state.messages) == 2
        assert len(result_state.workflow_steps) >= 2
        assert len(result_state.query_chains) > 0  # Should create query chain
        assert result_state.conversation_flow is not None
    
    @pytest.mark.asyncio
    async def test_process_multimodal_orchestration(self):
        """Test multimodal orchestration processing."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        mock_registry.invoke_tool = AsyncMock(return_value=[
            {"title": "Multimodal Result", "score": 0.88}
        ])
        
        engine = SmartOrchestrationEngine(mock_registry)
        
        state = SmartOrchestrationState(session_id="test")
        
        result_state = await engine.process_multimodal_orchestration(
            state,
            "find mecha anime",
            "base64_image_data",
            0.7
        )
        
        assert result_state.current_context is not None
        assert result_state.current_context.query == "find mecha anime"
        assert result_state.current_context.image_data == "base64_image_data"
        assert result_state.current_context.text_weight == 0.7
        assert result_state.conversation_flow is not None
        assert result_state.conversation_flow.flow_type == "multimodal"
    
    def test_assess_query_complexity(self):
        """Test query complexity assessment."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        engine = SmartOrchestrationEngine(mock_registry)
        
        state = SmartOrchestrationState(session_id="test")
        
        # Simple query
        simple_complexity = engine._assess_query_complexity("find action anime", state)
        assert simple_complexity < 0.5
        
        # Complex query with multiple factors
        complex_query = "find action anime and compare them but not horror first then show similar"
        complex_complexity = engine._assess_query_complexity(complex_query, state)
        assert complex_complexity > 0.7
        
        # Query with conversation history increases complexity
        state.messages = [None] * 5  # Simulate long conversation
        history_complexity = engine._assess_query_complexity("find anime", state)
        assert history_complexity > 0.2
    
    def test_decompose_complex_query(self):
        """Test complex query decomposition."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        engine = SmartOrchestrationEngine(mock_registry)
        
        # Query with 'and'
        and_query = "find action anime and romance series"
        and_decomposed = engine._decompose_complex_query(and_query)
        assert len(and_decomposed) == 2
        assert "action" in and_decomposed[0]
        assert "romance" in and_decomposed[1]
        
        # Query with 'but'
        but_query = "find action anime but not horror"
        but_decomposed = engine._decompose_complex_query(but_query)
        assert len(but_decomposed) == 2
        assert "action" in but_decomposed[0]
        assert "not horror" in but_decomposed[1]
        
        # Simple query
        simple_query = "find action anime"
        simple_decomposed = engine._decompose_complex_query(simple_query)
        assert len(simple_decomposed) == 1
        assert simple_decomposed[0] == simple_query
    
    def test_generate_orchestrated_response(self):
        """Test orchestrated response generation."""
        mock_registry = Mock(spec=MCPAdapterRegistry)
        engine = SmartOrchestrationEngine(mock_registry)
        
        state = SmartOrchestrationState(session_id="test")
        state.current_context = AnimeSearchContext()
        state.current_context.results = [
            {"title": "Naruto", "score": 0.95, "tags": ["action", "shounen"]},
            {"title": "One Piece", "score": 0.90, "tags": ["action", "adventure"]}
        ]
        
        chain = QueryChain(
            chain_id="test_chain",
            queries=["action anime", "shounen series"],
            confidence_scores={"action anime": 0.9, "shounen series": 0.85}
        )
        
        response = engine._generate_orchestrated_response(state, chain)
        
        assert "2 anime" in response
        assert "Naruto" in response
        assert "One Piece" in response
        assert "2 query steps" in response
        assert "90%" in response  # Confidence scores


class TestSmartOrchestrationModels:
    """Test smart orchestration model classes."""
    
    def test_query_chain_creation(self):
        """Test QueryChain model creation and usage."""
        chain = QueryChain(
            chain_id="test_chain",
            queries=["query1", "query2"]
        )
        
        assert chain.chain_id == "test_chain"
        assert len(chain.queries) == 2
        assert len(chain.relationships) == 0
        assert len(chain.results_mapping) == 0
    
    def test_refinement_criteria_defaults(self):
        """Test RefinementCriteria default values."""
        criteria = RefinementCriteria()
        
        assert criteria.min_confidence == 0.7
        assert criteria.max_iterations == 3
        assert criteria.target_result_count == 10
        assert len(criteria.focus_areas) == 0
        assert len(criteria.exclusion_criteria) == 0
    
    def test_orchestration_plan_creation(self):
        """Test OrchestrationPlan model."""
        from src.langgraph.models import OrchestrationPlan
        
        plan = OrchestrationPlan(
            plan_id="test_plan",
            steps=[{"step": "search"}, {"step": "refine"}],
            dependencies={"refine": ["search"]}
        )
        
        assert plan.plan_id == "test_plan"
        assert len(plan.steps) == 2
        assert "refine" in plan.dependencies
        assert plan.dependencies["refine"] == ["search"]
    
    def test_smart_orchestration_state_extensions(self):
        """Test SmartOrchestrationState extended functionality."""
        state = SmartOrchestrationState(session_id="test")
        
        # Test query chain creation
        chain = state.create_query_chain("initial query")
        assert len(state.query_chains) == 1
        assert chain.chain_id == "chain_1"
        assert chain.queries == ["initial query"]
        
        # Test adding to chain
        success = state.add_to_chain("chain_1", "follow-up query", {"type": "follow_up"})
        assert success is True
        assert len(state.query_chains[0].queries) == 2
        assert len(state.query_chains[0].relationships) == 1
        
        # Test adding to non-existent chain
        success = state.add_to_chain("nonexistent", "query", {})
        assert success is False
    
    def test_adaptive_preferences_interaction(self):
        """Test AdaptivePreferences interaction tracking."""
        from src.langgraph.models import AdaptivePreferences, UserPreferences
        
        base_prefs = UserPreferences(favorite_genres=["action"])
        adaptive_prefs = AdaptivePreferences(base_preferences=base_prefs)
        
        interaction = {
            "query": "find romance anime",
            "selected_results": [{"tags": ["romance"]}],
            "feedback": "positive"
        }
        
        adaptive_prefs.adapt_from_interaction(interaction)
        
        assert len(adaptive_prefs.interaction_history) == 1
        assert adaptive_prefs.interaction_history[0] == interaction
        assert adaptive_prefs.last_update > 0