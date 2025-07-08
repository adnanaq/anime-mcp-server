"""
Comprehensive tests for Task #89: Stateful Routing Memory and Context Learning.

Tests the core functionality of the stateful routing system including:
- Conversation context memory across user sessions  
- Agent handoff sequence learning and optimization
- Query pattern embedding and similarity matching
- User preference learning for personalized routing
- Integration with existing ReactAgent workflow
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.langgraph.stateful_routing_memory import (
    StatefulRoutingEngine,
    RoutingMemoryStore,
    ConversationContextMemory,
    AgentHandoffOptimizer,
    QueryPattern,
    AgentSequencePattern,
    UserPreferenceProfile,
    RoutingStrategy,
    MemoryScope,
    get_stateful_routing_engine
)
from src.langgraph.react_agent_workflow import (
    ReactAgentWorkflowEngine,
    ExecutionMode,
    LLMProvider,
    create_react_agent_workflow_engine
)


class TestQueryPattern:
    """Test QueryPattern data class functionality."""
    
    def test_query_pattern_creation(self):
        """Test creating and converting QueryPattern objects."""
        pattern = QueryPattern(
            pattern_id="test_001",
            query_text="find mecha anime",
            intent_type="search",
            successful_agent_sequences=[["search_agent", "enhancement_agent"]],
            success_rate=0.85,
            usage_count=10
        )
        
        assert pattern.pattern_id == "test_001"
        assert pattern.query_text == "find mecha anime"
        assert pattern.success_rate == 0.85
        assert len(pattern.successful_agent_sequences) == 1
        
        # Test to_dict conversion
        pattern_dict = pattern.to_dict()
        assert isinstance(pattern_dict, dict)
        assert pattern_dict["pattern_id"] == "test_001"
        
        # Test from_dict conversion
        restored_pattern = QueryPattern.from_dict(pattern_dict)
        assert restored_pattern.pattern_id == pattern.pattern_id
        assert restored_pattern.success_rate == pattern.success_rate


class TestAgentSequencePattern:
    """Test AgentSequencePattern learning functionality."""
    
    def test_agent_sequence_pattern_creation(self):
        """Test creating AgentSequencePattern objects."""
        pattern = AgentSequencePattern(
            sequence_id="mal->anilist->enhancement",
            agent_sequence=["mal_agent", "anilist_agent", "enhancement_agent"],
            context_triggers=["search", "detailed_info"],
            success_rate=0.9,
            usage_count=25
        )
        
        assert pattern.sequence_id == "mal->anilist->enhancement"
        assert len(pattern.agent_sequence) == 3
        assert pattern.success_rate == 0.9
        assert "search" in pattern.context_triggers


class TestUserPreferenceProfile:
    """Test UserPreferenceProfile learning functionality."""
    
    def test_user_preference_profile_creation(self):
        """Test creating and updating UserPreferenceProfile."""
        profile = UserPreferenceProfile(
            user_id="user_123",
            preferred_platforms=["myanimelist", "anilist"],
            preferred_genres=["Action", "Mecha"],
            interaction_count=15,
            confidence_score=0.6
        )
        
        assert profile.user_id == "user_123"
        assert "myanimelist" in profile.preferred_platforms
        assert "Action" in profile.preferred_genres
        assert profile.confidence_score == 0.6


class TestConversationContextMemory:
    """Test conversation context memory functionality."""
    
    def test_session_context_storage(self):
        """Test storing and retrieving session contexts."""
        memory = ConversationContextMemory(max_sessions=10, session_ttl_hours=1)
        
        # Store session context
        session_id = "session_001"
        context = {
            "last_query": "find attack on titan",
            "preferred_platforms": ["myanimelist"],
            "query_count": 3
        }
        
        memory.store_session_context(session_id, context)
        
        # Retrieve session context
        retrieved_context = memory.get_session_context(session_id)
        assert retrieved_context is not None
        assert retrieved_context["last_query"] == "find attack on titan"
        assert len(retrieved_context["preferred_platforms"]) == 1
    
    def test_session_context_expiry(self):
        """Test session context expiry functionality."""
        memory = ConversationContextMemory(max_sessions=10, session_ttl_hours=1)
        
        session_id = "session_002"
        context = {"test": "data"}
        
        memory.store_session_context(session_id, context)
        
        # Manually set an old timestamp to simulate expiry
        from datetime import datetime, timedelta
        memory.session_timestamps[session_id] = datetime.now() - timedelta(hours=2)
        
        # Force cleanup
        memory._cleanup_expired_sessions()
        
        # Context should be expired
        retrieved_context = memory.get_session_context(session_id)
        assert retrieved_context is None
    
    def test_user_profile_updates(self):
        """Test user preference profile updates."""
        memory = ConversationContextMemory()
        
        user_id = "user_456"
        interaction_data = {
            "platforms_used": ["myanimelist", "anilist"],
            "query_complexity": "complex",
            "successful_agents": ["search_agent", "enhancement_agent"]
        }
        
        # Update user profile
        memory.update_user_profile(user_id, interaction_data)
        
        # Check profile was created and updated
        profile = memory.get_user_profile(user_id)
        assert profile is not None
        assert profile.user_id == user_id
        assert profile.interaction_count == 1
        assert "myanimelist" in profile.preferred_platforms
        assert profile.typical_query_complexity == "complex"


class TestAgentHandoffOptimizer:
    """Test agent handoff sequence learning and optimization."""
    
    def test_handoff_optimization_learning(self):
        """Test learning from agent execution sequences."""
        optimizer = AgentHandoffOptimizer(max_patterns=100)
        
        # Simulate successful execution
        agent_sequence = ["search_agent", "mal_agent", "enhancement_agent"]
        execution_result = {
            "execution_time_ms": 250,
            "anime_results": [{"title": "Attack on Titan"}],
            "platforms_queried": ["myanimelist"]
        }
        context = {"intent_type": "search"}
        
        # Learn from execution
        optimizer.learn_from_execution(agent_sequence, execution_result, context)
        
        # Check pattern was learned
        sequence_key = "->".join(agent_sequence)
        assert sequence_key in optimizer.sequence_patterns
        
        pattern = optimizer.sequence_patterns[sequence_key]
        assert pattern.usage_count == 1
        assert pattern.success_rate == 1.0  # Successful execution
        assert "search" in pattern.optimal_for_intents
    
    def test_optimal_sequence_retrieval(self):
        """Test retrieving optimal agent sequences."""
        optimizer = AgentHandoffOptimizer()
        
        # Learn a successful pattern
        agent_sequence = ["search_agent", "anilist_agent"]
        execution_result = {"anime_results": [{"title": "One Piece"}]}
        context = {"intent_type": "discovery"}
        
        optimizer.learn_from_execution(agent_sequence, execution_result, context)
        
        # Get optimal sequence for same intent
        optimal_sequence = optimizer.get_optimal_sequence("discovery", context)
        assert optimal_sequence == agent_sequence
    
    def test_best_handoff_target(self):
        """Test finding best handoff targets."""
        optimizer = AgentHandoffOptimizer()
        
        # Simulate multiple successful handoffs
        for _ in range(5):
            optimizer.agent_handoff_success[("search_agent", "mal_agent")] = 0.8
            optimizer.agent_handoff_success[("search_agent", "anilist_agent")] = 0.6
        
        # Get best handoff target
        best_target = optimizer.get_best_handoff_target("search_agent", {})
        assert best_target == "mal_agent"  # Higher success rate


class TestRoutingMemoryStore:
    """Test routing memory store functionality."""
    
    def test_pattern_storage_and_retrieval(self):
        """Test storing and retrieving query patterns."""
        store = RoutingMemoryStore(max_patterns=100)
        
        # Create test pattern
        pattern = QueryPattern(
            pattern_id="pattern_001",
            query_text="find action anime",
            intent_type="search",
            success_rate=0.8,
            usage_count=5
        )
        
        # Store pattern
        store.store_pattern(pattern)
        
        # Check pattern was stored
        assert pattern.pattern_id in store.query_patterns
        assert "search" in store.intent_patterns
        assert pattern.pattern_id in store.intent_patterns["search"]
    
    def test_similar_pattern_finding(self):
        """Test finding similar query patterns."""
        store = RoutingMemoryStore()
        
        # Store multiple patterns
        patterns = [
            QueryPattern("p1", query_text="find mecha anime", intent_type="search", success_rate=0.9),
            QueryPattern("p2", query_text="search mecha robots", intent_type="search", success_rate=0.8),
            QueryPattern("p3", query_text="comedy anime", intent_type="search", success_rate=0.7)
        ]
        
        for pattern in patterns:
            store.store_pattern(pattern)
        
        # Find similar patterns to "mecha"
        similar_patterns = store.find_similar_patterns("mecha robots", "search", limit=2)
        
        assert len(similar_patterns) >= 1
        # Should find mecha-related patterns first
        assert any("mecha" in p.query_text for p in similar_patterns)
    
    def test_best_patterns_for_intent(self):
        """Test getting best patterns for specific intent."""
        store = RoutingMemoryStore()
        
        # Store patterns with different success rates
        patterns = [
            QueryPattern("p1", intent_type="search", success_rate=0.9, usage_count=10),
            QueryPattern("p2", intent_type="search", success_rate=0.7, usage_count=5),
            QueryPattern("p3", intent_type="discovery", success_rate=0.8, usage_count=8)
        ]
        
        for pattern in patterns:
            store.store_pattern(pattern)
        
        # Get best patterns for search intent
        best_patterns = store.get_best_patterns_for_intent("search", limit=2)
        
        assert len(best_patterns) == 2
        # Should be sorted by success rate
        assert best_patterns[0].success_rate >= best_patterns[1].success_rate


class TestStatefulRoutingEngine:
    """Test the main stateful routing engine functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create a StatefulRoutingEngine for testing."""
        with patch('src.langgraph.stateful_routing_memory.get_settings'):
            return StatefulRoutingEngine()
    
    @pytest.mark.asyncio
    async def test_optimal_routing_decision(self, engine):
        """Test getting optimal routing decisions."""
        query = "find attack on titan episodes"
        intent = "search"
        session_id = "session_test"
        
        # Get optimal routing
        routing_decision = await engine.get_optimal_routing(
            query=query,
            intent=intent,
            session_id=session_id
        )
        
        assert isinstance(routing_decision, dict)
        assert "strategy" in routing_decision
        assert "confidence" in routing_decision
        assert "reasoning" in routing_decision
        assert routing_decision["strategy"] in [s.value for s in RoutingStrategy]
    
    @pytest.mark.asyncio
    async def test_learning_from_execution(self, engine):
        """Test learning from execution results."""
        query = "find one piece episodes"
        intent = "search"
        agent_sequence = ["search_agent", "mal_agent"]
        execution_result = {
            "anime_results": [{"title": "One Piece", "episodes": 1000}],
            "execution_time_ms": 180,
            "platforms_queried": ["myanimelist"]
        }
        session_id = "session_learn"
        
        # Learn from execution
        await engine.learn_from_execution(
            query=query,
            intent=intent,
            agent_sequence=agent_sequence,
            execution_result=execution_result,
            session_id=session_id
        )
        
        # Check that learning occurred
        assert len(engine.memory_store.query_patterns) > 0
        assert len(engine.handoff_optimizer.sequence_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_routing_with_user_preferences(self, engine):
        """Test routing decisions with user preference profiles."""
        user_id = "user_789"
        
        # Create user profile
        profile = UserPreferenceProfile(
            user_id=user_id,
            preferred_platforms=["anilist", "kitsu"],
            optimal_agent_sequences=[["anilist_agent", "enhancement_agent"]],
            confidence_score=0.8
        )
        engine.context_memory.user_profiles[user_id] = profile
        
        # Get routing decision
        routing_decision = await engine.get_optimal_routing(
            query="find romance anime",
            intent="search",
            user_id=user_id
        )
        
        # Should use user preferences
        assert routing_decision["confidence"] == 0.8
        assert routing_decision["agent_sequence"] == ["anilist_agent", "enhancement_agent"]
    
    def test_memory_stats(self, engine):
        """Test getting memory system statistics."""
        # Add some test data
        pattern = QueryPattern("test", query_text="test", intent_type="search")
        engine.memory_store.store_pattern(pattern)
        
        # Get stats
        stats = engine.get_memory_stats()
        
        assert isinstance(stats, dict)
        assert "query_patterns_stored" in stats
        assert "active_sessions" in stats
        assert "user_profiles" in stats
        assert "recent_success_rate" in stats
        assert stats["query_patterns_stored"] >= 1


class TestReactAgentIntegration:
    """Test integration with ReactAgent workflow system."""
    
    @pytest.fixture
    def mock_mcp_tools(self):
        """Create mock MCP tools for testing."""
        return {
            "search_anime": AsyncMock(return_value={"results": [{"title": "Test Anime"}]}),
            "get_anime_details": AsyncMock(return_value={"title": "Test Details"}),
            "find_similar_anime": AsyncMock(return_value={"similar": []})
        }
    
    @patch('src.langgraph.react_agent_workflow.get_settings')
    @patch('src.langgraph.react_agent_workflow.ChatOpenAI')
    def test_stateful_execution_mode_creation(self, mock_openai, mock_settings, mock_mcp_tools):
        """Test creating ReactAgent with stateful execution mode."""
        # Mock settings
        mock_settings.return_value.openai_api_key = "test_key"
        
        # Create engine with stateful mode
        engine = ReactAgentWorkflowEngine(
            mcp_tools=mock_mcp_tools,
            llm_provider=LLMProvider.OPENAI,
            execution_mode=ExecutionMode.STATEFUL
        )
        
        assert engine.execution_mode == ExecutionMode.STATEFUL
        assert engine.stateful_routing_engine is not None
        assert engine.super_step_executor is None  # Should not have super-step executor
    
    @patch('src.langgraph.react_agent_workflow.get_settings')
    @patch('src.langgraph.react_agent_workflow.ChatOpenAI')
    @pytest.mark.asyncio
    async def test_stateful_workflow_execution(self, mock_openai, mock_settings, mock_mcp_tools):
        """Test executing workflow with stateful routing."""
        # Mock settings and LLM
        mock_settings.return_value.openai_api_key = "test_key"
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        # Create engine with stateful mode
        engine = ReactAgentWorkflowEngine(
            mcp_tools=mock_mcp_tools,
            execution_mode=ExecutionMode.STATEFUL
        )
        
        # Mock the agent.ainvoke method
        mock_result = {
            "messages": ["Query: find naruto", "Results: Found Naruto anime series"]
        }
        engine.agent.ainvoke = AsyncMock(return_value=mock_result)
        
        # Test stateful execution method
        input_data = {"messages": ["find naruto"]}
        config = {"configurable": {"thread_id": "test_thread"}}
        
        result = await engine._execute_stateful_workflow(
            input_data=input_data,
            config=config,
            enhanced_message="find naruto",
            session_id="test_session",
            thread_id="test_thread"
        )
        
        assert result == mock_result
        assert engine.agent.ainvoke.called


class TestGlobalStatefulEngine:
    """Test global stateful routing engine functionality."""
    
    def test_global_engine_singleton(self):
        """Test that global engine returns same instance."""
        engine1 = get_stateful_routing_engine()
        engine2 = get_stateful_routing_engine()
        
        assert engine1 is engine2  # Should be same instance
    
    def test_global_engine_functionality(self):
        """Test that global engine has expected functionality."""
        engine = get_stateful_routing_engine()
        
        assert hasattr(engine, 'memory_store')
        assert hasattr(engine, 'context_memory')
        assert hasattr(engine, 'handoff_optimizer')
        assert hasattr(engine, 'get_optimal_routing')
        assert hasattr(engine, 'learn_from_execution')


class TestErrorHandling:
    """Test error handling in stateful routing system."""
    
    @pytest.mark.asyncio
    async def test_routing_error_fallback(self):
        """Test that routing errors fallback gracefully."""
        engine = StatefulRoutingEngine()
        
        # Force an error in routing decision
        with patch.object(engine.memory_store, 'find_similar_patterns', side_effect=Exception("Test error")):
            routing_decision = await engine.get_optimal_routing(
                query="test query",
                intent="search"
            )
            
            # Should fallback to standard strategy
            assert routing_decision["strategy"] == RoutingStrategy.FALLBACK_STANDARD
            assert "Fallback due to error" in str(routing_decision["reasoning"])
    
    @pytest.mark.asyncio
    async def test_learning_error_handling(self):
        """Test that learning errors are handled gracefully."""
        engine = StatefulRoutingEngine()
        
        # Force an error in learning
        with patch.object(engine.memory_store, 'store_pattern', side_effect=Exception("Learning error")):
            # Should not raise exception
            await engine.learn_from_execution(
                query="test",
                intent="search",
                agent_sequence=["agent1"],
                execution_result={"anime_results": []},
                session_id="test"
            )


# Performance and Memory Tests
class TestPerformanceAndMemory:
    """Test performance and memory management."""
    
    def test_memory_cleanup(self):
        """Test that memory cleanup works properly."""
        store = RoutingMemoryStore(max_patterns=5)  # Small limit for testing
        
        # Add more patterns than the limit
        for i in range(10):
            pattern = QueryPattern(
                pattern_id=f"pattern_{i}",
                query_text=f"query {i}",
                intent_type="search",
                success_rate=0.5
            )
            store.store_pattern(pattern)
        
        # Should not exceed max_patterns
        assert len(store.query_patterns) <= 5
    
    def test_session_memory_limits(self):
        """Test session memory respects limits."""
        memory = ConversationContextMemory(max_sessions=3)
        
        # Add more sessions than limit
        for i in range(5):
            memory.store_session_context(f"session_{i}", {"data": f"session {i}"})
        
        # Should be within limits due to cleanup
        assert len(memory.session_contexts) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])