"""Tests for Super-Step Parallel Execution Engine.

Comprehensive test suite for Google Pregel-inspired super-step execution
with transactional rollback and fault tolerance validation.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.langgraph.parallel_supersteps import (
    SuperStepParallelExecutor,
    SuperStepPhase,
    SuperStepStatus,
    SuperStepResult,
    SuperStepConfig,
    SuperStepState,
    QueryComplexity,
    execute_super_step_query
)
from src.langgraph.react_agent_workflow import LLMProvider


@pytest.fixture
def mock_mcp_tools():
    """Create mock MCP tools for testing."""
    return {
        "search_anime": AsyncMock(return_value={"results": ["anime1", "anime2"]}),
        "get_anime_details": AsyncMock(return_value={"title": "Test Anime"}),
        "find_similar_anime": AsyncMock(return_value={"similar": ["anime3", "anime4"]}),
        "search_anime_by_image": AsyncMock(return_value={"image_results": ["anime5"]}),
    }


@pytest.fixture
def mock_super_step_executor(mock_mcp_tools):
    """Create a SuperStepParallelExecutor with mocked dependencies."""
    with patch('src.langgraph.parallel_supersteps.SendAPIParallelRouter'), \
         patch('src.langgraph.parallel_supersteps.MultiAgentSwarm'):
        
        executor = SuperStepParallelExecutor(mock_mcp_tools, LLMProvider.OPENAI)
        
        # Mock the component engines
        executor.send_api_router._analyze_query_complexity = AsyncMock(
            return_value=QueryComplexity.MODERATE
        )
        executor.swarm_agents._execute_single_agent = AsyncMock(
            return_value={"status": "success", "data": {"results": ["test_result"]}}
        )
        
        return executor


class TestSuperStepConfiguration:
    """Test super-step configuration and initialization."""

    def test_super_step_configs_initialization(self, mock_super_step_executor):
        """Test that all super-step configurations are properly initialized."""
        configs = mock_super_step_executor.super_step_configs
        
        # Verify all phases are configured
        expected_phases = [
            SuperStepPhase.FAST_BATCH,
            SuperStepPhase.COMPREHENSIVE_BATCH,
            SuperStepPhase.ENHANCEMENT_BATCH,
            SuperStepPhase.FINALIZATION
        ]
        
        assert len(configs) == len(expected_phases)
        for phase in expected_phases:
            assert phase in configs
            
        # Verify fast batch configuration
        fast_config = configs[SuperStepPhase.FAST_BATCH]
        assert fast_config.timeout_ms == 250
        assert fast_config.max_retries == 2
        assert len(fast_config.agent_batch) == 3
        assert "MAL_Agent" in fast_config.agent_batch
        assert fast_config.rollback_enabled is True
        
        # Verify comprehensive batch configuration
        comp_config = configs[SuperStepPhase.COMPREHENSIVE_BATCH]
        assert comp_config.timeout_ms == 1000
        assert len(comp_config.agent_batch) == 5
        assert comp_config.parallel_limit == 5
        
        # Verify enhancement batch configuration
        enhance_config = configs[SuperStepPhase.ENHANCEMENT_BATCH]
        assert "RatingCorrelation_Agent" in enhance_config.agent_batch
        assert enhance_config.rollback_enabled is True
        
        # Verify finalization configuration
        final_config = configs[SuperStepPhase.FINALIZATION]
        assert len(final_config.agent_batch) == 1
        assert "ResultMerger_Agent" in final_config.agent_batch
        assert final_config.rollback_enabled is False

    def test_execution_graph_creation(self, mock_super_step_executor):
        """Test that the execution graph is properly built."""
        graph = mock_super_step_executor.execution_graph
        
        # Verify graph exists and is compiled
        assert graph is not None
        assert hasattr(graph, 'ainvoke')  # CompiledStateGraph should have ainvoke


class TestComplexityAnalysis:
    """Test query complexity analysis and routing."""

    @pytest.mark.asyncio
    async def test_analyze_complexity_success(self, mock_super_step_executor):
        """Test successful complexity analysis."""
        state = SuperStepState(
            query="find mecha anime from 2020",
            complexity=QueryComplexity.SIMPLE,
            current_phase=SuperStepPhase.FAST_BATCH,
            super_step_results=[],
            cumulative_results=[],
            execution_history=[],
            checkpoint_stack=[],
            total_execution_time=0.0,
            global_errors=[],
            rollback_triggered=False,
            final_result=None
        )
        
        # Execute complexity analysis
        result_state = await mock_super_step_executor._analyze_complexity(state)
        
        # Verify complexity was updated
        assert result_state["complexity"] == QueryComplexity.MODERATE
        assert len(result_state["execution_history"]) == 1
        assert result_state["execution_history"][0]["phase"] == "complexity_analysis"
        assert result_state["execution_history"][0]["complexity"] == "moderate"

    @pytest.mark.asyncio
    async def test_analyze_complexity_failure(self, mock_super_step_executor):
        """Test complexity analysis with failure fallback."""
        # Mock complexity analysis to fail
        mock_super_step_executor.send_api_router._analyze_query_complexity.side_effect = Exception("Analysis failed")
        
        state = SuperStepState(
            query="test query",
            complexity=QueryComplexity.SIMPLE,
            current_phase=SuperStepPhase.FAST_BATCH,
            super_step_results=[],
            cumulative_results=[],
            execution_history=[],
            checkpoint_stack=[],
            total_execution_time=0.0,
            global_errors=[],
            rollback_triggered=False,
            final_result=None
        )
        
        result_state = await mock_super_step_executor._analyze_complexity(state)
        
        # Verify fallback to SIMPLE complexity
        assert result_state["complexity"] == QueryComplexity.SIMPLE
        assert len(result_state["global_errors"]) == 1
        assert "Analysis failed" in result_state["global_errors"][0]

    def test_route_complexity_simple(self, mock_super_step_executor):
        """Test complexity routing for simple queries."""
        state = SuperStepState(
            query="test",
            complexity=QueryComplexity.SIMPLE,
            current_phase=SuperStepPhase.FAST_BATCH,
            super_step_results=[],
            cumulative_results=[],
            execution_history=[],
            checkpoint_stack=[],
            total_execution_time=0.0,
            global_errors=[],
            rollback_triggered=False,
            final_result=None
        )
        
        route = mock_super_step_executor._route_complexity(state)
        assert route == "fast_batch"

    def test_route_complexity_complex(self, mock_super_step_executor):
        """Test complexity routing for complex queries."""
        state = SuperStepState(
            query="test",
            complexity=QueryComplexity.COMPLEX,
            current_phase=SuperStepPhase.FAST_BATCH,
            super_step_results=[],
            cumulative_results=[],
            execution_history=[],
            checkpoint_stack=[],
            total_execution_time=0.0,
            global_errors=[],
            rollback_triggered=False,
            final_result=None
        )
        
        route = mock_super_step_executor._route_complexity(state)
        assert route == "comprehensive_batch"


class TestSuperStepExecution:
    """Test individual super-step execution phases."""

    @pytest.mark.asyncio
    async def test_execute_fast_batch_success(self, mock_super_step_executor):
        """Test successful fast batch execution."""
        state = SuperStepState(
            query="find anime",
            complexity=QueryComplexity.MODERATE,
            current_phase=SuperStepPhase.FAST_BATCH,
            super_step_results=[],
            cumulative_results=[],
            execution_history=[],
            checkpoint_stack=[],
            total_execution_time=0.0,
            global_errors=[],
            rollback_triggered=False,
            final_result=None
        )
        
        result_state = await mock_super_step_executor._execute_fast_batch(state)
        
        # Verify super-step result was added
        assert len(result_state["super_step_results"]) == 1
        
        super_step_result = result_state["super_step_results"][0]
        assert super_step_result.phase == SuperStepPhase.FAST_BATCH
        assert super_step_result.agent_count == 3  # Fast batch has 3 agents
        assert super_step_result.success_count >= 0
        assert super_step_result.execution_time > 0
        
        # Verify cumulative results were updated
        assert len(result_state["cumulative_results"]) >= 0

    @pytest.mark.asyncio
    async def test_execute_agent_with_timeout_success(self, mock_super_step_executor):
        """Test successful agent execution with timeout."""
        result = await mock_super_step_executor._execute_agent_with_timeout(
            "MAL_Agent", "test query", 1000
        )
        
        assert result["agent"] == "MAL_Agent"
        assert result["status"] == "success"
        assert "data" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_execute_agent_with_timeout_failure(self, mock_super_step_executor):
        """Test agent execution failure handling."""
        # Mock agent execution to fail
        mock_super_step_executor.swarm_agents._execute_single_agent.side_effect = Exception("Agent failed")
        
        result = await mock_super_step_executor._execute_agent_with_timeout(
            "MAL_Agent", "test query", 1000
        )
        
        assert result["agent"] == "MAL_Agent"
        assert result["status"] == "error"
        assert "error" in result
        assert "Agent failed" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_super_step_with_partial_failure(self, mock_super_step_executor):
        """Test super-step execution with some agent failures."""
        # Mock partial failure by directly mocking _execute_agent_with_timeout
        async def mock_execute_agent_with_timeout(agent_name, query, timeout_ms):
            if agent_name == "MAL_Agent":
                return {
                    "agent": agent_name,
                    "status": "success",
                    "data": {"results": ["anime1"]},
                    "timestamp": 1234567890.0
                }
            else:
                return {
                    "agent": agent_name,
                    "status": "error",
                    "error": f"{agent_name} failed",
                    "timestamp": 1234567890.0
                }
        
        mock_super_step_executor._execute_agent_with_timeout = mock_execute_agent_with_timeout
        
        state = SuperStepState(
            query="test query",
            complexity=QueryComplexity.MODERATE,
            current_phase=SuperStepPhase.FAST_BATCH,
            super_step_results=[],
            cumulative_results=[],
            execution_history=[],
            checkpoint_stack=[],
            total_execution_time=0.0,
            global_errors=[],
            rollback_triggered=False,
            final_result=None
        )
        
        result_state = await mock_super_step_executor._execute_super_step(state, SuperStepPhase.FAST_BATCH)
        
        # Verify partial success handling
        super_step_result = result_state["super_step_results"][0]
        assert super_step_result.success_count == 1  # Only MAL_Agent succeeded
        assert super_step_result.error_count == 2    # Other agents failed
        assert super_step_result.status == SuperStepStatus.COMPLETED  # Still completed (partial success)
        assert not result_state["rollback_triggered"]  # Partial success shouldn't trigger rollback


class TestCheckpointingAndRollback:
    """Test checkpointing and transactional rollback functionality."""

    def test_create_checkpoint(self, mock_super_step_executor):
        """Test checkpoint creation."""
        state = SuperStepState(
            query="test",
            complexity=QueryComplexity.MODERATE,
            current_phase=SuperStepPhase.FAST_BATCH,
            super_step_results=[],
            cumulative_results=[{"result": "test1"}],
            execution_history=[{"step": "test"}],
            checkpoint_stack=[],
            total_execution_time=100.0,
            global_errors=[],
            rollback_triggered=False,
            final_result=None
        )
        
        checkpoint = mock_super_step_executor._create_checkpoint(state)
        
        assert "timestamp" in checkpoint
        assert checkpoint["current_phase"] == SuperStepPhase.FAST_BATCH
        assert len(checkpoint["cumulative_results"]) == 1
        assert len(checkpoint["execution_history"]) == 1
        assert checkpoint["total_execution_time"] == 100.0

    @pytest.mark.asyncio
    async def test_handle_rollback_success(self, mock_super_step_executor):
        """Test successful rollback to checkpoint."""
        # Create state with checkpoint
        checkpoint = {
            "timestamp": 1234567890.0,
            "current_phase": SuperStepPhase.FAST_BATCH,
            "cumulative_results": [{"original": "result"}],
            "execution_history": [{"original": "step"}],
            "total_execution_time": 50.0
        }
        
        state = SuperStepState(
            query="test",
            complexity=QueryComplexity.MODERATE,
            current_phase=SuperStepPhase.COMPREHENSIVE_BATCH,
            super_step_results=[],
            cumulative_results=[{"corrupted": "result"}],
            execution_history=[{"corrupted": "step"}],
            checkpoint_stack=[checkpoint],
            total_execution_time=200.0,
            global_errors=[],
            rollback_triggered=True,
            final_result=None
        )
        
        result_state = await mock_super_step_executor._handle_rollback(state)
        
        # Verify rollback restored state
        assert result_state["current_phase"] == SuperStepPhase.FAST_BATCH
        assert result_state["cumulative_results"] == [{"original": "result"}]
        assert result_state["total_execution_time"] == 50.0
        assert not result_state["rollback_triggered"]
        assert result_state["final_result"]["status"] == "rollback_completed"
        
        # Verify rollback was logged
        rollback_entry = next(
            (entry for entry in result_state["execution_history"] if entry.get("phase") == "rollback"),
            None
        )
        assert rollback_entry is not None

    @pytest.mark.asyncio
    async def test_handle_rollback_no_checkpoint(self, mock_super_step_executor):
        """Test rollback with no available checkpoints."""
        state = SuperStepState(
            query="test",
            complexity=QueryComplexity.MODERATE,
            current_phase=SuperStepPhase.FAST_BATCH,
            super_step_results=[],
            cumulative_results=[],
            execution_history=[],
            checkpoint_stack=[],  # No checkpoints
            total_execution_time=0.0,
            global_errors=[],
            rollback_triggered=True,
            final_result=None
        )
        
        result_state = await mock_super_step_executor._handle_rollback(state)
        
        # State should remain unchanged when no checkpoints available
        assert result_state["rollback_triggered"] is True
        assert result_state["final_result"] is None


class TestRoutingLogic:
    """Test super-step routing and progression logic."""

    def test_route_after_fast_batch_sufficient_results(self, mock_super_step_executor):
        """Test routing after fast batch with sufficient results."""
        state = SuperStepState(
            query="test",
            complexity=QueryComplexity.MODERATE,
            current_phase=SuperStepPhase.FAST_BATCH,
            super_step_results=[],
            cumulative_results=[{"r1": 1}, {"r2": 2}, {"r3": 3}],  # 3 results
            execution_history=[],
            checkpoint_stack=[],
            total_execution_time=0.0,
            global_errors=[],
            rollback_triggered=False,
            final_result=None
        )
        
        route = mock_super_step_executor._route_after_fast_batch(state)
        assert route == "finalize"

    def test_route_after_fast_batch_insufficient_results(self, mock_super_step_executor):
        """Test routing after fast batch with insufficient results."""
        state = SuperStepState(
            query="test",
            complexity=QueryComplexity.MODERATE,
            current_phase=SuperStepPhase.FAST_BATCH,
            super_step_results=[],
            cumulative_results=[{"r1": 1}],  # Only 1 result
            execution_history=[],
            checkpoint_stack=[],
            total_execution_time=0.0,
            global_errors=[],
            rollback_triggered=False,
            final_result=None
        )
        
        route = mock_super_step_executor._route_after_fast_batch(state)
        assert route == "enhancement"

    def test_route_after_fast_batch_rollback_triggered(self, mock_super_step_executor):
        """Test routing after fast batch with rollback triggered."""
        state = SuperStepState(
            query="test",
            complexity=QueryComplexity.MODERATE,
            current_phase=SuperStepPhase.FAST_BATCH,
            super_step_results=[],
            cumulative_results=[],
            execution_history=[],
            checkpoint_stack=[],
            total_execution_time=0.0,
            global_errors=[],
            rollback_triggered=True,
            final_result=None
        )
        
        route = mock_super_step_executor._route_after_fast_batch(state)
        assert route == "rollback"


class TestResultFinalization:
    """Test result finalization and merging."""

    @pytest.mark.asyncio
    async def test_finalize_results_success(self, mock_super_step_executor):
        """Test successful result finalization."""
        state = SuperStepState(
            query="test",
            complexity=QueryComplexity.MODERATE,
            current_phase=SuperStepPhase.FINALIZATION,
            super_step_results=[],
            cumulative_results=[{"r1": 1}, {"r2": 2}],
            execution_history=[],
            checkpoint_stack=[],
            total_execution_time=100.0,
            global_errors=[],
            rollback_triggered=False,
            final_result=None
        )
        
        result_state = await mock_super_step_executor._finalize_results(state)
        
        # Verify finalization completed
        assert result_state["final_result"] is not None
        assert "agent" in result_state["final_result"]
        assert result_state["total_execution_time"] > 100.0  # Should have increased

    @pytest.mark.asyncio
    async def test_finalize_results_failure(self, mock_super_step_executor):
        """Test result finalization with failure fallback."""
        # Mock finalization to fail by replacing _execute_agent_with_timeout
        async def mock_failing_agent(agent_name, query, timeout_ms):
            return {
                "agent": agent_name,
                "status": "error",
                "error": "Finalization failed",
                "timestamp": 1234567890.0
            }
        
        mock_super_step_executor._execute_agent_with_timeout = mock_failing_agent
        
        state = SuperStepState(
            query="test",
            complexity=QueryComplexity.MODERATE,
            current_phase=SuperStepPhase.FINALIZATION,
            super_step_results=[],
            cumulative_results=[{"r1": 1}],
            execution_history=[],
            checkpoint_stack=[],
            total_execution_time=100.0,
            global_errors=[],
            rollback_triggered=False,
            final_result=None
        )
        
        result_state = await mock_super_step_executor._finalize_results(state)
        
        # Verify fallback result
        assert result_state["final_result"] is not None
        assert result_state["final_result"]["status"] == "partial_success"
        assert result_state["final_result"]["results"] == [{"r1": 1}]
        assert len(result_state["global_errors"]) == 1


class TestFullWorkflowExecution:
    """Test complete super-step workflow execution."""

    @pytest.mark.asyncio
    async def test_execute_super_step_workflow_success(self, mock_super_step_executor):
        """Test successful complete workflow execution."""
        # Mock the graph execution
        mock_final_state = SuperStepState(
            query="find mecha anime",
            complexity=QueryComplexity.MODERATE,
            current_phase=SuperStepPhase.FINALIZATION,
            super_step_results=[
                SuperStepResult(
                    phase=SuperStepPhase.FAST_BATCH,
                    status=SuperStepStatus.COMPLETED,
                    results=[{"result": 1}],
                    execution_time=0.25,
                    agent_count=3,
                    success_count=3,
                    error_count=0,
                    errors=[]
                )
            ],
            cumulative_results=[{"result": 1}],
            execution_history=[{"step": "completed"}],
            checkpoint_stack=[],
            total_execution_time=0.5,
            global_errors=[],
            rollback_triggered=False,
            final_result={"status": "success", "data": "final_result"}
        )
        
        mock_super_step_executor.execution_graph.ainvoke = AsyncMock(return_value=mock_final_state)
        
        result = await mock_super_step_executor.execute_super_step_workflow("find mecha anime")
        
        # Verify successful execution result
        assert result["status"] == "success"
        assert result["query"] == "find mecha anime"
        assert result["final_result"]["status"] == "success"
        
        # Verify performance metrics
        metrics = result["performance_metrics"]
        assert metrics["super_step_count"] == 1
        assert metrics["total_agents_executed"] == 3
        assert metrics["successful_agents"] == 3
        assert metrics["error_count"] == 0
        assert not metrics["rollback_triggered"]
        
        # Verify execution history and super-step results
        assert len(result["execution_history"]) == 1
        assert len(result["super_step_results"]) == 1

    @pytest.mark.asyncio
    async def test_execute_super_step_workflow_failure(self, mock_super_step_executor):
        """Test workflow execution with failure handling."""
        # Mock graph execution to fail
        mock_super_step_executor.execution_graph.ainvoke = AsyncMock(
            side_effect=Exception("Workflow execution failed")
        )
        
        result = await mock_super_step_executor.execute_super_step_workflow("test query")
        
        # Verify error handling
        assert result["status"] == "error"
        assert result["query"] == "test query"
        assert "Workflow execution failed" in result["error"]
        assert "execution_time" in result


class TestConvenienceFunction:
    """Test the convenience function for easy integration."""

    @pytest.mark.asyncio
    async def test_execute_super_step_query_success(self, mock_mcp_tools):
        """Test the convenience function execution."""
        with patch('src.langgraph.parallel_supersteps.SuperStepParallelExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor.execute_super_step_workflow = AsyncMock(
                return_value={"status": "success", "result": "test"}
            )
            mock_executor_class.return_value = mock_executor
            
            result = await execute_super_step_query(
                "test query", 
                mock_mcp_tools, 
                "test_session",
                LLMProvider.OPENAI
            )
            
            # Verify convenience function worked
            assert result["status"] == "success"
            mock_executor_class.assert_called_once_with(mock_mcp_tools, LLMProvider.OPENAI)
            mock_executor.execute_super_step_workflow.assert_called_once_with("test query", "test_session")


class TestPerformanceMetrics:
    """Test performance metrics collection and reporting."""

    def test_super_step_result_creation(self):
        """Test SuperStepResult data structure creation."""
        result = SuperStepResult(
            phase=SuperStepPhase.FAST_BATCH,
            status=SuperStepStatus.COMPLETED,
            results=[{"test": "result"}],
            execution_time=0.123,
            agent_count=3,
            success_count=2,
            error_count=1,
            errors=["Agent failed"],
            checkpoint_data={"checkpoint": "data"}
        )
        
        assert result.phase == SuperStepPhase.FAST_BATCH
        assert result.status == SuperStepStatus.COMPLETED
        assert len(result.results) == 1
        assert result.execution_time == 0.123
        assert result.agent_count == 3
        assert result.success_count == 2
        assert result.error_count == 1
        assert len(result.errors) == 1

    def test_super_step_config_creation(self):
        """Test SuperStepConfig data structure creation."""
        config = SuperStepConfig(
            phase=SuperStepPhase.ENHANCEMENT_BATCH,
            timeout_ms=800,
            max_retries=2,
            agent_batch=["Agent1", "Agent2"],
            parallel_limit=2,
            rollback_enabled=True,
            checkpoint_enabled=True
        )
        
        assert config.phase == SuperStepPhase.ENHANCEMENT_BATCH
        assert config.timeout_ms == 800
        assert config.max_retries == 2
        assert len(config.agent_batch) == 2
        assert config.parallel_limit == 2
        assert config.rollback_enabled is True
        assert config.checkpoint_enabled is True


# Only apply asyncio mark to async tests individually