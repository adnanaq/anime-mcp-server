"""Super-Step Parallel Execution Engine for LangGraph.

This module implements Google Pregel-inspired super-step execution with
transactional rollback for parallel agent coordination and fault tolerance.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Union, Tuple
from dataclasses import dataclass
import json

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.types import Send
from langgraph.graph.state import CompiledStateGraph

from ..config import get_settings
from .langchain_tools import create_anime_langchain_tools
from .react_agent_workflow import LLMProvider
from .send_api_router import SendAPIParallelRouter, QueryComplexity, RouteStrategy
from .swarm_agents import MultiAgentSwarm, AgentSpecialization

logger = logging.getLogger(__name__)


class SuperStepPhase(Enum):
    """Super-step execution phases for Pregel-inspired coordination."""
    
    FAST_BATCH = "fast_batch"           # Super-step 1: Fast parallel agents (50-250ms)
    COMPREHENSIVE_BATCH = "comprehensive_batch"  # Super-step 2: All agents + enhancement (300-1000ms)
    ENHANCEMENT_BATCH = "enhancement_batch"      # Super-step 3: Pure enhancement agents
    FINALIZATION = "finalization"       # Super-step 4: Result merging and validation


class SuperStepStatus(Enum):
    """Status of super-step execution."""
    
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class SuperStepResult:
    """Result from a single super-step execution."""
    
    phase: SuperStepPhase
    status: SuperStepStatus
    results: List[Dict[str, Any]]
    execution_time: float
    agent_count: int
    success_count: int
    error_count: int
    errors: List[str]
    checkpoint_data: Optional[Dict[str, Any]] = None


@dataclass
class SuperStepConfig:
    """Configuration for super-step execution."""
    
    phase: SuperStepPhase
    timeout_ms: int
    max_retries: int
    agent_batch: List[str]
    parallel_limit: int
    rollback_enabled: bool = True
    checkpoint_enabled: bool = True


class SuperStepState(TypedDict):
    """State for super-step workflow coordination."""
    
    query: str
    complexity: QueryComplexity
    current_phase: SuperStepPhase
    super_step_results: List[SuperStepResult]
    cumulative_results: List[Dict[str, Any]]
    execution_history: List[Dict[str, Any]]
    checkpoint_stack: List[Dict[str, Any]]
    total_execution_time: float
    global_errors: List[str]
    rollback_triggered: bool
    final_result: Optional[Dict[str, Any]]


class SuperStepParallelExecutor:
    """Google Pregel-inspired super-step parallel execution engine.
    
    Implements transactional super-step execution with automatic rollback
    and checkpointing for fault-tolerant parallel agent coordination.
    """

    def __init__(
        self, 
        mcp_tools: Dict[str, Any], 
        llm_provider: LLMProvider = LLMProvider.OPENAI
    ):
        """Initialize the super-step parallel executor.

        Args:
            mcp_tools: Dictionary mapping tool names to their callable functions
            llm_provider: LLM provider to use (OpenAI or Anthropic)
        """
        self.mcp_tools = mcp_tools
        self.llm_provider = llm_provider
        self.settings = get_settings()
        
        # Initialize component engines
        self.send_api_router = SendAPIParallelRouter(mcp_tools, llm_provider)
        self.swarm_agents = MultiAgentSwarm(mcp_tools, llm_provider)
        
        # Super-step configuration
        self.super_step_configs = self._initialize_super_step_configs()
        
        # Execution state
        self.memory_saver = MemorySaver()
        self.execution_graph = self._build_execution_graph()

        logger.info(
            f"Initialized SuperStepParallelExecutor with {len(self.super_step_configs)} super-step phases"
        )

    def _initialize_super_step_configs(self) -> Dict[SuperStepPhase, SuperStepConfig]:
        """Initialize super-step execution configurations.
        
        Returns:
            Dict mapping super-step phases to their configurations
        """
        return {
            SuperStepPhase.FAST_BATCH: SuperStepConfig(
                phase=SuperStepPhase.FAST_BATCH,
                timeout_ms=250,
                max_retries=2,
                agent_batch=["MAL_Agent", "AniList_Agent", "Offline_Agent"],
                parallel_limit=3,
                rollback_enabled=True,
                checkpoint_enabled=True
            ),
            SuperStepPhase.COMPREHENSIVE_BATCH: SuperStepConfig(
                phase=SuperStepPhase.COMPREHENSIVE_BATCH,
                timeout_ms=1000,
                max_retries=3,
                agent_batch=["MAL_Agent", "AniList_Agent", "Jikan_Agent", "Offline_Agent", "Kitsu_Agent"],
                parallel_limit=5,
                rollback_enabled=True,
                checkpoint_enabled=True
            ),
            SuperStepPhase.ENHANCEMENT_BATCH: SuperStepConfig(
                phase=SuperStepPhase.ENHANCEMENT_BATCH,
                timeout_ms=800,
                max_retries=2,
                agent_batch=["RatingCorrelation_Agent", "StreamingAvailability_Agent", "ReviewAggregation_Agent"],
                parallel_limit=3,
                rollback_enabled=True,
                checkpoint_enabled=True
            ),
            SuperStepPhase.FINALIZATION: SuperStepConfig(
                phase=SuperStepPhase.FINALIZATION,
                timeout_ms=500,
                max_retries=1,
                agent_batch=["ResultMerger_Agent"],
                parallel_limit=1,
                rollback_enabled=False,
                checkpoint_enabled=True
            )
        }

    def _build_execution_graph(self) -> CompiledStateGraph:
        """Build the super-step execution graph.
        
        Returns:
            Compiled LangGraph state graph for super-step execution
        """
        workflow = StateGraph(SuperStepState)
        
        # Add super-step execution nodes
        workflow.add_node("analyze_complexity", self._analyze_complexity)
        workflow.add_node("execute_fast_batch", self._execute_fast_batch)
        workflow.add_node("execute_comprehensive_batch", self._execute_comprehensive_batch)
        workflow.add_node("execute_enhancement_batch", self._execute_enhancement_batch)
        workflow.add_node("finalize_results", self._finalize_results)
        workflow.add_node("handle_rollback", self._handle_rollback)
        
        # Add conditional edges for super-step progression
        workflow.add_conditional_edges(
            "analyze_complexity",
            self._route_complexity,
            {
                "fast_batch": "execute_fast_batch",
                "comprehensive_batch": "execute_comprehensive_batch"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_fast_batch",
            self._route_after_fast_batch,
            {
                "enhancement": "execute_enhancement_batch",
                "finalize": "finalize_results",
                "rollback": "handle_rollback"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_comprehensive_batch",
            self._route_after_comprehensive_batch,
            {
                "enhancement": "execute_enhancement_batch",
                "finalize": "finalize_results",
                "rollback": "handle_rollback"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_enhancement_batch",
            self._route_after_enhancement,
            {
                "finalize": "finalize_results",
                "rollback": "handle_rollback"
            }
        )
        
        workflow.add_edge("finalize_results", "__end__")
        workflow.add_edge("handle_rollback", "__end__")
        
        # Set entry point
        workflow.set_entry_point("analyze_complexity")
        
        return workflow.compile(checkpointer=self.memory_saver)

    async def _analyze_complexity(self, state: SuperStepState) -> SuperStepState:
        """Analyze query complexity for super-step routing.
        
        Args:
            state: Current super-step state
            
        Returns:
            Updated state with complexity analysis
        """
        start_time = time.time()
        
        try:
            # Use existing SendAPI router for complexity analysis
            complexity = await self.send_api_router._analyze_query_complexity(state["query"])
            
            state["complexity"] = complexity
            state["current_phase"] = SuperStepPhase.FAST_BATCH
            state["execution_history"].append({
                "phase": "complexity_analysis",
                "timestamp": time.time(),
                "complexity": complexity.value,
                "execution_time": time.time() - start_time
            })
            
            logger.info(f"Query complexity analyzed: {complexity.value}")
            
        except Exception as e:
            error_msg = f"Complexity analysis failed: {str(e)}"
            logger.error(error_msg)
            state["global_errors"].append(error_msg)
            # Default to SIMPLE for fallback
            state["complexity"] = QueryComplexity.SIMPLE
            
        return state

    async def _execute_fast_batch(self, state: SuperStepState) -> SuperStepState:
        """Execute fast batch super-step with parallel agents.
        
        Args:
            state: Current super-step state
            
        Returns:
            Updated state with fast batch results
        """
        return await self._execute_super_step(state, SuperStepPhase.FAST_BATCH)

    async def _execute_comprehensive_batch(self, state: SuperStepState) -> SuperStepState:
        """Execute comprehensive batch super-step with all platform agents.
        
        Args:
            state: Current super-step state
            
        Returns:
            Updated state with comprehensive batch results
        """
        return await self._execute_super_step(state, SuperStepPhase.COMPREHENSIVE_BATCH)

    async def _execute_enhancement_batch(self, state: SuperStepState) -> SuperStepState:
        """Execute enhancement batch super-step with enrichment agents.
        
        Args:
            state: Current super-step state
            
        Returns:
            Updated state with enhancement batch results
        """
        return await self._execute_super_step(state, SuperStepPhase.ENHANCEMENT_BATCH)

    async def _execute_super_step(self, state: SuperStepState, phase: SuperStepPhase) -> SuperStepState:
        """Execute a single super-step with transactional rollback.
        
        Args:
            state: Current super-step state
            phase: Super-step phase to execute
            
        Returns:
            Updated state with super-step results
        """
        start_time = time.time()
        config = self.super_step_configs[phase]
        
        logger.info(f"Executing super-step: {phase.value} with {len(config.agent_batch)} agents")
        
        # Create checkpoint before execution
        checkpoint_data = None
        if config.checkpoint_enabled:
            checkpoint_data = self._create_checkpoint(state)
            state["checkpoint_stack"].append(checkpoint_data)
        
        # Execute parallel agents with timeout
        results = []
        errors = []
        success_count = 0
        
        try:
            # Use asyncio.gather with timeout for parallel execution
            agent_tasks = []
            for agent_name in config.agent_batch:
                task = self._execute_agent_with_timeout(
                    agent_name, state["query"], config.timeout_ms
                )
                agent_tasks.append(task)
            
            # Execute all agents in parallel with overall timeout
            timeout_seconds = config.timeout_ms / 1000.0
            agent_results = await asyncio.wait_for(
                asyncio.gather(*agent_tasks, return_exceptions=True),
                timeout=timeout_seconds
            )
            
            # Process results and exceptions
            for i, result in enumerate(agent_results):
                agent_name = config.agent_batch[i]
                
                if isinstance(result, Exception):
                    error_msg = f"Agent {agent_name} failed: {str(result)}"
                    errors.append(error_msg)
                    logger.warning(error_msg)
                else:
                    # Check if the result indicates success or error
                    if result.get("status") == "error":
                        error_msg = f"Agent {agent_name} failed: {result.get('error', 'Unknown error')}"
                        errors.append(error_msg)
                        logger.warning(error_msg)
                    else:
                        results.append(result)
                        success_count += 1
                        logger.debug(f"Agent {agent_name} completed successfully")
                    
        except asyncio.TimeoutError:
            error_msg = f"Super-step {phase.value} timed out after {config.timeout_ms}ms"
            errors.append(error_msg)
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"Super-step {phase.value} execution failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        # Create super-step result
        execution_time = time.time() - start_time
        super_step_result = SuperStepResult(
            phase=phase,
            status=SuperStepStatus.COMPLETED if success_count > 0 else SuperStepStatus.FAILED,
            results=results,
            execution_time=execution_time,
            agent_count=len(config.agent_batch),
            success_count=success_count,
            error_count=len(errors),
            errors=errors,
            checkpoint_data=checkpoint_data
        )
        
        # Update state
        state["super_step_results"].append(super_step_result)
        state["cumulative_results"].extend(results)
        state["total_execution_time"] += execution_time
        state["global_errors"].extend(errors)
        
        # Trigger rollback if critical failure and rollback enabled
        if config.rollback_enabled and success_count == 0 and len(config.agent_batch) > 1:
            state["rollback_triggered"] = True
            logger.warning(f"Super-step {phase.value} failed completely, triggering rollback")
        
        logger.info(
            f"Super-step {phase.value} completed: {success_count}/{len(config.agent_batch)} agents succeeded"
        )
        
        return state

    async def _execute_agent_with_timeout(
        self, agent_name: str, query: str, timeout_ms: int
    ) -> Dict[str, Any]:
        """Execute a single agent with timeout protection.
        
        Args:
            agent_name: Name of the agent to execute
            query: Query to process
            timeout_ms: Timeout in milliseconds
            
        Returns:
            Agent execution result
        """
        try:
            # Use swarm agents for execution
            result = await self.swarm_agents._execute_single_agent(agent_name, query)
            return {
                "agent": agent_name,
                "status": "success",
                "data": result,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "agent": agent_name,
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }

    def _create_checkpoint(self, state: SuperStepState) -> Dict[str, Any]:
        """Create a checkpoint of the current state.
        
        Args:
            state: Current super-step state
            
        Returns:
            Checkpoint data for rollback
        """
        return {
            "timestamp": time.time(),
            "current_phase": state["current_phase"],
            "cumulative_results": state["cumulative_results"].copy(),
            "execution_history": state["execution_history"].copy(),
            "total_execution_time": state["total_execution_time"]
        }

    async def _finalize_results(self, state: SuperStepState) -> SuperStepState:
        """Finalize and merge all super-step results.
        
        Args:
            state: Current super-step state
            
        Returns:
            Updated state with final merged results
        """
        start_time = time.time()
        
        try:
            # Use result merger agent for intelligent merging
            merged_result = await self._execute_agent_with_timeout(
                "ResultMerger_Agent", 
                json.dumps(state["cumulative_results"]), 
                500
            )
            
            # Check if merging was successful
            if merged_result.get("status") == "error":
                error_msg = f"Result finalization failed: {merged_result.get('error', 'Unknown error')}"
                state["global_errors"].append(error_msg)
                logger.error(error_msg)
                
                # Fallback: use best available results
                state["final_result"] = {
                    "status": "partial_success",
                    "results": state["cumulative_results"],
                    "error": error_msg
                }
            else:
                state["final_result"] = merged_result
                logger.info(f"Results finalized with {len(state['cumulative_results'])} total results")
            
            state["total_execution_time"] += time.time() - start_time
            
        except Exception as e:
            error_msg = f"Result finalization failed: {str(e)}"
            state["global_errors"].append(error_msg)
            logger.error(error_msg)
            
            # Fallback: use best available results
            state["final_result"] = {
                "status": "partial_success",
                "results": state["cumulative_results"],
                "error": error_msg
            }
        
        return state

    async def _handle_rollback(self, state: SuperStepState) -> SuperStepState:
        """Handle transactional rollback to previous checkpoint.
        
        Args:
            state: Current super-step state
            
        Returns:
            Updated state with rollback applied
        """
        if not state["checkpoint_stack"]:
            logger.warning("Rollback requested but no checkpoints available")
            return state
        
        # Rollback to most recent checkpoint
        checkpoint = state["checkpoint_stack"][-1]
        
        logger.info(f"Rolling back to checkpoint at {checkpoint['timestamp']}")
        
        # Restore state from checkpoint
        state["current_phase"] = checkpoint["current_phase"]
        state["cumulative_results"] = checkpoint["cumulative_results"]
        state["execution_history"] = checkpoint["execution_history"]
        state["total_execution_time"] = checkpoint["total_execution_time"]
        
        # Mark rollback as completed
        state["rollback_triggered"] = False
        state["execution_history"].append({
            "phase": "rollback",
            "timestamp": time.time(),
            "checkpoint_timestamp": checkpoint["timestamp"]
        })
        
        # Provide fallback result
        state["final_result"] = {
            "status": "rollback_completed",
            "results": state["cumulative_results"],
            "message": "Execution rolled back to previous checkpoint"
        }
        
        return state

    def _route_complexity(self, state: SuperStepState) -> str:
        """Route based on query complexity.
        
        Args:
            state: Current super-step state
            
        Returns:
            Next node name
        """
        complexity = state["complexity"]
        
        if complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]:
            return "fast_batch"
        else:
            return "comprehensive_batch"

    def _route_after_fast_batch(self, state: SuperStepState) -> str:
        """Route after fast batch execution.
        
        Args:
            state: Current super-step state
            
        Returns:
            Next node name
        """
        if state["rollback_triggered"]:
            return "rollback"
        
        # Check if we have sufficient results
        if len(state["cumulative_results"]) >= 3:
            return "finalize"
        else:
            return "enhancement"

    def _route_after_comprehensive_batch(self, state: SuperStepState) -> str:
        """Route after comprehensive batch execution.
        
        Args:
            state: Current super-step state
            
        Returns:
            Next node name
        """
        if state["rollback_triggered"]:
            return "rollback"
        
        return "enhancement"

    def _route_after_enhancement(self, state: SuperStepState) -> str:
        """Route after enhancement batch execution.
        
        Args:
            state: Current super-step state
            
        Returns:
            Next node name
        """
        if state["rollback_triggered"]:
            return "rollback"
        
        return "finalize"

    async def execute_super_step_workflow(
        self, 
        query: str, 
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """Execute the complete super-step workflow.
        
        Args:
            query: Query to process
            session_id: Session ID for conversation memory
            
        Returns:
            Final workflow result with performance metrics
        """
        start_time = time.time()
        
        # Initialize state
        initial_state = SuperStepState(
            query=query,
            complexity=QueryComplexity.SIMPLE,  # Will be updated
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
        
        try:
            # Execute workflow
            config = RunnableConfig(configurable={"thread_id": session_id})
            final_state = await self.execution_graph.ainvoke(initial_state, config)
            
            # Calculate performance metrics
            total_time = time.time() - start_time
            
            return {
                "status": "success",
                "query": query,
                "final_result": final_state["final_result"],
                "performance_metrics": {
                    "total_execution_time": total_time,
                    "super_step_count": len(final_state["super_step_results"]),
                    "total_agents_executed": sum(r.agent_count for r in final_state["super_step_results"]),
                    "successful_agents": sum(r.success_count for r in final_state["super_step_results"]),
                    "error_count": len(final_state["global_errors"]),
                    "rollback_triggered": final_state["rollback_triggered"]
                },
                "execution_history": final_state["execution_history"],
                "super_step_results": [
                    {
                        "phase": r.phase.value,
                        "status": r.status.value,
                        "execution_time": r.execution_time,
                        "agent_count": r.agent_count,
                        "success_count": r.success_count,
                        "error_count": r.error_count
                    }
                    for r in final_state["super_step_results"]
                ]
            }
            
        except Exception as e:
            error_msg = f"Super-step workflow execution failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "status": "error",
                "query": query,
                "error": error_msg,
                "execution_time": time.time() - start_time
            }


# Convenience function for easy integration
async def execute_super_step_query(
    query: str, 
    mcp_tools: Dict[str, Any], 
    session_id: str = "default",
    llm_provider: LLMProvider = LLMProvider.OPENAI
) -> Dict[str, Any]:
    """Execute a query using super-step parallel execution.
    
    Args:
        query: Query to process
        mcp_tools: Dictionary mapping tool names to their callable functions
        session_id: Session ID for conversation memory
        llm_provider: LLM provider to use
        
    Returns:
        Super-step execution result with performance metrics
    """
    executor = SuperStepParallelExecutor(mcp_tools, llm_provider)
    return await executor.execute_super_step_workflow(query, session_id)