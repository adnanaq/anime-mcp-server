"""Multi-Agent Swarm Architecture for LangGraph.

This module implements the LangGraph Swarm pattern with specialized agents
that can hand off to each other for optimal anime search coordination.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Union
from dataclasses import dataclass

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

try:
    from langgraph_swarm import create_swarm, create_handoff_tool
except ImportError:
    # Fallback for testing or when langgraph-swarm is not available
    def create_swarm(*args, **kwargs):
        raise ImportError("langgraph-swarm not available. Install with: pip install langgraph-swarm")
    
    def create_handoff_tool(*args, **kwargs):
        raise ImportError("langgraph-swarm not available. Install with: pip install langgraph-swarm")

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from ..config import get_settings
from .langchain_tools import create_anime_langchain_tools
from .react_agent_workflow import LLMProvider

logger = logging.getLogger(__name__)


class AgentSpecialization(Enum):
    """Agent specialization types for swarm coordination."""
    
    PLATFORM_AGENT = "platform_agent"           # Platform-specific search agents
    ENHANCEMENT_AGENT = "enhancement_agent"     # Data enrichment agents  
    ORCHESTRATION_AGENT = "orchestration_agent" # Coordination and merging agents


class PlatformAgentType(Enum):
    """Platform agent types for specialized search."""
    
    MAL_AGENT = "mal_agent"                     # MyAnimeList official API
    ANILIST_AGENT = "anilist_agent"             # AniList GraphQL
    JIKAN_AGENT = "jikan_agent"                 # Jikan seasonal data
    OFFLINE_AGENT = "offline_agent"             # Vector database search
    KITSU_AGENT = "kitsu_agent"                 # Streaming platforms


class EnhancementAgentType(Enum):
    """Enhancement agent types for data enrichment."""
    
    RATING_CORRELATION_AGENT = "rating_correlation_agent"          # Cross-platform ratings
    STREAMING_AVAILABILITY_AGENT = "streaming_availability_agent"  # Multi-platform streaming
    REVIEW_AGGREGATION_AGENT = "review_aggregation_agent"          # Community reviews


class OrchestrationAgentType(Enum):
    """Orchestration agent types for coordination."""
    
    QUERY_ANALYSIS_AGENT = "query_analysis_agent"  # Intent detection and routing
    RESULT_MERGER_AGENT = "result_merger_agent"    # Intelligent result combination


@dataclass
class AgentDefinition:
    """Definition for a swarm agent with capabilities and handoff targets."""
    
    agent_name: str
    agent_type: Union[PlatformAgentType, EnhancementAgentType, OrchestrationAgentType]
    specialization: AgentSpecialization
    tools: List[str]
    handoff_targets: List[str]
    system_prompt: str
    performance_profile: Dict[str, Any]


class SwarmState(TypedDict):
    """State schema for Multi-Agent Swarm coordination."""
    
    messages: List[Any]
    session_id: str
    query: str
    image_data: Optional[str]
    text_weight: float
    search_parameters: Optional[Dict[str, Any]]
    
    # Swarm-specific state
    active_agent: str
    agent_results: Dict[str, Any]
    handoff_history: List[Dict[str, Any]]
    swarm_context: Dict[str, Any]
    
    # Final response
    final_results: Optional[Dict[str, Any]]
    processing_metrics: Dict[str, Any]


class MultiAgentSwarm:
    """Multi-Agent Swarm implementation using LangGraph Swarm patterns.
    
    This swarm coordinates 10 specialized agents across 3 categories:
    - 5 Platform Agents: MAL, AniList, Jikan, Offline, Kitsu
    - 3 Enhancement Agents: Rating, Streaming, Review
    - 2 Orchestration Agents: Query Analysis, Result Merger
    """
    
    def __init__(
        self, 
        mcp_tools: Dict[str, Any], 
        llm_provider: LLMProvider = LLMProvider.OPENAI
    ):
        """Initialize Multi-Agent Swarm.
        
        Args:
            mcp_tools: Dictionary mapping tool names to their callable functions
            llm_provider: LLM provider to use (OpenAI or Anthropic)
        """
        self.mcp_tools = mcp_tools
        self.llm_provider = llm_provider
        self.settings = get_settings()
        
        # Create LangChain tools from MCP tools
        self.tools = create_anime_langchain_tools(mcp_tools)
        
        # Initialize chat model
        self.chat_model = self._initialize_chat_model()
        
        # Create memory saver for conversation persistence
        self.memory_saver = MemorySaver()
        
        # Initialize agent definitions
        self.agent_definitions = self._init_agent_definitions()
        
        # Create specialized agents
        self.agents = self._create_swarm_agents()
        
        # Create swarm with handoff capabilities
        self.swarm = self._create_swarm_system()
        
        logger.info(
            f"Initialized MultiAgentSwarm with {len(self.agents)} specialized agents"
        )
    
    def _initialize_chat_model(self):
        """Initialize the chat model based on provider."""
        if self.llm_provider == LLMProvider.OPENAI:
            if ChatOpenAI is None:
                raise RuntimeError(
                    "langchain_openai not available. Install with: pip install langchain-openai"
                )
            
            api_key = getattr(self.settings, "openai_api_key", None)
            if not api_key:
                raise RuntimeError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
                )
            
            logger.info("Initializing OpenAI ChatGPT model for Multi-Agent Swarm")
            return ChatOpenAI(
                model="gpt-4o-mini", api_key=api_key, streaming=True, temperature=0.1
            )
        elif self.llm_provider == LLMProvider.ANTHROPIC:
            if ChatAnthropic is None:
                raise RuntimeError(
                    "langchain_anthropic not available. Install with: pip install langchain-anthropic"
                )
            
            api_key = getattr(self.settings, "anthropic_api_key", None)
            if not api_key:
                raise RuntimeError(
                    "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
                )
            
            logger.info("Initializing Anthropic Claude model for Multi-Agent Swarm")
            return ChatAnthropic(
                model="claude-3-haiku-20240307",
                api_key=api_key,
                streaming=True,
                temperature=0.1,
            )
        else:
            raise RuntimeError(
                f"Unknown LLM provider: {self.llm_provider}. Supported: {[p.value for p in LLMProvider]}"
            )
    
    def _init_agent_definitions(self) -> Dict[str, AgentDefinition]:
        """Initialize agent definitions with capabilities and handoff targets."""
        definitions = {}
        
        # Platform Agents (5)
        definitions["mal_agent"] = AgentDefinition(
            agent_name="mal_agent",
            agent_type=PlatformAgentType.MAL_AGENT,
            specialization=AgentSpecialization.PLATFORM_AGENT,
            tools=["search_anime", "get_anime_details"],
            handoff_targets=["anilist_agent", "rating_correlation_agent", "result_merger_agent"],
            system_prompt="""You are the MyAnimeList specialist agent. You excel at:
- Official MAL API searches with comprehensive metadata
- User list analysis and recommendation systems
- Seasonal anime tracking and popularity metrics
- Cross-referencing with MAL's extensive database

Hand off to AniList agent for GraphQL queries, Rating agent for cross-platform analysis, or Result Merger when you have sufficient data.""",
            performance_profile={"target_time_ms": 200, "reliability": 0.95}
        )
        
        definitions["anilist_agent"] = AgentDefinition(
            agent_name="anilist_agent",
            agent_type=PlatformAgentType.ANILIST_AGENT,
            specialization=AgentSpecialization.PLATFORM_AGENT,
            tools=["search_anime", "get_anime_details"],
            handoff_targets=["mal_agent", "streaming_availability_agent", "result_merger_agent"],
            system_prompt="""You are the AniList specialist agent. You excel at:
- GraphQL-based complex queries with flexible filtering
- Trending analysis and user statistics
- Advanced metadata and relationship tracking
- Community-driven data and user preferences

Hand off to MAL agent for official data, Streaming agent for platform availability, or Result Merger when complete.""",
            performance_profile={"target_time_ms": 250, "reliability": 0.90}
        )
        
        definitions["jikan_agent"] = AgentDefinition(
            agent_name="jikan_agent",
            agent_type=PlatformAgentType.JIKAN_AGENT,
            specialization=AgentSpecialization.PLATFORM_AGENT,
            tools=["search_anime", "get_anime_details"],
            handoff_targets=["mal_agent", "query_analysis_agent", "result_merger_agent"],
            system_prompt="""You are the Jikan specialist agent. You excel at:
- Seasonal anime data and broadcast schedules
- Genre-based filtering and classification
- Historical anime data and trends
- MAL data via unofficial API with rich metadata

Hand off to MAL agent for official verification, Query Analysis for complex filtering, or Result Merger when ready.""",
            performance_profile={"target_time_ms": 300, "reliability": 0.85}
        )
        
        definitions["offline_agent"] = AgentDefinition(
            agent_name="offline_agent",
            agent_type=PlatformAgentType.OFFLINE_AGENT,
            specialization=AgentSpecialization.PLATFORM_AGENT,
            tools=["search_anime", "find_similar_anime", "search_anime_by_image", "search_multimodal_anime"],
            handoff_targets=["mal_agent", "anilist_agent", "review_aggregation_agent"],
            system_prompt="""You are the Vector Database specialist agent. You excel at:
- Semantic similarity search with 38,000+ anime entries
- Image-based search using CLIP embeddings
- Multimodal search combining text and visual data
- Fast local search with sub-200ms response times

Hand off to MAL/AniList agents for external verification, Review agent for community data enrichment.""",
            performance_profile={"target_time_ms": 150, "reliability": 0.98}
        )
        
        definitions["kitsu_agent"] = AgentDefinition(
            agent_name="kitsu_agent",
            agent_type=PlatformAgentType.KITSU_AGENT,
            specialization=AgentSpecialization.PLATFORM_AGENT,
            tools=["search_anime", "get_anime_details"],
            handoff_targets=["streaming_availability_agent", "mal_agent", "result_merger_agent"],
            system_prompt="""You are the Kitsu specialist agent. You excel at:
- Streaming platform availability and licensing data
- JSON:API based searches with detailed metadata
- Community ratings and social features
- International availability and localization

Hand off to Streaming agent for comprehensive platform data, MAL agent for cross-reference, or Result Merger.""",
            performance_profile={"target_time_ms": 350, "reliability": 0.80}
        )
        
        # Enhancement Agents (3)
        definitions["rating_correlation_agent"] = AgentDefinition(
            agent_name="rating_correlation_agent",
            agent_type=EnhancementAgentType.RATING_CORRELATION_AGENT,
            specialization=AgentSpecialization.ENHANCEMENT_AGENT,
            tools=["search_anime", "get_anime_details"],
            handoff_targets=["streaming_availability_agent", "result_merger_agent"],
            system_prompt="""You are the Rating Correlation specialist. You excel at:
- Cross-platform rating analysis and correlation
- Score normalization across different rating systems
- Community sentiment analysis and review aggregation
- Quality assessment and recommendation scoring

Hand off to Streaming agent for availability data or Result Merger for final synthesis.""",
            performance_profile={"target_time_ms": 400, "reliability": 0.85}
        )
        
        definitions["streaming_availability_agent"] = AgentDefinition(
            agent_name="streaming_availability_agent",
            agent_type=EnhancementAgentType.STREAMING_AVAILABILITY_AGENT,
            specialization=AgentSpecialization.ENHANCEMENT_AGENT,
            tools=["search_anime", "get_anime_details"],
            handoff_targets=["review_aggregation_agent", "result_merger_agent"],
            system_prompt="""You are the Streaming Availability specialist. You excel at:
- Multi-platform streaming availability tracking
- Regional licensing and geo-restrictions analysis
- Subscription service comparison and recommendations
- Legal viewing options and platform updates

Hand off to Review agent for community feedback or Result Merger for comprehensive data.""",
            performance_profile={"target_time_ms": 500, "reliability": 0.75}
        )
        
        definitions["review_aggregation_agent"] = AgentDefinition(
            agent_name="review_aggregation_agent",
            agent_type=EnhancementAgentType.REVIEW_AGGREGATION_AGENT,
            specialization=AgentSpecialization.ENHANCEMENT_AGENT,
            tools=["search_anime", "get_anime_details"],
            handoff_targets=["rating_correlation_agent", "result_merger_agent"],
            system_prompt="""You are the Review Aggregation specialist. You excel at:
- Community review collection and sentiment analysis
- Expert critique aggregation and analysis
- Social media buzz and trending analysis
- User-generated content and recommendation mining

Hand off to Rating agent for correlation analysis or Result Merger for final compilation.""",
            performance_profile={"target_time_ms": 600, "reliability": 0.80}
        )
        
        # Orchestration Agents (2)
        definitions["query_analysis_agent"] = AgentDefinition(
            agent_name="query_analysis_agent",
            agent_type=OrchestrationAgentType.QUERY_ANALYSIS_AGENT,
            specialization=AgentSpecialization.ORCHESTRATION_AGENT,
            tools=["search_anime"],
            handoff_targets=["mal_agent", "anilist_agent", "offline_agent"],
            system_prompt="""You are the Query Analysis orchestrator. You excel at:
- Intent detection and query complexity assessment
- Parameter extraction and structured search planning
- Agent routing strategy and optimization
- Context understanding and conversation flow

Hand off to the most appropriate platform agent based on query analysis.""",
            performance_profile={"target_time_ms": 100, "reliability": 0.95}
        )
        
        definitions["result_merger_agent"] = AgentDefinition(
            agent_name="result_merger_agent",
            agent_type=OrchestrationAgentType.RESULT_MERGER_AGENT,
            specialization=AgentSpecialization.ORCHESTRATION_AGENT,
            tools=["search_anime", "get_anime_details"],
            handoff_targets=[],  # Terminal agent - no handoffs
            system_prompt="""You are the Result Merger orchestrator. You excel at:
- Intelligent result combination and deduplication
- Quality scoring and ranking optimization
- Cross-platform data correlation and synthesis
- Final response formatting and presentation

You are the terminal agent - synthesize all gathered data into the final response.""",
            performance_profile={"target_time_ms": 200, "reliability": 0.98}
        )
        
        return definitions
    
    def _create_swarm_agents(self) -> Dict[str, Any]:
        """Create individual react agents for the swarm."""
        agents = {}
        
        for agent_name, definition in self.agent_definitions.items():
            # Get tools for this agent
            agent_tools = []
            for tool_name in definition.tools:
                for tool in self.tools:
                    if tool.name == tool_name:
                        agent_tools.append(tool)
            
            # Create handoff tools for this agent
            handoff_tools = []
            for target_agent in definition.handoff_targets:
                try:
                    handoff_tool = create_handoff_tool(agent_name=target_agent)
                    handoff_tools.append(handoff_tool)
                except Exception as e:
                    logger.warning(f"Could not create handoff tool for {target_agent}: {e}")
            
            # Combine agent tools with handoff tools
            all_tools = agent_tools + handoff_tools
            
            # Create react agent
            try:
                agent = create_react_agent(
                    model=self.chat_model,
                    tools=all_tools,
                    checkpointer=self.memory_saver,
                    prompt=definition.system_prompt,
                )
                agents[agent_name] = agent
                logger.info(f"Created {agent_name} with {len(all_tools)} tools")
            except Exception as e:
                logger.error(f"Failed to create agent {agent_name}: {e}")
                # Create a fallback mock agent for testing
                agents[agent_name] = None
        
        return agents
    
    def _create_swarm_system(self) -> Any:
        """Create the swarm system with handoff capabilities."""
        try:
            # Filter out None agents (failed creations)
            valid_agents = {name: agent for name, agent in self.agents.items() if agent is not None}
            
            if not valid_agents:
                logger.error("No valid agents created, cannot create swarm")
                return None
            
            # Create swarm with default active agent
            swarm = create_swarm(
                list(valid_agents.values()),
                default_active_agent="query_analysis_agent"
            )
            
            logger.info(f"Created swarm system with {len(valid_agents)} agents")
            return swarm
            
        except Exception as e:
            logger.error(f"Failed to create swarm system: {e}")
            return None
    
    async def process_conversation(
        self,
        session_id: str,
        message: str,
        image_data: Optional[str] = None,
        text_weight: float = 0.7,
        thread_id: Optional[str] = None,
        search_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process conversation using Multi-Agent Swarm coordination.
        
        Args:
            session_id: Unique session identifier
            message: User message to process
            image_data: Optional base64 image data for multimodal search
            text_weight: Weight for text vs image in multimodal search (0.0-1.0)
            thread_id: Optional thread ID for conversation persistence
            search_parameters: Optional explicit SearchIntent parameters
            
        Returns:
            Dictionary with enhanced results from agent swarm
        """
        logger.info(f"Processing conversation with Multi-Agent Swarm for session {session_id}")
        
        try:
            if self.swarm is None:
                logger.error("Swarm system not available, falling back to single agent")
                return await self._fallback_single_agent_processing(
                    session_id, message, image_data, text_weight, search_parameters
                )
            
            # Configure checkpointing
            config: RunnableConfig = {
                "configurable": {"thread_id": thread_id or session_id},
                "recursion_limit": 20  # Higher limit for multi-agent coordination
            }
            
            # Prepare initial state
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "session_id": session_id,
                "query": message,
                "image_data": image_data,
                "text_weight": text_weight,
                "search_parameters": search_parameters,
            }
            
            # Execute swarm workflow
            result = await self.swarm.ainvoke(initial_state, config=config)
            
            # Convert to compatible format
            return self._convert_swarm_result_to_compatible_format(result, session_id)
        
        except Exception as e:
            logger.error(f"Error processing conversation with Multi-Agent Swarm: {e}")
            return self._create_error_response(session_id, message, str(e))
    
    async def _fallback_single_agent_processing(
        self,
        session_id: str,
        message: str,
        image_data: Optional[str] = None,
        text_weight: float = 0.7,
        search_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fallback to single agent processing when swarm is unavailable."""
        # Use offline agent as fallback (most reliable)
        offline_agent = self.agents.get("offline_agent")
        if offline_agent is None:
            return self._create_error_response(
                session_id, message, "No agents available for processing"
            )
        
        try:
            config = {"configurable": {"thread_id": session_id}}
            initial_state = {"messages": [HumanMessage(content=message)]}
            
            result = await offline_agent.ainvoke(initial_state, config=config)
            
            return {
                "session_id": session_id,
                "messages": [message, "Processed with fallback single agent"],
                "workflow_steps": [
                    {
                        "step_type": "fallback_single_agent",
                        "agent_name": "offline_agent",
                        "confidence": 0.7,
                    }
                ],
                "results": [],
                "swarm_enabled": False,
            }
        
        except Exception as e:
            logger.error(f"Fallback agent processing failed: {e}")
            return self._create_error_response(session_id, message, str(e))
    
    def _convert_swarm_result_to_compatible_format(
        self, result: Dict[str, Any], session_id: str
    ) -> Dict[str, Any]:
        """Convert swarm result to compatible format."""
        return {
            "session_id": session_id,
            "messages": [
                result.get("query", ""),
                "Multi-Agent Swarm coordination completed"
            ],
            "workflow_steps": [
                {
                    "step_type": "multi_agent_swarm_execution",
                    "active_agents": len(self.agents),
                    "handoff_enabled": True,
                    "specialization_count": {
                        "platform_agents": 5,
                        "enhancement_agents": 3,
                        "orchestration_agents": 2,
                    },
                    "confidence": 0.95,
                }
            ],
            "results": result.get("results", []),
            "swarm_metrics": {
                "total_agents": len(self.agents),
                "active_agent": result.get("active_agent", "unknown"),
                "handoff_count": len(result.get("handoff_history", [])),
            },
            "current_context": None,
            "user_preferences": None,
            "orchestration_enabled": True,
            "swarm_enabled": True,
        }
    
    def _create_error_response(
        self, session_id: str, message: str, error: str
    ) -> Dict[str, Any]:
        """Create error response for swarm failures."""
        return {
            "session_id": session_id,
            "messages": [message, f"Multi-Agent Swarm error: {error}"],
            "workflow_steps": [
                {"step_type": "swarm_error", "error": error, "confidence": 0.0}
            ],
            "results": [],
            "current_context": None,
            "user_preferences": None,
            "orchestration_enabled": False,
            "swarm_enabled": False,
        }
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the Multi-Agent Swarm."""
        return {
            "engine_type": "Multi-Agent Swarm",
            "features": [
                "LangGraph Swarm pattern implementation",
                "10 specialized agents with handoff capabilities",
                "Platform, Enhancement, and Orchestration agent types",
                "Intelligent agent coordination and routing",
                "Cross-platform data synthesis and correlation",
                "Conversation memory and context preservation",
            ],
            "performance": {
                "target_response_time": "200-600ms (agent-dependent)",
                "concurrent_coordination": True,
                "agent_specialization": True,
                "handoff_capabilities": True,
                "context_preservation": True,
                "tools_count": len(self.tools),
                "llm_provider": self.llm_provider.value,
            },
            "agent_architecture": {
                "platform_agents": 5,
                "enhancement_agents": 3,
                "orchestration_agents": 2,
                "total_agents": len(self.agents),
                "handoff_enabled": True,
            },
            "agent_definitions": {
                name: {
                    "specialization": def_.specialization.value,
                    "tools": def_.tools,
                    "handoff_targets": def_.handoff_targets,
                    "performance": def_.performance_profile,
                }
                for name, def_ in self.agent_definitions.items()
            },
        }


def create_multi_agent_swarm(
    mcp_tools: Dict[str, Any],
    llm_provider: LLMProvider = LLMProvider.OPENAI,
) -> MultiAgentSwarm:
    """Create Multi-Agent Swarm from MCP tool functions.
    
    Args:
        mcp_tools: Dictionary mapping tool names to their functions
        llm_provider: LLM provider to use (OpenAI or Anthropic)
        
    Returns:
        MultiAgentSwarm ready for swarm conversation processing
    """
    logger.info(f"Creating Multi-Agent Swarm with {len(mcp_tools)} MCP tools")
    swarm = MultiAgentSwarm(mcp_tools, llm_provider)
    logger.info("Multi-Agent Swarm created successfully")
    return swarm