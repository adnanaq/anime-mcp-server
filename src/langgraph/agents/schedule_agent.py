"""
ScheduleAgent for broadcast scheduling and streaming platform enrichment.

Specialized agent that provides real-time broadcast information, streaming availability,
and temporal anime data using AnimeSchedule and platform-specific tools.
"""

from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool

from ...anime_mcp.tools import (
    search_anime_schedule, get_schedule_data, get_currently_airing,
    search_streaming_platforms, get_anime_kitsu
)
from ..workflow_state import AnimeSwarmState, QueryIntent
from ...config import get_settings

settings = get_settings()


class ScheduleAgent:
    """
    Broadcast scheduling and streaming information specialist.
    
    Provides:
    - Real-time airing schedules and broadcast times
    - Streaming platform availability and regional info
    - Episode release calendars and countdowns
    - Cross-platform scheduling data enrichment
    """
    
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=settings.openai_api_key
        )
        
        # Schedule and streaming focused tools
        self.schedule_tools = [
            search_anime_schedule,
            get_schedule_data, 
            get_currently_airing
        ]
        
        self.streaming_tools = [
            search_streaming_platforms,
            get_anime_kitsu
        ]
        
        # Create agent with schedule specialization
        self.agent = self._create_schedule_agent()
    
    def _create_schedule_agent(self) -> Any:
        """Create reactive schedule agent with temporal and streaming tools."""
        
        # Combine schedule and streaming tools
        available_tools = self.schedule_tools + self.streaming_tools
        
        # Add handoff tools for agent coordination
        handoff_tools = [
            create_handoff_tool(
                agent_name="SearchAgent",
                description="Transfer to SearchAgent for anime discovery and platform search"
            ),
            create_handoff_tool(
                agent_name="EnrichmentAgent",
                description="Transfer to EnrichmentAgent for combining schedule data with other platforms"
            )
        ]
        
        all_tools = available_tools + handoff_tools
        
        system_prompt = self._build_system_prompt()
        
        return create_react_agent(
            model=self.model,
            tools=all_tools,
            state_modifier=system_prompt,
            name="ScheduleAgent"
        )
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for schedule agent."""
        return """You are ScheduleAgent, a specialist in anime broadcast schedules and streaming platform information.

## Your Expertise
- Real-time broadcast schedules and airing times
- Streaming platform availability and regional access
- Episode release calendars and countdown timers
- Cross-platform scheduling data enrichment

## Tools & When to Use:

**AnimeSchedule Tools**:
- `search_anime_schedule`: Comprehensive scheduling with 25+ streaming platforms
- `get_schedule_data`: Detailed schedule info by external IDs (MAL, AniList, etc.)
- `get_currently_airing`: Real-time airing anime with broadcast times

**Streaming Platform Tools**:
- `search_streaming_platforms`: Kitsu's specialized streaming platform search
- `get_anime_kitsu`: Detailed streaming data with regional availability

## Core Capabilities:

1. **Real-Time Scheduling**:
   - Current airing anime with exact broadcast times
   - Next episode information and countdown timers
   - Weekly broadcast calendars by day/timezone

2. **Streaming Intelligence**:
   - Platform availability across 25+ services
   - Regional streaming restrictions and availability
   - Subscription vs. free platform identification

3. **Cross-Platform Enrichment**:
   - Enhance search results with scheduling data
   - Connect anime from other platforms to broadcast info
   - Combine multiple streaming sources for complete picture

## Response Patterns:

**For "What's airing tonight?"**:
→ Use `get_currently_airing` with day/timezone filters

**For "Where can I watch [anime]?"**:
→ Use `search_streaming_platforms` or `get_anime_kitsu`

**For "When does [anime] air?"**:
→ Use `get_schedule_data` with anime IDs from other platforms

**For enriching search results**:
→ Take anime IDs and enhance with scheduling/streaming data

## Handoff Decisions:

- **SearchAgent**: When users need anime discovery or general search
- **EnrichmentAgent**: When schedule data needs combination with other platform data

Always provide:
- Precise broadcast times with timezone info
- Streaming platform availability with regional notes
- Next episode information when available
- Clear streaming URLs when possible

Be the definitive source for "when" and "where" anime questions."""

    async def enrich_with_schedule_data(
        self, 
        state: AnimeSwarmState,
        anime_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Enrich anime results with broadcast schedules and streaming information.
        
        Args:
            state: Current swarm state
            anime_results: Anime results to enrich with schedule data
            
        Returns:
            Updated state with enriched scheduling information
        """
        enrichment_message = HumanMessage(
            content=f"""
Enrich these {len(anime_results)} anime results with broadcast schedules and streaming information:

{self._format_anime_list_for_enrichment(anime_results)}

For each anime, please:
1. Find broadcast schedule and airing time information
2. Identify streaming platform availability 
3. Get next episode information if currently airing
4. Note regional streaming restrictions

Use the anime IDs (MAL, AniList, etc.) to look up scheduling data across platforms.
"""
        )
        
        # Add enrichment message to conversation
        updated_messages = state["messages"] + [enrichment_message]
        
        # Execute enrichment using the reactive agent
        result = await self.agent.ainvoke({
            "messages": updated_messages
        })
        
        # Extract enrichment data
        enrichment_data = self._extract_enrichment_data(result)
        
        return {
            "messages": result["messages"],
            "enrichment_data": enrichment_data,
            "current_step": "schedule_enrichment_completed",
            "completed_steps": state.get("completed_steps", []) + ["schedule_enrichment"]
        }
    
    async def get_currently_airing_anime(self, state: AnimeSwarmState) -> Dict[str, Any]:
        """
        Get currently airing anime with real-time broadcast information.
        
        Args:
            state: Current swarm state with query context
            
        Returns:
            Updated state with currently airing anime results
        """
        query_intent = state.get("query_intent", {})
        temporal_context = query_intent.get("temporal_context", {})
        
        # Build context-aware message
        airing_message = HumanMessage(
            content=f"""
Get currently airing anime with broadcast schedule information.

Context:
- User Query: "{state.get('user_query', '')}"
- Temporal Context: {temporal_context}

Please provide:
1. Currently airing anime with broadcast times
2. Next episode information and countdowns
3. Streaming platform availability
4. Regional broadcast information

Filter based on user preferences if specified (day of week, timezone, platforms).
"""
        )
        
        # Add to conversation
        updated_messages = state["messages"] + [airing_message]
        
        # Execute using reactive agent
        result = await self.agent.ainvoke({
            "messages": updated_messages
        })
        
        # Extract airing anime results
        airing_results = self._extract_airing_results(result)
        
        return {
            "messages": result["messages"],
            "anime_results": airing_results,
            "current_step": "airing_schedule_completed", 
            "completed_steps": state.get("completed_steps", []) + ["airing_schedule"]
        }
    
    async def find_streaming_platforms(
        self, 
        state: AnimeSwarmState,
        platform_query: str
    ) -> Dict[str, Any]:
        """
        Find anime available on specific streaming platforms.
        
        Args:
            state: Current swarm state
            platform_query: Streaming platform requirements
            
        Returns:
            Updated state with platform-specific anime results
        """
        streaming_message = HumanMessage(
            content=f"""
Find anime available on streaming platforms based on this request: "{platform_query}"

User's original query: "{state.get('user_query', '')}"

Please:
1. Identify specific streaming platforms mentioned
2. Search for anime available on those platforms
3. Include regional availability information
4. Note subscription requirements vs. free options
5. Provide streaming URLs when available

Use both AnimeSchedule and Kitsu data for comprehensive streaming coverage.
"""
        )
        
        # Add to conversation
        updated_messages = state["messages"] + [streaming_message]
        
        # Execute streaming search
        result = await self.agent.ainvoke({
            "messages": updated_messages
        })
        
        # Extract streaming results
        streaming_results = self._extract_streaming_results(result)
        
        return {
            "messages": result["messages"],
            "anime_results": streaming_results,
            "current_step": "streaming_search_completed",
            "completed_steps": state.get("completed_steps", []) + ["streaming_search"]
        }
    
    def _format_anime_list_for_enrichment(self, anime_results: List[Dict[str, Any]]) -> str:
        """Format anime list for enrichment context."""
        formatted = []
        for i, anime in enumerate(anime_results[:10], 1):  # Limit for context
            formatted.append(
                f"{i}. {anime.get('title', 'Unknown')} "
                f"(MAL ID: {anime.get('mal_id', 'N/A')}, "
                f"AniList ID: {anime.get('anilist_id', 'N/A')})"
            )
        
        return "\n".join(formatted)
    
    def _extract_enrichment_data(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract enrichment data from agent response."""
        # Parse agent tool call results for enrichment data
        return {
            "broadcast_schedules": [],
            "streaming_platforms": [],
            "next_episodes": []
        }
    
    def _extract_airing_results(self, agent_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract currently airing anime from agent response."""
        # Parse tool results for airing anime
        return []
    
    def _extract_streaming_results(self, agent_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract streaming platform results from agent response."""
        # Parse tool results for streaming availability
        return []
        
    def get_agent(self):
        """Get the configured schedule agent for swarm integration."""
        return self.agent