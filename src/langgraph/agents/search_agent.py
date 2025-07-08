"""
SearchAgent for intelligent anime discovery across multiple platforms.

Specialized agent that routes queries to optimal platform-specific MCP tools
based on query characteristics and platform strengths.
"""

from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool

from ...anime_mcp.tools import (
    search_anime_mal, get_anime_by_id_mal, get_seasonal_anime_mal,
    search_anime_anilist, get_anime_anilist,
    search_anime_jikan, get_anime_jikan, get_jikan_seasonal,
    search_anime_kitsu, get_anime_kitsu,
    anime_semantic_search, anime_similar
)
from ..workflow_state import AnimeSwarmState, QueryIntent
from ...config import get_settings

settings = get_settings()


class SearchAgent:
    """
    Intelligent anime search agent with platform-specific tool routing.
    
    Routes queries to optimal platform tools based on query intent and platform strengths:
    - MAL: Community data, ratings, official content
    - AniList: International content, advanced filtering (70+ params)
    - Jikan: MAL data without API keys, comprehensive metadata
    - Kitsu: Streaming platform specialization, JSON:API
    - Semantic: AI-powered similarity and natural language search
    """
    
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=settings.openai_api_key
        )
        
        # Platform-specific tool groups
        self.mal_tools = [search_anime_mal, get_anime_by_id_mal, get_seasonal_anime_mal]
        self.anilist_tools = [search_anime_anilist, get_anime_anilist]
        self.jikan_tools = [search_anime_jikan, get_anime_jikan, get_jikan_seasonal]
        self.kitsu_tools = [search_anime_kitsu, get_anime_kitsu]
        self.semantic_tools = [anime_semantic_search, anime_similar]
        
        # Create agent with handoff capabilities
        self.agent = self._create_search_agent()
    
    def _create_search_agent(self) -> Any:
        """Create reactive search agent with platform-specific tools."""
        
        # Select tools based on configuration and availability
        available_tools = []
        
        # Always include semantic search (no API key required)
        available_tools.extend(self.semantic_tools)
        
        # Add Jikan tools (no API key required)
        available_tools.extend(self.jikan_tools)
        
        # Add platform tools based on API key availability
        if hasattr(settings, 'mal_api_key') and settings.mal_api_key:
            available_tools.extend(self.mal_tools)
        
        if hasattr(settings, 'anilist_token') and settings.anilist_token:
            available_tools.extend(self.anilist_tools)
            
        # Kitsu doesn't require API key
        available_tools.extend(self.kitsu_tools)
        
        # Add handoff tools for agent coordination
        handoff_tools = [
            create_handoff_tool(
                agent_name="ScheduleAgent",
                description="Transfer to ScheduleAgent for broadcast schedules and streaming information"
            ),
            create_handoff_tool(
                agent_name="EnrichmentAgent", 
                description="Transfer to EnrichmentAgent for cross-platform data combination"
            )
        ]
        
        all_tools = available_tools + handoff_tools
        
        system_prompt = self._build_system_prompt()
        
        return create_react_agent(
            model=self.model,
            tools=all_tools,
            state_modifier=system_prompt,
            name="SearchAgent"
        )
    
    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt for search agent."""
        return """You are SearchAgent, an expert anime discovery specialist with access to multiple platform APIs.

## Your Mission
Help users discover anime through intelligent platform routing and comprehensive search capabilities.

## Platform Strengths & When to Use:

**Semantic Search (anime_semantic_search, anime_similar)**:
- Natural language queries: "dark fantasy with strong protagonist"
- Finding similar anime: "anime like Attack on Titan"
- Thematic discovery: "psychological horror anime"
- When keywords don't capture the essence

**Jikan (MAL Unofficial API)**:
- No API key required - always available
- Rich community data (scores, popularity, members)
- Seasonal anime discovery
- Detailed metadata (characters, staff, episodes)
- When you need reliable MAL data

**MAL Official API** (if available):
- Official community statistics
- Content filtering (NSFW, ratings)
- Broadcast scheduling
- When you need authoritative MAL data

**AniList**:
- 70+ advanced search parameters
- International content focus
- Complex filtering needs
- GraphQL-powered precision

**Kitsu**:
- Streaming platform specialization
- Range syntax filtering (80.., ..90, 80..90)
- JSON:API standards
- When streaming availability is key

## Search Strategy:

1. **Analyze Query Intent**: Understand what the user really wants
2. **Select Optimal Platform(s)**: Choose based on query characteristics
3. **Execute Smart Search**: Use platform strengths effectively
4. **Provide Rich Results**: Include relevant metadata and context

## Handoff Decisions:

- **ScheduleAgent**: When users need broadcast times, airing schedules, or streaming details
- **EnrichmentAgent**: When results need cross-platform data combination or quality enhancement

## Response Format:
- Lead with the most relevant results
- Explain your platform choice reasoning
- Include data quality indicators
- Suggest related searches or refinements
- Offer handoffs when appropriate

Be helpful, precise, and always explain your reasoning for platform selection."""

    def get_tools_for_intent(self, intent: QueryIntent) -> List[BaseTool]:
        """Select optimal tools based on query intent analysis."""
        selected_tools = []
        
        # Semantic search for natural language and similarity queries
        if intent.needs_semantic_search or intent.needs_similarity_search:
            selected_tools.extend(self.semantic_tools)
        
        # Seasonal data requirements
        if intent.needs_seasonal_data:
            selected_tools.extend([get_jikan_seasonal, get_seasonal_anime_mal])
        
        # Streaming platform focus
        if intent.needs_streaming_info:
            selected_tools.extend(self.kitsu_tools)
        
        # Platform-specific routing based on hints
        platform_hints = intent.platform_hints
        if "myanimelist" in platform_hints or "mal" in platform_hints:
            selected_tools.extend(self.mal_tools + self.jikan_tools)
        elif "anilist" in platform_hints:
            selected_tools.extend(self.anilist_tools)
        elif "kitsu" in platform_hints:
            selected_tools.extend(self.kitsu_tools)
        else:
            # Default comprehensive search
            selected_tools.extend(self.jikan_tools)  # Always available
            selected_tools.extend(self.semantic_tools)  # Always available
            
            # Add available platform tools
            if hasattr(settings, 'mal_api_key') and settings.mal_api_key:
                selected_tools.extend(self.mal_tools)
            if hasattr(settings, 'anilist_token') and settings.anilist_token:
                selected_tools.extend(self.anilist_tools)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tools = []
        for tool in selected_tools:
            if tool.name not in seen:
                seen.add(tool.name)
                unique_tools.append(tool)
        
        return unique_tools
    
    async def search(self, state: AnimeSwarmState) -> Dict[str, Any]:
        """
        Execute intelligent anime search based on query intent.
        
        Args:
            state: Current swarm state with query and intent
            
        Returns:
            Updated state with search results and metadata
        """
        query = state.get("user_query", "")
        intent = state.get("query_intent", {})
        
        # Build search context message
        search_message = HumanMessage(
            content=f"""
Search for anime based on this query: "{query}"

Query Intent Analysis:
- Intent Type: {intent.get('intent_type', 'search')}
- Needs Semantic Search: {intent.get('needs_semantic_search', False)}
- Needs Streaming Info: {intent.get('needs_streaming_info', False)}
- Platform Hints: {intent.get('platform_hints', [])}
- Suggested Tools: {intent.get('suggested_tools', [])}

Please select the most appropriate search strategy and execute the search.
Explain your reasoning for platform selection and provide comprehensive results.
"""
        )
        
        # Add search message to conversation
        updated_messages = state["messages"] + [search_message]
        
        # Execute search using the reactive agent
        result = await self.agent.ainvoke({
            "messages": updated_messages
        })
        
        # Extract search results from agent response
        search_results = self._extract_search_results(result)
        
        return {
            "messages": result["messages"],
            "anime_results": search_results,
            "current_step": "search_completed",
            "completed_steps": state.get("completed_steps", []) + ["search"],
            "platforms_used": self._extract_platforms_used(result)
        }
    
    def _extract_search_results(self, agent_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract structured anime results from agent response."""
        # This would parse the agent's tool call results
        # For now, return empty list as placeholder
        # In real implementation, we'd parse the actual tool results
        return []
    
    def _extract_platforms_used(self, agent_result: Dict[str, Any]) -> List[str]:
        """Extract which platforms were used in the search."""
        # Parse tool calls to determine which platforms were queried
        return ["jikan", "semantic"]  # Placeholder
        
    def get_agent(self):
        """Get the configured search agent for swarm integration."""
        return self.agent