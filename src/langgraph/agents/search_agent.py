"""
SearchAgent for intelligent anime discovery across multiple platforms.

Specialized agent that routes queries to optimal platform-specific MCP tools
based on query characteristics and platform strengths.
"""

import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph_swarm import create_handoff_tool

from langgraph.prebuilt import create_react_agent

# Removed get_recommended_tier import to avoid circular dependency
from ...config import get_settings
from ...integrations.clients.anilist_client import AniListClient
from ...integrations.clients.jikan_client import JikanClient
from ...integrations.clients.kitsu_client import KitsuClient
from ...integrations.clients.mal_client import MALClient
from ...vector.qdrant_client import QdrantClient
from ..workflow_state import AnimeSwarmState, QueryIntent

settings = get_settings()
logger = logging.getLogger(__name__)


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
            model="gpt-4o-mini", temperature=0.1, api_key=settings.openai_api_key
        )

        # Initialize clients
        self.jikan_client = JikanClient()
        self.anilist_client = AniListClient()
        self.mal_client = self._init_mal_client()
        self.kitsu_client = KitsuClient()
        self.qdrant_client = QdrantClient()

        # Create tiered tool groups - progressive complexity
        self.basic_tools = self._create_basic_tools()
        self.standard_tools = self._create_standard_tools()
        self.detailed_tools = self._create_detailed_tools()
        self.comprehensive_tools = self._create_comprehensive_tools()
        self.semantic_tools = self._create_semantic_tools()

        # Create agent with handoff capabilities
        self.agent = self._create_search_agent()

    def _init_mal_client(self):
        """Initialize MAL client safely."""
        try:
            if hasattr(settings, "mal_client_id") and settings.mal_client_id:
                return MALClient(
                    client_id=settings.mal_client_id,
                    client_secret=getattr(settings, "mal_client_secret", None),
                )
        except Exception as e:
            logger.warning(f"MAL client initialization failed: {e}")
        return None

    def _create_basic_tools(self):
        """Create basic tier LangChain tools."""

        @tool
        async def search_anime_basic(query: str, limit: int = 20) -> Dict[str, Any]:
            """Search for anime using basic search with essential information only."""
            try:
                results = await self.jikan_client.search_anime(q=query, limit=limit)
                return {"results": results[:limit], "tier": "basic"}
            except Exception as e:
                return {"error": str(e), "tier": "basic"}

        @tool
        async def get_seasonal_anime_basic(
            season: str, year: int, limit: int = 20
        ) -> Dict[str, Any]:
            """Get seasonal anime with basic information."""
            try:
                results = await self.jikan_client.get_seasonal_anime(season, year)
                return {"results": results.get("data", [])[:limit], "tier": "basic"}
            except Exception as e:
                return {"error": str(e), "tier": "basic"}

        return [search_anime_basic, get_seasonal_anime_basic]

    def _create_standard_tools(self):
        """Create standard tier LangChain tools."""

        @tool
        async def search_anime_standard(
            query: str, limit: int = 20, **kwargs
        ) -> Dict[str, Any]:
            """Search for anime using standard search with enhanced filtering."""
            try:
                results = await self.jikan_client.search_anime(
                    q=query, limit=limit, **kwargs
                )
                return {"results": results[:limit], "tier": "standard"}
            except Exception as e:
                return {"error": str(e), "tier": "standard"}

        return [search_anime_standard]

    def _create_detailed_tools(self):
        """Create detailed tier LangChain tools."""

        @tool
        async def search_anime_detailed(
            query: str, limit: int = 20, **kwargs
        ) -> Dict[str, Any]:
            """Search for anime using detailed search with comprehensive information."""
            try:
                results = await self.jikan_client.search_anime(
                    q=query, limit=limit, **kwargs
                )
                return {"results": results[:limit], "tier": "detailed"}
            except Exception as e:
                return {"error": str(e), "tier": "detailed"}

        return [search_anime_detailed]

    def _create_comprehensive_tools(self):
        """Create comprehensive tier LangChain tools."""

        @tool
        async def search_anime_comprehensive(
            query: str, limit: int = 20, **kwargs
        ) -> Dict[str, Any]:
            """Search for anime using comprehensive search with complete analytics."""
            try:
                results = await self.jikan_client.search_anime(
                    q=query, limit=limit, **kwargs
                )
                return {"results": results[:limit], "tier": "comprehensive"}
            except Exception as e:
                return {"error": str(e), "tier": "comprehensive"}

        return [search_anime_comprehensive]

    def _create_semantic_tools(self):
        """Create semantic search LangChain tools including image search capabilities."""
        
        # Store current image data for tools to access
        self.current_image_data = None

        @tool
        async def anime_semantic_search(query: str, limit: int = 20) -> Dict[str, Any]:
            """Perform semantic search across anime database using natural language."""
            try:
                results = await self.qdrant_client.search(
                    query=query, limit=limit, search_type="combined"
                )
                return {"results": results, "tier": "semantic"}
            except Exception as e:
                return {"error": str(e), "tier": "semantic"}

        @tool
        async def search_anime_by_image(query: str = "image search", limit: int = 20) -> Dict[str, Any]:
            """Search for anime using image similarity with CLIP embeddings. Uses image data from context."""
            try:
                if not self.current_image_data:
                    return {"error": "No image data available in context", "tier": "image_search"}
                    
                results = await self.qdrant_client.search_by_image(
                    image_data=self.current_image_data, limit=limit
                )
                return {"results": results, "tier": "image_search"}
            except Exception as e:
                return {"error": str(e), "tier": "image_search"}

        @tool
        async def search_multimodal_anime(
            query: str, text_weight: float = 0.7, limit: int = 20
        ) -> Dict[str, Any]:
            """Search for anime using both text query and image with adjustable weights. Uses image data from context."""
            try:
                if not self.current_image_data:
                    return {"error": "No image data available in context", "tier": "multimodal_search"}
                    
                results = await self.qdrant_client.search_multimodal(
                    query=query, image_data=self.current_image_data, text_weight=text_weight, limit=limit
                )
                return {"results": results, "tier": "multimodal_search"}
            except Exception as e:
                return {"error": str(e), "tier": "multimodal_search"}

        @tool
        async def find_visually_similar_anime(anime_id: str, limit: int = 20) -> Dict[str, Any]:
            """Find anime visually similar to a given anime ID using image embeddings."""
            try:
                results = await self.qdrant_client.find_visually_similar_anime(
                    anime_id=anime_id, limit=limit
                )
                return {"results": results, "tier": "visual_similarity"}
            except Exception as e:
                return {"error": str(e), "tier": "visual_similarity"}

        return [anime_semantic_search, search_anime_by_image, search_multimodal_anime, find_visually_similar_anime]

    def _create_search_agent(self) -> Any:
        """Create reactive search agent with platform-specific tools."""

        # Select tools based on configuration and availability
        available_tools = []

        # Always include semantic search (no API key required)
        available_tools.extend(self.semantic_tools)

        # Add tiered tools based on configuration
        # Basic tools are always available (fastest, essential info)
        available_tools.extend(self.basic_tools)

        # Standard tools for enhanced filtering
        available_tools.extend(self.standard_tools)

        # Detailed tools for comprehensive analysis
        available_tools.extend(self.detailed_tools)

        # Comprehensive tools for complete analytics
        available_tools.extend(self.comprehensive_tools)

        # Add handoff tools for agent coordination
        handoff_tools = [create_handoff_tool(agent_name="ScheduleAgent")]

        all_tools = available_tools + handoff_tools

        system_prompt = self._build_system_prompt()

        return create_react_agent(
            model=self.model, tools=all_tools, prompt=system_prompt, name="SearchAgent"
        )

    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt for search agent."""
        return """You are SearchAgent, an expert anime discovery specialist with access to tiered anime search tools.

## Your Mission
Help users discover anime through intelligent tier selection and comprehensive search capabilities.

## Tiered Tool Architecture & When to Use:

**Basic Tools (Tier 1) - 8 fields, 80% coverage**:
- search_anime_basic, get_anime_basic, find_similar_anime_basic, get_seasonal_anime_basic
- Use for: Quick searches, simple filtering, basic recommendations
- Performance: Fastest response times
- When: User needs quick answers or basic information

**Standard Tools (Tier 2) - 15 fields, 95% coverage**:
- search_anime_standard, get_anime_standard, find_similar_anime_standard, get_seasonal_anime_standard, search_by_genre_standard
- Use for: Advanced filtering, detailed search, genre-based discovery
- Performance: Fast with enhanced metadata
- When: User needs enhanced filtering or more detailed results

**Detailed Tools (Tier 3) - 25 fields, 99% coverage**:
- search_anime_detailed, get_anime_detailed, find_similar_anime_detailed, get_seasonal_anime_detailed, advanced_anime_analysis
- Use for: Cross-platform analysis, detailed comparison, research
- Performance: Moderate, comprehensive data
- When: User needs detailed analysis or cross-platform information

**Comprehensive Tools (Tier 4) - 40+ fields, 100% coverage**:
- search_anime_comprehensive, get_anime_comprehensive, find_similar_anime_comprehensive, comprehensive_anime_analytics
- Use for: Complete analysis, market research, predictive analytics
- Performance: Thorough, complete data set
- When: User needs exhaustive information or analytical insights

**Semantic & Image Search Tools**:
- **anime_semantic_search**: Natural language queries ("dark fantasy with strong protagonist")
- **search_anime_by_image**: Search using image similarity with CLIP embeddings
- **search_multimodal_anime**: Combined text + image search with adjustable weights
- **find_visually_similar_anime**: Find visually similar anime by anime ID
- Use when: Keywords don't capture essence, user provides images, or needs visual similarity

## Search Strategy:

1. **Analyze Query Complexity**: Determine if user needs basic info or detailed analysis
2. **Select Optimal Tier**: Choose appropriate tier based on query requirements
3. **Execute Smart Search**: Use tier strengths effectively
4. **Provide Rich Results**: Include relevant metadata and context
5. **Suggest Tier Upgrades**: Recommend higher tiers if user needs more info

## Tier Selection Guidelines:
- Simple queries or speed priority → Basic (Tier 1)
- Enhanced filtering needs → Standard (Tier 2) 
- Research or comparison → Detailed (Tier 3)
- Complete analysis → Comprehensive (Tier 4)

## Handoff Decisions:

- **ScheduleAgent**: When users need broadcast times, airing schedules, or streaming details
- **EnrichmentAgent**: When results need cross-platform data combination or quality enhancement

## Response Format:
- Lead with the most relevant results
- Explain your tier selection reasoning
- Include data quality indicators
- Suggest tier upgrades for more comprehensive results
- Offer handoffs when appropriate

Be helpful, precise, and always explain your reasoning for tier selection."""

    def get_tools_for_intent(self, intent: QueryIntent) -> List[BaseTool]:
        """Select optimal tools based on query intent analysis."""
        selected_tools = []

        # Semantic search for natural language and similarity queries
        if intent.needs_semantic_search or intent.needs_similarity_search:
            selected_tools.extend(self.semantic_tools)

        # Image search for image-based queries
        if intent.needs_image_search or intent.has_image_data:
            selected_tools.extend(self.semantic_tools)

        # Determine appropriate tier based on query complexity
        query_complexity = getattr(intent, "complexity", "medium")
        response_time_priority = getattr(intent, "response_time_priority", "balanced")

        # Get recommended tier (inline implementation to avoid circular imports)
        recommended_tier = self._get_recommended_tier(
            query_complexity, response_time_priority
        )

        # Add tools based on recommended tier
        if recommended_tier == "basic":
            selected_tools.extend(self.basic_tools)
        elif recommended_tier == "standard":
            selected_tools.extend(self.basic_tools + self.standard_tools)
        elif recommended_tier == "detailed":
            selected_tools.extend(
                self.basic_tools + self.standard_tools + self.detailed_tools
            )
        elif recommended_tier == "comprehensive":
            selected_tools.extend(
                self.basic_tools
                + self.standard_tools
                + self.detailed_tools
                + self.comprehensive_tools
            )

        # Always include semantic search for flexibility
        selected_tools.extend(self.semantic_tools)

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

        # Build search context message with image data if available
        user_context = state.get("user_context", {})
        image_data = user_context.get("image_data")
        
        # Set current image data for tools to access
        self.current_image_data = image_data
        
        search_content = f"""
Search for anime based on this query: "{query}"

Query Intent Analysis:
- Intent Type: {intent.get('intent_type', 'search')}
- Needs Semantic Search: {intent.get('needs_semantic_search', False)}
- Needs Image Search: {intent.get('needs_image_search', False)}
- Has Image Data: {intent.get('has_image_data', False)}
- Needs Streaming Info: {intent.get('needs_streaming_info', False)}
- Query Complexity: {intent.get('complexity', 'medium')}
- Response Time Priority: {intent.get('response_time_priority', 'balanced')}
- Suggested Tools: {intent.get('suggested_tools', [])}
"""

        if image_data:
            search_content += f"\n[SYSTEM: Image data is available with {len(image_data)} characters of base64 data. Use search_anime_by_image or search_multimodal_anime tools for image-based searches.]"

        search_content += "\nPlease select the most appropriate tier and search strategy based on the query complexity and available data.\nExplain your reasoning for tier selection and provide comprehensive results."

        search_message = HumanMessage(content=search_content)

        # Add search message to conversation
        updated_messages = state["messages"] + [search_message]

        # Execute search using the reactive agent
        result = await self.agent.ainvoke({"messages": updated_messages})

        # Extract search results from agent response
        search_results = self._extract_search_results(result)

        return {
            "messages": result["messages"],
            "anime_results": search_results,
            "current_step": "search_completed",
            "completed_steps": state.get("completed_steps", []) + ["search"],
            "tiers_used": self._extract_tiers_used(result),
        }

    def _extract_search_results(
        self, agent_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract structured anime results from agent response."""
        # This would parse the agent's tool call results
        # For now, return empty list as placeholder
        # In real implementation, we'd parse the actual tool results
        return []

    def _extract_tiers_used(self, agent_result: Dict[str, Any]) -> List[str]:
        """Extract which tiers were used in the search."""
        # Parse tool calls to determine which tiers were queried
        return ["basic", "semantic"]  # Placeholder

    def _get_recommended_tier(
        self, query_complexity: str = "medium", response_time_priority: str = "balanced"
    ) -> str:
        """
        Get recommended tier based on query complexity and response time priority.

        Args:
            query_complexity: "simple", "medium", "complex", "analytical"
            response_time_priority: "speed", "balanced", "completeness"

        Returns:
            Recommended tier: "basic", "standard", "detailed", or "comprehensive"
        """
        if query_complexity == "simple":
            if response_time_priority == "speed":
                return "basic"
            elif response_time_priority == "balanced":
                return "standard"
            else:
                return "detailed"

        elif query_complexity == "medium":
            if response_time_priority == "speed":
                return "standard"
            elif response_time_priority == "balanced":
                return "detailed"
            else:
                return "comprehensive"

        elif query_complexity == "complex":
            if response_time_priority == "speed":
                return "detailed"
            else:
                return "comprehensive"

        elif query_complexity == "analytical":
            return "comprehensive"

        # Default to standard for unknown inputs
        return "standard"

    def get_agent(self):
        """Get the configured search agent for swarm integration."""
        return self.agent
