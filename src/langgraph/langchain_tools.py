"""LangGraph ToolNode integration for anime conversations.

LangGraph's native ToolNode implementation, providing better integration.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, TypedDict, cast

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Import tiered response models
from ..models.structured_responses import (
    BasicAnimeResult,
    StandardAnimeResult,
    DetailedAnimeResult,
    ComprehensiveAnimeResult,
    BasicSearchResponse,
    StandardSearchResponse,
    DetailedSearchResponse,
    ComprehensiveSearchResponse,
    AnimeType,
    AnimeStatus,
    AnimeRating
)

logger = logging.getLogger(__name__)


# Tiered Input Schemas

class BasicSearchInput(BaseModel):
    """Input schema for basic anime search (Tier 1)."""
    query: str = Field(description="Search query for anime titles")
    limit: int = Field(default=20, ge=1, le=50, description="Maximum number of results")
    year: Optional[int] = Field(None, ge=1900, le=2030, description="Release year filter")
    type: Optional[AnimeType] = Field(None, description="Anime type filter")
    status: Optional[AnimeStatus] = Field(None, description="Anime status filter")


class StandardSearchInput(BaseModel):
    """Input schema for standard anime search (Tier 2)."""
    query: str = Field(description="Search query for anime titles")
    limit: int = Field(default=20, ge=1, le=50, description="Maximum number of results")
    year: Optional[int] = Field(None, ge=1900, le=2030, description="Release year filter")
    type: Optional[AnimeType] = Field(None, description="Anime type filter")
    status: Optional[AnimeStatus] = Field(None, description="Anime status filter")
    genres: Optional[List[str]] = Field(None, description="List of anime genres to filter by")
    studios: Optional[List[str]] = Field(None, description="List of animation studios")
    rating: Optional[AnimeRating] = Field(None, description="Content rating filter")
    min_score: Optional[float] = Field(None, ge=0, le=10, description="Minimum score filter")
    max_score: Optional[float] = Field(None, ge=0, le=10, description="Maximum score filter")


class DetailedSearchInput(BaseModel):
    """Input schema for detailed anime search (Tier 3)."""
    query: str = Field(description="Search query for anime titles")
    limit: int = Field(default=20, ge=1, le=50, description="Maximum number of results")
    year: Optional[int] = Field(None, ge=1900, le=2030, description="Release year filter")
    type: Optional[AnimeType] = Field(None, description="Anime type filter")
    status: Optional[AnimeStatus] = Field(None, description="Anime status filter")
    genres: Optional[List[str]] = Field(None, description="List of anime genres to filter by")
    studios: Optional[List[str]] = Field(None, description="List of animation studios")
    rating: Optional[AnimeRating] = Field(None, description="Content rating filter")
    min_score: Optional[float] = Field(None, ge=0, le=10, description="Minimum score filter")
    max_score: Optional[float] = Field(None, ge=0, le=10, description="Maximum score filter")
    year_range: Optional[List[int]] = Field(None, description="Year range as [start_year, end_year]")
    exclusions: Optional[List[str]] = Field(None, description="List of genres/themes to exclude")
    mood_keywords: Optional[List[str]] = Field(None, description="List of mood descriptors")
    cross_platform: bool = Field(default=True, description="Include cross-platform data")


class ComprehensiveSearchInput(BaseModel):
    """Input schema for comprehensive anime search (Tier 4)."""
    query: str = Field(description="Search query for anime titles")
    limit: int = Field(default=20, ge=1, le=50, description="Maximum number of results")
    year: Optional[int] = Field(None, ge=1900, le=2030, description="Release year filter")
    type: Optional[AnimeType] = Field(None, description="Anime type filter")
    status: Optional[AnimeStatus] = Field(None, description="Anime status filter")
    genres: Optional[List[str]] = Field(None, description="List of anime genres to filter by")
    studios: Optional[List[str]] = Field(None, description="List of animation studios")
    rating: Optional[AnimeRating] = Field(None, description="Content rating filter")
    min_score: Optional[float] = Field(None, ge=0, le=10, description="Minimum score filter")
    max_score: Optional[float] = Field(None, ge=0, le=10, description="Maximum score filter")
    year_range: Optional[List[int]] = Field(None, description="Year range as [start_year, end_year]")
    exclusions: Optional[List[str]] = Field(None, description="List of genres/themes to exclude")
    mood_keywords: Optional[List[str]] = Field(None, description="List of mood descriptors")
    cross_platform: bool = Field(default=True, description="Include cross-platform data")
    include_analytics: bool = Field(default=True, description="Include analytics and predictions")
    include_market_data: bool = Field(default=True, description="Include market analysis")




class AnimeDetailsInput(BaseModel):
    """Input schema for anime details tools."""
    anime_id: str = Field(description="Unique anime identifier")


class SimilarAnimeInput(BaseModel):
    """Input schema for similarity search tools."""
    anime_id: str = Field(description="Reference anime ID")
    limit: int = Field(default=10, description="Maximum number of results (1-20)")


class LangChainToolAdapter:
    """Adapter to wrap MCP functions as properties for LangChain tool creation."""

    def __init__(self, name: str, func: Callable[..., Any], description: str):
        self.name = name
        self.func = func
        self.description = description

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the underlying MCP function."""
        return await self.func(**kwargs)


def create_anime_langchain_tools(mcp_tools: Dict[str, Callable[..., Any]]) -> List[Any]:
    """Create LangChain tools from MCP tool functions.

    This function converts MCP tool functions into LangChain-compatible tools
    using the @tool decorator, supporting both legacy and tiered tools.

    Args:
        mcp_tools: Dictionary mapping tool names to their callable functions

    Returns:
        List of LangChain tools ready for use with ToolNode
    """
    tools = []
    
    # Create tiered search tools
    _create_tiered_search_tools(mcp_tools, tools)
    
    # Create tiered details tools
    _create_tiered_details_tools(mcp_tools, tools)
    
    # Create tiered similarity tools
    _create_tiered_similarity_tools(mcp_tools, tools)
    
    # Create tiered seasonal tools
    _create_tiered_seasonal_tools(mcp_tools, tools)
    
    
    logger.info(f"Created {len(tools)} LangChain tools from MCP functions")
    return tools


def _create_tiered_search_tools(mcp_tools: Dict[str, Callable[..., Any]], tools: List[Any]):
    """Create tiered search tools."""
    
    # Basic search tool
    if "search_anime_basic" in mcp_tools:
        @tool("search_anime_basic", args_schema=BasicSearchInput)
        async def search_anime_basic(
            query: str,
            limit: int = 20,
            year: Optional[int] = None,
            type: Optional[AnimeType] = None,
            status: Optional[AnimeStatus] = None,
        ) -> BasicSearchResponse:
            """Basic anime search with 8 essential fields, optimized for speed."""
            validated_input = BasicSearchInput(
                query=query, limit=limit, year=year, type=type, status=status
            )
            return await _execute_tool(mcp_tools["search_anime_basic"], validated_input)
        
        tools.append(search_anime_basic)
    
    # Standard search tool
    if "search_anime_standard" in mcp_tools:
        @tool("search_anime_standard", args_schema=StandardSearchInput)
        async def search_anime_standard(
            query: str,
            limit: int = 20,
            year: Optional[int] = None,
            type: Optional[AnimeType] = None,
            status: Optional[AnimeStatus] = None,
            genres: Optional[List[str]] = None,
            studios: Optional[List[str]] = None,
            rating: Optional[AnimeRating] = None,
            min_score: Optional[float] = None,
            max_score: Optional[float] = None,
        ) -> StandardSearchResponse:
            """Standard anime search with 15 fields and advanced filtering."""
            validated_input = StandardSearchInput(
                query=query, limit=limit, year=year, type=type, status=status,
                genres=genres, studios=studios, rating=rating, 
                min_score=min_score, max_score=max_score
            )
            return await _execute_tool(mcp_tools["search_anime_standard"], validated_input)
        
        tools.append(search_anime_standard)
    
    # Detailed search tool
    if "search_anime_detailed" in mcp_tools:
        @tool("search_anime_detailed", args_schema=DetailedSearchInput)
        async def search_anime_detailed(
            query: str,
            limit: int = 20,
            year: Optional[int] = None,
            type: Optional[AnimeType] = None,
            status: Optional[AnimeStatus] = None,
            genres: Optional[List[str]] = None,
            studios: Optional[List[str]] = None,
            rating: Optional[AnimeRating] = None,
            min_score: Optional[float] = None,
            max_score: Optional[float] = None,
            year_range: Optional[List[int]] = None,
            exclusions: Optional[List[str]] = None,
            mood_keywords: Optional[List[str]] = None,
            cross_platform: bool = True,
        ) -> DetailedSearchResponse:
            """Detailed anime search with 25 fields and cross-platform data."""
            validated_input = DetailedSearchInput(
                query=query, limit=limit, year=year, type=type, status=status,
                genres=genres, studios=studios, rating=rating, 
                min_score=min_score, max_score=max_score, year_range=year_range,
                exclusions=exclusions, mood_keywords=mood_keywords, 
                cross_platform=cross_platform
            )
            return await _execute_tool(mcp_tools["search_anime_detailed"], validated_input)
        
        tools.append(search_anime_detailed)
    
    # Comprehensive search tool
    if "search_anime_comprehensive" in mcp_tools:
        @tool("search_anime_comprehensive", args_schema=ComprehensiveSearchInput)
        async def search_anime_comprehensive(
            query: str,
            limit: int = 20,
            year: Optional[int] = None,
            type: Optional[AnimeType] = None,
            status: Optional[AnimeStatus] = None,
            genres: Optional[List[str]] = None,
            studios: Optional[List[str]] = None,
            rating: Optional[AnimeRating] = None,
            min_score: Optional[float] = None,
            max_score: Optional[float] = None,
            year_range: Optional[List[int]] = None,
            exclusions: Optional[List[str]] = None,
            mood_keywords: Optional[List[str]] = None,
            cross_platform: bool = True,
            include_analytics: bool = True,
            include_market_data: bool = True,
        ) -> ComprehensiveSearchResponse:
            """Comprehensive anime search with 40+ fields and complete analytics."""
            validated_input = ComprehensiveSearchInput(
                query=query, limit=limit, year=year, type=type, status=status,
                genres=genres, studios=studios, rating=rating, 
                min_score=min_score, max_score=max_score, year_range=year_range,
                exclusions=exclusions, mood_keywords=mood_keywords, 
                cross_platform=cross_platform, include_analytics=include_analytics,
                include_market_data=include_market_data
            )
            return await _execute_tool(mcp_tools["search_anime_comprehensive"], validated_input)
        
        tools.append(search_anime_comprehensive)


def _create_tiered_details_tools(mcp_tools: Dict[str, Callable[..., Any]], tools: List[Any]):
    """Create tiered anime details tools."""
    
    # Basic details tool
    if "get_anime_basic" in mcp_tools:
        @tool("get_anime_basic", args_schema=AnimeDetailsInput)
        async def get_anime_basic(anime_id: str) -> BasicAnimeResult:
            """Get basic anime details with 8 essential fields."""
            validated_input = AnimeDetailsInput(anime_id=anime_id)
            return await _execute_tool(mcp_tools["get_anime_basic"], validated_input)
        
        tools.append(get_anime_basic)
    
    # Standard details tool
    if "get_anime_standard" in mcp_tools:
        @tool("get_anime_standard", args_schema=AnimeDetailsInput)
        async def get_anime_standard(anime_id: str) -> StandardAnimeResult:
            """Get standard anime details with 15 fields."""
            validated_input = AnimeDetailsInput(anime_id=anime_id)
            return await _execute_tool(mcp_tools["get_anime_standard"], validated_input)
        
        tools.append(get_anime_standard)
    
    # Detailed details tool
    if "get_anime_detailed" in mcp_tools:
        @tool("get_anime_detailed", args_schema=AnimeDetailsInput)
        async def get_anime_detailed(anime_id: str) -> DetailedAnimeResult:
            """Get detailed anime details with 25 fields."""
            validated_input = AnimeDetailsInput(anime_id=anime_id)
            return await _execute_tool(mcp_tools["get_anime_detailed"], validated_input)
        
        tools.append(get_anime_detailed)
    
    # Comprehensive details tool
    if "get_anime_comprehensive" in mcp_tools:
        @tool("get_anime_comprehensive", args_schema=AnimeDetailsInput)
        async def get_anime_comprehensive(anime_id: str) -> ComprehensiveAnimeResult:
            """Get comprehensive anime details with 40+ fields."""
            validated_input = AnimeDetailsInput(anime_id=anime_id)
            return await _execute_tool(mcp_tools["get_anime_comprehensive"], validated_input)
        
        tools.append(get_anime_comprehensive)


def _create_tiered_similarity_tools(mcp_tools: Dict[str, Callable[..., Any]], tools: List[Any]):
    """Create tiered similarity tools."""
    
    # Basic similarity tool
    if "find_similar_anime_basic" in mcp_tools:
        @tool("find_similar_anime_basic", args_schema=SimilarAnimeInput)
        async def find_similar_anime_basic(anime_id: str, limit: int = 10) -> BasicSearchResponse:
            """Find similar anime with basic information."""
            validated_input = SimilarAnimeInput(anime_id=anime_id, limit=limit)
            return await _execute_tool(mcp_tools["find_similar_anime_basic"], validated_input)
        
        tools.append(find_similar_anime_basic)
    
    # Standard similarity tool
    if "find_similar_anime_standard" in mcp_tools:
        @tool("find_similar_anime_standard", args_schema=SimilarAnimeInput)
        async def find_similar_anime_standard(anime_id: str, limit: int = 10) -> StandardSearchResponse:
            """Find similar anime with standard information."""
            validated_input = SimilarAnimeInput(anime_id=anime_id, limit=limit)
            return await _execute_tool(mcp_tools["find_similar_anime_standard"], validated_input)
        
        tools.append(find_similar_anime_standard)
    
    # Detailed similarity tool
    if "find_similar_anime_detailed" in mcp_tools:
        @tool("find_similar_anime_detailed", args_schema=SimilarAnimeInput)
        async def find_similar_anime_detailed(anime_id: str, limit: int = 10) -> DetailedSearchResponse:
            """Find similar anime with detailed information."""
            validated_input = SimilarAnimeInput(anime_id=anime_id, limit=limit)
            return await _execute_tool(mcp_tools["find_similar_anime_detailed"], validated_input)
        
        tools.append(find_similar_anime_detailed)
    
    # Comprehensive similarity tool
    if "find_similar_anime_comprehensive" in mcp_tools:
        @tool("find_similar_anime_comprehensive", args_schema=SimilarAnimeInput)
        async def find_similar_anime_comprehensive(anime_id: str, limit: int = 10) -> ComprehensiveSearchResponse:
            """Find similar anime with comprehensive information."""
            validated_input = SimilarAnimeInput(anime_id=anime_id, limit=limit)
            return await _execute_tool(mcp_tools["find_similar_anime_comprehensive"], validated_input)
        
        tools.append(find_similar_anime_comprehensive)


def _create_tiered_seasonal_tools(mcp_tools: Dict[str, Callable[..., Any]], tools: List[Any]):
    """Create tiered seasonal anime tools."""
    
    # Basic seasonal tool
    if "get_seasonal_anime_basic" in mcp_tools:
        @tool("get_seasonal_anime_basic")
        async def get_seasonal_anime_basic() -> BasicSearchResponse:
            """Get currently airing anime with basic information."""
            return await _execute_tool(mcp_tools["get_seasonal_anime_basic"], {})
        
        tools.append(get_seasonal_anime_basic)
    
    # Standard seasonal tool
    if "get_seasonal_anime_standard" in mcp_tools:
        @tool("get_seasonal_anime_standard")
        async def get_seasonal_anime_standard() -> StandardSearchResponse:
            """Get currently airing anime with standard information."""
            return await _execute_tool(mcp_tools["get_seasonal_anime_standard"], {})
        
        tools.append(get_seasonal_anime_standard)
    
    # Detailed seasonal tool
    if "get_seasonal_anime_detailed" in mcp_tools:
        @tool("get_seasonal_anime_detailed")
        async def get_seasonal_anime_detailed() -> DetailedSearchResponse:
            """Get currently airing anime with detailed information."""
            return await _execute_tool(mcp_tools["get_seasonal_anime_detailed"], {})
        
        tools.append(get_seasonal_anime_detailed)


def _create_legacy_tools(mcp_tools: Dict[str, Callable[..., Any]], tools: List[Any]):
    """Create legacy tools for backward compatibility."""

    # Legacy search anime tool
    if "search_anime" in mcp_tools:
        @tool("search_anime", args_schema=SearchAnimeInput)
        async def search_anime(
            query: str,
            limit: int = 10,
            genres: Optional[List[str]] = None,
            year_range: Optional[List[int]] = None,
            anime_types: Optional[List[str]] = None,
            studios: Optional[List[str]] = None,
            exclusions: Optional[List[str]] = None,
            mood_keywords: Optional[List[str]] = None,
        ) -> Any:
            """Search for anime using semantic search with natural language queries (legacy)."""
            validated_input = SearchAnimeInput(
                query=query, limit=limit, genres=genres, year_range=year_range,
                anime_types=anime_types, studios=studios, exclusions=exclusions,
                mood_keywords=mood_keywords,
            )
            return await _execute_tool(mcp_tools["search_anime"], validated_input)
        
        tools.append(search_anime)
    
    # Legacy get anime details tool
    if "get_anime_details" in mcp_tools:
        @tool("get_anime_details", args_schema=GetAnimeDetailsInput)
        async def get_anime_details(anime_id: str) -> Any:
            """Get detailed information about a specific anime by its ID (legacy)."""
            validated_input = GetAnimeDetailsInput(anime_id=anime_id)
            return await _execute_tool(mcp_tools["get_anime_details"], validated_input)
        
        tools.append(get_anime_details)
    
    # Legacy find similar anime tool
    if "find_similar_anime" in mcp_tools:
        @tool("find_similar_anime", args_schema=FindSimilarAnimeInput)
        async def find_similar_anime(anime_id: str, limit: int = 10) -> Any:
            """Find anime similar to a given anime based on content and metadata (legacy)."""
            validated_input = FindSimilarAnimeInput(anime_id=anime_id, limit=limit)
            return await _execute_tool(mcp_tools["find_similar_anime"], validated_input)
        
        tools.append(find_similar_anime)
    
    # Legacy get anime stats tool
    if "get_anime_stats" in mcp_tools:
        @tool("get_anime_stats")
        async def get_anime_stats() -> Any:
            """Get database statistics and health information (legacy)."""
            return await _execute_tool(mcp_tools["get_anime_stats"], {})
        
        tools.append(get_anime_stats)
    
    # Legacy search anime by image tool
    if "search_anime_by_image" in mcp_tools:
        @tool("search_anime_by_image", args_schema=SearchAnimeByImageInput)
        async def search_anime_by_image(image_data: str, limit: int = 10) -> Any:
            """Search for anime using image similarity with CLIP embeddings (legacy)."""
            validated_input = SearchAnimeByImageInput(image_data=image_data, limit=limit)
            return await _execute_tool(mcp_tools["search_anime_by_image"], validated_input)
        
        tools.append(search_anime_by_image)
    
    # Legacy find visually similar anime tool
    if "find_visually_similar_anime" in mcp_tools:
        @tool("find_visually_similar_anime", args_schema=FindVisuallySimilarAnimeInput)
        async def find_visually_similar_anime(anime_id: str, limit: int = 10) -> Any:
            """Find anime with visually similar poster images (legacy)."""
            validated_input = FindVisuallySimilarAnimeInput(anime_id=anime_id, limit=limit)
            return await _execute_tool(mcp_tools["find_visually_similar_anime"], validated_input)
        
        tools.append(find_visually_similar_anime)
    
    # Legacy search multimodal anime tool
    if "search_multimodal_anime" in mcp_tools:
        @tool("search_multimodal_anime", args_schema=SearchMultimodalAnimeInput)
        async def search_multimodal_anime(
            query: str,
            image_data: Optional[str] = None,
            text_weight: float = 0.7,
            limit: int = 10,
        ) -> Any:
            """Perform combined text and image search with weighted results (legacy)."""
            validated_input = SearchMultimodalAnimeInput(
                query=query, image_data=image_data, text_weight=text_weight, limit=limit
            )
            return await _execute_tool(mcp_tools["search_multimodal_anime"], validated_input)
        
        tools.append(search_multimodal_anime)


async def _execute_tool(tool_func: Callable[..., Any], validated_input: BaseModel) -> Any:
    """Execute a tool function with proper error handling."""
    try:
        validated_params = validated_input.model_dump()
        # Execute MCP tool function
        if hasattr(tool_func, "ainvoke"):
            result = await tool_func.ainvoke(validated_params)
        else:
            result = await tool_func(**validated_params)
        return result
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        raise


class ToolNodeConversationState(TypedDict):
    """State schema for ToolNode-based StateGraph workflow."""

    messages: List[Any]  # LangChain messages
    session_id: str
    current_context: Optional[Dict[str, Any]]
    user_preferences: Optional[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]


class AnimeToolNodeWorkflow:
    """LangGraph StateGraph workflow using ToolNode for anime conversations."""

    def __init__(self, mcp_tools: Dict[str, Callable[..., Any]]):
        """Initialize the workflow with MCP tools.

        Args:
            mcp_tools: Dictionary mapping tool names to their callable functions
        """
        self.mcp_tools = mcp_tools
        self.tools = create_anime_langchain_tools(mcp_tools)
        self.tool_node = ToolNode(self.tools)
        self.memory_saver = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        """Build the StateGraph workflow with ToolNode integration."""
        # Create StateGraph with message-based state
        workflow = StateGraph(ToolNodeConversationState)

        # Add nodes
        workflow.add_node("chatbot", self._chatbot_node)
        workflow.add_node("tools", self.tool_node)

        # Add edges with tools_condition for automatic routing
        workflow.add_conditional_edges(
            "chatbot",
            tools_condition,  # Built-in condition that checks for tool calls
            path_map=["tools", "__end__"],
        )

        # Return to chatbot after tool execution
        workflow.add_edge("tools", "chatbot")

        # Set entry point
        workflow.set_entry_point("chatbot")

        # Compile with checkpointer
        return workflow.compile(checkpointer=self.memory_saver)

    async def _chatbot_node(
        self, state: ToolNodeConversationState
    ) -> ToolNodeConversationState:
        """Chatbot node that determines whether to use tools or provide final response.

        This is a simplified chatbot that analyzes the conversation and decides
        whether to call tools or provide a final response.
        """
        from langchain_core.messages import AIMessage, HumanMessage

        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]

        # If the last message is a ToolMessage (result from tool execution),
        # we should provide a summary response
        from langchain_core.messages import ToolMessage

        if isinstance(last_message, ToolMessage):
            # Parse tool results and create summary
            tool_result = last_message.content
            summary_message = AIMessage(
                content=self._create_response_from_tool_result(
                    tool_result, last_message.tool_call_id
                )
            )

            return {**state, "messages": messages + [summary_message]}

        # If it's a human message, determine if we need to call tools
        elif isinstance(last_message, HumanMessage):
            # Analyze the message to determine if tools are needed
            content = last_message.content
            if isinstance(content, list):
                # If content is a list, extract text content
                text_content = ""
                for item in content:
                    if isinstance(item, str):
                        text_content += item + " "
                    elif isinstance(item, dict) and "text" in item:
                        text_content += item["text"] + " "
                content = text_content.strip()
            elif not isinstance(content, str):
                content = str(content)

            needs_search = self._needs_anime_search(content)

            if needs_search:
                # Create tool call for anime search
                tool_call_id = f"call_{len(messages)}"
                search_params = self._extract_search_parameters(content)

                ai_message = AIMessage(
                    content="I'll search for anime based on your request.",
                    tool_calls=[
                        {
                            "name": "search_anime",
                            "args": search_params,
                            "id": tool_call_id,
                        }
                    ],
                )

                return {**state, "messages": messages + [ai_message]}
            else:
                # Provide direct response without tools
                response = AIMessage(
                    content="I can help you search for anime. Please provide more specific criteria like genres, titles, or descriptions."
                )

                return {**state, "messages": messages + [response]}

        # Default: return state unchanged
        return state

    def _needs_anime_search(self, message_content: str) -> bool:
        """Determine if the message requires anime search."""
        search_keywords = [
            "find",
            "search",
            "recommend",
            "anime",
            "show",
            "series",
            "movie",
            "genre",
            "action",
            "romance",
            "comedy",
            "drama",
            "fantasy",
            "sci-fi",
        ]

        message_lower = message_content.lower()
        return any(keyword in message_lower for keyword in search_keywords)

    def _extract_search_parameters(self, message_content: str) -> Dict[str, Any]:
        """Extract search parameters from message content."""
        # Simple parameter extraction - in production this would use LLM
        params = {"query": message_content, "limit": 10}

        # Extract limit if mentioned
        import re

        limit_match = re.search(
            r"\b(\d+)\s*(?:results?|anime|shows?)", message_content.lower()
        )
        if limit_match:
            params["limit"] = min(int(limit_match.group(1)), 50)

        return params

    def _create_response_from_tool_result(
        self, tool_result: Any, tool_call_id: str
    ) -> str:
        """Create a natural language response from tool results."""
        if isinstance(tool_result, list) and tool_result:
            # Search results
            count = len(tool_result)
            if count == 1:
                anime = tool_result[0]
                return f"I found this anime: {anime.get('title', 'Unknown')} - {anime.get('synopsis', 'No description available')[:100]}..."
            else:
                titles = [anime.get("title", "Unknown") for anime in tool_result[:3]]
                response = f"I found {count} anime. Here are the top results:\n"
                for i, title in enumerate(titles, 1):
                    response += f"{i}. {title}\n"
                return response
        elif isinstance(tool_result, dict):
            # Single anime details
            title = tool_result.get("title", "Unknown")
            synopsis = tool_result.get("synopsis", "No description available")
            return f"Here are the details for {title}: {synopsis[:200]}..."
        else:
            return "I couldn't find any anime matching your criteria. Please try a different search."

    async def process_conversation(
        self, session_id: str, message: str, thread_id: Optional[str] = None
    ) -> ToolNodeConversationState:
        """Process a conversation message using ToolNode workflow.

        Args:
            session_id: Unique session identifier
            message: User message to process
            thread_id: Optional thread ID for conversation persistence

        Returns:
            Updated conversation state after processing
        """
        from langchain_core.messages import HumanMessage

        logger.info(f"Processing conversation for session {session_id} using ToolNode")

        try:
            # Prepare initial state
            initial_state = ToolNodeConversationState(
                messages=[HumanMessage(content=message)],
                session_id=session_id,
                current_context=None,
                user_preferences=None,
                tool_results=[],
            )

            # Configure checkpointing
            config = cast(Any, {"configurable": {"thread_id": thread_id or session_id}})

            # Execute workflow
            result = await self.graph.ainvoke(initial_state, config=config)

            logger.info(f"ToolNode processed message for session {session_id}")
            return cast(ToolNodeConversationState, result)

        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
            raise

    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the ToolNode workflow.

        Returns:
            Dictionary with workflow metadata
        """
        return {
            "engine_type": "StateGraph+ToolNode",
            "features": [
                "LangChain ToolNode integration",
                "Native tool binding with bind_tools()",
                "Automatic tool routing with tools_condition",
                "Memory persistence with MemorySaver",
                "Reduced boilerplate (no custom adapters)",
                "Type-safe tool schemas",
            ],
            "performance": {
                "target_response_time": "150ms",  # Improved from adapter pattern
                "memory_persistence": True,
                "conversation_threading": True,
                "tools_count": len(self.tools),
            },
            "tools": [tool.name for tool in self.tools],
        }


def create_toolnode_workflow_from_mcp_tools(
    mcp_tools: Dict[str, Callable[..., Any]],
) -> AnimeToolNodeWorkflow:
    """Create ToolNode workflow from MCP tool functions.

    This is the main factory function that replaces create_adapter_registry_from_mcp_tools.

    Args:
        mcp_tools: Dictionary mapping tool names to their functions

    Returns:
        AnimeToolNodeWorkflow ready for conversation processing
    """
    logger.info(f"Creating ToolNode workflow with {len(mcp_tools)} MCP tools")
    workflow = AnimeToolNodeWorkflow(mcp_tools)
    logger.info("ToolNode workflow created successfully")
    return workflow
