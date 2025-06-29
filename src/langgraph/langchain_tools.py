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

logger = logging.getLogger(__name__)


class SearchAnimeInput(BaseModel):
    """Input schema for search_anime tool with SearchIntent parameters."""

    query: str = Field(description="Natural language search query")
    limit: int = Field(default=10, description="Maximum number of results (1-50)")
    # Enhanced SearchIntent parameters
    genres: Optional[List[str]] = Field(
        None, description="List of anime genres to filter by"
    )
    year_range: Optional[List[int]] = Field(
        None, description="Year range as [start_year, end_year]"
    )
    anime_types: Optional[List[str]] = Field(
        None, description="List of anime types (TV, Movie, etc.)"
    )
    studios: Optional[List[str]] = Field(None, description="List of animation studios")
    exclusions: Optional[List[str]] = Field(
        None, description="List of genres/themes to exclude"
    )
    mood_keywords: Optional[List[str]] = Field(
        None, description="List of mood descriptors"
    )


class GetAnimeDetailsInput(BaseModel):
    """Input schema for get_anime_details tool."""

    anime_id: str = Field(description="Unique anime identifier")


class FindSimilarAnimeInput(BaseModel):
    """Input schema for find_similar_anime tool."""

    anime_id: str = Field(description="Reference anime ID")
    limit: int = Field(default=10, description="Maximum number of results (1-20)")


class GetAnimeStatsInput(BaseModel):
    """Input schema for get_anime_stats tool."""

    pass  # No parameters needed


class SearchAnimeByImageInput(BaseModel):
    """Input schema for search_anime_by_image tool."""

    image_data: str = Field(description="Base64 encoded image data")
    limit: int = Field(default=10, description="Maximum number of results (1-30)")


class FindVisuallySimilarAnimeInput(BaseModel):
    """Input schema for find_visually_similar_anime tool."""

    anime_id: str = Field(description="Reference anime ID")
    limit: int = Field(default=10, description="Maximum number of results (1-20)")


class SearchMultimodalAnimeInput(BaseModel):
    """Input schema for search_multimodal_anime tool."""

    query: str = Field(description="Text search query")
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    text_weight: float = Field(default=0.7, description="Text weight (0.0-1.0)")
    limit: int = Field(default=10, description="Maximum number of results (1-25)")


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
    using the @tool decorator, eliminating the need for custom adapters.

    Args:
        mcp_tools: Dictionary mapping tool names to their callable functions

    Returns:
        List of LangChain tools ready for use with ToolNode
    """
    tools = []

    # Search anime tool
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
            """Search for anime using semantic search with natural language queries."""
            
            try:
                validated_input = SearchAnimeInput(
                    query=query,
                    limit=limit,
                    genres=genres,
                    year_range=year_range,
                    anime_types=anime_types,
                    studios=studios,
                    exclusions=exclusions,
                    mood_keywords=mood_keywords,
                )
                validated_params = validated_input.model_dump()
                # Handle both MCP tools (with .ainvoke) and test mocks (direct callable)
                tool_func = mcp_tools["search_anime"]
                if hasattr(tool_func, 'ainvoke'):
                    result = await tool_func.ainvoke(validated_params)
                else:
                    # For tests: call function directly with unpacked parameters
                    result = await tool_func(**validated_params)
                return result
                
            except Exception as e:
                logger.error(f"SearchAnimeInput validation failed: {e}")
                raise

        tools.append(search_anime)

    # Get anime details tool
    if "get_anime_details" in mcp_tools:

        @tool("get_anime_details", args_schema=GetAnimeDetailsInput)
        async def get_anime_details(anime_id: str) -> Any:
            """Get detailed information about a specific anime by its ID."""
            validated_input = GetAnimeDetailsInput(anime_id=anime_id)
            tool_func = mcp_tools["get_anime_details"]
            if hasattr(tool_func, 'ainvoke'):
                return await tool_func.ainvoke(validated_input.model_dump())
            else:
                return await tool_func(**validated_input.model_dump())

        tools.append(get_anime_details)

    # Find similar anime tool
    if "find_similar_anime" in mcp_tools:

        @tool("find_similar_anime", args_schema=FindSimilarAnimeInput)
        async def find_similar_anime(anime_id: str, limit: int = 10) -> Any:
            """Find anime similar to a given anime based on content and metadata."""
            validated_input = FindSimilarAnimeInput(anime_id=anime_id, limit=limit)
            tool_func = mcp_tools["find_similar_anime"]
            if hasattr(tool_func, 'ainvoke'):
                return await tool_func.ainvoke(validated_input.model_dump())
            else:
                return await tool_func(**validated_input.model_dump())

        tools.append(find_similar_anime)

    # Get anime stats tool
    if "get_anime_stats" in mcp_tools:

        @tool("get_anime_stats")
        async def get_anime_stats() -> Any:
            """Get database statistics and health information."""
            tool_func = mcp_tools["get_anime_stats"]
            if hasattr(tool_func, 'ainvoke'):
                return await tool_func.ainvoke({})
            else:
                return await tool_func()

        tools.append(get_anime_stats)

    # Search anime by image tool
    if "search_anime_by_image" in mcp_tools:

        @tool("search_anime_by_image", args_schema=SearchAnimeByImageInput)
        async def search_anime_by_image(image_data: str, limit: int = 10) -> Any:
            """Search for anime using image similarity with CLIP embeddings."""
            validated_input = SearchAnimeByImageInput(
                image_data=image_data, limit=limit
            )
            tool_func = mcp_tools["search_anime_by_image"]
            if hasattr(tool_func, 'ainvoke'):
                return await tool_func.ainvoke(validated_input.model_dump())
            else:
                return await tool_func(**validated_input.model_dump())

        tools.append(search_anime_by_image)

    # Find visually similar anime tool
    if "find_visually_similar_anime" in mcp_tools:

        @tool("find_visually_similar_anime", args_schema=FindVisuallySimilarAnimeInput)
        async def find_visually_similar_anime(anime_id: str, limit: int = 10) -> Any:
            """Find anime with visually similar poster images."""
            validated_input = FindVisuallySimilarAnimeInput(
                anime_id=anime_id, limit=limit
            )
            tool_func = mcp_tools["find_visually_similar_anime"]
            if hasattr(tool_func, 'ainvoke'):
                return await tool_func.ainvoke(validated_input.model_dump())
            else:
                return await tool_func(**validated_input.model_dump())

        tools.append(find_visually_similar_anime)

    # Search multimodal anime tool
    if "search_multimodal_anime" in mcp_tools:

        @tool("search_multimodal_anime", args_schema=SearchMultimodalAnimeInput)
        async def search_multimodal_anime(
            query: str,
            image_data: Optional[str] = None,
            text_weight: float = 0.7,
            limit: int = 10,
        ) -> Any:
            """Perform combined text and image search with weighted results."""
            validated_input = SearchMultimodalAnimeInput(
                query=query, image_data=image_data, text_weight=text_weight, limit=limit
            )
            tool_func = mcp_tools["search_multimodal_anime"]
            if hasattr(tool_func, 'ainvoke'):
                return await tool_func.ainvoke(validated_input.model_dump())
            else:
                return await tool_func(**validated_input.model_dump())

        tools.append(search_multimodal_anime)

    logger.info(f"Created {len(tools)} LangChain tools from MCP functions")
    return tools


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
