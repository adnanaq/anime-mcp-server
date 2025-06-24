"""MCP tool adapter layer for LangGraph integration."""

import logging
from abc import ABC
from typing import Any, Awaitable, Callable, Dict, List

logger = logging.getLogger(__name__)


class MCPToolAdapter(ABC):
    """Base adapter for MCP tools."""

    def __init__(
        self,
        tool_name: str,
        tool_function: Callable[[Dict[str, Any]], Awaitable[Any]],
        description: str,
    ):
        self.tool_name = tool_name
        self.tool_function = tool_function
        self.description = description

    async def invoke(self, parameters: Dict[str, Any]) -> Any:
        """Invoke the underlying MCP tool."""
        try:
            logger.debug(f"Invoking {self.tool_name} with parameters: {parameters}")
            # MCP tools expect individual parameters, not a dictionary
            result = await self.tool_function(**parameters)
            logger.debug(
                f"{self.tool_name} returned: {type(result)} with {len(result) if isinstance(result, (list, dict)) else 'N/A'} items"
            )
            return result
        except Exception as e:
            logger.error(f"Error invoking {self.tool_name}: {e}")
            raise


class AnimeSearchAdapter(MCPToolAdapter):
    """Adapter for semantic anime search."""

    def __init__(
        self,
        search_function: Callable[[Dict[str, Any]], Awaitable[List[Dict[str, Any]]]],
    ):
        super().__init__(
            tool_name="search_anime",
            tool_function=search_function,
            description="Perform semantic search over anime database using natural language queries",
        )


class AnimeDetailsAdapter(MCPToolAdapter):
    """Adapter for getting anime details."""

    def __init__(
        self, details_function: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
    ):
        super().__init__(
            tool_name="get_anime_details",
            tool_function=details_function,
            description="Get detailed information about a specific anime by its ID",
        )


class SimilarityAdapter(MCPToolAdapter):
    """Adapter for finding similar anime."""

    def __init__(
        self,
        similarity_function: Callable[
            [Dict[str, Any]], Awaitable[List[Dict[str, Any]]]
        ],
    ):
        super().__init__(
            tool_name="find_similar_anime",
            tool_function=similarity_function,
            description="Find anime similar to a given anime based on content and metadata",
        )


class StatsAdapter(MCPToolAdapter):
    """Adapter for database statistics."""

    def __init__(
        self, stats_function: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
    ):
        super().__init__(
            tool_name="get_anime_stats",
            tool_function=stats_function,
            description="Get database statistics and health information",
        )


class RecommendationAdapter(MCPToolAdapter):
    """Adapter for anime recommendations."""

    def __init__(
        self,
        recommendation_function: Callable[
            [Dict[str, Any]], Awaitable[List[Dict[str, Any]]]
        ],
    ):
        super().__init__(
            tool_name="recommend_anime",
            tool_function=recommendation_function,
            description="Get personalized anime recommendations based on preferences",
        )


class ImageSearchAdapter(MCPToolAdapter):
    """Adapter for image-based anime search."""

    def __init__(
        self,
        image_search_function: Callable[
            [Dict[str, Any]], Awaitable[List[Dict[str, Any]]]
        ],
    ):
        super().__init__(
            tool_name="search_anime_by_image",
            tool_function=image_search_function,
            description="Search for anime using image similarity with CLIP embeddings",
        )


class VisualSimilarityAdapter(MCPToolAdapter):
    """Adapter for visual similarity search."""

    def __init__(
        self,
        visual_similarity_function: Callable[
            [Dict[str, Any]], Awaitable[List[Dict[str, Any]]]
        ],
    ):
        super().__init__(
            tool_name="find_visually_similar_anime",
            tool_function=visual_similarity_function,
            description="Find anime with visually similar poster images",
        )


class MultimodalSearchAdapter(MCPToolAdapter):
    """Adapter for multimodal search (text + image)."""

    def __init__(
        self,
        multimodal_function: Callable[
            [Dict[str, Any]], Awaitable[List[Dict[str, Any]]]
        ],
    ):
        super().__init__(
            tool_name="search_multimodal_anime",
            tool_function=multimodal_function,
            description="Perform combined text and image search with weighted results",
        )


class MCPAdapterRegistry:
    """Registry for managing MCP tool adapters."""

    def __init__(self):
        self._adapters: Dict[str, MCPToolAdapter] = {}

    def register(self, adapter: MCPToolAdapter) -> None:
        """Register an adapter."""
        self._adapters[adapter.tool_name] = adapter
        logger.info(f"Registered adapter: {adapter.tool_name}")

    def get_adapter(self, tool_name: str) -> MCPToolAdapter:
        """Get an adapter by tool name."""
        if tool_name not in self._adapters:
            raise ValueError(f"Adapter not found: {tool_name}")
        return self._adapters[tool_name]

    def get_all_adapters(self) -> Dict[str, MCPToolAdapter]:
        """Get all registered adapters."""
        return self._adapters.copy()

    def list_tool_names(self) -> List[str]:
        """Get list of all registered tool names."""
        return list(self._adapters.keys())

    async def invoke_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Invoke a tool by name."""
        adapter = self.get_adapter(tool_name)
        return await adapter.invoke(parameters)


def create_adapter_registry_from_mcp_tools(
    mcp_tools: Dict[str, Callable],
) -> MCPAdapterRegistry:
    """Create adapter registry from MCP tool functions.

    Args:
        mcp_tools: Dictionary mapping tool names to their functions

    Returns:
        MCPAdapterRegistry with all tools registered
    """
    registry = MCPAdapterRegistry()

    adapter_classes = {
        "search_anime": AnimeSearchAdapter,
        "get_anime_details": AnimeDetailsAdapter,
        "find_similar_anime": SimilarityAdapter,
        "get_anime_stats": StatsAdapter,
        "recommend_anime": RecommendationAdapter,
        "search_anime_by_image": ImageSearchAdapter,
        "find_visually_similar_anime": VisualSimilarityAdapter,
        "search_multimodal_anime": MultimodalSearchAdapter,
    }

    for tool_name, tool_function in mcp_tools.items():
        if tool_name in adapter_classes:
            adapter_class = adapter_classes[tool_name]
            adapter = adapter_class(tool_function)
            registry.register(adapter)
        else:
            logger.warning(f"No adapter class found for tool: {tool_name}")

    logger.info(
        f"Created adapter registry with {len(registry.list_tool_names())} tools"
    )
    return registry
