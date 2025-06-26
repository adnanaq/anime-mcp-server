"""Tests for LangGraph ToolNode migration replacing MCPAdapterRegistry."""

from unittest.mock import AsyncMock, Mock

import pytest

from langgraph.prebuilt import tools_condition
from src.langgraph.langchain_tools import (
    AnimeToolNodeWorkflow,
    LangChainToolAdapter,
    create_anime_langchain_tools,
)


class TestCreateAnimeLangChainTools:
    """Test conversion of MCP functions to LangChain tools."""

    @pytest.mark.asyncio
    async def test_create_tools_from_mcp_functions(self):
        """Test that MCP functions are properly converted to LangChain tools."""
        # Mock MCP functions
        mock_search = AsyncMock(
            return_value=[{"anime_id": "test1", "title": "Test Anime"}]
        )
        mock_details = AsyncMock(
            return_value={"anime_id": "test1", "title": "Test Anime"}
        )
        mock_stats = AsyncMock(return_value={"total_documents": 100})

        mcp_tools = {
            "search_anime": mock_search,
            "get_anime_details": mock_details,
            "get_anime_stats": mock_stats,
        }

        # Create LangChain tools
        tools = create_anime_langchain_tools(mcp_tools)

        # Should have 3 tools
        assert len(tools) == 3

        # All tools should be callable LangChain tools
        tool_names = [tool.name for tool in tools]
        assert "search_anime" in tool_names
        assert "get_anime_details" in tool_names
        assert "get_anime_stats" in tool_names

        # Tools should have proper descriptions
        search_tool = next(t for t in tools if t.name == "search_anime")
        assert "semantic search" in search_tool.description.lower()

    @pytest.mark.asyncio
    async def test_tool_execution_with_parameters(self):
        """Test that tools execute with proper parameter passing."""
        # Mock search function
        mock_search = AsyncMock(
            return_value=[{"anime_id": "test1", "title": "Action Anime"}]
        )

        mcp_tools = {"search_anime": mock_search}
        tools = create_anime_langchain_tools(mcp_tools)

        search_tool = next(t for t in tools if t.name == "search_anime")

        # Execute tool with parameters
        result = await search_tool.ainvoke({"query": "action", "limit": 5})

        # Should call underlying function with unpacked parameters (including enhanced parameters as None)
        mock_search.assert_called_once_with(
            query="action",
            limit=5,
            genres=None,
            year_range=None,
            anime_types=None,
            studios=None,
            exclusions=None,
            mood_keywords=None,
        )
        assert result[0]["title"] == "Action Anime"

    @pytest.mark.asyncio
    async def test_multimodal_tool_creation(self):
        """Test multimodal search tool creation and parameter handling."""
        mock_multimodal = AsyncMock(
            return_value=[
                {"anime_id": "mm1", "title": "Multimodal Match", "score": 0.9}
            ]
        )

        mcp_tools = {"search_multimodal_anime": mock_multimodal}
        tools = create_anime_langchain_tools(mcp_tools)

        multimodal_tool = next(t for t in tools if t.name == "search_multimodal_anime")

        # Execute with multimodal parameters
        result = await multimodal_tool.ainvoke(
            {
                "query": "mecha robots",
                "image_data": "base64_image_data",
                "text_weight": 0.6,
                "limit": 8,
            }
        )

        mock_multimodal.assert_called_once_with(
            query="mecha robots",
            image_data="base64_image_data",
            text_weight=0.6,
            limit=8,
        )
        assert result[0]["score"] == 0.9

    def test_tools_have_proper_schemas(self):
        """Test that tools have proper input schemas for LangGraph."""
        mock_search = Mock()
        mcp_tools = {"search_anime": mock_search}
        tools = create_anime_langchain_tools(mcp_tools)

        search_tool = next(t for t in tools if t.name == "search_anime")

        # Should have input schema
        assert hasattr(search_tool, "args_schema")
        assert search_tool.args_schema is not None


class TestLangChainToolAdapter:
    """Test the LangChain tool adapter wrapper."""

    @pytest.mark.asyncio
    async def test_adapter_wraps_mcp_function(self):
        """Test that adapter properly wraps MCP function."""
        mock_func = AsyncMock(return_value={"result": "success"})

        adapter = LangChainToolAdapter(
            name="test_tool", func=mock_func, description="Test tool description"
        )

        result = await adapter.execute(param1="value1", param2="value2")

        mock_func.assert_called_once_with(param1="value1", param2="value2")
        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_adapter_error_handling(self):
        """Test adapter error handling."""
        mock_func = AsyncMock(side_effect=Exception("Tool error"))

        adapter = LangChainToolAdapter(
            name="failing_tool", func=mock_func, description="Failing tool"
        )

        with pytest.raises(Exception, match="Tool error"):
            await adapter.execute(param="value")

    def test_adapter_properties(self):
        """Test adapter exposes correct properties."""
        mock_func = Mock()

        adapter = LangChainToolAdapter(
            name="prop_tool", func=mock_func, description="Property test tool"
        )

        assert adapter.name == "prop_tool"
        assert adapter.description == "Property test tool"
        assert adapter.func == mock_func


class TestAnimeToolNodeWorkflow:
    """Test the new ToolNode-based workflow."""

    @pytest.fixture
    def mock_mcp_tools(self):
        """Mock MCP tools for testing."""
        return {
            "search_anime": AsyncMock(
                return_value=[{"anime_id": "t1", "title": "Test"}]
            ),
            "get_anime_details": AsyncMock(
                return_value={"anime_id": "t1", "details": "test"}
            ),
            "get_anime_stats": AsyncMock(return_value={"total": 100}),
        }

    @pytest.fixture
    def workflow_engine(self, mock_mcp_tools):
        """Create workflow engine with mocked tools."""
        return AnimeToolNodeWorkflow(mock_mcp_tools)

    def test_workflow_initialization(self, workflow_engine):
        """Test workflow engine initializes properly."""
        assert workflow_engine.tools is not None
        assert len(workflow_engine.tools) == 3
        assert workflow_engine.tool_node is not None
        assert workflow_engine.graph is not None

    def test_tools_are_langchain_compatible(self, workflow_engine):
        """Test that tools are LangChain compatible."""
        for tool in workflow_engine.tools:
            # Should have LangChain tool interface
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "invoke") or hasattr(tool, "ainvoke")

    def test_tool_node_creation(self, workflow_engine):
        """Test that ToolNode is created properly."""
        assert workflow_engine.tool_node is not None

        # ToolNode should contain our tools in tools_by_name
        # Note: ToolNode internal structure may vary, so we test basic properties
        assert hasattr(workflow_engine.tool_node, "tools_by_name")
        assert len(workflow_engine.tool_node.tools_by_name) == 3

    @pytest.mark.asyncio
    async def test_workflow_tool_execution(self, workflow_engine, mock_mcp_tools):
        """Test tool execution through the workflow."""
        # This tests the integration between ToolNode and our tools
        tools = workflow_engine.tools
        search_tool = next(t for t in tools if t.name == "search_anime")

        result = await search_tool.ainvoke({"query": "test query", "limit": 5})

        # Should call the underlying MCP function (including enhanced parameters as None)
        mock_mcp_tools["search_anime"].assert_called_once_with(
            query="test query",
            limit=5,
            genres=None,
            year_range=None,
            anime_types=None,
            studios=None,
            exclusions=None,
            mood_keywords=None,
        )
        assert result[0]["title"] == "Test"

    def test_graph_has_tool_nodes(self, workflow_engine):
        """Test that the StateGraph includes tool nodes."""
        graph = workflow_engine.graph

        # Graph should be compiled
        assert graph is not None
        assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")

    @pytest.mark.asyncio
    async def test_tool_condition_routing(self, workflow_engine):
        """Test that tools_condition works with our setup."""
        # Create a mock AI message with tool calls
        from langchain_core.messages import AIMessage

        ai_message = AIMessage(
            content="I'll search for anime",
            tool_calls=[
                {
                    "name": "search_anime",
                    "args": {"query": "action", "limit": 5},
                    "id": "call_123",
                }
            ],
        )

        # Test that tools_condition identifies this as needing tool execution
        condition_result = tools_condition({"messages": [ai_message]})

        # Should route to tools node (not END)
        assert condition_result == "tools"

    def test_workflow_info_updated(self, workflow_engine):
        """Test that workflow info reflects ToolNode usage."""
        info = workflow_engine.get_workflow_info()

        assert info["engine_type"] == "StateGraph+ToolNode"
        assert "LangChain ToolNode integration" in info["features"]
        assert "Native tool binding with bind_tools()" in info["features"]


class TestToolConditionIntegration:
    """Test tools_condition integration with anime tools."""

    def test_tools_condition_with_no_tool_calls(self):
        """Test routing when no tool calls are needed."""
        from langchain_core.messages import HumanMessage

        human_message = HumanMessage(content="Hello")
        condition_result = tools_condition({"messages": [human_message]})

        # Should route to end since no tools needed
        assert condition_result == "__end__"

    def test_tools_condition_with_tool_calls(self):
        """Test routing when tool calls are present."""
        from langchain_core.messages import AIMessage

        ai_message = AIMessage(
            content="",
            tool_calls=[
                {"name": "search_anime", "args": {"query": "test"}, "id": "call_456"}
            ],
        )

        condition_result = tools_condition({"messages": [ai_message]})

        # Should route to tools node
        assert condition_result == "tools"


class TestEnhancedSearchIntentParameters:
    """Test enhanced SearchIntent parameters in LangChain tool schemas."""

    def test_search_anime_schema_has_search_intent_parameters(self):
        """Test that SearchAnimeInput schema has all SearchIntent parameters."""
        mock_search = Mock()
        mcp_tools = {"search_anime": mock_search}

        tools = create_anime_langchain_tools(mcp_tools)
        search_tool = next(t for t in tools if t.name == "search_anime")

        # Verify schema has enhanced parameters
        schema_fields = search_tool.args_schema.model_fields

        # Basic parameters should still exist
        assert "query" in schema_fields
        assert "limit" in schema_fields

        # Enhanced SearchIntent parameters should be present
        assert "genres" in schema_fields
        assert "year_range" in schema_fields
        assert "anime_types" in schema_fields
        assert "studios" in schema_fields
        assert "exclusions" in schema_fields
        assert "mood_keywords" in schema_fields

    # Note: recommend_anime functionality now handled by search_anime with SearchIntent parameters

    @pytest.mark.asyncio
    async def test_search_anime_tool_accepts_enhanced_parameters(self):
        """Test that search_anime tool accepts and passes enhanced parameters."""
        mock_search = AsyncMock(
            return_value=[{"anime_id": "test1", "title": "Enhanced Test"}]
        )
        mcp_tools = {"search_anime": mock_search}

        tools = create_anime_langchain_tools(mcp_tools)
        search_tool = next(t for t in tools if t.name == "search_anime")

        # Execute with enhanced parameters
        result = await search_tool.ainvoke(
            {
                "query": "mecha anime",
                "limit": 5,
                "genres": ["Action", "Mecha"],
                "year_range": [2020, 2023],
                "anime_types": ["TV"],
                "studios": ["Mappa"],
                "exclusions": ["Horror"],
                "mood_keywords": ["serious"],
            }
        )

        # Verify the underlying MCP function was called with all parameters
        mock_search.assert_called_once_with(
            query="mecha anime",
            limit=5,
            genres=["Action", "Mecha"],
            year_range=[2020, 2023],
            anime_types=["TV"],
            studios=["Mappa"],
            exclusions=["Horror"],
            mood_keywords=["serious"],
        )

        assert result[0]["title"] == "Enhanced Test"

    # Note: recommend_anime functionality migrated to search_anime

    @pytest.mark.asyncio
    async def test_search_anime_backward_compatibility(self):
        """Test that search_anime maintains backward compatibility."""
        mock_search = AsyncMock(
            return_value=[{"anime_id": "compat1", "title": "Compatible"}]
        )
        mcp_tools = {"search_anime": mock_search}

        tools = create_anime_langchain_tools(mcp_tools)
        search_tool = next(t for t in tools if t.name == "search_anime")

        # Execute with only basic parameters (backward compatibility)
        result = await search_tool.ainvoke({"query": "action anime", "limit": 10})

        # Should call with None values for enhanced parameters
        mock_search.assert_called_once_with(
            query="action anime",
            limit=10,
            genres=None,
            year_range=None,
            anime_types=None,
            studios=None,
            exclusions=None,
            mood_keywords=None,
        )

        assert result[0]["title"] == "Compatible"


class TestMigrationCompatibility:
    """Test compatibility between old and new implementations."""

    @pytest.mark.asyncio
    async def test_toolnode_produces_correct_results(self):
        """Test that ToolNode produces correct results."""
        # Mock MCP function
        mock_search = AsyncMock(
            return_value=[{"anime_id": "comp1", "title": "Compatibility Test"}]
        )

        mcp_tools = {"search_anime": mock_search}

        # Test ToolNode implementation
        new_tools = create_anime_langchain_tools(mcp_tools)
        search_tool = next(t for t in new_tools if t.name == "search_anime")
        result = await search_tool.ainvoke({"query": "test", "limit": 5})

        # Result should match expected format
        expected = [{"anime_id": "comp1", "title": "Compatibility Test"}]
        assert result == expected

        # Verify mock was called correctly (with enhanced parameters as None)
        mock_search.assert_called_once_with(
            query="test",
            limit=5,
            genres=None,
            year_range=None,
            anime_types=None,
            studios=None,
            exclusions=None,
            mood_keywords=None,
        )

    def test_tool_names_compatibility(self):
        """Test that tool names are as expected (recommend_anime removed)."""
        mock_tools = {
            "search_anime": Mock(),
            "get_anime_details": Mock(),
            "find_similar_anime": Mock(),
            "get_anime_stats": Mock(),
            # "recommend_anime": Mock(),  # Removed - functionality moved to search_anime
            "search_anime_by_image": Mock(),
            "find_visually_similar_anime": Mock(),
            "search_multimodal_anime": Mock(),
        }

        # Create LangChain tools
        tools = create_anime_langchain_tools(mock_tools)
        tool_names = {tool.name for tool in tools}

        # Should have all the expected tool names (7 instead of 8)
        expected_names = set(mock_tools.keys())
        assert tool_names == expected_names
        assert "recommend_anime" not in tool_names  # Confirm removal
        assert len(tool_names) == 7  # Confirm total count

    def test_parameter_schemas_compatibility(self):
        """Test that parameter schemas are compatible."""
        mock_search = Mock()
        mcp_tools = {"search_anime": mock_search}

        tools = create_anime_langchain_tools(mcp_tools)
        search_tool = next(t for t in tools if t.name == "search_anime")

        # Should accept the same parameters as before
        # This is more of a smoke test - the actual schema validation
        # would be tested through integration tests
        assert hasattr(search_tool, "args_schema")
        assert search_tool.args_schema is not None
