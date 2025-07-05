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

        # Mock search function (create a simple callable without .ainvoke method)
        async def mock_search(**kwargs):
            return [{"anime_id": "test1", "title": "Action Anime"}]

        # Wrap it to track calls
        mock_search = AsyncMock(side_effect=mock_search)
        # Remove ainvoke attribute to force direct call path
        if hasattr(mock_search, "ainvoke"):
            delattr(mock_search, "ainvoke")

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

        async def mock_multimodal(**kwargs):
            return [{"anime_id": "mm1", "title": "Multimodal Match", "score": 0.9}]

        mock_multimodal = AsyncMock(side_effect=mock_multimodal)
        if hasattr(mock_multimodal, "ainvoke"):
            delattr(mock_multimodal, "ainvoke")

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
        search_mock = AsyncMock(return_value=[{"anime_id": "t1", "title": "Test"}])
        details_mock = AsyncMock(return_value={"anime_id": "t1", "details": "test"})
        stats_mock = AsyncMock(return_value={"total": 100})

        # Remove ainvoke to force direct call path for testing
        if hasattr(search_mock, "ainvoke"):
            delattr(search_mock, "ainvoke")
        if hasattr(details_mock, "ainvoke"):
            delattr(details_mock, "ainvoke")
        if hasattr(stats_mock, "ainvoke"):
            delattr(stats_mock, "ainvoke")

        return {
            "search_anime": search_mock,
            "get_anime_details": details_mock,
            "get_anime_stats": stats_mock,
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

        async def mock_search(**kwargs):
            return [{"anime_id": "test1", "title": "Enhanced Test"}]

        mock_search = AsyncMock(side_effect=mock_search)
        if hasattr(mock_search, "ainvoke"):
            delattr(mock_search, "ainvoke")

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

        async def mock_search(**kwargs):
            return [{"anime_id": "compat1", "title": "Compatible"}]

        mock_search = AsyncMock(side_effect=mock_search)
        if hasattr(mock_search, "ainvoke"):
            delattr(mock_search, "ainvoke")

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
        async def mock_search(**kwargs):
            return [{"anime_id": "comp1", "title": "Compatibility Test"}]

        mock_search = AsyncMock(side_effect=mock_search)
        if hasattr(mock_search, "ainvoke"):
            delattr(mock_search, "ainvoke")

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


class TestExceptionHandling:
    """Test exception handling in tool validation and execution."""

    @pytest.mark.asyncio
    async def test_search_anime_exception_handling(self):
        """Test exception handling in search anime tool execution."""

        # Test by simulating an exception in the underlying function
        async def failing_search(**kwargs):
            raise Exception("Search function failed")

        mock_search = AsyncMock(side_effect=failing_search)
        if hasattr(mock_search, "ainvoke"):
            delattr(mock_search, "ainvoke")

        mcp_tools = {"search_anime": mock_search}
        tools = create_anime_langchain_tools(mcp_tools)
        search_tool = next(t for t in tools if t.name == "search_anime")

        # Test that exceptions during execution are properly raised
        with pytest.raises(Exception, match="Search function failed"):
            await search_tool.ainvoke({"query": "test", "limit": 5})

    @pytest.mark.asyncio
    async def test_get_anime_details_validation_error(self):
        """Test validation error handling in get_anime_details."""
        mock_details = AsyncMock()
        mcp_tools = {"get_anime_details": mock_details}
        tools = create_anime_langchain_tools(mcp_tools)
        details_tool = next(t for t in tools if t.name == "get_anime_details")

        # Test with missing anime_id
        with pytest.raises(Exception):
            await details_tool.ainvoke({})

    @pytest.mark.asyncio
    async def test_search_anime_by_image_validation_error(self):
        """Test validation error handling in search_anime_by_image."""
        mock_image_search = AsyncMock()
        mcp_tools = {"search_anime_by_image": mock_image_search}
        tools = create_anime_langchain_tools(mcp_tools)
        image_tool = next(t for t in tools if t.name == "search_anime_by_image")

        # Test with missing image_data
        with pytest.raises(Exception):
            await image_tool.ainvoke({"limit": 5})


class TestMCPToolsWithAinvokeMethod:
    """Test MCP tools that have .ainvoke method."""

    @pytest.mark.asyncio
    async def test_search_anime_with_ainvoke_method(self):
        """Test search_anime tool execution when MCP function has .ainvoke method."""
        mock_search = AsyncMock()
        mock_search.ainvoke = AsyncMock(
            return_value=[{"anime_id": "test", "title": "Test Anime"}]
        )

        mcp_tools = {"search_anime": mock_search}
        tools = create_anime_langchain_tools(mcp_tools)
        search_tool = next(t for t in tools if t.name == "search_anime")

        result = await search_tool.ainvoke({"query": "test", "limit": 5})

        # Should call .ainvoke method with model_dump() output
        mock_search.ainvoke.assert_called_once()
        assert result[0]["title"] == "Test Anime"

    @pytest.mark.asyncio
    async def test_get_anime_details_with_ainvoke_method(self):
        """Test get_anime_details with .ainvoke method."""
        mock_details = AsyncMock()
        mock_details.ainvoke = AsyncMock(
            return_value={"anime_id": "test", "title": "Test Details"}
        )

        mcp_tools = {"get_anime_details": mock_details}
        tools = create_anime_langchain_tools(mcp_tools)
        details_tool = next(t for t in tools if t.name == "get_anime_details")

        result = await details_tool.ainvoke({"anime_id": "test123"})

        mock_details.ainvoke.assert_called_once_with({"anime_id": "test123"})
        assert result["title"] == "Test Details"

    @pytest.mark.asyncio
    async def test_find_similar_anime_with_ainvoke_method(self):
        """Test find_similar_anime with .ainvoke method."""
        mock_similar = AsyncMock()
        mock_similar.ainvoke = AsyncMock(return_value=[{"anime_id": "similar1"}])

        mcp_tools = {"find_similar_anime": mock_similar}
        tools = create_anime_langchain_tools(mcp_tools)
        similar_tool = next(t for t in tools if t.name == "find_similar_anime")

        result = await similar_tool.ainvoke({"anime_id": "ref123", "limit": 8})

        mock_similar.ainvoke.assert_called_once_with({"anime_id": "ref123", "limit": 8})
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_anime_stats_with_ainvoke_method(self):
        """Test get_anime_stats with .ainvoke method."""
        mock_stats = AsyncMock()
        mock_stats.ainvoke = AsyncMock(return_value={"total_documents": 500})

        mcp_tools = {"get_anime_stats": mock_stats}
        tools = create_anime_langchain_tools(mcp_tools)
        stats_tool = next(t for t in tools if t.name == "get_anime_stats")

        result = await stats_tool.ainvoke({})

        mock_stats.ainvoke.assert_called_once_with({})
        assert result["total_documents"] == 500

    @pytest.mark.asyncio
    async def test_search_anime_by_image_with_ainvoke_method(self):
        """Test search_anime_by_image with .ainvoke method."""
        mock_image_search = AsyncMock()
        mock_image_search.ainvoke = AsyncMock(return_value=[{"anime_id": "img1"}])

        mcp_tools = {"search_anime_by_image": mock_image_search}
        tools = create_anime_langchain_tools(mcp_tools)
        image_tool = next(t for t in tools if t.name == "search_anime_by_image")

        result = await image_tool.ainvoke({"image_data": "base64data", "limit": 5})

        mock_image_search.ainvoke.assert_called_once_with(
            {"image_data": "base64data", "limit": 5}
        )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_find_visually_similar_anime_with_ainvoke_method(self):
        """Test find_visually_similar_anime with .ainvoke method."""
        mock_visual = AsyncMock()
        mock_visual.ainvoke = AsyncMock(return_value=[{"anime_id": "visual1"}])

        mcp_tools = {"find_visually_similar_anime": mock_visual}
        tools = create_anime_langchain_tools(mcp_tools)
        visual_tool = next(t for t in tools if t.name == "find_visually_similar_anime")

        result = await visual_tool.ainvoke({"anime_id": "ref456", "limit": 12})

        mock_visual.ainvoke.assert_called_once_with({"anime_id": "ref456", "limit": 12})
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_search_multimodal_anime_with_ainvoke_method(self):
        """Test search_multimodal_anime with .ainvoke method."""
        mock_multimodal = AsyncMock()
        mock_multimodal.ainvoke = AsyncMock(
            return_value=[{"anime_id": "multi1", "score": 0.85}]
        )

        mcp_tools = {"search_multimodal_anime": mock_multimodal}
        tools = create_anime_langchain_tools(mcp_tools)
        multimodal_tool = next(t for t in tools if t.name == "search_multimodal_anime")

        result = await multimodal_tool.ainvoke(
            {
                "query": "mecha robots",
                "image_data": "base64img",
                "text_weight": 0.8,
                "limit": 15,
            }
        )

        expected_params = {
            "query": "mecha robots",
            "image_data": "base64img",
            "text_weight": 0.8,
            "limit": 15,
        }
        mock_multimodal.ainvoke.assert_called_once_with(expected_params)
        assert result[0]["score"] == 0.85


class TestChatbotNodeComplexMessageHandling:
    """Test chatbot node complex message content processing."""

    @pytest.fixture
    def simple_workflow(self):
        """Create simple workflow for testing."""
        return AnimeToolNodeWorkflow({})

    @pytest.mark.asyncio
    async def test_chatbot_node_empty_messages(self, simple_workflow):
        """Test chatbot node with empty messages."""
        state = {"messages": []}
        result = await simple_workflow._chatbot_node(state)

        # Should return state unchanged
        assert result == state

    @pytest.mark.asyncio
    async def test_chatbot_node_list_content_processing(self, simple_workflow):
        """Test chatbot node handling of list content."""
        from langchain_core.messages import HumanMessage

        message_with_list = HumanMessage(
            content=[{"text": "find anime"}, "about mechs", {"text": " with robots"}]
        )

        state = {"messages": [message_with_list]}
        result = await simple_workflow._chatbot_node(state)

        # Should extract text and process correctly
        assert len(result["messages"]) > 1

    @pytest.mark.asyncio
    async def test_chatbot_node_non_string_content(self, simple_workflow):
        """Test chatbot node with non-string content conversion."""
        from langchain_core.messages import HumanMessage

        # Create a message that gets to the non-string content conversion part
        # We'll modify the message after creation to test the code path
        message = HumanMessage(content="initial content")
        # Simulate what would happen if content was not a string
        message.content = {"type": "request", "text": "find anime"}

        state = {"messages": [message]}
        result = await simple_workflow._chatbot_node(state)

        # Should convert to string and process
        assert len(result["messages"]) > 1

    @pytest.mark.asyncio
    async def test_chatbot_node_tool_message_processing(self, simple_workflow):
        """Test chatbot node processing ToolMessage results."""
        from langchain_core.messages import ToolMessage

        tool_message = ToolMessage(
            content=[{"title": "Test Anime", "synopsis": "A test anime"}],
            tool_call_id="call_123",
        )

        state = {"messages": [tool_message]}
        result = await simple_workflow._chatbot_node(state)

        # Should add AI summary message
        assert len(result["messages"]) == 2
        # Check that AI message was added
        from langchain_core.messages import AIMessage

        assert isinstance(result["messages"][1], AIMessage)

    @pytest.mark.asyncio
    async def test_chatbot_node_direct_response(self, simple_workflow):
        """Test chatbot node providing direct response without tools."""
        from langchain_core.messages import HumanMessage

        # Message that doesn't trigger anime search
        human_message = HumanMessage(content="Hello, how are you?")

        state = {"messages": [human_message]}
        result = await simple_workflow._chatbot_node(state)

        # Should add direct response
        assert len(result["messages"]) == 2
        from langchain_core.messages import AIMessage

        assert isinstance(result["messages"][1], AIMessage)
        assert "search for anime" in result["messages"][1].content


class TestSearchParameterExtraction:
    """Test search parameter extraction and anime search detection."""

    @pytest.fixture
    def workflow(self):
        """Create workflow for testing."""
        return AnimeToolNodeWorkflow({})

    def test_needs_anime_search_positive_cases(self, workflow):
        """Test _needs_anime_search method with positive cases."""
        assert workflow._needs_anime_search("find action anime") == True
        assert workflow._needs_anime_search("recommend comedy shows") == True
        assert workflow._needs_anime_search("search for romance") == True
        assert workflow._needs_anime_search("I want to watch anime") == True
        assert workflow._needs_anime_search("show me some sci-fi series") == True
        assert workflow._needs_anime_search("fantasy movies please") == True
        assert workflow._needs_anime_search("drama recommendations") == True

    def test_needs_anime_search_negative_cases(self, workflow):
        """Test _needs_anime_search method with negative cases."""
        assert workflow._needs_anime_search("hello world") == False
        assert workflow._needs_anime_search("what is your name") == False
        assert workflow._needs_anime_search("how does this work") == False
        assert workflow._needs_anime_search("thank you") == False

    def test_extract_search_parameters_basic(self, workflow):
        """Test basic parameter extraction."""
        params = workflow._extract_search_parameters("find some anime")
        assert params["query"] == "find some anime"
        assert params["limit"] == 10

    def test_extract_search_parameters_with_limit(self, workflow):
        """Test parameter extraction with limit parsing."""
        # Test limit extraction with "anime"
        params = workflow._extract_search_parameters("find 15 anime")
        assert params["limit"] == 15
        assert params["query"] == "find 15 anime"

        # Test with "results"
        params = workflow._extract_search_parameters("show me 8 results")
        assert params["limit"] == 8

        # Test with "shows"
        params = workflow._extract_search_parameters("find 25 shows")
        assert params["limit"] == 25

    def test_extract_search_parameters_limit_capping(self, workflow):
        """Test limit capping at 50."""
        params = workflow._extract_search_parameters("find 100 anime")
        assert params["limit"] == 50  # Should be capped

    def test_extract_search_parameters_no_limit(self, workflow):
        """Test parameter extraction without explicit limit."""
        params = workflow._extract_search_parameters("find action anime")
        assert params["limit"] == 10  # Default value


class TestToolResultResponseCreation:
    """Test response creation from different tool result types."""

    @pytest.fixture
    def workflow(self):
        """Create workflow for testing."""
        return AnimeToolNodeWorkflow({})

    def test_create_response_single_result(self, workflow):
        """Test response creation from single result."""
        single_result = [
            {"title": "Test Anime", "synopsis": "A test anime about heroes"}
        ]
        response = workflow._create_response_from_tool_result(single_result, "call_1")

        assert "Test Anime" in response
        assert "A test anime about heroes" in response

    def test_create_response_multiple_results(self, workflow):
        """Test response creation from multiple results."""
        multiple_results = [
            {"title": "Anime 1"},
            {"title": "Anime 2"},
            {"title": "Anime 3"},
            {"title": "Anime 4"},
        ]
        response = workflow._create_response_from_tool_result(
            multiple_results, "call_2"
        )

        assert "4 anime" in response
        assert "Anime 1" in response
        assert "Anime 2" in response
        assert "Anime 3" in response

    def test_create_response_dict_result(self, workflow):
        """Test response creation from dict result."""
        dict_result = {"title": "Single Anime", "synopsis": "Description of the anime"}
        response = workflow._create_response_from_tool_result(dict_result, "call_3")

        assert "Single Anime" in response
        assert "Description of the anime" in response

    def test_create_response_empty_result(self, workflow):
        """Test response creation from empty/invalid result."""
        response = workflow._create_response_from_tool_result([], "call_4")
        assert "couldn't find" in response

        response = workflow._create_response_from_tool_result(None, "call_5")
        assert "couldn't find" in response

        response = workflow._create_response_from_tool_result("invalid", "call_6")
        assert "couldn't find" in response


class TestConversationProcessing:
    """Test conversation processing with thread IDs and error handling."""

    @pytest.fixture
    def mock_mcp_tools(self):
        """Mock MCP tools for testing."""
        search_mock = AsyncMock(return_value=[{"anime_id": "t1", "title": "Test"}])
        if hasattr(search_mock, "ainvoke"):
            delattr(search_mock, "ainvoke")

        return {"search_anime": search_mock}

    @pytest.fixture
    def workflow(self, mock_mcp_tools):
        """Create workflow with mocked tools."""
        return AnimeToolNodeWorkflow(mock_mcp_tools)

    @pytest.mark.asyncio
    async def test_process_conversation_basic(self, workflow):
        """Test basic conversation processing."""
        result = await workflow.process_conversation(
            session_id="test_session", message="find anime"
        )

        assert result["session_id"] == "test_session"
        assert len(result["messages"]) > 0

    @pytest.mark.asyncio
    async def test_process_conversation_with_thread_id(self, workflow):
        """Test conversation processing with custom thread ID."""
        result = await workflow.process_conversation(
            session_id="test_session", message="find anime", thread_id="custom_thread"
        )

        assert result["session_id"] == "test_session"
        assert len(result["messages"]) > 0

    @pytest.mark.asyncio
    async def test_process_conversation_error_handling(self):
        """Test error handling in process_conversation."""
        # Mock a workflow that raises an exception in graph execution
        failing_mock = AsyncMock(side_effect=Exception("Graph processing error"))
        workflow = AnimeToolNodeWorkflow({"search_anime": failing_mock})

        # Mock the graph to raise an exception
        workflow.graph.ainvoke = AsyncMock(side_effect=Exception("Processing error"))

        with pytest.raises(Exception, match="Processing error"):
            await workflow.process_conversation("session", "test message")


class TestFactoryFunction:
    """Test factory function for creating workflow from MCP tools."""

    def test_create_toolnode_workflow_from_mcp_tools(self):
        """Test factory function for creating workflow from MCP tools."""
        from src.langgraph.langchain_tools import (
            create_toolnode_workflow_from_mcp_tools,
        )

        mock_tools = {"search_anime": AsyncMock(), "get_anime_stats": AsyncMock()}
        workflow = create_toolnode_workflow_from_mcp_tools(mock_tools)

        assert isinstance(workflow, AnimeToolNodeWorkflow)
        assert len(workflow.tools) == 2
        assert workflow.mcp_tools == mock_tools


class TestInputSchemas:
    """Test input schema classes and validation."""

    def test_empty_input_schemas(self):
        """Test input schemas with no parameters."""
        from src.langgraph.langchain_tools import GetAnimeStatsInput

        # Should create successfully with no parameters
        schema = GetAnimeStatsInput()
        assert schema is not None

    def test_input_schema_validation(self):
        """Test input schema field validation."""
        from src.langgraph.langchain_tools import (
            FindSimilarAnimeInput,
            FindVisuallySimilarAnimeInput,
            GetAnimeDetailsInput,
            SearchAnimeByImageInput,
            SearchAnimeInput,
            SearchMultimodalAnimeInput,
        )

        # Test SearchAnimeInput
        search_input = SearchAnimeInput(query="test", limit=5)
        assert search_input.query == "test"
        assert search_input.limit == 5

        # Test GetAnimeDetailsInput
        details_input = GetAnimeDetailsInput(anime_id="test123")
        assert details_input.anime_id == "test123"

        # Test FindSimilarAnimeInput
        similar_input = FindSimilarAnimeInput(anime_id="ref456", limit=8)
        assert similar_input.anime_id == "ref456"
        assert similar_input.limit == 8

        # Test SearchAnimeByImageInput
        image_input = SearchAnimeByImageInput(image_data="base64data", limit=12)
        assert image_input.image_data == "base64data"
        assert image_input.limit == 12

        # Test FindVisuallySimilarAnimeInput
        visual_input = FindVisuallySimilarAnimeInput(anime_id="vis789")
        assert visual_input.anime_id == "vis789"
        assert visual_input.limit == 10  # Default value

        # Test SearchMultimodalAnimeInput
        multimodal_input = SearchMultimodalAnimeInput(
            query="test query", image_data="imgdata", text_weight=0.6, limit=20
        )
        assert multimodal_input.query == "test query"
        assert multimodal_input.image_data == "imgdata"
        assert multimodal_input.text_weight == 0.6
        assert multimodal_input.limit == 20


class TestDirectFunctionCallPaths:
    """Test the else branches for direct function calls (without .ainvoke)."""

    @pytest.mark.asyncio
    async def test_get_anime_details_direct_call(self):
        """Test get_anime_details direct function call path (line 167)."""

        async def mock_details(**kwargs):
            return {"anime_id": kwargs["anime_id"], "title": "Direct Call Test"}

        mock_details_func = AsyncMock(side_effect=mock_details)
        # Ensure it doesn't have .ainvoke to force direct call path
        if hasattr(mock_details_func, "ainvoke"):
            delattr(mock_details_func, "ainvoke")

        mcp_tools = {"get_anime_details": mock_details_func}
        tools = create_anime_langchain_tools(mcp_tools)
        details_tool = next(t for t in tools if t.name == "get_anime_details")

        result = await details_tool.ainvoke({"anime_id": "test123"})

        # Should call direct function
        mock_details_func.assert_called_once_with(anime_id="test123")
        assert result["title"] == "Direct Call Test"

    @pytest.mark.asyncio
    async def test_find_similar_anime_direct_call(self):
        """Test find_similar_anime direct function call path (line 182)."""

        async def mock_similar(**kwargs):
            return [{"anime_id": "similar_" + kwargs["anime_id"]}]

        mock_similar_func = AsyncMock(side_effect=mock_similar)
        if hasattr(mock_similar_func, "ainvoke"):
            delattr(mock_similar_func, "ainvoke")

        mcp_tools = {"find_similar_anime": mock_similar_func}
        tools = create_anime_langchain_tools(mcp_tools)
        similar_tool = next(t for t in tools if t.name == "find_similar_anime")

        result = await similar_tool.ainvoke({"anime_id": "ref456", "limit": 10})

        mock_similar_func.assert_called_once_with(anime_id="ref456", limit=10)
        assert result[0]["anime_id"] == "similar_ref456"

    @pytest.mark.asyncio
    async def test_get_anime_stats_direct_call(self):
        """Test get_anime_stats direct function call path (line 196)."""

        async def mock_stats():
            return {"total_documents": 12345}

        mock_stats_func = AsyncMock(side_effect=mock_stats)
        if hasattr(mock_stats_func, "ainvoke"):
            delattr(mock_stats_func, "ainvoke")

        mcp_tools = {"get_anime_stats": mock_stats_func}
        tools = create_anime_langchain_tools(mcp_tools)
        stats_tool = next(t for t in tools if t.name == "get_anime_stats")

        result = await stats_tool.ainvoke({})

        # Should call function with no arguments
        mock_stats_func.assert_called_once_with()
        assert result["total_documents"] == 12345

    @pytest.mark.asyncio
    async def test_search_anime_by_image_direct_call(self):
        """Test search_anime_by_image direct function call path (line 213)."""

        async def mock_image_search(**kwargs):
            return [{"anime_id": "img_result", "image_data": kwargs["image_data"]}]

        mock_image_func = AsyncMock(side_effect=mock_image_search)
        if hasattr(mock_image_func, "ainvoke"):
            delattr(mock_image_func, "ainvoke")

        mcp_tools = {"search_anime_by_image": mock_image_func}
        tools = create_anime_langchain_tools(mcp_tools)
        image_tool = next(t for t in tools if t.name == "search_anime_by_image")

        result = await image_tool.ainvoke({"image_data": "base64test", "limit": 15})

        mock_image_func.assert_called_once_with(image_data="base64test", limit=15)
        assert result[0]["image_data"] == "base64test"

    @pytest.mark.asyncio
    async def test_find_visually_similar_anime_direct_call(self):
        """Test find_visually_similar_anime direct function call path (line 230)."""

        async def mock_visual_similar(**kwargs):
            return [{"anime_id": "visual_" + kwargs["anime_id"], "similarity": 0.9}]

        mock_visual_func = AsyncMock(side_effect=mock_visual_similar)
        if hasattr(mock_visual_func, "ainvoke"):
            delattr(mock_visual_func, "ainvoke")

        mcp_tools = {"find_visually_similar_anime": mock_visual_func}
        tools = create_anime_langchain_tools(mcp_tools)
        visual_tool = next(t for t in tools if t.name == "find_visually_similar_anime")

        result = await visual_tool.ainvoke({"anime_id": "ref789", "limit": 20})

        mock_visual_func.assert_called_once_with(anime_id="ref789", limit=20)
        assert result[0]["anime_id"] == "visual_ref789"
        assert result[0]["similarity"] == 0.9


class TestChatbotNodeDefaultReturn:
    """Test the default return path in chatbot node (line 385)."""

    @pytest.fixture
    def simple_workflow(self):
        """Create simple workflow for testing."""
        return AnimeToolNodeWorkflow({})

    @pytest.mark.asyncio
    async def test_chatbot_node_default_return_path(self, simple_workflow):
        """Test chatbot node default return path (line 385)."""
        from langchain_core.messages import AIMessage

        # Create an AI message that doesn't match any specific handling paths
        ai_message = AIMessage(content="This is a response from AI")

        state = {"messages": [ai_message]}
        result = await simple_workflow._chatbot_node(state)

        # Should return state unchanged (default path)
        assert result == state
