"""Integration tests for FastMCP client adapter workflow integration."""

from unittest.mock import AsyncMock, Mock, patch

import pytest


class TestFastMCPWorkflowIntegration:
    """Test FastMCP client adapter integration with workflow engine."""

    @pytest.fixture
    def mock_fastmcp_tools(self):
        """Mock FastMCP tools that would be discovered."""
        tools = {}
        tool_names = [
            "search_anime",
            "get_anime_details",
            "find_similar_anime",
            "get_anime_stats",
            # "recommend_anime",  # Removed - functionality moved to search_anime
            "search_anime_by_image",
            "find_visually_similar_anime",
            "search_multimodal_anime",
        ]

        for name in tool_names:
            mock_tool = Mock()
            mock_tool.name = name
            tools[name] = mock_tool

        return tools

    @pytest.mark.asyncio
    async def test_workflow_engine_initialization_with_fastmcp(
        self, mock_fastmcp_tools
    ):
        """Test that workflow engine initializes with FastMCP client adapter."""
        from src.api.workflow import get_workflow_engine

        with patch(
            "src.mcp.fastmcp_client_adapter.get_all_mcp_tools"
        ) as mock_get_tools:
            mock_get_tools.return_value = mock_fastmcp_tools

            engine = await get_workflow_engine()

            assert engine is not None
            assert engine.tools is not None
            assert len(engine.tools) == 7  # 7 tools after recommend_anime removal

            # Verify get_all_mcp_tools was called (FastMCP client)
            mock_get_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_engine_singleton_behavior(self, mock_fastmcp_tools):
        """Test that workflow engine is a singleton."""
        # Clear singleton
        import src.api.workflow
        from src.api.workflow import get_workflow_engine

        src.api.workflow._workflow_engine = None

        with patch(
            "src.mcp.fastmcp_client_adapter.get_all_mcp_tools"
        ) as mock_get_tools:
            mock_get_tools.return_value = mock_fastmcp_tools

            # First call should initialize
            engine1 = await get_workflow_engine()

            # Second call should return same instance
            engine2 = await get_workflow_engine()

            assert engine1 is engine2
            # FastMCP client should only be called once due to singleton
            mock_get_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_engine_tools_compatibility(self, mock_fastmcp_tools):
        """Test that FastMCP tools are compatible with workflow engine."""
        from src.api.workflow import get_workflow_engine

        with patch(
            "src.mcp.fastmcp_client_adapter.get_all_mcp_tools"
        ) as mock_get_tools:
            mock_get_tools.return_value = mock_fastmcp_tools

            engine = await get_workflow_engine()

            # Get workflow info to check tool compatibility
            info = engine.get_workflow_info()

            assert "tools" in info
            assert len(info["tools"]) == 8
            assert "search_anime" in info["tools"]
            assert "get_anime_details" in info["tools"]

    @pytest.mark.asyncio
    async def test_fastmcp_error_handling(self):
        """Test error handling when FastMCP client fails."""
        # Clear singleton first
        import src.api.workflow
        from src.api.workflow import get_workflow_engine

        src.api.workflow._workflow_engine = None

        with patch(
            "src.mcp.fastmcp_client_adapter.get_all_mcp_tools"
        ) as mock_get_tools:
            mock_get_tools.side_effect = Exception("FastMCP connection failed")

            with pytest.raises(Exception, match="FastMCP connection failed"):
                await get_workflow_engine()

    @pytest.mark.asyncio
    async def test_backward_compatibility_with_tool_format(self, mock_fastmcp_tools):
        """Test that FastMCP client returns tools in expected format."""
        from src.mcp.fastmcp_client_adapter import get_all_mcp_tools

        with patch(
            "src.mcp.fastmcp_client_adapter.create_fastmcp_adapter"
        ) as mock_create:
            mock_adapter = AsyncMock()
            mock_adapter.get_tools_dict.return_value = mock_fastmcp_tools

            # Setup context manager
            mock_adapter.__aenter__ = AsyncMock(return_value=mock_adapter)
            mock_adapter.__aexit__ = AsyncMock(return_value=None)
            mock_adapter.connected.return_value.__aenter__ = AsyncMock(
                return_value=mock_adapter
            )
            mock_adapter.connected.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_create.return_value = mock_adapter

            tools = await get_all_mcp_tools()

            # Should return dict mapping tool names to callables
            assert isinstance(tools, dict)
            assert len(tools) == 8
            assert "search_anime" in tools

            for tool_name, tool_func in tools.items():
                assert isinstance(tool_name, str)
                # FastMCP returns BaseTool instances which are callable
                assert hasattr(tool_func, "name")

    def test_workflow_health_check_includes_fastmcp_info(self, mock_fastmcp_tools):
        """Test that health check includes FastMCP client information."""
        # This would be tested in the actual API integration test
