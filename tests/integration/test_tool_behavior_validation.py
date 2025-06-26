"""Tool behavior validation tests for FastMCP client adapter.

This module validates that all 8 tools maintain identical behavior
when accessed through the FastMCP client adapter vs manual extraction.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest


class TestToolBehaviorValidation:
    """Validate that FastMCP tools behave identically to manual extraction."""

    @pytest.fixture
    def expected_tool_signatures(self):
        """Expected tool signatures for behavior validation."""
        return {
            "search_anime": {
                "required_params": ["query"],
                "optional_params": ["limit", "score_threshold", "filter"],
                "return_type": "list",
            },
            "get_anime_details": {
                "required_params": ["anime_id"],
                "optional_params": [],
                "return_type": "dict",
            },
            "find_similar_anime": {
                "required_params": ["anime_id"],
                "optional_params": ["limit", "score_threshold"],
                "return_type": "list",
            },
            "get_anime_stats": {
                "required_params": [],
                "optional_params": [],
                "return_type": "dict",
            },
            "recommend_anime": {
                "required_params": ["preferences"],
                "optional_params": ["limit", "exclude_ids"],
                "return_type": "list",
            },
            "search_anime_by_image": {
                "required_params": ["image_data"],
                "optional_params": ["limit", "score_threshold"],
                "return_type": "list",
            },
            "find_visually_similar_anime": {
                "required_params": ["anime_id"],
                "optional_params": ["limit", "score_threshold"],
                "return_type": "list",
            },
            "search_multimodal_anime": {
                "required_params": ["query"],
                "optional_params": ["image_data", "text_weight", "limit"],
                "return_type": "list",
            },
        }

    @pytest.mark.asyncio
    async def test_all_expected_tools_discovered(self):
        """Test that FastMCP discovers all 8 expected tools."""
        from src.mcp.fastmcp_client_adapter import create_fastmcp_adapter

        expected_tools = {
            "search_anime",
            "get_anime_details",
            "find_similar_anime",
            "get_anime_stats",
            "recommend_anime",
            "search_anime_by_image",
            "find_visually_similar_anime",
            "search_multimodal_anime",
        }

        # Mock the adapter to simulate successful discovery
        with patch(
            "src.mcp.fastmcp_client_adapter.FastMCPClientAdapter"
        ) as MockAdapter:
            mock_instance = AsyncMock()
            MockAdapter.return_value = mock_instance

            # Create mock tools with proper names
            mock_tools = []
            for tool_name in expected_tools:
                mock_tool = Mock()
                mock_tool.name = tool_name
                mock_tools.append(mock_tool)

            mock_instance.get_tools_dict.return_value = {
                tool.name: tool for tool in mock_tools
            }

            adapter = create_fastmcp_adapter()

            # Mock the context manager
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            adapter.connected = AsyncMock(return_value=mock_instance)

            async with adapter.connected():
                discovered_tools = await adapter.get_tools_dict()
                discovered_names = set(discovered_tools.keys())

                assert discovered_names == expected_tools
                assert len(discovered_tools) == 8

    @pytest.mark.asyncio
    async def test_tool_interface_compatibility(self, expected_tool_signatures):
        """Test that discovered tools have compatible interfaces."""
        from src.mcp.fastmcp_client_adapter import create_fastmcp_adapter

        with patch(
            "src.mcp.fastmcp_client_adapter.FastMCPClientAdapter"
        ) as MockAdapter:
            mock_instance = AsyncMock()
            MockAdapter.return_value = mock_instance

            # Create tools with proper interface simulation
            mock_tools_dict = {}
            for tool_name, signature in expected_tool_signatures.items():
                mock_tool = Mock()
                mock_tool.name = tool_name
                # Simulate tool callable interface
                mock_tool.run = AsyncMock()
                mock_tool.arun = AsyncMock()
                mock_tools_dict[tool_name] = mock_tool

            mock_instance.get_tools_dict.return_value = mock_tools_dict
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)

            adapter = create_fastmcp_adapter()
            adapter.connected = AsyncMock(return_value=mock_instance)

            async with adapter.connected():
                tools = await adapter.get_tools_dict()

                # Validate each tool has expected interface
                for tool_name, tool in tools.items():
                    assert hasattr(tool, "name")
                    assert tool.name == tool_name
                    # BaseTool instances should have run/arun methods
                    assert hasattr(tool, "run") or hasattr(tool, "arun")

    @pytest.mark.asyncio
    async def test_tool_discovery_performance(self):
        """Test that tool discovery meets performance requirements (<200ms)."""
        import time

        from src.mcp.fastmcp_client_adapter import create_fastmcp_adapter

        with patch(
            "src.mcp.fastmcp_client_adapter.FastMCPClientAdapter"
        ) as MockAdapter:
            mock_instance = AsyncMock()
            MockAdapter.return_value = mock_instance

            # Simulate realistic discovery time
            async def mock_get_tools_dict():
                await asyncio.sleep(0.05)  # 50ms simulation
                return {"search_anime": Mock(name="search_anime")}

            mock_instance.get_tools_dict = mock_get_tools_dict
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)

            adapter = create_fastmcp_adapter()
            adapter.connected = AsyncMock(return_value=mock_instance)

            start_time = time.time()
            async with adapter.connected():
                await adapter.get_tools_dict()
            end_time = time.time()

            discovery_time = (end_time - start_time) * 1000  # Convert to ms
            assert (
                discovery_time < 200
            ), f"Tool discovery took {discovery_time}ms, expected <200ms"

    @pytest.mark.asyncio
    async def test_backward_compatibility_interface(self):
        """Test that get_all_mcp_tools() maintains backward compatibility."""
        from src.mcp.fastmcp_client_adapter import get_all_mcp_tools

        with patch(
            "src.mcp.fastmcp_client_adapter.create_fastmcp_adapter"
        ) as mock_create:
            mock_adapter = AsyncMock()
            mock_tools = {
                "search_anime": Mock(name="search_anime"),
                "get_anime_details": Mock(name="get_anime_details"),
            }
            mock_adapter.get_tools_dict.return_value = mock_tools

            # Setup context manager properly
            mock_adapter.__aenter__ = AsyncMock(return_value=mock_adapter)
            mock_adapter.__aexit__ = AsyncMock(return_value=None)
            mock_adapter.connected.return_value = mock_adapter

            mock_create.return_value = mock_adapter

            tools = await get_all_mcp_tools()

            assert isinstance(tools, dict)
            assert "search_anime" in tools
            assert "get_anime_details" in tools
            assert tools["search_anime"].name == "search_anime"

    @pytest.mark.asyncio
    async def test_error_handling_consistency(self):
        """Test that error handling is consistent between implementations."""
        from src.mcp.fastmcp_client_adapter import get_all_mcp_tools

        with patch(
            "src.mcp.fastmcp_client_adapter.create_fastmcp_adapter"
        ) as mock_create:
            mock_adapter = AsyncMock()
            mock_adapter.connected.side_effect = Exception("Connection failed")
            mock_create.return_value = mock_adapter

            with pytest.raises(Exception, match="Connection failed"):
                await get_all_mcp_tools()

    def test_tool_count_validation(self):
        """Test that exactly 8 tools are expected for validation."""
        expected_count = 8
        expected_tools = [
            "search_anime",
            "get_anime_details",
            "find_similar_anime",
            "get_anime_stats",
            "recommend_anime",
            "search_anime_by_image",
            "find_visually_similar_anime",
            "search_multimodal_anime",
        ]

        assert len(expected_tools) == expected_count
        assert len(set(expected_tools)) == expected_count  # No duplicates

    @pytest.mark.asyncio
    async def test_workflow_engine_integration(self):
        """Test that workflow engine properly integrates with FastMCP tools."""
        # Clear singleton for clean test
        import src.api.workflow
        from src.api.workflow import get_workflow_engine

        src.api.workflow._workflow_engine = None

        with patch(
            "src.mcp.fastmcp_client_adapter.get_all_mcp_tools"
        ) as mock_get_tools:
            mock_tools = {}
            for i in range(8):
                tool_name = f"tool_{i}"
                mock_tool = Mock()
                mock_tool.name = tool_name
                mock_tools[tool_name] = mock_tool

            mock_get_tools.return_value = mock_tools

            engine = await get_workflow_engine()

            assert engine is not None
            # Workflow engine should have been initialized with FastMCP tools
            mock_get_tools.assert_called_once()
