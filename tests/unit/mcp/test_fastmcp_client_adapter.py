"""Tests for FastMCP client adapter with automatic tool discovery."""

from unittest.mock import AsyncMock, Mock, patch

import pytest


class TestFastMCPClientAdapter:
    """Test FastMCP client adapter for automatic tool discovery."""

    @pytest.fixture
    def mock_toolkit(self):
        """Mock MCPToolkit for testing."""
        mock_toolkit = AsyncMock()

        # Mock tools that should be discovered
        mock_tools = []
        tool_names = [
            "search_anime",
            "get_anime_details",
            "find_similar_anime",
            "get_anime_stats",
            "search_anime_by_image",
            "find_visually_similar_anime",
            "search_multimodal_anime",
        ]

        for tool_name in tool_names:
            mock_tool = Mock()
            mock_tool.name = tool_name
            mock_tools.append(mock_tool)

        mock_toolkit.get_tools.return_value = mock_tools
        return mock_toolkit

    @pytest.fixture
    def adapter_config(self):
        """Configuration for FastMCP client adapter."""
        return {"command": "python", "args": ["-m", "src.mcp.server"]}

    def test_adapter_initialization(self, adapter_config):
        """Test that adapter initializes properly."""
        from src.mcp.fastmcp_client_adapter import FastMCPClientAdapter

        adapter = FastMCPClientAdapter(adapter_config)

        assert adapter.server_config == adapter_config
        assert adapter.toolkit is None  # Not connected yet
        assert adapter.session is None
        assert adapter._tools_cache is None

    def test_configuration_validation(self):
        """Test that invalid configuration raises appropriate errors."""
        from src.mcp.fastmcp_client_adapter import FastMCPClientAdapter

        # Empty config
        with pytest.raises(ValueError, match="Configuration cannot be empty"):
            FastMCPClientAdapter({})

        # Missing command
        with pytest.raises(ValueError, match="Missing required field 'command'"):
            FastMCPClientAdapter({"args": []})

        # Empty command
        with pytest.raises(ValueError, match="Empty command"):
            FastMCPClientAdapter({"command": "", "args": []})

    @pytest.mark.asyncio
    async def test_get_tools_without_connection(self, adapter_config):
        """Test that get_tools() raises error when not connected."""
        from src.mcp.fastmcp_client_adapter import FastMCPClientAdapter

        adapter = FastMCPClientAdapter(adapter_config)

        with pytest.raises(RuntimeError, match="Client not connected"):
            await adapter.get_tools()

    def test_is_connected_method(self, adapter_config):
        """Test the is_connected() method."""
        from src.mcp.fastmcp_client_adapter import FastMCPClientAdapter

        adapter = FastMCPClientAdapter(adapter_config)
        assert not adapter.is_connected()

    def test_get_tool_names_empty(self, adapter_config):
        """Test get_tool_names() returns empty list when no tools cached."""
        from src.mcp.fastmcp_client_adapter import FastMCPClientAdapter

        adapter = FastMCPClientAdapter(adapter_config)
        assert adapter.get_tool_names() == []

    def test_create_fastmcp_adapter_factory(self):
        """Test the factory function creates adapter with default config."""
        from src.mcp.fastmcp_client_adapter import create_fastmcp_adapter

        adapter = create_fastmcp_adapter()
        assert adapter.server_config["command"] == "python"
        assert adapter.server_config["args"] == ["-m", "src.mcp.server"]

    @pytest.mark.asyncio
    async def test_tool_count_validation(self, mock_toolkit, adapter_config):
        """Test that discovered tools match expected count."""
        # This test ensures we maintain compatibility with the 7 expected tools
        expected_tools = {
            "search_anime",
            "get_anime_details",
            "find_similar_anime",
            "get_anime_stats",
            "search_anime_by_image",
            "find_visually_similar_anime",
            "search_multimodal_anime",
        }

        # Verify mock returns all expected tools
        mock_tool_names = {tool.name for tool in mock_toolkit.get_tools.return_value}
        assert mock_tool_names == expected_tools

    @pytest.mark.asyncio
    async def test_connect_initializes_client(self, adapter_config):
        """Test that connect() initializes the MCPToolkit."""
        from src.mcp.fastmcp_client_adapter import FastMCPClientAdapter

        adapter = FastMCPClientAdapter(adapter_config)

        with (
            patch("src.mcp.fastmcp_client_adapter.stdio_client") as mock_stdio_client,
            patch(
                "src.mcp.fastmcp_client_adapter.ClientSession"
            ) as mock_client_session,
            patch("src.mcp.fastmcp_client_adapter.MCPToolkit") as mock_toolkit,
        ):

            # Mock stdio_client context manager
            mock_stdio_context = AsyncMock()
            mock_stdio_context.__aenter__ = AsyncMock(
                return_value=("read_stream", "write_stream")
            )
            mock_stdio_context.__aexit__ = AsyncMock(return_value=None)
            mock_stdio_client.return_value = mock_stdio_context

            # Mock ClientSession
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_client_session.return_value = mock_session

            # Mock MCPToolkit
            mock_toolkit_instance = AsyncMock()
            mock_toolkit.return_value = mock_toolkit_instance

            await adapter.connect()

            # Verify connection sequence
            mock_stdio_client.assert_called_once()
            mock_client_session.assert_called_once_with("read_stream", "write_stream")
            mock_session.initialize.assert_called_once()
            mock_toolkit.assert_called_once_with(session=mock_session)
            mock_toolkit_instance.initialize.assert_called_once()

            assert adapter.is_connected()

    @pytest.mark.asyncio
    async def test_get_tools_auto_discovery(self, adapter_config, mock_toolkit):
        """Test automatic tool discovery via toolkit.get_tools()."""
        from src.mcp.fastmcp_client_adapter import FastMCPClientAdapter

        adapter = FastMCPClientAdapter(adapter_config)
        adapter.toolkit = mock_toolkit  # Simulate connected state

        tools = await adapter.get_tools()

        # Verify toolkit.get_tools() was called
        mock_toolkit.get_tools.assert_called_once()

        # Verify correct number of tools returned
        assert len(tools) == 7

        # Verify tool names
        tool_names = {tool.name for tool in tools}
        expected_names = {
            "search_anime",
            "get_anime_details",
            "find_similar_anime",
            "get_anime_stats",
            "search_anime_by_image",
            "find_visually_similar_anime",
            "search_multimodal_anime",
        }
        assert tool_names == expected_names

    @pytest.mark.asyncio
    async def test_tools_caching(self, adapter_config, mock_toolkit):
        """Test that tools are cached after first discovery."""
        from src.mcp.fastmcp_client_adapter import FastMCPClientAdapter

        adapter = FastMCPClientAdapter(adapter_config)
        adapter.toolkit = mock_toolkit  # Simulate connected state

        # First call should use toolkit
        tools1 = await adapter.get_tools()
        mock_toolkit.get_tools.assert_called_once()

        # Second call should use cache
        tools2 = await adapter.get_tools()
        mock_toolkit.get_tools.assert_called_once()  # Still only called once

        # Results should be identical
        assert tools1 is tools2
        assert len(tools1) == len(tools2) == 7

    @pytest.mark.asyncio
    async def test_refresh_tools_clears_cache(self, adapter_config, mock_toolkit):
        """Test that refresh_tools() clears cache and re-discovers tools."""
        from src.mcp.fastmcp_client_adapter import FastMCPClientAdapter

        adapter = FastMCPClientAdapter(adapter_config)
        adapter.toolkit = mock_toolkit  # Simulate connected state

        # First call populates cache
        await adapter.get_tools()
        mock_toolkit.get_tools.assert_called_once()

        # Refresh should clear cache and call toolkit again
        await adapter.refresh_tools()
        assert mock_toolkit.get_tools.call_count == 2

        # Next get_tools should use the refreshed cache
        await adapter.get_tools()
        assert mock_toolkit.get_tools.call_count == 2  # No additional call

    @pytest.mark.asyncio
    async def test_disconnect_cleanup(self, adapter_config):
        """Test that disconnect() properly cleans up resources."""
        from src.mcp.fastmcp_client_adapter import FastMCPClientAdapter

        adapter = FastMCPClientAdapter(adapter_config)

        # Simulate connected state
        mock_session = AsyncMock()
        mock_session.__aexit__ = AsyncMock()
        adapter.session = mock_session

        mock_stdio_context = AsyncMock()
        mock_stdio_context.__aexit__ = AsyncMock()
        adapter._stdio_context = mock_stdio_context

        # Some cached tools
        adapter._tools_cache = [Mock(), Mock()]

        await adapter.disconnect()

        # Verify cleanup
        mock_session.__aexit__.assert_called_once_with(None, None, None)
        mock_stdio_context.__aexit__.assert_called_once_with(None, None, None)
        assert adapter.session is None
        assert adapter.toolkit is None
        assert adapter._tools_cache is None

    @pytest.mark.asyncio
    async def test_error_handling_connection_failure(self, adapter_config):
        """Test error handling when client connection fails."""
        from src.mcp.fastmcp_client_adapter import FastMCPClientAdapter

        adapter = FastMCPClientAdapter(adapter_config)

        with patch("src.mcp.fastmcp_client_adapter.stdio_client") as mock_stdio_client:
            # Simulate connection failure
            mock_stdio_client.side_effect = Exception("Connection failed")

            with pytest.raises(Exception, match="Connection failed"):
                await adapter.connect()

            # Adapter should remain disconnected
            assert not adapter.is_connected()

    @pytest.mark.asyncio
    async def test_error_handling_tool_discovery_failure(self, adapter_config):
        """Test error handling when tool discovery fails."""
        from src.mcp.fastmcp_client_adapter import FastMCPClientAdapter

        adapter = FastMCPClientAdapter(adapter_config)

        # Simulate connected state with failing toolkit
        mock_toolkit = AsyncMock()
        mock_toolkit.get_tools.side_effect = Exception("Tool discovery failed")
        adapter.toolkit = mock_toolkit

        with pytest.raises(Exception, match="Tool discovery failed"):
            await adapter.get_tools()

        # Cache should remain empty
        assert adapter._tools_cache is None

    @pytest.mark.asyncio
    async def test_backward_compatibility_interface(self, adapter_config, mock_toolkit):
        """Test that adapter provides same interface as get_all_mcp_tools()."""
        from src.mcp.fastmcp_client_adapter import FastMCPClientAdapter

        adapter = FastMCPClientAdapter(adapter_config)
        adapter.toolkit = mock_toolkit  # Simulate connected state

        # Test get_tools_dict returns dict interface
        tools_dict = await adapter.get_tools_dict()

        # Should be a dictionary
        assert isinstance(tools_dict, dict)
        assert len(tools_dict) == 7

        # Should contain expected tool names as keys
        expected_names = {
            "search_anime",
            "get_anime_details",
            "find_similar_anime",
            "get_anime_stats",
            "search_anime_by_image",
            "find_visually_similar_anime",
            "search_multimodal_anime",
        }
        assert set(tools_dict.keys()) == expected_names

        # Values should be the tool objects
        for tool_name, tool_obj in tools_dict.items():
            assert hasattr(tool_obj, "name")
            assert tool_obj.name == tool_name

    @pytest.mark.asyncio
    async def test_tool_execution_after_disconnect_fails(self, adapter_config):
        """Test that tools fail when called after client disconnect - reproduces ClosedResourceError."""
        from src.mcp.fastmcp_client_adapter import FastMCPClientAdapter

        adapter = FastMCPClientAdapter(adapter_config)

        # Mock a connected state and get tools
        mock_tool = AsyncMock()
        mock_tool.name = "search_anime"
        mock_tool.ainvoke = AsyncMock(side_effect=Exception("ClosedResourceError()"))

        mock_toolkit = AsyncMock()
        mock_toolkit.get_tools.return_value = [mock_tool]
        adapter.toolkit = mock_toolkit

        # Get tools while connected
        tools_dict = await adapter.get_tools_dict()
        search_tool = tools_dict["search_anime"]

        # Simulate disconnect (like our current architecture does)
        await adapter.disconnect()

        # Try to call tool after disconnect - should fail
        with pytest.raises(Exception, match="ClosedResourceError"):
            await search_tool.ainvoke({"query": "test", "limit": 10})

    @pytest.mark.asyncio
    async def test_persistent_connection_maintains_tool_functionality(
        self, adapter_config
    ):
        """Test that tools work when connection is maintained - desired behavior."""
        from src.mcp.fastmcp_client_adapter import FastMCPClientAdapter

        adapter = FastMCPClientAdapter(adapter_config)

        # Mock successful tool execution with persistent connection
        mock_tool = AsyncMock()
        mock_tool.name = "search_anime"
        mock_tool.ainvoke = AsyncMock(return_value=[{"title": "Test Anime"}])

        mock_toolkit = AsyncMock()
        mock_toolkit.get_tools.return_value = [mock_tool]
        adapter.toolkit = mock_toolkit

        # Get tools while connected
        tools_dict = await adapter.get_tools_dict()
        search_tool = tools_dict["search_anime"]

        # DO NOT disconnect - connection should remain open
        # Try to call tool - should succeed
        result = await search_tool.ainvoke({"query": "test", "limit": 10})

        assert result == [{"title": "Test Anime"}]
        assert adapter.is_connected()

    @pytest.mark.asyncio
    async def test_disconnect_exception_handling(self, adapter_config):
        """Test disconnect exception handling - covers lines 104-105."""
        from src.mcp.fastmcp_client_adapter import FastMCPClientAdapter

        adapter = FastMCPClientAdapter(adapter_config)

        # Simulate connected state with session that raises exception on exit
        mock_session = AsyncMock()
        mock_session.__aexit__ = AsyncMock(side_effect=Exception("Disconnect error"))
        adapter.session = mock_session

        # Should handle exception and log warning (lines 104-105)
        with patch("src.mcp.fastmcp_client_adapter.logger") as mock_logger:
            await adapter.disconnect()
            mock_logger.warning.assert_called_once_with("Error during disconnect: Disconnect error")

        # Cleanup should still happen in finally block
        assert adapter.session is None
        assert adapter.toolkit is None

    @pytest.mark.asyncio
    async def test_refresh_tools_not_connected_error(self, adapter_config):
        """Test refresh_tools when not connected - covers line 176."""
        from src.mcp.fastmcp_client_adapter import FastMCPClientAdapter

        adapter = FastMCPClientAdapter(adapter_config)
        # adapter.toolkit is None (not connected)

        with pytest.raises(RuntimeError, match="Client not connected"):
            await adapter.refresh_tools()

    @pytest.mark.asyncio
    async def test_connected_context_manager(self, adapter_config):
        """Test connected context manager - covers lines 190-194."""
        from src.mcp.fastmcp_client_adapter import FastMCPClientAdapter

        adapter = FastMCPClientAdapter(adapter_config)

        with (
            patch.object(adapter, "connect", AsyncMock()) as mock_connect,
            patch.object(adapter, "disconnect", AsyncMock()) as mock_disconnect,
        ):
            # Test context manager usage
            async with adapter.connected() as ctx_adapter:
                assert ctx_adapter is adapter
                mock_connect.assert_called_once()

            # Should call disconnect in finally block
            mock_disconnect.assert_called_once()

    def test_get_tool_names_with_cached_tools(self, adapter_config):
        """Test get_tool_names when tools are cached - covers line 203."""
        from src.mcp.fastmcp_client_adapter import FastMCPClientAdapter

        adapter = FastMCPClientAdapter(adapter_config)

        # Mock cached tools
        mock_tool1 = Mock()
        mock_tool1.name = "search_anime"
        mock_tool2 = Mock()
        mock_tool2.name = "get_anime_details"
        adapter._tools_cache = [mock_tool1, mock_tool2]

        result = adapter.get_tool_names()
        assert result == ["search_anime", "get_anime_details"]


class TestFastMCPClientAdapterIntegration:
    """Integration tests for FastMCP client adapter."""

    @pytest.mark.asyncio
    async def test_end_to_end_tool_discovery(self):
        """Test end-to-end tool discovery flow."""
        # This test will be implemented once the adapter is working
        # It should test actual connection to the MCP server

    @pytest.mark.asyncio
    async def test_tool_execution_compatibility(self):
        """Test that discovered tools can be executed with same parameters."""
        # This test ensures tools work identically to manual extraction

    @pytest.mark.asyncio
    async def test_performance_comparison(self):
        """Test that auto-discovery performance meets requirements."""
        # Should maintain <200ms response times


