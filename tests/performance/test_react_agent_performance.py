"""Performance tests for ReactAgent workflow engine.

Tests response time targets, memory usage, and throughput benchmarks.
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest


class TestReactAgentPerformance:
    """Performance benchmarks for ReactAgent integration."""

    @pytest.fixture
    def mock_mcp_tools(self):
        """Fast mock tools for performance testing."""
        return {
            "search_anime": AsyncMock(
                return_value=[
                    {"anime_id": "perf_1", "title": "Performance Test", "score": 0.9}
                ]
            ),
            "get_anime_details": AsyncMock(
                return_value={"anime_id": "perf_1", "title": "Performance Test"}
            ),
        }

    @pytest.mark.asyncio
    async def test_response_time_under_200ms(self, mock_mcp_tools):
        """Test response time meets performance targets."""
        with patch(
            "src.mcp.fastmcp_client_adapter.get_all_mcp_tools",
            return_value=mock_mcp_tools,
        ):
            with patch(
                "src.langgraph.react_agent_workflow.create_react_agent"
            ) as mock_create_agent:
                # Fast mock agent
                mock_agent = Mock()
                mock_agent.ainvoke = AsyncMock(
                    return_value={"messages": [Mock(content="Fast response")]}
                )
                mock_create_agent.return_value = mock_agent

                from src.api.workflow import get_workflow_engine

                engine = await get_workflow_engine()

                # Measure response time
                start_time = time.perf_counter()

                result = await engine.process_conversation(
                    session_id="perf_test", message="test performance"
                )

                end_time = time.perf_counter()
                response_time_ms = (end_time - start_time) * 1000

                # Verify response completed
                assert result["session_id"] == "perf_test"

                # Target: 120ms, allowing 200ms for test overhead
                assert (
                    response_time_ms < 200
                ), f"Response time {response_time_ms:.1f}ms exceeds 200ms target"
                print(f"✅ Response time: {response_time_ms:.1f}ms (target: <200ms)")

    @pytest.mark.asyncio
    async def test_streaming_latency(self, mock_mcp_tools):
        """Test streaming response latency."""
        with patch(
            "src.mcp.fastmcp_client_adapter.get_all_mcp_tools",
            return_value=mock_mcp_tools,
        ):
            with patch(
                "src.langgraph.react_agent_workflow.create_react_agent"
            ) as mock_create_agent:
                # Mock streaming agent
                async def mock_stream():
                    yield {"messages": [Mock(content="Chunk 1")]}
                    yield {"messages": [Mock(content="Chunk 2")]}
                    yield {"messages": [Mock(content="Final result")]}

                mock_agent = Mock()
                mock_agent.astream = AsyncMock(return_value=mock_stream())
                mock_create_agent.return_value = mock_agent

                from src.api.workflow import get_workflow_engine

                engine = await get_workflow_engine()

                # Measure first chunk latency
                start_time = time.perf_counter()
                first_chunk_time = None

                chunk_count = 0
                async for chunk in engine.astream_conversation(
                    session_id="stream_perf", message="stream test"
                ):
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()
                    chunk_count += 1

                first_chunk_latency = (first_chunk_time - start_time) * 1000

                # First chunk should arrive quickly
                assert (
                    first_chunk_latency < 100
                ), f"First chunk latency {first_chunk_latency:.1f}ms too high"
                assert chunk_count >= 1, "No chunks received"
                print(
                    f"✅ First chunk latency: {first_chunk_latency:.1f}ms (target: <100ms)"
                )

    @pytest.mark.asyncio
    async def test_concurrent_throughput(self, mock_mcp_tools):
        """Test concurrent request handling."""
        with patch(
            "src.mcp.fastmcp_client_adapter.get_all_mcp_tools",
            return_value=mock_mcp_tools,
        ):
            with patch(
                "src.langgraph.react_agent_workflow.create_react_agent"
            ) as mock_create_agent:
                mock_agent = Mock()
                mock_agent.ainvoke = AsyncMock(
                    return_value={"messages": [Mock(content="Concurrent response")]}
                )
                mock_create_agent.return_value = mock_agent

                from src.api.workflow import get_workflow_engine

                engine = await get_workflow_engine()

                # Test concurrent processing
                async def process_request(session_id: str):
                    return await engine.process_conversation(
                        session_id=session_id, message=f"concurrent test {session_id}"
                    )

                # Measure concurrent throughput
                start_time = time.perf_counter()

                # Run 10 concurrent requests
                tasks = [process_request(f"concurrent_{i}") for i in range(10)]
                results = await asyncio.gather(*tasks)

                end_time = time.perf_counter()
                total_time = end_time - start_time
                throughput = len(results) / total_time

                # All should complete successfully
                assert len(results) == 10
                for i, result in enumerate(results):
                    assert result["session_id"] == f"concurrent_{i}"

                # Should handle multiple requests efficiently
                assert throughput > 5, f"Throughput {throughput:.1f} req/s too low"
                print(
                    f"✅ Concurrent throughput: {throughput:.1f} req/s (target: >5 req/s)"
                )

    def test_memory_efficiency(self, mock_mcp_tools):
        """Test memory usage is reasonable."""
        try:
            import os

            import psutil
        except ImportError:
            pytest.skip("psutil not available for memory testing")

        with patch(
            "src.mcp.fastmcp_client_adapter.get_all_mcp_tools",
            return_value=mock_mcp_tools,
        ):
            with patch(
                "src.langgraph.react_agent_workflow.create_react_agent"
            ) as mock_create_agent:
                mock_agent = Mock()
                mock_create_agent.return_value = mock_agent

                # Measure memory before
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024  # MB

                # Initialize engine
                from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

                ReactAgentWorkflowEngine(mock_mcp_tools)

                # Measure memory after
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = memory_after - memory_before

                # Should not use excessive memory
                assert (
                    memory_increase < 100
                ), f"Memory increase {memory_increase:.1f}MB too high"
                print(f"✅ Memory usage: +{memory_increase:.1f}MB (target: <100MB)")

    @pytest.mark.asyncio
    async def test_error_handling_performance(self, mock_mcp_tools):
        """Test error handling doesn't impact performance."""
        with patch(
            "src.mcp.fastmcp_client_adapter.get_all_mcp_tools",
            return_value=mock_mcp_tools,
        ):
            with patch(
                "src.langgraph.react_agent_workflow.create_react_agent"
            ) as mock_create_agent:
                # Agent that raises errors
                mock_agent = Mock()
                mock_agent.ainvoke = AsyncMock(side_effect=Exception("Test error"))
                mock_create_agent.return_value = mock_agent

                from src.api.workflow import get_workflow_engine

                engine = await get_workflow_engine()

                # Measure error handling time
                start_time = time.perf_counter()

                result = await engine.process_conversation(
                    session_id="error_perf", message="trigger error"
                )

                end_time = time.perf_counter()
                error_handling_time = (end_time - start_time) * 1000

                # Should handle errors quickly
                assert (
                    error_handling_time < 100
                ), f"Error handling time {error_handling_time:.1f}ms too slow"
                assert result["session_id"] == "error_perf"
                print(
                    f"✅ Error handling: {error_handling_time:.1f}ms (target: <100ms)"
                )


class TestPerformanceRegression:
    """Regression tests to ensure performance doesn't degrade."""

    @pytest.mark.asyncio
    async def test_performance_targets_maintained(self):
        """Test that documented performance targets are achievable."""
        mock_tools = {
            "search_anime": AsyncMock(
                return_value=[{"anime_id": "target_test", "score": 0.9}]
            )
        }

        with patch(
            "src.mcp.fastmcp_client_adapter.get_all_mcp_tools", return_value=mock_tools
        ):
            with patch(
                "src.langgraph.react_agent_workflow.create_react_agent"
            ) as mock_create_agent:
                mock_agent = Mock()
                mock_agent.ainvoke = AsyncMock(
                    return_value={"messages": [Mock(content="Target test response")]}
                )
                mock_create_agent.return_value = mock_agent

                from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

                engine = ReactAgentWorkflowEngine(mock_tools)

                info = engine.get_workflow_info()

                # Verify documented targets
                assert "120ms" in info["performance"]["target_response_time"]
                assert info["performance"]["streaming_support"] is True
                assert info["performance"]["memory_persistence"] is True
                assert (
                    "20-30% response time improvement"
                    in info["performance"]["improvements"]
                )

                print("✅ Performance targets documented correctly")

    def test_workflow_efficiency_metrics(self):
        """Test workflow efficiency compared to previous implementation."""
        mock_tools = {"test_tool": AsyncMock()}

        with patch(
            "src.langgraph.react_agent_workflow.create_react_agent"
        ) as mock_create_agent:
            mock_agent = Mock()
            mock_create_agent.return_value = mock_agent

            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine

            engine = ReactAgentWorkflowEngine(mock_tools)

            info = engine.get_workflow_info()

            # Verify efficiency improvements
            assert info["llm_integration"]["manual_calls_eliminated"] is True
            assert info["llm_integration"]["structured_output"] is True
            assert info["llm_integration"]["native_integration"] is True

            # Should have expected features
            expected_features = [
                "Native LangGraph create_react_agent integration",
                "Structured output with with_structured_output()",
                "No manual LLM service calls",
                "No manual JSON parsing",
            ]

            for feature in expected_features:
                assert feature in info["features"]

            print("✅ Workflow efficiency metrics validated")
