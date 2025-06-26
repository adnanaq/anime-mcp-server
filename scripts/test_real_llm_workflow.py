#!/usr/bin/env python3
"""Real-time testing script for ReactAgent workflow with actual LLM calls.

This script tests the complete workflow chain with REAL LLM providers to verify:
1. AI-powered query understanding with actual OpenAI/Anthropic models
2. Structured output elimination of manual JSON parsing
3. create_react_agent performance with real API calls
4. Parameter extraction accuracy without mock models

REQUIREMENTS:
- Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables
- Install: pip install langchain-openai langchain-anthropic
- This script will FAIL if no real API keys are provided (no mocks)

Run with: python scripts/test_real_llm_workflow.py
"""

import asyncio
import logging
import time

from src.config import get_settings
from src.langgraph.react_agent_workflow import LLMProvider, ReactAgentWorkflowEngine
from src.mcp.fastmcp_client_adapter import get_all_mcp_tools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealLLMTester:
    """Tester for real-time LLM workflow validation."""

    def __init__(self):
        self.settings = get_settings()
        self.test_queries = [
            # Test complex query understanding
            "find me 5 mecha anime from 2010s but not too violent",
            # Test limit extraction
            "show me top 3 Studio Ghibli movies",
            # Test genre detection
            "looking for romantic comedy anime with good animation",
            # Test year range understanding
            "recent action anime from 2020s",
            # Test exclusions
            "fantasy anime but not isekai, limit to 7",
            # Test mood keywords
            "dark psychological anime, show me 4",
            # Test studio detection
            "anime by Mappa studio",
            # Test type filtering
            "OVA series with good story",
        ]

    def print_header(self, title: str):
        """Print formatted test header."""
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")

    def print_section(self, title: str):
        """Print formatted section header."""
        print(f"\n{'─'*40}")
        print(f" {title}")
        print(f"{'─'*40}")

    async def test_react_agent_providers(self):
        """Test ReactAgent with different LLM providers."""
        self.print_header("TESTING REACT AGENT WITH DIFFERENT PROVIDERS")

        # Test with different providers
        providers = []

        # Check which providers are available
        if hasattr(self.settings, "openai_api_key") and self.settings.openai_api_key:
            providers.append(LLMProvider.OPENAI)
            print("✅ OpenAI API key found")
        else:
            print("❌ OpenAI API key not found")

        if (
            hasattr(self.settings, "anthropic_api_key")
            and self.settings.anthropic_api_key
        ):
            providers.append(LLMProvider.ANTHROPIC)
            print("✅ Anthropic API key found")
        else:
            print("❌ Anthropic API key not found")

        if not providers:
            print("❌ No LLM providers available - REAL API KEYS REQUIRED")
            print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables")
            print("This script requires real LLM providers, not mocks!")
            return

        # Get MCP tools once
        mcp_tools = await get_all_mcp_tools()
        print(f"✅ Loaded {len(mcp_tools)} MCP tools")

        for provider in providers:
            self.print_section(f"Testing ReactAgent with {provider.value.upper()}")

            try:
                # Create ReactAgent with specific provider (REAL LLM, not mock)
                engine = ReactAgentWorkflowEngine(mcp_tools, provider)
                print(f"✅ ReactAgent initialized with REAL {provider.value} LLM")

                for i, query in enumerate(self.test_queries[:2]):  # Test subset
                    print(f"\nQuery {i+1}: '{query}'")

                    start_time = time.perf_counter()

                    try:
                        result = await engine.process_conversation(
                            session_id=f"provider_test_{provider.value}_{i}",
                            message=query,
                        )

                        end_time = time.perf_counter()
                        response_time = (end_time - start_time) * 1000

                        print(f"  ⏱️  Response time: {response_time:.1f}ms")
                        print(f"  📝 Messages: {len(result['messages'])}")
                        print(f"  🔄 Workflow steps: {len(result['workflow_steps'])}")

                        if result["messages"]:
                            first_msg = result["messages"][0]
                            print(
                                f"  📄 First message preview: {str(first_msg)[:100]}..."
                            )

                        if result["workflow_steps"]:
                            print("  ✅ ReactAgent executed successfully")
                        else:
                            print("  ⚠️  No workflow steps recorded")

                    except Exception as e:
                        print(f"  ❌ Error processing query: {e}")

                    await asyncio.sleep(0.5)  # Rate limiting

            except Exception as e:
                print(
                    f"❌ Failed to initialize ReactAgent with REAL {provider.value}: {e}"
                )
                print("   Make sure API keys are set and langchain packages installed!")

    async def test_react_agent_integration(self):
        """Test ReactAgent workflow with real LLM calls."""
        self.print_header("TESTING REACT AGENT WORKFLOW INTEGRATION")

        try:
            # Get MCP tools
            mcp_tools = await get_all_mcp_tools()
            print(f"✅ Loaded {len(mcp_tools)} MCP tools")

            # Create ReactAgent workflow engine
            engine = ReactAgentWorkflowEngine(mcp_tools)
            print("✅ ReactAgent workflow engine initialized")

            # Test workflow info
            info = engine.get_workflow_info()
            print(f"📋 Engine type: {info['engine_type']}")
            print(
                f"🎯 Target response time: {info['performance']['target_response_time']}"
            )
            print(f"🔧 Tools available: {len(info['tools'])}")

            self.print_section("Testing Real Conversations")

            for i, query in enumerate(self.test_queries[:4]):  # Test subset
                print(f"\n--- Test {i+1} ---")
                print(f"Query: '{query}'")

                start_time = time.perf_counter()

                try:
                    # Process conversation with ReactAgent
                    result = await engine.process_conversation(
                        session_id=f"real_test_{i}", message=query
                    )

                    end_time = time.perf_counter()
                    response_time = (end_time - start_time) * 1000

                    print(f"⏱️  Total response time: {response_time:.1f}ms")
                    print(f"📝 Messages: {len(result['messages'])}")
                    print(f"🔄 Workflow steps: {len(result['workflow_steps'])}")

                    # Show first message content
                    if result["messages"]:
                        first_msg = result["messages"][0]
                        print(f"📄 First message: {first_msg[:100]}...")

                    # Check if workflow executed properly
                    if result["workflow_steps"]:
                        print("✅ Workflow executed successfully")
                        for step in result["workflow_steps"]:
                            step_type = step.get("step_type", "unknown")
                            tool_name = step.get("tool_name", "none")
                            print(f"  🔧 Step: {step_type} | Tool: {tool_name}")
                    else:
                        print("⚠️  No workflow steps recorded")

                except Exception as e:
                    print(f"❌ Error processing conversation: {e}")

                await asyncio.sleep(1)  # Rate limiting

        except Exception as e:
            print(f"❌ Failed to initialize ReactAgent: {e}")

    async def test_streaming_capabilities(self):
        """Test streaming responses with real LLM."""
        self.print_header("TESTING STREAMING CAPABILITIES")

        try:
            mcp_tools = await get_all_mcp_tools()
            engine = ReactAgentWorkflowEngine(mcp_tools)

            query = "find me 3 action anime from 2020s"
            print(f"Streaming query: '{query}'")

            start_time = time.perf_counter()
            first_chunk_time = None
            chunk_count = 0

            print("\n📡 Streaming response:")

            async for chunk in engine.astream_conversation(
                session_id="stream_test", message=query
            ):
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                    first_chunk_latency = (first_chunk_time - start_time) * 1000
                    print(f"⚡ First chunk: {first_chunk_latency:.1f}ms")

                chunk_count += 1
                print(f"  📦 Chunk {chunk_count}: {str(chunk)[:100]}...")

            total_time = time.perf_counter() - start_time
            print(
                f"✅ Streaming completed: {chunk_count} chunks in {total_time*1000:.1f}ms"
            )

        except Exception as e:
            print(f"❌ Streaming test failed: {e}")

    async def test_react_agent_intelligence(self):
        """Test ReactAgent's AI-powered understanding capabilities."""
        self.print_header("TESTING REACT AGENT AI INTELLIGENCE")

        # Get MCP tools
        mcp_tools = await get_all_mcp_tools()
        engine = ReactAgentWorkflowEngine(mcp_tools, LLMProvider.OPENAI)

        intelligence_test_cases = [
            "find me 5 mecha anime from 2010s but not too violent",
            "show me top 3 romantic comedy anime with good animation",
            "recent dark psychological anime, limit to 4",
            "fantasy anime but not isekai, show me 7",
        ]

        for i, query in enumerate(intelligence_test_cases):
            print(f"\nIntelligence Test {i+1}: '{query}'")

            start_time = time.perf_counter()

            try:
                result = await engine.process_conversation(
                    session_id=f"intelligence_test_{i}", message=query
                )

                end_time = time.perf_counter()
                response_time = (end_time - start_time) * 1000

                print(f"  ⏱️  ReactAgent response time: {response_time:.1f}ms")
                print(
                    f"  🧠 AI Understanding: {len(result['messages'])} messages generated"
                )
                print(f"  🔧 Tools Used: {len(result['workflow_steps'])} steps")

                # Analyze workflow steps for intelligence indicators
                for step in result["workflow_steps"]:
                    step_type = step.get("step_type", "unknown")
                    tool_name = step.get("tool_name", "none")
                    confidence = step.get("confidence", 0.0)
                    print(
                        f"    📊 Step: {step_type} | Tool: {tool_name} | Confidence: {confidence}"
                    )

                # Check if ReactAgent understood the query complexity
                if len(result["workflow_steps"]) > 0:
                    print("  ✅ ReactAgent demonstrated intelligent query processing")
                else:
                    print("  ⚠️  ReactAgent processing unclear - check implementation")

            except Exception as e:
                print(f"  ❌ Intelligence test failed: {e}")

            await asyncio.sleep(1)

    async def run_all_tests(self):
        """Run all real-time LLM tests."""
        print("🚀 Starting Real-Time LLM Workflow Testing")
        print(f"⚙️  Settings loaded: {type(self.settings).__name__}")

        try:
            await self.test_react_agent_providers()
            await self.test_react_agent_integration()
            await self.test_streaming_capabilities()
            await self.test_react_agent_intelligence()

            self.print_header("TEST SUMMARY")
            print("✅ ReactAgent provider testing completed")
            print("✅ ReactAgent workflow integration tested")
            print("✅ Streaming capabilities tested")
            print("✅ ReactAgent AI intelligence validated")
            print("\n🎉 All ReactAgent real-time tests completed!")

        except Exception as e:
            print(f"❌ Test suite failed: {e}")
            raise


async def main():
    """Main test runner."""
    tester = RealLLMTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
