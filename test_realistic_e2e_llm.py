#!/usr/bin/env python3
"""
Realistic End-to-End MCP Server LLM Flow Test.

Tests the actual MCP tools we have in the codebase:
Natural Language Query → MCP Tool Call → Parameter Processing → API Integration → Results

Based on actual codebase tools, not assumptions.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from typing import Dict, Any, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealisticMCPTester:
    """Realistic tester based on actual MCP server implementation."""
    
    def __init__(self):
        """Initialize with actual tools from codebase."""
        
        # Core MCP Server tools (from src/anime_mcp/server.py)
        self.core_server_tools = {
            "search_anime": {
                "params": ["query", "limit", "genres", "year_range", "anime_types", "studios", "exclusions", "mood_keywords"],
                "description": "Semantic search for anime with filtering"
            },
            "search_anime_advanced": {
                "params": ["search_params"],  # Takes SearchAnimeInput model
                "description": "Advanced search with validation"
            },
            "get_anime_details": {
                "params": ["anime_id"],
                "description": "Get detailed anime information by ID"
            },
            "find_similar_anime": {
                "params": ["anime_id", "limit"],
                "description": "Find anime similar to reference"
            },
            "search_anime_by_image": {
                "params": ["image_data", "limit"],
                "description": "Visual similarity search using images"
            },
            "get_anime_stats": {
                "params": [],
                "description": "Get database statistics"
            }
        }
        
        # Modern MCP Server tools (from src/anime_mcp/modern_server.py)  
        self.modern_server_tools = {
            "discover_anime": {
                "params": ["query", "user_preferences", "session_id"],
                "description": "Intelligent anime discovery using multi-agent workflow"
            },
            "get_currently_airing_anime": {
                "params": ["season", "year", "include_upcoming"],
                "description": "Get currently airing anime with real-time schedules"
            },
            "find_similar_anime_workflow": {
                "params": ["reference_anime", "similarity_preferences", "session_id"],
                "description": "AI-powered similarity analysis workflow"
            },
            "search_by_streaming_platform": {
                "params": ["platform", "query", "session_id"],
                "description": "Search anime on specific streaming platforms"
            }
        }
        
        # Test scenarios based on actual tools
        self.test_scenarios = [
            {
                "name": "Core Server - Basic Search",
                "server": "core",
                "tool": "search_anime",
                "natural_query": "Find action anime with ninjas",
                "expected_params": {
                    "query": "action anime ninjas",
                    "genres": ["Action"],
                    "mood_keywords": ["ninja"]
                },
                "test_params": {
                    "query": "action anime ninjas",
                    "limit": 5
                },
                "success_criteria": "Should return anime results with action/ninja themes"
            },
            {
                "name": "Core Server - Similarity Search",
                "server": "core", 
                "tool": "find_similar_anime",
                "natural_query": "Find anime similar to Attack on Titan",
                "expected_params": {
                    "anime_id": "attack_on_titan_id"
                },
                "test_params": {
                    "anime_id": "16498",  # Attack on Titan MAL ID
                    "limit": 5
                },
                "success_criteria": "Should return anime similar to Attack on Titan"
            },
            {
                "name": "Core Server - Get Details",
                "server": "core",
                "tool": "get_anime_details", 
                "natural_query": "Get details for One Piece",
                "expected_params": {
                    "anime_id": "one_piece_id"
                },
                "test_params": {
                    "anime_id": "21"  # One Piece MAL ID
                },
                "success_criteria": "Should return detailed information about One Piece"
            },
            {
                "name": "Modern Server - Discovery Workflow",
                "server": "modern",
                "tool": "discover_anime",
                "natural_query": "I want dark psychological anime like Death Note",
                "expected_params": {
                    "query": "dark psychological anime like Death Note",
                    "user_preferences": {"themes": ["dark", "psychological"]}
                },
                "test_params": {
                    "query": "dark psychological anime like Death Note",
                    "user_preferences": {"themes": ["dark", "psychological"]},
                    "session_id": "test_session"
                },
                "success_criteria": "Should use AI workflow to find dark psychological anime"
            },
            {
                "name": "Modern Server - Currently Airing",
                "server": "modern",
                "tool": "get_currently_airing_anime",
                "natural_query": "What anime is airing this season?",
                "expected_params": {
                    "season": "current",
                    "include_upcoming": False
                },
                "test_params": {
                    "season": "fall",
                    "year": 2024,
                    "include_upcoming": True
                },
                "success_criteria": "Should return currently airing anime with schedules"
            }
        ]
        
        self.mcp_process = None
        
    async def setup_environment(self):
        """Setup and verify actual environment requirements."""
        print("🔧 Setting up realistic E2E test environment...")
        
        # Load environment
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check actual required environment variables
        env_checks = {
            "MAL_CLIENT_ID": os.getenv('MAL_CLIENT_ID'),
            "QDRANT_URL": os.getenv('QDRANT_URL', 'http://localhost:6333'),
            "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY'),
            "ANTHROPIC_API_KEY": os.getenv('ANTHROPIC_API_KEY')
        }
        
        for key, value in env_checks.items():
            status = "✅" if value else "❌"
            print(f"{status} {key}: {'Set' if value else 'Missing'}")
            
        # Check Qdrant availability
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:6333/", timeout=3.0)
                if response.status_code == 200:
                    print("✅ Qdrant: Running")
                else:
                    print("❌ Qdrant: Not responding")
                    return False
        except Exception as e:
            print(f"❌ Qdrant: Connection failed - {e}")
            print("   Start with: docker compose up -d qdrant")
            return False
            
        # Check for LLM provider (needed for modern server)
        has_llm = env_checks["OPENAI_API_KEY"] or env_checks["ANTHROPIC_API_KEY"]
        if not has_llm:
            print("⚠️  No LLM provider - modern server tests will be limited")
            
        return True
        
    async def start_mcp_server(self, server_type: str):
        """Start the specified MCP server."""
        print(f"🚀 Starting {server_type} MCP server...")
        
        if server_type == "core":
            server_module = "src.anime_mcp.server"
        elif server_type == "modern":
            server_module = "src.anime_mcp.modern_server"
        else:
            raise ValueError(f"Unknown server type: {server_type}")
            
        # Start server process in stdio mode (default for MCP)
        try:
            self.mcp_process = subprocess.Popen(
                [sys.executable, "-m", server_module],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
                cwd=os.getcwd(),
                env=dict(os.environ, PYTHONPATH=os.getcwd())
            )
            
            # Wait for server initialization
            await asyncio.sleep(3)
            
            # Check if server started
            if self.mcp_process.poll() is not None:
                stderr_output = self.mcp_process.stderr.read()
                stdout_output = self.mcp_process.stdout.read()
                print(f"❌ Server failed to start")
                print(f"STDERR: {stderr_output}")
                print(f"STDOUT: {stdout_output}")
                return False
                
            print("✅ MCP server started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
            
    async def call_mcp_tool(self, tool_name: str, params: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        """Call an MCP tool and get the response."""
        if not self.mcp_process:
            raise RuntimeError("MCP server not running")
            
        # Create MCP JSON-RPC request
        request = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            }
        }
        
        # Send request
        request_json = json.dumps(request) + "\n"
        print(f"📡 Sending: {request_json.strip()}")
        
        self.mcp_process.stdin.write(request_json)
        self.mcp_process.stdin.flush()
        
        # Wait for response
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if process is still alive
            if self.mcp_process.poll() is not None:
                stderr_output = self.mcp_process.stderr.read()
                raise RuntimeError(f"Server died: {stderr_output}")
                
            # Try to read response
            line = self.mcp_process.stdout.readline()
            if line.strip():
                try:
                    response = json.loads(line.strip())
                    print(f"📨 Received: {json.dumps(response, indent=2)}")
                    return response
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON decode error: {e}, line: {line}")
                    continue
                    
            await asyncio.sleep(0.1)
            
        raise TimeoutError(f"No response within {timeout} seconds")
        
    async def test_mcp_tool_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single MCP tool scenario."""
        print(f"\n📋 Testing: {scenario['name']}")
        print(f"🗣️  Natural Query: '{scenario['natural_query']}'")
        print(f"🔧 Tool: {scenario['tool']}")
        print(f"📊 Test Params: {scenario['test_params']}")
        
        result = {
            "scenario": scenario["name"],
            "tool": scenario["tool"],
            "natural_query": scenario["natural_query"],
            "success": False,
            "response": {},
            "analysis": "",
            "errors": []
        }
        
        try:
            # Call the MCP tool
            response = await self.call_mcp_tool(scenario["tool"], scenario["test_params"])
            result["response"] = response
            
            # Analyze response
            if "result" in response:
                tool_result = response["result"]
                
                # Check if we got content
                if isinstance(tool_result, dict) and "content" in tool_result:
                    content = tool_result["content"]
                    if isinstance(content, list) and len(content) > 0:
                        # Extract text content
                        text_content = content[0].get("text", "")
                        
                        # Try to parse as JSON if possible
                        try:
                            if text_content.strip().startswith('[') or text_content.strip().startswith('{'):
                                parsed_data = json.loads(text_content)
                                result["parsed_data"] = parsed_data
                                
                                # Analyze results
                                if isinstance(parsed_data, list):
                                    result["result_count"] = len(parsed_data)
                                    result["success"] = len(parsed_data) > 0
                                    result["analysis"] = f"Returned {len(parsed_data)} results"
                                elif isinstance(parsed_data, dict):
                                    result["success"] = bool(parsed_data)
                                    result["analysis"] = f"Returned data object with {len(parsed_data)} fields"
                            else:
                                result["success"] = len(text_content) > 10
                                result["analysis"] = f"Returned text response ({len(text_content)} chars)"
                                
                        except json.JSONDecodeError:
                            result["success"] = len(text_content) > 10
                            result["analysis"] = f"Returned text response ({len(text_content)} chars)"
                            
                elif isinstance(tool_result, (list, dict)):
                    # Direct result without content wrapper
                    result["parsed_data"] = tool_result
                    if isinstance(tool_result, list):
                        result["result_count"] = len(tool_result)
                        result["success"] = len(tool_result) > 0
                        result["analysis"] = f"Direct result: {len(tool_result)} items"
                    else:
                        result["success"] = bool(tool_result)
                        result["analysis"] = f"Direct result: {len(tool_result)} fields"
                        
            elif "error" in response:
                error_info = response["error"]
                result["errors"].append(f"MCP Error: {error_info}")
                result["analysis"] = f"MCP Error: {error_info.get('message', str(error_info))}"
            else:
                result["errors"].append("Unexpected response format")
                result["analysis"] = "No result or error in response"
                
            print(f"📊 Analysis: {result['analysis']}")
            print(f"🎯 Result: {'✅ PASSED' if result['success'] else '❌ FAILED'}")
            
        except Exception as e:
            result["errors"].append(str(e))
            result["analysis"] = f"Exception: {str(e)}"
            print(f"❌ Test failed: {e}")
            
        return result
        
    async def test_server_tools(self, server_type: str) -> List[Dict[str, Any]]:
        """Test all tools for a specific server."""
        print(f"\n🧪 Testing {server_type} server tools")
        print("=" * 50)
        
        # Start the appropriate server
        if not await self.start_mcp_server(server_type):
            return []
            
        results = []
        
        try:
            # Get scenarios for this server
            scenarios = [s for s in self.test_scenarios if s["server"] == server_type]
            
            for scenario in scenarios:
                result = await self.test_mcp_tool_scenario(scenario)
                results.append(result)
                
        finally:
            # Cleanup server
            if self.mcp_process:
                self.mcp_process.terminate()
                try:
                    await asyncio.wait_for(self._wait_for_process(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.mcp_process.kill()
                self.mcp_process = None
                
        return results
        
    async def _wait_for_process(self):
        """Wait for process to terminate."""
        while self.mcp_process and self.mcp_process.poll() is None:
            await asyncio.sleep(0.1)
            
    async def run_comprehensive_test(self):
        """Run comprehensive test of actual MCP server implementation."""
        print("🚀 Starting Realistic End-to-End MCP Server Test")
        print("Testing actual tools from codebase, not assumptions")
        print("=" * 70)
        
        # Setup environment
        if not await self.setup_environment():
            print("❌ Environment setup failed")
            return False
            
        # Test results
        all_results = {
            "core_server_tests": [],
            "modern_server_tests": [],
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "success_rate": 0.0
            }
        }
        
        # Test core server
        print(f"\n🔧 Testing Core MCP Server")
        print(f"Tools: {list(self.core_server_tools.keys())}")
        
        core_results = await self.test_server_tools("core")
        all_results["core_server_tests"] = core_results
        
        # Test modern server (if LLM available)
        has_llm = os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        if has_llm:
            print(f"\n🤖 Testing Modern MCP Server (with LLM)")
            print(f"Tools: {list(self.modern_server_tools.keys())}")
            
            modern_results = await self.test_server_tools("modern")
            all_results["modern_server_tests"] = modern_results
        else:
            print(f"\n⚠️  Skipping Modern MCP Server (no LLM provider)")
            
        # Calculate summary
        all_test_results = core_results + all_results["modern_server_tests"]
        
        for result in all_test_results:
            all_results["summary"]["total_tests"] += 1
            if result["success"]:
                all_results["summary"]["passed_tests"] += 1
            else:
                all_results["summary"]["failed_tests"] += 1
                
        total = all_results["summary"]["total_tests"]
        passed = all_results["summary"]["passed_tests"]
        all_results["summary"]["success_rate"] = (passed / total) if total > 0 else 0
        
        # Print final summary
        print(f"\n📊 REALISTIC E2E TEST SUMMARY")
        print("=" * 50)
        print(f"Core Server Tests: {len(core_results)}")
        print(f"Modern Server Tests: {len(all_results['modern_server_tests'])}")
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {all_results['summary']['failed_tests']}")
        print(f"Success Rate: {all_results['summary']['success_rate']:.1%}")
        
        if all_results["summary"]["success_rate"] >= 0.6:
            print("🎉 OVERALL RESULT: ✅ PASSED (≥60% success rate)")
            return True
        else:
            print("💥 OVERALL RESULT: ❌ FAILED (<60% success rate)")
            return False

async def main():
    """Main test execution."""
    tester = RealisticMCPTester()
    success = await tester.run_comprehensive_test()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())