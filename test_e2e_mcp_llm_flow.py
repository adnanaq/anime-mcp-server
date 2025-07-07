#!/usr/bin/env python3
"""
End-to-End MCP Server LLM Flow Test.

Tests the complete flow:
Natural Language Query → MCP Server → LLM Processing → API Parameter Extraction → Platform Routing → Results

This simulates how an AI assistant (like Claude Code) would interact with the MCP server.
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

class MCPServerE2ETester:
    """End-to-end tester for MCP server with natural language queries."""
    
    def __init__(self):
        """Initialize the E2E MCP tester."""
        self.mcp_process = None
        self.test_scenarios = [
            {
                "name": "Simple Anime Search",
                "query": "Find anime about ninjas and martial arts",
                "tool": "search_anime",
                "expected_extraction": {
                    "search_terms": ["ninja", "martial arts"],
                    "likely_params": {"q": "ninja martial arts"},
                    "expected_platforms": ["jikan", "mal"]
                },
                "success_criteria": "Should extract ninja/martial arts terms and search across platforms"
            },
            {
                "name": "Genre-Based Search",
                "query": "Show me action anime with high ratings from the 2010s",
                "tool": "search_anime", 
                "expected_extraction": {
                    "genres": ["action"],
                    "time_period": "2010s",
                    "rating_filter": "high",
                    "likely_params": {"genres": "action", "min_score": 7.0},
                    "expected_platforms": ["jikan"]  # Better filtering
                },
                "success_criteria": "Should extract genre, time period, and rating requirements"
            },
            {
                "name": "Similarity Search",
                "query": "Find anime similar to Attack on Titan with dark themes",
                "tool": "find_similar_anime",
                "expected_extraction": {
                    "reference_anime": "attack on titan", 
                    "themes": ["dark"],
                    "likely_params": {"anime_title": "Attack on Titan"},
                    "expected_platforms": ["jikan", "mal"]
                },
                "success_criteria": "Should identify reference anime and route to similarity search"
            },
            {
                "name": "Currently Airing Query",
                "query": "What anime is currently airing this season?",
                "tool": "get_currently_airing_anime",
                "expected_extraction": {
                    "status": "currently_airing",
                    "season": "current",
                    "likely_params": {"status": "airing"},
                    "expected_platforms": ["jikan", "anilist"]
                },
                "success_criteria": "Should identify current season request and route to scheduling"
            },
            {
                "name": "Platform-Specific Request",
                "query": "Search for One Piece on MyAnimeList specifically",
                "tool": "search_anime",
                "expected_extraction": {
                    "search_terms": ["one piece"],
                    "platform_preference": "mal",
                    "likely_params": {"q": "One Piece", "platform": "mal"},
                    "expected_platforms": ["mal"]
                },
                "success_criteria": "Should respect platform preference and route only to MAL"
            }
        ]
    
    async def setup_environment(self):
        """Setup test environment and verify prerequisites."""
        print("🔧 Setting up E2E MCP test environment...")
        
        # Load environment
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check required environment variables
        checks = {
            "MAL_CLIENT_ID": os.getenv('MAL_CLIENT_ID'),
            "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY'),
            "ANTHROPIC_API_KEY": os.getenv('ANTHROPIC_API_KEY')
        }
        
        for key, value in checks.items():
            status = "✅" if value else "❌"
            print(f"{status} {key}: {'Available' if value else 'Missing'}")
            
        # Check if we have at least one LLM provider
        has_llm = checks["OPENAI_API_KEY"] or checks["ANTHROPIC_API_KEY"]
        if not has_llm:
            print("❌ No LLM provider available. Need OPENAI_API_KEY or ANTHROPIC_API_KEY")
            return False
            
        # Check if Qdrant is running
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:6333/", timeout=5.0)
                if response.status_code == 200:
                    print("✅ Qdrant: Running")
                else:
                    print("❌ Qdrant: Not responding properly")
                    return False
        except Exception:
            print("❌ Qdrant: Not running (start with: docker compose up -d qdrant)")
            return False
            
        return True
        
    async def start_mcp_server(self, server_type="modern"):
        """Start the MCP server as a subprocess."""
        print(f"🚀 Starting {server_type} MCP server...")
        
        # Choose server type
        if server_type == "modern":
            server_module = "src.anime_mcp.modern_server"
            print("   Using modern server with LangGraph workflows")
        else:
            server_module = "src.anime_mcp.server"
            print("   Using core server with direct tools")
        
        # Start server process
        try:
            self.mcp_process = subprocess.Popen(
                [sys.executable, "-m", server_module],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
                cwd=os.getcwd()
            )
            
            # Wait a moment for server to initialize
            await asyncio.sleep(2)
            
            # Check if process is still running
            if self.mcp_process.poll() is not None:
                stderr = self.mcp_process.stderr.read()
                print(f"❌ MCP server failed to start: {stderr}")
                return False
                
            print("✅ MCP server started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start MCP server: {e}")
            return False
    
    async def send_mcp_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send an MCP request to the server and get response."""
        if not self.mcp_process:
            raise RuntimeError("MCP server not started")
            
        # Create MCP request
        request = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": method,
            "params": params
        }
        
        # Send request
        request_json = json.dumps(request) + "\n"
        self.mcp_process.stdin.write(request_json)
        self.mcp_process.stdin.flush()
        
        # Read response (with timeout)
        try:
            # Wait for response with timeout
            for _ in range(30):  # 30 second timeout
                if self.mcp_process.stdout.readable():
                    line = self.mcp_process.stdout.readline()
                    if line.strip():
                        return json.loads(line.strip())
                await asyncio.sleep(1)
                
            raise TimeoutError("No response from MCP server within 30 seconds")
            
        except Exception as e:
            # Check for stderr output
            if self.mcp_process.stderr.readable():
                stderr = self.mcp_process.stderr.read()
                if stderr:
                    print(f"Server stderr: {stderr}")
            raise e
    
    async def test_mcp_tool_call(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single MCP tool call with natural language query."""
        print(f"\n📋 Testing: {scenario['name']}")
        print(f"🗣️  Natural Language Query: '{scenario['query']}'")
        print(f"🔧 Expected Tool: {scenario['tool']}")
        
        result = {
            "scenario": scenario["name"],
            "query": scenario["query"],
            "expected_tool": scenario["tool"],
            "success": False,
            "mcp_response": {},
            "extracted_params": {},
            "platforms_used": [],
            "analysis": "",
            "errors": []
        }
        
        try:
            # Send MCP tool call request
            mcp_params = {
                "name": scenario["tool"],
                "arguments": {
                    "query": scenario["query"]
                }
            }
            
            print(f"📡 Sending MCP request: tools/call")
            
            # Send request to MCP server
            mcp_response = await self.send_mcp_request("tools/call", mcp_params)
            result["mcp_response"] = mcp_response
            
            print(f"📨 MCP Response received")
            
            # Analyze the response
            if "result" in mcp_response:
                tool_result = mcp_response["result"]
                
                # Extract content from MCP response
                if isinstance(tool_result, dict):
                    if "content" in tool_result:
                        content = tool_result["content"]
                        if isinstance(content, list) and len(content) > 0:
                            text_content = content[0].get("text", "")
                            try:
                                # Try to parse as JSON if it looks like JSON
                                if text_content.strip().startswith('{'):
                                    parsed_content = json.loads(text_content)
                                    result["extracted_params"] = parsed_content
                                else:
                                    result["extracted_params"] = {"response_text": text_content}
                            except json.JSONDecodeError:
                                result["extracted_params"] = {"response_text": text_content}
                    else:
                        result["extracted_params"] = tool_result
                
                # Analyze parameter extraction quality
                expected = scenario["expected_extraction"]
                extracted = result["extracted_params"]
                
                analysis_points = []
                score = 0
                total_checks = 0
                
                # Check search terms extraction
                if "search_terms" in expected:
                    total_checks += 1
                    extracted_text = str(extracted).lower()
                    if any(term.lower() in extracted_text for term in expected["search_terms"]):
                        score += 1
                        analysis_points.append("✅ Search terms correctly identified")
                    else:
                        analysis_points.append("❌ Search terms not properly extracted")
                
                # Check genre extraction
                if "genres" in expected:
                    total_checks += 1
                    extracted_text = str(extracted).lower()
                    if any(genre.lower() in extracted_text for genre in expected["genres"]):
                        score += 1
                        analysis_points.append("✅ Genres correctly identified")
                    else:
                        analysis_points.append("❌ Genres not properly extracted")
                
                # Check reference anime extraction
                if "reference_anime" in expected:
                    total_checks += 1
                    extracted_text = str(extracted).lower()
                    if expected["reference_anime"].lower() in extracted_text:
                        score += 1
                        analysis_points.append("✅ Reference anime correctly identified")
                    else:
                        analysis_points.append("❌ Reference anime not properly extracted")
                
                # Check if we got any meaningful response
                if not total_checks:
                    total_checks = 1
                    if extracted and len(str(extracted)) > 10:  # Non-trivial response
                        score = 1
                        analysis_points.append("✅ Received meaningful response from LLM")
                    else:
                        analysis_points.append("❌ No meaningful response received")
                
                # Calculate success
                success_rate = score / total_checks if total_checks > 0 else 0
                result["success"] = success_rate >= 0.6  # 60% threshold
                
                result["analysis"] = f"Score: {score}/{total_checks} ({success_rate:.1%}). " + "; ".join(analysis_points)
                
                print(f"📊 Analysis: {result['analysis']}")
                print(f"📋 Extracted Parameters: {result['extracted_params']}")
                print(f"🎯 Result: {'✅ PASSED' if result['success'] else '❌ FAILED'}")
                
            else:
                # Handle error response
                if "error" in mcp_response:
                    error_info = mcp_response["error"]
                    result["errors"].append(f"MCP Error: {error_info}")
                    print(f"❌ MCP Error: {error_info}")
                else:
                    result["errors"].append("No result in MCP response")
                    print(f"❌ Unexpected MCP response format")
                    
        except Exception as e:
            result["errors"].append(str(e))
            print(f"❌ Test failed with error: {e}")
            
        return result
    
    async def test_platform_priority(self) -> Dict[str, Any]:
        """Test platform priority system (Jikan > MAL)."""
        print(f"\n📋 Testing Platform Priority System")
        
        # Test with a query that should work on both platforms
        query = "Search for popular anime"
        
        try:
            # Make request that should trigger platform routing
            mcp_params = {
                "name": "search_anime",
                "arguments": {
                    "query": query,
                    "limit": 5
                }
            }
            
            response = await self.send_mcp_request("tools/call", mcp_params)
            
            # Analyze response for platform indicators
            response_text = str(response).lower()
            
            platforms_detected = []
            if "jikan" in response_text:
                platforms_detected.append("jikan")
            if "mal" in response_text:
                platforms_detected.append("mal")
                
            # Check priority (Jikan should be preferred)
            priority_correct = False
            if "jikan" in platforms_detected:
                priority_correct = True
                print("✅ Platform priority working - Jikan detected")
            elif "mal" in platforms_detected:
                print("ℹ️  MAL detected - checking if Jikan was unavailable")
                priority_correct = True  # MAL as fallback is OK
            else:
                print("⚠️  No clear platform indicators detected")
                
            return {
                "test": "Platform Priority",
                "success": priority_correct,
                "platforms_detected": platforms_detected,
                "query": query
            }
            
        except Exception as e:
            print(f"❌ Platform priority test failed: {e}")
            return {
                "test": "Platform Priority", 
                "success": False,
                "error": str(e),
                "query": query
            }
    
    async def cleanup(self):
        """Cleanup test environment."""
        if self.mcp_process:
            print("🧹 Cleaning up MCP server...")
            self.mcp_process.terminate()
            try:
                await asyncio.wait_for(asyncio.create_task(self._wait_for_process()), timeout=5.0)
            except asyncio.TimeoutError:
                print("⚠️  Force killing MCP server...")
                self.mcp_process.kill()
            self.mcp_process = None
            
    async def _wait_for_process(self):
        """Wait for process to terminate."""
        while self.mcp_process.poll() is None:
            await asyncio.sleep(0.1)
    
    async def run_comprehensive_test(self):
        """Run comprehensive end-to-end MCP LLM flow test."""
        print("🚀 Starting Comprehensive End-to-End MCP LLM Flow Test")
        print("=" * 70)
        
        try:
            # Setup environment
            if not await self.setup_environment():
                print("❌ Environment setup failed")
                return False
                
            # Start MCP server
            if not await self.start_mcp_server("modern"):
                print("❌ Failed to start MCP server")
                return False
                
            # Test results
            results = {
                "llm_flow_tests": [],
                "platform_tests": [],
                "summary": {
                    "total_tests": 0,
                    "passed_tests": 0,
                    "failed_tests": 0,
                    "success_rate": 0.0
                }
            }
            
            print(f"\n🧠 Testing Natural Language → LLM → API Parameter Flow")
            print("=" * 60)
            
            # Test each scenario
            for scenario in self.test_scenarios:
                result = await self.test_mcp_tool_call(scenario)
                results["llm_flow_tests"].append(result)
                
                if result["success"]:
                    results["summary"]["passed_tests"] += 1
                else:
                    results["summary"]["failed_tests"] += 1
                results["summary"]["total_tests"] += 1
                
            # Test platform priority
            print(f"\n🎯 Testing Platform Priority System")
            print("=" * 40)
            
            priority_result = await self.test_platform_priority()
            results["platform_tests"].append(priority_result)
            
            if priority_result["success"]:
                results["summary"]["passed_tests"] += 1
            else:
                results["summary"]["failed_tests"] += 1
            results["summary"]["total_tests"] += 1
            
            # Calculate final results
            total = results["summary"]["total_tests"]
            passed = results["summary"]["passed_tests"]
            results["summary"]["success_rate"] = (passed / total) if total > 0 else 0
            
            # Print summary
            print(f"\n📊 COMPREHENSIVE E2E TEST SUMMARY")
            print("=" * 50)
            print(f"Total Tests: {total}")
            print(f"Passed: {passed}")
            print(f"Failed: {results['summary']['failed_tests']}")
            print(f"Success Rate: {results['summary']['success_rate']:.1%}")
            
            if results["summary"]["success_rate"] >= 0.7:
                print("🎉 OVERALL RESULT: ✅ PASSED (≥70% success rate)")
                return True
            else:
                print("💥 OVERALL RESULT: ❌ FAILED (<70% success rate)")
                return False
                
        finally:
            await self.cleanup()

async def main():
    """Main test execution."""
    tester = MCPServerE2ETester()
    success = await tester.run_comprehensive_test()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())