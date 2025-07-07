#!/usr/bin/env python3
"""
End-to-End LLM Workflow Test for Anime MCP Server.

Tests the complete flow: Natural Language Query → LLM Parsing → API Parameter Extraction → Platform Routing → Results
"""

import asyncio
import json
import logging
import os
import sys
from typing import Dict, Any, List, Optional

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class E2ELLMWorkflowTester:
    """End-to-end tester for LLM workflow with natural language queries."""
    
    def __init__(self):
        """Initialize the E2E tester."""
        self.test_scenarios = [
            {
                "name": "Simple Title Search",
                "query": "Find anime similar to Attack on Titan",
                "expected_params": {
                    "search_terms": ["attack on titan", "similar"],
                    "api_calls": ["search_anime", "find_similar_anime"],
                    "platforms": ["jikan", "mal"]
                },
                "expected_processing": "Should extract 'Attack on Titan' and route to similarity search"
            },
            {
                "name": "Genre and Year Filter",
                "query": "Show me action anime from 2020 with high ratings",
                "expected_params": {
                    "genres": ["action"],
                    "year": 2020,
                    "min_score": "> 7.0",
                    "api_calls": ["search_anime"],
                    "platforms": ["jikan"]  # Jikan has better filtering
                },
                "expected_processing": "Should extract genre filter, year filter, and rating requirement"
            },
            {
                "name": "Complex Narrative Query",
                "query": "I want dark psychological anime like Death Note with complex characters",
                "expected_params": {
                    "themes": ["dark", "psychological"],
                    "reference_anime": "death note",
                    "characteristics": ["complex characters"],
                    "api_calls": ["search_anime", "find_similar_anime"],
                    "platforms": ["jikan", "mal"]
                },
                "expected_processing": "Should extract thematic elements and use reference anime for similarity"
            },
            {
                "name": "Currently Airing Query",
                "query": "What anime is airing this season?",
                "expected_params": {
                    "status": "currently_airing",
                    "season": "current",
                    "api_calls": ["get_seasonal_anime", "get_currently_airing_anime"],
                    "platforms": ["jikan", "anilist"]
                },
                "expected_processing": "Should identify current season request and route to scheduling tools"
            },
            {
                "name": "Platform Preference",
                "query": "Search for One Piece on MyAnimeList",
                "expected_params": {
                    "search_terms": ["one piece"],
                    "platform_preference": "mal",
                    "api_calls": ["search_anime"],
                    "platforms": ["mal"]
                },
                "expected_processing": "Should respect platform preference and route specifically to MAL"
            }
        ]
        
    async def setup_test_environment(self):
        """Setup test environment with required dependencies."""
        print("🔧 Setting up E2E test environment...")
        
        # Load environment
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check required environment variables
        mal_client_id = os.getenv('MAL_CLIENT_ID')
        openai_key = os.getenv('OPENAI_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        
        print(f"✅ MAL Client ID: {'✓' if mal_client_id else '✗'}")
        print(f"✅ OpenAI API Key: {'✓' if openai_key else '✗'}")  
        print(f"✅ Anthropic API Key: {'✓' if anthropic_key else '✗'}")
        
        if not mal_client_id:
            print("⚠️  MAL_CLIENT_ID required for platform testing")
            
        if not (openai_key or anthropic_key):
            print("⚠️  Either OPENAI_API_KEY or ANTHROPIC_API_KEY required for LLM testing")
            return False
            
        return True
        
    async def test_llm_query_parsing(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Test LLM's ability to parse natural language query into structured parameters."""
        print(f"\n📋 Testing: {scenario['name']}")
        print(f"🗣️  Query: '{scenario['query']}'")
        
        result = {
            "scenario": scenario["name"],
            "query": scenario["query"],
            "success": False,
            "extracted_params": {},
            "api_calls_made": [],
            "platforms_used": [],
            "errors": [],
            "llm_reasoning": ""
        }
        
        try:
            # Import workflow components
            from src.langgraph.react_agent_workflow import ReactAgentWorkflowEngine, LLMProvider
            from src.anime_mcp.server import get_mcp_tools
            
            # Get available MCP tools
            mcp_tools = await get_mcp_tools()
            print(f"📚 Available MCP tools: {list(mcp_tools.keys())}")
            
            # Choose LLM provider
            llm_provider = LLMProvider.ANTHROPIC if os.getenv('ANTHROPIC_API_KEY') else LLMProvider.OPENAI
            print(f"🤖 Using LLM provider: {llm_provider.value}")
            
            # Initialize workflow engine
            workflow_engine = ReactAgentWorkflowEngine(
                mcp_tools=mcp_tools,
                llm_provider=llm_provider
            )
            
            # Execute query through LLM workflow
            session_id = f"test-session-{scenario['name'].lower().replace(' ', '-')}"
            
            print(f"⚙️  Executing LLM workflow...")
            
            # Stream the workflow execution to capture reasoning
            reasoning_steps = []
            api_calls = []
            
            async for chunk in workflow_engine.stream_workflow(
                query=scenario["query"],
                session_id=session_id
            ):
                if isinstance(chunk, dict):
                    # Capture LLM reasoning
                    if "messages" in chunk:
                        for msg in chunk["messages"]:
                            if hasattr(msg, 'content'):
                                reasoning_steps.append(str(msg.content))
                    
                    # Capture tool calls
                    if "tools" in chunk:
                        for tool_call in chunk["tools"]:
                            api_calls.append({
                                "tool": tool_call.get("name", "unknown"),
                                "args": tool_call.get("args", {})
                            })
                    
                    # Capture final result
                    if "result" in chunk:
                        result["extracted_params"] = chunk["result"]
            
            result["llm_reasoning"] = " ".join(reasoning_steps)
            result["api_calls_made"] = api_calls
            
            # Analyze extracted parameters against expectations
            expected = scenario["expected_params"]
            extracted = result["extracted_params"]
            
            print(f"📊 Expected parameters: {expected}")
            print(f"📊 Extracted parameters: {extracted}")
            
            # Verify parameter extraction
            extraction_score = 0
            total_checks = 0
            
            # Check search terms extraction
            if "search_terms" in expected:
                total_checks += 1
                if any(term in str(extracted).lower() for term in expected["search_terms"]):
                    extraction_score += 1
                    print("✅ Search terms correctly extracted")
                else:
                    print("❌ Search terms not properly extracted")
                    
            # Check genre extraction
            if "genres" in expected:
                total_checks += 1
                if any(genre in str(extracted).lower() for genre in expected["genres"]):
                    extraction_score += 1
                    print("✅ Genres correctly extracted")
                else:
                    print("❌ Genres not properly extracted")
                    
            # Check API call routing
            if "api_calls" in expected:
                total_checks += 1
                api_calls_made = [call["tool"] for call in result["api_calls_made"]]
                if any(api_call in api_calls_made for api_call in expected["api_calls"]):
                    extraction_score += 1
                    print(f"✅ Correct API calls: {api_calls_made}")
                else:
                    print(f"❌ Expected API calls {expected['api_calls']}, got {api_calls_made}")
                    
            # Check platform routing
            if "platforms" in expected:
                total_checks += 1
                # Analyze which platforms were actually used (if we can detect this)
                platforms_detected = []
                for call in result["api_calls_made"]:
                    if "jikan" in str(call).lower():
                        platforms_detected.append("jikan")
                    if "mal" in str(call).lower():
                        platforms_detected.append("mal")
                        
                result["platforms_used"] = list(set(platforms_detected))
                
                if any(platform in result["platforms_used"] for platform in expected["platforms"]):
                    extraction_score += 1
                    print(f"✅ Correct platforms used: {result['platforms_used']}")
                else:
                    print(f"❌ Expected platforms {expected['platforms']}, detected {result['platforms_used']}")
            
            # Calculate success score
            if total_checks > 0:
                success_rate = extraction_score / total_checks
                result["success"] = success_rate >= 0.6  # 60% threshold
                print(f"📈 Parameter extraction score: {extraction_score}/{total_checks} ({success_rate:.1%})")
            else:
                result["success"] = len(result["api_calls_made"]) > 0
                print(f"📈 Basic execution success: {result['success']}")
                
            print(f"🎯 Test result: {'✅ PASSED' if result['success'] else '❌ FAILED'}")
            
        except Exception as e:
            result["errors"].append(str(e))
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            
        return result
        
    async def test_direct_api_routing(self) -> Dict[str, Any]:
        """Test direct API routing and parameter passing."""
        print(f"\n📋 Testing Direct API Parameter Routing")
        
        test_cases = [
            {
                "name": "Jikan Search with Filters",
                "api": "jikan",
                "method": "search_anime",
                "params": {
                    "q": "Attack on Titan",
                    "genres": [1],  # Action
                    "min_score": 8.0,
                    "limit": 5
                },
                "expected": "Should return filtered results with action genre and high scores"
            },
            {
                "name": "MAL Search with Fields",
                "api": "mal", 
                "method": "search_anime",
                "params": {
                    "q": "One Piece",
                    "fields": "id,title,mean,num_episodes,status",
                    "limit": 3
                },
                "expected": "Should return results with specified fields only"
            }
        ]
        
        results = []
        
        for case in test_cases:
            print(f"\n🔍 Testing {case['name']}")
            
            try:
                if case["api"] == "jikan":
                    from src.integrations.clients.jikan_client import JikanClient
                    client = JikanClient()
                    
                    if case["method"] == "search_anime":
                        result = await client.search_anime(**case["params"])
                        
                elif case["api"] == "mal":
                    from src.integrations.clients.mal_client import MALClient
                    mal_client_id = os.getenv('MAL_CLIENT_ID')
                    client = MALClient(client_id=mal_client_id)
                    
                    if case["method"] == "search_anime":
                        result = await client.search_anime(**case["params"])
                        
                success = len(result) > 0 if result else False
                print(f"📊 Results: {len(result) if result else 0} items")
                print(f"🎯 Success: {'✅ PASSED' if success else '❌ FAILED'}")
                
                results.append({
                    "test": case["name"],
                    "success": success,
                    "result_count": len(result) if result else 0,
                    "params_used": case["params"]
                })
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
                results.append({
                    "test": case["name"],
                    "success": False,
                    "error": str(e),
                    "params_used": case["params"]
                })
                
        return {"direct_api_tests": results}
        
    async def run_comprehensive_test(self):
        """Run comprehensive end-to-end LLM workflow test."""
        print("🚀 Starting Comprehensive End-to-End LLM Workflow Test")
        print("=" * 70)
        
        # Setup test environment
        if not await self.setup_test_environment():
            print("❌ Test environment setup failed")
            return False
            
        # Test results storage
        all_results = {
            "llm_workflow_tests": [],
            "direct_api_tests": [],
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "success_rate": 0.0
            }
        }
        
        print(f"\n🧠 Testing LLM Natural Language Query Processing")
        print("=" * 50)
        
        # Test each LLM scenario
        for scenario in self.test_scenarios:
            result = await self.test_llm_query_parsing(scenario)
            all_results["llm_workflow_tests"].append(result)
            
            if result["success"]:
                all_results["summary"]["passed_tests"] += 1
            else:
                all_results["summary"]["failed_tests"] += 1
            all_results["summary"]["total_tests"] += 1
            
        print(f"\n🔗 Testing Direct API Parameter Routing")  
        print("=" * 50)
        
        # Test direct API routing
        direct_results = await self.test_direct_api_routing()
        all_results["direct_api_tests"] = direct_results["direct_api_tests"]
        
        for test in direct_results["direct_api_tests"]:
            if test["success"]:
                all_results["summary"]["passed_tests"] += 1
            else:
                all_results["summary"]["failed_tests"] += 1
            all_results["summary"]["total_tests"] += 1
            
        # Calculate final success rate
        total = all_results["summary"]["total_tests"]
        passed = all_results["summary"]["passed_tests"]
        all_results["summary"]["success_rate"] = (passed / total) if total > 0 else 0
        
        # Print final summary
        print(f"\n📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 50)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {all_results['summary']['failed_tests']}")
        print(f"Success Rate: {all_results['summary']['success_rate']:.1%}")
        
        if all_results['summary']['success_rate'] >= 0.7:
            print("🎉 OVERALL RESULT: ✅ PASSED (≥70% success rate)")
            return True
        else:
            print("💥 OVERALL RESULT: ❌ FAILED (<70% success rate)")
            return False

async def main():
    """Main test execution."""
    tester = E2ELLMWorkflowTester()
    success = await tester.run_comprehensive_test()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())