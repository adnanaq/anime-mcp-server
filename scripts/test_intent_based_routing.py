#!/usr/bin/env python3
"""
Intent-based comprehensive test suite for intelligent routing system.

Tests all 9 intent types with 20 queries each (180 total queries) to thoroughly
analyze routing performance by intent classification.
"""

import asyncio
import json
import sys
import os
import re
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import logging
from collections import defaultdict

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our routing and analysis components
from src.langgraph.intelligent_router import IntelligentRouter, QueryIntent, RoutingDecision
from src.langgraph.query_analyzer import QueryAnalyzer
from src.config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentBasedTestSuite:
    """Comprehensive intent-based test suite for routing analysis."""
    
    def __init__(self):
        self.router = IntelligentRouter()
        self.analyzer = QueryAnalyzer()
        
        # Generate 20 queries for each intent type
        self.intent_queries = {
            QueryIntent.SEARCH: [
                "Find anime about samurai warriors",
                "Search for anime with magic and wizards",
                "Show me anime about high school romance",
                "Look for anime with robots and mecha",
                "Find anime set in medieval times",
                "Search for anime about cooking and food",
                "Show me anime with supernatural powers",
                "Find anime about music and bands",
                "Look for anime set in space",
                "Search for anime with detective stories",
                "Find anime about martial arts",
                "Show me anime with vampire characters",
                "Search for anime about time travel",
                "Find anime with dragon characters",
                "Look for anime about school life",
                "Search for anime with comedy elements",
                "Find anime about pirates and adventure",
                "Show me anime with magical girls",
                "Search for anime about sports competitions",
                "Find anime with post-apocalyptic settings"
            ],
            QueryIntent.SIMILAR: [
                "Show me anime similar to Death Note",
                "Find anime like Attack on Titan",
                "Recommend anime similar to Naruto",
                "I want anime like Your Name",
                "Find something similar to Demon Slayer",
                "Show me anime like Spirited Away",
                "Recommend anime similar to One Piece",
                "Find anime like Fullmetal Alchemist",
                "I want something similar to Tokyo Ghoul",
                "Show me anime like Cowboy Bebop",
                "Find anime similar to My Hero Academia",
                "Recommend something like Princess Mononoke",
                "I want anime similar to Steins;Gate",
                "Show me anime like Dragon Ball Z",
                "Find something similar to Akira",
                "Recommend anime like Hunter x Hunter",
                "I want anime similar to Evangelion",
                "Show me anime like Bleach",
                "Find something similar to Mob Psycho 100",
                "Recommend anime like Jujutsu Kaisen"
            ],
            QueryIntent.SCHEDULE: [
                "What anime is airing today?",
                "Show me the anime broadcast schedule",
                "What time does Jujutsu Kaisen air?",
                "Which anime are currently airing?",
                "What anime comes out tomorrow?",
                "Show me this week's anime schedule",
                "When does the next episode of Attack on Titan air?",
                "What anime is broadcasting tonight?",
                "Show me upcoming anime releases",
                "What anime finished airing last week?",
                "When does Demon Slayer air?",
                "Show me anime airing this season",
                "What anime episodes come out this weekend?",
                "When is the next My Hero Academia episode?",
                "Show me anime broadcast times",
                "What anime is airing on Saturday?",
                "When does One Piece air each week?",
                "Show me anime release schedule for this month",
                "What anime aired yesterday?",
                "When do new anime episodes typically release?"
            ],
            QueryIntent.STREAMING: [
                "Where can I watch Death Note?",
                "What anime is available on Netflix?",
                "Show me anime on Crunchyroll",
                "Where can I stream Attack on Titan?",
                "What anime is on Funimation?",
                "Find anime available on Hulu",
                "Where can I watch Demon Slayer legally?",
                "What anime is streaming on Disney+?",
                "Show me anime on Amazon Prime",
                "Where can I watch Your Name?",
                "What anime is available for free streaming?",
                "Find anime on HBO Max",
                "Where can I stream Spirited Away?",
                "What anime is on VRV?",
                "Show me anime available on Tubi",
                "Where can I watch One Piece online?",
                "What anime is streaming on Paramount+?",
                "Find anime available on Apple TV+",
                "Where can I watch Akira?",
                "What anime is on Peacock?"
            ],
            QueryIntent.COMPARISON: [
                "Compare Death Note ratings across platforms",
                "Show differences between MAL and AniList scores",
                "Compare Attack on Titan reviews on different sites",
                "How do critics rate Demon Slayer vs audience scores?",
                "Compare Your Name ratings internationally",
                "Show rating differences between Eastern and Western audiences",
                "Compare Spirited Away scores across databases",
                "How do MAL and Kitsu rate One Piece differently?",
                "Compare Akira ratings over time",
                "Show score differences for Evangelion across platforms",
                "Compare My Hero Academia ratings by region",
                "How do different sites rate Studio Ghibli films?",
                "Compare seasonal anime ratings across platforms",
                "Show rating discrepancies for controversial anime",
                "Compare user vs critic scores for popular anime",
                "How do streaming platforms rate anime differently?",
                "Compare anime ratings before and after English dub",
                "Show rating differences for long-running vs short anime",
                "Compare anime movie vs series ratings",
                "How do anime ratings vary by demographic?"
            ],
            QueryIntent.ENRICHMENT: [
                "Get comprehensive data for Death Note from all sources",
                "Show me complete information about Attack on Titan",
                "Gather all available data on Demon Slayer",
                "Get detailed information about Your Name from multiple platforms",
                "Show comprehensive data for Spirited Away",
                "Gather complete information about One Piece",
                "Get all available data on Akira",
                "Show comprehensive information about Evangelion",
                "Gather detailed data on My Hero Academia",
                "Get complete information about Studio Ghibli films",
                "Show all available data on Cowboy Bebop",
                "Gather comprehensive information about Fullmetal Alchemist",
                "Get detailed data on Tokyo Ghoul from all sources",
                "Show complete information about Steins;Gate",
                "Gather all available data on Dragon Ball Z",
                "Get comprehensive information about Hunter x Hunter",
                "Show detailed data on Bleach",
                "Gather complete information about Mob Psycho 100",
                "Get all available data on Jujutsu Kaisen",
                "Show comprehensive information about Princess Mononoke"
            ],
            QueryIntent.DISCOVERY: [
                "Recommend anime based on my preferences",
                "Help me discover new anime to watch",
                "What anime should I watch next?",
                "Recommend anime for beginners",
                "Help me find anime I might like",
                "What are some underrated anime gems?",
                "Recommend anime based on my mood",
                "Help me discover anime from different genres",
                "What anime should I watch if I like action?",
                "Recommend anime for someone who likes romance",
                "Help me find anime with strong female characters",
                "What anime should I watch if I like psychological themes?",
                "Recommend anime with beautiful animation",
                "Help me discover anime with unique art styles",
                "What anime should I watch for emotional stories?",
                "Recommend anime with great soundtracks",
                "Help me find anime with complex plots",
                "What anime should I watch if I like slice of life?",
                "Recommend anime with philosophical themes",
                "Help me discover anime that will make me think"
            ],
            QueryIntent.SEASONAL: [
                "What anime aired in Winter 2024?",
                "Show me Spring 2023 anime",
                "What's airing in the current season?",
                "Show me Summer 2022 anime releases",
                "What anime came out in Fall 2021?",
                "Show me this season's most popular anime",
                "What anime is airing in Winter 2025?",
                "Show me last season's best anime",
                "What anime aired in Spring 2020?",
                "Show me Summer 2019 anime lineup",
                "What anime came out in Fall 2018?",
                "Show me Winter 2017 anime",
                "What anime aired in Spring 2016?",
                "Show me Summer 2015 anime releases",
                "What anime came out in Fall 2014?",
                "Show me Winter 2013 anime",
                "What anime aired in Spring 2012?",
                "Show me Summer 2011 anime lineup",
                "What anime came out in Fall 2010?",
                "Show me seasonal anime trends over time"
            ],
            QueryIntent.TRENDING: [
                "What anime is trending right now?",
                "Show me the most popular anime this week",
                "What anime is everyone talking about?",
                "Show me trending anime on social media",
                "What anime is most watched currently?",
                "Show me the hottest anime right now",
                "What anime is viral this month?",
                "Show me the most discussed anime",
                "What anime has the most buzz?",
                "Show me trending anime hashtags",
                "What anime is popular on TikTok?",
                "Show me the most shared anime content",
                "What anime is trending on Twitter?",
                "Show me popular anime memes",
                "What anime is everyone recommending?",
                "Show me the most liked anime posts",
                "What anime is trending in different countries?",
                "Show me viral anime clips",
                "What anime is popular among influencers?",
                "Show me the most talked about anime episodes"
            ]
        }
    
    async def run_comprehensive_intent_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis on all intent types."""
        print("🚀 Starting comprehensive intent-based routing analysis...")
        print("=" * 70)
        
        results = {
            "test_summary": {
                "total_queries": 0,
                "total_intents": len(self.intent_queries),
                "successful_routings": 0,
                "failed_routings": 0,
                "average_confidence": 0.0,
                "intent_accuracy": 0.0,
            },
            "intent_analysis": {},
            "intent_performance": {},
            "routing_patterns": {},
            "optimization_opportunities": [],
            "detailed_results": []
        }
        
        # Test each intent type
        for intent, queries in self.intent_queries.items():
            print(f"🎯 Testing Intent: {intent.value.upper()} ({len(queries)} queries)")
            intent_results = await self._test_intent_category(intent, queries)
            results["intent_analysis"][intent.value] = intent_results
            results["detailed_results"].extend(intent_results["detailed_results"])
            results["test_summary"]["total_queries"] += len(queries)
        
        # Calculate summary statistics
        self._calculate_intent_summary_stats(results)
        
        # Analyze intent-specific patterns
        self._analyze_intent_patterns(results)
        
        # Identify intent-specific optimization opportunities
        self._identify_intent_optimization_opportunities(results)
        
        print("✅ Comprehensive intent-based routing analysis completed")
        return results
    
    async def _test_intent_category(self, expected_intent: QueryIntent, queries: List[str]) -> Dict[str, Any]:
        """Test a specific intent category with corrected evaluation logic."""
        intent_results = {
            "expected_intent": expected_intent.value,
            "total_queries": len(queries),
            "successful_routings": 0,
            "failed_routings": 0,
            "intent_accuracy": 0.0,
            "average_confidence": 0.0,
            "complexity_distribution": {},
            "strategy_distribution": {},
            "platform_distribution": {},
            "tool_usage": {},
            "detailed_results": []
        }
        
        confidences = []
        intent_matches = 0
        
        for i, query in enumerate(queries):
            print(f"    🔄 Testing query {i+1}/{len(queries)}: {query[:50]}...")
            
            try:
                # Test intelligent router
                routing_decision = await self.router.route_query(query)
                
                # --- CORRECTED LOGIC ---
                # The original script had flawed logic here. We now directly check the intent from the router's reasoning.
                classified_intent = "unknown"
                if routing_decision.reasoning:
                    # Extracts intent from a string like "Classified query intent as 'search'"
                    match = re.search(r"\'(.*?)\'", routing_decision.reasoning[0])
                    if match:
                        classified_intent = match.group(1)
                
                intent_correct = (expected_intent.value == classified_intent)
                # --- END CORRECTION ---

                if intent_correct:
                    intent_matches += 1
                
                # Record results
                result = {
                    "query": query,
                    "expected_intent": expected_intent.value,
                    "classified_intent": classified_intent,
                    "intent_correct": intent_correct,
                    "routing_decision": asdict(routing_decision),
                    "analysis_result": {}, # This was from a different component, removing for clarity
                    "success": True,
                    "error": None
                }
                
                intent_results["successful_routings"] += 1
                confidences.append(routing_decision.confidence)
                
                # Track distributions
                complexity = routing_decision.estimated_complexity
                strategy = routing_decision.execution_strategy
                
                intent_results["complexity_distribution"][complexity] = intent_results["complexity_distribution"].get(complexity, 0) + 1
                intent_results["strategy_distribution"][strategy] = intent_results["strategy_distribution"].get(strategy, 0) + 1
                
                # Track tool usage
                for tool in routing_decision.primary_tools:
                    intent_results["tool_usage"][tool] = intent_results["tool_usage"].get(tool, 0) + 1
                
                for tool in routing_decision.secondary_tools:
                    intent_results["tool_usage"][tool] = intent_results["tool_usage"].get(tool, 0) + 1
                
                # Track platform usage
                for platform, priority in routing_decision.platform_priorities.items():
                    if platform not in intent_results["platform_distribution"]:
                        intent_results["platform_distribution"][platform] = []
                    intent_results["platform_distribution"][platform].append(priority)
                
                print(f"      ✅ {complexity} complexity, {routing_decision.confidence:.2f} confidence, intent {'✓' if intent_correct else '✗'}")
                
            except Exception as e:
                result = {
                    "query": query,
                    "expected_intent": expected_intent.value,
                    "classified_intent": None,
                    "intent_correct": False,
                    "routing_decision": None,
                    "analysis_result": None,
                    "success": False,
                    "error": str(e)
                }
                intent_results["failed_routings"] += 1
                print(f"      ❌ Failed: {e}")
                logger.error(f"Failed to route query '{query}': {e}")
            
            intent_results["detailed_results"].append(result)
        
        # Calculate averages
        if confidences:
            intent_results["average_confidence"] = sum(confidences) / len(confidences)
        
        intent_results["intent_accuracy"] = intent_matches / len(queries) if queries else 0
        
        # Calculate average platform priorities
        for platform in intent_results["platform_distribution"]:
            priorities = intent_results["platform_distribution"][platform]
            intent_results["platform_distribution"][platform] = {
                "average_priority": sum(priorities) / len(priorities),
                "usage_count": len(priorities)
            }
        
        print(f"    📊 Intent Accuracy: {intent_results['intent_accuracy']:.1%}, Avg Confidence: {intent_results['average_confidence']:.2f}")
        
        return intent_results
    
    
    
    def _calculate_intent_summary_stats(self, results: Dict[str, Any]):
        """Calculate overall intent-based summary statistics."""
        total_successful = sum(intent["successful_routings"] for intent in results["intent_analysis"].values())
        total_failed = sum(intent["failed_routings"] for intent in results["intent_analysis"].values())
        
        results["test_summary"]["successful_routings"] = total_successful
        results["test_summary"]["failed_routings"] = total_failed
        
        # Calculate average confidence and intent accuracy
        confidences = []
        intent_accuracies = []
        
        for intent_result in results["intent_analysis"].values():
            if intent_result["average_confidence"] > 0:
                confidences.append(intent_result["average_confidence"])
            intent_accuracies.append(intent_result["intent_accuracy"])
        
        if confidences:
            results["test_summary"]["average_confidence"] = sum(confidences) / len(confidences)
        
        if intent_accuracies:
            results["test_summary"]["intent_accuracy"] = sum(intent_accuracies) / len(intent_accuracies)
    
    def _analyze_intent_patterns(self, results: Dict[str, Any]):
        """Analyze intent-specific routing patterns."""
        patterns = {
            "intent_tool_preferences": {},
            "intent_platform_preferences": {},
            "intent_complexity_distribution": {},
            "intent_strategy_preferences": {},
            "intent_confidence_levels": {}
        }
        
        for intent, data in results["intent_analysis"].items():
            # Tool preferences by intent
            patterns["intent_tool_preferences"][intent] = data["tool_usage"]
            
            # Platform preferences by intent
            patterns["intent_platform_preferences"][intent] = data["platform_distribution"]
            
            # Complexity distribution by intent
            patterns["intent_complexity_distribution"][intent] = data["complexity_distribution"]
            
            # Strategy preferences by intent
            patterns["intent_strategy_preferences"][intent] = data["strategy_distribution"]
            
            # Confidence levels by intent
            patterns["intent_confidence_levels"][intent] = {
                "average_confidence": data["average_confidence"],
                "intent_accuracy": data["intent_accuracy"]
            }
        
        results["routing_patterns"] = patterns
    
    def _identify_intent_optimization_opportunities(self, results: Dict[str, Any]):
        """Identify intent-specific optimization opportunities."""
        opportunities = []
        
        # Check for low intent accuracy
        low_accuracy_threshold = 0.7
        for intent, data in results["intent_analysis"].items():
            if data["intent_accuracy"] < low_accuracy_threshold:
                opportunities.append({
                    "type": "low_intent_accuracy",
                    "intent": intent,
                    "description": f"{intent} intent accuracy is {data['intent_accuracy']:.1%} (below {low_accuracy_threshold:.0%})",
                    "impact": "high",
                    "accuracy": data["intent_accuracy"],
                    "sample_misclassifications": self._get_misclassified_samples(intent, data)
                })
        
        # Check for low confidence by intent
        low_confidence_threshold = 0.8
        for intent, data in results["intent_analysis"].items():
            if data["average_confidence"] < low_confidence_threshold:
                opportunities.append({
                    "type": "low_intent_confidence",
                    "intent": intent,
                    "description": f"{intent} intent has low confidence {data['average_confidence']:.2f} (below {low_confidence_threshold:.1f})",
                    "impact": "medium",
                    "confidence": data["average_confidence"]
                })
        
        # Check for routing failures by intent
        for intent, data in results["intent_analysis"].items():
            if data["failed_routings"] > 0:
                opportunities.append({
                    "type": "intent_routing_failures",
                    "intent": intent,
                    "description": f"{intent} intent has {data['failed_routings']} routing failures",
                    "impact": "critical",
                    "failure_count": data["failed_routings"]
                })
        
        # Check for tool usage imbalances by intent
        for intent, data in results["intent_analysis"].items():
            tools = data["tool_usage"]
            if len(tools) > 1:
                max_usage = max(tools.values())
                min_usage = min(tools.values())
                if max_usage > min_usage * 3:  # 3x imbalance
                    opportunities.append({
                        "type": "intent_tool_imbalance",
                        "intent": intent,
                        "description": f"{intent} intent has imbalanced tool usage",
                        "impact": "low",
                        "tool_distribution": tools
                    })
        
        results["optimization_opportunities"] = opportunities
    
    def _get_misclassified_samples(self, intent: str, data: Dict[str, Any]) -> List[str]:
        """Get sample misclassified queries for an intent."""
        misclassified = []
        for result in data["detailed_results"]:
            if not result["intent_correct"] and result["success"]:
                misclassified.append(result["query"])
        return misclassified[:3]  # Return first 3 samples

async def main():
    """Main function to run the comprehensive intent-based routing analysis."""
    print("🎯 Starting comprehensive intent-based routing analysis...")
    print("=" * 70)
    
    # Initialize test suite
    test_suite = IntentBasedTestSuite()
    
    # Run comprehensive analysis
    results = await test_suite.run_comprehensive_intent_analysis()
    
    # Print detailed summary
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE INTENT-BASED ROUTING ANALYSIS RESULTS")
    print("=" * 70)
    
    summary = results['test_summary']
    print(f"📈 Total Queries Tested: {summary['total_queries']}")
    print(f"🎯 Total Intent Types: {summary['total_intents']}")
    print(f"✅ Successful Routings: {summary['successful_routings']}")
    print(f"❌ Failed Routings: {summary['failed_routings']}")
    print(f"🎯 Success Rate: {(summary['successful_routings'] / summary['total_queries'] * 100):.1f}%")
    print(f"🔍 Average Confidence: {summary['average_confidence']:.3f}")
    print(f"🎯 Intent Accuracy: {summary['intent_accuracy']:.1%}")
    
    print("\n🎯 INTENT-SPECIFIC PERFORMANCE")
    print("-" * 50)
    for intent, data in results['intent_analysis'].items():
        success_rate = (data['successful_routings'] / data['total_queries'] * 100)
        print(f"  {intent.upper()}: {data['intent_accuracy']:.1%} accuracy, {data['average_confidence']:.2f} confidence, {success_rate:.1f}% success")
    
    print("\n🔧 OPTIMIZATION OPPORTUNITIES")
    print("-" * 50)
    if results['optimization_opportunities']:
        for i, opp in enumerate(results['optimization_opportunities'], 1):
            print(f"  {i}. {opp['type']} - {opp['intent']} ({opp['impact']} impact)")
            print(f"     {opp['description']}")
    else:
        print("  ✅ No major optimization opportunities identified!")
    
    print("\n🛠️ INTENT TOOL PREFERENCES")
    print("-" * 50)
    for intent, tools in results['routing_patterns']['intent_tool_preferences'].items():
        if tools:
            top_tool = max(tools.items(), key=lambda x: x[1])
            print(f"  {intent.upper()}: {top_tool[0]} ({top_tool[1]} uses)")
    
    print("\n🌐 INTENT PLATFORM PREFERENCES")
    print("-" * 50)
    for intent, platforms in results['routing_patterns']['intent_platform_preferences'].items():
        if platforms:
            top_platform = max(platforms.items(), key=lambda x: x[1]['usage_count'])
            print(f"  {intent.upper()}: {top_platform[0]} ({top_platform[1]['usage_count']} uses, {top_platform[1]['average_priority']:.2f} priority)")
    
    # Save detailed results
    output_file = '/home/dani/code/anime-mcp-server/tests/results/intent_based_routing_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Detailed results saved to: {output_file}")
    
    # Generate specific recommendations
    print("\n🎯 SPECIFIC RECOMMENDATIONS")
    print("-" * 50)
    
    # Analyze results and provide specific recommendations
    if summary['intent_accuracy'] < 0.85:
        print("  1. Improve intent classification - accuracy below 85%")
        
        # Find lowest performing intents
        lowest_accuracy = min(results['intent_analysis'].values(), key=lambda x: x['intent_accuracy'])
        print(f"     Focus on {lowest_accuracy['expected_intent']} intent ({lowest_accuracy['intent_accuracy']:.1%} accuracy)")
    
    if summary['average_confidence'] < 0.85:
        print("  2. Enhance confidence scoring - average below 85%")
        
        # Find lowest confidence intents
        lowest_confidence = min(results['intent_analysis'].values(), key=lambda x: x['average_confidence'])
        print(f"     Focus on {lowest_confidence['expected_intent']} intent ({lowest_confidence['average_confidence']:.2f} confidence)")
    
    if summary['failed_routings'] > 0:
        print("  3. Address routing failures")
        
        # Find intents with failures
        failed_intents = [intent for intent, data in results['intent_analysis'].items() if data['failed_routings'] > 0]
        if failed_intents:
            print(f"     Failed intents: {', '.join(failed_intents)}")
    
    # Check for intent-specific issues
    patterns = results['routing_patterns']
    
    # Check for tool preference imbalances
    tool_usage_counts = defaultdict(int)
    for intent_tools in patterns['intent_tool_preferences'].values():
        for tool, count in intent_tools.items():
            tool_usage_counts[tool] += count
    
    if tool_usage_counts:
        most_used = max(tool_usage_counts.items(), key=lambda x: x[1])
        least_used = min(tool_usage_counts.items(), key=lambda x: x[1])
        if most_used[1] > least_used[1] * 5:  # 5x difference
            print("  4. Rebalance tool usage across intents")
            print(f"     Most used: {most_used[0]} ({most_used[1]} uses)")
            print(f"     Least used: {least_used[0]} ({least_used[1]} uses)")
    
    print("\n🎉 Intent-based analysis complete! Check detailed results for comprehensive routing behavior by intent.")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())