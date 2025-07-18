#!/usr/bin/env python3
"""
Comprehensive test script for intelligent routing and query analyzer.

Tests routing decisions, query analysis, and optimization opportunities
using complex real-world queries from docs/query.txt plus ultra-complex scenarios.
"""

import asyncio
import json
import sys
import os
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our routing and analysis components
from src.langgraph.intelligent_router import IntelligentRouter, QueryIntent, RoutingDecision
from src.langgraph.query_analyzer import QueryAnalyzer
from src.config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RoutingTestSuite:
    """Comprehensive test suite for intelligent routing analysis."""
    
    def __init__(self):
        self.router = IntelligentRouter()
        self.analyzer = QueryAnalyzer()
        
        # Test queries from docs/query.txt
        self.docs_queries = {
            "simple_search": [
                "Show me anime similar to Death Note",
                "Find anime where the main character is overpowered",
                "List anime released in 2021",
                "Search for romance anime with high ratings",
                "What are the top 5 shonen anime?",
            ],
            "medium_complexity": [
                "Find anime with a female protagonist who is a bounty hunter",
                "Show anime like Naruto but with a darker tone", 
                "List anime set in post-apocalyptic worlds with less than 24 episodes",
                "Which anime have a musical theme and are set in high school?",
                "Recommend anime with psychological elements and minimal action",
            ],
            "high_complexity": [
                "Find anime featuring AI or sentient robots that explore human emotions",
                "I want anime like Steins;Gate but with a female lead and more romance",
                "Search for anime based on novels, not manga, released after 2010",
                "Give me anime where the main character dies and is reincarnated in a fantasy world, but the twist is they become the villain",
                "Which anime have stories that span multiple generations of the same family?",
            ],
            "very_complex": [
                "I'm looking for an anime I watched years ago. The protagonist starts out as a hero but eventually becomes the antagonist due to moral conflict. The art style was realistic, and the soundtrack had jazz influences. I think it aired between 2006 and 2011.",
                "Find anime that blend sci-fi and slice-of-life, where the protagonist interacts with an alien or time traveler in a slow-paced, emotionally rich story. Preferably under 20 episodes.",
                "Search for anime where two timelines run in parallel and intersect in the final episode to reveal a tragic connection between the characters.",
                "Recommend anime with strong social commentary, minimal violence, a realistic art style, and no fantasy elements. Something akin to a political or economic thriller.",
            ],
            "multi_source_queries": [
                "Compare Death Note's ratings across MAL, AniList, and Anime-Planet",
                "Show me how LiveChart describes Frieren vs what AniSearch says",
                "What's the German synopsis from AniSearch for Attack on Titan?",
                "Get me the streaming info from Kitsu but the rating from MAL",
            ],
            "time_based_conditional": [
                "Find anime airing tomorrow on Crunchyroll according to LiveChart",
                "Show me what's trending on AniList but has high scores on MAL",
                "Which anime finished airing last week according to AnimeSchedule?",
                "Find anime that Kitsu says is on Netflix but MAL rates above 8",
            ],
            "cross_platform_verification": [
                "Is Chainsaw Man season 2 confirmed on any of the anime news sites?",
                "Check if this anime has different episode counts on different platforms",
                "Find discrepancies in scores between Eastern (MAL) and Western (Anime-Planet) sites",
            ],
        }
        
        # Add ultra-complex queries for comprehensive testing
        self.ultra_complex_queries = {
            "multi_dimensional_analysis": [
                "Find anime that simultaneously satisfy: (1) rated 8+ on MAL, (2) available on Netflix according to Kitsu, (3) has psychological themes per AniSearch, (4) aired 2015-2020, (5) has under 24 episodes, (6) features adult protagonists, (7) has minimal fanservice, (8) received critical acclaim in Western reviews",
                "I need anime that perfectly blend: sci-fi worldbuilding like Ghost in the Shell, character development like Monster, visual aesthetics like Akira, philosophical depth like Serial Experiments Lain, but in a slice-of-life setting with romance elements and a female protagonist who works in technology",
                "Search for anime where: the opening theme is by Yoko Kanno OR the studio is Madhouse OR A-1 Pictures, the genre includes 'thriller' but excludes 'supernatural', the protagonist has a morally ambiguous profession, the series commentary addresses social inequality, and it has a conclusive ending (not open-ended)",
            ],
            "temporal_cross_reference": [
                "Find anime that were popular on MAL during 2019-2020 but are now trending on AniList in 2024, compare their rating evolution, and identify which ones gained Western popularity through Netflix or Crunchyroll releases",
                "I watched an anime 10-15 years ago: mecha setting, protagonist pilots giant robots, has philosophical undertones about human consciousness, aired during fall/winter season, opening had orchestral music, and I remember a specific scene where the protagonist questions reality in a cockpit. Cross-reference all platforms for matches",
            ],
            "semantic_similarity_complex": [
                "Find anime that have similar 'emotional resonance' to Your Name but different narrative structure, similar character chemistry to Toradora but in a non-school setting, and similar philosophical depth to Evangelion but without mecha elements",
                "I want anime that evoke the same 'nostalgic melancholy' as 5 Centimeters per Second, the same 'existential dread' as Paranoia Agent, and the same 'bittersweet hope' as Grave of the Fireflies, but in a modern urban setting with adult characters",
            ],
            "cross_platform_intelligence": [
                "Based on my viewing history of [Monster, Steins;Gate, Death Note, Psycho-Pass], predict anime I haven't watched yet that would score 9+ for me, considering: my preference for psychological complexity, dislike of excessive fanservice, appreciation for mature themes, cross-reference across all platforms",
                "I'm looking for 'hidden gems' - anime that are highly rated by critics but have under 100,000 MAL members, with unique animation styles, unconventional narratives, check AniList and Anime-Planet for additional ratings",
            ],
            "technical_metadata": [
                "Find anime where the color palettes and visual composition demonstrate clear influence from specific art movements (impressionism, expressionism, surrealism), and analyze how these choices support the narrative themes using cross-platform reviews",
                "Search for anime with innovative sound design and mixing techniques, where the audio significantly enhances the storytelling experience, and identify the composers responsible using AniSearch and MAL data",
            ]
        }
    
    async def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis on all query categories."""
        print("🔍 Starting comprehensive routing analysis...")
        
        results = {
            "test_summary": {
                "total_queries": 0,
                "successful_routings": 0,
                "failed_routings": 0,
                "average_confidence": 0.0,
                "complexity_distribution": {},
                "intent_distribution": {},
                "strategy_distribution": {},
                "platform_distribution": {},
            },
            "category_analysis": {},
            "routing_patterns": {},
            "optimization_opportunities": [],
            "detailed_results": []
        }
        
        # Test all query categories
        all_queries = {**self.docs_queries, **self.ultra_complex_queries}
        
        for category, queries in all_queries.items():
            print(f"📝 Testing category: {category} ({len(queries)} queries)")
            category_results = await self._test_category(category, queries)
            results["category_analysis"][category] = category_results
            results["detailed_results"].extend(category_results["detailed_results"])
            results["test_summary"]["total_queries"] += len(queries)
        
        # Calculate summary statistics
        self._calculate_summary_stats(results)
        
        # Analyze routing patterns
        self._analyze_routing_patterns(results)
        
        # Identify optimization opportunities
        self._identify_optimization_opportunities(results)
        
        print("✅ Comprehensive routing analysis completed")
        return results
    
    async def _test_category(self, category: str, queries: List[str]) -> Dict[str, Any]:
        """Test a specific category of queries."""
        category_results = {
            "category": category,
            "total_queries": len(queries),
            "successful_routings": 0,
            "failed_routings": 0,
            "average_confidence": 0.0,
            "complexity_distribution": {},
            "intent_distribution": {},
            "strategy_distribution": {},
            "platform_distribution": {},
            "detailed_results": []
        }
        
        confidences = []
        
        for i, query in enumerate(queries):
            print(f"  🔄 Testing query {i+1}/{len(queries)}: {query[:60]}...")
            
            try:
                # Test intelligent router
                routing_decision = await self.router.route_query(query)
                
                # Test query analyzer
                analysis_result = await self.analyzer.analyze_query(query)
                
                # Record results
                result = {
                    "query": query,
                    "routing_decision": asdict(routing_decision),
                    "analysis_result": analysis_result,
                    "success": True,
                    "error": None
                }
                
                category_results["successful_routings"] += 1
                confidences.append(routing_decision.confidence)
                
                # Track distributions
                complexity = routing_decision.estimated_complexity
                primary_intent = routing_decision.primary_tools[0] if routing_decision.primary_tools else "unknown"
                strategy = routing_decision.execution_strategy
                
                category_results["complexity_distribution"][complexity] = category_results["complexity_distribution"].get(complexity, 0) + 1
                category_results["intent_distribution"][primary_intent] = category_results["intent_distribution"].get(primary_intent, 0) + 1
                category_results["strategy_distribution"][strategy] = category_results["strategy_distribution"].get(strategy, 0) + 1
                
                # Track platform usage
                for platform, priority in routing_decision.platform_priorities.items():
                    if platform not in category_results["platform_distribution"]:
                        category_results["platform_distribution"][platform] = []
                    category_results["platform_distribution"][platform].append(priority)
                
                print(f"    ✅ Success: {complexity} complexity, {len(routing_decision.primary_tools)} primary tools, {routing_decision.confidence:.2f} confidence")
                
            except Exception as e:
                result = {
                    "query": query,
                    "routing_decision": None,
                    "analysis_result": None,
                    "success": False,
                    "error": str(e)
                }
                category_results["failed_routings"] += 1
                print(f"    ❌ Failed: {e}")
                logger.error(f"Failed to route query '{query}': {e}")
            
            category_results["detailed_results"].append(result)
        
        # Calculate average confidence
        if confidences:
            category_results["average_confidence"] = sum(confidences) / len(confidences)
        
        # Calculate average platform priorities
        for platform in category_results["platform_distribution"]:
            priorities = category_results["platform_distribution"][platform]
            category_results["platform_distribution"][platform] = {
                "average_priority": sum(priorities) / len(priorities),
                "usage_count": len(priorities)
            }
        
        return category_results
    
    def _calculate_summary_stats(self, results: Dict[str, Any]):
        """Calculate overall summary statistics."""
        total_successful = sum(cat["successful_routings"] for cat in results["category_analysis"].values())
        total_failed = sum(cat["failed_routings"] for cat in results["category_analysis"].values())
        
        results["test_summary"]["successful_routings"] = total_successful
        results["test_summary"]["failed_routings"] = total_failed
        
        # Calculate average confidence across all categories
        all_confidences = []
        for cat_result in results["category_analysis"].values():
            if cat_result["average_confidence"] > 0:
                all_confidences.append(cat_result["average_confidence"])
        
        if all_confidences:
            results["test_summary"]["average_confidence"] = sum(all_confidences) / len(all_confidences)
        
        # Aggregate distributions
        for cat_result in results["category_analysis"].values():
            for complexity, count in cat_result["complexity_distribution"].items():
                results["test_summary"]["complexity_distribution"][complexity] = results["test_summary"]["complexity_distribution"].get(complexity, 0) + count
            
            for intent, count in cat_result["intent_distribution"].items():
                results["test_summary"]["intent_distribution"][intent] = results["test_summary"]["intent_distribution"].get(intent, 0) + count
            
            for strategy, count in cat_result["strategy_distribution"].items():
                results["test_summary"]["strategy_distribution"][strategy] = results["test_summary"]["strategy_distribution"].get(strategy, 0) + count
            
            for platform, data in cat_result["platform_distribution"].items():
                if platform not in results["test_summary"]["platform_distribution"]:
                    results["test_summary"]["platform_distribution"][platform] = {
                        "total_usage": 0,
                        "average_priority": 0.0
                    }
                results["test_summary"]["platform_distribution"][platform]["total_usage"] += data["usage_count"]
                results["test_summary"]["platform_distribution"][platform]["average_priority"] = (
                    results["test_summary"]["platform_distribution"][platform]["average_priority"] + data["average_priority"]
                ) / 2
    
    def _analyze_routing_patterns(self, results: Dict[str, Any]):
        """Analyze routing patterns and identify common behaviors."""
        patterns = {
            "tool_usage_frequency": {},
            "intent_accuracy": {},
            "complexity_confidence_correlation": {},
            "platform_effectiveness": {},
            "strategy_effectiveness": {}
        }
        
        # Analyze detailed results
        for result in results["detailed_results"]:
            if not result["success"]:
                continue
                
            routing = result["routing_decision"]
            
            # Tool usage frequency
            for tool in routing["primary_tools"]:
                patterns["tool_usage_frequency"][tool] = patterns["tool_usage_frequency"].get(tool, 0) + 1
            
            for tool in routing["secondary_tools"]:
                patterns["tool_usage_frequency"][tool] = patterns["tool_usage_frequency"].get(tool, 0) + 1
            
            # Complexity vs confidence correlation
            complexity = routing["estimated_complexity"]
            confidence = routing["confidence"]
            if complexity not in patterns["complexity_confidence_correlation"]:
                patterns["complexity_confidence_correlation"][complexity] = []
            patterns["complexity_confidence_correlation"][complexity].append(confidence)
            
            # Strategy effectiveness
            strategy = routing["execution_strategy"]
            if strategy not in patterns["strategy_effectiveness"]:
                patterns["strategy_effectiveness"][strategy] = []
            patterns["strategy_effectiveness"][strategy].append(confidence)
        
        # Calculate averages
        for complexity in patterns["complexity_confidence_correlation"]:
            confidences = patterns["complexity_confidence_correlation"][complexity]
            patterns["complexity_confidence_correlation"][complexity] = {
                "average_confidence": sum(confidences) / len(confidences),
                "query_count": len(confidences)
            }
        
        for strategy in patterns["strategy_effectiveness"]:
            confidences = patterns["strategy_effectiveness"][strategy]
            patterns["strategy_effectiveness"][strategy] = {
                "average_confidence": sum(confidences) / len(confidences),
                "usage_count": len(confidences)
            }
        
        results["routing_patterns"] = patterns
    
    def _identify_optimization_opportunities(self, results: Dict[str, Any]):
        """Identify areas for optimization based on test results."""
        opportunities = []
        
        # Check for low confidence routings
        low_confidence_threshold = 0.75
        low_confidence_queries = [
            result for result in results["detailed_results"] 
            if result["success"] and result["routing_decision"]["confidence"] < low_confidence_threshold
        ]
        
        if low_confidence_queries:
            opportunities.append({
                "type": "low_confidence_routing",
                "description": f"Found {len(low_confidence_queries)} queries with confidence < {low_confidence_threshold}",
                "impact": "high",
                "count": len(low_confidence_queries),
                "sample_queries": [q["query"][:100] for q in low_confidence_queries[:3]]
            })
        
        # Check for failed routings
        failed_queries = [result for result in results["detailed_results"] if not result["success"]]
        if failed_queries:
            opportunities.append({
                "type": "routing_failures",
                "description": f"Found {len(failed_queries)} queries that failed to route",
                "impact": "critical",
                "count": len(failed_queries),
                "errors": list(set([q["error"] for q in failed_queries]))
            })
        
        # Check for underutilized tools
        tool_usage = results["routing_patterns"]["tool_usage_frequency"]
        total_successful = results["test_summary"]["successful_routings"]
        underutilized_tools = [
            tool for tool, count in tool_usage.items() 
            if count < total_successful * 0.05  # Less than 5% usage
        ]
        
        if underutilized_tools:
            opportunities.append({
                "type": "underutilized_tools",
                "description": f"Found {len(underutilized_tools)} tools used in <5% of successful queries",
                "impact": "medium",
                "tools": underutilized_tools[:10]  # Show first 10
            })
        
        # Check for complexity handling effectiveness
        complexity_stats = results["routing_patterns"]["complexity_confidence_correlation"]
        for complexity, stats in complexity_stats.items():
            if stats["average_confidence"] < 0.75:
                opportunities.append({
                    "type": "complexity_handling_weakness",
                    "description": f"Low confidence ({stats['average_confidence']:.2f}) for {complexity} queries",
                    "impact": "high",
                    "complexity": complexity,
                    "query_count": stats["query_count"]
                })
        
        # Check for strategy effectiveness
        strategy_stats = results["routing_patterns"]["strategy_effectiveness"]
        if len(strategy_stats) > 1:
            avg_confidences = [stats["average_confidence"] for stats in strategy_stats.values()]
            max_conf = max(avg_confidences)
            min_conf = min(avg_confidences)
            
            if max_conf - min_conf > 0.15:  # 15% difference
                opportunities.append({
                    "type": "strategy_effectiveness_gap",
                    "description": f"Large confidence gap ({max_conf - min_conf:.2f}) between execution strategies",
                    "impact": "medium",
                    "strategy_performance": {
                        strategy: stats["average_confidence"] 
                        for strategy, stats in strategy_stats.items()
                    }
                })
        
        results["optimization_opportunities"] = opportunities

async def main():
    """Main function to run the comprehensive routing analysis."""
    print("🚀 Starting comprehensive intelligent routing analysis...")
    print("=" * 60)
    
    # Initialize test suite
    test_suite = RoutingTestSuite()
    
    # Run comprehensive analysis
    results = await test_suite.run_comprehensive_analysis()
    
    # Print detailed summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE ROUTING ANALYSIS RESULTS")
    print("=" * 60)
    
    summary = results['test_summary']
    print(f"📈 Total Queries Tested: {summary['total_queries']}")
    print(f"✅ Successful Routings: {summary['successful_routings']}")
    print(f"❌ Failed Routings: {summary['failed_routings']}")
    print(f"🎯 Success Rate: {(summary['successful_routings'] / summary['total_queries'] * 100):.1f}%")
    print(f"🔍 Average Confidence: {summary['average_confidence']:.3f}")
    
    print("\n🎲 COMPLEXITY DISTRIBUTION")
    print("-" * 30)
    for complexity, count in summary['complexity_distribution'].items():
        percentage = (count / summary['total_queries'] * 100)
        print(f"  {complexity.capitalize()}: {count} queries ({percentage:.1f}%)")
    
    print("\n🛠️ EXECUTION STRATEGY DISTRIBUTION")
    print("-" * 30)
    for strategy, count in summary['strategy_distribution'].items():
        percentage = (count / summary['total_queries'] * 100)
        print(f"  {strategy.capitalize()}: {count} queries ({percentage:.1f}%)")
    
    print("\n🌐 PLATFORM USAGE PATTERNS")
    print("-" * 30)
    for platform, data in summary['platform_distribution'].items():
        print(f"  {platform}: {data['total_usage']} uses, avg priority: {data['average_priority']:.2f}")
    
    print("\n🔧 OPTIMIZATION OPPORTUNITIES")
    print("-" * 30)
    if results['optimization_opportunities']:
        for i, opp in enumerate(results['optimization_opportunities'], 1):
            print(f"  {i}. {opp['type']} ({opp['impact']} impact)")
            print(f"     {opp['description']}")
            if 'sample_queries' in opp:
                print(f"     Sample: {opp['sample_queries'][0][:80]}...")
    else:
        print("  ✅ No major optimization opportunities identified!")
    
    print("\n📋 CATEGORY PERFORMANCE BREAKDOWN")
    print("-" * 30)
    for category, data in results['category_analysis'].items():
        success_rate = (data['successful_routings'] / data['total_queries'] * 100)
        print(f"  {category}: {success_rate:.1f}% success, {data['average_confidence']:.2f} avg confidence")
    
    # Save detailed results
    output_file = '/home/dani/code/anime-mcp-server/tests/results/routing_analysis_detailed.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\n📁 Detailed results saved to: {output_file}")
    
    # Generate recommendations
    print("\n🎯 RECOMMENDATIONS")
    print("-" * 30)
    
    # Analyze results and provide specific recommendations
    if summary['average_confidence'] < 0.8:
        print("  1. Consider improving query intent classification - confidence is below 80%")
    
    if summary['failed_routings'] > 0:
        print("  2. Address routing failures - some queries cannot be properly routed")
    
    complexity_stats = results['routing_patterns']['complexity_confidence_correlation']
    if 'complex' in complexity_stats and complexity_stats['complex']['average_confidence'] < 0.75:
        print("  3. Enhance handling of complex queries - they show lower confidence")
    
    tool_usage = results['routing_patterns']['tool_usage_frequency']
    if len(tool_usage) > 0:
        most_used = max(tool_usage.items(), key=lambda x: x[1])
        least_used = min(tool_usage.items(), key=lambda x: x[1])
        if most_used[1] > least_used[1] * 5:  # 5x difference
            print("  4. Consider rebalancing tool usage - some tools are heavily underutilized")
    
    print("\n🎉 Analysis complete! Check detailed results for in-depth routing behavior.")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())