"""
Comprehensive test suite for intelligent routing and query analyzer.

Tests routing decisions, query analysis, and optimization opportunities
using complex real-world queries from docs/query.txt plus ultra-complex scenarios.
"""

import pytest
import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import logging

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
        
        # Test query categories with expected routing patterns
        self.test_queries = {
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
                "I saw an anime years ago that involved underground tunnels, conspiracy theories, and a protagonist who communicates with a mysterious figure through dreams. It may have aired around 2010. Can you identify possible matches and rank them by likelihood?",
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
            "narrative_contextual": [
                "I watched an anime 5 years ago about time loops, check all sources",
                "Find me something like Death Note but darker according to AniSearch",
                "What does Anime-Planet say about anime similar to my favorites?",
                "Show me psychological anime that LiveChart says are currently airing",
            ],
            "cross_platform_verification": [
                "Is Chainsaw Man season 2 confirmed on any of the anime news sites?",
                "Check if this anime has different episode counts on different platforms",
                "Find discrepancies in scores between Eastern (MAL) and Western (Anime-Planet) sites",
            ],
            "ultra_specific": [
                "Get the exact airing time in JST from LiveChart for Jujutsu Kaisen",
                "What's the Anime News Network encyclopedia entry for Studio Trigger?",
                "Find anime that SIMKL users are currently watching the most",
                "Show me AniSearch's German voice actor list for Spy x Family",
            ]
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
                "Track the seasonal popularity of isekai anime from 2015-2024 across MAL, AniList, and Anime-Planet, identify which ones maintained consistent ratings vs those that declined, and correlate with their streaming platform availability",
            ],
            "semantic_similarity_complex": [
                "Find anime that have similar 'emotional resonance' to Your Name but different narrative structure, similar character chemistry to Toradora but in a non-school setting, and similar philosophical depth to Evangelion but without mecha elements",
                "I want anime that evoke the same 'nostalgic melancholy' as 5 Centimeters per Second, the same 'existential dread' as Paranoia Agent, and the same 'bittersweet hope' as Grave of the Fireflies, but in a modern urban setting with adult characters",
                "Search for anime that capture the 'atmospheric tension' of Death Note, the 'psychological complexity' of Monster, the 'moral ambiguity' of Psycho-Pass, but set in a historical period (not modern/futuristic)",
            ],
            "metadata_correlation": [
                "Analyze anime where the director previously worked on acclaimed series, the source material won literary awards, the voice acting includes at least 3 veteran seiyuu with 20+ years experience, and the animation studio has a track record of handling complex narratives",
                "Find anime where the average episode runtime is 22-24 minutes, the series aired across multiple seasons with consistent quality ratings, the merchandise sales exceeded 100 million yen, and the international distribution rights were acquired by major streaming platforms within 6 months of airing",
                "Cross-reference anime where the composer has worked on film soundtracks, the art director has a background in traditional animation, the series addresses mental health themes, and the character designer has previously worked on mature-rated content",
            ],
            "speculative_discovery": [
                "Based on my viewing history of [Monster, Steins;Gate, Death Note, Psycho-Pass], predict anime I haven't watched yet that would score 9+ for me, considering: my preference for psychological complexity, dislike of excessive fanservice, appreciation for mature themes, and tendency to prefer complete stories over ongoing series",
                "I'm looking for 'hidden gems' - anime that are highly rated by critics but have under 100,000 MAL members, preferably with unique animation styles, unconventional narratives, or experimental storytelling techniques, released between 2010-2023",
                "Find anime that are 'culturally significant' but may not have mainstream popularity - series that influenced the medium, addressed taboo subjects, or pioneered new techniques, with particular focus on works that gained retrospective appreciation",
            ],
            "technical_analysis": [
                "Identify anime with 'sakuga' (exceptional animation) episodes, cross-reference with the specific animators who worked on those episodes, and find other series where the same animators contributed to standout sequences",
                "Find anime where the color palettes and visual composition techniques demonstrate clear influence from specific art movements (impressionism, expressionism, surrealism), and analyze how these choices support the narrative themes",
                "Search for anime with innovative sound design and mixing techniques, where the audio significantly enhances the storytelling experience, and identify the sound engineers/composers responsible for these innovations",
            ]
        }
    
    async def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis on all query categories."""
        logger.info("Starting comprehensive routing analysis...")
        
        results = {
            "test_summary": {
                "total_queries": 0,
                "successful_routings": 0,
                "failed_routings": 0,
                "average_confidence": 0.0,
                "complexity_distribution": {},
                "intent_distribution": {},
                "strategy_distribution": {},
            },
            "category_analysis": {},
            "routing_patterns": {},
            "optimization_opportunities": [],
            "detailed_results": []
        }
        
        # Test all query categories
        all_queries = {**self.test_queries, **self.ultra_complex_queries}
        
        for category, queries in all_queries.items():
            logger.info(f"Testing category: {category}")
            category_results = await self._test_category(category, queries)
            results["category_analysis"][category] = category_results
            results["detailed_results"].extend(category_results["detailed_results"])
            results["test_summary"]["total_queries"] += len(queries)
        
        # Calculate summary statistics
        await self._calculate_summary_stats(results)
        
        # Identify routing patterns
        await self._analyze_routing_patterns(results)
        
        # Identify optimization opportunities
        await self._identify_optimization_opportunities(results)
        
        logger.info("Comprehensive routing analysis completed")
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
            "detailed_results": []
        }
        
        confidences = []
        
        for query in queries:
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
                intent = routing_decision.primary_tools[0] if routing_decision.primary_tools else "unknown"
                strategy = routing_decision.execution_strategy
                
                category_results["complexity_distribution"][complexity] = category_results["complexity_distribution"].get(complexity, 0) + 1
                category_results["intent_distribution"][intent] = category_results["intent_distribution"].get(intent, 0) + 1
                category_results["strategy_distribution"][strategy] = category_results["strategy_distribution"].get(strategy, 0) + 1
                
            except Exception as e:
                result = {
                    "query": query,
                    "routing_decision": None,
                    "analysis_result": None,
                    "success": False,
                    "error": str(e)
                }
                category_results["failed_routings"] += 1
                logger.error(f"Failed to route query '{query}': {e}")
            
            category_results["detailed_results"].append(result)
        
        # Calculate average confidence
        if confidences:
            category_results["average_confidence"] = sum(confidences) / len(confidences)
        
        return category_results
    
    async def _calculate_summary_stats(self, results: Dict[str, Any]):
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
    
    async def _analyze_routing_patterns(self, results: Dict[str, Any]):
        """Analyze routing patterns and identify common behaviors."""
        patterns = {
            "most_common_intents": {},
            "most_common_strategies": {},
            "most_common_complexities": {},
            "tool_usage_frequency": {},
            "platform_preferences": {},
            "confidence_by_complexity": {}
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
            
            # Platform preferences
            for platform, priority in routing["platform_priorities"].items():
                if platform not in patterns["platform_preferences"]:
                    patterns["platform_preferences"][platform] = []
                patterns["platform_preferences"][platform].append(priority)
            
            # Confidence by complexity
            complexity = routing["estimated_complexity"]
            if complexity not in patterns["confidence_by_complexity"]:
                patterns["confidence_by_complexity"][complexity] = []
            patterns["confidence_by_complexity"][complexity].append(routing["confidence"])
        
        # Calculate averages for platform preferences and confidence by complexity
        for platform in patterns["platform_preferences"]:
            priorities = patterns["platform_preferences"][platform]
            patterns["platform_preferences"][platform] = {
                "average_priority": sum(priorities) / len(priorities),
                "usage_count": len(priorities)
            }
        
        for complexity in patterns["confidence_by_complexity"]:
            confidences = patterns["confidence_by_complexity"][complexity]
            patterns["confidence_by_complexity"][complexity] = {
                "average_confidence": sum(confidences) / len(confidences),
                "query_count": len(confidences)
            }
        
        results["routing_patterns"] = patterns
    
    async def _identify_optimization_opportunities(self, results: Dict[str, Any]):
        """Identify areas for optimization based on test results."""
        opportunities = []
        
        # Check for low confidence routings
        low_confidence_threshold = 0.7
        low_confidence_queries = [
            result for result in results["detailed_results"] 
            if result["success"] and result["routing_decision"]["confidence"] < low_confidence_threshold
        ]
        
        if low_confidence_queries:
            opportunities.append({
                "type": "low_confidence_routing",
                "description": f"Found {len(low_confidence_queries)} queries with confidence < {low_confidence_threshold}",
                "impact": "high",
                "queries": [q["query"] for q in low_confidence_queries[:5]]  # Sample
            })
        
        # Check for failed routings
        failed_queries = [result for result in results["detailed_results"] if not result["success"]]
        if failed_queries:
            opportunities.append({
                "type": "routing_failures",
                "description": f"Found {len(failed_queries)} queries that failed to route",
                "impact": "critical",
                "errors": list(set([q["error"] for q in failed_queries]))
            })
        
        # Check for underutilized tools
        tool_usage = results["routing_patterns"]["tool_usage_frequency"]
        total_queries = results["test_summary"]["total_queries"]
        underutilized_tools = [
            tool for tool, count in tool_usage.items() 
            if count < total_queries * 0.05  # Less than 5% usage
        ]
        
        if underutilized_tools:
            opportunities.append({
                "type": "underutilized_tools",
                "description": f"Found {len(underutilized_tools)} tools used in <5% of queries",
                "impact": "medium",
                "tools": underutilized_tools
            })
        
        # Check for strategy distribution imbalances
        strategy_dist = results["test_summary"]["strategy_distribution"]
        if len(strategy_dist) > 1:
            max_strategy = max(strategy_dist.values())
            min_strategy = min(strategy_dist.values())
            if max_strategy > min_strategy * 3:  # 3x imbalance
                opportunities.append({
                    "type": "strategy_imbalance",
                    "description": "Execution strategy distribution is imbalanced",
                    "impact": "medium",
                    "distribution": strategy_dist
                })
        
        # Check for complexity handling effectiveness
        complexity_confidence = results["routing_patterns"]["confidence_by_complexity"]
        for complexity, stats in complexity_confidence.items():
            if stats["average_confidence"] < 0.75:
                opportunities.append({
                    "type": "complexity_handling",
                    "description": f"Low confidence ({stats['average_confidence']:.2f}) for {complexity} queries",
                    "impact": "high",
                    "complexity": complexity,
                    "query_count": stats["query_count"]
                })
        
        results["optimization_opportunities"] = opportunities

async def run_routing_analysis():
    """Main function to run the comprehensive routing analysis."""
    print("🔍 Starting comprehensive intelligent routing analysis...")
    
    # Initialize test suite
    test_suite = RoutingTestSuite()
    
    # Run comprehensive analysis
    results = await test_suite.run_comprehensive_analysis()
    
    # Print summary
    print("\n📊 ROUTING ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total Queries Tested: {results['test_summary']['total_queries']}")
    print(f"Successful Routings: {results['test_summary']['successful_routings']}")
    print(f"Failed Routings: {results['test_summary']['failed_routings']}")
    print(f"Average Confidence: {results['test_summary']['average_confidence']:.3f}")
    
    print("\n🎯 COMPLEXITY DISTRIBUTION")
    for complexity, count in results['test_summary']['complexity_distribution'].items():
        print(f"{complexity}: {count} queries")
    
    print("\n🛠️ STRATEGY DISTRIBUTION")
    for strategy, count in results['test_summary']['strategy_distribution'].items():
        print(f"{strategy}: {count} queries")
    
    print("\n📈 OPTIMIZATION OPPORTUNITIES")
    for i, opp in enumerate(results['optimization_opportunities'], 1):
        print(f"{i}. {opp['type']} ({opp['impact']} impact)")
        print(f"   {opp['description']}")
    
    # Save detailed results
    with open('/home/dani/code/anime-mcp-server/tests/results/routing_analysis_detailed.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Detailed results saved to: tests/results/routing_analysis_detailed.json")
    
    return results

if __name__ == "__main__":
    # Run the analysis
    asyncio.run(run_routing_analysis())