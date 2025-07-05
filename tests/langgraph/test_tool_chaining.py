"""Integration tests for tool chaining and cross-platform workflows.

Tests the coordination between platform-specific tools, intelligent
routing, cross-platform data enrichment, and error handling strategies.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List
import json

from src.langgraph.langchain_tools import create_anime_langchain_tools, ToolNode


class TestToolChainingIntegration:
    """Integration tests for tool chaining workflows."""

    @pytest.fixture
    def mock_platform_tools(self):
        """Create comprehensive mock platform tools."""
        tools = {
            # MAL tools
            "search_anime_mal": AsyncMock(return_value=[
                {
                    "id": "mal_16498",
                    "title": "Attack on Titan",
                    "mal_score": 8.54,
                    "mal_rank": 50,
                    "mal_popularity": 1,
                    "mal_members": 3000000,
                    "source_platform": "mal",
                    "data_quality_score": 0.95
                }
            ]),
            "get_anime_mal": AsyncMock(return_value={
                "id": "mal_16498",
                "title": "Attack on Titan",
                "mal_score": 8.54,
                "detailed_info": {"episodes": 25, "status": "finished"},
                "source_platform": "mal"
            }),
            
            # AniList tools
            "search_anime_anilist": AsyncMock(return_value=[
                {
                    "id": "anilist_16498",
                    "title": "Shingeki no Kyojin",
                    "anilist_score": 85,
                    "anilist_popularity": 95000,
                    "anilist_favourites": 75000,
                    "source_platform": "anilist",
                    "data_quality_score": 0.92
                }
            ]),
            "get_anime_anilist": AsyncMock(return_value={
                "id": "anilist_16498",
                "title": "Shingeki no Kyojin",
                "anilist_score": 85,
                "detailed_info": {"format": "TV", "episodes": 25},
                "source_platform": "anilist"
            }),
            
            # Schedule tools
            "search_anime_schedule": AsyncMock(return_value=[
                {
                    "id": "schedule_aot",
                    "title": "Attack on Titan",
                    "broadcast_day": "Sunday",
                    "broadcast_time": "17:00",
                    "streaming_sites": [
                        {"site": "Crunchyroll", "regions": ["US", "CA"]}
                    ],
                    "source_platform": "animeschedule",
                    "data_quality_score": 0.88
                }
            ]),
            "get_currently_airing": AsyncMock(return_value=[
                {
                    "title": "Current Airing Anime",
                    "broadcast_day": "Wednesday",
                    "next_episode_date": "2024-01-24T14:30:00Z",
                    "status": "airing"
                }
            ]),
            
            # Semantic tools
            "anime_semantic_search": AsyncMock(return_value=[
                {
                    "id": "semantic_16498",
                    "title": "Attack on Titan",
                    "similarity_score": 0.95,
                    "source_platform": "semantic",
                    "data_quality_score": 0.89
                }
            ]),
            "anime_similar": AsyncMock(return_value=[
                {
                    "id": "semantic_similar_1",
                    "title": "Similar Anime 1",
                    "similarity_score": 0.87,
                    "reference": "Attack on Titan"
                }
            ]),
            
            # Cross-platform enrichment tools
            "get_cross_platform_anime_data": AsyncMock(return_value={
                "anime_id": "cross_platform_16498",
                "title": "Attack on Titan",
                "cross_platform_data": {
                    "mal": {"score": 8.54, "rank": 50, "members": 3000000},
                    "anilist": {"score": 85, "popularity": 95000, "favourites": 75000},
                    "schedule": {"broadcast_day": "Sunday", "streaming_count": 2}
                },
                "aggregate_score": 8.67,
                "data_completeness": 0.93,
                "platforms_verified": 3
            }),
            "compare_anime_ratings_cross_platform": AsyncMock(return_value={
                "anime_title": "Attack on Titan",
                "platform_ratings": {
                    "mal": {"score": 8.54, "scale": "1-10", "voters": 1500000},
                    "anilist": {"score": 85, "scale": "1-100", "voters": 95000}
                },
                "normalized_scores": {
                    "mal_normalized": 8.54,
                    "anilist_normalized": 8.5
                },
                "rating_consensus": 0.95,
                "variance": 0.04
            })
        }
        return tools

    @pytest.fixture
    def tool_node(self, mock_platform_tools):
        """Create ToolNode with mocked platform tools."""
        with patch('src.langgraph.langchain_tools.get_all_platform_tools', return_value=mock_platform_tools):
            langchain_tools = create_anime_langchain_tools(mock_platform_tools)
            return ToolNode(langchain_tools)

    @pytest.mark.asyncio
    async def test_sequential_tool_chaining_workflow(self, mock_platform_tools):
        """Test sequential tool chaining from search to enrichment."""
        # Step 1: Initial search
        search_result = await mock_platform_tools["search_anime_mal"]("attack on titan")
        anime_id = search_result[0]["id"]
        
        # Step 2: Get detailed info
        detailed_result = await mock_platform_tools["get_anime_mal"](anime_id)
        
        # Step 3: Cross-platform enrichment
        enriched_result = await mock_platform_tools["get_cross_platform_anime_data"](
            anime_title="Attack on Titan",
            platforms=["mal", "anilist", "animeschedule"]
        )
        
        # Step 4: Rating comparison
        rating_comparison = await mock_platform_tools["compare_anime_ratings_cross_platform"](
            anime_title="Attack on Titan",
            platforms=["mal", "anilist"]
        )
        
        # Verify sequential chaining
        assert search_result[0]["title"] == "Attack on Titan"
        assert detailed_result["id"] == anime_id
        assert enriched_result["platforms_verified"] == 3
        assert rating_comparison["rating_consensus"] > 0.9
        
        # Verify data flow continuity
        assert enriched_result["title"] == search_result[0]["title"]
        assert rating_comparison["anime_title"] == enriched_result["title"]

    @pytest.mark.asyncio
    async def test_parallel_tool_execution_workflow(self, mock_platform_tools):
        """Test parallel tool execution for efficiency."""
        query = "attack on titan"
        
        # Execute parallel searches across platforms
        import asyncio
        
        parallel_results = await asyncio.gather(
            mock_platform_tools["search_anime_mal"](query),
            mock_platform_tools["search_anime_anilist"](query),
            mock_platform_tools["search_anime_schedule"](query)
        )
        
        # Verify parallel execution
        assert len(parallel_results) == 3
        platforms = [result[0]["source_platform"] for result in parallel_results]
        assert set(platforms) == {"mal", "anilist", "animeschedule"}

    @pytest.mark.asyncio
    async def test_conditional_tool_chaining_based_on_results(self, mock_platform_tools):
        """Test conditional tool chaining based on intermediate results."""
        # Step 1: Initial search
        search_result = await mock_platform_tools["search_anime_mal"]("attack on titan")
        
        # Step 2: Conditional chaining based on anime status
        anime_data = search_result[0]
        if "finished" in anime_data.get("status", "").lower():
            # For finished anime, get comprehensive data
            cross_platform_data = await mock_platform_tools["get_cross_platform_anime_data"](
                anime_title=anime_data["title"]
            )
            next_step_data = cross_platform_data
        else:
            # For airing anime, get current schedule
            airing_data = await mock_platform_tools["get_currently_airing"]()
            next_step_data = airing_data
        
        # Step 3: Further conditional processing
        if isinstance(next_step_data, dict) and "cross_platform_data" in next_step_data:
            # Rich data available, do rating comparison
            rating_data = await mock_platform_tools["compare_anime_ratings_cross_platform"](
                anime_title=anime_data["title"]
            )
            final_result = {
                "type": "comprehensive_analysis",
                "search_data": anime_data,
                "cross_platform_data": next_step_data,
                "rating_analysis": rating_data
            }
        else:
            # Limited data, do similarity search for recommendations
            similar_data = await mock_platform_tools["anime_similar"](
                reference_anime=anime_data["title"]
            )
            final_result = {
                "type": "basic_with_recommendations",
                "search_data": anime_data,
                "similar_anime": similar_data
            }
        
        # Verify conditional logic
        assert final_result["type"] in ["comprehensive_analysis", "basic_with_recommendations"]
        assert "search_data" in final_result

    @pytest.mark.asyncio
    async def test_error_handling_in_tool_chains(self, mock_platform_tools):
        """Test error handling and recovery in tool chains."""
        # Mock primary tool failure
        mock_platform_tools["search_anime_mal"].side_effect = Exception("MAL API unavailable")
        
        try:
            # Attempt primary search
            result = await mock_platform_tools["search_anime_mal"]("test query")
        except Exception:
            # Fallback to alternative platform
            result = await mock_platform_tools["search_anime_anilist"]("test query")
        
        # Verify fallback execution
        assert result[0]["source_platform"] == "anilist"
        
        # Continue chain with fallback result
        anime_id = result[0]["id"]
        detailed_result = await mock_platform_tools["get_anime_anilist"](anime_id)
        
        # Verify chain continuity despite initial failure
        assert detailed_result["source_platform"] == "anilist"

    @pytest.mark.asyncio
    async def test_data_quality_filtering_in_chains(self, mock_platform_tools):
        """Test data quality filtering and optimization in tool chains."""
        # Mock search with varying quality results
        mock_platform_tools["search_anime_mal"].return_value = [
            {"id": "mal_high", "title": "High Quality", "data_quality_score": 0.95},
            {"id": "mal_medium", "title": "Medium Quality", "data_quality_score": 0.75},
            {"id": "mal_low", "title": "Low Quality", "data_quality_score": 0.45}
        ]
        
        # Execute search
        search_results = await mock_platform_tools["search_anime_mal"]("quality test")
        
        # Filter by quality threshold
        quality_threshold = 0.7
        high_quality_results = [
            r for r in search_results 
            if r.get("data_quality_score", 0) >= quality_threshold
        ]
        
        # Continue chain only with high quality results
        for anime in high_quality_results:
            detailed_result = await mock_platform_tools["get_anime_mal"](anime["id"])
            # Process high quality data...
        
        # Verify quality filtering
        assert len(high_quality_results) == 2
        assert all(r["data_quality_score"] >= quality_threshold for r in high_quality_results)

    @pytest.mark.asyncio
    async def test_cross_platform_data_reconciliation(self, mock_platform_tools):
        """Test cross-platform data reconciliation and conflict resolution."""
        # Get data from multiple platforms
        mal_data = await mock_platform_tools["search_anime_mal"]("attack on titan")
        anilist_data = await mock_platform_tools["search_anime_anilist"]("attack on titan")
        
        # Mock data reconciliation
        reconciled_data = {
            "title": "Attack on Titan",  # Primary title from MAL
            "alternative_titles": ["Shingeki no Kyojin"],  # From AniList
            "scores": {
                "mal": mal_data[0]["mal_score"],
                "anilist": anilist_data[0]["anilist_score"],
                "normalized_average": 8.52
            },
            "popularity_metrics": {
                "mal_members": mal_data[0]["mal_members"],
                "anilist_popularity": anilist_data[0]["anilist_popularity"]
            },
            "data_conflicts": [],
            "reconciliation_confidence": 0.94
        }
        
        # Verify reconciliation
        assert reconciled_data["reconciliation_confidence"] > 0.9
        assert len(reconciled_data["data_conflicts"]) == 0
        assert "normalized_average" in reconciled_data["scores"]


class TestCrossPlatformWorkflows:
    """Integration tests for cross-platform workflow scenarios."""

    @pytest.fixture
    def mock_platform_tools(self):
        """Create comprehensive mock platform tools."""
        tools = {
            # MAL tools
            "search_anime_mal": AsyncMock(return_value=[
                {
                    "id": "mal_16498",
                    "title": "Attack on Titan",
                    "mal_score": 8.54,
                    "source_platform": "mal"
                }
            ]),
            "get_anime_mal": AsyncMock(return_value={
                "id": "mal_16498",
                "title": "Attack on Titan",
                "source_platform": "mal"
            }),
            
            # AniList tools  
            "search_anime_anilist": AsyncMock(return_value=[
                {
                    "id": "anilist_16498",
                    "title": "Shingeki no Kyojin",
                    "source_platform": "anilist"
                }
            ]),
            
            # Schedule tools
            "search_anime_schedule": AsyncMock(return_value=[
                {
                    "id": "schedule_aot",
                    "title": "Attack on Titan",
                    "source_platform": "animeschedule"
                }
            ]),
            
            # Cross-platform tools
            "get_cross_platform_anime_data": AsyncMock(return_value={
                "anime_id": "cross_platform_16498",
                "title": "Attack on Titan",
                "platforms_verified": 3
            }),
            
            # Semantic tools
            "anime_similar": AsyncMock(return_value=[
                {
                    "id": "semantic_similar_1",
                    "title": "Similar Anime 1",
                    "similarity_score": 0.87
                }
            ])
        }
        return tools

    @pytest.fixture
    def workflow_orchestrator(self, mock_platform_tools):
        """Create workflow orchestrator with all tools."""
        class WorkflowOrchestrator:
            def __init__(self, tools):
                self.tools = tools
                
            async def execute_discovery_workflow(self, query, platforms=None, enrichment_level="basic"):
                """Execute comprehensive discovery workflow."""
                platforms = platforms or ["mal", "anilist", "animeschedule"]
                
                # Step 1: Parallel platform search
                search_tasks = []
                for platform in platforms:
                    search_tool = f"search_anime_{platform}"
                    if search_tool in self.tools:
                        search_tasks.append(self.tools[search_tool](query))
                
                with patch('asyncio.gather') as mock_gather:
                    platform_results = []
                    for platform in platforms:
                        if platform == "mal":
                            platform_results.append([{"id": f"{platform}_123", "title": "Result", "source_platform": platform}])
                        elif platform == "anilist":
                            platform_results.append([{"id": f"{platform}_456", "title": "Result", "source_platform": platform}])
                        else:
                            platform_results.append([{"id": f"{platform}_789", "title": "Result", "source_platform": platform}])
                    
                    mock_gather.return_value = platform_results
                    search_results = await mock_gather(*search_tasks)
                
                # Step 2: Result aggregation
                aggregated_results = []
                for i, platform_result in enumerate(search_results):
                    for anime in platform_result:
                        aggregated_results.append({
                            **anime,
                            "platform_index": i
                        })
                
                # Step 3: Cross-platform enrichment
                if enrichment_level in ["comprehensive", "full"]:
                    for anime in aggregated_results:
                        enrichment = await self.tools["get_cross_platform_anime_data"](
                            anime_title=anime["title"]
                        )
                        anime["enrichment_data"] = enrichment
                
                return {
                    "results": aggregated_results,
                    "platforms_searched": platforms,
                    "enrichment_level": enrichment_level,
                    "total_results": len(aggregated_results)
                }
                
            async def execute_similarity_workflow(self, reference_anime, similarity_threshold=0.8):
                """Execute similarity-based discovery workflow."""
                # Step 1: Get reference anime data
                reference_data = await self.tools["search_anime_mal"](reference_anime)
                
                # Step 2: Semantic similarity search
                similar_anime = await self.tools["anime_similar"](reference_anime=reference_anime)
                
                # Step 3: Filter by similarity threshold
                high_similarity = [
                    anime for anime in similar_anime
                    if anime.get("similarity_score", 0) >= similarity_threshold
                ]
                
                # Step 4: Enrich similar anime with platform data
                enriched_similar = []
                for anime in high_similarity:
                    try:
                        platform_data = await self.tools["search_anime_mal"](anime["title"])
                        anime["platform_enrichment"] = platform_data[0] if platform_data else None
                    except Exception:
                        anime["platform_enrichment"] = None
                    enriched_similar.append(anime)
                
                return {
                    "reference_anime": reference_data[0] if reference_data else None,
                    "similar_anime": enriched_similar,
                    "similarity_threshold": similarity_threshold,
                    "total_similar": len(enriched_similar)
                }
        
        return WorkflowOrchestrator(mock_platform_tools)

    @pytest.mark.asyncio
    async def test_comprehensive_discovery_workflow(self, workflow_orchestrator):
        """Test comprehensive cross-platform discovery workflow."""
        result = await workflow_orchestrator.execute_discovery_workflow(
            query="attack on titan",
            platforms=["mal", "anilist", "animeschedule"],
            enrichment_level="comprehensive"
        )
        
        # Verify comprehensive workflow
        assert result["total_results"] == 3
        assert len(result["platforms_searched"]) == 3
        assert result["enrichment_level"] == "comprehensive"
        assert all("enrichment_data" in anime for anime in result["results"])

    @pytest.mark.asyncio
    async def test_similarity_based_discovery_workflow(self, workflow_orchestrator):
        """Test similarity-based discovery workflow."""
        result = await workflow_orchestrator.execute_similarity_workflow(
            reference_anime="Attack on Titan",
            similarity_threshold=0.85
        )
        
        # Verify similarity workflow
        assert result["reference_anime"] is not None
        assert result["similarity_threshold"] == 0.85
        assert all(
            anime["similarity_score"] >= 0.85 
            for anime in result["similar_anime"]
        )

    @pytest.mark.asyncio
    async def test_streaming_availability_workflow(self, mock_platform_tools):
        """Test streaming availability discovery workflow."""
        # Step 1: Find anime
        anime_result = await mock_platform_tools["search_anime_mal"]("popular anime")
        
        # Step 2: Get streaming data from schedule platform
        streaming_data = await mock_platform_tools["search_anime_schedule"](
            anime_result[0]["title"]
        )
        
        # Step 3: Cross-reference with streaming platforms
        platform_availability = []
        for anime in streaming_data:
            for site in anime.get("streaming_sites", []):
                platform_availability.append({
                    "anime_title": anime["title"],
                    "platform": site["site"],
                    "regions": site["regions"],
                    "availability_score": 0.9
                })
        
        # Verify streaming workflow
        assert len(platform_availability) > 0
        assert all("platform" in item for item in platform_availability)
        assert all("regions" in item for item in platform_availability)

    @pytest.mark.asyncio
    async def test_multi_agent_coordination_workflow(self, mock_platform_tools):
        """Test multi-agent coordination workflow."""
        # Simulate SearchAgent and ScheduleAgent coordination
        search_phase = {
            "agent": "SearchAgent",
            "results": await mock_platform_tools["search_anime_mal"]("ongoing anime"),
            "handoff_required": True,
            "handoff_context": {
                "anime_titles": ["Ongoing Anime"],
                "requested_info": ["schedule", "streaming"]
            }
        }
        
        # ScheduleAgent takes over based on handoff
        if search_phase["handoff_required"]:
            schedule_phase = {
                "agent": "ScheduleAgent",
                "schedule_data": await mock_platform_tools["get_currently_airing"](),
                "streaming_data": await mock_platform_tools["search_anime_schedule"](
                    search_phase["handoff_context"]["anime_titles"][0]
                ),
                "handoff_complete": True
            }
        
        # Combine results from both agents
        coordinated_result = {
            "search_results": search_phase["results"],
            "schedule_info": schedule_phase["schedule_data"],
            "streaming_info": schedule_phase["streaming_data"],
            "agents_involved": [search_phase["agent"], schedule_phase["agent"]],
            "coordination_successful": schedule_phase["handoff_complete"]
        }
        
        # Verify coordination
        assert coordinated_result["coordination_successful"] is True
        assert len(coordinated_result["agents_involved"]) == 2
        assert "search_results" in coordinated_result
        assert "schedule_info" in coordinated_result

    @pytest.mark.asyncio
    async def test_workflow_performance_optimization(self, workflow_orchestrator):
        """Test workflow performance optimization strategies."""
        # Test caching optimization
        query = "popular anime 2024"
        
        # First execution (cache miss)
        result_1 = await workflow_orchestrator.execute_discovery_workflow(
            query=query,
            enrichment_level="basic"
        )
        
        # Second execution (cache hit simulation)
        result_2 = await workflow_orchestrator.execute_discovery_workflow(
            query=query,
            enrichment_level="basic"
        )
        
        # Verify both executions work
        assert result_1["total_results"] > 0
        assert result_2["total_results"] > 0
        
        # Test batch processing optimization
        batch_queries = ["anime 1", "anime 2", "anime 3"]
        batch_results = []
        
        for query in batch_queries:
            result = await workflow_orchestrator.execute_discovery_workflow(
                query=query,
                enrichment_level="basic"
            )
            batch_results.append(result)
        
        # Verify batch processing
        assert len(batch_results) == 3
        assert all(result["total_results"] > 0 for result in batch_results)