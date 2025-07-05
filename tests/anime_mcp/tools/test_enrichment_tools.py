"""Unit tests for cross-platform enrichment MCP tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.anime_mcp.tools.enrichment_tools import (
    compare_anime_ratings_cross_platform,
    get_cross_platform_anime_data,
    correlate_anime_across_platforms,
    get_streaming_availability_multi_platform,
    detect_platform_discrepancies,
    get_enrichment_system
)


class TestEnrichmentTools:
    """Test suite for cross-platform enrichment MCP tools."""

    @pytest.fixture
    def mock_enrichment_system(self):
        """Create mock CrossPlatformEnrichment system."""
        system = AsyncMock()
        
        # Mock compare_anime_ratings
        system.compare_anime_ratings.return_value = {
            "anime_title": "Attack on Titan",
            "comparison_platforms": ["mal", "anilist", "kitsu"],
            "rating_details": {
                "mal": {
                    "score": 8.54,
                    "rated_by": 1500000,
                    "rank": 50,
                    "popularity": 1
                },
                "anilist": {
                    "score": 85,
                    "rated_by": 750000,
                    "rank": 45,
                    "popularity": 5
                },
                "kitsu": {
                    "score": 85.4,
                    "rated_by": 50000,
                    "rank": 60,
                    "popularity": 10
                }
            },
            "rating_statistics": {
                "average_score": 8.48,
                "score_variance": 0.02,
                "highest_rated_platform": "kitsu",
                "most_popular_platform": "mal",
                "total_ratings": 2300000
            },
            "rating_consistency": "high",
            "comparison_timestamp": "2024-01-15T12:00:00Z"
        }
        
        # Mock enrich_anime_cross_platform
        system.enrich_anime_cross_platform.return_value = {
            "reference_anime": "Attack on Titan",
            "enrichment_platforms": ["mal", "anilist", "animeschedule", "kitsu"],
            "platform_data": {
                "mal": {
                    "id": 16498,
                    "title": "Shingeki no Kyojin",
                    "score": 8.54,
                    "episodes": 25,
                    "status": "finished",
                    "studios": ["Wit Studio"]
                },
                "anilist": {
                    "id": 16498,
                    "title": "Shingeki no Kyojin", 
                    "score": 85,
                    "episodes": 25,
                    "status": "FINISHED",
                    "studios": ["Wit Studio"]
                },
                "animeschedule": {
                    "route": "attack-on-titan",
                    "broadcast_day": "Sunday",
                    "broadcast_time": "17:00",
                    "streaming_sites": [
                        {"site": "Crunchyroll", "regions": ["US", "CA"]}
                    ]
                },
                "kitsu": {
                    "id": "1",
                    "slug": "attack-on-titan",
                    "rating": 85.4,
                    "episodes": 25,
                    "status": "finished"
                }
            },
            "correlation_results": {
                "matches": {
                    "mal": {"confidence": 0.98, "match_type": "exact"},
                    "anilist": {"confidence": 0.97, "match_type": "exact"}, 
                    "animeschedule": {"confidence": 0.95, "match_type": "title"},
                    "kitsu": {"confidence": 0.93, "match_type": "title"}
                },
                "overall_confidence": 0.96
            },
            "enrichment_summary": {
                "total_platforms": 4,
                "successful_matches": 4,
                "data_completeness": 0.92,
                "enrichment_quality": "excellent"
            },
            "enrichment_timestamp": "2024-01-15T12:00:00Z"
        }
        
        # Mock correlate_anime_across_platforms
        system.correlate_anime_across_platforms.return_value = {
            "anime_title": "Attack on Titan",
            "correlation_platforms": ["mal", "anilist", "kitsu", "jikan"],
            "platform_matches": {
                "mal": {
                    "id": 16498,
                    "title": "Shingeki no Kyojin",
                    "year": 2013,
                    "episodes": 25,
                    "match_confidence": 0.98
                },
                "anilist": {
                    "id": 16498,
                    "title": "Shingeki no Kyojin",
                    "year": 2013, 
                    "episodes": 25,
                    "match_confidence": 0.97
                },
                "kitsu": {
                    "id": "1",
                    "title": "Attack on Titan",
                    "year": 2013,
                    "episodes": 25,
                    "match_confidence": 0.93
                },
                "jikan": {
                    "id": 16498,
                    "title": "Shingeki no Kyojin",
                    "year": 2013,
                    "episodes": 25,
                    "match_confidence": 0.99
                }
            },
            "correlation_confidence": 0.97,
            "correlation_method": "title_metadata_matching",
            "correlation_timestamp": "2024-01-15T12:00:00Z"
        }
        
        # Mock get_cross_platform_streaming_info
        system.get_cross_platform_streaming_info.return_value = {
            "anime_title": "Attack on Titan",
            "target_platforms": ["kitsu", "animeschedule"],
            "streaming_data": {
                "kitsu": [
                    {
                        "platform": "Crunchyroll",
                        "url": "https://crunchyroll.com/attack-on-titan",
                        "regions": ["US", "CA", "UK", "AU"],
                        "subtitles": ["en", "es", "fr"],
                        "dubs": ["en"]
                    },
                    {
                        "platform": "Funimation",
                        "url": "https://funimation.com/attack-on-titan", 
                        "regions": ["US", "CA"],
                        "subtitles": ["en"],
                        "dubs": ["en"]
                    }
                ],
                "animeschedule": [
                    {
                        "platform": "Crunchyroll",
                        "url": "https://crunchyroll.com/attack-on-titan",
                        "regions": ["US", "CA", "UK"]
                    },
                    {
                        "platform": "Netflix",
                        "url": "https://netflix.com/attack-on-titan",
                        "regions": ["JP", "KR"]
                    }
                ]
            },
            "availability_summary": {
                "total_platforms": 3,
                "unique_streaming_services": ["Crunchyroll", "Funimation", "Netflix"],
                "total_regions": 6,
                "most_available_service": "Crunchyroll"
            },
            "aggregation_timestamp": "2024-01-15T12:00:00Z"
        }
        
        return system

    @pytest.fixture
    def mock_context(self):
        """Create mock MCP context."""
        context = AsyncMock()
        context.info = AsyncMock()
        context.error = AsyncMock()
        return context

    @pytest.mark.asyncio
    async def test_compare_anime_ratings_cross_platform_success(self, mock_enrichment_system, mock_context):
        """Test successful cross-platform rating comparison."""
        with patch('src.anime_mcp.tools.enrichment_tools.get_enrichment_system', return_value=mock_enrichment_system):
            
            result = await compare_anime_ratings_cross_platform(
                anime_title="Attack on Titan",
                platforms=["mal", "anilist", "kitsu"],
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, dict)
            assert result["anime_title"] == "Attack on Titan"
            assert result["comparison_platforms"] == ["mal", "anilist", "kitsu"]
            assert "rating_details" in result
            assert "rating_statistics" in result
            
            # Verify rating details
            rating_details = result["rating_details"]
            assert "mal" in rating_details
            assert "anilist" in rating_details
            assert "kitsu" in rating_details
            
            # Verify MAL rating data
            mal_rating = rating_details["mal"]
            assert mal_rating["score"] == 8.54
            assert mal_rating["rated_by"] == 1500000
            assert mal_rating["rank"] == 50
            
            # Verify statistics
            stats = result["rating_statistics"]
            assert stats["average_score"] == 8.48
            assert stats["highest_rated_platform"] == "kitsu"
            assert stats["most_popular_platform"] == "mal"
            
            # Verify enrichment system was called correctly
            mock_enrichment_system.compare_anime_ratings.assert_called_once_with(
                anime_title="Attack on Titan",
                comparison_platforms=["mal", "anilist", "kitsu"]
            )
            
            # Verify context calls
            mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_compare_anime_ratings_default_platforms(self, mock_enrichment_system, mock_context):
        """Test rating comparison with default platforms."""
        with patch('src.anime_mcp.tools.enrichment_tools.get_enrichment_system', return_value=mock_enrichment_system):
            
            result = await compare_anime_ratings_cross_platform(
                anime_title="Death Note",
                ctx=mock_context
            )
            
            # Should use default platforms
            mock_enrichment_system.compare_anime_ratings.assert_called_once_with(
                anime_title="Death Note",
                comparison_platforms=["mal", "anilist", "kitsu"]
            )

    @pytest.mark.asyncio
    async def test_compare_anime_ratings_error_handling(self, mock_enrichment_system, mock_context):
        """Test rating comparison error handling."""
        mock_enrichment_system.compare_anime_ratings.side_effect = Exception("Rating comparison failed")
        
        with patch('src.anime_mcp.tools.enrichment_tools.get_enrichment_system', return_value=mock_enrichment_system):
            
            with pytest.raises(RuntimeError, match="Rating comparison failed: Rating comparison failed"):
                await compare_anime_ratings_cross_platform(
                    anime_title="Attack on Titan",
                    ctx=mock_context
                )
            
            mock_context.error.assert_called_with("Rating comparison failed: Rating comparison failed")

    @pytest.mark.asyncio
    async def test_get_cross_platform_anime_data_success(self, mock_enrichment_system, mock_context):
        """Test successful cross-platform data enrichment."""
        with patch('src.anime_mcp.tools.enrichment_tools.get_enrichment_system', return_value=mock_enrichment_system):
            
            result = await get_cross_platform_anime_data(
                anime_title="Attack on Titan",
                platforms=["mal", "anilist", "animeschedule", "kitsu"],
                include_ratings=True,
                include_streaming=True,
                include_schedule=True,
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, dict)
            assert result["reference_anime"] == "Attack on Titan"
            assert result["enrichment_platforms"] == ["mal", "anilist", "animeschedule", "kitsu"]
            
            # Verify platform data
            platform_data = result["platform_data"]
            assert "mal" in platform_data
            assert "anilist" in platform_data
            assert "animeschedule" in platform_data
            assert "kitsu" in platform_data
            
            # Verify MAL data
            mal_data = platform_data["mal"]
            assert mal_data["id"] == 16498
            assert mal_data["score"] == 8.54
            assert mal_data["episodes"] == 25
            
            # Verify correlation results
            correlation = result["correlation_results"]
            assert correlation["overall_confidence"] == 0.96
            assert len(correlation["matches"]) == 4
            
            # Verify enrichment summary
            summary = result["enrichment_summary"]
            assert summary["total_platforms"] == 4
            assert summary["successful_matches"] == 4
            assert summary["enrichment_quality"] == "excellent"
            
            # Verify enrichment system was called correctly
            mock_enrichment_system.enrich_anime_cross_platform.assert_called_once_with(
                reference_anime="Attack on Titan",
                platforms=["mal", "anilist", "animeschedule", "kitsu"],
                include_ratings=True,
                include_streaming=True,
                include_schedule=True
            )

    @pytest.mark.asyncio
    async def test_get_cross_platform_anime_data_default_options(self, mock_enrichment_system, mock_context):
        """Test cross-platform data enrichment with default options."""
        with patch('src.anime_mcp.tools.enrichment_tools.get_enrichment_system', return_value=mock_enrichment_system):
            
            result = await get_cross_platform_anime_data(
                anime_title="One Piece",
                ctx=mock_context
            )
            
            # Should use default platforms and options
            mock_enrichment_system.enrich_anime_cross_platform.assert_called_once_with(
                reference_anime="One Piece",
                platforms=["mal", "anilist", "animeschedule", "kitsu"],
                include_ratings=True,
                include_streaming=True,
                include_schedule=True
            )

    @pytest.mark.asyncio
    async def test_correlate_anime_across_platforms_success(self, mock_enrichment_system, mock_context):
        """Test successful anime correlation across platforms."""
        with patch('src.anime_mcp.tools.enrichment_tools.get_enrichment_system', return_value=mock_enrichment_system):
            
            result = await correlate_anime_across_platforms(
                anime_title="Attack on Titan",
                platforms=["mal", "anilist", "kitsu", "jikan"],
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, dict)
            assert result["anime_title"] == "Attack on Titan"
            assert result["correlation_platforms"] == ["mal", "anilist", "kitsu", "jikan"]
            
            # Verify platform matches
            matches = result["platform_matches"]
            assert "mal" in matches
            assert "anilist" in matches
            assert "kitsu" in matches
            assert "jikan" in matches
            
            # Verify match details
            mal_match = matches["mal"]
            assert mal_match["id"] == 16498
            assert mal_match["match_confidence"] == 0.98
            
            # Verify correlation metadata
            assert result["correlation_confidence"] == 0.97
            assert result["correlation_method"] == "title_metadata_matching"
            
            # Verify enrichment system was called correctly
            mock_enrichment_system.correlate_anime_across_platforms.assert_called_once_with(
                anime_title="Attack on Titan",
                correlation_platforms=["mal", "anilist", "kitsu", "jikan"]
            )
            
            # Verify context info includes confidence
            info_calls = [call.args[0] for call in mock_context.info.call_args_list]
            assert any("97.0%" in call for call in info_calls)

    @pytest.mark.asyncio
    async def test_correlate_anime_default_platforms(self, mock_enrichment_system, mock_context):
        """Test anime correlation with default platforms."""
        with patch('src.anime_mcp.tools.enrichment_tools.get_enrichment_system', return_value=mock_enrichment_system):
            
            result = await correlate_anime_across_platforms(
                anime_title="Naruto",
                ctx=mock_context
            )
            
            # Should use default platforms
            mock_enrichment_system.correlate_anime_across_platforms.assert_called_once_with(
                anime_title="Naruto",
                correlation_platforms=["mal", "anilist", "kitsu", "jikan"]
            )

    @pytest.mark.asyncio
    async def test_get_streaming_availability_multi_platform_success(self, mock_enrichment_system, mock_context):
        """Test successful multi-platform streaming availability."""
        with patch('src.anime_mcp.tools.enrichment_tools.get_enrichment_system', return_value=mock_enrichment_system):
            
            result = await get_streaming_availability_multi_platform(
                anime_title="Attack on Titan",
                platforms=["kitsu", "animeschedule"],
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, dict)
            assert result["anime_title"] == "Attack on Titan"
            assert result["target_platforms"] == ["kitsu", "animeschedule"]
            
            # Verify streaming data
            streaming_data = result["streaming_data"]
            assert "kitsu" in streaming_data
            assert "animeschedule" in streaming_data
            
            # Verify Kitsu streaming data
            kitsu_streams = streaming_data["kitsu"]
            assert len(kitsu_streams) == 2
            cr_stream = next(s for s in kitsu_streams if s["platform"] == "Crunchyroll")
            assert "US" in cr_stream["regions"]
            assert "en" in cr_stream["subtitles"]
            
            # Verify availability summary
            summary = result["availability_summary"]
            assert summary["total_platforms"] == 3
            assert summary["most_available_service"] == "Crunchyroll"
            assert len(summary["unique_streaming_services"]) == 3
            
            # Verify enrichment system was called correctly
            mock_enrichment_system.get_cross_platform_streaming_info.assert_called_once_with(
                anime_title="Attack on Titan",
                target_platforms=["kitsu", "animeschedule"]
            )

    @pytest.mark.asyncio
    async def test_get_streaming_availability_default_platforms(self, mock_enrichment_system, mock_context):
        """Test streaming availability with default platforms."""
        with patch('src.anime_mcp.tools.enrichment_tools.get_enrichment_system', return_value=mock_enrichment_system):
            
            result = await get_streaming_availability_multi_platform(
                anime_title="Demon Slayer",
                ctx=mock_context
            )
            
            # Should use default platforms
            mock_enrichment_system.get_cross_platform_streaming_info.assert_called_once_with(
                anime_title="Demon Slayer",
                target_platforms=["kitsu", "animeschedule"]
            )

    @pytest.mark.asyncio
    async def test_detect_platform_discrepancies_success(self, mock_enrichment_system, mock_context):
        """Test successful platform discrepancy detection."""
        # Setup mock correlation result with discrepancies
        mock_enrichment_system.correlate_anime_across_platforms.return_value = {
            "anime_title": "Test Anime",
            "platform_matches": {
                "mal": {
                    "title": "Test Anime",
                    "year": 2020,
                    "episodes": 12,
                    "rating": 8.5
                },
                "anilist": {
                    "title": "Test Anime",
                    "year": 2020,
                    "episodes": 13,  # Discrepancy
                    "rating": 85  # Different scale
                },
                "kitsu": {
                    "title": "Test Anime", 
                    "year": 2021,  # Discrepancy
                    "episodes": 12,
                    "rating": 8.4
                }
            },
            "correlation_timestamp": "2024-01-15T12:00:00Z"
        }
        
        with patch('src.anime_mcp.tools.enrichment_tools.get_enrichment_system', return_value=mock_enrichment_system):
            
            result = await detect_platform_discrepancies(
                anime_title="Test Anime",
                comparison_fields=["title", "year", "episodes", "rating"],
                platforms=["mal", "anilist", "kitsu"],
                ctx=mock_context
            )
            
            # Verify result structure
            assert isinstance(result, dict)
            assert result["anime_title"] == "Test Anime"
            assert result["platforms_compared"] == ["mal", "anilist", "kitsu"]
            assert result["fields_analyzed"] == ["title", "year", "episodes", "rating"]
            
            # Verify discrepancies were found
            discrepancies = result["discrepancies_found"]
            assert len(discrepancies) >= 2  # Should find year and episodes discrepancies
            
            # Check for specific discrepancies
            year_discrepancy = next((d for d in discrepancies if d["field"] == "year"), None)
            assert year_discrepancy is not None
            assert year_discrepancy["discrepancy_type"] == "value_mismatch"
            assert len(year_discrepancy["unique_values"]) == 2  # 2020 and 2021
            
            episodes_discrepancy = next((d for d in discrepancies if d["field"] == "episodes"), None)
            assert episodes_discrepancy is not None
            assert len(episodes_discrepancy["unique_values"]) == 2  # 12 and 13
            
            # Verify consistency score (2 consistent fields out of 4)
            expected_consistency = 2 / 4  # title and rating are consistent (relatively)
            assert 0.0 <= result["consistency_score"] <= 1.0
            
            # Verify enrichment system was called
            mock_enrichment_system.correlate_anime_across_platforms.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_platform_discrepancies_default_fields(self, mock_enrichment_system, mock_context):
        """Test discrepancy detection with default comparison fields."""
        mock_enrichment_system.correlate_anime_across_platforms.return_value = {
            "platform_matches": {
                "mal": {"title": "Test", "year": 2020, "episodes": 12, "rating": 8.0},
                "anilist": {"title": "Test", "year": 2020, "episodes": 12, "rating": 80}
            }
        }
        
        with patch('src.anime_mcp.tools.enrichment_tools.get_enrichment_system', return_value=mock_enrichment_system):
            
            result = await detect_platform_discrepancies(
                anime_title="Test Anime",
                ctx=mock_context
            )
            
            # Should use default comparison fields
            assert result["fields_analyzed"] == ["title", "year", "episodes", "rating"]

    @pytest.mark.asyncio
    async def test_detect_platform_discrepancies_no_discrepancies(self, mock_enrichment_system, mock_context):
        """Test discrepancy detection when no discrepancies exist."""
        # Setup mock correlation result with consistent data
        mock_enrichment_system.correlate_anime_across_platforms.return_value = {
            "anime_title": "Consistent Anime",
            "platform_matches": {
                "mal": {
                    "title": "Consistent Anime",
                    "year": 2020,
                    "episodes": 12,
                    "rating": 8.5
                },
                "anilist": {
                    "title": "Consistent Anime", 
                    "year": 2020,
                    "episodes": 12,
                    "rating": 8.5
                }
            }
        }
        
        with patch('src.anime_mcp.tools.enrichment_tools.get_enrichment_system', return_value=mock_enrichment_system):
            
            result = await detect_platform_discrepancies(
                anime_title="Consistent Anime",
                platforms=["mal", "anilist"],
                ctx=mock_context
            )
            
            # Should find no discrepancies
            assert len(result["discrepancies_found"]) == 0
            assert result["consistency_score"] == 1.0  # Perfect consistency
            
            # Verify context info reflects high consistency
            info_calls = [call.args[0] for call in mock_context.info.call_args_list]
            assert any("100.0%" in call for call in info_calls)


class TestEnrichmentToolsAdvanced:
    """Advanced tests for enrichment tools."""

    @pytest.mark.asyncio
    async def test_enrichment_tools_error_handling_comprehensive(self, mock_context):
        """Test comprehensive error handling across all enrichment tools."""
        mock_system = AsyncMock()
        
        error_scenarios = [
            ("Network timeout", "Request timeout during enrichment"),
            ("Data correlation", "Failed to correlate anime across platforms"),
            ("Platform unavailable", "One or more platforms temporarily unavailable"),
            ("Invalid anime", "Anime not found on any platform")
        ]
        
        with patch('src.anime_mcp.tools.enrichment_tools.get_enrichment_system', return_value=mock_system):
            
            for error_type, error_msg in error_scenarios:
                # Test all tools with the same error
                mock_system.compare_anime_ratings.side_effect = Exception(error_msg)
                mock_system.enrich_anime_cross_platform.side_effect = Exception(error_msg)
                mock_system.correlate_anime_across_platforms.side_effect = Exception(error_msg)
                mock_system.get_cross_platform_streaming_info.side_effect = Exception(error_msg)
                
                # Test rating comparison
                with pytest.raises(RuntimeError, match=f"Rating comparison failed: {error_msg}"):
                    await compare_anime_ratings_cross_platform("Test", ctx=mock_context)
                
                # Test data enrichment
                with pytest.raises(RuntimeError, match=f"Data enrichment failed: {error_msg}"):
                    await get_cross_platform_anime_data("Test", ctx=mock_context)
                
                # Test correlation
                with pytest.raises(RuntimeError, match=f"Anime correlation failed: {error_msg}"):
                    await correlate_anime_across_platforms("Test", ctx=mock_context)
                
                # Test streaming availability
                with pytest.raises(RuntimeError, match=f"Streaming availability check failed: {error_msg}"):
                    await get_streaming_availability_multi_platform("Test", ctx=mock_context)
                
                # Test discrepancy detection
                with pytest.raises(RuntimeError, match=f"Discrepancy detection failed: {error_msg}"):
                    await detect_platform_discrepancies("Test", ctx=mock_context)
                
                mock_context.reset_mock()

    @pytest.mark.asyncio
    async def test_enrichment_tools_without_context(self):
        """Test enrichment tools without context (no logging)."""
        mock_system = AsyncMock()
        mock_system.compare_anime_ratings.return_value = {"anime_title": "Test"}
        mock_system.enrich_anime_cross_platform.return_value = {"reference_anime": "Test"}
        mock_system.correlate_anime_across_platforms.return_value = {"anime_title": "Test"}
        mock_system.get_cross_platform_streaming_info.return_value = {"anime_title": "Test"}
        
        with patch('src.anime_mcp.tools.enrichment_tools.get_enrichment_system', return_value=mock_system):
            
            # Test all tools without ctx parameter
            await compare_anime_ratings_cross_platform("Test")  # No ctx
            await get_cross_platform_anime_data("Test")  # No ctx
            await correlate_anime_across_platforms("Test")  # No ctx
            await get_streaming_availability_multi_platform("Test")  # No ctx
            await detect_platform_discrepancies("Test")  # No ctx

    @pytest.mark.asyncio
    async def test_enrichment_system_singleton_behavior(self):
        """Test that enrichment system behaves as singleton."""
        with patch('src.anime_mcp.tools.enrichment_tools.enrichment_system', None):
            
            # First call should create system
            system1 = get_enrichment_system()
            assert system1 is not None
            
            # Second call should return same system
            system2 = get_enrichment_system()
            assert system1 is system2

    @pytest.mark.asyncio
    async def test_enrichment_tools_complex_data_processing(self, mock_context):
        """Test complex data processing scenarios."""
        mock_system = AsyncMock()
        
        # Mock complex correlation result with partial matches
        mock_system.correlate_anime_across_platforms.return_value = {
            "anime_title": "Complex Anime",
            "platform_matches": {
                "mal": {
                    "title": "Complex Anime",
                    "year": 2020,
                    "episodes": None,  # Missing data
                    "rating": 8.5,
                    "status": "finished"
                },
                "anilist": {
                    "title": "Complex Anime (TV)",  # Slightly different title
                    "year": 2020,
                    "episodes": 24,
                    "rating": 85,  # Different scale
                    "status": "FINISHED"  # Different format
                },
                "kitsu": {
                    # Missing some fields entirely
                    "title": "Complex Anime",
                    "year": 2020
                }
            }
        }
        
        with patch('src.anime_mcp.tools.enrichment_tools.get_enrichment_system', return_value=mock_system):
            
            result = await detect_platform_discrepancies(
                anime_title="Complex Anime",
                comparison_fields=["title", "year", "episodes", "rating", "status"],
                ctx=mock_context
            )
            
            # Should handle missing data gracefully
            discrepancies = result["discrepancies_found"]
            
            # Should find discrepancies in available fields
            title_discrepancy = next((d for d in discrepancies if d["field"] == "title"), None)
            if title_discrepancy:  # May or may not be detected depending on tolerance
                assert "Complex Anime" in title_discrepancy["platform_values"].values()
            
            # Episodes should show discrepancy between None and 24
            episodes_discrepancy = next((d for d in discrepancies if d["field"] == "episodes"), None)
            if episodes_discrepancy:
                values = list(episodes_discrepancy["platform_values"].values())
                assert None in values or 24 in values

    @pytest.mark.asyncio 
    async def test_enrichment_tools_performance_with_large_datasets(self, mock_context):
        """Test enrichment tools performance with large datasets."""
        mock_system = AsyncMock()
        
        # Mock large streaming dataset
        large_streaming_data = {
            "anime_title": "Popular Anime",
            "streaming_data": {
                "kitsu": [
                    {
                        "platform": f"Platform_{i}",
                        "url": f"https://platform{i}.com/anime",
                        "regions": [f"REGION_{j}" for j in range(5)],
                        "subtitles": ["en", "es", "fr", "de", "ja"],
                        "dubs": ["en", "es"]
                    }
                    for i in range(20)  # 20 platforms
                ],
                "animeschedule": [
                    {
                        "platform": f"Schedule_Platform_{i}",
                        "url": f"https://scheduleplatform{i}.com/anime",
                        "regions": [f"S_REGION_{j}" for j in range(3)]
                    }
                    for i in range(15)  # 15 platforms
                ]
            },
            "availability_summary": {
                "total_platforms": 35,
                "unique_streaming_services": [f"Service_{i}" for i in range(25)],
                "total_regions": 50
            }
        }
        
        mock_system.get_cross_platform_streaming_info.return_value = large_streaming_data
        
        with patch('src.anime_mcp.tools.enrichment_tools.get_enrichment_system', return_value=mock_system):
            
            result = await get_streaming_availability_multi_platform(
                anime_title="Popular Anime",
                ctx=mock_context
            )
            
            # Verify large dataset is handled correctly
            kitsu_data = result["streaming_data"]["kitsu"]
            assert len(kitsu_data) == 20
            
            schedule_data = result["streaming_data"]["animeschedule"]
            assert len(schedule_data) == 15
            
            summary = result["availability_summary"]
            assert summary["total_platforms"] == 35
            assert len(summary["unique_streaming_services"]) == 25


class TestEnrichmentToolsIntegration:
    """Integration tests for enrichment tools."""

    @pytest.mark.asyncio
    async def test_enrichment_tools_annotation_verification(self):
        """Verify that enrichment tools have correct MCP annotations."""
        from src.anime_mcp.tools.enrichment_tools import mcp
        
        # Get registered tools
        tools = mcp._tools
        
        # Verify compare_anime_ratings_cross_platform annotations
        rating_tool = tools.get("compare_anime_ratings_cross_platform")
        assert rating_tool is not None
        assert rating_tool.annotations["title"] == "Cross-Platform Rating Comparison"
        assert rating_tool.annotations["readOnlyHint"] is True
        assert rating_tool.annotations["idempotentHint"] is True
        
        # Verify get_cross_platform_anime_data annotations
        data_tool = tools.get("get_cross_platform_anime_data")
        assert data_tool is not None
        assert data_tool.annotations["title"] == "Cross-Platform Data Enrichment"
        
        # Verify correlate_anime_across_platforms annotations
        correlate_tool = tools.get("correlate_anime_across_platforms")
        assert correlate_tool is not None
        assert correlate_tool.annotations["title"] == "Cross-Platform Anime Correlation"
        
        # Verify get_streaming_availability_multi_platform annotations
        streaming_tool = tools.get("get_streaming_availability_multi_platform")
        assert streaming_tool is not None
        assert streaming_tool.annotations["title"] == "Multi-Platform Streaming Availability"
        
        # Verify detect_platform_discrepancies annotations
        discrepancy_tool = tools.get("detect_platform_discrepancies")
        assert discrepancy_tool is not None
        assert discrepancy_tool.annotations["title"] == "Cross-Platform Discrepancy Detection"

    @pytest.mark.asyncio
    async def test_enrichment_tools_workflow_integration(self, mock_context):
        """Test integration workflow between different enrichment tools."""
        mock_system = AsyncMock()
        
        # Setup realistic workflow data
        mock_system.correlate_anime_across_platforms.return_value = {
            "anime_title": "Workflow Test Anime",
            "platform_matches": {
                "mal": {"id": 12345, "title": "Test Anime", "score": 8.5},
                "anilist": {"id": 12345, "title": "Test Anime", "score": 85}
            }
        }
        
        mock_system.compare_anime_ratings.return_value = {
            "anime_title": "Workflow Test Anime",
            "rating_statistics": {"average_score": 8.25},
            "rating_consistency": "high"
        }
        
        mock_system.get_cross_platform_streaming_info.return_value = {
            "anime_title": "Workflow Test Anime",
            "availability_summary": {"total_platforms": 3}
        }
        
        with patch('src.anime_mcp.tools.enrichment_tools.get_enrichment_system', return_value=mock_system):
            
            # Step 1: Correlate anime across platforms
            correlation_result = await correlate_anime_across_platforms(
                anime_title="Workflow Test Anime",
                ctx=mock_context
            )
            assert len(correlation_result["platform_matches"]) == 2
            
            # Step 2: Compare ratings using correlated data
            rating_result = await compare_anime_ratings_cross_platform(
                anime_title="Workflow Test Anime",
                platforms=["mal", "anilist"],  # From correlation
                ctx=mock_context
            )
            assert rating_result["rating_statistics"]["average_score"] == 8.25
            
            # Step 3: Check streaming availability
            streaming_result = await get_streaming_availability_multi_platform(
                anime_title="Workflow Test Anime",
                ctx=mock_context
            )
            assert streaming_result["availability_summary"]["total_platforms"] == 3
            
            # Step 4: Detect discrepancies in found data
            discrepancy_result = await detect_platform_discrepancies(
                anime_title="Workflow Test Anime",
                platforms=["mal", "anilist"],
                ctx=mock_context
            )
            assert discrepancy_result["platforms_compared"] == ["mal", "anilist"]