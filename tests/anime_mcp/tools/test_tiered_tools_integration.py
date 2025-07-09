"""Integration tests for modern tiered MCP tools."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Dict, Any

from src.anime_mcp.tools import TIER_INFO
from src.models.structured_responses import (
    BasicSearchResponse, 
    StandardSearchResponse, 
    DetailedSearchResponse,
    ComprehensiveSearchResponse,
    BasicAnimeResult,
    StandardAnimeResult,
    DetailedAnimeResult,
    ComprehensiveAnimeResult,
    AnimeType,
    AnimeStatus
)


class TestTieredToolsIntegration:
    """Test suite for tiered MCP tools integration."""

    @pytest.fixture
    def mock_clients(self):
        """Create mock clients for all platforms."""
        mocks = {}
        
        # Mock Jikan client
        jikan_mock = AsyncMock()
        jikan_mock.search_anime.return_value = [
            {
                "mal_id": 1,
                "title": "Test Anime",
                "type": "TV",
                "episodes": 12,
                "status": "Finished Airing",
                "score": 8.5,
                "year": 2023,
                "genres": [{"name": "Action"}],
                "studios": [{"name": "Test Studio"}],
                "synopsis": "Test synopsis",
                "images": {"jpg": {"image_url": "https://example.com/test.jpg"}}
            }
        ]
        mocks['jikan'] = jikan_mock
        
        # Mock MAL client
        mal_mock = AsyncMock()
        mal_mock.search_anime.return_value = {
            "data": [
                {
                    "node": {
                        "id": 1,
                        "title": "Test Anime MAL",
                        "main_picture": {"medium": "https://example.com/mal.jpg"},
                        "mean": 8.0,
                        "media_type": "tv",
                        "status": "finished_airing",
                        "genres": [{"name": "Action"}],
                        "studios": [{"name": "MAL Studio"}],
                        "synopsis": "MAL synopsis",
                        "start_date": "2023-01-01",
                        "num_episodes": 12
                    }
                }
            ]
        }
        mocks['mal'] = mal_mock
        
        return mocks

    def test_tier_info_structure(self):
        """Test that TIER_INFO has the correct structure."""
        assert "basic" in TIER_INFO
        assert "standard" in TIER_INFO
        assert "detailed" in TIER_INFO
        assert "comprehensive" in TIER_INFO
        
        # Check each tier has required fields
        for tier_name, tier_info in TIER_INFO.items():
            assert "name" in tier_info
            assert "description" in tier_info
            assert "fields" in tier_info
            assert "coverage" in tier_info
            assert "tools" in tier_info
            assert isinstance(tier_info["tools"], list)
            assert len(tier_info["tools"]) > 0

    def test_basic_tier_tools(self):
        """Test basic tier tools are properly defined."""
        basic_tools = TIER_INFO["basic"]["tools"]
        expected_tools = [
            "search_anime_basic",
            "get_anime_basic",
            "find_similar_anime_basic",
            "get_seasonal_anime_basic"
        ]
        
        for tool in expected_tools:
            assert tool in basic_tools

    def test_standard_tier_tools(self):
        """Test standard tier tools are properly defined."""
        standard_tools = TIER_INFO["standard"]["tools"]
        expected_tools = [
            "search_anime_standard",
            "get_anime_standard",
            "find_similar_anime_standard",
            "get_seasonal_anime_standard",
            "search_by_genre_standard"
        ]
        
        for tool in expected_tools:
            assert tool in standard_tools

    def test_detailed_tier_tools(self):
        """Test detailed tier tools are properly defined."""
        detailed_tools = TIER_INFO["detailed"]["tools"]
        expected_tools = [
            "search_anime_detailed",
            "get_anime_detailed",
            "find_similar_anime_detailed",
            "get_seasonal_anime_detailed",
            "advanced_anime_analysis"
        ]
        
        for tool in expected_tools:
            assert tool in detailed_tools

    def test_comprehensive_tier_tools(self):
        """Test comprehensive tier tools are properly defined."""
        comprehensive_tools = TIER_INFO["comprehensive"]["tools"]
        expected_tools = [
            "search_anime_comprehensive",
            "get_anime_comprehensive",
            "find_similar_anime_comprehensive",
            "comprehensive_anime_analytics"
        ]
        
        for tool in expected_tools:
            assert tool in comprehensive_tools

    def test_tier_progression(self):
        """Test that tiers have proper progression in complexity."""
        basic_fields = TIER_INFO["basic"]["fields"]
        standard_fields = TIER_INFO["standard"]["fields"]
        detailed_fields = TIER_INFO["detailed"]["fields"]
        
        # Fields should increase across tiers
        assert basic_fields < standard_fields < detailed_fields
        
        # Coverage should increase across tiers
        basic_coverage = int(TIER_INFO["basic"]["coverage"].split("%")[0])
        standard_coverage = int(TIER_INFO["standard"]["coverage"].split("%")[0])
        detailed_coverage = int(TIER_INFO["detailed"]["coverage"].split("%")[0])
        comprehensive_coverage = int(TIER_INFO["comprehensive"]["coverage"].split("%")[0])
        
        assert basic_coverage < standard_coverage < detailed_coverage < comprehensive_coverage

    def test_tier_tool_registration(self):
        """Test that tier tools can be imported and registered."""
        from src.anime_mcp.tools.tier1_basic_tools import register_basic_tools
        from src.anime_mcp.tools.tier2_standard_tools import register_standard_tools
        from src.anime_mcp.tools.tier3_detailed_tools import register_detailed_tools
        from src.anime_mcp.tools.tier4_comprehensive_tools import register_comprehensive_tools
        
        # These should import without errors
        assert register_basic_tools is not None
        assert register_standard_tools is not None
        assert register_detailed_tools is not None
        assert register_comprehensive_tools is not None

    def test_response_models_exist(self):
        """Test that all response models are properly defined."""
        # Basic response models
        assert BasicSearchResponse is not None
        assert BasicAnimeResult is not None
        
        # Standard response models
        assert StandardSearchResponse is not None
        assert StandardAnimeResult is not None
        
        # Detailed response models
        assert DetailedSearchResponse is not None
        assert DetailedAnimeResult is not None
        
        # Comprehensive response models
        assert ComprehensiveSearchResponse is not None
        assert ComprehensiveAnimeResult is not None

    def test_enum_types(self):
        """Test that enum types are properly defined."""
        # Test AnimeType enum
        assert AnimeType.TV is not None
        assert AnimeType.MOVIE is not None
        assert AnimeType.OVA is not None
        assert AnimeType.SPECIAL is not None
        
        # Test AnimeStatus enum
        assert AnimeStatus.FINISHED is not None
        assert AnimeStatus.RELEASING is not None
        assert AnimeStatus.NOT_YET_RELEASED is not None

    @pytest.mark.asyncio
    async def test_basic_search_response_structure(self):
        """Test BasicSearchResponse structure."""
        result = BasicAnimeResult(
            id="test_1",
            title="Test Anime",
            score=8.5,
            year=2023,
            type=AnimeType.TV,
            genres=["Action"],
            image_url="https://example.com/test.jpg",
            synopsis="Test synopsis"
        )
        
        response = BasicSearchResponse(
            results=[result],
            total=1,
            query="test query",
            processing_time_ms=150.0
        )
        
        assert response.results[0].title == "Test Anime"
        assert response.total == 1
        assert response.query == "test query"
        assert response.processing_time_ms == 150.0

    def test_tier_helper_functions(self):
        """Test tier helper functions."""
        from src.anime_mcp.tools import get_recommended_tier, get_tier_tools, get_all_tiers
        
        # Test get_recommended_tier
        assert get_recommended_tier("simple", "speed") == "basic"
        assert get_recommended_tier("complex", "completeness") == "comprehensive"
        
        # Test get_tier_tools
        basic_tools = get_tier_tools("basic")
        assert len(basic_tools) > 0
        assert "search_anime_basic" in basic_tools
        
        # Test get_all_tiers
        all_tiers = get_all_tiers()
        assert "basic" in all_tiers
        assert "standard" in all_tiers
        assert "detailed" in all_tiers
        assert "comprehensive" in all_tiers

    def test_no_universal_parameter_references(self):
        """Test that no Universal parameter references remain."""
        # This test ensures the modernization removed all Universal parameter dependencies
        
        # Try to import Universal parameter models - should fail
        with pytest.raises(ImportError):
            from src.models.universal_anime import UniversalSearchParams
            
        with pytest.raises(ImportError):
            from src.integrations.mapper_registry import MapperRegistry
            
        with pytest.raises(ImportError):
            from src.integrations.service_manager import ServiceManager

    def test_modern_architecture_imports(self):
        """Test that modern architecture components import correctly."""
        from src.anime_mcp.tools import register_all_tiered_tools
        from src.models.structured_responses import BasicSearchResponse
        from src.langgraph.langchain_tools import create_anime_langchain_tools
        
        # All modern components should import without errors
        assert register_all_tiered_tools is not None
        assert BasicSearchResponse is not None
        assert create_anime_langchain_tools is not None

    def test_langchain_tools_integration(self):
        """Test that LangChain tools integration works with tiered tools."""
        from src.langgraph.langchain_tools import create_anime_langchain_tools
        
        # Create mock tool registry
        mock_tools = {
            "search_anime_basic": lambda **kwargs: {"result": "basic"},
            "search_anime_standard": lambda **kwargs: {"result": "standard"},
            "get_anime_detailed": lambda **kwargs: {"result": "detailed"},
        }
        
        # Test LangChain tools creation
        langchain_tools = create_anime_langchain_tools(mock_tools)
        
        # Verify tools were created
        assert len(langchain_tools) > 0
        tool_names = [tool.name for tool in langchain_tools]
        assert "search_anime_basic" in tool_names
        assert "search_anime_standard" in tool_names
        assert "get_anime_detailed" in tool_names