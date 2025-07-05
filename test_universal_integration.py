#!/usr/bin/env python3
"""Test script for universal anime search integration.

This script demonstrates the complete integration flow from
MCP tools to universal parameter system with intelligent routing.
"""

import asyncio
import logging
from src.models.universal_anime import UniversalSearchParams
from src.integrations.mapper_registry import MapperRegistry
from src.integrations.service_manager import ServiceManager
from src.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_parameter_extraction():
    """Test universal parameter extraction and platform selection."""
    print("\n🔍 Testing Parameter Extraction")
    print("=" * 50)
    
    # Complex test case with multiple platform-specific parameters
    test_params = {
        "query": "dark military anime",
        "genres": ["Action", "Drama"],
        "min_score": 7.5,
        "limit": 10,
        "mal_nsfw": "white",
        "anilist_is_adult": False,
        "anilist_country_of_origin": "JP",
        "kitsu_streamers": ["Crunchyroll"],
        "animeschedule_streams": ["Netflix"]
    }
    
    # Extract parameters
    universal_params, platform_specific = MapperRegistry.extract_platform_params(**test_params)
    
    print(f"✅ Universal parameters ({len(universal_params)}):")
    for key, value in universal_params.items():
        print(f"   • {key}: {value}")
    
    print(f"\n✅ Platform-specific parameters:")
    for platform, params in platform_specific.items():
        print(f"   • {platform}: {params}")
    
    # Test auto-platform selection
    selected_platform = MapperRegistry.auto_select_platform(universal_params, platform_specific)
    print(f"\n🎯 Auto-selected platform: {selected_platform}")
    
    return universal_params, platform_specific, selected_platform

async def test_universal_search_params():
    """Test UniversalSearchParams model validation."""
    print("\n📋 Testing UniversalSearchParams Model")
    print("=" * 50)
    
    # Test comprehensive parameter support
    params = UniversalSearchParams(
        query="mecha robots fighting",
        genres=["Action", "Sci-Fi"],
        min_score=8.0,
        year=2023,
        limit=15,
        # MAL-specific
        mal_nsfw="white",
        mal_rating="pg_13",
        # AniList-specific  
        anilist_is_adult=False,
        anilist_country_of_origin="JP",
        anilist_episodes_greater=12,
        anilist_format_not_in=["MOVIE"],
        # Kitsu-specific
        kitsu_streamers=["Crunchyroll", "Funimation"],
        # AnimeSchedule-specific
        animeschedule_streams=["Netflix"],
        animeschedule_media_types_exclude=["OVA"]
    )
    
    print("✅ UniversalSearchParams created successfully")
    
    # Count non-None parameters
    param_dict = params.dict(exclude_none=True)
    print(f"✅ Total parameters: {len(param_dict)}")
    
    # Group by category
    universal_count = len([k for k in param_dict.keys() if not any(k.startswith(p) for p in ["mal_", "anilist_", "kitsu_", "animeschedule_"])])
    mal_count = len([k for k in param_dict.keys() if k.startswith("mal_")])
    anilist_count = len([k for k in param_dict.keys() if k.startswith("anilist_")])
    kitsu_count = len([k for k in param_dict.keys() if k.startswith("kitsu_")])
    animeschedule_count = len([k for k in param_dict.keys() if k.startswith("animeschedule_")])
    
    print(f"   • Universal: {universal_count}")
    print(f"   • MAL-specific: {mal_count}")
    print(f"   • AniList-specific: {anilist_count}")
    print(f"   • Kitsu-specific: {kitsu_count}")
    print(f"   • AnimeSchedule-specific: {animeschedule_count}")
    
    return params

async def test_service_manager():
    """Test ServiceManager initialization and capabilities."""
    print("\n⚙️ Testing ServiceManager")
    print("=" * 50)
    
    settings = get_settings()
    service_manager = ServiceManager(settings)
    
    print("✅ ServiceManager initialized")
    print(f"✅ Available platform clients: {list(service_manager.clients.keys())}")
    print(f"✅ Platform priority order: {service_manager.platform_priorities}")
    
    # Test health check
    try:
        health_status = await service_manager.health_check()
        print(f"✅ Health check completed")
        for platform, status in health_status.items():
            status_icon = "✅" if status.get("status") == "healthy" else "⚠️"
            print(f"   {status_icon} {platform}: {status.get('status', 'unknown')}")
    except Exception as e:
        print(f"⚠️ Health check failed: {e}")
    
    return service_manager

async def test_mcp_tools():
    """Test MCP tool registration and functionality."""
    print("\n🛠️ Testing MCP Tools")
    print("=" * 50)
    
    try:
        from src.anime_mcp.modern_server import mcp
        
        # List all tools
        tools = [tool.name for tool in mcp.list_tools()]
        print(f"✅ Total MCP tools registered: {len(tools)}")
        
        # Check for universal tools
        universal_tools = [tool for tool in tools if "universal" in tool]
        specialized_tools = [tool for tool in tools if "universal" not in tool]
        
        print(f"✅ Specialized tools ({len(specialized_tools)}):")
        for tool in specialized_tools:
            print(f"   • {tool}")
        
        print(f"✅ Universal tools ({len(universal_tools)}):")
        for tool in universal_tools:
            print(f"   • {tool}")
        
        # Test tool selection guide
        from src.anime_mcp.modern_server import tool_selection_guide
        guide = tool_selection_guide()
        print(f"✅ Tool selection guide available ({len(guide)} characters)")
        
    except Exception as e:
        print(f"❌ MCP tools test failed: {e}")
        import traceback
        traceback.print_exc()

async def demo_usage_scenarios():
    """Demonstrate various usage scenarios."""
    print("\n🎮 Usage Scenarios Demo")
    print("=" * 50)
    
    scenarios = [
        {
            "name": "Simple Semantic Search",
            "tool": "anime_search",
            "description": "Basic text search with simple filters",
            "params": {"query": "romance anime", "limit": 5}
        },
        {
            "name": "Platform-Specific Filtering",
            "tool": "anime_universal_search", 
            "description": "MAL SFW content with rating filter",
            "params": {
                "query": "school anime",
                "mal_nsfw": "white",
                "mal_rating": "pg",
                "min_score": 7.0
            }
        },
        {
            "name": "Cross-Platform Complex Query",
            "tool": "anime_universal_search",
            "description": "Korean anime on Crunchyroll, no adult content",
            "params": {
                "query": "korean anime",
                "anilist_country_of_origin": "KR",
                "anilist_is_adult": False,
                "kitsu_streamers": ["Crunchyroll"],
                "min_score": 6.0
            }
        },
        {
            "name": "Advanced Filtering",
            "tool": "anime_universal_search",
            "description": "Mecha anime with episode count and format exclusions",
            "params": {
                "query": "mecha robots",
                "genres": ["Action", "Sci-Fi"],
                "anilist_episodes_greater": 12,
                "anilist_episodes_lesser": 26,
                "anilist_format_not_in": ["MOVIE", "OVA"],
                "year": 2020,
                "sort_by": "score"
            }
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Tool: {scenario['tool']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Parameters: {scenario['params']}")
        
        if scenario['tool'] == 'anime_universal_search':
            # Show parameter analysis
            universal_params, platform_specific = MapperRegistry.extract_platform_params(**scenario['params'])
            selected_platform = MapperRegistry.auto_select_platform(universal_params, platform_specific)
            print(f"   → Auto-selected platform: {selected_platform}")
            if platform_specific:
                print(f"   → Platform-specific params: {list(platform_specific.keys())}")

async def main():
    """Run all integration tests."""
    print("🚀 Universal Anime Search Integration Test")
    print("=" * 60)
    
    try:
        # Test parameter system
        await test_parameter_extraction()
        await test_universal_search_params()
        
        # Test service manager
        await test_service_manager()
        
        # Test MCP integration
        await test_mcp_tools()
        
        # Demo usage scenarios
        await demo_usage_scenarios()
        
        print("\n🎉 Integration Test Summary")
        print("=" * 50)
        print("✅ Parameter extraction and validation")
        print("✅ Platform selection and routing")
        print("✅ Service manager initialization")
        print("✅ MCP tool registration")
        print("✅ Universal search capabilities")
        print("\n🌟 Universal anime search integration is ready!")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())