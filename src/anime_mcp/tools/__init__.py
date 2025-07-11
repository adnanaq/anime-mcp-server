"""
Tiered MCP Tools - Progressive complexity anime search tools.

This module provides a 4-tier architecture for anime search tools:
- Tier 1 (Basic): 8 fields, 80% of queries - Fast, essential information
- Tier 2 (Standard): 15 fields, 95% of queries - Enhanced filtering and details
- Tier 3 (Detailed): 25 fields, 99% of queries - Cross-platform data and comprehensive info
- Tier 4 (Comprehensive): 40+ fields, 100% of queries - Complete data with analytics

Each tier is optimized for different use cases, with higher tiers providing more
comprehensive information at the cost of increased processing time and complexity.
"""

from .tier1_basic_tools import register_basic_tools
from .tier2_standard_tools import register_standard_tools
from .tier3_detailed_tools import register_detailed_tools
from .tier4_comprehensive_tools import register_comprehensive_tools

# Specialized tools (kept for active functionality)
# Note: schedule_tools and enrichment_tools are actively used by QueryAnalyzer and AnimeSwarm
# from .schedule_tools import register_schedule_tools
# from .enrichment_tools import register_enrichment_tools

__all__ = [
    # Tiered tools (new architecture)
    "register_basic_tools",
    "register_standard_tools",
    "register_detailed_tools",
    "register_comprehensive_tools",
    # Specialized tools (kept for active functionality)
    # "register_schedule_tools",
    # "register_enrichment_tools",
]


# Tool registration functions for each tier
def register_all_tiered_tools(mcp):
    """Register all tiered tools with the MCP server."""
    register_basic_tools(mcp)
    register_standard_tools(mcp)
    register_detailed_tools(mcp)
    register_comprehensive_tools(mcp)

# Tier information for documentation and selection
TIER_INFO = {
    "basic": {
        "name": "Basic (Tier 1)",
        "description": "Essential anime information for quick searches",
        "fields": 8,
        "coverage": "80% of queries",
        "use_cases": ["Quick searches", "Basic recommendations", "Simple filtering"],
        "performance": "Fastest",
        "tools": [
            "search_anime_basic",
            "get_anime_basic",
            "find_similar_anime_basic",
            "get_seasonal_anime_basic",
        ],
    },
    "standard": {
        "name": "Standard (Tier 2)",
        "description": "Enhanced anime information with advanced filtering",
        "fields": 15,
        "coverage": "95% of queries",
        "use_cases": ["Advanced filtering", "Detailed search", "Genre-based discovery"],
        "performance": "Fast",
        "tools": [
            "search_anime_standard",
            "get_anime_standard",
            "find_similar_anime_standard",
            "get_seasonal_anime_standard",
            "search_by_genre_standard",
        ],
    },
    "detailed": {
        "name": "Detailed (Tier 3)",
        "description": "Comprehensive anime information with cross-platform data",
        "fields": 25,
        "coverage": "99% of queries",
        "use_cases": ["Cross-platform analysis", "Detailed comparison", "Research"],
        "performance": "Moderate",
        "tools": [
            "search_anime_detailed",
            "get_anime_detailed",
            "find_similar_anime_detailed",
            "get_seasonal_anime_detailed",
            "advanced_anime_analysis",
        ],
    },
    "comprehensive": {
        "name": "Comprehensive (Tier 4)",
        "description": "Complete anime information with analytics and predictions",
        "fields": "40+",
        "coverage": "100% of queries",
        "use_cases": ["Complete analysis", "Market research", "Predictive analytics"],
        "performance": "Thorough",
        "tools": [
            "search_anime_comprehensive",
            "get_anime_comprehensive",
            "find_similar_anime_comprehensive",
            "comprehensive_anime_analytics",
        ],
    },
}


# Tool selection helper
def get_recommended_tier(
    query_complexity: str = "medium", response_time_priority: str = "balanced"
) -> str:
    """
    Get recommended tier based on query complexity and response time priority.

    Args:
        query_complexity: "simple", "medium", "complex", "analytical"
        response_time_priority: "speed", "balanced", "completeness"

    Returns:
        Recommended tier: "basic", "standard", "detailed", or "comprehensive"
    """
    if query_complexity == "simple":
        if response_time_priority == "speed":
            return "basic"
        elif response_time_priority == "balanced":
            return "standard"
        else:
            return "detailed"

    elif query_complexity == "medium":
        if response_time_priority == "speed":
            return "standard"
        elif response_time_priority == "balanced":
            return "detailed"
        else:
            return "comprehensive"

    elif query_complexity == "complex":
        if response_time_priority == "speed":
            return "detailed"
        else:
            return "comprehensive"

    elif query_complexity == "analytical":
        return "comprehensive"

    # Default to standard for unknown inputs
    return "standard"


def get_tier_tools(tier: str) -> list:
    """Get list of tools available in a specific tier."""
    if tier in TIER_INFO:
        return TIER_INFO[tier]["tools"]
    return []


def get_all_tiers() -> dict:
    """Get information about all available tiers."""
    return TIER_INFO
