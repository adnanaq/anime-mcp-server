"""
Platform-specific MCP tools for anime search and discovery.

This module provides focused, schema-clear tools for each anime platform,
following MCP best practices for tool design and capability exposure.
"""

from .mal_tools import search_anime_mal, get_anime_mal, get_mal_seasonal_anime
from .anilist_tools import search_anime_anilist, get_anime_anilist  
from .schedule_tools import search_anime_schedule, get_schedule_data, get_currently_airing
from .kitsu_tools import search_anime_kitsu, get_anime_kitsu, search_streaming_platforms
from .jikan_tools import search_anime_jikan, get_anime_jikan, get_jikan_seasonal
from .semantic_tools import anime_semantic_search, anime_similar, anime_vector_stats
from .enrichment_tools import (
    compare_anime_ratings_cross_platform,
    get_cross_platform_anime_data,
    correlate_anime_across_platforms,
    get_streaming_availability_multi_platform,
    detect_platform_discrepancies,
)

__all__ = [
    # MAL tools
    "search_anime_mal",
    "get_anime_mal",
    "get_mal_seasonal_anime",
    
    # AniList tools  
    "search_anime_anilist",
    "get_anime_anilist",
    
    # AnimeSchedule tools
    "search_anime_schedule", 
    "get_schedule_data",
    "get_currently_airing",
    
    # Kitsu tools
    "search_anime_kitsu",
    "get_anime_kitsu",
    "search_streaming_platforms",
    
    # Jikan tools
    "search_anime_jikan",
    "get_anime_jikan",
    "get_jikan_seasonal",
    
    # Semantic search tools
    "anime_semantic_search",
    "anime_similar",
    "anime_vector_stats",
    
    # Cross-platform enrichment tools
    "compare_anime_ratings_cross_platform",
    "get_cross_platform_anime_data",
    "correlate_anime_across_platforms", 
    "get_streaming_availability_multi_platform",
    "detect_platform_discrepancies",
]