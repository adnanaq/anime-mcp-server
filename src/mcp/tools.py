"""
MCP Tools - Utility functions for FastMCP server

This module provides helper functions and utilities for the FastMCP server.
The actual tools are defined in server.py using @mcp.tool decorators.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def validate_search_query(query: str) -> str:
    """Validate and clean search query.
    
    Args:
        query: Raw search query string
        
    Returns:
        Cleaned and validated query string
        
    Raises:
        ValueError: If query is invalid
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")
    
    # Clean the query
    cleaned_query = query.strip()
    
    # Limit query length
    if len(cleaned_query) > 500:
        cleaned_query = cleaned_query[:500]
        logger.warning(f"Query truncated to 500 characters: {cleaned_query[:50]}...")
    
    return cleaned_query


def validate_anime_id(anime_id: str) -> str:
    """Validate anime ID format.
    
    Args:
        anime_id: Anime identifier string
        
    Returns:
        Validated anime ID
        
    Raises:
        ValueError: If anime ID is invalid
    """
    if not anime_id or not anime_id.strip():
        raise ValueError("Anime ID cannot be empty")
    
    cleaned_id = anime_id.strip()
    
    # Basic format validation - should be non-empty string
    if len(cleaned_id) < 3:
        raise ValueError("Anime ID too short")
    
    if len(cleaned_id) > 100:
        raise ValueError("Anime ID too long")
    
    return cleaned_id


def validate_limit(limit: int, max_limit: int = 50) -> int:
    """Validate and normalize limit parameter.
    
    Args:
        limit: Requested limit value
        max_limit: Maximum allowed limit
        
    Returns:
        Validated limit within acceptable range
    """
    if limit is None:
        return 10  # Default limit
    
    if not isinstance(limit, int):
        try:
            limit = int(limit)
        except (ValueError, TypeError):
            logger.warning(f"Invalid limit value, using default: {10}")
            return 10
    
    # Clamp to valid range
    if limit < 1:
        limit = 1
    elif limit > max_limit:
        limit = max_limit
        logger.info(f"Limit clamped to maximum: {max_limit}")
    
    return limit


def format_anime_result(anime: Dict[str, Any]) -> Dict[str, Any]:
    """Format anime result for MCP response.
    
    Args:
        anime: Raw anime data from database
        
    Returns:
        Formatted anime data suitable for MCP clients
    """
    if not anime:
        return {}
    
    # Extract essential fields with defaults
    formatted = {
        "anime_id": anime.get("anime_id", ""),
        "title": anime.get("title", "Unknown Title"),
        "synopsis": anime.get("synopsis", "No synopsis available"),
        "type": anime.get("type", "Unknown"),
        "episodes": anime.get("episodes", 0),
        "year": anime.get("year"),
        "season": anime.get("season"),
        "tags": anime.get("tags", []),
        "studios": anime.get("studios", []),
        "picture": anime.get("picture", ""),
        "data_quality_score": anime.get("data_quality_score", 0.0)
    }
    
    # Add platform IDs if available
    platform_ids = {}
    for platform in ["myanimelist_id", "anilist_id", "kitsu_id", "anidb_id"]:
        if anime.get(platform):
            platform_ids[platform] = anime[platform]
    
    if platform_ids:
        formatted["platform_ids"] = platform_ids
    
    # Add similarity score if present (for similarity search results)
    if "similarity_score" in anime:
        formatted["similarity_score"] = anime["similarity_score"]
    
    return formatted


def format_search_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format list of anime search results.
    
    Args:
        results: List of raw anime data from database
        
    Returns:
        List of formatted anime data suitable for MCP clients
    """
    if not results:
        return []
    
    return [format_anime_result(anime) for anime in results]


def build_recommendation_query(
    genres: Optional[str] = None,
    year: Optional[int] = None,
    anime_type: Optional[str] = None
) -> str:
    """Build search query from recommendation parameters.
    
    Args:
        genres: Comma-separated genre list
        year: Release year
        anime_type: Anime type (TV, Movie, etc.)
        
    Returns:
        Search query string for recommendations
    """
    query_parts = []
    
    # Add genres
    if genres:
        genre_list = [g.strip() for g in genres.split(",") if g.strip()]
        query_parts.extend(genre_list)
    
    # Add year
    if year:
        query_parts.append(str(year))
    
    # Add type
    if anime_type:
        query_parts.append(anime_type.strip())
    
    # Default query if no preferences provided
    return " ".join(query_parts) if query_parts else "popular anime"


def validate_recommendation_filters(
    anime: Dict[str, Any],
    genres: Optional[str] = None,
    year: Optional[int] = None,
    anime_type: Optional[str] = None
) -> bool:
    """Check if anime matches recommendation filters.
    
    Args:
        anime: Anime data to check
        genres: Required genres (comma-separated)
        year: Required release year
        anime_type: Required anime type
        
    Returns:
        True if anime matches all specified filters
    """
    # Check year filter
    if year and anime.get("year") != year:
        return False
    
    # Check type filter
    if anime_type:
        anime_type_lower = anime_type.lower().strip()
        actual_type = anime.get("type", "").lower().strip()
        if actual_type != anime_type_lower:
            return False
    
    # Check genre filter
    if genres:
        anime_tags = [tag.lower() for tag in anime.get("tags", [])]
        requested_genres = [g.strip().lower() for g in genres.split(",") if g.strip()]
        
        # Check if any requested genre is present in anime tags
        if not any(genre in anime_tags for genre in requested_genres):
            return False
    
    return True


def get_anime_summary(anime: Dict[str, Any]) -> str:
    """Generate a brief summary of an anime for logging/display.
    
    Args:
        anime: Anime data dictionary
        
    Returns:
        Brief summary string
    """
    if not anime:
        return "Unknown anime"
    
    title = anime.get("title", "Unknown")
    year = anime.get("year", "Unknown year")
    anime_type = anime.get("type", "Unknown type")
    
    return f"{title} ({year}, {anime_type})"


class MCPToolError(Exception):
    """Custom exception for MCP tool errors."""
    
    def __init__(self, message: str, error_code: str = "TOOL_ERROR"):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class ValidationError(MCPToolError):
    """Exception raised for input validation errors."""
    
    def __init__(self, message: str):
        super().__init__(message, "VALIDATION_ERROR")


class DatabaseError(MCPToolError):
    """Exception raised for database-related errors."""
    
    def __init__(self, message: str):
        super().__init__(message, "DATABASE_ERROR")


class SearchError(MCPToolError):
    """Exception raised for search-related errors."""
    
    def __init__(self, message: str):
        super().__init__(message, "SEARCH_ERROR")