"""Modern structured response models for MCP tools.

This module defines clean, tiered response schemas that replace the
over-engineered Universal parameter system. Follows 2025 LLM best practices:
- Direct tool calls with focused parameters
- Structured outputs for consistent consumption
- Progressive complexity through tiered models
- JSON Schema compatibility for excellent LLM integration

DESIGN PRINCIPLES:
1. Usage-Driven Design: Based on actual usage patterns, not theoretical completeness
2. Progressive Complexity: 4 tiers for different query complexity levels
3. Structured Outputs: Mandatory for modern LLM consumption
4. Performance First: Optimized for common use cases (80% of queries)
5. Modern LLM Standards: Direct tools, clear parameters, excellent documentation
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class AnimeStatus(str, Enum):
    """Standard anime status enum."""
    FINISHED = "finished"
    RELEASING = "releasing"
    NOT_YET_RELEASED = "not_yet_released"
    CANCELLED = "cancelled"
    HIATUS = "hiatus"


class AnimeType(str, Enum):
    """Standard anime type/format enum."""
    TV = "tv"
    MOVIE = "movie"
    OVA = "ova"
    ONA = "ona"
    SPECIAL = "special"
    MUSIC = "music"


class AnimeRating(str, Enum):
    """Standard content rating enum."""
    G = "g"
    PG = "pg"
    PG_13 = "pg_13"
    R = "r"
    R_PLUS = "r_plus"
    RX = "rx"


# =============================================================================
# TIER 1: BASIC RESPONSE - Core search results (80% of queries)
# =============================================================================

class BasicAnimeResult(BaseModel):
    """Basic anime result - 8 essential fields for core search functionality.
    
    Covers 80% of use cases with minimal overhead.
    Perfect for simple searches like "anime similar to Death Note".
    """
    
    id: str = Field(..., description="Unique anime identifier")
    title: str = Field(..., description="Primary anime title")
    score: Optional[float] = Field(None, ge=0, le=10, description="User rating (0-10)")
    year: Optional[int] = Field(None, ge=1900, le=2030, description="Release year")
    type: Optional[AnimeType] = Field(None, description="Media type (TV, Movie, OVA, etc.)")
    genres: List[str] = Field(default=[], description="Primary genres")
    image_url: Optional[str] = Field(None, description="Cover image URL")
    synopsis: Optional[str] = Field(None, description="Short plot summary", max_length=300)


class BasicSearchResponse(BaseModel):
    """Response for Tier 1 basic search tools."""
    
    results: List[BasicAnimeResult] = Field(..., description="Search results")
    total: int = Field(..., ge=0, description="Total results available")
    query: str = Field(..., description="Original search query")
    processing_time_ms: float = Field(..., ge=0, description="Response time in milliseconds")


# =============================================================================
# TIER 2: STANDARD RESPONSE - Advanced filtering (95% of queries)
# =============================================================================

class StandardAnimeResult(BaseModel):
    """Standard anime result - 15 fields for advanced filtering.
    
    Handles 95% of use cases with moderate detail.
    Perfect for queries like "action anime from 2020 with high ratings".
    """
    
    # Core fields (from BasicAnimeResult)
    id: str = Field(..., description="Unique anime identifier")
    title: str = Field(..., description="Primary anime title")
    score: Optional[float] = Field(None, ge=0, le=10, description="User rating (0-10)")
    year: Optional[int] = Field(None, ge=1900, le=2030, description="Release year")
    type: Optional[AnimeType] = Field(None, description="Media type")
    genres: List[str] = Field(default=[], description="Primary genres")
    image_url: Optional[str] = Field(None, description="Cover image URL")
    synopsis: Optional[str] = Field(None, description="Plot summary", max_length=500)
    
    # Extended fields for advanced filtering
    status: Optional[AnimeStatus] = Field(None, description="Release status")
    episodes: Optional[int] = Field(None, ge=0, description="Episode count")
    duration: Optional[int] = Field(None, ge=0, description="Episode duration (minutes)")
    studios: List[str] = Field(default=[], description="Animation studios")
    rating: Optional[AnimeRating] = Field(None, description="Content rating")
    popularity: Optional[int] = Field(None, ge=0, description="Popularity ranking")
    aired_from: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")


class StandardSearchResponse(BaseModel):
    """Response for Tier 2 standard search tools."""
    
    results: List[StandardAnimeResult] = Field(..., description="Search results")
    total: int = Field(..., ge=0, description="Total results available")
    query: str = Field(..., description="Original search query")
    filters_applied: Dict[str, Any] = Field(default={}, description="Filters used")
    processing_time_ms: float = Field(..., ge=0, description="Response time in milliseconds")


# =============================================================================
# TIER 3: DETAILED RESPONSE - Cross-platform comparison (99% of queries)
# =============================================================================

class DetailedAnimeResult(BaseModel):
    """Detailed anime result - 25 fields for comprehensive queries.
    
    Handles 99% of use cases with full detail.
    Perfect for cross-platform comparisons and detailed analysis.
    """
    
    # Core fields (from StandardAnimeResult)
    id: str = Field(..., description="Unique anime identifier")
    title: str = Field(..., description="Primary anime title")
    score: Optional[float] = Field(None, ge=0, le=10, description="User rating (0-10)")
    year: Optional[int] = Field(None, ge=1900, le=2030, description="Release year")
    type: Optional[AnimeType] = Field(None, description="Media type")
    genres: List[str] = Field(default=[], description="Primary genres")
    image_url: Optional[str] = Field(None, description="Cover image URL")
    synopsis: Optional[str] = Field(None, description="Full plot summary")
    status: Optional[AnimeStatus] = Field(None, description="Release status")
    episodes: Optional[int] = Field(None, ge=0, description="Episode count")
    duration: Optional[int] = Field(None, ge=0, description="Episode duration (minutes)")
    studios: List[str] = Field(default=[], description="Animation studios")
    rating: Optional[AnimeRating] = Field(None, description="Content rating")
    popularity: Optional[int] = Field(None, ge=0, description="Popularity ranking")
    aired_from: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    
    # Extended fields for detailed analysis
    title_english: Optional[str] = Field(None, description="English title")
    title_japanese: Optional[str] = Field(None, description="Japanese title")
    title_synonyms: List[str] = Field(default=[], description="Alternative titles")
    aired_to: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    season: Optional[str] = Field(None, description="Release season")
    source: Optional[str] = Field(None, description="Source material")
    producers: List[str] = Field(default=[], description="Producer companies")
    licensors: List[str] = Field(default=[], description="Licensing companies")
    themes: List[str] = Field(default=[], description="Thematic elements")
    demographics: List[str] = Field(default=[], description="Target demographics")


class DetailedSearchResponse(BaseModel):
    """Response for Tier 3 detailed search tools."""
    
    results: List[DetailedAnimeResult] = Field(..., description="Search results")
    total: int = Field(..., ge=0, description="Total results available")
    query: str = Field(..., description="Original search query")
    filters_applied: Dict[str, Any] = Field(default={}, description="Filters used")
    sources_used: List[str] = Field(default=[], description="Data sources queried")
    processing_time_ms: float = Field(..., ge=0, description="Response time in milliseconds")


# =============================================================================
# TIER 4: COMPREHENSIVE RESPONSE - Ultra-complex queries (100% coverage)
# =============================================================================

class PlatformRating(BaseModel):
    """Platform-specific rating information."""
    
    platform: str = Field(..., description="Platform name (MAL, AniList, etc.)")
    score: Optional[float] = Field(None, ge=0, le=10, description="Normalized score (0-10)")
    raw_score: Optional[float] = Field(None, description="Original platform score")
    votes: Optional[int] = Field(None, ge=0, description="Number of votes")
    rank: Optional[int] = Field(None, ge=0, description="Platform ranking")


class Character(BaseModel):
    """Character information."""
    
    name: str = Field(..., description="Character name")
    role: str = Field(..., description="Character role (main, supporting, etc.)")
    image_url: Optional[str] = Field(None, description="Character image URL")


class StaffMember(BaseModel):
    """Staff member information."""
    
    name: str = Field(..., description="Staff member name")
    role: str = Field(..., description="Staff role (director, writer, etc.)")
    image_url: Optional[str] = Field(None, description="Staff member image URL")


class ComprehensiveAnimeResult(BaseModel):
    """Comprehensive anime result - Full platform data for ultra-complex queries.
    
    Handles 100% of use cases with complete information.
    Perfect for advanced AI analysis and discovery workflows.
    """
    
    # Core identification
    id: str = Field(..., description="Unique anime identifier")
    platform_ids: Dict[str, Union[int, str]] = Field(default={}, description="Platform-specific IDs")
    
    # Titles and metadata
    title: str = Field(..., description="Primary anime title")
    title_english: Optional[str] = Field(None, description="English title")
    title_japanese: Optional[str] = Field(None, description="Japanese title")
    title_synonyms: List[str] = Field(default=[], description="Alternative titles")
    
    # Core information
    type: Optional[AnimeType] = Field(None, description="Media type")
    status: Optional[AnimeStatus] = Field(None, description="Release status")
    episodes: Optional[int] = Field(None, ge=0, description="Episode count")
    duration: Optional[int] = Field(None, ge=0, description="Episode duration (minutes)")
    
    # Content classification
    genres: List[str] = Field(default=[], description="Primary genres")
    themes: List[str] = Field(default=[], description="Thematic elements")
    demographics: List[str] = Field(default=[], description="Target demographics")
    rating: Optional[AnimeRating] = Field(None, description="Content rating")
    
    # Production information
    studios: List[str] = Field(default=[], description="Animation studios")
    producers: List[str] = Field(default=[], description="Producer companies")
    licensors: List[str] = Field(default=[], description="Licensing companies")
    
    # Temporal information
    year: Optional[int] = Field(None, ge=1900, le=2030, description="Release year")
    season: Optional[str] = Field(None, description="Release season")
    aired_from: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    aired_to: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    
    # Content and media
    synopsis: Optional[str] = Field(None, description="Full plot summary")
    image_url: Optional[str] = Field(None, description="Cover image URL")
    image_large: Optional[str] = Field(None, description="Large cover image URL")
    source: Optional[str] = Field(None, description="Source material")
    
    # Ratings and popularity
    platform_ratings: List[PlatformRating] = Field(default=[], description="Multi-platform ratings")
    average_score: Optional[float] = Field(None, ge=0, le=10, description="Average across platforms")
    popularity: Optional[int] = Field(None, ge=0, description="Popularity ranking")
    
    # People and characters
    characters: List[Character] = Field(default=[], description="Main characters")
    staff: List[StaffMember] = Field(default=[], description="Staff members")
    
    # Metadata
    data_quality: Optional[float] = Field(None, ge=0, le=1, description="Data completeness score")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")


class ComprehensiveSearchResponse(BaseModel):
    """Response for Tier 4 comprehensive search tools."""
    
    results: List[ComprehensiveAnimeResult] = Field(..., description="Search results")
    total: int = Field(..., ge=0, description="Total results available")
    query: str = Field(..., description="Original search query")
    filters_applied: Dict[str, Any] = Field(default={}, description="Filters used")
    sources_used: List[str] = Field(default=[], description="Data sources queried")
    processing_time_ms: float = Field(..., ge=0, description="Response time in milliseconds")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Result confidence")


# =============================================================================
# UTILITY MODELS
# =============================================================================

class SearchError(BaseModel):
    """Structured error response."""
    
    error_type: str = Field(..., description="Error category")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class SearchStats(BaseModel):
    """Search statistics and performance metrics."""
    
    total_queries: int = Field(..., ge=0, description="Total queries processed")
    average_response_time_ms: float = Field(..., ge=0, description="Average response time")
    cache_hit_rate: float = Field(..., ge=0, le=1, description="Cache hit rate")
    sources_health: Dict[str, bool] = Field(default={}, description="Source availability")
    last_updated: datetime = Field(default_factory=datetime.now, description="Stats timestamp")


# =============================================================================
# RESPONSE TYPE UNION
# =============================================================================

# Union type for all response tiers
TieredResponse = Union[
    BasicSearchResponse,
    StandardSearchResponse,
    DetailedSearchResponse,
    ComprehensiveSearchResponse
]

# Union type for all result tiers
TieredResult = Union[
    BasicAnimeResult,
    StandardAnimeResult,
    DetailedAnimeResult,
    ComprehensiveAnimeResult
]