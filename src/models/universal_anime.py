"""Universal anime schema models for cross-platform data integration.

This module defines the universal schema that abstracts away differences
between ALL 9 major anime data sources, providing a consistent interface 
for LLM tools and multi-source queries.

SUPPORTED DATA SOURCES (9 total):
1. Anime Offline Database - Static JSON baseline (38,894 entries)
2. MyAnimeList API v2 - Official MAL REST API with OAuth2  
3. MAL/Jikan - Unofficial MAL REST API (no auth required)
4. AniList GraphQL API - Modern GraphQL with rich relationships
5. Kitsu JSON:API - Standard JSON:API with comprehensive metadata
6. AniDB API - Detailed episode-level information
7. Anime-Planet - Web scraping with rich structured data
8. AnimeSchedule API - Broadcasting schedules and timing data  
9. AniSearch - European anime database via web scraping

Based on comprehensive property mapping analysis identifying guaranteed 
universal properties (12), high-confidence properties (9), and 
medium-confidence properties (3) across all sources.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class AnimeStatus(str, Enum):
    """Universal anime status enum mapped from all data sources."""
    
    # Standardized status values
    FINISHED = "FINISHED"           # completed/finished/ended
    RELEASING = "RELEASING"         # airing/ongoing/current/publishing
    NOT_YET_RELEASED = "NOT_YET_RELEASED"  # upcoming/not_yet_aired/tba
    CANCELLED = "CANCELLED"         # cancelled/discontinued
    HIATUS = "HIATUS"              # on_hiatus/paused
    
    @classmethod
    def from_source_value(cls, value: str, source: str = "") -> "AnimeStatus":
        """Convert source-specific status values to universal status.
        
        This handles the mapping chaos identified in our architectural analysis across ALL 9 sources:
        - Offline DB: "FINISHED", "RELEASING", "UPCOMING"
        - MAL API v2: "finished", "currently_airing", "not_yet_aired" 
        - MAL/Jikan: "complete", "airing", "upcoming"
        - AniList: "FINISHED", "RELEASING", "NOT_YET_RELEASED", "CANCELLED", "HIATUS"
        - Kitsu: "finished", "current", "upcoming"
        - AniDB: "complete", "ongoing", "upcoming"  
        - Anime-Planet: "finished", "ongoing", "not yet aired"
        - AnimeSchedule: "finished", "airing", "upcoming"
        - AniSearch: "finished", "currently airing", "upcoming"
        """
        value_lower = value.lower().strip().replace("_", " ")
        
        # Finished/Completed variants (all sources)
        if value_lower in [
            "finished", "completed", "complete", "ended", 
            "finished airing", "finished airing"
        ]:
            return cls.FINISHED
            
        # Currently airing/releasing variants (all sources)
        elif value_lower in [
            "airing", "releasing", "current", "ongoing", "publishing",
            "currently airing", "currently releasing", "currently publishing"
        ]:
            return cls.RELEASING
            
        # Not yet released variants (all sources)
        elif value_lower in [
            "not yet released", "not yet aired", "upcoming", "tba", "announced",
            "not yet airing", "not yet published", "to be aired", "to be announced"
        ]:
            return cls.NOT_YET_RELEASED
            
        # Cancelled variants
        elif value_lower in ["cancelled", "canceled", "discontinued"]:
            return cls.CANCELLED
            
        # Hiatus variants (mainly AniList)
        elif value_lower in ["hiatus", "on hiatus", "paused"]:
            return cls.HIATUS
            
        # Default fallback
        else:
            return cls.NOT_YET_RELEASED


class AnimeFormat(str, Enum):
    """Universal anime format/type enum mapped from all data sources."""
    
    TV = "TV"                      # TV series
    MOVIE = "MOVIE"                # Theatrical films
    OVA = "OVA"                    # Original Video Animation
    ONA = "ONA"                    # Original Net Animation
    SPECIAL = "SPECIAL"            # TV specials/extras
    MUSIC = "MUSIC"                # Music videos
    
    @classmethod
    def from_source_value(cls, value: str, source: str = "") -> "AnimeFormat":
        """Convert source-specific format values to universal format across ALL 9 sources.
        
        Format mappings across sources:
        - Offline DB: "TV", "Movie", "Special", "OVA", "ONA", "Music"
        - MAL API v2: "tv", "movie", "ova", "special", "ona", "music"  
        - MAL/Jikan: "TV", "Movie", "OVA", "Special", "ONA", "Music"
        - AniList: "TV", "TV_SHORT", "MOVIE", "SPECIAL", "OVA", "ONA", "MUSIC"
        - Kitsu: "TV", "movie", "OVA", "ONA", "special", "music"
        - AniDB: "TV Series", "Movie", "OVA", "Web", "TV Special", "Music Video"
        - Anime-Planet: "TV", "Movie", "OVA", "ONA", "Special", "Music Video"
        - AnimeSchedule: "TV", "Movie", "OVA", "ONA", "Special"
        - AniSearch: "TV", "Movie", "OVA", "ONA", "Special"
        """
        value_normalized = value.upper().strip().replace("_", " ").replace("-", " ")
        
        # TV variants (all sources)
        if value_normalized in [
            "TV", "TV SERIES", "TV SHORT", "TV ANIME", "TELEVISION", "SERIES"
        ]:
            return cls.TV
            
        # Movie variants (all sources)  
        elif value_normalized in [
            "MOVIE", "FILM", "THEATRICAL", "CINEMA", "FEATURE FILM"
        ]:
            return cls.MOVIE
            
        # OVA variants (all sources)
        elif value_normalized in [
            "OVA", "ORIGINAL VIDEO ANIMATION", "VIDEO"
        ]:
            return cls.OVA
            
        # ONA variants (all sources)
        elif value_normalized in [
            "ONA", "ORIGINAL NET ANIMATION", "WEB", "WEB ANIME", "ONLINE", "NET"
        ]:
            return cls.ONA
            
        # Special variants (all sources)
        elif value_normalized in [
            "SPECIAL", "TV SPECIAL", "EXTRA", "BONUS", "SP"
        ]:
            return cls.SPECIAL
            
        # Music variants (all sources)  
        elif value_normalized in [
            "MUSIC", "MUSIC VIDEO", "MV", "PV", "PROMOTIONAL VIDEO"
        ]:
            return cls.MUSIC
            
        # Default fallback
        else:
            return cls.TV


class AnimeRating(str, Enum):
    """Universal content rating enum."""
    
    G = "G"                        # General audiences
    PG = "PG"                      # Parental guidance
    PG13 = "PG13"                  # Parents strongly cautioned  
    R = "R"                        # Restricted
    R_PLUS = "R_PLUS"              # 17+ recommended
    RX = "RX"                      # Adult content


class AnimeSeason(str, Enum):
    """Universal season enum."""
    
    WINTER = "WINTER"              # December-February
    SPRING = "SPRING"              # March-May  
    SUMMER = "SUMMER"              # June-August
    FALL = "FALL"                  # September-November


class UniversalAnime(BaseModel):
    """Universal anime model supporting ALL properties from Universal Schema Foundation.
    
    This model represents the unified schema that can be populated from any
    of the 9 supported anime data sources. Properties are categorized by
    confidence level based on comprehensive cross-platform availability analysis.
    
    Coverage Analysis:
    - GUARANTEED (9/9 sources): 12 properties  
    - HIGH-CONFIDENCE (7-8/9 sources): 9 properties
    - MEDIUM-CONFIDENCE (4-6/9 sources): 3 properties
    Total: 24 universal properties across all anime platforms
    """
    
    # GUARANTEED UNIVERSAL PROPERTIES (Available in ALL 9/9 sources)
    id: str = Field(..., description="Universal unique identifier")
    title: str = Field(..., description="Primary anime title") 
    type_format: AnimeFormat = Field(..., description="Media format (TV, Movie, OVA, etc.)")
    episodes: Optional[int] = Field(None, ge=0, description="Total episode count")
    status: AnimeStatus = Field(..., description="Current release status")
    genres: List[str] = Field(default=[], description="Genre/category tags")
    score: Optional[float] = Field(None, ge=0, le=10, description="Average user rating (0-10 scale)")
    image_url: Optional[str] = Field(None, description="Cover/poster image URL")
    image_large: Optional[str] = Field(None, description="Large format cover image")
    year: Optional[int] = Field(None, ge=1900, le=2030, description="Release year")
    synonyms: List[str] = Field(default=[], description="Alternative titles")
    studios: List[str] = Field(default=[], description="Animation/production studios")
    
    # HIGH-CONFIDENCE PROPERTIES (Available in 7-8/9 sources)
    description: Optional[str] = Field(None, description="Synopsis/plot summary")
    url: Optional[str] = Field(None, description="Canonical URL to anime page")
    score_count: Optional[int] = Field(None, ge=0, description="Number of user ratings")
    title_english: Optional[str] = Field(None, description="Official English title")
    title_native: Optional[str] = Field(None, description="Native language title (usually Japanese)")
    start_date: Optional[str] = Field(None, description="Start date (ISO 8601 format)")
    season: Optional[AnimeSeason] = Field(None, description="Release season")
    end_date: Optional[str] = Field(None, description="End date (ISO 8601 format)")
    duration: Optional[int] = Field(None, ge=0, description="Episode duration in minutes")
    
    # MEDIUM-CONFIDENCE PROPERTIES (Available in 4-6/9 sources)  
    source: Optional[str] = Field(None, description="Source material type (manga, novel, original, etc.)")
    rank: Optional[int] = Field(None, ge=0, description="Overall ranking/popularity rank")
    staff: List[Dict[str, Any]] = Field(default=[], description="Staff information (directors, writers, etc.)")
    
    # ADDITIONAL UNIVERSAL PROPERTIES (Lower confidence but still valuable)
    characters: List[Dict[str, Any]] = Field(default=[], description="Character information")
    image_small: Optional[str] = Field(None, description="Small/thumbnail image")
    rating: Optional[AnimeRating] = Field(None, description="Content rating (G, PG, PG-13, etc.)")
    themes: List[str] = Field(default=[], description="Thematic tags")
    demographics: List[str] = Field(default=[], description="Target demographic tags") 
    producers: List[str] = Field(default=[], description="Production companies")
    popularity: Optional[int] = Field(None, ge=0, description="Popularity score/metric")
    
    # SOURCE PLATFORM IDS - For cross-referencing and data enrichment
    platform_ids: Dict[str, Union[int, str]] = Field(
        default={},
        description="Platform-specific IDs for cross-referencing"
    )
    
    # METADATA
    data_quality_score: Optional[float] = Field(
        None, 
        ge=0, 
        le=1, 
        description="Data completeness quality score (0-1)"
    )
    last_updated: Optional[datetime] = Field(None, description="Last data update timestamp")
    source_priority: List[str] = Field(
        default=[],
        description="Ordered list of sources used to populate this record"
    )
    
    @field_validator("platform_ids")
    @classmethod
    def validate_platform_ids(cls, v):
        """Ensure platform IDs are in expected format."""
        if not isinstance(v, dict):
            return {}
        return v
    
    def get_platform_id(self, platform: str) -> Optional[Union[int, str]]:
        """Get platform-specific ID for this anime."""
        return self.platform_ids.get(platform.lower())
    
    def set_platform_id(self, platform: str, platform_id: Union[int, str]) -> None:
        """Set platform-specific ID for this anime."""
        self.platform_ids[platform.lower()] = platform_id
    
    def calculate_quality_score(self) -> float:
        """Calculate data quality score based on populated fields."""
        total_fields = 0
        populated_fields = 0
        
        # Count guaranteed properties (highest weight)
        guaranteed_props = [
            self.title, self.type_format, self.status, 
            self.image_url, self.description, self.year
        ]
        total_fields += len(guaranteed_props) * 3
        populated_fields += sum(1 for prop in guaranteed_props if prop is not None) * 3
        
        # Count high-confidence properties (medium weight) 
        high_conf_props = [
            self.duration, self.season, self.rating,
            self.start_date, self.end_date
        ]
        total_fields += len(high_conf_props) * 2
        populated_fields += sum(1 for prop in high_conf_props if prop is not None) * 2
        
        # Count medium-confidence properties (low weight)
        medium_conf_props = [
            self.english_title, self.japanese_title, self.source_material
        ]
        total_fields += len(medium_conf_props)
        populated_fields += sum(1 for prop in medium_conf_props if prop is not None)
        
        # Count list properties (if non-empty)
        list_props = [self.genres, self.studios, self.synonyms, self.themes, self.demographics, self.producers]
        total_fields += len(list_props) * 2
        populated_fields += sum(2 for prop in list_props if prop) * 2
        
        # Calculate score
        if total_fields == 0:
            return 0.0
        return min(1.0, populated_fields / total_fields)


class UniversalSearchParams(BaseModel):
    """Universal search parameters that can be mapped to any anime data source.
    
    These parameters represent the comprehensive search capabilities across all 9 
    supported anime platforms, covering all properties from Universal Schema Foundation.
    The LLM will use these parameters, and mappers will convert them to source-specific formats.
    """
    
    # TEXT SEARCH PARAMETERS
    query: Optional[str] = Field(None, description="Search query string (title, description)")
    title: Optional[str] = Field(None, description="Specific title search")
    title_english: Optional[str] = Field(None, description="English title search")
    title_native: Optional[str] = Field(None, description="Native title search")
    
    # CONTENT CLASSIFICATION FILTERS
    genres: Optional[List[str]] = Field(None, description="Genre filters (include)")
    genres_exclude: Optional[List[str]] = Field(None, description="Genre filters (exclude)")
    status: Optional[AnimeStatus] = Field(None, description="Release status filter")
    type_format: Optional[AnimeFormat] = Field(None, description="Media format filter")
    rating: Optional[AnimeRating] = Field(None, description="Content rating filter")
    source: Optional[str] = Field(None, description="Source material filter (manga, novel, original, etc.)")
    
    # TEMPORAL FILTERS
    year: Optional[int] = Field(None, ge=1900, le=2030, description="Release year")
    season: Optional[AnimeSeason] = Field(None, description="Release season")
    start_date: Optional[str] = Field(None, description="Minimum start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="Maximum end date (YYYY-MM-DD)")
    
    # NUMERIC RANGE FILTERS
    min_score: Optional[float] = Field(None, ge=0, le=10, description="Minimum user score")
    max_score: Optional[float] = Field(None, ge=0, le=10, description="Maximum user score")
    min_score_count: Optional[int] = Field(None, ge=0, description="Minimum number of ratings")
    max_score_count: Optional[int] = Field(None, ge=0, description="Maximum number of ratings")
    min_episodes: Optional[int] = Field(None, ge=0, description="Minimum episode count")
    max_episodes: Optional[int] = Field(None, ge=0, description="Maximum episode count")
    min_duration: Optional[int] = Field(None, ge=0, description="Minimum episode duration (minutes)")
    max_duration: Optional[int] = Field(None, ge=0, description="Maximum episode duration (minutes)")
    min_rank: Optional[int] = Field(None, ge=1, description="Minimum ranking position")
    max_rank: Optional[int] = Field(None, ge=1, description="Maximum ranking position")
    min_popularity: Optional[int] = Field(None, ge=0, description="Minimum popularity score")
    max_popularity: Optional[int] = Field(None, ge=0, description="Maximum popularity score")
    
    # PRODUCTION FILTERS
    studios: Optional[List[str]] = Field(None, description="Animation studio filters")
    producers: Optional[List[str]] = Field(None, description="Producer/production company filters")
    staff: Optional[List[str]] = Field(None, description="Staff member filters (directors, writers, etc.)")
    
    # CONTENT FILTERS
    themes: Optional[List[str]] = Field(None, description="Thematic tag filters")
    demographics: Optional[List[str]] = Field(None, description="Target demographic filters")
    characters: Optional[List[str]] = Field(None, description="Character name filters")
    
    # RESULT CONTROL
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of results")
    offset: int = Field(default=0, ge=0, description="Results offset for pagination")
    sort_by: Optional[str] = Field(
        None, 
        description="Sort field (score, popularity, title, year, rank, episodes, duration, start_date)"
    )
    sort_order: Optional[str] = Field(None, description="Sort direction (asc, desc)")
    
    # PLATFORM & CONTENT OPTIONS
    include_adult: bool = Field(default=False, description="Include adult content")
    include_unaired: bool = Field(default=True, description="Include upcoming/unaired anime")
    preferred_source: Optional[str] = Field(None, description="Preferred data source platform")
    require_image: bool = Field(default=False, description="Only return results with images")
    require_description: bool = Field(default=False, description="Only return results with descriptions")
    
    @field_validator("sort_order")
    @classmethod
    def validate_sort_order(cls, v):
        """Validate sort order parameter."""
        if v is not None and v.lower() not in ["asc", "desc"]:
            raise ValueError("sort_order must be 'asc' or 'desc'")
        return v.lower() if v else v
    
    @field_validator("min_score", "max_score")
    @classmethod
    def validate_score_range(cls, v, info):
        """Validate score ranges."""
        if v is not None and (v < 0 or v > 10):
            raise ValueError("Score must be between 0 and 10")
        return v
    
    def to_source_params(self, source: str) -> Dict[str, Any]:
        """Convert universal parameters to source-specific parameters.
        
        This method provides a base conversion that mappers can override.
        It handles the fundamental parameter mapping challenge identified
        in our architectural analysis.
        """
        params = {}
        
        # Basic parameter mapping (source-specific mappers will extend this)
        if self.query:
            params["q"] = self.query
        if self.limit:
            params["limit"] = self.limit
        if self.offset:
            params["offset"] = self.offset
            
        # Status mapping for ALL 9 sources (this is where universal schema prevents chaos)
        if self.status:
            status_map = {
                # 1. Offline Database
                "offline_db": {
                    AnimeStatus.FINISHED: "FINISHED",
                    AnimeStatus.RELEASING: "RELEASING", 
                    AnimeStatus.NOT_YET_RELEASED: "UPCOMING"
                },
                # 2. MAL API v2 (official)
                "mal_api": {
                    AnimeStatus.FINISHED: "finished",
                    AnimeStatus.RELEASING: "currently_airing",
                    AnimeStatus.NOT_YET_RELEASED: "not_yet_aired"
                },
                # 3. MAL/Jikan (unofficial)
                "mal": {
                    AnimeStatus.FINISHED: "complete",
                    AnimeStatus.RELEASING: "airing", 
                    AnimeStatus.NOT_YET_RELEASED: "upcoming"
                },
                "jikan": {
                    AnimeStatus.FINISHED: "complete",
                    AnimeStatus.RELEASING: "airing", 
                    AnimeStatus.NOT_YET_RELEASED: "upcoming"
                },
                # 4. AniList GraphQL
                "anilist": {
                    AnimeStatus.FINISHED: "FINISHED",
                    AnimeStatus.RELEASING: "RELEASING",
                    AnimeStatus.NOT_YET_RELEASED: "NOT_YET_RELEASED",
                    AnimeStatus.CANCELLED: "CANCELLED",
                    AnimeStatus.HIATUS: "HIATUS"
                },
                # 5. Kitsu JSON:API
                "kitsu": {
                    AnimeStatus.FINISHED: "finished", 
                    AnimeStatus.RELEASING: "current",
                    AnimeStatus.NOT_YET_RELEASED: "upcoming"
                },
                # 6. AniDB
                "anidb": {
                    AnimeStatus.FINISHED: "complete",
                    AnimeStatus.RELEASING: "ongoing",
                    AnimeStatus.NOT_YET_RELEASED: "upcoming"
                },
                # 7. Anime-Planet (scraped)
                "anime_planet": {
                    AnimeStatus.FINISHED: "finished",
                    AnimeStatus.RELEASING: "ongoing",
                    AnimeStatus.NOT_YET_RELEASED: "not yet aired"
                },
                # 8. AnimeSchedule API
                "animeschedule": {
                    AnimeStatus.FINISHED: "finished",
                    AnimeStatus.RELEASING: "airing",
                    AnimeStatus.NOT_YET_RELEASED: "upcoming"
                },
                # 9. AniSearch (scraped)
                "anisearch": {
                    AnimeStatus.FINISHED: "finished",
                    AnimeStatus.RELEASING: "currently airing",
                    AnimeStatus.NOT_YET_RELEASED: "upcoming"
                }
            }
            source_map = status_map.get(source.lower(), {})
            if self.status in source_map:
                params["status"] = source_map[self.status]
                
        return params


class UniversalSearchResult(BaseModel):
    """Universal search result model."""
    
    anime: UniversalAnime = Field(..., description="Anime data in universal format")
    relevance_score: float = Field(..., ge=0, le=1, description="Search relevance score")
    source: str = Field(..., description="Primary data source used")
    enrichment_sources: List[str] = Field(default=[], description="Additional sources used for enrichment")


class UniversalSearchResponse(BaseModel):
    """Universal search response model."""
    
    query_params: UniversalSearchParams = Field(..., description="Original search parameters")
    results: List[UniversalSearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., ge=0, description="Total number of results available")
    processing_time_ms: float = Field(..., ge=0, description="Query processing time")
    sources_used: List[str] = Field(..., description="Data sources queried")
    cache_hit: bool = Field(default=False, description="Whether results came from cache")