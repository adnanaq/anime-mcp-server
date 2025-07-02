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
    


class AnimeFormat(str, Enum):
    """Universal anime format/type enum mapped from all data sources.
    
    Comprehensive coverage of all format types across 9 anime data sources.
    LLM can use any format - mappers will handle source-specific conversions.
    """
    
    TV = "TV"                      # TV series (standard length episodes)
    TV_SHORT = "TV_SHORT"          # TV series with short episodes (AniList specific)
    TV_SPECIAL = "TV_SPECIAL"      # TV specials/extras (Jikan specific)
    MOVIE = "MOVIE"                # Theatrical films
    OVA = "OVA"                    # Original Video Animation
    ONA = "ONA"                    # Original Net Animation  
    SPECIAL = "SPECIAL"            # TV specials/extras
    MUSIC = "MUSIC"                # Music videos/promotional videos
    CM = "CM"                      # Commercial/advertisement (Jikan specific)
    PV = "PV"                      # Promotional video (Jikan specific)
    


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
            self.title_english, self.title_native, self.source
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
    """Universal search parameters for comprehensive anime queries across all platforms.
    
    COMPREHENSIVE COVERAGE - LLM can use any parameter that makes sense:
    - Text search: query, title variants
    - Content filters: genres, themes, demographics, characters (with exclude options)
    - Metadata filters: status, format, rating, year, season, dates
    - Numeric filters: score, episodes, duration (with min/max ranges)
    - Studio/Staff filters: studios, producers, staff (with exclude options)
    - Result control: sorting, pagination, content preferences
    
    PLATFORM COMPATIBILITY - Mappers handle automatically:
    - AniList: Supports most parameters including themes, demographics, advanced filtering
    - MAL/Jikan: Core parameters supported, advanced features gracefully ignored
    - Other sources: Essential parameters supported, extras ignored
    
    USAGE: Use any parameter - mappers convert to platform-specific formats automatically.
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
    themes_exclude: Optional[List[str]] = Field(None, description="Exclude thematic tags")
    demographics: Optional[List[str]] = Field(None, description="Target demographic filters")
    demographics_exclude: Optional[List[str]] = Field(None, description="Exclude demographics")
    characters: Optional[List[str]] = Field(None, description="Character name filters")
    characters_exclude: Optional[List[str]] = Field(None, description="Exclude characters")
    
    # DURATION FILTERS
    min_duration: Optional[int] = Field(None, ge=1, description="Minimum episode duration in minutes")
    max_duration: Optional[int] = Field(None, ge=1, description="Maximum episode duration in minutes")
    
    # DATE FILTERS
    
    # STUDIO/PRODUCER EXCLUDES
    studios_exclude: Optional[List[str]] = Field(None, description="Exclude studios")
    producers_exclude: Optional[List[str]] = Field(None, description="Exclude producers")
    staff_exclude: Optional[List[str]] = Field(None, description="Exclude staff members")
    
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
    
    # MAL-SPECIFIC PROPERTIES (clearly documented as platform-specific, type-safe)
    mal_rating: Optional[str] = Field(None, description="MAL-specific: Content rating (g, pg, pg_13, r, r+, rx)")
    mal_nsfw: Optional[str] = Field(None, pattern="^(white|gray|black)$", description="MAL-specific: Content filter (white=SFW, gray=questionable, black=NSFW)")
    mal_source: Optional[str] = Field(None, pattern="^(other|original|manga|4_koma_manga|web_manga|digital_manga|novel|light_novel|visual_novel|game|card_game|book|picture_book|radio|music)$", description="MAL-specific: Source material")
    mal_num_list_users: Optional[int] = Field(None, ge=0, description="MAL-specific: Minimum number of users who have anime in their list")
    mal_num_scoring_users: Optional[int] = Field(None, ge=0, description="MAL-specific: Minimum number of users who scored the anime")
    mal_created_at: Optional[str] = Field(None, pattern="^\\d{4}-\\d{2}-\\d{2}$", description="MAL-specific: Creation date filter (YYYY-MM-DD)")
    mal_updated_at: Optional[str] = Field(None, pattern="^\\d{4}-\\d{2}-\\d{2}$", description="MAL-specific: Last update filter (YYYY-MM-DD)")
    mal_broadcast: Optional[str] = Field(None, description="MAL-specific: Broadcast information filter")
    mal_main_picture: Optional[str] = Field(None, description="MAL-specific: Main picture URL filter")
    mal_start_date: Optional[str] = Field(None, pattern="^\\d{4}-\\d{2}-\\d{2}$", description="MAL-specific: Precise start date (YYYY-MM-DD)")
    mal_start_season: Optional[str] = Field(None, pattern="^\\d{4},(winter|spring|summer|fall)$", description="MAL-specific: Start season as 'year,season' (e.g., '2024,winter')")
    
    # CORE RESPONSE FIELDS (Available response fields from all platforms)
    # These indicate what fields each platform can return in responses (not filtering capabilities)
    id_field: Optional[bool] = Field(None, description="Request ID field in response")
    title_field: Optional[bool] = Field(None, description="Request title field in response")
    status_field: Optional[bool] = Field(None, description="Request status field in response")
    format_field: Optional[bool] = Field(None, description="Request format/media type field in response")
    episodes_field: Optional[bool] = Field(None, description="Request episodes count field in response")
    score_field: Optional[bool] = Field(None, description="Request score/rating field in response")
    genres_field: Optional[bool] = Field(None, description="Request genres field in response")
    start_date_field: Optional[bool] = Field(None, description="Request start date field in response")
    end_date_field: Optional[bool] = Field(None, description="Request end date field in response")
    synopsis_field: Optional[bool] = Field(None, description="Request synopsis/description field in response")
    popularity_field: Optional[bool] = Field(None, description="Request popularity field in response")
    rank_field: Optional[bool] = Field(None, description="Request rank field in response")
    source_field: Optional[bool] = Field(None, description="Request source material field in response")
    rating_field: Optional[bool] = Field(None, description="Request content rating field in response")
    studios_field: Optional[bool] = Field(None, description="Request studios field in response")
    
    # MAL-SPECIFIC RESPONSE FIELDS (Available only from MAL API v2)
    mal_alternative_titles_field: Optional[bool] = Field(None, description="MAL-specific: Request alternative titles object")
    mal_my_list_status_field: Optional[bool] = Field(None, description="MAL-specific: Request user's list status (requires auth)")
    mal_num_list_users_field: Optional[bool] = Field(None, description="MAL-specific: Request number of users with anime in list")
    mal_num_scoring_users_field: Optional[bool] = Field(None, description="MAL-specific: Request number of users who scored anime")
    mal_nsfw_field: Optional[bool] = Field(None, description="MAL-specific: Request content safety rating")
    mal_average_episode_duration_field: Optional[bool] = Field(None, description="MAL-specific: Request episode duration in seconds")
    mal_start_season_field: Optional[bool] = Field(None, description="MAL-specific: Request season information object")
    mal_broadcast_field: Optional[bool] = Field(None, description="MAL-specific: Request broadcast information")
    mal_main_picture_field: Optional[bool] = Field(None, description="MAL-specific: Request main picture URLs")
    mal_created_at_field: Optional[bool] = Field(None, description="MAL-specific: Request creation timestamp")
    mal_updated_at_field: Optional[bool] = Field(None, description="MAL-specific: Request last update timestamp")
    
    # ANILIST-SPECIFIC PROPERTIES (All 69+ GraphQL Media query parameters)
    # Basic filters
    anilist_id: Optional[int] = Field(None, ge=1, description="AniList-specific: Filter by AniList media ID")
    anilist_id_mal: Optional[int] = Field(None, ge=1, description="AniList-specific: Filter by MyAnimeList ID")
    anilist_start_date: Optional[int] = Field(None, description="AniList-specific: Start date as FuzzyDateInt (YYYYMMDD)")
    anilist_end_date: Optional[int] = Field(None, description="AniList-specific: End date as FuzzyDateInt (YYYYMMDD)")
    anilist_season: Optional[str] = Field(None, pattern="^(WINTER|SPRING|SUMMER|FALL)$", description="AniList-specific: Season filter")
    anilist_season_year: Optional[int] = Field(None, ge=1900, le=2030, description="AniList-specific: Season year")
    anilist_episodes: Optional[int] = Field(None, ge=0, description="AniList-specific: Exact episode count")
    anilist_duration: Optional[int] = Field(None, ge=0, description="AniList-specific: Exact episode duration (minutes)")
    anilist_chapters: Optional[int] = Field(None, ge=0, description="AniList-specific: Exact chapter count")
    anilist_volumes: Optional[int] = Field(None, ge=0, description="AniList-specific: Exact volume count")
    anilist_is_adult: Optional[bool] = Field(None, description="AniList-specific: Include adult content")
    anilist_format: Optional[str] = Field(None, description="AniList-specific: Single format filter")
    anilist_genre: Optional[str] = Field(None, description="AniList-specific: Single genre filter")
    anilist_tag: Optional[str] = Field(None, description="AniList-specific: Single tag filter")
    anilist_minimum_tag_rank: Optional[int] = Field(None, ge=1, le=100, description="AniList-specific: Minimum tag rank (1-100)")
    anilist_tag_category: Optional[str] = Field(None, description="AniList-specific: Tag category filter")
    anilist_on_list: Optional[bool] = Field(None, description="AniList-specific: Filter by user's list status")
    anilist_licensed_by: Optional[str] = Field(None, description="AniList-specific: Licensing site name")
    anilist_licensed_by_id: Optional[int] = Field(None, ge=1, description="AniList-specific: Licensing site ID")
    anilist_average_score: Optional[int] = Field(None, ge=0, le=100, description="AniList-specific: Exact average score")
    anilist_popularity: Optional[int] = Field(None, ge=0, description="AniList-specific: Exact popularity count")
    anilist_source: Optional[str] = Field(None, pattern="^(ORIGINAL|MANGA|LIGHT_NOVEL|VISUAL_NOVEL|VIDEO_GAME|OTHER|NOVEL|DOUJINSHI|ANIME|WEB_NOVEL|LIVE_ACTION|GAME|BOOK|MULTIMEDIA_PROJECT|PICTURE_BOOK|COMIC)$", description="AniList-specific: Source material type")
    anilist_country_of_origin: Optional[str] = Field(None, pattern="^[A-Z]{2}$", description="AniList-specific: Country code (JP, KR, CN, etc.)")
    anilist_is_licensed: Optional[bool] = Field(None, description="AniList-specific: Official licensing status")
    
    # Negation filters
    anilist_id_not: Optional[int] = Field(None, ge=1, description="AniList-specific: Exclude AniList ID")
    anilist_id_mal_not: Optional[int] = Field(None, ge=1, description="AniList-specific: Exclude MAL ID")
    anilist_format_not: Optional[str] = Field(None, pattern="^(TV|TV_SHORT|MOVIE|SPECIAL|OVA|ONA|MUSIC|MANGA|NOVEL|ONE_SHOT)$", description="AniList-specific: Exclude format")
    anilist_status_not: Optional[str] = Field(None, pattern="^(FINISHED|RELEASING|NOT_YET_RELEASED|CANCELLED|HIATUS)$", description="AniList-specific: Exclude status")
    anilist_average_score_not: Optional[int] = Field(None, ge=0, le=100, description="AniList-specific: Exclude average score")
    anilist_popularity_not: Optional[int] = Field(None, ge=0, description="AniList-specific: Exclude popularity count")
    
    # Array inclusion filters
    anilist_id_in: Optional[List[int]] = Field(None, description="AniList-specific: Include AniList IDs")
    anilist_id_not_in: Optional[List[int]] = Field(None, description="AniList-specific: Exclude AniList IDs")
    anilist_id_mal_in: Optional[List[int]] = Field(None, description="AniList-specific: Include MAL IDs")
    anilist_id_mal_not_in: Optional[List[int]] = Field(None, description="AniList-specific: Exclude MAL IDs")
    anilist_format_in: Optional[List[str]] = Field(None, description="AniList-specific: Include formats")
    anilist_format_not_in: Optional[List[str]] = Field(None, description="AniList-specific: Exclude formats")
    anilist_status_in: Optional[List[str]] = Field(None, description="AniList-specific: Include statuses")
    anilist_status_not_in: Optional[List[str]] = Field(None, description="AniList-specific: Exclude statuses")
    anilist_genre_in: Optional[List[str]] = Field(None, description="AniList-specific: Include genres")
    anilist_genre_not_in: Optional[List[str]] = Field(None, description="AniList-specific: Exclude genres")
    anilist_tag_in: Optional[List[str]] = Field(None, description="AniList-specific: Include tags")
    anilist_tag_not_in: Optional[List[str]] = Field(None, description="AniList-specific: Exclude tags")
    anilist_tag_category_in: Optional[List[str]] = Field(None, description="AniList-specific: Include tag categories")
    anilist_tag_category_not_in: Optional[List[str]] = Field(None, description="AniList-specific: Exclude tag categories")
    anilist_licensed_by_in: Optional[List[str]] = Field(None, description="AniList-specific: Include licensing sites")
    anilist_licensed_by_not_in: Optional[List[str]] = Field(None, description="AniList-specific: Exclude licensing sites")
    anilist_licensed_by_id_in: Optional[List[int]] = Field(None, description="AniList-specific: Include licensing site IDs")
    anilist_source_in: Optional[List[str]] = Field(None, description="AniList-specific: Include source types")
    anilist_source_not_in: Optional[List[str]] = Field(None, description="AniList-specific: Exclude source types")
    anilist_format_range: Optional[List[str]] = Field(None, description="AniList-specific: Format range filter")
    
    # Range filters
    anilist_start_date_greater: Optional[int] = Field(None, description="AniList-specific: Start date greater than (YYYYMMDD)")
    anilist_start_date_lesser: Optional[int] = Field(None, description="AniList-specific: Start date lesser than (YYYYMMDD)")
    anilist_start_date_like: Optional[str] = Field(None, description="AniList-specific: Start date pattern match")
    anilist_end_date_greater: Optional[int] = Field(None, description="AniList-specific: End date greater than (YYYYMMDD)")
    anilist_end_date_lesser: Optional[int] = Field(None, description="AniList-specific: End date lesser than (YYYYMMDD)")
    anilist_end_date_like: Optional[str] = Field(None, description="AniList-specific: End date pattern match")
    anilist_episodes_greater: Optional[int] = Field(None, ge=0, description="AniList-specific: Episodes greater than")
    anilist_episodes_lesser: Optional[int] = Field(None, ge=0, description="AniList-specific: Episodes lesser than")
    anilist_duration_greater: Optional[int] = Field(None, ge=0, description="AniList-specific: Duration greater than (minutes)")
    anilist_duration_lesser: Optional[int] = Field(None, ge=0, description="AniList-specific: Duration lesser than (minutes)")
    anilist_chapters_greater: Optional[int] = Field(None, ge=0, description="AniList-specific: Chapters greater than")
    anilist_chapters_lesser: Optional[int] = Field(None, ge=0, description="AniList-specific: Chapters lesser than")
    anilist_volumes_greater: Optional[int] = Field(None, ge=0, description="AniList-specific: Volumes greater than")
    anilist_volumes_lesser: Optional[int] = Field(None, ge=0, description="AniList-specific: Volumes lesser than")
    anilist_average_score_greater: Optional[int] = Field(None, ge=0, le=100, description="AniList-specific: Average score greater than")
    anilist_average_score_lesser: Optional[int] = Field(None, ge=0, le=100, description="AniList-specific: Average score lesser than")
    anilist_popularity_greater: Optional[int] = Field(None, ge=0, description="AniList-specific: Popularity greater than")
    anilist_popularity_lesser: Optional[int] = Field(None, ge=0, description="AniList-specific: Popularity lesser than")
    
    # Special sorting
    anilist_sort: Optional[List[str]] = Field(None, description="AniList-specific: Sort options (ID, TITLE_ROMAJI, SCORE_DESC, etc.)")
    
    # JIKAN-SPECIFIC PROPERTIES (API v4 verified, only unique features, type-safe)
    jikan_letter: Optional[str] = Field(None, pattern="^[A-Za-z]$", description="Jikan-specific: Alphabetical filter (unique to Jikan)")
    jikan_unapproved: Optional[bool] = Field(None, description="Jikan-specific: Include unapproved entries (unique to Jikan)")
    
    # KITSU-SPECIFIC PROPERTIES
    kitsu_age_rating: Optional[str] = Field(None, description="Kitsu-specific: Age rating filter")
    kitsu_subtype: Optional[str] = Field(None, description="Kitsu-specific: Media subtype filter")
    
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
    
    @field_validator("mal_rating")
    @classmethod
    def validate_mal_rating(cls, v):
        """Validate and convert MAL rating."""
        if v is not None:
            # Convert common formats to MAL format
            rating_mapping = {
                "pg-13": "pg_13",
                "pg13": "pg_13", 
                "r+": "r+",
                "r_plus": "r+"
            }
            converted = rating_mapping.get(v.lower(), v.lower())
            
            # Validate the final value
            valid_ratings = ["g", "pg", "pg_13", "r", "r+", "rx"]
            if converted not in valid_ratings:
                raise ValueError(f"Invalid MAL rating. Must be one of: {valid_ratings}")
            
            return converted
        return v
    
    @field_validator("anilist_format_in", "anilist_format_not_in")
    @classmethod
    def validate_anilist_formats(cls, v):
        """Validate AniList format arrays."""
        if v is not None:
            valid_formats = ["TV", "TV_SHORT", "MOVIE", "SPECIAL", "OVA", "ONA", "MUSIC", "MANGA", "NOVEL", "ONE_SHOT"]
            for format_val in v:
                if format_val not in valid_formats:
                    raise ValueError(f"Invalid AniList format: {format_val}. Must be one of: {valid_formats}")
        return v
    
    @field_validator("anilist_status_in", "anilist_status_not_in")
    @classmethod
    def validate_anilist_statuses(cls, v):
        """Validate AniList status arrays."""
        if v is not None:
            valid_statuses = ["FINISHED", "RELEASING", "NOT_YET_RELEASED", "CANCELLED", "HIATUS"]
            for status_val in v:
                if status_val not in valid_statuses:
                    raise ValueError(f"Invalid AniList status: {status_val}. Must be one of: {valid_statuses}")
        return v
    
    @field_validator("anilist_source_in")
    @classmethod
    def validate_anilist_sources(cls, v):
        """Validate AniList source arrays."""
        if v is not None:
            valid_sources = ["ORIGINAL", "MANGA", "LIGHT_NOVEL", "VISUAL_NOVEL", "VIDEO_GAME", "OTHER", 
                           "NOVEL", "DOUJINSHI", "ANIME", "WEB_NOVEL", "LIVE_ACTION", "GAME", "BOOK", 
                           "MULTIMEDIA_PROJECT", "PICTURE_BOOK", "COMIC"]
            for source_val in v:
                if source_val not in valid_sources:
                    raise ValueError(f"Invalid AniList source: {source_val}. Must be one of: {valid_sources}")
        return v
    
    @field_validator("anilist_sort")
    @classmethod
    def validate_anilist_sort(cls, v):
        """Validate AniList sort arrays."""
        if v is not None:
            valid_sorts = ["ID", "ID_DESC", "TITLE_ROMAJI", "TITLE_ROMAJI_DESC", "TITLE_ENGLISH", "TITLE_ENGLISH_DESC",
                          "TITLE_NATIVE", "TITLE_NATIVE_DESC", "TYPE", "TYPE_DESC", "FORMAT", "FORMAT_DESC",
                          "START_DATE", "START_DATE_DESC", "END_DATE", "END_DATE_DESC", "SCORE", "SCORE_DESC",
                          "POPULARITY", "POPULARITY_DESC", "TRENDING", "TRENDING_DESC", "EPISODES", "EPISODES_DESC",
                          "DURATION", "DURATION_DESC", "STATUS", "STATUS_DESC", "CHAPTERS", "CHAPTERS_DESC",
                          "VOLUMES", "VOLUMES_DESC", "UPDATED_AT", "UPDATED_AT_DESC", "SEARCH_MATCH", 
                          "FAVOURITES", "FAVOURITES_DESC"]
            for sort_val in v:
                if sort_val not in valid_sorts:
                    raise ValueError(f"Invalid AniList sort: {sort_val}. Must be one of: {valid_sorts}")
        return v
    


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