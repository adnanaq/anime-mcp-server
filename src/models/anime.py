# src/models/anime.py - Pydantic Models for Anime Data
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator


class CharacterEntry(BaseModel):
    """Character information from external APIs with multi-source support"""
    
    # Primary identification
    name: str = Field(..., description="Character name")
    role: str = Field(..., description="Character role (Main, Supporting, etc.)")
    
    # Name variations from different sources
    name_variations: List[str] = Field(default_factory=list, description="All name spellings and variations")
    name_kanji: Optional[str] = Field(None, description="Character name in Kanji/Japanese")
    name_native: Optional[str] = Field(None, description="Native language name")
    
    # Platform IDs (replaces sources/is_merged metadata)
    character_ids: Dict[str, int] = Field(default_factory=dict, description="Character IDs across platforms (mal, anilist, etc.)")
    
    # Images from different sources
    images: Dict[str, str] = Field(default_factory=dict, description="Character images from different sources")
    
    # Character details (prefer most detailed source)
    description: Optional[str] = Field(None, description="Character description/biography")
    age: Optional[str] = Field(None, description="Character age")
    gender: Optional[str] = Field(None, description="Character gender")
    
    # Voice actors from all sources
    voice_actors: List[Dict[str, Any]] = Field(default_factory=list, description="Voice actor information")


class TrailerEntry(BaseModel):
    """Trailer information from external APIs"""
    
    youtube_url: Optional[str] = Field(None, description="YouTube video URL")
    title: Optional[str] = Field(None, description="Trailer title")
    thumbnail_url: Optional[str] = Field(None, description="Trailer thumbnail URL")


class EnrichmentMetadata(BaseModel):
    """Metadata about data enrichment process"""
    
    source: str = Field(..., description="Source of enrichment (mal, anilist, multi-source, etc.)")
    enriched_at: datetime = Field(..., description="When enrichment was performed")
    character_count: int = Field(default=0, description="Number of characters enriched")
    success: bool = Field(default=True, description="Whether enrichment was successful")
    error_message: Optional[str] = Field(None, description="Error message if enrichment failed")


class ImageEntry(BaseModel):
    """Image entry with source attribution"""
    url: str = Field(..., description="Image URL")
    source: str = Field(..., description="Source platform")
    type: str = Field(..., description="Image type (cover, banner, promotional)")


class RelationEntry(BaseModel):
    """Related anime entry with multi-platform URLs"""
    anime_id: str = Field(..., description="Related anime ID")
    relation_type: str = Field(..., description="Relation type (sequel, prequel, etc.)")
    title: Optional[str] = Field(None, description="Related anime title")
    title_english: Optional[str] = Field(None, description="Related anime English title")
    urls: Dict[str, str] = Field(default_factory=dict, description="URLs from different platforms")


class RelatedAnimeEntry(BaseModel):
    """Related anime entry from URL processing"""
    relation_type: str = Field(..., description="Relation type (Sequel, Prequel, Other, etc.)")
    title: str = Field(..., description="Related anime title extracted from URL")
    url: str = Field(..., description="Original URL")


class StreamingEntry(BaseModel):
    """Streaming platform entry"""
    platform: str = Field(..., description="Streaming platform name")
    url: str = Field(..., description="Streaming URL")
    region: Optional[str] = Field(None, description="Available regions")
    free: Optional[bool] = Field(None, description="Free to watch")
    premium_required: Optional[bool] = Field(None, description="Premium subscription required")
    dub_available: Optional[bool] = Field(None, description="Dub available")
    subtitle_languages: List[str] = Field(default_factory=list, description="Available subtitle languages")


class ThemeEntry(BaseModel):
    """Theme entry with description"""
    name: str = Field(..., description="Theme name")
    description: Optional[str] = Field(None, description="Theme description")


class StaffEntry(BaseModel):
    """Staff member entry"""
    name: str = Field(..., description="Staff member name")
    role: str = Field(..., description="Primary role")
    positions: List[str] = Field(default_factory=list, description="All positions held")


class StatisticsEntry(BaseModel):
    """Standardized statistics entry - AI maps all platforms to these uniform properties"""
    score: Optional[float] = Field(None, description="Rating score (normalized to 0-10 scale)")
    scored_by: Optional[int] = Field(None, description="Number of users who rated")
    rank: Optional[int] = Field(None, description="Overall ranking position")
    popularity_rank: Optional[int] = Field(None, description="Popularity ranking position")
    members: Optional[int] = Field(None, description="Total members/users tracking")
    favorites: Optional[int] = Field(None, description="Number of users who favorited")


class AnimeEntry(BaseModel):
    """Anime entry from anime-offline-database with comprehensive enhancement support"""

    # =====================================================================
    # EXISTING ANIME-OFFLINE-DATABASE FIELDS (Keep Unchanged)
    # =====================================================================
    sources: List[str] = Field(..., description="Source URLs from various providers")
    title: str = Field(..., description="Primary anime title")
    type: str = Field(..., description="TV, Movie, OVA, etc.")
    episodes: int = Field(default=0, description="Number of episodes")
    status: str = Field(..., description="Airing status")
    animeSeason: Optional[Dict[str, Any]] = Field(None, description="Season and year")
    picture: Optional[str] = Field(None, description="Cover image URL")
    thumbnail: Optional[str] = Field(None, description="Thumbnail URL")
    duration: Optional[Union[int, Dict[str, Any]]] = Field(
        None, description="Episode duration in seconds"
    )
    score: Optional[Dict[str, float]] = Field(
        None,
        description="Anime scoring data with arithmeticGeometricMean, arithmeticMean, median",
    )
    synonyms: List[str] = Field(default_factory=list, description="Alternative titles")
    tags: List[str] = Field(default_factory=list, description="Original tags from offline database")
    studios: List[str] = Field(default_factory=list, description="Animation studios")
    producers: List[str] = Field(default_factory=list, description="Production companies")

    # =====================================================================
    # CURRENT ENRICHMENT FIELDS (Already Enhanced)
    # =====================================================================
    synopsis: Optional[str] = Field(None, description="Detailed anime synopsis from external sources")
    characters: List[CharacterEntry] = Field(default_factory=list, description="Character information with multi-source support")
    trailers: List[TrailerEntry] = Field(default_factory=list, description="Trailer information from external APIs")
    platform_ids: Optional[Dict[str, Optional[int]]] = Field(None, description="Platform IDs extracted from sources")
    enrichment_metadata: Optional[EnrichmentMetadata] = Field(None, description="Metadata about enrichment process")

    # =====================================================================
    # NEW ENHANCEMENT FIELDS (Add to existing structure)
    # =====================================================================
    genres: List[str] = Field(default_factory=list, description="Anime genres from AniList/other sources")
    demographics: List[str] = Field(default_factory=list, description="Target demographics (Shounen, Seinen, etc.)")
    themes: List[ThemeEntry] = Field(default_factory=list, description="Thematic elements with descriptions")
    source_material: Optional[str] = Field(None, description="Source material (manga, light novel, etc.)")
    rating: Optional[str] = Field(None, description="Content rating (PG-13, R, etc.)")
    content_warnings: List[str] = Field(default_factory=list, description="Content warnings")
    
    # Title variations from different sources
    title_japanese: Optional[str] = Field(None, description="Japanese title")
    title_english: Optional[str] = Field(None, description="English title")
    
    # Detailed timing information
    aired_dates: Optional[Dict[str, Any]] = Field(None, description="Detailed airing dates")
    broadcast: Optional[Dict[str, Any]] = Field(None, description="Broadcast schedule information")
    month: Optional[str] = Field(None, description="Premiere month from AnimSchedule")
    background: Optional[str] = Field(None, description="Background information from MAL")
    
    # Streaming and availability
    streaming_info: List[StreamingEntry] = Field(default_factory=list, description="Streaming platform information")
    licensors: List[str] = Field(default_factory=list, description="Licensing companies")
    streaming_licenses: List[str] = Field(default_factory=list, description="Streaming licenses")
    
    # Staff and music
    staff: List[StaffEntry] = Field(default_factory=list, description="Staff information")
    opening_themes: List[Dict[str, Any]] = Field(default_factory=list, description="Opening theme songs")
    ending_themes: List[Dict[str, Any]] = Field(default_factory=list, description="Ending theme songs")
    
    # Statistics from multiple platforms with standardized schema
    statistics: Dict[str, StatisticsEntry] = Field(default_factory=dict, description="Standardized statistics from different platforms (mal, anilist, kitsu, animeschedule)")
    
    # External links
    external_links: Dict[str, str] = Field(default_factory=dict, description="External links (official site, social media)")
    
    # Enhanced images with source attribution
    images: Dict[str, List[ImageEntry]] = Field(default_factory=dict, description="Images from multiple sources")
    
    # Episode details
    episode_details: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed episode information")
    
    # Relations with multi-platform URLs
    relations: List[RelationEntry] = Field(default_factory=list, description="Related anime with platform URLs")
    
    # Related anime from URL processing (different from relations)
    relatedAnime: List[RelatedAnimeEntry] = Field(default_factory=list, description="Related anime entries from URL processing")
    
    # Awards and recognition
    awards: List[Dict[str, Any]] = Field(default_factory=list, description="Awards and recognition")
    
    # Popularity trends
    popularity_trends: Optional[Dict[str, Any]] = Field(None, description="Popularity trend data")
    
    # Enhanced metadata
    enhanced_metadata: Optional[Dict[str, Any]] = Field(None, description="Enhanced enrichment metadata")
    
    def has_enrichment_data(self) -> bool:
        """Check if this entry has any enrichment data"""
        return (
            self.synopsis is not None or 
            len(self.characters) > 0 or 
            len(self.trailers) > 0 or
            len(self.genres) > 0 or
            len(self.themes) > 0 or
            len(self.streaming_info) > 0 or
            len(self.staff) > 0 or
            len(self.relations) > 0 or
            len(self.relatedAnime) > 0
        )
    
    def is_enrichable(self) -> bool:
        """Check if this entry can be enriched (has MAL or AniList sources)"""
        if not self.platform_ids:
            return any('myanimelist.net' in source or 'anilist.co' in source for source in self.sources)
        return self.platform_ids.get('mal_id') is not None or self.platform_ids.get('anilist_id') is not None
    
    def get_primary_enrichment_id(self) -> Optional[Tuple[str, int]]:
        """Get the primary ID for enrichment (prefer MAL, fallback to AniList)"""
        if self.platform_ids:
            if self.platform_ids.get('mal_id'):
                return ('mal', self.platform_ids['mal_id'])
            elif self.platform_ids.get('anilist_id'):
                return ('anilist', self.platform_ids['anilist_id'])
        return None
    
    def should_update_enrichment(self, max_age_days: int = 30) -> bool:
        """Check if enrichment data should be updated"""
        if not self.enrichment_metadata:
            return True
        
        # Check age
        age_days = (datetime.utcnow() - self.enrichment_metadata.enriched_at).days
        if age_days > max_age_days:
            return True
        
        # Check if previous enrichment failed
        if not self.enrichment_metadata.success:
            return True
        
        # Check if no data was actually enriched
        if not self.has_enrichment_data():
            return True
        
        return False
    
    def get_enrichment_sources(self) -> List[str]:
        """Get list of sources that contributed to enrichment"""
        sources = set()
        
        # Check metadata source
        if self.enrichment_metadata and self.enrichment_metadata.source:
            sources.add(self.enrichment_metadata.source)
        
        # Check character sources (from character_ids platforms)
        for character in self.characters:
            sources.update(character.character_ids.keys())
        
        # Check streaming info sources
        for stream in self.streaming_info:
            if hasattr(stream, 'source'):
                sources.add(stream.source)
        
        # Check image sources
        for image_list in self.images.values():
            for image in image_list:
                sources.add(image.source)
        
        
        return list(sources)
    
    def get_character_stats(self) -> Dict[str, Any]:
        """Get statistics about character data"""
        total_chars = len(self.characters)
        merged_chars = sum(1 for char in self.characters if len(char.character_ids) > 1)
        chars_with_description = sum(1 for char in self.characters if char.description)
        chars_with_voice_actors = sum(1 for char in self.characters if char.voice_actors)
        main_chars = sum(1 for char in self.characters if char.role.lower() in ['main', 'protagonist'])
        
        return {
            "total_characters": total_chars,
            "merged_characters": merged_chars,
            "characters_with_description": chars_with_description,
            "characters_with_voice_actors": chars_with_voice_actors,
            "main_characters": main_chars,
            "merge_rate": merged_chars / total_chars if total_chars > 0 else 0.0,
        }
    
    def has_multi_source_enrichment(self) -> bool:
        """Check if this anime has data from multiple enrichment sources"""
        sources = self.get_enrichment_sources()
        return len(sources) > 1 or any(len(char.character_ids) > 1 for char in self.characters)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all enhanced data"""
        return {
            # Basic stats
            "has_enrichment": self.has_enrichment_data(),
            "enrichment_sources": self.get_enrichment_sources(),
            "multi_source_enriched": self.has_multi_source_enrichment(),
            
            # Content counts
            "character_count": len(self.characters),
            "trailer_count": len(self.trailers),
            "genre_count": len(self.genres),
            "theme_count": len(self.themes),
            "streaming_platform_count": len(self.streaming_info),
            "staff_count": len(self.staff),
            "relation_count": len(self.relations),
            "related_anime_count": len(self.relatedAnime),
            "award_count": len(self.awards),
            
            # Data quality indicators
            "has_synopsis": self.synopsis is not None,
            "has_detailed_timing": self.aired_dates is not None and self.broadcast is not None,
            "has_streaming_info": len(self.streaming_info) > 0,
            "has_staff_info": len(self.staff) > 0,
            "has_theme_info": len(self.opening_themes) > 0 or len(self.ending_themes) > 0,
            "has_episode_details": len(self.episode_details) > 0,
            "has_external_links": len(self.external_links) > 0,
            "has_multi_platform_images": sum(len(imgs) for imgs in self.images.values()) > 1,
            "has_content_warnings": len(self.content_warnings) > 0,
            
            # Platform coverage
            "platform_statistics_count": len(self.statistics),
            "relation_platform_coverage": sum(len(rel.urls) for rel in self.relations),
            "image_source_count": len(set(img.source for imgs in self.images.values() for img in imgs)),
            
            # Completeness score (0-1)
            "completeness_score": self._calculate_completeness_score()
        }
    
    def _calculate_completeness_score(self) -> float:
        """Calculate overall data completeness score (0-1)"""
        score = 0.0
        total_possible = 20.0  # Total possible enhancement areas
        
        # Basic enrichment (4 points)
        if self.synopsis: score += 1.0
        if len(self.characters) > 0: score += 1.0
        if len(self.trailers) > 0: score += 1.0
        if len(self.genres) > 0: score += 1.0
        
        # Detailed metadata (6 points)
        if len(self.themes) > 0: score += 1.0
        if self.source_material: score += 1.0
        if self.rating: score += 1.0
        if self.aired_dates and self.broadcast: score += 1.0
        if len(self.demographics) > 0: score += 1.0
        if len(self.content_warnings) > 0: score += 1.0
        
        # Rich content (5 points)
        if len(self.streaming_info) > 0: score += 1.0
        if len(self.staff) > 0: score += 1.0
        if len(self.opening_themes) > 0 or len(self.ending_themes) > 0: score += 1.0
        if len(self.episode_details) > 0: score += 1.0
        if len(self.relations) > 0: score += 1.0
        
        # External integration (3 points)
        if len(self.external_links) > 0: score += 1.0
        if len(self.statistics) > 1: score += 1.0  # Multiple platform stats
        if sum(len(imgs) for imgs in self.images.values()) > 2: score += 1.0  # Multiple images
        
        # Quality indicators (2 points)
        if len(self.awards) > 0: score += 1.0
        if self.popularity_trends: score += 1.0
        
        return min(score / total_possible, 1.0)

    @field_validator("duration")
    @classmethod
    def validate_duration(cls, v):
        """Convert duration dict to int seconds"""
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, dict) and "value" in v:
            return v["value"]
        return None



class SearchRequest(BaseModel):
    """Search request model"""

    query: str = Field(..., description="Search query")
    limit: int = Field(default=20, ge=1, le=100, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")


class UnifiedSearchRequest(BaseModel):
    """Unified search request model for all search types"""
    
    # Text search fields
    query: Optional[str] = Field(None, description="Search query for text search")
    
    # ID-based search fields  
    anime_id: Optional[str] = Field(None, description="Anime ID for similarity search")
    
    # Image search fields
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    
    # Visual similarity flag
    visual_similarity: Optional[bool] = Field(False, description="Enable visual similarity search")
    
    # Multimodal search weight
    text_weight: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Text vs image weight for multimodal")
    
    # Common fields
    limit: int = Field(default=20, ge=1, le=100, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")


class SearchResult(BaseModel):
    """Search result model"""

    anime_id: str
    title: str
    synopsis: Optional[str] = None
    type: str
    episodes: int
    tags: List[str]
    studios: List[str]
    picture: Optional[str] = None
    relevance_score: float = Field(..., description="Search relevance score (0-1)")
    anime_score: Optional[float] = Field(None, description="Anime rating score (1-10)")
    year: Optional[int] = None
    season: Optional[str] = None

    # Platform IDs for cross-referencing
    myanimelist_id: Optional[int] = None
    anilist_id: Optional[int] = None
    kitsu_id: Optional[int] = None
    anidb_id: Optional[int] = None
    anisearch_id: Optional[int] = None
    simkl_id: Optional[int] = None
    livechart_id: Optional[int] = None
    animenewsnetwork_id: Optional[int] = None
    animeplanet_id: Optional[str] = None
    notify_id: Optional[str] = None
    animecountdown_id: Optional[int] = None


class SearchResponse(BaseModel):
    """Search response model"""

    query: str
    results: List[SearchResult]
    total_results: int
    processing_time_ms: float


class DatabaseStats(BaseModel):
    """Database statistics"""

    total_anime: int
    indexed_anime: int
    last_updated: datetime
    index_health: str
    average_quality_score: float
