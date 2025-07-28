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
    nicknames: List[str] = Field(default_factory=list, description="Character nicknames from Jikan API")
    
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


class EpisodeThumbnail(BaseModel):
    """Episode thumbnail with source attribution"""
    
    url: str = Field(..., description="Thumbnail image URL")
    source: str = Field(..., description="Source platform (anilist, kitsu, etc.)")
    platform: Optional[str] = Field(None, description="Streaming platform (crunchyroll, funimation, etc.)")


class EpisodeDetailEntry(BaseModel):
    """Comprehensive episode details with multi-source integration"""
    
    # Primary identification
    episode_number: int = Field(..., description="Episode number")
    season_number: Optional[int] = Field(None, description="Season number from Kitsu")
    
    # Episode titles from different sources
    title: str = Field(..., description="Primary episode title")
    title_japanese: Optional[str] = Field(None, description="Japanese episode title")
    title_romaji: Optional[str] = Field(None, description="Romanized episode title")
    
    # Episode content
    synopsis: Optional[str] = Field(None, description="Episode synopsis/description")
    
    # Visual content and streaming
    thumbnails: List[EpisodeThumbnail] = Field(default_factory=list, description="Episode thumbnails from all sources")
    streaming: Dict[str, str] = Field(default_factory=dict, description="Streaming platforms and URLs {platform: url}")
    
    # Technical metadata
    aired: Optional[str] = Field(None, description="Episode air date with timezone")
    duration: Optional[int] = Field(None, description="Episode duration in seconds")
    score: Optional[float] = Field(None, description="Episode rating score")
    
    # Episode flags
    filler: bool = Field(default=False, description="Whether episode is filler")
    recap: bool = Field(default=False, description="Whether episode is recap")
    
    # Source attribution
    url: Optional[str] = Field(None, description="Episode page URL (typically MAL)")


class TrailerEntry(BaseModel):
    """Trailer information from external APIs"""
    
    youtube_url: Optional[str] = Field(None, description="YouTube video URL")
    title: Optional[str] = Field(None, description="Trailer title")
    thumbnail_url: Optional[str] = Field(None, description="Trailer thumbnail URL")


class EnrichmentMetadata(BaseModel):
    """Metadata about data enrichment process"""
    
    source: str = Field(..., description="Source of enrichment (mal, anilist, multi-source, etc.)")
    enriched_at: datetime = Field(..., description="When enrichment was performed")
    success: bool = Field(default=True, description="Whether enrichment was successful")
    error_message: Optional[str] = Field(None, description="Error message if enrichment failed")


class ImageEntry(BaseModel):
    """Image entry with source attribution"""
    url: str = Field(..., description="Image URL")
    source: str = Field(..., description="Source platform")
    type: Optional[str] = Field(None, description="Image type (cover, banner, promotional) - optional when organized by type arrays")


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


class StaffMember(BaseModel):
    """Individual staff member with multi-source integration"""
    staff_ids: Dict[str, str] = Field(default_factory=dict, description="Staff IDs across platforms (anidb, anilist)")
    name: str = Field(..., description="Staff member name")
    native_name: Optional[str] = Field(None, description="Native language name")
    role: str = Field(..., description="Primary role")
    image: Optional[str] = Field(None, description="Staff member image URL")
    biography: Optional[str] = Field(None, description="Staff member biography")
    birth_date: Optional[str] = Field(None, description="Birth date")
    hometown: Optional[str] = Field(None, description="Hometown")
    primary_occupations: List[str] = Field(default_factory=list, description="Primary occupations")
    years_active: List[int] = Field(default_factory=list, description="Years active")
    gender: Optional[str] = Field(None, description="Gender")
    blood_type: Optional[str] = Field(None, description="Blood type")
    community_favorites: Optional[int] = Field(None, description="Community favorites count")
    enhancement_status: Optional[str] = Field(None, description="Enhancement status from AniList matching")


class VoiceActor(BaseModel):
    """Voice actor with character assignments"""
    staff_ids: Dict[str, str] = Field(default_factory=dict, description="Staff IDs across platforms")
    name: str = Field(..., description="Voice actor name")
    native_name: Optional[str] = Field(None, description="Native language name")
    character_assignments: List[str] = Field(default_factory=list, description="Characters voiced")
    image: Optional[str] = Field(None, description="Voice actor image URL")
    biography: Optional[str] = Field(None, description="Voice actor biography")
    birth_date: Optional[str] = Field(None, description="Birth date")
    blood_type: Optional[str] = Field(None, description="Blood type")


class CompanyEntry(BaseModel):
    """Studio/Producer/Licensor company entry"""
    name: str = Field(..., description="Company name")
    type: str = Field(..., description="Company type (animation_studio, producer, licensor)")
    url: Optional[str] = Field(None, description="Company URL")


class ProductionStaff(BaseModel):
    """Production staff organized by role"""
    directors: List[StaffMember] = Field(default_factory=list, description="Directors")
    music_composers: List[StaffMember] = Field(default_factory=list, description="Music composers")
    character_designers: List[StaffMember] = Field(default_factory=list, description="Character designers")
    series_writers: List[StaffMember] = Field(default_factory=list, description="Series writers")
    animation_directors: List[StaffMember] = Field(default_factory=list, description="Animation directors")
    original_creators: List[StaffMember] = Field(default_factory=list, description="Original creators")


class VoiceActors(BaseModel):
    """Voice actors organized by language"""
    japanese: List[VoiceActor] = Field(default_factory=list, description="Japanese voice actors")


class StaffData(BaseModel):
    """Comprehensive staff data structure"""
    production_staff: ProductionStaff = Field(default_factory=ProductionStaff, description="Production staff by role")
    studios: List[CompanyEntry] = Field(default_factory=list, description="Animation studios")
    producers: List[CompanyEntry] = Field(default_factory=list, description="Producers")
    licensors: List[CompanyEntry] = Field(default_factory=list, description="Licensors")
    voice_actors: VoiceActors = Field(default_factory=VoiceActors, description="Voice actors by language")




class ContextualRank(BaseModel):
    """Contextual ranking information from platforms like AniList"""
    rank: int = Field(..., description="Rank position")
    type: str = Field(..., description="Ranking type (POPULAR, RATED, etc.)")
    format: Optional[str] = Field(None, description="Format context (TV, Movie, etc.)")
    year: Optional[int] = Field(None, description="Year context")
    season: Optional[str] = Field(None, description="Season context (SPRING, SUMMER, FALL, WINTER)")
    all_time: Optional[bool] = Field(None, description="Whether this is an all-time ranking")


class StatisticsEntry(BaseModel):
    """Standardized statistics entry - AI maps all platforms to these uniform properties"""
    score: Optional[float] = Field(None, description="Rating score (normalized to 0-10 scale)")
    scored_by: Optional[int] = Field(None, description="Number of users who rated")
    rank: Optional[int] = Field(None, description="Overall ranking position")
    popularity_rank: Optional[int] = Field(None, description="Popularity ranking position")
    members: Optional[int] = Field(None, description="Total members/users tracking")
    favorites: Optional[int] = Field(None, description="Number of users who favorited")
    contextual_ranks: Optional[List[ContextualRank]] = Field(None, description="Contextual ranking achievements (e.g., 'Best of 2021')")


class AnimeEntry(BaseModel):
    """Anime entry from anime-offline-database with comprehensive enhancement support"""

    # =====================================================================
    # SCALAR FIELDS (alphabetical)
    # =====================================================================
    background: Optional[str] = Field(None, description="Background information from MAL")
    episodes: int = Field(default=0, description="Number of episodes")
    month: Optional[str] = Field(None, description="Premiere month from AnimSchedule")
    nsfw: Optional[bool] = Field(None, description="Not Safe For Work flag from Kitsu")
    picture: Optional[str] = Field(None, description="Cover image URL")
    rating: Optional[str] = Field(None, description="Content rating (PG-13, R, etc.)")
    source_material: Optional[str] = Field(None, description="Source material (manga, light novel, etc.)")
    status: str = Field(..., description="Airing status")
    synopsis: Optional[str] = Field(None, description="Detailed anime synopsis from external sources")
    thumbnail: Optional[str] = Field(None, description="Thumbnail URL")
    title: str = Field(..., description="Primary anime title")
    title_english: Optional[str] = Field(None, description="English title")
    title_japanese: Optional[str] = Field(None, description="Japanese title")
    type: str = Field(..., description="TV, Movie, OVA, etc.")

    # =====================================================================
    # ARRAY FIELDS (alphabetical)
    # =====================================================================
    awards: List[Dict[str, Any]] = Field(default_factory=list, description="Awards and recognition")
    characters: List[CharacterEntry] = Field(default_factory=list, description="Character information with multi-source support")
    content_warnings: List[str] = Field(default_factory=list, description="Content warnings")
    demographics: List[str] = Field(default_factory=list, description="Target demographics (Shounen, Seinen, etc.)")
    ending_themes: List[Dict[str, Any]] = Field(default_factory=list, description="Ending theme songs")
    episode_details: List[EpisodeDetailEntry] = Field(default_factory=list, description="Detailed episode information with multi-source integration")
    genres: List[str] = Field(default_factory=list, description="Anime genres from AniList/other sources")
    licensors: List[str] = Field(default_factory=list, description="Licensing companies")
    opening_themes: List[Dict[str, Any]] = Field(default_factory=list, description="Opening theme songs")
    related_anime: List[RelatedAnimeEntry] = Field(default_factory=list, description="Related anime entries from URL processing")
    relations: List[RelationEntry] = Field(default_factory=list, description="Related anime with platform URLs")
    sources: List[str] = Field(..., description="Source URLs from various providers")
    streaming_info: List[StreamingEntry] = Field(default_factory=list, description="Streaming platform information")
    streaming_licenses: List[str] = Field(default_factory=list, description="Streaming licenses")
    synonyms: List[str] = Field(default_factory=list, description="Alternative titles")
    tags: List[str] = Field(default_factory=list, description="Original tags from offline database")
    themes: List[ThemeEntry] = Field(default_factory=list, description="Thematic elements with descriptions")
    trailers: List[TrailerEntry] = Field(default_factory=list, description="Trailer information from external APIs")

    # =====================================================================
    # OBJECT/DICT FIELDS (alphabetical)
    # =====================================================================
    aired_dates: Optional[Dict[str, Any]] = Field(None, description="Detailed airing dates")
    anime_season: Optional[Dict[str, Any]] = Field(None, description="Season and year")
    broadcast: Optional[Dict[str, Any]] = Field(None, description="Broadcast schedule information")
    broadcast_schedule: Optional[Dict[str, Any]] = Field(None, description="Broadcast timing for different versions (jpn_time, sub_time, dub_time)")
    delay_information: Optional[Dict[str, Any]] = Field(None, description="Current delay status and reasons")
    duration: Optional[Union[int, Dict[str, Any]]] = Field(None, description="Episode duration in seconds")
    enhanced_metadata: Optional[Dict[str, Any]] = Field(None, description="Enhanced enrichment metadata")
    enrichment_metadata: Optional[EnrichmentMetadata] = Field(None, description="Metadata about enrichment process")
    episode_overrides: Optional[Dict[str, Any]] = Field(None, description="Episode override information for different versions (main_override, sub_override, dub_override)")
    external_links: Dict[str, str] = Field(default_factory=dict, description="External links (official site, social media)")
    images: Dict[str, List[ImageEntry]] = Field(default_factory=dict, description="Images from multiple sources")
    popularity_trends: Optional[Dict[str, Any]] = Field(None, description="Popularity trend data")
    premiere_dates: Optional[Dict[str, Any]] = Field(None, description="Premiere dates for different versions (original, sub, dub)")
    score: Optional[Dict[str, float]] = Field(None, description="Anime scoring data with arithmeticGeometricMean, arithmeticMean, median")
    staff_data: Optional[StaffData] = Field(None, description="Comprehensive staff data with multi-source integration")
    statistics: Dict[str, StatisticsEntry] = Field(default_factory=dict, description="Standardized statistics from different platforms (mal, anilist, kitsu, animeschedule)")
    
    def has_enrichment_data(self) -> bool:
        """Check if this entry has any enrichment data"""
        return (
            self.synopsis is not None or 
            len(self.characters) > 0 or 
            len(self.trailers) > 0 or
            len(self.episode_details) > 0 or
            len(self.genres) > 0 or
            len(self.themes) > 0 or
            len(self.streaming_info) > 0 or
            self.staff_data is not None or
            len(self.relations) > 0 or
            len(self.related_anime) > 0
        )
    
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
            "staff_count": self._get_staff_data_count(),
            "relation_count": len(self.relations),
            "related_anime_count": len(self.related_anime),
            "award_count": len(self.awards),
            
            # Data quality indicators
            "has_synopsis": self.synopsis is not None,
            "has_detailed_timing": self.aired_dates is not None and self.broadcast is not None,
            "has_broadcast_schedule": self.broadcast_schedule is not None,
            "has_premiere_dates": self.premiere_dates is not None,
            "has_delay_information": self.delay_information is not None,
            "has_episode_overrides": self.episode_overrides is not None,
            "has_streaming_info": len(self.streaming_info) > 0,
            "has_staff_info": self.staff_data is not None,
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
        total_possible = 24.0  # Total possible enhancement areas
        
        # Basic enrichment (5 points)
        if self.synopsis: score += 1.0
        if len(self.characters) > 0: score += 1.0
        if len(self.trailers) > 0: score += 1.0
        if len(self.episode_details) > 0: score += 1.0
        if len(self.genres) > 0: score += 1.0
        
        # Detailed metadata (9 points)
        if len(self.themes) > 0: score += 1.0
        if self.source_material: score += 1.0
        if self.rating: score += 1.0
        if self.aired_dates and self.broadcast: score += 1.0
        if len(self.demographics) > 0: score += 1.0
        if len(self.content_warnings) > 0: score += 1.0
        if self.broadcast_schedule: score += 1.0
        if self.premiere_dates: score += 1.0
        if self.episode_overrides: score += 1.0
        
        # Rich content (5 points)
        if len(self.streaming_info) > 0: score += 1.0
        if self.staff_data is not None: score += 1.0
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
    
    def _get_staff_data_count(self) -> int:
        """Get total count of staff members in comprehensive staff data"""
        if not self.staff_data:
            return 0
        
        count = 0
        # Production staff
        production = self.staff_data.production_staff
        count += len(production.directors)
        count += len(production.music_composers)
        count += len(production.character_designers)
        count += len(production.series_writers)
        count += len(production.animation_directors)
        count += len(production.original_creators)
        
        # Voice actors
        count += len(self.staff_data.voice_actors.japanese)
        
        # Companies
        count += len(self.staff_data.studios)
        count += len(self.staff_data.producers)
        count += len(self.staff_data.licensors)
        
        return count

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
