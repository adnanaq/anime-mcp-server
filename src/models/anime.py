# src/models/anime.py - Pydantic Models for Anime Data
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class AnimeEntry(BaseModel):
    """Anime entry from anime-offline-database"""

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
        None, description="Anime scoring data with arithmeticGeometricMean, arithmeticMean, median"
    )

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

    synonyms: List[str] = Field(default_factory=list, description="Alternative titles")
    relatedAnime: List[str] = Field(
        default_factory=list, description="Related anime URLs"
    )
    tags: List[str] = Field(default_factory=list, description="Genre and theme tags")
    studios: List[str] = Field(default_factory=list, description="Animation studios")
    producers: List[str] = Field(
        default_factory=list, description="Production companies"
    )
    synopsis: Optional[str] = Field(None, description="Anime synopsis/description")


class SearchRequest(BaseModel):
    """Search request model"""

    query: str = Field(..., description="Search query")
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
