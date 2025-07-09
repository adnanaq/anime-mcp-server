"""
Pydantic schemas for MCP anime server tools.

Consolidated input/output validation following modern MCP patterns.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class SearchAnimeInput(BaseModel):
    """Input schema for anime search with comprehensive validation."""

    query: str = Field(
        description="Natural language search query (e.g., 'romantic comedy school anime')",
        min_length=1,
        max_length=500,
    )
    limit: int = Field(
        default=10, ge=1, le=50, description="Maximum number of results to return"
    )
    genres: Optional[List[str]] = Field(
        None,
        description="List of anime genres to filter by (e.g., ['Action', 'Comedy'])",
    )
    year_range: Optional[List[int]] = Field(
        None,
        description="Year range as [start_year, end_year] (e.g., [2020, 2023])",
        min_length=2,
        max_length=2,
    )
    anime_types: Optional[List[str]] = Field(
        None, description="List of anime types (e.g., ['TV', 'Movie', 'OVA'])"
    )
    studios: Optional[List[str]] = Field(
        None, description="List of animation studios (e.g., ['Mappa', 'Studio Ghibli'])"
    )
    exclusions: Optional[List[str]] = Field(
        None, description="List of genres/themes to exclude (e.g., ['Horror', 'Ecchi'])"
    )
    mood_keywords: Optional[List[str]] = Field(
        None,
        description="List of mood descriptors (e.g., ['dark', 'serious', 'funny'])",
    )


class AnimeDetailsInput(BaseModel):
    """Input schema for getting anime details."""

    anime_id: str = Field(description="Unique anime identifier", min_length=1)


class SimilarAnimeInput(BaseModel):
    """Input schema for finding similar anime."""

    anime_id: str = Field(
        description="Reference anime ID to find similar anime for", min_length=1
    )
    limit: int = Field(
        default=10, ge=1, le=20, description="Maximum number of similar anime to return"
    )


class ImageSearchInput(BaseModel):
    """Input schema for image-based anime search."""

    image_data: str = Field(
        description="Base64 encoded image data (supports JPG, PNG, WebP formats)",
        min_length=1,
    )
    limit: int = Field(
        default=10, ge=1, le=30, description="Maximum number of results to return"
    )


class MultimodalSearchInput(BaseModel):
    """Input schema for multimodal anime search."""

    query: str = Field(
        description="Text search query (e.g., 'mecha robots fighting')",
        min_length=1,
        max_length=500,
    )
    image_data: Optional[str] = Field(
        None, description="Optional base64 encoded image for visual similarity"
    )
    limit: int = Field(
        default=10, ge=1, le=25, description="Maximum number of results to return"
    )
    text_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for text similarity (0.0-1.0), default 0.7 means 70% text, 30% image",
    )


class VisualSimilarInput(BaseModel):
    """Input schema for visual similarity search."""

    anime_id: str = Field(
        description="Reference anime ID to find visually similar anime for",
        min_length=1,
    )
    limit: int = Field(
        default=10, ge=1, le=20, description="Maximum number of similar anime to return"
    )
