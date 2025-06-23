"""
Centralized configuration management for Anime MCP Server.

This module provides type-safe configuration handling with validation,
eliminating hardcoded values and DRY violations across the codebase.
"""

from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation and type safety."""

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    debug: bool = Field(default=True, description="Enable debug mode")

    # Qdrant Vector Database Configuration
    qdrant_url: str = Field(
        default="http://localhost:6333", description="Qdrant server URL"
    )
    qdrant_collection_name: str = Field(
        default="anime_database", description="Qdrant collection name"
    )
    qdrant_vector_size: int = Field(
        default=384, description="Vector embedding dimensions"
    )
    qdrant_distance_metric: str = Field(
        default="cosine", description="Distance metric for similarity"
    )

    # FastEmbed Configuration
    fastembed_model: str = Field(
        default="BAAI/bge-small-en-v1.5", description="FastEmbed model for embeddings"
    )
    fastembed_cache_dir: Optional[str] = Field(
        default=None, description="FastEmbed model cache directory"
    )

    # Multi-Vector Configuration (Phase 4)
    enable_multi_vector: bool = Field(
        default=False, description="Enable multi-vector support (text + image)"
    )
    image_vector_size: int = Field(
        default=512, description="Image embedding dimensions (CLIP)"
    )
    clip_model: str = Field(
        default="ViT-B/32", description="CLIP model for image embeddings"
    )

    # Data Processing Configuration
    batch_size: int = Field(
        default=1000, ge=1, description="Batch size for document processing"
    )
    max_concurrent_batches: int = Field(
        default=3, ge=1, description="Maximum concurrent processing batches"
    )
    processing_timeout: int = Field(
        default=300, ge=30, description="Processing timeout in seconds"
    )

    # API Configuration
    api_title: str = Field(default="Anime MCP Server", description="API title")
    api_description: str = Field(
        default="Semantic search API for anime database with MCP integration",
        description="API description",
    )
    api_version: str = Field(default="1.0.0", description="API version")
    max_search_limit: int = Field(
        default=100, ge=1, le=1000, description="Maximum search results limit"
    )

    # CORS Configuration
    allowed_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
        ],
        description="Allowed CORS origins",
    )
    allowed_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE"], description="Allowed HTTP methods"
    )
    allowed_headers: List[str] = Field(
        default=["*"], description="Allowed HTTP headers"
    )

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )

    # Data Source Configuration
    anime_database_url: str = Field(
        default="https://github.com/manami-project/anime-offline-database/raw/master/anime-offline-database.json",
        description="URL for anime offline database",
    )
    data_cache_ttl: int = Field(
        default=86400, ge=300, description="Data cache TTL in seconds"
    )

    # Health Check Configuration
    health_check_timeout: int = Field(
        default=10, ge=1, description="Health check timeout in seconds"
    )

    @validator("qdrant_url")
    def validate_qdrant_url(cls, v):
        """Validate Qdrant URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Qdrant URL must start with http:// or https://")
        return v

    @validator("qdrant_distance_metric")
    def validate_distance_metric(cls, v):
        """Validate distance metric is supported by Qdrant."""
        valid_metrics = ["cosine", "euclid", "dot"]
        if v.lower() not in valid_metrics:
            raise ValueError(f"Distance metric must be one of: {valid_metrics}")
        return v.lower()

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    @validator("fastembed_model")
    def validate_fastembed_model(cls, v):
        """Validate FastEmbed model format."""
        # Common FastEmbed models for validation
        valid_models = [
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "sentence-transformers/all-MiniLM-L6-v2",
            "intfloat/e5-small-v2",
            "intfloat/e5-base-v2",
        ]
        if v not in valid_models:
            # Allow custom models but warn
            import warnings

            warnings.warn(
                f"FastEmbed model '{v}' not in validated list. Ensure it's compatible."
            )
        return v

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True
        extra = "ignore"  # Ignore unknown environment variables


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get application settings instance.

    Returns:
        Settings: Application configuration
    """
    return settings


def reload_settings() -> Settings:
    """
    Reload settings from environment variables.

    Returns:
        Settings: Reloaded application configuration
    """
    global settings
    settings = Settings()
    return settings
