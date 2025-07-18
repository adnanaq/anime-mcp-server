"""
Centralized configuration management for Anime MCP Server.

This module provides type-safe configuration handling with validation,
eliminating hardcoded values and DRY violations across the codebase.
"""

from typing import List, Optional

from pydantic import ConfigDict, Field, field_validator
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

    # Multi-Vector Configuration (always enabled)
    image_vector_size: int = Field(
        default=512, description="Image embedding dimensions (CLIP)"
    )
    clip_model: str = Field(
        default="ViT-B/32", description="CLIP model for image embeddings"
    )
    
    # Modern Embedding Configuration
    # Text Embedding Model Selection
    text_embedding_provider: str = Field(
        default="huggingface", description="Text embedding provider: fastembed, huggingface, sentence-transformers"
    )
    text_embedding_model: str = Field(
        default="BAAI/bge-m3", description="Modern text embedding model name"
    )
    
    # Image Embedding Model Selection  
    image_embedding_provider: str = Field(
        default="jinaclip", description="Image embedding provider: clip, siglip, jinaclip"
    )
    image_embedding_model: str = Field(
        default="jinaai/jina-clip-v2", description="Modern image embedding model name"
    )
    
    # Model-Specific Configuration
    siglip_input_resolution: int = Field(
        default=384, description="SigLIP input image resolution"
    )
    
    jinaclip_input_resolution: int = Field(
        default=512, description="JinaCLIP input image resolution"
    )
    jinaclip_text_max_length: int = Field(
        default=77, description="JinaCLIP maximum text sequence length"
    )
    
    # BGE Configuration
    bge_model_version: str = Field(
        default="m3", description="BGE model version: v1.5, m3, reranker"
    )
    bge_model_size: str = Field(
        default="base", description="BGE model size: small, base, large"
    )
    bge_max_length: int = Field(
        default=8192, description="BGE maximum input sequence length"
    )
    
    model_cache_dir: Optional[str] = Field(
        default=None, description="Custom cache directory for embedding models"
    )
    model_warm_up: bool = Field(
        default=False, description="Pre-load and warm up models during initialization"
    )
    

    # Qdrant Performance Optimization Configuration
    # Vector Quantization (40x speedup potential, 60% storage reduction)
    qdrant_enable_quantization: bool = Field(
        default=False, description="Enable vector quantization for performance optimization"
    )
    qdrant_quantization_type: str = Field(
        default="scalar", description="Quantization type: binary, scalar, product"
    )
    qdrant_quantization_always_ram: Optional[bool] = Field(
        default=None, description="Keep quantized vectors in RAM for better performance"
    )
    
    # GPU Acceleration Configuration (10x indexing performance)
    qdrant_enable_gpu: bool = Field(
        default=False, description="Enable GPU acceleration for indexing (requires CUDA)"
    )
    qdrant_gpu_device: Optional[int] = Field(
        default=None, description="GPU device ID for acceleration (0, 1, etc.)"
    )
    
    # HNSW Performance Tuning (optimized for anime search patterns)
    qdrant_hnsw_ef_construct: Optional[int] = Field(
        default=None, description="HNSW ef_construct parameter (higher = better accuracy, slower indexing)"
    )
    qdrant_hnsw_m: Optional[int] = Field(
        default=None, description="HNSW M parameter (higher = better accuracy, more memory)"
    )
    qdrant_hnsw_max_indexing_threads: Optional[int] = Field(
        default=None, description="Maximum threads for HNSW indexing"
    )
    
    # Payload Indexing Configuration (faster filtering)
    qdrant_enable_payload_indexing: bool = Field(
        default=True, description="Enable automatic payload field indexing for faster filtering"
    )
    qdrant_indexed_payload_fields: List[str] = Field(
        default=["type", "year", "status", "genres", "studios"], 
        description="Payload fields to index for filtering optimization"
    )
    
    # Storage and Memory Optimization
    qdrant_enable_wal: Optional[bool] = Field(
        default=None, description="Enable Write-Ahead Logging (None = Qdrant default)"
    )
    qdrant_memory_mapping_threshold: Optional[int] = Field(
        default=None, description="Memory mapping threshold in KB (None = Qdrant default)"
    )
    qdrant_storage_compression_ratio: Optional[float] = Field(
        default=None, description="Storage compression ratio (None = Qdrant default)"
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

    # MCP Server Configuration
    server_mode: str = Field(
        default="stdio",
        description="MCP server transport mode: stdio, http, sse, streamable",
    )
    mcp_host: str = Field(
        default="0.0.0.0", description="MCP server host (for HTTP modes)"
    )
    mcp_port: int = Field(
        default=8001, ge=1024, le=65535, description="MCP server port (for HTTP modes)"
    )

    # Data Source Configuration
    anime_database_url: str = Field(
        default="https://github.com/manami-project/anime-offline-database/raw/master/anime-offline-database.json",
        description="URL for anime offline database",
    )
    data_cache_ttl: int = Field(
        default=86400, ge=300, description="Data cache TTL in seconds"
    )

    # External API Configuration
    mal_client_id: Optional[str] = Field(
        default=None, description="MyAnimeList OAuth2 Client ID for official API"
    )
    mal_client_secret: Optional[str] = Field(
        default=None, description="MyAnimeList OAuth2 Client Secret for official API"
    )
    anilist_client_id: Optional[str] = Field(
        default=None, description="AniList OAuth2 Client ID for user data"
    )
    anilist_client_secret: Optional[str] = Field(
        default=None, description="AniList OAuth2 Client Secret for user data"
    )
    anilist_auth_token: Optional[str] = Field(
        default=None,
        description="AniList OAuth2 Bearer Token for authenticated requests",
    )
    anidb_client: Optional[str] = Field(
        default="dimeapi", description="AniDB client name for API requests"
    )
    anidb_clientver: Optional[str] = Field(
        default="2", description="AniDB client version for API requests"
    )
    anidb_protover: Optional[str] = Field(
        default="1", description="AniDB protocol version for API requests"
    )
    anime_schedule_token: Optional[str] = Field(
        default=None, description="AnimeSchedule API token for authenticated requests"
    )

    # Health Check Configuration
    health_check_timeout: int = Field(
        default=10, ge=1, description="Health check timeout in seconds"
    )

    # LLM Configuration (Phase 6C)
    openai_api_key: Optional[str] = Field(
        default=None, description="OpenAI API key for LLM services"
    )
    anthropic_api_key: Optional[str] = Field(
        default=None, description="Anthropic API key for LLM services"
    )
    llm_provider: str = Field(
        default="openai", description="Default LLM provider: openai, anthropic"
    )

    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, v):
        """Validate Qdrant URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Qdrant URL must start with http:// or https://")
        return v

    @field_validator("qdrant_distance_metric")
    @classmethod
    def validate_distance_metric(cls, v):
        """Validate distance metric is supported by Qdrant."""
        valid_metrics = ["cosine", "euclid", "dot"]
        if v.lower() not in valid_metrics:
            raise ValueError(f"Distance metric must be one of: {valid_metrics}")
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    @field_validator("server_mode")
    @classmethod
    def validate_server_mode(cls, v):
        """Validate MCP server transport mode."""
        valid_modes = ["stdio", "http", "sse", "streamable"]
        if v.lower() not in valid_modes:
            raise ValueError(f"Server mode must be one of: {valid_modes}")
        return v.lower()

    @field_validator("fastembed_model")
    @classmethod
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

    @field_validator("qdrant_quantization_type")
    @classmethod
    def validate_quantization_type(cls, v):
        """Validate Qdrant quantization type."""
        valid_types = ["binary", "scalar", "product"]
        if v.lower() not in valid_types:
            raise ValueError(f"Quantization type must be one of: {valid_types}")
        return v.lower()

    @field_validator("qdrant_hnsw_ef_construct")
    @classmethod
    def validate_hnsw_ef_construct(cls, v):
        """Validate HNSW ef_construct parameter."""
        if v is not None and (v < 4 or v > 2000):
            raise ValueError("HNSW ef_construct must be between 4 and 2000")
        return v

    @field_validator("qdrant_hnsw_m")
    @classmethod
    def validate_hnsw_m(cls, v):
        """Validate HNSW M parameter."""
        if v is not None and (v < 2 or v > 100):
            raise ValueError("HNSW M must be between 2 and 100")
        return v

    @field_validator("text_embedding_provider")
    @classmethod
    def validate_text_embedding_provider(cls, v):
        """Validate text embedding provider."""
        valid_providers = ["fastembed", "huggingface", "sentence-transformers"]
        if v.lower() not in valid_providers:
            raise ValueError(f"Text embedding provider must be one of: {valid_providers}")
        return v.lower()

    @field_validator("image_embedding_provider")
    @classmethod
    def validate_image_embedding_provider(cls, v):
        """Validate image embedding provider."""
        valid_providers = ["clip", "siglip", "jinaclip"]
        if v.lower() not in valid_providers:
            raise ValueError(f"Image embedding provider must be one of: {valid_providers}")
        return v.lower()

    @field_validator("bge_model_version")
    @classmethod
    def validate_bge_model_version(cls, v):
        """Validate BGE model version."""
        valid_versions = ["v1.5", "m3", "reranker"]
        if v.lower() not in valid_versions:
            raise ValueError(f"BGE model version must be one of: {valid_versions}")
        return v.lower()

    @field_validator("bge_model_size")
    @classmethod
    def validate_bge_model_size(cls, v):
        """Validate BGE model size."""
        valid_sizes = ["small", "base", "large"]
        if v.lower() not in valid_sizes:
            raise ValueError(f"BGE model size must be one of: {valid_sizes}")
        return v.lower()

    @field_validator("siglip_input_resolution")
    @classmethod
    def validate_siglip_input_resolution(cls, v):
        """Validate SigLIP input resolution."""
        valid_resolutions = [224, 256, 384, 512]
        if v not in valid_resolutions:
            raise ValueError(f"SigLIP input resolution must be one of: {valid_resolutions}")
        return v

    @field_validator("jinaclip_input_resolution")
    @classmethod
    def validate_jinaclip_input_resolution(cls, v):
        """Validate JinaCLIP input resolution."""
        valid_resolutions = [224, 256, 384, 512]
        if v not in valid_resolutions:
            raise ValueError(f"JinaCLIP input resolution must be one of: {valid_resolutions}")
        return v

    # Domain-Specific Fine-Tuning Configuration (Task #118)
    enable_fine_tuning: bool = Field(
        default=False, description="Enable domain-specific fine-tuning for anime content"
    )
    fine_tuning_data_path: str = Field(
        default="data/anime_fine_tuning_data.json", description="Path to fine-tuning dataset"
    )
    fine_tuning_model_dir: str = Field(
        default="models/anime_finetuned", description="Directory to save fine-tuned models"
    )
    fine_tuning_use_lora: bool = Field(
        default=True, description="Use LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning"
    )
    fine_tuning_lora_r: int = Field(
        default=8, ge=1, le=64, description="LoRA rank parameter"
    )
    fine_tuning_lora_alpha: int = Field(
        default=32, ge=1, le=128, description="LoRA alpha parameter"
    )
    fine_tuning_lora_dropout: float = Field(
        default=0.1, ge=0.0, le=0.5, description="LoRA dropout rate"
    )
    fine_tuning_batch_size: int = Field(
        default=16, ge=1, le=128, description="Fine-tuning batch size"
    )
    fine_tuning_learning_rate: float = Field(
        default=1e-4, ge=1e-6, le=1e-2, description="Fine-tuning learning rate"
    )
    fine_tuning_num_epochs: int = Field(
        default=3, ge=1, le=20, description="Number of fine-tuning epochs"
    )
    fine_tuning_warmup_steps: int = Field(
        default=100, ge=0, le=1000, description="Number of warmup steps"
    )
    
    # Character Recognition Fine-Tuning
    character_recognition_enabled: bool = Field(
        default=True, description="Enable character recognition fine-tuning"
    )
    character_recognition_weight: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Weight for character recognition task in multi-task training"
    )
    character_min_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold for character predictions"
    )
    
    # Art Style Classification Fine-Tuning
    art_style_classification_enabled: bool = Field(
        default=True, description="Enable art style classification fine-tuning"
    )
    art_style_classification_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Weight for art style classification task in multi-task training"
    )
    art_style_min_confidence: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum confidence threshold for art style predictions"
    )
    
    # Genre Enhancement Fine-Tuning
    genre_enhancement_enabled: bool = Field(
        default=True, description="Enable genre understanding enhancement fine-tuning"
    )
    genre_enhancement_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Weight for genre enhancement task in multi-task training"
    )
    genre_min_confidence: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Minimum confidence threshold for genre predictions"
    )
    
    # Fine-Tuning Data Configuration
    fine_tuning_train_split: float = Field(
        default=0.8, ge=0.5, le=0.95, description="Training data split ratio"
    )
    fine_tuning_validation_split: float = Field(
        default=0.1, ge=0.05, le=0.3, description="Validation data split ratio"
    )
    fine_tuning_test_split: float = Field(
        default=0.1, ge=0.05, le=0.3, description="Test data split ratio"
    )
    fine_tuning_augment_data: bool = Field(
        default=True, description="Enable data augmentation for fine-tuning"
    )
    fine_tuning_max_samples: Optional[int] = Field(
        default=None, description="Maximum number of samples to use for fine-tuning (None for all)"
    )
    
    # Fine-Tuning Performance Configuration
    fine_tuning_checkpoint_steps: int = Field(
        default=500, ge=100, le=5000, description="Steps between model checkpoints"
    )
    fine_tuning_eval_steps: int = Field(
        default=250, ge=50, le=2000, description="Steps between evaluation runs"
    )
    fine_tuning_save_best_model: bool = Field(
        default=True, description="Save the best performing model during training"
    )
    fine_tuning_early_stopping_patience: int = Field(
        default=3, ge=1, le=10, description="Early stopping patience (epochs without improvement)"
    )

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_assignment=True,
        extra="ignore",  # Ignore unknown environment variables
    )


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
