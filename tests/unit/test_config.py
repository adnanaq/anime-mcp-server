"""Unit tests for configuration management."""

import warnings

import pytest
from pydantic import ValidationError

from src.config import Settings, get_settings, reload_settings


class TestConfiguration:
    """Test configuration loading and validation."""

    def test_config_valid_defaults(self):
        """Test configuration with valid default values."""
        settings = Settings()
        assert settings.qdrant_url == "http://localhost:6333"
        assert settings.qdrant_collection_name == "anime_database"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000

    def test_qdrant_url_validation_invalid(self):
        """Test invalid Qdrant URL validation."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(qdrant_url="invalid-url")

        assert "Qdrant URL must start with http:// or https://" in str(exc_info.value)

    def test_qdrant_url_validation_valid(self):
        """Test valid Qdrant URL validation."""
        # Test http://
        settings = Settings(qdrant_url="http://localhost:6333")
        assert settings.qdrant_url == "http://localhost:6333"

        # Test https://
        settings = Settings(qdrant_url="https://cloud.qdrant.io")
        assert settings.qdrant_url == "https://cloud.qdrant.io"

    def test_distance_metric_validation_invalid(self):
        """Test invalid distance metric validation."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(qdrant_distance_metric="invalid")

        assert "Distance metric must be one of: ['cosine', 'euclid', 'dot']" in str(
            exc_info.value
        )

    def test_distance_metric_validation_valid(self):
        """Test valid distance metric validation."""
        settings = Settings(qdrant_distance_metric="COSINE")
        assert settings.qdrant_distance_metric == "cosine"  # Should be lowercased

    def test_log_level_validation_invalid(self):
        """Test invalid log level validation."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(log_level="INVALID")

        assert (
            "Log level must be one of: ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']"
            in str(exc_info.value)
        )

    def test_log_level_validation_valid(self):
        """Test valid log level validation."""
        settings = Settings(log_level="debug")
        assert settings.log_level == "DEBUG"  # Should be uppercased

    def test_server_mode_validation_invalid(self):
        """Test invalid server mode validation."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(server_mode="invalid")

        assert (
            "Server mode must be one of: ['stdio', 'http', 'sse', 'streamable']"
            in str(exc_info.value)
        )

    def test_server_mode_validation_valid(self):
        """Test valid server mode validation."""
        settings = Settings(server_mode="HTTP")
        assert settings.server_mode == "http"  # Should be lowercased

    def test_fastembed_model_validation_valid(self):
        """Test valid FastEmbed model validation."""
        settings = Settings(fastembed_model="BAAI/bge-small-en-v1.5")
        assert settings.fastembed_model == "BAAI/bge-small-en-v1.5"

    def test_fastembed_model_validation_custom_with_warning(self):
        """Test custom FastEmbed model validation with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            settings = Settings(fastembed_model="custom/model")

            assert settings.fastembed_model == "custom/model"
            assert len(w) == 1
            assert "not in validated list" in str(w[0].message)

    def test_get_settings_singleton(self):
        """Test that get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_reload_settings(self):
        """Test settings reload functionality."""
        original_settings = get_settings()
        reloaded_settings = reload_settings()

        # Should return new Settings instance
        assert reloaded_settings is not original_settings
        # But new get_settings() should return the reloaded one
        current_settings = get_settings()
        assert current_settings is reloaded_settings
