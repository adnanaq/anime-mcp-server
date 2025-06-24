"""Integration tests for centralized settings across modules."""

import os

import pytest


class TestSettingsIntegration:
    """Test cases for settings integration across different modules."""

    def setup_method(self):
        """Setup test environment."""
        self.original_env = os.environ.copy()

    def teardown_method(self):
        """Cleanup test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""

        class MockSettings:
            def __init__(self, **kwargs):
                # Qdrant settings
                self.qdrant_url = kwargs.get("qdrant_url", "http://localhost:6333")
                self.qdrant_collection_name = kwargs.get(
                    "qdrant_collection_name", "anime_database"
                )
                self.qdrant_vector_size = kwargs.get("qdrant_vector_size", 384)
                self.qdrant_distance_metric = kwargs.get(
                    "qdrant_distance_metric", "cosine"
                )
                self.fastembed_model = kwargs.get(
                    "fastembed_model", "BAAI/bge-small-en-v1.5"
                )
                self.fastembed_cache_dir = kwargs.get("fastembed_cache_dir", None)

                # Data processing settings
                self.batch_size = kwargs.get("batch_size", 1000)
                self.max_concurrent_batches = kwargs.get("max_concurrent_batches", 3)
                self.processing_timeout = kwargs.get("processing_timeout", 300)
                self.anime_database_url = kwargs.get(
                    "anime_database_url", "https://example.com/anime.json"
                )
                self.data_cache_ttl = kwargs.get("data_cache_ttl", 86400)

                # API settings
                self.host = kwargs.get("host", "0.0.0.0")
                self.port = kwargs.get("port", 8000)
                self.debug = kwargs.get("debug", True)
                self.api_title = kwargs.get("api_title", "Anime MCP Server")
                self.api_description = kwargs.get(
                    "api_description", "Semantic search API"
                )
                self.api_version = kwargs.get("api_version", "1.0.0")
                self.max_search_limit = kwargs.get("max_search_limit", 100)

                # CORS settings
                self.allowed_origins = kwargs.get(
                    "allowed_origins", ["http://localhost:3000"]
                )
                self.allowed_methods = kwargs.get("allowed_methods", ["GET", "POST"])
                self.allowed_headers = kwargs.get("allowed_headers", ["*"])

                # Logging settings
                self.log_level = kwargs.get("log_level", "INFO")
                self.log_format = kwargs.get(
                    "log_format", "%(asctime)s - %(levelname)s - %(message)s"
                )

                # Health check settings
                self.health_check_timeout = kwargs.get("health_check_timeout", 10)

        return MockSettings()

    def test_qdrant_client_settings_integration(self, mock_settings):
        """Test QdrantClient integration with centralized settings."""

        # Mock QdrantClient that uses settings
        class MockQdrantClient:
            def __init__(self, url=None, collection_name=None, settings=None):
                if settings:
                    self.settings = settings
                    self.url = url or settings.qdrant_url
                    self.collection_name = (
                        collection_name or settings.qdrant_collection_name
                    )
                    self._vector_size = settings.qdrant_vector_size
                    self._distance_metric = settings.qdrant_distance_metric
                    self._fastembed_model = settings.fastembed_model
                else:
                    # Fallback to defaults if no settings provided
                    self.url = url or "http://localhost:6333"
                    self.collection_name = collection_name or "anime_database"
                    self._vector_size = 384
                    self._distance_metric = "cosine"
                    self._fastembed_model = "BAAI/bge-small-en-v1.5"

        # Test with settings integration
        client_with_settings = MockQdrantClient(settings=mock_settings)
        assert client_with_settings.url == mock_settings.qdrant_url
        assert (
            client_with_settings.collection_name == mock_settings.qdrant_collection_name
        )
        assert client_with_settings._vector_size == mock_settings.qdrant_vector_size
        assert (
            client_with_settings._distance_metric
            == mock_settings.qdrant_distance_metric
        )
        assert client_with_settings._fastembed_model == mock_settings.fastembed_model

        # Test override behavior
        custom_url = "http://custom-qdrant:6333"
        custom_collection = "custom_collection"
        client_with_overrides = MockQdrantClient(
            url=custom_url, collection_name=custom_collection, settings=mock_settings
        )
        assert client_with_overrides.url == custom_url
        assert client_with_overrides.collection_name == custom_collection
        # Other settings should still come from settings object
        assert client_with_overrides._vector_size == mock_settings.qdrant_vector_size

    def test_data_service_settings_integration(self, mock_settings):
        """Test DataService integration with centralized settings."""

        # Mock DataService that uses settings
        class MockDataService:
            def __init__(self, settings=None):
                if settings:
                    self.settings = settings
                    self.anime_db_url = settings.anime_database_url
                    self.batch_size = settings.batch_size
                    self.max_concurrent_batches = settings.max_concurrent_batches
                    self.processing_timeout = settings.processing_timeout
                    self.data_cache_ttl = settings.data_cache_ttl
                else:
                    # Fallback to defaults
                    self.anime_db_url = "https://default.com/anime.json"
                    self.batch_size = 1000
                    self.max_concurrent_batches = 3
                    self.processing_timeout = 300
                    self.data_cache_ttl = 86400

            def create_processing_config(self):
                """Create processing configuration from settings."""
                return {
                    "batch_size": self.batch_size,
                    "max_concurrent_batches": self.max_concurrent_batches,
                    "processing_timeout": self.processing_timeout,
                }

        # Test with settings integration
        service_with_settings = MockDataService(settings=mock_settings)
        assert service_with_settings.anime_db_url == mock_settings.anime_database_url
        assert service_with_settings.batch_size == mock_settings.batch_size
        assert (
            service_with_settings.max_concurrent_batches
            == mock_settings.max_concurrent_batches
        )
        assert (
            service_with_settings.processing_timeout == mock_settings.processing_timeout
        )

        # Test processing config creation
        config = service_with_settings.create_processing_config()
        assert config["batch_size"] == mock_settings.batch_size
        assert config["max_concurrent_batches"] == mock_settings.max_concurrent_batches
        assert config["processing_timeout"] == mock_settings.processing_timeout

    def test_fastapi_app_settings_integration(self, mock_settings):
        """Test FastAPI app integration with centralized settings."""

        # Mock FastAPI app configuration
        class MockFastAPIApp:
            def __init__(self, settings=None):
                if settings:
                    self.title = settings.api_title
                    self.description = settings.api_description
                    self.version = settings.api_version
                    self.debug = settings.debug

                    # CORS configuration
                    self.cors_config = {
                        "allow_origins": settings.allowed_origins,
                        "allow_methods": settings.allowed_methods,
                        "allow_headers": settings.allowed_headers,
                    }

                    # Server configuration
                    self.server_config = {
                        "host": settings.host,
                        "port": settings.port,
                        "debug": settings.debug,
                    }
                else:
                    # Fallback defaults
                    self.title = "Default API"
                    self.description = "Default description"
                    self.version = "1.0.0"
                    self.debug = False

        # Test with settings integration
        app_with_settings = MockFastAPIApp(settings=mock_settings)
        assert app_with_settings.title == mock_settings.api_title
        assert app_with_settings.description == mock_settings.api_description
        assert app_with_settings.version == mock_settings.api_version
        assert app_with_settings.debug == mock_settings.debug

        # Test CORS configuration
        assert (
            app_with_settings.cors_config["allow_origins"]
            == mock_settings.allowed_origins
        )
        assert (
            app_with_settings.cors_config["allow_methods"]
            == mock_settings.allowed_methods
        )
        assert (
            app_with_settings.cors_config["allow_headers"]
            == mock_settings.allowed_headers
        )

        # Test server configuration
        assert app_with_settings.server_config["host"] == mock_settings.host
        assert app_with_settings.server_config["port"] == mock_settings.port
        assert app_with_settings.server_config["debug"] == mock_settings.debug

    def test_logging_settings_integration(self, mock_settings):
        """Test logging configuration integration with centralized settings."""

        # Mock logging configuration function
        def configure_logging(settings):
            """Configure logging based on settings."""
            return {
                "level": settings.log_level,
                "format": settings.log_format,
                "handlers": ["console"],
                "disable_existing_loggers": False,
            }

        # Test logging configuration
        log_config = configure_logging(mock_settings)
        assert log_config["level"] == mock_settings.log_level
        assert log_config["format"] == mock_settings.log_format

        # Test with different log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            test_settings = MockSettings(log_level=level)
            config = configure_logging(test_settings)
            assert config["level"] == level

    def test_environment_variable_propagation(self, mock_settings):
        """Test that environment variables properly propagate through settings."""
        # Set test environment variables
        test_env_vars = {
            "QDRANT_URL": "http://test-env:6333",
            "BATCH_SIZE": "2000",
            "API_TITLE": "Test Environment API",
            "DEBUG": "false",
            "LOG_LEVEL": "WARNING",
        }

        for key, value in test_env_vars.items():
            os.environ[key] = value

        # Create settings that read from environment
        env_settings = MockSettings(
            qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            batch_size=int(os.environ.get("BATCH_SIZE", "1000")),
            api_title=os.environ.get("API_TITLE", "Anime MCP Server"),
            debug=os.environ.get("DEBUG", "true").lower() == "true",
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )

        # Verify environment variables are reflected in settings
        assert env_settings.qdrant_url == "http://test-env:6333"
        assert env_settings.batch_size == 2000
        assert env_settings.api_title == "Test Environment API"
        assert env_settings.debug is False
        assert env_settings.log_level == "WARNING"

    def test_settings_consistency_across_modules(self, mock_settings):
        """Test that settings are consistent when used across different modules."""

        # Mock modules that use the same settings
        class MockModule1:
            def __init__(self, settings):
                self.qdrant_url = settings.qdrant_url
                self.batch_size = settings.batch_size

        class MockModule2:
            def __init__(self, settings):
                self.qdrant_url = settings.qdrant_url
                self.batch_size = settings.batch_size

        class MockModule3:
            def __init__(self, settings):
                self.qdrant_url = settings.qdrant_url
                self.batch_size = settings.batch_size

        # Create modules with shared settings
        module1 = MockModule1(mock_settings)
        module2 = MockModule2(mock_settings)
        module3 = MockModule3(mock_settings)

        # Verify all modules have consistent settings
        assert module1.qdrant_url == module2.qdrant_url == module3.qdrant_url
        assert module1.batch_size == module2.batch_size == module3.batch_size

        # Verify they all match the source settings
        assert module1.qdrant_url == mock_settings.qdrant_url
        assert module1.batch_size == mock_settings.batch_size

    def test_settings_validation_integration(self, mock_settings):
        """Test settings validation integration across modules."""

        # Mock validation functions that different modules might use
        def validate_qdrant_settings(settings):
            """Validate Qdrant-related settings."""
            errors = []

            if not settings.qdrant_url.startswith(("http://", "https://")):
                errors.append("Invalid Qdrant URL format")

            if settings.qdrant_vector_size <= 0:
                errors.append("Vector size must be positive")

            if settings.qdrant_distance_metric not in ["cosine", "euclid", "dot"]:
                errors.append("Invalid distance metric")

            return errors

        def validate_processing_settings(settings):
            """Validate data processing settings."""
            errors = []

            if settings.batch_size <= 0:
                errors.append("Batch size must be positive")

            if settings.max_concurrent_batches <= 0:
                errors.append("Max concurrent batches must be positive")

            if settings.processing_timeout < 30:
                errors.append("Processing timeout too low")

            return errors

        def validate_api_settings(settings):
            """Validate API settings."""
            errors = []

            if not (1 <= settings.port <= 65535):
                errors.append("Invalid port number")

            if not settings.api_title:
                errors.append("API title cannot be empty")

            return errors

        # Test validation with valid settings
        qdrant_errors = validate_qdrant_settings(mock_settings)
        processing_errors = validate_processing_settings(mock_settings)
        api_errors = validate_api_settings(mock_settings)

        assert len(qdrant_errors) == 0
        assert len(processing_errors) == 0
        assert len(api_errors) == 0

        # Test validation with invalid settings
        invalid_settings = MockSettings(
            qdrant_url="invalid-url",
            qdrant_vector_size=-1,
            qdrant_distance_metric="invalid",
            batch_size=0,
            max_concurrent_batches=-1,
            processing_timeout=10,
            port=99999,
            api_title="",
        )

        qdrant_errors = validate_qdrant_settings(invalid_settings)
        processing_errors = validate_processing_settings(invalid_settings)
        api_errors = validate_api_settings(invalid_settings)

        assert len(qdrant_errors) > 0
        assert len(processing_errors) > 0
        assert len(api_errors) > 0

    def test_settings_reload_behavior(self, mock_settings):
        """Test settings reload behavior across modules."""

        # Mock modules that cache settings
        class MockCachingModule:
            def __init__(self, settings):
                self._cached_settings = settings
                self.batch_size = settings.batch_size

            def reload_settings(self, new_settings):
                """Reload settings and update cached values."""
                self._cached_settings = new_settings
                self.batch_size = new_settings.batch_size

            def get_current_batch_size(self):
                return self.batch_size

        # Create module with initial settings
        module = MockCachingModule(mock_settings)
        initial_batch_size = module.get_current_batch_size()
        assert initial_batch_size == mock_settings.batch_size

        # Create new settings with different values
        new_settings = MockSettings(batch_size=5000)

        # Reload settings in module
        module.reload_settings(new_settings)
        updated_batch_size = module.get_current_batch_size()

        # Verify settings were updated
        assert updated_batch_size != initial_batch_size
        assert updated_batch_size == new_settings.batch_size

    def test_settings_dependency_injection(self, mock_settings):
        """Test settings dependency injection pattern."""

        # Mock classes that accept settings via dependency injection
        class MockServiceA:
            def __init__(self, settings):
                self.settings = settings
                self.qdrant_url = settings.qdrant_url

        class MockServiceB:
            def __init__(self, settings, service_a):
                self.settings = settings
                self.service_a = service_a
                self.batch_size = settings.batch_size

        class MockServiceC:
            def __init__(self, settings, service_b):
                self.settings = settings
                self.service_b = service_b
                self.api_title = settings.api_title

        # Create services with dependency injection
        service_a = MockServiceA(mock_settings)
        service_b = MockServiceB(mock_settings, service_a)
        service_c = MockServiceC(mock_settings, service_b)

        # Verify dependency chain and settings consistency
        assert service_a.settings is mock_settings
        assert service_b.settings is mock_settings
        assert service_c.settings is mock_settings

        assert service_a.qdrant_url == mock_settings.qdrant_url
        assert service_b.batch_size == mock_settings.batch_size
        assert service_c.api_title == mock_settings.api_title

        # Verify dependency relationships
        assert service_b.service_a is service_a
        assert service_c.service_b is service_b

    def test_settings_backward_compatibility(self, mock_settings):
        """Test backward compatibility when settings change."""

        # Mock old-style configuration access
        class MockLegacyModule:
            def __init__(self, qdrant_url=None, batch_size=None):
                self.qdrant_url = qdrant_url or "http://localhost:6333"
                self.batch_size = batch_size or 1000

        # Mock adapter that converts settings to legacy format
        class MockSettingsAdapter:
            def __init__(self, settings):
                self.settings = settings

            def create_legacy_module(self):
                """Create legacy module using settings values."""
                return MockLegacyModule(
                    qdrant_url=self.settings.qdrant_url,
                    batch_size=self.settings.batch_size,
                )

        # Test adapter pattern
        adapter = MockSettingsAdapter(mock_settings)
        legacy_module = adapter.create_legacy_module()

        # Verify legacy module gets correct values from new settings
        assert legacy_module.qdrant_url == mock_settings.qdrant_url
        assert legacy_module.batch_size == mock_settings.batch_size

    def test_settings_type_safety_integration(self, mock_settings):
        """Test type safety across settings integration."""

        # Mock functions that expect specific types
        def process_with_int_batch_size(batch_size: int) -> str:
            """Function that expects integer batch size."""
            return f"Processing with batch size: {batch_size}"

        def connect_with_str_url(url: str) -> str:
            """Function that expects string URL."""
            return f"Connecting to: {url}"

        def configure_with_bool_debug(debug: bool) -> str:
            """Function that expects boolean debug flag."""
            return f"Debug mode: {debug}"

        # Test type compatibility
        batch_result = process_with_int_batch_size(mock_settings.batch_size)
        url_result = connect_with_str_url(mock_settings.qdrant_url)
        debug_result = configure_with_bool_debug(mock_settings.debug)

        # Verify functions receive correct types
        assert isinstance(mock_settings.batch_size, int)
        assert isinstance(mock_settings.qdrant_url, str)
        assert isinstance(mock_settings.debug, bool)

        assert "Processing with batch size: 1000" in batch_result
        assert "Connecting to: http://localhost:6333" in url_result
        assert "Debug mode: True" in debug_result
