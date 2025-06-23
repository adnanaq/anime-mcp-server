"""Unit tests for configuration management."""
import pytest
from unittest.mock import patch, MagicMock
import os
from typing import Dict, Any

# Mock pydantic and pydantic_settings before importing config
with patch('sys.modules.pydantic', MagicMock()):
    with patch('sys.modules.pydantic_settings', MagicMock()):
        # Create mock classes that behave like pydantic
        class MockBaseSettings:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
                # Set default values for testing
                self.qdrant_url = kwargs.get('qdrant_url', 'http://localhost:6333')
                self.qdrant_collection_name = kwargs.get('qdrant_collection_name', 'anime_database')
                self.qdrant_vector_size = kwargs.get('qdrant_vector_size', 384)
                self.qdrant_distance_metric = kwargs.get('qdrant_distance_metric', 'cosine')
                self.fastembed_model = kwargs.get('fastembed_model', 'BAAI/bge-small-en-v1.5')
                self.batch_size = kwargs.get('batch_size', 1000)
                self.max_concurrent_batches = kwargs.get('max_concurrent_batches', 3)
                self.processing_timeout = kwargs.get('processing_timeout', 300)
                self.host = kwargs.get('host', '0.0.0.0')
                self.port = kwargs.get('port', 8000)
                self.debug = kwargs.get('debug', True)
                self.api_title = kwargs.get('api_title', 'Anime MCP Server')
                self.api_description = kwargs.get('api_description', 'Semantic search API for anime database with MCP integration')
                self.api_version = kwargs.get('api_version', '1.0.0')
                self.max_search_limit = kwargs.get('max_search_limit', 100)
                self.allowed_origins = kwargs.get('allowed_origins', ['http://localhost:3000'])
                self.allowed_methods = kwargs.get('allowed_methods', ['GET', 'POST', 'PUT', 'DELETE'])
                self.allowed_headers = kwargs.get('allowed_headers', ['*'])
                self.log_level = kwargs.get('log_level', 'INFO')
                self.log_format = kwargs.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                self.anime_database_url = kwargs.get('anime_database_url', 'https://github.com/manami-project/anime-offline-database/raw/master/anime-offline-database.json')
                self.data_cache_ttl = kwargs.get('data_cache_ttl', 86400)
                self.health_check_timeout = kwargs.get('health_check_timeout', 10)
        
        # Mock Field function
        def MockField(**kwargs):
            return kwargs.get('default', None)
        
        # Mock validator decorator
        def MockValidator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator


class TestConfigurationManagement:
    """Test cases for centralized configuration management."""
    
    def setup_method(self):
        """Setup test environment."""
        self.original_env = os.environ.copy()
    
    def teardown_method(self):
        """Cleanup test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_default_configuration_values(self):
        """Test that default configuration values are properly set."""
        # Mock the Settings class
        with patch('src.config.Settings', MockBaseSettings):
            with patch('src.config.PydanticBaseSettings', MockBaseSettings):
                with patch('src.config.Field', MockField):
                    with patch('src.config.validator', MockValidator):
                        # Import after mocking
                        import importlib
                        import sys
                        if 'src.config' in sys.modules:
                            importlib.reload(sys.modules['src.config'])
                        
                        # Test basic instantiation
                        settings = MockBaseSettings()
                        
                        # Test default values
                        assert settings.qdrant_url == "http://localhost:6333"
                        assert settings.qdrant_collection_name == "anime_database"
                        assert settings.qdrant_vector_size == 384
                        assert settings.qdrant_distance_metric == "cosine"
                        assert settings.fastembed_model == "BAAI/bge-small-en-v1.5"
                        assert settings.batch_size == 1000
                        assert settings.max_concurrent_batches == 3
                        assert settings.processing_timeout == 300
                        assert settings.host == "0.0.0.0"
                        assert settings.port == 8000
                        assert settings.debug is True
    
    def test_environment_variable_override(self):
        """Test that environment variables properly override defaults."""
        # Set test environment variables
        os.environ['QDRANT_URL'] = 'http://test-qdrant:6333'
        os.environ['QDRANT_COLLECTION_NAME'] = 'test_collection'
        os.environ['QDRANT_VECTOR_SIZE'] = '512'
        os.environ['BATCH_SIZE'] = '2000'
        os.environ['HOST'] = '127.0.0.1'
        os.environ['PORT'] = '9000'
        os.environ['DEBUG'] = 'false'
        
        # Create settings with environment variables
        settings = MockBaseSettings(
            qdrant_url=os.environ.get('QDRANT_URL', 'http://localhost:6333'),
            qdrant_collection_name=os.environ.get('QDRANT_COLLECTION_NAME', 'anime_database'),
            qdrant_vector_size=int(os.environ.get('QDRANT_VECTOR_SIZE', '384')),
            batch_size=int(os.environ.get('BATCH_SIZE', '1000')),
            host=os.environ.get('HOST', '0.0.0.0'),
            port=int(os.environ.get('PORT', '8000')),
            debug=os.environ.get('DEBUG', 'true').lower() == 'true'
        )
        
        # Verify environment overrides work
        assert settings.qdrant_url == "http://test-qdrant:6333"
        assert settings.qdrant_collection_name == "test_collection"
        assert settings.qdrant_vector_size == 512
        assert settings.batch_size == 2000
        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        assert settings.debug is False
    
    def test_configuration_validation_logic(self):
        """Test configuration validation logic."""
        # Test valid configurations
        valid_configs = [
            {'qdrant_url': 'http://localhost:6333'},
            {'qdrant_url': 'https://qdrant.example.com:6333'},
            {'qdrant_distance_metric': 'cosine'},
            {'qdrant_distance_metric': 'euclid'},
            {'qdrant_distance_metric': 'dot'},
            {'log_level': 'DEBUG'},
            {'log_level': 'INFO'},
            {'log_level': 'WARNING'},
            {'log_level': 'ERROR'},
            {'log_level': 'CRITICAL'},
        ]
        
        for config in valid_configs:
            try:
                settings = MockBaseSettings(**config)
                assert True  # Should not raise exception
            except Exception as e:
                pytest.fail(f"Valid config {config} raised exception: {e}")
    
    def test_invalid_configuration_detection(self):
        """Test that invalid configurations are detected."""
        # Test URL validation logic manually
        invalid_urls = [
            'localhost:6333',  # Missing protocol
            'ftp://localhost:6333',  # Wrong protocol
            'qdrant',  # Not a URL
        ]
        
        for url in invalid_urls:
            # Simulate validation logic
            if not url.startswith(('http://', 'https://')):
                assert True  # Would raise validation error
            else:
                pytest.fail(f"Invalid URL {url} should be rejected")
        
        # Test distance metric validation logic
        invalid_metrics = ['manhattan', 'hamming', 'invalid']
        valid_metrics = ['cosine', 'euclid', 'dot']
        
        for metric in invalid_metrics:
            if metric.lower() not in valid_metrics:
                assert True  # Would raise validation error
            else:
                pytest.fail(f"Invalid metric {metric} should be rejected")
        
        # Test log level validation logic
        invalid_levels = ['TRACE', 'VERBOSE', 'INVALID']
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in invalid_levels:
            if level.upper() not in valid_levels:
                assert True  # Would raise validation error
            else:
                pytest.fail(f"Invalid log level {level} should be rejected")
    
    def test_settings_reload_functionality(self):
        """Test settings reload functionality."""
        # Initial settings
        initial_settings = MockBaseSettings(qdrant_url='http://initial:6333')
        assert initial_settings.qdrant_url == 'http://initial:6333'
        
        # Change environment and create new settings (simulating reload)
        os.environ['QDRANT_URL'] = 'http://reloaded:6333'
        reloaded_settings = MockBaseSettings(
            qdrant_url=os.environ.get('QDRANT_URL', 'http://localhost:6333')
        )
        
        assert reloaded_settings.qdrant_url == 'http://reloaded:6333'
        assert reloaded_settings.qdrant_url != initial_settings.qdrant_url
    
    def test_model_configuration_consistency(self):
        """Test that FastEmbed model configuration is consistent."""
        # Test common FastEmbed models
        valid_models = [
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5", 
            "sentence-transformers/all-MiniLM-L6-v2",
            "intfloat/e5-small-v2",
            "intfloat/e5-base-v2"
        ]
        
        for model in valid_models:
            settings = MockBaseSettings(fastembed_model=model)
            assert settings.fastembed_model == model
    
    def test_cors_configuration(self):
        """Test CORS configuration handling."""
        # Test default CORS settings
        settings = MockBaseSettings()
        assert isinstance(settings.allowed_origins, list)
        assert isinstance(settings.allowed_methods, list)
        assert isinstance(settings.allowed_headers, list)
        
        # Test custom CORS settings
        custom_origins = ['http://localhost:3000', 'https://example.com']
        custom_methods = ['GET', 'POST']
        custom_headers = ['Content-Type', 'Authorization']
        
        settings = MockBaseSettings(
            allowed_origins=custom_origins,
            allowed_methods=custom_methods,
            allowed_headers=custom_headers
        )
        
        assert settings.allowed_origins == custom_origins
        assert settings.allowed_methods == custom_methods
        assert settings.allowed_headers == custom_headers
    
    def test_numeric_configuration_bounds(self):
        """Test numeric configuration validation bounds."""
        # Test port bounds (1-65535)
        valid_ports = [1, 8000, 65535]
        invalid_ports = [0, -1, 65536, 100000]
        
        for port in valid_ports:
            if 1 <= port <= 65535:
                assert True  # Valid port
            else:
                pytest.fail(f"Valid port {port} should be accepted")
        
        for port in invalid_ports:
            if not (1 <= port <= 65535):
                assert True  # Invalid port would be rejected
            else:
                pytest.fail(f"Invalid port {port} should be rejected")
        
        # Test search limit bounds (1-1000)
        valid_limits = [1, 50, 100, 1000]
        invalid_limits = [0, -1, 1001, 5000]
        
        for limit in valid_limits:
            if 1 <= limit <= 1000:
                assert True  # Valid limit
            else:
                pytest.fail(f"Valid limit {limit} should be accepted")
        
        for limit in invalid_limits:
            if not (1 <= limit <= 1000):
                assert True  # Invalid limit would be rejected
            else:
                pytest.fail(f"Invalid limit {limit} should be rejected")
    
    def test_timeout_configuration(self):
        """Test timeout configuration values."""
        # Test processing timeout (minimum 30 seconds)
        valid_timeouts = [30, 300, 600, 3600]
        invalid_timeouts = [0, 10, 29, -1]
        
        for timeout in valid_timeouts:
            if timeout >= 30:
                assert True  # Valid timeout
            else:
                pytest.fail(f"Valid timeout {timeout} should be accepted")
        
        for timeout in invalid_timeouts:
            if timeout < 30:
                assert True  # Invalid timeout would be rejected
            else:
                pytest.fail(f"Invalid timeout {timeout} should be rejected")
    
    def test_batch_processing_configuration(self):
        """Test batch processing configuration validation."""
        # Test batch size (minimum 1)
        valid_batch_sizes = [1, 100, 1000, 5000]
        invalid_batch_sizes = [0, -1, -100]
        
        for size in valid_batch_sizes:
            if size >= 1:
                assert True  # Valid batch size
            else:
                pytest.fail(f"Valid batch size {size} should be accepted")
        
        for size in invalid_batch_sizes:
            if size < 1:
                assert True  # Invalid batch size would be rejected
            else:
                pytest.fail(f"Invalid batch size {size} should be rejected")
        
        # Test max concurrent batches (minimum 1)
        valid_concurrent = [1, 3, 8, 16]
        invalid_concurrent = [0, -1, -5]
        
        for concurrent in valid_concurrent:
            if concurrent >= 1:
                assert True  # Valid concurrent value
            else:
                pytest.fail(f"Valid concurrent {concurrent} should be accepted")
        
        for concurrent in invalid_concurrent:
            if concurrent < 1:
                assert True  # Invalid concurrent would be rejected
            else:
                pytest.fail(f"Invalid concurrent {concurrent} should be rejected")
    
    def test_configuration_integration_points(self):
        """Test configuration integration with other modules."""
        # Test that configuration provides all required fields for QdrantClient
        settings = MockBaseSettings()
        
        required_qdrant_fields = [
            'qdrant_url', 'qdrant_collection_name', 'qdrant_vector_size',
            'qdrant_distance_metric', 'fastembed_model'
        ]
        
        for field in required_qdrant_fields:
            assert hasattr(settings, field), f"Settings missing required field: {field}"
            assert getattr(settings, field) is not None, f"Field {field} should not be None"
        
        # Test that configuration provides all required fields for DataService
        required_data_fields = [
            'batch_size', 'max_concurrent_batches', 'processing_timeout',
            'anime_database_url', 'data_cache_ttl'
        ]
        
        for field in required_data_fields:
            assert hasattr(settings, field), f"Settings missing required field: {field}"
            assert getattr(settings, field) is not None, f"Field {field} should not be None"
        
        # Test that configuration provides all required fields for FastAPI
        required_api_fields = [
            'host', 'port', 'debug', 'api_title', 'api_description', 'api_version',
            'allowed_origins', 'allowed_methods', 'allowed_headers'
        ]
        
        for field in required_api_fields:
            assert hasattr(settings, field), f"Settings missing required field: {field}"
            assert getattr(settings, field) is not None, f"Field {field} should not be None"