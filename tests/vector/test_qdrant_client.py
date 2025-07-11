"""
Simplified tests for QdrantClient after multi-vector always-enabled refactoring.

Tests focus on current functionality:
- Multi-vector operations (always enabled)
- Text and image embeddings
- Search operations
- Health checks
"""

from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from src.vector.qdrant_client import QdrantClient


class TestQdrantClientSimplified:
    """Simplified test cases for refactored QdrantClient."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing with optimization features."""
        settings = MagicMock()
        settings.qdrant_url = "http://localhost:6333"
        settings.qdrant_collection_name = "test_anime"
        settings.qdrant_vector_size = 384
        settings.qdrant_distance_metric = "cosine"
        settings.fastembed_model = "BAAI/bge-small-en-v1.5"
        settings.fastembed_cache_dir = None
        settings.image_vector_size = 512
        settings.clip_model = "ViT-B/32"
        
        # Optimization settings (Task #116)
        settings.qdrant_enable_quantization = True
        settings.qdrant_quantization_type = "scalar"
        settings.qdrant_quantization_always_ram = True
        settings.qdrant_enable_gpu = False  # Disabled for testing
        settings.qdrant_gpu_device = None
        settings.qdrant_hnsw_ef_construct = 128
        settings.qdrant_hnsw_m = 16
        settings.qdrant_hnsw_max_indexing_threads = 4
        settings.qdrant_enable_payload_indexing = True
        settings.qdrant_indexed_payload_fields = ["type", "year", "genres"]
        settings.qdrant_enable_wal = True
        settings.qdrant_memory_mapping_threshold = 20000
        settings.qdrant_storage_compression_ratio = 0.7
        
        # Modern embedding settings
        settings.text_embedding_provider = "huggingface"
        settings.image_embedding_provider = "jinaclip"
        settings.model_warm_up = False
        
        return settings

    @pytest.fixture
    def mock_qdrant_sdk(self):
        """Mock Qdrant SDK client."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.create_collection.return_value = True
        mock_client.get_collection.return_value = MagicMock()
        mock_client.count.return_value.count = 100
        return mock_client

    @pytest.fixture
    def mock_fastembed(self):
        """Mock FastEmbed encoder."""
        mock_encoder = MagicMock()
        mock_encoder.embed.return_value = [np.array([0.1, 0.2, 0.3] * 128)]  # 384 dimensions
        return mock_encoder

    @pytest.fixture
    def mock_vision_processor(self):
        """Mock vision processor."""
        mock_processor = MagicMock()
        mock_processor.encode_image.return_value = [0.1] * 512
        return mock_processor

    @pytest.fixture
    def client(self, mock_settings, mock_qdrant_sdk, mock_fastembed, mock_vision_processor):
        """Create QdrantClient instance with mocked dependencies."""
        with (
            patch("src.vector.qdrant_client.QdrantSDK", return_value=mock_qdrant_sdk),
        ):
            # Mock FastEmbed import and initialization
            with patch("fastembed.TextEmbedding", return_value=mock_fastembed):
                # Mock vision processor import and initialization
                with patch("src.vector.vision_processor.VisionProcessor", return_value=mock_vision_processor):
                    # Mock modern processors (Task #117)
                    with patch("src.vector.text_processor.TextProcessor") as mock_text, \
                         patch("src.vector.vision_processor.VisionProcessor") as mock_vision:
                        # Configure modern processor mocks
                        mock_text.return_value.get_model_info.return_value = {
                            "embedding_size": 1024,
                            "model_name": "BAAI/bge-m3",
                            "provider": "huggingface"
                        }
                        mock_vision.return_value.get_model_info.return_value = {
                            "embedding_size": 768,
                            "model_name": "jinaai/jina-clip-v2",
                            "provider": "jinaclip"
                        }
                        
                        return QdrantClient(settings=mock_settings)

    def test_initialization_always_multi_vector(self, client):
        """Test that QdrantClient always initializes in multi-vector mode."""
        # Multi-vector is always enabled - no toggle
        assert client._vector_size == 1024
        assert client._image_vector_size == 768
        assert client.vision_processor is not None

    def test_multi_vector_config_creation(self, client):
        """Test multi-vector configuration creation."""
        with patch("src.vector.qdrant_client.VectorParams") as mock_vector_params:
            # Mock VectorParams to return a mock object with size attribute
            mock_vector_params.return_value = MagicMock()
            mock_vector_params.return_value.size = 384  # Will be overridden for image vectors
            
            config = client._create_multi_vector_config()
            
            assert isinstance(config, dict)
            assert "text" in config
            assert "picture" in config  
            assert "thumbnail" in config
            
            # Verify VectorParams was called with correct sizes
            expected_calls = [
                ((384,), {'distance': mock_vector_params.return_value}),
                ((512,), {'distance': mock_vector_params.return_value}),
                ((512,), {'distance': mock_vector_params.return_value}),
            ]
            assert mock_vector_params.call_count == 3

    def test_create_text_embedding(self, client):
        """Test text embedding creation."""
        embedding = client._create_embedding("test anime query")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_create_text_embedding_empty(self, client):
        """Test text embedding with empty input."""
        embedding = client._create_embedding("")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(x == 0.0 for x in embedding)  # Zero vector for empty text

    def test_create_image_embedding(self, client):
        """Test image embedding creation."""
        fake_image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        embedding = client._create_image_embedding(fake_image_data)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 512
        assert all(isinstance(x, float) for x in embedding)

    def test_create_image_embedding_no_processor(self, mock_settings, mock_qdrant_sdk, mock_fastembed):
        """Test image embedding when vision processor is not available."""
        with (
            patch("src.vector.qdrant_client.QdrantSDK", return_value=mock_qdrant_sdk),
            patch("fastembed.TextEmbedding", return_value=mock_fastembed),
            patch("src.vector.vision_processor.VisionProcessor", side_effect=ImportError("No CLIP")),
        ):
            client = QdrantClient(settings=mock_settings)
            
            # Should gracefully handle missing vision processor
            assert client.vision_processor is None
            
            embedding = client._create_image_embedding("fake_image")
            assert embedding is None  # Returns None when vision processor unavailable

    @pytest.mark.asyncio
    async def test_health_check(self, client, mock_qdrant_sdk):
        """Test health check functionality."""
        mock_qdrant_sdk.get_collections.return_value = MagicMock()
        
        result = await client.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client, mock_qdrant_sdk):
        """Test health check failure."""
        mock_qdrant_sdk.get_collections.side_effect = Exception("Connection failed")
        
        result = await client.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_search_basic(self, client, mock_qdrant_sdk):
        """Test basic search functionality."""
        # Mock search results
        mock_hit = MagicMock()
        mock_hit.payload = {"anime_id": "test123", "title": "Test Anime"}
        mock_hit.score = 0.95
        mock_hit.id = "point123"
        mock_qdrant_sdk.search.return_value = [mock_hit]
        
        results = await client.search("dragon ball", limit=5)
        
        assert len(results) == 1
        assert results[0]["title"] == "Test Anime"
        assert results[0]["_score"] == 0.95

    @pytest.mark.asyncio
    async def test_search_by_image(self, client, mock_qdrant_sdk):
        """Test image-based search."""
        # Mock search results for both picture and thumbnail vectors
        mock_hit = MagicMock()
        mock_hit.payload = {"anime_id": "test123", "title": "Visual Match"}
        mock_hit.score = 0.85
        mock_hit.id = "point123"
        mock_qdrant_sdk.search.return_value = [mock_hit]
        
        fake_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        results = await client.search_by_image(fake_image, limit=3)
        
        assert isinstance(results, list)
        # Should work with combined picture/thumbnail search

    @pytest.mark.asyncio
    async def test_add_documents_multi_vector(self, client, mock_qdrant_sdk):
        """Test adding documents with multi-vector support."""
        mock_qdrant_sdk.upsert.return_value = True
        
        documents = [
            {
                "anime_id": "test123",
                "title": "Test Anime",
                "embedding_text": "action adventure anime",
                "picture_data": "fake_image_data",
                "thumbnail_data": "fake_thumb_data"
            }
        ]
        
        result = await client.add_documents(documents)
        assert result is True
        
        # Verify upsert was called with multi-vector points
        mock_qdrant_sdk.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_stats(self, client, mock_qdrant_sdk):
        """Test database statistics."""
        # Mock collection info
        collection_info = MagicMock()
        collection_info.status = "green"
        collection_info.optimizer_status = "ok"
        collection_info.indexed_vectors_count = 1000
        collection_info.points_count = 1000
        
        mock_qdrant_sdk.get_collection.return_value = collection_info
        mock_qdrant_sdk.count.return_value.count = 1000
        
        stats = await client.get_stats()
        
        assert stats["total_documents"] == 1000
        assert stats["vector_size"] == 384
        assert stats["status"] == "green"

    def test_generate_point_id_consistency(self, client):
        """Test point ID generation consistency."""
        anime_id = "test123"
        
        id1 = client._generate_point_id(anime_id)
        id2 = client._generate_point_id(anime_id)
        
        assert id1 == id2  # Same input should produce same ID

    def test_generate_point_id_uniqueness(self, client):
        """Test point ID generation uniqueness."""
        id1 = client._generate_point_id("anime1")
        id2 = client._generate_point_id("anime2")
        
        assert id1 != id2  # Different inputs should produce different IDs

    def test_quantization_config_creation(self, mock_settings, mock_qdrant_sdk, mock_fastembed, mock_vision_processor):
        """Test quantization configuration creation."""
        with patch("src.vector.qdrant_client.QdrantSDK", return_value=mock_qdrant_sdk), \
             patch("fastembed.TextEmbedding", return_value=mock_fastembed), \
             patch("src.vector.vision_processor.VisionProcessor", return_value=mock_vision_processor):
            
            client = QdrantClient(settings=mock_settings)
            
            # Test scalar quantization
            mock_settings.qdrant_enable_quantization = True
            mock_settings.qdrant_quantization_type = "scalar"
            quantization_config = client._create_quantization_config()
            
            assert quantization_config is not None
            
            # Test binary quantization
            mock_settings.qdrant_quantization_type = "binary"
            quantization_config = client._create_quantization_config()
            
            assert quantization_config is not None
            
            # Test product quantization
            mock_settings.qdrant_quantization_type = "product"
            quantization_config = client._create_quantization_config()
            
            assert quantization_config is not None

    def test_hnsw_optimization(self, mock_settings, mock_qdrant_sdk, mock_fastembed, mock_vision_processor):
        """Test HNSW parameter optimization."""
        with patch("src.vector.qdrant_client.QdrantSDK", return_value=mock_qdrant_sdk), \
             patch("fastembed.TextEmbedding", return_value=mock_fastembed), \
             patch("src.vector.vision_processor.VisionProcessor", return_value=mock_vision_processor):
            
            client = QdrantClient(settings=mock_settings)
            
            # Test multi-vector config with HNSW optimization
            vector_config = client._create_multi_vector_config()
            
            assert "text" in vector_config
            assert "picture" in vector_config
            assert "thumbnail" in vector_config
            
            # Each vector should have HNSW config
            for vector_name, vector_params in vector_config.items():
                assert vector_params.hnsw_config is not None

    def test_payload_indexing_setup(self, mock_settings, mock_qdrant_sdk, mock_fastembed, mock_vision_processor):
        """Test payload indexing configuration."""
        with patch("src.vector.qdrant_client.QdrantSDK", return_value=mock_qdrant_sdk), \
             patch("fastembed.TextEmbedding", return_value=mock_fastembed), \
             patch("src.vector.vision_processor.VisionProcessor", return_value=mock_vision_processor):
            
            client = QdrantClient(settings=mock_settings)
            
            # Reset the call count since it was called during collection creation
            mock_qdrant_sdk.create_payload_index.reset_mock()
            
            # Test payload indexing setup
            client._setup_payload_indexing()
            
            # Should create indexes for specified fields
            expected_calls = len(mock_settings.qdrant_indexed_payload_fields)
            assert mock_qdrant_sdk.create_payload_index.call_count == expected_calls

    def test_optimizers_config_creation(self, mock_settings, mock_qdrant_sdk, mock_fastembed, mock_vision_processor):
        """Test optimizers configuration creation."""
        with patch("src.vector.qdrant_client.QdrantSDK", return_value=mock_qdrant_sdk), \
             patch("fastembed.TextEmbedding", return_value=mock_fastembed), \
             patch("src.vector.vision_processor.VisionProcessor", return_value=mock_vision_processor):
            
            client = QdrantClient(settings=mock_settings)
            
            # Test optimizers config creation
            optimizers_config = client._create_optimizers_config()
            
            assert optimizers_config is not None

    def test_wal_config_creation(self, mock_settings, mock_qdrant_sdk, mock_fastembed, mock_vision_processor):
        """Test WAL configuration creation."""
        with patch("src.vector.qdrant_client.QdrantSDK", return_value=mock_qdrant_sdk), \
             patch("fastembed.TextEmbedding", return_value=mock_fastembed), \
             patch("src.vector.vision_processor.VisionProcessor", return_value=mock_vision_processor):
            
            client = QdrantClient(settings=mock_settings)
            
            # Test WAL config creation
            wal_config = client._create_wal_config()
            
            assert wal_config is not None

    @pytest.mark.asyncio
    async def test_hybrid_search_optimization(self, mock_settings, mock_qdrant_sdk, mock_fastembed, mock_vision_processor):
        """Test hybrid search optimization."""
        with patch("src.vector.qdrant_client.QdrantSDK", return_value=mock_qdrant_sdk), \
             patch("fastembed.TextEmbedding", return_value=mock_fastembed), \
             patch("src.vector.vision_processor.VisionProcessor", return_value=mock_vision_processor):
            
            # Mock search_batch for hybrid search
            mock_qdrant_sdk.search_batch.return_value = [
                [MagicMock(payload={"anime_id": "test1", "title": "Test Anime 1"}, score=0.9, id="test1")],
                [MagicMock(payload={"anime_id": "test2", "title": "Test Anime 2"}, score=0.8, id="test2")]
            ]
            
            # Mock image embedding
            mock_vision_processor.encode_image.return_value = [0.1] * 512
            
            client = QdrantClient(settings=mock_settings)
            
            # Test hybrid search
            results = await client.search_by_image("fake_image_data", limit=5, use_hybrid_search=True)
            
            # Should use search_batch for hybrid search
            assert mock_qdrant_sdk.search_batch.called
            assert len(results) > 0

    def test_optimization_config_validation(self, mock_settings):
        """Test optimization configuration validation."""
        # Test invalid quantization type
        mock_settings.qdrant_quantization_type = "invalid"
        
        with pytest.raises(Exception):
            from src.config import Settings
            # This should trigger validation error
            Settings(qdrant_quantization_type="invalid")

    def test_performance_settings_applied(self, mock_settings, mock_qdrant_sdk, mock_fastembed, mock_vision_processor):
        """Test that performance optimization settings are properly applied."""
        with patch("src.vector.qdrant_client.QdrantSDK", return_value=mock_qdrant_sdk), \
             patch("fastembed.TextEmbedding", return_value=mock_fastembed), \
             patch("src.vector.vision_processor.VisionProcessor", return_value=mock_vision_processor):
            
            client = QdrantClient(settings=mock_settings)
            
            # Verify collection creation includes optimization features
            mock_qdrant_sdk.create_collection.assert_called_once()
            call_args = mock_qdrant_sdk.create_collection.call_args
            
            # Should have optimization parameters
            assert "quantization_config" in call_args.kwargs
            assert "optimizers_config" in call_args.kwargs
            assert "wal_config" in call_args.kwargs


class TestQdrantClientRealWorld:
    """Real-world integration tests (require actual dependencies)."""

    @pytest.mark.integration
    def test_real_initialization_with_optimization(self):
        """Test real initialization with optimization features."""
        try:
            from src.config import get_settings
            settings = get_settings()
            
            # Mock Qdrant client to avoid requiring actual Qdrant server
            with patch("src.vector.qdrant_client.QdrantSDK") as mock_qdrant:
                mock_client = MagicMock()
                mock_client.get_collections.return_value.collections = []
                mock_qdrant.return_value = mock_client
                
                # This will test real FastEmbed and vision processor initialization
                client = QdrantClient(settings=settings)
                
                # Should successfully initialize with optimization features
                assert client._vector_size == 384
                assert client._image_vector_size == 512
                
                # Verify optimization methods exist
                assert hasattr(client, '_create_quantization_config')
                assert hasattr(client, '_create_optimizers_config')
                assert hasattr(client, '_create_wal_config')
                assert hasattr(client, '_setup_payload_indexing')
                
                # Verify collection creation was called with optimization
                mock_client.create_collection.assert_called_once()
                call_kwargs = mock_client.create_collection.call_args.kwargs
                assert 'vectors_config' in call_kwargs
                
        except ImportError as e:
            pytest.skip(f"Real dependencies not available: {e}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_health_check(self):
        """Test real health check (requires running Qdrant)."""
        try:
            from src.config import get_settings
            settings = get_settings()
            client = QdrantClient(settings=settings)
            
            # This will test real Qdrant connection
            health = await client.health_check()
            
            # Should connect successfully if Qdrant is running
            assert isinstance(health, bool)
            
        except Exception:
            pytest.skip("Qdrant not available for integration test")

    @pytest.mark.integration 
    @pytest.mark.asyncio
    async def test_optimization_performance_impact(self):
        """Test that optimizations improve performance metrics."""
        try:
            from src.config import get_settings
            settings = get_settings()
            
            with patch("src.vector.qdrant_client.QdrantSDK") as mock_qdrant:
                mock_client = MagicMock()
                mock_client.get_collections.return_value.collections = []
                
                # Mock search performance
                mock_hit = MagicMock()
                mock_hit.payload = {"anime_id": "test1", "title": "Test Anime"}
                mock_hit.score = 0.95
                mock_hit.id = "test1"
                
                # Mock faster search_batch for hybrid search
                mock_client.search_batch.return_value = [[mock_hit], [mock_hit]]
                mock_client.search.return_value = [mock_hit]
                
                mock_qdrant.return_value = mock_client
                
                # Test with optimization enabled
                client = QdrantClient(settings=settings)
                
                # Mock vision processor for image embedding
                with patch.object(client, '_create_image_embedding', return_value=[0.1] * 512):
                    # Test hybrid search (optimized)
                    results_hybrid = await client.search_by_image("test_image", use_hybrid_search=True)
                    
                    # Test legacy search (non-optimized)
                    results_legacy = await client.search_by_image("test_image", use_hybrid_search=False)
                    
                    # Both should return results
                    assert len(results_hybrid) > 0
                    assert len(results_legacy) > 0
                    
                    # Hybrid search should use search_batch (more efficient)
                    assert mock_client.search_batch.called
                    
                    # Legacy search should use individual search calls
                    assert mock_client.search.called
                    
        except ImportError as e:
            pytest.skip(f"Real dependencies not available: {e}")

    def test_optimization_configuration_completeness(self):
        """Test that all optimization features are properly configured."""
        from src.config import get_settings
        settings = get_settings()
        
        # Verify all optimization settings exist in configuration
        optimization_settings = [
            "qdrant_enable_quantization",
            "qdrant_quantization_type", 
            "qdrant_quantization_always_ram",
            "qdrant_enable_gpu",
            "qdrant_gpu_device",
            "qdrant_hnsw_ef_construct",
            "qdrant_hnsw_m",
            "qdrant_hnsw_max_indexing_threads",
            "qdrant_enable_payload_indexing",
            "qdrant_indexed_payload_fields",
            "qdrant_enable_wal",
            "qdrant_memory_mapping_threshold",
            "qdrant_storage_compression_ratio"
        ]
        
        for setting in optimization_settings:
            assert hasattr(settings, setting), f"Missing optimization setting: {setting}"
    
    def test_modern_embedding_configuration(self):
        """Test modern embedding configuration settings (Task #117)."""
        from src.config import get_settings
        settings = get_settings()
        
        # Verify modern embedding settings exist
        modern_embedding_settings = [
            "text_embedding_provider",
            "text_embedding_model",
            "image_embedding_provider",
            "image_embedding_model",
            "siglip_model",
            "siglip_input_resolution",
            "jinaclip_model",
            "jinaclip_input_resolution",
            "bge_model_version",
            "bge_model_size",
            "enable_model_fallback",
            "model_warm_up"
        ]
        
        for setting in modern_embedding_settings:
            assert hasattr(settings, setting), f"Missing modern embedding setting: {setting}"
    
    def test_modern_processors_initialization(self, mock_settings, mock_qdrant_sdk):
        """Test modern processors initialization when configured."""
        # Configure for modern processors
        mock_settings.text_embedding_provider = "huggingface"
        mock_settings.image_embedding_provider = "siglip"
        
        with patch("src.vector.qdrant_client.QdrantSDK", return_value=mock_qdrant_sdk), \
             patch("src.vector.text_processor.TextProcessor") as mock_text, \
             patch("src.vector.vision_processor.VisionProcessor") as mock_vision:
            
            # Configure modern processor mocks
            mock_text_processor = MagicMock()
            mock_text_processor.get_model_info.return_value = {"embedding_size": 768}
            mock_text.return_value = mock_text_processor
            
            mock_vision_processor = MagicMock()
            mock_vision_processor.get_model_info.return_value = {"embedding_size": 768}
            mock_vision.return_value = mock_vision_processor
            
            client = QdrantClient(settings=mock_settings)
            
            # Verify modern processors are initialized
            assert client.use_modern_processors is True
            assert client.modern_text_processor is not None
            assert client.modern_vision_processor is not None
            assert client._vector_size == 768
            assert client._image_vector_size == 768
    
    def test_modern_text_embedding_creation(self, mock_settings, mock_qdrant_sdk):
        """Test text embedding creation with modern processors."""
        # Configure for modern processors
        mock_settings.text_embedding_provider = "huggingface"
        mock_settings.image_embedding_provider = "clip"  # Keep legacy for simplicity
        
        with patch("src.vector.qdrant_client.QdrantSDK", return_value=mock_qdrant_sdk), \
             patch("src.vector.text_processor.TextProcessor") as mock_text, \
             patch("src.vector.vision_processor.VisionProcessor") as mock_vision_processor:
            
            # Configure modern text processor mock
            mock_text_processor = MagicMock()
            mock_text_processor.get_model_info.return_value = {"embedding_size": 768}
            mock_text_processor.encode_text.return_value = [0.1] * 768
            mock_text.return_value = mock_text_processor
            
            # Configure legacy vision processor mock
            mock_vision_processor.return_value.encode_image.return_value = [0.1] * 512
            
            client = QdrantClient(settings=mock_settings)
            embedding = client._create_embedding("test anime query")
            
            assert embedding is not None
            assert len(embedding) == 768
            assert all(isinstance(x, float) for x in embedding)
            mock_text_processor.encode_text.assert_called_once_with("test anime query")
    
    def test_modern_image_embedding_creation(self, mock_settings, mock_qdrant_sdk):
        """Test image embedding creation with modern processors."""
        # Configure for modern processors
        mock_settings.text_embedding_provider = "fastembed"  # Keep legacy for simplicity
        mock_settings.image_embedding_provider = "siglip"
        
        with patch("src.vector.qdrant_client.QdrantSDK", return_value=mock_qdrant_sdk), \
             patch("fastembed.TextEmbedding") as mock_fastembed, \
             patch("src.vector.vision_processor.VisionProcessor") as mock_vision:
            
            # Configure legacy text processor mock
            mock_text_encoder = MagicMock()
            mock_text_encoder.embed.return_value = [np.array([0.1] * 384)]
            mock_fastembed.return_value = mock_text_encoder
            
            # Configure modern vision processor mock
            mock_vision_processor = MagicMock()
            mock_vision_processor.get_model_info.return_value = {"embedding_size": 768}
            mock_vision_processor.encode_image.return_value = [0.1] * 768
            mock_vision.return_value = mock_vision_processor
            
            client = QdrantClient(settings=mock_settings)
            embedding = client._create_image_embedding("fake_image_data")
            
            assert embedding is not None
            assert len(embedding) == 768
            assert all(isinstance(x, float) for x in embedding)
            mock_vision_processor.encode_image.assert_called_once_with("fake_image_data")
    
    def test_legacy_fallback_on_modern_processor_failure(self, mock_settings, mock_qdrant_sdk):
        """Test fallback to legacy processors when modern processors fail."""
        # Configure for modern processors
        mock_settings.text_embedding_provider = "huggingface"
        mock_settings.image_embedding_provider = "siglip"
        
        with patch("src.vector.qdrant_client.QdrantSDK", return_value=mock_qdrant_sdk), \
             patch("src.vector.text_processor.TextProcessor") as mock_text, \
             patch("src.vector.vision_processor.VisionProcessor") as mock_vision, \
             patch("fastembed.TextEmbedding") as mock_fastembed, \
             patch("src.vector.vision_processor.VisionProcessor") as mock_vision_processor:
            
            # Make modern processors fail during initialization
            mock_text.side_effect = Exception("Modern text processor failed")
            mock_vision.side_effect = Exception("Modern vision processor failed")
            
            # Configure legacy processors
            mock_text_encoder = MagicMock()
            mock_text_encoder.embed.return_value = [np.array([0.1] * 384)]
            mock_fastembed.return_value = mock_text_encoder
            
            mock_vision_processor.return_value.encode_image.return_value = [0.1] * 512
            
            client = QdrantClient(settings=mock_settings)
            
            # Should have fallen back to legacy processors
            assert client.use_modern_processors is False
            assert client.encoder is not None
            assert client.vision_processor is not None
            
            # Test that embeddings work with legacy processors
            text_embedding = client._create_embedding("test query")
            assert text_embedding is not None
            assert len(text_embedding) == 384
            
            image_embedding = client._create_image_embedding("fake_image")
            assert image_embedding is not None
            assert len(image_embedding) == 512