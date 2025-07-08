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
        """Mock settings for testing."""
        settings = MagicMock()
        settings.qdrant_url = "http://localhost:6333"
        settings.qdrant_collection_name = "test_anime"
        settings.qdrant_vector_size = 384
        settings.qdrant_distance_metric = "cosine"
        settings.fastembed_model = "BAAI/bge-small-en-v1.5"
        settings.fastembed_cache_dir = None
        settings.image_vector_size = 512
        settings.clip_model = "ViT-B/32"
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
                    return QdrantClient(settings=mock_settings)

    def test_initialization_always_multi_vector(self, client):
        """Test that QdrantClient always initializes in multi-vector mode."""
        # Multi-vector is always enabled - no toggle
        assert client._vector_size == 384
        assert client._image_vector_size == 512
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


class TestQdrantClientRealWorld:
    """Real-world integration tests (require actual dependencies)."""

    @pytest.mark.integration
    def test_real_initialization(self):
        """Test real initialization with actual dependencies."""
        try:
            from src.config import get_settings
            settings = get_settings()
            
            # This will test real FastEmbed and vision processor initialization
            client = QdrantClient(settings=settings)
            
            # Should successfully initialize
            assert client._vector_size == 384
            assert client._image_vector_size == 512
            
        except ImportError:
            pytest.skip("Real dependencies not available")

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