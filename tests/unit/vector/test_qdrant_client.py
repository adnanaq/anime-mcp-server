"""Unit tests for QdrantClient with FastEmbed integration."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.vector.qdrant_client import QdrantClient


class TestQdrantClient:
    """Test cases for QdrantClient with FastEmbed."""

    @pytest.fixture
    def mock_fastembed(self):
        """Mock FastEmbed encoder."""
        mock_encoder = MagicMock()
        # Mock embedding generation
        mock_encoder.embed.return_value = [
            np.array([0.1, 0.2, 0.3] * 128)
        ]  # 384 dimensions
        return mock_encoder

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
    def client(self, mock_fastembed, mock_qdrant_sdk):
        """Create QdrantClient instance with mocked dependencies."""
        with (
            patch(
                "src.vector.qdrant_client.TextEmbedding", return_value=mock_fastembed
            ),
            patch("src.vector.qdrant_client.QdrantSDK", return_value=mock_qdrant_sdk),
        ):
            return QdrantClient(
                url="http://test-qdrant:6333", collection_name="test_anime"
            )

    @pytest.mark.asyncio
    async def test_health_check_success(self, client, mock_qdrant_sdk):
        """Test successful health check."""
        mock_qdrant_sdk.get_collections.return_value = MagicMock()

        result = await client.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client, mock_qdrant_sdk):
        """Test health check failure."""
        mock_qdrant_sdk.get_collections.side_effect = Exception("Connection failed")

        result = await client.health_check()
        assert result is False

    def test_create_embedding_success(self, client, mock_fastembed):
        """Test successful embedding creation with FastEmbed."""
        text = "action adventure anime with robots"

        embedding = client._create_embedding(text)

        assert len(embedding) == 384
        assert all(isinstance(val, float) for val in embedding)
        mock_fastembed.embed.assert_called_once_with([text])

    def test_create_embedding_empty_text(self, client):
        """Test embedding creation with empty text."""
        embedding = client._create_embedding("")

        assert len(embedding) == 384
        assert all(val == 0.0 for val in embedding)

    def test_create_embedding_whitespace_only(self, client):
        """Test embedding creation with whitespace only."""
        embedding = client._create_embedding("   \n\t   ")

        assert len(embedding) == 384
        assert all(val == 0.0 for val in embedding)

    def test_create_embedding_none_text(self, client):
        """Test embedding creation with None text."""
        embedding = client._create_embedding(None)

        assert len(embedding) == 384
        assert all(val == 0.0 for val in embedding)

    def test_create_embedding_fastembed_failure(self, client, mock_fastembed):
        """Test embedding creation when FastEmbed fails."""
        mock_fastembed.embed.side_effect = Exception("FastEmbed error")

        embedding = client._create_embedding("test text")

        # Should return zero vector on error
        assert len(embedding) == 384
        assert all(val == 0.0 for val in embedding)

    def test_create_embedding_wrong_dimensions(self, client, mock_fastembed):
        """Test embedding creation with wrong dimensions from FastEmbed."""
        # Mock FastEmbed returning wrong size
        mock_fastembed.embed.return_value = [np.array([0.1, 0.2])]  # Only 2 dimensions

        embedding = client._create_embedding("test text")

        # Should pad to correct size
        assert len(embedding) == 384
        assert embedding[0] == 0.1
        assert embedding[1] == 0.2
        assert all(val == 0.0 for val in embedding[2:])

    def test_create_embedding_too_large_dimensions(self, client, mock_fastembed):
        """Test embedding creation with too many dimensions from FastEmbed."""
        # Mock FastEmbed returning too many dimensions
        large_embedding = np.array([0.1] * 500)  # 500 dimensions
        mock_fastembed.embed.return_value = [large_embedding]

        embedding = client._create_embedding("test text")

        # Should truncate to correct size
        assert len(embedding) == 384
        assert all(val == 0.1 for val in embedding)

    @pytest.mark.asyncio
    async def test_add_documents_success(self, client, mock_qdrant_sdk):
        """Test successful document addition."""
        documents = [
            {
                "anime_id": "test123",
                "title": "Test Anime",
                "embedding_text": "action adventure anime",
                "synopsis": "A test anime",
                "tags": ["action", "adventure"],
            },
            {
                "anime_id": "test456",
                "title": "Another Anime",
                "embedding_text": "romance comedy school",
                "synopsis": "A romantic comedy",
                "tags": ["romance", "comedy"],
            },
        ]

        mock_qdrant_sdk.upsert.return_value = True

        result = await client.add_documents(documents)

        assert result is True
        mock_qdrant_sdk.upsert.assert_called()

    @pytest.mark.asyncio
    async def test_add_documents_empty_list(self, client):
        """Test adding empty document list."""
        result = await client.add_documents([])
        assert result is True

    @pytest.mark.asyncio
    async def test_add_documents_missing_embedding_text(self, client, mock_qdrant_sdk):
        """Test adding documents with missing embedding_text."""
        documents = [
            {
                "anime_id": "test123",
                "title": "Test Anime",
                # Missing embedding_text
                "synopsis": "A test anime",
            }
        ]

        mock_qdrant_sdk.upsert.return_value = True

        result = await client.add_documents(documents)

        # Should still succeed but skip documents without embedding_text
        assert result is True

    @pytest.mark.asyncio
    async def test_add_documents_qdrant_failure(self, client, mock_qdrant_sdk):
        """Test document addition when Qdrant fails."""
        documents = [{"anime_id": "test123", "embedding_text": "test"}]

        mock_qdrant_sdk.upsert.side_effect = Exception("Qdrant error")

        result = await client.add_documents(documents)
        assert result is False

    @pytest.mark.asyncio
    async def test_search_success(self, client, mock_qdrant_sdk):
        """Test successful search."""
        # Mock search results
        mock_hit = MagicMock()
        mock_hit.payload = {"anime_id": "test123", "title": "Test Anime"}
        mock_hit.score = 0.95
        mock_hit.id = "point123"

        mock_qdrant_sdk.search.return_value = [mock_hit]

        results = await client.search("action anime", limit=5)

        assert len(results) == 1
        assert results[0]["title"] == "Test Anime"
        assert results[0]["_score"] == 0.95
        assert results[0]["_id"] == "point123"

    @pytest.mark.asyncio
    async def test_search_no_results(self, client, mock_qdrant_sdk):
        """Test search with no results."""
        mock_qdrant_sdk.search.return_value = []

        results = await client.search("nonexistent anime")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_with_filters(self, client, mock_qdrant_sdk):
        """Test search with metadata filters."""
        mock_qdrant_sdk.search.return_value = []

        filters = {"type": "TV", "year": 2023}
        await client.search("test query", filters=filters)

        # Verify search was called with filters
        mock_qdrant_sdk.search.assert_called()
        call_args = mock_qdrant_sdk.search.call_args
        assert call_args[1]["query_filter"] is not None

    @pytest.mark.asyncio
    async def test_search_qdrant_failure(self, client, mock_qdrant_sdk):
        """Test search when Qdrant fails."""
        mock_qdrant_sdk.search.side_effect = Exception("Search failed")

        results = await client.search("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_get_similar_anime_success(self, client, mock_qdrant_sdk):
        """Test successful similar anime retrieval."""
        # Mock retrieve result
        mock_point = MagicMock()
        mock_point.vector = [0.1] * 384
        mock_qdrant_sdk.retrieve.return_value = [mock_point]

        # Mock search results for similar anime
        mock_hit = MagicMock()
        mock_hit.payload = {"anime_id": "similar123", "title": "Similar Anime"}
        mock_hit.score = 0.85
        mock_qdrant_sdk.search.return_value = [mock_hit]

        results = await client.get_similar_anime("test123", limit=5)

        assert len(results) == 1
        assert results[0]["title"] == "Similar Anime"
        assert results[0]["similarity_score"] == 0.85

    @pytest.mark.asyncio
    async def test_get_similar_anime_not_found(self, client, mock_qdrant_sdk):
        """Test similar anime when target anime not found."""
        mock_qdrant_sdk.retrieve.return_value = []

        results = await client.get_similar_anime("nonexistent123")
        assert results == []

    @pytest.mark.asyncio
    async def test_get_similar_anime_qdrant_failure(self, client, mock_qdrant_sdk):
        """Test similar anime when Qdrant fails."""
        mock_qdrant_sdk.retrieve.side_effect = Exception("Retrieve failed")

        results = await client.get_similar_anime("test123")
        assert results == []

    @pytest.mark.asyncio
    async def test_get_stats_success(self, client, mock_qdrant_sdk):
        """Test successful stats retrieval."""
        # Mock collection info
        mock_collection = MagicMock()
        mock_collection.status = "green"
        mock_collection.optimizer_status = "ok"
        mock_collection.indexed_vectors_count = 1000
        mock_collection.points_count = 1000
        mock_qdrant_sdk.get_collection.return_value = mock_collection

        # Mock count result
        mock_count = MagicMock()
        mock_count.count = 1000
        mock_qdrant_sdk.count.return_value = mock_count

        stats = await client.get_stats()

        assert stats["collection_name"] == "test_anime"
        assert stats["total_documents"] == 1000
        assert stats["vector_size"] == 384
        assert stats["distance_metric"] == "cosine"

    @pytest.mark.asyncio
    async def test_get_stats_failure(self, client, mock_qdrant_sdk):
        """Test stats retrieval failure."""
        mock_qdrant_sdk.get_collection.side_effect = Exception("Stats failed")

        stats = await client.get_stats()
        assert "error" in stats

    def test_build_filter_simple(self, client):
        """Test building simple filters."""
        filters = {"type": "TV", "year": 2023}

        qdrant_filter = client._build_filter(filters)

        assert qdrant_filter is not None
        assert len(qdrant_filter.must) == 2

    def test_build_filter_range(self, client):
        """Test building range filters."""
        filters = {"year": {"gte": 2020, "lte": 2023}}

        qdrant_filter = client._build_filter(filters)

        assert qdrant_filter is not None
        assert len(qdrant_filter.must) == 1

    def test_build_filter_list_values(self, client):
        """Test building filters with list values."""
        filters = {"tags": ["action", "adventure"]}

        qdrant_filter = client._build_filter(filters)

        assert qdrant_filter is not None
        assert len(qdrant_filter.must) == 1

    def test_build_filter_empty(self, client):
        """Test building empty filters."""
        qdrant_filter = client._build_filter({})
        assert qdrant_filter is None

        qdrant_filter = client._build_filter(None)
        assert qdrant_filter is None

    @pytest.mark.asyncio
    async def test_clear_index_success(self, client, mock_qdrant_sdk):
        """Test successful index clearing."""
        mock_qdrant_sdk.delete_collection.return_value = True

        result = await client.clear_index()
        assert result is True

    @pytest.mark.asyncio
    async def test_clear_index_failure(self, client, mock_qdrant_sdk):
        """Test index clearing failure."""
        mock_qdrant_sdk.delete_collection.side_effect = Exception("Delete failed")

        result = await client.clear_index()
        assert result is False

    def test_generate_point_id_consistency(self, client):
        """Test that point ID generation is consistent."""
        anime_id = "test123"

        id1 = client._generate_point_id(anime_id)
        id2 = client._generate_point_id(anime_id)

        assert id1 == id2
        assert len(id1) == 32  # MD5 hash length

    def test_generate_point_id_uniqueness(self, client):
        """Test that different anime IDs generate different point IDs."""
        id1 = client._generate_point_id("anime1")
        id2 = client._generate_point_id("anime2")

        assert id1 != id2

    def test_client_initialization_default(self):
        """Test client initialization with default parameters."""
        with (
            patch("src.vector.qdrant_client.TextEmbedding"),
            patch("src.vector.qdrant_client.QdrantSDK"),
        ):
            client = QdrantClient()

            assert client.url == "http://localhost:6333"
            assert client.collection_name == "anime_database"
            assert client._vector_size == 384

    def test_client_initialization_custom(self):
        """Test client initialization with custom parameters."""
        with (
            patch("src.vector.qdrant_client.TextEmbedding"),
            patch("src.vector.qdrant_client.QdrantSDK"),
        ):
            client = QdrantClient(
                url="http://custom:6333", collection_name="custom_collection"
            )

            assert client.url == "http://custom:6333"
            assert client.collection_name == "custom_collection"

    def test_encoder_initialization_failure(self):
        """Test behavior when FastEmbed initialization fails."""
        with patch("src.vector.qdrant_client.TextEmbedding") as mock_text_embedding:
            mock_text_embedding.side_effect = Exception("FastEmbed init failed")

            with pytest.raises(Exception):
                QdrantClient()


# ============================================================================
# MULTI-VECTOR FUNCTIONALITY TESTS (PHASE 4)
# ============================================================================

from qdrant_client.models import (
    PointStruct,
    ScoredPoint,
)


class TestQdrantMultiVector:
    """Test suite for multi-vector QdrantClient functionality."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        from src.config import Settings

        settings = MagicMock(spec=Settings)
        settings.qdrant_url = "http://localhost:6333"
        settings.qdrant_collection_name = "test_anime_database"
        settings.qdrant_vector_size = 384
        settings.qdrant_distance_metric = "cosine"
        settings.fastembed_model = "BAAI/bge-small-en-v1.5"
        settings.fastembed_cache_dir = None
        return settings

    @pytest.fixture
    def mock_qdrant_sdk(self):
        """Mock QdrantSDK client."""
        with patch("src.vector.qdrant_client.QdrantSDK") as mock_sdk:
            mock_client = MagicMock()
            mock_sdk.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def mock_text_embedding(self):
        """Mock FastEmbed TextEmbedding."""
        with patch("src.vector.qdrant_client.TextEmbedding") as mock_embedding:
            mock_encoder = MagicMock()
            mock_embedding.return_value = mock_encoder
            # Mock embedding output - 384 dimensions
            mock_encoder.embed.return_value = [np.random.random(384).tolist()]
            yield mock_encoder

    @pytest.fixture
    def mock_vision_processor(self):
        """Mock VisionProcessor."""
        with patch("src.vector.qdrant_client.VisionProcessor") as mock_processor:
            processor = MagicMock()
            mock_processor.return_value = processor
            # Mock image embedding output - 512 dimensions
            processor.encode_image.return_value = np.random.random(512).tolist()
            yield processor

    @pytest.fixture
    def multi_vector_client(
        self, mock_settings, mock_qdrant_sdk, mock_text_embedding, mock_vision_processor
    ):
        """Create QdrantClient with multi-vector support enabled."""
        client = QdrantClient(mock_settings)
        # Enable multi-vector support manually for testing
        client._supports_multi_vector = True
        client._vision_processor = mock_vision_processor
        return client

    def test_multi_vector_initialization(self, multi_vector_client):
        """Test that multi-vector support can be enabled."""
        assert multi_vector_client._supports_multi_vector is True
        assert multi_vector_client._vision_processor is not None

    @pytest.mark.asyncio
    async def test_create_multi_vector_collection(
        self, multi_vector_client, mock_qdrant_sdk
    ):
        """Test creation of multi-vector collection with named vectors."""
        # Mock collection doesn't exist
        mock_qdrant_sdk.collection_exists.return_value = False
        mock_qdrant_sdk.create_collection.return_value = True

        await multi_vector_client._ensure_collection_exists()

        # Verify create_collection was called with multi-vector config
        mock_qdrant_sdk.create_collection.assert_called_once()
        call_args = mock_qdrant_sdk.create_collection.call_args
        vectors_config = call_args[1]["vectors_config"]

        # Should have both text and image vectors
        assert "text" in vectors_config
        assert "image" in vectors_config
        assert vectors_config["text"].size == 384
        assert vectors_config["image"].size == 512

    @pytest.mark.asyncio
    async def test_search_by_image(
        self, multi_vector_client, mock_qdrant_sdk, mock_vision_processor
    ):
        """Test image-based search functionality."""
        # Mock search results
        mock_points = [
            ScoredPoint(id="anime1", score=0.95, payload={"title": "Attack on Titan"}),
            ScoredPoint(id="anime2", score=0.87, payload={"title": "Death Note"}),
        ]
        mock_qdrant_sdk.search.return_value = mock_points

        # Test image search
        test_image_data = "base64_image_data_here"
        results = await multi_vector_client.search_by_image(test_image_data, limit=5)

        # Verify vision processor was called
        mock_vision_processor.encode_image.assert_called_once_with(test_image_data)

        # Verify search was called with image vector
        mock_qdrant_sdk.search.assert_called_once()
        search_call = mock_qdrant_sdk.search.call_args
        assert search_call[1]["using"] == "image"  # Named vector
        assert len(search_call[1]["query_vector"]) == 512  # Image embedding size

        # Verify results
        assert len(results) == 2
        assert results[0]["title"] == "Attack on Titan"
        assert results[0]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_find_visually_similar_anime(
        self, multi_vector_client, mock_qdrant_sdk, mock_vision_processor
    ):
        """Test finding visually similar anime by anime ID."""
        # Mock getting the reference anime first
        reference_anime = {"anime_id": "ref123", "title": "Reference Anime"}
        mock_qdrant_sdk.retrieve.return_value = [
            PointStruct(
                id="ref123", payload=reference_anime, vector={"image": [0.1] * 512}
            )
        ]

        # Mock similar anime search results
        mock_points = [
            ScoredPoint(id="anime1", score=0.93, payload={"title": "Similar Anime 1"}),
            ScoredPoint(id="anime2", score=0.89, payload={"title": "Similar Anime 2"}),
        ]
        mock_qdrant_sdk.search.return_value = mock_points

        results = await multi_vector_client.find_visually_similar_anime(
            "ref123", limit=3
        )

        # Verify retrieve was called to get reference anime
        mock_qdrant_sdk.retrieve.assert_called_once_with(
            collection_name=multi_vector_client.collection_name,
            ids=["ref123"],
            with_vectors=["image"],
        )

        # Verify search was called with reference image vector
        mock_qdrant_sdk.search.assert_called_once()
        search_call = mock_qdrant_sdk.search.call_args
        assert search_call[1]["using"] == "image"

        # Verify results
        assert len(results) == 2
        assert results[0]["title"] == "Similar Anime 1"

    @pytest.mark.asyncio
    async def test_search_multimodal(
        self,
        multi_vector_client,
        mock_qdrant_sdk,
        mock_text_embedding,
        mock_vision_processor,
    ):
        """Test multimodal search combining text and image."""
        # Mock text and image search results
        text_points = [
            ScoredPoint(id="anime1", score=0.91, payload={"title": "Text Match 1"}),
            ScoredPoint(id="anime2", score=0.85, payload={"title": "Text Match 2"}),
        ]
        image_points = [
            ScoredPoint(
                id="anime1", score=0.88, payload={"title": "Text Match 1"}
            ),  # Same anime
            ScoredPoint(id="anime3", score=0.82, payload={"title": "Image Match 1"}),
        ]

        # Mock multiple search calls
        mock_qdrant_sdk.search.side_effect = [text_points, image_points]

        results = await multi_vector_client.search_multimodal(
            text_query="mecha robots",
            image_data="base64_image",
            text_weight=0.7,
            limit=5,
        )

        # Verify both text and image embeddings were generated
        mock_text_embedding.embed.assert_called_once_with(["mecha robots"])
        mock_vision_processor.encode_image.assert_called_once_with("base64_image")

        # Verify two searches were performed
        assert mock_qdrant_sdk.search.call_count == 2

        # Verify combined results are returned
        assert len(results) > 0
        # anime1 should have highest combined score (0.91 * 0.7 + 0.88 * 0.3)

    @pytest.mark.asyncio
    async def test_add_documents_multi_vector(
        self,
        multi_vector_client,
        mock_qdrant_sdk,
        mock_text_embedding,
        mock_vision_processor,
    ):
        """Test adding documents with both text and image vectors."""
        # Mock successful upsert
        mock_qdrant_sdk.upsert.return_value = True

        documents = [
            {
                "anime_id": "anime1",
                "title": "Test Anime",
                "embedding_text": "Test anime about robots",
                "image_data": "base64_image_data",
            }
        ]

        await multi_vector_client.add_documents(documents)

        # Verify both text and image embeddings were generated
        mock_text_embedding.embed.assert_called_once()
        mock_vision_processor.encode_image.assert_called_once()

        # Verify upsert was called with multi-vector point
        mock_qdrant_sdk.upsert.assert_called_once()
        upsert_call = mock_qdrant_sdk.upsert.call_args
        points = upsert_call[1]["points"]

        assert len(points) == 1
        point = points[0]
        assert "text" in point.vector
        assert "image" in point.vector
        assert len(point.vector["text"]) == 384  # Text embedding size
        assert len(point.vector["image"]) == 512  # Image embedding size

    @pytest.mark.asyncio
    async def test_backward_compatibility_single_vector(
        self, mock_settings, mock_qdrant_sdk, mock_text_embedding
    ):
        """Test that existing single-vector functionality still works."""
        # Create client without multi-vector support
        client = QdrantClient(mock_settings)
        # Explicitly disable multi-vector support
        client._supports_multi_vector = False

        # Mock single-vector collection exists
        mock_qdrant_sdk.collection_exists.return_value = True

        # Mock search results
        mock_points = [
            ScoredPoint(id="anime1", score=0.95, payload={"title": "Attack on Titan"})
        ]
        mock_qdrant_sdk.search.return_value = mock_points

        # Test regular text search
        results = await client.search("mecha anime", limit=5)

        # Verify search used single vector (no 'using' parameter)
        search_call = mock_qdrant_sdk.search.call_args
        assert "using" not in search_call[1]  # Single vector mode

        # Verify results
        assert len(results) == 1
        assert results[0]["title"] == "Attack on Titan"

    @pytest.mark.asyncio
    async def test_graceful_fallback_missing_vision_processor(
        self, mock_settings, mock_qdrant_sdk, mock_text_embedding
    ):
        """Test graceful handling when VisionProcessor is unavailable."""
        with patch("src.vector.qdrant_client.VisionProcessor") as mock_vision_import:
            # Mock VisionProcessor import failure
            mock_vision_import.side_effect = ImportError(
                "VisionProcessor not available"
            )

            client = QdrantClient(mock_settings)

            # Should fall back to single-vector mode
            assert client._supports_multi_vector is False
            assert client._vision_processor is None

    def test_multi_vector_settings_validation(self):
        """Test that multi-vector settings are properly validated."""
        from src.config import Settings

        # Test with valid multi-vector settings
        settings = Settings()
        client = QdrantClient(settings)

        # Verify multi-vector support is based on VisionProcessor availability
        if hasattr(client, "_vision_processor") and client._vision_processor:
            assert client._supports_multi_vector is True
        else:
            assert client._supports_multi_vector is False

    @pytest.mark.asyncio
    async def test_image_search_error_handling(
        self, multi_vector_client, mock_vision_processor
    ):
        """Test error handling in image search functionality."""
        # Mock vision processor failure
        mock_vision_processor.encode_image.side_effect = Exception(
            "Image processing failed"
        )

        with pytest.raises(Exception, match="Image processing failed"):
            await multi_vector_client.search_by_image("invalid_image_data")

    @pytest.mark.asyncio
    async def test_multimodal_search_partial_failure(
        self,
        multi_vector_client,
        mock_qdrant_sdk,
        mock_text_embedding,
        mock_vision_processor,
    ):
        """Test multimodal search when one modality fails."""
        # Mock text search success, image search failure
        text_points = [
            ScoredPoint(id="anime1", score=0.9, payload={"title": "Text Result"})
        ]
        mock_qdrant_sdk.search.side_effect = [
            text_points,
            Exception("Image search failed"),
        ]

        # Should fall back to text-only search
        results = await multi_vector_client.search_multimodal(
            text_query="mecha anime", image_data="base64_image", limit=5
        )

        # Should return text results even if image search failed
        assert len(results) == 1
        assert results[0]["title"] == "Text Result"

    def test_collection_migration_compatibility(self, multi_vector_client):
        """Test that collection migration maintains compatibility."""
        # Verify the client can handle both single and multi-vector collections
        assert hasattr(multi_vector_client, "_ensure_collection_exists")
        assert hasattr(multi_vector_client, "_supports_multi_vector")

        # Test migration flag
        if multi_vector_client._supports_multi_vector:
            # Should create multi-vector collection
            assert multi_vector_client._vision_processor is not None
        else:
            # Should create single-vector collection
            assert multi_vector_client._vision_processor is None
