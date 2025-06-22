"""Unit tests for QdrantClient with FastEmbed integration."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import numpy as np

from src.vector.qdrant_client import QdrantClient


class TestQdrantClient:
    """Test cases for QdrantClient with FastEmbed."""

    @pytest.fixture
    def mock_fastembed(self):
        """Mock FastEmbed encoder."""
        mock_encoder = MagicMock()
        # Mock embedding generation
        mock_encoder.embed.return_value = [np.array([0.1, 0.2, 0.3] * 128)]  # 384 dimensions
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
        with patch('src.vector.qdrant_client.TextEmbedding', return_value=mock_fastembed), \
             patch('src.vector.qdrant_client.QdrantSDK', return_value=mock_qdrant_sdk):
            return QdrantClient(url="http://test-qdrant:6333", collection_name="test_anime")

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
                "tags": ["action", "adventure"]
            },
            {
                "anime_id": "test456", 
                "title": "Another Anime",
                "embedding_text": "romance comedy school",
                "synopsis": "A romantic comedy",
                "tags": ["romance", "comedy"]
            }
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
                "synopsis": "A test anime"
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
        with patch('src.vector.qdrant_client.TextEmbedding'), \
             patch('src.vector.qdrant_client.QdrantSDK'):
            client = QdrantClient()
            
            assert client.url == "http://localhost:6333"
            assert client.collection_name == "anime_database"
            assert client._vector_size == 384

    def test_client_initialization_custom(self):
        """Test client initialization with custom parameters."""
        with patch('src.vector.qdrant_client.TextEmbedding'), \
             patch('src.vector.qdrant_client.QdrantSDK'):
            client = QdrantClient(
                url="http://custom:6333", 
                collection_name="custom_collection"
            )
            
            assert client.url == "http://custom:6333"
            assert client.collection_name == "custom_collection"

    def test_encoder_initialization_failure(self):
        """Test behavior when FastEmbed initialization fails."""
        with patch('src.vector.qdrant_client.TextEmbedding') as mock_text_embedding:
            mock_text_embedding.side_effect = Exception("FastEmbed init failed")
            
            with pytest.raises(Exception):
                QdrantClient()