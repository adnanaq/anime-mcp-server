"""Unit tests for QdrantClient."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.vector.qdrant_client import QdrantClient


class TestQdrantClient:
    """Test cases for QdrantClient."""

    @pytest.fixture
    def client(self):
        """Create QdrantClient instance."""
        return QdrantClient(url="http://test-marqo:8882", index_name="test_anime")

    @pytest.fixture
    def mock_marqo_instance(self):
        """Mock marqo client instance.""" 
        mock_client = MagicMock()
        mock_client.index.return_value = MagicMock()
        return mock_client

    @pytest.mark.asyncio
    async def test_health_check_success(self, client: QdrantClient, mock_marqo_instance):
        """Test successful health check."""
        mock_marqo_instance.health.return_value = {"status": "green"}
        
        with patch('marqo.Client', return_value=mock_marqo_instance):
            result = await client.health_check()
            
            assert result is True
            mock_marqo_instance.health.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client: QdrantClient):
        """Test health check failure."""
        with patch('marqo.Client') as mock_client_class:
            mock_client_class.side_effect = Exception("Connection failed")
            
            result = await client.health_check()
            
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_status(self, client: QdrantClient, mock_marqo_instance):
        """Test health check with unhealthy status."""
        mock_marqo_instance.health.return_value = {"status": "red"}
        
        with patch('marqo.Client', return_value=mock_marqo_instance):
            result = await client.health_check()
            
            assert result is False

    @pytest.mark.asyncio
    async def test_create_index_success(self, client: QdrantClient, mock_marqo_instance):
        """Test successful index creation."""
        mock_index = MagicMock()
        mock_marqo_instance.index.return_value = mock_index
        mock_index.get_stats.side_effect = Exception("Index doesn't exist")  # First call fails
        mock_marqo_instance.create_index.return_value = {"status": "created"}
        
        with patch('marqo.Client', return_value=mock_marqo_instance):
            result = await client.create_index()
            
            assert result is True
            mock_marqo_instance.create_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_index_already_exists(self, client: QdrantClient, mock_marqo_instance):
        """Test index creation when index already exists."""
        mock_index = MagicMock()
        mock_marqo_instance.index.return_value = mock_index
        mock_index.get_stats.return_value = {"numberOfDocuments": 100}  # Index exists
        
        with patch('marqo.Client', return_value=mock_marqo_instance):
            result = await client.create_index()
            
            assert result is True
            mock_marqo_instance.create_index.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_index_failure(self, client: QdrantClient, mock_marqo_instance):
        """Test index creation failure."""
        mock_index = MagicMock()
        mock_marqo_instance.index.return_value = mock_index
        mock_index.get_stats.side_effect = Exception("Index doesn't exist")
        mock_marqo_instance.create_index.side_effect = Exception("Creation failed")
        
        with patch('marqo.Client', return_value=mock_marqo_instance):
            result = await client.create_index()
            
            assert result is False

    @pytest.mark.asyncio
    async def test_add_documents_success(self, client: QdrantClient, mock_marqo_instance):
        """Test successful document addition."""
        documents = [
            {
                "anime_id": "test123",
                "title": "Test Anime",
                "embedding_text": "Test anime about testing",
                "myanimelist_id": 123,
                "anilist_id": 456
            }
        ]
        
        mock_index = MagicMock()
        mock_marqo_instance.index.return_value = mock_index
        mock_index.add_documents.return_value = {
            "errors": [],
            "processingTimeMs": 100,
            "index_name": "test_anime",
            "items": [{"_id": "test123", "status": 201}]
        }
        
        with patch('marqo.Client', return_value=mock_marqo_instance):
            result = await client.add_documents(documents)
            
            assert result is True
            mock_index.add_documents.assert_called_once_with(
                documents, tensor_fields=["embedding_text"]
            )

    @pytest.mark.asyncio
    async def test_add_documents_with_errors(self, client: QdrantClient, mock_marqo_instance):
        """Test document addition with some errors."""
        documents = [{"anime_id": "test123", "title": "Test"}]
        
        mock_index = MagicMock()
        mock_marqo_instance.index.return_value = mock_index
        mock_index.add_documents.return_value = {
            "errors": [{"error": "Invalid document", "id": "test123"}],
            "processingTimeMs": 100
        }
        
        with patch('marqo.Client', return_value=mock_marqo_instance):
            result = await client.add_documents(documents)
            
            assert result is False

    @pytest.mark.asyncio
    async def test_add_documents_empty_list(self, client: QdrantClient, mock_marqo_instance):
        """Test adding empty document list."""
        with patch('marqo.Client', return_value=mock_marqo_instance):
            result = await client.add_documents([])
            
            assert result is True  # Empty list should succeed trivially

    @pytest.mark.asyncio
    async def test_add_documents_exception(self, client: QdrantClient, mock_marqo_instance):
        """Test document addition with exception."""
        documents = [{"anime_id": "test123"}]
        
        mock_index = MagicMock()
        mock_marqo_instance.index.return_value = mock_index
        mock_index.add_documents.side_effect = Exception("Network error")
        
        with patch('marqo.Client', return_value=mock_marqo_instance):
            result = await client.add_documents(documents)
            
            assert result is False

    @pytest.mark.asyncio
    async def test_search_success(self, client: QdrantClient, mock_marqo_instance):
        """Test successful search."""
        mock_index = MagicMock()
        mock_marqo_instance.index.return_value = mock_index
        mock_index.search.return_value = {
            "hits": [
                {
                    "anime_id": "test123",
                    "title": "Test Anime",
                    "myanimelist_id": 123,
                    "anilist_id": 456,
                    "_score": 0.95,
                    "_highlights": {}
                }
            ],
            "processingTimeMs": 50,
            "query": "test anime"
        }
        
        with patch('marqo.Client', return_value=mock_marqo_instance):
            results = await client.search("test anime", limit=10)
            
            assert len(results) == 1
            assert results[0]["anime_id"] == "test123"
            assert results[0]["_score"] == 0.95
            assert results[0]["processing_time_ms"] == 50
            
            mock_index.search.assert_called_once_with(
                q="test anime", limit=10
            )

    @pytest.mark.asyncio
    async def test_search_no_results(self, client: QdrantClient, mock_marqo_instance):
        """Test search with no results."""
        mock_index = MagicMock()
        mock_marqo_instance.index.return_value = mock_index
        mock_index.search.return_value = {
            "hits": [],
            "processingTimeMs": 25,
            "query": "nonexistent"
        }
        
        with patch('marqo.Client', return_value=mock_marqo_instance):
            results = await client.search("nonexistent", limit=10)
            
            assert results == []

    @pytest.mark.asyncio
    async def test_search_exception(self, client: QdrantClient, mock_marqo_instance):
        """Test search with exception."""
        mock_index = MagicMock()
        mock_marqo_instance.index.return_value = mock_index
        mock_index.search.side_effect = Exception("Search failed")
        
        with patch('marqo.Client', return_value=mock_marqo_instance):
            results = await client.search("test", limit=10)
            
            assert results == []

    @pytest.mark.asyncio
    async def test_search_with_filters(self, client: QdrantClient, mock_marqo_instance):
        """Test search with filters."""
        mock_index = MagicMock()
        mock_marqo_instance.index.return_value = mock_index
        mock_index.search.return_value = {
            "hits": [],
            "processingTimeMs": 30,
            "query": "test"
        }
        
        filters = {"type": "TV", "year": 2023}
        
        with patch('marqo.Client', return_value=mock_marqo_instance):
            await client.search("test", limit=5, filters=filters)
            
            mock_index.search.assert_called_once_with(
                q="test", limit=5, filter_string="type:TV AND year:2023"
            )

    @pytest.mark.asyncio
    async def test_get_similar_anime_success(self, client: QdrantClient, mock_marqo_instance):
        """Test successful similar anime retrieval."""
        mock_index = MagicMock()
        mock_marqo_instance.index.return_value = mock_index
        
        # Mock get document by ID
        mock_index.get_document.return_value = {
            "anime_id": "target123",
            "title": "Target Anime",
            "embedding_text": "Target anime description"
        }
        
        # Mock search for similar
        mock_index.search.return_value = {
            "hits": [
                {
                    "anime_id": "similar123",
                    "title": "Similar Anime",
                    "_score": 0.85
                }
            ],
            "processingTimeMs": 75
        }
        
        with patch('marqo.Client', return_value=mock_marqo_instance):
            results = await client.get_similar_anime("target123", limit=5)
            
            assert len(results) == 1
            assert results[0]["anime_id"] == "similar123"
            assert results[0]["_score"] == 0.85

    @pytest.mark.asyncio
    async def test_get_similar_anime_document_not_found(self, client: QdrantClient, mock_marqo_instance):
        """Test similar anime when target document not found."""
        mock_index = MagicMock()
        mock_marqo_instance.index.return_value = mock_index
        mock_index.get_document.side_effect = Exception("Document not found")
        
        with patch('marqo.Client', return_value=mock_marqo_instance):
            results = await client.get_similar_anime("nonexistent123", limit=5)
            
            assert results == []

    @pytest.mark.asyncio
    async def test_get_stats_success(self, client: QdrantClient, mock_marqo_instance):
        """Test successful stats retrieval."""
        mock_index = MagicMock()
        mock_marqo_instance.index.return_value = mock_index
        mock_index.get_stats.return_value = {
            "numberOfDocuments": 38894,
            "numberOfVectors": 38894,
            "backend": {
                "memoryUsage": "2.1GB",
                "storageUsage": "5.4GB"
            }
        }
        
        with patch('marqo.Client', return_value=mock_marqo_instance):
            stats = await client.get_stats()
            
            assert stats.total_anime == 38894
            assert stats.indexed_anime == 38894
            assert stats.index_health == "green"
            assert isinstance(stats.last_updated, datetime)

    @pytest.mark.asyncio
    async def test_get_stats_partial_index(self, client: QdrantClient, mock_marqo_instance):
        """Test stats with partial indexing."""
        mock_index = MagicMock()
        mock_marqo_instance.index.return_value = mock_index
        mock_index.get_stats.return_value = {
            "numberOfDocuments": 17931,
            "numberOfVectors": 17931
        }
        
        with patch('marqo.Client', return_value=mock_marqo_instance):
            stats = await client.get_stats()
            
            assert stats.total_anime == 17931  # Only what's indexed
            assert stats.indexed_anime == 17931
            assert stats.index_health == "yellow"  # Partial index

    @pytest.mark.asyncio
    async def test_get_stats_exception(self, client: QdrantClient, mock_marqo_instance):
        """Test stats retrieval with exception."""
        mock_index = MagicMock()
        mock_marqo_instance.index.return_value = mock_index
        mock_index.get_stats.side_effect = Exception("Stats unavailable")
        
        with patch('marqo.Client', return_value=mock_marqo_instance):
            stats = await client.get_stats()
            
            assert stats.total_anime == 0
            assert stats.indexed_anime == 0
            assert stats.index_health == "red"

    @pytest.mark.asyncio
    async def test_delete_documents_success(self, client: QdrantClient, mock_marqo_instance):
        """Test successful document deletion."""
        mock_index = MagicMock()
        mock_marqo_instance.index.return_value = mock_index
        mock_index.delete_documents.return_value = {
            "errors": [],
            "items": [
                {"_id": "doc1", "status": 200},
                {"_id": "doc2", "status": 200}
            ]
        }
        
        with patch('marqo.Client', return_value=mock_marqo_instance):
            result = await client.delete_documents(["doc1", "doc2"])
            
            assert result is True
            mock_index.delete_documents.assert_called_once_with(
                ids=["doc1", "doc2"]
            )

    @pytest.mark.asyncio
    async def test_delete_documents_with_errors(self, client: QdrantClient, mock_marqo_instance):
        """Test document deletion with errors."""
        mock_index = MagicMock()
        mock_marqo_instance.index.return_value = mock_index
        mock_index.delete_documents.return_value = {
            "errors": [{"error": "Document not found", "id": "doc1"}],
            "items": [{"_id": "doc2", "status": 200}]
        }
        
        with patch('marqo.Client', return_value=mock_marqo_instance):
            result = await client.delete_documents(["doc1", "doc2"])
            
            assert result is False

    @pytest.mark.asyncio
    async def test_delete_documents_empty_list(self, client: QdrantClient):
        """Test deleting empty document list."""
        result = await client.delete_documents([])
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_documents_exception(self, client: QdrantClient, mock_marqo_instance):
        """Test document deletion with exception."""
        mock_index = MagicMock()
        mock_marqo_instance.index.return_value = mock_index
        mock_index.delete_documents.side_effect = Exception("Deletion failed")
        
        with patch('marqo.Client', return_value=mock_marqo_instance):
            result = await client.delete_documents(["doc1"])
            
            assert result is False

    def test_client_initialization(self):
        """Test client initialization with different parameters."""
        # Default initialization
        client1 = QdrantClient()
        assert client1.url == "http://localhost:8882"
        assert client1.index_name == "anime_database"
        
        # Custom initialization
        client2 = QdrantClient(url="http://custom:9999", index_name="custom_index")
        assert client2.url == "http://custom:9999"
        assert client2.index_name == "custom_index"

    def test_build_filter_string(self, client: QdrantClient):
        """Test filter string building."""
        # Single filter
        filters = {"type": "TV"}
        filter_string = client._build_filter_string(filters)
        assert filter_string == "type:TV"
        
        # Multiple filters
        filters = {"type": "TV", "year": 2023, "status": "FINISHED"}
        filter_string = client._build_filter_string(filters)
        # Should combine with AND
        assert "type:TV" in filter_string
        assert "year:2023" in filter_string
        assert "status:FINISHED" in filter_string
        assert " AND " in filter_string
        
        # Empty filters
        assert client._build_filter_string({}) == ""
        assert client._build_filter_string(None) == ""

    def test_build_filter_string_special_values(self, client: QdrantClient):
        """Test filter string with special values."""
        # String values with spaces
        filters = {"studio": "Studio Ghibli"}
        filter_string = client._build_filter_string(filters)
        assert 'studio:"Studio Ghibli"' in filter_string
        
        # Numeric values
        filters = {"episodes": 12, "year": 2023}
        filter_string = client._build_filter_string(filters)
        assert "episodes:12" in filter_string
        assert "year:2023" in filter_string
        
        # Boolean values
        filters = {"completed": True}
        filter_string = client._build_filter_string(filters)
        assert "completed:true" in filter_string