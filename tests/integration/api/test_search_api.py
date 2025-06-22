"""Integration tests for search API endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.main import app


class TestSearchAPI:
    """Test cases for search API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_search_results(self):
        """Mock search results with platform IDs."""
        return [
            {
                "anime_id": "abc123def456",
                "title": "Test Anime",
                "synopsis": "A test anime",
                "type": "TV",
                "episodes": 12,
                "tags": ["Action", "Adventure"],
                "studios": ["Test Studio"],
                "picture": "https://example.com/image.jpg",
                "_score": 0.95,
                "year": 2023,
                "season": "spring",
                
                # Platform IDs
                "myanimelist_id": 12345,
                "anilist_id": 67890,
                "kitsu_id": 111,
                "anidb_id": 222,
                "anisearch_id": 333,
                "simkl_id": 444,
                "livechart_id": 555,
                "animenewsnetwork_id": 666,
                "animeplanet_id": "test-anime-slug",
                "notify_id": "ABC123DEF",
                "animecountdown_id": 777
            }
        ]

    @pytest.mark.integration
    def test_semantic_search_success(self, client: TestClient, mock_search_results):
        """Test successful semantic search."""
        with patch('src.main.qdrant_client') as mock_client:
            mock_client.search.return_value = mock_search_results
            
            response = client.post(
                "/api/search/semantic",
                json={"query": "action anime", "limit": 10}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["query"] == "action anime"
            assert len(data["results"]) == 1
            assert data["total_results"] == 1
            
            result = data["results"][0]
            assert result["anime_id"] == "abc123def456"
            assert result["title"] == "Test Anime"
            
            # Verify all platform IDs are included
            assert result["myanimelist_id"] == 12345
            assert result["anilist_id"] == 67890
            assert result["kitsu_id"] == 111
            assert result["animeplanet_id"] == "test-anime-slug"
            assert result["notify_id"] == "ABC123DEF"

    @pytest.mark.integration
    def test_semantic_search_no_marqo_client(self, client: TestClient):
        """Test semantic search when Qdrant client is unavailable."""
        with patch('src.main.qdrant_client', None):
            response = client.post(
                "/api/search/semantic",
                json={"query": "test", "limit": 10}
            )
            
            assert response.status_code == 503
            assert "Vector database not available" in response.json()["detail"]

    @pytest.mark.integration
    def test_semantic_search_invalid_request(self, client: TestClient):
        """Test semantic search with invalid request data."""
        response = client.post(
            "/api/search/semantic",
            json={"invalid": "data"}
        )
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.integration
    def test_simple_search_success(self, client: TestClient, mock_search_results):
        """Test successful simple GET search."""
        with patch('src.main.qdrant_client') as mock_client:
            mock_client.search.return_value = mock_search_results
            
            response = client.get("/api/search/?q=test&limit=5")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["query"] == "test"
            assert len(data["results"]) == 1
            assert data["results"][0]["title"] == "Test Anime"

    @pytest.mark.integration
    def test_simple_search_missing_query(self, client: TestClient):
        """Test simple search without query parameter."""
        response = client.get("/api/search/")
        
        assert response.status_code == 422  # Missing required parameter

    @pytest.mark.integration
    def test_simple_search_limit_validation(self, client: TestClient, mock_search_results):
        """Test simple search with limit validation."""
        with patch('src.main.qdrant_client') as mock_client:
            mock_client.search.return_value = mock_search_results
            
            # Test valid limit
            response = client.get("/api/search/?q=test&limit=50")
            assert response.status_code == 200
            
            # Test limit too high
            response = client.get("/api/search/?q=test&limit=200")
            assert response.status_code == 422
            
            # Test limit too low
            response = client.get("/api/search/?q=test&limit=0")
            assert response.status_code == 422

    @pytest.mark.integration
    def test_similar_anime_success(self, client: TestClient):
        """Test successful similar anime search."""
        mock_similar = [
            {
                "anime_id": "def456ghi789",
                "title": "Similar Anime",
                "similarity_score": 0.85
            }
        ]
        
        with patch('src.main.qdrant_client') as mock_client:
            mock_client.get_similar_anime.return_value = mock_similar
            
            response = client.get("/api/search/similar/abc123def456?limit=10")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["anime_id"] == "abc123def456"
            assert data["count"] == 1
            assert len(data["similar_anime"]) == 1
            assert data["similar_anime"][0]["title"] == "Similar Anime"

    @pytest.mark.integration
    def test_similar_anime_no_marqo_client(self, client: TestClient):
        """Test similar anime search when Qdrant client is unavailable."""
        with patch('src.main.qdrant_client', None):
            response = client.get("/api/search/similar/abc123def456")
            
            assert response.status_code == 503
            assert "Vector database not available" in response.json()["detail"]

    @pytest.mark.integration
    def test_search_error_handling(self, client: TestClient):
        """Test search error handling."""
        with patch('src.main.qdrant_client') as mock_client:
            mock_client.search.side_effect = Exception("Database error")
            
            response = client.post(
                "/api/search/semantic",
                json={"query": "test", "limit": 10}
            )
            
            assert response.status_code == 500
            assert "Search failed" in response.json()["detail"]

    @pytest.mark.integration
    def test_search_results_platform_id_types(self, client: TestClient):
        """Test that platform IDs have correct types in response."""
        mock_results = [
            {
                "anime_id": "test123",
                "title": "Test Anime",
                "synopsis": "Test",
                "type": "TV",
                "episodes": 12,
                "tags": ["Action"],
                "studios": ["Test Studio"],
                "_score": 0.9,
                
                # Mix of numeric and string IDs
                "myanimelist_id": 12345,
                "anilist_id": 67890,
                "animeplanet_id": "test-slug",
                "notify_id": "ABC123",
                "animecountdown_id": None  # Test null handling
            }
        ]
        
        with patch('src.main.qdrant_client') as mock_client:
            mock_client.search.return_value = mock_results
            
            response = client.post(
                "/api/search/semantic",
                json={"query": "test", "limit": 10}
            )
            
            assert response.status_code == 200
            result = response.json()["results"][0]
            
            # Verify types are preserved
            assert isinstance(result["myanimelist_id"], int)
            assert isinstance(result["anilist_id"], int)
            assert isinstance(result["animeplanet_id"], str)
            assert isinstance(result["notify_id"], str)
            assert result["animecountdown_id"] is None