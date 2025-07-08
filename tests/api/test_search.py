"""Comprehensive tests for search API endpoints - merged unit and integration tests."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api.search import (
    find_visually_similar_anime,
    get_similar_anime,
    search_anime,
    search_anime_by_image,
    search_multimodal_anime,
    semantic_search,
)
from src.main import app
from src.models.anime import SearchRequest


# ============================================================================
# UNIT TESTS - Direct function testing
# ============================================================================

@pytest.fixture
def sample_anime_results():
    """Sample anime search results."""
    return [
        {
            "anime_id": "anime_001",
            "title": "Attack on Titan",
            "synopsis": "Humanity fights against giant titans",
            "type": "TV",
            "episodes": 25,
            "year": 2013,
            "tags": ["action", "drama"],
            "studios": ["Studio Pierrot"],
            "_score": 0.95,
        },
        {
            "anime_id": "anime_002",
            "title": "Death Note",
            "synopsis": "A student finds a supernatural notebook",
            "type": "TV",
            "episodes": 37,
            "year": 2006,
            "tags": ["thriller", "supernatural"],
            "studios": ["Madhouse"],
            "_score": 0.98,
        },
    ]


class TestSemanticSearchUnit:
    """Unit tests for semantic search endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_semantic_search_success(self, sample_anime_results):
        """Test successful semantic search."""
        with patch("src.main.qdrant_client") as mock_client:
            mock_client.search = AsyncMock(return_value=sample_anime_results)

            request = SearchRequest(query="attack titan", limit=10)
            result = await semantic_search(request)

            assert len(result.results) == 2
            assert result.query == "attack titan"
            assert result.total_results == 2
            assert result.results[0].title == "Attack on Titan"
            mock_client.search.assert_called_once_with(query="attack titan", limit=10)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_semantic_search_no_client(self):
        """Test semantic search when client is not available."""
        with patch("src.main.qdrant_client", None):
            request = SearchRequest(query="test", limit=10)
            with pytest.raises(HTTPException) as exc_info:
                await semantic_search(request)
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_semantic_search_client_error(self):
        """Test semantic search with client error."""
        with patch("src.main.qdrant_client") as mock_client:
            mock_client.search = AsyncMock(side_effect=Exception("Search failed"))

            request = SearchRequest(query="test", limit=10)
            with pytest.raises(HTTPException) as exc_info:
                await semantic_search(request)
            assert exc_info.value.status_code == 500


class TestSimpleSearchUnit:
    """Unit tests for simple GET search endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_search_anime_success(self, sample_anime_results):
        """Test successful anime search."""
        with patch("src.main.qdrant_client") as mock_client:
            mock_client.search = AsyncMock(return_value=sample_anime_results)

            result = await search_anime(q="attack titan", limit=10)

            assert len(result.results) == 2
            assert result.query == "attack titan"
            assert result.total_results == 2
            mock_client.search.assert_called_once_with(query="attack titan", limit=10)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_search_anime_no_client(self):
        """Test search anime when client is not available."""
        with patch("src.main.qdrant_client", None):
            with pytest.raises(HTTPException) as exc_info:
                await search_anime(q="test", limit=10)
            assert exc_info.value.status_code == 500


class TestSimilarAnimeUnit:
    """Unit tests for similar anime endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_similar_anime_success(self, sample_anime_results):
        """Test successful similar anime search."""
        with patch("src.main.qdrant_client") as mock_client:
            mock_client.get_similar_anime = AsyncMock(
                return_value=sample_anime_results[1:]
            )

            result = await get_similar_anime(anime_id="anime_001", limit=5)

            assert len(result["similar_anime"]) == 1
            assert result["anime_id"] == "anime_001"
            assert result["count"] == 1
            mock_client.get_similar_anime.assert_called_once_with("anime_001", 5)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_similar_anime_no_client(self):
        """Test similar anime when client is not available."""
        with patch("src.main.qdrant_client", None):
            with pytest.raises(HTTPException) as exc_info:
                await get_similar_anime(anime_id="test", limit=5)
            assert exc_info.value.status_code == 500


class TestImageSearchUnit:
    """Unit tests for image search endpoints."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_search_anime_by_image_success(self, sample_anime_results):
        """Test successful image search."""
        from unittest.mock import AsyncMock, Mock

        # Create mock image file
        image_data = b"fake image data"
        image_file = Mock()
        image_file.filename = "test.jpg"
        image_file.content_type = "image/jpeg"
        image_file.read = AsyncMock(return_value=image_data)

        with patch("src.main.qdrant_client") as mock_client:
            mock_client.search_by_image = AsyncMock(return_value=sample_anime_results)

            result = await search_anime_by_image(image=image_file, limit=10)

            assert len(result["results"]) == 2
            assert result["search_type"] == "image_similarity"
            assert result["total_results"] == 2
            assert result["uploaded_file"] == "test.jpg"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_search_anime_by_image_not_supported(self):
        """Test image search when multi-vector is not supported."""
        from unittest.mock import AsyncMock, Mock

        image_file = Mock()
        image_file.filename = "test.jpg"
        image_file.content_type = "image/jpeg"
        image_file.read = AsyncMock(return_value=b"fake data")

        with patch("src.main.qdrant_client") as mock_client:

            with pytest.raises(HTTPException) as exc_info:
                await search_anime_by_image(image=image_file, limit=10)
            assert exc_info.value.status_code == 501

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_search_anime_by_image_invalid_file(self):
        """Test image search with invalid file type."""
        from unittest.mock import AsyncMock, Mock

        text_file = Mock()
        text_file.filename = "test.txt"
        text_file.content_type = "text/plain"
        text_file.read = AsyncMock(return_value=b"not an image")

        with patch("src.main.qdrant_client") as mock_client:

            with pytest.raises(HTTPException) as exc_info:
                await search_anime_by_image(image=text_file, limit=10)
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_find_visually_similar_anime_success(self, sample_anime_results):
        """Test successful visual similarity search."""
        with patch("src.main.qdrant_client") as mock_client:
            mock_client.find_visually_similar_anime = AsyncMock(
                return_value=sample_anime_results[1:]
            )

            result = await find_visually_similar_anime(anime_id="anime_001", limit=5)

            assert len(result["similar_anime"]) == 1
            assert result["reference_anime_id"] == "anime_001"
            assert result["search_type"] == "visual_similarity"
            mock_client.find_visually_similar_anime.assert_called_once_with(
                anime_id="anime_001", limit=5
            )


class TestMultimodalSearchUnit:
    """Unit tests for multimodal search endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_search_multimodal_anime_with_image(self, sample_anime_results):
        """Test multimodal search with text and image."""
        from unittest.mock import AsyncMock, Mock

        image_file = Mock()
        image_file.filename = "test.jpg"
        image_file.content_type = "image/jpeg"
        image_file.read = AsyncMock(return_value=b"fake image data")

        with patch("src.main.qdrant_client") as mock_client:
            mock_client.search_multimodal = AsyncMock(return_value=sample_anime_results)

            result = await search_multimodal_anime(
                query="action anime", image=image_file, limit=10, text_weight=0.7
            )

            assert len(result["results"]) == 2
            assert result["search_type"] == "multimodal"
            assert result["text_weight"] == 0.7
            assert result["has_image"] is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_search_multimodal_anime_text_only(self, sample_anime_results):
        """Test multimodal search with text only."""
        with patch("src.main.qdrant_client") as mock_client:
            mock_client.search_multimodal = AsyncMock(return_value=sample_anime_results)

            result = await search_multimodal_anime(
                query="action anime", image=None, limit=10, text_weight=0.7
            )

            assert len(result["results"]) == 2
            assert result["search_type"] == "multimodal"
            assert result["has_image"] is False

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_search_multimodal_anime_invalid_image(self):
        """Test multimodal search with invalid image file."""
        from unittest.mock import AsyncMock, Mock

        text_file = Mock()
        text_file.filename = "test.txt"
        text_file.content_type = "text/plain"
        text_file.read = AsyncMock(return_value=b"not an image")

        with patch("src.main.qdrant_client") as mock_client:
            with pytest.raises(HTTPException) as exc_info:
                await search_multimodal_anime(
                    query="action anime", image=text_file, limit=10
                )
            assert exc_info.value.status_code == 400


class TestErrorHandlingUnit:
    """Unit tests for comprehensive error handling."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_all_endpoints_handle_client_errors(self, sample_anime_results):
        """Test that all endpoints handle client errors properly."""
        from unittest.mock import AsyncMock, Mock

        image_file = Mock()
        image_file.filename = "test.jpg"
        image_file.content_type = "image/jpeg"
        image_file.read = AsyncMock(return_value=b"fake data")

        with patch("src.main.qdrant_client") as mock_client:
            # Make all client methods raise exceptions
            mock_client.search = AsyncMock(side_effect=Exception("Client error"))
            mock_client.get_similar_anime = AsyncMock(
                side_effect=Exception("Client error")
            )
            mock_client.search_by_image = AsyncMock(
                side_effect=Exception("Client error")
            )
            mock_client.find_visually_similar_anime = AsyncMock(
                side_effect=Exception("Client error")
            )
            mock_client.search_multimodal = AsyncMock(
                side_effect=Exception("Client error")
            )

            # Test all endpoints raise 500 errors on client exceptions
            with pytest.raises(HTTPException) as exc_info:
                await search_anime(q="test", limit=10)
            assert exc_info.value.status_code == 500

            with pytest.raises(HTTPException) as exc_info:
                await get_similar_anime(anime_id="test", limit=5)
            assert exc_info.value.status_code == 500

            with pytest.raises(HTTPException) as exc_info:
                await search_anime_by_image(image=image_file, limit=10)
            assert exc_info.value.status_code == 500

            with pytest.raises(HTTPException) as exc_info:
                await find_visually_similar_anime(anime_id="test", limit=5)
            assert exc_info.value.status_code == 500

            with pytest.raises(HTTPException) as exc_info:
                await search_multimodal_anime(query="test", image=image_file, limit=10)
            assert exc_info.value.status_code == 500


class TestMissingCoverageEdgeCasesUnit:
    """Unit tests for edge cases that were missing from coverage."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_search_by_image_no_client(self):
        """Test search by image when client is None."""
        from unittest.mock import AsyncMock, Mock

        image_file = Mock()
        image_file.filename = "test.jpg"
        image_file.content_type = "image/jpeg"
        image_file.read = AsyncMock(return_value=b"fake data")

        with patch("src.main.qdrant_client", None):
            with pytest.raises(HTTPException) as exc_info:
                await search_anime_by_image(image=image_file, limit=10)
            assert exc_info.value.status_code == 503
            assert "Vector database not available" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_search_by_image_base64_success_path(self, sample_anime_results):
        """Test successful base64 image search."""
        import base64

        from src.api.search import search_anime_by_image_base64

        # Create base64 encoded fake image data
        fake_image_data = b"fake image data"
        base64_data = base64.b64encode(fake_image_data).decode("utf-8")

        with patch("src.main.qdrant_client") as mock_client:
            mock_client.search_by_image = AsyncMock(return_value=sample_anime_results)

            result = await search_anime_by_image_base64(
                image_data=base64_data, limit=10
            )

            assert len(result["results"]) == 2
            assert result["search_type"] == "image_similarity_base64"
            assert result["total_results"] == 2
            mock_client.search_by_image.assert_called_once()


# ============================================================================
# INTEGRATION TESTS - HTTP client testing via FastAPI TestClient
# ============================================================================

class TestSearchAPIIntegration:
    """Integration tests for search API endpoints."""

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
                "animecountdown_id": 777,
            }
        ]

    @pytest.mark.integration
    def test_semantic_search_success(self, client: TestClient, mock_search_results):
        """Test successful semantic search."""
        with patch("src.main.qdrant_client") as mock_client:
            mock_client.search = AsyncMock(return_value=mock_search_results)

            response = client.post(
                "/api/search/semantic", json={"query": "action anime", "limit": 10}
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
    def test_semantic_search_no_qdrant_client(self, client: TestClient):
        """Test semantic search when Qdrant client is unavailable."""
        with patch("src.main.qdrant_client", None):
            response = client.post(
                "/api/search/semantic", json={"query": "test", "limit": 10}
            )

            assert response.status_code == 503
            assert "Vector database not available" in response.json()["detail"]

    @pytest.mark.integration
    def test_semantic_search_invalid_request(self, client: TestClient):
        """Test semantic search with invalid request data."""
        response = client.post("/api/search/semantic", json={"invalid": "data"})

        assert response.status_code == 422  # Validation error

    @pytest.mark.integration
    def test_simple_search_success(self, client: TestClient, mock_search_results):
        """Test successful simple GET search."""
        with patch("src.main.qdrant_client") as mock_client:
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
    def test_simple_search_limit_validation(
        self, client: TestClient, mock_search_results
    ):
        """Test simple search with limit validation."""
        with patch("src.main.qdrant_client") as mock_client:
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
                "similarity_score": 0.85,
            }
        ]

        with patch("src.main.qdrant_client") as mock_client:
            mock_client.get_similar_anime.return_value = mock_similar

            response = client.get("/api/search/similar/abc123def456?limit=10")

            assert response.status_code == 200
            data = response.json()

            assert data["anime_id"] == "abc123def456"
            assert data["count"] == 1
            assert len(data["similar_anime"]) == 1
            assert data["similar_anime"][0]["title"] == "Similar Anime"

    @pytest.mark.integration
    def test_similar_anime_no_qdrant_client(self, client: TestClient):
        """Test similar anime search when Qdrant client is unavailable."""
        with patch("src.main.qdrant_client", None):
            response = client.get("/api/search/similar/abc123def456")

            assert response.status_code == 503
            assert "Vector database not available" in response.json()["detail"]

    @pytest.mark.integration
    def test_search_error_handling(self, client: TestClient):
        """Test search error handling."""
        with patch("src.main.qdrant_client") as mock_client:
            mock_client.search.side_effect = Exception("Database error")

            response = client.post(
                "/api/search/semantic", json={"query": "test", "limit": 10}
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
                "animecountdown_id": None,  # Test null handling
            }
        ]

        with patch("src.main.qdrant_client") as mock_client:
            mock_client.search.return_value = mock_results

            response = client.post(
                "/api/search/semantic", json={"query": "test", "limit": 10}
            )

            assert response.status_code == 200
            result = response.json()["results"][0]

            # Verify types are preserved
            assert isinstance(result["myanimelist_id"], int)
            assert isinstance(result["anilist_id"], int)
            assert isinstance(result["animeplanet_id"], str)
            assert isinstance(result["notify_id"], str)
            assert result["animecountdown_id"] is None

    @pytest.mark.integration
    def test_fastembed_semantic_quality(self, client: TestClient):
        """Test that FastEmbed provides semantically relevant results."""
        # Mock semantically relevant results for anime queries
        mock_results = [
            {
                "anime_id": "mecha123",
                "title": "Gundam Wing",
                "synopsis": "Mecha anime with giant robots",
                "type": "TV",
                "episodes": 49,
                "tags": ["mecha", "action", "robots"],
                "studios": ["Sunrise"],
                "_score": 0.85,
                "year": 1995,
            },
            {
                "anime_id": "mecha456",
                "title": "Evangelion",
                "synopsis": "Psychological mecha series",
                "type": "TV",
                "episodes": 26,
                "tags": ["mecha", "psychological", "drama"],
                "studios": ["Studio Gainax"],
                "_score": 0.82,
                "year": 1995,
            },
        ]

        with patch("src.main.qdrant_client") as mock_client:
            mock_client.search.return_value = mock_results

            # Test semantic query for mecha anime
            response = client.post(
                "/api/search/semantic",
                json={"query": "giant robots fighting mecha", "limit": 5},
            )

            assert response.status_code == 200
            data = response.json()

            # Verify results are semantically relevant to mecha
            assert len(data["results"]) == 2

            for result in data["results"]:
                # Should find mecha-related anime
                assert any(
                    tag in ["mecha", "action", "robots"] for tag in result["tags"]
                )
                assert result["_score"] > 0.8  # High semantic similarity

            # Verify search was called with semantic query
            mock_client.search.assert_called_once_with(
                "giant robots fighting mecha", limit=5, filters=None
            )

    @pytest.mark.integration
    def test_fastembed_embedding_edge_cases(self, client: TestClient):
        """Test FastEmbed handles edge cases in text processing."""
        edge_case_queries = [
            "",  # Empty query
            "   ",  # Whitespace only
            "龍珠",  # Non-English characters
            "action" * 100,  # Very long query
            "special chars: !@#$%^&*()",  # Special characters
        ]

        with patch("src.main.qdrant_client") as mock_client:
            mock_client.search.return_value = []

            for query in edge_case_queries:
                response = client.post(
                    "/api/search/semantic", json={"query": query, "limit": 5}
                )

                # Should handle gracefully without errors
                assert response.status_code == 200
                data = response.json()
                assert "results" in data
                assert data["query"] == query

    @pytest.mark.integration
    def test_fastembed_similarity_consistency(self, client: TestClient):
        """Test that similar anime search provides consistent similarity scores."""
        mock_similar = [
            {
                "anime_id": "similar1",
                "title": "Similar Anime 1",
                "synopsis": "Very similar content",
                "similarity_score": 0.92,
                "tags": ["action", "adventure"],
            },
            {
                "anime_id": "similar2",
                "title": "Similar Anime 2",
                "synopsis": "Somewhat similar content",
                "similarity_score": 0.78,
                "tags": ["action", "drama"],
            },
            {
                "anime_id": "similar3",
                "title": "Similar Anime 3",
                "synopsis": "Less similar content",
                "similarity_score": 0.65,
                "tags": ["romance", "comedy"],
            },
        ]

        with patch("src.main.qdrant_client") as mock_client:
            mock_client.get_similar_anime.return_value = mock_similar

            response = client.get("/api/search/similar/reference123?limit=5")

            assert response.status_code == 200
            data = response.json()

            # Verify similarity scores are in descending order
            scores = [result["similarity_score"] for result in data["results"]]
            assert scores == sorted(scores, reverse=True)

            # Verify all scores are valid similarity values (0-1)
            for score in scores:
                assert 0 <= score <= 1

    @pytest.mark.integration
    def test_fastembed_filter_integration(self, client: TestClient):
        """Test that FastEmbed works correctly with metadata filters."""
        mock_results = [
            {
                "anime_id": "tv_action1",
                "title": "TV Action Anime",
                "type": "TV",
                "year": 2023,
                "tags": ["action", "adventure"],
                "_score": 0.88,
            }
        ]

        with patch("src.main.qdrant_client") as mock_client:
            mock_client.search.return_value = mock_results

            # Test semantic search with filters
            response = client.post(
                "/api/search/semantic",
                json={
                    "query": "action packed adventure",
                    "limit": 10,
                    "filters": {"type": "TV", "year": 2023},
                },
            )

            assert response.status_code == 200
            data = response.json()

            # Verify FastEmbed search was called with filters
            mock_client.search.assert_called_once()
            call_args = mock_client.search.call_args
            assert call_args[1]["filters"] == {"type": "TV", "year": 2023}

            # Verify results match filters
            for result in data["results"]:
                assert result["type"] == "TV"
                assert result["year"] == 2023