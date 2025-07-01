"""Tests for search API endpoints - corrected to match actual API."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from src.api.search import (
    find_visually_similar_anime,
    get_similar_anime,
    search_anime,
    search_anime_by_image,
    search_multimodal_anime,
    semantic_search,
)
from src.models.anime import SearchRequest


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


class TestSemanticSearch:
    """Test semantic search endpoint."""

    @pytest.mark.asyncio
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
    async def test_semantic_search_no_client(self):
        """Test semantic search when client is not available."""
        with patch("src.main.qdrant_client", None):
            request = SearchRequest(query="test", limit=10)
            with pytest.raises(HTTPException) as exc_info:
                await semantic_search(request)
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_semantic_search_client_error(self):
        """Test semantic search with client error."""
        with patch("src.main.qdrant_client") as mock_client:
            mock_client.search = AsyncMock(side_effect=Exception("Search failed"))

            request = SearchRequest(query="test", limit=10)
            with pytest.raises(HTTPException) as exc_info:
                await semantic_search(request)
            assert exc_info.value.status_code == 500


class TestSimpleSearch:
    """Test simple GET search endpoint."""

    @pytest.mark.asyncio
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
    async def test_search_anime_no_client(self):
        """Test search anime when client is not available."""
        with patch("src.main.qdrant_client", None):
            with pytest.raises(HTTPException) as exc_info:
                await search_anime(q="test", limit=10)
            assert exc_info.value.status_code == 500


class TestSimilarAnime:
    """Test similar anime endpoint."""

    @pytest.mark.asyncio
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
    async def test_get_similar_anime_no_client(self):
        """Test similar anime when client is not available."""
        with patch("src.main.qdrant_client", None):
            with pytest.raises(HTTPException) as exc_info:
                await get_similar_anime(anime_id="test", limit=5)
            assert exc_info.value.status_code == 500


class TestImageSearch:
    """Test image search endpoints."""

    @pytest.mark.asyncio
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
            mock_client._supports_multi_vector = True

            result = await search_anime_by_image(image=image_file, limit=10)

            assert len(result["results"]) == 2
            assert result["search_type"] == "image_similarity"
            assert result["total_results"] == 2
            assert result["uploaded_file"] == "test.jpg"

    @pytest.mark.asyncio
    async def test_search_anime_by_image_not_supported(self):
        """Test image search when multi-vector is not supported."""
        from unittest.mock import AsyncMock, Mock

        image_file = Mock()
        image_file.filename = "test.jpg"
        image_file.content_type = "image/jpeg"
        image_file.read = AsyncMock(return_value=b"fake data")

        with patch("src.main.qdrant_client") as mock_client:
            mock_client._supports_multi_vector = False

            with pytest.raises(HTTPException) as exc_info:
                await search_anime_by_image(image=image_file, limit=10)
            assert exc_info.value.status_code == 501

    @pytest.mark.asyncio
    async def test_search_anime_by_image_invalid_file(self):
        """Test image search with invalid file type."""
        from unittest.mock import AsyncMock, Mock

        text_file = Mock()
        text_file.filename = "test.txt"
        text_file.content_type = "text/plain"
        text_file.read = AsyncMock(return_value=b"not an image")

        with patch("src.main.qdrant_client") as mock_client:
            mock_client._supports_multi_vector = True

            with pytest.raises(HTTPException) as exc_info:
                await search_anime_by_image(image=text_file, limit=10)
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_find_visually_similar_anime_success(self, sample_anime_results):
        """Test successful visual similarity search."""
        with patch("src.main.qdrant_client") as mock_client:
            mock_client.find_visually_similar_anime = AsyncMock(
                return_value=sample_anime_results[1:]
            )
            mock_client._supports_multi_vector = True

            result = await find_visually_similar_anime(anime_id="anime_001", limit=5)

            assert len(result["similar_anime"]) == 1
            assert result["reference_anime_id"] == "anime_001"
            assert result["search_type"] == "visual_similarity"
            mock_client.find_visually_similar_anime.assert_called_once_with(
                anime_id="anime_001", limit=5
            )


class TestMultimodalSearch:
    """Test multimodal search endpoint."""

    @pytest.mark.asyncio
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


class TestErrorHandling:
    """Test comprehensive error handling."""

    @pytest.mark.asyncio
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
            mock_client._supports_multi_vector = True

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


class TestMissingCoverageEdgeCases:
    """Test edge cases that were missing from coverage."""

    @pytest.mark.asyncio
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
    async def test_search_by_image_base64_no_client(self):
        """Test search by image base64 when client is None."""
        from src.api.search import search_anime_by_image_base64

        with patch("src.main.qdrant_client", None):
            with pytest.raises(HTTPException) as exc_info:
                await search_anime_by_image_base64(image_data="dGVzdA==", limit=10)
            assert exc_info.value.status_code == 503
            assert "Vector database not available" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_search_by_image_base64_no_multi_vector(self):
        """Test search by image base64 when multi-vector is not supported."""
        from src.api.search import search_anime_by_image_base64

        with patch("src.main.qdrant_client") as mock_client:
            mock_client._supports_multi_vector = False

            with pytest.raises(HTTPException) as exc_info:
                await search_anime_by_image_base64(image_data="dGVzdA==", limit=10)
            assert exc_info.value.status_code == 501
            assert "Image search not enabled" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_search_by_image_base64_empty_data(self):
        """Test search by image base64 with empty image data."""
        from src.api.search import search_anime_by_image_base64

        with patch("src.main.qdrant_client") as mock_client:
            mock_client._supports_multi_vector = True

            # Test with empty string
            with pytest.raises(HTTPException) as exc_info:
                await search_anime_by_image_base64(image_data="", limit=10)
            assert exc_info.value.status_code == 400
            assert "Image data cannot be empty" in str(exc_info.value.detail)

            # Test with whitespace only
            with pytest.raises(HTTPException) as exc_info:
                await search_anime_by_image_base64(image_data="   ", limit=10)
            assert exc_info.value.status_code == 400
            assert "Image data cannot be empty" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_find_visually_similar_no_client(self):
        """Test find visually similar when client is None."""
        with patch("src.main.qdrant_client", None):
            with pytest.raises(HTTPException) as exc_info:
                await find_visually_similar_anime(anime_id="test", limit=5)
            assert exc_info.value.status_code == 500
            assert "Visual similarity search failed" in str(exc_info.value.detail)
            assert "Vector database not available" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_find_visually_similar_no_multi_vector(self):
        """Test find visually similar when multi-vector is not supported."""
        with patch("src.main.qdrant_client") as mock_client:
            mock_client._supports_multi_vector = False

            with pytest.raises(HTTPException) as exc_info:
                await find_visually_similar_anime(anime_id="test", limit=5)
            assert exc_info.value.status_code == 500
            assert "Visual similarity search failed" in str(exc_info.value.detail)
            assert "Visual similarity search not enabled" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_search_multimodal_no_client(self):
        """Test multimodal search when client is None."""
        with patch("src.main.qdrant_client", None):
            with pytest.raises(HTTPException) as exc_info:
                await search_multimodal_anime(query="test", image=None, limit=10)
            assert exc_info.value.status_code == 503
            assert "Vector database not available" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_search_by_image_base64_success_path(self, sample_anime_results):
        """Test successful base64 image search - covers missing lines 188, 192."""
        import base64

        from src.api.search import search_anime_by_image_base64

        # Create base64 encoded fake image data
        fake_image_data = b"fake image data"
        base64_data = base64.b64encode(fake_image_data).decode("utf-8")

        with patch("src.main.qdrant_client") as mock_client:
            mock_client._supports_multi_vector = True
            # This will cover line 188: results = await qdrant_client.search_by_image(
            mock_client.search_by_image = AsyncMock(return_value=sample_anime_results)

            # This call will cover the success path and line 192: return {
            result = await search_anime_by_image_base64(
                image_data=base64_data, limit=10
            )

            assert len(result["results"]) == 2
            assert result["search_type"] == "image_similarity_base64"
            assert result["total_results"] == 2
            mock_client.search_by_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_by_image_base64_generic_exception(self):
        """Test base64 image search generic exception handling - covers lines 201-203."""
        import base64

        from src.api.search import search_anime_by_image_base64

        fake_image_data = b"fake image data"
        base64_data = base64.b64encode(fake_image_data).decode("utf-8")

        with patch("src.main.qdrant_client") as mock_client:
            mock_client._supports_multi_vector = True
            # This will trigger lines 201-203: except Exception as e: logger.error... raise HTTPException
            mock_client.search_by_image = AsyncMock(
                side_effect=Exception("Search failed")
            )

            with pytest.raises(HTTPException) as exc_info:
                await search_anime_by_image_base64(image_data=base64_data, limit=10)

            assert exc_info.value.status_code == 500
            assert "Base64 image search failed" in str(exc_info.value.detail)
