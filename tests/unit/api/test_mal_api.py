"""Tests for MAL API endpoints."""

from unittest.mock import AsyncMock, patch
import uuid

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.api.external.mal import router


class TestMALAPI:
    """Test cases for MAL API endpoints."""

    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_anime_data(self):
        """Sample anime data for testing."""
        return {
            "mal_id": 21,
            "title": "One Piece",
            "title_english": "One Piece",
            "title_japanese": "ワンピース",
            "synopsis": "Gol D. Roger was known as the Pirate King...",
            "episodes": 1000,
            "status": "Currently Airing",
            "genres": [{"name": "Action"}, {"name": "Adventure"}],
            "score": 8.7,
        }

    @patch("src.api.external.mal._mal_service")
    def test_search_anime_success(self, mock_service, client, sample_anime_data):
        """Test successful anime search endpoint."""
        # Setup
        mock_service.search_anime = AsyncMock(return_value=[sample_anime_data])

        # Execute
        response = client.get("/external/mal/search?q=one piece&limit=10")

        # Verify
        assert response.status_code == 200
        data = response.json()

        assert data["source"] == "mal"
        assert data["query"] == "one piece"
        assert data["limit"] == 10
        assert len(data["results"]) == 1
        assert data["results"][0] == sample_anime_data
        assert data["total_results"] == 1
        # Enhanced endpoint includes correlation_id in headers and enhanced_parameters in body
        assert "X-Correlation-ID" in response.headers
        assert "enhanced_parameters" in data

        # Verify service was called with enhanced signature
        mock_service.search_anime.assert_called_once()
        call_args = mock_service.search_anime.call_args
        assert call_args.kwargs["query"] == "one piece"
        assert call_args.kwargs["limit"] == 10
        assert call_args.kwargs["status"] is None
        assert call_args.kwargs["genres"] is None
        assert "correlation_id" in call_args.kwargs

    @patch("src.api.external.mal._mal_service")
    def test_search_anime_with_filters(self, mock_service, client, sample_anime_data):
        """Test search endpoint with filters."""
        # Setup
        mock_service.search_anime = AsyncMock(return_value=[sample_anime_data])

        # Execute
        response = client.get(
            "/external/mal/search?q=action&limit=20&status=airing&genres=1,2"
        )

        # Verify
        assert response.status_code == 200
        data = response.json()

        assert data["source"] == "mal"
        assert data["query"] == "action"
        assert data["limit"] == 20
        assert data["status"] == "airing"
        assert data["genres"] == [1, 2]
        # Enhanced endpoint includes correlation_id in headers and additional fields in body
        assert "X-Correlation-ID" in response.headers
        assert "enhanced_parameters" in data

        # Verify service was called with enhanced signature
        mock_service.search_anime.assert_called_once()
        call_args = mock_service.search_anime.call_args
        assert call_args.kwargs["query"] == "action"
        assert call_args.kwargs["limit"] == 20
        assert call_args.kwargs["status"] == "airing"
        assert call_args.kwargs["genres"] == [1, 2]
        assert "correlation_id" in call_args.kwargs

    @patch("src.api.external.mal._mal_service")
    def test_search_anime_default_params(self, mock_service, client):
        """Test search endpoint with default parameters."""
        # Setup
        mock_service.search_anime = AsyncMock(return_value=[])

        # Execute
        response = client.get("/external/mal/search?q=test")

        # Verify
        assert response.status_code == 200
        # Verify service was called with enhanced signature
        mock_service.search_anime.assert_called_once()
        call_args = mock_service.search_anime.call_args
        assert call_args.kwargs["query"] == "test"
        assert call_args.kwargs["limit"] == 10
        assert call_args.kwargs["status"] is None
        assert call_args.kwargs["genres"] is None
        assert "correlation_id" in call_args.kwargs

    @patch("src.api.external.mal._mal_service")
    def test_search_anime_service_error(self, mock_service, client):
        """Test search endpoint when service fails."""
        # Setup
        mock_service.search_anime = AsyncMock(side_effect=Exception("Service error"))

        # Execute
        response = client.get("/external/mal/search?q=test")

        # Verify - general service errors return 500, specific network/connection errors return 503
        assert response.status_code == 500
        error_data = response.json()
        # The error structure is nested within detail
        assert "temporarily unavailable" in error_data["detail"]["detail"]
        # Verify comprehensive error context structure
        assert "error_context" in error_data["detail"]
        assert "Service error" in error_data["detail"]["error_context"]["debug_info"]

    # OLD TEST REMOVED - replaced with comprehensive TDD test below

    # OLD TESTS REMOVED - replaced with comprehensive TDD tests below

    # OLD ENDPOINT TESTS REMOVED - replaced with comprehensive TDD tests in TestMALAPIEnhancedErrorHandling

    @patch("src.api.external.mal._mal_service")
    def test_health_check_healthy(self, mock_service, client):
        """Test health check endpoint when service is healthy."""
        # Setup
        health_data = {
            "service": "mal",
            "status": "healthy",
            "circuit_breaker_open": False,
        }
        mock_service.health_check = AsyncMock(return_value=health_data)

        # Execute
        response = client.get("/external/mal/health")

        # Verify
        assert response.status_code == 200
        assert response.json() == health_data

        mock_service.health_check.assert_called_once()

    @patch("src.api.external.mal._mal_service")
    def test_health_check_unhealthy(self, mock_service, client):
        """Test health check endpoint when service is unhealthy."""
        # Setup
        health_data = {
            "service": "mal",
            "status": "unhealthy",
            "error": "Service down",
            "circuit_breaker_open": True,
        }
        mock_service.health_check = AsyncMock(return_value=health_data)

        # Execute
        response = client.get("/external/mal/health")

        # Verify
        assert response.status_code == 200
        assert response.json() == health_data

    def test_search_anime_validation_errors(self, client):
        """Test search endpoint parameter validation."""
        # Query is now optional, so this should work
        # Test missing query parameter - should work (empty query is allowed)
        response = client.get("/external/mal/search")
        assert response.status_code == 200  # Changed: query is now optional

        # Test invalid limit (too high)
        response = client.get("/external/mal/search?q=test&limit=100")
        assert response.status_code == 422

        # Test invalid status
        response = client.get("/external/mal/search?q=test&status=invalid")
        assert response.status_code == 422

    def test_search_anime_invalid_genre_format(self, client):
        """Test search endpoint with invalid genre format."""
        # Test invalid genre format (non-numeric)
        response = client.get("/external/mal/search?q=test&genres=action,comedy")
        assert response.status_code == 422
        error_data = response.json()
        # Check comprehensive error structure (nested within detail)
        assert "error_context" in error_data["detail"]
        assert "Invalid genre format" in error_data["detail"]["error_context"]["debug_info"]

    def test_seasonal_anime_validation_errors(self, client):
        """Test seasonal endpoint parameter validation."""
        # Test invalid year
        response = client.get("/external/mal/seasonal/1900/winter")
        assert response.status_code == 422

        # Test invalid season
        response = client.get("/external/mal/seasonal/2024/invalid")
        assert response.status_code == 422


class TestMALAPIFullParameters:
    """Test cases for MAL API endpoints with full Jikan parameter support."""

    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def full_sample_data(self):
        """Full sample anime data."""
        return {
            "mal_id": 1535,
            "title": "Death Note",
            "score": 8.62,
            "type": "TV",
            "rating": "R - 17+ (violence & profanity)",
            "genres": [{"mal_id": 40, "name": "Psychological"}],
            "studios": [{"mal_id": 11, "name": "Madhouse"}],
            "year": 2006
        }

    @patch("src.api.external.mal._mal_service")
    def test_search_anime_with_type_filter(self, mock_service, client, full_sample_data):
        """Test search endpoint with anime type filter - NOW IMPLEMENTED."""
        mock_service.search_anime = AsyncMock(return_value=[full_sample_data])

        # Execute - this should now work with enhanced endpoint
        response = client.get("/external/mal/search?q=psychological&anime_type=TV")

        # Verify - this should now pass (200) instead of failing (422)
        assert response.status_code == 200
        data = response.json()
        
        # Verify enhanced response structure
        assert data["source"] == "mal"
        assert data["query"] == "psychological"
        assert "X-Correlation-ID" in response.headers
        assert "enhanced_parameters" in data
        assert data["enhanced_parameters"]["anime_type"] == "TV"
        assert len(data["results"]) == 1
        assert data["results"][0] == full_sample_data
        
        # Verify service was called with enhanced parameters
        mock_service.search_anime.assert_called_once()
        call_args = mock_service.search_anime.call_args
        assert call_args.kwargs["query"] == "psychological"
        assert call_args.kwargs["anime_type"] == "TV"
        assert "correlation_id" in call_args.kwargs

    @patch("src.api.external.mal._mal_service")
    def test_search_anime_with_score_filters(self, mock_service, client, full_sample_data):
        """Test search endpoint with score range filters - NOW IMPLEMENTED."""
        mock_service.search_anime = AsyncMock(return_value=[full_sample_data])

        # Execute - this should now work with enhanced endpoint
        response = client.get("/external/mal/search?q=thriller&min_score=8.0&max_score=9.5")

        # Verify - this should now pass (200)
        assert response.status_code == 200
        data = response.json()
        
        # Verify enhanced response structure
        assert data["source"] == "mal"
        assert data["query"] == "thriller"
        assert data["enhanced_parameters"]["min_score"] == 8.0
        assert data["enhanced_parameters"]["max_score"] == 9.5
        
        # Verify service was called with score parameters
        mock_service.search_anime.assert_called_once()
        call_args = mock_service.search_anime.call_args
        assert call_args.kwargs["min_score"] == 8.0
        assert call_args.kwargs["max_score"] == 9.5

    @patch("src.api.external.mal._mal_service")
    def test_search_anime_with_date_range(self, mock_service, client, full_sample_data):
        """Test search endpoint with date range filters - NOW IMPLEMENTED."""
        mock_service.search_anime = AsyncMock(return_value=[full_sample_data])

        # Execute - this should now work with enhanced endpoint
        response = client.get("/external/mal/search?q=mecha&start_date=2010&end_date=2020")

        # Verify - this should now pass (200)
        assert response.status_code == 200
        data = response.json()
        
        # Verify enhanced response structure
        assert data["source"] == "mal"
        assert data["query"] == "mecha"
        assert data["enhanced_parameters"]["start_date"] == "2010"
        assert data["enhanced_parameters"]["end_date"] == "2020"
        
        # Verify service was called with date parameters
        mock_service.search_anime.assert_called_once()
        call_args = mock_service.search_anime.call_args
        assert call_args.kwargs["start_date"] == "2010"
        assert call_args.kwargs["end_date"] == "2020"

    @patch("src.api.external.mal._mal_service")
    def test_search_anime_with_genre_exclusions(self, mock_service, client, full_sample_data):
        """Test search endpoint with genre exclusion filters - NOW IMPLEMENTED."""
        mock_service.search_anime = AsyncMock(return_value=[full_sample_data])

        # Execute - this should now work with enhanced endpoint
        response = client.get("/external/mal/search?q=action&genres=1,2&genres_exclude=14,26")

        # Verify - this should now pass (200)
        assert response.status_code == 200
        data = response.json()
        
        # Verify enhanced response structure
        assert data["source"] == "mal"
        assert data["query"] == "action"
        assert data["genres"] == [1, 2]
        assert data["enhanced_parameters"]["genres_exclude"] == [14, 26]
        
        # Verify service was called with genre parameters
        mock_service.search_anime.assert_called_once()
        call_args = mock_service.search_anime.call_args
        assert call_args.kwargs["genres"] == [1, 2]
        assert call_args.kwargs["genres_exclude"] == [14, 26]

    @patch("src.api.external.mal._mal_service")
    def test_search_anime_with_producer_filter(self, mock_service, client, full_sample_data):
        """Test search endpoint with producer/studio filters - NOW IMPLEMENTED."""
        mock_service.search_anime = AsyncMock(return_value=[full_sample_data])

        # Execute - this should now work with enhanced endpoint
        response = client.get("/external/mal/search?q=psychological&producers=11")

        # Verify - this should now pass (200)
        assert response.status_code == 200
        data = response.json()
        
        # Verify enhanced response structure
        assert data["source"] == "mal"
        assert data["query"] == "psychological"
        assert data["enhanced_parameters"]["producers"] == [11]
        
        # Verify service was called with producers parameter
        mock_service.search_anime.assert_called_once()
        call_args = mock_service.search_anime.call_args
        assert call_args.kwargs["producers"] == [11]

    @patch("src.api.external.mal._mal_service")
    def test_search_anime_with_rating_filter(self, mock_service, client, full_sample_data):
        """Test search endpoint with content rating filter - NOW IMPLEMENTED."""
        mock_service.search_anime = AsyncMock(return_value=[full_sample_data])

        # Execute - this should now work with enhanced endpoint
        response = client.get("/external/mal/search?q=family&rating=PG-13&sfw=true")

        # Verify - this should now pass (200)
        assert response.status_code == 200
        data = response.json()
        
        # Verify enhanced response structure
        assert data["source"] == "mal"
        assert data["query"] == "family"
        assert data["enhanced_parameters"]["rating"] == "PG-13"
        assert data["enhanced_parameters"]["sfw"] == True
        
        # Verify service was called with rating parameters
        mock_service.search_anime.assert_called_once()
        call_args = mock_service.search_anime.call_args
        assert call_args.kwargs["rating"] == "PG-13"
        assert call_args.kwargs["sfw"] == True

    @patch("src.api.external.mal._mal_service")
    def test_search_anime_with_sorting(self, mock_service, client, full_sample_data):
        """Test search endpoint with ordering and sorting - NOW IMPLEMENTED."""
        mock_service.search_anime = AsyncMock(return_value=[full_sample_data])

        # Execute - this should now work with enhanced endpoint
        response = client.get("/external/mal/search?q=action&order_by=score&sort=desc")

        # Verify - this should now pass (200)
        assert response.status_code == 200
        data = response.json()
        
        # Verify enhanced response structure
        assert data["source"] == "mal"
        assert data["query"] == "action"
        assert data["enhanced_parameters"]["order_by"] == "score"
        assert data["enhanced_parameters"]["sort"] == "desc"
        
        # Verify service was called with sorting parameters
        mock_service.search_anime.assert_called_once()
        call_args = mock_service.search_anime.call_args
        assert call_args.kwargs["order_by"] == "score"
        assert call_args.kwargs["sort"] == "desc"

    @patch("src.api.external.mal._mal_service")
    def test_search_anime_with_pagination(self, mock_service, client, full_sample_data):
        """Test search endpoint with pagination support - NOW IMPLEMENTED."""
        mock_service.search_anime = AsyncMock(return_value=[full_sample_data])

        # Execute - this should now work with enhanced endpoint
        response = client.get("/external/mal/search?q=romance&page=2&limit=25")

        # Verify - this should now pass (200)
        assert response.status_code == 200
        data = response.json()
        
        # Verify enhanced response structure
        assert data["source"] == "mal"
        assert data["query"] == "romance"
        assert data["limit"] == 25
        assert data["enhanced_parameters"]["page"] == 2
        
        # Verify service was called with pagination parameters
        mock_service.search_anime.assert_called_once()
        call_args = mock_service.search_anime.call_args
        assert call_args.kwargs["page"] == 2
        assert call_args.kwargs["limit"] == 25

    @patch("src.api.external.mal._mal_service")
    def test_search_anime_with_letter_filter(self, mock_service, client, full_sample_data):
        """Test search endpoint with starting letter filter - NOW IMPLEMENTED."""
        mock_service.search_anime = AsyncMock(return_value=[full_sample_data])

        # Execute - this should now work with enhanced endpoint
        response = client.get("/external/mal/search?letter=D")

        # Verify - this should now pass (200)
        assert response.status_code == 200
        data = response.json()
        
        # Verify enhanced response structure
        assert data["source"] == "mal"
        assert data["enhanced_parameters"]["letter"] == "D"
        
        # Verify service was called with letter parameter
        mock_service.search_anime.assert_called_once()
        call_args = mock_service.search_anime.call_args
        assert call_args.kwargs["letter"] == "D"

    @patch("src.api.external.mal._mal_service")
    def test_search_anime_complex_query(self, mock_service, client, full_sample_data):
        """Test search endpoint with complex parameter combination - NOW IMPLEMENTED."""
        mock_service.search_anime = AsyncMock(return_value=[full_sample_data])

        # Execute - this should now work with enhanced endpoint
        response = client.get(
            "/external/mal/search?"
            "q=psychological%20thriller&"
            "anime_type=TV&"
            "min_score=8.0&"
            "genres=40,41&"
            "genres_exclude=1,14&"
            "producers=11&"
            "start_date=2005&"
            "end_date=2015&"
            "rating=R&"
            "order_by=score&"
            "sort=desc&"
            "limit=10"
        )

        # Verify - this should now pass (200)
        assert response.status_code == 200
        data = response.json()
        
        # Verify enhanced response structure
        assert data["source"] == "mal"
        assert data["query"] == "psychological thriller"
        assert data["limit"] == 10
        enhanced_params = data["enhanced_parameters"]
        assert enhanced_params["anime_type"] == "TV"
        assert enhanced_params["min_score"] == 8.0
        assert enhanced_params["genres_exclude"] == [1, 14]
        assert enhanced_params["producers"] == [11]
        assert enhanced_params["rating"] == "R"
        assert enhanced_params["order_by"] == "score"
        assert enhanced_params["sort"] == "desc"
        
        # Verify service was called with all parameters
        mock_service.search_anime.assert_called_once()
        call_args = mock_service.search_anime.call_args
        assert call_args.kwargs["query"] == "psychological thriller"
        assert call_args.kwargs["anime_type"] == "TV"
        assert call_args.kwargs["min_score"] == 8.0

    @patch("src.api.external.mal._mal_service")
    def test_search_anime_correlation_id_header_missing(self, mock_service, client, full_sample_data):
        """Test search endpoint without correlation ID header - AUTO-GENERATION."""
        mock_service.search_anime = AsyncMock(return_value=[full_sample_data])
        
        # Execute - request without correlation ID header
        response = client.get("/external/mal/search?q=test")

        # Verify - response should auto-generate correlation ID
        assert response.status_code == 200
        data = response.json()
        
        # Should auto-generate correlation ID in headers only
        assert "X-Correlation-ID" in response.headers
        auto_generated_id = response.headers["X-Correlation-ID"]
        assert auto_generated_id.startswith("mal-api-")
        
        # Verify service was called with auto-generated correlation ID
        mock_service.search_anime.assert_called_once()
        call_args = mock_service.search_anime.call_args
        assert call_args.kwargs["correlation_id"] == auto_generated_id

    @patch("src.api.external.mal._mal_service")
    def test_search_anime_with_correlation_id_header(self, mock_service, client, full_sample_data):
        """Test search endpoint with provided correlation ID header - PROPAGATION."""
        mock_service.search_anime = AsyncMock(return_value=[full_sample_data])
        
        correlation_id = str(uuid.uuid4())
        headers = {"X-Correlation-ID": correlation_id}
        
        # Execute - request with correlation ID header
        response = client.get("/external/mal/search?q=test", headers=headers)

        # Verify - response should echo back the provided correlation ID
        assert response.status_code == 200
        data = response.json()
        
        # Should use provided correlation ID in headers only
        assert response.headers["X-Correlation-ID"] == correlation_id
        
        # Verify service was called with provided correlation ID
        mock_service.search_anime.assert_called_once()
        call_args = mock_service.search_anime.call_args
        assert call_args.kwargs["correlation_id"] == correlation_id

    @patch("src.api.external.mal._mal_service")
    def test_search_anime_correlation_id_propagation(self, mock_service, client, full_sample_data):
        """Test that correlation ID is properly propagated to service layer - IMPLEMENTED."""
        
        correlation_id = str(uuid.uuid4())
        headers = {"X-Correlation-ID": correlation_id}
        
        mock_service.search_anime = AsyncMock(return_value=[full_sample_data])

        # Execute - this should now work with enhanced endpoint
        response = client.get("/external/mal/search?q=test&anime_type=TV&min_score=8.0", headers=headers)

        # Verify - this should now pass with proper correlation ID propagation
        assert response.status_code == 200
        data = response.json()
        
        # Verify correlation ID propagation in headers only
        assert response.headers["X-Correlation-ID"] == correlation_id
        
        # Verify service was called with all parameters including correlation ID
        mock_service.search_anime.assert_called_once()
        call_args = mock_service.search_anime.call_args
        assert call_args.kwargs["query"] == "test"
        assert call_args.kwargs["anime_type"] == "TV"
        assert call_args.kwargs["min_score"] == 8.0
        assert call_args.kwargs["correlation_id"] == correlation_id

    def test_search_anime_parameter_validation_edge_cases(self, client):
        """Test search endpoint parameter validation edge cases - NOW IMPLEMENTED."""
        
        # Test invalid anime_type
        response = client.get("/external/mal/search?q=test&anime_type=INVALID")
        assert response.status_code == 422
        assert "String should match pattern" in response.json()["detail"][0]["msg"]
        
        # Test invalid score range (min > max)
        response = client.get("/external/mal/search?q=test&min_score=9.0&max_score=8.0")
        assert response.status_code == 422
        error_data = response.json()
        # This goes through our comprehensive error handling (nested structure)
        assert "min_score must be less than or equal to max_score" in error_data["detail"]["error_context"]["debug_info"]
        
        # Test invalid rating
        response = client.get("/external/mal/search?q=test&rating=INVALID")
        assert response.status_code == 422
        
        # Test invalid order_by
        response = client.get("/external/mal/search?q=test&order_by=INVALID")
        assert response.status_code == 422
        
        # Test invalid sort
        response = client.get("/external/mal/search?q=test&sort=INVALID")
        assert response.status_code == 422
        
        # Test invalid date format
        response = client.get("/external/mal/search?q=test&start_date=invalid-date")
        assert response.status_code == 422
        
        # Test invalid letter (multiple characters)
        response = client.get("/external/mal/search?letter=ABC")
        assert response.status_code == 422
        
        # Test invalid score bounds
        response = client.get("/external/mal/search?q=test&min_score=11.0")
        assert response.status_code == 422
        
        response = client.get("/external/mal/search?q=test&max_score=-1.0")
        assert response.status_code == 422

    @patch("src.api.external.mal._mal_service")
    def test_successful_search_with_correlation_id(self, mock_service, client, full_sample_data):
        """Test successful search with correlation ID - FULLY IMPLEMENTED."""
        correlation_id = str(uuid.uuid4())
        headers = {"X-Correlation-ID": correlation_id}
        
        # Setup mock to return data when called with correlation_id
        mock_service.search_anime = AsyncMock(return_value=[full_sample_data])
        
        # Execute with complex enhanced parameters + correlation ID
        response = client.get(
            "/external/mal/search?q=psychological&anime_type=TV&min_score=8.0&genres_exclude=1,14&order_by=score&sort=desc", 
            headers=headers
        )
        
        # This should now pass with full implementation
        assert response.status_code == 200
        assert response.headers.get("X-Correlation-ID") == correlation_id
        
        data = response.json()
        assert "X-Correlation-ID" in response.headers
        assert response.headers["X-Correlation-ID"] == correlation_id
        
        # Verify enhanced response structure
        assert data["source"] == "mal"
        assert data["query"] == "psychological"
        assert "enhanced_parameters" in data
        enhanced_params = data["enhanced_parameters"]
        assert enhanced_params["anime_type"] == "TV"
        assert enhanced_params["min_score"] == 8.0
        assert enhanced_params["genres_exclude"] == [1, 14]
        assert enhanced_params["order_by"] == "score"
        assert enhanced_params["sort"] == "desc"
        
        # Verify service call with all enhanced parameters
        mock_service.search_anime.assert_called_once()
        call_args = mock_service.search_anime.call_args
        assert call_args.kwargs["query"] == "psychological"
        assert call_args.kwargs["anime_type"] == "TV"
        assert call_args.kwargs["min_score"] == 8.0
        assert call_args.kwargs["genres_exclude"] == [1, 14]
        assert call_args.kwargs["order_by"] == "score"
        assert call_args.kwargs["sort"] == "desc"
        assert call_args.kwargs["correlation_id"] == correlation_id

    @patch("src.api.external.mal._mal_service")
    def test_search_anime_all_enhanced_parameters(self, mock_service, client, full_sample_data):
        """Test search endpoint with ALL 17 enhanced parameters."""
        mock_service.search_anime = AsyncMock(return_value=[full_sample_data])
        
        # Execute with ALL enhanced parameters
        response = client.get(
            "/external/mal/search?"
            "q=complex%20query&"
            "limit=25&"
            "status=complete&"
            "genres=1,2,3&"
            "anime_type=TV&"
            "score=8.5&"
            "min_score=8.0&"
            "max_score=9.0&"
            "rating=R&"
            "sfw=true&"
            "genres_exclude=14,26&"
            "order_by=score&"
            "sort=desc&"
            "letter=D&"
            "producers=11,29&"
            "start_date=2020-01-01&"
            "end_date=2023-12-31&"
            "page=2&"
            "unapproved=false"
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        # Verify all parameters in response
        enhanced_params = data["enhanced_parameters"]
        assert enhanced_params["anime_type"] == "TV"
        assert enhanced_params["score"] == 8.5
        assert enhanced_params["min_score"] == 8.0
        assert enhanced_params["max_score"] == 9.0
        assert enhanced_params["rating"] == "R"
        assert enhanced_params["sfw"] == True
        assert enhanced_params["genres_exclude"] == [14, 26]
        assert enhanced_params["order_by"] == "score"
        assert enhanced_params["sort"] == "desc"
        assert enhanced_params["letter"] == "D"
        assert enhanced_params["producers"] == [11, 29]
        assert enhanced_params["start_date"] == "2020-01-01"
        assert enhanced_params["end_date"] == "2023-12-31"
        assert enhanced_params["page"] == 2
        assert enhanced_params["unapproved"] == False
        
        # Verify service was called with all parameters
        mock_service.search_anime.assert_called_once()
        call_args = mock_service.search_anime.call_args
        assert call_args.kwargs["query"] == "complex query"
        assert call_args.kwargs["limit"] == 25
        assert call_args.kwargs["status"] == "complete"
        assert call_args.kwargs["genres"] == [1, 2, 3]
        assert call_args.kwargs["anime_type"] == "TV"
        assert call_args.kwargs["score"] == 8.5
        assert call_args.kwargs["min_score"] == 8.0
        assert call_args.kwargs["max_score"] == 9.0
        assert call_args.kwargs["rating"] == "R"
        assert call_args.kwargs["sfw"] == True
        assert call_args.kwargs["genres_exclude"] == [14, 26]
        assert call_args.kwargs["order_by"] == "score"
        assert call_args.kwargs["sort"] == "desc"
        assert call_args.kwargs["letter"] == "D"
        assert call_args.kwargs["producers"] == [11, 29]
        assert call_args.kwargs["start_date"] == "2020-01-01"
        assert call_args.kwargs["end_date"] == "2023-12-31"
        assert call_args.kwargs["page"] == 2
        assert call_args.kwargs["unapproved"] == False

    def test_search_anime_invalid_comma_separated_lists(self, client):
        """Test validation of comma-separated list parameters."""
        # Test invalid genres format
        response = client.get("/external/mal/search?q=test&genres=action,comedy")
        assert response.status_code == 422
        error_data = response.json()
        assert "Invalid genre format" in error_data["detail"]["error_context"]["debug_info"]
        
        # Test invalid genres_exclude format
        response = client.get("/external/mal/search?q=test&genres_exclude=horror,ecchi")
        assert response.status_code == 422
        error_data = response.json()
        assert "Invalid genres_exclude format" in error_data["detail"]["error_context"]["debug_info"]
        
        # Test invalid producers format
        response = client.get("/external/mal/search?q=test&producers=madhouse,toei")
        assert response.status_code == 422
        error_data = response.json()
        assert "Invalid producers format" in error_data["detail"]["error_context"]["debug_info"]



class TestMALAPIEnhancedErrorHandling:
    """Test cases for MAL API with comprehensive error handling infrastructure integration."""

    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_anime_data(self):
        """Sample anime data for testing."""
        return {
            "mal_id": 1535,
            "title": "Death Note",
            "score": 8.62,
            "type": "TV"
        }

    @patch("src.api.external.mal._mal_service")
    def test_enhanced_error_handling_service_failure(self, mock_service, client, sample_anime_data):
        """Test that service failures use comprehensive error handling infrastructure - WILL FAIL INITIALLY."""
        # This test will FAIL because we're not using ErrorContext, CircuitBreaker, etc. yet
        
        # Setup service to raise an error
        mock_service.search_anime = AsyncMock(side_effect=Exception("Database connection failed"))
        
        correlation_id = str(uuid.uuid4())
        headers = {"X-Correlation-ID": correlation_id}
        
        # Execute
        response = client.get("/external/mal/search?q=test", headers=headers)
        
        # Verify comprehensive error handling features that should be implemented:
        assert response.status_code == 503
        assert response.headers["X-Correlation-ID"] == correlation_id
        
        # These assertions will FAIL initially - we expect them to fail
        # After implementation, these should pass:
        error_detail = response.json()["detail"]
        
        # Should have comprehensive error structure with ErrorContext
        assert isinstance(error_detail, dict)
        assert "error_context" in error_detail
        
        error_context = error_detail["error_context"]
        user_message = error_context["user_message"]
        
        # Should have user-friendly message from ErrorContext
        assert "temporarily unavailable" in user_message.lower() or "service unavailable" in user_message.lower()
        
        # Should NOT expose internal details like "Database connection failed" in user message
        assert "Database connection failed" not in user_message
        
        # Should have comprehensive error context fields
        assert error_context["correlation_id"] == correlation_id
        assert error_context["severity"] in ["critical", "error", "warning", "info", "debug"] 
        assert "timestamp" in error_context
        assert "operation" in error_context

    @patch("src.api.external.mal._handle_circuit_breaker_check")
    @patch("src.api.external.mal._mal_service")
    def test_circuit_breaker_integration(self, mock_service, mock_circuit_breaker_check, client):
        """Test circuit breaker integration in API endpoint."""
        
        # Setup circuit breaker check to return "open" response
        circuit_breaker_response = {
            "error_context": {
                "user_message": "The anime search service is temporarily experiencing issues and has been disabled to prevent further problems. Please try again in a few minutes.",
                "debug_info": "Circuit breaker is open due to repeated failures",
                "correlation_id": "test-correlation-123",
                "severity": "warning",
                "timestamp": "2024-01-01T00:00:00Z",
                "breadcrumbs": [],
                "operation": "circuit_breaker_check"
            },
            "detail": "Service temporarily unavailable - circuit breaker is open"
        }
        # Fix: Use AsyncMock for async function and return the response
        mock_circuit_breaker_check.return_value = circuit_breaker_response
        mock_service.search_anime = AsyncMock(side_effect=Exception("Service down"))
        
        # Execute
        response = client.get("/external/mal/search?q=test")
        
        # Should get circuit breaker response (faster failure)
        assert response.status_code == 503
        
        # Response should indicate circuit breaker is open
        error_data = response.json()
        assert "circuit breaker" in str(error_data.get("detail", "")).lower()
        
        # Verify circuit breaker check was called
        mock_circuit_breaker_check.assert_called_once()

    @patch("src.api.external.mal._graceful_degradation")
    @patch("src.api.external.mal._mal_service")
    def test_graceful_degradation_fallback(self, mock_service, mock_graceful_degradation, client, sample_anime_data):
        """Test graceful degradation with fallback responses - WILL FAIL INITIALLY."""
        # This test will FAIL because we don't have graceful degradation integration yet
        
        # Setup service failure that triggers graceful degradation
        mock_service.search_anime = AsyncMock(side_effect=Exception("Connection timeout"))
        
        # Setup graceful degradation to return fallback data
        correlation_id = str(uuid.uuid4())
        fallback_data = {
            "source": "mal_degraded",
            "correlation_id": correlation_id,
            "query": "popular anime",
            "results": [],
            "total_results": 0,
            "degraded": True,
            "fallback_reason": "Service temporarily unavailable"
        }
        mock_graceful_degradation.handle_failure = AsyncMock(return_value=fallback_data)
        
        headers = {"X-Correlation-ID": correlation_id}
        
        # Execute
        response = client.get("/external/mal/search?q=popular anime", headers=headers)
        
        # Should get graceful degradation response instead of hard failure
        assert response.status_code == 200  # Graceful degradation provides fallback
        
        data = response.json()
        assert "results" in data
        assert response.headers["X-Correlation-ID"] == correlation_id
        assert data["degraded"] == True
        assert data["source"] == "mal_degraded"
        assert data["fallback_reason"] == "Service temporarily unavailable"
        assert len(data["results"]) == 0  # Empty results due to fallback
        
        # Should have degradation indicator
        assert "degraded" in data or "fallback" in data.get("source", "")

    @patch("src.api.external.mal._mal_service")
    def test_execution_tracer_integration(self, mock_service, client, sample_anime_data):
        """Test execution tracer integration in API endpoint - WILL FAIL INITIALLY."""
        # This test will FAIL because we don't have execution tracer integration yet
        
        mock_service.search_anime = AsyncMock(return_value=[sample_anime_data])
        
        correlation_id = str(uuid.uuid4())
        headers = {"X-Correlation-ID": correlation_id}
        
        # Execute
        response = client.get("/external/mal/search?q=test&anime_type=TV", headers=headers)
        
        # Should have trace information in response
        assert response.status_code == 200
        data = response.json()
        
        # This will FAIL initially - we expect it to
        assert "trace_info" in data or "execution_trace" in data
        
        # Trace should include timing and steps
        if "trace_info" in data:
            trace = data["trace_info"]
            assert "duration_ms" in trace
            assert "steps" in trace or "trace_id" in trace

    @patch("src.api.external.mal._mal_service")
    def test_correlation_logger_integration(self, mock_service, client, sample_anime_data):
        """Test correlation logger integration via header tracking."""
        # The correlation logger is used internally for structured logging,
        # not for exposing log context in the response body
        
        mock_service.search_anime = AsyncMock(return_value=[sample_anime_data])
        
        correlation_id = str(uuid.uuid4())
        headers = {"X-Correlation-ID": correlation_id}
        
        # Execute
        response = client.get("/external/mal/search?q=test", headers=headers)
        
        # Should have successful response with correlation tracking
        assert response.status_code == 200
        data = response.json()
        
        # Correlation ID should be properly tracked in headers (industry best practice)
        assert "X-Correlation-ID" in response.headers
        assert response.headers["X-Correlation-ID"] == correlation_id
        
        # Response should have proper structure without exposing internal logging
        assert data["source"] == "mal"
        assert data["query"] == "test"
        assert "results" in data
        
        # Verify service was called with correlation ID for internal logging
        mock_service.search_anime.assert_called_once()
        call_args = mock_service.search_anime.call_args
        assert call_args.kwargs["correlation_id"] == correlation_id

    @patch("src.api.external.mal._circuit_breaker")
    @patch("src.api.external.mal._mal_service")
    def test_error_severity_classification(self, mock_service, mock_circuit_breaker, client):
        """Test error severity classification from ErrorContext - WILL FAIL INITIALLY."""
        # This test will FAIL because we don't use ErrorContext yet
        
        # Mock circuit breaker to be closed (not triggered)
        mock_circuit_breaker.is_open.return_value = False
        async def mock_call_with_breaker(func):
            return await func()
        mock_circuit_breaker.call_with_breaker = mock_call_with_breaker
        
        # Test different error types should have different severities
        test_cases = [
            ("Connection timeout", "error", 503),      # connection + timeout -> ERROR, timeout -> 503
            ("Database corruption", "critical", 503),  # critical -> CRITICAL, database -> 503
            ("Invalid parameter", "info", 500),        # invalid -> INFO, no 503 keywords -> 500  
            ("Service unavailable", "error", 503)      # unavailable -> ERROR, unavailable -> 503
        ]
        
        for error_msg, expected_severity, expected_status in test_cases:
            mock_service.search_anime = AsyncMock(side_effect=Exception(error_msg))
            
            response = client.get("/external/mal/search?q=test")
            
            # Should classify error severity properly
            assert response.status_code == expected_status
            
            # This will FAIL initially - we expect it to
            error_data = response.json()
            # Should have severity classification from ErrorContext (nested structure)
            assert "error_context" in error_data["detail"]
            
            error_context = error_data["detail"]["error_context"]
            assert "severity" in error_context
            assert error_context["severity"] == expected_severity  # already lowercase

    @patch("src.api.external.mal._circuit_breaker")
    def test_comprehensive_error_context_structure(self, mock_circuit_breaker, client):
        """Test that error responses have comprehensive ErrorContext structure."""
        # Test comprehensive error handling with a custom validation error that goes through our handler
        
        # Mock circuit breaker to be closed (not triggered)
        mock_circuit_breaker.is_open.return_value = False
        
        # Trigger a custom validation error (genre format) that uses our comprehensive error handling
        response = client.get("/external/mal/search?q=test&genres=invalid,format")  # Invalid genre format
        
        assert response.status_code == 422
        error_data = response.json()
        
        # Should have comprehensive ErrorContext structure (nested)
        assert "error_context" in error_data["detail"]
        
        error_ctx = error_data["detail"]["error_context"]
        assert "user_message" in error_ctx
        assert "debug_info" in error_ctx
        assert "correlation_id" in error_ctx
        assert "severity" in error_ctx
        assert "timestamp" in error_ctx
        assert "breadcrumbs" in error_ctx
        assert "operation" in error_ctx
        
        # Verify specific values for our comprehensive error handling
        assert error_ctx["operation"] == "mal_search_validation"
        assert error_ctx["severity"] in ["critical", "error", "warning", "info", "debug"]
        assert "Invalid genre format" in error_ctx["debug_info"]

    # =================== NEW ENDPOINT TESTS (TDD) ===================
    
    @patch("src.api.external.mal._mal_service")
    def test_get_anime_details_success(self, mock_service, client, sample_anime_data):
        """Test successful anime details endpoint with correlation tracking."""
        # Setup
        mock_service.get_anime_details = AsyncMock(return_value=sample_anime_data)
        
        # Execute
        response = client.get("/external/mal/anime/21")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "mal"
        assert data["anime_id"] == 21
        assert data["data"] == sample_anime_data
        assert "X-Correlation-ID" in response.headers
        
        # Verify service was called with correlation_id
        mock_service.get_anime_details.assert_called_once()
        call_args = mock_service.get_anime_details.call_args
        assert call_args.args[0] == 21  # anime_id
        assert "correlation_id" in call_args.kwargs

    @patch("src.api.external.mal._mal_service")
    def test_get_anime_details_with_correlation_id(self, mock_service, client, sample_anime_data):
        """Test anime details endpoint with provided correlation ID."""
        # Setup
        mock_service.get_anime_details = AsyncMock(return_value=sample_anime_data)
        correlation_id = "test-correlation-123"
        headers = {"X-Correlation-ID": correlation_id}
        
        # Execute
        response = client.get("/external/mal/anime/21", headers=headers)
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        assert response.headers["X-Correlation-ID"] == correlation_id
        
        # Verify service was called with provided correlation_id
        call_args = mock_service.get_anime_details.call_args
        assert call_args.kwargs["correlation_id"] == correlation_id

    @patch("src.api.external.mal._mal_service")
    @patch("src.api.external.mal._circuit_breaker")
    def test_get_anime_details_circuit_breaker_open(self, mock_circuit_breaker, mock_service, client):
        """Test anime details endpoint when circuit breaker is open."""
        # Setup - circuit breaker is open
        mock_circuit_breaker.is_open.return_value = True
        mock_circuit_breaker.failure_count = 5
        mock_circuit_breaker.recovery_timeout = 30
        
        # Execute
        response = client.get("/external/mal/anime/21")
        
        # Verify
        assert response.status_code == 503
        assert "X-Correlation-ID" in response.headers
        
        # Service should not be called when circuit breaker is open
        mock_service.get_anime_details.assert_not_called()

    @patch("src.api.external.mal._mal_service")
    def test_get_anime_details_not_found(self, mock_service, client):
        """Test anime details endpoint when anime not found."""
        # Setup
        mock_service.get_anime_details = AsyncMock(return_value=None)
        
        # Execute
        response = client.get("/external/mal/anime/99999")
        
        # Verify
        assert response.status_code == 404
        error_data = response.json()
        
        # 404 errors go through comprehensive error handling
        assert "error_context" in error_data["detail"]
        error_ctx = error_data["detail"]["error_context"]
        assert "not found" in error_ctx["user_message"].lower()

    @patch("src.api.external.mal._mal_service")
    def test_get_anime_details_service_error(self, mock_service, client):
        """Test anime details endpoint with service error and comprehensive error handling."""
        # Setup
        mock_service.get_anime_details = AsyncMock(side_effect=Exception("Connection timeout"))
        
        # Execute
        response = client.get("/external/mal/anime/21")
        
        # Verify
        assert response.status_code == 503  # timeout should be 503
        error_data = response.json()
        
        # Should have comprehensive error context
        assert "error_context" in error_data["detail"]
        error_ctx = error_data["detail"]["error_context"]
        assert "user_message" in error_ctx
        assert "correlation_id" in error_ctx
        assert "operation" in error_ctx
        assert error_ctx["operation"] == "mal_anime_details"

    @patch("src.api.external.mal._mal_service")
    def test_get_seasonal_anime_success(self, mock_service, client, sample_anime_data):
        """Test successful seasonal anime endpoint with correlation tracking."""
        # Setup
        mock_service.get_seasonal_anime = AsyncMock(return_value=[sample_anime_data])
        
        # Execute
        response = client.get("/external/mal/seasonal/2024/winter")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "mal"
        assert data["year"] == 2024
        assert data["season"] == "winter"
        assert data["results"] == [sample_anime_data]
        assert data["total_results"] == 1
        assert "X-Correlation-ID" in response.headers
        
        # Verify service was called with correlation_id
        mock_service.get_seasonal_anime.assert_called_once()
        call_args = mock_service.get_seasonal_anime.call_args
        assert call_args.args == (2024, "winter")  # year, season
        assert "correlation_id" in call_args.kwargs

    @patch("src.api.external.mal._mal_service")
    def test_get_seasonal_anime_invalid_season(self, mock_service, client):
        """Test seasonal anime endpoint with invalid season parameter."""
        # Execute - invalid season should be caught by path validation
        response = client.get("/external/mal/seasonal/2024/invalid_season")
        
        # Verify
        assert response.status_code == 422
        # Service should not be called for invalid parameters
        mock_service.get_seasonal_anime.assert_not_called()

    @patch("src.api.external.mal._mal_service")
    def test_get_seasonal_anime_service_error(self, mock_service, client):
        """Test seasonal anime endpoint with service error and comprehensive error handling."""
        # Setup
        mock_service.get_seasonal_anime = AsyncMock(side_effect=Exception("Network unavailable"))
        
        # Execute
        response = client.get("/external/mal/seasonal/2024/winter")
        
        # Verify
        assert response.status_code == 503  # network unavailable should be 503
        error_data = response.json()
        
        # Should have comprehensive error context
        assert "error_context" in error_data["detail"]
        error_ctx = error_data["detail"]["error_context"]
        assert "user_message" in error_ctx
        assert "correlation_id" in error_ctx
        assert "operation" in error_ctx
        assert error_ctx["operation"] == "mal_seasonal_anime"

    @patch("src.api.external.mal._mal_service")
    def test_get_current_season_anime_success(self, mock_service, client, sample_anime_data):
        """Test successful current season anime endpoint with correlation tracking."""
        # Setup
        mock_service.get_current_season = AsyncMock(return_value=[sample_anime_data])
        
        # Execute
        response = client.get("/external/mal/current-season")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "mal"
        assert data["type"] == "current_season"
        assert data["results"] == [sample_anime_data]
        assert data["total_results"] == 1
        assert "X-Correlation-ID" in response.headers
        
        # Verify service was called with correlation_id
        mock_service.get_current_season.assert_called_once()
        call_args = mock_service.get_current_season.call_args
        assert "correlation_id" in call_args.kwargs

    @patch("src.api.external.mal._mal_service")
    def test_get_current_season_anime_with_correlation_id(self, mock_service, client, sample_anime_data):
        """Test current season anime endpoint with provided correlation ID."""
        # Setup
        mock_service.get_current_season = AsyncMock(return_value=[sample_anime_data])
        correlation_id = "current-season-123"
        headers = {"X-Correlation-ID": correlation_id}
        
        # Execute
        response = client.get("/external/mal/current-season", headers=headers)
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert response.headers["X-Correlation-ID"] == correlation_id
        
        # Verify service was called with provided correlation_id
        call_args = mock_service.get_current_season.call_args
        assert call_args.kwargs["correlation_id"] == correlation_id

    @patch("src.api.external.mal._mal_service")
    def test_get_current_season_anime_service_error(self, mock_service, client):
        """Test current season anime endpoint with service error and comprehensive error handling."""
        # Setup
        mock_service.get_current_season = AsyncMock(side_effect=Exception("Database connection failed"))
        
        # Execute
        response = client.get("/external/mal/current-season")
        
        # Verify
        assert response.status_code == 503  # database connection should be 503
        error_data = response.json()
        
        # Should have comprehensive error context
        assert "error_context" in error_data["detail"]
        error_ctx = error_data["detail"]["error_context"]
        assert "user_message" in error_ctx
        assert "correlation_id" in error_ctx
        assert "operation" in error_ctx
        assert error_ctx["operation"] == "mal_current_season"

    @patch("src.api.external.mal._mal_service")
    def test_get_anime_statistics_success(self, mock_service, client):
        """Test successful anime statistics endpoint with correlation tracking."""
        # Setup
        stats_data = {
            "watching": 100000,
            "completed": 500000,
            "on_hold": 50000,
            "dropped": 25000,
            "plan_to_watch": 200000
        }
        mock_service.get_anime_statistics = AsyncMock(return_value=stats_data)
        
        # Execute
        response = client.get("/external/mal/anime/21/statistics")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "mal"
        assert data["anime_id"] == 21
        assert data["statistics"] == stats_data
        assert "X-Correlation-ID" in response.headers
        
        # Verify service was called with correlation_id
        mock_service.get_anime_statistics.assert_called_once()
        call_args = mock_service.get_anime_statistics.call_args
        assert call_args.args[0] == 21  # anime_id
        assert "correlation_id" in call_args.kwargs

    @patch("src.api.external.mal._mal_service")
    def test_get_anime_statistics_not_found(self, mock_service, client):
        """Test anime statistics endpoint when statistics not found."""
        # Setup
        mock_service.get_anime_statistics = AsyncMock(return_value=None)
        
        # Execute
        response = client.get("/external/mal/anime/99999/statistics")
        
        # Verify
        assert response.status_code == 404
        error_data = response.json()
        
        # 404 errors go through comprehensive error handling
        assert "error_context" in error_data["detail"]
        error_ctx = error_data["detail"]["error_context"]
        assert "not found" in error_ctx["user_message"].lower()

    @patch("src.api.external.mal._mal_service")
    def test_get_anime_statistics_service_error(self, mock_service, client):
        """Test anime statistics endpoint with service error and comprehensive error handling."""
        # Setup
        mock_service.get_anime_statistics = AsyncMock(side_effect=Exception("Service timeout"))
        
        # Execute
        response = client.get("/external/mal/anime/21/statistics")
        
        # Verify
        assert response.status_code == 503  # timeout should be 503
        error_data = response.json()
        
        # Should have comprehensive error context
        assert "error_context" in error_data["detail"]
        error_ctx = error_data["detail"]["error_context"]
        assert "user_message" in error_ctx
        assert "correlation_id" in error_ctx
        assert "operation" in error_ctx
        assert error_ctx["operation"] == "mal_anime_statistics"

    @patch("src.api.external.mal._mal_service")
    @patch("src.api.external.mal._circuit_breaker")
    def test_all_endpoints_circuit_breaker_protection(self, mock_circuit_breaker, mock_service, client):
        """Test that all enhanced endpoints respect circuit breaker protection."""
        # Setup - circuit breaker is open
        mock_circuit_breaker.is_open.return_value = True
        
        endpoints = [
            "/external/mal/anime/21",
            "/external/mal/seasonal/2024/winter", 
            "/external/mal/current-season",
            "/external/mal/anime/21/statistics"
        ]
        
        for endpoint in endpoints:
            # Execute
            response = client.get(endpoint)
            
            # Verify - all should return 503 when circuit breaker is open
            assert response.status_code == 503, f"Endpoint {endpoint} should return 503 when circuit breaker is open"
            assert "X-Correlation-ID" in response.headers
            
        # Verify no service calls were made
        mock_service.get_anime_details.assert_not_called()
        mock_service.get_seasonal_anime.assert_not_called()
        mock_service.get_current_season.assert_not_called()
        mock_service.get_anime_statistics.assert_not_called()

    # =================== COVERAGE COMPLETION TESTS (TDD) ===================
    
    @patch("src.api.external.mal._mal_service")
    @patch("src.api.external.mal._circuit_breaker")
    def test_rate_limit_error_classification_and_message(self, mock_circuit_breaker, mock_service, client):
        """Test rate limit error gets WARNING severity and appropriate user message - COVERAGE for lines 124, 140."""
        # Setup - circuit breaker closed
        mock_circuit_breaker.is_open.return_value = False
        async def mock_call_with_breaker(func):
            return await func()
        mock_circuit_breaker.call_with_breaker = mock_call_with_breaker
        
        # Test rate limit error
        mock_service.search_anime = AsyncMock(side_effect=Exception("Rate limit exceeded"))
        
        response = client.get("/external/mal/search?q=test")
        
        assert response.status_code == 500  # rate limit doesn't map to 503
        error_data = response.json()
        
        # Should have comprehensive error context with WARNING severity
        assert "error_context" in error_data["detail"]
        error_ctx = error_data["detail"]["error_context"]
        assert error_ctx["severity"] == "warning"  # Line 124 coverage
        assert "too many requests" in error_ctx["user_message"].lower()  # Line 140 coverage
        
    @patch("src.api.external.mal._mal_service")
    @patch("src.api.external.mal._circuit_breaker") 
    def test_timeout_error_user_message(self, mock_circuit_breaker, mock_service, client):
        """Test timeout error gets appropriate user message - COVERAGE for line 142."""
        # Setup - circuit breaker closed
        mock_circuit_breaker.is_open.return_value = False
        async def mock_call_with_breaker(func):
            return await func()
        mock_circuit_breaker.call_with_breaker = mock_call_with_breaker
        
        # Test timeout error
        mock_service.search_anime = AsyncMock(side_effect=Exception("Request timeout"))
        
        response = client.get("/external/mal/search?q=test")
        
        assert response.status_code == 503  # timeout should be 503
        error_data = response.json()
        
        # Should have comprehensive error context with timeout message
        assert "error_context" in error_data["detail"]
        error_ctx = error_data["detail"]["error_context"]
        assert "timed out" in error_ctx["user_message"].lower()  # Line 142 coverage
        
    @patch("src.api.external.mal._mal_service")
    @patch("src.api.external.mal._circuit_breaker")
    def test_critical_severity_user_message(self, mock_circuit_breaker, mock_service, client):
        """Test critical severity error gets appropriate user message - COVERAGE for line 146."""
        # Setup - circuit breaker closed
        mock_circuit_breaker.is_open.return_value = False
        async def mock_call_with_breaker(func):
            return await func()
        mock_circuit_breaker.call_with_breaker = mock_call_with_breaker
        
        # Test critical error - use "critical" keyword without database/connection keywords
        mock_service.search_anime = AsyncMock(side_effect=Exception("Critical system failure occurred"))
        
        response = client.get("/external/mal/search?q=test")
        
        assert response.status_code == 500  # critical error without 503 keywords should be 500
        error_data = response.json()
        
        # Should have comprehensive error context with CRITICAL severity
        assert "error_context" in error_data["detail"]
        error_ctx = error_data["detail"]["error_context"]
        assert error_ctx["severity"] == "critical"  # Should be classified as CRITICAL
        assert "critical service error" in error_ctx["user_message"].lower()  # Line 146 coverage
        assert "contact support" in error_ctx["user_message"].lower()

    # =================== 100% COVERAGE COMPLETION TESTS ===================
    
    @patch("src.api.external.mal._mal_service")
    def test_seasonal_anime_422_http_error_handling(self, mock_service, client):
        """Test seasonal anime 422 HTTP error handling - COVERAGE for lines 821-829."""
        # Setup - mock a 422 validation error from the service layer
        mock_service.get_seasonal_anime = AsyncMock(side_effect=HTTPException(status_code=422, detail="Invalid season parameter"))
        
        # Execute - trigger a 422 error (not circuit breaker related)
        response = client.get("/external/mal/seasonal/2024/winter")
        
        # Verify - should get 422 with comprehensive error context
        assert response.status_code == 422
        error_data = response.json()
        
        # Should have comprehensive error context (lines 821-829 coverage)
        assert "error_context" in error_data["detail"]
        error_ctx = error_data["detail"]["error_context"]
        assert "operation" in error_ctx
        assert error_ctx["operation"] == "mal_seasonal_anime"
        assert "X-Correlation-ID" in response.headers

    @patch("src.api.external.mal._mal_service")
    def test_current_season_non_503_http_error_handling(self, mock_service, client):
        """Test current season non-503 HTTP error handling - COVERAGE for lines 927-935."""
        # Setup - mock a 400 HTTP error from the service layer
        mock_service.get_current_season = AsyncMock(side_effect=HTTPException(status_code=400, detail="Bad request"))
        
        # Execute - trigger a 400 error (not 503)
        response = client.get("/external/mal/current-season")
        
        # Verify - should get 400 with comprehensive error context
        assert response.status_code == 400
        error_data = response.json()
        
        # Should have comprehensive error context (lines 927-935 coverage)
        assert "error_context" in error_data["detail"]
        error_ctx = error_data["detail"]["error_context"]
        assert "operation" in error_ctx
        assert error_ctx["operation"] == "mal_current_season"
        assert "X-Correlation-ID" in response.headers

    @patch("src.api.external.mal._mal_service")
    def test_health_check_exception_handling(self, mock_service, client):
        """Test health check exception handling - COVERAGE for lines 1082-1084."""
        # Setup - mock service health check to raise exception
        mock_service.health_check = AsyncMock(side_effect=Exception("Health check service error"))
        
        # Execute
        response = client.get("/external/mal/health")
        
        # Verify - should return unhealthy status (lines 1082-1084 coverage)
        assert response.status_code == 200  # Health endpoint returns 200 even for errors
        data = response.json()
        
        assert data["service"] == "mal"
        assert data["status"] == "unhealthy" 
        assert "Health check service error" in data["error"]
        assert data["circuit_breaker_open"] is True  # Line 1084 coverage
