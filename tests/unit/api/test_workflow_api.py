"""Tests for workflow API endpoints with enhanced SearchIntent parameters.

This module tests the enhanced workflow API that allows direct specification
of SearchIntent parameters alongside natural language queries.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app


class TestWorkflowAPIEnhancements:
    """Test enhanced workflow API with SearchIntent parameters."""

    @pytest.fixture
    def client(self):
        """Create test client for workflow API."""
        return TestClient(app)

    @pytest.fixture
    def mock_workflow_engine(self):
        """Mock workflow engine for testing."""
        engine = AsyncMock()
        engine.process_conversation.return_value = {
            "session_id": "test_session",
            "messages": ["I'll search for anime based on your parameters"],
            "workflow_steps": [
                {
                    "step_type": "search",
                    "tool_name": "search_anime",
                    "result": {"found": 5},
                    "confidence": 0.95,
                }
            ],
            "current_context": None,
            "user_preferences": None,
        }
        engine.get_workflow_info.return_value = {
            "engine_type": "create_react_agent+LangGraph",
            "features": ["AI-powered query understanding"],
            "performance": {"target_response_time": "120ms"},
        }
        return engine

    @pytest.mark.asyncio
    async def test_enhanced_conversation_with_search_intent_parameters(
        self, client, mock_workflow_engine
    ):
        """Test existing conversation endpoint with SearchIntent parameters."""
        with patch(
            "src.api.workflow.get_workflow_engine", return_value=mock_workflow_engine
        ):
            response = client.post(
                "/api/workflow/conversation",
                json={
                    "message": "find mecha anime",
                    "session_id": "test_session",
                    "search_parameters": {
                        "genres": ["Mecha", "Action"],
                        "year_range": [2020, 2023],
                        "limit": 5,
                        "anime_types": ["TV"],
                        "studios": ["Mappa"],
                        "exclusions": ["Horror"],
                        "mood_keywords": ["serious"],
                    },
                },
            )

            assert response.status_code == 200
            data = response.json()

            assert data["session_id"] == "test_session"
            assert len(data["messages"]) >= 1
            assert len(data["workflow_steps"]) >= 1

            # Verify that enhanced parameters were passed to the workflow engine
            mock_workflow_engine.process_conversation.assert_called_once()
            call_args = mock_workflow_engine.process_conversation.call_args
            assert "search_parameters" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_enhanced_conversation_backward_compatibility(
        self, client, mock_workflow_engine
    ):
        """Test that enhanced endpoint maintains backward compatibility."""
        with patch(
            "src.api.workflow.get_workflow_engine", return_value=mock_workflow_engine
        ):
            # Test without search_parameters (should work like regular conversation)
            response = client.post(
                "/api/workflow/conversation",
                json={
                    "message": "find action anime",
                    "session_id": "backward_test",
                },
            )

            assert response.status_code == 200
            data = response.json()

            assert data["session_id"] == "backward_test"
            mock_workflow_engine.process_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhanced_multimodal_with_search_parameters(
        self, client, mock_workflow_engine
    ):
        """Test enhanced multimodal endpoint with SearchIntent parameters."""
        with patch(
            "src.api.workflow.get_workflow_engine", return_value=mock_workflow_engine
        ):
            response = client.post(
                "/api/workflow/multimodal",
                json={
                    "message": "find similar anime",
                    "image_data": "base64_encoded_image_data",
                    "text_weight": 0.6,
                    "session_id": "multimodal_test",
                    "search_parameters": {
                        "genres": ["Action"],
                        "year_range": [2015, 2025],
                        "limit": 8,
                    },
                },
            )

            assert response.status_code == 200
            data = response.json()

            assert data["session_id"] == "multimodal_test"
            mock_workflow_engine.process_multimodal_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_parameters_validation(self, client, mock_workflow_engine):
        """Test validation of SearchIntent parameters."""
        with patch(
            "src.api.workflow.get_workflow_engine", return_value=mock_workflow_engine
        ):
            # Test with invalid year_range format
            response = client.post(
                "/api/workflow/conversation",
                json={
                    "message": "test query",
                    "search_parameters": {
                        "year_range": [2020],  # Should be [start, end]
                        "limit": 0,  # Should be >= 1
                    },
                },
            )

            # Should still work but validate parameters
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_search_parameters_processing(self, client, mock_workflow_engine):
        """Test that search parameters are properly processed."""
        with patch(
            "src.api.workflow.get_workflow_engine", return_value=mock_workflow_engine
        ):
            test_parameters = {
                "genres": ["Action", "Drama"],
                "year_range": [2020, 2023],
                "anime_types": ["TV", "Movie"],
                "studios": ["Mappa", "Studio Ghibli"],
                "exclusions": ["Horror", "Ecchi"],
                "mood_keywords": ["dark", "serious"],
                "limit": 10,
            }

            response = client.post(
                "/api/workflow/conversation",
                json={
                    "message": "find anime matching my criteria",
                    "search_parameters": test_parameters,
                },
            )

            assert response.status_code == 200

            # Verify parameters were passed through
            call_args = mock_workflow_engine.process_conversation.call_args
            passed_params = call_args.kwargs.get("search_parameters", {})

            # Should preserve all the parameters
            assert passed_params["genres"] == ["Action", "Drama"]
            assert passed_params["year_range"] == [2020, 2023]
            assert passed_params["limit"] == 10

    @pytest.mark.asyncio
    async def test_ai_powered_parameter_extraction_override(
        self, client, mock_workflow_engine
    ):
        """Test that explicit parameters can override AI extraction."""
        with patch(
            "src.api.workflow.get_workflow_engine", return_value=mock_workflow_engine
        ):
            response = client.post(
                "/api/workflow/conversation",
                json={
                    "message": "find horror anime from 1990s",  # AI would extract Horror + 1990s
                    "search_parameters": {
                        "genres": ["Comedy"],  # Override: Comedy instead of Horror
                        "year_range": [2020, 2023],  # Override: 2020s instead of 1990s
                        "limit": 3,
                    },
                },
            )

            assert response.status_code == 200

            # Should use explicit parameters, not AI-extracted ones
            call_args = mock_workflow_engine.process_conversation.call_args
            passed_params = call_args.kwargs.get("search_parameters", {})

            assert passed_params["genres"] == ["Comedy"]
            assert passed_params["year_range"] == [2020, 2023]

    @pytest.mark.asyncio
    async def test_empty_search_parameters_ignored(self, client, mock_workflow_engine):
        """Test that empty search parameters are properly handled."""
        with patch(
            "src.api.workflow.get_workflow_engine", return_value=mock_workflow_engine
        ):
            response = client.post(
                "/api/workflow/conversation",
                json={
                    "message": "find anime",
                    "search_parameters": {
                        "genres": [],  # Empty list
                        "year_range": None,  # None value
                        "studios": ["Mappa"],  # Valid value
                        "limit": 5,
                    },
                },
            )

            assert response.status_code == 200

            # Should only pass non-empty parameters
            call_args = mock_workflow_engine.process_conversation.call_args
            passed_params = call_args.kwargs.get("search_parameters", {})

            assert "genres" not in passed_params or not passed_params["genres"]
            assert (
                "year_range" not in passed_params or passed_params["year_range"] is None
            )
            assert passed_params["studios"] == ["Mappa"]

    def test_enhanced_conversation_request_model_validation(self):
        """Test that the enhanced conversation request model validates properly."""
        from src.api.workflow import ConversationRequest, SearchIntentParameters

        # Test valid request
        search_params = SearchIntentParameters(
            genres=["Action"],
            limit=5,
        )
        valid_request = ConversationRequest(
            message="test message",
            search_parameters=search_params,
        )
        assert valid_request.message == "test message"
        assert valid_request.search_parameters.genres == ["Action"]
        assert valid_request.search_parameters.limit == 5

        # Test request without search_parameters (should be optional)
        minimal_request = ConversationRequest(message="test")
        assert minimal_request.message == "test"
        assert minimal_request.search_parameters is None

    def test_search_intent_parameters_model_validation(self):
        """Test SearchIntentParameters model validation."""
        from src.api.workflow import SearchIntentParameters

        # Test valid parameters
        valid_params = SearchIntentParameters(
            genres=["Action", "Drama"],
            year_range=[2020, 2023],
            limit=10,
        )
        assert valid_params.genres == ["Action", "Drama"]
        assert valid_params.year_range == [2020, 2023]
        assert valid_params.limit == 10

        # Test with all optional parameters as None
        minimal_params = SearchIntentParameters()
        assert minimal_params.genres is None
        assert minimal_params.year_range is None
        assert minimal_params.limit == 10  # Default value


class TestWorkflowAPIBackwardCompatibility:
    """Test that existing workflow API endpoints maintain compatibility."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_workflow_engine(self):
        """Mock workflow engine."""
        engine = AsyncMock()
        engine.process_conversation.return_value = {
            "session_id": "compat_test",
            "messages": ["Response message"],
            "workflow_steps": [],
            "current_context": None,
            "user_preferences": None,
        }
        return engine

    @pytest.mark.asyncio
    async def test_existing_conversation_endpoint_unchanged(
        self, client, mock_workflow_engine
    ):
        """Test that existing /conversation endpoint is unchanged."""
        with patch(
            "src.api.workflow.get_workflow_engine", return_value=mock_workflow_engine
        ):
            response = client.post(
                "/api/workflow/conversation",
                json={
                    "message": "find anime",
                    "session_id": "compat_test",
                },
            )

            assert response.status_code == 200
            data = response.json()

            assert data["session_id"] == "compat_test"
            assert "messages" in data
            assert "workflow_steps" in data

    @pytest.mark.asyncio
    async def test_existing_multimodal_endpoint_unchanged(
        self, client, mock_workflow_engine
    ):
        """Test that existing /multimodal endpoint is unchanged."""
        with patch(
            "src.api.workflow.get_workflow_engine", return_value=mock_workflow_engine
        ):
            response = client.post(
                "/api/workflow/multimodal",
                json={
                    "message": "find similar anime",
                    "image_data": "base64_data",
                    "text_weight": 0.7,
                },
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_endpoint_includes_enhancement_info(
        self, client, mock_workflow_engine
    ):
        """Test that health endpoint shows enhancement capabilities."""
        with patch(
            "src.api.workflow.get_workflow_engine", return_value=mock_workflow_engine
        ):
            response = client.get("/api/workflow/health")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "healthy"
            assert "workflow_engine" in data
            assert "enhanced_search_parameters" in data.get("features", [])
