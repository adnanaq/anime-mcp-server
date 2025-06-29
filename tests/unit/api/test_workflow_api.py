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
        from unittest.mock import Mock, AsyncMock
        engine = Mock()
        
        async def dynamic_response(*args, **kwargs):
            # Extract session_id from the call arguments
            session_id = kwargs.get("session_id", "test_session")
            return {
                "session_id": session_id,
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
        
        engine.process_conversation = AsyncMock(side_effect=dynamic_response)
        engine.process_multimodal_conversation = AsyncMock(side_effect=dynamic_response)
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
        async def mock_async_get_engine():
            return mock_workflow_engine
        
        with patch(
            "src.api.workflow.get_workflow_engine", side_effect=mock_async_get_engine
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
        async def mock_async_get_engine():
            return mock_workflow_engine
        
        with patch(
            "src.api.workflow.get_workflow_engine", side_effect=mock_async_get_engine
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
        async def mock_async_get_engine():
            return mock_workflow_engine
        
        with patch(
            "src.api.workflow.get_workflow_engine", side_effect=mock_async_get_engine
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
        async def mock_async_get_engine():
            return mock_workflow_engine
        
        with patch(
            "src.api.workflow.get_workflow_engine", side_effect=mock_async_get_engine
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
        async def mock_async_get_engine():
            return mock_workflow_engine
        
        with patch(
            "src.api.workflow.get_workflow_engine", side_effect=mock_async_get_engine
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
        async def mock_async_get_engine():
            return mock_workflow_engine
        
        with patch(
            "src.api.workflow.get_workflow_engine", side_effect=mock_async_get_engine
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
        async def mock_async_get_engine():
            return mock_workflow_engine
        
        with patch(
            "src.api.workflow.get_workflow_engine", side_effect=mock_async_get_engine
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
        from unittest.mock import Mock, AsyncMock
        engine = Mock()
        
        async def mock_response(*args, **kwargs):
            return {
                "session_id": "compat_test",
                "messages": ["Response message"],
                "workflow_steps": [],
                "current_context": None,
                "user_preferences": None,
            }
        
        engine.process_conversation = AsyncMock(side_effect=mock_response)
        engine.process_multimodal_conversation = AsyncMock(side_effect=mock_response)
        engine.get_workflow_info.return_value = {
            "engine_type": "create_react_agent+LangGraph",
            "features": ["AI-powered query understanding"],
            "performance": {"target_response_time": "120ms"},
        }
        return engine

    @pytest.mark.asyncio
    async def test_existing_conversation_endpoint_unchanged(
        self, client, mock_workflow_engine
    ):
        """Test that existing /conversation endpoint is unchanged."""
        async def mock_async_get_engine():
            return mock_workflow_engine
        
        with patch(
            "src.api.workflow.get_workflow_engine", side_effect=mock_async_get_engine
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
        async def mock_async_get_engine():
            return mock_workflow_engine
        
        with patch(
            "src.api.workflow.get_workflow_engine", side_effect=mock_async_get_engine
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
        async def mock_async_get_engine():
            return mock_workflow_engine
        
        with patch(
            "src.api.workflow.get_workflow_engine", side_effect=mock_async_get_engine
        ):
            response = client.get("/api/workflow/health")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "healthy"
            assert "workflow_engine" in data
            assert "enhanced_search_parameters" in data.get("features", [])


class TestWorkflowAPIErrorHandling:
    """Test error handling and edge cases in workflow API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.mark.asyncio
    async def test_conversation_error_handling(self, client):
        """Test error handling in conversation endpoint."""
        with patch("src.api.workflow.get_workflow_engine") as mock_get_engine:
            # Mock engine to raise exception
            mock_engine = AsyncMock()
            mock_engine.process_conversation.side_effect = Exception("Test error")
            mock_get_engine.return_value = mock_engine

            response = client.post(
                "/api/workflow/conversation",
                json={"message": "test message"},
            )

            assert response.status_code == 500
            assert "Conversation processing error" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_multimodal_conversation_error_handling(self, client):
        """Test error handling in multimodal endpoint."""
        with patch("src.api.workflow.get_workflow_engine") as mock_get_engine:
            # Mock engine to raise exception
            mock_engine = AsyncMock()
            mock_engine.process_multimodal_conversation.side_effect = Exception("Multimodal error")
            mock_get_engine.return_value = mock_engine

            response = client.post(
                "/api/workflow/multimodal",
                json={
                    "message": "test message",
                    "image_data": "base64_data",
                },
            )

            assert response.status_code == 500
            assert "Multimodal conversation processing error" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_smart_conversation_endpoint(self, client):
        """Test smart conversation endpoint functionality."""
        with patch("src.api.workflow.get_workflow_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.process_conversation.return_value = {
                "session_id": "smart_test_session",
                "messages": ["Smart response"],
                "workflow_steps": [{"step_type": "analysis", "result": {"analyzed": True}}],
                "current_context": {"smart": True},
                "user_preferences": {"ai_assistance": True},
            }
            mock_engine.get_conversation_summary.return_value = "Smart conversation summary"
            mock_get_engine.return_value = mock_engine

            response = client.post(
                "/api/workflow/smart-conversation",
                json={
                    "message": "find complex anime recommendations",
                    "session_id": "smart_test_session",
                    "enable_smart_orchestration": True,
                    "max_discovery_depth": 3,
                    "limit": 10,
                },
            )

            assert response.status_code == 200
            data = response.json()
            
            assert data["session_id"] == "smart_test_session"
            assert data["summary"] == "Smart conversation summary"
            assert len(data["messages"]) > 0
            assert len(data["workflow_steps"]) > 0

            # Verify smart conversation was processed
            mock_engine.process_conversation.assert_called_once()
            mock_engine.get_conversation_summary.assert_called_once_with("smart_test_session")

    @pytest.mark.asyncio
    async def test_smart_conversation_error_handling(self, client):
        """Test error handling in smart conversation endpoint."""
        with patch("src.api.workflow.get_workflow_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.process_conversation.side_effect = Exception("Smart processing error")
            mock_get_engine.return_value = mock_engine

            response = client.post(
                "/api/workflow/smart-conversation",
                json={"message": "test message"},
            )

            assert response.status_code == 500
            assert "Smart conversation processing error" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_conversation_history_endpoint(self, client):
        """Test conversation history retrieval."""
        with patch("src.api.workflow.get_workflow_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.get_conversation_summary.return_value = "Conversation summary"
            mock_get_engine.return_value = mock_engine

            response = client.get("/api/workflow/conversation/test_session_123")

            assert response.status_code == 200
            data = response.json()
            
            assert data["session_id"] == "test_session_123"
            assert data["summary"] == "Conversation summary"
            assert isinstance(data["messages"], list)
            assert isinstance(data["workflow_steps"], list)

            mock_engine.get_conversation_summary.assert_called_once_with("test_session_123")

    @pytest.mark.asyncio
    async def test_conversation_history_error_handling(self, client):
        """Test error handling in conversation history endpoint."""
        with patch("src.api.workflow.get_workflow_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.get_conversation_summary.side_effect = Exception("History error")
            mock_get_engine.return_value = mock_engine

            response = client.get("/api/workflow/conversation/error_session")

            assert response.status_code == 500
            assert "Error retrieving conversation" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_delete_conversation_endpoint(self, client):
        """Test conversation deletion endpoint."""
        response = client.delete("/api/workflow/conversation/delete_session_123")

        assert response.status_code == 200
        data = response.json()
        assert "delete request acknowledged" in data["message"]
        assert "delete_session_123" in data["message"]

    @pytest.mark.asyncio
    async def test_delete_conversation_error_handling(self, client):
        """Test error handling in delete conversation endpoint - covers lines 265-267."""
        # We'll patch the internal delete operation to raise an exception
        with patch("src.api.workflow.logger") as mock_logger:
            # Force an error by making logger.info raise an exception
            mock_logger.info.side_effect = Exception("Delete operation failed")
            
            response = client.delete("/api/workflow/conversation/error_session")
            
            # This should cover lines 265-267: except Exception as e: logger.error... raise HTTPException
            assert response.status_code == 500
            assert "Error deleting conversation" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_workflow_stats_endpoint(self, client):
        """Test workflow statistics endpoint."""
        response = client.get("/api/workflow/stats")

        assert response.status_code == 200
        data = response.json()
        
        assert "total_conversations" in data
        assert "active_sessions" in data
        assert "average_messages_per_session" in data
        assert "total_workflow_steps" in data

    @pytest.mark.asyncio
    async def test_workflow_stats_error_handling(self, client):
        """Test error handling in workflow stats endpoint."""
        # This test covers the exception path, though it's unlikely to fail in practice
        with patch("src.api.workflow.ConversationStats") as mock_stats:
            mock_stats.side_effect = Exception("Stats error")
            
            response = client.get("/api/workflow/stats")
            
            assert response.status_code == 500
            assert "Error retrieving stats" in response.json()["detail"]


class TestWorkflowEngineInitialization:
    """Test workflow engine initialization and caching."""

    @pytest.mark.asyncio
    async def test_workflow_engine_initialization(self):
        """Test workflow engine initialization with MCP tools."""
        with patch("src.api.workflow.get_all_mcp_tools") as mock_get_tools, \
             patch("src.api.workflow.create_react_agent_workflow_engine") as mock_create_engine:
            
            # Mock MCP tools discovery
            mock_tools = [{"name": "search_anime"}, {"name": "get_anime_details"}]
            mock_get_tools.return_value = mock_tools
            
            # Mock engine creation
            mock_engine = AsyncMock()
            mock_create_engine.return_value = mock_engine
            
            # Import the function to test
            from src.api.workflow import get_workflow_engine
            
            # Reset global engine state
            import src.api.workflow
            src.api.workflow._workflow_engine = None
            
            # Test first call - should initialize
            engine = await get_workflow_engine()
            
            assert engine == mock_engine
            mock_get_tools.assert_called_once()
            mock_create_engine.assert_called_once_with(mock_tools)
            
            # Test second call - should use cached engine
            engine2 = await get_workflow_engine()
            
            assert engine2 == mock_engine
            # Should not call initialization again
            assert mock_get_tools.call_count == 1
            assert mock_create_engine.call_count == 1

    def test_workflow_health_endpoint_with_engine_info(self, ):
        """Test workflow health endpoint returns engine information."""
        from fastapi.testclient import TestClient
        from src.main import app
        
        client = TestClient(app)
        
        # Create a mock async function that returns our mock engine
        async def mock_async_get_engine():
            from unittest.mock import Mock
            mock_engine = Mock()
            mock_engine.get_workflow_info.return_value = {
                "engine_type": "create_react_agent+LangGraph",
                "features": ["AI-powered query understanding", "conversation memory"],
                "performance": {"target_response_time": "120ms"},
            }
            return mock_engine
        
        with patch("src.api.workflow.get_workflow_engine", side_effect=mock_async_get_engine):
            response = client.get("/api/workflow/health")

            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "healthy"
            assert data["workflow_engine"] == "create_react_agent+LangGraph"
            assert data["engine_type"] == "create_react_agent+LangGraph"
            assert "enhanced_search_parameters" in data["features"]
            assert "AI-powered query understanding" in data["features"]
            assert data["memory_persistence"] is True
            assert data["checkpointing"] == "MemorySaver"

    def test_workflow_health_error_handling(self):
        """Test workflow health endpoint error handling."""
        from fastapi.testclient import TestClient
        from src.main import app
        
        client = TestClient(app)
        
        # Create a mock async function that raises an exception
        async def mock_async_get_engine_error():
            raise Exception("Engine initialization failed")
        
        with patch("src.api.workflow.get_workflow_engine", side_effect=mock_async_get_engine_error):
            response = client.get("/api/workflow/health")

            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "unhealthy"
            assert "Engine initialization failed" in data["error"]
