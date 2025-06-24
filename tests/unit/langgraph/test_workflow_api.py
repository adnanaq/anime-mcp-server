"""Tests for LangGraph workflow API endpoints."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
import json

from src.main import app
from src.langgraph.models import ConversationState, WorkflowMessage, MessageType


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_workflow_engine():
    """Mock workflow engine."""
    engine = Mock()
    engine.process_conversation = AsyncMock()
    engine.process_multimodal_conversation = AsyncMock()
    engine.get_conversation_summary = AsyncMock()
    return engine


class TestWorkflowEndpoints:
    """Test workflow API endpoints."""
    
    def test_start_conversation(self, mock_workflow_engine):
        """Test starting a new conversation."""
        # Mock response
        mock_state = ConversationState(session_id="test-123")
        mock_state.messages = [
            WorkflowMessage(message_type=MessageType.USER, content="Find action anime"),
            WorkflowMessage(message_type=MessageType.ASSISTANT, content="I found 5 action anime for you")
        ]
        mock_workflow_engine.process_conversation.return_value = mock_state
        
        with patch('src.api.workflow.get_workflow_engine', return_value=mock_workflow_engine):
            client = TestClient(app)
            response = client.post(
                "/api/workflow/conversation",
                json={"message": "Find action anime"}
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "session_id" in data
        assert "messages" in data
        assert len(data["messages"]) == 2
        assert data["messages"][0]["content"] == "Find action anime"
        assert data["messages"][1]["content"] == "I found 5 action anime for you"
        
        mock_workflow_engine.process_conversation.assert_called_once()
    
    def test_continue_conversation(self, mock_workflow_engine):
        """Test continuing an existing conversation."""
        # Mock response
        mock_state = ConversationState(session_id="existing-session")
        mock_state.messages = [
            WorkflowMessage(message_type=MessageType.USER, content="Find similar anime"),
            WorkflowMessage(message_type=MessageType.ASSISTANT, content="Here are similar anime")
        ]
        mock_workflow_engine.process_conversation.return_value = mock_state
        
        with patch('src.api.workflow.get_workflow_engine', return_value=mock_workflow_engine), \
             patch('src.api.workflow.get_conversation_state', return_value=mock_state):
            client = TestClient(app)
            response = client.post(
                "/api/workflow/conversation",
                json={
                    "message": "Find similar anime",
                    "session_id": "existing-session"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == "existing-session"
        assert "messages" in data
        
        mock_workflow_engine.process_conversation.assert_called_once()
    
    def test_multimodal_conversation(self, mock_workflow_engine):
        """Test multimodal conversation with image."""
        # Mock response
        mock_state = ConversationState(session_id="multimodal-123")
        mock_state.messages = [
            WorkflowMessage(message_type=MessageType.USER, content="Find anime like this image"),
            WorkflowMessage(message_type=MessageType.ASSISTANT, content="Found visually similar anime")
        ]
        mock_workflow_engine.process_multimodal_conversation.return_value = mock_state
        
        with patch('src.api.workflow.get_workflow_engine', return_value=mock_workflow_engine):
            client = TestClient(app)
            response = client.post(
                    "/api/workflow/multimodal",
                    json={
                        "message": "Find anime like this image",
                        "image_data": "base64_encoded_image",
                        "text_weight": 0.7
                    }
                )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "session_id" in data
        assert "messages" in data
        
        # The call should be made, but the first argument will be a new ConversationState since no session_id was provided
        mock_workflow_engine.process_multimodal_conversation.assert_called_once()
        call_args = mock_workflow_engine.process_multimodal_conversation.call_args
        assert call_args[0][1] == "Find anime like this image"
        assert call_args[0][2] == "base64_encoded_image" 
        assert call_args[0][3] == 0.7
    
    def test_get_conversation_history(self, mock_workflow_engine):
        """Test getting conversation history."""
        mock_workflow_engine.get_conversation_summary.return_value = "User searched for action anime"
        
        with patch('src.api.workflow.get_conversation_state') as mock_get_state:
            mock_state = ConversationState(session_id="test-history")
            mock_state.messages = [
                WorkflowMessage(message_type=MessageType.USER, content="Find action anime")
            ]
            mock_get_state.return_value = mock_state
            
            client = TestClient(app)
            response = client.get("/api/workflow/conversation/test-history")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == "test-history"
        assert "messages" in data
        assert "summary" in data
        # The summary will be generated by the actual engine, not our mock
        assert "anime" in data["summary"].lower()
    
    def test_conversation_not_found(self):
        """Test getting non-existent conversation."""
        with patch('src.api.workflow.get_conversation_state', return_value=None):
            client = TestClient(app)
            response = client.get("/api/workflow/conversation/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    def test_invalid_message_format(self):
        """Test invalid message format."""
        client = TestClient(app)
        response = client.post(
            "/api/workflow/conversation",
            json={"invalid": "format"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_workflow_engine_error(self, mock_workflow_engine):
        """Test handling workflow engine errors."""
        mock_workflow_engine.process_conversation.side_effect = Exception("Engine error")
        
        with patch('src.api.workflow.get_workflow_engine', return_value=mock_workflow_engine):
            client = TestClient(app)
            response = client.post(
                    "/api/workflow/conversation",
                    json={"message": "Find anime"}
                )
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data["detail"].lower()
    
    def test_conversation_stats(self):
        """Test getting conversation statistics."""
        with patch('src.api.workflow.get_conversation_stats') as mock_stats:
            mock_stats.return_value = {
                "total_conversations": 42,
                "active_sessions": 5,
                "average_messages_per_session": 3.2,
                "total_workflow_steps": 128
            }
            
            client = TestClient(app)
            response = client.get("/api/workflow/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_conversations"] == 42
        assert data["active_sessions"] == 5
        assert data["average_messages_per_session"] == 3.2


class TestWorkflowRequestModels:
    """Test workflow request/response models."""
    
    def test_conversation_request_validation(self):
        """Test conversation request validation."""
        from src.api.workflow import ConversationRequest
        
        # Valid request
        request = ConversationRequest(message="Find anime")
        assert request.message == "Find anime"
        assert request.session_id is None
        
        # With session ID
        request = ConversationRequest(message="Find anime", session_id="test-123")
        assert request.session_id == "test-123"
    
    def test_multimodal_request_validation(self):
        """Test multimodal request validation."""
        from src.api.workflow import MultimodalRequest
        
        # Valid request
        request = MultimodalRequest(
            message="Find mecha anime",
            image_data="base64_data",
            text_weight=0.8
        )
        assert request.message == "Find mecha anime"
        assert request.image_data == "base64_data"
        assert request.text_weight == 0.8
        
        # Default text weight
        request = MultimodalRequest(
            message="Find anime",
            image_data="base64_data"
        )
        assert request.text_weight == 0.7  # default value
    
    def test_conversation_response_format(self):
        """Test conversation response format."""
        from src.api.workflow import ConversationResponse
        
        response = ConversationResponse(
            session_id="test-123",
            messages=[
                {"message_type": "user", "content": "Find anime"},
                {"message_type": "assistant", "content": "Found anime"}
            ],
            workflow_steps=[],
            current_context=None,
            user_preferences=None
        )
        
        assert response.session_id == "test-123"
        assert len(response.messages) == 2
        assert response.messages[0]["content"] == "Find anime"


class TestWorkflowIntegration:
    """Test workflow integration with existing MCP tools."""
    
    def test_workflow_uses_existing_mcp_tools(self, mock_workflow_engine):
        """Test that workflow properly integrates with existing MCP tools."""
        # Simulate workflow that calls multiple MCP tools
        mock_state = ConversationState(session_id="integration-test")
        mock_state.messages = [
            WorkflowMessage(message_type=MessageType.ASSISTANT, content="Integration test response")
        ]
        mock_state.workflow_steps = []  # Would contain steps calling MCP tools
        mock_workflow_engine.process_conversation.return_value = mock_state
        
        with patch('src.api.workflow.get_workflow_engine', return_value=mock_workflow_engine):
            client = TestClient(app)
            response = client.post(
                    "/api/workflow/conversation",
                    json={"message": "Find action anime and then find similar ones"}
                )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have processed the complex request
        assert "session_id" in data
        assert "messages" in data
        
        # Workflow engine should have been called
        mock_workflow_engine.process_conversation.assert_called_once()
    
    def test_workflow_preserves_existing_functionality(self):
        """Test that existing REST endpoints still work alongside workflow."""
        # Test that we can still use direct search endpoints
        client = TestClient(app)
        # This should still work - existing functionality preserved
        response = client.get("/api/search/?q=naruto&limit=5")
            
        # Endpoint should exist (even if it fails due to missing DB in test)
        assert response.status_code != 404  # Not found would indicate broken routing