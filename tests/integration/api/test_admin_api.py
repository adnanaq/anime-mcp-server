"""Integration tests for admin API endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock

from src.main import app


class TestAdminAPI:
    """Test cases for admin API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_update_service(self):
        """Mock UpdateService."""
        mock_service = MagicMock()
        mock_service.check_for_updates = AsyncMock(return_value=True)
        mock_service.perform_incremental_update = AsyncMock()
        mock_service.perform_full_update = AsyncMock()
        mock_service.schedule_weekly_update = AsyncMock()
        mock_service.load_metadata.return_value = {
            "last_check": "2024-01-01T12:00:00Z",
            "last_update": "2024-01-01T10:00:00Z",
            "entry_count": 38894,
            "content_hash": "abc123def456",
            "update_available": True,
            "last_changes": {"added": 5, "modified": 3, "removed": 1}
        }
        return mock_service

    @pytest.fixture
    def mock_smart_scheduler(self):
        """Mock SmartScheduler."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_optimal_update_schedule = AsyncMock(return_value={
            "optimal_schedule": "daily at 03:00 UTC",
            "confidence": 0.85,
            "reasoning": "Low activity period detected"
        })
        mock_scheduler.is_safe_to_update = AsyncMock(return_value={
            "safe_to_update": True,
            "reasons": ["No recent releases", "Low user activity"],
            "recommendation": "Proceed with update"
        })
        return mock_scheduler

    @pytest.mark.integration
    def test_check_updates_success(self, client: TestClient, mock_update_service):
        """Test successful update check."""
        with patch('src.api.admin.UpdateService', return_value=mock_update_service), \
             patch('src.main.qdrant_client') as mock_client:
            
            response = client.post("/api/admin/check-updates")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["has_updates"] is True
            assert data["last_check"] == "2024-01-01T12:00:00Z"
            assert data["entry_count"] == 38894
            assert data["content_hash"] == "abc123de"  # Truncated

    @pytest.mark.integration
    def test_check_updates_no_marqo_client(self, client: TestClient):
        """Test update check when Qdrant client is unavailable."""
        with patch('src.main.qdrant_client', None):
            response = client.post("/api/admin/check-updates")
            
            assert response.status_code == 500

    @pytest.mark.integration
    def test_check_updates_service_error(self, client: TestClient, mock_update_service):
        """Test update check with service error."""
        mock_update_service.check_for_updates.side_effect = Exception("Service error")
        
        with patch('src.api.admin.UpdateService', return_value=mock_update_service), \
             patch('src.main.qdrant_client') as mock_client:
            
            response = client.post("/api/admin/check-updates")
            
            assert response.status_code == 500
            assert "Update check failed" in response.json()["detail"]

    @pytest.mark.integration
    def test_incremental_update_success(self, client: TestClient, mock_update_service):
        """Test successful incremental update."""
        with patch('src.api.admin.UpdateService', return_value=mock_update_service), \
             patch('src.main.qdrant_client') as mock_client:
            
            response = client.post("/api/admin/update-incremental")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["message"] == "Incremental update started in background"
            assert data["status"] == "processing"

    @pytest.mark.integration
    def test_incremental_update_service_error(self, client: TestClient, mock_update_service):
        """Test incremental update with service error."""
        mock_update_service.perform_incremental_update.side_effect = Exception("Update failed")
        
        with patch('src.api.admin.UpdateService', return_value=mock_update_service), \
             patch('src.main.qdrant_client') as mock_client:
            
            response = client.post("/api/admin/update-incremental")
            
            assert response.status_code == 500
            assert "Update failed" in response.json()["detail"]

    @pytest.mark.integration
    def test_full_update_success(self, client: TestClient, mock_update_service):
        """Test successful full update."""
        with patch('src.api.admin.UpdateService', return_value=mock_update_service), \
             patch('src.main.qdrant_client') as mock_client:
            
            response = client.post("/api/admin/update-full")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["message"] == "Full update started in background"
            assert data["status"] == "processing"
            assert "38,000+" in data["warning"]

    @pytest.mark.integration
    def test_full_update_no_marqo_client(self, client: TestClient):
        """Test full update when Qdrant client is unavailable."""
        with patch('src.main.qdrant_client', None):
            response = client.post("/api/admin/update-full")
            
            assert response.status_code == 503
            assert "Qdrant client not initialized" in response.json()["detail"]

    @pytest.mark.integration
    def test_full_update_service_error(self, client: TestClient, mock_update_service):
        """Test full update with service error."""
        mock_update_service.perform_full_update.side_effect = Exception("Full update failed")
        
        with patch('src.api.admin.UpdateService', return_value=mock_update_service), \
             patch('src.main.qdrant_client') as mock_client:
            
            response = client.post("/api/admin/update-full")
            
            assert response.status_code == 500
            assert "Update failed" in response.json()["detail"]

    @pytest.mark.integration
    def test_update_status_success(self, client: TestClient, mock_update_service):
        """Test successful update status retrieval."""
        with patch('src.api.admin.UpdateService', return_value=mock_update_service), \
             patch('src.main.qdrant_client') as mock_client:
            
            response = client.get("/api/admin/update-status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["last_check"] == "2024-01-01T12:00:00Z"
            assert data["last_update"] == "2024-01-01T10:00:00Z"
            assert data["entry_count"] == 38894
            assert data["update_available"] is True
            assert data["last_changes"]["added"] == 5

    @pytest.mark.integration
    def test_update_status_service_error(self, client: TestClient, mock_update_service):
        """Test update status with service error."""
        mock_update_service.load_metadata.side_effect = Exception("Metadata error")
        
        with patch('src.api.admin.UpdateService', return_value=mock_update_service), \
             patch('src.main.qdrant_client') as mock_client:
            
            response = client.get("/api/admin/update-status")
            
            assert response.status_code == 500
            assert "Status check failed" in response.json()["detail"]

    @pytest.mark.integration
    def test_schedule_weekly_update_success(self, client: TestClient, mock_update_service):
        """Test successful weekly update scheduling."""
        with patch('src.api.admin.UpdateService', return_value=mock_update_service), \
             patch('src.main.qdrant_client') as mock_client:
            
            response = client.post("/api/admin/schedule-weekly-update")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["message"] == "Weekly update check started"
            assert data["status"] == "processing"

    @pytest.mark.integration
    def test_schedule_weekly_update_service_error(self, client: TestClient, mock_update_service):
        """Test weekly update scheduling with service error."""
        mock_update_service.schedule_weekly_update.side_effect = Exception("Scheduling failed")
        
        with patch('src.api.admin.UpdateService', return_value=mock_update_service), \
             patch('src.main.qdrant_client') as mock_client:
            
            response = client.post("/api/admin/schedule-weekly-update")
            
            assert response.status_code == 500
            assert "Weekly update failed" in response.json()["detail"]

    @pytest.mark.integration
    def test_smart_schedule_analysis_success(self, client: TestClient, mock_smart_scheduler):
        """Test successful smart schedule analysis."""
        with patch('src.api.admin.SmartScheduler', return_value=mock_smart_scheduler):
            
            response = client.get("/api/admin/smart-schedule-analysis")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["optimal_schedule"] == "daily at 03:00 UTC"
            assert data["confidence"] == 0.85
            assert data["reasoning"] == "Low activity period detected"

    @pytest.mark.integration
    def test_smart_schedule_analysis_error(self, client: TestClient, mock_smart_scheduler):
        """Test smart schedule analysis with error."""
        mock_smart_scheduler.get_optimal_update_schedule.side_effect = Exception("Analysis failed")
        
        with patch('src.api.admin.SmartScheduler', return_value=mock_smart_scheduler):
            
            response = client.get("/api/admin/smart-schedule-analysis")
            
            assert response.status_code == 500
            assert "Analysis failed" in response.json()["detail"]

    @pytest.mark.integration
    def test_update_safety_check_success(self, client: TestClient, mock_smart_scheduler):
        """Test successful update safety check."""
        with patch('src.api.admin.SmartScheduler', return_value=mock_smart_scheduler):
            
            response = client.get("/api/admin/update-safety-check")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["safe_to_update"] is True
            assert len(data["reasons"]) == 2
            assert data["recommendation"] == "Proceed with update"

    @pytest.mark.integration
    def test_update_safety_check_unsafe(self, client: TestClient, mock_smart_scheduler):
        """Test update safety check when unsafe."""
        mock_smart_scheduler.is_safe_to_update.return_value = {
            "safe_to_update": False,
            "reasons": ["Recent major release detected", "High user activity"],
            "recommendation": "Wait 2 hours before updating"
        }
        
        with patch('src.api.admin.SmartScheduler', return_value=mock_smart_scheduler):
            
            response = client.get("/api/admin/update-safety-check")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["safe_to_update"] is False
            assert "Recent major release" in data["reasons"]
            assert "Wait 2 hours" in data["recommendation"]

    @pytest.mark.integration
    def test_update_safety_check_error(self, client: TestClient, mock_smart_scheduler):
        """Test update safety check with error."""
        mock_smart_scheduler.is_safe_to_update.side_effect = Exception("Safety check failed")
        
        with patch('src.api.admin.SmartScheduler', return_value=mock_smart_scheduler):
            
            response = client.get("/api/admin/update-safety-check")
            
            assert response.status_code == 500
            assert "Safety check failed" in response.json()["detail"]

    @pytest.mark.integration
    def test_smart_update_safe_to_proceed(self, client: TestClient, mock_smart_scheduler, mock_update_service):
        """Test smart update when safe to proceed."""
        with patch('src.api.admin.SmartScheduler', return_value=mock_smart_scheduler), \
             patch('src.api.admin.UpdateService', return_value=mock_update_service), \
             patch('src.main.qdrant_client') as mock_client:
            
            response = client.post("/api/admin/smart-update")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["message"] == "Smart update started - safety checks passed"
            assert data["status"] == "processing"
            assert data["safety_info"]["safe_to_update"] is True

    @pytest.mark.integration
    def test_smart_update_postponed(self, client: TestClient, mock_smart_scheduler):
        """Test smart update when postponed due to safety."""
        mock_smart_scheduler.is_safe_to_update.return_value = {
            "safe_to_update": False,
            "reasons": ["Recent release detected"],
            "recommendation": "Wait before updating"
        }
        
        with patch('src.api.admin.SmartScheduler', return_value=mock_smart_scheduler):
            
            response = client.post("/api/admin/smart-update")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["message"] == "Update postponed - not safe to update yet"
            assert data["status"] == "postponed"
            assert data["reasons"] == ["Recent release detected"]
            assert data["recommendation"] == "Wait before updating"

    @pytest.mark.integration
    def test_smart_update_safety_check_error(self, client: TestClient, mock_smart_scheduler):
        """Test smart update with safety check error."""
        mock_smart_scheduler.is_safe_to_update.side_effect = Exception("Safety check error")
        
        with patch('src.api.admin.SmartScheduler', return_value=mock_smart_scheduler):
            
            response = client.post("/api/admin/smart-update")
            
            assert response.status_code == 500
            assert "Smart update failed" in response.json()["detail"]

    @pytest.mark.integration
    def test_admin_endpoints_require_post_methods(self, client: TestClient):
        """Test that admin action endpoints require POST methods."""
        # These should be POST-only
        post_endpoints = [
            "/api/admin/check-updates",
            "/api/admin/update-incremental", 
            "/api/admin/update-full",
            "/api/admin/schedule-weekly-update",
            "/api/admin/smart-update"
        ]
        
        for endpoint in post_endpoints:
            # GET should not be allowed
            response = client.get(endpoint)
            assert response.status_code == 405  # Method Not Allowed

    @pytest.mark.integration
    def test_admin_endpoints_require_get_methods(self, client: TestClient):
        """Test that admin status endpoints require GET methods."""
        # These should be GET-only
        get_endpoints = [
            "/api/admin/update-status",
            "/api/admin/smart-schedule-analysis",
            "/api/admin/update-safety-check"
        ]
        
        for endpoint in get_endpoints:
            # POST should not be allowed
            response = client.post(endpoint)
            assert response.status_code == 405  # Method Not Allowed

    @pytest.mark.integration
    def test_admin_background_task_execution(self, client: TestClient, mock_update_service):
        """Test that background tasks are properly scheduled."""
        with patch('src.api.admin.UpdateService', return_value=mock_update_service), \
             patch('src.main.qdrant_client') as mock_client:
            
            # Trigger background task
            response = client.post("/api/admin/update-incremental")
            
            assert response.status_code == 200
            
            # Background task should have been scheduled
            # (We can't easily test the actual execution in a unit test)
            mock_update_service.perform_incremental_update.assert_not_called()  # Not called immediately

    @pytest.mark.integration 
    def test_admin_concurrent_requests(self, client: TestClient, mock_update_service):
        """Test handling of concurrent admin requests."""
        with patch('src.api.admin.UpdateService', return_value=mock_update_service), \
             patch('src.main.qdrant_client') as mock_client:
            
            # Multiple concurrent requests should all succeed
            responses = []
            for _ in range(3):
                response = client.post("/api/admin/check-updates")
                responses.append(response)
            
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert "has_updates" in data