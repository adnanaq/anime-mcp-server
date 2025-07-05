"""Tests for admin API endpoints - corrected to match actual API."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException

from src.api.admin import (
    analyze_optimal_schedule,
    check_for_updates,
    check_update_safety,
    get_update_status,
    perform_full_update,
    perform_incremental_update,
    schedule_weekly_update,
    smart_update,
)


class TestUpdateChecks:
    """Test update check endpoints."""

    @pytest.mark.asyncio
    async def test_check_for_updates_success(self):
        """Test successful update check."""
        with patch("src.api.admin.get_qdrant_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            with patch("src.api.admin.UpdateService") as mock_update_service:
                mock_service = Mock()
                mock_service.check_for_updates = AsyncMock(return_value=True)
                mock_service.load_metadata.return_value = {
                    "last_check": "2024-01-01",
                    "last_update": "2024-01-01",
                    "entry_count": 38894,
                    "content_hash": "abc123def456",
                }
                mock_update_service.return_value = mock_service

                result = await check_for_updates()

                assert result["has_updates"] is True
                assert result["entry_count"] == 38894
                mock_service.check_for_updates.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_for_updates_service_error(self):
        """Test update check with service error."""
        with patch(
            "src.api.admin.get_qdrant_client",
            side_effect=Exception("Update check failed"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await check_for_updates()
            assert exc_info.value.status_code == 500


class TestUpdateOperations:
    """Test update operation endpoints."""

    @pytest.mark.asyncio
    async def test_perform_incremental_update_success(self):
        """Test successful incremental update."""
        with patch("src.api.admin.get_qdrant_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            with patch("src.api.admin.UpdateService") as mock_update_service:
                mock_service = Mock()
                mock_update_service.return_value = mock_service

                background_tasks = BackgroundTasks()
                result = await perform_incremental_update(background_tasks)

                assert result["status"] == "processing"
                assert "incremental update started" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_perform_full_update_success(self):
        """Test successful full update."""
        with patch("src.api.admin.get_qdrant_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            with patch("src.api.admin.UpdateService") as mock_update_service:
                mock_service = Mock()
                mock_update_service.return_value = mock_service

                background_tasks = BackgroundTasks()
                result = await perform_full_update(background_tasks)

                assert result["status"] == "processing"
                assert "full update started" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_perform_full_update_no_client(self):
        """Test full update when client is not available."""
        with patch("src.api.admin.get_qdrant_client", return_value=None):
            background_tasks = BackgroundTasks()

            with pytest.raises(HTTPException) as exc_info:
                await perform_full_update(background_tasks)
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_get_update_status_success(self):
        """Test successful update status check."""
        with patch("src.api.admin.get_qdrant_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            with patch("src.api.admin.UpdateService") as mock_update_service:
                mock_service = Mock()
                mock_service.load_metadata.return_value = {
                    "last_check": "2024-01-01",
                    "last_update": "2024-01-01",
                    "entry_count": 38894,
                    "update_available": False,
                }
                mock_update_service.return_value = mock_service

                result = await get_update_status()

                assert "last_check" in result
                assert result["entry_count"] == 38894


class TestSchedulingOperations:
    """Test scheduling operation endpoints."""

    @pytest.mark.asyncio
    async def test_schedule_weekly_update_success(self):
        """Test successful weekly update scheduling."""
        with patch("src.api.admin.get_qdrant_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            with patch("src.api.admin.UpdateService") as mock_update_service:
                mock_service = Mock()
                mock_update_service.return_value = mock_service

                background_tasks = BackgroundTasks()
                result = await schedule_weekly_update(background_tasks)

                assert result["status"] == "processing"
                assert "weekly update check started" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_analyze_optimal_schedule_success(self):
        """Test successful schedule analysis."""
        with patch("src.api.admin.SmartScheduler") as mock_scheduler:
            mock_instance = Mock()
            mock_instance.get_optimal_update_schedule = AsyncMock(
                return_value={
                    "recommended_frequency": "weekly",
                    "optimal_time": "Sunday 03:00 UTC",
                }
            )
            mock_scheduler.return_value = mock_instance

            result = await analyze_optimal_schedule()

            assert "recommended_frequency" in result
            mock_instance.get_optimal_update_schedule.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_update_safety_success(self):
        """Test successful update safety check."""
        with patch("src.api.admin.SmartScheduler") as mock_scheduler:
            mock_instance = Mock()
            mock_instance.is_safe_to_update = AsyncMock(
                return_value={"safe_to_update": True, "reasons": []}
            )
            mock_scheduler.return_value = mock_instance

            result = await check_update_safety()

            assert "safe_to_update" in result
            mock_instance.is_safe_to_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_smart_update_success(self):
        """Test successful smart update."""
        with patch("src.api.admin.SmartScheduler") as mock_scheduler:
            mock_scheduler_instance = Mock()
            mock_scheduler_instance.is_safe_to_update = AsyncMock(
                return_value={"safe_to_update": True, "reasons": []}
            )
            mock_scheduler.return_value = mock_scheduler_instance

            with patch("src.api.admin.get_qdrant_client") as mock_get_client:
                mock_client = Mock()
                mock_get_client.return_value = mock_client

                with patch("src.api.admin.UpdateService") as mock_update_service:
                    mock_service = Mock()
                    mock_update_service.return_value = mock_service

                    background_tasks = BackgroundTasks()
                    result = await smart_update(background_tasks)

                    assert result["status"] == "processing"
                    assert "smart update started" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_smart_update_not_safe(self):
        """Test smart update when not safe to update."""
        with patch("src.api.admin.SmartScheduler") as mock_scheduler:
            mock_scheduler_instance = Mock()
            mock_scheduler_instance.is_safe_to_update = AsyncMock(
                return_value={
                    "safe_to_update": False,
                    "reasons": ["Recent release detected"],
                    "recommendation": "Wait 2 hours",
                }
            )
            mock_scheduler.return_value = mock_scheduler_instance

            background_tasks = BackgroundTasks()
            result = await smart_update(background_tasks)

            assert result["status"] == "postponed"
            assert "not safe to update" in result["message"].lower()


class TestErrorHandling:
    """Test error handling across all admin endpoints."""

    @pytest.mark.asyncio
    async def test_all_endpoints_handle_exceptions(self):
        """Test that all endpoints handle exceptions properly."""
        background_tasks = BackgroundTasks()

        # Test endpoints that might raise exceptions
        with patch(
            "src.api.admin.get_qdrant_client", side_effect=Exception("Test error")
        ):
            with pytest.raises(HTTPException):
                await check_for_updates()

            with pytest.raises(HTTPException):
                await perform_incremental_update(background_tasks)

            with pytest.raises(HTTPException):
                await get_update_status()

            with pytest.raises(HTTPException):
                await schedule_weekly_update(background_tasks)

        # Test scheduler endpoints
        with patch(
            "src.api.admin.SmartScheduler", side_effect=Exception("Scheduler error")
        ):
            with pytest.raises(HTTPException):
                await analyze_optimal_schedule()

            with pytest.raises(HTTPException):
                await check_update_safety()
