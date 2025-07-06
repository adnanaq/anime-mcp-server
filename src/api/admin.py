# src/api/admin.py - Admin Endpoints for Database Management
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from ..services.smart_scheduler import SmartScheduler
from ..services.update_service import UpdateService
from ..vector.qdrant_client import QdrantClient

router = APIRouter()
logger = logging.getLogger(__name__)


# Access global qdrant client
def get_qdrant_client() -> Optional[QdrantClient]:
    """Get the global Qdrant client instance."""
    from .. import main

    return main.qdrant_client


@router.post("/check-updates")
async def check_for_updates(request: Request) -> Dict[str, Any]:
    """Check if anime database has updates available"""
    correlation_id = getattr(request.state, 'correlation_id', None)
    
    try:
        logger.info(
            "Starting database update check",
            extra={"correlation_id": correlation_id}
        )
        
        qdrant_client = get_qdrant_client()
        update_service = UpdateService(qdrant_client=qdrant_client)
        has_updates = await update_service.check_for_updates()

        metadata = update_service.load_metadata()

        logger.info(
            f"Update check completed - has_updates: {has_updates}",
            extra={
                "correlation_id": correlation_id,
                "has_updates": has_updates,
                "entry_count": metadata.get("entry_count"),
            }
        )

        return {
            "has_updates": has_updates,
            "last_check": metadata.get("last_check"),
            "last_update": metadata.get("last_update"),
            "entry_count": metadata.get("entry_count"),
            "content_hash": metadata.get("content_hash", "")[:8],
        }

    except Exception as e:
        logger.error(
            f"Update check failed: {e}",
            extra={
                "correlation_id": correlation_id,
                "error_type": type(e).__name__,
            }
        )
        raise HTTPException(status_code=500, detail=f"Update check failed: {str(e)}")


@router.post("/update-incremental")
async def perform_incremental_update(
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Perform incremental database update"""
    try:
        qdrant_client = get_qdrant_client()
        update_service = UpdateService(qdrant_client=qdrant_client)

        # Run in background to avoid timeout
        background_tasks.add_task(update_service.perform_incremental_update)

        return {
            "message": "Incremental update started in background",
            "status": "processing",
        }

    except Exception as e:
        logger.error(f"Incremental update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@router.post("/update-full")
async def perform_full_update(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Perform full database refresh"""
    try:
        # Use global qdrant client with correct Docker networking
        qdrant_client = get_qdrant_client()
        if not qdrant_client:
            raise HTTPException(status_code=503, detail="Qdrant client not initialized")

        update_service = UpdateService(qdrant_client=qdrant_client)

        # Run in background to avoid timeout
        # Create a simple background task that maintains the client context
        def run_full_update() -> None:
            """Run full update in background thread."""
            import asyncio

            # Use the working client in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(update_service.perform_full_update())
            finally:
                loop.close()

        background_tasks.add_task(run_full_update)

        return {
            "message": "Full update started in background",
            "status": "processing",
            "warning": "This will re-index all 38,000+ entries and may take 2-3 hours",
        }

    except Exception as e:
        logger.error(f"Full update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@router.get("/update-status")
async def get_update_status() -> Dict[str, Any]:
    """Get current update status and metadata"""
    try:
        qdrant_client = get_qdrant_client()
        update_service = UpdateService(qdrant_client=qdrant_client)
        metadata = update_service.load_metadata()

        return {
            "last_check": metadata.get("last_check"),
            "last_update": metadata.get("last_update"),
            "last_incremental_update": metadata.get("last_incremental_update"),
            "last_full_update": metadata.get("last_full_update"),
            "entry_count": metadata.get("entry_count"),
            "update_available": metadata.get("update_available", False),
            "last_changes": metadata.get("last_changes", {}),
            "content_hash": metadata.get("content_hash", "")[:8],
        }

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.post("/schedule-weekly-update")
async def schedule_weekly_update(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Manually trigger weekly update process"""
    try:
        qdrant_client = get_qdrant_client()
        update_service = UpdateService(qdrant_client=qdrant_client)

        # Run in background
        background_tasks.add_task(update_service.schedule_weekly_update)

        return {"message": "Weekly update check started", "status": "processing"}

    except Exception as e:
        logger.error(f"Weekly update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Weekly update failed: {str(e)}")


@router.get("/smart-schedule-analysis")
async def analyze_optimal_schedule() -> Dict[str, Any]:
    """Analyze release patterns and suggest optimal update schedule"""
    try:
        scheduler = SmartScheduler()
        analysis = await scheduler.get_optimal_update_schedule()

        return analysis

    except Exception as e:
        logger.error(f"Schedule analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/update-safety-check")
async def check_update_safety() -> Dict[str, Any]:
    """Check if it's currently safe to update based on recent activity"""
    try:
        scheduler = SmartScheduler()
        safety_check = await scheduler.is_safe_to_update(min_hours_after_release=2)

        return safety_check

    except Exception as e:
        logger.error(f"Safety check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Safety check failed: {str(e)}")


@router.post("/smart-update")
async def smart_update(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Perform update only if it's safe to do so"""
    try:
        scheduler = SmartScheduler()
        safety_check = await scheduler.is_safe_to_update(min_hours_after_release=2)

        if not safety_check["safe_to_update"]:
            return {
                "message": "Update postponed - not safe to update yet",
                "status": "postponed",
                "reasons": safety_check["reasons"],
                "recommendation": safety_check["recommendation"],
            }

        # Safe to update - proceed
        qdrant_client = get_qdrant_client()
        update_service = UpdateService(qdrant_client=qdrant_client)
        background_tasks.add_task(update_service.perform_incremental_update)

        return {
            "message": "Smart update started - safety checks passed",
            "status": "processing",
            "safety_info": safety_check,
        }

    except Exception as e:
        logger.error(f"Smart update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Smart update failed: {str(e)}")
