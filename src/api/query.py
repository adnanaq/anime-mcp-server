"""FastAPI universal query endpoint using the Anime Discovery Swarm."""

import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..langgraph.anime_swarm import AnimeDiscoverySwarm
from ..langgraph.workflow_state import WorkflowResult

logger = logging.getLogger(__name__)

router = APIRouter()

# Global swarm instance
_anime_swarm = AnimeDiscoverySwarm()


class QueryRequest(BaseModel):
    """Universal query request for natural language anime queries."""

    message: str = Field(..., description="Natural language query message")
    image_data: Optional[str] = Field(
        None, description="Base64 encoded image data for multimodal queries"
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation continuity",
    )
    user_context: Optional[Dict[str, Any]] = Field(
        None, description="Optional user preferences and context"
    )


@router.post("", response_model=WorkflowResult)
async def process_query(
    request: QueryRequest, http_request: Request
) -> WorkflowResult:
    """
    Universal query endpoint that processes natural language anime queries
    using the Anime Discovery Swarm.
    """
    correlation_id = getattr(http_request.state, "correlation_id", None)
    session_id = request.session_id or str(uuid.uuid4())

    try:
        logger.info(
            f"Processing query with Anime Discovery Swarm: {request.message[:100]}...",
            extra={
                "correlation_id": correlation_id,
                "session_id": session_id,
                "has_image": bool(request.image_data),
            },
        )

        # The swarm's discover_anime method is the entry point for all queries.
        # It internally handles intent analysis and routing to the correct agent.
        # Image data can be passed in the user_context.
        user_context = request.user_context or {}
        if request.image_data:
            user_context["image_data"] = request.image_data

        result = await _anime_swarm.discover_anime(
            query=request.message,
            user_context=user_context,
            session_id=session_id,
        )

        logger.info(
            "Anime Discovery Swarm processing completed.",
            extra={"correlation_id": correlation_id, "session_id": session_id},
        )

        return result

    except Exception as e:
        logger.error(
            f"Error processing query with Anime Discovery Swarm: {e}",
            exc_info=True,
            extra={"correlation_id": correlation_id, "session_id": session_id},
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during query processing.",
            headers={"X-Correlation-ID": correlation_id} if correlation_id else {},
        )


@router.get("/health")
async def query_health_check(http_request: Request) -> Dict[str, Any]:
    """Health check for the Anime Discovery Swarm."""
    correlation_id = getattr(http_request.state, "correlation_id", None)
    try:
        health_report = _anime_swarm.get_health_report()
        logger.info(
            "Anime Discovery Swarm health check successful.",
            extra={"correlation_id": correlation_id},
        )
        return health_report
    except Exception as e:
        logger.error(
            f"Anime Discovery Swarm health check failed: {e}",
            exc_info=True,
            extra={"correlation_id": correlation_id},
        )
        return {
            "status": "unhealthy",
            "error": str(e),
            "correlation_id": correlation_id,
        }
