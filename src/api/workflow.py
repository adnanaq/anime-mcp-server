"""FastAPI endpoints for LangGraph workflow operations."""

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# ReactAgent-based workflow engine with create_react_agent (Phase 2.5)
from ..langgraph.react_agent_workflow import create_react_agent_workflow_engine
from ..mcp.fastmcp_client_adapter import get_all_mcp_tools

logger = logging.getLogger(__name__)

router = APIRouter()

# StateGraph handles conversation persistence internally with MemorySaver checkpointing


class SearchIntentParameters(BaseModel):
    """SearchIntent parameters for enhanced search control."""

    genres: Optional[List[str]] = Field(
        None, description="List of anime genres to filter by"
    )
    year_range: Optional[List[int]] = Field(
        None, description="Year range as [start_year, end_year]"
    )
    anime_types: Optional[List[str]] = Field(
        None, description="List of anime types (TV, Movie, OVA, etc.)"
    )
    studios: Optional[List[str]] = Field(None, description="List of animation studios")
    exclusions: Optional[List[str]] = Field(
        None, description="List of genres/themes to exclude"
    )
    mood_keywords: Optional[List[str]] = Field(
        None, description="List of mood descriptors"
    )
    limit: int = Field(10, description="Maximum number of results (1-50)")


class ConversationRequest(BaseModel):
    """Enhanced conversation request with optional SearchIntent parameters."""

    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Existing session ID")
    search_parameters: Optional[SearchIntentParameters] = Field(
        None,
        description="Optional explicit search parameters to override AI extraction",
    )


class MultimodalRequest(BaseModel):
    """Enhanced multimodal request with optional SearchIntent parameters."""

    message: str = Field(..., description="User message")
    image_data: str = Field(..., description="Base64 encoded image data")
    text_weight: float = Field(0.7, description="Weight for text vs image (0.0-1.0)")
    session_id: Optional[str] = Field(None, description="Existing session ID")
    search_parameters: Optional[SearchIntentParameters] = Field(
        None,
        description="Optional explicit search parameters to override AI extraction",
    )


class SmartConversationRequest(BaseModel):
    """Request model for smart orchestration conversation."""

    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Existing session ID")
    enable_smart_orchestration: bool = Field(
        True, description="Enable smart orchestration features"
    )
    max_discovery_depth: int = Field(
        3, description="Maximum discovery depth for complex queries"
    )
    limit: Optional[int] = Field(
        None, description="Result limit (1-50), extracted from message if not specified"
    )


class ConversationResponse(BaseModel):
    """Response model for conversation endpoints."""

    session_id: str
    messages: List[Dict[str, Any]]
    workflow_steps: List[Dict[str, Any]]
    current_context: Optional[Dict[str, Any]]
    user_preferences: Optional[Dict[str, Any]]
    summary: Optional[str] = None


class ConversationStats(BaseModel):
    """Statistics about conversations."""

    total_conversations: int
    active_sessions: int
    average_messages_per_session: float
    total_workflow_steps: int


# Global workflow engine instance
_workflow_engine = None


async def get_workflow_engine():
    """Get or create the ReactAgent workflow engine instance."""
    global _workflow_engine

    if _workflow_engine is None:
        logger.info("Initializing ReactAgent workflow engine with FastMCP client...")

        # Get all MCP tools using FastMCP client adapter
        mcp_tools = await get_all_mcp_tools()
        logger.info(f"Discovered {len(mcp_tools)} MCP tools via FastMCP client")

        # Create ReactAgent workflow engine (replaces ToolNode + manual LLM service)
        _workflow_engine = create_react_agent_workflow_engine(mcp_tools)
        logger.info("ReactAgent workflow engine initialized successfully")

    return _workflow_engine


# StateGraph handles conversation state management internally


@router.post("/conversation", response_model=ConversationResponse)
async def process_conversation(request: ConversationRequest) -> ConversationResponse:
    """Process a conversation message with optional SearchIntent parameters.

    Enhanced to support explicit search parameters while maintaining full
    backward compatibility with existing API contracts.
    """
    try:
        engine = await get_workflow_engine()

        # Use session ID or create new one
        session_id = request.session_id or str(uuid.uuid4())

        # Process conversation with optional search parameters
        result_state = await engine.process_conversation(
            session_id=session_id,
            message=request.message,
            search_parameters=_prepare_search_parameters(request.search_parameters),
        )

        # Convert ReactAgent result to response format
        return _convert_result_to_response(result_state, request.message)

    except Exception as e:
        logger.error(f"Error processing conversation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Conversation processing error: {str(e)}"
        )


@router.post("/multimodal", response_model=ConversationResponse)
async def process_multimodal_conversation(
    request: MultimodalRequest,
) -> ConversationResponse:
    """Process multimodal conversation with optional SearchIntent parameters.

    Enhanced to support explicit search parameters while maintaining full
    backward compatibility with existing multimodal API contracts.
    """
    try:
        engine = await get_workflow_engine()

        # Use session ID or create new one
        session_id = request.session_id or str(uuid.uuid4())

        # Process multimodal conversation with optional search parameters
        result_state = await engine.process_multimodal_conversation(
            session_id=session_id,
            message=request.message,
            image_data=request.image_data,
            text_weight=request.text_weight,
            search_parameters=_prepare_search_parameters(request.search_parameters),
        )

        # Convert ReactAgent result to response format
        return _convert_result_to_response(result_state, request.message)

    except Exception as e:
        logger.error(f"Error processing multimodal conversation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Multimodal conversation processing error: {str(e)}",
        )


@router.post("/smart-conversation", response_model=ConversationResponse)
async def process_smart_conversation(
    request: SmartConversationRequest,
) -> ConversationResponse:
    """Process a conversation with smart orchestration features using ReactAgent."""
    try:
        engine = await get_workflow_engine()

        # Use session ID or create new one
        session_id = request.session_id or str(uuid.uuid4())

        # ReactAgent already includes smart orchestration features
        # Process the conversation normally - ReactAgent handles orchestration internally
        result_state = await engine.process_conversation(
            session_id=session_id,
            message=request.message,
            search_parameters=None,  # Smart conversation relies on AI extraction
        )

        # Convert ReactAgent result to response format
        response = _convert_result_to_response(result_state, request.message)

        # Generate summary using ReactAgent for smart conversation
        summary = await engine.get_conversation_summary(session_id)
        response.summary = summary

        return response

    except Exception as e:
        logger.error(f"Error processing smart conversation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Smart conversation processing error: {str(e)}"
        )


@router.get("/conversation/{session_id}", response_model=ConversationResponse)
async def get_conversation_history(session_id: str) -> ConversationResponse:
    """Get conversation history for a session using ReactAgent memory."""
    try:
        engine = await get_workflow_engine()

        # Generate summary from ReactAgent memory
        summary = await engine.get_conversation_summary(session_id)

        # For now, return minimal response since ReactAgent manages conversation history internally
        # In a full implementation, we would retrieve the conversation from ReactAgent checkpointer
        return ConversationResponse(
            session_id=session_id,
            messages=[],  # Would be retrieved from ReactAgent checkpointer
            workflow_steps=[],  # Would be retrieved from ReactAgent checkpointer
            current_context=None,
            user_preferences=None,
            summary=summary,
        )

    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving conversation: {str(e)}"
        )


@router.delete("/conversation/{session_id}")
async def delete_conversation(session_id: str) -> Dict[str, str]:
    """Delete a conversation session from ReactAgent memory."""
    try:
        # Note: ReactAgent conversation deletion would require accessing the checkpointer
        # For now, we'll just acknowledge the request
        logger.info(f"Delete request for conversation session: {session_id}")

        return {"message": f"Conversation {session_id} delete request acknowledged"}

    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error deleting conversation: {str(e)}"
        )


@router.get("/stats", response_model=ConversationStats)
async def get_workflow_stats() -> ConversationStats:
    """Get workflow and conversation statistics from ReactAgent."""
    try:
        # Since ReactAgent handles conversation persistence internally,
        # we return basic stats for now
        return ConversationStats(
            total_conversations=0,  # Would be retrieved from ReactAgent checkpointer
            active_sessions=0,  # Would be retrieved from ReactAgent checkpointer
            average_messages_per_session=0.0,
            total_workflow_steps=0,
        )

    except Exception as e:
        logger.error(f"Error getting workflow stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")


def _prepare_search_parameters(
    search_params: Optional[SearchIntentParameters],
) -> Optional[Dict[str, Any]]:
    """Prepare SearchIntent parameters for workflow engine.

    Converts Pydantic model to dictionary and filters out None/empty values.
    """
    if not search_params:
        return None

    params = search_params.model_dump(exclude_none=True)

    # Filter out empty lists
    filtered_params = {}
    for key, value in params.items():
        if isinstance(value, list) and len(value) == 0:
            continue
        if value is not None:
            filtered_params[key] = value

    return filtered_params if filtered_params else None


def _convert_result_to_response(
    result_state: Dict[str, Any], original_message: str
) -> ConversationResponse:
    """Convert ReactAgent result to ConversationResponse format."""
    messages = []
    for msg_content in result_state.get("messages", []):
        messages.append(
            {
                "message_type": (
                    "user" if msg_content == original_message else "assistant"
                ),
                "content": msg_content,
                "timestamp": None,
                "tool_call_id": None,
                "tool_results": None,
                "metadata": None,
            }
        )

    workflow_steps = []
    for step in result_state.get("workflow_steps", []):
        workflow_steps.append(
            {
                "step_type": step.get("step_type", "unknown"),
                "tool_name": step.get("tool_name"),
                "parameters": step.get("parameters", {}),
                "result": step.get("result", {}),
                "reasoning": step.get("reasoning"),
                "confidence": step.get("confidence", 0.0),
                "execution_time_ms": None,
                "error": step.get("error"),
                "timestamp": None,
            }
        )

    return ConversationResponse(
        session_id=result_state["session_id"],
        messages=messages,
        workflow_steps=workflow_steps,
        current_context=result_state.get("current_context"),
        user_preferences=result_state.get("user_preferences"),
    )


@router.get("/health")
async def workflow_health_check() -> Dict[str, Any]:
    """Health check for workflow system."""
    try:
        engine = await get_workflow_engine()

        # Check ReactAgent workflow info
        workflow_info = engine.get_workflow_info()

        return {
            "status": "healthy",
            "workflow_engine": "create_react_agent+LangGraph",
            "engine_type": workflow_info["engine_type"],
            "features": workflow_info["features"] + ["enhanced_search_parameters"],
            "performance": workflow_info["performance"],
            "memory_persistence": True,
            "checkpointing": "MemorySaver",
            "enhanced_api": {
                "search_intent_parameters": True,
                "ai_parameter_override": True,
                "backward_compatibility": True,
            },
        }

    except Exception as e:
        logger.error(f"Workflow health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}
