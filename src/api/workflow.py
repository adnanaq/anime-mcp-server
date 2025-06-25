"""FastAPI endpoints for LangGraph workflow operations."""

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# ToolNode-based workflow engine (replaces MCPAdapterRegistry + StateGraph)
from ..langgraph.workflow_engine import create_anime_workflow_engine
from ..mcp.tools import get_all_mcp_tools

logger = logging.getLogger(__name__)

router = APIRouter()

# StateGraph handles conversation persistence internally with MemorySaver checkpointing


class ConversationRequest(BaseModel):
    """Request model for conversation endpoint."""

    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Existing session ID")


class MultimodalRequest(BaseModel):
    """Request model for multimodal conversation."""

    message: str = Field(..., description="User message")
    image_data: str = Field(..., description="Base64 encoded image data")
    text_weight: float = Field(0.7, description="Weight for text vs image (0.0-1.0)")
    session_id: Optional[str] = Field(None, description="Existing session ID")


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


def get_workflow_engine():
    """Get or create the ToolNode workflow engine instance."""
    global _workflow_engine

    if _workflow_engine is None:
        logger.info("Initializing ToolNode workflow engine...")

        # Get all MCP tools
        mcp_tools = get_all_mcp_tools()
        logger.info(f"Found {len(mcp_tools)} MCP tools")

        # Create ToolNode workflow engine (replaces adapter registry + StateGraph)
        _workflow_engine = create_anime_workflow_engine(mcp_tools)
        logger.info("ToolNode workflow engine initialized successfully")

    return _workflow_engine


# StateGraph handles conversation state management internally


@router.post("/conversation", response_model=ConversationResponse)
async def process_conversation(request: ConversationRequest) -> ConversationResponse:
    """Process a conversation message through the StateGraph workflow engine."""
    try:
        engine = get_workflow_engine()

        # Use session ID or create new one
        session_id = request.session_id or str(uuid.uuid4())

        # Process the conversation using StateGraph
        result_state = await engine.process_conversation(
            session_id=session_id,
            message=request.message
        )

        # Convert ToolNode result to response format
        # ToolNode returns a dictionary in StateGraph-compatible format
        messages = []
        for msg_content in result_state.get("messages", []):
            messages.append({
                "message_type": "user" if msg_content == request.message else "assistant",
                "content": msg_content,
                "timestamp": None,
                "tool_call_id": None,
                "tool_results": None,
                "metadata": None,
            })

        workflow_steps = []
        for step in result_state.get("workflow_steps", []):
            workflow_steps.append({
                "step_type": step.get("step_type", "unknown"),
                "tool_name": step.get("tool_name"),
                "parameters": step.get("parameters", {}),
                "result": step.get("result", {}),
                "reasoning": step.get("reasoning"),
                "confidence": step.get("confidence", 0.0),
                "execution_time_ms": None,
                "error": step.get("error"),
                "timestamp": None,
            })

        return ConversationResponse(
            session_id=result_state["session_id"],
            messages=messages,
            workflow_steps=workflow_steps,
            current_context=result_state.get("current_context"),
            user_preferences=result_state.get("user_preferences"),
        )

    except Exception as e:
        logger.error(f"Error processing conversation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Conversation processing error: {str(e)}"
        )


@router.post("/multimodal", response_model=ConversationResponse)
async def process_multimodal_conversation(
    request: MultimodalRequest,
) -> ConversationResponse:
    """Process a multimodal conversation with text and image using StateGraph."""
    try:
        engine = get_workflow_engine()

        # Use session ID or create new one
        session_id = request.session_id or str(uuid.uuid4())

        # Process the multimodal conversation using ToolNode
        result_state = await engine.process_multimodal_conversation(
            session_id=session_id,
            message=request.message,
            image_data=request.image_data,
            text_weight=request.text_weight
        )

        # Convert ToolNode result to response format
        messages = []
        for msg_content in result_state.get("messages", []):
            messages.append({
                "message_type": "user" if msg_content == request.message else "assistant",
                "content": msg_content,
                "timestamp": None,
                "tool_call_id": None,
                "tool_results": None,
                "metadata": None,
            })

        workflow_steps = []
        for step in result_state.get("workflow_steps", []):
            workflow_steps.append({
                "step_type": step.get("step_type", "unknown"),
                "tool_name": step.get("tool_name"),
                "parameters": step.get("parameters", {}),
                "result": step.get("result", {}),
                "reasoning": step.get("reasoning"),
                "confidence": step.get("confidence", 0.0),
                "execution_time_ms": None,
                "error": step.get("error"),
                "timestamp": None,
            })

        return ConversationResponse(
            session_id=result_state["session_id"],
            messages=messages,
            workflow_steps=workflow_steps,
            current_context=result_state.get("current_context"),
            user_preferences=result_state.get("user_preferences"),
        )

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
    """Process a conversation with smart orchestration features using StateGraph."""
    try:
        engine = get_workflow_engine()

        # Use session ID or create new one
        session_id = request.session_id or str(uuid.uuid4())

        # ToolNode already includes smart orchestration features
        # Process the conversation normally - ToolNode handles orchestration internally
        result_state = await engine.process_conversation(
            session_id=session_id,
            message=request.message
        )

        # Convert ToolNode result to response format
        messages = []
        for msg_content in result_state.get("messages", []):
            messages.append({
                "message_type": "user" if msg_content == request.message else "assistant",
                "content": msg_content,
                "timestamp": None,
                "tool_call_id": None,
                "tool_results": None,
                "metadata": None,
            })

        workflow_steps = []
        for step in result_state.get("workflow_steps", []):
            workflow_steps.append({
                "step_type": step.get("step_type", "unknown"),
                "tool_name": step.get("tool_name"),
                "parameters": step.get("parameters", {}),
                "result": step.get("result", {}),
                "reasoning": step.get("reasoning"),
                "confidence": step.get("confidence", 0.0),
                "execution_time_ms": None,
                "error": step.get("error"),
                "timestamp": None,
            })

        # Generate summary using ToolNode
        summary = await engine.get_conversation_summary(session_id)

        return ConversationResponse(
            session_id=result_state["session_id"],
            messages=messages,
            workflow_steps=workflow_steps,
            current_context=result_state.get("current_context"),
            user_preferences=result_state.get("user_preferences"),
            summary=summary,
        )

    except Exception as e:
        logger.error(f"Error processing smart conversation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Smart conversation processing error: {str(e)}"
        )


@router.get("/conversation/{session_id}", response_model=ConversationResponse)
async def get_conversation_history(session_id: str) -> ConversationResponse:
    """Get conversation history for a session using ToolNode memory."""
    try:
        engine = get_workflow_engine()
        
        # Generate summary from ToolNode memory
        summary = await engine.get_conversation_summary(session_id)

        # For now, return minimal response since ToolNode manages conversation history internally
        # In a full implementation, we would retrieve the conversation from ToolNode checkpointer
        return ConversationResponse(
            session_id=session_id,
            messages=[],  # Would be retrieved from ToolNode checkpointer
            workflow_steps=[],  # Would be retrieved from ToolNode checkpointer
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
    """Delete a conversation session from ToolNode memory."""
    try:
        # Note: ToolNode conversation deletion would require accessing the checkpointer
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
    """Get workflow and conversation statistics from ToolNode."""
    try:
        # Since ToolNode handles conversation persistence internally,
        # we return basic stats for now
        return ConversationStats(
            total_conversations=0,  # Would be retrieved from ToolNode checkpointer
            active_sessions=0,  # Would be retrieved from ToolNode checkpointer
            average_messages_per_session=0.0,
            total_workflow_steps=0,
        )

    except Exception as e:
        logger.error(f"Error getting workflow stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")


@router.get("/health")
async def workflow_health_check() -> Dict[str, Any]:
    """Health check for workflow system."""
    try:
        engine = get_workflow_engine()

        # Check ToolNode workflow info
        workflow_info = engine.get_workflow_info()

        return {
            "status": "healthy",
            "workflow_engine": "ToolNode+StateGraph",
            "engine_type": workflow_info["engine_type"],
            "features": workflow_info["features"],
            "performance": workflow_info["performance"],
            "memory_persistence": True,
            "checkpointing": "MemorySaver",
        }

    except Exception as e:
        logger.error(f"Workflow health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}
