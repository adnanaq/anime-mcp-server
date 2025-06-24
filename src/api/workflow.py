"""FastAPI endpoints for LangGraph workflow operations."""

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..langgraph.adapters import (
    create_adapter_registry_from_mcp_tools,
)
from ..langgraph.models import (
    AnimeSearchContext,
    ConversationState,
    SmartOrchestrationState,
)
from ..langgraph.workflow_engine import AnimeWorkflowEngine
from ..mcp.tools import get_all_mcp_tools

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory conversation storage (would be Redis/PostgreSQL in production)
_conversation_storage: Dict[str, ConversationState] = {}


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
_workflow_engine: Optional[AnimeWorkflowEngine] = None


def get_workflow_engine() -> AnimeWorkflowEngine:
    """Get or create the workflow engine instance."""
    global _workflow_engine

    if _workflow_engine is None:
        logger.info("Initializing workflow engine...")

        # Get all MCP tools
        mcp_tools = get_all_mcp_tools()
        logger.info(f"Found {len(mcp_tools)} MCP tools")

        # Create adapter registry
        adapter_registry = create_adapter_registry_from_mcp_tools(mcp_tools)

        # Create workflow engine
        _workflow_engine = AnimeWorkflowEngine(adapter_registry)
        logger.info("Workflow engine initialized successfully")

    return _workflow_engine


def get_conversation_state(session_id: str) -> Optional[ConversationState]:
    """Get conversation state by session ID."""
    return _conversation_storage.get(session_id)


def save_conversation_state(state: ConversationState) -> None:
    """Save conversation state."""
    _conversation_storage[state.session_id] = state


def get_conversation_stats() -> Dict[str, Any]:
    """Get conversation statistics."""
    if not _conversation_storage:
        return {
            "total_conversations": 0,
            "active_sessions": 0,
            "average_messages_per_session": 0.0,
            "total_workflow_steps": 0,
        }

    total_messages = sum(
        len(state.messages) for state in _conversation_storage.values()
    )
    total_steps = sum(
        len(state.workflow_steps) for state in _conversation_storage.values()
    )

    return {
        "total_conversations": len(_conversation_storage),
        "active_sessions": len(
            _conversation_storage
        ),  # All stored sessions are considered active
        "average_messages_per_session": (
            total_messages / len(_conversation_storage)
            if _conversation_storage
            else 0.0
        ),
        "total_workflow_steps": total_steps,
    }


@router.post("/conversation", response_model=ConversationResponse)
async def process_conversation(request: ConversationRequest) -> ConversationResponse:
    """Process a conversation message through the workflow engine."""
    try:
        engine = get_workflow_engine()

        # Get or create conversation state
        if request.session_id:
            state = get_conversation_state(request.session_id)
            if not state:
                raise HTTPException(
                    status_code=404, detail=f"Session {request.session_id} not found"
                )
        else:
            # Create new session
            session_id = str(uuid.uuid4())
            state = ConversationState(session_id=session_id)

        # Process the conversation
        result_state = await engine.process_conversation(state, request.message)

        # Save the updated state
        save_conversation_state(result_state)

        # Convert to response format
        messages = [
            {
                "message_type": msg.message_type.value,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "tool_call_id": msg.tool_call_id,
                "tool_results": msg.tool_results,
                "metadata": msg.metadata,
            }
            for msg in result_state.messages
        ]

        workflow_steps = [
            {
                "step_type": step.step_type.value,
                "tool_name": step.tool_name,
                "parameters": step.parameters,
                "result": step.result,
                "reasoning": step.reasoning,
                "confidence": step.confidence,
                "execution_time_ms": step.execution_time_ms,
                "error": step.error,
                "timestamp": step.timestamp,
            }
            for step in result_state.workflow_steps
        ]

        return ConversationResponse(
            session_id=result_state.session_id,
            messages=messages,
            workflow_steps=workflow_steps,
            current_context=(
                result_state.current_context.model_dump()
                if result_state.current_context
                else None
            ),
            user_preferences=(
                result_state.user_preferences.model_dump()
                if result_state.user_preferences
                else None
            ),
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
    """Process a multimodal conversation with text and image."""
    try:
        engine = get_workflow_engine()

        # Get or create conversation state
        if request.session_id:
            state = get_conversation_state(request.session_id)
            if not state:
                raise HTTPException(
                    status_code=404, detail=f"Session {request.session_id} not found"
                )
        else:
            # Create new session
            session_id = str(uuid.uuid4())
            state = ConversationState(session_id=session_id)

        # Process the multimodal conversation
        result_state = await engine.process_multimodal_conversation(
            state, request.message, request.image_data, request.text_weight
        )

        # Save the updated state
        save_conversation_state(result_state)

        # Convert to response format (same as regular conversation)
        messages = [
            {
                "message_type": msg.message_type.value,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "tool_call_id": msg.tool_call_id,
                "tool_results": msg.tool_results,
                "metadata": msg.metadata,
            }
            for msg in result_state.messages
        ]

        workflow_steps = [
            {
                "step_type": step.step_type.value,
                "tool_name": step.tool_name,
                "parameters": step.parameters,
                "result": step.result,
                "reasoning": step.reasoning,
                "confidence": step.confidence,
                "execution_time_ms": step.execution_time_ms,
                "error": step.error,
                "timestamp": step.timestamp,
            }
            for step in result_state.workflow_steps
        ]

        return ConversationResponse(
            session_id=result_state.session_id,
            messages=messages,
            workflow_steps=workflow_steps,
            current_context=(
                result_state.current_context.model_dump()
                if result_state.current_context
                else None
            ),
            user_preferences=(
                result_state.user_preferences.model_dump()
                if result_state.user_preferences
                else None
            ),
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
    """Process a conversation with smart orchestration features."""
    try:
        # Get or create conversation state with smart orchestration
        if request.session_id:
            state = get_conversation_state(request.session_id)
            if state and not isinstance(state, SmartOrchestrationState):
                # Upgrade existing state to smart orchestration
                smart_state = SmartOrchestrationState(
                    session_id=state.session_id,
                    messages=state.messages,
                    current_context=state.current_context,
                    user_preferences=state.user_preferences,
                    workflow_steps=state.workflow_steps,
                    current_step_index=state.current_step_index,
                    conversation_summary=state.conversation_summary,
                    metadata=state.metadata,
                    created_at=state.created_at,
                    updated_at=state.updated_at,
                )
                smart_state.max_discovery_depth = request.max_discovery_depth
                state = smart_state
        else:
            # Create new smart orchestration state
            state = SmartOrchestrationState(
                session_id=str(uuid.uuid4()),
                max_discovery_depth=request.max_discovery_depth,
            )

        # Set limit in context if provided
        if request.limit:
            if not state.current_context:
                state.current_context = AnimeSearchContext()
            state.current_context.limit = request.limit

        # Process the conversation with smart orchestration
        engine = get_workflow_engine()
        result_state = await engine.process_conversation(state, request.message)

        # Save the updated state
        save_conversation_state(result_state)

        # Convert to response format
        messages = [
            {
                "message_type": msg.message_type.value,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "tool_call_id": msg.tool_call_id,
                "tool_results": msg.tool_results,
                "metadata": msg.metadata,
            }
            for msg in result_state.messages
        ]

        workflow_steps = [
            {
                "step_type": step.step_type.value,
                "tool_name": step.tool_name,
                "parameters": step.parameters,
                "result": step.result,
                "reasoning": step.reasoning,
                "confidence": step.confidence,
                "execution_time_ms": step.execution_time_ms,
                "error": step.error,
                "timestamp": step.timestamp,
            }
            for step in result_state.workflow_steps
        ]

        # Add smart orchestration specific fields to response
        response_data = {
            "session_id": result_state.session_id,
            "messages": messages,
            "workflow_steps": workflow_steps,
            "current_context": (
                result_state.current_context.model_dump()
                if result_state.current_context
                else None
            ),
            "user_preferences": (
                result_state.user_preferences.model_dump()
                if result_state.user_preferences
                else None
            ),
        }

        # Add summary if available
        if (
            hasattr(result_state, "conversation_summary")
            and result_state.conversation_summary
        ):
            response_data["summary"] = result_state.conversation_summary

        return ConversationResponse(**response_data)

    except Exception as e:
        logger.error(f"Error processing smart conversation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Smart conversation processing error: {str(e)}"
        )


@router.get("/conversation/{session_id}", response_model=ConversationResponse)
async def get_conversation_history(session_id: str) -> ConversationResponse:
    """Get conversation history for a session."""
    try:
        state = get_conversation_state(session_id)
        if not state:
            raise HTTPException(
                status_code=404, detail=f"Conversation {session_id} not found"
            )

        engine = get_workflow_engine()
        summary = await engine.get_conversation_summary(state)

        # Convert to response format
        messages = [
            {
                "message_type": msg.message_type.value,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "tool_call_id": msg.tool_call_id,
                "tool_results": msg.tool_results,
                "metadata": msg.metadata,
            }
            for msg in state.messages
        ]

        workflow_steps = [
            {
                "step_type": step.step_type.value,
                "tool_name": step.tool_name,
                "parameters": step.parameters,
                "result": step.result,
                "reasoning": step.reasoning,
                "confidence": step.confidence,
                "execution_time_ms": step.execution_time_ms,
                "error": step.error,
                "timestamp": step.timestamp,
            }
            for step in state.workflow_steps
        ]

        return ConversationResponse(
            session_id=state.session_id,
            messages=messages,
            workflow_steps=workflow_steps,
            current_context=(
                state.current_context.model_dump() if state.current_context else None
            ),
            user_preferences=(
                state.user_preferences.model_dump() if state.user_preferences else None
            ),
            summary=summary,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving conversation: {str(e)}"
        )


@router.delete("/conversation/{session_id}")
async def delete_conversation(session_id: str) -> Dict[str, str]:
    """Delete a conversation session."""
    try:
        if session_id not in _conversation_storage:
            raise HTTPException(
                status_code=404, detail=f"Conversation {session_id} not found"
            )

        del _conversation_storage[session_id]
        logger.info(f"Deleted conversation session: {session_id}")

        return {"message": f"Conversation {session_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error deleting conversation: {str(e)}"
        )


@router.get("/stats", response_model=ConversationStats)
async def get_workflow_stats() -> ConversationStats:
    """Get workflow and conversation statistics."""
    try:
        stats = get_conversation_stats()
        return ConversationStats(**stats)

    except Exception as e:
        logger.error(f"Error getting workflow stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")


@router.get("/health")
async def workflow_health_check() -> Dict[str, Any]:
    """Health check for workflow system."""
    try:
        engine = get_workflow_engine()

        # Check adapter registry
        adapter_count = len(engine.adapter_registry.get_all_adapters())

        # Check conversation storage
        active_sessions = len(_conversation_storage)

        return {
            "status": "healthy",
            "workflow_engine": "initialized",
            "adapters_loaded": adapter_count,
            "active_sessions": active_sessions,
            "storage": "in_memory",  # Would be Redis/PostgreSQL in production
        }

    except Exception as e:
        logger.error(f"Workflow health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}
