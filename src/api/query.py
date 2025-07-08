"""FastAPI universal query endpoint using LangGraph workflow operations."""

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

# ReactAgent-based workflow engine with create_react_agent (Phase 2.5)
from ..langgraph.react_agent_workflow import create_react_agent_workflow_engine
from ..anime_mcp.modern_client import get_all_mcp_tools

logger = logging.getLogger(__name__)

router = APIRouter()

# StateGraph handles conversation persistence internally with MemorySaver checkpointing


# SearchIntentParameters removed - LLM handles intent processing via system prompt


class QueryRequest(BaseModel):
    """Universal query request for natural language anime queries."""

    message: str = Field(..., description="Natural language query message")
    image_data: Optional[str] = Field(None, description="Base64 encoded image data for multimodal queries")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity (only used if enable_conversation=True)")
    enable_conversation: bool = Field(False, description="Enable conversation flow and session memory")
    enable_super_step: bool = Field(False, description="Enable Google Pregel-inspired super-step parallel execution for performance")


class ConversationResponse(BaseModel):
    """Response model for conversation endpoints."""

    session_id: str
    messages: List[Dict[str, Any]]
    workflow_steps: List[Dict[str, Any]]
    current_context: Optional[Dict[str, Any]]
    user_preferences: Optional[Dict[str, Any]]
    summary: Optional[str] = None


class QueryStats(BaseModel):
    """Statistics about queries."""

    total_queries: int
    active_sessions: int
    average_queries_per_session: float
    total_workflow_steps: int


# Global workflow engine instance
_workflow_engine = None


async def get_workflow_engine(enable_super_step: bool = False):
    """Get or create the ReactAgent workflow engine instance.
    
    Args:
        enable_super_step: Whether to enable super-step execution mode
        
    Returns:
        ReactAgent workflow engine with appropriate execution mode
    """
    global _workflow_engine

    # Always create new engine if super-step mode is requested (stateless approach)
    if enable_super_step or _workflow_engine is None:
        logger.info(f"Initializing ReactAgent workflow engine with FastMCP client (super_step={enable_super_step})...")

        # Get all MCP tools using FastMCP client adapter
        mcp_tools = await get_all_mcp_tools()
        logger.info(f"Discovered {len(mcp_tools)} MCP tools via FastMCP client")

        # Import execution mode enum
        from ..langgraph.react_agent_workflow import ExecutionMode
        
        execution_mode = ExecutionMode.SUPER_STEP if enable_super_step else ExecutionMode.STANDARD

        # Create ReactAgent workflow engine with appropriate execution mode
        engine = create_react_agent_workflow_engine(mcp_tools, execution_mode=execution_mode)
        logger.info(f"ReactAgent workflow engine initialized successfully in {execution_mode.value} mode")
        
        # Store standard engine for reuse, but always create fresh super-step engines
        if not enable_super_step:
            _workflow_engine = engine
            
        return engine

    return _workflow_engine


# StateGraph handles conversation state management internally


@router.post("", response_model=ConversationResponse)
async def process_query(request: QueryRequest, http_request: Request) -> ConversationResponse:
    """Universal query endpoint that processes natural language anime queries.
    
    Automatically detects query type (text-only vs multimodal) based on presence of image_data.
    Uses LangGraph ReactAgent with LLM-powered intent processing to understand user queries
    and route to appropriate MCP tools.
    
    Conversation flow can be enabled with enable_conversation=True for multi-turn interactions.
    """
    # Extract correlation ID from middleware
    correlation_id = getattr(http_request.state, 'correlation_id', None)
    
    try:
        logger.info(
            f"Processing universal query: {request.message[:100]}{'...' if len(request.message) > 100 else ''}",
            extra={
                "correlation_id": correlation_id,
                "query_length": len(request.message),
                "has_image": bool(request.image_data),
                "conversation_enabled": request.enable_conversation,
                "super_step_enabled": request.enable_super_step,
                "session_id": request.session_id,
            }
        )
        
        engine = await get_workflow_engine(enable_super_step=request.enable_super_step)
        
        # Handle session management based on conversation setting
        if request.enable_conversation:
            # Conversation mode: use provided session_id or create new one
            session_id = request.session_id or str(uuid.uuid4())
            thread_id = session_id  # Use session_id as thread_id for persistence
            logger.info(
                f"Conversation mode enabled for session {session_id}",
                extra={
                    "correlation_id": correlation_id,
                    "session_id": session_id,
                    "mode": "conversation"
                }
            )
        else:
            # Single query mode: use unique session for each request (no persistence)
            session_id = str(uuid.uuid4())
            thread_id = None  # No thread persistence for single queries
            logger.info(
                f"Single query mode - session {session_id}",
                extra={
                    "correlation_id": correlation_id,
                    "session_id": session_id,
                    "mode": "single"
                }
            )
        
        # Auto-detect query type and process accordingly
        if request.image_data:
            # Multimodal query - image + text
            logger.info(
                f"Processing multimodal query for session {session_id}",
                extra={
                    "correlation_id": correlation_id,
                    "session_id": session_id,
                    "query_type": "multimodal",
                    "image_size": len(request.image_data) if request.image_data else 0
                }
            )
            result_state = await engine.process_multimodal_conversation(
                session_id=session_id,
                message=request.message,
                image_data=request.image_data,
                text_weight=0.7,  # Default text weight
                thread_id=thread_id,  # Pass thread_id for conversation persistence
                search_parameters=None,  # Let LLM handle intent processing
            )
        else:
            # Text-only query
            logger.info(
                f"Processing text query for session {session_id}",
                extra={
                    "correlation_id": correlation_id,
                    "session_id": session_id,
                    "query_type": "text",
                    "message_length": len(request.message)
                }
            )
            result_state = await engine.process_conversation(
                session_id=session_id,
                message=request.message,
                thread_id=thread_id,  # Pass thread_id for conversation persistence
                search_parameters=None,  # Let LLM handle intent processing
            )
        
        logger.info(
            f"Query processing completed successfully for session {session_id}",
            extra={
                "correlation_id": correlation_id,
                "session_id": session_id,
                "message_count": len(result_state.get("messages", [])),
                "workflow_steps": len(result_state.get("workflow_steps", [])),
            }
        )
        
        # Convert ReactAgent result to response format
        return _convert_result_to_response(result_state, request.message)
        
    except Exception as e:
        logger.error(
            f"Error processing query: {str(e)}",
            extra={
                "correlation_id": correlation_id,
                "session_id": getattr(locals(), "session_id", "unknown"),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "query_type": "multimodal" if request.image_data else "text",
                "conversation_enabled": request.enable_conversation,
            }
        )
        raise HTTPException(
            status_code=500, 
            detail=f"Query processing error: {str(e)}",
            headers={"X-Correlation-ID": correlation_id} if correlation_id else {}
        )


@router.get("/session/{session_id}", response_model=ConversationResponse)
async def get_query_session_history(session_id: str, http_request: Request) -> ConversationResponse:
    """Get query session history using ReactAgent memory."""
    correlation_id = getattr(http_request.state, 'correlation_id', None)
    
    try:
        logger.info(
            f"Retrieving session history for {session_id}",
            extra={
                "correlation_id": correlation_id,
                "session_id": session_id,
            }
        )
        
        engine = await get_workflow_engine()

        # Generate summary from ReactAgent memory
        summary = await engine.get_conversation_summary(session_id)
        
        logger.info(
            f"Session history retrieved successfully for {session_id}",
            extra={
                "correlation_id": correlation_id,
                "session_id": session_id,
                "has_summary": bool(summary),
            }
        )

        # For now, return minimal response since ReactAgent manages session history internally
        # In a full implementation, we would retrieve the session from ReactAgent checkpointer
        return ConversationResponse(
            session_id=session_id,
            messages=[],  # Would be retrieved from ReactAgent checkpointer
            workflow_steps=[],  # Would be retrieved from ReactAgent checkpointer
            current_context=None,
            user_preferences=None,
            summary=summary,
        )

    except Exception as e:
        logger.error(
            f"Error getting query session history: {str(e)}",
            extra={
                "correlation_id": correlation_id,
                "session_id": session_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
        )
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving session: {str(e)}",
            headers={"X-Correlation-ID": correlation_id} if correlation_id else {}
        )


@router.delete("/session/{session_id}")
async def delete_query_session(session_id: str, http_request: Request) -> Dict[str, str]:
    """Delete a query session from ReactAgent memory."""
    correlation_id = getattr(http_request.state, 'correlation_id', None)
    
    try:
        logger.info(
            f"Delete request for query session: {session_id}",
            extra={
                "correlation_id": correlation_id,
                "session_id": session_id,
            }
        )
        
        # Note: ReactAgent session deletion would require accessing the checkpointer
        # For now, we'll just acknowledge the request
        
        logger.info(
            f"Query session delete acknowledged for {session_id}",
            extra={
                "correlation_id": correlation_id,
                "session_id": session_id,
            }
        )

        return {"message": f"Query session {session_id} delete request acknowledged"}

    except Exception as e:
        logger.error(
            f"Error deleting query session: {str(e)}",
            extra={
                "correlation_id": correlation_id,
                "session_id": session_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
        )
        raise HTTPException(
            status_code=500, 
            detail=f"Error deleting query session: {str(e)}",
            headers={"X-Correlation-ID": correlation_id} if correlation_id else {}
        )


@router.get("/stats", response_model=QueryStats)
async def get_query_stats(http_request: Request) -> QueryStats:
    """Get query processing statistics from ReactAgent."""
    correlation_id = getattr(http_request.state, 'correlation_id', None)
    
    try:
        logger.info(
            "Retrieving query processing statistics",
            extra={"correlation_id": correlation_id}
        )
        
        # Since ReactAgent handles session persistence internally,
        # we return basic stats for now
        stats = QueryStats(
            total_queries=0,  # Would be retrieved from ReactAgent checkpointer
            active_sessions=0,  # Would be retrieved from ReactAgent checkpointer
            average_queries_per_session=0.0,
            total_workflow_steps=0,
        )
        
        logger.info(
            "Query statistics retrieved successfully",
            extra={
                "correlation_id": correlation_id,
                "total_queries": stats.total_queries,
                "active_sessions": stats.active_sessions,
            }
        )
        
        return stats

    except Exception as e:
        logger.error(
            f"Error getting query stats: {str(e)}",
            extra={
                "correlation_id": correlation_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
        )
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving stats: {str(e)}",
            headers={"X-Correlation-ID": correlation_id} if correlation_id else {}
        )


# _prepare_search_parameters removed - LLM handles intent processing automatically


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
async def query_health_check(http_request: Request) -> Dict[str, Any]:
    """Health check for universal query system."""
    correlation_id = getattr(http_request.state, 'correlation_id', None)
    
    try:
        logger.info(
            "Performing query system health check",
            extra={"correlation_id": correlation_id}
        )
        
        engine = await get_workflow_engine()

        # Check ReactAgent workflow info
        workflow_info = engine.get_workflow_info()
        
        health_status = {
            "status": "healthy",
            "query_engine": "create_react_agent+LangGraph",
            "engine_type": workflow_info["engine_type"],
            "features": workflow_info["features"] + ["universal_query_interface", "auto_query_detection", "correlation_tracking"],
            "performance": workflow_info["performance"],
            "memory_persistence": True,
            "checkpointing": "MemorySaver",
            "query_api": {
                "natural_language_processing": True,
                "multimodal_detection": True,
                "llm_intent_processing": True,
                "conversation_flow": True,
                "correlation_tracking": True,
            },
        }
        
        logger.info(
            "Query health check completed successfully",
            extra={
                "correlation_id": correlation_id,
                "status": health_status["status"],
                "engine_type": health_status["engine_type"],
            }
        )

        return health_status

    except Exception as e:
        logger.error(
            f"Query health check failed: {str(e)}",
            extra={
                "correlation_id": correlation_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
        )
        return {
            "status": "unhealthy", 
            "error": str(e),
            "correlation_id": correlation_id
        }
