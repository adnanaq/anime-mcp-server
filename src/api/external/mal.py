"""MAL API endpoints for external anime data."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Path, Query, Request, Response
from fastapi.responses import JSONResponse

from ...config import get_settings
from ...integrations.error_handling import (
    CircuitBreaker,
    ErrorContext,
    ErrorSeverity,
    ExecutionTracer,
    GracefulDegradation,
)
from ...services.external.mal_service import MALService

logger = logging.getLogger(__name__)

# Create router for MAL endpoints
router = APIRouter(prefix="/external/mal", tags=["External APIs", "MAL"])

# Get settings for MAL configuration
settings = get_settings()

# Initialize comprehensive error handling infrastructure
_circuit_breaker = CircuitBreaker(
    api_name="mal_api", failure_threshold=5, recovery_timeout=30
)
_graceful_degradation = GracefulDegradation()
_execution_tracer = ExecutionTracer()

# Initialize MAL service (singleton pattern) with error handling infrastructure
_mal_service = MALService(
    client_id=settings.mal_client_id, client_secret=settings.mal_client_secret
)


async def _handle_endpoint_common_setup(
    request: Request,
    x_correlation_id: Optional[str],
    response: Response,
    operation: str,
) -> str:
    """Handle common endpoint setup: correlation ID generation and circuit breaker check."""
    # Use middleware correlation ID first, then header, then generate
    correlation_id = (
        getattr(request.state, "correlation_id", None)
        or x_correlation_id
        or f"mal-{operation}-{uuid.uuid4().hex[:12]}"
    )
    response.headers["X-Correlation-ID"] = correlation_id

    # Check circuit breaker
    circuit_breaker_response = await _handle_circuit_breaker_check(correlation_id)
    if circuit_breaker_response:
        raise HTTPException(
            status_code=503,
            detail=circuit_breaker_response["detail"],
            headers={"X-Correlation-ID": correlation_id},
        )

    return correlation_id


async def _handle_endpoint_logging(
    correlation_id: str,
    operation: str,
    context: Dict[str, Any],
    is_success: bool = False,
) -> None:
    """Handle endpoint logging for start and success."""
    if is_success:
        f"MAL {operation} completed successfully"
        log_context = {"operation": f"mal_{operation}", "success": True, **context}
    else:
        f"MAL {operation} request started"
        log_context = {"operation": f"mal_{operation}", **context}


# correlation logging removed


async def _create_error_response(
    error: Exception,
    correlation_id: str,
    operation: str = "mal_search",
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create enhanced error response using comprehensive error handling infrastructure.

    Args:
        error: The exception that occurred
        correlation_id: Correlation ID for tracking
        operation: Operation being performed
        context: Additional context for error handling

    Returns:
        Enhanced error response with comprehensive error context
    """
    # Classify error severity based on error type and message
    severity = _classify_error_severity(error)

    # Create enhanced ErrorContext
    error_context = ErrorContext(
        user_message=_get_user_friendly_message(error, severity),
        debug_info=f"Operation: {operation}, Error: {str(error)}",
        correlation_id=correlation_id,
        severity=severity,
        context=context or {},
        timestamp=datetime.now(timezone.utc),
    )

    # Add breadcrumb for debugging
    error_context.add_breadcrumb(
        f"API endpoint error in {operation}",
        {"error_type": type(error).__name__, "correlation_id": correlation_id},
    )

    # Return comprehensive error structure
    return {
        "error_context": {
            "user_message": error_context.user_message,
            "debug_info": error_context.debug_info,
            "correlation_id": error_context.correlation_id,
            "severity": error_context.severity.value,
            "timestamp": error_context.timestamp.isoformat(),
            "breadcrumbs": error_context.breadcrumbs,
            "operation": operation,
        },
        "detail": error_context.user_message,  # For FastAPI compatibility
    }


def _classify_error_severity(error: Exception) -> ErrorSeverity:
    """Classify error severity based on error type and message."""
    error_msg = str(error).lower()

    if "database corruption" in error_msg or "critical" in error_msg:
        return ErrorSeverity.CRITICAL
    elif any(
        term in error_msg
        for term in ["connection", "network", "timeout", "unavailable"]
    ):
        return ErrorSeverity.ERROR
    elif any(term in error_msg for term in ["rate limit", "throttle", "quota"]):
        return ErrorSeverity.WARNING
    elif any(term in error_msg for term in ["invalid", "parameter", "validation"]):
        return ErrorSeverity.INFO
    else:
        return ErrorSeverity.ERROR


def _get_user_friendly_message(error: Exception, severity: ErrorSeverity) -> str:
    """Get user-friendly error message based on error type and severity."""
    error_msg = str(error).lower()

    if "not found" in error_msg:
        return "The requested anime or resource was not found. Please check the ID and try again."
    elif "database" in error_msg or "connection" in error_msg:
        return "The anime search service is temporarily unavailable. Please try again in a few moments."
    elif "rate limit" in error_msg or "throttle" in error_msg:
        return "Too many requests. Please wait a moment before searching again."
    elif "timeout" in error_msg:
        return "The search request timed out. Please try again with a simpler query."
    elif "invalid" in error_msg or "validation" in error_msg:
        return "Invalid search parameters provided. Please check your request and try again."
    elif severity == ErrorSeverity.CRITICAL:
        return "A critical service error occurred. Please contact support if this persists."
    else:
        return "The anime search service is temporarily unavailable. Please try again later."


async def _handle_circuit_breaker_check(
    correlation_id: str,
) -> Optional[Dict[str, Any]]:
    """Check circuit breaker and return appropriate response if open."""
    if _circuit_breaker.is_open():

        # Return circuit breaker response
        error_context = ErrorContext(
            user_message="The anime search service is temporarily experiencing issues and has been disabled to prevent further problems. Please try again in a few minutes.",
            debug_info="Circuit breaker is open due to repeated failures",
            correlation_id=correlation_id,
            severity=ErrorSeverity.WARNING,
            context={"circuit_breaker_open": True},
        )

        return {
            "error_context": {
                "user_message": error_context.user_message,
                "debug_info": error_context.debug_info,
                "correlation_id": error_context.correlation_id,
                "severity": error_context.severity.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "breadcrumbs": [],
                "operation": "circuit_breaker_check",
            },
            "detail": "Service temporarily unavailable - circuit breaker is open",
        }

    return None


@router.get("/search")
async def search_anime(
    request: Request,
    response: Response,
    q: str = Query("", description="Search query string"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    status: Optional[str] = Query(
        None, pattern="^(airing|complete|upcoming)$", description="Anime status filter"
    ),
    genres: Optional[str] = Query(
        None, description="Comma-separated genre IDs (e.g., '1,2,3')"
    ),
    # Enhanced Jikan parameters
    anime_type: Optional[str] = Query(
        None,
        pattern="^(TV|Movie|OVA|ONA|Special)$",
        description="Anime type filter (TV, Movie, OVA, ONA, Special)",
    ),
    score: Optional[float] = Query(
        None, ge=0.0, le=10.0, description="Exact score filter (0.0-10.0)"
    ),
    min_score: Optional[float] = Query(
        None, ge=0.0, le=10.0, description="Minimum score filter (0.0-10.0)"
    ),
    max_score: Optional[float] = Query(
        None, ge=0.0, le=10.0, description="Maximum score filter (0.0-10.0)"
    ),
    rating: Optional[str] = Query(
        None,
        pattern="^(G|PG|PG-13|R|R\\+|Rx)$",
        description="Content rating (G, PG, PG-13, R, R+, Rx)",
    ),
    sfw: Optional[bool] = Query(
        None, description="Filter out adult content (Safe For Work)"
    ),
    genres_exclude: Optional[str] = Query(
        None, description="Comma-separated genre IDs to exclude (e.g., '14,26')"
    ),
    order_by: Optional[str] = Query(
        None,
        pattern="^(title|score|rank|popularity|members|favorites|start_date|end_date|episodes|type)$",
        description="Order results by field (title, score, rank, popularity, etc.)",
    ),
    sort: Optional[str] = Query(
        None, pattern="^(asc|desc)$", description="Sort direction (asc, desc)"
    ),
    letter: Optional[str] = Query(
        None, pattern="^[A-Za-z]$", description="Return entries starting with letter"
    ),
    producers: Optional[str] = Query(
        None, description="Comma-separated producer/studio IDs"
    ),
    start_date: Optional[str] = Query(
        None,
        pattern="^\\d{4}(-\\d{2})?(-\\d{2})?$",
        description="Start date filter (YYYY, YYYY-MM, or YYYY-MM-DD)",
    ),
    end_date: Optional[str] = Query(
        None,
        pattern="^\\d{4}(-\\d{2})?(-\\d{2})?$",
        description="End date filter (YYYY, YYYY-MM, or YYYY-MM-DD)",
    ),
    page: Optional[int] = Query(None, ge=1, description="Page number for pagination"),
    unapproved: Optional[bool] = Query(None, description="Include unapproved entries"),
    # Correlation ID support
    x_correlation_id: Optional[str] = Header(
        None, alias="X-Correlation-ID", description="Correlation ID for request tracing"
    ),
) -> Dict[str, Any]:
    """Search for anime on MAL/Jikan with full parameter support and correlation tracking.

    Args:
        q: Search query string (optional for letter-based search)
        limit: Maximum number of results (1-50)
        status: Anime status filter (airing, complete, upcoming)
        genres: Comma-separated genre IDs to include
        anime_type: Anime type filter (TV, Movie, OVA, ONA, Special)
        score: Exact score filter (0.0-10.0)
        min_score: Minimum score filter (0.0-10.0)
        max_score: Maximum score filter (0.0-10.0)
        rating: Content rating (G, PG, PG-13, R, R+, Rx)
        sfw: Filter out adult content
        genres_exclude: Comma-separated genre IDs to exclude
        order_by: Order results by field
        sort: Sort direction (asc, desc)
        letter: Return entries starting with letter
        producers: Comma-separated producer/studio IDs
        start_date: Start date filter (YYYY, YYYY-MM, YYYY-MM-DD)
        end_date: End date filter (YYYY, YYYY-MM, YYYY-MM-DD)
        page: Page number for pagination
        unapproved: Include unapproved entries
        x_correlation_id: Correlation ID for request tracing

    Returns:
        Enhanced search results with metadata and correlation tracking

    Raises:
        HTTPException: If search service fails or validation errors occur
    """
    try:
        # Handle common setup (correlation ID + circuit breaker)
        correlation_id = await _handle_endpoint_common_setup(
            request, x_correlation_id, response, "api"
        )

        # Start execution tracing
        trace_id = await _execution_tracer.start_trace(
            operation="mal_api_search",
            context={
                "correlation_id": correlation_id,
                "query": q,
                "enhanced_parameters_count": sum(
                    1
                    for p in [
                        anime_type,
                        score,
                        min_score,
                        max_score,
                        rating,
                        sfw,
                        genres_exclude,
                        order_by,
                        sort,
                        letter,
                        producers,
                        start_date,
                        end_date,
                        page,
                        unapproved,
                    ]
                    if p is not None
                ),
                "api_endpoint": "/external/mal/search",
            },
        )

        # Log request start
        await _handle_endpoint_logging(
            correlation_id,
            "API search",
            {
                "query": q,
                "limit": limit,
                "trace_id": trace_id,
                "enhanced_parameters": {
                    "anime_type": anime_type,
                    "score_filters": bool(score or min_score or max_score),
                    "genre_filters": bool(genres or genres_exclude),
                    "date_filters": bool(start_date or end_date),
                    "sorting": bool(order_by or sort),
                },
            },
        )

        # Parse comma-separated lists
        genre_list = None
        if genres:
            try:
                genre_list = [int(g.strip()) for g in genres.split(",") if g.strip()]
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail="Invalid genre format. Use comma-separated integers.",
                )

        genres_exclude_list = None
        if genres_exclude:
            try:
                genres_exclude_list = [
                    int(g.strip()) for g in genres_exclude.split(",") if g.strip()
                ]
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail="Invalid genres_exclude format. Use comma-separated integers.",
                )

        producers_list = None
        if producers:
            try:
                producers_list = [
                    int(p.strip()) for p in producers.split(",") if p.strip()
                ]
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail="Invalid producers format. Use comma-separated integers.",
                )

        # Additional validation for score range
        if min_score is not None and max_score is not None and min_score > max_score:
            raise HTTPException(
                status_code=422,
                detail="min_score must be less than or equal to max_score",
            )

        logger.info(
            "Enhanced MAL search: query='%s', limit=%d, correlation_id=%s, enhanced_params=%s",
            q,
            limit,
            correlation_id,
            {
                "anime_type": anime_type,
                "score_range": (
                    f"{min_score}-{max_score}" if min_score or max_score else None
                ),
                "rating": rating,
                "order_by": order_by,
                "genres_exclude": bool(genres_exclude_list),
                "producers": bool(producers_list),
            },
        )

        # Add trace step for service call
        if trace_id:
            await _execution_tracer.add_trace_step(
                trace_id=trace_id,
                step_name="service_call_start",
                step_data={
                    "service": "mal_service",
                    "method": "search_anime",
                    "parameters_count": len(
                        [
                            p
                            for p in [
                                q,
                                anime_type,
                                score,
                                min_score,
                                max_score,
                                rating,
                                sfw,
                                genres_exclude_list,
                                order_by,
                                sort,
                                letter,
                                producers_list,
                                start_date,
                                end_date,
                                page,
                                unapproved,
                            ]
                            if p is not None
                        ]
                    ),
                },
            )

        # Call service with all parameters (with circuit breaker protection)
        try:
            results = await _circuit_breaker.call_with_breaker(
                lambda: _mal_service.search_anime(
                    query=q,
                    limit=limit,
                    status=status,
                    genres=genre_list,
                    # Parameters
                    anime_type=anime_type,
                    score=score,
                    min_score=min_score,
                    max_score=max_score,
                    rating=rating,
                    sfw=sfw,
                    genres_exclude=genres_exclude_list,
                    order_by=order_by,
                    sort=sort,
                    letter=letter,
                    producers=producers_list,
                    start_date=start_date,
                    end_date=end_date,
                    page=page,
                    unapproved=unapproved,
                    correlation_id=correlation_id,
                )
            )
        except Exception as service_error:
            # Circuit breaker failure is automatically recorded by call_with_breaker()
            raise service_error

        # Add trace step for successful service call
        if trace_id:
            await _execution_tracer.add_trace_step(
                trace_id=trace_id,
                step_name="service_call_success",
                step_data={
                    "results_count": len(results),
                    "service_response_time_category": (
                        "normal" if len(results) > 0 else "empty"
                    ),
                },
            )

        await _handle_endpoint_logging(
            correlation_id,
            "API search",
            {"results_count": len(results), "query": q, "trace_id": trace_id},
            is_success=True,
        )

        # Enhanced response with execution trace
        response_data = {
            "source": "mal",
            "query": q,
            "limit": limit,
            "status": status,
            "genres": genre_list,
            "results": results,
            "total_results": len(results),
            "enhanced_parameters": {
                "anime_type": anime_type,
                "score": score,
                "min_score": min_score,
                "max_score": max_score,
                "rating": rating,
                "sfw": sfw,
                "genres_exclude": genres_exclude_list,
                "order_by": order_by,
                "sort": sort,
                "letter": letter,
                "producers": producers_list,
                "start_date": start_date,
                "end_date": end_date,
                "page": page,
                "unapproved": unapproved,
            },
        }

        # Add execution trace info if available
        if trace_id:
            trace_info = await _execution_tracer.end_trace(
                trace_id=trace_id,
                status="success",
                result={"results_count": len(results), "query": q},
            )
            if trace_info:
                response_data["trace_info"] = {
                    "trace_id": trace_id,
                    "duration_ms": trace_info.get("duration_ms", 0),
                    "steps": len(trace_info.get("steps", [])),
                    "operation": "mal_search",
                }

        # Add operation metadata (but keep correlation_id in headers only)
        response_data["operation_metadata"] = {
            "operation": "mal_search",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Return JSONResponse to preserve headers set on response object
        return JSONResponse(
            content=response_data, headers={"X-Correlation-ID": correlation_id}
        )

    except HTTPException as http_exc:
        # Handle different types of HTTP exceptions appropriately
        error_correlation_id = x_correlation_id or f"validation-{uuid.uuid4().hex[:8]}"

        # End trace if active
        if "trace_id" in locals() and trace_id:
            await _execution_tracer.end_trace(
                trace_id=trace_id,
                status=(
                    "validation_error"
                    if http_exc.status_code == 422
                    else "service_error"
                ),
                error=http_exc,
            )

        # For 503 status codes (circuit breaker, service unavailable), re-raise as-is
        if http_exc.status_code == 503:
            response.headers["X-Correlation-ID"] = error_correlation_id
            raise http_exc

        # For validation errors (422), use comprehensive error handling
        validation_error_response = await _create_error_response(
            error=http_exc,
            correlation_id=error_correlation_id,
            operation="mal_search_validation",
            context={"validation_error": True, "status_code": http_exc.status_code},
        )

        response.headers["X-Correlation-ID"] = error_correlation_id

        # For validation errors, return 422 with comprehensive error context
        raise HTTPException(
            status_code=422,
            detail=validation_error_response,
            headers={"X-Correlation-ID": error_correlation_id},
        )

    except Exception as e:
        # Use comprehensive error handling infrastructure for service errors
        error_correlation_id = x_correlation_id or f"error-{uuid.uuid4().hex[:8]}"

        # End trace if active
        if "trace_id" in locals() and trace_id:
            await _execution_tracer.end_trace(
                trace_id=trace_id, status="error", error=e
            )

        # Circuit breaker failure is automatically recorded by call_with_breaker()

        # Try graceful degradation for specific error types
        if any(term in str(e).lower() for term in ["timeout", "connection", "network"]):
            try:
                degraded_response = await _graceful_degradation.handle_failure(
                    primary_func=lambda: None,  # We already failed
                    fallback_data={
                        "source": "mal_degraded",
                        "correlation_id": error_correlation_id,
                        "query": q,
                        "results": [],
                        "total_results": 0,
                        "degraded": True,
                        "fallback_reason": "Service temporarily unavailable",
                    },
                    context={
                        "operation": "mal_search",
                        "query": q,
                        "correlation_id": error_correlation_id,
                    },
                )

                response.headers["X-Correlation-ID"] = error_correlation_id
                return degraded_response

            except Exception:
                # If graceful degradation fails, continue to comprehensive error handling
                pass

        # Create comprehensive error response
        error_response = await _create_error_response(
            error=e,
            correlation_id=error_correlation_id,
            operation="mal_search",
            context={
                "query": q,
                "circuit_breaker_failures": _circuit_breaker.failure_count,
                "service_error": True,
            },
        )

        response.headers["X-Correlation-ID"] = error_correlation_id

        # Return comprehensive error with proper status code
        status_code = (
            503
            if any(
                term in str(e).lower()
                for term in [
                    "unavailable",
                    "connection",
                    "database",
                    "timeout",
                    "network",
                ]
            )
            else 500
        )
        raise HTTPException(
            status_code=status_code,
            detail=error_response,
            headers={"X-Correlation-ID": error_correlation_id},
        )


@router.get("/anime/{anime_id}")
async def get_anime_details(
    request: Request,
    response: Response,
    anime_id: int = Path(..., ge=1, description="MAL anime ID"),
    x_correlation_id: Optional[str] = Header(
        None, alias="X-Correlation-ID", description="Correlation ID for request tracing"
    ),
) -> Dict[str, Any]:
    """Get detailed anime information by MAL ID.

    Args:
        anime_id: MAL anime ID
        x_correlation_id: Correlation ID for request tracing

    Returns:
        Detailed anime information with correlation tracking

    Raises:
        HTTPException: If anime not found or service fails
    """
    try:
        # Handle common setup (correlation ID + circuit breaker)
        correlation_id = await _handle_endpoint_common_setup(
            request, x_correlation_id, response, "details"
        )

        # Log request start
        await _handle_endpoint_logging(
            correlation_id, "anime details", {"anime_id": anime_id}
        )

        anime_data = await _mal_service.get_anime_details(
            anime_id, correlation_id=correlation_id
        )

        if not anime_data:
            raise HTTPException(
                status_code=404, detail=f"Anime with ID {anime_id} not found on MAL"
            )

        await _handle_endpoint_logging(
            correlation_id, "anime details", {"anime_id": anime_id}, is_success=True
        )

        response_data = {"source": "mal", "anime_id": anime_id, "data": anime_data}

        # Add operation metadata (but keep correlation_id in headers only)
        response_data["operation_metadata"] = {
            "operation": "mal_anime_details",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return JSONResponse(
            content=response_data, headers={"X-Correlation-ID": correlation_id}
        )

    except HTTPException as http_exc:
        error_correlation_id = (
            x_correlation_id or f"details-error-{uuid.uuid4().hex[:8]}"
        )

        # For 503 status codes (circuit breaker, service unavailable), re-raise as-is
        if http_exc.status_code == 503:
            response.headers["X-Correlation-ID"] = error_correlation_id
            raise http_exc

        # For other HTTP errors, use comprehensive error handling
        error_response = await _create_error_response(
            error=http_exc,
            correlation_id=error_correlation_id,
            operation="mal_anime_details",
            context={"anime_id": anime_id, "http_error": True},
        )

        response.headers["X-Correlation-ID"] = error_correlation_id
        raise HTTPException(
            status_code=http_exc.status_code,
            detail=error_response,
            headers={"X-Correlation-ID": error_correlation_id},
        )

    except Exception as e:
        error_correlation_id = (
            x_correlation_id or f"details-error-{uuid.uuid4().hex[:8]}"
        )

        # Create comprehensive error response
        error_response = await _create_error_response(
            error=e,
            correlation_id=error_correlation_id,
            operation="mal_anime_details",
            context={"anime_id": anime_id, "service_error": True},
        )

        response.headers["X-Correlation-ID"] = error_correlation_id

        status_code = (
            503
            if any(
                term in str(e).lower()
                for term in ["unavailable", "connection", "timeout"]
            )
            else 500
        )
        raise HTTPException(
            status_code=status_code,
            detail=error_response,
            headers={"X-Correlation-ID": error_correlation_id},
        )


@router.get("/seasonal/{year}/{season}")
async def get_seasonal_anime(
    request: Request,
    response: Response,
    year: int = Path(..., ge=1990, le=2030, description="Year (1990-2030)"),
    season: str = Path(
        ...,
        pattern="^(winter|spring|summer|fall)$",
        description="Season (winter, spring, summer, fall)",
    ),
    x_correlation_id: Optional[str] = Header(
        None, alias="X-Correlation-ID", description="Correlation ID for request tracing"
    ),
) -> Dict[str, Any]:
    """Get seasonal anime for a specific year and season.

    Args:
        year: Year (1990-2030)
        season: Season (winter, spring, summer, fall)
        x_correlation_id: Correlation ID for request tracing

    Returns:
        Seasonal anime list with correlation tracking

    Raises:
        HTTPException: If service fails
    """
    try:
        # Handle common setup (correlation ID + circuit breaker)
        correlation_id = await _handle_endpoint_common_setup(
            request, x_correlation_id, response, "seasonal"
        )

        # Log request start
        await _handle_endpoint_logging(
            correlation_id, "seasonal anime", {"year": year, "season": season}
        )

        results = await _mal_service.get_seasonal_anime(
            year, season, correlation_id=correlation_id
        )

        await _handle_endpoint_logging(
            correlation_id,
            "seasonal anime",
            {"year": year, "season": season, "results_count": len(results)},
            is_success=True,
        )

        response_data = {
            "source": "mal",
            "year": year,
            "season": season,
            "results": results,
            "total_results": len(results),
        }

        # Add operation metadata (but keep correlation_id in headers only)
        response_data["operation_metadata"] = {
            "operation": "mal_seasonal_anime",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return JSONResponse(
            content=response_data, headers={"X-Correlation-ID": correlation_id}
        )

    except HTTPException as http_exc:
        error_correlation_id = (
            x_correlation_id or f"seasonal-error-{uuid.uuid4().hex[:8]}"
        )

        # For 503 status codes (circuit breaker, service unavailable), re-raise as-is
        if http_exc.status_code == 503:
            response.headers["X-Correlation-ID"] = error_correlation_id
            raise http_exc

        # For validation errors (422), use comprehensive error handling
        error_response = await _create_error_response(
            error=http_exc,
            correlation_id=error_correlation_id,
            operation="mal_seasonal_anime",
            context={"year": year, "season": season, "validation_error": True},
        )

        response.headers["X-Correlation-ID"] = error_correlation_id
        raise HTTPException(
            status_code=http_exc.status_code,
            detail=error_response,
            headers={"X-Correlation-ID": error_correlation_id},
        )

    except Exception as e:
        error_correlation_id = (
            x_correlation_id or f"seasonal-error-{uuid.uuid4().hex[:8]}"
        )

        # Create comprehensive error response
        error_response = await _create_error_response(
            error=e,
            correlation_id=error_correlation_id,
            operation="mal_seasonal_anime",
            context={"year": year, "season": season, "service_error": True},
        )

        response.headers["X-Correlation-ID"] = error_correlation_id

        status_code = (
            503
            if any(
                term in str(e).lower()
                for term in ["unavailable", "connection", "timeout"]
            )
            else 500
        )
        raise HTTPException(
            status_code=status_code,
            detail=error_response,
            headers={"X-Correlation-ID": error_correlation_id},
        )


@router.get("/current-season")
async def get_current_season_anime(
    response: Response,
    x_correlation_id: Optional[str] = Header(
        None, alias="X-Correlation-ID", description="Correlation ID for request tracing"
    ),
) -> Dict[str, Any]:
    """Get current season anime.

    Args:
        x_correlation_id: Correlation ID for request tracing

    Returns:
        Current season anime list with correlation tracking

    Raises:
        HTTPException: If service fails
    """
    try:
        # Generate or use provided correlation ID
        correlation_id = x_correlation_id or f"mal-current-{uuid.uuid4().hex[:12]}"
        response.headers["X-Correlation-ID"] = correlation_id

        # Check circuit breaker first
        circuit_breaker_response = await _handle_circuit_breaker_check(correlation_id)
        if circuit_breaker_response:
            raise HTTPException(
                status_code=503,
                detail=circuit_breaker_response["detail"],
                headers={"X-Correlation-ID": correlation_id},
            )

        results = await _mal_service.get_current_season(correlation_id=correlation_id)

        response_data = {
            "source": "mal",
            "type": "current_season",
            "results": results,
            "total_results": len(results),
        }

        # Add operation metadata (but keep correlation_id in headers only)
        response_data["operation_metadata"] = {
            "operation": "mal_current_season",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return JSONResponse(
            content=response_data, headers={"X-Correlation-ID": correlation_id}
        )

    except HTTPException as http_exc:
        error_correlation_id = (
            x_correlation_id or f"current-error-{uuid.uuid4().hex[:8]}"
        )

        # For 503 status codes (circuit breaker, service unavailable), re-raise as-is
        if http_exc.status_code == 503:
            response.headers["X-Correlation-ID"] = error_correlation_id
            raise http_exc

        # For other HTTP errors, use comprehensive error handling
        error_response = await _create_error_response(
            error=http_exc,
            correlation_id=error_correlation_id,
            operation="mal_current_season",
            context={"http_error": True},
        )

        response.headers["X-Correlation-ID"] = error_correlation_id
        raise HTTPException(
            status_code=http_exc.status_code,
            detail=error_response,
            headers={"X-Correlation-ID": error_correlation_id},
        )

    except Exception as e:
        error_correlation_id = (
            x_correlation_id or f"current-error-{uuid.uuid4().hex[:8]}"
        )

        # Create comprehensive error response
        error_response = await _create_error_response(
            error=e,
            correlation_id=error_correlation_id,
            operation="mal_current_season",
            context={"service_error": True},
        )

        response.headers["X-Correlation-ID"] = error_correlation_id

        status_code = (
            503
            if any(
                term in str(e).lower()
                for term in ["unavailable", "connection", "timeout"]
            )
            else 500
        )
        raise HTTPException(
            status_code=status_code,
            detail=error_response,
            headers={"X-Correlation-ID": error_correlation_id},
        )


@router.get("/anime/{anime_id}/statistics")
async def get_anime_statistics(
    response: Response,
    anime_id: int = Path(..., ge=1, description="MAL anime ID"),
    x_correlation_id: Optional[str] = Header(
        None, alias="X-Correlation-ID", description="Correlation ID for request tracing"
    ),
) -> Dict[str, Any]:
    """Get anime statistics (watching, completed, etc.).

    Args:
        anime_id: MAL anime ID
        x_correlation_id: Correlation ID for request tracing

    Returns:
        Anime statistics with correlation tracking

    Raises:
        HTTPException: If anime not found or service fails
    """
    try:
        # Generate or use provided correlation ID
        correlation_id = x_correlation_id or f"mal-stats-{uuid.uuid4().hex[:12]}"
        response.headers["X-Correlation-ID"] = correlation_id

        # Check circuit breaker first
        circuit_breaker_response = await _handle_circuit_breaker_check(correlation_id)
        if circuit_breaker_response:
            raise HTTPException(
                status_code=503,
                detail=circuit_breaker_response["detail"],
                headers={"X-Correlation-ID": correlation_id},
            )

        stats = await _mal_service.get_anime_statistics(
            anime_id, correlation_id=correlation_id
        )

        if not stats:
            raise HTTPException(
                status_code=404,
                detail=f"Statistics for anime ID {anime_id} not found on MAL",
            )

        response_data = {"source": "mal", "anime_id": anime_id, "statistics": stats}

        # Add operation metadata (but keep correlation_id in headers only)
        response_data["operation_metadata"] = {
            "operation": "mal_statistics",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return JSONResponse(
            content=response_data, headers={"X-Correlation-ID": correlation_id}
        )

    except HTTPException as http_exc:
        error_correlation_id = x_correlation_id or f"stats-error-{uuid.uuid4().hex[:8]}"

        # For 503 status codes (circuit breaker, service unavailable), re-raise as-is
        if http_exc.status_code == 503:
            response.headers["X-Correlation-ID"] = error_correlation_id
            raise http_exc

        # For other HTTP errors, use comprehensive error handling
        error_response = await _create_error_response(
            error=http_exc,
            correlation_id=error_correlation_id,
            operation="mal_anime_statistics",
            context={"anime_id": anime_id, "http_error": True},
        )

        response.headers["X-Correlation-ID"] = error_correlation_id
        raise HTTPException(
            status_code=http_exc.status_code,
            detail=error_response,
            headers={"X-Correlation-ID": error_correlation_id},
        )

    except Exception as e:
        error_correlation_id = x_correlation_id or f"stats-error-{uuid.uuid4().hex[:8]}"

        # Create comprehensive error response
        error_response = await _create_error_response(
            error=e,
            correlation_id=error_correlation_id,
            operation="mal_anime_statistics",
            context={"anime_id": anime_id, "service_error": True},
        )

        response.headers["X-Correlation-ID"] = error_correlation_id

        status_code = (
            503
            if any(
                term in str(e).lower()
                for term in ["unavailable", "connection", "timeout"]
            )
            else 500
        )
        raise HTTPException(
            status_code=status_code,
            detail=error_response,
            headers={"X-Correlation-ID": error_correlation_id},
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check MAL service health.

    Returns:
        Service health status
    """
    try:
        logger.info("MAL health check")
        return await _mal_service.health_check()

    except Exception as e:
        logger.error("MAL health check failed: %s", e)
        return {
            "service": "mal",
            "status": "unhealthy",
            "error": str(e),
            "circuit_breaker_open": True,
        }
