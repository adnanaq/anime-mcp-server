"""Base client for API integrations with error handling foundation."""

import logging
from typing import Any, Callable, Dict, Optional, Union

import aiohttp

from src.exceptions import APIError

from ..error_handling import (
    CircuitBreaker,
    CorrelationLogger,
    ErrorContext,
    ErrorSeverity,
    ExecutionTracer,
    GracefulDegradation,
)
from ..rate_limiter import rate_limit_manager

logger = logging.getLogger(__name__)


class BaseClient:
    """Base client that integrates all error handling components."""

    def __init__(
        self,
        service_name: str,
        circuit_breaker: Optional[CircuitBreaker] = None,
        cache_manager=None,
        error_handler=None,
        correlation_logger: Optional[CorrelationLogger] = None,
        execution_tracer: Optional[ExecutionTracer] = None,
        timeout: float = 30.0,
    ):
        """Initialize BaseClient with error handling dependencies.

        Args:
            service_name: Name of the service for rate limiting
            circuit_breaker: Circuit breaker instance
            cache_manager: Cache manager instance
            error_handler: Error handler instance
            correlation_logger: Correlation logger instance
            execution_tracer: Execution tracer instance
            timeout: Request timeout in seconds
        """
        self.service_name = service_name
        self.circuit_breaker = circuit_breaker or CircuitBreaker(api_name=service_name)
        self.cache_manager = cache_manager
        self.error_handler = error_handler
        self.correlation_logger = correlation_logger
        self.execution_tracer = execution_tracer
        self.timeout = timeout

        # Simple logging
        self.logger = logging.getLogger(f"integrations.{service_name}")

        # Note: rate_limiter removed since we use global rate_limit_manager directly

    async def make_request(
        self,
        url: str,
        method: str = "GET",
        priority: int = 1,
        endpoint: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """Make HTTP request with rate limiting and basic error handling.

        Args:
            url: URL to request
            method: HTTP method
            priority: Request priority (0=highest, 2=lowest)
            endpoint: Endpoint identifier for monitoring
            **kwargs: Additional request parameters

        Returns:
            Response data

        Raises:
            APIError: If request fails
        """
        # Set timeout if not provided
        if "timeout" not in kwargs:
            kwargs["timeout"] = aiohttp.ClientTimeout(total=self.timeout)

        # Acquire rate limit permission
        if not await rate_limit_manager.acquire(
            service_name=self.service_name, priority=priority, endpoint=endpoint
        ):
            self.logger.warning(f"Rate limit exceeded for {self.service_name}")
            raise APIError(f"Rate limit exceeded for {self.service_name}")

        # Execute request with circuit breaker
        async def _request():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(method, url, **kwargs) as response:
                        # Record response for rate limit adaptation
                        rate_limit_manager.record_response(
                            self.service_name, response.status, 0  # Simple timing
                        )

                        # Parse response
                        if (
                            response.headers.get("content-type", "")
                            .lower()
                            .startswith("application/json")
                        ):
                            response_data = await response.json()
                        else:
                            response_data = await response.text()

                        # Check for HTTP errors
                        if response.status >= 400:
                            error_msg = (
                                f"{self.service_name} API error: {response.status}"
                            )
                            if isinstance(response_data, dict):
                                error_msg += f" - {response_data.get('error', response_data.get('message', ''))}"
                            self.logger.error(error_msg)
                            raise APIError(error_msg)

                        return response_data

            except aiohttp.ClientError as e:
                error_msg = f"{self.service_name} connection error: {e}"
                self.logger.error(error_msg)
                raise APIError(error_msg)

        # Use circuit breaker if available
        if self.circuit_breaker:
            try:
                return await self.circuit_breaker.call_with_breaker(_request)
            except Exception as e:
                # Log circuit breaker errors with correlation if available
                if hasattr(self, "_current_correlation_id") and self.correlation_logger:
                    await self.correlation_logger.log_with_correlation(
                        correlation_id=self._current_correlation_id,
                        level="error",
                        message=f"Circuit breaker error for {self.service_name}: {str(e)}",
                        context={
                            "service": self.service_name,
                            "circuit_breaker": "open",
                        },
                    )
                raise
        else:
            return await _request()

    async def _parse_response(
        self, response: aiohttp.ClientResponse
    ) -> Union[Dict[str, Any], str]:
        """Parse response based on content type."""
        content_type = response.headers.get("content-type", "").lower()

        try:
            if "json" in content_type:
                return await response.json()
            elif "xml" in content_type:
                return await response.text()
            else:
                return await response.text()
        except Exception:
            # If parsing fails, return raw text
            try:
                return await response.text()
            except:
                return {"raw_response": "Unable to parse response"}

    async def with_circuit_breaker(self, func: Callable) -> Any:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute

        Returns:
            Function result
        """
        if self.circuit_breaker:
            return await self.circuit_breaker.call_with_breaker(func)
        else:
            return await func()

    async def handle_graceful_degradation(
        self,
        primary_func: Callable,
        fallback_func: Optional[Callable] = None,
        cache_key: Optional[str] = None,
    ) -> Any:
        """Handle graceful degradation with fallback strategies.

        Args:
            primary_func: Primary function to execute
            fallback_func: Fallback function if primary fails
            cache_key: Cache key for cached fallback

        Returns:
            Result from primary function or fallback
        """
        try:
            # Try primary function
            result = await primary_func()
            return result

        except APIError as e:
            # Log degradation event
            fallback_strategy = "none"
            data_quality = None

            # Try cache fallback first
            if cache_key and self.cache_manager:
                try:
                    cached_result = await self.cache_manager.get(cache_key)
                    if cached_result:
                        fallback_strategy = "cache"
                        data_quality = 0.7  # Cached data quality

                        self.logger.info(
                            f"Degradation fallback: {fallback_strategy}, reason: {str(e)}"
                        )

                        return cached_result
                except Exception:
                    pass

            # Try fallback function
            if fallback_func:
                try:
                    fallback_result = await fallback_func()
                    fallback_strategy = "fallback_service"
                    data_quality = 0.5  # Fallback service quality

                    self.logger.info(
                        f"Degradation fallback: {fallback_strategy}, reason: {str(e)}"
                    )

                    return fallback_result
                except Exception:
                    pass

            # No fallback available, re-raise original error
            self.logger.error(f"Degradation failed: {str(e)}")

            raise e

    async def make_request_with_correlation(
        self,
        url: str,
        correlation_id: str,
        method: str = "GET",
        priority: int = 1,
        endpoint: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """Make HTTP request with correlation logging.

        Args:
            url: URL to request
            correlation_id: Correlation ID for request tracking
            method: HTTP method
            priority: Request priority
            endpoint: Endpoint identifier
            **kwargs: Additional request parameters

        Returns:
            Response data
        """
        if self.correlation_logger:
            await self.correlation_logger.log_with_correlation(
                correlation_id=correlation_id,
                level="info",
                message=f"Starting {method} request to {url}",
                context={
                    "service": self.service_name,
                    "endpoint": endpoint,
                    "priority": priority,
                },
            )

        try:
            result = await self.make_request(
                url=url, method=method, priority=priority, endpoint=endpoint, **kwargs
            )

            if self.correlation_logger:
                await self.correlation_logger.log_with_correlation(
                    correlation_id=correlation_id,
                    level="info",
                    message=f"Completed {method} request to {url}",
                    context={"service": self.service_name, "status": "success"},
                )

            return result

        except Exception as e:
            if self.correlation_logger:
                await self.correlation_logger.log_with_correlation(
                    correlation_id=correlation_id,
                    level="error",
                    message=f"Failed {method} request to {url}",
                    context={"service": self.service_name, "error": str(e)},
                    error_details={
                        "exception_type": type(e).__name__,
                        "message": str(e),
                    },
                )
            raise

    async def make_request_with_tracing(
        self,
        url: str,
        operation: str,
        method: str = "GET",
        priority: int = 1,
        endpoint: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """Make HTTP request with execution tracing.

        Args:
            url: URL to request
            operation: Operation name for tracing
            method: HTTP method
            priority: Request priority
            endpoint: Endpoint identifier
            **kwargs: Additional request parameters

        Returns:
            Response data
        """
        trace_id = None
        if self.execution_tracer:
            trace_id = await self.execution_tracer.start_trace(
                operation=operation,
                context={"url": url, "service": self.service_name, "method": method},
            )

        try:
            if self.execution_tracer and trace_id:
                await self.execution_tracer.add_trace_step(
                    trace_id=trace_id,
                    step_name="request_start",
                    step_data={"endpoint": endpoint, "priority": priority},
                )

            result = await self.make_request(
                url=url, method=method, priority=priority, endpoint=endpoint, **kwargs
            )

            if self.execution_tracer and trace_id:
                await self.execution_tracer.end_trace(
                    trace_id=trace_id,
                    status="success",
                    result={"response_type": type(result).__name__},
                )

            return result

        except Exception as e:
            if self.execution_tracer and trace_id:
                await self.execution_tracer.end_trace(
                    trace_id=trace_id, status="error", error=e
                )
            raise

    async def create_error_context(
        self,
        error: Exception,
        correlation_id: str,
        user_message: str,
        trace_data: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
    ) -> ErrorContext:
        """Create enhanced ErrorContext from exception.

        Args:
            error: Exception that occurred
            correlation_id: Correlation ID for tracking
            user_message: User-friendly error message
            trace_data: Additional trace data
            severity: Error severity level

        Returns:
            ErrorContext with enhanced information
        """
        error_context = ErrorContext(
            user_message=user_message,
            debug_info=f"{type(error).__name__}: {str(error)}",
            trace_data=trace_data or {},
            correlation_id=correlation_id,
            severity=severity,
        )

        # Add breadcrumb for error creation
        error_context.add_breadcrumb(
            "error_context_created",
            {
                "service": self.service_name,
                "error_type": type(error).__name__,
                "severity": severity.value,
            },
        )

        # Log error with correlation
        if self.correlation_logger:
            await self.correlation_logger.log_with_correlation(
                correlation_id=correlation_id,
                level="error",
                message=f"Error context created: {user_message}",
                context={
                    "service": self.service_name,
                    "error_type": type(error).__name__,
                    "severity": severity.value,
                },
                error_details={
                    "debug_info": error_context.debug_info,
                    "trace_data": trace_data,
                },
            )

        return error_context

    async def handle_enhanced_graceful_degradation(
        self,
        primary_func: Callable,
        correlation_id: str,
        context: Dict[str, Any],
        fallback_func: Optional[Callable] = None,
        cache_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle graceful degradation with enhanced strategy cascade.

        Args:
            primary_func: Primary function to execute
            correlation_id: Correlation ID for tracking
            context: Context information for degradation
            fallback_func: Optional fallback function
            cache_key: Optional cache key

        Returns:
            Result with degradation metadata
        """
        # Add correlation and service info to context
        enhanced_context = {
            **context,
            "correlation_id": correlation_id,
            "service": self.service_name,
            "cache_key": cache_key,
            "cache_manager": self.cache_manager,
            "fallback_func": fallback_func,
            "primary_func": primary_func,
        }

        try:
            # Try primary function first
            result = await primary_func()
            return {"data": result, "strategy": "primary", "quality_score": 1.0}

        except Exception as e:
            # Log degradation attempt
            if self.correlation_logger:
                await self.correlation_logger.log_with_correlation(
                    correlation_id=correlation_id,
                    level="warning",
                    message=f"Primary function failed, attempting degradation: {str(e)}",
                    context={
                        "service": self.service_name,
                        "operation": context.get("operation"),
                    },
                )

            # Execute enhanced degradation cascade
            degradation_result = await GracefulDegradation.execute_degradation_cascade(
                enhanced_context
            )

            # Log successful degradation
            if self.correlation_logger:
                await self.correlation_logger.log_with_correlation(
                    correlation_id=correlation_id,
                    level="info",
                    message=f"Degradation successful with strategy: {degradation_result.get('strategy')}",
                    context={
                        "service": self.service_name,
                        "strategy": degradation_result.get("strategy"),
                        "quality_score": degradation_result.get("quality_score"),
                    },
                )

            return degradation_result

    async def make_request_with_enhanced_error_handling(
        self,
        url: str,
        correlation_id: str,
        method: str = "GET",
        priority: int = 1,
        endpoint: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """Make HTTP request with full enhanced error handling.

        Args:
            url: URL to request
            correlation_id: Correlation ID for tracking
            method: HTTP method
            priority: Request priority
            endpoint: Endpoint identifier
            **kwargs: Additional request parameters

        Returns:
            Response data

        Raises:
            APIError: Enhanced with correlation and trace context
        """
        trace_id = None
        if self.execution_tracer:
            trace_id = await self.execution_tracer.start_trace(
                operation="enhanced_api_request",
                context={
                    "url": url,
                    "service": self.service_name,
                    "correlation_id": correlation_id,
                    "method": method,
                },
            )

        try:
            # Log request start
            if self.correlation_logger:
                await self.correlation_logger.log_with_correlation(
                    correlation_id=correlation_id,
                    level="info",
                    message=f"Enhanced {method} request to {url}",
                    context={"service": self.service_name, "trace_id": trace_id},
                )

            result = await self.make_request(
                url=url, method=method, priority=priority, endpoint=endpoint, **kwargs
            )

            # Complete successful trace
            if self.execution_tracer and trace_id:
                await self.execution_tracer.end_trace(
                    trace_id=trace_id,
                    status="success",
                    result={
                        "response_size": len(str(result)),
                        "response_type": type(result).__name__,
                    },
                )

            return result

        except Exception as e:
            # Create enhanced error context
            error_context = await self.create_error_context(
                error=e,
                correlation_id=correlation_id,
                user_message=f"Request to {self.service_name} failed",
                trace_data={"url": url, "method": method, "endpoint": endpoint},
                severity=(
                    ErrorSeverity.ERROR
                    if isinstance(e, APIError)
                    else ErrorSeverity.CRITICAL
                ),
            )

            # Complete error trace
            if self.execution_tracer and trace_id:
                await self.execution_tracer.end_trace(
                    trace_id=trace_id, status="error", error=e
                )

            raise e

    async def make_request_with_correlation_chain(
        self,
        url: str,
        parent_correlation_id: Optional[str] = None,
        method: str = "GET",
        priority: int = 1,
        endpoint: str = "",
        operation_type: Optional[str] = None,
        service_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make HTTP request with enhanced correlation ID chaining.

        Features:
        - HTTP header propagation for distributed tracing
        - Chain depth calculation and tracking  
        - Integration with execution tracer
        - Performance metrics collection
        - Enhanced metadata enrichment

        Args:
            url: URL to request
            parent_correlation_id: Parent correlation ID for chaining
            method: HTTP method
            priority: Request priority
            endpoint: Endpoint identifier
            operation_type: Type of operation for tracing
            service_name: Override service name for this request
            **kwargs: Additional request parameters

        Returns:
            Response data
        """
        from datetime import datetime, timezone
        import uuid

        # Performance tracking
        start_time = datetime.now(timezone.utc)
        
        # Generate new correlation ID for this request
        child_correlation_id = f"req-{uuid.uuid4().hex[:12]}"
        
        # Calculate chain depth
        chain_depth = 0 if parent_correlation_id is None else 1
        
        # Generate request sequence number for ordering
        import time
        request_sequence = int(time.time() * 1000000)  # microsecond precision
        
        # Get or override service name
        current_service_name = service_name or self.service_name
        
        # Prepare correlation headers for HTTP propagation
        correlation_headers = {
            "X-Correlation-ID": child_correlation_id,
            "X-Request-Chain-Depth": str(chain_depth),
        }
        
        if parent_correlation_id:
            correlation_headers["X-Parent-Correlation-ID"] = parent_correlation_id
            
        # Start execution tracing if available
        trace_id = None
        if self.execution_tracer:
            trace_operation = f"chained_{operation_type}" if operation_type else f"chained_{method.lower()}_request"
            trace_id = await self.execution_tracer.start_trace(
                operation=trace_operation,
                context={
                    "url": url,
                    "method": method,
                    "parent_correlation_id": parent_correlation_id,
                    "correlation_id": child_correlation_id,
                    "chain_depth": chain_depth,
                    "service": current_service_name,
                    "operation_type": operation_type,
                }
            )
            if trace_id:
                correlation_headers["X-Trace-ID"] = trace_id

        # Merge correlation headers with existing headers
        headers = kwargs.get("headers", {})
        headers.update(correlation_headers)
        kwargs["headers"] = headers

        # Enhanced correlation logging with metadata
        if self.correlation_logger:
            log_context = {
                "service": current_service_name,
                "endpoint": endpoint,
                "parent_correlation_id": parent_correlation_id,
                "operation_type": operation_type,
                "chain_depth": chain_depth,
                "request_sequence": request_sequence,
                "trace_id": trace_id,
            }
            
            await self.correlation_logger.log_with_correlation(
                correlation_id=child_correlation_id,
                level="info",
                message=f"Chained {method} request to {url}",
                context=log_context,
                parent_correlation_id=parent_correlation_id,
            )

        try:
            result = await self.make_request(
                url=url, method=method, priority=priority, endpoint=endpoint, **kwargs
            )

            # Calculate performance metrics
            end_time = datetime.now(timezone.utc)
            request_duration_ms = (end_time - start_time).total_seconds() * 1000

            # Enhanced completion logging with performance metrics
            if self.correlation_logger:
                completion_context = {
                    "service": current_service_name,
                    "status": "success",
                    "parent_correlation_id": parent_correlation_id,
                    "operation_type": operation_type,
                    "chain_depth": chain_depth,
                    "request_duration_ms": request_duration_ms,
                    "chain_sequence_number": request_sequence,
                    "trace_id": trace_id,
                }
                
                await self.correlation_logger.log_with_correlation(
                    correlation_id=child_correlation_id,
                    level="info",
                    message=f"Chained request completed successfully",
                    context=completion_context,
                    parent_correlation_id=parent_correlation_id,
                )

            # End execution tracing if started
            if self.execution_tracer and trace_id:
                await self.execution_tracer.end_trace(
                    trace_id=trace_id,
                    status="success",
                    result={"correlation_id": child_correlation_id, "duration_ms": request_duration_ms}
                )

            return result

        except Exception as e:
            # Calculate duration even on error
            end_time = datetime.now(timezone.utc)
            request_duration_ms = (end_time - start_time).total_seconds() * 1000

            # Enhanced error logging with chain context
            if self.correlation_logger:
                error_context = {
                    "service": current_service_name,
                    "error": str(e),
                    "parent_correlation_id": parent_correlation_id,
                    "operation_type": operation_type,
                    "chain_depth": chain_depth,
                    "request_duration_ms": request_duration_ms,
                    "error_type": type(e).__name__,
                    "trace_id": trace_id,
                }
                
                await self.correlation_logger.log_with_correlation(
                    correlation_id=child_correlation_id,
                    level="error",
                    message=f"Chained request failed: {str(e)}",
                    context=error_context,
                    parent_correlation_id=parent_correlation_id,
                )

            # End execution tracing with error
            if self.execution_tracer and trace_id:
                await self.execution_tracer.end_trace(
                    trace_id=trace_id,
                    status="error",
                    error=e
                )

            raise
