"""
api/main.py
Stage 5 FastAPI Integration Engine

This module implements the complete REST API layer for Stage 5 processing,
providing enterprise-grade HTTP endpoints for both Stage 5.1 (complexity analysis)
and Stage 5.2 (solver selection) with comprehensive error handling, validation,
and structured logging.

API Architecture:
- POST /stage5/1/analyze: Execute Stage 5.1 complexity parameter analysis
- POST /stage5/2/select: Execute Stage 5.2 solver selection optimization
- GET /health: System health check and readiness verification
- GET /stage5/info: Stage capabilities and version information

The API follows enterprise patterns with:
- Pydantic models for comprehensive request/response validation
- Structured JSON logging for production monitoring and debugging
- Comprehensive error handling with detailed context for troubleshooting
- Atomic operations with rollback capabilities on failure
- Performance monitoring with execution time tracking
- Schema validation ensuring compliance with foundational design specifications

Integration Points:
- Input: HTTP requests with file paths and configuration parameters
- Processing: Orchestrates stage_5_1 and stage_5_2 module execution
- Output: JSON responses with results or detailed error information
- Logging: Structured JSON logs for monitoring and audit trails

Performance Characteristics:
- Request validation: O(1) with Pydantic model validation
- Processing delegation: Direct pass-through to computational modules
- Response serialization: O(n) where n = result size
- Concurrent handling: AsyncIO-based for high throughput
- Memory efficiency: Streaming responses for large result sets

For detailed theoretical foundations and API specifications, see:
- Stage5-FOUNDATIONAL-DESIGN-IMPLEMENTATION-PLAN.md
- JSON schema definitions in api/schemas.py
- Integration contracts in stage_5_1 and stage_5_2 modules
"""

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
import traceback

from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn

from .schemas import (
    Stage51AnalysisRequest, Stage51AnalysisResponse,
    Stage52SelectionRequest, Stage52SelectionResponse,
    HealthResponse, Stage5InfoResponse,
    ErrorResponse, ValidationErrorResponse
)
from ..common.logging import get_logger, setup_structured_logging
from ..common.exceptions import (
    Stage5ValidationError, Stage5ComputationError, Stage5PerformanceError
)
from ..common.config import Stage5Config
from ..stage_5_1.runner import run_stage_5_1_complete
from ..stage_5_2.runner import run_stage_5_2_complete

# Global configuration and logging setup
config = Stage5Config()
logger = get_logger("stage5_api")

# API metadata and versioning aligned with foundational design
API_VERSION = "1.0.0"
API_TITLE = "Stage 5 Complexity Analysis & Solver Selection API"
API_DESCRIPTION = """
Enterprise-grade REST API for Stage 5 processing in the HEI Timetabling Engine.

**Stage 5.1 - Complexity Analysis**: Computes 16-parameter complexity vector from Stage 3 outputs
**Stage 5.2 - Solver Selection**: Performs L2 normalization + LP weight learning for optimal solver selection

Features:
- Mathematical rigor with exact theoretical framework implementation
- Enterprise-grade error handling with structured validation
- Comprehensive audit logging for production monitoring
- Performance tracking with execution time measurement
- Schema-compliant JSON I/O matching foundational design specifications
"""

# Application state and performance monitoring
app_state = {
    "startup_time": None,
    "request_count": 0,
    "error_count": 0,
    "stage_5_1_executions": 0,
    "stage_5_2_executions": 0,
    "average_execution_time": 0.0,
    "last_execution_time": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management for startup and shutdown procedures.
    
    Handles:
    - Structured logging initialization
    - Configuration validation
    - Performance monitoring setup
    - Resource cleanup on shutdown
    """
    # Startup procedures
    logger.info("Starting Stage 5 API server...")
    
    # Initialize structured logging
    setup_structured_logging(
        level=config.log_level,
        json_format=config.json_logs
    )
    
    # Validate configuration
    try:
        config.validate_configuration()
        logger.info("Configuration validation successful")
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise
    
    # Set startup time for health checks
    app_state["startup_time"] = datetime.now(timezone.utc)
    
    logger.info(
        f"Stage 5 API server started successfully: version={API_VERSION}, "
        f"config_valid=True, logging_initialized=True"
    )
    
    yield
    
    # Shutdown procedures
    logger.info("Shutting down Stage 5 API server...")
    logger.info(
        f"Final statistics: requests={app_state['request_count']}, "
        f"errors={app_state['error_count']}, "
        f"avg_execution_time={app_state['average_execution_time']:.3f}s"
    )


# Initialize FastAPI application with enterprise configuration
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS middleware for development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
if config.allowed_hosts:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=config.allowed_hosts
    )


async def update_performance_metrics(execution_time: float):
    """
    Update global performance metrics with new execution time.
    
    Args:
        execution_time: Execution time in seconds for metric aggregation
    """
    app_state["last_execution_time"] = execution_time
    
    # Update moving average
    total_executions = app_state["stage_5_1_executions"] + app_state["stage_5_2_executions"]
    if total_executions > 0:
        # Exponential moving average for performance smoothing
        alpha = 0.1
        app_state["average_execution_time"] = (
            alpha * execution_time + 
            (1 - alpha) * app_state["average_execution_time"]
        )


async def log_request_info(request: Request, response_time: Optional[float] = None):
    """
    Log structured request information for monitoring and debugging.
    
    Args:
        request: FastAPI Request object with client and endpoint information
        response_time: Optional response time for performance logging
    """
    app_state["request_count"] += 1
    
    log_data = {
        "method": request.method,
        "url": str(request.url),
        "client": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown"),
        "request_id": app_state["request_count"],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    if response_time is not None:
        log_data["response_time_ms"] = int(response_time * 1000)
    
    logger.info("API request processed", extra=log_data)


async def handle_stage_error(error: Exception, stage: str, context: Dict[str, Any]) -> ErrorResponse:
    """
    Handle Stage 5 errors with comprehensive logging and structured responses.
    
    Args:
        error: Exception that occurred during processing
        stage: Stage identifier (5.1 or 5.2) for context
        context: Additional context information for debugging
        
    Returns:
        ErrorResponse: Structured error response with debugging information
    """
    app_state["error_count"] += 1
    
    error_context = {
        "stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "traceback": traceback.format_exc() if config.debug_mode else None
    }
    
    logger.error(f"Stage {stage} execution failed", extra=error_context)
    
    # Map internal exceptions to appropriate HTTP status codes
    if isinstance(error, Stage5ValidationError):
        status_code = 400
        error_type = "validation_error"
        error_detail = "Input validation failed - check file paths and data formats"
    elif isinstance(error, Stage5ComputationError):
        status_code = 422
        error_type = "computation_error"  
        error_detail = "Mathematical computation failed - check data quality and algorithm parameters"
    elif isinstance(error, Stage5PerformanceError):
        status_code = 408
        error_type = "performance_error"
        error_detail = "Execution exceeded performance limits - reduce problem size or increase timeouts"
    else:
        status_code = 500
        error_type = "internal_error"
        error_detail = "Internal server error during processing"
    
    return ErrorResponse(
        error=error_type,
        detail=error_detail,
        message=str(error),
        context=context if config.debug_mode else {},
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    System health check endpoint for monitoring and readiness verification.
    
    Returns system status, uptime, performance metrics, and configuration information
    for monitoring systems and load balancers.
    
    Returns:
        HealthResponse: Comprehensive system health information
    """
    try:
        # Calculate uptime
        if app_state["startup_time"]:
            uptime_seconds = (datetime.now(timezone.utc) - app_state["startup_time"]).total_seconds()
        else:
            uptime_seconds = 0.0
        
        # Validate critical components
        stage_5_1_ready = True
        stage_5_2_ready = True
        
        try:
            from ..stage_5_1 import get_module_status
            stage_5_1_status = get_module_status()
            stage_5_1_ready = stage_5_1_status.get("ready_for_production", False)
        except Exception:
            stage_5_1_ready = False
        
        try:
            from ..stage_5_2 import get_module_status
            stage_5_2_status = get_module_status()
            stage_5_2_ready = stage_5_2_status.get("ready_for_production", False)
        except Exception:
            stage_5_2_ready = False
        
        # Determine overall system health
        system_healthy = stage_5_1_ready and stage_5_2_ready
        
        health_response = HealthResponse(
            status="healthy" if system_healthy else "degraded",
            uptime_seconds=uptime_seconds,
            version=API_VERSION,
            stage_5_1_ready=stage_5_1_ready,
            stage_5_2_ready=stage_5_2_ready,
            request_count=app_state["request_count"],
            error_count=app_state["error_count"],
            average_execution_time=app_state["average_execution_time"],
            last_execution_time=app_state["last_execution_time"],
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        return health_response
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Health check system failure"
        )


@app.get("/stage5/info", response_model=Stage5InfoResponse, tags=["Information"])
async def get_stage5_info():
    """
    Get comprehensive Stage 5 capabilities and version information.
    
    Provides detailed information about supported operations, mathematical framework
    compliance, performance characteristics, and integration specifications.
    
    Returns:
        Stage5InfoResponse: Complete Stage 5 system information
    """
    try:
        # Get module information
        try:
            from ..stage_5_1 import get_stage_info as get_stage_5_1_info
            stage_5_1_info = get_stage_5_1_info()
        except Exception:
            stage_5_1_info = {"available": False}
        
        try:
            from ..stage_5_2 import get_stage_info as get_stage_5_2_info
            stage_5_2_info = get_stage_5_2_info()
        except Exception:
            stage_5_2_info = {"available": False}
        
        info_response = Stage5InfoResponse(
            api_version=API_VERSION,
            stage_5_1_info=stage_5_1_info,
            stage_5_2_info=stage_5_2_info,
            supported_endpoints=[
                "/stage5/1/analyze",
                "/stage5/2/select",
                "/health",
                "/stage5/info"
            ],
            supported_input_formats=[
                "L_raw.parquet",
                "L_rel.graphml", 
                "L_idx.{pkl,parquet,feather,idx,bin}",
                "complexity_metrics.json",
                "solver_capabilities.json"
            ],
            supported_output_formats=[
                "complexity_metrics.json",
                "selection_decision.json"
            ],
            mathematical_framework="Stage-5-FOUNDATIONAL-DESIGN-IMPLEMENTATION-PLAN.md",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        return info_response
        
    except Exception as e:
        logger.error(f"Stage 5 info retrieval failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="System information retrieval failure"
        )


@app.post("/stage5/1/analyze", 
         response_model=Stage51AnalysisResponse, 
         responses={
             400: {"model": ValidationErrorResponse},
             422: {"model": ErrorResponse},
             408: {"model": ErrorResponse},
             500: {"model": ErrorResponse}
         },
         tags=["Stage 5.1"])
async def execute_stage_5_1_analysis(request: Stage51AnalysisRequest, http_request: Request):
    """
    Execute Stage 5.1 complexity parameter analysis.
    
    Computes 16-parameter complexity vector from Stage 3 outputs using exact
    mathematical definitions from the theoretical framework. Performs comprehensive
    analysis including dimensionality, constraint density, faculty specialization,
    and other complexity metrics.
    
    Args:
        request: Stage51AnalysisRequest with input file paths and configuration
        http_request: FastAPI Request object for logging context
        
    Returns:
        Stage51AnalysisResponse: Complete complexity analysis results
        
    Raises:
        HTTPException: Various error conditions with detailed context
        
    Mathematical Processing:
    - Parameter computation using formal definitions P1-P16
    - Composite index calculation with statistical aggregation
    - Metadata extraction and validation
    - JSON schema compliance verification
    """
    start_time = time.perf_counter()
    
    try:
        await log_request_info(http_request)
        
        logger.info(
            f"Starting Stage 5.1 analysis: l_raw={request.l_raw_path}, "
            f"l_rel={request.l_rel_path}, l_idx={request.l_idx_path}"
        )
        
        # Convert request paths to Path objects
        l_raw_path = Path(request.l_raw_path)
        l_rel_path = Path(request.l_rel_path)  
        l_idx_path = Path(request.l_idx_path)
        output_dir = Path(request.output_dir)
        
        # Validate input files exist
        for path_name, path in [
            ("l_raw_path", l_raw_path),
            ("l_rel_path", l_rel_path), 
            ("l_idx_path", l_idx_path)
        ]:
            if not path.exists():
                raise Stage5ValidationError(
                    f"Input file does not exist: {path}",
                    validation_type="file_existence",
                    field_name=path_name,
                    actual_value=str(path)
                )
        
        # Execute Stage 5.1 computation
        results = await asyncio.create_task(
            asyncio.to_thread(
                run_stage_5_1_complete,
                l_raw_path=l_raw_path,
                l_rel_path=l_rel_path,
                l_idx_path=l_idx_path,
                output_dir=output_dir,
                config_overrides=request.config_overrides or {}
            )
        )
        
        execution_time = time.perf_counter() - start_time
        app_state["stage_5_1_executions"] += 1
        await update_performance_metrics(execution_time)
        
        # Create successful response
        response = Stage51AnalysisResponse(
            success=True,
            complexity_metrics=results.complexity_metrics.dict(),
            output_files=[str(path) for path in results.output_files],
            execution_time_ms=int(execution_time * 1000),
            metadata=results.execution_metadata,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        logger.info(
            f"Stage 5.1 analysis completed successfully: execution_time={execution_time:.3f}s, "
            f"composite_index={results.complexity_metrics.composite_index:.6f}"
        )
        
        await log_request_info(http_request, execution_time)
        
        return response
        
    except Exception as e:
        execution_time = time.perf_counter() - start_time
        await log_request_info(http_request, execution_time)
        
        error_context = {
            "l_raw_path": request.l_raw_path,
            "l_rel_path": request.l_rel_path,
            "l_idx_path": request.l_idx_path,
            "output_dir": request.output_dir,
            "execution_time": execution_time
        }
        
        error_response = await handle_stage_error(e, "5.1", error_context)
        
        # Map error types to appropriate HTTP status codes
        if isinstance(e, Stage5ValidationError):
            raise HTTPException(status_code=400, detail=error_response.dict())
        elif isinstance(e, Stage5ComputationError):
            raise HTTPException(status_code=422, detail=error_response.dict())
        elif isinstance(e, Stage5PerformanceError):
            raise HTTPException(status_code=408, detail=error_response.dict())
        else:
            raise HTTPException(status_code=500, detail=error_response.dict())


@app.post("/stage5/2/select",
         response_model=Stage52SelectionResponse,
         responses={
             400: {"model": ValidationErrorResponse},
             422: {"model": ErrorResponse},
             408: {"model": ErrorResponse}, 
             500: {"model": ErrorResponse}
         },
         tags=["Stage 5.2"])
async def execute_stage_5_2_selection(request: Stage52SelectionRequest, http_request: Request):
    """
    Execute Stage 5.2 solver selection optimization.
    
    Performs L2 normalization and LP-based weight learning for optimal solver selection
    using the two-stage optimization framework from theoretical foundations.
    
    Args:
        request: Stage52SelectionRequest with complexity metrics and solver capabilities
        http_request: FastAPI Request object for logging context
        
    Returns:
        Stage52SelectionResponse: Complete solver selection results
        
    Raises:
        HTTPException: Various error conditions with detailed context
        
    Mathematical Processing:
    - L2 parameter normalization with boundedness guarantees
    - LP-based weight learning with convergence proofs
    - Optimal solver selection with confidence scoring
    - Complete ranking generation with margin analysis
    """
    start_time = time.perf_counter()
    
    try:
        await log_request_info(http_request)
        
        logger.info(
            f"Starting Stage 5.2 selection: complexity_metrics={request.complexity_metrics_path}, "
            f"solver_capabilities={request.solver_capabilities_path}"
        )
        
        # Convert request paths to Path objects
        complexity_metrics_path = Path(request.complexity_metrics_path)
        solver_capabilities_path = Path(request.solver_capabilities_path)
        output_dir = Path(request.output_dir)
        
        # Validate input files exist
        for path_name, path in [
            ("complexity_metrics_path", complexity_metrics_path),
            ("solver_capabilities_path", solver_capabilities_path)
        ]:
            if not path.exists():
                raise Stage5ValidationError(
                    f"Input file does not exist: {path}",
                    validation_type="file_existence",
                    field_name=path_name,
                    actual_value=str(path)
                )
        
        # Execute Stage 5.2 optimization
        results = await asyncio.create_task(
            asyncio.to_thread(
                run_stage_5_2_complete,
                complexity_metrics_path=complexity_metrics_path,
                solver_capabilities_path=solver_capabilities_path,
                output_dir=output_dir,
                config_overrides=request.config_overrides or {}
            )
        )
        
        execution_time = time.perf_counter() - start_time
        app_state["stage_5_2_executions"] += 1
        await update_performance_metrics(execution_time)
        
        # Create successful response
        response = Stage52SelectionResponse(
            success=True,
            selection_decision=results.selection_decision.dict(),
            output_files=[str(path) for path in results.output_files],
            execution_time_ms=int(execution_time * 1000),
            metadata=results.execution_metadata,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        logger.info(
            f"Stage 5.2 selection completed successfully: execution_time={execution_time:.3f}s, "
            f"chosen_solver={results.selection_decision.chosen_solver.solver_id}, "
            f"confidence={results.selection_decision.chosen_solver.confidence:.4f}"
        )
        
        await log_request_info(http_request, execution_time)
        
        return response
        
    except Exception as e:
        execution_time = time.perf_counter() - start_time
        await log_request_info(http_request, execution_time)
        
        error_context = {
            "complexity_metrics_path": request.complexity_metrics_path,
            "solver_capabilities_path": request.solver_capabilities_path,
            "output_dir": request.output_dir,
            "execution_time": execution_time
        }
        
        error_response = await handle_stage_error(e, "5.2", error_context)
        
        # Map error types to appropriate HTTP status codes
        if isinstance(e, Stage5ValidationError):
            raise HTTPException(status_code=400, detail=error_response.dict())
        elif isinstance(e, Stage5ComputationError):
            raise HTTPException(status_code=422, detail=error_response.dict())
        elif isinstance(e, Stage5PerformanceError):
            raise HTTPException(status_code=408, detail=error_response.dict())
        else:
            raise HTTPException(status_code=500, detail=error_response.dict())


# Custom exception handlers for comprehensive error management
@app.exception_handler(Stage5ValidationError)
async def validation_error_handler(request: Request, exc: Stage5ValidationError):
    """Handle Stage 5 validation errors with structured responses."""
    error_response = await handle_stage_error(exc, "validation", {"url": str(request.url)})
    return JSONResponse(
        status_code=400,
        content=error_response.dict()
    )


@app.exception_handler(Stage5ComputationError)
async def computation_error_handler(request: Request, exc: Stage5ComputationError):
    """Handle Stage 5 computation errors with structured responses."""
    error_response = await handle_stage_error(exc, "computation", {"url": str(request.url)})
    return JSONResponse(
        status_code=422,
        content=error_response.dict()
    )


@app.exception_handler(Stage5PerformanceError) 
async def performance_error_handler(request: Request, exc: Stage5PerformanceError):
    """Handle Stage 5 performance errors with structured responses."""
    error_response = await handle_stage_error(exc, "performance", {"url": str(request.url)})
    return JSONResponse(
        status_code=408,
        content=error_response.dict()
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Handle internal server errors with structured responses."""
    error_response = await handle_stage_error(exc, "internal", {"url": str(request.url)})
    return JSONResponse(
        status_code=500,
        content=error_response.dict()
    )


# Development server entry point
def run_development_server():
    """
    Run development server with hot reload and debugging enabled.
    
    This function provides a development entry point with appropriate
    configuration for local testing and development workflows.
    """
    logger.info("Starting Stage 5 API development server...")
    
    uvicorn.run(
        "api.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug_mode,
        log_level=config.log_level.lower(),
        access_log=True,
        use_colors=not config.json_logs
    )


# Production server entry point
def run_production_server():
    """
    Run production server with optimized configuration.
    
    This function provides a production entry point with appropriate
    configuration for deployment environments.
    """
    logger.info("Starting Stage 5 API production server...")
    
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        access_log=True,
        use_colors=False,
        loop="asyncio",
        workers=1  # Single worker for prototype deployment
    )


if __name__ == "__main__":
    # Auto-detect development vs production mode
    import sys
    
    if "--dev" in sys.argv or config.debug_mode:
        run_development_server()
    else:
        run_production_server()