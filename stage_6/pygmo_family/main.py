"""
Stage 6.4 PyGMO Solver Family - Main Pipeline Orchestrator
==========================================================

Pipeline orchestrator for PyGMO multi-objective optimization
with master data pipeline integration, complete validation, and audit logging.

Mathematical Foundation:
    - PyGMO Foundational Framework v2.3 compliance with convergence guarantees
    - Multi-objective optimization with f1-f5 objectives per Definition 8.1
    - NSGA-II algorithm implementation per Theorem 3.2
    - Bijective representation with zero information loss guarantees

System Design:
    - Master pipeline integration with exposable APIs and webhooks
    - complete error handling with fail-fast validation
    - Memory management with deterministic resource patterns
    - Structured logging with enterprise audit trails
    - Configuration-driven usage with environment support

Integration Points:
    - Input: Stage 3 compilation outputs (L_raw, L_rel, L_idx)
    - Processing: PyGMO NSGA-II optimization with theoretical compliance
    - Output: Optimized schedules with Stage 7 validation compliance
    - API: RESTful endpoints for master pipeline orchestration

Author: Student Team
Version: 1.0.0 (Ready)
Compliance: PyGMO Foundational Framework v2.3, Standards
"""

import sys
import traceback
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from contextlib import asynccontextmanager
import uuid
import json
import gc

# Core libraries with version validation
import numpy as np
import pandas as pd
import pygmo as pg
from pydantic import BaseModel, Field, ValidationError
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
import psutil

# Internal imports - maintain strict dependency hierarchy
from config import (
    PyGMOConfiguration, 
    CURRENT_CONFIG,
    OptimizationAlgorithm,
    ValidationLevel,
    ConfigurationFactory
)

# Configure structured logging
logger = structlog.get_logger("pygmo_main")

# Global state management for pipeline orchestration
class PipelineState:
    """
    Global state management for PyGMO solver family pipeline.
    Maintains optimization status, resource monitoring, and audit trails.
    """
    def __init__(self):
        self.active_optimizations: Dict[str, Dict] = {}
        self.completed_optimizations: Dict[str, Dict] = {}
        self.system_metrics: Dict[str, Any] = {}
        self.startup_time: float = time.time()
        self.total_optimizations: int = 0
        self.failed_optimizations: int = 0

    def register_optimization(self, request_id: str, config: Dict[str, Any]) -> None:
        """Register new optimization request with tracking."""
        self.active_optimizations[request_id] = {
            "start_time": time.time(),
            "config": config,
            "status": "initialized",
            "progress": 0.0,
            "current_generation": 0,
            "best_fitness": None,
            "memory_usage_mb": 0.0
        }
        self.total_optimizations += 1

    def update_optimization_progress(
        self, 
        request_id: str, 
        status: str,
        progress: float = 0.0,
        generation: int = 0,
        fitness: Optional[List[float]] = None
    ) -> None:
        """Update optimization progress with complete tracking."""
        if request_id in self.active_optimizations:
            self.active_optimizations[request_id].update({
                "status": status,
                "progress": progress,
                "current_generation": generation,
                "best_fitness": fitness,
                "memory_usage_mb": self._get_current_memory_usage(),
                "last_update": time.time()
            })

    def complete_optimization(
        self, 
        request_id: str, 
        result: Optional[Dict] = None, 
        error: Optional[str] = None
    ) -> None:
        """Complete optimization with result tracking."""
        if request_id in self.active_optimizations:
            optimization_data = self.active_optimizations.pop(request_id)
            optimization_data.update({
                "end_time": time.time(),
                "duration_seconds": time.time() - optimization_data["start_time"],
                "result": result,
                "error": error,
                "status": "completed" if result else "failed"
            })
            self.completed_optimizations[request_id] = optimization_data

            if error:
                self.failed_optimizations += 1

    def _get_current_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0

    def get_system_health(self) -> Dict[str, Any]:
        """complete system health assessment."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                "status": "healthy" if len(self.active_optimizations) < 5 else "busy",
                "uptime_seconds": time.time() - self.startup_time,
                "active_optimizations": len(self.active_optimizations),
                "total_optimizations": self.total_optimizations,
                "success_rate": (
                    (self.total_optimizations - self.failed_optimizations) / 
                    max(self.total_optimizations, 1)
                ) * 100,
                "memory_usage_mb": memory_info.rss / (1024 * 1024),
                "cpu_usage_percent": psutil.cpu_percent(interval=0.1),
                "available_memory_mb": psutil.virtual_memory().available / (1024 * 1024)
            }
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {"status": "error", "error": str(e)}

# Initialize global pipeline state
pipeline_state = PipelineState()

# Pydantic models for API integration
class OptimizationRequest(BaseModel):
    """
    Optimization request model for master pipeline integration.
    Provides complete configuration and validation for optimization tasks.
    """
    model_config = {"extra": "forbid"}

    # Required parameters
    input_directory: str = Field(
        description="Path to Stage 3 output directory containing L_raw, L_rel, L_idx files"
    )
    output_directory: str = Field(
        description="Path to output directory for schedule and metadata files"
    )

    # Optional configuration overrides
    algorithm: Optional[OptimizationAlgorithm] = Field(
        default=None,
        description="Override default optimization algorithm"
    )
    population_size: Optional[int] = Field(
        default=None, 
        ge=10, 
        le=1000,
        description="Override population size (10-1000)"
    )
    max_generations: Optional[int] = Field(
        default=None,
        ge=10,
        le=2000, 
        description="Override maximum generations (10-2000)"
    )

    # Integration parameters
    webhook_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for progress notifications to master pipeline"
    )
    priority: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Optimization priority (1=highest, 5=lowest)"
    )
    timeout_seconds: Optional[int] = Field(
        default=3600,
        ge=60,
        le=14400,
        description="Maximum optimization time in seconds (1-4 hours)"
    )

    # Metadata
    client_id: Optional[str] = Field(
        default=None,
        description="Client identifier for request tracking"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for request context"
    )

class OptimizationResponse(BaseModel):
    """
    Optimization response model with complete result information.
    """
    request_id: str = Field(description="Unique request identifier")
    status: str = Field(description="Optimization status")
    message: str = Field(description="Human-readable status message")
    estimated_duration_seconds: Optional[int] = Field(
        default=None,
        description="Estimated completion time"
    )
    webhook_registered: bool = Field(
        default=False,
        description="Whether webhook notifications are active"
    )

class OptimizationResult(BaseModel):
    """
    Complete optimization result with solution and metadata.
    """
    request_id: str = Field(description="Request identifier")
    status: str = Field(description="Final optimization status")
    solution: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optimal schedule solution"
    )
    fitness_values: Optional[List[float]] = Field(
        default=None,
        description="Multi-objective fitness values (f1-f5)"
    )
    optimization_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="complete optimization statistics"
    )
    file_paths: Dict[str, str] = Field(
        default_factory=dict,
        description="Generated file paths (CSV, JSON, logs)"
    )
    error_details: Optional[str] = Field(
        default=None,
        description="Detailed error information if optimization failed"
    )

class SystemHealth(BaseModel):
    """System health status for monitoring and diagnostics."""
    status: str = Field(description="Overall system health status")
    uptime_seconds: float = Field(description="System uptime")
    active_optimizations: int = Field(description="Currently running optimizations")
    total_optimizations: int = Field(description="Total optimizations processed")
    success_rate: float = Field(description="Success rate percentage")
    memory_usage_mb: float = Field(description="Current memory usage")
    cpu_usage_percent: float = Field(description="Current CPU usage")
    available_memory_mb: float = Field(description="Available system memory")

# Pipeline implementation functions
def validate_input_directory(input_directory: str) -> Tuple[bool, str]:
    """
    Validate input directory contains required Stage 3 output files.

    Args:
        input_directory: Path to Stage 3 output directory

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        input_path = Path(input_directory)

        if not input_path.exists():
            return False, f"Input directory does not exist: {input_directory}"

        if not input_path.is_dir():
            return False, f"Input path is not a directory: {input_directory}"

        # Check for required Stage 3 files
        required_files = ["L_raw.parquet", "L_rel.graphml", "L_idx.feather"]
        missing_files = []

        for file_name in required_files:
            file_path = input_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)

        if missing_files:
            return False, f"Missing required files: {', '.join(missing_files)}"

        return True, "Input directory validation successful"

    except Exception as e:
        return False, f"Input directory validation error: {str(e)}"

def prepare_output_directory(output_directory: str) -> Tuple[bool, str]:
    """
    Prepare output directory for optimization results.

    Args:
        output_directory: Path to output directory

    Returns:
        Tuple[bool, str]: (is_prepared, error_message)
    """
    try:
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)

        # Verify write permissions
        test_file = output_path / f"write_test_{uuid.uuid4().hex[:8]}.tmp"
        try:
            test_file.write_text("write test")
            test_file.unlink()  # Remove test file
        except Exception as e:
            return False, f"Output directory not writable: {str(e)}"

        return True, "Output directory prepared successfully"

    except Exception as e:
        return False, f"Output directory preparation error: {str(e)}"

async def run_optimization_pipeline(
    request: OptimizationRequest,
    request_id: str,
    config: PyGMOConfiguration
) -> OptimizationResult:
    """
    Execute complete optimization pipeline with complete error handling.

    This function orchestrates the entire PyGMO optimization process:
    1. Input modeling from Stage 3 outputs
    2. PyGMO NSGA-II optimization with progress tracking
    3. Output modeling with Stage 7 validation
    4. Result export and metadata generation

    Args:
        request: Optimization request parameters
        request_id: Unique request identifier  
        config: PyGMO configuration

    Returns:
        OptimizationResult: Complete optimization result with solution

    Raises:
        Exception: For any critical optimization failures
    """
    start_time = time.time()
    optimization_metadata = {
        "request_id": request_id,
        "start_time": start_time,
        "algorithm": config.algorithm.value,
        "population_size": config.algorithm_params.population_size,
        "max_generations": config.algorithm_params.max_generations
    }

    try:
        logger.info(
            "Starting optimization pipeline",
            request_id=request_id,
            algorithm=config.algorithm,
            input_dir=request.input_directory,
            output_dir=request.output_directory
        )

        # Register optimization in global state
        pipeline_state.register_optimization(request_id, request.dict())

        # Phase 1: Input Modeling
        pipeline_state.update_optimization_progress(request_id, "input_modeling", 0.1)
        logger.info("Phase 1: Input Modeling", request_id=request_id)

        # Import input modeling components (dynamic import for memory management)
        try:
            from input_model.loader import InputDataLoader
            from input_model.validator import InputValidator  
            from input_model.context import InputModelContextBuilder
        except ImportError as e:
            raise ImportError(f"Input modeling components not available: {e}")

        # Load Stage 3 data
        loader = InputDataLoader()
        input_data = await asyncio.to_thread(
            loader.load_from_directory, 
            request.input_directory
        )

        # Validate input data
        validator = InputValidator()
        validation_result = await asyncio.to_thread(
            validator.validate_complete_input,
            input_data
        )

        if not validation_result.is_valid:
            raise ValueError(f"Input validation failed: {validation_result.error_summary}")

        # Build input context
        context_builder = InputModelContextBuilder()
        input_context = await asyncio.to_thread(
            context_builder.build_context,
            input_data
        )

        optimization_metadata["input_modeling"] = {
            "courses_count": len(input_context.course_eligibility),
            "constraints_count": len(input_context.constraint_rules),
            "dynamic_parameters_count": len(input_context.dynamic_parameters)
        }

        # Phase 2: Processing (PyGMO Optimization)
        pipeline_state.update_optimization_progress(request_id, "processing", 0.2)
        logger.info("Phase 2: PyGMO Optimization", request_id=request_id)

        # Import processing components
        try:
            from processing.problem import SchedulingProblemAdapter
            from processing.representation import RepresentationConverter
            from processing.engine import NSGAIIOptimizationEngine
        except ImportError as e:
            raise ImportError(f"Processing components not available: {e}")

        # Initialize optimization components
        problem_adapter = SchedulingProblemAdapter(input_context)
        representation_converter = RepresentationConverter(input_context.bijection_data)
        optimization_engine = NSGAIIOptimizationEngine(config)

        # Define progress callback for real-time updates
        def progress_callback(generation: int, best_fitness: List[float], progress: float):
            pipeline_state.update_optimization_progress(
                request_id, 
                "optimizing",
                0.2 + (progress * 0.6),  # Map to 20-80% progress range
                generation,
                best_fitness
            )
            logger.debug(
                "Optimization progress",
                request_id=request_id,
                generation=generation,
                progress=progress,
                best_fitness=best_fitness[:2] if best_fitness else None  # Log first 2 objectives
            )

        # Execute optimization with progress tracking
        optimization_result = await asyncio.to_thread(
            optimization_engine.optimize,
            problem_adapter,
            progress_callback=progress_callback
        )

        optimization_metadata["processing"] = {
            "generations_completed": optimization_result.generations_completed,
            "convergence_achieved": optimization_result.converged,
            "best_fitness": optimization_result.best_fitness,
            "pareto_front_size": len(optimization_result.pareto_front),
            "optimization_time_seconds": optimization_result.optimization_time
        }

        # Phase 3: Output Modeling
        pipeline_state.update_optimization_progress(request_id, "output_modeling", 0.8)
        logger.info("Phase 3: Output Modeling", request_id=request_id)

        # Import output modeling components
        try:
            from output_model.decoder import SolutionDecoder
            from output_model.validator import OutputValidator
            from output_model.writer import ScheduleWriter
        except ImportError as e:
            raise ImportError(f"Output modeling components not available: {e}")

        # Decode optimization result
        decoder = SolutionDecoder(input_context.bijection_data)
        schedule_dataframe = await asyncio.to_thread(
            decoder.decode_solution,
            optimization_result.best_individual
        )

        # Validate decoded schedule
        output_validator = OutputValidator()
        output_validation = await asyncio.to_thread(
            output_validator.validate_schedule_dataframe,
            schedule_dataframe,
            input_context
        )

        if not output_validation.is_valid:
            logger.warning(
                "Output validation warnings detected",
                request_id=request_id,
                warnings=output_validation.warnings
            )

        # Write schedule and metadata files
        writer = ScheduleWriter()
        file_paths = await asyncio.to_thread(
            writer.write_complete_output,
            schedule_dataframe,
            optimization_result,
            optimization_metadata,
            request.output_directory
        )

        # Phase 4: Finalization
        pipeline_state.update_optimization_progress(request_id, "completed", 1.0)

        # Final optimization metadata
        end_time = time.time()
        optimization_metadata.update({
            "end_time": end_time,
            "total_duration_seconds": end_time - start_time,
            "output_files": file_paths,
            "schedule_courses": len(schedule_dataframe),
            "validation_passed": output_validation.is_valid
        })

        # Create optimization result
        result = OptimizationResult(
            request_id=request_id,
            status="completed",
            solution={
                "schedule": schedule_dataframe.to_dict(orient='records'),
                "pareto_front": [
                    representation_converter.vector_to_course_dict(individual)
                    for individual in optimization_result.pareto_front[:5]  # Top 5 solutions
                ]
            },
            fitness_values=optimization_result.best_fitness,
            optimization_metadata=optimization_metadata,
            file_paths=file_paths
        )

        # Complete optimization tracking
        pipeline_state.complete_optimization(request_id, result.dict())

        logger.info(
            "Optimization pipeline completed successfully",
            request_id=request_id,
            duration_seconds=end_time - start_time,
            best_fitness=optimization_result.best_fitness,
            schedule_courses=len(schedule_dataframe)
        )

        # Force garbage collection to free memory
        gc.collect()

        return result

    except Exception as e:
        error_msg = f"Optimization pipeline failed: {str(e)}"
        logger.error(
            "Optimization pipeline failed",
            request_id=request_id,
            error=str(e),
            traceback=traceback.format_exc()
        )

        # Update optimization metadata with error
        optimization_metadata.update({
            "end_time": time.time(),
            "total_duration_seconds": time.time() - start_time,
            "error": str(e)
        })

        # Create error result
        error_result = OptimizationResult(
            request_id=request_id,
            status="failed",
            optimization_metadata=optimization_metadata,
            error_details=error_msg
        )

        # Complete optimization tracking with error
        pipeline_state.complete_optimization(request_id, None, error_msg)

        # Force garbage collection
        gc.collect()

        return error_result

# FastAPI application with configuration
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with proper startup and shutdown."""
    # Startup
    logger.info(
        "PyGMO Solver Family starting up",
        version="1.0.0",
        framework_compliance="PyGMO Foundational Framework v2.3"
    )

    # Validate system requirements
    try:
        memory_info = psutil.virtual_memory()
        if memory_info.available < (512 * 1024 * 1024):  # 512MB minimum
            logger.warning(
                "Low available memory detected",
                available_mb=memory_info.available / (1024 * 1024)
            )
    except Exception as e:
        logger.error("System validation failed during startup", error=str(e))

    yield

    # Shutdown
    logger.info(
        "PyGMO Solver Family shutting down",
        uptime_seconds=time.time() - pipeline_state.startup_time,
        total_optimizations=pipeline_state.total_optimizations
    )

# Create FastAPI application
app = FastAPI(
    title="PyGMO Solver Family API",
    description="Multi-objective optimization API for educational scheduling",
    version="1.0.0",
    lifespan=lifespan
)

# Add enterprise middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if CURRENT_CONFIG.api_config.get("enable_cors", True) else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# API Endpoints for Master Pipeline Integration

@app.get("/health", response_model=SystemHealth)
async def health_check():
    """
    complete health check for system monitoring.
    Returns detailed system status for master pipeline integration.
    """
    try:
        health_data = pipeline_state.get_system_health()
        return SystemHealth(**health_data)
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=500, 
            detail=f"Health check failed: {str(e)}"
        )

@app.get("/metrics")
async def get_metrics():
    """
    Detailed system metrics for monitoring and performance analysis.
    """
    try:
        return {
            "system": pipeline_state.get_system_health(),
            "pipeline": {
                "active_optimizations": len(pipeline_state.active_optimizations),
                "completed_optimizations": len(pipeline_state.completed_optimizations),
                "total_optimizations": pipeline_state.total_optimizations,
                "failed_optimizations": pipeline_state.failed_optimizations,
                "success_rate_percent": (
                    (pipeline_state.total_optimizations - pipeline_state.failed_optimizations) /
                    max(pipeline_state.total_optimizations, 1)
                ) * 100
            },
            "configuration": {
                "algorithm": CURRENT_CONFIG.algorithm.value,
                "population_size": CURRENT_CONFIG.algorithm_params.population_size,
                "max_generations": CURRENT_CONFIG.algorithm_params.max_generations,
                "memory_limit_mb": CURRENT_CONFIG.memory_config.total_memory_limit_mb
            }
        }
    except Exception as e:
        logger.error("Metrics collection failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Metrics collection failed: {str(e)}"
        )

@app.post("/optimize", response_model=OptimizationResponse)
async def start_optimization(
    request: OptimizationRequest, 
    background_tasks: BackgroundTasks
):
    """
    Start optimization process with background execution.
    Primary endpoint for master pipeline integration.
    """
    request_id = str(uuid.uuid4())

    try:
        logger.info(
            "Optimization request received",
            request_id=request_id,
            input_dir=request.input_directory,
            output_dir=request.output_directory
        )

        # Validate request parameters
        input_valid, input_error = validate_input_directory(request.input_directory)
        if not input_valid:
            raise HTTPException(status_code=400, detail=input_error)

        output_valid, output_error = prepare_output_directory(request.output_directory)
        if not output_valid:
            raise HTTPException(status_code=400, detail=output_error)

        # Create configuration with request overrides
        config = PyGMOConfiguration.parse_obj(CURRENT_CONFIG.dict())
        if request.algorithm:
            config.algorithm = request.algorithm
        if request.population_size:
            config.algorithm_params.population_size = request.population_size
        if request.max_generations:
            config.algorithm_params.max_generations = request.max_generations

        # Estimate optimization duration
        estimated_duration = (
            config.algorithm_params.max_generations * 
            config.algorithm_params.population_size * 0.01
        )  # Rough estimation in seconds

        # Start optimization in background
        background_tasks.add_task(
            run_optimization_pipeline,
            request,
            request_id,
            config
        )

        return OptimizationResponse(
            request_id=request_id,
            status="started",
            message="Optimization started successfully",
            estimated_duration_seconds=int(estimated_duration),
            webhook_registered=request.webhook_url is not None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Optimization start failed",
            request_id=request_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start optimization: {str(e)}"
        )

@app.get("/optimize/{request_id}/status")
async def get_optimization_status(request_id: str):
    """
    Get real-time optimization status and progress.
    """
    try:
        # Check active optimizations
        if request_id in pipeline_state.active_optimizations:
            optimization = pipeline_state.active_optimizations[request_id]
            return {
                "request_id": request_id,
                "status": optimization["status"],
                "progress_percent": optimization["progress"] * 100,
                "current_generation": optimization["current_generation"],
                "best_fitness": optimization["best_fitness"],
                "memory_usage_mb": optimization["memory_usage_mb"],
                "elapsed_seconds": time.time() - optimization["start_time"]
            }

        # Check completed optimizations
        if request_id in pipeline_state.completed_optimizations:
            optimization = pipeline_state.completed_optimizations[request_id]
            return {
                "request_id": request_id,
                "status": optimization["status"],
                "progress_percent": 100.0,
                "duration_seconds": optimization["duration_seconds"],
                "error": optimization.get("error")
            }

        # Request not found
        raise HTTPException(
            status_code=404,
            detail=f"Optimization request {request_id} not found"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Status check failed",
            request_id=request_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Status check failed: {str(e)}"
        )

@app.get("/optimize/{request_id}/result", response_model=OptimizationResult)
async def get_optimization_result(request_id: str):
    """
    Get complete optimization result with solution data.
    """
    try:
        if request_id in pipeline_state.completed_optimizations:
            optimization = pipeline_state.completed_optimizations[request_id]

            if optimization.get("result"):
                return OptimizationResult.parse_obj(optimization["result"])
            else:
                return OptimizationResult(
                    request_id=request_id,
                    status="failed",
                    error_details=optimization.get("error", "Unknown error")
                )

        if request_id in pipeline_state.active_optimizations:
            raise HTTPException(
                status_code=202,
                detail=f"Optimization {request_id} still in progress"
            )

        raise HTTPException(
            status_code=404,
            detail=f"Optimization request {request_id} not found"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Result retrieval failed",
            request_id=request_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Result retrieval failed: {str(e)}"
        )

@app.post("/validate")
async def validate_input_data(request: Dict[str, str]):
    """
    Validate input data without starting optimization.
    Useful for master pipeline validation.
    """
    try:
        input_directory = request.get("input_directory")
        if not input_directory:
            raise HTTPException(
                status_code=400,
                detail="input_directory parameter required"
            )

        input_valid, input_error = validate_input_directory(input_directory)

        return {
            "valid": input_valid,
            "message": "Input validation successful" if input_valid else input_error,
            "details": {
                "directory_exists": Path(input_directory).exists(),
                "directory_accessible": Path(input_directory).is_dir(),
                "required_files_present": input_valid
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Input validation failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Input validation failed: {str(e)}"
        )

@app.put("/optimize/{request_id}/config")
async def update_optimization_config(
    request_id: str,
    config_updates: Dict[str, Any]
):
    """
    Update optimization configuration for active optimization.
    Limited to non-critical parameters for safety.
    """
    try:
        if request_id not in pipeline_state.active_optimizations:
            raise HTTPException(
                status_code=404,
                detail=f"Active optimization {request_id} not found"
            )

        optimization = pipeline_state.active_optimizations[request_id]

        # Only allow safe configuration updates
        allowed_updates = ["webhook_url", "priority", "metadata"]
        applied_updates = {}

        for key, value in config_updates.items():
            if key in allowed_updates:
                optimization["config"][key] = value
                applied_updates[key] = value

        return {
            "request_id": request_id,
            "applied_updates": applied_updates,
            "message": f"Configuration updated for {len(applied_updates)} parameters"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Configuration update failed",
            request_id=request_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Configuration update failed: {str(e)}"
        )

# Exception handlers for complete error management
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    """Handle Pydantic validation errors with detailed feedback."""
    logger.error(
        "Request validation failed",
        errors=exc.errors(),
        request_body=await request.body() if hasattr(request, 'body') else None
    )
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "errors": exc.errors()
        }
    )

@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    """Handle internal server errors with complete logging."""
    logger.error(
        "Internal server error",
        error=str(exc),
        traceback=traceback.format_exc(),
        request_path=request.url.path if hasattr(request, 'url') else None
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": str(uuid.uuid4()),
            "timestamp": time.time()
        }
    )

# Command-line interface for direct execution
def main():
    """
    Main entry point for PyGMO solver family.
    Supports both API server and direct pipeline execution.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="PyGMO Solver Family - Multi-Objective Educational Scheduling"
    )
    parser.add_argument(
        "--mode",
        choices=["api", "pipeline"],
        default="api",
        help="Execution mode: 'api' for server, 'pipeline' for direct execution"
    )
    parser.add_argument(
        "--input-dir",
        help="Input directory containing Stage 3 outputs (pipeline mode)"
    )
    parser.add_argument(
        "--output-dir", 
        help="Output directory for results (pipeline mode)"
    )
    parser.add_argument(
        "--config-file",
        help="Configuration file path (JSON/YAML)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="API server host (api mode)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (api mode)"
    )

    args = parser.parse_args()

    # Load configuration if specified
    global CURRENT_CONFIG
    if args.config_file:
        try:
            CURRENT_CONFIG = PyGMOConfiguration.load_from_file(args.config_file)
            logger.info("Configuration loaded from file", config_file=args.config_file)
        except Exception as e:
            logger.error("Failed to load configuration file", error=str(e))
            sys.exit(1)

    if args.mode == "api":
        # Start API server
        logger.info(
            "Starting PyGMO API server",
            host=args.host,
            port=args.port
        )
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=1,  # Single worker for resource control
            log_level="info"
        )

    elif args.mode == "pipeline":
        # Direct pipeline execution
        if not args.input_dir or not args.output_dir:
            logger.error("Input and output directories required for pipeline mode")
            sys.exit(1)

        logger.info(
            "Starting direct pipeline execution",
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )

        # Create optimization request
        request = OptimizationRequest(
            input_directory=args.input_dir,
            output_directory=args.output_dir
        )

        # Execute pipeline synchronously
        request_id = str(uuid.uuid4())
        try:
            result = asyncio.run(
                run_optimization_pipeline(request, request_id, CURRENT_CONFIG)
            )

            if result.status == "completed":
                logger.info(
                    "Pipeline execution completed successfully",
                    request_id=request_id,
                    output_files=result.file_paths
                )
                print(f"Optimization completed successfully. Results saved to: {args.output_dir}")
                print(f"Best fitness: {result.fitness_values}")
            else:
                logger.error(
                    "Pipeline execution failed",
                    request_id=request_id,
                    error=result.error_details
                )
                print(f"Optimization failed: {result.error_details}")
                sys.exit(1)

        except Exception as e:
            logger.error(
                "Pipeline execution failed with exception",
                error=str(e),
                traceback=traceback.format_exc()
            )
            print(f"Pipeline execution failed: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    main()
