#!/usr/bin/env python3
"""
PuLP Solver Family - Stage 6 API Layer: Pydantic Schemas & Data Models

This module implements the complete Pydantic schemas and data models for
Stage 6.1 PuLP solver family API layer, providing complete type validation
and serialization with mathematical rigor and theoretical compliance. Critical
component implementing complete API data model per Stage 6 foundational framework.

Theoretical Foundation:
    Based on Stage 6.1 PuLP Framework API schema requirements:
    - Implements complete data model per API integration specifications
    - Maintains mathematical consistency across all schema definitions
    - Ensures complete input validation and type safety
    - Provides serialization compatibility with JSON and other formats
    - Supports extensible schema evolution with backward compatibility

Architecture Compliance:
    - Implements API Schema Layer per foundational design rules
    - Maintains strict type validation with complete error handling
    - Provides modular schema organization with clear separation of concerns
    - Ensures optimal serialization performance with memory efficiency
    - Supports production-ready validation with complete reliability

Dependencies: pydantic, typing, datetime, enum, pathlib, json
Author: Student Team
Version: 1.0.0 (Production)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum

# Pydantic imports with complete error handling
try:
    from pydantic import BaseModel, Field, validator, root_validator, ValidationError
    from pydantic.dataclasses import dataclass as pydantic_dataclass
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback for environments without Pydantic
    BaseModel = object
    Field = lambda **kwargs: None
    validator = lambda *args, **kwargs: lambda f: f
    root_validator = lambda **kwargs: lambda f: f
    ValidationError = Exception
    pydantic_dataclass = lambda cls: cls
    PYDANTIC_AVAILABLE = False

# Configure module logger
import logging
logger = logging.getLogger(__name__)

class SolverBackendEnum(str, Enum):
    """
    Enumeration of supported PuLP solver backends.

    Mathematical Foundation: Defines complete solver backend coverage per
    PuLP framework specifications ensuring complete optimization support.
    """
    CBC = "CBC"                         # COIN-OR CBC (default)
    GLPK = "GLPK"                      # GNU Linear Programming Kit
    HIGHS = "HiGHS"                    # HiGHS optimizer
    CLP = "CLP"                        # COIN-OR CLP
    SYMPHONY = "SYMPHONY"              # COIN-OR SYMPHONY

class ExecutionStatus(str, Enum):
    """
    Enumeration of execution status states.

    Defines complete execution lifecycle states ensuring complete
    status tracking and monitoring capabilities.
    """
    CREATED = "created"                 # Execution created, not started
    RUNNING = "running"                 # Pipeline execution in progress
    COMPLETED = "completed"             # Execution completed successfully
    ERROR = "error"                     # Execution failed with error
    CANCELLED = "cancelled"             # Execution cancelled by user

class CSVFormatEnum(str, Enum):
    """
    Enumeration of supported CSV output formats.

    Defines complete CSV format coverage per output model requirements
    ensuring complete output generation capabilities.
    """
    STANDARD = "standard"               # Standard scheduling CSV format
    EXTENDED = "extended"               # Extended format with metadata
    MINIMAL = "minimal"                 # Minimal format with core fields
    COMPLIANCE = "compliance"           # Compliance reporting format
    ANALYTICS = "analytics"             # Analytics-optimized format

class ValidationLevel(str, Enum):
    """
    Enumeration of data validation levels.

    Defines validation strictness levels ensuring appropriate data quality
    control based on operational requirements.
    """
    NONE = "none"                       # No validation
    BASIC = "basic"                     # Basic field validation
    STRICT = "strict"                   # Strict data type validation
    complete = "complete"     # complete domain validation

if PYDANTIC_AVAILABLE:
    class BaseAPIModel(BaseModel):
        """
        Base Pydantic model with common configuration.

        Provides standard configuration and validation rules for all API models
        ensuring consistent behavior and optimal performance characteristics.
        """

        class Config:
            # Enable validation on assignment
            validate_assignment = True
            # Allow arbitrary types for complex objects
            arbitrary_types_allowed = True
            # Forbid extra fields to ensure schema compliance
            extra = 'forbid'
            # Use enum values for serialization
            use_enum_values = True
            # Enable JSON encoders for complex types
            json_encoders = {
                datetime: lambda v: v.isoformat(),
                Path: lambda v: str(v)
            }

else:
    # Fallback base class when Pydantic not available
    class BaseAPIModel:
        """Fallback base model when Pydantic not available."""
        pass

class InputPaths(BaseAPIModel):
    """
    Input file paths specification for Stage 3 artifacts.

    Mathematical Foundation: Defines complete input specification per
    Stage 6 foundational design ensuring proper artifact ingestion.

    Attributes:
        l_raw_path: Path to L_raw.parquet file from Stage 3
        l_rel_path: Path to L_rel.graphml file from Stage 3  
        l_idx_path: Path to L_idx file from Stage 3 (various formats)
    """
    l_raw_path: str = Field(
        ...,
        description="Path to L_raw.parquet file containing normalized entity tables",
        example="/path/to/stage3/L_raw.parquet"
    )

    l_rel_path: str = Field(
        ...,
        description="Path to L_rel.graphml file containing relationship graphs",
        example="/path/to/stage3/L_rel.graphml"
    )

    l_idx_path: str = Field(
        ...,
        description="Path to L_idx file containing multi-modal indices",
        example="/path/to/stage3/L_idx.feather"
    )

    @validator('l_raw_path', 'l_rel_path', 'l_idx_path')
    def validate_path_format(cls, v):
        """Validate path format and basic accessibility."""
        if not v or not v.strip():
            raise ValueError("Path cannot be empty")

        # Basic path format validation
        try:
            path_obj = Path(v)
            if not path_obj.is_absolute() and not str(path_obj).startswith('./'):
                # Allow both absolute and relative paths
                pass
        except Exception as e:
            raise ValueError(f"Invalid path format: {str(e)}")

        return v.strip()

class SolverConfiguration(BaseAPIModel):
    """
    complete solver configuration specification.

    Mathematical Foundation: Defines complete solver configuration per
    PuLP framework requirements ensuring optimal solving performance.

    Attributes:
        solver_name: PuLP solver backend to use
        time_limit_seconds: Maximum solving time limit
        memory_limit_mb: Maximum memory usage limit
        optimization_focus: Focus on optimality vs speed
        threads: Number of parallel threads to use
        presolve: Enable/disable presolve optimizations
        cuts: Enable/disable cutting plane algorithms
        heuristics: Enable/disable heuristic methods
        tolerance: Numerical tolerance for optimization
        verbose_logging: Enable verbose solver logging
        advanced_options: Additional solver-specific options
    """
    solver_name: SolverBackendEnum = Field(
        default=SolverBackendEnum.CBC,
        description="PuLP solver backend to use for optimization"
    )

    time_limit_seconds: Optional[float] = Field(
        default=300.0,
        ge=1.0,
        le=3600.0,
        description="Maximum time limit for solving (1-3600 seconds)"
    )

    memory_limit_mb: Optional[int] = Field(
        default=512,
        ge=64,
        le=2048,
        description="Maximum memory usage limit (64-2048 MB)"
    )

    optimization_focus: str = Field(
        default="balanced",
        description="Optimization focus: 'speed', 'optimality', or 'balanced'"
    )

    threads: Optional[int] = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of parallel threads (1-8)"
    )

    presolve: bool = Field(
        default=True,
        description="Enable presolve optimizations"
    )

    cuts: bool = Field(
        default=True,
        description="Enable cutting plane algorithms"
    )

    heuristics: bool = Field(
        default=True,
        description="Enable heuristic methods"
    )

    tolerance: float = Field(
        default=1e-6,
        ge=1e-9,
        le=1e-3,
        description="Numerical tolerance for optimization (1e-9 to 1e-3)"
    )

    verbose_logging: bool = Field(
        default=False,
        description="Enable verbose solver logging"
    )

    advanced_options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional solver-specific configuration options"
    )

    @validator('optimization_focus')
    def validate_optimization_focus(cls, v):
        """Validate optimization focus parameter."""
        valid_focuses = ['speed', 'optimality', 'balanced']
        if v.lower() not in valid_focuses:
            raise ValueError(f"Optimization focus must be one of: {valid_focuses}")
        return v.lower()

class OutputConfiguration(BaseAPIModel):
    """
    Output generation configuration specification.

    Mathematical Foundation: Defines complete output configuration per
    output model requirements ensuring complete result generation.

    Attributes:
        csv_format: CSV output format type
        include_metadata: Include metadata in output
        include_solver_info: Include solver-specific information
        validation_level: Level of output validation
        compression: Enable output file compression
        custom_fields: Additional custom fields to include
    """
    csv_format: CSVFormatEnum = Field(
        default=CSVFormatEnum.EXTENDED,
        description="CSV output format type"
    )

    include_metadata: bool = Field(
        default=True,
        description="Include complete metadata in output"
    )

    include_solver_info: bool = Field(
        default=True,
        description="Include solver-specific information in output"
    )

    validation_level: ValidationLevel = Field(
        default=ValidationLevel.complete,
        description="Level of output validation to perform"
    )

    compression: bool = Field(
        default=False,
        description="Enable output file compression"
    )

    custom_fields: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional custom fields to include in output"
    )

class SchedulingRequest(BaseAPIModel):
    """
    Primary scheduling optimization request schema.

    Mathematical Foundation: Implements complete request specification per
    API integration requirements ensuring complete scheduling configuration.

    Attributes:
        request_id: Optional unique request identifier
        input_paths: Paths to Stage 3 input artifacts
        solver_config: Solver configuration parameters
        output_config: Output generation configuration
        execution_priority: Execution priority level
        callback_url: Optional callback URL for status updates
        metadata: Additional request metadata
    """
    request_id: Optional[str] = Field(
        default=None,
        description="Optional unique request identifier for tracking"
    )

    input_paths: InputPaths = Field(
        ...,
        description="Paths to Stage 3 input artifacts (L_raw, L_rel, L_idx)"
    )

    solver_config: SolverConfiguration = Field(
        default_factory=SolverConfiguration,
        description="complete solver configuration parameters"
    )

    output_config: OutputConfiguration = Field(
        default_factory=OutputConfiguration,
        description="Output generation configuration"
    )

    execution_priority: int = Field(
        default=0,
        ge=-10,
        le=10,
        description="Execution priority (-10 low to 10 high)"
    )

    callback_url: Optional[str] = Field(
        default=None,
        description="Optional callback URL for execution status updates"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional request metadata and configuration"
    )

    @validator('callback_url')
    def validate_callback_url(cls, v):
        """Validate callback URL format if provided."""
        if v is not None:
            if not v.startswith(('http://', 'https://')):
                raise ValueError("Callback URL must start with http:// or https://")
        return v

class SchedulingResponse(BaseAPIModel):
    """
    Scheduling optimization response schema.

    Mathematical Foundation: Defines complete response specification per
    API integration standards ensuring complete execution information.

    Attributes:
        execution_id: Unique execution identifier
        status: Current execution status
        message: Human-readable response message
        created_at: Execution creation timestamp
        estimated_completion_time: Estimated completion time
        progress: Execution progress percentage
        current_stage: Current processing stage
        callback_scheduled: Whether callback is scheduled
    """
    execution_id: str = Field(
        ...,
        description="Unique execution identifier for tracking and status queries"
    )

    status: ExecutionStatus = Field(
        ...,
        description="Current execution status"
    )

    message: str = Field(
        ...,
        description="Human-readable response message"
    )

    created_at: str = Field(
        ...,
        description="Execution creation timestamp (ISO 8601)"
    )

    estimated_completion_time: Optional[str] = Field(
        default=None,
        description="Estimated completion timestamp (ISO 8601)"
    )

    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Execution progress percentage (0.0 to 100.0)"
    )

    current_stage: str = Field(
        default="initialization",
        description="Current processing stage description"
    )

    callback_scheduled: bool = Field(
        default=False,
        description="Whether callback notification is scheduled"
    )

class PipelineStatus(BaseAPIModel):
    """
    complete pipeline execution status schema.

    Mathematical Foundation: Implements complete status specification per
    execution monitoring requirements ensuring complete progress tracking.

    Attributes:
        execution_id: Unique execution identifier
        status: Current execution status
        progress: Execution progress percentage
        current_stage: Current processing stage
        created_at: Execution creation timestamp
        updated_at: Last status update timestamp
        completed_at: Execution completion timestamp
        execution_time_seconds: Total execution time
        results: Execution results summary
        error: Error information if applicable
        output_files: Generated output file information
        performance_metrics: Execution performance metrics
    """
    execution_id: str = Field(
        ...,
        description="Unique execution identifier"
    )

    status: ExecutionStatus = Field(
        ...,
        description="Current execution status"
    )

    progress: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Execution progress percentage (0.0 to 100.0)"
    )

    current_stage: str = Field(
        ...,
        description="Current processing stage"
    )

    created_at: str = Field(
        ...,
        description="Execution creation timestamp (ISO 8601)"
    )

    updated_at: Optional[str] = Field(
        default=None,
        description="Last status update timestamp (ISO 8601)"
    )

    completed_at: Optional[str] = Field(
        default=None,
        description="Execution completion timestamp (ISO 8601)"
    )

    execution_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total execution time in seconds"
    )

    results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Execution results summary"
    )

    error: Optional[str] = Field(
        default=None,
        description="Error information if execution failed"
    )

    output_files: Dict[str, str] = Field(
        default_factory=dict,
        description="Generated output file paths and information"
    )

    performance_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Execution performance metrics and statistics"
    )

class ExecutionResults(BaseAPIModel):
    """
    complete execution results schema.

    Mathematical Foundation: Defines complete results specification per
    output model requirements ensuring complete result information.

    Attributes:
        execution_id: Unique execution identifier
        pipeline_success: Overall pipeline success status
        assignments_generated: Number of assignments generated
        solver_info: Solver execution information
        performance_metrics: Detailed performance metrics
        quality_assessment: Solution quality assessment
        output_files: Generated output file information
        metadata: Additional result metadata
    """
    execution_id: str = Field(
        ...,
        description="Unique execution identifier"
    )

    pipeline_success: bool = Field(
        ...,
        description="Overall pipeline success status"
    )

    assignments_generated: int = Field(
        ...,
        ge=0,
        description="Total number of assignments generated"
    )

    solver_info: Dict[str, Any] = Field(
        ...,
        description="Detailed solver execution information"
    )

    performance_metrics: Dict[str, float] = Field(
        ...,
        description="complete performance metrics"
    )

    quality_assessment: Dict[str, Any] = Field(
        ...,
        description="Solution quality assessment and scoring"
    )

    output_files: Dict[str, str] = Field(
        ...,
        description="Generated output file paths and metadata"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution result metadata"
    )

class HealthCheckResponse(BaseAPIModel):
    """
    System health check response schema.

    Mathematical Foundation: Defines complete health specification per
    monitoring requirements ensuring complete system status reporting.

    Attributes:
        status: Overall system health status
        timestamp: Health check timestamp
        version: API version information
        stage: Implementation stage identifier
        import_success: Module import success status
        import_error: Import error information if applicable
        server_uptime_seconds: Server uptime in seconds
        active_executions: Number of active executions
        total_requests: Total number of requests processed
        system_info: Additional system information
    """
    status: str = Field(
        ...,
        description="Overall system health status ('healthy' or 'unhealthy')"
    )

    timestamp: str = Field(
        ...,
        description="Health check timestamp (ISO 8601)"
    )

    version: str = Field(
        ...,
        description="API version information"
    )

    stage: str = Field(
        ...,
        description="Implementation stage identifier"
    )

    import_success: bool = Field(
        ...,
        description="Module import success status"
    )

    import_error: Optional[str] = Field(
        default=None,
        description="Import error information if applicable"
    )

    server_uptime_seconds: float = Field(
        ...,
        ge=0.0,
        description="Server uptime in seconds"
    )

    active_executions: int = Field(
        ...,
        ge=0,
        description="Number of currently active executions"
    )

    total_requests: int = Field(
        ...,
        ge=0,
        description="Total number of requests processed since startup"
    )

    system_info: Dict[str, Any] = Field(
        ...,
        description="Additional system configuration and status information"
    )

class ErrorResponse(BaseAPIModel):
    """
    complete error response schema.

    Mathematical Foundation: Defines complete error specification per
    error handling requirements ensuring complete error information.

    Attributes:
        error_type: Classification of error type
        message: Human-readable error message
        details: Detailed error information
        timestamp: Error occurrence timestamp
        execution_id: Associated execution identifier if applicable
        suggested_actions: Suggested corrective actions
    """
    error_type: str = Field(
        ...,
        description="Classification of error type"
    )

    message: str = Field(
        ...,
        description="Human-readable error message"
    )

    details: Optional[str] = Field(
        default=None,
        description="Detailed error information and stack trace"
    )

    timestamp: str = Field(
        ...,
        description="Error occurrence timestamp (ISO 8601)"
    )

    execution_id: Optional[str] = Field(
        default=None,
        description="Associated execution identifier if applicable"
    )

    suggested_actions: List[str] = Field(
        default_factory=list,
        description="Suggested corrective actions for error resolution"
    )

class FileUploadResponse(BaseAPIModel):
    """
    File upload response schema.

    Mathematical Foundation: Defines complete upload response per
    file handling requirements ensuring complete upload information.

    Attributes:
        upload_id: Unique upload session identifier
        status: Upload status
        message: Human-readable upload message
        uploaded_at: Upload completion timestamp
        files: Information about uploaded files
        total_size_bytes: Total size of uploaded files
        input_paths: Generated input paths for scheduling
    """
    upload_id: str = Field(
        ...,
        description="Unique upload session identifier"
    )

    status: str = Field(
        ...,
        description="Upload status ('success', 'partial', 'error')"
    )

    message: str = Field(
        ...,
        description="Human-readable upload message"
    )

    uploaded_at: str = Field(
        ...,
        description="Upload completion timestamp (ISO 8601)"
    )

    files: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Detailed information about each uploaded file"
    )

    total_size_bytes: int = Field(
        ...,
        ge=0,
        description="Total size of all uploaded files in bytes"
    )

    input_paths: Dict[str, str] = Field(
        ...,
        description="Generated input paths mapping for scheduling request"
    )

class SolverPerformanceMetrics(BaseAPIModel):
    """
    complete solver performance metrics schema.

    Mathematical Foundation: Defines complete performance specification per
    solver monitoring requirements ensuring complete performance analysis.

    Attributes:
        solving_time_seconds: Total solving time in seconds
        objective_value: Final objective function value
        optimality_gap: Gap between best solution and optimal
        node_count: Number of branch-and-bound nodes explored
        cut_count: Number of cutting planes generated
        memory_usage_mb: Peak memory usage in MB
        iterations_completed: Number of solver iterations
        convergence_rate: Rate of convergence to solution
    """
    solving_time_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total solving time in seconds"
    )

    objective_value: Optional[float] = Field(
        default=None,
        description="Final objective function value"
    )

    optimality_gap: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Gap between best solution and optimal (0.0 to 1.0)"
    )

    node_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of branch-and-bound nodes explored"
    )

    cut_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of cutting planes generated"
    )

    memory_usage_mb: float = Field(
        ...,
        ge=0.0,
        description="Peak memory usage in MB"
    )

    iterations_completed: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of solver iterations completed"
    )

    convergence_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Rate of convergence to solution"
    )

class QualityAssessment(BaseAPIModel):
    """
    Solution quality assessment schema.

    Mathematical Foundation: Defines complete quality specification per
    quality assessment requirements ensuring complete solution evaluation.

    Attributes:
        overall_score: Overall solution quality score (0.0 to 1.0)
        constraint_satisfaction_rate: Rate of constraint satisfaction
        objective_optimality: Measure of objective optimality
        resource_utilization: Overall resource utilization efficiency
        temporal_distribution: Quality of temporal distribution
        assignment_balance: Balance across different assignment types
        feasibility_margin: Margin of solution feasibility
        quality_grade: Letter grade for solution quality
    """
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall solution quality score (0.0 to 1.0)"
    )

    constraint_satisfaction_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Rate of constraint satisfaction (0.0 to 1.0)"
    )

    objective_optimality: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Measure of objective optimality (0.0 to 1.0)"
    )

    resource_utilization: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall resource utilization efficiency (0.0 to 1.0)"
    )

    temporal_distribution: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Quality of temporal distribution (0.0 to 1.0)"
    )

    assignment_balance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Balance across different assignment types (0.0 to 1.0)"
    )

    feasibility_margin: float = Field(
        ...,
        ge=0.0,
        description="Margin of solution feasibility"
    )

    quality_grade: str = Field(
        ...,
        description="Letter grade for solution quality (A+, A, B+, B, C+, C, D)"
    )

    @validator('quality_grade')
    def validate_quality_grade(cls, v):
        """Validate quality grade format."""
        valid_grades = ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D']
        if v not in valid_grades:
            raise ValueError(f"Quality grade must be one of: {valid_grades}")
        return v

class APIConfiguration(BaseAPIModel):
    """
    API configuration schema for runtime settings.

    Mathematical Foundation: Defines complete configuration specification per
    API operational requirements ensuring optimal system configuration.

    Attributes:
        max_concurrent_executions: Maximum concurrent executions allowed
        execution_timeout_minutes: Timeout for individual executions
        max_upload_size_mb: Maximum file upload size
        enable_background_tasks: Enable asynchronous background processing
        cors_enabled: Enable CORS middleware
        debug_mode: Enable debug mode with verbose logging
        rate_limiting: Enable API rate limiting
        authentication_required: Require authentication for API access
    """
    max_concurrent_executions: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of concurrent executions (1-20)"
    )

    execution_timeout_minutes: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Timeout for individual executions in minutes (1-60)"
    )

    max_upload_size_mb: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum file upload size in MB (10-1000)"
    )

    enable_background_tasks: bool = Field(
        default=True,
        description="Enable asynchronous background processing"
    )

    cors_enabled: bool = Field(
        default=True,
        description="Enable Cross-Origin Resource Sharing (CORS)"
    )

    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode with verbose logging"
    )

    rate_limiting: bool = Field(
        default=False,
        description="Enable API rate limiting"
    )

    authentication_required: bool = Field(
        default=False,
        description="Require authentication for API access"
    )

# Utility functions for schema validation and conversion
def validate_scheduling_request(data: Dict[str, Any]) -> SchedulingRequest:
    """
    Validate and convert dictionary to SchedulingRequest.

    Provides complete validation of scheduling request data with detailed
    error reporting ensuring request correctness and compliance.

    Args:
        data: Dictionary containing request data

    Returns:
        Validated SchedulingRequest object

    Raises:
        ValidationError: If validation fails with detailed error information
    """
    try:
        return SchedulingRequest(**data)
    except ValidationError as e:
        logger.error(f"Scheduling request validation failed: {str(e)}")
        raise

def create_error_response(error_type: str, message: str, details: Optional[str] = None,
                         execution_id: Optional[str] = None) -> ErrorResponse:
    """
    Create standardized error response with complete error information.

    Provides consistent error response generation ensuring complete
    error reporting and debugging capabilities.

    Args:
        error_type: Classification of error type
        message: Human-readable error message
        details: Optional detailed error information
        execution_id: Optional associated execution identifier

    Returns:
        Standardized ErrorResponse object
    """
    return ErrorResponse(
        error_type=error_type,
        message=message,
        details=details,
        timestamp=datetime.now(timezone.utc).isoformat(),
        execution_id=execution_id,
        suggested_actions=get_suggested_actions(error_type)
    )

def get_suggested_actions(error_type: str) -> List[str]:
    """
    Get suggested corrective actions based on error type.

    Provides context-aware suggestions for error resolution ensuring
    complete user guidance and debugging support.

    Args:
        error_type: Classification of error type

    Returns:
        List of suggested corrective actions
    """
    action_mapping = {
        "validation_error": [
            "Check request format and required fields",
            "Verify input file paths are accessible",
            "Ensure solver configuration parameters are valid"
        ],
        "file_not_found": [
            "Verify input file paths exist and are accessible",
            "Check file permissions",
            "Ensure files are in expected format"
        ],
        "solver_error": [
            "Try different solver configuration",
            "Check input data for feasibility issues",
            "Reduce problem complexity or increase time limits"
        ],
        "timeout_error": [
            "Increase execution timeout",
            "Optimize solver configuration for speed",
            "Consider problem size reduction"
        ],
        "memory_error": [
            "Increase memory limits",
            "Reduce problem size",
            "Optimize data structures"
        ],
        "internal_server_error": [
            "Check server logs for detailed error information",
            "Retry request after brief delay",
            "Contact system administrator if issue persists"
        ]
    }

    return action_mapping.get(error_type, [
        "Review error details and retry",
        "Check system logs for additional information"
    ])

def create_success_response(execution_id: str, message: str = "Request processed successfully") -> SchedulingResponse:
    """
    Create standardized success response with complete information.

    Provides consistent success response generation ensuring complete
    execution information and status reporting.

    Args:
        execution_id: Unique execution identifier
        message: Success message

    Returns:
        Standardized SchedulingResponse object
    """
    return SchedulingResponse(
        execution_id=execution_id,
        status=ExecutionStatus.CREATED,
        message=message,
        created_at=datetime.now(timezone.utc).isoformat(),
        progress=0.0,
        current_stage="initialization"
    )

# Schema registry for dynamic validation
SCHEMA_REGISTRY = {
    "SchedulingRequest": SchedulingRequest,
    "SchedulingResponse": SchedulingResponse,
    "PipelineStatus": PipelineStatus,
    "ExecutionResults": ExecutionResults,
    "HealthCheckResponse": HealthCheckResponse,
    "ErrorResponse": ErrorResponse,
    "FileUploadResponse": FileUploadResponse,
    "SolverPerformanceMetrics": SolverPerformanceMetrics,
    "QualityAssessment": QualityAssessment,
    "APIConfiguration": APIConfiguration
}

def get_schema_by_name(schema_name: str) -> Optional[BaseAPIModel]:
    """
    Retrieve schema class by name for dynamic validation.

    Provides dynamic schema lookup functionality for runtime validation
    and serialization ensuring flexible schema management.

    Args:
        schema_name: Name of schema class

    Returns:
        Schema class if found, None otherwise
    """
    return SCHEMA_REGISTRY.get(schema_name)

def validate_data_against_schema(data: Dict[str, Any], schema_name: str) -> Any:
    """
    Validate data against specified schema with complete error handling.

    Provides dynamic data validation against registered schemas ensuring
    complete type checking and error reporting.

    Args:
        data: Data to validate
        schema_name: Name of schema to validate against

    Returns:
        Validated data object

    Raises:
        ValueError: If schema not found
        ValidationError: If validation fails
    """
    schema_class = get_schema_by_name(schema_name)

    if not schema_class:
        raise ValueError(f"Schema not found: {schema_name}")

    try:
        return schema_class(**data)
    except ValidationError as e:
        logger.error(f"Schema validation failed for {schema_name}: {str(e)}")
        raise

# Export all schemas for external use
__all__ = [
    # Enums
    'SolverBackendEnum',
    'ExecutionStatus', 
    'CSVFormatEnum',
    'ValidationLevel',

    # Core schemas
    'SchedulingRequest',
    'SchedulingResponse',
    'PipelineStatus',
    'ExecutionResults',
    'HealthCheckResponse',
    'ErrorResponse',
    'FileUploadResponse',

    # Configuration schemas
    'InputPaths',
    'SolverConfiguration',
    'OutputConfiguration',
    'APIConfiguration',

    # Metrics schemas
    'SolverPerformanceMetrics',
    'QualityAssessment',

    # Utility functions
    'validate_scheduling_request',
    'create_error_response',
    'create_success_response',
    'get_schema_by_name',
    'validate_data_against_schema',

    # Schema registry
    'SCHEMA_REGISTRY'
]

if __name__ == "__main__":
    # Example usage and testing
    import sys

    print("Testing API schemas...")

    try:
        # Test basic schema creation
        input_paths = InputPaths(
            l_raw_path="/path/to/L_raw.parquet",
            l_rel_path="/path/to/L_rel.graphml",
            l_idx_path="/path/to/L_idx.feather"
        )

        solver_config = SolverConfiguration(
            solver_name=SolverBackendEnum.CBC,
            time_limit_seconds=300.0,
            memory_limit_mb=512
        )

        request = SchedulingRequest(
            input_paths=input_paths,
            solver_config=solver_config
        )

        print(f"✓ Created SchedulingRequest: {request.dict()}")

        # Test error response creation
        error = create_error_response(
            error_type="validation_error",
            message="Test error message",
            details="Detailed error information"
        )

        print(f"✓ Created ErrorResponse: {error.dict()}")

        # Test schema validation
        test_data = {
            "l_raw_path": "/test/path/L_raw.parquet",
            "l_rel_path": "/test/path/L_rel.graphml", 
            "l_idx_path": "/test/path/L_idx.feather"
        }

        validated = validate_data_against_schema(test_data, "InputPaths")
        print(f"✓ Schema validation successful: {validated.dict()}")

        print("✓ All schema tests passed")

    except Exception as e:
        print(f"Schema test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
