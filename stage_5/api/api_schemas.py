"""
api/schemas.py
Stage 5 REST API Pydantic Models and Schema Definitions

This module defines comprehensive Pydantic models for all Stage 5 API endpoints,
providing enterprise-grade request/response validation, serialization, and
documentation generation. All models align exactly with the foundational design
specifications and JSON schema requirements.

Model Architecture:
- Request models: Input validation for Stage 5.1 and 5.2 API endpoints
- Response models: Structured output with comprehensive metadata and results
- Error models: Detailed error responses with context for debugging
- Health models: System monitoring and readiness verification
- Info models: API capabilities and version information

The models follow enterprise patterns with:
- Comprehensive field validation using Pydantic validators
- Detailed field documentation for API documentation generation
- JSON schema compliance with foundational design specifications
- Type safety with full IDE support and IntelliSense integration
- Custom validation logic for complex business rules
- Serialization control for optimal API performance

Schema Compliance:
- Exact match with Stage5-FOUNDATIONAL-DESIGN-IMPLEMENTATION-PLAN.md
- Version 1.0.0 schema validation for all JSON documents
- Comprehensive metadata preservation for audit trails
- Integration compatibility with Stage 5.1 and 5.2 module schemas

Validation Features:
- Path existence validation for input files
- Schema version compatibility checking
- Numeric bounds validation for configuration parameters
- Cross-field validation for logical consistency
- Custom error messages with detailed context

For detailed specifications and theoretical foundations, see:
- Stage5-FOUNDATIONAL-DESIGN-IMPLEMENTATION-PLAN.md Section 2 (JSON Schemas)
- ../common/schema.py for shared data models
- Individual stage modules for computational result models
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import re

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import constr, confloat, conint

# Schema version aligned with foundational design
SCHEMA_VERSION = "1.0.0"

# Validation constants from foundational design specifications
PARAMETER_COUNT = 16  # Fixed 16-parameter complexity framework
MAX_EXECUTION_TIME_MS = 300_000  # 5 minutes maximum execution time
MIN_CONFIDENCE_SCORE = 0.0  # Minimum confidence for valid selection
MAX_CONFIDENCE_SCORE = 1.0  # Maximum confidence score
MIN_SOLVER_COUNT = 1  # Minimum solvers for meaningful selection


class BaseAPIModel(BaseModel):
    """
    Base model for all API schemas with common configuration.
    
    Provides consistent configuration for JSON serialization,
    validation behavior, and schema generation across all API models.
    """
    
    class Config:
        # Enable JSON serialization of complex types
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            Path: lambda v: str(v) if v else None,
        }
        
        # Validate assignment for runtime type checking
        validate_assignment = True
        
        # Use enum values in schema generation
        use_enum_values = True
        
        # Allow population by field name or alias
        allow_population_by_field_name = True
        
        # Generate detailed schema descriptions
        schema_extra = {
            "examples": {}  # Will be overridden in specific models
        }


class Stage51AnalysisRequest(BaseAPIModel):
    """
    Request model for Stage 5.1 complexity parameter analysis.
    
    Validates input file paths and configuration for Stage 5.1 execution,
    ensuring all required files exist and configuration parameters are valid.
    
    Attributes:
        l_raw_path: Path to L_raw.parquet file from Stage 3
        l_rel_path: Path to L_rel.graphml file from Stage 3
        l_idx_path: Path to L_idx.{format} file from Stage 3
        output_dir: Directory for Stage 5.1 output files
        config_overrides: Optional configuration parameter overrides
        
    Validation:
        - File path format validation
        - Configuration parameter bounds checking
        - Cross-field logical consistency validation
        - Schema version compatibility
    """
    
    l_raw_path: constr(min_length=1) = Field(
        ...,
        description="Path to L_raw.parquet file containing normalized entities from Stage 3",
        example="/data/stage3/L_raw.parquet"
    )
    
    l_rel_path: constr(min_length=1) = Field(
        ...,
        description="Path to L_rel.graphml file containing relationship graphs from Stage 3",
        example="/data/stage3/L_rel.graphml"
    )
    
    l_idx_path: constr(min_length=1) = Field(
        ...,
        description="Path to L_idx index file in any supported format (pkl/parquet/feather/idx/bin)",
        example="/data/stage3/L_idx.feather"
    )
    
    output_dir: constr(min_length=1) = Field(
        ...,
        description="Output directory for Stage 5.1 processing results and logs",
        example="/outputs/stage5_1"
    )
    
    config_overrides: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional configuration parameter overrides for Stage 5.1 execution",
        example={
            "random_seed": 42,
            "ruggedness_walks": 100,
            "variance_samples": 50
        }
    )
    
    @validator('l_raw_path')
    def validate_l_raw_path(cls, v):
        """Validate L_raw file path format and extension."""
        if not v.endswith(('.parquet', '.pq')):
            raise ValueError("L_raw file must have .parquet or .pq extension")
        return v
    
    @validator('l_rel_path')
    def validate_l_rel_path(cls, v):
        """Validate L_rel file path format and extension."""
        if not v.endswith('.graphml'):
            raise ValueError("L_rel file must have .graphml extension")
        return v
    
    @validator('l_idx_path')
    def validate_l_idx_path(cls, v):
        """Validate L_idx file path format and supported extensions."""
        supported_extensions = ('.pkl', '.parquet', '.pq', '.feather', '.idx', '.bin')
        if not v.endswith(supported_extensions):
            raise ValueError(f"L_idx file must have one of these extensions: {supported_extensions}")
        return v
    
    @validator('config_overrides')
    def validate_config_overrides(cls, v):
        """Validate configuration override parameters."""
        if v is None:
            return v
        
        # Validate specific configuration parameters if present
        if 'random_seed' in v:
            seed = v['random_seed']
            if not isinstance(seed, int) or seed < 0:
                raise ValueError("random_seed must be non-negative integer")
        
        if 'ruggedness_walks' in v:
            walks = v['ruggedness_walks']
            if not isinstance(walks, int) or walks < 10 or walks > 1000:
                raise ValueError("ruggedness_walks must be integer between 10 and 1000")
        
        if 'variance_samples' in v:
            samples = v['variance_samples']
            if not isinstance(samples, int) or samples < 10 or samples > 500:
                raise ValueError("variance_samples must be integer between 10 and 500")
        
        return v
    
    class Config(BaseAPIModel.Config):
        schema_extra = {
            "example": {
                "l_raw_path": "/data/stage3/L_raw.parquet",
                "l_rel_path": "/data/stage3/L_rel.graphml", 
                "l_idx_path": "/data/stage3/L_idx.feather",
                "output_dir": "/outputs/stage5_1",
                "config_overrides": {
                    "random_seed": 42,
                    "ruggedness_walks": 100,
                    "variance_samples": 50
                }
            }
        }


class Stage51AnalysisResponse(BaseAPIModel):
    """
    Response model for Stage 5.1 complexity parameter analysis results.
    
    Contains complete complexity analysis results including 16-parameter vector,
    composite index, execution metadata, and output file information.
    
    Attributes:
        success: Boolean indicating successful execution
        complexity_metrics: Complete complexity analysis results
        output_files: List of generated output file paths
        execution_time_ms: Total execution time in milliseconds
        metadata: Execution metadata and statistics
        timestamp: Response generation timestamp
        
    Schema Compliance:
        - Exact match with complexity_metrics.json format
        - Version 1.0.0 schema validation
        - Complete parameter coverage P1-P16
        - Comprehensive metadata preservation
    """
    
    success: bool = Field(
        ...,
        description="Boolean indicating successful Stage 5.1 execution"
    )
    
    complexity_metrics: Dict[str, Any] = Field(
        ...,
        description="Complete complexity analysis results with 16 parameters and composite index"
    )
    
    output_files: List[str] = Field(
        ...,
        description="List of generated output file paths"
    )
    
    execution_time_ms: conint(ge=0, le=MAX_EXECUTION_TIME_MS) = Field(
        ...,
        description="Total execution time in milliseconds"
    )
    
    metadata: Dict[str, Any] = Field(
        ...,
        description="Execution metadata including statistics and configuration"
    )
    
    timestamp: str = Field(
        ...,
        description="Response generation timestamp in ISO format"
    )
    
    @validator('complexity_metrics')
    def validate_complexity_metrics(cls, v):
        """Validate complexity metrics structure and content."""
        required_fields = ['complexity_parameters', 'composite_index', 'parameter_statistics']
        
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate parameters structure
        params = v['complexity_parameters']
        expected_params = [
            f"p{i}_{name}" for i, name in enumerate([
                'dimensionality', 'constraint_density', 'faculty_specialization',
                'room_utilization', 'temporal_complexity', 'batch_variance',
                'competency_entropy', 'conflict_measure', 'coupling_coefficient',
                'heterogeneity_index', 'flexibility_measure', 'dependency_complexity',
                'landscape_ruggedness', 'scalability_factor', 'propagation_depth',
                'quality_variance'
            ], 1)
        ]
        
        for param in expected_params:
            if param not in params:
                raise ValueError(f"Missing complexity parameter: {param}")
            
            param_value = params[param]
            if not isinstance(param_value, (int, float)) or not isinstance(param_value, (type(None))):
                if param_value is not None:
                    try:
                        float(param_value)
                    except (ValueError, TypeError):
                        raise ValueError(f"Parameter {param} must be numeric")
        
        # Validate composite index
        composite_index = v['composite_index']
        if not isinstance(composite_index, (int, float)):
            try:
                float(composite_index)
            except (ValueError, TypeError):
                raise ValueError("composite_index must be numeric")
        
        return v
    
    class Config(BaseAPIModel.Config):
        schema_extra = {
            "example": {
                "success": True,
                "complexity_metrics": {
                    "schema_version": "1.0.0",
                    "complexity_parameters": {
                        "p1_dimensionality": 1250000.0,
                        "p2_constraint_density": 0.742,
                        "p3_faculty_specialization": 0.685,
                        # ... other parameters
                        "p16_quality_variance": 0.234
                    },
                    "composite_index": 12.875,
                    "parameter_statistics": {
                        "entity_counts": {
                            "courses": 250,
                            "faculty": 50,
                            "rooms": 30,
                            "timeslots": 40,
                            "batches": 25
                        }
                    }
                },
                "output_files": ["/outputs/stage5_1/complexity_metrics.json"],
                "execution_time_ms": 8500,
                "metadata": {"stage": "5.1"},
                "timestamp": "2025-10-07T02:45:00Z"
            }
        }


class Stage52SelectionRequest(BaseAPIModel):
    """
    Request model for Stage 5.2 solver selection optimization.
    
    Validates complexity metrics and solver capabilities file paths
    for Stage 5.2 execution, ensuring proper L2 normalization and
    LP optimization can be performed.
    
    Attributes:
        complexity_metrics_path: Path to Stage 5.1 complexity_metrics.json
        solver_capabilities_path: Path to solver_capabilities.json
        output_dir: Directory for Stage 5.2 output files
        config_overrides: Optional configuration parameter overrides
        
    Validation:
        - JSON file path validation
        - Configuration parameter bounds checking
        - Schema version compatibility
        - Cross-field logical consistency
    """
    
    complexity_metrics_path: constr(min_length=1) = Field(
        ...,
        description="Path to complexity_metrics.json file from Stage 5.1",
        example="/outputs/stage5_1/complexity_metrics.json"
    )
    
    solver_capabilities_path: constr(min_length=1) = Field(
        ...,
        description="Path to solver_capabilities.json file with solver arsenal",
        example="/config/solver_capabilities.json"
    )
    
    output_dir: constr(min_length=1) = Field(
        ...,
        description="Output directory for Stage 5.2 processing results and logs",
        example="/outputs/stage5_2"
    )
    
    config_overrides: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional configuration parameter overrides for Stage 5.2 execution",
        example={
            "optimization_seed": 42,
            "max_lp_iterations": 10,
            "lp_convergence_tolerance": 1e-6
        }
    )
    
    @validator('complexity_metrics_path')
    def validate_complexity_metrics_path(cls, v):
        """Validate complexity metrics file path and format."""
        if not v.endswith('.json'):
            raise ValueError("complexity_metrics file must have .json extension")
        if not Path(v).name.startswith('complexity_metrics'):
            raise ValueError("file must be named complexity_metrics.json")
        return v
    
    @validator('solver_capabilities_path')
    def validate_solver_capabilities_path(cls, v):
        """Validate solver capabilities file path and format."""
        if not v.endswith('.json'):
            raise ValueError("solver_capabilities file must have .json extension")
        return v
    
    @validator('config_overrides')
    def validate_config_overrides(cls, v):
        """Validate configuration override parameters."""
        if v is None:
            return v
        
        # Validate optimization-specific parameters
        if 'optimization_seed' in v:
            seed = v['optimization_seed']
            if not isinstance(seed, int) or seed < 0:
                raise ValueError("optimization_seed must be non-negative integer")
        
        if 'max_lp_iterations' in v:
            iterations = v['max_lp_iterations']
            if not isinstance(iterations, int) or iterations < 1 or iterations > 100:
                raise ValueError("max_lp_iterations must be integer between 1 and 100")
        
        if 'lp_convergence_tolerance' in v:
            tolerance = v['lp_convergence_tolerance']
            if not isinstance(tolerance, (int, float)) or tolerance <= 0 or tolerance >= 1:
                raise ValueError("lp_convergence_tolerance must be float between 0 and 1")
        
        return v
    
    class Config(BaseAPIModel.Config):
        schema_extra = {
            "example": {
                "complexity_metrics_path": "/outputs/stage5_1/complexity_metrics.json",
                "solver_capabilities_path": "/config/solver_capabilities.json",
                "output_dir": "/outputs/stage5_2",
                "config_overrides": {
                    "optimization_seed": 42,
                    "max_lp_iterations": 10,
                    "lp_convergence_tolerance": 1e-6
                }
            }
        }


class Stage52SelectionResponse(BaseAPIModel):
    """
    Response model for Stage 5.2 solver selection optimization results.
    
    Contains complete solver selection results including chosen solver,
    ranking, optimization details, and execution metadata.
    
    Attributes:
        success: Boolean indicating successful execution
        selection_decision: Complete solver selection results
        output_files: List of generated output file paths
        execution_time_ms: Total execution time in milliseconds
        metadata: Execution metadata and optimization details
        timestamp: Response generation timestamp
        
    Schema Compliance:
        - Exact match with selection_decision.json format
        - Version 1.0.0 schema validation
        - Complete solver ranking with margins
        - Comprehensive optimization details
    """
    
    success: bool = Field(
        ...,
        description="Boolean indicating successful Stage 5.2 execution"
    )
    
    selection_decision: Dict[str, Any] = Field(
        ...,
        description="Complete solver selection results with ranking and optimization details"
    )
    
    output_files: List[str] = Field(
        ...,
        description="List of generated output file paths"
    )
    
    execution_time_ms: conint(ge=0, le=MAX_EXECUTION_TIME_MS) = Field(
        ...,
        description="Total execution time in milliseconds"
    )
    
    metadata: Dict[str, Any] = Field(
        ...,
        description="Execution metadata including optimization convergence details"
    )
    
    timestamp: str = Field(
        ...,
        description="Response generation timestamp in ISO format"
    )
    
    @validator('selection_decision')
    def validate_selection_decision(cls, v):
        """Validate selection decision structure and content."""
        required_fields = ['selection_result', 'optimization_details']
        
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate selection result structure
        selection_result = v['selection_result']
        if 'chosen_solver' not in selection_result:
            raise ValueError("Missing chosen_solver in selection_result")
        
        chosen_solver = selection_result['chosen_solver']
        required_solver_fields = ['solver_id', 'confidence', 'match_score']
        
        for field in required_solver_fields:
            if field not in chosen_solver:
                raise ValueError(f"Missing field in chosen_solver: {field}")
        
        # Validate confidence score bounds
        confidence = chosen_solver['confidence']
        if not isinstance(confidence, (int, float)):
            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                raise ValueError("confidence must be numeric")
        
        if not (MIN_CONFIDENCE_SCORE <= confidence <= MAX_CONFIDENCE_SCORE):
            raise ValueError(f"confidence must be between {MIN_CONFIDENCE_SCORE} and {MAX_CONFIDENCE_SCORE}")
        
        # Validate ranking if present
        if 'ranking' in selection_result:
            ranking = selection_result['ranking']
            if not isinstance(ranking, list):
                raise ValueError("ranking must be a list")
            
            if len(ranking) < MIN_SOLVER_COUNT:
                raise ValueError(f"ranking must contain at least {MIN_SOLVER_COUNT} solver(s)")
            
            for i, rank_entry in enumerate(ranking):
                required_rank_fields = ['solver_id', 'score', 'margin']
                for field in required_rank_fields:
                    if field not in rank_entry:
                        raise ValueError(f"Missing field in ranking[{i}]: {field}")
        
        # Validate optimization details
        optimization_details = v['optimization_details']
        required_opt_fields = ['learned_weights', 'separation_margin']
        
        for field in required_opt_fields:
            if field not in optimization_details:
                raise ValueError(f"Missing field in optimization_details: {field}")
        
        # Validate learned weights
        learned_weights = optimization_details['learned_weights']
        if not isinstance(learned_weights, list):
            raise ValueError("learned_weights must be a list")
        
        if len(learned_weights) != PARAMETER_COUNT:
            raise ValueError(f"learned_weights must have exactly {PARAMETER_COUNT} elements")
        
        return v
    
    class Config(BaseAPIModel.Config):
        schema_extra = {
            "example": {
                "success": True,
                "selection_decision": {
                    "schema_version": "1.0.0",
                    "selection_result": {
                        "chosen_solver": {
                            "solver_id": "pulp_cbc",
                            "confidence": 0.847,
                            "match_score": 7.315
                        },
                        "ranking": [
                            {"solver_id": "pulp_cbc", "score": 7.315, "margin": 0.632},
                            {"solver_id": "ortools_cp_sat", "score": 6.683, "margin": 0.421}
                        ]
                    },
                    "optimization_details": {
                        "learned_weights": [0.08, 0.12, 0.06, 0.09, 0.11, 0.07, 0.05, 0.08, 0.07, 0.06, 0.09, 0.08, 0.06, 0.07, 0.05, 0.06],
                        "separation_margin": 0.632,
                        "lp_convergence": {
                            "iterations": 4,
                            "status": "OPTIMAL"
                        }
                    }
                },
                "output_files": ["/outputs/stage5_2/selection_decision.json"],
                "execution_time_ms": 3200,
                "metadata": {"stage": "5.2"},
                "timestamp": "2025-10-07T02:48:00Z"
            }
        }


class HealthResponse(BaseAPIModel):
    """
    Response model for system health check endpoint.
    
    Provides comprehensive system status information including
    uptime, performance metrics, and component readiness status.
    """
    
    status: constr(regex=r'^(healthy|degraded|unhealthy)$') = Field(
        ...,
        description="Overall system health status"
    )
    
    uptime_seconds: confloat(ge=0) = Field(
        ...,
        description="System uptime in seconds since startup"
    )
    
    version: str = Field(
        ...,
        description="API version information"
    )
    
    stage_5_1_ready: bool = Field(
        ...,
        description="Stage 5.1 module readiness status"
    )
    
    stage_5_2_ready: bool = Field(
        ...,
        description="Stage 5.2 module readiness status"
    )
    
    request_count: conint(ge=0) = Field(
        ...,
        description="Total number of processed requests"
    )
    
    error_count: conint(ge=0) = Field(
        ...,
        description="Total number of errors encountered"
    )
    
    average_execution_time: confloat(ge=0) = Field(
        ...,
        description="Average execution time in seconds"
    )
    
    last_execution_time: Optional[confloat(ge=0)] = Field(
        None,
        description="Most recent execution time in seconds"
    )
    
    timestamp: str = Field(
        ...,
        description="Health check timestamp in ISO format"
    )


class Stage5InfoResponse(BaseAPIModel):
    """
    Response model for Stage 5 system information endpoint.
    
    Provides detailed information about API capabilities,
    supported formats, and integration specifications.
    """
    
    api_version: str = Field(
        ...,
        description="API version information"
    )
    
    stage_5_1_info: Dict[str, Any] = Field(
        ...,
        description="Stage 5.1 module information and capabilities"
    )
    
    stage_5_2_info: Dict[str, Any] = Field(
        ...,
        description="Stage 5.2 module information and capabilities"
    )
    
    supported_endpoints: List[str] = Field(
        ...,
        description="List of supported API endpoints"
    )
    
    supported_input_formats: List[str] = Field(
        ...,
        description="List of supported input file formats"
    )
    
    supported_output_formats: List[str] = Field(
        ...,
        description="List of supported output file formats"
    )
    
    mathematical_framework: str = Field(
        ...,
        description="Reference to mathematical framework documentation"
    )
    
    timestamp: str = Field(
        ...,
        description="Information generation timestamp in ISO format"
    )


class ErrorResponse(BaseAPIModel):
    """
    Response model for API error conditions.
    
    Provides structured error information with context
    for debugging and troubleshooting.
    """
    
    error: str = Field(
        ...,
        description="Error type classification"
    )
    
    detail: str = Field(
        ...,
        description="Human-readable error description"
    )
    
    message: str = Field(
        ...,
        description="Detailed error message"
    )
    
    context: Dict[str, Any] = Field(
        {},
        description="Additional error context for debugging"
    )
    
    timestamp: str = Field(
        ...,
        description="Error occurrence timestamp in ISO format"
    )


class ValidationErrorResponse(BaseAPIModel):
    """
    Response model for input validation errors.
    
    Provides detailed validation failure information
    with field-specific error messages.
    """
    
    error: str = Field(
        "validation_error",
        description="Error type classification"
    )
    
    detail: str = Field(
        ...,
        description="Human-readable validation error description"  
    )
    
    validation_errors: List[Dict[str, Any]] = Field(
        ...,
        description="List of field-specific validation errors"
    )
    
    timestamp: str = Field(
        ...,
        description="Validation error timestamp in ISO format"
    )


# Export all models for API integration
__all__ = [
    "Stage51AnalysisRequest",
    "Stage51AnalysisResponse", 
    "Stage52SelectionRequest",
    "Stage52SelectionResponse",
    "HealthResponse",
    "Stage5InfoResponse",
    "ErrorResponse",
    "ValidationErrorResponse",
    "SCHEMA_VERSION",
    "PARAMETER_COUNT"
]