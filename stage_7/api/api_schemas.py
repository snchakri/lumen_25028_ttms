# api/schemas.py
"""
Stage 7 API Schemas - Pydantic Models for Request/Response Validation

This module defines comprehensive Pydantic models for all Stage 7 API endpoints,
providing strict request/response validation, automatic documentation generation,
and type safety for the complete timetable validation and formatting system.

CRITICAL DESIGN PHILOSOPHY: EXHAUSTIVE CONFIGURATION OPTIONS
These schemas provide extensive configuration parameters and validation options
to enable comprehensive customization of the validation and formatting process
according to institutional requirements and deployment scenarios.

Mathematical Foundation:
- Based on Stage 7 Complete Framework schema requirements
- Implements 12-parameter threshold validation data models
- Supports multi-format output configuration with institutional standards
- Provides comprehensive audit and monitoring data structures

Theoretical Compliance:
- Stage 7.1 Validation Engine request/response models
- Stage 7.2 Human-Readable Format Generation configuration models
- Complete API documentation with OpenAPI/Swagger integration
- Comprehensive error handling with structured error responses

Schema Categories:
1. Validation Schemas: Request/response models for schedule validation
2. Format Conversion Schemas: Models for human-readable format generation
3. Configuration Schemas: System configuration and customization models
4. Monitoring Schemas: Performance metrics and audit trail models
5. Utility Schemas: File upload, schema validation, and diagnostic models

Quality Assurance:
- Comprehensive field validation using Pydantic validators
- Detailed field documentation for API documentation generation
- Type safety with strict validation and error reporting
- Integration with FastAPI automatic documentation generation

IDE Integration:
This module is optimized for Cursor IDE and JetBrains IDEs with comprehensive
type hints, detailed docstrings, and intelligent code completion support.

Authors: Perplexity Labs AI - SIH 2025 Implementation
Version: Stage 7 API Schemas - Phase 4 Implementation
License: SIH 2025 Project - Educational Use Only
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from enum import Enum
from pathlib import Path
import time

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass


# === ENUMERATION CLASSES ===

class ValidationStatus(str, Enum):
    """Validation status enumeration for Stage 7.1 operations"""
    ACCEPTED = "accepted"
    REJECTED = "rejected" 
    PENDING = "pending"
    ERROR = "error"


class ProcessingStatus(str, Enum):
    """Processing status enumeration for Stage 7.2 operations"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class OutputFormat(str, Enum):
    """Output format enumeration for format conversion"""
    CSV = "csv"
    EXCEL = "excel"
    TSV = "tsv"
    JSON = "json"
    HTML = "html"


class InstitutionalStandard(str, Enum):
    """Institutional standard enumeration for format compliance"""
    UNIVERSITY = "university"
    COLLEGE = "college"
    SCHOOL = "school"
    INSTITUTE = "institute"
    CUSTOM = "custom"


class SortingStrategy(str, Enum):
    """Sorting strategy enumeration for human-readable format generation"""
    STANDARD_ACADEMIC = "standard_academic"
    FACULTY_CENTRIC = "faculty_centric"
    ROOM_CENTRIC = "room_centric"
    DEPARTMENT_CENTRIC = "department_centric"
    TIME_OPTIMIZED = "time_optimized"


class ErrorCategory(str, Enum):
    """Error category enumeration for 4-tier error classification"""
    CRITICAL = "critical"
    QUALITY = "quality"
    PREFERENCE = "preference"
    COMPUTATIONAL = "computational"


class SystemHealthStatus(str, Enum):
    """System health status enumeration for monitoring"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


# === BASE CONFIGURATION MODELS ===

class ThresholdBoundsModel(BaseModel):
    """
    Threshold bounds model for individual validation parameters
    
    Represents the lower and upper bounds for a single threshold parameter
    with mathematical foundation references and validation descriptions.
    """
    lower_bound: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Lower threshold bound (0.0 to 1.0)"
    )
    upper_bound: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Upper threshold bound (0.0 to 1.0)"
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Human-readable description of the threshold parameter"
    )
    mathematical_foundation: Optional[str] = Field(
        None,
        max_length=200,
        description="Reference to theoretical framework section or theorem"
    )
    
    @validator('upper_bound')
    def validate_bounds(cls, v, values):
        if 'lower_bound' in values and v < values['lower_bound']:
            raise ValueError('upper_bound must be >= lower_bound')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "lower_bound": 0.95,
                "upper_bound": 1.0,
                "description": "Minimum 95% of required courses must be scheduled",
                "mathematical_foundation": "Theorem 3.1: Course Coverage Adequacy"
            }
        }


class ValidationConfigModel(BaseModel):
    """
    Comprehensive validation configuration model for Stage 7.1
    
    Encapsulates all configuration parameters for the 12-parameter threshold
    validation system with customizable bounds and processing options.
    """
    threshold_bounds: Optional[Dict[str, ThresholdBoundsModel]] = Field(
        None,
        description="Custom threshold bounds for all 12 validation parameters"
    )
    enable_correlation_analysis: bool = Field(
        True,
        description="Enable threshold interdependency analysis per Section 16.1"
    )
    fail_fast: bool = Field(
        True,
        description="Enable fail-fast validation with immediate rejection"
    )
    generate_advisory_messages: bool = Field(
        True,
        description="Generate actionable advisory messages for violations"
    )
    custom_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Custom weights for global quality score calculation"
    )
    global_quality_threshold: float = Field(
        0.75,
        ge=0.0,
        le=1.0,
        description="Minimum global quality score for acceptance"
    )
    enable_detailed_logging: bool = Field(
        True,
        description="Enable comprehensive audit logging for validation process"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "threshold_bounds": {
                    "course_coverage_ratio": {
                        "lower_bound": 0.95,
                        "upper_bound": 1.0,
                        "description": "Course coverage adequacy"
                    }
                },
                "fail_fast": True,
                "generate_advisory_messages": True,
                "global_quality_threshold": 0.75
            }
        }


class FormatConfigModel(BaseModel):
    """
    Format configuration model for Stage 7.2 human-readable generation
    
    Encapsulates all configuration parameters for human-readable timetable
    generation with educational domain optimization and institutional compliance.
    """
    output_format: OutputFormat = Field(
        OutputFormat.CSV,
        description="Output format for human-readable timetable"
    )
    institutional_standard: InstitutionalStandard = Field(
        InstitutionalStandard.UNIVERSITY,
        description="Institutional standard for format compliance"
    )
    sorting_strategy: SortingStrategy = Field(
        SortingStrategy.STANDARD_ACADEMIC,
        description="Sorting strategy for educational domain optimization"
    )
    department_order: Optional[List[str]] = Field(
        None,
        max_items=50,
        description="Custom department priority ordering for sorting"
    )
    enable_time_optimization: bool = Field(
        True,
        description="Enable time-based sorting optimization"
    )
    include_duration_formatting: bool = Field(
        True,
        description="Include human-readable duration formatting"
    )
    enable_utf8_encoding: bool = Field(
        True,
        description="Enable UTF-8 encoding for international character support"
    )
    custom_column_names: Optional[Dict[str, str]] = Field(
        None,
        description="Custom column names for institutional requirements"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "output_format": "csv",
                "institutional_standard": "university",
                "sorting_strategy": "standard_academic",
                "department_order": ["CSE", "ME", "CHE", "EE"],
                "enable_time_optimization": True
            }
        }


# === VALIDATION REQUEST/RESPONSE MODELS ===

class ValidationRequest(BaseModel):
    """
    Complete schedule validation request model
    
    Defines the request structure for comprehensive timetable validation
    using the Stage 7.1 12-parameter framework with configurable options.
    """
    schedule_csv_path: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Path to validated schedule CSV file from Stage 6"
    )
    output_model_json_path: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Path to output model JSON metadata from Stage 6"
    )
    stage3_reference_path: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Path to Stage 3 compiled reference data directory"
    )
    validation_config: Optional[ValidationConfigModel] = Field(
        None,
        description="Optional validation configuration for customization"
    )
    output_path: Optional[str] = Field(
        None,
        max_length=1000,
        description="Optional path for validation analysis output"
    )
    enable_async_processing: bool = Field(
        False,
        description="Enable asynchronous processing for large datasets"
    )
    
    @validator('schedule_csv_path', 'output_model_json_path', 'stage3_reference_path')
    def validate_paths(cls, v):
        if not v.strip():
            raise ValueError('Path cannot be empty')
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "schedule_csv_path": "/data/stage6/schedule.csv",
                "output_model_json_path": "/data/stage6/output_model.json",
                "stage3_reference_path": "/data/stage3/compiled_data/",
                "validation_config": {
                    "fail_fast": True,
                    "generate_advisory_messages": True
                }
            }
        }


class ValidationResultModel(BaseModel):
    """
    Detailed validation result model with comprehensive metrics
    
    Encapsulates the complete validation result including all 12 threshold
    values, violation details, and advisory messages for failed validations.
    """
    status: ValidationStatus = Field(
        ...,
        description="Overall validation status (accepted/rejected)"
    )
    global_quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weighted global quality score (Σ wi·φi(S))"
    )
    threshold_values: Dict[str, float] = Field(
        ...,
        description="Calculated values for all 12 threshold parameters"
    )
    threshold_status: Dict[str, bool] = Field(
        ...,
        description="Pass/fail status for each threshold parameter"
    )
    violated_thresholds: List[str] = Field(
        default_factory=list,
        description="List of threshold parameters that failed validation"
    )
    error_category: Optional[ErrorCategory] = Field(
        None,
        description="Primary error category for failed validations"
    )
    advisory_message: Optional[str] = Field(
        None,
        max_length=1000,
        description="Actionable advisory message for violation remediation"
    )
    correlation_analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="Threshold interdependency analysis results"
    )
    detailed_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Detailed mathematical metrics and calculations"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "accepted",
                "global_quality_score": 0.87,
                "threshold_values": {
                    "course_coverage_ratio": 0.98,
                    "conflict_resolution_rate": 1.0,
                    "faculty_workload_balance": 0.89
                },
                "threshold_status": {
                    "course_coverage_ratio": True,
                    "conflict_resolution_rate": True,
                    "faculty_workload_balance": True
                },
                "violated_thresholds": [],
                "error_category": None,
                "advisory_message": None
            }
        }


class ValidationResponse(BaseModel):
    """
    Complete validation response model with operation metadata
    
    Provides comprehensive validation results with processing metadata,
    performance metrics, and audit trail information.
    """
    operation_id: str = Field(
        ...,
        min_length=1,
        description="Unique operation identifier for tracking and auditing"
    )
    status: ValidationStatus = Field(
        ...,
        description="Overall validation status"
    )
    validation_result: ValidationResultModel = Field(
        ...,
        description="Detailed validation results and metrics"
    )
    processing_time_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total processing time in seconds"
    )
    memory_usage_mb: Optional[float] = Field(
        None,
        ge=0.0,
        description="Peak memory usage in megabytes"
    )
    timestamp: float = Field(
        ...,
        description="Response generation timestamp (Unix time)"
    )
    api_version: str = Field(
        default="7.0.0",
        description="API version used for validation"
    )
    theoretical_framework: str = Field(
        default="Stage 7 Output Validation Framework",
        description="Theoretical framework reference"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "operation_id": "val_12345678-1234-5678-9abc-123456789abc",
                "status": "accepted",
                "validation_result": {
                    "status": "accepted",
                    "global_quality_score": 0.87,
                    "threshold_values": {},
                    "threshold_status": {},
                    "violated_thresholds": []
                },
                "processing_time_seconds": 4.23,
                "memory_usage_mb": 145.7,
                "timestamp": 1699123456.789
            }
        }


# === FORMAT CONVERSION REQUEST/RESPONSE MODELS ===

class FormatConversionRequest(BaseModel):
    """
    Human-readable format conversion request model
    
    Defines the request structure for converting validated technical schedules
    to human-readable timetables with educational domain optimization.
    """
    validated_schedule_path: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Path to validated schedule CSV (TRUSTED - NO RE-VALIDATION)"
    )
    stage3_reference_path: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Path to Stage 3 reference data for metadata enrichment"
    )
    output_path: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Path where final_timetable file should be written"
    )
    format_config: Optional[FormatConfigModel] = Field(
        None,
        description="Optional format configuration for customization"
    )
    enable_quality_validation: bool = Field(
        True,
        description="Enable output quality validation and metrics collection"
    )
    
    @validator('validated_schedule_path', 'stage3_reference_path', 'output_path')
    def validate_paths(cls, v):
        if not v.strip():
            raise ValueError('Path cannot be empty')
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "validated_schedule_path": "/data/stage7/validated_schedule.csv",
                "stage3_reference_path": "/data/stage3/compiled_data/",
                "output_path": "/output/final_timetable.csv",
                "format_config": {
                    "output_format": "csv",
                    "sorting_strategy": "standard_academic",
                    "department_order": ["CSE", "ME", "CHE"]
                }
            }
        }


class FormatConversionResponse(BaseModel):
    """
    Format conversion response model with comprehensive metadata
    
    Provides complete format conversion results with processing metadata,
    quality metrics, and educational domain optimization details.
    """
    operation_id: str = Field(
        ...,
        min_length=1,
        description="Unique operation identifier for tracking"
    )
    status: ProcessingStatus = Field(
        ...,
        description="Format conversion processing status"
    )
    output_file_path: Optional[str] = Field(
        None,
        description="Path to generated human-readable timetable file"
    )
    conversion_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Detailed conversion process metadata"
    )
    sorting_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Multi-level sorting process metadata"
    )
    formatting_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Output formatting process metadata"
    )
    quality_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Output quality assessment metrics"
    )
    processing_time_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total processing time in seconds"
    )
    memory_usage_mb: Optional[float] = Field(
        None,
        ge=0.0,
        description="Peak memory usage in megabytes"
    )
    timestamp: float = Field(
        ...,
        description="Response generation timestamp (Unix time)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "operation_id": "fmt_87654321-4321-8765-dcba-987654321cba",
                "status": "completed",
                "output_file_path": "/output/final_timetable.csv",
                "conversion_metadata": {
                    "records_processed": 1247,
                    "departments_included": ["CSE", "ME", "CHE"],
                    "conversion_time_ms": 1234
                },
                "processing_time_seconds": 2.45,
                "timestamp": 1699123456.789
            }
        }


# === BATCH PROCESSING MODELS ===

class BatchValidationRequest(BaseModel):
    """
    Batch validation request model for multiple schedule processing
    
    Enables processing of multiple timetable validation requests with
    parallel processing support and comprehensive result aggregation.
    """
    schedule_requests: List[ValidationRequest] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="List of individual validation requests (max 50)"
    )
    enable_parallel_processing: bool = Field(
        True,
        description="Enable parallel processing for improved performance"
    )
    max_concurrent_validations: int = Field(
        5,
        ge=1,
        le=20,
        description="Maximum number of concurrent validation operations"
    )
    aggregation_strategy: str = Field(
        "comprehensive",
        regex="^(comprehensive|summary|detailed)$",
        description="Result aggregation strategy"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "schedule_requests": [
                    {
                        "schedule_csv_path": "/data/schedule1.csv",
                        "output_model_json_path": "/data/output1.json",
                        "stage3_reference_path": "/data/stage3/"
                    },
                    {
                        "schedule_csv_path": "/data/schedule2.csv",
                        "output_model_json_path": "/data/output2.json",
                        "stage3_reference_path": "/data/stage3/"
                    }
                ],
                "enable_parallel_processing": True,
                "max_concurrent_validations": 3
            }
        }


class BatchValidationResponse(BaseModel):
    """
    Batch validation response model with aggregated results
    
    Provides comprehensive results for batch validation operations with
    success/failure statistics and detailed individual results.
    """
    operation_id: str = Field(
        ...,
        description="Unique batch operation identifier"
    )
    total_schedules: int = Field(
        ...,
        ge=0,
        description="Total number of schedules processed"
    )
    successful_validations: int = Field(
        ...,
        ge=0,
        description="Number of successfully validated schedules"
    )
    failed_validations: int = Field(
        ...,
        ge=0,
        description="Number of failed validation attempts"
    )
    results: List[ValidationResponse] = Field(
        default_factory=list,
        description="Individual validation results for each schedule"
    )
    aggregate_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Aggregated metrics across all validations"
    )
    processing_time_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total batch processing time"
    )
    timestamp: float = Field(
        ...,
        description="Batch completion timestamp"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "operation_id": "batch_11111111-1111-1111-1111-111111111111",
                "total_schedules": 5,
                "successful_validations": 4,
                "failed_validations": 1,
                "results": [],
                "processing_time_seconds": 23.45,
                "timestamp": 1699123456.789
            }
        }


# === CONFIGURATION REQUEST/RESPONSE MODELS ===

class ThresholdConfigurationRequest(BaseModel):
    """
    Threshold configuration update request model
    
    Enables customization of validation threshold bounds for institutional
    requirements while maintaining mathematical validity.
    """
    threshold_updates: Dict[str, ThresholdBoundsModel] = Field(
        ...,
        min_items=1,
        max_items=12,
        description="Threshold parameter updates (max 12 parameters)"
    )
    validation_mode: str = Field(
        "strict",
        regex="^(strict|lenient|custom)$",
        description="Validation mode for bound checking"
    )
    institutional_context: Optional[str] = Field(
        None,
        max_length=500,
        description="Institutional context for threshold customization"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "threshold_updates": {
                    "course_coverage_ratio": {
                        "lower_bound": 0.90,
                        "upper_bound": 1.0,
                        "description": "Relaxed course coverage for small institutions"
                    }
                },
                "validation_mode": "custom",
                "institutional_context": "Small private college with flexible requirements"
            }
        }


class ThresholdConfigurationResponse(BaseModel):
    """
    Threshold configuration response model with current settings
    
    Provides comprehensive threshold configuration information including
    mathematical foundations and institutional customizations.
    """
    thresholds: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Complete threshold configuration with bounds and metadata"
    )
    last_updated: float = Field(
        ...,
        description="Last configuration update timestamp"
    )
    theoretical_framework: str = Field(
        ...,
        description="Theoretical framework reference"
    )
    version: str = Field(
        ...,
        description="Configuration version"
    )
    institutional_customizations: Optional[List[str]] = Field(
        None,
        description="List of parameters with institutional customizations"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "thresholds": {
                    "course_coverage_ratio": {
                        "lower_bound": 0.95,
                        "upper_bound": 1.0,
                        "description": "Course coverage adequacy",
                        "mathematical_foundation": "Theorem 3.1"
                    }
                },
                "last_updated": 1699123456.789,
                "theoretical_framework": "Stage 7 Output Validation Framework",
                "version": "7.0.0"
            }
        }


class DepartmentOrderingRequest(BaseModel):
    """
    Department ordering configuration request model
    
    Enables customization of department priority ordering for human-readable
    format generation with educational domain optimization.
    """
    department_order: List[str] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="Department priority order for sorting (max 50 departments)"
    )
    ordering_strategy: str = Field(
        "institutional_priority",
        regex="^(institutional_priority|alphabetical|enrollment_based|custom)$",
        description="Ordering strategy rationale"
    )
    institutional_context: Optional[str] = Field(
        None,
        max_length=500,
        description="Institutional context for ordering decision"
    )
    
    @validator('department_order')
    def validate_unique_departments(cls, v):
        if len(v) != len(set(v)):
            raise ValueError('Department order contains duplicate entries')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "department_order": ["CSE", "ME", "CHE", "EE", "ECE", "CE"],
                "ordering_strategy": "institutional_priority",
                "institutional_context": "Engineering-focused university with CSE as primary department"
            }
        }


class DepartmentOrderingResponse(BaseModel):
    """
    Department ordering configuration response model
    
    Provides current department ordering configuration with optimization
    metadata and institutional compliance information.
    """
    department_order: List[str] = Field(
        ...,
        description="Current department priority ordering"
    )
    ordering_strategy: str = Field(
        ...,
        description="Applied ordering strategy"
    )
    last_updated: float = Field(
        ...,
        description="Last ordering update timestamp"
    )
    description: str = Field(
        ...,
        description="Ordering description and rationale"
    )
    optimization_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Educational domain optimization metadata"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "department_order": ["CSE", "ME", "CHE", "EE", "ECE", "CE"],
                "ordering_strategy": "institutional_priority",
                "last_updated": 1699123456.789,
                "description": "Engineering-focused priority ordering",
                "optimization_metadata": {
                    "sort_stability": "guaranteed",
                    "performance_impact": "minimal"
                }
            }
        }


# === MONITORING AND SYSTEM STATUS MODELS ===

class HealthCheckResponse(BaseModel):
    """
    System health check response model
    
    Provides comprehensive system health information including component
    status, performance metrics, and diagnostic information.
    """
    status: SystemHealthStatus = Field(
        ...,
        description="Overall system health status"
    )
    timestamp: float = Field(
        ...,
        description="Health check timestamp"
    )
    uptime_seconds: float = Field(
        ...,
        ge=0.0,
        description="System uptime in seconds"
    )
    version: str = Field(
        ...,
        description="API version"
    )
    components: Dict[str, bool] = Field(
        ...,
        description="Individual component health status"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current system metrics"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if status is error"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": 1699123456.789,
                "uptime_seconds": 86400.0,
                "version": "7.0.0",
                "components": {
                    "stage_7_1_validation": True,
                    "stage_7_2_formatting": True,
                    "database": True
                },
                "metrics": {
                    "total_requests": 1247,
                    "error_count": 3,
                    "active_operations": 2
                }
            }
        }


class APIMetricsResponse(BaseModel):
    """
    API performance metrics response model
    
    Provides detailed performance analytics including request patterns,
    response times, error rates, and resource utilization statistics.
    """
    time_period_hours: int = Field(
        ...,
        ge=1,
        description="Time period for metrics calculation (hours)"
    )
    total_requests: int = Field(
        ...,
        ge=0,
        description="Total requests in time period"
    )
    successful_requests: int = Field(
        ...,
        ge=0,
        description="Successful requests in time period"
    )
    failed_requests: int = Field(
        ...,
        ge=0,
        description="Failed requests in time period"
    )
    average_response_time: float = Field(
        ...,
        ge=0.0,
        description="Average response time in seconds"
    )
    p95_response_time: float = Field(
        ...,
        ge=0.0,
        description="95th percentile response time in seconds"
    )
    requests_per_hour: float = Field(
        ...,
        ge=0.0,
        description="Average requests per hour"
    )
    error_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Error rate as fraction (0.0 to 1.0)"
    )
    endpoint_statistics: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-endpoint performance statistics"
    )
    timestamp: float = Field(
        ...,
        description="Metrics calculation timestamp"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "time_period_hours": 24,
                "total_requests": 1247,
                "successful_requests": 1201,
                "failed_requests": 46,
                "average_response_time": 2.34,
                "p95_response_time": 8.67,
                "requests_per_hour": 51.96,
                "error_rate": 0.037,
                "endpoint_statistics": {
                    "/validate/complete": {
                        "total_requests": 892,
                        "average_response_time": 4.12
                    }
                },
                "timestamp": 1699123456.789
            }
        }


class AuditTrailRequest(BaseModel):
    """
    Audit trail request model for filtering and pagination
    
    Enables filtered retrieval of audit log entries with pagination
    support for compliance and debugging purposes.
    """
    operation_filter: Optional[str] = Field(
        None,
        max_length=100,
        description="Filter by operation type (partial match)"
    )
    start_time: Optional[float] = Field(
        None,
        description="Start time filter (Unix timestamp)"
    )
    end_time: Optional[float] = Field(
        None,
        description="End time filter (Unix timestamp)"
    )
    page: int = Field(
        1,
        ge=1,
        description="Page number for pagination (starting from 1)"
    )
    page_size: int = Field(
        50,
        ge=1,
        le=1000,
        description="Number of entries per page (max 1000)"
    )
    
    @validator('end_time')
    def validate_time_range(cls, v, values):
        if v is not None and 'start_time' in values and values['start_time'] is not None:
            if v <= values['start_time']:
                raise ValueError('end_time must be after start_time')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "operation_filter": "validation",
                "start_time": 1699000000.0,
                "end_time": 1699123456.789,
                "page": 1,
                "page_size": 100
            }
        }


class AuditTrailResponse(BaseModel):
    """
    Audit trail response model with paginated entries
    
    Provides paginated audit log entries with comprehensive metadata
    for compliance monitoring and system debugging.
    """
    total_entries: int = Field(
        ...,
        ge=0,
        description="Total number of matching audit entries"
    )
    page: int = Field(
        ...,
        ge=1,
        description="Current page number"
    )
    page_size: int = Field(
        ...,
        ge=1,
        description="Entries per page"
    )
    entries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Paginated audit log entries"
    )
    timestamp: float = Field(
        ...,
        description="Response generation timestamp"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "total_entries": 1247,
                "page": 1,
                "page_size": 50,
                "entries": [
                    {
                        "timestamp": 1699123456.789,
                        "operation": "validation_request",
                        "details": {"operation_id": "val_123"},
                        "user_id": None,
                        "request_id": "req_456"
                    }
                ],
                "timestamp": 1699123456.789
            }
        }


# === UTILITY AND DIAGNOSTIC MODELS ===

class FileUploadResponse(BaseModel):
    """
    File upload response model with storage information
    
    Provides file upload confirmation with storage metadata
    for subsequent processing operations.
    """
    file_id: str = Field(
        ...,
        description="Unique file identifier for processing"
    )
    original_filename: Optional[str] = Field(
        None,
        description="Original filename from upload"
    )
    stored_path: str = Field(
        ...,
        description="Server storage path for uploaded file"
    )
    file_size: Optional[int] = Field(
        None,
        ge=0,
        description="File size in bytes"
    )
    content_type: Optional[str] = Field(
        None,
        description="MIME content type"
    )
    upload_timestamp: float = Field(
        ...,
        description="Upload completion timestamp"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "file_id": "file_12345678-1234-5678-9abc-123456789abc",
                "original_filename": "schedule.csv",
                "stored_path": "/tmp/stage7_uploads/file_123_schedule.csv",
                "file_size": 2048576,
                "content_type": "text/csv",
                "upload_timestamp": 1699123456.789
            }
        }


class SchemaValidationRequest(BaseModel):
    """
    Data schema validation request model
    
    Enables validation of uploaded files against expected schemas
    for compliance with Stage 7 data requirements.
    """
    schedule_csv_path: Optional[str] = Field(
        None,
        max_length=1000,
        description="Path to schedule CSV file for schema validation"
    )
    output_model_json_path: Optional[str] = Field(
        None,
        max_length=1000,
        description="Path to output model JSON file for schema validation"
    )
    stage3_reference_path: Optional[str] = Field(
        None,
        max_length=1000,
        description="Path to Stage 3 reference data for schema validation"
    )
    strict_validation: bool = Field(
        True,
        description="Enable strict schema validation with all requirements"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "schedule_csv_path": "/data/schedule.csv",
                "output_model_json_path": "/data/output_model.json",
                "strict_validation": True
            }
        }


class SchemaValidationResponse(BaseModel):
    """
    Schema validation response model with detailed results
    
    Provides comprehensive schema validation results with specific
    error details and compliance information for each validated file.
    """
    overall_valid: bool = Field(
        ...,
        description="Overall schema validation status"
    )
    validation_results: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Detailed validation results per file type"
    )
    timestamp: float = Field(
        ...,
        description="Validation completion timestamp"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "overall_valid": True,
                "validation_results": {
                    "schedule_csv": {
                        "valid": True,
                        "missing_columns": [],
                        "total_records": 1247,
                        "column_count": 14
                    },
                    "output_model_json": {
                        "valid": True,
                        "missing_fields": [],
                        "field_count": 8
                    }
                },
                "timestamp": 1699123456.789
            }
        }


class SystemDiagnosticsResponse(BaseModel):
    """
    System diagnostics response model with comprehensive information
    
    Provides detailed system diagnostics including platform information,
    component status, configuration, and performance data.
    """
    timestamp: float = Field(
        ...,
        description="Diagnostics generation timestamp"
    )
    system_info: Dict[str, Any] = Field(
        ...,
        description="System platform and hardware information"
    )
    component_info: Dict[str, Any] = Field(
        ...,
        description="Component versions and availability status"
    )
    configuration_info: Dict[str, Any] = Field(
        ...,
        description="Current system configuration summary"
    )
    performance_info: Dict[str, Any] = Field(
        ...,
        description="Performance metrics and statistics"
    )
    uptime_seconds: float = Field(
        ...,
        ge=0.0,
        description="System uptime in seconds"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if diagnostics collection failed"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": 1699123456.789,
                "system_info": {
                    "platform": "Linux-5.15.0-89-generic-x86_64",
                    "python_version": "3.11.6",
                    "cpu_count": 8,
                    "memory_total_mb": 16384.0
                },
                "component_info": {
                    "fastapi_available": True,
                    "pandas_version": "2.1.3",
                    "stage_7_1_available": True
                },
                "configuration_info": {
                    "api_version": "7.0.0",
                    "debug_mode": False,
                    "authentication_enabled": True
                },
                "performance_info": {
                    "recent_requests": 100,
                    "average_response_time": 2.34,
                    "success_rate": 0.96
                },
                "uptime_seconds": 86400.0
            }
        }


# === ASYNC OPERATION MODELS ===

class AsyncOperationResponse(BaseModel):
    """
    Asynchronous operation response model
    
    Provides operation tracking information for long-running async
    validation and formatting operations with status monitoring.
    """
    operation_id: str = Field(
        ...,
        description="Unique operation identifier for status tracking"
    )
    status: str = Field(
        ...,
        regex="^(pending|running|completed|failed|cancelled)$",
        description="Current operation status"
    )
    progress_percentage: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Operation progress percentage (0-100)"
    )
    estimated_completion_time: Optional[float] = Field(
        None,
        description="Estimated completion timestamp (Unix time)"
    )
    result_url: Optional[str] = Field(
        None,
        description="URL for result retrieval once completed"
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if operation failed"
    )
    created_timestamp: float = Field(
        ...,
        description="Operation creation timestamp"
    )
    last_updated: float = Field(
        ...,
        description="Last status update timestamp"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "operation_id": "async_11111111-1111-1111-1111-111111111111",
                "status": "running",
                "progress_percentage": 67.5,
                "estimated_completion_time": 1699123556.789,
                "result_url": "/api/results/async_111111...",
                "created_timestamp": 1699123456.789,
                "last_updated": 1699123500.123
            }
        }


# === ERROR RESPONSE MODELS ===

class ErrorResponse(BaseModel):
    """
    Standardized error response model
    
    Provides consistent error response structure with detailed error
    information, troubleshooting guidance, and operation context.
    """
    error: str = Field(
        ...,
        description="Error message"
    )
    error_code: Optional[str] = Field(
        None,
        description="Machine-readable error code"
    )
    status_code: int = Field(
        ...,
        ge=400,
        le=599,
        description="HTTP status code"
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details and context"
    )
    troubleshooting: Optional[str] = Field(
        None,
        description="Troubleshooting guidance for error resolution"
    )
    timestamp: float = Field(
        ...,
        description="Error occurrence timestamp"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request identifier for error tracking"
    )
    endpoint: Optional[str] = Field(
        None,
        description="API endpoint where error occurred"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Validation failed: Course coverage ratio below threshold",
                "error_code": "VALIDATION_THRESHOLD_VIOLATION",
                "status_code": 400,
                "details": {
                    "threshold_parameter": "course_coverage_ratio",
                    "actual_value": 0.89,
                    "required_minimum": 0.95
                },
                "troubleshooting": "Ensure all required courses are included in schedule",
                "timestamp": 1699123456.789,
                "request_id": "req_12345",
                "endpoint": "/validate/complete"
            }
        }


# === CONVENIENCE TYPE ALIASES ===

# Request/Response type aliases for common operations
ValidationRequestType = ValidationRequest
ValidationResponseType = ValidationResponse
FormatConversionRequestType = FormatConversionRequest
FormatConversionResponseType = FormatConversionResponse

# Configuration type aliases
ThresholdConfigType = ThresholdConfigurationRequest
DepartmentOrderType = DepartmentOrderingRequest

# Monitoring type aliases
HealthCheckType = HealthCheckResponse
MetricsResponseType = APIMetricsResponse
AuditResponseType = AuditTrailResponse

# Error handling type aliases
StandardErrorType = ErrorResponse
ValidationErrorType = ErrorResponse

# Export all models for API integration
__all__ = [
    # Enums
    "ValidationStatus",
    "ProcessingStatus", 
    "OutputFormat",
    "InstitutionalStandard",
    "SortingStrategy",
    "ErrorCategory",
    "SystemHealthStatus",
    
    # Base models
    "ThresholdBoundsModel",
    "ValidationConfigModel", 
    "FormatConfigModel",
    
    # Validation models
    "ValidationRequest",
    "ValidationResultModel",
    "ValidationResponse",
    
    # Format conversion models
    "FormatConversionRequest",
    "FormatConversionResponse",
    
    # Batch processing models
    "BatchValidationRequest",
    "BatchValidationResponse",
    
    # Configuration models
    "ThresholdConfigurationRequest",
    "ThresholdConfigurationResponse",
    "DepartmentOrderingRequest", 
    "DepartmentOrderingResponse",
    
    # Monitoring models
    "HealthCheckResponse",
    "APIMetricsResponse",
    "AuditTrailRequest",
    "AuditTrailResponse",
    
    # Utility models
    "FileUploadResponse",
    "SchemaValidationRequest",
    "SchemaValidationResponse", 
    "SystemDiagnosticsResponse",
    "AsyncOperationResponse",
    
    # Error models
    "ErrorResponse",
    
    # Type aliases
    "ValidationRequestType",
    "ValidationResponseType",
    "FormatConversionRequestType",
    "FormatConversionResponseType",
    "ThresholdConfigType",
    "DepartmentOrderType",
    "HealthCheckType",
    "MetricsResponseType",
    "AuditResponseType",
    "StandardErrorType",
    "ValidationErrorType"
]