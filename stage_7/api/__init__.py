# api/__init__.py
"""
Stage 7 API Package - FastAPI Integration Module for Schedule Validation System

This package provides complete REST API endpoints for the Stage 7 Output Validation
system, enabling web-based integration and configuration of the timetable validation and
human-readable format generation pipeline.

Following the theoretical framework from Stage-7-OUTPUT-VALIDATION-Theoretical-Foundation-
Mathematical-Framework.pdf, this API module exposes all critical Stage 7 functionality
through well-defined RESTful endpoints with complete configuration options.

CRITICAL DESIGN PHILOSOPHY: EXHAUSTIVE CONFIGURATION OPTIONS
This API module provides an extensive array of configuration endpoints and options to
enable the development team to customize every aspect of the validation and formatting
process according to institutional requirements and usage scenarios.

Mathematical Foundation:
- Based on Stage 7 Complete Framework (Algorithms 15.1, 3.2, 4.3)
- Implements 12-parameter threshold validation per theoretical requirements
- Supports multi-format output generation with institutional compliance
- Provides complete audit logging and performance monitoring

Theoretical Compliance:
- Stage 7.1 Validation Engine (12-parameter threshold validation)
- Stage 7.2 Human-Readable Format Generation (educational domain optimization)
- Fail-fast validation philosophy with complete error reporting
- Master pipeline integration with downward communication support

API Architecture:
1. app.py: Main FastAPI application with complete endpoint definitions
2. schemas.py: Pydantic models for request/response validation and documentation

Endpoint Categories:
1. Validation Endpoints: Complete schedule validation with configurable thresholds
2. Format Conversion Endpoints: Human-readable timetable generation
3. Configuration Endpoints: Threshold bounds, department ordering, output formats
4. Monitoring Endpoints: Performance metrics, audit trails, system health
5. Utility Endpoints: Schema validation, data integrity checks, system information

Integration Points:
- Input: Schedule.csv and output_model.json from Stage 6
- Reference: Stage 3 compiled data for validation context
- Output: Validated schedule, validation analysis JSON, human-readable timetable
- Pipeline: Master orchestrator communication with configurable parameters

Performance Requirements:
- API Response Time: <10 seconds for complete validation pipeline
- Concurrent Requests: Support for multiple simultaneous validation operations
- Memory Usage: <512MB per validation request
- Scalability: Horizontal scaling support with stateless operation design

Quality Assurance:
- complete request/response validation using Pydantic models
- Detailed API documentation with OpenAPI/Swagger integration
- Extensive error handling with structured error responses
- Complete audit logging for all API operations and configurations

type hints, detailed docstrings, and cross-file references for intelligent
code completion and API development support.

Author: Student Team
Version: Stage 7 API - Phase 4 Implementation

"""

from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass
from enum import Enum
import warnings

# Configure logging for Stage 7 API operations
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress warnings for cleaner API response handling
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Import FastAPI and related dependencies with availability checking
try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
    from fastapi.openapi.utils import get_openapi
    _FASTAPI_AVAILABLE = True
    logger.info("FastAPI dependencies imported successfully")
except ImportError as e:
    logger.error(f"Failed to import FastAPI dependencies: {e}")
    _FASTAPI_AVAILABLE = False

# Import Pydantic models with availability checking
try:
    from pydantic import BaseModel, Field, validator, root_validator
    from pydantic.typing import Union as PydanticUnion
    from pydantic.dataclasses import dataclass as pydantic_dataclass
    _PYDANTIC_AVAILABLE = True
    logger.info("Pydantic dependencies imported successfully")
except ImportError as e:
    logger.error(f"Failed to import Pydantic dependencies: {e}")
    _PYDANTIC_AVAILABLE = False

# Import uvicorn for standalone server support
try:
    import uvicorn
    _UVICORN_AVAILABLE = True
    logger.info("Uvicorn server support available")
except ImportError as e:
    logger.warning(f"Uvicorn not available for standalone server: {e}")
    _UVICORN_AVAILABLE = False

# Import Stage 7 core modules with error handling
try:
    from ..stage_7_1_validation import (
        Stage71ValidationEngine,
        ValidationConfig,
        ValidationResult,
        ThresholdBounds,
        ErrorCategory,
        ValidationError as Stage71ValidationError
    )
    _STAGE_71_AVAILABLE = True
    logger.info("Stage 7.1 validation module imported successfully")
except ImportError as e:
    logger.error(f"Failed to import Stage 7.1 validation module: {e}")
    _STAGE_71_AVAILABLE = False

try:
    from ..stage_7_2_finalformat import (
        Stage72Pipeline,
        Stage72Config,
        Stage72Result,
        ProcessingStatus,
        HumanFormatError
    )
    _STAGE_72_AVAILABLE = True
    logger.info("Stage 7.2 final format module imported successfully")
except ImportError as e:
    logger.error(f"Failed to import Stage 7.2 final format module: {e}")
    _STAGE_72_AVAILABLE = False

# Import schemas module
try:
    from .schemas import (
        ValidationRequest,
        ValidationResponse,
        FormatConversionRequest,
        FormatConversionResponse,
        ConfigurationRequest,
        ConfigurationResponse,
        SystemStatusResponse,
        ErrorResponse,
        AuditLogResponse,
        PerformanceMetricsResponse
    )
    _SCHEMAS_AVAILABLE = True
    logger.info("API schemas imported successfully")
except ImportError as e:
    logger.error(f"Failed to import API schemas: {e}")
    _SCHEMAS_AVAILABLE = False

# Import main app module
try:
    from .app import (
        create_stage7_app,
        Stage7APIConfig,
        APISecurityConfig,
        CORSConfig,
        RateLimitConfig
    )
    _APP_AVAILABLE = True
    logger.info("Main API application imported successfully")
except ImportError as e:
    logger.error(f"Failed to import main API application: {e}")
    _APP_AVAILABLE = False

# API-level enums and constants
class APIStatus(Enum):
    """
    API status enumeration for Stage 7 system status tracking
    
    Provides complete status tracking for all API operations
    including validation, formatting, and system health monitoring.
    """
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    UNAVAILABLE = "unavailable"
    ERROR = "error"

class EndpointCategory(Enum):
    """
    Endpoint category enumeration for API organization and routing
    
    Categorizes API endpoints based on functionality for improved
    documentation, monitoring, and access control.
    """
    VALIDATION = "validation"
    FORMATTING = "formatting"
    CONFIGURATION = "configuration"
    MONITORING = "monitoring"
    UTILITY = "utility"
    HEALTH = "health"

class ResponseFormat(Enum):
    """
    Response format enumeration for API output customization
    
    Supports multiple response formats to accommodate different
    integration requirements and client preferences.
    """
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    EXCEL = "excel"
    BINARY = "binary"

@dataclass
class APIConfiguration:
    """
    complete API configuration class for Stage 7 system
    
    Encapsulates all configuration parameters required for API operation
    including security, performance, monitoring, and integration settings.
    
    Mathematical Foundation:
    Based on Stage 7 theoretical framework requirements for API integration
    and configuration management with institutional customization support.
    """
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 1
    
    # API metadata
    title: str = "Stage 7 Timetable Validation API"
    description: str = "complete timetable validation and formatting API"
    version: str = "7.0.0"
    contact_name: str = "Team LUMEN"
    contact_email: str = ""
    
    # Security configuration
    enable_authentication: bool = False
    api_key_header: str = "X-API-Key"
    cors_origins: List[str] = None
    cors_methods: List[str] = None
    
    # Performance configuration
    max_request_size_mb: int = 100
    request_timeout_seconds: int = 300
    enable_compression: bool = True
    enable_rate_limiting: bool = False
    
    # Validation configuration
    default_threshold_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    default_department_order: List[str] = None
    enable_async_processing: bool = False
    
    # Monitoring configuration
    enable_metrics_collection: bool = True
    enable_audit_logging: bool = True
    log_level: str = "INFO"
    metrics_retention_days: int = 30
    
    def __post_init__(self):
        """Post-initialization validation and default value assignment"""
        if self.cors_origins is None:
            self.cors_origins = ["*"]  # Allow all origins for development
        
        if self.cors_methods is None:
            self.cors_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        
        if self.default_department_order is None:
            self.default_department_order = [
                "CSE", "ME", "CHE", "EE", "ECE", "CE", "IT", "BT", "MT", 
                "PI", "EP", "IC", "AE", "AS", "CH", "CY", "PH", "MA", "HS"
            ]
        
        if self.default_threshold_bounds is None:
            # Default threshold bounds per Stage 7 theoretical framework
            self.default_threshold_bounds = {
                "course_coverage_ratio": (0.95, 1.0),
                "conflict_resolution_rate": (1.0, 1.0),
                "faculty_workload_balance": (0.85, 1.0),
                "room_utilization_efficiency": (0.60, 0.85),
                "student_schedule_density": (0.70, 0.95),
                "pedagogical_sequence_compliance": (1.0, 1.0),
                "faculty_preference_satisfaction": (0.75, 1.0),
                "resource_diversity_index": (0.30, 0.70),
                "constraint_violation_penalty": (0.0, 0.20),
                "solution_stability_index": (0.90, 1.0),
                "computational_quality_score": (0.70, 0.95),
                "multi_objective_balance": (0.85, 1.0)
            }
        
        # Validate configuration parameters
        if self.port < 1 or self.port > 65535:
            raise ValueError("Port must be between 1 and 65535")
        
        if self.max_request_size_mb <= 0:
            raise ValueError("max_request_size_mb must be positive")
        
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")

class Stage7API:
    """
    Main Stage 7 API orchestrator class
    
    Coordinates the complete API infrastructure including FastAPI application
    creation, middleware configuration, endpoint registration, and system
    monitoring for the Stage 7 timetable validation and formatting system.
    
    Theoretical Foundation:
    Implements Stage 7 complete framework API integration requirements with
    complete configuration support and institutional customization.
    
    Integration Architecture:
    - Stage 7.1: 12-parameter validation engine integration
    - Stage 7.2: Human-readable format generation integration  
    - Pipeline: Master orchestrator communication support
    - Monitoring: Performance metrics and audit trail collection
    """
    
    def __init__(self, config: APIConfiguration = None):
        """
        Initialize Stage 7 API with complete configuration
        
        Args:
            config: APIConfiguration instance with system parameters
                   If None, uses default configuration optimized for development
        """
        self.config = config or APIConfiguration()
        self.logger = logging.getLogger(f"{__name__}.Stage7API")
        
        # Verify dependency availability
        self._verify_dependencies()
        
        # Initialize API application
        self.app = None
        self._initialize_application()
        
        # Setup monitoring and metrics
        self._setup_monitoring()
        
        self.logger.info("Stage 7 API initialized successfully")
    
    def _verify_dependencies(self):
        """Verify all required dependencies are available"""
        missing_dependencies = []
        
        if not _FASTAPI_AVAILABLE:
            missing_dependencies.append("FastAPI")
        if not _PYDANTIC_AVAILABLE:
            missing_dependencies.append("Pydantic")
        if not _STAGE_71_AVAILABLE:
            missing_dependencies.append("Stage 7.1 Validation")
        if not _STAGE_72_AVAILABLE:
            missing_dependencies.append("Stage 7.2 Final Format")
        if not _SCHEMAS_AVAILABLE:
            missing_dependencies.append("API Schemas")
        if not _APP_AVAILABLE:
            missing_dependencies.append("Main Application")
        
        if missing_dependencies:
            error_msg = f"Critical API dependencies unavailable: {missing_dependencies}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _initialize_application(self):
        """Initialize FastAPI application with complete configuration"""
        try:
            # Create Stage 7 API configuration
            api_config = Stage7APIConfig(
                title=self.config.title,
                description=self.config.description,
                version=self.config.version,
                contact={
                    "name": self.config.contact_name,
                    "email": self.config.contact_email
                },
                debug=self.config.debug
            )
            
            # Create FastAPI application
            self.app = create_stage7_app(api_config)
            
            # Configure middleware
            self._configure_middleware()
            
            self.logger.info("FastAPI application initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize API application: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _configure_middleware(self):
        """Configure FastAPI middleware for security, performance, and monitoring"""
        if self.app is None:
            return
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=self.config.cors_methods,
            allow_headers=["*"]
        )
        
        # Add compression middleware
        if self.config.enable_compression:
            self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        self.logger.info("API middleware configured successfully")
    
    def _setup_monitoring(self):
        """Setup performance monitoring and metrics collection"""
        self.metrics = {
            'requests_total': 0,
            'requests_successful': 0,
            'requests_failed': 0,
            'average_response_time': 0.0,
            'peak_memory_usage_mb': 0.0,
            'validation_operations': 0,
            'formatting_operations': 0
        }
        
        self.audit_log = []
        
        if self.config.enable_metrics_collection:
            self.logger.info("Performance monitoring enabled")
    
    def run_server(self, **kwargs):
        """
        Run the Stage 7 API server using uvicorn
        
        Args:
            **kwargs: Additional uvicorn configuration options
        """
        if not _UVICORN_AVAILABLE:
            raise RuntimeError("Uvicorn not available for standalone server operation")
        
        if self.app is None:
            raise RuntimeError("API application not initialized")
        
        # Merge configuration with provided kwargs
        server_config = {
            "app": self.app,
            "host": self.config.host,
            "port": self.config.port,
            "reload": self.config.reload,
            "workers": self.config.workers,
            "log_level": self.config.log_level.lower()
        }
        server_config.update(kwargs)
        
        self.logger.info(f"Starting Stage 7 API server on {self.config.host}:{self.config.port}")
        uvicorn.run(**server_config)
    
    def get_application(self) -> Optional['FastAPI']:
        """
        Get the FastAPI application instance for external WSGI/ASGI usage
        
        Returns:
            FastAPI application instance or None if not initialized
        """
        return self.app
    
    def get_openapi_schema(self) -> Dict[str, Any]:
        """
        Get the OpenAPI schema for the API
        
        Returns:
            OpenAPI schema dictionary for documentation and client generation
        """
        if self.app is None:
            raise RuntimeError("API application not initialized")
        
        return get_openapi(
            title=self.config.title,
            version=self.config.version,
            description=self.config.description,
            routes=self.app.routes
        )
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform complete API health check
        
        Returns:
            Health status dictionary with system information and diagnostics
        """
        health_status = {
            "status": APIStatus.HEALTHY.value,
            "timestamp": time.time(),
            "version": self.config.version,
            "dependencies": {
                "fastapi": _FASTAPI_AVAILABLE,
                "pydantic": _PYDANTIC_AVAILABLE,
                "uvicorn": _UVICORN_AVAILABLE,
                "stage_7_1": _STAGE_71_AVAILABLE,
                "stage_7_2": _STAGE_72_AVAILABLE,
                "schemas": _SCHEMAS_AVAILABLE,
                "app": _APP_AVAILABLE
            },
            "metrics": self.metrics.copy() if hasattr(self, 'metrics') else {},
            "configuration": {
                "debug": self.config.debug,
                "authentication_enabled": self.config.enable_authentication,
                "compression_enabled": self.config.enable_compression,
                "rate_limiting_enabled": self.config.enable_rate_limiting
            }
        }
        
        # Determine overall health status based on critical dependencies
        critical_deps = ["fastapi", "stage_7_1", "stage_7_2"]
        if not all(health_status["dependencies"][dep] for dep in critical_deps):
            health_status["status"] = APIStatus.DEGRADED.value
        
        return health_status

# Package-level convenience functions
def create_api_instance(config: APIConfiguration = None) -> Stage7API:
    """
    Convenience function for creating Stage 7 API instance
    
    Args:
        config: Optional APIConfiguration for customization
    
    Returns:
        Stage7API: Configured API instance ready for usage
    """
    return Stage7API(config)

def run_development_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = True,
    debug: bool = True
):
    """
    Convenience function for running development server
    
    Args:
        host: Server host address
        port: Server port number
        reload: Enable auto-reload for development
        debug: Enable debug mode
    """
    config = APIConfiguration(
        host=host,
        port=port,
        reload=reload,
        debug=debug
    )
    
    api = Stage7API(config)
    api.run_server()

# Package metadata and version information
__version__ = "7.0.0"
__author__ = "Student Team"
__description__ = "Stage 7 API Package - FastAPI Integration Module"
__theoretical_foundation__ = "Stage 7 Complete Framework API Integration"

# Export public interface
__all__ = [
    # Main classes
    'Stage7API',
    'APIConfiguration',
    'APIStatus',
    'EndpointCategory',
    'ResponseFormat',
    
    # FastAPI components (if available)
    'FastAPI',
    'HTTPException',
    'Depends',
    'Request',
    'Response',
    
    # App components (if available)
    'create_stage7_app',
    'Stage7APIConfig',
    'APISecurityConfig',
    'CORSConfig',
    'RateLimitConfig',
    
    # Schema components (if available)
    'ValidationRequest',
    'ValidationResponse',
    'FormatConversionRequest',
    'FormatConversionResponse',
    'ConfigurationRequest',
    'ConfigurationResponse',
    'SystemStatusResponse',
    'ErrorResponse',
    'AuditLogResponse',
    'PerformanceMetricsResponse',
    
    # Convenience functions
    'create_api_instance',
    'run_development_server',
    
    # Availability flags
    '_FASTAPI_AVAILABLE',
    '_PYDANTIC_AVAILABLE',
    '_UVICORN_AVAILABLE',
    '_STAGE_71_AVAILABLE',
    '_STAGE_72_AVAILABLE',
    '_SCHEMAS_AVAILABLE',
    '_APP_AVAILABLE'
]

# Conditional exports based on availability
if _FASTAPI_AVAILABLE:
    __all__.extend([
        'CORSMiddleware',
        'GZipMiddleware',
        'JSONResponse',
        'FileResponse',
        'StreamingResponse',
        'HTTPBearer',
        'HTTPAuthorizationCredentials'
    ])

if _PYDANTIC_AVAILABLE:
    __all__.extend([
        'BaseModel',
        'Field',
        'validator',
        'root_validator'
    ])

if _UVICORN_AVAILABLE:
    __all__.extend(['uvicorn'])

# Package initialization logging
logger.info("Stage 7 API package initialized successfully")
logger.info(f"Dependency availability - FastAPI: {_FASTAPI_AVAILABLE}, Pydantic: {_PYDANTIC_AVAILABLE}, "
           f"Uvicorn: {_UVICORN_AVAILABLE}, Stage 7.1: {_STAGE_71_AVAILABLE}, Stage 7.2: {_STAGE_72_AVAILABLE}")

# Final availability check and warning
if not all([_FASTAPI_AVAILABLE, _PYDANTIC_AVAILABLE, _STAGE_71_AVAILABLE, _STAGE_72_AVAILABLE]):
    logger.warning("Some critical dependencies are unavailable. API functionality may be limited.")
else:
    logger.info("All critical dependencies available. Full API functionality enabled.")