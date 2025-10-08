"""
Stage 6.4 PyGMO Solver Family - API Package
====================================================

Enterprise FastAPI-based REST API endpoints for PyGMO solver family integration 
with master data pipeline orchestration system.

Mathematical Framework Integration:
- Pydantic schema validation per Definition 12.3
- RESTful endpoint design per Algorithm 13.1  
- Master pipeline webhook support per Specification 13.2
- Enterprise error handling with structured audit trails

Author: Student Team
"""

from typing import Dict, Any, Optional
import logging
from fastapi import FastAPI
from .app import create_pygmo_api_app
from .schemas import (
    # Request/Response models
    OptimizationRequest, OptimizationResponse,
    HealthCheckResponse, ConfigurationRequest,
    ValidationRequest, ValidationResponse,
    # Status and metadata models
    StatusResponse, MetricsResponse,
    ErrorResponse, LogResponse
)

# Configure enterprise logging for API layer
logger = logging.getLogger(__name__)

# API metadata and version information
__version__ = "1.0.0"
__api_title__ = "PyGMO Solver Family API"
__description__ = "Enterprise REST API for PyGMO-based scheduling optimization"

# Export all public API components
__all__ = [
    # Core application factory
    "create_pygmo_api_app",
    "get_api_metadata",
    # Schema models
    "OptimizationRequest", "OptimizationResponse", 
    "HealthCheckResponse", "ConfigurationRequest",
    "ValidationRequest", "ValidationResponse",
    "StatusResponse", "MetricsResponse",
    "ErrorResponse", "LogResponse",
    # Utility functions
    "validate_api_configuration",
    "initialize_api_logging"
]

def get_api_metadata() -> Dict[str, Any]:
    """
    Return complete API metadata for master pipeline integration.
    
    Implements Algorithm 13.3 (API Metadata Specification) ensuring
    complete compatibility information for external orchestration.
    
    Returns:
        Dict containing version, endpoints, schemas, and integration specs
    """
    return {
        "version": __version__,
        "title": __api_title__,
        "description": __description__,
        "solver_family": "pygmo",
        "supported_algorithms": ["nsga2", "moead", "mopso", "de", "sa"],
        "api_specification": "OpenAPI 3.0",
        "integration_type": "REST + Webhooks",
        "master_pipeline_compatible": True,
        "theoretical_compliance": "PyGMO Framework v2.3",
        "mathematical_guarantees": {
            "convergence": "Theorem 3.2 (NSGA-II)",
            "pareto_optimality": "Theorem 3.6 (MOEA/D)", 
            "hypervolume_monotonicity": "Theorem 7.2"
        }
    }

def validate_api_configuration(config: Dict[str, Any]) -> bool:
    """
    Validate API configuration against PyGMO framework requirements.
    
    Args:
        config: API configuration dictionary
        
    Returns:
        bool: True if configuration valid, raises ValidationError otherwise
        
    Raises:
        ValueError: For invalid configuration parameters
        TypeError: For incorrect configuration types
    """
    required_keys = [
        "host", "port", "debug_mode", "log_level",
        "cors_origins", "request_timeout", "max_request_size"
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate specific configuration constraints
    if not isinstance(config["port"], int) or not (1024 <= config["port"] <= 65535):
        raise ValueError("Port must be integer between 1024-65535")
        
    if config["log_level"] not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError("Invalid log_level specified")
        
    if config["request_timeout"] <= 0 or config["request_timeout"] > 3600:
        raise ValueError("Request timeout must be between 1-3600 seconds")
    
    logger.info("API configuration validated successfully")
    return True

def initialize_api_logging(log_level: str = "INFO") -> None:
    """
    Initialize structured logging for API operations.
    
    Configures logging per Enterprise Logging Standard with:
    - Structured JSON formatting for machine readability
    - complete context information for debugging
    - Integration with master pipeline audit systems
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    import structlog
    from pythonjsonlogger import jsonlogger
    
    # Configure structured logging with enterprise format
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Add JSON formatter for machine-readable logs
    json_handler = logging.StreamHandler()
    json_handler.setFormatter(jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
    ))
    
    # Configure structured logging processor chain
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True
    )
    
    logger.info("Enterprise API logging initialized", 
                log_level=log_level, api_version=__version__)