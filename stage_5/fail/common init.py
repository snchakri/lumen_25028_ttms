# STAGE 5 - COMMON/__INIT__.PY
# Package initialization for Stage 5 common utilities

"""
STAGE 5 COMMON PACKAGE
Enterprise-Grade Common Utilities for Stage 5 Input-Complexity Analysis and Solver Selection

This package provides the foundational components for Stage 5's rigorous mathematical framework
implementation, including schema definitions, exception handling, structured logging, and 
utility functions for file I/O and configuration management.

Package Components:
- schema.py: Pydantic V2 models for Stage 5.1 and 5.2 data contracts
- exceptions.py: Comprehensive exception hierarchy for fail-fast error handling
- logging.py: Structured JSON logging framework for audit trails and debugging
- utils.py: Enterprise-grade file I/O, validation, and configuration utilities

Critical Implementation Notes:
- NO MOCK IMPLEMENTATIONS: All components perform real operations with full validation
- ENTERPRISE RELIABILITY: Fail-fast error handling prevents corruption propagation
- THEORETICAL COMPLIANCE: All schemas align with mathematical framework specifications
- AUDIT TRAIL SUPPORT: Complete execution tracking for debugging and compliance
- PRODUCTION READY: Suitable for enterprise deployment with comprehensive error handling

Usage Example:
```python
from stage5.common import (
    ComplexityMetricsSchema, SolverSelectionSchema,
    Stage5FileHandler, Stage5ConfigManager,
    setup_stage5_logging, get_stage5_logger
)

# Setup logging
logger = setup_stage5_logging(log_directory='./logs', execution_id='exec_001')

# Initialize file handler
file_handler = Stage5FileHandler(logger=logger)

# Load Stage 3 outputs
l_raw_data, l_rel_graph, l_idx_data = file_handler.load_stage3_outputs(
    l_raw_path='./data/L_raw.parquet',
    l_rel_path='./data/L_rel.graphml', 
    l_idx_path='./data/L_idx.feather'
)

# Process complexity metrics (Stage 5.1 implementation)
complexity_metrics = ComplexityMetricsSchema(...)

# Save results
file_handler.save_json_atomically(
    data=complexity_metrics.model_dump(),
    output_file=Path('./output/complexity_metrics.json')
)
```

References:
- Stage5-FOUNDATIONAL-DESIGN-IMPLEMENTATION-PLAN.md: Complete implementation specifications
- Stage-5.1-INPUT-COMPLEXITY-ANALYSIS: Mathematical parameter definitions
- Stage-5.2-SOLVER-SELECTION-ARSENAL-MODULARITY: L2 normalization and LP optimization
"""

# Import all schema definitions
from .schema import (
    # Enumerations
    SolverParadigmEnum, LPConvergenceStatus, FileFormatEnum,
    
    # Stage 5.1 Schemas
    ExecutionMetadata, EntityCounts, ComputationNotes,
    ComplexityParameters, ComplexityMetricsSchema,
    
    # Stage 5.2 Schemas  
    SolverLimits, SolverCapability, SolverArsenalSchema,
    SolverChoice, SolverRanking, LPConvergenceInfo,
    OptimizationDetails, SelectionResult, SolverSelectionSchema,
    
    # Shared Schemas
    Stage3OutputPaths, ExecutionContext
)

# Import exception hierarchy
from .exceptions import (
    # Base Exception Classes
    Stage5BaseException, Stage5ValidationError, Stage5ComputationError,
    
    # Stage 5.1 Exceptions
    Stage51InputError, Stage51ParameterComputationError, Stage51OutputError,
    
    # Stage 5.2 Exceptions
    Stage52SolverCapabilityError, Stage52NormalizationError,
    Stage52LPOptimizationError, Stage52SelectionOutputError,
    
    # Integration Exceptions
    Stage5ConfigurationError, Stage5IntegrationError, Stage5TimeoutError,
    
    # Utility Functions
    format_validation_error, format_computation_error
)

# Import logging framework
from .logging import (
    Stage5JSONFormatter, Stage5LoggingConfig, Stage5LoggerAdapter,
    performance_timing_context, setup_stage5_logging, get_stage5_logger,
    log_stage5_exception
)

# Import utility functions
from .utils import (
    # Exception Classes (from utils)
    Stage5UtilsError, Stage5FileError, Stage5ValidationUtilsError, Stage5ConfigError,
    
    # File Handler Class
    Stage5FileHandler,
    
    # Schema Validation
    validate_pydantic_model, extract_validation_errors,
    
    # Configuration Management  
    Stage5ConfigManager,
    
    # Mathematical Utilities
    safe_divide, safe_log, compute_coefficient_of_variation, compute_entropy,
    
    # Path Utilities
    ensure_directory_exists, generate_execution_id, validate_file_permissions
)

# Package version and metadata
__version__ = "1.0.0"
__author__ = "LUMEN Team - SIH 2025"
__description__ = "Stage 5 Common Utilities for Input-Complexity Analysis and Solver Selection"

# Define package-level exports for clean imports
__all__ = [
    # Version and Metadata
    "__version__", "__author__", "__description__",
    
    # Schema Components
    "SolverParadigmEnum", "LPConvergenceStatus", "FileFormatEnum",
    "ExecutionMetadata", "EntityCounts", "ComputationNotes",
    "ComplexityParameters", "ComplexityMetricsSchema",
    "SolverLimits", "SolverCapability", "SolverArsenalSchema", 
    "SolverChoice", "SolverRanking", "LPConvergenceInfo",
    "OptimizationDetails", "SelectionResult", "SolverSelectionSchema",
    "Stage3OutputPaths", "ExecutionContext",
    
    # Exception Components
    "Stage5BaseException", "Stage5ValidationError", "Stage5ComputationError",
    "Stage51InputError", "Stage51ParameterComputationError", "Stage51OutputError",
    "Stage52SolverCapabilityError", "Stage52NormalizationError",
    "Stage52LPOptimizationError", "Stage52SelectionOutputError",
    "Stage5ConfigurationError", "Stage5IntegrationError", "Stage5TimeoutError",
    "format_validation_error", "format_computation_error",
    
    # Logging Components
    "Stage5JSONFormatter", "Stage5LoggingConfig", "Stage5LoggerAdapter", 
    "performance_timing_context", "setup_stage5_logging", "get_stage5_logger",
    "log_stage5_exception",
    
    # Utility Components
    "Stage5UtilsError", "Stage5FileError", "Stage5ValidationUtilsError", "Stage5ConfigError",
    "Stage5FileHandler", "validate_pydantic_model", "extract_validation_errors",
    "Stage5ConfigManager", "safe_divide", "safe_log", "compute_coefficient_of_variation", 
    "compute_entropy", "ensure_directory_exists", "generate_execution_id", 
    "validate_file_permissions"
]

# Package initialization logging
import logging
_logger = logging.getLogger(__name__)
_logger.info(f"Stage 5 Common Package initialized - Version {__version__}")
_logger.debug(f"Exported components: {len(__all__)} total")

# Validate critical dependencies at package import
try:
    import pandas
    import numpy  
    import networkx
    import pyarrow
    import pydantic
    _logger.debug("All critical dependencies available")
except ImportError as e:
    _logger.warning(f"Some dependencies missing: {e}")
    _logger.warning("Stage 5 functionality may be limited - ensure all required packages are installed")

print("✅ STAGE 5 COMMON/__INIT__.PY - COMPLETE")
print("   - Package initialization with comprehensive component exports")
print("   - Version metadata and dependency validation") 
print("   - Clean import interface for Stage 5 common utilities")
print(f"   - Total exported components: {len(__all__)}")