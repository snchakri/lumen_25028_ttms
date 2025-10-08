"""
stage_5_1/__init__.py
Stage 5.1 Input Complexity Analysis Module

This module implements the Stage 5.1 component of the scheduling engine,
providing mathematically rigorous 16-parameter complexity analysis.

Key Components:
- ComplexityParameterComputer: Core algorithm implementation for P1-P16
- Stage3DataLoader: Robust data loading from Stage 3 outputs
- CLI runner: Command-line interface for standalone execution  
- I/O utilities: JSON serialization and file management

The module follows enterprise-grade patterns with:
- Fail-fast error handling with structured exceptions
- Comprehensive logging with JSON output support
- Performance monitoring and resource constraints
- Mathematical accuracy verification and bounds checking

Integration Points:
- Input: Stage 3 outputs (L_raw.parquet, L_rel.graphml, L_idx.*)
- Output: complexity_metrics.json with 16 parameters + composite index
- Downstream: Stage 5.2 solver selection consumes this module's output

For detailed theoretical foundations, see:
- Stage-5.1-INPUT-COMPLEXITY-ANALYSIS-Theoretical-Foundations-Mathematical-Framework.pdf
- 16-Parameter mathematical definitions with exact formulations
- Composite index PCA-weighted calculation from empirical validation
"""

from .compute import ComplexityParameterComputer, Stage3DataLoader
from .io import load_stage3_inputs, write_complexity_metrics
from .runner import (
    run_stage_5_1_complete,
    execute_stage_5_1_complexity_analysis,
    create_execution_context,
)

# Version information aligned with foundational design document
__version__ = "1.0.0"
__stage__ = "5.1"
__description__ = "Input Complexity Analysis - 16 Parameter Mathematical Framework"

# Public API exports for external consumers
__all__ = [
    # Core computation classes
    "ComplexityParameterComputer",
    "Stage3DataLoader",
    
    # I/O functions
    "load_stage3_inputs", 
    "write_complexity_metrics",
    
    # Execution interfaces
    "run_stage_5_1_complete",
    "execute_stage_5_1_complexity_analysis",
    "create_execution_context",
    
    # Module metadata
    "__version__",
    "__stage__", 
    "__description__",
]

# Theoretical compliance validation - ensures exact mathematical implementations
# These constants are derived from the foundational framework document
PARAMETER_COUNT = 16
COMPOSITE_INDEX_PCA_DIMENSIONS = 16  
MATHEMATICAL_PRECISION_EPSILON = 1e-12
PERFORMANCE_TARGET_COMPLEXITY = "O(N log N)"
MEMORY_LIMIT_MB = 512
EXECUTION_TIME_LIMIT_SECONDS = 600

# Input format support matrix from foundational design
SUPPORTED_L_RAW_FORMATS = [".parquet"]
SUPPORTED_L_REL_FORMATS = [".graphml"] 
SUPPORTED_L_IDX_FORMATS = [".pkl", ".parquet", ".feather", ".idx", ".bin"]

# Output schema version - must match foundational design specification
OUTPUT_SCHEMA_VERSION = "1.0.0"

def get_stage_info():
    """
    Get comprehensive Stage 5.1 module information.
    
    Returns:
        Dict containing version, capabilities, and compliance information
    """
    return {
        "version": __version__,
        "stage": __stage__,
        "description": __description__,
        "capabilities": {
            "parameter_count": PARAMETER_COUNT,
            "composite_index_dimensions": COMPOSITE_INDEX_PCA_DIMENSIONS,
            "supported_input_formats": {
                "l_raw": SUPPORTED_L_RAW_FORMATS,
                "l_rel": SUPPORTED_L_REL_FORMATS, 
                "l_idx": SUPPORTED_L_IDX_FORMATS,
            },
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
        },
        "performance_characteristics": {
            "computational_complexity": PERFORMANCE_TARGET_COMPLEXITY,
            "memory_limit_mb": MEMORY_LIMIT_MB,
            "execution_time_limit_seconds": EXECUTION_TIME_LIMIT_SECONDS,
            "mathematical_precision": MATHEMATICAL_PRECISION_EPSILON,
        },
        "theoretical_compliance": {
            "framework_document": "Stage-5.1-INPUT-COMPLEXITY-ANALYSIS-Theoretical-Foundations-Mathematical-Framework.pdf",
            "parameter_definitions": "Exact mathematical formulations implemented",
            "composite_index": "PCA-weighted from empirical 500-problem validation",
            "numerical_methods": "Information theory, graph analysis, statistical sampling",
        }
    }


# Module-level validation to ensure proper import structure
def _validate_module_imports():
    """
    Validate that all required dependencies are available and functional.
    
    This performs basic import validation to catch configuration issues early
    rather than failing during execution.
    """
    try:
        # Validate core computation dependencies
        import numpy
        import pandas  
        import networkx
        import scipy
        
        # Validate that our internal modules can import properly
        from ..common.schema import ComplexityParameterVector
        from ..common.exceptions import Stage5ValidationError
        from ..common.logging import get_logger
        
        return True
        
    except ImportError as e:
        import warnings
        warnings.warn(
            f"Stage 5.1 module dependency validation failed: {e}. "
            f"Some functionality may not be available.",
            ImportWarning
        )
        return False


# Perform import validation when module is loaded
_IMPORTS_VALID = _validate_module_imports()

# Provide programmatic access to validation status
def is_fully_functional():
    """
    Check if Stage 5.1 module is fully functional with all dependencies.
    
    Returns:
        bool: True if all dependencies are available and module is ready for use
    """
    return _IMPORTS_VALID


# Enterprise-grade module initialization logging
def _log_module_initialization():
    """Log module initialization for debugging and audit purposes."""
    try:
        from ..common.logging import get_logger
        logger = get_logger("stage5_1.init")
        
        logger.info(
            f"Stage 5.1 module initialized: version={__version__}, "
            f"functional={_IMPORTS_VALID}, parameter_count={PARAMETER_COUNT}"
        )
    except Exception:
        # Silently fail if logging is not available during import
        pass


# Initialize module logging
_log_module_initialization()