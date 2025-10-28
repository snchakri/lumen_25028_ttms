"""
Stage 4 Utilities Package
Contains logging, error handling, metrics calculation, and report generation
"""

from .logger import StructuredLogger, create_logger
from .error_handler import (
    ErrorHandler,
    FeasibilityError,
    LayerValidationError,
    TheoremViolationError,
    Stage3InputError,
    MathematicalProofError,
    ErrorSeverity,
    ErrorCategory
)
from .metrics_calculator import CrossLayerMetricsCalculator
from .report_generator import FeasibilityReportGenerator

__all__ = [
    # Logger
    "StructuredLogger",
    "create_logger",
    
    # Error Handling
    "ErrorHandler",
    "FeasibilityError",
    "LayerValidationError",
    "TheoremViolationError",
    "Stage3InputError",
    "MathematicalProofError",
    "ErrorSeverity",
    "ErrorCategory",
    
    # Metrics and Reporting
    "CrossLayerMetricsCalculator",
    "FeasibilityReportGenerator"
]