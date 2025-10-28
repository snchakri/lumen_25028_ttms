"""
Error Handling and Reporting for Stage 5

Implements comprehensive error reporting with JSON/TXT outputs.

Author: LUMEN TTMS
Version: 2.0.0
"""

from .error_reporter import ErrorReporter, ErrorReport
from .foundation_exceptions import (
    FoundationGapError,
    TheoremViolationError,
    ParameterValidationError,
    DataLoadingError,
    SolverSelectionError
)

__all__ = [
    'ErrorReporter',
    'ErrorReport',
    'FoundationGapError',
    'TheoremViolationError',
    'ParameterValidationError',
    'DataLoadingError',
    'SolverSelectionError'
]


