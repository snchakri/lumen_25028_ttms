"""
Error Handling Package for DEAP Solver Family

Implements comprehensive error handling, recovery mechanisms, and reporting
as per Stage 6.3 foundational requirements.

Author: LUMEN Team [TEAM-ID: 93912]
"""

from .error_reporter import ErrorReporter
from .recovery_manager import RecoveryManager
from .validation_errors import ValidationError, InputValidationError, SolverError

__all__ = [
    'ErrorReporter',
    'RecoveryManager', 
    'ValidationError',
    'InputValidationError',
    'SolverError'
]

