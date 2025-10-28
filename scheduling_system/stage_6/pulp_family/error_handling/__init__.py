"""
Error Handling - Rigorous Error Management

Implements comprehensive error handling with JSON/TXT reports.

Author: LUMEN Team [TEAM-ID: 93912]
"""

from .error_reporter import ErrorReporter
from .recovery import SolverFailureRecovery

__all__ = [
    'ErrorReporter',
    'SolverFailureRecovery'
]



