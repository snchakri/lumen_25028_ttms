"""Error handling system per Definition 9.4."""

from .error_types import (
    ValidationSyntaxError,
    ValidationStructuralError,
    ValidationSemanticError,
    ValidationDomainError,
)
from .error_collector import ErrorCollector
from .error_reporter import ErrorReporter

__all__ = [
    "ValidationSyntaxError",
    "ValidationStructuralError",
    "ValidationSemanticError",
    "ValidationDomainError",
    "ErrorCollector",
    "ErrorReporter",
]





