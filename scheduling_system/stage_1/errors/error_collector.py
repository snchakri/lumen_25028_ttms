"""
Error collector for aggregating validation errors across all stages.

Collects errors from all 7 validation stages before terminating,
enabling comprehensive error reporting per Theorem 12.4.
"""

from typing import List, Dict
from collections import defaultdict
from ..models.validation_types import ErrorReport, ErrorSeverity, ErrorCategory


class ErrorCollector:
    """
    Collect and organize validation errors.
    
    Per Theorem 12.4 (Error Handling Correctness), provides:
    - Completeness: Detects all constraint violations
    - Precision: No false positives
    - Clarity: Structured messages with locations and fixes
    """
    
    def __init__(self):
        """Initialize error collector."""
        self.errors: List[ErrorReport] = []
        self._errors_by_category: Dict[ErrorCategory, List[ErrorReport]] = defaultdict(list)
        self._errors_by_severity: Dict[ErrorSeverity, List[ErrorReport]] = defaultdict(list)
        self._errors_by_file: Dict[str, List[ErrorReport]] = defaultdict(list)
    
    @property
    def errors_by_severity(self) -> Dict[ErrorSeverity, List[ErrorReport]]:
        """Get errors grouped by severity (read-only)."""
        return self._errors_by_severity
    
    def add_error(self, error: ErrorReport):
        """
        Add an error to the collection.
        
        Args:
            error: ErrorReport to add
        """
        self.errors.append(error)
        self._errors_by_category[error.category].append(error)
        self._errors_by_severity[error.severity].append(error)
        
        if error.file_path:
            self._errors_by_file[error.file_path].append(error)
    
    def add_errors(self, errors: List[ErrorReport]):
        """Add multiple errors."""
        for error in errors:
            self.add_error(error)
    
    def has_errors(self) -> bool:
        """Check if any errors collected."""
        return len(self.errors) > 0
    
    def has_critical_errors(self) -> bool:
        """Check if any critical errors collected."""
        return len(self._errors_by_severity[ErrorSeverity.CRITICAL]) > 0
    
    def error_count(self) -> int:
        """Get total error count."""
        return len(self.errors)
    
    def critical_count(self) -> int:
        """Get critical error count."""
        return len(self._errors_by_severity[ErrorSeverity.CRITICAL])
    
    def error_count_by_severity(self, severity: ErrorSeverity) -> int:
        """Get error count for specific severity."""
        return len(self._errors_by_severity[severity])
    
    def error_count_by_category(self, category: ErrorCategory) -> int:
        """Get error count for specific category."""
        return len(self._errors_by_category[category])
    
    def get_all_errors(self) -> List[ErrorReport]:
        """Get all collected errors."""
        return self.errors.copy()
    
    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[ErrorReport]:
        """Get errors of specific severity."""
        return self._errors_by_severity[severity].copy()
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[ErrorReport]:
        """Get errors of specific category."""
        return self._errors_by_category[category].copy()
    
    def get_errors_by_file(self, file_path: str) -> List[ErrorReport]:
        """Get errors for specific file."""
        return self._errors_by_file[file_path].copy()
    
    def get_summary(self) -> Dict[str, int]:
        """
        Get error summary statistics.
        
        Returns:
            Dictionary with error counts by severity and category
        """
        return {
            "total_errors": len(self.errors),
            "critical": len(self._errors_by_severity[ErrorSeverity.CRITICAL]),
            "errors": len(self._errors_by_severity[ErrorSeverity.ERROR]),
            "warnings": len(self._errors_by_severity[ErrorSeverity.WARNING]),
            "info": len(self._errors_by_severity[ErrorSeverity.INFO]),
            "by_category": {
                cat.value: len(errs) for cat, errs in self._errors_by_category.items()
            },
            "files_with_errors": len(self._errors_by_file),
        }
    
    def clear(self):
        """Clear all collected errors."""
        self.errors.clear()
        self._errors_by_category.clear()
        self._errors_by_severity.clear()
        self._errors_by_file.clear()
    
    def __len__(self) -> int:
        """Return number of errors collected."""
        return len(self.errors)
    
    def __bool__(self) -> bool:
        """Return True if errors exist."""
        return len(self.errors) > 0





