"""
Validation types per Definition 9.4: Error Classification.

Implements hierarchical error classification and validation result structures
following theoretical foundations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime


class ValidationStatus(Enum):
    """Overall validation status."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"


class ErrorSeverity(Enum):
    """Error severity levels per Definition 9.4."""
    CRITICAL = "CRITICAL"  # Prevents further processing
    ERROR = "ERROR"  # Violation of hard constraints
    WARNING = "WARNING"  # Soft constraint or quality issue
    INFO = "INFO"  # Informational message


class ErrorCategory(Enum):
    """Error category classification per Definition 9.4."""
    SYNTAX = "SYNTAX"  # Stage 1: CSV format, encoding
    STRUCTURAL = "STRUCTURAL"  # Stage 2: Schema conformance, types
    REFERENTIAL = "REFERENTIAL"  # Stage 3: Foreign keys, cycles
    SEMANTIC = "SEMANTIC"  # Stage 4: Domain constraints, axioms
    TEMPORAL = "TEMPORAL"  # Stage 5: Time consistency
    CROSS_TABLE = "CROSS_TABLE"  # Stage 6: Multi-table consistency
    DOMAIN = "DOMAIN"  # Stage 7: Educational policy compliance


@dataclass
class ErrorReport:
    """
    Structured error report with location tracking and fix suggestions.
    
    Per Definition 9.4, provides completeness, precision, and clarity in
    error detection and reporting.
    """
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    field_name: Optional[str] = None
    record_id: Optional[str] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    constraint_violated: Optional[str] = None
    fix_suggestion: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "field_name": self.field_name,
            "record_id": self.record_id,
            "expected_value": str(self.expected_value) if self.expected_value is not None else None,
            "actual_value": str(self.actual_value) if self.actual_value is not None else None,
            "constraint_violated": self.constraint_violated,
            "fix_suggestion": self.fix_suggestion,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def to_human_readable(self) -> str:
        """Generate human-readable error message."""
        lines = []
        lines.append(f"[{self.severity.value}] {self.category.value}: {self.message}")
        
        if self.file_path:
            location = f"  Location: {self.file_path}"
            if self.line_number:
                location += f", line {self.line_number}"
            if self.column_number:
                location += f", column {self.column_number}"
            lines.append(location)
        
        if self.field_name:
            lines.append(f"  Field: {self.field_name}")
        
        if self.record_id:
            lines.append(f"  Record ID: {self.record_id}")
        
        if self.expected_value is not None:
            lines.append(f"  Expected: {self.expected_value}")
        
        if self.actual_value is not None:
            lines.append(f"  Actual: {self.actual_value}")
        
        if self.constraint_violated:
            lines.append(f"  Constraint: {self.constraint_violated}")
        
        if self.fix_suggestion:
            lines.append(f"  Fix: {self.fix_suggestion}")
        
        return "\n".join(lines)


@dataclass
class StageResult:
    """Result from a single validation stage."""
    stage_number: int
    stage_name: str
    status: ValidationStatus
    errors: List[ErrorReport] = field(default_factory=list)
    warnings: List[ErrorReport] = field(default_factory=list)
    info: List[ErrorReport] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    records_processed: int = 0
    records_validated: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_all_errors(self) -> List[ErrorReport]:
        """Get all error reports including warnings and info."""
        return self.errors + self.warnings + self.info
    
    def has_critical_errors(self) -> bool:
        """Check if stage has critical errors."""
        return any(e.severity == ErrorSeverity.CRITICAL for e in self.errors)
    
    def has_errors(self) -> bool:
        """Check if stage has any errors (excluding warnings and info)."""
        return len(self.errors) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "stage_number": self.stage_number,
            "stage_name": self.stage_name,
            "status": self.status.value,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "info": [i.to_dict() for i in self.info],
            "metrics": self.metrics,
            "execution_time_seconds": self.execution_time_seconds,
            "records_processed": self.records_processed,
            "records_validated": self.records_validated,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ValidationResult:
    """
    Complete validation result from Stage-1 pipeline.
    
    Implements Theorem 11.3 (Validation Soundness) and Theorem 11.4
    (Validation Completeness) verification.
    """
    overall_status: ValidationStatus
    stage_results: List[StageResult] = field(default_factory=list)
    files_validated: List[str] = field(default_factory=list)
    files_skipped: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    complexity_metrics: Dict[str, Any] = field(default_factory=dict)
    total_execution_time_seconds: float = 0.0
    total_records_processed: int = 0
    total_errors: int = 0
    total_warnings: int = 0
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    validation_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def should_abort(self) -> bool:
        """
        Determine if validation failures require aborting the pipeline.
        
        Returns True if critical errors exist that prevent further processing.
        """
        if self.overall_status == ValidationStatus.FAIL:
            # Check if any stage has critical errors
            for stage in self.stage_results:
                if stage.has_critical_errors():
                    return True
            # If no critical errors but status is FAIL, still abort
            return True
        return False
    
    def get_all_errors(self) -> List[ErrorReport]:
        """Get all error reports from all stages."""
        all_errors = []
        for stage in self.stage_results:
            all_errors.extend(stage.get_all_errors())
        return all_errors
    
    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[ErrorReport]:
        """Get all errors of a specific severity."""
        return [e for e in self.get_all_errors() if e.severity == severity]
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[ErrorReport]:
        """Get all errors of a specific category."""
        return [e for e in self.get_all_errors() if e.category == category]
    
    def get_stage_result(self, stage_number: int) -> Optional[StageResult]:
        """Get result for a specific stage."""
        for stage in self.stage_results:
            if stage.stage_number == stage_number:
                return stage
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_status": self.overall_status.value,
            "should_abort": self.should_abort(),
            "stage_results": [s.to_dict() for s in self.stage_results],
            "files_validated": self.files_validated,
            "files_skipped": self.files_skipped,
            "quality_metrics": self.quality_metrics,
            "complexity_metrics": self.complexity_metrics,
            "total_execution_time_seconds": self.total_execution_time_seconds,
            "total_records_processed": self.total_records_processed,
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
            "start_timestamp": self.start_timestamp.isoformat() if self.start_timestamp else None,
            "end_timestamp": self.end_timestamp.isoformat() if self.end_timestamp else None,
            "validation_metadata": self.validation_metadata,
        }
    
    def get_summary(self) -> str:
        """Get a concise summary of validation results."""
        lines = []
        lines.append(f"Validation Status: {self.overall_status.value}")
        lines.append(f"Files Validated: {len(self.files_validated)}")
        lines.append(f"Total Records Processed: {self.total_records_processed}")
        lines.append(f"Total Errors: {self.total_errors}")
        lines.append(f"Total Warnings: {self.total_warnings}")
        lines.append(f"Execution Time: {self.total_execution_time_seconds:.2f}s")
        
        if self.should_abort():
            lines.append("\n[CRITICAL] Validation failed - pipeline should be aborted")
        
        # Stage summary
        lines.append("\nStage Results:")
        for stage in self.stage_results:
            status_icon = "[PASS]" if stage.status == ValidationStatus.PASS else "[FAIL]"
            lines.append(
                f"  {status_icon} Stage {stage.stage_number}: {stage.stage_name} "
                f"({stage.status.value}) - {len(stage.errors)} errors, "
                f"{len(stage.warnings)} warnings"
            )
        
        return "\n".join(lines)


def create_error(
    category: ErrorCategory,
    severity: ErrorSeverity,
    message: str,
    **kwargs
) -> ErrorReport:
    """
    Factory function to create error reports with consistent structure.
    
    Args:
        category: Error category
        severity: Error severity
        message: Error message
        **kwargs: Additional error details (file_path, line_number, etc.)
    
    Returns:
        ErrorReport instance
    """
    return ErrorReport(
        category=category,
        severity=severity,
        message=message,
        **kwargs
    )



