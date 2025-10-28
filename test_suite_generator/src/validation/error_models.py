"""
Error Models for Validation Framework

Defines error classification, severity levels, and validation result structures
as specified in DESIGN_PART_4_VALIDATION_FRAMEWORK.md Section 10.

Compliance:
    - Error categories: Configuration, Generation, Validation, System
    - Severity levels: CRITICAL, ERROR, WARNING, INFO
    - Structured error reporting with full context
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class ErrorSeverity(str, Enum):
    """
    Error severity levels as per design specification.
    
    CRITICAL: Stops generation immediately, blocks all output
    ERROR: Blocks output but continues validation
    WARNING: Logged but doesn't block output
    INFO: Informational, no action required
    """
    
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class ErrorCategory(str, Enum):
    """
    Error classification categories as per design specification.
    
    CONFIGURATION: Invalid configuration values or conflicting settings
    GENERATION: Failed to generate valid entity after retries
    VALIDATION: Generated data violates constraints
    SYSTEM: File I/O, memory, or library errors
    """
    
    CONFIGURATION = "CONFIGURATION"
    GENERATION = "GENERATION"
    VALIDATION = "VALIDATION"
    SYSTEM = "SYSTEM"


@dataclass
class ValidationError:
    """
    Structured validation error with full context.
    
    Attributes:
        error_id: Unique identifier for this error instance
        timestamp: When error was detected
        category: Error category (config, generation, validation, system)
        severity: Error severity level
        layer: Validation layer that detected error (L1-L7)
        message: Human-readable error message
        entity_type: Type of entity with error (e.g., 'students', 'courses')
        entity_id: UUID of specific entity with error (if applicable)
        field_name: Specific field with error (if applicable)
        expected_value: Expected value or constraint
        actual_value: Actual value that violated constraint
        constraint_name: Name of violated constraint
        suggestion: Suggested fix for the error
        related_entities: Related entity IDs for context
        stack_trace: Stack trace for system errors
        metadata: Additional context and diagnostic information
    """
    
    error_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)
    category: ErrorCategory = ErrorCategory.VALIDATION
    severity: ErrorSeverity = ErrorSeverity.ERROR
    layer: str = ""
    message: str = ""
    entity_type: Optional[str] = None
    entity_id: Optional[UUID] = None
    field_name: Optional[str] = None
    expected_value: Any = None
    actual_value: Any = None
    constraint_name: Optional[str] = None
    suggestion: Optional[str] = None
    related_entities: List[UUID] = field(default_factory=list)
    stack_trace: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for JSON serialization."""
        return {
            "error_id": str(self.error_id),
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "severity": self.severity.value,
            "layer": self.layer,
            "message": self.message,
            "entity_type": self.entity_type,
            "entity_id": str(self.entity_id) if self.entity_id else None,
            "field_name": self.field_name,
            "expected_value": str(self.expected_value) if self.expected_value is not None else None,
            "actual_value": str(self.actual_value) if self.actual_value is not None else None,
            "constraint_name": self.constraint_name,
            "suggestion": self.suggestion,
            "related_entities": [str(e) for e in self.related_entities],
            "stack_trace": self.stack_trace,
            "metadata": self.metadata,
        }
    
    def get_console_message(self) -> str:
        """
        Format error for console display with Rich formatting.
        
        Returns:
            Formatted error message with color codes
        """
        severity_colors = {
            ErrorSeverity.CRITICAL: "bold red",
            ErrorSeverity.ERROR: "red",
            ErrorSeverity.WARNING: "yellow",
            ErrorSeverity.INFO: "blue",
        }
        
        color = severity_colors.get(self.severity, "white")
        msg = f"[{color}]{self.severity.value}[/{color}] [{self.layer}] {self.message}"
        
        if self.entity_type and self.entity_id:
            msg += f"\n  Entity: {self.entity_type}/{self.entity_id}"
        
        if self.field_name:
            msg += f"\n  Field: {self.field_name}"
        
        if self.actual_value is not None:
            msg += f"\n  Actual: {self.actual_value}"
        
        if self.expected_value is not None:
            msg += f"\n  Expected: {self.expected_value}"
        
        if self.suggestion:
            msg += f"\n  [cyan]Suggestion:[/cyan] {self.suggestion}"
        
        return msg


@dataclass
class ValidationResult:
    """
    Result of validation execution for a layer or entire pipeline.
    
    Attributes:
        layer_name: Name of validation layer (e.g., "L1_Structural")
        passed: Whether validation passed overall
        errors: List of all errors detected
        warnings: List of all warnings detected
        info_messages: List of informational messages
        entities_validated: Total number of entities validated
        violations_detected: Total violations detected
        execution_time_seconds: Time taken for validation
        metadata: Additional validation statistics and context
    """
    
    layer_name: str
    passed: bool = True
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    info_messages: List[ValidationError] = field(default_factory=list)
    entities_validated: int = 0
    violations_detected: int = 0
    execution_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: ValidationError) -> None:
        """Add error to appropriate list based on severity."""
        self.violations_detected += 1
        
        if error.severity == ErrorSeverity.CRITICAL or error.severity == ErrorSeverity.ERROR:
            self.errors.append(error)
            self.passed = False
        elif error.severity == ErrorSeverity.WARNING:
            self.warnings.append(error)
        else:
            self.info_messages.append(error)
    
    def get_critical_count(self) -> int:
        """Count critical errors."""
        return sum(1 for e in self.errors if e.severity == ErrorSeverity.CRITICAL)
    
    def get_error_count(self) -> int:
        """Count non-critical errors."""
        return sum(1 for e in self.errors if e.severity == ErrorSeverity.ERROR)
    
    def get_warning_count(self) -> int:
        """Count warnings."""
        return len(self.warnings)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "layer_name": self.layer_name,
            "passed": self.passed,
            "entities_validated": self.entities_validated,
            "violations_detected": self.violations_detected,
            "critical_errors": self.get_critical_count(),
            "errors": self.get_error_count(),
            "warnings": self.get_warning_count(),
            "info_messages": len(self.info_messages),
            "execution_time_seconds": self.execution_time_seconds,
            "errors_detail": [e.to_dict() for e in self.errors[:100]],  # First 100
            "warnings_detail": [w.to_dict() for w in self.warnings[:100]],
            "metadata": self.metadata,
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary of validation result."""
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        return (
            f"{self.layer_name}: {status}\n"
            f"  Entities validated: {self.entities_validated}\n"
            f"  Critical: {self.get_critical_count()}, "
            f"Errors: {self.get_error_count()}, "
            f"Warnings: {self.get_warning_count()}\n"
            f"  Time: {self.execution_time_seconds:.2f}s"
        )


@dataclass
class ValidationReport:
    """
    Comprehensive validation report for entire pipeline execution.
    
    Attributes:
        report_id: Unique identifier for this validation run
        timestamp: When validation started
        configuration: Configuration used for validation
        layer_results: Results from each validation layer
        overall_passed: Whether entire validation passed
        total_entities: Total entities validated across all layers
        total_violations: Total violations detected
        total_time_seconds: Total execution time
        mode: Validation mode (strict, lenient, adversarial)
        recommendations: List of recommendations for fixing errors
    """
    
    report_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)
    configuration: Dict[str, Any] = field(default_factory=dict)
    layer_results: List[ValidationResult] = field(default_factory=list)
    overall_passed: bool = True
    total_entities: int = 0
    total_violations: int = 0
    total_time_seconds: float = 0.0
    mode: str = "strict"
    recommendations: List[str] = field(default_factory=list)
    
    def add_layer_result(self, result: ValidationResult) -> None:
        """Add layer result and update overall statistics."""
        self.layer_results.append(result)
        self.total_entities += result.entities_validated
        self.total_violations += result.violations_detected
        self.total_time_seconds += result.execution_time_seconds
        
        if not result.passed and self.mode == "strict":
            self.overall_passed = False
    
    def get_all_errors(self) -> List[ValidationError]:
        """Get all errors from all layers."""
        all_errors = []
        for result in self.layer_results:
            all_errors.extend(result.errors)
        return all_errors
    
    def get_all_warnings(self) -> List[ValidationError]:
        """Get all warnings from all layers."""
        all_warnings = []
        for result in self.layer_results:
            all_warnings.extend(result.warnings)
        return all_warnings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "report_id": str(self.report_id),
            "timestamp": self.timestamp.isoformat(),
            "mode": self.mode,
            "overall_passed": self.overall_passed,
            "total_entities": self.total_entities,
            "total_violations": self.total_violations,
            "total_time_seconds": self.total_time_seconds,
            "configuration": self.configuration,
            "layer_results": [r.to_dict() for r in self.layer_results],
            "recommendations": self.recommendations,
            "summary": self.get_summary(),
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary of validation report."""
        status = "✓ PASSED" if self.overall_passed else "✗ FAILED"
        total_critical = sum(r.get_critical_count() for r in self.layer_results)
        total_errors = sum(r.get_error_count() for r in self.layer_results)
        total_warnings = sum(r.get_warning_count() for r in self.layer_results)
        
        summary = f"""
{'='*70}
VALIDATION REPORT: {status}
{'='*70}
Report ID: {self.report_id}
Timestamp: {self.timestamp.isoformat()}
Mode: {self.mode.upper()}

OVERALL STATISTICS:
  Total Entities Validated: {self.total_entities}
  Total Violations: {self.total_violations}
  Critical Errors: {total_critical}
  Errors: {total_errors}
  Warnings: {total_warnings}
  Total Time: {self.total_time_seconds:.2f}s

LAYER RESULTS:
"""
        for result in self.layer_results:
            summary += f"\n{result.get_summary()}\n"
        
        if self.recommendations:
            summary += "\nRECOMMENDATIONS:\n"
            for i, rec in enumerate(self.recommendations, 1):
                summary += f"  {i}. {rec}\n"
        
        summary += "=" * 70
        return summary
