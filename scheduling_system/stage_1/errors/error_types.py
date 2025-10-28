"""
Error type definitions per Definition 9.4: Hierarchical Error Classification.

Implements custom exception classes for validation errors:
- SyntaxError: Stage 1 (CSV format, encoding)
- StructuralError: Stage 2 (Schema conformance, types)
- SemanticError: Stage 4 (Domain constraints, axioms)
- DomainError: Stage 7 (Educational policy compliance)
"""

from typing import Optional, Any


class ValidationError(Exception):
    """Base class for all validation errors."""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        column_number: Optional[int] = None,
        field_name: Optional[str] = None,
        expected_value: Optional[Any] = None,
        actual_value: Optional[Any] = None,
        fix_suggestion: Optional[str] = None
    ):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            file_path: File where error occurred
            line_number: Line number in file
            column_number: Column number in line
            field_name: Field/column name
            expected_value: Expected value
            actual_value: Actual value found
            fix_suggestion: Suggested fix
        """
        super().__init__(message)
        self.message = message
        self.file_path = file_path
        self.line_number = line_number
        self.column_number = column_number
        self.field_name = field_name
        self.expected_value = expected_value
        self.actual_value = actual_value
        self.fix_suggestion = fix_suggestion
    
    def __str__(self) -> str:
        """Generate human-readable error message."""
        msg = f"{self.__class__.__name__}: {self.message}"
        
        if self.file_path:
            msg += f"\n  File: {self.file_path}"
            if self.line_number:
                msg += f", line {self.line_number}"
            if self.column_number:
                msg += f", column {self.column_number}"
        
        if self.field_name:
            msg += f"\n  Field: {self.field_name}"
        
        if self.expected_value is not None:
            msg += f"\n  Expected: {self.expected_value}"
        
        if self.actual_value is not None:
            msg += f"\n  Actual: {self.actual_value}"
        
        if self.fix_suggestion:
            msg += f"\n  Suggestion: {self.fix_suggestion}"
        
        return msg


class ValidationSyntaxError(ValidationError):
    """
    Syntactic validation errors per Stage 1.
    
    Includes:
    - Malformed CSV format
    - Encoding issues
    - Missing delimiters
    - Unmatched quotes
    - Invalid line endings
    """
    pass


class ValidationStructuralError(ValidationError):
    """
    Structural validation errors per Stage 2.
    
    Includes:
    - Schema conformance violations
    - Type mismatches
    - Missing required fields
    - Invalid ENUM values
    - Format violations (email, UUID, etc.)
    """
    pass


class ValidationReferentialError(ValidationError):
    """
    Referential integrity errors per Stage 3.
    
    Includes:
    - Foreign key violations (orphan records)
    - Circular dependencies
    - Self-references
    - Missing primary keys
    """
    pass


class ValidationSemanticError(ValidationError):
    """
    Semantic validation errors per Stage 4.
    
    Includes:
    - Competency axiom violations (Axiom 4.3)
    - Capacity constraint violations
    - Domain constraint violations
    - Hierarchical consistency violations
    """
    pass


class ValidationTemporalError(ValidationError):
    """
    Temporal consistency errors per Stage 5.
    
    Includes:
    - Overlapping timeslots
    - Invalid time ranges (end_time <= start_time)
    - Duration mismatches
    - Circular temporal dependencies
    """
    pass


class ValidationCrossTableError(ValidationError):
    """
    Cross-table consistency errors per Stage 6.
    
    Includes:
    - Capacity mismatches across tables
    - Resource conflicts
    - Enrollment inconsistencies
    - Cardinality violations
    """
    pass


class ValidationDomainError(ValidationError):
    """
    Educational domain compliance errors per Stage 7.
    
    Includes:
    - Policy violations
    - Constraint expression errors
    - Parameter validation errors
    - Academic rule violations
    """
    pass


class ValidationCriticalError(ValidationError):
    """
    Critical errors that prevent further processing.
    
    Includes:
    - Missing mandatory files
    - Unreadable files
    - System errors
    - Corrupted data
    """
    pass


def create_syntax_error(
    message: str,
    file_path: Optional[str] = None,
    line_number: Optional[int] = None,
    fix_suggestion: Optional[str] = None,
    **kwargs
) -> ValidationSyntaxError:
    """Factory function for syntax errors."""
    return ValidationSyntaxError(
        message=message,
        file_path=file_path,
        line_number=line_number,
        fix_suggestion=fix_suggestion,
        **kwargs
    )


def create_structural_error(
    message: str,
    field_name: Optional[str] = None,
    expected_value: Optional[Any] = None,
    actual_value: Optional[Any] = None,
    fix_suggestion: Optional[str] = None,
    **kwargs
) -> ValidationStructuralError:
    """Factory function for structural errors."""
    return ValidationStructuralError(
        message=message,
        field_name=field_name,
        expected_value=expected_value,
        actual_value=actual_value,
        fix_suggestion=fix_suggestion,
        **kwargs
    )


def create_referential_error(
    message: str,
    field_name: Optional[str] = None,
    actual_value: Optional[Any] = None,
    fix_suggestion: Optional[str] = None,
    **kwargs
) -> ValidationReferentialError:
    """Factory function for referential errors."""
    return ValidationReferentialError(
        message=message,
        field_name=field_name,
        actual_value=actual_value,
        fix_suggestion=fix_suggestion,
        **kwargs
    )


def create_semantic_error(
    message: str,
    field_name: Optional[str] = None,
    expected_value: Optional[Any] = None,
    actual_value: Optional[Any] = None,
    fix_suggestion: Optional[str] = None,
    **kwargs
) -> ValidationSemanticError:
    """Factory function for semantic errors."""
    return ValidationSemanticError(
        message=message,
        field_name=field_name,
        expected_value=expected_value,
        actual_value=actual_value,
        fix_suggestion=fix_suggestion,
        **kwargs
    )


def create_temporal_error(
    message: str,
    field_name: Optional[str] = None,
    actual_value: Optional[Any] = None,
    fix_suggestion: Optional[str] = None,
    **kwargs
) -> ValidationTemporalError:
    """Factory function for temporal errors."""
    return ValidationTemporalError(
        message=message,
        field_name=field_name,
        actual_value=actual_value,
        fix_suggestion=fix_suggestion,
        **kwargs
    )


def create_cross_table_error(
    message: str,
    fix_suggestion: Optional[str] = None,
    **kwargs
) -> ValidationCrossTableError:
    """Factory function for cross-table errors."""
    return ValidationCrossTableError(
        message=message,
        fix_suggestion=fix_suggestion,
        **kwargs
    )


def create_domain_error(
    message: str,
    field_name: Optional[str] = None,
    actual_value: Optional[Any] = None,
    fix_suggestion: Optional[str] = None,
    **kwargs
) -> ValidationDomainError:
    """Factory function for domain errors."""
    return ValidationDomainError(
        message=message,
        field_name=field_name,
        actual_value=actual_value,
        fix_suggestion=fix_suggestion,
        **kwargs
    )


def create_critical_error(
    message: str,
    fix_suggestion: Optional[str] = None,
    **kwargs
) -> ValidationCriticalError:
    """Factory function for critical errors."""
    return ValidationCriticalError(
        message=message,
        fix_suggestion=fix_suggestion,
        **kwargs
    )





