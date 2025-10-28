"""
Unified Error Handling System
Implements fail-fast architecture with no recovery attempts
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = "CRITICAL"      # Must abort pipeline immediately
    HIGH = "HIGH"              # Stage failure, abort pipeline
    MEDIUM = "MEDIUM"          # Component failure, abort stage
    LOW = "LOW"                # Warning, continue with degraded functionality

class ErrorCategory(Enum):
    """Error categories for classification"""
    VALIDATION = "VALIDATION"
    MATHEMATICAL = "MATHEMATICAL"
    MEMORY = "MEMORY"
    INTEGRATION = "INTEGRATION"
    SOLVER = "SOLVER"
    DATA = "DATA"
    CONFIGURATION = "CONFIGURATION"

@dataclass
class ErrorContext:
    """Context information for errors"""
    stage: str
    component: str
    operation: str
    details: Dict[str, Any]
    timestamp: str

class SystemError(Exception):
    """Base system error with fail-fast behavior"""
    
    def __init__(
        self, 
        message: str, 
        severity: ErrorSeverity,
        category: ErrorCategory,
        context: ErrorContext,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.severity = severity
        self.category = category
        self.context = context
        self.original_error = original_error
        
        # Log the error immediately
        self._log_error()
        
        # Format error message
        error_msg = f"[{severity.value}:{category.value}] {message}"
        error_msg += f" in {context.stage}:{context.component}"
        
        super().__init__(error_msg)
    
    def _log_error(self):
        """Log error with structured logging"""
        logger.error(
            "System error occurred",
            severity=self.severity.value,
            category=self.category.value,
            stage=self.context.stage,
            component=self.context.component,
            operation=self.context.operation,
            message=self.message,
            details=self.context.details,
            original_error=str(self.original_error) if self.original_error else None
        )

class CriticalSystemError(SystemError):
    """Critical errors that must abort the pipeline immediately"""
    
    def __init__(self, message: str, context: ErrorContext, original_error: Optional[Exception] = None):
        super().__init__(
            message=message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.VALIDATION,
            context=context,
            original_error=original_error
        )

class ValidationError(CriticalSystemError):
    """Data validation failures - must abort pipeline"""
    
    def __init__(self, message: str, context: ErrorContext, validation_details: Dict[str, Any]):
        context.details.update(validation_details)
        super().__init__(message, context)

class MathematicalError(CriticalSystemError):
    """Mathematical framework violations - must abort pipeline"""
    
    def __init__(self, message: str, context: ErrorContext, mathematical_details: Dict[str, Any]):
        context.details.update(mathematical_details)
        super().__init__(message, context)

class MemoryError(CriticalSystemError):
    """Memory constraint violations - must abort pipeline"""
    
    def __init__(self, message: str, context: ErrorContext, memory_details: Dict[str, Any]):
        context.details.update(memory_details)
        super().__init__(message, context)

class IntegrationError(SystemError):
    """Integration failures - abort stage"""
    
    def __init__(self, message: str, context: ErrorContext, original_error: Optional[Exception] = None):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.INTEGRATION,
            context=context,
            original_error=original_error
        )

class SolverError(SystemError):
    """Solver failures - abort stage"""
    
    def __init__(self, message: str, context: ErrorContext, solver_details: Dict[str, Any]):
        context.details.update(solver_details)
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SOLVER,
            context=context
        )

class DataError(SystemError):
    """Data processing failures - abort stage"""
    
    def __init__(self, message: str, context: ErrorContext, data_details: Dict[str, Any]):
        context.details.update(data_details)
        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA,
            context=context
        )

def create_error_context(stage: str, component: str, operation: str, **details) -> ErrorContext:
    """Create error context with current timestamp"""
    from datetime import datetime
    return ErrorContext(
        stage=stage,
        component=component,
        operation=operation,
        details=details,
        timestamp=datetime.utcnow().isoformat()
    )

def fail_fast_on_error(error: Exception, stage: str, component: str, operation: str):
    """Fail-fast handler - immediately abort on any error"""
    context = create_error_context(stage, component, operation)
    
    if isinstance(error, SystemError):
        # Re-raise system errors as-is
        raise error
    else:
        # Convert generic errors to critical system errors
        raise CriticalSystemError(
            message=f"Unexpected error: {str(error)}",
            context=context,
            original_error=error
        )

def validate_fail_fast(condition: bool, message: str, stage: str, component: str, operation: str, **details):
    """Validate condition and fail-fast if false"""
    if not condition:
        context = create_error_context(stage, component, operation, **details)
        raise ValidationError(message, context, details)

def mathematical_fail_fast(condition: bool, message: str, stage: str, component: str, operation: str, **details):
    """Validate mathematical condition and fail-fast if false"""
    if not condition:
        context = create_error_context(stage, component, operation, **details)
        raise MathematicalError(message, context, details)

def memory_fail_fast(current_mb: float, limit_mb: float, stage: str, component: str, operation: str):
    """Validate memory constraint and fail-fast if exceeded"""
    if current_mb > limit_mb:
        context = create_error_context(
            stage, component, operation,
            current_mb=current_mb,
            limit_mb=limit_mb,
            violation_mb=current_mb - limit_mb
        )
        raise MemoryError(
            f"Memory limit exceeded: {current_mb:.2f}MB > {limit_mb}MB",
            context,
            {"current_mb": current_mb, "limit_mb": limit_mb}
        )
