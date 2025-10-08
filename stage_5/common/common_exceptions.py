"""
STAGE 5 - COMMON/EXCEPTIONS.PY
Enterprise-Grade Exception Handling & Error Management

This module defines comprehensive exception hierarchy for Stage 5 operations with
detailed error context, recovery guidance, and integration with logging and monitoring
systems. All exceptions follow enterprise standards for error handling and provide
actionable information for debugging and system recovery.

CRITICAL IMPLEMENTATION NOTES:
- NO MOCK EXCEPTIONS: All exceptions represent real error conditions with actionable context
- FAIL-FAST PHILOSOPHY: Immediate error raising prevents data corruption and invalid state
- STRUCTURED ERROR CONTEXT: Machine-readable error information for monitoring systems
- RECOVERY GUIDANCE: Actionable error messages with suggested resolution steps
- AUDIT TRAIL INTEGRATION: Exception correlation with logging and execution tracking

References:
- Stage5-FOUNDATIONAL-DESIGN-IMPLEMENTATION-PLAN.md: Error handling requirements
- Python exception hierarchy: Best practices for custom exception design
- Enterprise error handling patterns: Context preservation and error correlation
- Monitoring integration: Structured error reporting for alert systems

Cross-Module Dependencies:
- common.logging: Exception logging with structured context
- common.utils: File operation error handling and validation failures
- common.schema: Data validation error context and schema violations

IDE Integration Notes:
- Exception hierarchy enables intelligent error handling in try/except blocks
- Comprehensive docstrings provide context for error handling strategies
- Type hints support static analysis of exception handling code paths
"""

from typing import Dict, List, Optional, Union, Any, Type
from datetime import datetime
from pathlib import Path
import traceback
import sys

# =============================================================================
# MODULE METADATA AND CONFIGURATION
# =============================================================================

__version__ = "1.0.0"
__author__ = "LUMEN Team (Team ID: 93912)"
__description__ = "Stage 5 Exception Hierarchy & Error Management"

# Error severity levels for monitoring and alerting
ERROR_SEVERITY_LOW = "LOW"
ERROR_SEVERITY_MEDIUM = "MEDIUM" 
ERROR_SEVERITY_HIGH = "HIGH"
ERROR_SEVERITY_CRITICAL = "CRITICAL"

# Error categories for classification and handling
ERROR_CATEGORY_VALIDATION = "VALIDATION"
ERROR_CATEGORY_FILE_OPERATION = "FILE_OPERATION"
ERROR_CATEGORY_COMPUTATION = "COMPUTATION"
ERROR_CATEGORY_CONFIGURATION = "CONFIGURATION"
ERROR_CATEGORY_INTEGRATION = "INTEGRATION"
ERROR_CATEGORY_PERFORMANCE = "PERFORMANCE"

# =============================================================================
# BASE STAGE 5 EXCEPTION - Root Exception Class
# =============================================================================

class Stage5BaseException(Exception):
    """
    Base exception class for all Stage 5 operations.
    
    Provides comprehensive error context and metadata for all Stage 5 exceptions:
    - Error categorization and severity classification
    - Structured error context for debugging and monitoring
    - Recovery guidance and suggested resolution steps
    - Correlation with execution context and audit trails
    - Integration with logging and monitoring systems
    
    Error Context Structure:
    - error_code: Unique identifier for error type
    - severity: Error severity level (LOW, MEDIUM, HIGH, CRITICAL)  
    - category: Error category for classification and routing
    - context: Additional context dictionary with error-specific information
    - recovery_guidance: Human-readable guidance for error resolution
    - correlation_id: Execution correlation identifier for tracing
    
    Monitoring Integration:
    - Structured error data for automated monitoring and alerting
    - Error correlation across distributed systems and components
    - Performance impact assessment and resource utilization tracking
    """
    
    def __init__(self, 
                 message: str,
                 error_code: Optional[str] = None,
                 severity: str = ERROR_SEVERITY_MEDIUM,
                 category: str = ERROR_CATEGORY_VALIDATION,
                 context: Optional[Dict[str, Any]] = None,
                 recovery_guidance: Optional[str] = None,
                 correlation_id: Optional[str] = None,
                 inner_exception: Optional[Exception] = None):
        """
        Initialize Stage 5 base exception with comprehensive error context.
        
        Args:
            message: Primary error message describing the failure
            error_code: Unique error code for this exception type
            severity: Error severity level for monitoring and alerting
            category: Error category for classification and routing
            context: Additional context dictionary with error details
            recovery_guidance: Human-readable guidance for error resolution
            correlation_id: Execution correlation ID for distributed tracing
            inner_exception: Original exception that caused this error
            
        Example Usage:
            ```python
            raise Stage5BaseException(
                message="Parameter computation failed due to invalid input data",
                error_code="STAGE5_PARAM_COMPUTE_001",
                severity=ERROR_SEVERITY_HIGH,
                category=ERROR_CATEGORY_COMPUTATION,
                context={
                    "parameter": "p1_dimensionality",
                    "input_size": 0,
                    "expected_min_size": 1
                },
                recovery_guidance="Verify input data contains valid entity counts"
            )
            ```
        """
        super().__init__(message)
        
        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.recovery_guidance = recovery_guidance
        self.correlation_id = correlation_id
        self.inner_exception = inner_exception
        
        # Capture exception metadata
        self.timestamp = datetime.utcnow()
        self.traceback_info = traceback.format_exc() if sys.exc_info()[0] else None
        
        # Add class-specific context
        self.context.update({
            "exception_class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "timestamp": self.timestamp.isoformat(),
            "python_version": sys.version
        })
    
    def _generate_error_code(self) -> str:
        """
        Generate default error code based on exception class and timestamp.
        
        Returns:
            str: Generated error code for this exception
        """
        class_name = self.__class__.__name__.upper()
        timestamp_suffix = self.timestamp.strftime("%Y%m%d_%H%M%S")
        return f"{class_name}_{timestamp_suffix}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to structured dictionary for logging and monitoring.
        
        Returns:
            Dict[str, Any]: Structured exception data
            
        Example Output:
            ```python
            {
                "message": "Parameter computation failed",
                "error_code": "STAGE5_VALIDATION_ERROR_20251007_012000",
                "severity": "HIGH",
                "category": "COMPUTATION",
                "context": {...},
                "recovery_guidance": "Verify input data...",
                "timestamp": "2025-10-07T01:20:00.123456",
                "traceback": "Traceback (most recent call last)..."
            }
            ```
        """
        return {
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity,
            "category": self.category,
            "context": self.context,
            "recovery_guidance": self.recovery_guidance,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "traceback": self.traceback_info,
            "inner_exception": str(self.inner_exception) if self.inner_exception else None
        }
    
    def __str__(self) -> str:
        """
        String representation with error code and context.
        
        Returns:
            str: Formatted error message with metadata
        """
        parts = [f"[{self.error_code}] {self.message}"]
        
        if self.severity != ERROR_SEVERITY_MEDIUM:
            parts.append(f"Severity: {self.severity}")
        
        if self.context:
            context_str = ", ".join([f"{k}={v}" for k, v in self.context.items() 
                                   if k not in ["exception_class", "module", "timestamp", "python_version"]])
            if context_str:
                parts.append(f"Context: {context_str}")
        
        if self.recovery_guidance:
            parts.append(f"Resolution: {self.recovery_guidance}")
        
        return " | ".join(parts)

# =============================================================================
# VALIDATION EXCEPTIONS - Data and Schema Validation Errors
# =============================================================================

class Stage5ValidationError(Stage5BaseException):
    """
    Exception for data validation and schema compliance failures.
    
    Raised when:
    - Pydantic schema validation fails
    - Mathematical parameter bounds are violated
    - File format validation fails
    - Cross-reference validation fails between data structures
    - Input data integrity checks fail
    
    Validation Context:
    - validation_type: Type of validation that failed
    - expected_value: Expected value or format
    - actual_value: Actual value that caused validation failure
    - schema_path: JSON schema path where validation failed
    - field_name: Specific field that failed validation
    """
    
    def __init__(self, message: str,
                 validation_type: Optional[str] = None,
                 expected_value: Optional[Any] = None,
                 actual_value: Optional[Any] = None,
                 schema_path: Optional[str] = None,
                 field_name: Optional[str] = None,
                 **kwargs):
        """
        Initialize validation error with validation-specific context.
        
        Args:
            message: Primary validation error message
            validation_type: Type of validation (schema, bounds, format, etc.)
            expected_value: Expected value or format
            actual_value: Actual value that failed validation
            schema_path: JSON schema path where validation failed
            field_name: Specific field name that failed validation
            **kwargs: Additional arguments passed to base exception
        """
        # Set default values for validation errors
        kwargs.setdefault("severity", ERROR_SEVERITY_HIGH)
        kwargs.setdefault("category", ERROR_CATEGORY_VALIDATION)
        
        # Build validation-specific context
        validation_context = kwargs.get("context", {})
        validation_context.update({
            "validation_type": validation_type,
            "expected_value": expected_value,
            "actual_value": actual_value,
            "schema_path": schema_path,
            "field_name": field_name
        })
        kwargs["context"] = validation_context
        
        # Generate validation-specific error code
        if not kwargs.get("error_code"):
            kwargs["error_code"] = f"STAGE5_VALIDATION_{validation_type.upper() if validation_type else 'UNKNOWN'}"
        
        super().__init__(message, **kwargs)

class Stage5SchemaValidationError(Stage5ValidationError):
    """
    Specific exception for Pydantic schema validation failures.
    
    Raised when Pydantic model validation fails during:
    - JSON schema validation for input/output files
    - Parameter vector validation (16-parameter structure)
    - Solver capability profile validation
    - Configuration parameter validation
    """
    
    def __init__(self, message: str,
                 model_class: Optional[str] = None,
                 validation_errors: Optional[List[Dict[str, Any]]] = None,
                 **kwargs):
        """
        Initialize schema validation error with Pydantic-specific context.
        
        Args:
            message: Schema validation error message
            model_class: Name of Pydantic model class that failed validation
            validation_errors: List of Pydantic validation error details
            **kwargs: Additional arguments passed to validation error
        """
        # Add Pydantic-specific context
        schema_context = kwargs.get("context", {})
        schema_context.update({
            "model_class": model_class,
            "validation_errors": validation_errors,
            "validation_error_count": len(validation_errors) if validation_errors else 0
        })
        kwargs["context"] = schema_context
        kwargs["validation_type"] = "pydantic_schema"
        
        super().__init__(message, **kwargs)

class Stage5ParameterBoundsError(Stage5ValidationError):
    """
    Exception for mathematical parameter bounds validation failures.
    
    Raised when complexity parameters violate theoretical bounds:
    - P1-P16 parameter values outside expected ranges
    - Composite index exceeding empirical bounds
    - Statistical parameter validation failures
    - Mathematical constraint violations
    """
    
    def __init__(self, message: str,
                 parameter_name: Optional[str] = None,
                 parameter_value: Optional[float] = None,
                 min_bound: Optional[float] = None,
                 max_bound: Optional[float] = None,
                 theoretical_reference: Optional[str] = None,
                 **kwargs):
        """
        Initialize parameter bounds error with mathematical context.
        
        Args:
            message: Parameter bounds violation message
            parameter_name: Name of parameter that violated bounds
            parameter_value: Actual parameter value
            min_bound: Minimum allowed value
            max_bound: Maximum allowed value
            theoretical_reference: Reference to theoretical framework
            **kwargs: Additional arguments passed to validation error
        """
        bounds_context = kwargs.get("context", {})
        bounds_context.update({
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "min_bound": min_bound,
            "max_bound": max_bound,
            "theoretical_reference": theoretical_reference
        })
        kwargs["context"] = bounds_context
        kwargs["validation_type"] = "parameter_bounds"
        
        super().__init__(message, **kwargs)

# =============================================================================
# FILE OPERATION EXCEPTIONS - File I/O and Format Errors
# =============================================================================

class Stage5FileError(Stage5BaseException):
    """
    Exception for file operations and I/O failures.
    
    Raised when:
    - File does not exist or is not accessible
    - File format detection or validation fails
    - File reading or writing operations fail
    - File size exceeds prototype scale limits
    - File permission or security issues occur
    
    File Context:
    - file_path: Path to file that caused the error
    - operation: File operation being performed
    - file_size: File size if relevant to error
    - file_format: Expected or detected file format
    - permission_error: Whether error is permission-related
    """
    
    def __init__(self, message: str,
                 file_path: Optional[Union[str, Path]] = None,
                 operation: Optional[str] = None,
                 file_size: Optional[int] = None,
                 file_format: Optional[str] = None,
                 **kwargs):
        """
        Initialize file error with file operation context.
        
        Args:
            message: File operation error message
            file_path: Path to file that caused error
            operation: File operation being performed
            file_size: File size in bytes
            file_format: Expected or detected file format
            **kwargs: Additional arguments passed to base exception
        """
        kwargs.setdefault("severity", ERROR_SEVERITY_HIGH)
        kwargs.setdefault("category", ERROR_CATEGORY_FILE_OPERATION)
        
        file_context = kwargs.get("context", {})
        file_context.update({
            "file_path": str(file_path) if file_path else None,
            "operation": operation,
            "file_size": file_size,
            "file_format": file_format,
            "file_exists": Path(file_path).exists() if file_path else None
        })
        kwargs["context"] = file_context
        
        if not kwargs.get("error_code"):
            kwargs["error_code"] = f"STAGE5_FILE_{operation.upper() if operation else 'UNKNOWN'}"
        
        super().__init__(message, **kwargs)

class Stage5FileFormatError(Stage5FileError):
    """
    Specific exception for file format detection and validation failures.
    
    Raised when:
    - File extension doesn't match expected format
    - File contents don't match format specification
    - File format is not supported by Stage 5
    - File structure validation fails for known formats
    """
    
    def __init__(self, message: str,
                 expected_format: Optional[str] = None,
                 detected_format: Optional[str] = None,
                 supported_formats: Optional[List[str]] = None,
                 **kwargs):
        """
        Initialize file format error with format-specific context.
        
        Args:
            message: File format error message
            expected_format: Expected file format
            detected_format: Actually detected file format
            supported_formats: List of formats supported by operation
            **kwargs: Additional arguments passed to file error
        """
        format_context = kwargs.get("context", {})
        format_context.update({
            "expected_format": expected_format,
            "detected_format": detected_format,
            "supported_formats": supported_formats
        })
        kwargs["context"] = format_context
        kwargs["operation"] = "format_validation"
        
        super().__init__(message, **kwargs)

# =============================================================================
# COMPUTATION EXCEPTIONS - Mathematical and Algorithm Errors
# =============================================================================

class Stage5ComputationError(Stage5BaseException):
    """
    Exception for mathematical computation and algorithm failures.
    
    Raised when:
    - Mathematical parameter calculations fail
    - Numerical computation encounters invalid conditions
    - Algorithm convergence fails or times out
    - Linear programming optimization fails
    - Statistical computation produces invalid results
    
    Computation Context:
    - computation_type: Type of computation that failed
    - input_parameters: Input parameters to computation
    - intermediate_results: Intermediate computation results
    - algorithm_state: State of algorithm when failure occurred
    - numerical_issues: Specific numerical problems encountered
    """
    
    def __init__(self, message: str,
                 computation_type: Optional[str] = None,
                 input_parameters: Optional[Dict[str, Any]] = None,
                 algorithm_state: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize computation error with mathematical context.
        
        Args:
            message: Computation error message
            computation_type: Type of computation that failed
            input_parameters: Input parameters to failed computation
            algorithm_state: State of algorithm when failure occurred
            **kwargs: Additional arguments passed to base exception
        """
        kwargs.setdefault("severity", ERROR_SEVERITY_HIGH)
        kwargs.setdefault("category", ERROR_CATEGORY_COMPUTATION)
        
        computation_context = kwargs.get("context", {})
        computation_context.update({
            "computation_type": computation_type,
            "input_parameters": input_parameters,
            "algorithm_state": algorithm_state
        })
        kwargs["context"] = computation_context
        
        if not kwargs.get("error_code"):
            kwargs["error_code"] = f"STAGE5_COMPUTATION_{computation_type.upper() if computation_type else 'UNKNOWN'}"
        
        super().__init__(message, **kwargs)

class Stage5ConvergenceError(Stage5ComputationError):
    """
    Specific exception for algorithm convergence failures.
    
    Raised when:
    - LP weight learning fails to converge
    - Iterative algorithms exceed maximum iterations
    - Numerical optimization produces unstable results
    - Statistical sampling fails to reach significance
    """
    
    def __init__(self, message: str,
                 algorithm_name: Optional[str] = None,
                 max_iterations: Optional[int] = None,
                 current_iteration: Optional[int] = None,
                 convergence_criteria: Optional[Dict[str, float]] = None,
                 **kwargs):
        """
        Initialize convergence error with algorithm-specific context.
        
        Args:
            message: Convergence error message
            algorithm_name: Name of algorithm that failed to converge
            max_iterations: Maximum allowed iterations
            current_iteration: Iteration where convergence failed
            convergence_criteria: Convergence criteria and tolerances
            **kwargs: Additional arguments passed to computation error
        """
        convergence_context = kwargs.get("context", {})
        convergence_context.update({
            "algorithm_name": algorithm_name,
            "max_iterations": max_iterations,
            "current_iteration": current_iteration,
            "convergence_criteria": convergence_criteria
        })
        kwargs["context"] = convergence_context
        kwargs["computation_type"] = "algorithm_convergence"
        
        super().__init__(message, **kwargs)

# =============================================================================
# CONFIGURATION EXCEPTIONS - Setup and Parameter Errors
# =============================================================================

class Stage5ConfigurationError(Stage5BaseException):
    """
    Exception for configuration and setup failures.
    
    Raised when:
    - Configuration file parsing fails
    - Required configuration parameters are missing
    - Configuration parameter validation fails
    - Environment setup or dependency issues occur
    - Invalid configuration overrides are provided
    
    Configuration Context:
    - config_key: Specific configuration key that caused error
    - config_value: Invalid configuration value
    - config_source: Source of configuration (file, override, default)
    - expected_type: Expected configuration value type
    - validation_rule: Validation rule that was violated
    """
    
    def __init__(self, message: str,
                 config_key: Optional[str] = None,
                 config_value: Optional[Any] = None,
                 config_source: Optional[str] = None,
                 expected_type: Optional[str] = None,
                 **kwargs):
        """
        Initialize configuration error with setup context.
        
        Args:
            message: Configuration error message
            config_key: Configuration key that caused error
            config_value: Invalid configuration value
            config_source: Source of configuration
            expected_type: Expected configuration value type
            **kwargs: Additional arguments passed to base exception
        """
        kwargs.setdefault("severity", ERROR_SEVERITY_MEDIUM)
        kwargs.setdefault("category", ERROR_CATEGORY_CONFIGURATION)
        
        config_context = kwargs.get("context", {})
        config_context.update({
            "config_key": config_key,
            "config_value": config_value,
            "config_source": config_source,
            "expected_type": expected_type
        })
        kwargs["context"] = config_context
        
        if not kwargs.get("error_code"):
            kwargs["error_code"] = f"STAGE5_CONFIG_{config_key.upper() if config_key else 'UNKNOWN'}"
        
        super().__init__(message, **kwargs)

# =============================================================================
# INTEGRATION EXCEPTIONS - Cross-Stage and External System Errors
# =============================================================================

class Stage5IntegrationError(Stage5BaseException):
    """
    Exception for integration and cross-system failures.
    
    Raised when:
    - Stage 3 output format is incompatible
    - Solver arsenal configuration is invalid
    - Cross-stage data contract violations occur
    - External dependency failures affect Stage 5
    - API endpoint communication failures occur
    
    Integration Context:
    - integration_point: Specific integration that failed
    - external_system: External system or stage involved
    - data_contract: Data contract that was violated
    - compatibility_issue: Specific compatibility problem
    """
    
    def __init__(self, message: str,
                 integration_point: Optional[str] = None,
                 external_system: Optional[str] = None,
                 data_contract: Optional[str] = None,
                 **kwargs):
        """
        Initialize integration error with cross-system context.
        
        Args:
            message: Integration error message
            integration_point: Specific integration point that failed
            external_system: External system or stage involved
            data_contract: Data contract that was violated
            **kwargs: Additional arguments passed to base exception
        """
        kwargs.setdefault("severity", ERROR_SEVERITY_HIGH)
        kwargs.setdefault("category", ERROR_CATEGORY_INTEGRATION)
        
        integration_context = kwargs.get("context", {})
        integration_context.update({
            "integration_point": integration_point,
            "external_system": external_system,
            "data_contract": data_contract
        })
        kwargs["context"] = integration_context
        
        if not kwargs.get("error_code"):
            kwargs["error_code"] = f"STAGE5_INTEGRATION_{integration_point.upper() if integration_point else 'UNKNOWN'}"
        
        super().__init__(message, **kwargs)

# =============================================================================
# PERFORMANCE EXCEPTIONS - Resource and Timing Errors
# =============================================================================

class Stage5PerformanceError(Stage5BaseException):
    """
    Exception for performance and resource constraint violations.
    
    Raised when:
    - Computation time exceeds prototype limits
    - Memory usage exceeds available resources
    - File size exceeds processing limits
    - System resource exhaustion occurs
    - Performance SLA violations occur
    
    Performance Context:
    - resource_type: Type of resource that was exceeded
    - limit_value: Resource limit that was violated
    - actual_value: Actual resource usage
    - measurement_unit: Unit of measurement for resource
    - performance_impact: Impact on system performance
    """
    
    def __init__(self, message: str,
                 resource_type: Optional[str] = None,
                 limit_value: Optional[Union[int, float]] = None,
                 actual_value: Optional[Union[int, float]] = None,
                 measurement_unit: Optional[str] = None,
                 **kwargs):
        """
        Initialize performance error with resource constraint context.
        
        Args:
            message: Performance error message
            resource_type: Type of resource that was exceeded
            limit_value: Resource limit that was violated
            actual_value: Actual resource usage
            measurement_unit: Unit of measurement
            **kwargs: Additional arguments passed to base exception
        """
        kwargs.setdefault("severity", ERROR_SEVERITY_HIGH)
        kwargs.setdefault("category", ERROR_CATEGORY_PERFORMANCE)
        
        performance_context = kwargs.get("context", {})
        performance_context.update({
            "resource_type": resource_type,
            "limit_value": limit_value,
            "actual_value": actual_value,
            "measurement_unit": measurement_unit,
            "resource_utilization": (actual_value / limit_value * 100) if (limit_value and actual_value) else None
        })
        kwargs["context"] = performance_context
        
        if not kwargs.get("error_code"):
            kwargs["error_code"] = f"STAGE5_PERFORMANCE_{resource_type.upper() if resource_type else 'UNKNOWN'}"
        
        super().__init__(message, **kwargs)

# =============================================================================
# EXCEPTION UTILITIES - Error Handling Helpers
# =============================================================================

def handle_exception_with_context(exception: Exception,
                                 logger,
                                 operation: str,
                                 context: Optional[Dict[str, Any]] = None,
                                 reraise: bool = True) -> Dict[str, Any]:
    """
    Handle exception with comprehensive logging and context preservation.
    
    Provides standardized exception handling with:
    - Structured exception logging with context
    - Exception type classification and routing
    - Recovery guidance based on exception type
    - Correlation ID preservation for distributed tracing
    
    Args:
        exception: Exception to handle
        logger: Logger instance for exception reporting
        operation: Operation that was being performed
        context: Additional context for exception handling
        reraise: Whether to reraise exception after handling
        
    Returns:
        Dict[str, Any]: Structured exception data
        
    Raises:
        Exception: Original exception if reraise=True
        
    Example Usage:
        ```python
        try:
            result = risky_operation()
        except Exception as e:
            error_data = handle_exception_with_context(
                e, logger, "parameter_computation", 
                {"parameter": "p1_dimensionality"}
            )
        ```
    """
    # Convert to Stage5 exception if not already
    if not isinstance(exception, Stage5BaseException):
        stage5_exception = Stage5BaseException(
            message=str(exception),
            category=ERROR_CATEGORY_VALIDATION,
            context=context or {},
            inner_exception=exception
        )
    else:
        stage5_exception = exception
        if context:
            stage5_exception.context.update(context)
    
    # Add operation context
    stage5_exception.context["failed_operation"] = operation
    
    # Log exception with structured data
    logger.error(
        f"Exception in operation '{operation}': {stage5_exception.message}",
        extra={
            "exception_data": stage5_exception.to_dict(),
            "operation": operation,
            "error_code": stage5_exception.error_code,
            "severity": stage5_exception.severity
        },
        exc_info=True
    )
    
    # Reraise if requested
    if reraise:
        raise stage5_exception
    
    return stage5_exception.to_dict()

def create_recovery_guidance(exception_type: Type[Exception],
                           context: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate recovery guidance based on exception type and context.
    
    Args:
        exception_type: Type of exception for guidance generation
        context: Exception context for specific guidance
        
    Returns:
        str: Human-readable recovery guidance
    """
    guidance_map = {
        Stage5ValidationError: "Verify input data format and values against schema requirements",
        Stage5FileError: "Check file exists, is readable, and has correct format",
        Stage5ComputationError: "Review input parameters and mathematical constraints",
        Stage5ConfigurationError: "Validate configuration parameters and environment setup",
        Stage5IntegrationError: "Check data contracts and external system compatibility",
        Stage5PerformanceError: "Review resource limits and optimize data processing"
    }
    
    base_guidance = guidance_map.get(exception_type, "Review error context and system logs")
    
    # Add context-specific guidance if available
    if context and "parameter_name" in context:
        base_guidance += f" for parameter '{context['parameter_name']}'"
    
    if context and "file_path" in context:
        base_guidance += f" for file '{context['file_path']}'"
    
    return base_guidance

# =============================================================================
# MODULE EXPORTS - Public Exception API
# =============================================================================

__all__ = [
    # Base exception classes
    "Stage5BaseException",
    
    # Validation exceptions
    "Stage5ValidationError",
    "Stage5SchemaValidationError", 
    "Stage5ParameterBoundsError",
    
    # File operation exceptions
    "Stage5FileError",
    "Stage5FileFormatError",
    
    # Computation exceptions
    "Stage5ComputationError",
    "Stage5ConvergenceError",
    
    # Configuration exceptions
    "Stage5ConfigurationError",
    
    # Integration exceptions
    "Stage5IntegrationError",
    
    # Performance exceptions
    "Stage5PerformanceError",
    
    # Utility functions
    "handle_exception_with_context",
    "create_recovery_guidance",
    
    # Constants
    "ERROR_SEVERITY_LOW",
    "ERROR_SEVERITY_MEDIUM", 
    "ERROR_SEVERITY_HIGH",
    "ERROR_SEVERITY_CRITICAL",
    "ERROR_CATEGORY_VALIDATION",
    "ERROR_CATEGORY_FILE_OPERATION",
    "ERROR_CATEGORY_COMPUTATION",
    "ERROR_CATEGORY_CONFIGURATION",
    "ERROR_CATEGORY_INTEGRATION",
    "ERROR_CATEGORY_PERFORMANCE"
]

print("✅ STAGE 5 COMMON/EXCEPTIONS.PY - Enterprise exception hierarchy complete")
print("   - Comprehensive exception hierarchy with structured error context")
print("   - Enterprise-grade error handling with recovery guidance")
print("   - Integration with logging and monitoring systems")
print("   - Fail-fast philosophy with actionable error messages")
print("   - Audit trail correlation and distributed tracing support")