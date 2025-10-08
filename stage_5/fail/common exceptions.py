# STAGE 5 - COMMON/EXCEPTIONS.PY  
# Enterprise-Grade Exception Hierarchy for Rigorous Error Handling

"""
STAGE 5 COMMON EXCEPTION DEFINITIONS
Enterprise-Grade Exception Hierarchy for Fail-Fast Error Handling

This module defines comprehensive exception classes for Stage 5's rigorous error handling
framework. Every exception provides detailed context for debugging and maintains the
fail-fast discipline required for enterprise-grade reliability.

Critical Implementation Notes:
- NO MOCK EXCEPTIONS: All exceptions represent real failure conditions  
- DETAILED CONTEXT: Every exception includes specific error details and remedy guidance
- FAIL-FAST PHILOSOPHY: Exceptions halt execution immediately to prevent error propagation
- CURSOR/PyCharm IDE INTEGRATION: Full docstrings for intelligent error handling
- AUDIT TRAIL SUPPORT: All exceptions log context for debugging and analysis

Design Principles:
1. GRANULAR CLASSIFICATION: Specific exception types for precise error identification
2. CONTEXTUAL INFORMATION: Rich error messages with actionable remediation guidance  
3. HIERARCHICAL STRUCTURE: Base classes enable consistent error handling patterns
4. DEBUGGING SUPPORT: Stack traces and context preservation for rapid issue resolution
5. INTEGRATION COMPATIBILITY: Exception types align with logging and monitoring systems

References:
- Stage5-FOUNDATIONAL-DESIGN-IMPLEMENTATION-PLAN.md: Error handling specifications
- Python Exception Hierarchy: Built on standard Python exception patterns
- Enterprise Error Handling: Industry best practices for production systems
"""

from typing import Optional, Dict, Any, List
import traceback
from datetime import datetime

# =============================================================================
# BASE EXCEPTION CLASSES
# Foundation for Stage 5 error handling hierarchy
# =============================================================================

class Stage5BaseException(Exception):
    """
    Base exception class for all Stage 5 errors.
    Provides common functionality for error context, logging, and debugging support.
    
    This base class establishes the foundation for Stage 5's fail-fast error handling
    philosophy. All Stage 5 exceptions inherit from this base to ensure consistent
    error reporting, context preservation, and integration with logging systems.
    
    Attributes:
        message (str): Human-readable error description
        error_code (str): Unique error identifier for categorization
        context (Dict[str, Any]): Additional error context for debugging
        timestamp (datetime): When the error occurred for audit trails
        stage_component (str): Which Stage 5 component generated the error
        remediation_hint (str): Suggested remedy for the error condition
    
    Usage Pattern:
        try:
            risky_operation()
        except Stage5BaseException as e:
            logger.error(f"Stage 5 Error [{e.error_code}]: {e.message}", 
                        extra={'context': e.context})
            raise  # Re-raise for fail-fast behavior
    """
    
    def __init__(
        self, 
        message: str,
        error_code: str,
        context: Optional[Dict[str, Any]] = None,
        stage_component: str = "unknown",
        remediation_hint: Optional[str] = None
    ):
        """
        Initialize Stage 5 base exception with comprehensive error context.
        
        Args:
            message: Human-readable error description for developers
            error_code: Unique error identifier for monitoring/alerting systems
            context: Additional context dictionary for debugging support
            stage_component: Component that generated error (5.1, 5.2, common)
            remediation_hint: Suggested fix for the error condition
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now()
        self.stage_component = stage_component
        self.remediation_hint = remediation_hint
        
        # Capture stack trace for debugging
        self.stack_trace = traceback.format_stack()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for structured logging and JSON serialization.
        
        Returns:
            Dict containing all exception attributes for logging systems
        """
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'stage_component': self.stage_component,
            'remediation_hint': self.remediation_hint,
            'stack_trace': self.stack_trace[-3:] if self.stack_trace else None  # Last 3 frames
        }

    def __str__(self) -> str:
        """Enhanced string representation with error code and context."""
        base_msg = f"[{self.error_code}] {self.message}"
        if self.remediation_hint:
            base_msg += f" | Remedy: {self.remediation_hint}"
        return base_msg

class Stage5ValidationError(Stage5BaseException):
    """
    Base class for all validation-related errors in Stage 5.
    Covers input validation, schema validation, and mathematical bounds checking.
    
    This exception category handles all forms of data validation failures including:
    - Pydantic schema validation errors
    - Mathematical parameter bounds violations  
    - File format and structure validation failures
    - Cross-field validation constraint violations
    
    Validation errors are ALWAYS fail-fast conditions that prevent further processing
    to maintain data integrity and prevent corruption propagation.
    """
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        invalid_value: Any = None,
        expected_constraints: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize validation error with field-specific context.
        
        Args:
            message: Validation error description
            field_name: Name of field that failed validation
            invalid_value: The invalid value that caused the error
            expected_constraints: Description of expected value constraints
            **kwargs: Additional base exception arguments
        """
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'STAGE5_VALIDATION_ERROR'),
            stage_component=kwargs.get('stage_component', 'validation'),
            **{k: v for k, v in kwargs.items() if k not in ['error_code', 'stage_component']}
        )
        
        # Add validation-specific context
        self.context.update({
            'field_name': field_name,
            'invalid_value': str(invalid_value) if invalid_value is not None else None,
            'expected_constraints': expected_constraints
        })

class Stage5ComputationError(Stage5BaseException):
    """
    Base class for all mathematical computation errors in Stage 5.
    Covers numerical errors, algorithm failures, and mathematical constraint violations.
    
    This exception category handles computational failures including:
    - Division by zero in mathematical formulas
    - Numerical overflow/underflow conditions
    - Matrix operation failures (singular matrices, etc.)
    - Algorithm convergence failures
    - Mathematical constraint violations
    
    Computation errors indicate fundamental issues with input data or algorithm
    implementation and require immediate attention for resolution.
    """
    
    def __init__(
        self,
        message: str,
        computation_type: Optional[str] = None,
        input_data_summary: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize computation error with mathematical context.
        
        Args:
            message: Computation error description
            computation_type: Type of computation that failed (P1-P16, LP, etc.)
            input_data_summary: Summary of input data that caused failure
            **kwargs: Additional base exception arguments
        """
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'STAGE5_COMPUTATION_ERROR'),
            stage_component=kwargs.get('stage_component', 'computation'),
            **{k: v for k, v in kwargs.items() if k not in ['error_code', 'stage_component']}
        )
        
        # Add computation-specific context
        self.context.update({
            'computation_type': computation_type,
            'input_data_summary': input_data_summary or {}
        })

# =============================================================================
# STAGE 5.1 SPECIFIC EXCEPTIONS
# Input-Complexity Analysis error conditions
# =============================================================================

class Stage51InputError(Stage5ValidationError):
    """
    Stage 3 input file validation errors for Stage 5.1 complexity analysis.
    
    Covers all input file-related failures including:
    - Missing L_raw.parquet, L_rel.graphml, or L_idx files
    - Corrupted or unreadable input files
    - Incorrect file formats or extensions
    - Empty or malformed data structures
    - Schema mismatches between expected and actual data
    
    Input errors prevent Stage 5.1 from beginning complexity analysis and require
    resolution at the Stage 3 data compilation level.
    """
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,  # "L_raw", "L_rel", "L_idx"
        expected_format: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code='STAGE51_INPUT_ERROR',
            stage_component='stage_5_1',
            remediation_hint='Verify Stage 3 output files exist and are correctly formatted',
            **kwargs
        )
        
        self.context.update({
            'file_path': file_path,
            'file_type': file_type,
            'expected_format': expected_format
        })

class Stage51ParameterComputationError(Stage5ComputationError):
    """
    Mathematical parameter computation errors for the 16-parameter framework.
    
    Covers computation failures for specific complexity parameters including:
    - P1-P16 mathematical formula evaluation errors
    - Division by zero in parameter calculations  
    - Numerical instability in entropy/variance computations
    - Graph analysis failures for dependency parameters
    - Stochastic sampling errors for P13/P16 parameters
    
    Parameter computation errors indicate issues with input data quality or
    mathematical implementation requiring immediate debugging.
    """
    
    def __init__(
        self,
        message: str,
        parameter_id: Optional[str] = None,  # "P1", "P2", ..., "P16"
        mathematical_formula: Optional[str] = None,
        input_statistics: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            computation_type=f"parameter_{parameter_id}",
            error_code='STAGE51_PARAMETER_ERROR',
            stage_component='stage_5_1',
            remediation_hint='Check input data quality and parameter formula implementation',
            **kwargs
        )
        
        self.context.update({
            'parameter_id': parameter_id,
            'mathematical_formula': mathematical_formula,
            'input_statistics': input_statistics or {}
        })

class Stage51OutputError(Stage5ValidationError):
    """
    Stage 5.1 output generation and validation errors.
    
    Covers output file generation failures including:
    - complexity_metrics.json writing failures
    - Schema validation errors for output JSON
    - Directory creation or permission errors
    - Composite index calculation failures
    - Parameter statistics generation errors
    
    Output errors prevent Stage 5.2 from receiving valid complexity metrics and
    require resolution in Stage 5.1 implementation.
    """
    
    def __init__(
        self,
        message: str,
        output_file: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code='STAGE51_OUTPUT_ERROR', 
            stage_component='stage_5_1',
            remediation_hint='Verify output directory permissions and schema compliance',
            **kwargs
        )
        
        self.context.update({
            'output_file': output_file,
            'validation_errors': validation_errors or []
        })

# =============================================================================
# STAGE 5.2 SPECIFIC EXCEPTIONS  
# Solver Selection & Arsenal Modularity error conditions
# =============================================================================

class Stage52SolverCapabilityError(Stage5ValidationError):
    """
    Solver arsenal capability file validation and loading errors.
    
    Covers solver capability specification failures including:
    - Missing or corrupted solver_capabilities.json file
    - Invalid solver capability vector dimensions (not 16-D)
    - Capability values outside [0,10] effectiveness range
    - Missing or invalid solver paradigm classifications
    - Duplicate solver IDs within arsenal
    
    Capability errors prevent Stage 5.2 from performing solver selection and
    require correction of the solver arsenal configuration.
    """
    
    def __init__(
        self,
        message: str,
        solver_id: Optional[str] = None,
        capability_issue: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code='STAGE52_CAPABILITY_ERROR',
            stage_component='stage_5_2', 
            remediation_hint='Verify solver_capabilities.json format and solver specifications',
            **kwargs
        )
        
        self.context.update({
            'solver_id': solver_id,
            'capability_issue': capability_issue
        })

class Stage52NormalizationError(Stage5ComputationError):
    """
    L2 normalization computation errors in Stage 5.2 solver selection.
    
    Covers L2 normalization failures including:
    - Zero-norm capability vectors (all zeros)
    - Numerical instability in square root computations
    - Matrix dimension mismatches
    - Invalid complexity parameter values for normalization
    
    Normalization errors prevent proper solver-complexity matching and require
    correction of input data or normalization algorithm implementation.
    """
    
    def __init__(
        self,
        message: str,
        normalization_step: Optional[str] = None,
        parameter_index: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            computation_type='l2_normalization',
            error_code='STAGE52_NORMALIZATION_ERROR',
            stage_component='stage_5_2',
            remediation_hint='Check capability vectors and complexity parameters for numerical issues',
            **kwargs
        )
        
        self.context.update({
            'normalization_step': normalization_step,
            'parameter_index': parameter_index
        })

class Stage52LPOptimizationError(Stage5ComputationError):
    """
    Linear programming optimization errors in Stage 5.2 weight learning.
    
    Covers LP optimization failures including:
    - Infeasible LP formulation (no valid solution exists)
    - Unbounded LP problem (infinite separation possible)
    - LP solver crashes or timeouts
    - Convergence failures in iterative algorithm
    - Invalid weight vector results (negative weights, sum ≠ 1)
    
    LP optimization errors prevent optimal solver selection and may indicate
    fundamental issues with the solver arsenal or complexity parameters.
    """
    
    def __init__(
        self,
        message: str,
        lp_status: Optional[str] = None,
        iteration_count: Optional[int] = None,
        solver_count: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            computation_type='lp_weight_learning',
            error_code='STAGE52_LP_ERROR',
            stage_component='stage_5_2',
            remediation_hint='Review LP formulation and solver arsenal for feasibility issues',
            **kwargs
        )
        
        self.context.update({
            'lp_status': lp_status,
            'iteration_count': iteration_count,
            'solver_count': solver_count
        })

class Stage52SelectionOutputError(Stage5ValidationError):
    """
    Stage 5.2 solver selection output generation and validation errors.
    
    Covers selection output failures including:
    - selection_decision.json writing failures
    - Schema validation errors for selection output
    - Ranking consistency validation failures
    - Missing or invalid solver selection results
    - Output directory creation or permission errors
    
    Selection output errors prevent Stage 6 from receiving valid solver choices
    and require resolution in Stage 5.2 implementation.
    """
    
    def __init__(
        self,
        message: str,
        output_file: Optional[str] = None,
        selected_solver: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code='STAGE52_OUTPUT_ERROR',
            stage_component='stage_5_2',
            remediation_hint='Verify selection output format and directory permissions',
            **kwargs
        )
        
        self.context.update({
            'output_file': output_file,
            'selected_solver': selected_solver
        })

# =============================================================================
# INTEGRATION AND SYSTEM-LEVEL EXCEPTIONS
# Cross-stage and pipeline integration error conditions  
# =============================================================================

class Stage5ConfigurationError(Stage5ValidationError):
    """
    Configuration and setup errors affecting Stage 5 execution.
    
    Covers configuration failures including:
    - Missing or invalid configuration files
    - Incorrect path specifications
    - Environment variable issues
    - Dependency library import failures
    - Resource availability problems (memory, disk space)
    
    Configuration errors prevent Stage 5 from initializing properly and require
    system-level resolution before execution can begin.
    """
    
    def __init__(
        self,
        message: str,
        config_parameter: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code='STAGE5_CONFIG_ERROR',
            stage_component='configuration',
            remediation_hint='Review configuration files and system setup',
            **kwargs
        )
        
        self.context.update({
            'config_parameter': config_parameter,
            'config_file': config_file
        })

class Stage5IntegrationError(Stage5BaseException):
    """
    Pipeline integration errors affecting Stage 5 handoffs with other stages.
    
    Covers integration failures including:
    - Stage 3 → Stage 5.1 data handoff failures
    - Stage 5.1 → Stage 5.2 data format mismatches  
    - Stage 5.2 → Stage 6 solver selection handoff issues
    - Version incompatibilities between pipeline stages
    - Schema evolution and backward compatibility problems
    
    Integration errors indicate systemic issues requiring coordination between
    multiple pipeline stages for resolution.
    """
    
    def __init__(
        self,
        message: str,
        source_stage: Optional[str] = None,
        target_stage: Optional[str] = None,
        interface_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code='STAGE5_INTEGRATION_ERROR',
            stage_component='integration',
            remediation_hint='Verify stage interfaces and data format compatibility',
            **kwargs
        )
        
        self.context.update({
            'source_stage': source_stage,
            'target_stage': target_stage,
            'interface_type': interface_type
        })

class Stage5TimeoutError(Stage5BaseException):
    """
    Execution timeout errors for Stage 5 processing.
    
    Covers timeout failures including:
    - Stage 5.1 complexity analysis exceeding time limits
    - Stage 5.2 LP optimization taking too long
    - File I/O operations timing out
    - Network operations (if any) exceeding limits
    
    Timeout errors indicate performance issues or resource constraints requiring
    optimization or system scaling.
    """
    
    def __init__(
        self,
        message: str,
        operation_type: Optional[str] = None,
        timeout_duration: Optional[float] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code='STAGE5_TIMEOUT_ERROR',
            stage_component=kwargs.get('stage_component', 'execution'),
            remediation_hint='Review performance requirements and system resources',
            **kwargs
        )
        
        self.context.update({
            'operation_type': operation_type,
            'timeout_duration': timeout_duration
        })

# =============================================================================
# EXCEPTION UTILITY FUNCTIONS
# Helper functions for exception handling and error reporting
# =============================================================================

def format_validation_error(
    field_name: str,
    invalid_value: Any,
    constraint_description: str,
    stage_component: str = "validation"
) -> Stage5ValidationError:
    """
    Helper function to create consistently formatted validation errors.
    
    Args:
        field_name: Name of the field that failed validation
        invalid_value: The value that violated validation constraints
        constraint_description: Description of the validation constraint
        stage_component: Stage component where validation failed
        
    Returns:
        Properly formatted Stage5ValidationError with complete context
    """
    return Stage5ValidationError(
        message=f"Validation failed for field '{field_name}': {constraint_description}",
        field_name=field_name,
        invalid_value=invalid_value,
        expected_constraints=constraint_description,
        stage_component=stage_component
    )

def format_computation_error(
    computation_type: str,
    error_details: str,
    input_context: Optional[Dict[str, Any]] = None,
    stage_component: str = "computation"
) -> Stage5ComputationError:
    """
    Helper function to create consistently formatted computation errors.
    
    Args:
        computation_type: Type of computation that failed
        error_details: Detailed description of the computation failure
        input_context: Context about input data that caused the failure
        stage_component: Stage component where computation failed
        
    Returns:
        Properly formatted Stage5ComputationError with complete context
    """
    return Stage5ComputationError(
        message=f"Computation failed in {computation_type}: {error_details}",
        computation_type=computation_type,
        input_data_summary=input_context,
        stage_component=stage_component
    )

# Export all exception classes for module imports
__all__ = [
    # Base Exception Classes
    'Stage5BaseException', 'Stage5ValidationError', 'Stage5ComputationError',
    
    # Stage 5.1 Exceptions
    'Stage51InputError', 'Stage51ParameterComputationError', 'Stage51OutputError',
    
    # Stage 5.2 Exceptions
    'Stage52SolverCapabilityError', 'Stage52NormalizationError', 
    'Stage52LPOptimizationError', 'Stage52SelectionOutputError',
    
    # Integration Exceptions
    'Stage5ConfigurationError', 'Stage5IntegrationError', 'Stage5TimeoutError',
    
    # Utility Functions
    'format_validation_error', 'format_computation_error'
]

print("✅ STAGE 5 COMMON/EXCEPTIONS.PY - COMPLETE")
print("   - Comprehensive exception hierarchy with base classes and specializations")
print("   - Stage 5.1 and Stage 5.2 specific exception types for precise error handling")
print("   - Integration and system-level exceptions for pipeline coordination")
print("   - Rich error context and debugging support for all exception types")
print("   - Utility functions for consistent error formatting and handling")
print(f"   - Total exception classes and utilities exported: {len(__all__)}")