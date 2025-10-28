"""
Comprehensive Error Handling System for Stage 4 Feasibility Check
Implements error reporting with JSON and TXT outputs including fixes
"""

import json
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum


class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories"""
    INPUT_ERROR = "input_error"
    VALIDATION_ERROR = "validation_error"
    THEOREM_VIOLATION = "theorem_violation"
    MATHEMATICAL_ERROR = "mathematical_error"
    SYSTEM_ERROR = "system_error"
    CONFIGURATION_ERROR = "configuration_error"


@dataclass
class ErrorReport:
    """Comprehensive error report structure"""
    timestamp: str
    error_id: str
    category: str
    severity: str
    error_type: str
    message: str
    layer: Optional[str]
    raw_error: Dict[str, Any]
    human_readable_explanation: str
    suggested_fixes: List[str]
    mathematical_reasoning: Optional[str]
    context: Dict[str, Any]
    stack_trace: str


class FeasibilityError(Exception):
    """Base exception for feasibility checking errors"""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM_ERROR):
        self.message = message
        self.category = category
        super().__init__(self.message)


class LayerValidationError(FeasibilityError):
    """Exception raised when layer validation fails"""
    
    def __init__(self, message: str, layer: str, details: Dict[str, Any]):
        self.layer = layer
        self.details = details
        super().__init__(message, ErrorCategory.VALIDATION_ERROR)


class TheoremViolationError(FeasibilityError):
    """Exception raised when a theorem is violated"""
    
    def __init__(self, message: str, theorem: str, proof: str):
        self.theorem = theorem
        self.proof = proof
        super().__init__(message, ErrorCategory.THEOREM_VIOLATION)


class Stage3InputError(FeasibilityError):
    """Exception raised when Stage 3 input is invalid or missing"""
    
    def __init__(self, message: str, missing_files: List[str]):
        self.missing_files = missing_files
        super().__init__(message, ErrorCategory.INPUT_ERROR)


class MathematicalProofError(FeasibilityError):
    """Exception raised when mathematical proof validation fails"""
    
    def __init__(self, message: str, proof_statement: str, conditions: List[str]):
        self.proof_statement = proof_statement
        self.conditions = conditions
        super().__init__(message, ErrorCategory.MATHEMATICAL_ERROR)


class ErrorHandler:
    """
    Comprehensive error handling system for Stage 4
    
    Features:
    - Custom exception hierarchy
    - Error report generation (JSON + TXT)
    - Human-readable explanations
    - Suggested fixes based on error type
    - Mathematical reasoning for failures
    - Error aggregation across layers
    """
    
    def __init__(self, output_directory: Path):
        """
        Initialize error handler
        
        Args:
            output_directory: Directory for error reports
        """
        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.error_reports: List[ErrorReport] = []
        self.error_counter = 0
    
    def handle_error(
        self,
        exception: Exception,
        layer: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH
    ) -> ErrorReport:
        """
        Handle an exception and generate error report
        
        Args:
            exception: The exception that occurred
            layer: Layer where error occurred
            context: Additional context information
            severity: Error severity level
            
        Returns:
            ErrorReport with comprehensive error information
        """
        self.error_counter += 1
        error_id = f"ERR_{self.error_counter:04d}"
        
        # Determine error category
        if isinstance(exception, FeasibilityError):
            category = exception.category.value
        elif isinstance(exception, (ValueError, TypeError)):
            category = ErrorCategory.INPUT_ERROR.value
        elif isinstance(exception, (AssertionError, RuntimeError)):
            category = ErrorCategory.VALIDATION_ERROR.value
        else:
            category = ErrorCategory.SYSTEM_ERROR.value
        
        # Generate error report
        report = ErrorReport(
            timestamp=datetime.utcnow().isoformat(),
            error_id=error_id,
            category=category,
            severity=severity.value,
            error_type=type(exception).__name__,
            message=str(exception),
            layer=layer,
            raw_error=self._extract_raw_error(exception),
            human_readable_explanation=self._generate_explanation(exception, layer),
            suggested_fixes=self._suggest_fixes(exception, layer),
            mathematical_reasoning=self._generate_mathematical_reasoning(exception),
            context=context or {},
            stack_trace=traceback.format_exc()
        )
        
        self.error_reports.append(report)
        return report
    
    def _extract_raw_error(self, exception: Exception) -> Dict[str, Any]:
        """Extract raw error information"""
        return {
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "exception_args": exception.args if hasattr(exception, 'args') else []
        }
    
    def _generate_explanation(self, exception: Exception, layer: Optional[str]) -> str:
        """Generate human-readable explanation"""
        layer_info = f" in Layer {layer}" if layer else ""
        
        if isinstance(exception, Stage3InputError):
            return f"Stage 3 input data is invalid or incomplete{layer_info}. " \
                   f"Missing files: {', '.join(exception.missing_files)}"
        
        elif isinstance(exception, TheoremViolationError):
            return f"Mathematical theorem violation{layer_info}. " \
                   f"Theorem {exception.theorem} has been violated, indicating infeasibility."
        
        elif isinstance(exception, LayerValidationError):
            return f"Layer validation failed{layer_info}. " \
                   f"Validation details: {exception.details}"
        
        elif isinstance(exception, MathematicalProofError):
            return f"Mathematical proof validation failed{layer_info}. " \
                   f"Proof statement: {exception.proof_statement}"
        
        elif isinstance(exception, (ValueError, TypeError)):
            return f"Invalid input or data type{layer_info}. " \
                   f"Error: {str(exception)}"
        
        elif isinstance(exception, (AssertionError, RuntimeError)):
            return f"Runtime validation error{layer_info}. " \
                   f"Error: {str(exception)}"
        
        else:
            return f"Unexpected error occurred{layer_info}. " \
                   f"Error: {str(exception)}"
    
    def _suggest_fixes(self, exception: Exception, layer: Optional[str]) -> List[str]:
        """Generate suggested fixes based on error type"""
        fixes = []
        
        if isinstance(exception, Stage3InputError):
            fixes.extend([
                f"Verify that all required Stage 3 output files are present",
                f"Check that Stage 3 completed successfully before running Stage 4",
                f"Ensure Stage 3 output directory path is correct"
            ])
        
        elif isinstance(exception, TheoremViolationError):
            fixes.extend([
                f"Review input data for violations of {exception.theorem}",
                f"Check data consistency and constraints",
                f"Consider relaxing constraints if infeasibility is due to over-constraining"
            ])
        
        elif isinstance(exception, LayerValidationError):
            fixes.extend([
                f"Review layer {layer} validation details",
                f"Check input data quality and completeness",
                f"Verify that all required fields are present and valid"
            ])
        
        elif isinstance(exception, MathematicalProofError):
            fixes.extend([
                f"Review mathematical proof conditions",
                f"Verify that all mathematical constraints are satisfied",
                f"Check for numerical precision issues"
            ])
        
        elif isinstance(exception, (ValueError, TypeError)):
            fixes.extend([
                f"Check data types and formats",
                f"Verify that all required parameters are provided",
                f"Ensure data is in expected format"
            ])
        
        else:
            fixes.extend([
                f"Review error details and stack trace",
                f"Check system resources and configuration",
                f"Contact support if error persists"
            ])
        
        return fixes
    
    def _generate_mathematical_reasoning(self, exception: Exception) -> Optional[str]:
        """Generate mathematical reasoning for the error"""
        if isinstance(exception, TheoremViolationError):
            return f"Theorem {exception.theorem} states: {exception.proof}. " \
                   f"This violation indicates that the instance is infeasible."
        
        elif isinstance(exception, MathematicalProofError):
            return f"Mathematical proof failed. Conditions: {', '.join(exception.conditions)}"
        
        return None
    
    def save_error_reports(self, output_path: Optional[Path] = None):
        """
        Save all error reports to JSON and TXT files
        
        Args:
            output_path: Optional custom output path (defaults to output_directory)
        """
        if not self.error_reports:
            return
        
        if output_path is None:
            output_path = self.output_directory / "error_reports"
        
        # Save JSON report
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump([asdict(report) for report in self.error_reports], f, indent=2)
        
        # Save TXT report
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STAGE 4 FEASIBILITY CHECK - ERROR REPORTS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.utcnow().isoformat()}\n")
            f.write(f"Total Errors: {len(self.error_reports)}\n\n")
            
            for i, report in enumerate(self.error_reports, 1):
                f.write(f"\n{'=' * 80}\n")
                f.write(f"ERROR #{i}: {report.error_id}\n")
                f.write(f"{'=' * 80}\n\n")
                
                f.write(f"Timestamp: {report.timestamp}\n")
                f.write(f"Category: {report.category}\n")
                f.write(f"Severity: {report.severity}\n")
                f.write(f"Error Type: {report.error_type}\n")
                if report.layer:
                    f.write(f"Layer: {report.layer}\n")
                f.write(f"\n")
                
                f.write(f"MESSAGE:\n{report.message}\n\n")
                
                f.write(f"HUMAN-READABLE EXPLANATION:\n{report.human_readable_explanation}\n\n")
                
                if report.suggested_fixes:
                    f.write(f"SUGGESTED FIXES:\n")
                    for j, fix in enumerate(report.suggested_fixes, 1):
                        f.write(f"{j}. {fix}\n")
                    f.write(f"\n")
                
                if report.mathematical_reasoning:
                    f.write(f"MATHEMATICAL REASONING:\n{report.mathematical_reasoning}\n\n")
                
                if report.context:
                    f.write(f"CONTEXT:\n{json.dumps(report.context, indent=2)}\n\n")
                
                f.write(f"STACK TRACE:\n{report.stack_trace}\n\n")
        
        return json_path, txt_path
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors"""
        if not self.error_reports:
            return {"total_errors": 0}
        
        severity_counts = {}
        category_counts = {}
        layer_counts = {}
        
        for report in self.error_reports:
            severity_counts[report.severity] = severity_counts.get(report.severity, 0) + 1
            category_counts[report.category] = category_counts.get(report.category, 0) + 1
            if report.layer:
                layer_counts[report.layer] = layer_counts.get(report.layer, 0) + 1
        
        return {
            "total_errors": len(self.error_reports),
            "severity_distribution": severity_counts,
            "category_distribution": category_counts,
            "layer_distribution": layer_counts
        }
    
    def clear_errors(self):
        """Clear all error reports"""
        self.error_reports.clear()
        self.error_counter = 0
