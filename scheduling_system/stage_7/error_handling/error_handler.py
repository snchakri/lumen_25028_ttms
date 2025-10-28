"""
Error Handling and Reporting System
===================================

Implements comprehensive error handling with:
- Structured error reports
- Human-readable explanations
- Automated fix recommendations
- Error context preservation
"""

import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum


class ErrorCategory(Enum):
    """Error categories for classification."""
    INPUT_ERROR = "INPUT_ERROR"                    # Invalid input data
    THRESHOLD_VIOLATION = "THRESHOLD_VIOLATION"    # Threshold not met
    MATHEMATICAL_ERROR = "MATHEMATICAL_ERROR"      # Math validation failed
    CONFLICT_ERROR = "CONFLICT_ERROR"              # Schedule conflicts detected
    COVERAGE_ERROR = "COVERAGE_ERROR"              # Incomplete course coverage
    CONSTRAINT_ERROR = "CONSTRAINT_ERROR"          # Constraint violations
    DATA_INTEGRITY = "DATA_INTEGRITY"              # Data integrity issues
    COMPUTATION_ERROR = "COMPUTATION_ERROR"        # Computational errors
    SYSTEM_ERROR = "SYSTEM_ERROR"                  # System-level errors


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "CRITICAL"    # Must abort - solution unacceptable
    ERROR = "ERROR"          # Significant issue - quality compromised
    WARNING = "WARNING"      # Minor issue - acceptable but suboptimal
    INFO = "INFO"            # Informational - no action needed


@dataclass
class FixRecommendation:
    """Recommended fix for an error."""
    action: str                    # Action to take
    description: str               # Detailed description
    priority: str                  # Priority level (HIGH, MEDIUM, LOW)
    automated: bool = False        # Can be automatically fixed
    code_example: Optional[str] = None  # Code example if applicable
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ValidationError:
    """Structured validation error."""
    error_id: str                           # Unique error identifier
    category: ErrorCategory                 # Error category
    severity: ErrorSeverity                 # Severity level
    message: str                            # Short error message
    detailed_message: str                   # Detailed explanation
    threshold_id: Optional[str] = None      # Related threshold
    metric_value: Optional[float] = None    # Actual metric value
    expected_range: Optional[Dict[str, float]] = None  # Expected range
    affected_entities: Optional[List[str]] = None      # Affected entities
    error_context: Optional[Dict[str, Any]] = None     # Additional context
    fix_recommendations: List[FixRecommendation] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            'error_id': self.error_id,
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'detailed_message': self.detailed_message,
            'timestamp': self.timestamp
        }
        
        if self.threshold_id:
            data['threshold_id'] = self.threshold_id
        if self.metric_value is not None:
            data['metric_value'] = self.metric_value
        if self.expected_range:
            data['expected_range'] = self.expected_range
        if self.affected_entities:
            data['affected_entities'] = self.affected_entities
        if self.error_context:
            data['error_context'] = self.error_context
        if self.fix_recommendations:
            data['fix_recommendations'] = [fix.to_dict() for fix in self.fix_recommendations]
        
        return data
    
    def to_human_readable(self) -> str:
        """Convert to human-readable text."""
        lines = [
            "=" * 80,
            f"ERROR: {self.message}",
            "=" * 80,
            f"Category: {self.category.value}",
            f"Severity: {self.severity.value}",
            f"Timestamp: {self.timestamp}",
            ""
        ]
        
        if self.threshold_id:
            lines.append(f"Threshold: {self.threshold_id}")
        
        if self.metric_value is not None:
            lines.append(f"Actual Value: {self.metric_value:.6f}")
        
        if self.expected_range:
            lower = self.expected_range.get('lower', 'N/A')
            upper = self.expected_range.get('upper', 'N/A')
            lines.append(f"Expected Range: [{lower}, {upper}]")
        
        lines.extend([
            "",
            "Detailed Description:",
            "-" * 80,
            self.detailed_message,
            ""
        ])
        
        if self.affected_entities:
            lines.extend([
                "Affected Entities:",
                "-" * 80
            ])
            for entity in self.affected_entities:
                lines.append(f"  - {entity}")
            lines.append("")
        
        if self.fix_recommendations:
            lines.extend([
                "Recommended Fixes:",
                "-" * 80
            ])
            for i, fix in enumerate(self.fix_recommendations, 1):
                lines.extend([
                    f"{i}. {fix.action} (Priority: {fix.priority})",
                    f"   {fix.description}"
                ])
                if fix.code_example:
                    lines.extend([
                        "   Code Example:",
                        f"   {fix.code_example}"
                    ])
                lines.append("")
        
        return "\n".join(lines)


@dataclass
class ErrorReport:
    """Complete error report for validation session."""
    session_id: str
    validation_status: str = "PENDING"  # "PASSED", "FAILED", "PARTIAL", initial "PENDING"
    total_errors: int = 0
    critical_errors: int = 0
    errors: int = 0
    warnings: int = 0
    validation_errors: List[ValidationError] = field(default_factory=list)
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_error(self, error: ValidationError):
        """Add validation error to report."""
        self.validation_errors.append(error)
        self.total_errors += 1
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.critical_errors += 1
        elif error.severity == ErrorSeverity.ERROR:
            self.errors += 1
        elif error.severity == ErrorSeverity.WARNING:
            self.warnings += 1
    
    def generate_summary(self):
        """Generate summary of errors."""
        if self.total_errors == 0:
            self.summary = "All validation checks passed successfully."
            self.validation_status = "PASSED"
        elif self.critical_errors > 0:
            self.summary = (
                f"Validation FAILED with {self.critical_errors} critical error(s), "
                f"{self.errors} error(s), and {self.warnings} warning(s). "
                "Solution is unacceptable and must be rejected."
            )
            self.validation_status = "FAILED"
        else:
            self.summary = (
                f"Validation completed with {self.errors} error(s) and "
                f"{self.warnings} warning(s). Review required."
            )
            self.validation_status = "PARTIAL"
        
        # Generate recommendations
        self.recommendations = []
        if self.critical_errors > 0:
            self.recommendations.append(
                "CRITICAL: Reject this solution and regenerate schedule with corrected parameters."
            )
        
        # Collect unique fix recommendations
        unique_fixes = set()
        for error in self.validation_errors:
            for fix in error.fix_recommendations:
                if fix.priority == "HIGH":
                    unique_fixes.add(fix.action)
        
        self.recommendations.extend(list(unique_fixes))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        self.generate_summary()
        
        return {
            'session_id': self.session_id,
            'validation_status': self.validation_status,
            'timestamp': self.timestamp,
            'total_errors': self.total_errors,
            'critical_errors': self.critical_errors,
            'errors': self.errors,
            'warnings': self.warnings,
            'summary': self.summary,
            'recommendations': self.recommendations,
            'validation_errors': [error.to_dict() for error in self.validation_errors]
        }
    
    def to_json(self, filepath: Path):
        """Save report as JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_text(self, filepath: Path):
        """Save report as human-readable text file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.generate_summary()
        
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STAGE 7 OUTPUT VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Validation Status: {self.validation_status}\n\n")
            
            f.write("Error Summary:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Errors: {self.total_errors}\n")
            f.write(f"  - Critical: {self.critical_errors}\n")
            f.write(f"  - Errors: {self.errors}\n")
            f.write(f"  - Warnings: {self.warnings}\n\n")
            
            f.write("Summary:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{self.summary}\n\n")
            
            if self.recommendations:
                f.write("Recommendations:\n")
                f.write("-" * 80 + "\n")
                for i, rec in enumerate(self.recommendations, 1):
                    f.write(f"{i}. {rec}\n")
                f.write("\n")
            
            if self.validation_errors:
                f.write("Detailed Errors:\n")
                f.write("=" * 80 + "\n\n")
                for error in self.validation_errors:
                    f.write(error.to_human_readable())
                    f.write("\n\n")


class ErrorHandler:
    """
    Error handler for Stage 7 validation.
    
    Creates structured error reports with fix recommendations.
    """
    
    def __init__(self, session_id: str, report_dir: Path):
        """
        Initialize error handler.
        
        Args:
            session_id: Unique session identifier
            report_dir: Directory for error reports
        """
        self.session_id = session_id
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        self.error_report = ErrorReport(session_id=session_id)
        self.error_counter = 0
    
    def create_error(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        message: str,
        detailed_message: str,
        threshold_id: Optional[str] = None,
        metric_value: Optional[float] = None,
        expected_range: Optional[Dict[str, float]] = None,
        affected_entities: Optional[List[str]] = None,
        error_context: Optional[Dict[str, Any]] = None
    ) -> ValidationError:
        """
        Create validation error with auto-generated recommendations.
        
        Args:
            category: Error category
            severity: Error severity
            message: Short error message
            detailed_message: Detailed explanation
            threshold_id: Related threshold (if applicable)
            metric_value: Actual metric value
            expected_range: Expected range for metric
            affected_entities: List of affected entities
            error_context: Additional context
        
        Returns:
            ValidationError object
        """
        self.error_counter += 1
        error_id = f"E{self.error_counter:04d}"
        
        # Create error
        error = ValidationError(
            error_id=error_id,
            category=category,
            severity=severity,
            message=message,
            detailed_message=detailed_message,
            threshold_id=threshold_id,
            metric_value=metric_value,
            expected_range=expected_range,
            affected_entities=affected_entities,
            error_context=error_context
        )
        
        # Generate fix recommendations
        error.fix_recommendations = self._generate_fix_recommendations(error)
        
        # Add to report
        self.error_report.add_error(error)
        
        return error
    
    def _generate_fix_recommendations(self, error: ValidationError) -> List[FixRecommendation]:
        """Generate fix recommendations based on error type."""
        recommendations = []
        
        # Category-specific recommendations
        if error.category == ErrorCategory.THRESHOLD_VIOLATION:
            if error.threshold_id == 'tau1':  # Course Coverage
                recommendations.append(FixRecommendation(
                    action="Increase resource allocation",
                    description="Add more faculty, rooms, or time slots to accommodate all required courses.",
                    priority="HIGH"
                ))
                recommendations.append(FixRecommendation(
                    action="Review course requirements",
                    description="Verify that all required courses are properly defined and have competent faculty assigned.",
                    priority="HIGH"
                ))
            
            elif error.threshold_id == 'tau2':  # Conflict Resolution
                recommendations.append(FixRecommendation(
                    action="Re-run solver with stricter constraints",
                    description="Conflicts indicate solver produced invalid solution. Re-run with explicit conflict prevention constraints.",
                    priority="CRITICAL"
                ))
            
            elif error.threshold_id == 'tau3':  # Workload Balance
                recommendations.append(FixRecommendation(
                    action="Rebalance faculty assignments",
                    description="Redistribute courses among faculty to achieve more uniform workload distribution.",
                    priority="MEDIUM"
                ))
            
            elif error.threshold_id == 'tau4':  # Room Utilization
                recommendations.append(FixRecommendation(
                    action="Optimize room assignments",
                    description="Improve room allocation to better match batch sizes with room capacities.",
                    priority="MEDIUM"
                ))
            
            elif error.threshold_id == 'tau6':  # Pedagogical Sequence
                recommendations.append(FixRecommendation(
                    action="Fix prerequisite ordering",
                    description="Ensure prerequisite courses are scheduled before dependent courses.",
                    priority="CRITICAL"
                ))
        
        elif error.category == ErrorCategory.CONFLICT_ERROR:
            recommendations.append(FixRecommendation(
                action="Resolve scheduling conflicts",
                description="Eliminate overlapping assignments for faculty, rooms, or batches.",
                priority="CRITICAL"
            ))
        
        elif error.category == ErrorCategory.COVERAGE_ERROR:
            recommendations.append(FixRecommendation(
                action="Ensure complete curriculum coverage",
                description="Schedule all required courses for complete curriculum delivery.",
                priority="HIGH"
            ))
        
        elif error.category == ErrorCategory.INPUT_ERROR:
            recommendations.append(FixRecommendation(
                action="Fix input data",
                description="Correct malformed or invalid input data before re-running validation.",
                priority="CRITICAL"
            ))
        
        # Add generic recommendation
        if error.severity == ErrorSeverity.CRITICAL:
            recommendations.append(FixRecommendation(
                action="Regenerate schedule",
                description="Critical error detected. Reject current solution and regenerate with corrected parameters.",
                priority="CRITICAL"
            ))
        
        return recommendations
    
    def finalize_report(
        self,
        output_formats: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """
        Finalize and save error report.
        
        Args:
            output_formats: List of output formats ('json', 'txt')
        
        Returns:
            Dictionary mapping format to filepath
        """
        if output_formats is None:
            output_formats = ['json', 'txt']
        
        output_files = {}
        
        # Generate report files
        if 'json' in output_formats:
            json_path = self.report_dir / f"{self.session_id}_error_report.json"
            self.error_report.to_json(json_path)
            output_files['json'] = json_path
        
        if 'txt' in output_formats:
            txt_path = self.report_dir / f"{self.session_id}_error_report.txt"
            self.error_report.to_text(txt_path)
            output_files['txt'] = txt_path
        
        return output_files
    
    def should_abort(self) -> bool:
        """Check if validation should abort based on errors."""
        return self.error_report.critical_errors > 0
    
    def get_report(self) -> ErrorReport:
        """Get current error report."""
        return self.error_report


def create_error_handler(session_id: str, report_dir: Path) -> ErrorHandler:
    """Factory function to create ErrorHandler."""
    return ErrorHandler(session_id=session_id, report_dir=report_dir)
