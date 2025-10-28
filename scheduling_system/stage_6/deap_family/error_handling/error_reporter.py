"""
Error Reporter for DEAP Solver Family

Implements comprehensive error reporting with human-readable messages,
advised fixes, and dual output (console + JSON/TXT files).

Author: LUMEN Team [TEAM-ID: 93912]
"""

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ErrorReport:
    """Structured error report."""
    error_id: str
    timestamp: str
    error_type: str
    error_message: str
    phase: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    context: Dict[str, Any]
    stack_trace: Optional[str]
    advised_fixes: List[str]
    recovery_suggestions: List[str]
    foundation_compliance: Dict[str, str]  # foundation_section -> compliance_status


class ErrorReporter:
    """
    Comprehensive error reporting system.
    
    Provides dual output (console + files) with structured error data,
    human-readable messages, and advised fixes.
    """
    
    def __init__(self, error_report_path: Path):
        """
        Initialize error reporter.
        
        Args:
            error_report_path: Path to write error reports
        """
        self.error_report_path = Path(error_report_path)
        self.error_report_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Error counter
        self.error_count = 0
        
        # Error registry for tracking
        self.error_registry: List[ErrorReport] = []
    
    def create_error_report(
        self,
        error_type: str,
        error_message: str,
        phase: str,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "HIGH",
        stack_trace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive error report.
        
        Args:
            error_type: Type of error (InputError, SolverError, ValidationError, etc.)
            error_message: Human-readable error message
            phase: Pipeline phase where error occurred
            context: Additional context data
            severity: Error severity level
            stack_trace: Optional stack trace
        
        Returns:
            Error report dictionary
        """
        self.error_count += 1
        error_id = f"DEAP_ERROR_{self.error_count:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate advised fixes based on error type and context
        advised_fixes = self._generate_advised_fixes(error_type, error_message, context)
        
        # Generate recovery suggestions
        recovery_suggestions = self._generate_recovery_suggestions(error_type, phase, context)
        
        # Check foundation compliance
        foundation_compliance = self._check_foundation_compliance(error_type, phase)
        
        # Create error report
        error_report = ErrorReport(
            error_id=error_id,
            timestamp=datetime.now().isoformat(),
            error_type=error_type,
            error_message=error_message,
            phase=phase,
            severity=severity,
            context=context or {},
            stack_trace=stack_trace,
            advised_fixes=advised_fixes,
            recovery_suggestions=recovery_suggestions,
            foundation_compliance=foundation_compliance
        )
        
        # Register error
        self.error_registry.append(error_report)
        
        # Output to console
        self._output_to_console(error_report)
        
        # Output to files
        self._output_to_files(error_report)
        
        return asdict(error_report)
    
    def _generate_advised_fixes(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate advised fixes based on error analysis."""
        fixes = []
        
        if error_type == "InputError":
            fixes.extend([
                "Verify Stage 3 output files exist and are readable",
                "Check file formats: LRAW/LOPT (Parquet), LREL (GraphML), LIDX (Pickle)",
                "Validate data schema compliance with Stage 3 specifications",
                "Ensure dynamic_parameters.parquet contains required solver parameters"
            ])
            
        elif error_type == "SolverError":
            fixes.extend([
                "Check solver parameter validity and ranges",
                "Verify population size meets minimum requirements (≥50 per foundation)",
                "Validate constraint formulation and penalty weights",
                "Ensure fitness function components are properly normalized"
            ])
            
        elif error_type == "ValidationError":
            fixes.extend([
                "Review input data for missing or invalid values",
                "Check bijective mapping consistency",
                "Validate constraint satisfaction in generated solutions",
                "Verify genotype-phenotype mapping integrity"
            ])
            
        elif error_type == "ConfigurationError":
            fixes.extend([
                "Verify configuration file syntax and completeness",
                "Check parameter bounds against theoretical limits",
                "Validate solver selection criteria",
                "Ensure all required paths are accessible"
            ])
            
        elif error_type == "MemoryError":
            fixes.extend([
                "Increase available memory allocation",
                "Consider population size reduction (while maintaining foundation compliance)",
                "Implement memory-efficient data structures",
                "Use streaming processing for large datasets"
            ])
            
        elif error_type == "ConvergenceError":
            fixes.extend([
                "Increase maximum generations limit",
                "Adjust mutation and crossover rates",
                "Review diversity maintenance mechanisms",
                "Consider hybrid approaches or local search integration"
            ])
        
        # Add context-specific fixes
        if context:
            if "missing_files" in context:
                fixes.append(f"Ensure these files exist: {', '.join(context['missing_files'])}")
            
            if "invalid_parameters" in context:
                fixes.append(f"Fix invalid parameters: {', '.join(context['invalid_parameters'])}")
            
            if "constraint_violations" in context:
                fixes.append(f"Address constraint violations: {context['constraint_violations']}")
        
        return fixes
    
    def _generate_recovery_suggestions(
        self,
        error_type: str,
        phase: str,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate recovery suggestions."""
        suggestions = []
        
        if phase == "input_loading":
            suggestions.extend([
                "Retry with alternative file paths",
                "Regenerate Stage 3 outputs if corrupted",
                "Use backup data if available",
                "Contact data pipeline team for assistance"
            ])
            
        elif phase == "solver_selection":
            suggestions.extend([
                "Try alternative solver (fallback mechanism)",
                "Use default NSGA-II configuration",
                "Reduce problem complexity temporarily",
                "Manual solver specification override"
            ])
            
        elif phase == "evolution":
            suggestions.extend([
                "Restart with different random seed",
                "Reduce population size and increase generations",
                "Switch to more robust solver (ES or DE)",
                "Implement checkpoint-based recovery"
            ])
            
        elif phase == "output_generation":
            suggestions.extend([
                "Retry output writing with different format",
                "Check disk space and permissions",
                "Use alternative output directory",
                "Generate minimal output for debugging"
            ])
        
        return suggestions
    
    def _check_foundation_compliance(self, error_type: str, phase: str) -> Dict[str, str]:
        """Check compliance with foundational requirements."""
        compliance = {}
        
        # Check against Stage 6.3 foundations
        if error_type == "SolverError":
            compliance["Definition_2.1_EA_Framework"] = "REQUIRES_REVIEW"
            compliance["Section_13_Performance_Analysis"] = "REQUIRES_REVIEW"
            
        if phase == "input_loading":
            compliance["Section_12_Pipeline_Integration"] = "REQUIRES_REVIEW"
            compliance["Dynamic_Parametric_System"] = "REQUIRES_REVIEW"
            
        if error_type == "ValidationError":
            compliance["Definition_2.2_Genotype_Encoding"] = "REQUIRES_REVIEW"
            compliance["Definition_2.3_Phenotype_Mapping"] = "REQUIRES_REVIEW"
        
        return compliance
    
    def _output_to_console(self, error_report: ErrorReport):
        """Output error report to console."""
        print("\n" + "=" * 80)
        print(f"🚨 DEAP SOLVER ERROR REPORT - {error_report.error_id}")
        print("=" * 80)
        print(f"Timestamp: {error_report.timestamp}")
        print(f"Error Type: {error_report.error_type}")
        print(f"Phase: {error_report.phase}")
        print(f"Severity: {error_report.severity}")
        print(f"Message: {error_report.error_message}")
        
        if error_report.context:
            print(f"\nContext:")
            for key, value in error_report.context.items():
                print(f"  {key}: {value}")
        
        if error_report.advised_fixes:
            print(f"\n🔧 Advised Fixes:")
            for i, fix in enumerate(error_report.advised_fixes, 1):
                print(f"  {i}. {fix}")
        
        if error_report.recovery_suggestions:
            print(f"\n🔄 Recovery Suggestions:")
            for i, suggestion in enumerate(error_report.recovery_suggestions, 1):
                print(f"  {i}. {suggestion}")
        
        if error_report.foundation_compliance:
            print(f"\n📋 Foundation Compliance:")
            for section, status in error_report.foundation_compliance.items():
                print(f"  {section}: {status}")
        
        if error_report.stack_trace:
            print(f"\n📚 Stack Trace:")
            print(error_report.stack_trace)
        
        print("=" * 80)
        
        # Also log to logger
        self.logger.error(f"Error {error_report.error_id}: {error_report.error_message}")
    
    def _output_to_files(self, error_report: ErrorReport):
        """Output error report to JSON and TXT files."""
        # JSON file for structured data
        json_path = self.error_report_path / f"{error_report.error_id}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(error_report), f, indent=2, default=str)
        
        # TXT file for human-readable report
        txt_path = self.error_report_path / f"{error_report.error_id}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"DEAP SOLVER ERROR REPORT - {error_report.error_id}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {error_report.timestamp}\n")
            f.write(f"Error Type: {error_report.error_type}\n")
            f.write(f"Phase: {error_report.phase}\n")
            f.write(f"Severity: {error_report.severity}\n")
            f.write(f"Message: {error_report.error_message}\n\n")
            
            if error_report.context:
                f.write("Context:\n")
                for key, value in error_report.context.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            if error_report.advised_fixes:
                f.write("Advised Fixes:\n")
                for i, fix in enumerate(error_report.advised_fixes, 1):
                    f.write(f"  {i}. {fix}\n")
                f.write("\n")
            
            if error_report.recovery_suggestions:
                f.write("Recovery Suggestions:\n")
                for i, suggestion in enumerate(error_report.recovery_suggestions, 1):
                    f.write(f"  {i}. {suggestion}\n")
                f.write("\n")
            
            if error_report.foundation_compliance:
                f.write("Foundation Compliance:\n")
                for section, status in error_report.foundation_compliance.items():
                    f.write(f"  {section}: {status}\n")
                f.write("\n")
            
            if error_report.stack_trace:
                f.write("Stack Trace:\n")
                f.write(error_report.stack_trace)
                f.write("\n")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors."""
        if not self.error_registry:
            return {"total_errors": 0, "by_type": {}, "by_phase": {}, "by_severity": {}}
        
        by_type = {}
        by_phase = {}
        by_severity = {}
        
        for error in self.error_registry:
            # Count by type
            by_type[error.error_type] = by_type.get(error.error_type, 0) + 1
            
            # Count by phase
            by_phase[error.phase] = by_phase.get(error.phase, 0) + 1
            
            # Count by severity
            by_severity[error.severity] = by_severity.get(error.severity, 0) + 1
        
        return {
            "total_errors": len(self.error_registry),
            "by_type": by_type,
            "by_phase": by_phase,
            "by_severity": by_severity,
            "latest_error": asdict(self.error_registry[-1]) if self.error_registry else None
        }
    
    def has_critical_errors(self) -> bool:
        """Check if any critical errors occurred."""
        return any(error.severity == "CRITICAL" for error in self.error_registry)
    
    def clear_errors(self):
        """Clear error registry."""
        self.error_registry.clear()
        self.error_count = 0

