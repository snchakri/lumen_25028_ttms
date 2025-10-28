"""
Error Reporter for Stage 5 - Foundation-Compliant Error Handling and Reporting

This module implements comprehensive error reporting with both raw error data
and human-readable comprehensive reports with suggested fixes, as required
by the Stage 5 foundations.
"""

import json
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import structlog

logger = structlog.get_logger(__name__)


class Stage5ErrorReporter:
    """
    Comprehensive error reporting system for Stage 5 with foundation compliance.
    
    Generates both JSON and TXT reports with:
    - Raw error data
    - Human-readable comprehensive analysis
    - Suggested fixes based on error type
    - Diagnostic information for debugging
    """
    
    def __init__(self):
        self.logger = logger.bind(component="error_reporter")
    
    def generate_error_report(self, 
                            error: Exception, 
                            context: Dict[str, Any],
                            reports_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate comprehensive error report with both JSON and TXT outputs.
        
        Args:
            error: The exception that occurred
            context: Context information about the error
            reports_path: Path to write report files (optional)
            
        Returns:
            Complete error report dictionary
        """
        error_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Build comprehensive error report
        report = {
            "error_metadata": {
                "error_id": error_id,
                "timestamp": timestamp,
                "stage": context.get("stage", "5"),
                "substage": context.get("substage", "unknown"),
                "component": context.get("component", "unknown"),
                "foundation_document": context.get("foundation_document", 
                                                 "Stage-5.1 & Stage-5.2 Foundations")
            },
            "error_details": {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_category": self._categorize_error(error),
                "severity": self._assess_severity(error, context),
                "is_recoverable": self._is_recoverable(error)
            },
            "technical_details": {
                "stack_trace": traceback.format_exc(),
                "error_location": self._extract_error_location(error),
                "function_context": context.get("function", "unknown"),
                "input_parameters": context.get("inputs", {}),
                "system_state": self._gather_system_state(context)
            },
            "diagnostic_information": self._gather_diagnostics(error, context),
            "suggested_fixes": self._generate_suggested_fixes(error, context),
            "foundation_compliance": self._check_foundation_compliance(error, context),
            "context": context
        }
        
        # Write reports if path provided
        if reports_path:
            self._write_json_report(report, reports_path)
            self._write_txt_report(report, reports_path)
        
        # Log error occurrence
        self.logger.error("Stage 5 error occurred",
                         error_id=error_id,
                         error_type=type(error).__name__,
                         error_message=str(error),
                         severity=report["error_details"]["severity"],
                         recoverable=report["error_details"]["is_recoverable"])
        
        return report
    
    def _categorize_error(self, error: Exception) -> str:
        """Categorize error type for better handling."""
        error_categories = {
            "ValueError": "data_validation",
            "TypeError": "type_mismatch",
            "KeyError": "missing_data",
            "FileNotFoundError": "file_system",
            "PermissionError": "file_system",
            "RuntimeError": "runtime_failure",
            "AssertionError": "foundation_violation",
            "ImportError": "dependency_missing",
            "MemoryError": "resource_exhaustion",
            "TimeoutError": "performance_issue"
        }
        
        error_type = type(error).__name__
        return error_categories.get(error_type, "unknown")
    
    def _assess_severity(self, error: Exception, context: Dict[str, Any]) -> str:
        """Assess error severity level."""
        error_type = type(error).__name__
        
        # Critical errors that violate foundations
        if error_type in ["AssertionError"] and "theorem" in str(error).lower():
            return "CRITICAL"
        
        # High severity errors that prevent execution
        if error_type in ["FileNotFoundError", "MemoryError", "ImportError"]:
            return "HIGH"
        
        # Medium severity errors that may be recoverable
        if error_type in ["ValueError", "TypeError", "KeyError"]:
            return "MEDIUM"
        
        # Low severity errors
        return "LOW"
    
    def _is_recoverable(self, error: Exception) -> bool:
        """Determine if error is potentially recoverable."""
        non_recoverable_types = [
            "MemoryError",
            "ImportError", 
            "SystemExit",
            "KeyboardInterrupt"
        ]
        
        error_type = type(error).__name__
        
        # Foundation violations are generally not recoverable
        if error_type == "AssertionError" and "theorem" in str(error).lower():
            return False
        
        return error_type not in non_recoverable_types
    
    def _extract_error_location(self, error: Exception) -> Dict[str, Any]:
        """Extract detailed error location information."""
        tb = traceback.extract_tb(error.__traceback__)
        if not tb:
            return {"file": "unknown", "line": 0, "function": "unknown"}
        
        # Get the last frame (where error occurred)
        last_frame = tb[-1]
        
        return {
            "file": last_frame.filename,
            "line": last_frame.lineno,
            "function": last_frame.name,
            "code_context": last_frame.line,
            "full_traceback_frames": len(tb)
        }
    
    def _gather_system_state(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather relevant system state information."""
        import psutil
        import sys
        
        try:
            return {
                "python_version": sys.version,
                "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                "cpu_percent": psutil.cpu_percent(),
                "available_memory_mb": psutil.virtual_memory().available / 1024 / 1024,
                "working_directory": str(Path.cwd()),
                "stage3_path_exists": context.get("stage3_path") and Path(context["stage3_path"]).exists(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception:
            return {"error": "Could not gather system state"}
    
    def _gather_diagnostics(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather diagnostic information specific to error type."""
        diagnostics = {}
        error_type = type(error).__name__
        
        if error_type == "FileNotFoundError":
            diagnostics.update(self._diagnose_file_error(error, context))
        elif error_type == "ValueError":
            diagnostics.update(self._diagnose_value_error(error, context))
        elif error_type == "KeyError":
            diagnostics.update(self._diagnose_key_error(error, context))
        elif error_type == "AssertionError":
            diagnostics.update(self._diagnose_assertion_error(error, context))
        
        return diagnostics
    
    def _diagnose_file_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose file-related errors."""
        diagnostics = {"error_type": "file_system"}
        
        # Check if Stage 3 path exists
        stage3_path = context.get("stage3_path")
        if stage3_path:
            stage3_path = Path(stage3_path)
            diagnostics["stage3_path_exists"] = stage3_path.exists()
            
            if stage3_path.exists():
                diagnostics["stage3_contents"] = [
                    str(p.name) for p in stage3_path.iterdir() if p.is_dir()
                ]
                
                # Check for expected subdirectories
                expected_dirs = ["L_raw", "L_rel"]
                diagnostics["expected_dirs_present"] = {
                    dir_name: (stage3_path / dir_name).exists() 
                    for dir_name in expected_dirs
                }
        
        return diagnostics
    
    def _diagnose_value_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose value-related errors."""
        return {
            "error_type": "data_validation",
            "error_message_analysis": str(error),
            "input_data_types": {
                k: type(v).__name__ for k, v in context.get("inputs", {}).items()
            }
        }
    
    def _diagnose_key_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose key-related errors."""
        missing_key = str(error).strip("'\"")
        
        return {
            "error_type": "missing_data",
            "missing_key": missing_key,
            "available_keys": list(context.get("available_keys", [])),
            "data_structure_type": context.get("data_structure_type", "unknown")
        }
    
    def _diagnose_assertion_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose assertion errors (foundation violations)."""
        error_msg = str(error)
        
        diagnostics = {
            "error_type": "foundation_violation",
            "assertion_message": error_msg
        }
        
        # Check if it's a theorem violation
        if "theorem" in error_msg.lower():
            diagnostics["violation_type"] = "theorem_violation"
            diagnostics["requires_foundation_review"] = True
        
        return diagnostics
    
    def _generate_suggested_fixes(self, error: Exception, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate contextual suggested fixes."""
        fixes = []
        error_type = type(error).__name__
        error_msg = str(error)
        
        if error_type == "FileNotFoundError":
            fixes.extend([
                {
                    "priority": "HIGH",
                    "category": "data_path",
                    "description": "Verify Stage 3 output path is correct",
                    "action": "Check that stage3_output_path points to valid Stage 3 output directory"
                },
                {
                    "priority": "HIGH", 
                    "category": "data_structure",
                    "description": "Ensure Stage 3 completed successfully",
                    "action": "Run Stage 3 to completion before executing Stage 5"
                }
            ])
        
        elif error_type == "ValueError" and "parameter" in error_msg.lower():
            fixes.extend([
                {
                    "priority": "MEDIUM",
                    "category": "parameter_validation",
                    "description": "Check parameter value ranges",
                    "action": "Verify all complexity parameters are within expected bounds [0, 1]"
                },
                {
                    "priority": "MEDIUM",
                    "category": "data_quality",
                    "description": "Validate input data completeness",
                    "action": "Ensure all required entities (courses, faculty, rooms, etc.) are present"
                }
            ])
        
        elif error_type == "AssertionError" and "theorem" in error_msg.lower():
            fixes.extend([
                {
                    "priority": "CRITICAL",
                    "category": "foundation_compliance",
                    "description": "Foundation theorem violation detected",
                    "action": "Review mathematical implementation against theoretical foundations"
                },
                {
                    "priority": "HIGH",
                    "category": "algorithm_review",
                    "description": "Algorithm may not match foundation specification",
                    "action": "Compare implementation with exact formulas in foundation documents"
                }
            ])
        
        elif "LP" in error_msg or "linear programming" in error_msg.lower():
            fixes.extend([
                {
                    "priority": "HIGH",
                    "category": "solver_configuration",
                    "description": "Linear programming solver issue",
                    "action": "Check solver capabilities matrix and problem complexity vector dimensions"
                },
                {
                    "priority": "MEDIUM",
                    "category": "numerical_stability",
                    "description": "Numerical stability issue in LP optimization",
                    "action": "Verify normalization factors are non-zero and finite"
                }
            ])
        
        # Always add general debugging fix
        fixes.append({
            "priority": "LOW",
            "category": "debugging",
            "description": "Enable detailed logging for diagnosis",
            "action": "Set LOG_LEVEL=DEBUG and review detailed logs in stage5.json"
        })
        
        return fixes
    
    def _check_foundation_compliance(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if error indicates foundation compliance issues."""
        error_msg = str(error).lower()
        
        compliance_issues = []
        
        # Check for common foundation violations
        if "mock" in error_msg or "synthetic" in error_msg:
            compliance_issues.append({
                "issue": "mock_data_usage",
                "description": "Mock or synthetic data detected - violates foundation requirement",
                "foundation_section": "Phase 2.1 - Remove ALL Mock/Synthetic/Hardcoded Data"
            })
        
        if "ensemble" in error_msg or "voting" in error_msg:
            compliance_issues.append({
                "issue": "ensemble_usage",
                "description": "Ensemble methods detected - violates Stage 5.2 LP framework requirement",
                "foundation_section": "Phase 3 - Pure LP Framework (NO SOLVERS)"
            })
        
        if "l_opt" in error_msg or "l_idx" in error_msg:
            compliance_issues.append({
                "issue": "incorrect_data_loading",
                "description": "Loading L_opt or L_idx - not required by Stage 5 foundations",
                "foundation_section": "Phase 1.2 - Foundation-Compliant Data Loading"
            })
        
        return {
            "compliance_status": "VIOLATION" if compliance_issues else "COMPLIANT",
            "issues_detected": compliance_issues,
            "requires_foundation_review": len(compliance_issues) > 0
        }
    
    def _write_json_report(self, report: Dict[str, Any], reports_path: Path) -> None:
        """Write JSON error report."""
        reports_path.mkdir(parents=True, exist_ok=True)
        
        json_path = reports_path / f"error_{report['error_metadata']['error_id']}.json"
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info("JSON error report written", path=str(json_path))
        except Exception as e:
            self.logger.error("Failed to write JSON error report", error=str(e))
    
    def _write_txt_report(self, report: Dict[str, Any], reports_path: Path) -> None:
        """Write human-readable TXT error report."""
        reports_path.mkdir(parents=True, exist_ok=True)
        
        txt_path = reports_path / f"error_{report['error_metadata']['error_id']}.txt"
        
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(self._format_human_readable_report(report))
            
            self.logger.info("TXT error report written", path=str(txt_path))
        except Exception as e:
            self.logger.error("Failed to write TXT error report", error=str(e))
    
    def _format_human_readable_report(self, report: Dict[str, Any]) -> str:
        """Format comprehensive human-readable error report."""
        lines = []
        
        # Header
        lines.extend([
            "=" * 80,
            "STAGE 5 ERROR REPORT",
            "=" * 80,
            "",
            f"Error ID: {report['error_metadata']['error_id']}",
            f"Timestamp: {report['error_metadata']['timestamp']}",
            f"Stage: {report['error_metadata']['stage']}",
            f"Component: {report['error_metadata']['component']}",
            ""
        ])
        
        # Error Summary
        error_details = report['error_details']
        lines.extend([
            "ERROR SUMMARY",
            "-" * 40,
            f"Type: {error_details['error_type']}",
            f"Category: {error_details['error_category']}",
            f"Severity: {error_details['severity']}",
            f"Recoverable: {'Yes' if error_details['is_recoverable'] else 'No'}",
            f"Message: {error_details['error_message']}",
            ""
        ])
        
        # Technical Details
        tech_details = report['technical_details']
        lines.extend([
            "TECHNICAL DETAILS",
            "-" * 40,
            f"Location: {tech_details['error_location']['file']}:{tech_details['error_location']['line']}",
            f"Function: {tech_details['error_location']['function']}",
            f"Code Context: {tech_details['error_location']['code_context']}",
            ""
        ])
        
        # Diagnostic Information
        if report['diagnostic_information']:
            lines.extend([
                "DIAGNOSTIC INFORMATION",
                "-" * 40
            ])
            for key, value in report['diagnostic_information'].items():
                lines.append(f"{key}: {value}")
            lines.append("")
        
        # Suggested Fixes
        if report['suggested_fixes']:
            lines.extend([
                "SUGGESTED FIXES",
                "-" * 40
            ])
            for i, fix in enumerate(report['suggested_fixes'], 1):
                lines.extend([
                    f"{i}. [{fix['priority']}] {fix['description']}",
                    f"   Category: {fix['category']}",
                    f"   Action: {fix['action']}",
                    ""
                ])
        
        # Foundation Compliance
        compliance = report['foundation_compliance']
        lines.extend([
            "FOUNDATION COMPLIANCE",
            "-" * 40,
            f"Status: {compliance['compliance_status']}",
            f"Requires Review: {'Yes' if compliance['requires_foundation_review'] else 'No'}",
            ""
        ])
        
        if compliance['issues_detected']:
            lines.append("Compliance Issues:")
            for issue in compliance['issues_detected']:
                lines.extend([
                    f"  - {issue['issue']}: {issue['description']}",
                    f"    Foundation Section: {issue['foundation_section']}",
                    ""
                ])
        
        # Stack Trace
        lines.extend([
            "FULL STACK TRACE",
            "-" * 40,
            report['technical_details']['stack_trace'],
            "",
            "=" * 80
        ])
        
        return "\n".join(lines)


# Convenience function for quick error reporting
def report_stage5_error(error: Exception, 
                       context: Dict[str, Any], 
                       reports_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Convenience function for quick Stage 5 error reporting.
    
    Args:
        error: The exception that occurred
        context: Context information about the error
        reports_path: Optional path to write report files
        
    Returns:
        Complete error report dictionary
    """
    reporter = Stage5ErrorReporter()
    return reporter.generate_error_report(error, context, reports_path)

