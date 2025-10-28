"""
Comprehensive report generator.

Generates both machine-readable (JSON) and human-readable (TXT) reports
with complete validation results, error details, and fix suggestions.
"""

import json
from pathlib import Path
from typing import List
from ..models.validation_types import ValidationResult, ErrorReport


class ReportGenerator:
    """
    Generate comprehensive validation reports.
    
    Creates both JSON (machine-readable) and TXT (human-readable) formats
    with complete error details, fix suggestions, and actionable recommendations.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self, result: ValidationResult):
        """
        Generate comprehensive validation reports.
        
        Args:
            result: ValidationResult from pipeline
        """
        # Generate JSON report (machine-readable)
        self._generate_json_report(result)
        
        # Generate TXT report (human-readable)
        self._generate_txt_report(result)
    
    def _generate_json_report(self, result: ValidationResult):
        """Generate machine-readable JSON report."""
        filepath = self.output_dir / "validation_report.json"
        
        report = {
            "overall_status": result.overall_status.value,
            "should_abort": result.should_abort(),
            "total_errors": result.total_errors,
            "total_warnings": result.total_warnings,
            "validation_metadata": result.validation_metadata,
            "stage_results": [
                {
                    "stage_number": sr.stage_number,
                    "stage_name": sr.stage_name,
                    "status": sr.status.value,
                    "execution_time": sr.execution_time_seconds,
                    "records_processed": sr.records_processed,
                    "errors": [self._error_to_dict(e) for e in sr.errors],
                    "warnings": [self._error_to_dict(w) for w in sr.warnings]
                }
                for sr in result.stage_results
            ],
            "errors_by_category": dict(result.errors_by_category),
            "errors_by_severity": dict(result.errors_by_severity)
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    def _generate_txt_report(self, result: ValidationResult):
        """Generate human-readable TXT report."""
        filepath = self.output_dir / "validation_report.txt"
        
        lines = [
            "=" * 80,
            "STAGE-1 INPUT VALIDATION REPORT",
            "=" * 80,
            "",
            f"Overall Status: {result.overall_status.value}",
            f"Should Abort Pipeline: {result.should_abort()}",
            f"Total Errors: {result.total_errors}",
            f"Total Warnings: {result.total_warnings}",
            ""
        ]
        
        # Add validation metadata
        if hasattr(result, 'validation_metadata'):
            lines.append("Validation Metadata:")
            for key, value in result.validation_metadata.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        # Add stage results
        lines.append("=" * 80)
        lines.append("STAGE RESULTS")
        lines.append("=" * 80)
        lines.append("")
        
        for sr in result.stage_results:
            lines.append(f"Stage {sr.stage_number}: {sr.stage_name}")
            lines.append(f"  Status: {sr.status.value}")
            lines.append(f"  Execution Time: {sr.execution_time_seconds:.4f}s")
            lines.append(f"  Records Processed: {sr.records_processed}")
            lines.append(f"  Errors: {len(sr.errors)}")
            lines.append(f"  Warnings: {len(sr.warnings)}")
            lines.append("")
        
        # Add error details
        if result.total_errors > 0:
            lines.append("=" * 80)
            lines.append("ERROR DETAILS")
            lines.append("=" * 80)
            lines.append("")
            
            for i, sr in enumerate(result.stage_results, 1):
                if sr.errors:
                    lines.append(f"Stage {sr.stage_number}: {sr.stage_name}")
                    lines.append("-" * 80)
                    
                    for j, error in enumerate(sr.errors, 1):
                        lines.append(f"\nError {j}:")
                        lines.append(f"  Category: {error.category.value}")
                        lines.append(f"  Severity: {error.severity.value}")
                        lines.append(f"  Message: {error.message}")
                        lines.append(f"  File: {error.file_path}")
                        if error.line_number:
                            lines.append(f"  Line: {error.line_number}")
                        if error.field_name:
                            lines.append(f"  Field: {error.field_name}")
                        if error.expected_value:
                            lines.append(f"  Expected: {error.expected_value}")
                        if error.actual_value:
                            lines.append(f"  Actual: {error.actual_value}")
                        if error.fix_suggestion:
                            lines.append(f"  Fix: {error.fix_suggestion}")
                    
                    lines.append("")
        
        # Add warning details
        if result.total_warnings > 0:
            lines.append("=" * 80)
            lines.append("WARNING DETAILS")
            lines.append("=" * 80)
            lines.append("")
            
            for i, sr in enumerate(result.stage_results, 1):
                if sr.warnings:
                    lines.append(f"Stage {sr.stage_number}: {sr.stage_name}")
                    lines.append("-" * 80)
                    
                    for j, warning in enumerate(sr.warnings, 1):
                        lines.append(f"\nWarning {j}:")
                        lines.append(f"  Message: {warning.message}")
                        lines.append(f"  File: {warning.file_path}")
                        if warning.fix_suggestion:
                            lines.append(f"  Suggestion: {warning.fix_suggestion}")
                    
                    lines.append("")
        
        # Add summary and recommendations
        lines.append("=" * 80)
        lines.append("RECOMMENDATIONS")
        lines.append("=" * 80)
        lines.append("")
        
        if result.should_abort():
            lines.append("[FAIL] VALIDATION FAILED - Pipeline should be aborted")
            lines.append("")
            lines.append("Critical errors detected that must be fixed before proceeding.")
            lines.append("Please review the error details above and fix all issues.")
        elif result.total_warnings > 0:
            lines.append("[WARN] VALIDATION PASSED WITH WARNINGS")
            lines.append("")
            lines.append("Validation passed but warnings were detected.")
            lines.append("Review warnings above and consider addressing them for better data quality.")
        else:
            lines.append("[PASS] VALIDATION PASSED")
            lines.append("")
            lines.append("All validation checks passed successfully.")
            lines.append("Data is ready for Stage-2 processing.")
        
        lines.append("")
        lines.append("=" * 80)
        
        # Write report
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
    
    def _error_to_dict(self, error: ErrorReport) -> dict:
        """Convert ErrorReport to dictionary for JSON serialization."""
        return {
            "category": error.category.value,
            "severity": error.severity.value,
            "message": error.message,
            "file_path": error.file_path,
            "line_number": error.line_number,
            "field_name": error.field_name,
            "expected_value": error.expected_value,
            "actual_value": error.actual_value,
            "fix_suggestion": error.fix_suggestion
        }

