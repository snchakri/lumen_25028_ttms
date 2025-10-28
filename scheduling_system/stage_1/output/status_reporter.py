"""
Status reporter for validation results.

Provides status summary with abort signal for calling module.
"""

from typing import Dict, Any
from ..models.validation_types import ValidationResult


class StatusReporter:
    """
    Generate status report with abort signal.
    
    Provides calling module with clear status and decision on whether
    to abort the pipeline based on validation results.
    """
    
    def report(self, result: ValidationResult) -> Dict[str, Any]:
        """
        Generate status report.
        
        Args:
            result: ValidationResult from pipeline
        
        Returns:
            Dictionary with status, abort signal, and summary
        """
        return {
            "status": result.overall_status.value,
            "should_abort": result.should_abort(),
            "total_errors": result.total_errors,
            "total_warnings": result.total_warnings,
            "summary": self._generate_summary(result)
        }
    
    def _generate_summary(self, result: ValidationResult) -> str:
        """Generate human-readable summary."""
        summary_parts = [
            f"Validation Status: {result.overall_status.value}",
            f"Total Errors: {result.total_errors}",
            f"Total Warnings: {result.total_warnings}"
        ]
        
        if hasattr(result, 'validation_metadata'):
            metadata = result.validation_metadata
            if 'total_files' in metadata:
                summary_parts.append(f"Files Processed: {metadata['total_files']}")
            if 'total_records' in metadata:
                summary_parts.append(f"Records Processed: {metadata['total_records']}")
            if 'execution_time' in metadata:
                summary_parts.append(f"Execution Time: {metadata['execution_time']:.2f}s")
        
        return " | ".join(summary_parts)

