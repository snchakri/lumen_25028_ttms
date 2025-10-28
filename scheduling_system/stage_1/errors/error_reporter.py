"""
Error reporter for generating formatted error reports.

Generates error reports in both JSON (machine-readable) and TXT (human-readable)
formats with complete error details and fix suggestions.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from ..models.validation_types import ErrorReport, ErrorSeverity, ErrorCategory


class ErrorReporter:
    """
    Generate comprehensive error reports.
    
    Per Theorem 12.4, provides:
    - Completeness: All errors reported
    - Precision: Accurate error details
    - Clarity: Actionable fix suggestions
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize error reporter.
        
        Args:
            output_dir: Directory for error reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_json_report(
        self,
        errors: List[ErrorReport],
        metadata: Dict[str, Any]
    ) -> Path:
        """
        Generate JSON error report.
        
        Args:
            errors: List of error reports
            metadata: Additional metadata (timestamps, file info, etc.)
        
        Returns:
            Path to generated JSON file
        """
        report = {
            "report_type": "validation_error_report",
            "generated_at": datetime.now().isoformat(),
            "metadata": metadata,
            "summary": self._generate_summary(errors),
            "errors": [error.to_dict() for error in errors],
        }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"error_report_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def generate_text_report(
        self,
        errors: List[ErrorReport],
        metadata: Dict[str, Any]
    ) -> Path:
        """
        Generate human-readable text error report.
        
        Args:
            errors: List of error reports
            metadata: Additional metadata
        
        Returns:
            Path to generated text file
        """
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("STAGE-1 INPUT VALIDATION ERROR REPORT")
        lines.append("TEAM LUMEN [93912]")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Metadata
        if metadata:
            lines.append("METADATA")
            lines.append("-" * 80)
            for key, value in metadata.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        # Summary
        summary = self._generate_summary(errors)
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"  Total Errors: {summary['total_errors']}")
        lines.append(f"  Critical:     {summary['critical']}")
        lines.append(f"  Errors:       {summary['errors']}")
        lines.append(f"  Warnings:     {summary['warnings']}")
        lines.append(f"  Info:         {summary['info']}")
        lines.append("")
        
        # Errors by category
        lines.append("ERRORS BY CATEGORY")
        lines.append("-" * 80)
        for category, count in summary['by_category'].items():
            lines.append(f"  {category:20s}: {count}")
        lines.append("")
        
        # Detailed errors
        lines.append("DETAILED ERRORS")
        lines.append("=" * 80)
        lines.append("")
        
        # Group by category
        errors_by_category = self._group_by_category(errors)
        
        for category, cat_errors in errors_by_category.items():
            lines.append(f"[{category}]")
            lines.append("-" * 80)
            
            for idx, error in enumerate(cat_errors, 1):
                lines.append(f"\nError #{idx}")
                lines.append(error.to_human_readable())
                lines.append("")
        
        # Fix suggestions summary
        lines.append("")
        lines.append("FIX SUGGESTIONS SUMMARY")
        lines.append("=" * 80)
        
        suggestions = self._collect_fix_suggestions(errors)
        if suggestions:
            for idx, suggestion in enumerate(suggestions, 1):
                lines.append(f"{idx}. {suggestion}")
        else:
            lines.append("No specific fix suggestions available.")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        # Write to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"error_report_{timestamp}.txt"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        return filepath
    
    def generate_reports(
        self,
        errors: List[ErrorReport],
        metadata: Dict[str, Any]
    ) -> tuple[Path, Path]:
        """
        Generate both JSON and text reports.
        
        Args:
            errors: List of error reports
            metadata: Additional metadata
        
        Returns:
            Tuple of (json_path, text_path)
        """
        json_path = self.generate_json_report(errors, metadata)
        text_path = self.generate_text_report(errors, metadata)
        return json_path, text_path
    
    def _generate_summary(self, errors: List[ErrorReport]) -> Dict[str, Any]:
        """Generate error summary statistics."""
        from collections import defaultdict
        
        by_severity = defaultdict(int)
        by_category = defaultdict(int)
        
        for error in errors:
            by_severity[error.severity.value] += 1
            by_category[error.category.value] += 1
        
        return {
            "total_errors": len(errors),
            "critical": by_severity[ErrorSeverity.CRITICAL.value],
            "errors": by_severity[ErrorSeverity.ERROR.value],
            "warnings": by_severity[ErrorSeverity.WARNING.value],
            "info": by_severity[ErrorSeverity.INFO.value],
            "by_category": dict(by_category),
        }
    
    def _group_by_category(
        self,
        errors: List[ErrorReport]
    ) -> Dict[str, List[ErrorReport]]:
        """Group errors by category."""
        from collections import defaultdict
        
        grouped = defaultdict(list)
        for error in errors:
            grouped[error.category.value].append(error)
        
        return dict(grouped)
    
    def _collect_fix_suggestions(self, errors: List[ErrorReport]) -> List[str]:
        """Collect unique fix suggestions from errors."""
        suggestions = set()
        for error in errors:
            if error.fix_suggestion:
                suggestions.add(error.fix_suggestion)
        return sorted(suggestions)
    
    def generate_quick_summary(self, errors: List[ErrorReport]) -> str:
        """Generate a quick one-line summary."""
        summary = self._generate_summary(errors)
        return (
            f"Validation failed with {summary['total_errors']} errors: "
            f"{summary['critical']} critical, {summary['errors']} errors, "
            f"{summary['warnings']} warnings"
        )





