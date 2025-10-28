"""
Error Reporter - JSON/TXT Error Reports

Generates comprehensive error reports with raw data and human-readable fixes.

Requirements: JSON + TXT error reports

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ErrorReport:
    """Comprehensive error report."""
    
    timestamp: str
    error_type: str
    error_message: str
    traceback: str
    context: Dict[str, Any] = field(default_factory=dict)
    suggested_fixes: List[str] = field(default_factory=list)
    severity: str = "ERROR"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'traceback': self.traceback,
            'context': self.context,
            'suggested_fixes': self.suggested_fixes,
            'severity': self.severity
        }
    
    def to_text(self) -> str:
        """Convert to human-readable text."""
        text = f"ERROR REPORT\n"
        text += f"{'='*80}\n\n"
        text += f"Timestamp: {self.timestamp}\n"
        text += f"Error Type: {self.error_type}\n"
        text += f"Severity: {self.severity}\n\n"
        text += f"Error Message:\n{self.error_message}\n\n"
        
        if self.context:
            text += f"Context:\n"
            for key, value in self.context.items():
                text += f"  - {key}: {value}\n"
            text += "\n"
        
        if self.suggested_fixes:
            text += f"Suggested Fixes:\n"
            for i, fix in enumerate(self.suggested_fixes, 1):
                text += f"  {i}. {fix}\n"
            text += "\n"
        
        if self.traceback:
            text += f"Traceback:\n{self.traceback}\n"
        
        text += f"{'='*80}\n"
        
        return text


class ErrorReporter:
    """
    Generates comprehensive error reports with JSON and TXT formats.
    
    Requirements: JSON + TXT error reports
    """
    
    def __init__(self, error_report_path: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize error reporter.
        
        Args:
            error_report_path: Path for error reports
            logger: Logger instance
        """
        self.error_report_path = Path(error_report_path)
        self.error_report_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
    
    def create_error_report(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        suggested_fixes: Optional[List[str]] = None
    ) -> ErrorReport:
        """
        Create comprehensive error report.
        
        Args:
            error: Exception that occurred
            context: Context information
            suggested_fixes: List of suggested fixes
        
        Returns:
            ErrorReport
        """
        self.logger.error(f"Creating error report for: {type(error).__name__}")
        
        # Generate suggested fixes if not provided
        if suggested_fixes is None:
            suggested_fixes = self._generate_suggested_fixes(error, context)
        
        # Create error report
        report = ErrorReport(
            timestamp=datetime.now().isoformat(),
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            context=context or {},
            suggested_fixes=suggested_fixes,
            severity=self._determine_severity(error)
        )
        
        return report
    
    def save_error_report(self, report: ErrorReport) -> tuple[Path, Path]:
        """
        Save error report in both JSON and TXT formats.
        
        Args:
            report: ErrorReport to save
        
        Returns:
            (json_path, txt_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_file = self.error_report_path / f'error_report_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        self.logger.info(f"Saved JSON error report: {json_file}")
        
        # Save TXT report
        txt_file = self.error_report_path / f'error_report_{timestamp}.txt'
        with open(txt_file, 'w') as f:
            f.write(report.to_text())
        
        self.logger.info(f"Saved TXT error report: {txt_file}")
        
        return json_file, txt_file
    
    def _generate_suggested_fixes(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate suggested fixes based on error type."""
        fixes = []
        
        error_type = type(error).__name__
        
        if "FileNotFoundError" in error_type:
            fixes.append("Check that all required input files exist")
            fixes.append("Verify file paths are correct")
            fixes.append("Ensure Stage 3 output directory is properly generated")
        
        elif "KeyError" in error_type:
            fixes.append("Verify entity mappings are complete")
            fixes.append("Check that all required entities exist in input data")
            fixes.append("Ensure bijective mappings are properly created")
        
        elif "ValueError" in error_type:
            fixes.append("Verify input data format and types")
            fixes.append("Check for missing or invalid values")
            fixes.append("Ensure numerical values are within valid ranges")
        
        elif "MemoryError" in error_type:
            fixes.append("Increase available memory")
            fixes.append("Consider processing data in smaller batches")
            fixes.append("Optimize data structures for memory efficiency")
        
        elif "TimeoutError" in error_type:
            fixes.append("Increase solver time limit")
            fixes.append("Try a different solver algorithm")
            fixes.append("Simplify problem constraints if possible")
        
        else:
            fixes.append("Review error message and context")
            fixes.append("Check logs for additional details")
            fixes.append("Verify all dependencies are installed")
        
        return fixes
    
    def _determine_severity(self, error: Exception) -> str:
        """Determine error severity."""
        error_type = type(error).__name__
        
        if "MemoryError" in error_type or "TimeoutError" in error_type:
            return "CRITICAL"
        elif "FileNotFoundError" in error_type or "KeyError" in error_type:
            return "ERROR"
        elif "ValueError" in error_type or "TypeError" in error_type:
            return "WARNING"
        else:
            return "ERROR"



