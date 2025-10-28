"""
Error Reporter Module

Generates comprehensive error reports with:
- Raw error data
- Human-readable descriptions
- Suggested fixes
- Context information

Output formats: JSON and TXT
"""

import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from ..config import PyGMOConfig
from ..logging_system.logger import StructuredLogger


class ErrorReporter:
    """
    Generates comprehensive error reports for debugging and recovery.
    """
    
    def __init__(self, config: PyGMOConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.log_dir = config.log_dir
        
        self.logger.info("ErrorReporter initialized successfully.")
    
    def generate_report(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generates a comprehensive error report.
        
        Args:
            exception: The exception that occurred
            context: Additional context information
        
        Returns:
            Dictionary containing error report data
        """
        self.logger.error(f"Generating error report for: {type(exception).__name__}")
        
        # Extract error information
        error_type = type(exception).__name__
        error_message = str(exception)
        error_traceback = traceback.format_exc()
        
        # Generate report
        # Ensure context is JSON-serializable (convert Paths, etc.)
        safe_context = {}
        if context:
            for k, v in context.items():
                try:
                    json.dumps(v)
                    safe_context[k] = v
                except Exception:
                    # Fallback to string representation
                    safe_context[k] = str(v)

        report = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'error_traceback': error_traceback,
            'context': safe_context,
            'human_readable': self._generate_human_readable(exception, context),
            'suggested_fixes': self._suggest_fixes(exception, context),
            'severity': self._determine_severity(exception),
            'recoverable': self._is_recoverable(exception)
        }
        
        # Write reports to files
        self._write_json_report(report)
        self._write_txt_report(report)
        
        return report
    
    def _generate_human_readable(self, exception: Exception, context: Optional[Dict[str, Any]]) -> str:
        """
        Generates a human-readable description of the error.
        """
        error_type = type(exception).__name__
        error_message = str(exception)
        
        description = f"An error of type '{error_type}' occurred during PyGMO solver execution.\n\n"
        description += f"Error Message: {error_message}\n\n"
        
        if context:
            description += "Context Information:\n"
            for key, value in context.items():
                description += f"  - {key}: {value}\n"
        
        return description
    
    def _suggest_fixes(self, exception: Exception, context: Optional[Dict[str, Any]]) -> list[str]:
        """
        Suggests potential fixes based on the error type and context.
        """
        error_type = type(exception).__name__
        suggestions = []
        
        # Common error patterns and fixes
        if error_type == 'FileNotFoundError':
            suggestions.append("Check that all input files from Stage 3 are present in the input directory.")
            suggestions.append("Verify that the input directory path is correct.")
            suggestions.append("Ensure Stage 3 has completed successfully before running Stage 6.4.")
        
        elif error_type == 'ValueError':
            suggestions.append("Check that input data is in the correct format and range.")
            suggestions.append("Verify that all UUIDs are valid and properly formatted.")
            suggestions.append("Ensure that problem dimensions (courses, faculty, rooms, etc.) are non-zero.")
        
        elif error_type == 'RuntimeError':
            suggestions.append("Check system resources (memory, CPU).")
            suggestions.append("Verify that PyGMO is installed correctly.")
            suggestions.append("Try reducing population size or number of islands if memory is limited.")
        
        elif error_type == 'KeyError':
            suggestions.append("Check that all required data fields are present in Stage 3 outputs.")
            suggestions.append("Verify the schema of input Parquet/JSON files.")
            suggestions.append("Ensure that entity relationships are properly defined.")
        
        elif error_type == 'TypeError':
            suggestions.append("Check data type compatibility in input files.")
            suggestions.append("Verify that numeric fields contain valid numbers.")
            suggestions.append("Ensure that UUID fields contain valid UUID strings.")
        
        elif error_type == 'MemoryError':
            suggestions.append("Reduce population size in configuration.")
            suggestions.append("Reduce number of islands in archipelago.")
            suggestions.append("Consider using a machine with more RAM.")
            suggestions.append("Enable checkpointing to save intermediate results.")
        
        else:
            suggestions.append("Check the error traceback for specific details.")
            suggestions.append("Review the logs for additional context.")
            suggestions.append("Verify that all dependencies are installed correctly.")
        
        # Context-specific suggestions
        if context:
            phase = context.get('phase', '')
            if phase == 'input_loading':
                suggestions.append("Verify Stage 3 outputs are complete and uncorrupted.")
            elif phase == 'problem_initialization':
                suggestions.append("Check that problem dimensions are valid and non-zero.")
            elif phase == 'optimization':
                suggestions.append("Try using a different algorithm or adjusting hyperparameters.")
        
        return suggestions
    
    def _determine_severity(self, exception: Exception) -> str:
        """
        Determines the severity level of the error.
        """
        error_type = type(exception).__name__
        
        critical_errors = ['MemoryError', 'SystemError', 'OSError']
        high_errors = ['RuntimeError', 'ValueError', 'FileNotFoundError']
        medium_errors = ['KeyError', 'TypeError', 'AttributeError']
        
        if error_type in critical_errors:
            return 'CRITICAL'
        elif error_type in high_errors:
            return 'HIGH'
        elif error_type in medium_errors:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _is_recoverable(self, exception: Exception) -> bool:
        """
        Determines if the error is recoverable through fallback mechanisms.
        """
        error_type = type(exception).__name__
        
        # Unrecoverable errors
        unrecoverable = ['MemoryError', 'SystemError', 'KeyboardInterrupt']
        
        return error_type not in unrecoverable
    
    def _write_json_report(self, report: Dict[str, Any]):
        """
        Writes the error report to a JSON file.
        """
        output_path = self.log_dir / self.config.error_report_file_name
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Error report (JSON) written to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to write JSON error report: {e}")
    
    def _write_txt_report(self, report: Dict[str, Any]):
        """
        Writes the error report to a human-readable TXT file.
        """
        output_path = self.log_dir / self.config.error_report_txt_name
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("PYGMO SOLVER FAMILY - ERROR REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Timestamp: {report['timestamp']}\n")
                f.write(f"Error Type: {report['error_type']}\n")
                f.write(f"Severity: {report['severity']}\n")
                f.write(f"Recoverable: {report['recoverable']}\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("HUMAN-READABLE DESCRIPTION\n")
                f.write("=" * 80 + "\n\n")
                f.write(report['human_readable'] + "\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("SUGGESTED FIXES\n")
                f.write("=" * 80 + "\n\n")
                for i, fix in enumerate(report['suggested_fixes'], 1):
                    f.write(f"{i}. {fix}\n")
                f.write("\n")
                
                f.write("=" * 80 + "\n")
                f.write("RAW ERROR DATA\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Error Message: {report['error_message']}\n\n")
                f.write("Traceback:\n")
                f.write(report['error_traceback'] + "\n\n")
                
                if report['context']:
                    f.write("=" * 80 + "\n")
                    f.write("CONTEXT INFORMATION\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(json.dumps(report['context'], indent=2) + "\n")
            
            self.logger.info(f"Error report (TXT) written to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to write TXT error report: {e}")


