"""
Structured JSON logging system for Stage-1 validation.

Logs all validation operations in JSON Lines format for machine processing
and analysis. Includes complete trace of validation pipeline execution.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    """Log levels matching Python logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredLogger:
    """
    Structured JSON logger outputting JSON Lines format.
    
    Each log entry is a complete JSON object on a single line,
    enabling efficient streaming processing and analysis.
    """
    
    def __init__(self, log_file_path: Path, min_level: LogLevel = LogLevel.INFO):
        """
        Initialize structured logger.
        
        Args:
            log_file_path: Path to JSON log file
            min_level: Minimum log level to record
        """
        self.log_file_path = log_file_path
        self.min_level = min_level
        self.log_file = None
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._entry_count = 0
    
    def open(self):
        """Open log file for writing."""
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
        
        # Write session header
        self._write_entry({
            "session_id": self._session_id,
            "event": "session_start",
            "timestamp": datetime.now().isoformat(),
            "log_file": str(self.log_file_path),
        })
    
    def close(self):
        """Close log file."""
        if self.log_file:
            self._write_entry({
                "session_id": self._session_id,
                "event": "session_end",
                "timestamp": datetime.now().isoformat(),
                "total_entries": self._entry_count,
            })
            self.log_file.close()
            self.log_file = None
    
    def _write_entry(self, entry: Dict[str, Any]):
        """Write a single JSON log entry."""
        if self.log_file:
            json.dump(entry, self.log_file, ensure_ascii=False, default=str)
            self.log_file.write('\n')
            self.log_file.flush()
            self._entry_count += 1
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if log level should be recorded."""
        level_order = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 4,
        }
        return level_order[level] >= level_order[self.min_level]
    
    def log(
        self,
        level: LogLevel,
        message: str,
        **kwargs
    ):
        """
        Log a structured entry.
        
        Args:
            level: Log level
            message: Log message
            **kwargs: Additional structured data
        """
        if not self._should_log(level):
            return
        
        entry = {
            "session_id": self._session_id,
            "timestamp": datetime.now().isoformat(),
            "level": level.value,
            "message": message,
        }
        entry.update(kwargs)
        
        self._write_entry(entry)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.log(LogLevel.CRITICAL, message, **kwargs)
    
    def log_stage_start(self, stage_number: int, stage_name: str, **kwargs):
        """Log validation stage start."""
        self.info(
            f"Stage {stage_number} started: {stage_name}",
            event="stage_start",
            stage_number=stage_number,
            stage_name=stage_name,
            **kwargs
        )
    
    def log_stage_end(
        self,
        stage_number: int,
        stage_name: str,
        status: str,
        execution_time: float,
        **kwargs
    ):
        """Log validation stage completion."""
        self.info(
            f"Stage {stage_number} completed: {stage_name} ({status})",
            event="stage_end",
            stage_number=stage_number,
            stage_name=stage_name,
            status=status,
            execution_time_seconds=execution_time,
            **kwargs
        )
    
    def log_file_processing(self, filename: str, action: str, **kwargs):
        """Log file processing operation."""
        self.info(
            f"File {action}: {filename}",
            event="file_processing",
            filename=filename,
            action=action,
            **kwargs
        )
    
    def log_validation_error(
        self,
        category: str,
        severity: str,
        message: str,
        **kwargs
    ):
        """Log validation error."""
        level = LogLevel.ERROR if severity in ["CRITICAL", "ERROR"] else LogLevel.WARNING
        self.log(
            level,
            message,
            event="validation_error",
            error_category=category,
            error_severity=severity,
            **kwargs
        )
    
    def log_metrics(self, metrics_type: str, metrics: Dict[str, Any], **kwargs):
        """Log metrics data."""
        self.info(
            f"Metrics: {metrics_type}",
            event="metrics",
            metrics_type=metrics_type,
            metrics=metrics,
            **kwargs
        )
    
    def log_theorem_verification(
        self,
        theorem_id: str,
        verified: bool,
        details: Dict[str, Any],
        **kwargs
    ):
        """Log theorem verification result."""
        self.info(
            f"Theorem {theorem_id} verification: {'PASSED' if verified else 'FAILED'}",
            event="theorem_verification",
            theorem_id=theorem_id,
            verified=verified,
            details=details,
            **kwargs
        )
    
    def log_complexity_measurement(
        self,
        operation: str,
        data_size: int,
        time_seconds: float,
        complexity_class: str,
        **kwargs
    ):
        """Log complexity measurement."""
        self.debug(
            f"Complexity: {operation}",
            event="complexity_measurement",
            operation=operation,
            data_size=data_size,
            time_seconds=time_seconds,
            complexity_class=complexity_class,
            **kwargs
        )
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type:
            self.critical(
                f"Exception occurred: {exc_type.__name__}",
                event="exception",
                exception_type=exc_type.__name__,
                exception_message=str(exc_val),
            )
        self.close()
        return False  # Don't suppress exceptions


