"""
Log coordinator for managing both structured and console logging.

Provides unified logging interface that writes to both JSON log file
and rich console output simultaneously.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from .structured_logger import StructuredLogger, LogLevel
from .console_logger import ConsoleLogger


class LogCoordinator:
    """
    Coordinates structured JSON logging and console output.
    
    Provides single interface that logs to both systems simultaneously,
    ensuring complete traceability while maintaining user-friendly output.
    """
    
    def __init__(
        self,
        log_dir: Path,
        console_verbose: bool = False,
        json_min_level: LogLevel = LogLevel.DEBUG
    ):
        """
        Initialize log coordinator.
        
        Args:
            log_dir: Directory for log files
            console_verbose: Enable verbose console output
            json_min_level: Minimum level for JSON logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file path with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"validation_{timestamp}.jsonl"
        
        # Initialize both loggers
        self.structured_logger = StructuredLogger(log_file, min_level=json_min_level)
        self.console_logger = ConsoleLogger(verbose=console_verbose)
        
        self._active = False
    
    def start(self):
        """Start logging session."""
        self.structured_logger.open()
        self.console_logger.start_session()
        self._active = True
    
    def stop(self):
        """Stop logging session."""
        if self._active:
            self.structured_logger.close()
            self._active = False
    
    def info(self, message: str, **kwargs):
        """Log info message to both systems."""
        self.structured_logger.info(message, **kwargs)
        self.console_logger.info(message)
    
    def success(self, message: str, **kwargs):
        """Log success message."""
        self.structured_logger.info(message, **kwargs)
        self.console_logger.success(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message to both systems."""
        self.structured_logger.warning(message, **kwargs)
        self.console_logger.warning(message)
    
    def error(self, message: str, **kwargs):
        """Log error message to both systems."""
        self.structured_logger.error(message, **kwargs)
        self.console_logger.error(message)
    
    def critical(self, message: str, **kwargs):
        """Log critical message to both systems."""
        self.structured_logger.critical(message, **kwargs)
        self.console_logger.critical(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message to both systems."""
        self.structured_logger.debug(message, **kwargs)
        self.console_logger.debug(message)
    
    def start_progress(self, total_stages: int = 7):
        """Start progress tracking."""
        self.console_logger.start_progress(total_stages)
    
    def stop_progress(self):
        """Stop progress tracking."""
        self.console_logger.stop_progress()
    
    def start_stage(self, stage_number: int, stage_name: str, **kwargs):
        """Log stage start to both systems."""
        self.structured_logger.log_stage_start(stage_number, stage_name, **kwargs)
        self.console_logger.start_stage(stage_number, stage_name)
    
    def update_stage_progress(self, stage_number: int, advance: int = 1):
        """Update stage progress (console only)."""
        self.console_logger.update_stage_progress(stage_number, advance)
    
    def complete_stage(
        self,
        stage_number: int,
        stage_name: str,
        status: str,
        execution_time: float,
        error_count: int = 0,
        warning_count: int = 0,
        **kwargs
    ):
        """Log stage completion to both systems."""
        self.structured_logger.log_stage_end(
            stage_number, stage_name, status, execution_time,
            error_count=error_count,
            warning_count=warning_count,
            **kwargs
        )
        self.console_logger.complete_stage(
            stage_number, stage_name, status, error_count, warning_count
        )
    
    def log_file_processing(self, filename: str, action: str, **kwargs):
        """Log file processing operation."""
        self.structured_logger.log_file_processing(filename, action, **kwargs)
        if action == "validating":
            self.console_logger.debug(f"Processing file: {filename}")
    
    def log_validation_error(self, category: str, severity: str, message: str, **kwargs):
        """Log validation error to both systems."""
        self.structured_logger.log_validation_error(category, severity, message, **kwargs)
        # Don't spam console with every error (summary at end instead)
    
    def log_metrics(self, metrics_type: str, metrics: Dict[str, Any], **kwargs):
        """Log metrics to both systems."""
        self.structured_logger.log_metrics(metrics_type, metrics, **kwargs)
        if metrics_type == "quality_vector":
            self.console_logger.print_quality_vector(metrics)
        else:
            self.console_logger.print_metrics_table(metrics, title=f"{metrics_type} Metrics")
    
    def log_theorem_verification(
        self,
        theorem_id: str,
        verified: bool,
        details: Dict[str, Any],
        **kwargs
    ):
        """Log theorem verification to both systems."""
        self.structured_logger.log_theorem_verification(theorem_id, verified, details, **kwargs)
        details_str = details.get('summary', '') if details else ''
        self.console_logger.print_theorem_verification(theorem_id, verified, details_str)
    
    def print_file_list(self, title: str, files: List[str], style: str = "cyan"):
        """Print file list to console."""
        self.console_logger.print_file_list(title, files, style)
    
    def print_error_summary(self, errors: List, max_display: int = 10):
        """Print error summary to console."""
        self.console_logger.print_error_summary(errors, max_display)
    
    def end_session(self, status: str, total_errors: int, total_warnings: int):
        """End validation session."""
        self.structured_logger.info(
            f"Validation session ended: {status}",
            event="session_summary",
            status=status,
            total_errors=total_errors,
            total_warnings=total_warnings
        )
        self.console_logger.end_session(status, total_errors, total_warnings)
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type:
            self.critical(
                f"Unhandled exception: {exc_type.__name__}: {exc_val}",
                exception_type=exc_type.__name__,
                exception_message=str(exc_val)
            )
        self.stop()
        return False  # Don't suppress exceptions


# Global logger instance
_global_logger: Optional[LogCoordinator] = None


def initialize_logging(
    log_dir: Path,
    console_verbose: bool = False,
    json_min_level: LogLevel = LogLevel.DEBUG
) -> LogCoordinator:
    """
    Initialize global logger.
    
    Args:
        log_dir: Directory for log files
        console_verbose: Enable verbose console output
        json_min_level: Minimum level for JSON logging
    
    Returns:
        LogCoordinator instance
    """
    global _global_logger
    _global_logger = LogCoordinator(log_dir, console_verbose, json_min_level)
    _global_logger.start()
    return _global_logger


def get_logger() -> LogCoordinator:
    """
    Get global logger instance.
    
    Returns:
        LogCoordinator instance
    
    Raises:
        RuntimeError: If logging not initialized
    """
    if _global_logger is None:
        raise RuntimeError("Logging not initialized. Call initialize_logging() first.")
    return _global_logger


def finalize_logging():
    """Finalize and close global logger."""
    global _global_logger
    if _global_logger:
        _global_logger.stop()
        _global_logger = None


