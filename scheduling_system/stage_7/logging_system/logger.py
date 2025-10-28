


import numpy as np
import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict, field
from enum import Enum
import traceback

"""
Advanced Logging System for Stage 7
==================================

Implements rigorous logging with:
- Multi-level console output with color coding
- Structured JSON logging for analysis
- Mathematical validation logging
- Performance metrics tracking
- Error context capture
"""

def json_safe(obj):
    """Recursively convert numpy types and sets to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {json_safe(k): json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"




from dataclasses import dataclass, asdict

@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    message: str
    module: str
    function: Optional[str] = None
    line_number: Optional[int] = None
    session_id: Optional[str] = None
    threshold_id: Optional[str] = None
    metric_value: Optional[float] = None
    metric_bounds: Optional[Dict[str, float]] = None
    validation_result: Optional[bool] = None
    error_details: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    additional_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values and making JSON safe."""
        raw = {k: v for k, v in asdict(self).items() if v is not None}
        return json_safe(raw)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ConsoleFormatter(logging.Formatter):
    """Custom formatter for console output with color coding."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            record.levelname = colored_levelname
        
        # Format the message
        formatted = super().format(record)
        
        # Add additional context if available
        if hasattr(record, 'threshold_id'):
            formatted += f" | Threshold: {record.threshold_id}"
        if hasattr(record, 'metric_value'):
            formatted += f" | Value: {record.metric_value:.6f}"
        
        return formatted


class JSONFileHandler(logging.Handler):
    """Handler for writing JSON log files."""
    
    def __init__(self, filepath: Path):
        """Initialize JSON file handler."""
        super().__init__()
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file with empty array
        with open(self.filepath, 'w') as f:
            f.write('[\n')
        
        self.first_entry = True
    
    def emit(self, record):
        """Emit a log record as JSON."""
        try:
            # Create log entry
            entry = LogEntry(
                timestamp=datetime.now().isoformat(),
                level=record.levelname,
                message=record.getMessage(),
                module=record.module,
                function=record.funcName,
                line_number=record.lineno,
                session_id=getattr(record, 'session_id', None),
                threshold_id=getattr(record, 'threshold_id', None),
                metric_value=getattr(record, 'metric_value', None),
                metric_bounds=getattr(record, 'metric_bounds', None),
                validation_result=getattr(record, 'validation_result', None),
                error_details=getattr(record, 'error_details', None),
                performance_metrics=getattr(record, 'performance_metrics', None),
                additional_context=getattr(record, 'additional_context', None)
            )
            
            # Write to file
            with open(self.filepath, 'a') as f:
                if not self.first_entry:
                    f.write(',\n')
                else:
                    self.first_entry = False
                
                json.dump(entry.to_dict(), f, indent=2)
        
        except Exception:
            self.handleError(record)
    
    def close(self):
        """Close the JSON file properly."""
        try:
            with open(self.filepath, 'a') as f:
                f.write('\n]')
        except Exception:
            pass
        super().close()


class Stage7Logger:
    """
    Advanced logger for Stage 7 validation.
    
    Features:
    - Console and file logging
    - JSON structured logging
    - Mathematical validation logging
    - Performance tracking
    - Error context capture
    """
    
    def __init__(
        self,
        session_id: str,
        log_dir: Path,
        log_level: str = "INFO",
        enable_console: bool = True,
        enable_json: bool = True
    ):
        """
        Initialize Stage 7 logger.
        
        Args:
            session_id: Unique session identifier
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_console: Enable console output
            enable_json: Enable JSON file logging
        """
        self.session_id = session_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(f"stage7.{session_id}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.propagate = False
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_formatter = ConsoleFormatter(
                '%(asctime)s | %(levelname)s | %(module)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # JSON file handler
        if enable_json:
            json_filepath = self.log_dir / f"{session_id}_validation.json"
            json_handler = JSONFileHandler(json_filepath)
            json_handler.setLevel(logging.DEBUG)  # Capture all levels in JSON
            self.logger.addHandler(json_handler)
            self.json_filepath = json_filepath
        
        # Text file handler for human-readable logs
        text_filepath = self.log_dir / f"{session_id}_validation.log"
        file_handler = logging.FileHandler(text_filepath)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(module)-15s | %(funcName)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        self.text_filepath = text_filepath
        
        # Performance tracking
        self.start_time = datetime.now()
        self.performance_metrics: Dict[str, Any] = {
            'start_time': self.start_time.isoformat(),
            'threshold_validations': {},
            'errors': [],
            'warnings': []
        }
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)
        self.performance_metrics['warnings'].append({
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'context': kwargs
        })
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)
        self.performance_metrics['errors'].append({
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'context': kwargs
        })
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)
        self.performance_metrics['errors'].append({
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'context': kwargs,
            'severity': 'CRITICAL'
        })
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method."""
        # Add session_id to all log records
        kwargs['session_id'] = self.session_id
        
        # Create log record with extra fields
        self.logger.log(level, message, extra=kwargs)
    
    def log_threshold_validation(
        self,
        threshold_id: str,
        metric_value: float,
        lower_bound: float,
        upper_bound: float,
        passed: bool,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """
        Log threshold validation result.
        
        Args:
            threshold_id: Threshold identifier (e.g., 'tau1')
            metric_value: Computed metric value
            lower_bound: Lower threshold bound
            upper_bound: Upper threshold bound
            passed: Whether validation passed
            additional_context: Additional context information
        """
        level = logging.INFO if passed else logging.ERROR
        
        message = (
            f"Threshold {threshold_id}: "
            f"value={metric_value:.6f}, "
            f"bounds=[{lower_bound:.6f}, {upper_bound:.6f}], "
            f"status={'PASS' if passed else 'FAIL'}"
        )
        
        self._log(
            level,
            message,
            threshold_id=threshold_id,
            metric_value=metric_value,
            metric_bounds={'lower': lower_bound, 'upper': upper_bound},
            validation_result=passed,
            additional_context=additional_context
        )
        
        # Track in performance metrics
        self.performance_metrics['threshold_validations'][threshold_id] = {
            'value': metric_value,
            'bounds': {'lower': lower_bound, 'upper': upper_bound},
            'passed': passed,
            'timestamp': datetime.now().isoformat()
        }
    
    def log_mathematical_validation(
        self,
        theorem_name: str,
        validation_type: str,
        result: bool,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log mathematical theorem validation.
        
        Args:
            theorem_name: Name of theorem being validated
            validation_type: Type of validation (e.g., 'proof', 'bound_check')
            result: Validation result
            details: Additional validation details
        """
        level = logging.INFO if result else logging.ERROR
        
        message = f"Mathematical validation: {theorem_name} ({validation_type}) - {'PASS' if result else 'FAIL'}"
        
        self._log(
            level,
            message,
            additional_context={
                'theorem_name': theorem_name,
                'validation_type': validation_type,
                'result': result,
                'details': details or {}
            }
        )
    
    def log_performance(self, operation: str, duration_ms: float, details: Optional[Dict[str, Any]] = None):
        """
        Log performance metrics.
        
        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            details: Additional performance details
        """
        message = f"Performance: {operation} completed in {duration_ms:.2f}ms"
        
        self._log(
            logging.DEBUG,
            message,
            performance_metrics={
                'operation': operation,
                'duration_ms': duration_ms,
                'details': details or {}
            }
        )
    
    def log_exception(self, exception: Exception, context: Optional[str] = None):
        """
        Log exception with full traceback.
        
        Args:
            exception: Exception to log
            context: Additional context about where exception occurred
        """
        tb = traceback.format_exc()
        
        message = f"Exception occurred: {str(exception)}"
        if context:
            message = f"{context} - {message}"
        
        self._log(
            logging.ERROR,
            message,
            error_details={
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'traceback': tb,
                'context': context
            }
        )
    
    def finalize(self) -> Dict[str, Any]:
        """
        Finalize logging and return performance summary.
        
        Returns:
            Performance metrics dictionary
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self.performance_metrics['end_time'] = end_time.isoformat()
        self.performance_metrics['total_duration_seconds'] = duration
        
        # Write performance summary
        summary_file = self.log_dir / f"{self.session_id}_summary.json"
        from scheduling_engine_localized.stage_7.logging_system.logger import json_safe
        with open(summary_file, 'w') as f:
            json.dump(json_safe(self.performance_metrics), f, indent=2)
        
        self.info(f"Validation completed in {duration:.2f} seconds")
        self.info(f"Performance summary written to: {summary_file}")
        
        # Close JSON handler properly
        for handler in self.logger.handlers:
            if isinstance(handler, JSONFileHandler):
                handler.close()
        
        return self.performance_metrics
    
    def get_logger(self) -> logging.Logger:
        """Get underlying logger instance."""
        return self.logger


def create_logger(
    session_id: str,
    log_dir: Path,
    log_level: str = "INFO",
    enable_console: bool = True,
    enable_json: bool = True
) -> Stage7Logger:
    """
    Factory function to create Stage7Logger.
    
    Args:
        session_id: Unique session identifier
        log_dir: Directory for log files
        log_level: Logging level
        enable_console: Enable console output
        enable_json: Enable JSON logging
    
    Returns:
        Configured Stage7Logger instance
    """
    return Stage7Logger(
        session_id=session_id,
        log_dir=log_dir,
        log_level=log_level,
        enable_console=enable_console,
        enable_json=enable_json
    )
