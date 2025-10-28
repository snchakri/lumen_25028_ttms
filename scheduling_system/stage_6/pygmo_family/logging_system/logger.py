"""
Structured Logging System for PyGMO Solver Family

This module implements comprehensive logging with both console and JSON file output,
providing detailed debugging and monitoring capabilities.

Theoretical Foundation: Section 18 - Audit Requirements
"""

import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List
from enum import Enum
import traceback
from colorama import Fore, Style, init as colorama_init

# Initialize colorama for cross-platform colored output
colorama_init(autoreset=True)


class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredLogger:
    """
    Structured logging system with console and JSON file output.
    
    Features:
    - Colored console output for readability
    - JSON file logging for machine parsing
    - Hierarchical log structure
    - Performance metrics tracking
    - Error context capture
    """
    
    def __init__(
        self,
        name: str,
        log_dir: Path,
        log_level: str = "INFO",
        log_to_console: bool = True,
        log_to_file: bool = True,
        json_format: bool = True
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name (typically module name)
            log_dir: Directory for log files
            log_level: Minimum log level
            log_to_console: Enable console logging
            log_to_file: Enable file logging
            json_format: Use JSON format for file logs
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_level = getattr(logging, log_level.upper())
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file
        self.json_format = json_format
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Setup handlers
        if self.log_to_console:
            self._setup_console_handler()
        
        if self.log_to_file:
            self._setup_file_handler()
        
        # Log buffer for JSON output
        self.log_buffer: List[Dict[str, Any]] = []
        
        # Session metadata
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
    
    def _setup_console_handler(self):
        """Setup colored console handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        
        # Custom formatter with colors
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self):
        """Setup file handler"""
        if self.json_format:
            # JSON logs will be written in flush() method
            pass
        else:
            # Standard text log file
            log_file = self.log_dir / f"{self.name}_{self.session_id}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.log_level)
            
            file_formatter = logging.Formatter(
                fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def _create_log_entry(
        self,
        level: str,
        message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Create structured log entry"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "logger": self.name,
            "level": level,
            "message": message,
        }
        
        # Add additional context
        if kwargs:
            entry["context"] = kwargs
        
        return entry
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message)
        if self.json_format:
            self.log_buffer.append(self._create_log_entry("DEBUG", message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message)
        if self.json_format:
            self.log_buffer.append(self._create_log_entry("INFO", message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message)
        if self.json_format:
            self.log_buffer.append(self._create_log_entry("WARNING", message, **kwargs))
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception"""
        self.logger.error(message)
        
        if self.json_format:
            entry = self._create_log_entry("ERROR", message, **kwargs)
            if exception:
                entry["exception"] = {
                    "type": type(exception).__name__,
                    "message": str(exception),
                    "traceback": traceback.format_exc()
                }
            self.log_buffer.append(entry)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message"""
        self.logger.critical(message)
        
        if self.json_format:
            entry = self._create_log_entry("CRITICAL", message, **kwargs)
            if exception:
                entry["exception"] = {
                    "type": type(exception).__name__,
                    "message": str(exception),
                    "traceback": traceback.format_exc()
                }
            self.log_buffer.append(entry)
    
    def log_stage(self, stage_name: str, status: str, **metrics):
        """Log stage completion with metrics"""
        message = f"Stage: {stage_name} | Status: {status}"
        self.info(message, stage=stage_name, status=status, metrics=metrics)
    
    def log_performance(self, operation: str, duration_seconds: float, **metrics):
        """Log performance metrics"""
        message = f"Performance: {operation} completed in {duration_seconds:.2f}s"
        self.info(message, operation=operation, duration=duration_seconds, **metrics)
    
    def log_validation(self, validation_type: str, result: bool, **details):
        """Log validation results"""
        status = "PASS" if result else "FAIL"
        message = f"Validation: {validation_type} | Status: {status}"
        level = self.info if result else self.warning
        level(message, validation=validation_type, result=result, **details)
    
    def log_optimization_progress(
        self,
        generation: int,
        hypervolume: float,
        best_fitness: List[float],
        **metrics
    ):
        """Log optimization progress"""
        message = f"Generation {generation} | HV: {hypervolume:.6f} | Best: {best_fitness}"
        self.debug(
            message,
            generation=generation,
            hypervolume=hypervolume,
            best_fitness=best_fitness,
            **metrics
        )
    
    def flush(self):
        """Flush log buffer to JSON file"""
        if self.json_format and self.log_buffer:
            json_log_file = self.log_dir / f"{self.name}_{self.session_id}.json"
            with open(json_log_file, 'w') as f:
                json.dump({
                    "session_id": self.session_id,
                    "logger": self.name,
                    "start_time": self.start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "log_entries": self.log_buffer
                }, f, indent=2)
            
            self.info(f"Flushed {len(self.log_buffer)} log entries to {json_log_file}")
    
    def close(self):
        """Close logger and flush remaining logs"""
        self.flush()
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record):
        """Format log record with colors"""
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
        return super().format(record)


def create_logger(
    name: str,
    log_dir: Path,
    config: Optional[Any] = None
) -> StructuredLogger:
    """
    Factory function to create logger from configuration.
    
    Args:
        name: Logger name
        log_dir: Log directory
        config: Optional PyGMOConfig instance
    
    Returns:
        StructuredLogger instance
    """
    if config:
        return StructuredLogger(
            name=name,
            log_dir=log_dir,
            log_level=config.log_level,
            log_to_console=config.log_to_console,
            log_to_file=config.log_to_file,
            json_format=config.log_json_format
        )
    else:
        return StructuredLogger(
            name=name,
            log_dir=log_dir
        )


