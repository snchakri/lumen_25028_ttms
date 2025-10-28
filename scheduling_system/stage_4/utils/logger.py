"""
Structured Logging System for Stage 4 Feasibility Check
Implements comprehensive logging with JSON and console output
"""

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import sys


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: str
    level: str
    layer: Optional[str]
    message: str
    context: Dict[str, Any]
    execution_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None


class StructuredLogger:
    """
    Structured logging system for Stage 4 feasibility checking
    
    Features:
    - JSON logging for machine readability
    - Console logging with proper formatting (no emojis/special chars)
    - Layer-specific log contexts
    - Performance metrics logging
    - Mathematical operation logging
    - Log file rotation
    """
    
    def __init__(self, log_file: Optional[Path] = None, console_level: str = "INFO"):
        """
        Initialize structured logger
        
        Args:
            log_file: Path to JSON log file (if None, only console logging)
            console_level: Console logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_file = log_file
        self.console_level = console_level
        
        # Setup console handler
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(getattr(logging, console_level))
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.console_handler.setFormatter(console_formatter)
        
        # Setup JSON file handler if log file specified
        self.json_handler = None
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self.json_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10 MB
                backupCount=5
            )
            self.json_handler.setLevel(logging.DEBUG)
        
        # Setup root logger
        self.logger = logging.getLogger('stage4_feasibility')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.console_handler)
        if self.json_handler:
            self.logger.addHandler(self.json_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
        
        # Store structured logs for JSON output
        self.structured_logs = []
    
    def _create_log_entry(
        self,
        level: str,
        message: str,
        layer: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        execution_time_ms: Optional[float] = None,
        memory_usage_mb: Optional[float] = None
    ) -> LogEntry:
        """Create a structured log entry"""
        return LogEntry(
            timestamp=datetime.utcnow().isoformat(),
            level=level,
            layer=layer,
            message=message,
            context=context or {},
            execution_time_ms=execution_time_ms,
            memory_usage_mb=memory_usage_mb
        )
    
    def debug(self, message: str, layer: Optional[str] = None, **kwargs):
        """Log debug message"""
        entry = self._create_log_entry("DEBUG", message, layer, kwargs)
        self.structured_logs.append(entry)
        self.logger.debug(f"[{layer}] {message}" if layer else message)
    
    def info(self, message: str, layer: Optional[str] = None, **kwargs):
        """Log info message"""
        entry = self._create_log_entry("INFO", message, layer, kwargs)
        self.structured_logs.append(entry)
        self.logger.info(f"[{layer}] {message}" if layer else message)
    
    def warning(self, message: str, layer: Optional[str] = None, **kwargs):
        """Log warning message"""
        entry = self._create_log_entry("WARNING", message, layer, kwargs)
        self.structured_logs.append(entry)
        self.logger.warning(f"[{layer}] {message}" if layer else message)
    
    def error(self, message: str, layer: Optional[str] = None, **kwargs):
        """Log error message"""
        entry = self._create_log_entry("ERROR", message, layer, kwargs)
        self.structured_logs.append(entry)
        self.logger.error(f"[{layer}] {message}" if layer else message)
    
    def critical(self, message: str, layer: Optional[str] = None, **kwargs):
        """Log critical message"""
        entry = self._create_log_entry("CRITICAL", message, layer, kwargs)
        self.structured_logs.append(entry)
        self.logger.critical(f"[{layer}] {message}" if layer else message)
    
    def log_layer_start(self, layer_name: str, layer_number: int):
        """Log layer execution start"""
        self.info(
            f"Starting Layer {layer_number}: {layer_name}",
            layer=f"Layer{layer_number}",
            layer_name=layer_name,
            layer_number=layer_number
        )
    
    def log_layer_complete(
        self,
        layer_name: str,
        layer_number: int,
        status: str,
        execution_time_ms: float,
        memory_usage_mb: float
    ):
        """Log layer execution completion"""
        self.info(
            f"Layer {layer_number} completed: {status}",
            layer=f"Layer{layer_number}",
            layer_name=layer_name,
            status=status,
            execution_time_ms=execution_time_ms,
            memory_usage_mb=memory_usage_mb
        )
    
    def log_mathematical_operation(
        self,
        operation: str,
        theorem: str,
        result: Any,
        layer: Optional[str] = None
    ):
        """Log mathematical operation (theorem check, proof validation, etc.)"""
        self.debug(
            f"Mathematical operation: {operation}",
            layer=layer,
            operation=operation,
            theorem=theorem,
            result=str(result)
        )
    
    def log_performance_metrics(
        self,
        metrics: Dict[str, Any],
        layer: Optional[str] = None
    ):
        """Log performance metrics"""
        self.info(
            "Performance metrics",
            layer=layer,
            **metrics
        )
    
    def save_json_logs(self, output_path: Path):
        """Save structured logs to JSON file"""
        if not self.structured_logs:
            return
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump([asdict(entry) for entry in self.structured_logs], f, indent=2)
        
        self.info(f"Structured logs saved to {output_path}")
    
    def close(self):
        """Close all handlers"""
        if self.console_handler:
            self.console_handler.close()
        if self.json_handler:
            self.json_handler.close()


def create_logger(log_file: Optional[Path] = None, console_level: str = "INFO") -> StructuredLogger:
    """
    Factory function to create a structured logger
    
    Args:
        log_file: Path to JSON log file
        console_level: Console logging level
        
    Returns:
        Configured StructuredLogger instance
    """
    return StructuredLogger(log_file, console_level)
