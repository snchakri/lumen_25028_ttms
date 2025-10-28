"""
Comprehensive Logging System

Implements rigorous logging with console and JSON file outputs per requirements.

Requirements:
- Console output with color coding
- JSON log file with structured data
- Performance metrics tracking
- Mathematical validation results
- Solver statistics

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field, asdict
import time


@dataclass
class LogEntry:
    """Structured log entry."""
    
    timestamp: str
    level: str
    component: str
    message: str
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    mathematical_results: Dict[str, Any] = field(default_factory=dict)
    solver_statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'level': self.level,
            'component': self.component,
            'message': self.message,
            'performance_metrics': self.performance_metrics,
            'mathematical_results': self.mathematical_results,
            'solver_statistics': self.solver_statistics
        }


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'component': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'performance_metrics'):
            log_data['performance_metrics'] = record.performance_metrics
        
        if hasattr(record, 'mathematical_results'):
            log_data['mathematical_results'] = record.mathematical_results
        
        if hasattr(record, 'solver_statistics'):
            log_data['solver_statistics'] = record.solver_statistics
        
        return json.dumps(log_data)


class ColoredConsoleFormatter(logging.Formatter):
    """Color-coded console formatter."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format message
        formatted = super().format(record)
        
        # Add color
        return f"{color}{formatted}{reset}"


class ComprehensiveLogger:
    """
    Comprehensive logging system with console and JSON file outputs.
    
    Requirements: Console + JSON file logging
    """
    
    def __init__(
        self,
        log_path: Path,
        log_level: str = "INFO",
        log_console: bool = True,
        log_file: bool = True
    ):
        """
        Initialize comprehensive logger.
        
        Args:
            log_path: Path for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_console: Enable console logging
            log_file: Enable file logging
        """
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('PuLPSolver')
        self.logger.setLevel(getattr(logging, log_level))
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        if log_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level))
            console_formatter = ColoredConsoleFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # JSON file handler
        if log_file:
            json_log_file = self.log_path / 'pulp_solver.json.log'
            file_handler = logging.FileHandler(json_log_file)
            file_handler.setLevel(getattr(logging, log_level))
            json_formatter = JSONFormatter()
            file_handler.setFormatter(json_formatter)
            self.logger.addHandler(file_handler)
            
            # Also create text log for human readability
            text_log_file = self.log_path / 'pulp_solver.log'
            text_handler = logging.FileHandler(text_log_file)
            text_handler.setLevel(getattr(logging, log_level))
            text_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            text_handler.setFormatter(text_formatter)
            self.logger.addHandler(text_handler)
        
        # Store log entries for JSON export
        self.log_entries: List[LogEntry] = []
    
    def log_phase_start(self, phase_name: str):
        """Log start of a processing phase."""
        self.logger.info(f"{'='*80}")
        self.logger.info(f"STARTING PHASE: {phase_name}")
        self.logger.info(f"{'='*80}")
    
    def log_phase_end(self, phase_name: str, success: bool, execution_time: float):
        """Log end of a processing phase."""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"{'='*80}")
        self.logger.info(f"PHASE COMPLETE: {phase_name} - {status} ({execution_time:.3f}s)")
        self.logger.info(f"{'='*80}")
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics."""
        self.logger.info("Performance Metrics:")
        for key, value in metrics.items():
            self.logger.info(f"  - {key}: {value}")
    
    def log_mathematical_validation(self, validation_results: Dict[str, Any]):
        """Log mathematical validation results."""
        self.logger.info("Mathematical Validation:")
        for check_name, result in validation_results.items():
            status = "PASSED" if result.get('passed', False) else "FAILED"
            self.logger.info(f"  - {check_name}: {status}")
            if 'details' in result:
                for detail_key, detail_value in result['details'].items():
                    self.logger.info(f"    - {detail_key}: {detail_value}")
    
    def log_solver_statistics(self, solver_stats: Dict[str, Any]):
        """Log solver statistics."""
        self.logger.info("Solver Statistics:")
        for key, value in solver_stats.items():
            self.logger.info(f"  - {key}: {value}")
    
    def log_error_with_context(
        self,
        error: Exception,
        context: Dict[str, Any],
        suggested_fixes: List[str]
    ):
        """Log error with full context and suggested fixes."""
        self.logger.error(f"Error: {str(error)}")
        self.logger.error(f"Error Type: {type(error).__name__}")
        
        if context:
            self.logger.error("Context:")
            for key, value in context.items():
                self.logger.error(f"  - {key}: {value}")
        
        if suggested_fixes:
            self.logger.error("Suggested Fixes:")
            for i, fix in enumerate(suggested_fixes, 1):
                self.logger.error(f"  {i}. {fix}")
    
    def export_logs_to_json(self, output_file: Path):
        """Export all log entries to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(
                [entry.to_dict() for entry in self.log_entries],
                f,
                indent=2
            )
    
    def get_logger(self) -> logging.Logger:
        """Get the underlying logger instance."""
        return self.logger



