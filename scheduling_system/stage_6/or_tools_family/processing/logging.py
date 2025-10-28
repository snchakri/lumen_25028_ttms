"""
Comprehensive Logging System for OR-Tools Solver Family

Provides structured logging with console and JSON file output,
tracking all phases, metrics, and solver statistics.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    phase: str
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ComprehensiveLogger:
    """
    Comprehensive logging system with console and JSON file output.
    
    Tracks:
    - Phase timings
    - Solver statistics
    - Performance metrics
    - Memory usage
    - Solution quality metrics
    """
    
    def __init__(
        self,
        log_path: Path,
        log_level: str = "INFO",
        log_console: bool = True,
        log_file: bool = True
    ):
        self.log_path = Path(log_path)
        self.log_level = log_level
        self.log_console = log_console
        self.log_file = log_file
        
        # Create log directory
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize log entries list
        self.log_entries: list[LogEntry] = []
        
        # Setup Python logger
        self.logger = logging.getLogger("ORToolsSolver")
        self.logger.setLevel(getattr(logging, log_level))
        
        # Console handler
        if log_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level))
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler for standard logs
        if log_file:
            log_file_path = self.log_path / "ortools_solver.log"
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(getattr(logging, log_level))
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # JSON log file
        self.json_log_path = self.log_path / "ortools_solver.json"
        
        self.logger.info("Comprehensive Logger initialized")
        self.logger.info(f"Log path: {self.log_path}")
        self.logger.info(f"Log level: {log_level}")
    
    def log_phase_start(self, phase_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Log the start of a phase."""
        timestamp = datetime.now().isoformat()
        message = f"Starting phase: {phase_name}"
        
        log_entry = LogEntry(
            timestamp=timestamp,
            level="INFO",
            phase=phase_name,
            message=message,
            metadata=metadata or {}
        )
        
        self._add_log_entry(log_entry)
        self.logger.info(message)
    
    def log_phase_end(
        self,
        phase_name: str,
        success: bool,
        execution_time: float,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """Log the end of a phase with metrics."""
        timestamp = datetime.now().isoformat()
        status = "SUCCESS" if success else "FAILED"
        message = f"Completed phase: {phase_name} - {status} ({execution_time:.3f}s)"
        
        log_entry = LogEntry(
            timestamp=timestamp,
            level="INFO" if success else "ERROR",
            phase=phase_name,
            message=message,
            metrics={
                "execution_time": execution_time,
                "success": success,
                **(metrics or {})
            }
        )
        
        self._add_log_entry(log_entry)
        self.logger.info(message)
    
    def log_solver_stats(
        self,
        solver_type: str,
        stats: Dict[str, Any]
    ):
        """Log solver-specific statistics."""
        timestamp = datetime.now().isoformat()
        message = f"Solver statistics for {solver_type}"
        
        log_entry = LogEntry(
            timestamp=timestamp,
            level="INFO",
            phase="solver_execution",
            message=message,
            metrics=stats
        )
        
        self._add_log_entry(log_entry)
        self.logger.info(f"{message}: {stats}")
    
    def log_solution_quality(
        self,
        quality_metrics: Dict[str, float]
    ):
        """Log solution quality metrics."""
        timestamp = datetime.now().isoformat()
        message = "Solution quality metrics"
        
        log_entry = LogEntry(
            timestamp=timestamp,
            level="INFO",
            phase="solution_validation",
            message=message,
            metrics=quality_metrics
        )
        
        self._add_log_entry(log_entry)
        self.logger.info(f"{message}: {quality_metrics}")
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        error_data: Optional[Dict[str, Any]] = None
    ):
        """Log an error."""
        timestamp = datetime.now().isoformat()
        
        log_entry = LogEntry(
            timestamp=timestamp,
            level="ERROR",
            phase="error",
            message=f"{error_type}: {error_message}",
            metadata=error_data or {}
        )
        
        self._add_log_entry(log_entry)
        self.logger.error(f"{error_type}: {error_message}")
    
    def log_warning(self, warning_message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a warning."""
        timestamp = datetime.now().isoformat()
        
        log_entry = LogEntry(
            timestamp=timestamp,
            level="WARNING",
            phase="warning",
            message=warning_message,
            metadata=metadata or {}
        )
        
        self._add_log_entry(log_entry)
        self.logger.warning(warning_message)
    
    def log_debug(self, debug_message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a debug message."""
        timestamp = datetime.now().isoformat()
        
        log_entry = LogEntry(
            timestamp=timestamp,
            level="DEBUG",
            phase="debug",
            message=debug_message,
            metadata=metadata or {}
        )
        
        self._add_log_entry(log_entry)
        self.logger.debug(debug_message)
    
    def _add_log_entry(self, log_entry: LogEntry):
        """Add log entry to list and write to JSON file."""
        self.log_entries.append(log_entry)
        
        # Write to JSON file
        if self.log_file:
            with open(self.json_log_path, 'a') as f:
                json.dump(log_entry.to_dict(), f)
                f.write('\n')
    
    def get_logger(self) -> logging.Logger:
        """Get the underlying Python logger."""
        return self.logger
    
    def save_summary(self, summary: Dict[str, Any]):
        """Save execution summary to JSON."""
        summary_path = self.log_path / "execution_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Execution summary saved to {summary_path}")
    
    def get_all_logs(self) -> list[Dict[str, Any]]:
        """Get all log entries as list of dictionaries."""
        return [entry.to_dict() for entry in self.log_entries]
    
    def clear_logs(self):
        """Clear all log entries (useful for testing)."""
        self.log_entries.clear()
        self.logger.info("Logs cleared")


Comprehensive Logging System for OR-Tools Solver Family

Provides structured logging with console and JSON file output,
tracking all phases, metrics, and solver statistics.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    phase: str
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ComprehensiveLogger:
    """
    Comprehensive logging system with console and JSON file output.
    
    Tracks:
    - Phase timings
    - Solver statistics
    - Performance metrics
    - Memory usage
    - Solution quality metrics
    """
    
    def __init__(
        self,
        log_path: Path,
        log_level: str = "INFO",
        log_console: bool = True,
        log_file: bool = True
    ):
        self.log_path = Path(log_path)
        self.log_level = log_level
        self.log_console = log_console
        self.log_file = log_file
        
        # Create log directory
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize log entries list
        self.log_entries: list[LogEntry] = []
        
        # Setup Python logger
        self.logger = logging.getLogger("ORToolsSolver")
        self.logger.setLevel(getattr(logging, log_level))
        
        # Console handler
        if log_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level))
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler for standard logs
        if log_file:
            log_file_path = self.log_path / "ortools_solver.log"
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(getattr(logging, log_level))
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # JSON log file
        self.json_log_path = self.log_path / "ortools_solver.json"
        
        self.logger.info("Comprehensive Logger initialized")
        self.logger.info(f"Log path: {self.log_path}")
        self.logger.info(f"Log level: {log_level}")
    
    def log_phase_start(self, phase_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Log the start of a phase."""
        timestamp = datetime.now().isoformat()
        message = f"Starting phase: {phase_name}"
        
        log_entry = LogEntry(
            timestamp=timestamp,
            level="INFO",
            phase=phase_name,
            message=message,
            metadata=metadata or {}
        )
        
        self._add_log_entry(log_entry)
        self.logger.info(message)
    
    def log_phase_end(
        self,
        phase_name: str,
        success: bool,
        execution_time: float,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """Log the end of a phase with metrics."""
        timestamp = datetime.now().isoformat()
        status = "SUCCESS" if success else "FAILED"
        message = f"Completed phase: {phase_name} - {status} ({execution_time:.3f}s)"
        
        log_entry = LogEntry(
            timestamp=timestamp,
            level="INFO" if success else "ERROR",
            phase=phase_name,
            message=message,
            metrics={
                "execution_time": execution_time,
                "success": success,
                **(metrics or {})
            }
        )
        
        self._add_log_entry(log_entry)
        self.logger.info(message)
    
    def log_solver_stats(
        self,
        solver_type: str,
        stats: Dict[str, Any]
    ):
        """Log solver-specific statistics."""
        timestamp = datetime.now().isoformat()
        message = f"Solver statistics for {solver_type}"
        
        log_entry = LogEntry(
            timestamp=timestamp,
            level="INFO",
            phase="solver_execution",
            message=message,
            metrics=stats
        )
        
        self._add_log_entry(log_entry)
        self.logger.info(f"{message}: {stats}")
    
    def log_solution_quality(
        self,
        quality_metrics: Dict[str, float]
    ):
        """Log solution quality metrics."""
        timestamp = datetime.now().isoformat()
        message = "Solution quality metrics"
        
        log_entry = LogEntry(
            timestamp=timestamp,
            level="INFO",
            phase="solution_validation",
            message=message,
            metrics=quality_metrics
        )
        
        self._add_log_entry(log_entry)
        self.logger.info(f"{message}: {quality_metrics}")
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        error_data: Optional[Dict[str, Any]] = None
    ):
        """Log an error."""
        timestamp = datetime.now().isoformat()
        
        log_entry = LogEntry(
            timestamp=timestamp,
            level="ERROR",
            phase="error",
            message=f"{error_type}: {error_message}",
            metadata=error_data or {}
        )
        
        self._add_log_entry(log_entry)
        self.logger.error(f"{error_type}: {error_message}")
    
    def log_warning(self, warning_message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a warning."""
        timestamp = datetime.now().isoformat()
        
        log_entry = LogEntry(
            timestamp=timestamp,
            level="WARNING",
            phase="warning",
            message=warning_message,
            metadata=metadata or {}
        )
        
        self._add_log_entry(log_entry)
        self.logger.warning(warning_message)
    
    def log_debug(self, debug_message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a debug message."""
        timestamp = datetime.now().isoformat()
        
        log_entry = LogEntry(
            timestamp=timestamp,
            level="DEBUG",
            phase="debug",
            message=debug_message,
            metadata=metadata or {}
        )
        
        self._add_log_entry(log_entry)
        self.logger.debug(debug_message)
    
    def _add_log_entry(self, log_entry: LogEntry):
        """Add log entry to list and write to JSON file."""
        self.log_entries.append(log_entry)
        
        # Write to JSON file
        if self.log_file:
            with open(self.json_log_path, 'a') as f:
                json.dump(log_entry.to_dict(), f)
                f.write('\n')
    
    def get_logger(self) -> logging.Logger:
        """Get the underlying Python logger."""
        return self.logger
    
    def save_summary(self, summary: Dict[str, Any]):
        """Save execution summary to JSON."""
        summary_path = self.log_path / "execution_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Execution summary saved to {summary_path}")
    
    def get_all_logs(self) -> list[Dict[str, Any]]:
        """Get all log entries as list of dictionaries."""
        return [entry.to_dict() for entry in self.log_entries]
    
    def clear_logs(self):
        """Clear all log entries (useful for testing)."""
        self.log_entries.clear()
        self.logger.info("Logs cleared")




