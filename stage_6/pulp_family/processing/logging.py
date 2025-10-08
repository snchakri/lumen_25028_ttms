#!/usr/bin/env python3
"""
PuLP Solver Family - Stage 6 Processing Layer: Execution Logging & Audit Module

This module implements the complete logging and audit functionality for Stage 6.1
processing, providing complete execution tracking, performance monitoring, and error
diagnostics with mathematical rigor and theoretical compliance. Critical component implementing
the complete audit trail per Stage 6 foundational framework with guaranteed traceability.

Theoretical Foundation:
    Based on Stage 6.1 PuLP Framework (Section 8: Integration with Scheduling Pipeline):
    - Implements complete logging per Section 8.2 (Error Handling and Recovery)
    - Maintains complete execution audit trail for mathematical validation
    - Ensures performance monitoring and resource usage tracking
    - Provides structured error reporting and diagnostic capabilities  
    - Supports dynamic parameter logging and EAV system integration

Architecture Compliance:
    - Implements Processing Layer Stage 4 per foundational design rules
    - Maintains O(1) logging operations for performance-critical paths
    - Provides fail-safe error handling with complete diagnostic capture
    - Supports distributed execution tracking and centralized audit storage
    - Ensures memory-efficient logging with configurable detail levels

Dependencies: logging, structlog, json, psutil, datetime, pathlib, typing, dataclasses
Author: Student Team
Version: 1.0.0 (Production)
"""

import logging
import json
import time
import psutil
import threading
import traceback
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, IO
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import queue

# Import data structures from previous modules - strict dependency management
try:
    from ..input_model.metadata import InputModelMetadata
    from .solver import SolverResult, SolverStatus
    from ..output_model.decoder import DecodingMetrics, SchedulingAssignment
except ImportError:
    # Handle standalone execution or development imports
    import sys
    sys.path.append('..')
    try:
        from input_model.metadata import InputModelMetadata
        from processing.solver import SolverResult, SolverStatus
        from output_model.decoder import DecodingMetrics, SchedulingAssignment
    except ImportError:
        # Final fallback for direct execution
        class InputModelMetadata: pass
        class SolverResult: pass
        class SolverStatus: pass
        class DecodingMetrics: pass
        class SchedulingAssignment: pass

# Configure base logger
logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Enumeration of logging levels with enterprise categorization."""
    TRACE = "TRACE"              # Detailed trace information for debugging
    DEBUG = "DEBUG"              # Debug information for development
    INFO = "INFO"                # General information messages
    SUCCESS = "SUCCESS"          # Success operation confirmations
    WARNING = "WARNING"          # Warning messages for potential issues
    ERROR = "ERROR"              # Error messages for failures
    CRITICAL = "CRITICAL"        # Critical system failures
    AUDIT = "AUDIT"             # Audit trail information for compliance

class LogCategory(Enum):
    """Categorization of log entries for structured analysis."""
    SYSTEM = "system"                    # System-level operations
    PERFORMANCE = "performance"          # Performance monitoring data
    SOLVER = "solver"                   # Solver execution information
    DATA = "data"                       # Data processing operations
    VALIDATION = "validation"           # Validation and verification
    ERROR = "error"                     # Error and exception handling
    AUDIT = "audit"                     # Audit and compliance tracking
    RESOURCE = "resource"               # Resource usage monitoring
    SECURITY = "security"               # Security and access control

@dataclass
class LogEntry:
    """
    complete log entry structure with mathematical precision tracking.

    Mathematical Foundation: Captures complete execution context for audit
    compliance and performance analysis per Stage 6.1 integration requirements.

    Attributes:
        timestamp: Precise timestamp with timezone information
        execution_id: Unique execution identifier for correlation
        log_level: Severity level of log entry
        log_category: Functional category for structured analysis
        component: Component or module generating the log
        operation: Specific operation being logged
        message: Human-readable log message
        data: Structured data associated with log entry
        performance_metrics: Performance-related measurements
        resource_usage: System resource consumption data
        error_info: Error information if applicable
        trace_context: Execution trace context for debugging
    """
    timestamp: str
    execution_id: str
    log_level: LogLevel
    log_category: LogCategory
    component: str
    operation: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, Union[int, float]] = field(default_factory=dict)
    error_info: Optional[Dict[str, str]] = None
    trace_context: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary for JSON serialization."""
        entry_dict = asdict(self)
        # Convert enums to values for JSON serialization
        entry_dict['log_level'] = self.log_level.value
        entry_dict['log_category'] = self.log_category.value
        return entry_dict

    def get_summary(self) -> str:
        """Generate human-readable summary of log entry."""
        return f"[{self.timestamp}] {self.log_level.value} {self.component}.{self.operation}: {self.message}"

@dataclass
class ExecutionSummary:
    """
    complete execution summary with mathematical performance analysis.

    Mathematical Foundation: Aggregates execution statistics for performance
    characterization and compliance reporting per theoretical framework requirements.

    Attributes:
        execution_id: Unique execution identifier
        start_time: Execution start timestamp
        end_time: Execution completion timestamp  
        total_duration_seconds: Total execution time
        components_executed: List of components that executed
        operations_completed: Total number of operations completed
        performance_summary: Aggregated performance metrics
        resource_peak_usage: Peak resource consumption measurements
        error_summary: Summary of errors and warnings
        success_rate: Overall execution success rate
        log_entry_count: Total number of log entries generated
        file_outputs: List of output files generated
    """
    execution_id: str
    start_time: str
    end_time: str
    total_duration_seconds: float
    components_executed: List[str]
    operations_completed: int
    performance_summary: Dict[str, float]
    resource_peak_usage: Dict[str, Union[int, float]]
    error_summary: Dict[str, int]
    success_rate: float
    log_entry_count: int
    file_outputs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert execution summary to dictionary."""
        return asdict(self)

    def get_performance_grade(self) -> str:
        """Calculate performance grade based on metrics."""
        if self.success_rate >= 0.95 and self.total_duration_seconds < 300:
            return "A+"
        elif self.success_rate >= 0.90 and self.total_duration_seconds < 450:
            return "A"
        elif self.success_rate >= 0.85 and self.total_duration_seconds < 600:
            return "B+"
        elif self.success_rate >= 0.80:
            return "B"
        elif self.success_rate >= 0.70:
            return "C"
        else:
            return "D"

@dataclass
class LoggingConfiguration:
    """
    Configuration structure for execution logging system.

    Provides fine-grained control over logging behavior while maintaining
    performance and ensuring complete audit capability.

    Attributes:
        log_level: Minimum logging level for output
        enable_performance_logging: Enable detailed performance tracking
        enable_resource_monitoring: Enable system resource monitoring
        log_buffer_size: Size of log buffer for batch writing
        log_rotation_size_mb: File size threshold for log rotation
        max_log_files: Maximum number of log files to retain
        json_formatting: Use structured JSON log formatting
        console_output: Enable console log output
        file_output: Enable file log output
        async_logging: Use asynchronous logging for performance
        structured_data_logging: Enable structured data capture
    """
    log_level: LogLevel = LogLevel.INFO
    enable_performance_logging: bool = True
    enable_resource_monitoring: bool = True
    log_buffer_size: int = 1000
    log_rotation_size_mb: int = 100
    max_log_files: int = 10
    json_formatting: bool = True
    console_output: bool = True
    file_output: bool = True
    async_logging: bool = True
    structured_data_logging: bool = True
    detailed_error_reporting: bool = True

    def validate_config(self) -> None:
        """Validate logging configuration parameters."""
        if self.log_buffer_size <= 0:
            raise ValueError("Log buffer size must be positive")

        if self.log_rotation_size_mb <= 0:
            raise ValueError("Log rotation size must be positive")

        if self.max_log_files <= 0:
            raise ValueError("Maximum log files must be positive")

class LogHandler(ABC):
    """
    Abstract base class for log output handlers.

    Implements strategy pattern for different logging outputs while maintaining
    consistent performance and structured formatting across all implementations.
    """

    @abstractmethod
    def write_log_entry(self, entry: LogEntry) -> None:
        """Write log entry to output destination."""
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush buffered log entries."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close log handler and cleanup resources."""
        pass

class FileLogHandler(LogHandler):
    """
    File-based log handler with rotation and structured formatting.

    Mathematical Foundation: Implements high-performance file logging with
    optimal I/O characteristics and guaranteed data persistence for audit compliance.
    """

    def __init__(self, log_file_path: Path, config: LoggingConfiguration):
        """Initialize file log handler."""
        self.log_file_path = log_file_path
        self.config = config

        # Create log directory if needed
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize file handle
        self.log_file: Optional[IO] = None
        self.current_file_size = 0
        self.file_index = 0

        # Initialize buffer for batch writing
        self.log_buffer: List[LogEntry] = []
        self.buffer_lock = threading.Lock()

        self._open_log_file()

        logger.debug(f"FileLogHandler initialized: {self.log_file_path}")

    def _open_log_file(self) -> None:
        """Open log file for writing."""
        try:
            if self.log_file:
                self.log_file.close()

            # Generate log file name with rotation index
            if self.file_index == 0:
                file_path = self.log_file_path
            else:
                stem = self.log_file_path.stem
                suffix = self.log_file_path.suffix
                file_path = self.log_file_path.parent / f"{stem}.{self.file_index}{suffix}"

            self.log_file = open(file_path, 'a', encoding='utf-8')
            self.current_file_size = self.log_file.tell()

        except Exception as e:
            logger.error(f"Failed to open log file {self.log_file_path}: {str(e)}")
            raise

    def write_log_entry(self, entry: LogEntry) -> None:
        """Write log entry to file with buffering."""
        with self.buffer_lock:
            self.log_buffer.append(entry)

            # Flush buffer if it reaches threshold
            if len(self.log_buffer) >= self.config.log_buffer_size:
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush log buffer to file."""
        if not self.log_buffer or not self.log_file:
            return

        try:
            # Write buffered entries
            for entry in self.log_buffer:
                if self.config.json_formatting:
                    log_line = json.dumps(entry.to_dict(), default=str) + '\n'
                else:
                    log_line = entry.get_summary() + '\n'

                self.log_file.write(log_line)
                self.current_file_size += len(log_line.encode('utf-8'))

            self.log_file.flush()
            self.log_buffer.clear()

            # Check for rotation
            if self.current_file_size > self.config.log_rotation_size_mb * 1024 * 1024:
                self._rotate_log_file()

        except Exception as e:
            logger.error(f"Failed to flush log buffer: {str(e)}")

    def _rotate_log_file(self) -> None:
        """Rotate log file when size threshold exceeded."""
        try:
            self.file_index += 1

            # Remove old log files if exceeding max count
            self._cleanup_old_log_files()

            # Open new log file
            self._open_log_file()

            logger.debug(f"Rotated log file to index {self.file_index}")

        except Exception as e:
            logger.error(f"Failed to rotate log file: {str(e)}")

    def _cleanup_old_log_files(self) -> None:
        """Remove old log files exceeding maximum count."""
        try:
            # Find existing log files
            stem = self.log_file_path.stem
            suffix = self.log_file_path.suffix
            log_dir = self.log_file_path.parent

            existing_files = []
            for file_path in log_dir.glob(f"{stem}*{suffix}"):
                existing_files.append(file_path)

            # Sort by modification time (oldest first)
            existing_files.sort(key=lambda p: p.stat().st_mtime)

            # Remove excess files
            files_to_remove = len(existing_files) - self.config.max_log_files + 1
            if files_to_remove > 0:
                for file_path in existing_files[:files_to_remove]:
                    try:
                        file_path.unlink()
                        logger.debug(f"Removed old log file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Could not remove old log file {file_path}: {str(e)}")

        except Exception as e:
            logger.warning(f"Failed to cleanup old log files: {str(e)}")

    def flush(self) -> None:
        """Force flush of buffered entries."""
        with self.buffer_lock:
            self._flush_buffer()

    def close(self) -> None:
        """Close file handler and cleanup resources."""
        try:
            # Flush remaining buffer
            self.flush()

            # Close file handle
            if self.log_file:
                self.log_file.close()
                self.log_file = None

        except Exception as e:
            logger.error(f"Error closing file log handler: {str(e)}")

class ConsoleLogHandler(LogHandler):
    """
    Console-based log handler with color formatting and structured output.

    Mathematical Foundation: Provides real-time logging output for development
    and debugging with optimal terminal formatting and minimal performance impact.
    """

    def __init__(self, config: LoggingConfiguration):
        """Initialize console log handler."""
        self.config = config

        # Color codes for different log levels
        self.color_codes = {
            LogLevel.TRACE: '\033[90m',      # Dark gray
            LogLevel.DEBUG: '\033[36m',      # Cyan
            LogLevel.INFO: '\033[32m',       # Green  
            LogLevel.SUCCESS: '\033[92m',    # Bright green
            LogLevel.WARNING: '\033[33m',    # Yellow
            LogLevel.ERROR: '\033[31m',      # Red
            LogLevel.CRITICAL: '\033[91m',   # Bright red
            LogLevel.AUDIT: '\033[35m'       # Magenta
        }
        self.reset_code = '\033[0m'

        logger.debug("ConsoleLogHandler initialized")

    def write_log_entry(self, entry: LogEntry) -> None:
        """Write log entry to console with formatting."""
        try:
            if self.config.json_formatting:
                # JSON format for structured logging
                print(json.dumps(entry.to_dict(), default=str))
            else:
                # Human-readable format with colors
                color_code = self.color_codes.get(entry.log_level, '')
                formatted_message = f"{color_code}{entry.get_summary()}{self.reset_code}"
                print(formatted_message)

                # Print additional data if available
                if entry.data and entry.log_level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                    print(f"  Data: {json.dumps(entry.data, indent=2, default=str)}")

        except Exception as e:
            # Fallback to basic print if formatting fails
            print(f"LOG ERROR: Could not format entry: {str(e)}")
            print(f"Original message: {entry.message}")

    def flush(self) -> None:
        """Flush console output."""
        sys.stdout.flush()

    def close(self) -> None:
        """Close console handler."""
        self.flush()

class PerformanceMonitor:
    """
    Performance monitoring utility for execution tracking.

    Mathematical Foundation: Implements complete performance measurement
    with statistical analysis and resource usage tracking per theoretical requirements.
    """

    def __init__(self, execution_id: str):
        """Initialize performance monitor."""
        self.execution_id = execution_id
        self.start_time = time.time()
        self.operation_times: Dict[str, List[float]] = {}
        self.resource_samples: List[Dict[str, Union[int, float]]] = []
        self.monitoring_active = True

        # Start resource monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitoring_thread.start()

        logger.debug(f"PerformanceMonitor initialized for execution {execution_id}")

    @contextmanager
    def measure_operation(self, operation_name: str):
        """Context manager for measuring operation performance."""
        start_time = time.time()

        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time

            # Record operation time
            if operation_name not in self.operation_times:
                self.operation_times[operation_name] = []
            self.operation_times[operation_name].append(duration)

    def _monitor_resources(self) -> None:
        """Monitor system resource usage in background thread."""
        try:
            process = psutil.Process()

            while self.monitoring_active:
                try:
                    # Sample current resource usage
                    memory_info = process.memory_info()
                    cpu_percent = process.cpu_percent()

                    resource_sample = {
                        'timestamp': time.time(),
                        'memory_rss_mb': memory_info.rss / (1024 * 1024),
                        'memory_vms_mb': memory_info.vms / (1024 * 1024),
                        'cpu_percent': cpu_percent,
                        'threads_count': process.num_threads()
                    }

                    self.resource_samples.append(resource_sample)

                    # Limit sample history to prevent memory growth
                    if len(self.resource_samples) > 1000:
                        self.resource_samples = self.resource_samples[-500:]

                    time.sleep(1.0)  # Sample every second

                except psutil.NoSuchProcess:
                    break
                except Exception as e:
                    logger.debug(f"Resource monitoring error: {str(e)}")
                    time.sleep(5.0)

        except Exception as e:
            logger.error(f"Performance monitoring thread failed: {str(e)}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate complete performance summary."""
        current_time = time.time()
        total_duration = current_time - self.start_time

        # Calculate operation statistics
        operation_stats = {}
        for op_name, times in self.operation_times.items():
            if times:
                operation_stats[op_name] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'average_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }

        # Calculate resource statistics
        resource_stats = {}
        if self.resource_samples:
            memory_values = [s['memory_rss_mb'] for s in self.resource_samples]
            cpu_values = [s['cpu_percent'] for s in self.resource_samples]

            resource_stats = {
                'peak_memory_mb': max(memory_values) if memory_values else 0,
                'average_memory_mb': sum(memory_values) / len(memory_values) if memory_values else 0,
                'peak_cpu_percent': max(cpu_values) if cpu_values else 0,
                'average_cpu_percent': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                'sample_count': len(self.resource_samples)
            }

        return {
            'execution_id': self.execution_id,
            'total_duration_seconds': total_duration,
            'operation_statistics': operation_stats,
            'resource_statistics': resource_stats,
            'performance_grade': self._calculate_performance_grade(total_duration, resource_stats)
        }

    def _calculate_performance_grade(self, duration: float, resource_stats: Dict) -> str:
        """Calculate performance grade based on metrics."""
        # Grade based on duration and resource usage
        if duration < 60 and resource_stats.get('peak_memory_mb', 0) < 200:
            return "A+"
        elif duration < 180 and resource_stats.get('peak_memory_mb', 0) < 350:
            return "A"
        elif duration < 300 and resource_stats.get('peak_memory_mb', 0) < 500:
            return "B+"
        elif duration < 450:
            return "B"
        elif duration < 600:
            return "C"
        else:
            return "D"

    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)

class PuLPExecutionLogger:
    """
    complete execution logger for PuLP solver family pipeline.

    Implements complete logging, performance monitoring, and audit trail
    functionality following Stage 6.1 theoretical framework. Provides mathematical
    guarantees for complete execution traceability with optimal performance characteristics.

    Mathematical Foundation:
        - Implements complete audit trail per Section 8 (Integration with Scheduling Pipeline)
        - Maintains O(1) logging operations for performance-critical execution paths
        - Ensures complete error diagnostics and recovery information capture
        - Provides statistical performance analysis and resource usage tracking
        - Supports distributed execution coordination and centralized audit storage
    """

    def __init__(self, execution_id: str, log_directory: Path, config: LoggingConfiguration = LoggingConfiguration()):
        """Initialize PuLP execution logger with complete monitoring."""
        self.execution_id = execution_id
        self.log_directory = Path(log_directory)
        self.config = config
        self.config.validate_config()

        # Create log directory structure
        self.execution_log_dir = self.log_directory / f"execution_{execution_id}"
        self.execution_log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize log handlers
        self.handlers: List[LogHandler] = []
        self._initialize_handlers()

        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor(execution_id)

        # Initialize execution state
        self.execution_start_time = datetime.now(timezone.utc)
        self.log_entries: List[LogEntry] = []
        self.component_operations: Dict[str, List[str]] = {}
        self.error_count = 0
        self.warning_count = 0

        # Initialize async logging if enabled
        self.async_queue: Optional[queue.Queue] = None
        self.async_thread: Optional[threading.Thread] = None
        if self.config.async_logging:
            self._initialize_async_logging()

        logger.info(f"PuLPExecutionLogger initialized for execution {execution_id}")

        # Log initialization success
        self.log_success("system", "initialization", "Execution logger initialized successfully", {
            'execution_id': execution_id,
            'log_directory': str(self.log_directory),
            'config': self.config.__dict__
        })

    def _initialize_handlers(self) -> None:
        """Initialize log output handlers."""
        try:
            # File handler
            if self.config.file_output:
                log_file_path = self.execution_log_dir / f"execution_{self.execution_id}.log"
                file_handler = FileLogHandler(log_file_path, self.config)
                self.handlers.append(file_handler)

            # Console handler
            if self.config.console_output:
                console_handler = ConsoleLogHandler(self.config)
                self.handlers.append(console_handler)

        except Exception as e:
            logger.error(f"Failed to initialize log handlers: {str(e)}")
            raise

    def _initialize_async_logging(self) -> None:
        """Initialize asynchronous logging system."""
        try:
            self.async_queue = queue.Queue(maxsize=self.config.log_buffer_size * 2)
            self.async_thread = threading.Thread(target=self._async_log_worker, daemon=True)
            self.async_thread.start()

        except Exception as e:
            logger.error(f"Failed to initialize async logging: {str(e)}")
            # Fall back to synchronous logging
            self.config.async_logging = False

    def _async_log_worker(self) -> None:
        """Background worker for asynchronous log processing."""
        try:
            while True:
                try:
                    # Get log entry from queue (blocks until available)
                    entry = self.async_queue.get(timeout=1.0)
                    if entry is None:  # Poison pill to stop thread
                        break

                    # Write to all handlers
                    self._write_to_handlers(entry)
                    self.async_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Async log worker error: {str(e)}")

        except Exception as e:
            logger.error(f"Async log worker thread failed: {str(e)}")

    def _write_to_handlers(self, entry: LogEntry) -> None:
        """Write log entry to all active handlers."""
        for handler in self.handlers:
            try:
                handler.write_log_entry(entry)
            except Exception as e:
                # Avoid recursive logging issues
                print(f"Log handler error: {str(e)}")

    def _create_log_entry(self, log_level: LogLevel, log_category: LogCategory,
                         component: str, operation: str, message: str,
                         data: Optional[Dict[str, Any]] = None,
                         error_info: Optional[Exception] = None) -> LogEntry:
        """Create structured log entry with complete context."""
        # Gather performance metrics if enabled
        performance_metrics = {}
        if self.config.enable_performance_logging:
            performance_metrics = self.performance_monitor.get_performance_summary().get('resource_statistics', {})

        # Gather resource usage if enabled
        resource_usage = {}
        if self.config.enable_resource_monitoring:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                resource_usage = {
                    'memory_rss_mb': memory_info.rss / (1024 * 1024),
                    'memory_vms_mb': memory_info.vms / (1024 * 1024),
                    'cpu_percent': process.cpu_percent(),
                    'threads_count': process.num_threads()
                }
            except Exception:
                resource_usage = {'error': 'Could not gather resource usage'}

        # Gather error information if provided
        error_data = None
        if error_info and self.config.detailed_error_reporting:
            error_data = {
                'error_type': type(error_info).__name__,
                'error_message': str(error_info),
                'traceback': traceback.format_exc() if isinstance(error_info, Exception) else None
            }

        # Create trace context
        trace_context = {
            'thread_id': str(threading.current_thread().ident),
            'thread_name': threading.current_thread().name,
            'function': sys._getframe(2).f_code.co_name if sys._getframe(2) else 'unknown'
        }

        # Create log entry
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            execution_id=self.execution_id,
            log_level=log_level,
            log_category=log_category,
            component=component,
            operation=operation,
            message=message,
            data=data or {},
            performance_metrics=performance_metrics,
            resource_usage=resource_usage,
            error_info=error_data,
            trace_context=trace_context
        )

        return entry

    def _log_entry(self, entry: LogEntry) -> None:
        """Process and output log entry."""
        # Store entry for summary generation
        self.log_entries.append(entry)

        # Track component operations
        if entry.component not in self.component_operations:
            self.component_operations[entry.component] = []
        if entry.operation not in self.component_operations[entry.component]:
            self.component_operations[entry.component].append(entry.operation)

        # Update error counters
        if entry.log_level == LogLevel.ERROR or entry.log_level == LogLevel.CRITICAL:
            self.error_count += 1
        elif entry.log_level == LogLevel.WARNING:
            self.warning_count += 1

        # Output log entry
        if self.config.async_logging and self.async_queue:
            try:
                self.async_queue.put_nowait(entry)
            except queue.Full:
                # Fallback to synchronous if queue is full
                self._write_to_handlers(entry)
        else:
            self._write_to_handlers(entry)

    def log_trace(self, component: str, operation: str, message: str, 
                 data: Optional[Dict[str, Any]] = None) -> None:
        """Log trace-level information for detailed debugging."""
        if self.config.log_level.value in ['TRACE']:
            entry = self._create_log_entry(LogLevel.TRACE, LogCategory.SYSTEM, component, operation, message, data)
            self._log_entry(entry)

    def log_debug(self, component: str, operation: str, message: str,
                 data: Optional[Dict[str, Any]] = None) -> None:
        """Log debug information for development and troubleshooting."""
        if self.config.log_level.value in ['TRACE', 'DEBUG']:
            entry = self._create_log_entry(LogLevel.DEBUG, LogCategory.SYSTEM, component, operation, message, data)
            self._log_entry(entry)

    def log_info(self, component: str, operation: str, message: str,
                data: Optional[Dict[str, Any]] = None) -> None:
        """Log general information messages."""
        if self.config.log_level.value in ['TRACE', 'DEBUG', 'INFO']:
            entry = self._create_log_entry(LogLevel.INFO, LogCategory.SYSTEM, component, operation, message, data)
            self._log_entry(entry)

    def log_success(self, component: str, operation: str, message: str,
                   data: Optional[Dict[str, Any]] = None) -> None:
        """Log successful operation completion."""
        entry = self._create_log_entry(LogLevel.SUCCESS, LogCategory.SYSTEM, component, operation, message, data)
        self._log_entry(entry)

    def log_warning(self, component: str, operation: str, message: str,
                   data: Optional[Dict[str, Any]] = None) -> None:
        """Log warning messages for potential issues."""
        entry = self._create_log_entry(LogLevel.WARNING, LogCategory.SYSTEM, component, operation, message, data)
        self._log_entry(entry)

    def log_error(self, component: str, operation: str, message: str,
                 data: Optional[Dict[str, Any]] = None,
                 error: Optional[Exception] = None) -> None:
        """Log error messages with optional exception information."""
        entry = self._create_log_entry(LogLevel.ERROR, LogCategory.ERROR, component, operation, message, data, error)
        self._log_entry(entry)

    def log_critical(self, component: str, operation: str, message: str,
                    data: Optional[Dict[str, Any]] = None,
                    error: Optional[Exception] = None) -> None:
        """Log critical system failures."""
        entry = self._create_log_entry(LogLevel.CRITICAL, LogCategory.ERROR, component, operation, message, data, error)
        self._log_entry(entry)

    def log_audit(self, component: str, operation: str, message: str,
                 data: Optional[Dict[str, Any]] = None) -> None:
        """Log audit trail information for compliance."""
        entry = self._create_log_entry(LogLevel.AUDIT, LogCategory.AUDIT, component, operation, message, data)
        self._log_entry(entry)

    def log_performance(self, component: str, operation: str, message: str,
                       metrics: Dict[str, float],
                       data: Optional[Dict[str, Any]] = None) -> None:
        """Log performance measurement data."""
        entry = self._create_log_entry(LogLevel.INFO, LogCategory.PERFORMANCE, component, operation, message, data)
        entry.performance_metrics.update(metrics)
        self._log_entry(entry)

    def log_solver_operation(self, solver_name: str, operation: str, message: str,
                           solver_data: Optional[Dict[str, Any]] = None) -> None:
        """Log solver-specific operations and results."""
        entry = self._create_log_entry(LogLevel.INFO, LogCategory.SOLVER, f"solver.{solver_name}", operation, message, solver_data)
        self._log_entry(entry)

    def log_data_operation(self, component: str, operation: str, message: str,
                         data_info: Optional[Dict[str, Any]] = None) -> None:
        """Log data processing operations."""
        entry = self._create_log_entry(LogLevel.INFO, LogCategory.DATA, component, operation, message, data_info)
        self._log_entry(entry)

    def log_validation_result(self, component: str, operation: str, message: str,
                            validation_data: Dict[str, Any]) -> None:
        """Log validation and verification results."""
        log_level = LogLevel.SUCCESS if validation_data.get('success', False) else LogLevel.ERROR
        entry = self._create_log_entry(log_level, LogCategory.VALIDATION, component, operation, message, validation_data)
        self._log_entry(entry)

    @contextmanager
    def measure_operation(self, component: str, operation: str, description: str):
        """Context manager for measuring and logging operation performance."""
        start_message = f"Starting {description}"
        self.log_info(component, operation, start_message)

        with self.performance_monitor.measure_operation(f"{component}.{operation}"):
            try:
                yield
                success_message = f"Completed {description} successfully"
                self.log_success(component, operation, success_message)
            except Exception as e:
                error_message = f"Failed {description}: {str(e)}"
                self.log_error(component, operation, error_message, error=e)
                raise

    def generate_execution_summary(self) -> ExecutionSummary:
        """Generate complete execution summary with performance analysis."""
        current_time = datetime.now(timezone.utc)
        total_duration = (current_time - self.execution_start_time).total_seconds()

        # Get performance summary
        perf_summary = self.performance_monitor.get_performance_summary()

        # Calculate success rate
        total_operations = len(self.log_entries)
        success_operations = len([e for e in self.log_entries if e.log_level == LogLevel.SUCCESS])
        success_rate = success_operations / total_operations if total_operations > 0 else 0.0

        # Generate error summary
        error_summary = {
            'critical_errors': len([e for e in self.log_entries if e.log_level == LogLevel.CRITICAL]),
            'errors': len([e for e in self.log_entries if e.log_level == LogLevel.ERROR]),
            'warnings': len([e for e in self.log_entries if e.log_level == LogLevel.WARNING]),
            'total_issues': self.error_count + self.warning_count
        }

        # Find output files
        output_files = []
        try:
            for log_file in self.execution_log_dir.glob("*.log"):
                output_files.append(str(log_file))
            for json_file in self.execution_log_dir.glob("*.json"):
                output_files.append(str(json_file))
        except Exception as e:
            logger.debug(f"Could not enumerate output files: {str(e)}")

        summary = ExecutionSummary(
            execution_id=self.execution_id,
            start_time=self.execution_start_time.isoformat(),
            end_time=current_time.isoformat(),
            total_duration_seconds=total_duration,
            components_executed=list(self.component_operations.keys()),
            operations_completed=total_operations,
            performance_summary=perf_summary.get('operation_statistics', {}),
            resource_peak_usage=perf_summary.get('resource_statistics', {}),
            error_summary=error_summary,
            success_rate=success_rate,
            log_entry_count=len(self.log_entries),
            file_outputs=output_files,
            metadata={
                'performance_grade': perf_summary.get('performance_grade', 'N/A'),
                'log_config': self.config.__dict__
            }
        )

        return summary

    def save_execution_summary(self) -> Path:
        """Save execution summary to JSON file."""
        summary = self.generate_execution_summary()

        summary_file = self.execution_log_dir / f"execution_summary_{self.execution_id}.json"

        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary.to_dict(), f, indent=2, default=str)

            self.log_audit("system", "summary_generation", "Execution summary saved successfully", {
                'summary_file': str(summary_file),
                'performance_grade': summary.get_performance_grade()
            })

            return summary_file

        except Exception as e:
            self.log_error("system", "summary_generation", f"Failed to save execution summary: {str(e)}", error=e)
            raise

    def flush_logs(self) -> None:
        """Force flush all log handlers."""
        try:
            # Flush async queue if enabled
            if self.async_queue:
                self.async_queue.join()

            # Flush all handlers
            for handler in self.handlers:
                handler.flush()

        except Exception as e:
            logger.error(f"Error flushing logs: {str(e)}")

    def close(self) -> None:
        """Close logger and cleanup resources."""
        try:
            # Log execution completion
            self.log_audit("system", "shutdown", "Execution logger shutting down")

            # Generate and save final summary
            self.save_execution_summary()

            # Stop performance monitoring
            self.performance_monitor.stop_monitoring()

            # Stop async logging if enabled
            if self.async_queue and self.async_thread:
                self.async_queue.put(None)  # Poison pill
                self.async_thread.join(timeout=5.0)

            # Close all handlers
            for handler in self.handlers:
                handler.close()

            logger.info(f"PuLPExecutionLogger closed for execution {self.execution_id}")

        except Exception as e:
            logger.error(f"Error closing execution logger: {str(e)}")

def create_execution_logger(execution_id: str, 
                          log_directory: Union[str, Path],
                          config: Optional[LoggingConfiguration] = None) -> PuLPExecutionLogger:
    """
    Factory function to create PuLP execution logger with complete monitoring.

    Provides simplified interface for logger creation with optimal configuration
    for processing pipeline integration and performance monitoring.

    Args:
        execution_id: Unique execution identifier
        log_directory: Directory for log file storage
        config: Optional logging configuration

    Returns:
        Configured PuLPExecutionLogger instance

    Example:
        >>> logger = create_execution_logger("exec_001", "./logs")
        >>> with logger.measure_operation("solver", "optimization", "CBC solving"):
        ...     # Solver operations here
        ...     logger.log_success("solver", "optimization", "Solution found")
    """
    # Use default config if not provided
    if config is None:
        config = LoggingConfiguration()

    # Create and return logger
    execution_logger = PuLPExecutionLogger(
        execution_id=execution_id,
        log_directory=Path(log_directory),
        config=config
    )

    return execution_logger

if __name__ == "__main__":
    # Example usage and testing
    import tempfile
    import time

    if len(sys.argv) < 2:
        print("Usage: python logging.py <execution_id>")
        sys.exit(1)

    execution_id = sys.argv[1]

    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Create temporary log directory
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "test_logs"

            # Create execution logger
            config = LoggingConfiguration(
                log_level=LogLevel.DEBUG,
                enable_performance_logging=True,
                enable_resource_monitoring=True,
                console_output=True,
                file_output=True
            )

            exec_logger = create_execution_logger(execution_id, log_dir, config)

            print(f"✓ Execution logger created for execution {execution_id}")

            # Test various logging operations
            exec_logger.log_info("test", "initialization", "Starting logging test")

            # Test performance measurement
            with exec_logger.measure_operation("test", "sample_operation", "sample processing"):
                time.sleep(0.1)  # Simulate work
                exec_logger.log_debug("test", "processing", "Processing sample data", {
                    'sample_size': 1000,
                    'processing_method': 'standard'
                })

            # Test error logging
            try:
                raise ValueError("Sample error for testing")
            except Exception as e:
                exec_logger.log_error("test", "error_handling", "Caught sample error", error=e)

            # Test solver logging
            exec_logger.log_solver_operation("CBC", "optimization", "Solver completed successfully", {
                'objective_value': 42.5,
                'solving_time': 1.23,
                'status': 'optimal'
            })

            # Test validation logging
            exec_logger.log_validation_result("test", "validation", "Test validation completed", {
                'success': True,
                'checks_passed': 5,
                'checks_failed': 0
            })

            # Test audit logging
            exec_logger.log_audit("test", "completion", "Test logging completed successfully")

            # Generate summary
            summary = exec_logger.generate_execution_summary()

            print(f"  Execution duration: {summary.total_duration_seconds:.2f} seconds")
            print(f"  Components executed: {len(summary.components_executed)}")
            print(f"  Operations completed: {summary.operations_completed}")
            print(f"  Success rate: {summary.success_rate:.2f}")
            print(f"  Performance grade: {summary.get_performance_grade()}")
            print(f"  Log entries: {summary.log_entry_count}")
            print(f"  Errors: {summary.error_summary.get('errors', 0)}")
            print(f"  Warnings: {summary.error_summary.get('warnings', 0)}")

            # Save summary and close
            summary_file = exec_logger.save_execution_summary()
            print(f"  Summary saved: {summary_file}")

            exec_logger.close()
            print(f"✓ Execution logger test completed successfully")

    except Exception as e:
        print(f"Failed to test execution logger: {str(e)}")
        sys.exit(1)
