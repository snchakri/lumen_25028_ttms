# STAGE 5 - COMMON/LOGGING.PY
# complete Structured JSON Logging Framework

"""
STAGE 5 COMMON LOGGING FRAMEWORK
complete Structured JSON Logging for Audit Trails and Debugging

This module provides complete structured logging capabilities for Stage 5's
rigorous execution tracking and debugging requirements. Every log entry follows
JSON schema specifications for downstream analysis and monitoring integration.

Critical Implementation Notes:
- STRUCTURED JSON LOGGING: All log entries output as parseable JSON for aggregation
- complete CONTEXT: Every log includes execution context, performance metrics, and debugging info
- AUDIT TRAIL COMPLIANCE: Log entries enable full execution reconstruction and analysis
- CURSOR/PyCharm IDE SUPPORT: Full type hints and docstrings for development assistance
- ENTERPRISE INTEGRATION: Compatible with standard logging aggregation and monitoring systems

Logging Architecture:
1. HIERARCHICAL LOGGERS: Stage-specific loggers with inheritance (stage5, stage5.stage_5_1, stage5.stage_5_2)
2. STRUCTURED FORMATTING: JSON formatters with consistent schema for all log entries
3. MULTI-DESTINATION OUTPUT: Console output for development, file output for production
4. CONTEXTUAL ENRICHMENT: Automatic addition of execution context, timestamps, and metadata
5. PERFORMANCE TRACKING: Built-in performance metrics and execution timing

Integration Points:
- Exception handling: Automatic exception context logging with stack traces
- Parameter computation: Mathematical computation progress and result logging
- Solver selection: LP optimization iterations and convergence tracking
- File I/O operations: Input/output validation and processing status
- API endpoints: Request/response logging with performance metrics

References:
- Stage5-FOUNDATIONAL-DESIGN-IMPLEMENTATION-PLAN.md: JSON logging specifications
- Python logging module: Built on standard Python logging infrastructure
- Structured logging best practices: Industry standards for production logging
"""

import json
import logging
import logging.config
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from contextlib import contextmanager
import traceback
import os

# =============================================================================
# STRUCTURED JSON FORMATTER
# Core formatting class for consistent JSON log output
# =============================================================================

class Stage5JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for Stage 5 structured logging.
    Converts all log records to structured JSON format with consistent schema.
    
    This formatter ensures every log entry contains:
    - Standard fields: timestamp, level, logger_name, message
    - Stage 5 context: stage_component, execution_id, operation_type
    - Performance data: duration_ms, memory_usage_mb (when available)
    - Error context: exception_type, stack_trace (for ERROR/CRITICAL logs)
    - Custom fields: Any additional fields added via logging extra parameter
    
    The JSON schema is designed for compatibility with log aggregation systems
    like ELK stack, Splunk, or cloud logging services.
    """
    
    def __init__(self):
        """Initialize JSON formatter with Stage 5 specific configuration."""
        super().__init__()
        
    def format(self, record: logging.LogRecord) -> str:
        """
        Format logging record as structured JSON.
        
        Args:
            record: Python logging record to format
            
        Returns:
            JSON string with structured log data
        """
        # Base log entry structure
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger_name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line_number': record.lineno
        }
        
        # Add Stage 5 specific context if available
        stage_context = self._extract_stage_context(record)
        if stage_context:
            log_entry['stage5_context'] = stage_context
            
        # Add performance metrics if available
        performance_data = self._extract_performance_data(record)
        if performance_data:
            log_entry['performance'] = performance_data
            
        # Add exception information for error logs
        if record.exc_info:
            log_entry['exception'] = self._format_exception(record.exc_info)
            
        # Add any additional fields from log record extras
        extra_fields = self._extract_extra_fields(record)
        if extra_fields:
            log_entry['extra'] = extra_fields
            
        return json.dumps(log_entry, ensure_ascii=False, separators=(',', ':'))
    
    def _extract_stage_context(self, record: logging.LogRecord) -> Optional[Dict[str, Any]]:
        """Extract Stage 5 specific context from log record."""
        context = {}
        
        # Stage 5 context fields
        context_fields = [
            'execution_id', 'stage_component', 'operation_type', 
            'parameter_id', 'solver_id', 'file_path', 'computation_type'
        ]
        
        for field in context_fields:
            if hasattr(record, field):
                context[field] = getattr(record, field)
                
        return context if context else None
    
    def _extract_performance_data(self, record: logging.LogRecord) -> Optional[Dict[str, Any]]:
        """Extract performance metrics from log record."""
        performance = {}
        
        # Performance fields
        perf_fields = [
            'duration_ms', 'memory_usage_mb', 'cpu_usage_percent',
            'parameter_count', 'solver_count', 'iteration_count'
        ]
        
        for field in perf_fields:
            if hasattr(record, field):
                performance[field] = getattr(record, field)
                
        return performance if performance else None
    
    def _format_exception(self, exc_info) -> Dict[str, Any]:
        """Format exception information for structured logging."""
        exc_type, exc_value, exc_traceback = exc_info
        
        return {
            'type': exc_type.__name__ if exc_type else None,
            'message': str(exc_value) if exc_value else None,
            'stack_trace': traceback.format_exception(exc_type, exc_value, exc_traceback)
        }
    
    def _extract_extra_fields(self, record: logging.LogRecord) -> Optional[Dict[str, Any]]:
        """Extract additional fields from log record extras."""
        # Standard logging record attributes to exclude
        standard_attrs = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
            'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
            'thread', 'threadName', 'processName', 'process', 'exc_info', 'exc_text',
            'stack_info', 'message', 'taskName'
        }
        
        # Stage 5 context attributes (already handled)
        stage5_attrs = {
            'execution_id', 'stage_component', 'operation_type', 'parameter_id',
            'solver_id', 'file_path', 'computation_type', 'duration_ms',
            'memory_usage_mb', 'cpu_usage_percent', 'parameter_count',
            'solver_count', 'iteration_count'
        }
        
        extra = {}
        for key, value in record.__dict__.items():
            if key not in standard_attrs and key not in stage5_attrs:
                extra[key] = value
                
        return extra if extra else None

# =============================================================================
# LOGGING CONFIGURATION MANAGER
# Centralized logging setup and configuration management
# =============================================================================

class Stage5LoggingConfig:
    """
    Centralized logging configuration manager for Stage 5.
    Handles logger setup, file rotation, and environment-specific configurations.
    
    This class provides:
    - Environment-aware configuration (development vs production)
    - Hierarchical logger setup for Stage 5 components
    - File rotation and retention policies
    - Console output formatting for development
    - JSON file output for production and aggregation
    
    Logger Hierarchy:
    - stage5: Root logger for all Stage 5 operations
    - stage5.stage_5_1: Stage 5.1 complexity analysis operations
    - stage5.stage_5_2: Stage 5.2 solver selection operations  
    - stage5.common: Common utilities and shared operations
    - stage5.api: REST API endpoint operations
    """
    
    def __init__(self, log_directory: Optional[str] = None, log_level: str = "INFO"):
        """
        Initialize logging configuration manager.
        
        Args:
            log_directory: Directory for log files (None for console-only in dev)
            log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_directory = Path(log_directory) if log_directory else None
        self.log_level = log_level.upper()
        self.formatters = {}
        self.handlers = {}
        
        # Create log directory if specified
        if self.log_directory:
            self.log_directory.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self) -> None:
        """
        Configure all Stage 5 loggers with appropriate handlers and formatters.
        Sets up hierarchical logger structure with JSON formatting.
        """
        # Setup formatters
        self._setup_formatters()
        
        # Setup handlers
        self._setup_handlers()
        
        # Setup loggers
        self._setup_loggers()
        
        # Configure root Stage 5 logger
        self._configure_root_logger()
    
    def _setup_formatters(self) -> None:
        """Setup logging formatters for different output destinations."""
        # JSON formatter for structured logging
        self.formatters['json'] = Stage5JSONFormatter()
        
        # Console formatter for human-readable development output
        self.formatters['console'] = logging.Formatter(
            fmt='%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _setup_handlers(self) -> None:
        """Setup logging handlers for different output destinations."""
        # Console handler for development and immediate output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatters['console'])
        console_handler.setLevel(self.log_level)
        self.handlers['console'] = console_handler
        
        # File handlers for production logging (if log directory specified)
        if self.log_directory:
            # Main Stage 5 log file with JSON format
            main_log_file = self.log_directory / 'stage5_execution.log'
            main_handler = logging.FileHandler(main_log_file, mode='a', encoding='utf-8')
            main_handler.setFormatter(self.formatters['json'])
            main_handler.setLevel(self.log_level)
            self.handlers['main'] = main_handler
            
            # Error-specific log file for critical issues
            error_log_file = self.log_directory / 'stage5_errors.log'
            error_handler = logging.FileHandler(error_log_file, mode='a', encoding='utf-8')
            error_handler.setFormatter(self.formatters['json'])
            error_handler.setLevel('ERROR')
            self.handlers['error'] = error_handler
            
            # Performance log file for metrics and timing
            perf_log_file = self.log_directory / 'stage5_performance.log'
            perf_handler = logging.FileHandler(perf_log_file, mode='a', encoding='utf-8')
            perf_handler.setFormatter(self.formatters['json'])
            perf_handler.setLevel('INFO')
            # Add filter for performance-specific logs
            perf_handler.addFilter(lambda record: hasattr(record, 'performance_metric'))
            self.handlers['performance'] = perf_handler
    
    def _setup_loggers(self) -> None:
        """Setup hierarchical Stage 5 logger structure."""
        logger_configs = [
            # Root Stage 5 logger
            {
                'name': 'stage5',
                'level': self.log_level,
                'handlers': ['console'] + (['main', 'error'] if self.log_directory else [])
            },
            # Stage 5.1 complexity analysis logger
            {
                'name': 'stage5.stage_5_1',
                'level': self.log_level,
                'handlers': []  # Inherits from parent
            },
            # Stage 5.2 solver selection logger
            {
                'name': 'stage5.stage_5_2', 
                'level': self.log_level,
                'handlers': []  # Inherits from parent
            },
            # Common utilities logger
            {
                'name': 'stage5.common',
                'level': self.log_level,
                'handlers': []  # Inherits from parent
            },
            # API endpoint logger
            {
                'name': 'stage5.api',
                'level': self.log_level,
                'handlers': []  # Inherits from parent
            }
        ]
        
        for config in logger_configs:
            logger = logging.getLogger(config['name'])
            logger.setLevel(config['level'])
            
            # Add handlers if specified
            for handler_name in config['handlers']:
                if handler_name in self.handlers:
                    logger.addHandler(self.handlers[handler_name])
            
            # Prevent log propagation to root logger to avoid duplicates
            if config['name'] != 'stage5':
                logger.propagate = True  # Allow propagation to stage5 parent
    
    def _configure_root_logger(self) -> None:
        """Configure Python root logger to prevent interference."""
        # Ensure Stage 5 logs don't propagate to root logger
        stage5_logger = logging.getLogger('stage5')
        stage5_logger.propagate = False
        
        # Optionally configure root logger level to prevent other library noise
        logging.getLogger().setLevel('WARNING')

# =============================================================================
# CONTEXTUAL LOGGING UTILITIES
# Helper functions and context managers for enhanced logging
# =============================================================================

class Stage5LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter for automatic Stage 5 context injection.
    Automatically adds execution context to all log entries for consistency.
    
    This adapter ensures every log entry includes:
    - execution_id: Unique identifier for the current execution
    - stage_component: Which Stage 5 component is logging (5.1, 5.2, common)
    - operation_type: Type of operation being performed
    - Additional context fields as needed
    """
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]):
        """
        Initialize logger adapter with Stage 5 context.
        
        Args:
            logger: Base logger to adapt
            extra: Context fields to add to all log entries
        """
        super().__init__(logger, extra)
    
    def process(self, msg: Any, kwargs: Dict[str, Any]) -> tuple:
        """
        Process log message and kwargs to inject Stage 5 context.
        
        Args:
            msg: Log message
            kwargs: Log call keyword arguments
            
        Returns:
            Tuple of (message, kwargs) with context injected
        """
        # Ensure extra dict exists in kwargs
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        # Merge adapter context with call-specific context
        kwargs['extra'].update(self.extra)
        
        return msg, kwargs

@contextmanager
def performance_timing_context(
    logger: logging.Logger,
    operation_name: str,
    **context
):
    """
    Context manager for automatic performance timing and logging.
    
    Automatically logs operation start/completion with timing metrics.
    Useful for tracking Stage 5 operation performance and identifying bottlenecks.
    
    Args:
        logger: Logger instance to use for timing logs
        operation_name: Name of the operation being timed
        **context: Additional context to include in timing logs
    
    Usage:
        with performance_timing_context(logger, "parameter_computation", parameter_id="P1"):
            result = compute_parameter_p1(data)
    """
    start_time = time.time()
    start_memory = _get_memory_usage()
    
    # Log operation start
    logger.info(
        f"Starting operation: {operation_name}",
        extra={
            'operation_type': operation_name,
            'operation_status': 'started',
            **context
        }
    )
    
    try:
        yield
        
        # Log successful completion
        end_time = time.time()
        end_memory = _get_memory_usage()
        duration_ms = round((end_time - start_time) * 1000, 2)
        
        logger.info(
            f"Completed operation: {operation_name}",
            extra={
                'operation_type': operation_name,
                'operation_status': 'completed',
                'duration_ms': duration_ms,
                'memory_usage_mb': end_memory,
                'performance_metric': True,
                **context
            }
        )
        
    except Exception as e:
        # Log operation failure
        end_time = time.time()
        duration_ms = round((end_time - start_time) * 1000, 2)
        
        logger.error(
            f"Failed operation: {operation_name} - {str(e)}",
            extra={
                'operation_type': operation_name,
                'operation_status': 'failed',
                'duration_ms': duration_ms,
                'error_type': type(e).__name__,
                **context
            },
            exc_info=True
        )
        raise

def _get_memory_usage() -> float:
    """
    Get current memory usage in MB.
    
    Returns:
        Memory usage in megabytes, or 0.0 if unable to determine
    """
    try:
        import psutil
        process = psutil.Process()
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except ImportError:
        # psutil not available, return 0
        return 0.0
    except Exception:
        # Any other error in memory monitoring
        return 0.0

# =============================================================================
# CONVENIENCE FUNCTIONS
# Easy-to-use functions for Stage 5 logging setup and usage
# =============================================================================

def setup_stage5_logging(
    log_directory: Optional[str] = None,
    log_level: str = "INFO",
    execution_id: Optional[str] = None
) -> logging.Logger:
    """
    Convenience function to setup Stage 5 logging with standard configuration.
    
    Args:
        log_directory: Directory for log files (None for console-only)
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        execution_id: Unique execution identifier for log context
        
    Returns:
        Configured Stage 5 root logger with context adapter
    """
    # Setup logging configuration
    config = Stage5LoggingConfig(log_directory=log_directory, log_level=log_level)
    config.setup_logging()
    
    # Get base logger
    base_logger = logging.getLogger('stage5')
    
    # Add execution context if provided
    if execution_id:
        context = {'execution_id': execution_id}
        logger = Stage5LoggerAdapter(base_logger, context)
    else:
        logger = base_logger
    
    return logger

def get_stage5_logger(
    component: str,
    execution_id: Optional[str] = None,
    **context
) -> Union[logging.Logger, Stage5LoggerAdapter]:
    """
    Get a Stage 5 component-specific logger with optional context.
    
    Args:
        component: Stage 5 component name ('stage_5_1', 'stage_5_2', 'common', 'api')
        execution_id: Unique execution identifier
        **context: Additional context fields for the logger
        
    Returns:
        Logger instance with Stage 5 context
    """
    logger_name = f'stage5.{component}'
    base_logger = logging.getLogger(logger_name)
    
    # Add context if provided
    if execution_id or context:
        adapter_context = {}
        if execution_id:
            adapter_context['execution_id'] = execution_id
        adapter_context.update(context)
        
        return Stage5LoggerAdapter(base_logger, adapter_context)
    
    return base_logger

def log_stage5_exception(
    logger: logging.Logger,
    exception: Exception,
    operation_context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log Stage 5 exception with complete context and stack trace.
    
    Args:
        logger: Logger instance to use
        exception: Exception instance to log
        operation_context: Additional context about the operation that failed
    """
    context = {
        'exception_type': type(exception).__name__,
        'operation_status': 'failed'
    }
    
    if operation_context:
        context.update(operation_context)
    
    # Add Stage 5 exception context if available
    if hasattr(exception, 'to_dict'):
        context['stage5_exception_context'] = exception.to_dict()
    
    logger.error(
        f"Stage 5 Exception: {str(exception)}",
        extra=context,
        exc_info=True
    )

# Export all logging components
__all__ = [
    'Stage5JSONFormatter', 'Stage5LoggingConfig', 'Stage5LoggerAdapter',
    'performance_timing_context', 'setup_stage5_logging', 'get_stage5_logger',
    'log_stage5_exception'
]

print("✅ STAGE 5 COMMON/LOGGING.PY - COMPLETE")
print("   - complete structured JSON logging framework")
print("   - Hierarchical logger setup for Stage 5 components")
print("   - Performance timing context managers and memory tracking")
print("   - Stage 5 exception logging with complete context")
print("   - Production-ready file handlers and console output")
print(f"   - Total logging components exported: {len(__all__)}")