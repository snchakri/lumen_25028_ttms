"""
Structured Logging System
Comprehensive logging framework for the 7-stage scheduling engine with
mathematical rigor, performance monitoring, and audit trails.
"""

import logging
import structlog
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import sys
import traceback

class LogLevel(str, Enum):
    """Log levels for structured logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogCategory(str, Enum):
    """Log categories for different system components."""
    STAGE_1 = "STAGE_1_VALIDATION"
    STAGE_2 = "STAGE_2_BATCHING"
    STAGE_3 = "STAGE_3_COMPILATION"
    STAGE_4 = "STAGE_4_FEASIBILITY"
    STAGE_5 = "STAGE_5_COMPLEXITY"
    STAGE_6 = "STAGE_6_OPTIMIZATION"
    STAGE_7 = "STAGE_7_VALIDATION"
    PIPELINE = "PIPELINE_ORCHESTRATION"
    MATHEMATICAL = "MATHEMATICAL_FRAMEWORK"
    PERFORMANCE = "PERFORMANCE_MONITORING"
    AUDIT = "AUDIT_TRAIL"
    ERROR = "ERROR_HANDLING"

@dataclass
class LogEntry:
    """Structured log entry with comprehensive metadata."""
    timestamp: str
    level: str
    category: str
    stage: Optional[str]
    component: str
    message: str
    execution_id: str
    session_id: str
    performance_metrics: Optional[Dict[str, Any]] = None
    mathematical_metrics: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None

class StructuredLogger:
    """
    Structured logger with mathematical rigor and performance monitoring.
    Implements comprehensive logging for all 7 stages with audit trails.
    """
    
    def __init__(self, output_dir: Path = None, session_id: str = None):
        self.output_dir = output_dir or Path("logs")
        self.output_dir.mkdir(exist_ok=True)
        
        self.session_id = session_id or str(uuid.uuid4())
        self.execution_id = str(uuid.uuid4())
        
        # Create subdirectories for different log types
        self.stage_logs_dir = self.output_dir / "stage_logs"
        self.performance_logs_dir = self.output_dir / "performance_logs"
        self.audit_logs_dir = self.output_dir / "audit_logs"
        self.error_logs_dir = self.output_dir / "error_logs"
        self.mathematical_logs_dir = self.output_dir / "mathematical_logs"
        
        for log_dir in [self.stage_logs_dir, self.performance_logs_dir, 
                       self.audit_logs_dir, self.error_logs_dir, self.mathematical_logs_dir]:
            log_dir.mkdir(exist_ok=True)
        
        # Configure structlog
        self._configure_structlog()
        
        # Initialize loggers for different categories
        self.loggers = {
            category: structlog.get_logger(category.value)
            for category in LogCategory
        }
        
        # Performance tracking
        self.performance_tracker = {}
        self.mathematical_metrics = {}
        
        # Log session start
        self.log_session_start()
    
    def _configure_structlog(self):
        """Configure structlog with JSON formatting and file handlers."""
        
        # Configure processors
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ]
        
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Configure standard logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "main.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def log_session_start(self):
        """Log session start with system information."""
        session_info = {
            "session_id": self.session_id,
            "execution_id": self.execution_id,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": sys.platform,
            "log_directory": str(self.output_dir)
        }
        
        self.loggers[LogCategory.AUDIT].info(
            "Session started",
            **session_info,
            category="SESSION_START"
        )
    
    def log_stage_start(self, stage: int, stage_name: str, **context):
        """Log the start of a stage execution."""
        stage_info = {
            "stage": stage,
            "stage_name": stage_name,
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "start_time": datetime.now(timezone.utc).isoformat(),
            **context
        }
        
        self.loggers[LogCategory.PIPELINE].info(
            f"Stage {stage} ({stage_name}) started",
            **stage_info,
            category="STAGE_START"
        )
        
        # Track performance
        self.performance_tracker[f"stage_{stage}_start"] = time.time()
    
    def log_stage_completion(self, stage: int, stage_name: str, 
                           success: bool, **metrics):
        """Log the completion of a stage execution."""
        end_time = time.time()
        start_time = self.performance_tracker.get(f"stage_{stage}_start", end_time)
        execution_time = end_time - start_time
        
        stage_info = {
            "stage": stage,
            "stage_name": stage_name,
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "success": success,
            "execution_time_seconds": execution_time,
            **metrics
        }
        
        log_level = "info" if success else "error"
        self.loggers[LogCategory.PIPELINE].info(
            f"Stage {stage} ({stage_name}) completed",
            **stage_info,
            category="STAGE_COMPLETION"
        )
        
        # Log to performance logs
        self.loggers[LogCategory.PERFORMANCE].info(
            f"Stage {stage} performance metrics",
            **stage_info,
            category="PERFORMANCE_METRICS"
        )
    
    def log_mathematical_validation(self, stage: int, validation_type: str,
                                  result: Dict[str, Any], **context):
        """Log mathematical validation results."""
        math_info = {
            "stage": stage,
            "validation_type": validation_type,
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_result": result,
            **context
        }
        
        self.loggers[LogCategory.MATHEMATICAL].info(
            f"Mathematical validation: {validation_type}",
            **math_info,
            category="MATHEMATICAL_VALIDATION"
        )
        
        # Store mathematical metrics
        self.mathematical_metrics[f"{stage}_{validation_type}"] = result
    
    def log_performance_metrics(self, stage: int, component: str,
                              metrics: Dict[str, Any], **context):
        """Log detailed performance metrics."""
        perf_info = {
            "stage": stage,
            "component": component,
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "performance_metrics": metrics,
            **context
        }
        
        self.loggers[LogCategory.PERFORMANCE].info(
            f"Performance metrics: {component}",
            **perf_info,
            category="PERFORMANCE_METRICS"
        )
    
    def log_error(self, stage: int, component: str, error: Exception,
                 context: Dict[str, Any] = None):
        """Log errors with full context and stack trace."""
        error_info = {
            "stage": stage,
            "component": component,
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "stack_trace": traceback.format_exc(),
            "context": context or {}
        }
        
        self.loggers[LogCategory.ERROR].error(
            f"Error in {component}",
            **error_info,
            category="ERROR_OCCURRED"
        )
    
    def log_audit_event(self, event_type: str, details: Dict[str, Any],
                       **context):
        """Log audit events for compliance and tracking."""
        audit_info = {
            "event_type": event_type,
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
            **context
        }
        
        self.loggers[LogCategory.AUDIT].info(
            f"Audit event: {event_type}",
            **audit_info,
            category="AUDIT_EVENT"
        )
    
    def generate_session_report(self) -> Dict[str, Any]:
        """Generate comprehensive session report."""
        report = {
            "session_id": self.session_id,
            "execution_id": self.execution_id,
            "session_duration": time.time() - self.performance_tracker.get("session_start", time.time()),
            "stages_completed": len([k for k in self.performance_tracker.keys() if k.endswith("_start")]),
            "mathematical_validations": len(self.mathematical_metrics),
            "performance_summary": self._calculate_performance_summary(),
            "mathematical_summary": self._calculate_mathematical_summary(),
            "error_summary": self._calculate_error_summary(),
            "log_files": self._get_log_files(),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Save session report
        report_path = self.output_dir / f"session_report_{self.session_id}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _calculate_performance_summary(self) -> Dict[str, Any]:
        """Calculate performance summary from tracked metrics."""
        # Implementation for performance summary calculation
        return {
            "total_execution_time": sum(
                self.performance_tracker.get(f"stage_{i}_end", 0) - 
                self.performance_tracker.get(f"stage_{i}_start", 0)
                for i in range(1, 8)
            ),
            "average_stage_time": 0,  # Calculate from tracked data
            "peak_memory_usage": 0,   # Extract from performance logs
            "total_memory_usage": 0   # Extract from performance logs
        }
    
    def _calculate_mathematical_summary(self) -> Dict[str, Any]:
        """Calculate mathematical validation summary."""
        return {
            "total_validations": len(self.mathematical_metrics),
            "validation_types": list(set(
                k.split('_', 1)[1] for k in self.mathematical_metrics.keys()
            )),
            "compliance_score": self._calculate_compliance_score()
        }
    
    def _calculate_error_summary(self) -> Dict[str, Any]:
        """Calculate error summary."""
        return {
            "total_errors": 0,  # Count from error logs
            "error_types": [],  # Extract from error logs
            "critical_errors": 0  # Count critical errors
        }
    
    def _calculate_compliance_score(self) -> float:
        """Calculate overall mathematical framework compliance score."""
        if not self.mathematical_metrics:
            return 0.0
        
        # Calculate based on validation results
        total_score = 0
        count = 0
        
        for validation_result in self.mathematical_metrics.values():
            if isinstance(validation_result, dict) and 'score' in validation_result:
                total_score += validation_result['score']
                count += 1
        
        return total_score / count if count > 0 else 0.0
    
    def _get_log_files(self) -> Dict[str, str]:
        """Get list of generated log files."""
        log_files = {}
        
        for log_dir in [self.stage_logs_dir, self.performance_logs_dir,
                       self.audit_logs_dir, self.error_logs_dir, self.mathematical_logs_dir]:
            if log_dir.exists():
                log_files[log_dir.name] = [str(f) for f in log_dir.glob("*.log")]
        
        return log_files

class LoggingSystemManager:
    """
    Manager for the logging system across all stages.
    Provides centralized logging configuration and management.
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("logs")
        self.session_logger = None
        self.stage_loggers = {}
    
    def start_session(self, session_id: str = None) -> StructuredLogger:
        """Start a new logging session."""
        self.session_logger = StructuredLogger(self.output_dir, session_id)
        return self.session_logger
    
    def get_stage_logger(self, stage: int) -> StructuredLogger:
        """Get logger for specific stage."""
        if stage not in self.stage_loggers:
            stage_output_dir = self.output_dir / f"stage_{stage}_logs"
            self.stage_loggers[stage] = StructuredLogger(stage_output_dir)
        
        return self.stage_loggers[stage]
    
    def end_session(self) -> Dict[str, Any]:
        """End current session and generate final report."""
        if self.session_logger:
            return self.session_logger.generate_session_report()
        return {}

# Factory function
def create_logging_system(output_dir: Path = None) -> LoggingSystemManager:
    """
    Create a logging system manager.
    
    Args:
        output_dir: Directory for log files
        
    Returns:
        Configured LoggingSystemManager
    """
    return LoggingSystemManager(output_dir)

