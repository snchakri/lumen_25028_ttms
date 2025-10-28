"""
Structured Logger for Stage-2 Batching System
JSON logging system with console output
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional


class StructuredLogger:
    """
    Structured logging system with JSON file output and console logging.
    
    Provides comprehensive logging for debugging and monitoring.
    """
    
    def __init__(self, log_file_path: str):
        """
        Initialize structured logger.
        
        Args:
            log_file_path: Path to save JSON log file
        """
        self.log_file_path = log_file_path
        self.log_records = []
        
        # Configure console logger
        self.console_logger = logging.getLogger('Stage2')
        self.console_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.console_logger.handlers = []
        
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.console_logger.addHandler(console_handler)
    
    def log(
        self,
        level: str,
        stage: str,
        message: str,
        data: Optional[Dict] = None
    ) -> None:
        """
        Structured logging with both console and JSON file output.
        
        Args:
            level: DEBUG, INFO, WARNING, ERROR, CRITICAL
            stage: Processing stage (e.g., 'PREPROCESSING', 'OPTIMIZATION')
            message: Human-readable message
            data: Additional structured data (dict)
        """
        timestamp = datetime.now().isoformat()
        
        log_record = {
            'timestamp': timestamp,
            'level': level,
            'stage': stage,
            'message': message,
            'data': data or {}
        }
        
        self.log_records.append(log_record)
        
        # Console output
        console_message = f"[{stage}] {message}"
        if data:
            console_message += f" | Data: {json.dumps(data, indent=2)}"
        
        getattr(self.console_logger, level.lower())(console_message)
    
    def save_logs(self) -> None:
        """Write all logs to JSON file."""
        with open(self.log_file_path, 'w') as f:
            json.dump({
                'execution_summary': {
                    'total_records': len(self.log_records),
                    'start_time': self.log_records[0]['timestamp'] if self.log_records else None,
                    'end_time': self.log_records[-1]['timestamp'] if self.log_records else None
                },
                'logs': self.log_records
            }, f, indent=2)
    
    def log_preprocessing_start(self, n_students: int, n_courses: int) -> None:
        """Log preprocessing start."""
        self.log(
            'INFO', 'PREPROCESSING',
            f'Starting preprocessing: {n_students} students, {n_courses} courses'
        )
    
    def log_similarity_computation(self, matrix_size: int, computation_time: float) -> None:
        """Log similarity computation."""
        self.log(
            'INFO', 'SIMILARITY',
            f'Computed similarity matrix: {matrix_size}x{matrix_size}',
            {'computation_time_seconds': computation_time}
        )
    
    def log_optimization_start(self, n_variables: int, n_constraints: int) -> None:
        """Log optimization start."""
        self.log(
            'INFO', 'OPTIMIZATION',
            f'CP-SAT model built: {n_variables} variables, {n_constraints} constraints'
        )
    
    def log_optimization_progress(self, iteration: int, objective_value: float) -> None:
        """Log optimization progress."""
        self.log(
            'DEBUG', 'OPTIMIZATION',
            f'Iteration {iteration}: objective = {objective_value}'
        )
    
    def log_solution_found(self, status: str, objective_value: float, solve_time: float) -> None:
        """Log solution found."""
        self.log(
            'INFO', 'OPTIMIZATION',
            f'Solution found: status={status}, objective={objective_value}',
            {'solve_time_seconds': solve_time}
        )
    
    def log_validation_result(self, validation_name: str, passed: bool, details: Dict) -> None:
        """Log validation result."""
        level = 'INFO' if passed else 'WARNING'
        self.log(
            level, 'VALIDATION',
            f'{validation_name}: {"PASSED" if passed else "FAILED"}',
            details
        )
    
    def log_compliance_score(self, compliance_score: float) -> None:
        """Log foundation compliance score."""
        self.log(
            'INFO', 'COMPLIANCE',
            f'Foundation compliance score: {compliance_score:.2f}%'
        )

