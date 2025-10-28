"""
Comprehensive Logging System

Dual logging: console (human-readable) and JSON file (machine-parsable).

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime
import os


class ComprehensiveLogger:
    """
    Comprehensive logging system with dual output.
    
    - Console: Human-readable, INFO level, progress indicators
    - JSON File: Machine-parsable, DEBUG level, complete data
    """
    
    def __init__(self, log_path: Path, log_level: str = "INFO", log_console: bool = True, log_file: bool = True):
        self.log_path = Path(log_path)
        self.log_level = log_level
        self.log_console = log_console
        self.log_file = log_file
        
        # Create log directory
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Setup console logger
        self.console_logger = None
        if self.log_console:
            self._setup_console_logger()
        
        # Setup file logger
        self.file_logger = None
        self.json_log_file = None
        if self.log_file:
            self._setup_file_logger()
        
        # Statistics
        self.log_count = 0
        self.max_log_size_mb = 100  # Rotate at 100MB
    
    def _setup_console_logger(self):
        """Setup console logger."""
        self.console_logger = logging.getLogger('console')
        self.console_logger.setLevel(getattr(logging, self.log_level))
        
        # Remove existing handlers
        self.console_logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.log_level))
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        self.console_logger.addHandler(console_handler)
        self.console_logger.propagate = False
    
    def _setup_file_logger(self):
        """Setup file logger."""
        # JSON log file
        json_log_path = self.log_path / f"deap_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.json_log_file = open(json_log_path, 'w')
        
        # Write opening bracket
        self.json_log_file.write('[\n')
        self.json_log_file.flush()
    
    def get_logger(self) -> logging.Logger:
        """Get the console logger."""
        return self.console_logger
    
    def log(self, level: str, message: str, data: Dict[str, Any] = None):
        """
        Log message with data.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Human-readable message
            data: Additional data for JSON log
        """
        # Console log
        if self.console_logger:
            log_func = getattr(self.console_logger, level.lower(), self.console_logger.info)
            log_func(message)
        
        # JSON log
        if self.json_log_file:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'message': message,
                'data': data or {}
            }
            
            # Add comma if not first entry
            if self.log_count > 0:
                self.json_log_file.write(',\n')
            
            json.dump(log_entry, self.json_log_file, indent=2)
            self.json_log_file.flush()
            
            self.log_count += 1
    
    def log_phase_start(self, phase_name: str):
        """Log phase start."""
        self.log('INFO', f"Starting: {phase_name}")
    
    def log_phase_end(self, phase_name: str, success: bool, duration: float):
        """Log phase end."""
        status = "SUCCESS" if success else "FAILED"
        self.log('INFO', f"Completed: {phase_name} - {status} ({duration:.2f}s)")
    
    def log_generation(self, generation: int, stats: Dict[str, Any]):
        """
        Log generation statistics.
        
        Args:
            generation: Generation number
            stats: Statistics dictionary
        """
        self.log('INFO', f"Generation {generation}", {
            'generation': generation,
            'fitness_best': stats.get('fitness_best'),
            'fitness_mean': stats.get('fitness_mean'),
            'fitness_std': stats.get('fitness_std'),
            'diversity': stats.get('diversity'),
            'population_size': stats.get('size'),
        })
    
    def log_fitness_evolution(self, generation: int, fitness_components: Tuple[float, ...]):
        """Log fitness component evolution."""
        self.log('DEBUG', f"Fitness components at generation {generation}", {
            'generation': generation,
            'f_1': fitness_components[0] if len(fitness_components) > 0 else None,
            'f_2': fitness_components[1] if len(fitness_components) > 1 else None,
            'f_3': fitness_components[2] if len(fitness_components) > 2 else None,
            'f_4': fitness_components[3] if len(fitness_components) > 3 else None,
            'f_5': fitness_components[4] if len(fitness_components) > 4 else None,
        })
    
    def log_constraint_violations(self, generation: int, violations: Dict[str, int]):
        """Log constraint violations."""
        self.log('DEBUG', f"Constraint violations at generation {generation}", {
            'generation': generation,
            'violations': violations,
        })
    
    def log_operator_effectiveness(self, generation: int, crossover_success: float, mutation_success: float):
        """Log operator effectiveness."""
        self.log('DEBUG', f"Operator effectiveness at generation {generation}", {
            'generation': generation,
            'crossover_success_rate': crossover_success,
            'mutation_success_rate': mutation_success,
        })
    
    def log_theorem_validation(self, theorem_name: str, passed: bool, details: Dict[str, Any]):
        """Log theorem validation results."""
        status = "PASS" if passed else "FAIL"
        self.log('INFO', f"Theorem validation: {theorem_name} - {status}", {
            'theorem': theorem_name,
            'passed': passed,
            'details': details,
        })
    
    def close(self):
        """Close loggers and files."""
        # Close JSON log file
        if self.json_log_file:
            # Write closing bracket
            self.json_log_file.write('\n]')
            self.json_log_file.close()
    
    def __del__(self):
        """Destructor to ensure files are closed."""
        self.close()

