"""
Fallback Manager Module

Implements fallback mechanisms for graceful degradation when primary solvers fail.

Fallback Chain:
1. Primary: Specified solver
2. Secondary: NSGA-II (if not already)
3. Tertiary: Greedy heuristic for feasible solution
"""

from typing import Dict, Any, Optional
from pathlib import Path

from ..config import PyGMOConfig
from ..logging_system.logger import StructuredLogger


class FallbackManager:
    """
    Manages fallback mechanisms for solver failures.
    """
    
    def __init__(self, config: PyGMOConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        
        # Fallback chain
        self.fallback_chain = ['NSGA-II', 'Differential Evolution', 'PSO', 'Greedy']
        self.current_fallback_index = 0
        
        self.logger.info("FallbackManager initialized successfully.")
    
    def execute_fallback(self, error_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes fallback mechanism based on error report.
        
        Args:
            error_report: Error report from ErrorReporter
        
        Returns:
            Dictionary with fallback execution result
        """
        self.logger.info("Executing fallback mechanism...")
        
        error_type = error_report.get('error_type', '')
        severity = error_report.get('severity', 'UNKNOWN')
        
        # Determine fallback strategy based on error
        if 'Memory' in error_type:
            return self._handle_memory_error(error_report)
        elif 'Optimization' in error_type or 'Convergence' in error_type:
            return self._handle_optimization_error(error_report)
        elif 'Input' in error_type or 'Validation' in error_type:
            return self._handle_input_error(error_report)
        else:
            return self._handle_generic_error(error_report)
    
    def _handle_memory_error(self, error_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles memory-related errors by reducing resource usage.
        """
        self.logger.warning("Memory error detected. Applying memory reduction strategies.")
        
        # Reduce memory-intensive parameters
        adjustments = {
            'population_size': max(20, self.config.population_size // 2),
            'num_islands': max(2, self.config.num_islands // 2),
            'generations': self.config.generations  # Keep generations
        }
        
        # Apply adjustments
        for key, value in adjustments.items():
            if hasattr(self.config, key):
                original = getattr(self.config, key)
                setattr(self.config, key, value)
                self.logger.info(f"Adjusted {key}: {original} -> {value}")
        
        return {
            'success': True,
            'strategy': 'memory_reduction',
            'message': 'Memory parameters reduced',
            'adjustments': adjustments
        }
    
    def _handle_optimization_error(self, error_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles optimization convergence errors by switching algorithms.
        """
        self.logger.warning("Optimization error detected. Attempting algorithm switch.")
        
        # Try next fallback solver
        if self.current_fallback_index < len(self.fallback_chain):
            new_solver = self.fallback_chain[self.current_fallback_index]
            self.logger.info(f"Switching to fallback solver: {new_solver}")
            
            self.config.default_solver = new_solver
            self.current_fallback_index += 1
            
            return {
                'success': True,
                'strategy': 'algorithm_switch',
                'message': f'Switched to {new_solver}',
                'new_solver': new_solver
            }
        else:
            self.logger.error("All fallback algorithms exhausted.")
            return {
                'success': False,
                'strategy': 'algorithm_switch',
                'message': 'All fallback algorithms exhausted'
            }
    
    def _handle_input_error(self, error_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles input validation errors.
        """
        self.logger.error("Input error detected. Cannot proceed with optimization.")
        
        # Input errors are typically non-recoverable
        return {
            'success': False,
            'strategy': 'input_validation',
            'message': 'Input validation failed. Cannot proceed.',
            'required_action': 'Fix Stage 3 outputs and retry'
        }
    
    def _handle_generic_error(self, error_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles generic errors with default fallback strategy.
        """
        self.logger.warning("Generic error detected. Applying default fallback.")
        
        # Try algorithm switch as default
        return self._handle_optimization_error(error_report)
    
    def reset_fallback_chain(self):
        """
        Resets the fallback chain to initial state.
        """
        self.current_fallback_index = 0
        self.logger.debug("Fallback chain reset.")


