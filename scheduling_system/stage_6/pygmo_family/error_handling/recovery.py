"""
Recovery Manager Module

Implements recovery strategies for handling errors during optimization.

Recovery strategies:
- Checkpoint restoration
- Parameter adjustment
- Algorithm switching
- Partial result extraction
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pickle

from ..config import PyGMOConfig
from ..logging_system.logger import StructuredLogger


class RecoveryManager:
    """
    Manages error recovery strategies for the PyGMO solver.
    """
    
    def __init__(self, config: PyGMOConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.checkpoint_dir = config.checkpoint_dir
        self.logger.info("RecoveryManager initialized successfully.")
        self.logger.info("RecoveryManager initialized successfully.")
    
    def attempt_recovery(self, error_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempts to recover from an error using available strategies.
        
        Args:
            error_report: Error report from ErrorReporter
        
        Returns:
            Dictionary with recovery status and results
        """
        self.logger.info("Attempting error recovery...")
        
        error_type = error_report.get('error_type', '')
        severity = error_report.get('severity', 'UNKNOWN')
        recoverable = error_report.get('recoverable', False)
        
        if not recoverable:
            self.logger.error(f"Error is not recoverable: {error_type}")
            return {
                'success': False,
                'strategy': None,
                'message': 'Error is not recoverable'
            }
        
        # Try recovery strategies in order of preference
        strategies = [
            self._try_checkpoint_restoration,
            self._try_parameter_adjustment,
            self._try_algorithm_switch,
            self._try_partial_result_extraction
        ]
        
        for strategy in strategies:
            try:
                result = strategy(error_report)
                if result['success']:
                    self.logger.info(f"Recovery successful using strategy: {result['strategy']}")
                    return result
            except Exception as e:
                self.logger.warning(f"Recovery strategy failed: {e}")
                continue
        
        self.logger.error("All recovery strategies failed.")
        return {
            'success': False,
            'strategy': None,
            'message': 'All recovery strategies failed'
        }
    
    def _try_checkpoint_restoration(self, error_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempts to restore from the latest checkpoint.
        """
        self.logger.info("Trying checkpoint restoration...")
        
        if not self.config.enable_checkpoints:
            return {
                'success': False,
                'strategy': 'checkpoint_restoration',
                'message': 'Checkpointing is disabled'
            }
        
        # Find latest checkpoint
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_cycle_*.pkl'))
        if not checkpoint_files:
            return {
                'success': False,
                'strategy': 'checkpoint_restoration',
                'message': 'No checkpoints found'
            }
        
        # Sort by modification time (latest first)
        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_checkpoint, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.logger.info(f"Checkpoint restored from: {latest_checkpoint}")
            
            return {
                'success': True,
                'strategy': 'checkpoint_restoration',
                'message': f'Restored from checkpoint: {latest_checkpoint.name}',
                'checkpoint_data': checkpoint_data
            }
        except Exception as e:
            self.logger.error(f"Failed to restore checkpoint: {e}")
            return {
                'success': False,
                'strategy': 'checkpoint_restoration',
                'message': f'Checkpoint restoration failed: {e}'
            }
    
    def _try_parameter_adjustment(self, error_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempts to adjust parameters to avoid the error.
        """
        self.logger.info("Trying parameter adjustment...")
        
        error_type = error_report.get('error_type', '')
        
        adjusted_params = {}
        
        if error_type == 'MemoryError':
            # Reduce memory-intensive parameters
            adjusted_params['population_size'] = max(20, self.config.population_size // 2)
            adjusted_params['num_islands'] = max(2, self.config.num_islands // 2)
            self.logger.info(f"Adjusted parameters for MemoryError: {adjusted_params}")
        
        elif error_type == 'RuntimeError':
            # Reduce computational complexity
            adjusted_params['generations'] = max(100, self.config.generations // 2)
            self.logger.info(f"Adjusted parameters for RuntimeError: {adjusted_params}")
        
        else:
            return {
                'success': False,
                'strategy': 'parameter_adjustment',
                'message': f'No parameter adjustment strategy for {error_type}'
            }
        
        # Apply adjusted parameters to config
        for key, value in adjusted_params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        return {
            'success': True,
            'strategy': 'parameter_adjustment',
            'message': 'Parameters adjusted',
            'adjusted_params': adjusted_params
        }
    
    def _try_algorithm_switch(self, error_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempts to switch to a more robust algorithm.
        """
        self.logger.info("Trying algorithm switch...")
        
        current_algorithm = self.config.default_solver
        
        # Fallback algorithm hierarchy (from most complex to simplest)
        fallback_algorithms = ['NSGA-II', 'Differential Evolution', 'PSO', 'Simulated Annealing']
        
        # Find next algorithm in hierarchy
        if current_algorithm in fallback_algorithms:
            current_idx = fallback_algorithms.index(current_algorithm)
            if current_idx < len(fallback_algorithms) - 1:
                new_algorithm = fallback_algorithms[current_idx + 1]
                self.config.default_solver = new_algorithm
                
                self.logger.info(f"Switched algorithm from {current_algorithm} to {new_algorithm}")
                
                return {
                    'success': True,
                    'strategy': 'algorithm_switch',
                    'message': f'Switched to {new_algorithm}',
                    'new_algorithm': new_algorithm
                }
        
        return {
            'success': False,
            'strategy': 'algorithm_switch',
            'message': 'No fallback algorithm available'
        }
    
    def _try_partial_result_extraction(self, error_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempts to extract partial results from incomplete optimization.
        """
        self.logger.info("Trying partial result extraction...")
        
        # This would require access to the archipelago state
        # For now, returning a placeholder
        
        return {
            'success': False,
            'strategy': 'partial_result_extraction',
            'message': 'Partial result extraction not implemented'
        }


