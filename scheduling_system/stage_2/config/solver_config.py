"""
OR-Tools CP-SAT Solver Configuration
Implements Algorithm 5.1 from OR-Tools CP-SAT Bridge Foundation
"""

from typing import Dict
from dataclasses import dataclass


@dataclass
class SolverConfig:
    """
    CP-SAT solver configuration per Algorithm 5.1.
    
    Implements batching-optimized search strategy from Section 5.1.
    """
    
    # Search Strategy (Algorithm 5.1)
    search_branching: str = 'PORTFOLIO_SEARCH'
    cp_model_presolve: bool = True
    symmetry_level: int = 2
    
    # Performance Parameters
    num_search_workers: int = 4
    max_time_in_seconds: int = 300
    
    # Logging and Monitoring
    log_search_progress: bool = True
    log_to_response: bool = True
    
    # No artificial memory/time caps per requirements
    # Let mathematical bounds dictate resource usage
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'SolverConfig':
        """Create SolverConfig from dictionary."""
        return cls(**config)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'search_branching': self.search_branching,
            'cp_model_presolve': self.cp_model_presolve,
            'symmetry_level': self.symmetry_level,
            'num_search_workers': self.num_search_workers,
            'max_time_in_seconds': self.max_time_in_seconds,
            'log_search_progress': self.log_search_progress,
            'log_to_response': self.log_to_response
        }
    
    def apply_to_solver(self, solver) -> None:
        """
        Apply configuration to CP-SAT solver instance.
        
        Args:
            solver: ortools.sat.python.cp_model.CpSolver instance
        """
        from ortools.sat.python import cp_model
        
        # Algorithm 5.1: Batching-Optimized Search Strategy
        if self.search_branching == 'PORTFOLIO_SEARCH':
            solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
        
        solver.parameters.cp_model_presolve = self.cp_model_presolve
        solver.parameters.symmetry_level = self.symmetry_level
        solver.parameters.num_search_workers = self.num_search_workers
        solver.parameters.max_time_in_seconds = self.max_time_in_seconds
        solver.parameters.log_search_progress = self.log_search_progress
        solver.parameters.log_to_response = self.log_to_response

