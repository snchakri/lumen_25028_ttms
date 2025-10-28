"""
JSON Writer - Solution Metadata

Generates JSON outputs with S* = (x*, y*, z*, M) per Definition 4.1.

Compliance: Definition 4.1: Optimal Solution Representation

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from .decoder import Schedule


class JSONWriter:
    """Writes JSON outputs with solution metadata."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize JSON writer."""
        self.logger = logger or logging.getLogger(__name__)
    
    def write_solution_json(
        self,
        schedule: Schedule,
        solver_result,
        output_path: Path
    ) -> Path:
        """
        Write solution.json with S* = (x*, y*, z*, M).
        
        Compliance: Definition 4.1
        
        Args:
            schedule: Schedule
            solver_result: SolverResult
            output_path: Output directory
        
        Returns:
            Path to created JSON file
        """
        self.logger.info("Writing solution.json...")
        
        # Build solution structure per Definition 4.1
        solution = {
            'x_star': {
                'continuous_variables': {},  # Would extract from problem
                'description': 'Optimal continuous variable values'
            },
            'y_star': {
                'binary_variables': len(schedule.assignments),
                'assignments': [a.to_dict() for a in schedule.assignments],
                'description': 'Optimal binary variable values'
            },
            'z_star': schedule.objective_value,
            'M': {
                'solver_used': schedule.solver_used,
                'solve_time': schedule.solve_time,
                'status': solver_result.status,
                'iterations': solver_result.iterations,
                'nodes_explored': solver_result.nodes_explored,
                'optimality_gap': solver_result.optimality_gap,
                'metadata': solver_result.metadata
            }
        }
        
        # Write JSON
        json_file = output_path / 'solution.json'
        with open(json_file, 'w') as f:
            json.dump(solution, f, indent=2)
        
        self.logger.info(f"Wrote solution to {json_file}")
        
        return json_file
    
    def write_validation_json(
        self,
        schedule: Schedule,
        validation_results: Dict[str, Any],
        output_path: Path
    ) -> Path:
        """
        Write validation_analysis.json for Stage 7.
        
        Args:
            schedule: Schedule
            validation_results: Validation results
            output_path: Output directory
        
        Returns:
            Path to created JSON file
        """
        self.logger.info("Writing validation_analysis.json...")
        
        # Build validation structure
        validation = {
            'schedule_metrics': {
                'n_assignments': len(schedule.assignments),
                'n_conflicts': schedule.n_conflicts,
                'objective_value': schedule.objective_value
            },
            'validation_results': validation_results,
            'status': 'valid' if schedule.n_conflicts == 0 else 'invalid'
        }
        
        # Write JSON
        json_file = output_path / 'validation_analysis.json'
        with open(json_file, 'w') as f:
            json.dump(validation, f, indent=2)
        
        self.logger.info(f"Wrote validation to {json_file}")
        
        return json_file



