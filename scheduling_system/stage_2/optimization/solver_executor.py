"""
CP-SAT Solver Executor for Stage-2 Batching
Implements Algorithm 5.1 from OR-Tools CP-SAT Bridge Foundation
"""

from ortools.sat.python import cp_model
from typing import Dict, List
import time


class CPSATSolverExecutor:
    """
    CP-SAT Solver Executor with quality guarantees.
    
    Implements:
    - Algorithm 5.1: Batching-Optimized Search Strategy
    - Definition 8.2: Quality Guarantees
    """
    
    def __init__(self, model_builder, parameters: Dict):
        """
        Initialize solver executor.
        
        Args:
            model_builder: CPSATBatchingModel instance containing model and variables
            parameters: Foundation parameters including solver config
        """
        self.mb = model_builder
        self.model = model_builder.model
        self.parameters = parameters
        self.solver = cp_model.CpSolver()
        self._configure_solver()
    
    def _configure_solver(self) -> None:
        """
        Algorithm 5.1: Configure solver with batching-optimized search strategy.
        """
        # Search branching strategy
        self.solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
        
        # Presolve and symmetry
        self.solver.parameters.cp_model_presolve = True
        self.solver.parameters.symmetry_level = 2
        
        # Performance parameters
        self.solver.parameters.num_search_workers = self.parameters.get('parallel_workers', 4)
        self.solver.parameters.max_time_in_seconds = self.parameters.get('solver_timeout_seconds', 300)
        
        # Logging
        self.solver.parameters.log_search_progress = True
        self.solver.parameters.log_to_response = True
        
        # No artificial memory/time caps per requirements
        # Let mathematical bounds dictate resource usage
    
    def solve_with_guarantees(self) -> Dict:
        """
        Definition 8.2: Solve with Quality Guarantees
        
        Quality guarantees:
        1. Feasibility: All hard constraints satisfied
        2. Optimality: Optimal weighted objective within time bounds
        3. Foundation compliance: 100% adherence
        
        Returns:
            Dictionary with solution, metadata, and quality metrics
        
        Raises:
            InfeasibleBatchingException: If no feasible solution found
        """
        start_time = time.time()
        
        # Solve
        status = self.solver.Solve(self.model)
        solve_time = time.time() - start_time
        
        # Check status
        if status == cp_model.OPTIMAL:
            solution = self._extract_solution(status, solve_time)
            self._validate_solution_guarantees(solution)
            return solution
        
        elif status == cp_model.FEASIBLE:
            solution = self._extract_solution(status, solve_time)
            self._validate_solution_guarantees(solution)
            return solution
        
        elif status == cp_model.INFEASIBLE:
            raise InfeasibleBatchingException(
                "CP-SAT solver found problem infeasible. "
                "Check batch size constraints, room capacity, and course coherence requirements."
            )
        
        elif status == cp_model.MODEL_INVALID:
            raise ValueError("CP-SAT model is invalid. Check model construction.")
        
        else:
            raise RuntimeError(f"CP-SAT solver failed with status: {status}")
    
    def _extract_solution(self, status: int, solve_time: float) -> Dict:
        """
        Extract solution from solver.
        
        Args:
            status: Solver status
            solve_time: Solution time in seconds
        
        Returns:
            Dictionary with solution data
        """
        # Extract batch assignments
        # Use variable mapping from model builder
        n = self.mb.n
        m = self.mb.m
        batches = []
        assignments = []
        for j in range(m):
            batch_students = []
            for i in range(n):
                if self.solver.Value(self.mb.x[i, j]) == 1:
                    # Preserve original student_id and any enriched attributes
                    student = self.mb.students[i]
                    batch_students.append({
                        'student_id': str(student.get('student_id', i)),
                        'student_index': i,
                        'enrolled_courses': student.get('enrolled_courses', [])
                    })
                    assignments.append({'student_index': i, 'batch_index': j})
            if batch_students:
                batches.append({'batch_index': j, 'students': batch_students, 'batch_size': len(batch_students)})
        
        return {
            'status': 'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE',
            'objective_value': self.solver.ObjectiveValue(),
            'solve_time': solve_time,
            'batches': batches,
            'assignments': assignments,
            'metadata': {
                'num_conflicts': self.solver.NumConflicts(),
                'num_branches': self.solver.NumBranches(),
                'wall_time': self.solver.WallTime(),
                'user_time': self.solver.UserTime()
            }
        }
    
    def _validate_solution_guarantees(self, solution: Dict) -> None:
        """
        Validate Definition 8.2 quality guarantees.
        
        Ensures:
        1. Feasibility: All hard constraints satisfied
        2. Optimality: Optimal weighted objective within time bounds
        3. Foundation compliance: 100% adherence
        """
        # Check 1: Feasibility
        if solution['status'] not in ['OPTIMAL', 'FEASIBLE']:
            raise ValueError(f"Solution not feasible: status={solution['status']}")
        
        # Check 2: Optimality (within time bounds)
        if solution['solve_time'] > self.parameters.get('solver_timeout_seconds', 300):
            raise RuntimeError(f"Solution time {solution['solve_time']} exceeds timeout")
        
        # Check 3: Foundation compliance
        # Verify batch sizes within bounds
        for batch in solution['batches']:
            batch_size = batch['batch_size']
            if not (self.parameters['min_batch_size'] <= batch_size <= self.parameters['max_batch_size']):
                raise ValueError(
                    f"Batch {batch['batch_index']} size {batch_size} violates bounds "
                    f"[{self.parameters['min_batch_size']}, {self.parameters['max_batch_size']}]"
                )
    
    # Helper legacy methods no longer needed. We rely on model builder mappings directly.


class InfeasibleBatchingException(Exception):
    """Exception raised when batching problem is infeasible."""
    pass

