"""
Solver Selection Intelligence

Implements Definition 14.1 (Algorithm Selection Guidelines) with performance prediction.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from typing import Dict, Any, Type
from dataclasses import dataclass

# Support both package and script execution
try:
    from ..input_model.metadata import ProblemMetadata
except (ImportError, ValueError):
    from input_model.metadata import ProblemMetadata


@dataclass
class SolverRecommendation:
    """Solver recommendation with rationale."""
    solver_name: str
    solver_class: Type
    confidence: float
    rationale: str
    expected_complexity: str
    expected_performance: float


class SolverSelector:
    """
    Intelligent solver selection per Definition 14.1.
    
    Never chooses inefficient solver that violates complexity bounds.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def select_solver(self, metadata: ProblemMetadata, solver_type: str = None) -> SolverRecommendation:
        """
        Select optimal solver based on problem characteristics.
        
        Args:
            metadata: Problem metadata
            solver_type: Override solver type if specified
        
        Returns:
            SolverRecommendation with selected solver
        """
        self.logger.info("=" * 80)
        self.logger.info("SOLVER SELECTION ANALYSIS")
        self.logger.info("=" * 80)
        
        # If solver type is specified, validate and use it
        if solver_type:
            return self._validate_specified_solver(solver_type, metadata)
        
        # Intelligent selection based on problem characteristics
        return self._intelligent_selection(metadata)
    
    def _validate_specified_solver(self, solver_type: str, metadata: ProblemMetadata) -> SolverRecommendation:
        """Validate specified solver against problem characteristics."""
        valid_solvers = ['ga', 'gp', 'es', 'de', 'pso', 'nsga2']
        
        if solver_type not in valid_solvers:
            raise ValueError(f"Invalid solver type: {solver_type}. Must be one of {valid_solvers}")
        
        # Import solver classes dynamically
        solver_classes = self._get_solver_classes()
        solver_class = solver_classes[solver_type]
        
        # Check if solver is efficient for problem
        efficiency_check = self._check_solver_efficiency(solver_type, metadata)
        
        if not efficiency_check['efficient']:
            self.logger.warning(f"Solver {solver_type} may be inefficient: {efficiency_check['reason']}")
        
        return SolverRecommendation(
            solver_name=solver_type,
            solver_class=solver_class,
            confidence=0.8 if efficiency_check['efficient'] else 0.4,
            rationale=f"User specified {solver_type}. {efficiency_check['reason']}",
            expected_complexity=efficiency_check['complexity'],
            expected_performance=efficiency_check['performance']
        )
    
    def _intelligent_selection(self, metadata: ProblemMetadata) -> SolverRecommendation:
        """Intelligent solver selection per Definition 14.1."""
        
        # Multi-objective problems: NSGA-II recommended per Section 13.2
        if metadata.multi_objective:
            self.logger.info("Multi-objective problem detected: NSGA-II recommended")
            return SolverRecommendation(
                solver_name='nsga2',
                solver_class=self._get_solver_classes()['nsga2'],
                confidence=0.95,
                rationale="Multi-objective optimization (RECOMMENDED per Section 13.2)",
                expected_complexity="O(λ·T·n·m + λ²·k)",
                expected_performance=0.9
            )
        
        # Problem size analysis
        problem_size = metadata.n_courses * metadata.n_faculty * metadata.n_rooms
        constraint_density = metadata.constraint_density
        
        self.logger.info(f"Problem size: {problem_size}")
        self.logger.info(f"Constraint density: {constraint_density:.4f}")
        self.logger.info(f"Specialization index: {metadata.specialization_index:.4f}")
        
        # Selection rules per Definition 14.1
        if problem_size < 10000:  # Small problems
            if constraint_density < 0.5:
                return self._recommend_ga(metadata, "Small problem with low constraint density")
            else:
                return self._recommend_de(metadata, "Small problem with high constraint density")
        
        elif problem_size < 100000:  # Medium problems
            if constraint_density < 0.3:
                return self._recommend_ga(metadata, "Medium problem with low constraint density")
            elif constraint_density < 0.7:
                return self._recommend_de(metadata, "Medium problem with medium constraint density")
            else:
                return self._recommend_nsga2(metadata, "Medium problem with high constraint density")
        
        else:  # Large problems (>100000)
            if metadata.fitness_landscape_smooth:
                return self._recommend_pso(metadata, "Large problem with smooth fitness landscape")
            else:
                return self._recommend_es(metadata, "Large problem with complex fitness landscape")
    
    def _recommend_ga(self, metadata: ProblemMetadata, reason: str) -> SolverRecommendation:
        """Recommend Genetic Algorithm."""
        return SolverRecommendation(
            solver_name='ga',
            solver_class=self._get_solver_classes()['ga'],
            confidence=0.85,
            rationale=f"GA selected: {reason}",
            expected_complexity="O(λ·T·n·m)",
            expected_performance=0.8
        )
    
    def _recommend_de(self, metadata: ProblemMetadata, reason: str) -> SolverRecommendation:
        """Recommend Differential Evolution."""
        return SolverRecommendation(
            solver_name='de',
            solver_class=self._get_solver_classes()['de'],
            confidence=0.8,
            rationale=f"DE selected: {reason}",
            expected_complexity="O(λ·T·n·m)",
            expected_performance=0.82
        )
    
    def _recommend_pso(self, metadata: ProblemMetadata, reason: str) -> SolverRecommendation:
        """Recommend Particle Swarm Optimization."""
        return SolverRecommendation(
            solver_name='pso',
            solver_class=self._get_solver_classes()['pso'],
            confidence=0.75,
            rationale=f"PSO selected: {reason}",
            expected_complexity="O(λ·T·n·m)",
            expected_performance=0.78
        )
    
    def _recommend_es(self, metadata: ProblemMetadata, reason: str) -> SolverRecommendation:
        """Recommend Evolution Strategies."""
        return SolverRecommendation(
            solver_name='es',
            solver_class=self._get_solver_classes()['es'],
            confidence=0.7,
            rationale=f"ES selected: {reason}",
            expected_complexity="O(λ·T·n²)",
            expected_performance=0.85
        )
    
    def _recommend_nsga2(self, metadata: ProblemMetadata, reason: str) -> SolverRecommendation:
        """Recommend NSGA-II."""
        return SolverRecommendation(
            solver_name='nsga2',
            solver_class=self._get_solver_classes()['nsga2'],
            confidence=0.9,
            rationale=f"NSGA-II selected: {reason}",
            expected_complexity="O(λ·T·n·m + λ²·k)",
            expected_performance=0.88
        )
    
    def _check_solver_efficiency(self, solver_type: str, metadata: ProblemMetadata) -> Dict[str, Any]:
        """Check if solver is efficient for problem characteristics."""
        problem_size = metadata.n_courses * metadata.n_faculty * metadata.n_rooms
        
        # Efficiency rules per Theorem 10.1
        if solver_type == 'es' and problem_size < 1000:
            return {
                'efficient': False,
                'reason': 'ES has O(n²) complexity, inefficient for small problems',
                'complexity': 'O(λ·T·n²)',
                'performance': 0.3
            }
        
        if solver_type == 'pso' and not metadata.fitness_landscape_smooth:
            return {
                'efficient': False,
                'reason': 'PSO requires smooth fitness landscape for optimal performance',
                'complexity': 'O(λ·T·n·m)',
                'performance': 0.4
            }
        
        if solver_type == 'gp' and metadata.constraint_density > 0.8:
            return {
                'efficient': False,
                'reason': 'GP may struggle with highly constrained problems',
                'complexity': 'O(λ·T·d²·eval)',
                'performance': 0.5
            }
        
        # Default: solver is efficient
        complexity_map = {
            'ga': 'O(λ·T·n·m)',
            'gp': 'O(λ·T·d²·eval)',
            'es': 'O(λ·T·n²)',
            'de': 'O(λ·T·n·m)',
            'pso': 'O(λ·T·n·m)',
            'nsga2': 'O(λ·T·n·m + λ²·k)'
        }
        
        performance_map = {
            'ga': 0.8,
            'gp': 0.7,
            'es': 0.85,
            'de': 0.82,
            'pso': 0.78,
            'nsga2': 0.88
        }
        
        return {
            'efficient': True,
            'reason': 'Solver is efficient for problem characteristics',
            'complexity': complexity_map.get(solver_type, 'O(λ·T·n·m)'),
            'performance': performance_map.get(solver_type, 0.8)
        }
    
    def _get_solver_classes(self) -> Dict[str, Type]:
        """Get solver classes (lazy import to avoid circular dependencies)."""
        # Import here to avoid circular dependencies
        from .nsga2 import NSGA2Solver
        
        # Placeholder classes for other solvers (not yet implemented)
        class GASolver:
            pass
        
        class GPSolver:
            pass
        
        class ESSolver:
            pass
        
        class DESolver:
            pass
        
        class PSOSolver:
            pass
        
        return {
            'ga': GASolver,
            'gp': GPSolver,
            'es': ESSolver,
            'de': DESolver,
            'pso': PSOSolver,
            'nsga2': NSGA2Solver,
        }

