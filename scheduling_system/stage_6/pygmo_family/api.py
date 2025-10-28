"""
Programmatic API for PyGMO Solver Family

Provides clean interface for Stage 5 invocation.
"""

from pathlib import Path
from typing import Dict, Any, Optional

from .config import PyGMOConfig
from .processing.solver_orchestrator import SolverOrchestrator


def solve_pygmo(
    input_dir: str,
    output_dir: str,
    log_dir: str,
    solver: str = "NSGA-II",
    config: Optional[PyGMOConfig] = None
) -> Dict[str, Any]:
    """
    Programmatic API for PyGMO solver invocation.
    
    Args:
        input_dir: Path to Stage 3 outputs
        output_dir: Path for Stage 7 outputs
        log_dir: Path for logs
        solver: Solver name from Stage 5 ("NSGA-II", "MOEA/D", "PSO", "DE", "SA", "MIXED")
        config: Optional configuration override
    
    Returns:
        {
            "status": "success" | "error",
            "output_files": {
                "final_timetable": "path/to/final_timetable.csv",
                "solver_metadata": "path/to/solver_metadata.json",
                "pareto_front": "path/to/pareto_front.json",
                "performance_analytics": "path/to/performance_analytics.json"
            },
            "metrics": {
                "execution_time": 1234.56,
                "hypervolume": 0.92,
                "generations": 850
            },
            "error": None | "error message"
        }
    """
    # Create configuration
    if config is None:
        config = PyGMOConfig()
        config.input_dir = Path(input_dir)
        config.output_dir = Path(output_dir)
        config.log_dir = Path(log_dir)
        config.default_solver = solver
        config.__post_init__()
    else:
        # Override paths
        config.input_dir = Path(input_dir)
        config.output_dir = Path(output_dir)
        config.log_dir = Path(log_dir)
        if solver:
            config.default_solver = solver
    
    # Create orchestrator
    orchestrator = SolverOrchestrator(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        log_dir=Path(log_dir),
        base_config=config
    )
    
    # Execute optimization
    result = orchestrator.solve(solver_name=solver)
    
    return result


