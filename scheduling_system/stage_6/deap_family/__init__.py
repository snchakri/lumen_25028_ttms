"""
DEAP Solver Family Package

Stage 6.3 Implementation with Foundation Compliance.

Author: LUMEN Team [TEAM-ID: 93912]
"""

__version__ = "1.0.0"
__team__ = "LUMEN [TEAM-ID: 93912]"

from .main import run_deap_solver_pipeline, nsga2, ga, gp, es, de, pso

__all__ = [
    "run_deap_solver_pipeline",
    "nsga2",
    "ga",
    "gp",
    "es",
    "de",
    "pso",
]
