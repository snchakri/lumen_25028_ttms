"""
Stage 6.1 PuLP Solver Family - Mathematically Rigorous Implementation

This module implements the PuLP solver family (CBC, GLPK, HiGHS, CLP, Symphony)
for educational scheduling optimization with 101% compliance to theoretical foundations.

Compliance:
- Stage-6.1 PuLP SOLVER FAMILY - Foundational Framework
- Dynamic Parametric System - Formal Analysis
- Stage-7 OUTPUT VALIDATION - Theoretical Foundation
- Stage-3 DATA COMPILATION - Theoretical Foundations

Author: LUMEN Team [TEAM-ID: 93912]
Version: 1.0 - Rigorous Theoretical Implementation
"""

__version__ = "1.0.0"
__author__ = "LUMEN Team [TEAM-ID: 93912]"

from .config import PuLPSolverConfig
from .main import run_pulp_solver_pipeline

__all__ = [
    'PuLPSolverConfig',
    'run_pulp_solver_pipeline'
]



