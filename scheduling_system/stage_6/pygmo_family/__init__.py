"""
PyGMO Solver Family (Stage 6.4)
Multi-Objective Global Optimization for Educational Timetabling

This module implements the PyGMO-based solver family for the scheduling engine,
providing archipelago-based multi-objective optimization with rigorous mathematical
compliance to theoretical foundations.
"""

__version__ = "1.0.0"

from .config import PyGMOConfig
from .api import solve_pygmo

__all__ = [
    'PyGMOConfig',
    'solve_pygmo',
]


