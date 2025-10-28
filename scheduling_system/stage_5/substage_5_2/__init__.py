#!/usr/bin/env python3
"""
Substage 5.2: Optimal Solver Selection Module
=============================================

This module implements Substage 5.2 of the HEI Timetabling Engine, providing
intelligent solver selection based on complexity analysis results and solver
capabilities. It strictly adheres to the theoretical foundations and mathematical
frameworks defined in the project documentation.

MATHEMATICAL FOUNDATIONS COMPLIANCE:
- Implements solver selection based on theoretical complexity thresholds
- Uses validated solver capability matching algorithms
- No hardcoded solver preferences - all based on mathematical analysis
- Statistical confidence in solver selection decisions
- Adheres to modularity of solver arsenal theoretical foundations

Key Components:
- SelectionOrchestrator: Main orchestrator for the complete selection pipeline
- SolverSelectionEngine: Core engine for solver selection and validation
- SolverCapability: Data structure for solver capability information
- SolverSelectionResult: Comprehensive result with reasoning and alternatives

Usage:
The primary entry point is the `SelectionOrchestrator`, which coordinates the
complete selection process and produces structured output for downstream execution.

Example:
>>> from scheduling_engine_localized.stage_5.substage_5_2 import SelectionOrchestrator
>>> orchestrator = SelectionOrchestrator()
>>> result = orchestrator.execute_solver_selection(
...     complexity_analysis_path="stage_5/outputs/complexity_analysis.json",
...     solver_capabilities_path="config/solver_capabilities.json",
...     output_path="stage_5/outputs/solver_selection.json"
... )

Compliance:
- Modularity of Solver Arsenal - Theoretical Foundations & Mathematical Framework.md

Author: LUMEN TTMS - Theoretical Foundation Compliant Implementation
Version: 2.0.0
License: MIT
"""

import structlog

# Configure structured logging for the package
logger = structlog.get_logger(__name__)

# Main orchestrator
from .selection_orchestrator import SelectionOrchestrator

# Core selection components
from .solver_selection_engine import (
    SolverSelectionEngine,
    SolverCapability,
    SolverSelectionResult
)

# Public API
__all__ = [
    "SelectionOrchestrator",
    "SolverSelectionEngine",
    "SolverCapability",
    "SolverSelectionResult"
]

# Package metadata
__version__ = "2.0.0"
__author__ = "LUMEN TTMS - Theoretical Foundation Compliant Implementation"
__license__ = "MIT"


