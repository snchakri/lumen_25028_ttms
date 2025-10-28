#!/usr/bin/env python3
"""
Substage 5.1: Input Complexity Analysis Module
==============================================

This module implements Substage 5.1 of the HEI Timetabling Engine, providing
comprehensive 16-parameter complexity analysis of Stage 3 outputs. It strictly
adheres to the theoretical foundations and mathematical frameworks defined in
the project documentation.

MATHEMATICAL FOUNDATIONS COMPLIANCE:
- Implements all 16 complexity parameters with formal theorem compliance
- Uses theoretical O(N log N) complexity bounds
- No hardcoded values - all computed from actual data
- Statistical validation with 95% confidence intervals
- Composite complexity index with validated weights

Key Components:
- AnalysisOrchestrator: Main orchestrator for the complete analysis pipeline
- ComplexityAnalyzer: Core engine for parameter computation and validation
- ParameterComputations: Individual computation functions for each parameter

Usage:
The primary entry point is the `AnalysisOrchestrator`, which coordinates the
complete analysis and produces JSON output for downstream solver selection.

Example:
>>> from scheduling_engine_localized.stage_5.substage_5_1 import AnalysisOrchestrator
>>> orchestrator = AnalysisOrchestrator()
>>> result = orchestrator.execute_complexity_analysis(
...     stage3_output_path="stage_3/outputs",
...     output_path="stage_5/outputs/complexity_analysis.json"
... )

Compliance:
- Stage-5.1 INPUT-COMPLEXITY ANALYSIS - Theoretical Foundations & Mathematical Framework.md

Author: LUMEN TTMS - Theoretical Foundation Compliant Implementation
Version: 2.0.0
License: MIT
"""

import structlog

# Configure structured logging for the package
logger = structlog.get_logger(__name__)

# Main orchestrator
from .analysis_orchestrator import AnalysisOrchestrator

# Core analysis components
from .complexity_analyzer import ComplexityAnalyzer, DataStructures
from .parameter_computations import ParameterComputations

# Public API
__all__ = [
    "AnalysisOrchestrator",
    "ComplexityAnalyzer", 
    "DataStructures",
    "ParameterComputations"
]

# Package metadata
__version__ = "2.0.0"
__author__ = "LUMEN TTMS - Theoretical Foundation Compliant Implementation"
__license__ = "MIT"