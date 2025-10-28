#!/usr/bin/env python3
"""
Dynamic Parameter Definitions for Stage 5

Defines all 24 configurable parameters for Stage 5 with validation rules.

Author: LUMEN TTMS
Version: 2.0.0
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

class ParameterType(Enum):
    """Parameter data types."""
    FLOAT = "float"
    INTEGER = "int"
    STRING = "str"
    BOOLEAN = "bool"
    JSON = "json"

@dataclass
class ParameterDefinition:
    """Definition of a dynamic parameter."""
    code: str
    name: str
    param_type: ParameterType
    default_value: Any
    min_value: Any = None
    max_value: Any = None
    allowed_values: List[Any] = None
    description: str = ""
    validation_rule: str = ""

# 24 Stage 5 Parameters per stage5-dynamic-parameters-framework.md

# Complexity Analysis Parameters (8 params)
COMPLEXITY_WEIGHT_P1 = ParameterDefinition(
    code="COMPLEXITY_WEIGHT_P1",
    name="Problem Space Dimensionality Weight",
    param_type=ParameterType.FLOAT,
    default_value=0.15,
    min_value=0.0,
    max_value=1.0,
    description="Weight for Problem Space Dimensionality (P1) in composite index"
)

COMPLEXITY_WEIGHT_P2 = ParameterDefinition(
    code="COMPLEXITY_WEIGHT_P2",
    name="Constraint Density Weight",
    param_type=ParameterType.FLOAT,
    default_value=0.12,
    min_value=0.0,
    max_value=1.0,
    description="Weight for Constraint Density (P2) in composite index"
)

COMPLEXITY_WEIGHT_P3 = ParameterDefinition(
    code="COMPLEXITY_WEIGHT_P3",
    name="Faculty Specialization Weight",
    param_type=ParameterType.FLOAT,
    default_value=0.10,
    min_value=0.0,
    max_value=1.0,
    description="Weight for Faculty Specialization Index (P3) in composite index"
)

COMPLEXITY_WEIGHT_P4 = ParameterDefinition(
    code="COMPLEXITY_WEIGHT_P4",
    name="Room Utilization Weight",
    param_type=ParameterType.FLOAT,
    default_value=0.09,
    min_value=0.0,
    max_value=1.0,
    description="Weight for Room Utilization Factor (P4) in composite index"
)

COMPLEXITY_WEIGHT_P5 = ParameterDefinition(
    code="COMPLEXITY_WEIGHT_P5",
    name="Temporal Distribution Weight",
    param_type=ParameterType.FLOAT,
    default_value=0.08,
    min_value=0.0,
    max_value=1.0,
    description="Weight for Temporal Distribution Complexity (P5) in composite index"
)

COMPLEXITY_WEIGHT_P6 = ParameterDefinition(
    code="COMPLEXITY_WEIGHT_P6",
    name="Batch Size Variance Weight",
    param_type=ParameterType.FLOAT,
    default_value=0.07,
    min_value=0.0,
    max_value=1.0,
    description="Weight for Batch Size Variance (P6) in composite index"
)

COMPLEXITY_WEIGHT_P7 = ParameterDefinition(
    code="COMPLEXITY_WEIGHT_P7",
    name="Competency Entropy Weight",
    param_type=ParameterType.FLOAT,
    default_value=0.06,
    min_value=0.0,
    max_value=1.0,
    description="Weight for Competency Distribution Entropy (P7) in composite index"
)

COMPLEXITY_WEIGHT_P8 = ParameterDefinition(
    code="COMPLEXITY_WEIGHT_P8",
    name="Multi-Objective Conflict Weight",
    param_type=ParameterType.FLOAT,
    default_value=0.06,
    min_value=0.0,
    max_value=1.0,
    description="Weight for Multi-Objective Conflict Measure (P8) in composite index"
)

COMPLEXITY_WEIGHT_P9 = ParameterDefinition(
    code="COMPLEXITY_WEIGHT_P9",
    name="Constraint Coupling Weight",
    param_type=ParameterType.FLOAT,
    default_value=0.05,
    min_value=0.0,
    max_value=1.0,
    description="Weight for Constraint Coupling Coefficient (P9) in composite index"
)

COMPLEXITY_WEIGHT_P10 = ParameterDefinition(
    code="COMPLEXITY_WEIGHT_P10",
    name="Resource Heterogeneity Weight",
    param_type=ParameterType.FLOAT,
    default_value=0.05,
    min_value=0.0,
    max_value=1.0,
    description="Weight for Resource Heterogeneity Index (P10) in composite index"
)

COMPLEXITY_WEIGHT_P11 = ParameterDefinition(
    code="COMPLEXITY_WEIGHT_P11",
    name="Schedule Flexibility Weight",
    param_type=ParameterType.FLOAT,
    default_value=0.04,
    min_value=0.0,
    max_value=1.0,
    description="Weight for Schedule Flexibility Measure (P11) in composite index"
)

COMPLEXITY_WEIGHT_P12 = ParameterDefinition(
    code="COMPLEXITY_WEIGHT_P12",
    name="Dependency Graph Weight",
    param_type=ParameterType.FLOAT,
    default_value=0.04,
    min_value=0.0,
    max_value=1.0,
    description="Weight for Dependency Graph Complexity (P12) in composite index"
)

COMPLEXITY_WEIGHT_P13 = ParameterDefinition(
    code="COMPLEXITY_WEIGHT_P13",
    name="Landscape Ruggedness Weight",
    param_type=ParameterType.FLOAT,
    default_value=0.03,
    min_value=0.0,
    max_value=1.0,
    description="Weight for Optimization Landscape Ruggedness (P13) in composite index"
)

COMPLEXITY_WEIGHT_P14 = ParameterDefinition(
    code="COMPLEXITY_WEIGHT_P14",
    name="Scalability Projection Weight",
    param_type=ParameterType.FLOAT,
    default_value=0.03,
    min_value=0.0,
    max_value=1.0,
    description="Weight for Scalability Projection Factor (P14) in composite index"
)

COMPLEXITY_WEIGHT_P15 = ParameterDefinition(
    code="COMPLEXITY_WEIGHT_P15",
    name="Constraint Propagation Weight",
    param_type=ParameterType.FLOAT,
    default_value=0.02,
    min_value=0.0,
    max_value=1.0,
    description="Weight for Constraint Propagation Depth (P15) in composite index"
)

COMPLEXITY_WEIGHT_P16 = ParameterDefinition(
    code="COMPLEXITY_WEIGHT_P16",
    name="Solution Quality Variance Weight",
    param_type=ParameterType.FLOAT,
    default_value=0.02,
    min_value=0.0,
    max_value=1.0,
    description="Weight for Solution Quality Variance (P16) in composite index"
)

COMPLEXITY_THRESHOLD_LOW = ParameterDefinition(
    code="COMPLEXITY_THRESHOLD_LOW",
    name="Low Complexity Threshold",
    param_type=ParameterType.FLOAT,
    default_value=3.0,
    min_value=0.0,
    max_value=15.0,
    description="Threshold for low complexity (use heuristics)"
)

COMPLEXITY_THRESHOLD_MEDIUM = ParameterDefinition(
    code="COMPLEXITY_THRESHOLD_MEDIUM",
    name="Medium Complexity Threshold",
    param_type=ParameterType.FLOAT,
    default_value=6.0,
    min_value=0.0,
    max_value=15.0,
    description="Threshold for medium complexity (use local search)"
)

COMPLEXITY_THRESHOLD_HIGH = ParameterDefinition(
    code="COMPLEXITY_THRESHOLD_HIGH",
    name="High Complexity Threshold",
    param_type=ParameterType.FLOAT,
    default_value=9.0,
    min_value=0.0,
    max_value=15.0,
    description="Threshold for high complexity (use metaheuristics)"
)

# LP Optimization Parameters (8 params)
LP_CONVERGENCE_TOLERANCE = ParameterDefinition(
    code="LP_CONVERGENCE_TOLERANCE",
    name="LP Convergence Tolerance",
    param_type=ParameterType.FLOAT,
    default_value=1e-6,
    min_value=1e-10,
    max_value=1e-3,
    description="Convergence tolerance for LP iterative algorithm"
)

LP_MAX_ITERATIONS = ParameterDefinition(
    code="LP_MAX_ITERATIONS",
    name="LP Maximum Iterations",
    param_type=ParameterType.INTEGER,
    default_value=20,
    min_value=5,
    max_value=100,
    description="Maximum iterations for LP optimization"
)

LP_SOLVER_METHOD = ParameterDefinition(
    code="LP_SOLVER_METHOD",
    name="LP Solver Method",
    param_type=ParameterType.STRING,
    default_value="highs",
    allowed_values=["highs", "interior-point", "revised_simplex"],
    description="scipy.optimize.linprog solver method"
)

LP_PRESOLVE_ENABLED = ParameterDefinition(
    code="LP_PRESOLVE_ENABLED",
    name="LP Presolve Enabled",
    param_type=ParameterType.BOOLEAN,
    default_value=True,
    description="Enable LP presolve optimization"
)

SEPARATION_MARGIN_THRESHOLD = ParameterDefinition(
    code="SEPARATION_MARGIN_THRESHOLD",
    name="Separation Margin Threshold",
    param_type=ParameterType.FLOAT,
    default_value=0.001,
    min_value=1e-6,
    max_value=0.1,
    description="Minimum acceptable separation margin for solver selection"
)

WEIGHT_LEARNING_ENABLED = ParameterDefinition(
    code="WEIGHT_LEARNING_ENABLED",
    name="Weight Learning Enabled",
    param_type=ParameterType.BOOLEAN,
    default_value=True,
    description="Enable automated weight learning via LP"
)

NORMALIZATION_METHOD = ParameterDefinition(
    code="NORMALIZATION_METHOD",
    name="Normalization Method",
    param_type=ParameterType.STRING,
    default_value="l2_norm",
    allowed_values=["l2_norm", "min_max", "z_score"],
    description="Parameter normalization method (must be l2_norm per foundations)"
)

ROBUSTNESS_FACTOR = ParameterDefinition(
    code="ROBUSTNESS_FACTOR",
    name="Robustness Factor",
    param_type=ParameterType.FLOAT,
    default_value=0.1,
    min_value=0.0,
    max_value=1.0,
    description="Robustness weighting factor for selection"
)

# Selection Parameters (4 params)
SOLVER_CONFIDENCE_THRESHOLD = ParameterDefinition(
    code="SOLVER_CONFIDENCE_THRESHOLD",
    name="Solver Confidence Threshold",
    param_type=ParameterType.FLOAT,
    default_value=0.7,
    min_value=0.0,
    max_value=1.0,
    description="Minimum confidence for solver selection"
)

CAPABILITY_ASSESSMENT_METHOD = ParameterDefinition(
    code="CAPABILITY_ASSESSMENT_METHOD",
    name="Capability Assessment Method",
    param_type=ParameterType.STRING,
    default_value="benchmark_based",
    allowed_values=["benchmark_based", "expert_assessment", "hybrid"],
    description="Solver capability evaluation method"
)

SELECTION_BIAS_CORRECTION = ParameterDefinition(
    code="SELECTION_BIAS_CORRECTION",
    name="Selection Bias Correction",
    param_type=ParameterType.BOOLEAN,
    default_value=True,
    description="Apply bias correction in selection"
)

MULTI_SOLVER_ENABLED = ParameterDefinition(
    code="MULTI_SOLVER_ENABLED",
    name="Multi-Solver Enabled",
    param_type=ParameterType.BOOLEAN,
    default_value=False,
    description="Enable multi-solver orchestration"
)

# Normalization Parameters (4 params)
L2_NORM_EPSILON = ParameterDefinition(
    code="L2_NORM_EPSILON",
    name="L2 Norm Epsilon",
    param_type=ParameterType.FLOAT,
    default_value=1e-10,
    min_value=1e-15,
    max_value=1e-6,
    description="Epsilon for preventing division by zero in L2 normalization"
)

NORMALIZATION_BOUNDS_CHECK = ParameterDefinition(
    code="NORMALIZATION_BOUNDS_CHECK",
    name="Normalization Bounds Check",
    param_type=ParameterType.BOOLEAN,
    default_value=True,
    description="Enable validation of normalization bounds [0,1]"
)

SCALE_INVARIANCE_VALIDATION = ParameterDefinition(
    code="SCALE_INVARIANCE_VALIDATION",
    name="Scale Invariance Validation",
    param_type=ParameterType.BOOLEAN,
    default_value=True,
    description="Enable validation of scale invariance property"
)

DYNAMIC_ADAPTATION_MODE = ParameterDefinition(
    code="DYNAMIC_ADAPTATION_MODE",
    name="Dynamic Adaptation Mode",
    param_type=ParameterType.BOOLEAN,
    default_value=True,
    description="Enable dynamic adaptation to new solvers"
)

# All 24 parameters
STAGE5_PARAMETERS = {
    # Complexity weights (16 params)
    "COMPLEXITY_WEIGHT_P1": COMPLEXITY_WEIGHT_P1,
    "COMPLEXITY_WEIGHT_P2": COMPLEXITY_WEIGHT_P2,
    "COMPLEXITY_WEIGHT_P3": COMPLEXITY_WEIGHT_P3,
    "COMPLEXITY_WEIGHT_P4": COMPLEXITY_WEIGHT_P4,
    "COMPLEXITY_WEIGHT_P5": COMPLEXITY_WEIGHT_P5,
    "COMPLEXITY_WEIGHT_P6": COMPLEXITY_WEIGHT_P6,
    "COMPLEXITY_WEIGHT_P7": COMPLEXITY_WEIGHT_P7,
    "COMPLEXITY_WEIGHT_P8": COMPLEXITY_WEIGHT_P8,
    "COMPLEXITY_WEIGHT_P9": COMPLEXITY_WEIGHT_P9,
    "COMPLEXITY_WEIGHT_P10": COMPLEXITY_WEIGHT_P10,
    "COMPLEXITY_WEIGHT_P11": COMPLEXITY_WEIGHT_P11,
    "COMPLEXITY_WEIGHT_P12": COMPLEXITY_WEIGHT_P12,
    "COMPLEXITY_WEIGHT_P13": COMPLEXITY_WEIGHT_P13,
    "COMPLEXITY_WEIGHT_P14": COMPLEXITY_WEIGHT_P14,
    "COMPLEXITY_WEIGHT_P15": COMPLEXITY_WEIGHT_P15,
    "COMPLEXITY_WEIGHT_P16": COMPLEXITY_WEIGHT_P16,
    
    # Complexity thresholds (3 params)
    "COMPLEXITY_THRESHOLD_LOW": COMPLEXITY_THRESHOLD_LOW,
    "COMPLEXITY_THRESHOLD_MEDIUM": COMPLEXITY_THRESHOLD_MEDIUM,
    "COMPLEXITY_THRESHOLD_HIGH": COMPLEXITY_THRESHOLD_HIGH,
    
    # LP optimization (8 params)
    "LP_CONVERGENCE_TOLERANCE": LP_CONVERGENCE_TOLERANCE,
    "LP_MAX_ITERATIONS": LP_MAX_ITERATIONS,
    "LP_SOLVER_METHOD": LP_SOLVER_METHOD,
    "LP_PRESOLVE_ENABLED": LP_PRESOLVE_ENABLED,
    "SEPARATION_MARGIN_THRESHOLD": SEPARATION_MARGIN_THRESHOLD,
    "WEIGHT_LEARNING_ENABLED": WEIGHT_LEARNING_ENABLED,
    "NORMALIZATION_METHOD": NORMALIZATION_METHOD,
    "ROBUSTNESS_FACTOR": ROBUSTNESS_FACTOR,
    
    # Selection (4 params)
    "SOLVER_CONFIDENCE_THRESHOLD": SOLVER_CONFIDENCE_THRESHOLD,
    "CAPABILITY_ASSESSMENT_METHOD": CAPABILITY_ASSESSMENT_METHOD,
    "SELECTION_BIAS_CORRECTION": SELECTION_BIAS_CORRECTION,
    "MULTI_SOLVER_ENABLED": MULTI_SOLVER_ENABLED,
    
    # Normalization (4 params)
    "L2_NORM_EPSILON": L2_NORM_EPSILON,
    "NORMALIZATION_BOUNDS_CHECK": NORMALIZATION_BOUNDS_CHECK,
    "SCALE_INVARIANCE_VALIDATION": SCALE_INVARIANCE_VALIDATION,
    "DYNAMIC_ADAPTATION_MODE": DYNAMIC_ADAPTATION_MODE,
}

# Validate weight sum constraint
def validate_complexity_weights() -> bool:
    """
    Validate that complexity weights sum to 1.0 ± 1e-6
    
    Returns:
        True if weights are valid
    """
    weights = [
        STAGE5_PARAMETERS[f"COMPLEXITY_WEIGHT_P{i}"].default_value 
        for i in range(1, 17)
    ]
    weight_sum = sum(weights)
    
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"Complexity weights sum to {weight_sum}, expected 1.0")
    
    return True

# Validate at module import
validate_complexity_weights()


