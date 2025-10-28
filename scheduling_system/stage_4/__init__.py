"""
Stage 4 Feasibility Check - Rebuilt from Foundations & Frameworks
Team Lumen [Team ID: 93912] - SIH 2025
Theoretical Compliant Seven-Layer Mathematical Validation System

This module implements the seven-layer mathematical feasibility validation pipeline
as specified in "Stage-4 FEASIBILITY CHECK - Theoretical Foundation & Mathematical Framework.md"

Mathematical Foundation:
- Layer 1: BCNF compliance & schema consistency (Theorem 2.1)
- Layer 2: Relational integrity & cardinality (Theorem 3.1)  
- Layer 3: Resource capacity bounds (Theorem 4.1)
- Layer 4: Temporal window analysis (pigeonhole principle)
- Layer 5: Competency & availability (Hall's Marriage Theorem 6.1)
- Layer 6: Conflict graph & chromatic feasibility (Brooks' theorem)
- Layer 7: Global constraint propagation (Arc-consistency)

Integration Points:
- Input: Stage 3 compiled data (L_raw, L_rel, L_idx)
- Output: Feasibility certificate (JSON) or infeasibility report (JSON) with metrics (CSV)
- Performance: O(n²) complexity per layer, fail-fast termination
- Architecture: Modular, OR-Tools powered, theoretically compliant
"""

__version__ = "2.0.0"
__author__ = "Team Lumen [Team ID: 93912]"

# Core exports
from .core.data_structures import (
    FeasibilityInput,
    FeasibilityOutput,
    LayerResult,
    ValidationStatus,
    FeasibilityConfig,
    MathematicalProof,
    CrossLayerMetrics
)
from .core.orchestrator import FeasibilityOrchestrator
from .core.stage3_adapter import Stage3Adapter, Stage3Data

# Layer exports
from .layers.layer_1_bcnf import BCNFValidator
from .layers.layer_2_integrity import IntegrityValidator
from .layers.layer_3_capacity import CapacityValidator
from .layers.layer_4_temporal import TemporalValidator
from .layers.layer_5_competency import CompetencyValidator
from .layers.layer_6_conflict import ConflictValidator
from .layers.layer_7_propagation import PropagationValidator

# Utility exports
from .utils.metrics_calculator import CrossLayerMetricsCalculator
from .utils.report_generator import FeasibilityReportGenerator
from .utils.logger import StructuredLogger, create_logger
from .utils.error_handler import (
    ErrorHandler,
    FeasibilityError,
    LayerValidationError,
    TheoremViolationError,
    Stage3InputError,
    MathematicalProofError,
    ErrorSeverity,
    ErrorCategory
)

__all__ = [
    # Core
    "FeasibilityInput",
    "FeasibilityOutput", 
    "LayerResult",
    "ValidationStatus",
    "FeasibilityConfig",
    "MathematicalProof",
    "CrossLayerMetrics",
    "FeasibilityOrchestrator",
    "Stage3Adapter",
    "Stage3Data",
    
    # Layers
    "BCNFValidator",
    "IntegrityValidator", 
    "CapacityValidator",
    "TemporalValidator",
    "CompetencyValidator",
    "ConflictValidator",
    "PropagationValidator",
    
    # Utilities
    "CrossLayerMetricsCalculator",
    "FeasibilityReportGenerator",
    "StructuredLogger",
    "create_logger",
    "ErrorHandler",
    "FeasibilityError",
    "LayerValidationError",
    "TheoremViolationError",
    "Stage3InputError",
    "MathematicalProofError",
    "ErrorSeverity",
    "ErrorCategory"
]

def create_feasibility_orchestrator(config: FeasibilityConfig = None) -> FeasibilityOrchestrator:
    """Create a new feasibility orchestrator with the given configuration."""
    if config is None:
        config = FeasibilityConfig()
    return FeasibilityOrchestrator(config)

def execute_feasibility_check(
    input_directory: str,
    output_directory: str,
    config: FeasibilityConfig = None
) -> FeasibilityOutput:
    """Execute the complete seven-layer feasibility check pipeline."""
    orchestrator = create_feasibility_orchestrator(config)
    return orchestrator.execute_feasibility_check(input_directory, output_directory)