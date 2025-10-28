"""
Stage 4 Core Package
Contains data structures, orchestrator, and Stage 3 adapter
"""

from .data_structures import (
    FeasibilityInput,
    FeasibilityOutput,
    LayerResult,
    ValidationStatus,
    FeasibilityConfig,
    MathematicalProof,
    CrossLayerMetrics
)
from .orchestrator import FeasibilityOrchestrator
from .stage3_adapter import Stage3Adapter, Stage3Data

__all__ = [
    # Data Structures
    "FeasibilityInput",
    "FeasibilityOutput",
    "LayerResult",
    "ValidationStatus",
    "FeasibilityConfig",
    "MathematicalProof",
    "CrossLayerMetrics",
    
    # Core Components
    "FeasibilityOrchestrator",
    "Stage3Adapter",
    "Stage3Data"
]