"""
Core data structures for Stage 4 Feasibility Check
Theoretical compliant models based on mathematical frameworks
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import numpy as np
import pandas as pd


def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy/pandas types to JSON serializable types"""
    if isinstance(obj, (np.integer, pd.Int64Dtype)):
        return int(obj)
    elif isinstance(obj, (np.floating, pd.Float64Dtype)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    # Handle numpy scalars
    elif hasattr(obj, 'dtype') and hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj


class ValidationStatus(str, Enum):
    """Validation status enumeration"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class MathematicalProof:
    """Mathematical proof structure for feasibility violations"""
    theorem: str
    proof_statement: str
    conditions: List[str]
    conclusion: str
    complexity: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "theorem": self.theorem,
            "proof_statement": self.proof_statement,
            "conditions": self.conditions,
            "conclusion": self.conclusion,
            "complexity": self.complexity
        }


@dataclass
class LayerResult:
    """Result from a single validation layer"""
    layer_number: int
    layer_name: str
    status: ValidationStatus
    message: str
    details: Dict[str, Any]
    mathematical_proof: Optional[MathematicalProof] = None
    execution_time_ms: float = 0.0
    memory_used_mb: float = 0.0
    
    def is_valid(self) -> bool:
        """Check if layer validation passed"""
        return self.status == ValidationStatus.PASSED
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "layer_number": self.layer_number,
            "layer_name": self.layer_name,
            "status": self.status.value,
            "message": self.message,
            "details": convert_to_serializable(self.details),
            "execution_time_ms": self.execution_time_ms,
            "memory_used_mb": self.memory_used_mb
        }
        if self.mathematical_proof:
            result["mathematical_proof"] = self.mathematical_proof.to_dict()
        return result


@dataclass
class CrossLayerMetrics:
    """Cross-layer metrics as defined in the theoretical framework"""
    aggregate_load_ratio: float
    window_tightness_index: float
    conflict_density: float
    total_entities: int
    total_constraints: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregate_load_ratio": self.aggregate_load_ratio,
            "window_tightness_index": self.window_tightness_index,
            "conflict_density": self.conflict_density,
            "total_entities": self.total_entities,
            "total_constraints": self.total_constraints
        }


@dataclass
class FeasibilityInput:
    """Input configuration for feasibility checking"""
    input_directory: Union[str, Path]
    stage_3_artifacts: Dict[str, Path] = field(default_factory=dict)
    memory_limit_mb: Optional[int] = None
    timeout_seconds: Optional[int] = None
    
    def __post_init__(self):
        """Initialize stage 3 artifact paths"""
        if isinstance(self.input_directory, str):
            self.input_directory = Path(self.input_directory)
        
        base_path = self.input_directory / "files"
        self.stage_3_artifacts = {
            "L_raw": base_path / "L_raw",  # Directory containing entity parquet files
            "L_rel": base_path / "L_rel" / "relationship_graph.graphml", 
            "L_idx": base_path / "L_idx",  # Directory containing index pickle files
            "L_opt": base_path / "L_opt.pkl"  # Optional optimization data
        }


@dataclass
class FeasibilityOutput:
    """Output from feasibility checking"""
    is_feasible: bool
    layer_results: List[LayerResult]
    cross_layer_metrics: CrossLayerMetrics
    total_execution_time_ms: float
    peak_memory_mb: float
    failure_reason: Optional[str] = None
    mathematical_summary: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return convert_to_serializable({
            "is_feasible": self.is_feasible,
            "layer_results": [result.to_dict() for result in self.layer_results],
            "cross_layer_metrics": self.cross_layer_metrics.to_dict(),
            "total_execution_time_ms": self.total_execution_time_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "failure_reason": self.failure_reason,
            "mathematical_summary": self.mathematical_summary
        })
    
    def save_to_json(self, output_path: Path) -> None:
        """Save feasibility output to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class FeasibilityConfig:
    """Configuration for feasibility checking"""
    memory_limit_mb: Optional[int] = None
    timeout_seconds: Optional[int] = None
    enable_cross_layer_metrics: bool = True
    fail_fast: bool = True
    detailed_logging: bool = True
    
    # Layer-specific configurations
    layer_1_config: Dict[str, Any] = field(default_factory=dict)
    layer_2_config: Dict[str, Any] = field(default_factory=dict)
    layer_3_config: Dict[str, Any] = field(default_factory=dict)
    layer_4_config: Dict[str, Any] = field(default_factory=dict)
    layer_5_config: Dict[str, Any] = field(default_factory=dict)
    layer_6_config: Dict[str, Any] = field(default_factory=dict)
    layer_7_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default layer configurations"""
        if not self.layer_1_config:
            self.layer_1_config = {
                "strict_bcnf": True,
                "check_functional_dependencies": True
            }
        
        if not self.layer_2_config:
            self.layer_2_config = {
                "detect_cycles": True,
                "check_cardinality": True
            }
        
        if not self.layer_3_config:
            self.layer_3_config = {
                "resource_types": ["rooms", "faculty", "equipment"],
                "utilization_threshold": 0.85
            }
        
        if not self.layer_4_config:
            self.layer_4_config = {
                "strict_temporal": True,
                "soft_constraints": True
            }
        
        if not self.layer_5_config:
            self.layer_5_config = {
                "min_competency_score": 4.0,
                "check_availability": True
            }
        
        if not self.layer_6_config:
            self.layer_6_config = {
                "use_brooks_theorem": True,
                "check_cliques": True
            }
        
        if not self.layer_7_config:
            self.layer_7_config = {
                "arc_consistency": True,
                "forward_checking": True
            }
