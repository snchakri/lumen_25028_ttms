"""
Configuration Management for PuLP Solver Family

Implements Dynamic Parametric System integration with hierarchical parameter resolution,
solver-specific parameter extraction, and configuration validation per foundations.

Compliance:
- Dynamic Parametric System Section 6.3: Solver Configuration Parameters
- Definition 5.2: Solver Selection Mapping
- Theorem 10.2: No Universal Best Solver

"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np


class SolverType(Enum):
    """Supported PuLP solver types per foundations."""
    CBC = "CBC"
    GLPK = "GLPK"
    HIGHS = "HiGHS"
    CLP = "CLP"
    SYMPHONY = "Symphony"


class SolverStatus(Enum):
    """Solver execution status."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class SolverParameters:
    """Solver-specific parameters from dynamic parametric system."""
    
    # Time limits (no artificial caps per foundations)
    # None = no limit, computed from O(.) complexity bounds per Theorem 6.1
    time_limit_seconds: Optional[float] = None
    
    # Solution quality thresholds
    optimality_gap: float = 0.0  # 0.0 = optimal, 0.01 = 1% gap
    feasibility_tolerance: float = 1e-6
    optimality_tolerance: float = 1e-6
    
    # Solver preferences
    preferred_solver: Optional[SolverType] = None
    fallback_solvers: List[SolverType] = field(default_factory=lambda: [
        SolverType.CBC, SolverType.GLPK, SolverType.HIGHS
    ])
    
    # Resource limits (no artificial caps per foundations)
    memory_limit_mb: Optional[float] = None  # None = no limit
    
    # CBC-specific parameters
    cbc_threads: int = 1  # Sequential execution per foundations
    cbc_strong_branching: bool = True
    cbc_cuts: str = "on"  # "on", "off", "root"
    
    # GLPK-specific parameters
    glpk_presolve: bool = True
    glpk_scale: bool = True
    
    # HiGHS-specific parameters
    highs_presolve: bool = True
    highs_parallel: bool = False  # Sequential per foundations
    
    # CLP-specific parameters
    clp_dual_simplex: bool = True
    clp_primal_simplex: bool = False
    
    # Symphony-specific parameters
    symphony_threads: int = 1  # Sequential per foundations
    
    # Output preferences
    generate_csv: bool = True
    generate_parquet: bool = True
    generate_json: bool = True
    generate_metadata: bool = True
    
    # Logging preferences
    log_level: str = "INFO"
    log_console: bool = True
    log_file: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'time_limit_seconds': self.time_limit_seconds,
            'optimality_gap': self.optimality_gap,
            'feasibility_tolerance': self.feasibility_tolerance,
            'optimality_tolerance': self.optimality_tolerance,
            'preferred_solver': self.preferred_solver.value if self.preferred_solver else None,
            'fallback_solvers': [s.value for s in self.fallback_solvers],
            'memory_limit_mb': self.memory_limit_mb,
            'cbc_threads': self.cbc_threads,
            'cbc_strong_branching': self.cbc_strong_branching,
            'cbc_cuts': self.cbc_cuts,
            'glpk_presolve': self.glpk_presolve,
            'glpk_scale': self.glpk_scale,
            'highs_presolve': self.highs_presolve,
            'highs_parallel': self.highs_parallel,
            'clp_dual_simplex': self.clp_dual_simplex,
            'clp_primal_simplex': self.clp_primal_simplex,
            'symphony_threads': self.symphony_threads,
            'generate_csv': self.generate_csv,
            'generate_parquet': self.generate_parquet,
            'generate_json': self.generate_json,
            'generate_metadata': self.generate_metadata,
            'log_level': self.log_level,
            'log_console': self.log_console,
            'log_file': self.log_file
        }


@dataclass
class PuLPSolverConfig:
    """
    Complete configuration for PuLP solver execution.
    
    Compliance: Dynamic Parametric System Section 6.3
    """
    
    # Input paths
    stage3_output_path: Path
    output_path: Path
    log_path: Path
    error_report_path: Path
    
    # Solver parameters
    solver_params: SolverParameters = field(default_factory=SolverParameters)
    
    # Override parameters from calling module
    override_params: Optional[Dict[str, Any]] = None
    
    # Session metadata
    session_id: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        """Validate and apply overrides."""
        # Apply override parameters if provided
        if self.override_params:
            for key, value in self.override_params.items():
                if hasattr(self.solver_params, key):
                    setattr(self.solver_params, key, value)
        
        # Validate paths
        if not self.stage3_output_path.exists():
            raise ValueError(f"Stage 3 output path does not exist: {self.stage3_output_path}")
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.error_report_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PuLPSolverConfig':
        """Create configuration from dictionary."""
        return cls(
            stage3_output_path=Path(config_dict['stage3_output_path']),
            output_path=Path(config_dict['output_path']),
            log_path=Path(config_dict['log_path']),
            error_report_path=Path(config_dict['error_report_path']),
            solver_params=SolverParameters(**config_dict.get('solver_params', {})),
            override_params=config_dict.get('override_params'),
            session_id=config_dict.get('session_id', ''),
            timestamp=config_dict.get('timestamp', '')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'stage3_output_path': str(self.stage3_output_path),
            'output_path': str(self.output_path),
            'log_path': str(self.log_path),
            'error_report_path': str(self.error_report_path),
            'solver_params': self.solver_params.to_dict(),
            'override_params': self.override_params,
            'session_id': self.session_id,
            'timestamp': self.timestamp
        }
    
    def save_to_file(self, file_path: Path):
        """Save configuration to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> 'PuLPSolverConfig':
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            return cls.from_dict(json.load(f))


class ConfigValidator:
    """Validate configuration against theoretical foundations."""
    
    @staticmethod
    def validate(config: PuLPSolverConfig) -> tuple[bool, List[str]]:
        """
        Validate configuration compliance with foundations.
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Validate paths
        if not config.stage3_output_path.exists():
            errors.append(f"Stage 3 output path does not exist: {config.stage3_output_path}")
        
        # Validate solver parameters
        if config.solver_params.optimality_gap < 0 or config.solver_params.optimality_gap > 1:
            errors.append(f"Optimality gap must be in [0, 1]: {config.solver_params.optimality_gap}")
        
        if config.solver_params.feasibility_tolerance <= 0:
            errors.append(f"Feasibility tolerance must be positive: {config.solver_params.feasibility_tolerance}")
        
        if config.solver_params.optimality_tolerance <= 0:
            errors.append(f"Optimality tolerance must be positive: {config.solver_params.optimality_tolerance}")
        
        # Validate solver selection
        if config.solver_params.preferred_solver and config.solver_params.preferred_solver not in SolverType:
            errors.append(f"Invalid preferred solver: {config.solver_params.preferred_solver}")
        
        # Validate time limits (should be None or positive per foundations)
        if config.solver_params.time_limit_seconds is not None and config.solver_params.time_limit_seconds <= 0:
            errors.append(f"Time limit must be positive or None: {config.solver_params.time_limit_seconds}")
        
        # Validate memory limits (should be None or positive per foundations)
        if config.solver_params.memory_limit_mb is not None and config.solver_params.memory_limit_mb <= 0:
            errors.append(f"Memory limit must be positive or None: {config.solver_params.memory_limit_mb}")
        
        return len(errors) == 0, errors


