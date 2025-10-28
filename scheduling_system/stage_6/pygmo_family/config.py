"""
Configuration Management for PyGMO Solver Family (Stage 6.4)

This module provides centralized configuration management for the PyGMO solver,
including algorithm parameters, archipelago settings, and validation thresholds.

Theoretical Foundation: Dynamic Parametric System - Formal Analysis
Section: 10.2 Application Layer Integration
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
from enum import Enum


class SolverType(Enum):
    """Supported PyGMO solver algorithms"""
    NSGA_II = "NSGA-II"
    MOEA_D = "MOEA/D"
    PSO = "PSO"
    DE = "DE"
    SA = "SA"
    MIXED = "MIXED"  # Portfolio approach


class MigrationTopology(Enum):
    """Migration topology types"""
    RING = "ring"
    FULLY_CONNECTED = "fully_connected"
    SMALL_WORLD = "small_world"  # Optimal per Theorem 5.2
    STAR = "star"


@dataclass
class PyGMOConfig:
    """
    Centralized configuration for PyGMO Solver Family.
    
    Theoretical Foundation:
    - Section 2.3: Archipelago Architecture
    - Section 5.1-5.3: Island Model and Migration
    - Section 6.2: Hyperparameter Optimization
    - Section 13.2: Implementation Guidelines
    """
    
    # ============================================================================
    # INPUT/OUTPUT PATHS
    # ============================================================================
    input_dir: Path
    output_dir: Path
    log_dir: Path
    
    # ============================================================================
    # PYGMO ARCHIPELAGO PARAMETERS (Section 2.3, 5.1-5.3)
    # ============================================================================
    population_size: int = 100  # Population per island
    num_islands: int = 8  # Number of islands in archipelago
    generations: int = 1000  # Maximum generations
    
    # Migration Parameters (Section 5.2-5.3)
    migration_rate: float = 0.1  # Fraction of population to migrate
    migration_frequency: int = 10  # Generations between migrations
    migration_topology: MigrationTopology = MigrationTopology.SMALL_WORLD  # Theorem 5.2
    migration_selection: str = "best"  # "best" or "random" (Algorithm 5.3, 5.4)
    
    # ============================================================================
    # ALGORITHM CONFIGURATION (Section 13.2)
    # ============================================================================
    default_solver: SolverType = SolverType.NSGA_II  # Per Section 13.2
    
    # NSGA-II Parameters (Section 3.1)
    nsga2_crossover_prob: float = 0.9
    nsga2_mutation_prob: float = 0.1
    nsga2_crossover_eta: float = 20.0
    nsga2_mutation_eta: float = 20.0
    
    # MOEA/D Parameters (Section 3.2)
    moead_weight_generation: str = "grid"  # "grid" or "random"
    moead_neighbours: int = 20
    moead_decomposition: str = "tchebycheff"  # "tchebycheff" or "weighted"
    
    # PSO Parameters (Section 3.3)
    pso_omega: float = 0.7298  # Inertia weight
    pso_c1: float = 2.05  # Cognitive parameter
    pso_c2: float = 2.05  # Social parameter
    pso_variant: int = 5  # PSO variant (1-6)
    
    # Differential Evolution Parameters (Section 3.4)
    de_variant: int = 2  # DE/rand/1/bin
    de_f: float = 0.8  # Differential weight
    de_cr: float = 0.9  # Crossover probability
    
    # Simulated Annealing Parameters (Section 3.5)
    sa_temp_start: float = 10.0
    sa_temp_end: float = 0.01
    sa_n_temp_adj: int = 10
    sa_n_range_adj: int = 10
    sa_bin_size: int = 10
    
    # ============================================================================
    # CONSTRAINT HANDLING (Section 4)
    # ============================================================================
    penalty_multiplier: float = 1000.0  # Penalty for constraint violations
    constraint_tolerance: float = 1e-6  # Feasibility tolerance
    
    # ============================================================================
    # CONVERGENCE CRITERIA (Section 7.2)
    # ============================================================================
    convergence_tolerance: float = 1e-4  # Hypervolume stagnation threshold
    stagnation_generations: int = 50  # Generations without improvement
    target_hypervolume: Optional[float] = None  # Target HV (if known)
    hypervolume_ref_point: Optional[List[float]] = None  # Reference point for HV calculation
    
    # ============================================================================
    # CHECKPOINTING (Section 8.2)
    # ============================================================================
    enable_checkpoints: bool = True
    checkpoint_frequency: int = 100  # Generations between checkpoints
    checkpoint_dir: Optional[Path] = None
    
    # ============================================================================
    # LOGGING CONFIGURATION (Section 18)
    # ============================================================================
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    # Error report file names expected by ErrorReporter
    error_report_file_name: str = "error_report.json"
    error_report_txt_name: str = "error_report.txt"
    log_to_console: bool = True
    log_to_file: bool = True
    log_json_format: bool = True
    
    # ============================================================================
    # VALIDATION (INTERNAL ONLY - Stage 7 handles output validation)
    # ============================================================================
    validate_theorems: bool = True  # Validate mathematical theorems
    validate_constraints: bool = True  # Internal constraint validation
    # Fallback control for orchestrator error handling
    enable_fallback: bool = True
    
    # ============================================================================
    # PERFORMANCE OPTIMIZATION (Section 9)
    # ============================================================================
    enable_parallelism: bool = True  # Use archipelago parallelism
    max_threads: Optional[int] = None  # None = auto-detect
    memory_limit_mb: Optional[int] = None  # None = no limit (per foundations)
    
    # ============================================================================
    # DYNAMIC PARAMETERS (Dynamic Parametric System)
    # ============================================================================
    dynamic_params: Dict[str, Any] = field(default_factory=dict)
    
    # ============================================================================
    # ADVANCED FEATURES (Section 11)
    # ============================================================================
    enable_hyperparameter_optimization: bool = False
    enable_adaptive_migration: bool = True  # Section 5.3
    
    def __post_init__(self):
        """Validate configuration and set derived parameters"""
        # Convert string paths to Path objects
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        self.log_dir = Path(self.log_dir)
        
        # Set checkpoint directory if not specified
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.output_dir / "checkpoints"
        else:
            self.checkpoint_dir = Path(self.checkpoint_dir)
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate configuration parameters"""
        assert self.population_size > 0, "Population size must be positive"
        assert self.num_islands > 0, "Number of islands must be positive"
        assert self.generations > 0, "Generations must be positive"
        assert 0 < self.migration_rate < 1, "Migration rate must be in (0, 1)"
        assert self.migration_frequency > 0, "Migration frequency must be positive"
        assert 0 < self.convergence_tolerance < 1, "Convergence tolerance must be in (0, 1)"
        assert self.stagnation_generations > 0, "Stagnation generations must be positive"
        
        # Validate algorithm-specific parameters
        assert 0 <= self.nsga2_crossover_prob <= 1, "NSGA-II crossover probability must be in [0, 1]"
        assert 0 <= self.nsga2_mutation_prob <= 1, "NSGA-II mutation probability must be in [0, 1]"
        assert self.de_f > 0, "DE differential weight must be positive"
        assert 0 <= self.de_cr <= 1, "DE crossover probability must be in [0, 1]"
        assert self.sa_temp_start > self.sa_temp_end, "SA start temperature must be > end temperature"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "log_dir": str(self.log_dir),
            "population_size": self.population_size,
            "num_islands": self.num_islands,
            "generations": self.generations,
            "migration_rate": self.migration_rate,
            "migration_frequency": self.migration_frequency,
            "migration_topology": self.migration_topology.value,
            "migration_selection": self.migration_selection,
            "default_solver": self.default_solver.value,
            "convergence_tolerance": self.convergence_tolerance,
            "stagnation_generations": self.stagnation_generations,
            "enable_checkpoints": self.enable_checkpoints,
            "checkpoint_frequency": self.checkpoint_frequency,
            "log_level": self.log_level,
            "validate_theorems": self.validate_theorems,
            "validate_constraints": self.validate_constraints,
            "enable_parallelism": self.enable_parallelism,
            "dynamic_params": self.dynamic_params,
        }
    
    def to_json(self, filepath: Path):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PyGMOConfig':
        """Load configuration from dictionary"""
        # Convert enum strings back to enums
        if 'default_solver' in config_dict:
            config_dict['default_solver'] = SolverType(config_dict['default_solver'])
        if 'migration_topology' in config_dict:
            config_dict['migration_topology'] = MigrationTopology(config_dict['migration_topology'])
        
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, filepath: Path) -> 'PyGMOConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def apply_dynamic_parameters(self, dynamic_params: Dict[str, Any]):
        """
        Apply dynamic parameters from Stage 3 output.
        
        Theoretical Foundation: Dynamic Parametric System - Formal Analysis
        Section 5.2: Parameter Activation Mechanisms
        Section 6.3: Solver Configuration Parameters
        """
        # Extract PyGMO-specific parameters
        for key, value in dynamic_params.items():
            if key.startswith("solver.pygmo."):
                param_name = key.replace("solver.pygmo.", "")
                if hasattr(self, param_name):
                    setattr(self, param_name, value)
                    self.dynamic_params[param_name] = value
            elif key.startswith("optimization."):
                param_name = key.replace("optimization.", "")
                if hasattr(self, param_name):
                    setattr(self, param_name, value)
                    self.dynamic_params[param_name] = value


def load_config(
    input_dir: Path,
    output_dir: Path,
    log_dir: Path,
    config_file: Optional[Path] = None,
    **overrides
) -> PyGMOConfig:
    """
    Load configuration with optional overrides.
    
    Args:
        input_dir: Path to Stage 3 outputs
        output_dir: Path for Stage 6.4 outputs
        log_dir: Path for logs
        config_file: Optional JSON configuration file
        **overrides: Additional configuration overrides
    
    Returns:
        PyGMOConfig instance
    """
    if config_file and config_file.exists():
        config = PyGMOConfig.from_json(config_file)
        # Override paths
        config.input_dir = input_dir
        config.output_dir = output_dir
        config.log_dir = log_dir
    else:
        config = PyGMOConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            log_dir=log_dir
        )
    
    # Apply additional overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


