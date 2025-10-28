"""
Stage 6.3 DEAP Solver Family - Configuration System

Implements configuration management with dynamic parameter integration
per Dynamic Parametric System (Section 3.1) and DEAP Foundation (Section 3).

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd


@dataclass
class SolverParameters:
    """Evolutionary algorithm parameters from dynamic parameters."""
    
    # Population parameters
    population_size: int = 100
    max_generations: int = 1000
    convergence_threshold: float = 1e-6
    stagnation_generations: int = 50
    
    # Selection parameters
    selection_method: str = "tournament"
    tournament_size: int = 3
    selection_pressure: Optional[float] = None  # Calculated dynamically
    
    # Crossover parameters
    crossover_rate: float = 0.8
    crossover_operators: List[str] = field(default_factory=lambda: ["ox", "pmx", "cx", "uniform"])
    
    # Mutation parameters
    mutation_rate: Optional[float] = None  # Calculated per Theorem 3.8
    mutation_operators: List[str] = field(default_factory=lambda: ["swap", "insertion", "inversion", "scramble"])
    
    # DEAP-specific parameters
    elitism_count: int = 2
    diversity_threshold: float = 0.3
    
    # DE-specific parameters
    differential_weight: float = 0.8
    crossover_probability: float = 0.9
    
    # PSO-specific parameters
    inertia_weight: float = 0.7
    cognitive_coefficient: float = 1.5
    social_coefficient: float = 1.5
    
    # ES-specific parameters
    success_rate_threshold: float = 0.2  # 1/5 rule
    adaptation_constant: float = 0.817
    recombination_parents: int = 2
    
    # NSGA-II specific parameters
    crowding_distance_threshold: float = 1e-6
    
    # Logging parameters
    log_level: str = "INFO"
    log_console: bool = True
    log_file: bool = True


@dataclass
class DEAPConfig:
    """
    DEAP Solver Family Configuration
    
    Integrates dynamic parameters from Stage 3 with theoretical defaults
    per Stage-6.3 DEAP Foundation and Dynamic Parametric System.
    """
    
    # Paths
    stage3_output_path: Path
    output_path: Path
    log_path: Path
    error_report_path: Path
    
    # Solver selection
    solver_type: str = "nsga2"  # "ga", "gp", "es", "de", "pso", "nsga2"
    auto_select_solver: bool = True
    
    # Solver parameters
    solver_params: SolverParameters = field(default_factory=SolverParameters)
    
    # Override parameters from calling module
    override_params: Optional[Dict[str, Any]] = None
    
    # Execution parameters
    time_limit_seconds: Optional[float] = None
    memory_limit_mb: Optional[float] = None
    
    # Validation parameters
    enable_theorem_validation: bool = True
    enable_numerical_validation: bool = True
    validation_tolerance: float = 1e-6
    
    def __post_init__(self):
        """Post-initialization: Load dynamic parameters and apply overrides."""
        self._load_dynamic_parameters()
        self._apply_overrides()
        self._validate_configuration()
    
    def _load_dynamic_parameters(self):
        """Load dynamic parameters from Stage 3 outputs."""
        try:
            # Load dynamic_parameters.parquet from LRAW
            dynamic_params_path = self.stage3_output_path / "L_raw" / "dynamic_parameters.parquet"
            
            if dynamic_params_path.exists():
                df = pd.read_parquet(dynamic_params_path)
                
                # Filter for DEAP solver parameters (path starts with "solver.deap.")
                deap_params = df[df['path'].str.startswith('solver.deap.', na=False)]
                
                # Extract parameters
                for _, row in deap_params.iterrows():
                    param_name = row['code']
                    param_value = self._extract_parameter_value(row)
                    
                    # Map to solver parameters
                    self._set_solver_parameter(param_name, param_value)
                
                logging.info(f"Loaded {len(deap_params)} dynamic parameters")
            else:
                logging.warning(f"Dynamic parameters file not found: {dynamic_params_path}")
        
        except Exception as e:
            logging.error(f"Failed to load dynamic parameters: {e}")
            logging.info("Using default parameters")
    
    def _extract_parameter_value(self, row: pd.Series) -> Any:
        """Extract parameter value based on data type."""
        data_type = row.get('data_type', 'text')
        
        if data_type == 'integer':
            return row.get('value_integer')
        elif data_type == 'numeric':
            return row.get('value_numeric')
        elif data_type == 'boolean':
            return row.get('value_boolean')
        elif data_type == 'json':
            import json
            return json.loads(row.get('value_json', '{}'))
        else:
            return row.get('value_text')
    
    def _set_solver_parameter(self, param_name: str, param_value: Any):
        """Set solver parameter by name."""
        # Map parameter names to attributes
        param_mapping = {
            'population_size': 'population_size',
            'max_generations': 'max_generations',
            'convergence_threshold': 'convergence_threshold',
            'stagnation_generations': 'stagnation_generations',
            'selection_method': 'selection_method',
            'tournament_size': 'tournament_size',
            'crossover_rate': 'crossover_rate',
            'mutation_rate': 'mutation_rate',
            'elitism_count': 'elitism_count',
            'diversity_threshold': 'diversity_threshold',
            'differential_weight': 'differential_weight',
            'crossover_probability': 'crossover_probability',
            'inertia_weight': 'inertia_weight',
            'cognitive_coefficient': 'cognitive_coefficient',
            'social_coefficient': 'social_coefficient',
            'success_rate_threshold': 'success_rate_threshold',
            'adaptation_constant': 'adaptation_constant',
            'recombination_parents': 'recombination_parents',
            'crowding_distance_threshold': 'crowding_distance_threshold',
            'log_level': 'log_level',
        }
        
        if param_name in param_mapping:
            attr_name = param_mapping[param_name]
            setattr(self.solver_params, attr_name, param_value)
    
    def _apply_overrides(self):
        """Apply override parameters from calling module."""
        if self.override_params:
            for key, value in self.override_params.items():
                if hasattr(self.solver_params, key):
                    setattr(self.solver_params, key, value)
                elif hasattr(self, key):
                    setattr(self, key, value)
    
    def _validate_configuration(self):
        """Validate configuration against theoretical bounds."""
        # Validate population size
        if self.solver_params.population_size < 10:
            raise ValueError("Population size must be at least 10")
        
        # Validate tournament size
        if self.solver_params.tournament_size < 2:
            raise ValueError("Tournament size must be at least 2")
        
        # Validate crossover rate
        if not 0 <= self.solver_params.crossover_rate <= 1:
            raise ValueError("Crossover rate must be in [0, 1]")
        
        # Validate mutation rate (if specified)
        if self.solver_params.mutation_rate is not None:
            if not 0 <= self.solver_params.mutation_rate <= 1:
                raise ValueError("Mutation rate must be in [0, 1]")
        
        # Validate PSO parameters (Theorem 7.3)
        if self.solver_type == "pso":
            phi = self.solver_params.cognitive_coefficient + self.solver_params.social_coefficient
            if phi <= 4:
                raise ValueError(f"PSO convergence requires φ > 4, got φ = {phi}")
    
    def get_solver_config(self) -> Dict[str, Any]:
        """Get solver-specific configuration."""
        base_config = {
            'population_size': self.solver_params.population_size,
            'max_generations': self.solver_params.max_generations,
            'convergence_threshold': self.solver_params.convergence_threshold,
            'stagnation_generations': self.solver_params.stagnation_generations,
            'log_level': self.solver_params.log_level,
        }
        
        if self.solver_type in ['ga', 'gp', 'nsga2']:
            base_config.update({
                'selection_method': self.solver_params.selection_method,
                'tournament_size': self.solver_params.tournament_size,
                'crossover_rate': self.solver_params.crossover_rate,
                'mutation_rate': self.solver_params.mutation_rate,
                'crossover_operators': self.solver_params.crossover_operators,
                'mutation_operators': self.solver_params.mutation_operators,
                'elitism_count': self.solver_params.elitism_count,
                'diversity_threshold': self.solver_params.diversity_threshold,
            })
        
        if self.solver_type == 'de':
            base_config.update({
                'differential_weight': self.solver_params.differential_weight,
                'crossover_probability': self.solver_params.crossover_probability,
            })
        
        if self.solver_type == 'pso':
            base_config.update({
                'inertia_weight': self.solver_params.inertia_weight,
                'cognitive_coefficient': self.solver_params.cognitive_coefficient,
                'social_coefficient': self.solver_params.social_coefficient,
            })
        
        if self.solver_type == 'es':
            base_config.update({
                'success_rate_threshold': self.solver_params.success_rate_threshold,
                'adaptation_constant': self.solver_params.adaptation_constant,
                'recombination_parents': self.solver_params.recombination_parents,
            })
        
        if self.solver_type == 'nsga2':
            base_config.update({
                'crowding_distance_threshold': self.solver_params.crowding_distance_threshold,
            })
        
        return base_config


class ConfigValidator:
    """Configuration validator per theoretical foundations."""
    
    @staticmethod
    def validate(config: DEAPConfig) -> tuple[bool, List[str]]:
        """
        Validate configuration.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check paths exist
        if not config.stage3_output_path.exists():
            errors.append(f"Stage 3 output path does not exist: {config.stage3_output_path}")
        
        # Check solver type is valid
        valid_solvers = ['ga', 'gp', 'es', 'de', 'pso', 'nsga2']
        if config.solver_type not in valid_solvers:
            errors.append(f"Invalid solver type: {config.solver_type}. Must be one of {valid_solvers}")
        
        # Check time limit is positive
        if config.time_limit_seconds is not None and config.time_limit_seconds <= 0:
            errors.append("Time limit must be positive")
        
        # Check memory limit is positive
        if config.memory_limit_mb is not None and config.memory_limit_mb <= 0:
            errors.append("Memory limit must be positive")
        
        # Check validation tolerance is positive
        if config.validation_tolerance <= 0:
            errors.append("Validation tolerance must be positive")
        
        return len(errors) == 0, errors

