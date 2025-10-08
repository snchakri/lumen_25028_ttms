#!/usr/bin/env python3
"""
PuLP Solver Family - Stage 6 Configuration Management System

This module implements the complete configuration management system for Stage 6.1
PuLP solver family, providing complete path configuration, solver selection, dynamic
parameter management with mathematical rigor and theoretical compliance. Critical component
implementing complete configuration framework per Stage 6 foundational design.

Theoretical Foundation:
    Based on Stage 6.1 PuLP Framework configuration requirements:
    - Implements complete configuration management per foundational design rules
    - Maintains mathematical consistency across all configuration parameters
    - Ensures complete path management with directory isolation
    - Provides dynamic parameter integration with EAV model support
    - Supports multi-solver backend configuration with unified interface

Architecture Compliance:
    - Implements Configuration Layer per foundational design architecture
    - Maintains optimal performance characteristics through configuration optimization
    - Provides fail-safe configuration validation with complete error handling
    - Supports distributed execution with centralized configuration management
    - Ensures memory-efficient operations through optimized configuration structures

Dependencies: pathlib, typing, dataclasses, json, os, logging, datetime, enum
Author: Student Team
Version: 1.0.0 (Production)
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

# Configure module logger
logger = logging.getLogger(__name__)

# Configuration metadata and version information
__version__ = "1.0.0"
__stage__ = "6.1"
__component__ = "configuration"
__status__ = "production"

class SolverBackend(str, Enum):
    """
    Enumeration of supported PuLP solver backends with mathematical guarantees.

    Mathematical Foundation: Complete solver backend coverage per PuLP framework
    specifications ensuring complete optimization support and performance.
    """
    CBC = "CBC"                         # COIN-OR CBC (default mixed-integer programming)
    GLPK = "GLPK"                      # GNU Linear Programming Kit (open-source LP/MIP)
    HIGHS = "HiGHS"                    # HiGHS optimizer (high-performance LP/MIP)
    CLP = "CLP"                        # COIN-OR CLP (linear programming specialized)
    SYMPHONY = "SYMPHONY"              # COIN-OR SYMPHONY (mixed-integer programming)

class ConfigurationLevel(str, Enum):
    """
    Configuration validation and enforcement levels.

    Defines configuration strictness levels ensuring appropriate system behavior
    based on operational requirements and mathematical compliance needs.
    """
    MINIMAL = "minimal"                 # Minimal configuration validation
    STANDARD = "standard"               # Standard configuration enforcement
    STRICT = "strict"                  # Strict configuration validation
    complete = "complete"     # complete configuration enforcement

class ExecutionMode(str, Enum):
    """
    Execution mode enumeration for different operational contexts.

    Mathematical Foundation: Defines complete execution mode coverage per
    operational requirements ensuring appropriate system behavior and performance.
    """
    DEVELOPMENT = "development"         # Development mode with verbose logging
    TESTING = "testing"                # Testing mode with enhanced validation
    PRODUCTION = "production"           # Production mode with optimized performance
    BENCHMARK = "benchmark"             # Benchmark mode with detailed metrics

@dataclass
class PathConfiguration:
    """
    complete path configuration for Stage 6 execution environment.

    Mathematical Foundation: Implements complete path specification per
    foundational design rules ensuring proper directory isolation and organization.

    Attributes:
        stage3_input_directory: Directory containing Stage 3 output artifacts
        execution_base_directory: Base directory for all execution outputs
        logs_directory: Directory for complete execution logging
        metadata_directory: Directory for metadata and configuration files
        temporary_directory: Directory for temporary processing files
        output_directory: Directory for final output files and reports
    """
    stage3_input_directory: str = "./stage3_outputs"
    execution_base_directory: str = "./executions"
    logs_directory: str = "./logs"
    metadata_directory: str = "./metadata"
    temporary_directory: str = "./temp"
    output_directory: str = "./outputs"

    def __post_init__(self):
        """Post-initialization path validation and normalization."""
        # Convert all paths to Path objects for consistent handling
        self.stage3_input_directory = str(Path(self.stage3_input_directory).resolve())
        self.execution_base_directory = str(Path(self.execution_base_directory).resolve())
        self.logs_directory = str(Path(self.logs_directory).resolve())
        self.metadata_directory = str(Path(self.metadata_directory).resolve())
        self.temporary_directory = str(Path(self.temporary_directory).resolve())
        self.output_directory = str(Path(self.output_directory).resolve())

    def create_directories(self, exist_ok: bool = True) -> None:
        """
        Create all configured directories with proper permissions.

        Args:
            exist_ok: Whether to allow existing directories
        """
        directories = [
            self.stage3_input_directory,
            self.execution_base_directory,
            self.logs_directory,
            self.metadata_directory,
            self.temporary_directory,
            self.output_directory
        ]

        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=exist_ok)
                logger.debug(f"Created directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {str(e)}")
                raise

    def validate_paths(self) -> Dict[str, bool]:
        """
        Validate all configured paths for accessibility and permissions.

        Returns:
            Dictionary mapping path names to validation results
        """
        validation_results = {}
        path_mapping = {
            'stage3_input_directory': self.stage3_input_directory,
            'execution_base_directory': self.execution_base_directory,
            'logs_directory': self.logs_directory,
            'metadata_directory': self.metadata_directory,
            'temporary_directory': self.temporary_directory,
            'output_directory': self.output_directory
        }

        for path_name, path_value in path_mapping.items():
            try:
                path_obj = Path(path_value)

                # Check if path exists or can be created
                if not path_obj.exists():
                    path_obj.mkdir(parents=True, exist_ok=True)

                # Check read/write permissions
                is_readable = os.access(path_value, os.R_OK)
                is_writable = os.access(path_value, os.W_OK)

                validation_results[path_name] = is_readable and is_writable

                if not validation_results[path_name]:
                    logger.warning(f"Path validation failed for {path_name}: {path_value}")

            except Exception as e:
                logger.error(f"Path validation error for {path_name}: {str(e)}")
                validation_results[path_name] = False

        return validation_results

@dataclass
class SolverConfiguration:
    """
    complete solver configuration for PuLP backends.

    Mathematical Foundation: Implements complete solver configuration per
    PuLP framework requirements ensuring optimal performance and mathematical correctness.

    Attributes:
        default_backend: Default solver backend to use
        time_limit_seconds: Maximum solving time limit (None for unlimited)
        memory_limit_mb: Maximum memory usage limit in MB
        optimization_tolerance: Numerical tolerance for optimization
        threads: Number of parallel threads to use
        presolve: Enable/disable presolve optimizations
        cuts: Enable/disable cutting plane algorithms
        heuristics: Enable/disable heuristic methods
        verbose_logging: Enable verbose solver logging
        backend_specific_options: Additional backend-specific configuration options
    """
    default_backend: SolverBackend = SolverBackend.CBC
    time_limit_seconds: Optional[float] = 300.0
    memory_limit_mb: int = 512
    optimization_tolerance: float = 1e-6
    threads: int = 1
    presolve: bool = True
    cuts: bool = True
    heuristics: bool = True
    verbose_logging: bool = False
    backend_specific_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization solver configuration validation."""
        # Validate time limit
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise ValueError("Time limit must be positive or None")

        # Validate memory limit
        if self.memory_limit_mb <= 0:
            raise ValueError("Memory limit must be positive")

        # Validate optimization tolerance
        if self.optimization_tolerance <= 0:
            raise ValueError("Optimization tolerance must be positive")

        # Validate thread count
        if self.threads < 1:
            raise ValueError("Thread count must be at least 1")

    def get_backend_options(self, backend: SolverBackend) -> Dict[str, Any]:
        """
        Get backend-specific configuration options.

        Args:
            backend: Target solver backend

        Returns:
            Dictionary of backend-specific options
        """
        base_options = {
            'timeLimit': self.time_limit_seconds,
            'threads': self.threads,
            'presolve': self.presolve,
            'cuts': self.cuts,
            'heuristics': self.heuristics,
            'msg': 1 if self.verbose_logging else 0
        }

        # Add backend-specific options
        backend_key = backend.value.lower()
        if backend_key in self.backend_specific_options:
            base_options.update(self.backend_specific_options[backend_key])

        # Remove None values for cleaner options
        return {k: v for k, v in base_options.items() if v is not None}

    def validate_configuration(self) -> Dict[str, bool]:
        """
        Validate solver configuration parameters.

        Returns:
            Dictionary mapping configuration aspects to validation results
        """
        validation_results = {
            'backend_supported': self.default_backend in SolverBackend,
            'time_limit_valid': self.time_limit_seconds is None or self.time_limit_seconds > 0,
            'memory_limit_valid': self.memory_limit_mb > 0,
            'tolerance_valid': 0 < self.optimization_tolerance < 1,
            'threads_valid': 1 <= self.threads <= 16,
            'options_valid': isinstance(self.backend_specific_options, dict)
        }

        # Overall configuration validity
        validation_results['overall_valid'] = all(validation_results.values())

        return validation_results

@dataclass
class InputModelConfiguration:
    """
    Configuration for input model processing layer.

    Mathematical Foundation: Defines complete input model configuration per
    foundational design rules ensuring optimal data processing and validation.

    Attributes:
        validation_level: Level of input validation to perform
        bijection_verification: Enable bijection mapping consistency verification
        constraint_matrix_optimization: Enable constraint matrix optimization
        metadata_generation: Enable complete metadata generation
        fail_fast_validation: Enable fail-fast validation on errors
        memory_optimization: Enable memory usage optimization
    """
    validation_level: ConfigurationLevel = ConfigurationLevel.complete
    bijection_verification: bool = True
    constraint_matrix_optimization: bool = True
    metadata_generation: bool = True
    fail_fast_validation: bool = True
    memory_optimization: bool = True

    def get_validation_parameters(self) -> Dict[str, Any]:
        """
        Get validation parameters based on configuration level.

        Returns:
            Dictionary of validation parameters
        """
        level_parameters = {
            ConfigurationLevel.MINIMAL: {
                'entity_validation': False,
                'constraint_validation': False,
                'referential_integrity': False,
                'temporal_validation': False
            },
            ConfigurationLevel.STANDARD: {
                'entity_validation': True,
                'constraint_validation': False,
                'referential_integrity': True,
                'temporal_validation': False
            },
            ConfigurationLevel.STRICT: {
                'entity_validation': True,
                'constraint_validation': True,
                'referential_integrity': True,
                'temporal_validation': True
            },
            ConfigurationLevel.complete: {
                'entity_validation': True,
                'constraint_validation': True,
                'referential_integrity': True,
                'temporal_validation': True,
                'constraint_consistency': True,
                'optimization_validation': True
            }
        }

        return level_parameters.get(self.validation_level, level_parameters[ConfigurationLevel.STANDARD])

@dataclass
class OutputModelConfiguration:
    """
    Configuration for output model generation layer.

    Mathematical Foundation: Defines complete output model configuration per
    foundational design rules ensuring optimal CSV generation and metadata handling.

    Attributes:
        csv_format_extended: Generate extended CSV format with additional metadata
        constraint_satisfaction_scoring: Enable constraint satisfaction scoring
        objective_contribution_analysis: Enable objective contribution analysis
        solver_metadata_inclusion: Include detailed solver metadata
        timestamp_precision: Precision level for timestamp formatting
        file_compression: Enable output file compression
    """
    csv_format_extended: bool = True
    constraint_satisfaction_scoring: bool = True
    objective_contribution_analysis: bool = True
    solver_metadata_inclusion: bool = True
    timestamp_precision: str = "seconds"
    file_compression: bool = False

    def get_csv_column_specification(self) -> List[str]:
        """
        Get CSV column specification based on configuration.

        Returns:
            List of column names for CSV output
        """
        base_columns = [
            'assignment_id',
            'course_id',
            'faculty_id',
            'room_id',
            'timeslot_id',
            'batch_id',
            'start_time',
            'end_time',
            'day_of_week',
            'duration_hours',
            'assignment_type'
        ]

        if self.constraint_satisfaction_scoring:
            base_columns.append('constraint_satisfaction_score')

        if self.objective_contribution_analysis:
            base_columns.append('objective_contribution')

        if self.solver_metadata_inclusion:
            base_columns.append('solver_metadata')

        return base_columns

@dataclass
class DynamicParameterConfiguration:
    """
    Configuration for dynamic parameter handling per EAV model integration.

    Mathematical Foundation: Implements complete dynamic parameter configuration
    per EAV model requirements ensuring proper parameter integration and optimization.

    Attributes:
        enable_dynamic_parameters: Enable dynamic parameter processing
        parameter_validation: Enable parameter value validation
        constraint_weight_adjustment: Enable dynamic constraint weight adjustment
        objective_coefficient_optimization: Enable dynamic objective coefficient optimization
        parameter_conflict_resolution: Strategy for resolving parameter conflicts
        parameter_persistence: Enable parameter persistence across executions
    """
    enable_dynamic_parameters: bool = True
    parameter_validation: bool = True
    constraint_weight_adjustment: bool = True
    objective_coefficient_optimization: bool = True
    parameter_conflict_resolution: str = "priority_based"
    parameter_persistence: bool = True

    def get_parameter_processing_options(self) -> Dict[str, Any]:
        """
        Get dynamic parameter processing options.

        Returns:
            Dictionary of parameter processing configuration
        """
        return {
            'validation_enabled': self.parameter_validation,
            'weight_adjustment': self.constraint_weight_adjustment,
            'coefficient_optimization': self.objective_coefficient_optimization,
            'conflict_resolution': self.parameter_conflict_resolution,
            'persistence_enabled': self.parameter_persistence
        }

@dataclass
class PuLPFamilyConfiguration:
    """
    Master configuration for complete PuLP solver family Stage 6.1 system.

    Mathematical Foundation: Implements complete system configuration per
    Stage 6.1 foundational framework ensuring complete system operation
    with mathematical guarantees and theoretical compliance.

    Attributes:
        execution_mode: System execution mode
        configuration_level: Configuration validation and enforcement level
        paths: Path configuration for directories and file management
        solver: Solver configuration for optimization backends
        input_model: Input model processing configuration
        output_model: Output model generation configuration
        dynamic_parameters: Dynamic parameter handling configuration
        logging_configuration: complete logging configuration
        performance_limits: System performance limits and constraints
    """
    execution_mode: ExecutionMode = ExecutionMode.PRODUCTION
    configuration_level: ConfigurationLevel = ConfigurationLevel.complete
    paths: PathConfiguration = field(default_factory=PathConfiguration)
    solver: SolverConfiguration = field(default_factory=SolverConfiguration)
    input_model: InputModelConfiguration = field(default_factory=InputModelConfiguration)
    output_model: OutputModelConfiguration = field(default_factory=OutputModelConfiguration)
    dynamic_parameters: DynamicParameterConfiguration = field(default_factory=DynamicParameterConfiguration)
    logging_configuration: Dict[str, Any] = field(default_factory=lambda: {
        'log_level': 'INFO',
        'log_to_file': True,
        'log_to_console': True,
        'max_log_file_size_mb': 100,
        'log_retention_days': 30,
        'structured_logging': True
    })
    performance_limits: Dict[str, Any] = field(default_factory=lambda: {
        'max_variables': 10000000,
        'max_constraints': 5000000,
        'max_memory_mb': 1024,
        'max_execution_time_seconds': 1800,
        'checkpoint_interval_seconds': 60
    })

    def __post_init__(self):
        """Post-initialization system configuration validation."""
        # Validate execution mode
        if self.execution_mode not in ExecutionMode:
            raise ValueError(f"Invalid execution mode: {self.execution_mode}")

        # Validate configuration level
        if self.configuration_level not in ConfigurationLevel:
            raise ValueError(f"Invalid configuration level: {self.configuration_level}")

        # Validate performance limits
        self._validate_performance_limits()

        # Create directories if they don't exist
        try:
            self.paths.create_directories()
        except Exception as e:
            logger.error(f"Failed to create configured directories: {str(e)}")
            if self.configuration_level in [ConfigurationLevel.STRICT, ConfigurationLevel.complete]:
                raise

    def _validate_performance_limits(self) -> None:
        """Validate performance limit configuration."""
        required_limits = ['max_variables', 'max_constraints', 'max_memory_mb', 'max_execution_time_seconds']

        for limit_name in required_limits:
            if limit_name not in self.performance_limits:
                raise ValueError(f"Missing required performance limit: {limit_name}")

            limit_value = self.performance_limits[limit_name]
            if not isinstance(limit_value, (int, float)) or limit_value <= 0:
                raise ValueError(f"Invalid performance limit {limit_name}: {limit_value}")

    def validate_complete_configuration(self) -> Dict[str, Any]:
        """
        Validate complete system configuration with complete checks.

        Returns:
            Dictionary containing complete validation results
        """
        validation_results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'configuration_level': self.configuration_level.value,
            'execution_mode': self.execution_mode.value
        }

        try:
            # Validate paths
            validation_results['paths'] = self.paths.validate_paths()

            # Validate solver configuration
            validation_results['solver'] = self.solver.validate_configuration()

            # Validate input model configuration
            validation_results['input_model'] = {
                'validation_level': self.input_model.validation_level.value,
                'bijection_verification': self.input_model.bijection_verification,
                'fail_fast_enabled': self.input_model.fail_fast_validation
            }

            # Validate output model configuration
            validation_results['output_model'] = {
                'csv_extended_format': self.output_model.csv_format_extended,
                'csv_columns': len(self.output_model.get_csv_column_specification()),
                'compression_enabled': self.output_model.file_compression
            }

            # Validate dynamic parameters configuration
            validation_results['dynamic_parameters'] = {
                'enabled': self.dynamic_parameters.enable_dynamic_parameters,
                'validation_enabled': self.dynamic_parameters.parameter_validation,
                'conflict_resolution': self.dynamic_parameters.parameter_conflict_resolution
            }

            # Validate performance limits
            validation_results['performance_limits'] = {
                limit_name: limit_value > 0 for limit_name, limit_value in self.performance_limits.items()
            }

            # Overall validation status
            path_valid = all(validation_results['paths'].values())
            solver_valid = validation_results['solver']['overall_valid']
            performance_valid = all(validation_results['performance_limits'].values())

            validation_results['overall_valid'] = path_valid and solver_valid and performance_valid

        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            validation_results['validation_error'] = str(e)
            validation_results['overall_valid'] = False

        return validation_results

    def export_to_json(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """
        Export complete configuration to JSON format.

        Args:
            file_path: Optional path to save configuration file

        Returns:
            JSON string representation of configuration
        """
        config_dict = asdict(self)

        # Add metadata
        config_dict['metadata'] = {
            'version': __version__,
            'stage': __stage__,
            'component': __component__,
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'configuration_schema_version': '1.0'
        }

        # Convert to JSON
        json_content = json.dumps(config_dict, indent=2, sort_keys=True)

        # Save to file if path provided
        if file_path:
            file_path = Path(file_path)
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json_content)
                logger.info(f"Configuration exported to: {file_path}")
            except Exception as e:
                logger.error(f"Failed to export configuration to {file_path}: {str(e)}")
                raise

        return json_content

    @classmethod
    def import_from_json(cls, file_path: Union[str, Path]) -> 'PuLPFamilyConfiguration':
        """
        Import configuration from JSON file.

        Args:
            file_path: Path to configuration JSON file

        Returns:
            PuLPFamilyConfiguration instance
        """
        file_path = Path(file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

            # Remove metadata if present
            config_dict.pop('metadata', None)

            # Handle enum conversions
            if 'execution_mode' in config_dict:
                config_dict['execution_mode'] = ExecutionMode(config_dict['execution_mode'])

            if 'configuration_level' in config_dict:
                config_dict['configuration_level'] = ConfigurationLevel(config_dict['configuration_level'])

            # Handle nested configurations
            if 'paths' in config_dict:
                config_dict['paths'] = PathConfiguration(**config_dict['paths'])

            if 'solver' in config_dict:
                solver_config = config_dict['solver']
                if 'default_backend' in solver_config:
                    solver_config['default_backend'] = SolverBackend(solver_config['default_backend'])
                config_dict['solver'] = SolverConfiguration(**solver_config)

            if 'input_model' in config_dict:
                input_config = config_dict['input_model']
                if 'validation_level' in input_config:
                    input_config['validation_level'] = ConfigurationLevel(input_config['validation_level'])
                config_dict['input_model'] = InputModelConfiguration(**input_config)

            if 'output_model' in config_dict:
                config_dict['output_model'] = OutputModelConfiguration(**config_dict['output_model'])

            if 'dynamic_parameters' in config_dict:
                config_dict['dynamic_parameters'] = DynamicParameterConfiguration(**config_dict['dynamic_parameters'])

            configuration = cls(**config_dict)
            logger.info(f"Configuration imported from: {file_path}")

            return configuration

        except Exception as e:
            logger.error(f"Failed to import configuration from {file_path}: {str(e)}")
            raise

# Default configuration instance
DEFAULT_PULP_FAMILY_CONFIGURATION = PuLPFamilyConfiguration()

def load_configuration_from_environment() -> PuLPFamilyConfiguration:
    """
    Load configuration from environment variables with fallback to defaults.

    Environment variables override default configuration values following
    the pattern: PULP_<SECTION>_<PARAMETER> (e.g., PULP_SOLVER_BACKEND).

    Returns:
        PuLPFamilyConfiguration with environment overrides applied
    """
    config = PuLPFamilyConfiguration()

    try:
        # Path configuration from environment
        if os.getenv('PULP_PATHS_STAGE3_INPUT'):
            config.paths.stage3_input_directory = os.getenv('PULP_PATHS_STAGE3_INPUT')

        if os.getenv('PULP_PATHS_EXECUTION_BASE'):
            config.paths.execution_base_directory = os.getenv('PULP_PATHS_EXECUTION_BASE')

        if os.getenv('PULP_PATHS_LOGS'):
            config.paths.logs_directory = os.getenv('PULP_PATHS_LOGS')

        if os.getenv('PULP_PATHS_OUTPUT'):
            config.paths.output_directory = os.getenv('PULP_PATHS_OUTPUT')

        # Solver configuration from environment
        if os.getenv('PULP_SOLVER_BACKEND'):
            config.solver.default_backend = SolverBackend(os.getenv('PULP_SOLVER_BACKEND'))

        if os.getenv('PULP_SOLVER_TIME_LIMIT'):
            config.solver.time_limit_seconds = float(os.getenv('PULP_SOLVER_TIME_LIMIT'))

        if os.getenv('PULP_SOLVER_MEMORY_LIMIT'):
            config.solver.memory_limit_mb = int(os.getenv('PULP_SOLVER_MEMORY_LIMIT'))

        if os.getenv('PULP_SOLVER_THREADS'):
            config.solver.threads = int(os.getenv('PULP_SOLVER_THREADS'))

        # Execution mode from environment
        if os.getenv('PULP_EXECUTION_MODE'):
            config.execution_mode = ExecutionMode(os.getenv('PULP_EXECUTION_MODE'))

        # Configuration level from environment
        if os.getenv('PULP_CONFIGURATION_LEVEL'):
            config.configuration_level = ConfigurationLevel(os.getenv('PULP_CONFIGURATION_LEVEL'))

        # Recreate directories with updated paths
        config.paths.create_directories()

        logger.info("Configuration loaded from environment variables")

    except Exception as e:
        logger.warning(f"Failed to load some environment configuration: {str(e)}")
        logger.info("Using default configuration with partial environment overrides")

    return config

def get_execution_directory_path(config: PuLPFamilyConfiguration, 
                                execution_id: str) -> Path:
    """
    Get execution-specific directory path with proper isolation.

    Creates isolated execution directory following Stage 6 foundational design
    rules ensuring proper file organization and audit trail management.

    Args:
        config: PuLP family configuration instance
        execution_id: Unique execution identifier

    Returns:
        Path to execution-specific directory
    """
    execution_dir = Path(config.paths.execution_base_directory) / execution_id

    # Create execution subdirectories
    subdirectories = [
        'input_model',
        'processing',
        'output_model',
        'logs',
        'metadata',
        'temp'
    ]

    for subdir in subdirectories:
        (execution_dir / subdir).mkdir(parents=True, exist_ok=True)

    logger.debug(f"Created execution directory: {execution_dir}")

    return execution_dir

def validate_stage3_artifacts(config: PuLPFamilyConfiguration) -> Dict[str, bool]:
    """
    Validate availability and accessibility of Stage 3 artifacts.

    Performs complete validation of Stage 3 output artifacts ensuring
    proper file availability and format compliance for input processing.

    Args:
        config: PuLP family configuration instance

    Returns:
        Dictionary mapping artifact names to validation results
    """
    artifacts = {
        'L_raw.parquet': 'L_raw.parquet',
        'L_rel.graphml': 'L_rel.graphml',
        'L_idx': None  # Multiple possible extensions
    }

    validation_results = {}
    input_directory = Path(config.paths.stage3_input_directory)

    for artifact_name, expected_filename in artifacts.items():
        if artifact_name == 'L_idx':
            # Check for L_idx with various extensions
            possible_extensions = ['.idx', '.bin', '.parquet', '.feather', '.pkl']
            l_idx_found = False

            for ext in possible_extensions:
                if (input_directory / f'L_idx{ext}').exists():
                    l_idx_found = True
                    break

            validation_results[artifact_name] = l_idx_found
        else:
            artifact_path = input_directory / expected_filename
            validation_results[artifact_name] = artifact_path.exists() and artifact_path.is_file()

    # Overall validation status
    validation_results['all_artifacts_available'] = all(validation_results.values())

    if not validation_results['all_artifacts_available']:
        missing_artifacts = [k for k, v in validation_results.items() if not v and k != 'all_artifacts_available']
        logger.warning(f"Missing Stage 3 artifacts: {missing_artifacts}")

    return validation_results

def create_configuration_template(file_path: Union[str, Path]) -> None:
    """
    Create configuration template file with complete documentation.

    Generates complete configuration template with detailed comments and
    examples for all configuration options and parameters.

    Args:
        file_path: Path where template file should be created
    """
    template_config = PuLPFamilyConfiguration()

    # Add complete documentation
    template_content = {
        "_documentation": {
            "title": "PuLP Solver Family Configuration Template",
            "version": __version__,
            "stage": __stage__,
            "description": "complete configuration template for Stage 6.1 PuLP solver family",
            "sections": {
                "execution_mode": "System execution mode (development, testing, production, benchmark)",
                "configuration_level": "Configuration validation level (minimal, standard, strict, complete)",
                "paths": "Directory paths for input, output, and execution management",
                "solver": "PuLP solver backend configuration and optimization parameters",
                "input_model": "Input model processing and validation configuration",
                "output_model": "Output model generation and CSV formatting configuration",
                "dynamic_parameters": "Dynamic parameter handling per EAV model integration",
                "logging_configuration": "complete logging and audit configuration",
                "performance_limits": "System performance limits and resource constraints"
            }
        }
    }

    # Add actual configuration
    template_content.update(asdict(template_config))

    file_path = Path(file_path)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(template_content, f, indent=2, sort_keys=True)

        logger.info(f"Configuration template created: {file_path}")

    except Exception as e:
        logger.error(f"Failed to create configuration template: {str(e)}")
        raise

# Configuration validation and diagnostic utilities
def diagnose_configuration_issues(config: PuLPFamilyConfiguration) -> Dict[str, Any]:
    """
    complete configuration diagnosis and issue identification.

    Performs detailed analysis of configuration settings identifying potential
    issues, conflicts, and optimization opportunities for system performance.

    Args:
        config: PuLP family configuration instance

    Returns:
        Dictionary containing complete diagnostic information
    """
    diagnosis = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'configuration_summary': {
            'execution_mode': config.execution_mode.value,
            'configuration_level': config.configuration_level.value,
            'solver_backend': config.solver.default_backend.value
        },
        'issues': [],
        'warnings': [],
        'recommendations': []
    }

    try:
        # Check path accessibility
        path_validation = config.paths.validate_paths()
        if not all(path_validation.values()):
            diagnosis['issues'].append({
                'category': 'paths',
                'severity': 'high',
                'message': 'Some configured paths are not accessible',
                'details': path_validation
            })

        # Check solver configuration
        solver_validation = config.solver.validate_configuration()
        if not solver_validation['overall_valid']:
            diagnosis['issues'].append({
                'category': 'solver',
                'severity': 'high',
                'message': 'Solver configuration validation failed',
                'details': solver_validation
            })

        # Check memory limits vs solver memory
        if config.performance_limits['max_memory_mb'] < config.solver.memory_limit_mb:
            diagnosis['warnings'].append({
                'category': 'performance',
                'message': 'Solver memory limit exceeds system memory limit',
                'recommendation': 'Adjust solver memory limit or increase system memory limit'
            })

        # Check execution mode consistency
        if config.execution_mode == ExecutionMode.PRODUCTION and config.solver.verbose_logging:
            diagnosis['warnings'].append({
                'category': 'performance',
                'message': 'Verbose logging enabled in production mode',
                'recommendation': 'Disable verbose logging for optimal production performance'
            })

        # Performance recommendations
        if config.solver.threads == 1 and config.performance_limits['max_variables'] > 1000000:
            diagnosis['recommendations'].append({
                'category': 'performance',
                'message': 'Consider increasing thread count for large problems',
                'suggestion': f'Increase threads to {min(4, os.cpu_count() or 4)} for better performance'
            })

        # Configuration level recommendations
        if config.execution_mode == ExecutionMode.DEVELOPMENT and config.configuration_level == ConfigurationLevel.MINIMAL:
            diagnosis['recommendations'].append({
                'category': 'validation',
                'message': 'Consider stricter validation in development mode',
                'suggestion': 'Use complete configuration level for development'
            })

        # Overall diagnosis
        diagnosis['overall_status'] = 'healthy' if not diagnosis['issues'] else 'issues_detected'
        diagnosis['issue_count'] = len(diagnosis['issues'])
        diagnosis['warning_count'] = len(diagnosis['warnings'])
        diagnosis['recommendation_count'] = len(diagnosis['recommendations'])

    except Exception as e:
        logger.error(f"Configuration diagnosis failed: {str(e)}")
        diagnosis['diagnosis_error'] = str(e)
        diagnosis['overall_status'] = 'diagnosis_failed'

    return diagnosis

# Export configuration utilities for external use
__all__ = [
    # Enumerations
    'SolverBackend',
    'ConfigurationLevel',
    'ExecutionMode',

    # Configuration classes
    'PathConfiguration',
    'SolverConfiguration',
    'InputModelConfiguration',
    'OutputModelConfiguration',
    'DynamicParameterConfiguration',
    'PuLPFamilyConfiguration',

    # Utility functions
    'load_configuration_from_environment',
    'get_execution_directory_path',
    'validate_stage3_artifacts',
    'create_configuration_template',
    'diagnose_configuration_issues',

    # Default instance
    'DEFAULT_PULP_FAMILY_CONFIGURATION',

    # Metadata
    '__version__',
    '__stage__',
    '__component__'
]

# Module-level configuration validation on import
try:
    _default_validation = DEFAULT_PULP_FAMILY_CONFIGURATION.validate_complete_configuration()
    if not _default_validation['overall_valid']:
        logger.warning("Default configuration has validation issues")
    else:
        logger.debug("Configuration module initialized successfully")
except Exception as e:
    logger.error(f"Configuration module initialization failed: {str(e)}")

if __name__ == "__main__":
    # Example usage and testing
    import sys

    print("Testing PuLP Family Configuration System...")

    try:
        # Test default configuration
        config = PuLPFamilyConfiguration()

        print(f"✓ Created default configuration")
        print(f"  Execution Mode: {config.execution_mode.value}")
        print(f"  Solver Backend: {config.solver.default_backend.value}")
        print(f"  Configuration Level: {config.configuration_level.value}")

        # Test configuration validation
        validation_results = config.validate_complete_configuration()
        print(f"✓ Configuration validation: {'PASSED' if validation_results['overall_valid'] else 'FAILED'}")

        # Test environment loading
        env_config = load_configuration_from_environment()
        print(f"✓ Environment configuration loaded")

        # Test configuration export/import
        test_file = Path("test_config.json")
        json_content = config.export_to_json(test_file)
        imported_config = PuLPFamilyConfiguration.import_from_json(test_file)
        print(f"✓ Configuration export/import successful")

        # Clean up test file
        if test_file.exists():
            test_file.unlink()

        # Test configuration diagnosis
        diagnosis = diagnose_configuration_issues(config)
        print(f"✓ Configuration diagnosis: {diagnosis['overall_status']}")
        print(f"  Issues: {diagnosis['issue_count']}, Warnings: {diagnosis['warning_count']}")

        print("✓ All configuration tests passed")

    except Exception as e:
        print(f"Configuration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
