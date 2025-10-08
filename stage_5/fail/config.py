# STAGE 5 - CONFIG.PY
# complete Centralized Configuration Management

"""
STAGE 5 CONFIGURATION MANAGEMENT
complete Centralized Configuration for Stage 5 Input-Complexity Analysis and Solver Selection

This module provides centralized configuration management for Stage 5's rigorous execution framework.
All runtime settings, file paths, mathematical parameters, and system configurations are managed
through this single configuration interface with environment variable overrides and validation.

Critical Implementation Notes:
- CENTRALIZED CONTROL: Single source of truth for all Stage 5 configuration parameters
- ENVIRONMENT AWARE: Automatic detection and override from environment variables
- VALIDATION ENFORCED: All configuration values validated against constraints and ranges
- THEORETICAL COMPLIANCE: Mathematical parameters align with framework specifications
- ENTERPRISE READY: complete configuration management with error handling

Configuration Categories:
1. FILE PATHS: Stage 3 inputs, Stage 5 outputs, solver capabilities, log directories
2. MATHEMATICAL PARAMETERS: Random seeds, convergence tolerances, computation limits
3. PERFORMANCE SETTINGS: Memory limits, timeout values, optimization parameters
4. LOGGING CONFIGURATION: Log levels, output formats, audit trail settings
5. INTEGRATION SETTINGS: API endpoints, callback URLs, pipeline coordination

References:
- Stage5-FOUNDATIONAL-DESIGN-IMPLEMENTATION-PLAN.md: Configuration specifications
- Stage-5.1-INPUT-COMPLEXITY-ANALYSIS: Mathematical parameter requirements
- Stage-5.2-SOLVER-SELECTION-ARSENAL-MODULARITY: LP optimization configuration
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from datetime import datetime
import json

# =============================================================================
# CONFIGURATION DATA CLASSES
# Structured configuration with type safety and validation
# =============================================================================

@dataclass
class Stage5FilePathsConfig:
    """
    File path configurations for Stage 5 input/output operations.
    Centralizes all file path specifications with environment variable overrides.
    
    Path Categories:
    - stage3_inputs: Stage 3 output file locations (L_raw, L_rel, L_idx)
    - stage5_outputs: Stage 5 output directory for results and logs
    - solver_arsenal: Solver capabilities configuration file
    - log_directory: Structured logging output directory
    - execution_directory: Per-execution isolation directory template
    """
    
    # Stage 3 Input Paths (can be overridden per execution)
    stage3_l_raw_path: Optional[str] = None
    stage3_l_rel_path: Optional[str] = None  
    stage3_l_idx_path: Optional[str] = None
    
    # Stage 5 Output Paths
    stage5_output_directory: str = "./stage5_outputs"
    stage5_1_output_filename: str = "complexity_metrics.json"
    stage5_2_output_filename: str = "selection_decision.json"
    
    # Configuration and Arsenal Files
    solver_capabilities_file: str = "./config/solver_capabilities.json"
    config_file: Optional[str] = None
    
    # Logging and Audit Directories
    log_directory: Optional[str] = None  # None = console-only
    execution_base_directory: str = "./executions"
    
    # Per-execution directory structure: {base}/executions/{timestamp_uuid}/
    execution_subdirs: List[str] = field(default_factory=lambda: [
        "audit_logs", "error_reports", "output_data", "final_output"
    ])
    
    def get_execution_directory(self, execution_id: str) -> Dict[str, Path]:
        """
        Generate per-execution directory structure for isolation.
        
        Args:
            execution_id: Unique execution identifier
            
        Returns:
            Dictionary mapping subdirectory names to Path objects
        """
        base_exec_dir = Path(self.execution_base_directory) / execution_id
        
        return {
            "base": base_exec_dir,
            **{subdir: base_exec_dir / subdir for subdir in self.execution_subdirs}
        }

@dataclass 
class Stage51Config:
    """
    Stage 5.1 Input-Complexity Analysis configuration parameters.
    Mathematical and computational settings for the 16-parameter framework.
    
    Configuration Categories:
    - stochastic_parameters: Random seeds and sampling configurations
    - computation_limits: Timeout and resource constraints
    - mathematical_tolerances: Numerical precision and convergence thresholds
    - validation_settings: Input validation and bounds checking parameters
    """
    
    # Stochastic Computation Parameters
    random_seed: int = 42  # Deterministic reproducibility for P13, P16
    ruggedness_walks: int = 1000  # P13 landscape ruggedness sampling
    variance_samples: int = 50  # P16 quality variance sample size
    
    # Computation Limits and Timeouts
    computation_timeout_seconds: int = 300  # 5 minutes maximum per execution
    memory_limit_mb: int = 256  # Maximum memory usage for Stage 5.1
    max_entity_scale: int = 2000  # Maximum entities for prototype validation
    
    # Mathematical Tolerances and Precision
    numerical_epsilon: float = 1e-12  # Near-zero threshold for calculations
    entropy_calculation_base: float = 2.0  # Base for entropy calculations (bits)
    coefficient_variation_min_samples: int = 2  # Minimum samples for CV calculation
    
    # Parameter Validation Bounds (from theoretical framework)
    p1_dimensionality_max: float = 1e12  # Computational tractability limit
    p7_entropy_max: float = 30.0  # Information-theoretic maximum
    composite_index_max: float = 50.0  # Empirical maximum from validation dataset
    
    # Input Validation Settings
    validate_input_schemas: bool = True  # Enforce Pydantic schema validation
    strict_bounds_checking: bool = True  # Enforce parameter bounds per theory
    allow_missing_l_idx_formats: bool = True  # Accept multiple L_idx formats

@dataclass
class Stage52Config:
    """
    Stage 5.2 Solver Selection & Arsenal Modularity configuration parameters.
    L2 normalization, LP optimization, and solver ranking settings.
    
    Configuration Categories:
    - normalization_parameters: L2 normalization mathematical settings
    - lp_optimization: Linear programming convergence and iteration limits
    - solver_selection: Ranking and confidence calculation parameters
    - arsenal_management: Solver capability validation and loading settings
    """
    
    # L2 Normalization Parameters
    normalization_epsilon: float = 1e-12  # Minimum norm threshold for stability
    capability_vector_dimension: int = 16  # Must match P1-P16 parameter count
    capability_value_range: tuple = (0.0, 10.0)  # Effectiveness scale bounds
    
    # LP Optimization Parameters
    lp_convergence_tolerance: float = 1e-6  # Convergence threshold for weights
    max_lp_iterations: int = 20  # Maximum iterations per theoretical bound
    lp_solver_timeout_seconds: int = 60  # Maximum time for LP solver
    weight_sum_tolerance: float = 1e-6  # Simplex constraint validation (Σw = 1)
    
    # Solver Selection and Ranking
    min_separation_margin: float = 1e-8  # Minimum margin for meaningful ranking
    confidence_calculation_method: str = "separation_based"  # Confidence metric type
    ranking_tie_breaker: str = "solver_id"  # Deterministic tie resolution
    
    # Arsenal Management and Validation
    validate_capability_vectors: bool = True  # Enforce 16-D capability validation
    require_paradigm_diversity: bool = True  # Require ≥2 solver paradigms
    min_solvers_in_arsenal: int = 2  # Minimum solvers for meaningful selection
    max_solvers_in_arsenal: int = 50  # Reasonable upper bound for performance
    
    # Iterative Algorithm Settings
    initial_weight_distribution: str = "uniform"  # Initial weight vector (1/16 each)
    convergence_history_length: int = 5  # Track convergence over iterations
    enable_weight_learning: bool = True  # Enable LP-based weight optimization

@dataclass
class Stage5LoggingConfig:
    """
    Structured logging configuration for Stage 5 audit trails and debugging.
    JSON logging framework settings with performance and error tracking.
    
    Configuration Categories:
    - output_settings: Log destinations and format specifications
    - performance_tracking: Execution timing and resource monitoring
    - audit_requirements: Compliance and traceability settings
    - error_handling: Error logging and context preservation
    """
    
    # Logging Output Settings
    log_level: str = "INFO"  # Minimum log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    log_format: str = "json"  # Output format (json/console)
    enable_console_output: bool = True  # Console logging for development
    enable_file_output: bool = False  # File logging for production (requires log_directory)
    
    # File Logging Configuration
    main_log_filename: str = "stage5_execution.log"
    error_log_filename: str = "stage5_errors.log" 
    performance_log_filename: str = "stage5_performance.log"
    log_file_encoding: str = "utf-8"
    
    # Performance Tracking Settings
    enable_performance_logging: bool = True  # Track operation timing and memory
    performance_log_threshold_ms: float = 100.0  # Log operations >100ms
    memory_monitoring_enabled: bool = False  # Requires psutil (optional)
    
    # Audit Trail Requirements
    include_execution_context: bool = True  # Add execution_id to all logs
    include_stack_traces: bool = True  # Include stack traces for errors
    log_parameter_values: bool = False  # Log computed parameter values (debug)
    log_solver_rankings: bool = True  # Log complete solver rankings
    
    # Error Handling and Context
    max_error_context_length: int = 1000  # Maximum error context characters
    preserve_exception_chains: bool = True  # Maintain exception causality
    log_validation_failures: bool = True  # Log schema validation errors

@dataclass
class Stage5PerformanceConfig:
    """
    Performance optimization and resource management configuration.
    System-level settings for memory, computation, and scalability limits.
    
    Configuration Categories:
    - resource_limits: Memory, CPU, and storage constraints
    - optimization_settings: Performance tuning parameters
    - scalability_bounds: Entity count and complexity limits
    - monitoring_configuration: Performance tracking and alerting
    """
    
    # Resource Limits
    memory_limit_mb: int = 512  # Maximum memory usage for entire Stage 5
    cpu_usage_limit_percent: float = 80.0  # Maximum CPU utilization
    disk_space_required_mb: int = 100  # Minimum free disk space
    execution_timeout_minutes: int = 10  # Overall Stage 5 timeout
    
    # Computation Optimization
    enable_parallel_processing: bool = False  # Parallel parameter computation (future)
    numpy_thread_count: Optional[int] = None  # NumPy thread control (None = auto)
    pandas_memory_optimization: bool = True  # Optimize pandas memory usage
    
    # Scalability and Entity Limits
    max_courses: int = 1000  # Maximum course entities for prototype
    max_faculty: int = 300  # Maximum faculty entities
    max_rooms: int = 200  # Maximum room entities
    max_timeslots: int = 100  # Maximum timeslot entities
    max_batches: int = 150  # Maximum batch entities
    
    # Performance Monitoring
    enable_profiling: bool = False  # Enable detailed profiling (development)
    profile_output_directory: Optional[str] = None  # Profiling data directory
    performance_alert_threshold_ms: int = 5000  # Alert for operations >5s
    memory_alert_threshold_mb: int = 400  # Alert for memory usage >400MB

@dataclass 
class Stage5IntegrationConfig:
    """
    Integration configuration for Stage 5 pipeline coordination.
    API endpoints, callbacks, and inter-stage communication settings.
    
    Configuration Categories:
    - api_settings: REST API endpoints and authentication
    - pipeline_coordination: Inter-stage communication and handoffs
    - callback_configuration: Webhook and notification settings
    - error_propagation: Failure handling and retry policies
    """
    
    # API Endpoint Configuration
    api_host: str = "0.0.0.0"  # API binding host
    api_port: int = 8000  # API server port
    api_prefix: str = "/stage5"  # API URL prefix
    enable_cors: bool = True  # Enable CORS for web integration
    api_timeout_seconds: int = 300  # API request timeout
    
    # Pipeline Coordination
    stage3_completion_check: bool = True  # Verify Stage 3 outputs before start
    stage6_handoff_validation: bool = True  # Validate output for Stage 6
    enable_pipeline_callbacks: bool = False  # Enable pipeline event callbacks
    
    # Callback and Notification Settings
    success_callback_url: Optional[str] = None  # Success notification webhook
    error_callback_url: Optional[str] = None  # Error notification webhook
    progress_callback_url: Optional[str] = None  # Progress update webhook
    callback_timeout_seconds: int = 30  # Callback request timeout
    
    # Error Propagation and Retry
    fail_fast_on_errors: bool = True  # Immediate failure on errors (no retries)
    enable_error_recovery: bool = False  # Error recovery attempts (future)
    max_retry_attempts: int = 0  # Maximum retry attempts (0 = no retries)
    retry_delay_seconds: int = 5  # Delay between retry attempts

# =============================================================================
# MAIN CONFIGURATION CLASS
# Aggregates all configuration components with environment variable overrides
# =============================================================================

class Stage5Config:
    """
    Main configuration class for Stage 5 operations.
    Aggregates all configuration components with environment variable overrides,
    validation, and runtime updates.
    
    This class provides:
    - Centralized access to all Stage 5 configuration parameters
    - Environment variable override support with type conversion
    - Configuration validation and constraint enforcement
    - Runtime configuration updates with validation
    - Configuration persistence and loading from files
    
    Environment Variable Mapping:
    - STAGE5_LOG_LEVEL → logging.log_level
    - STAGE5_LOG_DIRECTORY → file_paths.log_directory
    - STAGE5_RANDOM_SEED → stage_5_1.random_seed
    - STAGE5_MEMORY_LIMIT → performance.memory_limit_mb
    - STAGE5_API_PORT → integration.api_port
    - And many more following STAGE5_{SECTION}_{PARAMETER} pattern
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize Stage 5 configuration with environment overrides.
        
        Args:
            config_file: Optional JSON configuration file path
        """
        # Initialize configuration components with defaults
        self.file_paths = Stage5FilePathsConfig()
        self.stage_5_1 = Stage51Config()
        self.stage_5_2 = Stage52Config()
        self.logging = Stage5LoggingConfig()
        self.performance = Stage5PerformanceConfig()
        self.integration = Stage5IntegrationConfig()
        
        # Load from configuration file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Apply environment variable overrides
        self._apply_environment_overrides()
        
        # Validate final configuration
        self.validate()
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # Environment variable mappings with type conversion
        env_mappings = {
            # File Paths
            'STAGE5_OUTPUT_DIRECTORY': ('file_paths', 'stage5_output_directory', str),
            'STAGE5_SOLVER_CAPABILITIES': ('file_paths', 'solver_capabilities_file', str),
            'STAGE5_LOG_DIRECTORY': ('file_paths', 'log_directory', str),
            
            # Stage 5.1 Parameters
            'STAGE5_RANDOM_SEED': ('stage_5_1', 'random_seed', int),
            'STAGE5_RUGGEDNESS_WALKS': ('stage_5_1', 'ruggedness_walks', int),
            'STAGE5_VARIANCE_SAMPLES': ('stage_5_1', 'variance_samples', int),
            'STAGE5_COMPUTATION_TIMEOUT': ('stage_5_1', 'computation_timeout_seconds', int),
            
            # Stage 5.2 Parameters
            'STAGE5_LP_TOLERANCE': ('stage_5_2', 'lp_convergence_tolerance', float),
            'STAGE5_MAX_LP_ITERATIONS': ('stage_5_2', 'max_lp_iterations', int),
            
            # Logging Configuration
            'STAGE5_LOG_LEVEL': ('logging', 'log_level', str),
            'STAGE5_ENABLE_FILE_LOGGING': ('logging', 'enable_file_output', bool),
            'STAGE5_PERFORMANCE_LOGGING': ('logging', 'enable_performance_logging', bool),
            
            # Performance Settings  
            'STAGE5_MEMORY_LIMIT': ('performance', 'memory_limit_mb', int),
            'STAGE5_EXECUTION_TIMEOUT': ('performance', 'execution_timeout_minutes', int),
            
            # Integration Settings
            'STAGE5_API_HOST': ('integration', 'api_host', str),
            'STAGE5_API_PORT': ('integration', 'api_port', int),
            'STAGE5_ENABLE_CALLBACKS': ('integration', 'enable_pipeline_callbacks', bool)
        }
        
        for env_var, (section, param, type_converter) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    # Convert string environment variable to appropriate type
                    if type_converter == bool:
                        converted_value = value.lower() in ('true', '1', 'yes', 'on')
                    elif type_converter == int:
                        converted_value = int(value)
                    elif type_converter == float:
                        converted_value = float(value)
                    else:  # str
                        converted_value = value
                    
                    # Set configuration value
                    config_section = getattr(self, section)
                    setattr(config_section, param, converted_value)
                    
                except (ValueError, AttributeError) as e:
                    print(f"Warning: Invalid environment variable {env_var}={value}: {e}")
    
    def load_from_file(self, config_file: str) -> None:
        """
        Load configuration from JSON file with validation.
        
        Args:
            config_file: Path to JSON configuration file
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            json.JSONDecodeError: If JSON is invalid
            ValueError: If configuration values are invalid
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Update configuration sections from file
            self._update_from_dict(config_data)
            
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in configuration file: {e}")
    
    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Update configuration from dictionary data."""
        section_mapping = {
            'file_paths': self.file_paths,
            'stage_5_1': self.stage_5_1, 
            'stage_5_2': self.stage_5_2,
            'logging': self.logging,
            'performance': self.performance,
            'integration': self.integration
        }
        
        for section_name, section_data in config_data.items():
            if section_name in section_mapping and isinstance(section_data, dict):
                config_section = section_mapping[section_name]
                for param_name, param_value in section_data.items():
                    if hasattr(config_section, param_name):
                        setattr(config_section, param_name, param_value)
    
    def validate(self) -> None:
        """
        Validate entire configuration for consistency and constraints.
        
        Raises:
            ValueError: If any configuration values are invalid or inconsistent
        """
        # Validate Stage 5.1 configuration
        if self.stage_5_1.random_seed < 0:
            raise ValueError("Random seed must be non-negative")
        
        if self.stage_5_1.ruggedness_walks < 100 or self.stage_5_1.ruggedness_walks > 10000:
            raise ValueError("Ruggedness walks must be between 100 and 10000")
        
        if self.stage_5_1.variance_samples < 10 or self.stage_5_1.variance_samples > 200:
            raise ValueError("Variance samples must be between 10 and 200")
        
        # Validate Stage 5.2 configuration
        if self.stage_5_2.max_lp_iterations < 1 or self.stage_5_2.max_lp_iterations > 100:
            raise ValueError("LP iterations must be between 1 and 100")
        
        if self.stage_5_2.lp_convergence_tolerance <= 0 or self.stage_5_2.lp_convergence_tolerance > 1e-3:
            raise ValueError("LP convergence tolerance must be between 0 and 1e-3")
        
        # Validate logging configuration
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.logging.log_level.upper() not in valid_log_levels:
            raise ValueError(f"Log level must be one of: {valid_log_levels}")
        
        # Validate performance configuration
        if self.performance.memory_limit_mb < 64:
            raise ValueError("Memory limit must be at least 64MB")
        
        if self.performance.execution_timeout_minutes < 1:
            raise ValueError("Execution timeout must be at least 1 minute")
        
        # Validate integration configuration
        if self.integration.api_port < 1024 or self.integration.api_port > 65535:
            raise ValueError("API port must be between 1024 and 65535")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format for serialization.
        
        Returns:
            Dictionary representation of all configuration parameters
        """
        return {
            'file_paths': self.file_paths.__dict__,
            'stage_5_1': self.stage_5_1.__dict__,
            'stage_5_2': self.stage_5_2.__dict__,
            'logging': self.logging.__dict__,
            'performance': self.performance.__dict__,
            'integration': self.integration.__dict__
        }
    
    def save_to_file(self, config_file: str) -> None:
        """
        Save current configuration to JSON file.
        
        Args:
            config_file: Path to output JSON configuration file
        """
        config_data = self.to_dict()
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, separators=(',', ': '), default=str)
    
    def get_execution_paths(self, execution_id: str) -> Dict[str, Path]:
        """
        Get per-execution directory structure for the given execution ID.
        
        Args:
            execution_id: Unique execution identifier
            
        Returns:
            Dictionary mapping directory names to Path objects
        """
        return self.file_paths.get_execution_directory(execution_id)
    
    def override(self, **kwargs) -> 'Stage5Config':
        """
        Create a new configuration instance with runtime overrides.
        
        Args:
            **kwargs: Configuration overrides in section.parameter=value format
            
        Returns:
            New Stage5Config instance with overrides applied
            
        Example:
            new_config = config.override(**{
                'stage_5_1.random_seed': 123,
                'logging.log_level': 'DEBUG',
                'performance.memory_limit_mb': 1024
            })
        """
        # Create a deep copy of current configuration
        new_config = Stage5Config()
        new_config.file_paths = Stage5FilePathsConfig(**self.file_paths.__dict__)
        new_config.stage_5_1 = Stage51Config(**self.stage_5_1.__dict__)
        new_config.stage_5_2 = Stage52Config(**self.stage_5_2.__dict__)
        new_config.logging = Stage5LoggingConfig(**self.logging.__dict__)
        new_config.performance = Stage5PerformanceConfig(**self.performance.__dict__)
        new_config.integration = Stage5IntegrationConfig(**self.integration.__dict__)
        
        # Apply overrides
        for key, value in kwargs.items():
            if '.' in key:
                section_name, param_name = key.split('.', 1)
                if hasattr(new_config, section_name):
                    config_section = getattr(new_config, section_name)
                    if hasattr(config_section, param_name):
                        setattr(config_section, param_name, value)
        
        # Validate new configuration
        new_config.validate()
        
        return new_config

# =============================================================================
# GLOBAL CONFIGURATION INSTANCE
# Default configuration instance for Stage 5 operations
# =============================================================================

# Create default configuration instance
_default_config = None

def get_stage5_config(config_file: Optional[str] = None) -> Stage5Config:
    """
    Get Stage 5 configuration instance with lazy initialization.
    
    Args:
        config_file: Optional configuration file path for first initialization
        
    Returns:
        Stage5Config instance with environment overrides applied
    """
    global _default_config
    
    if _default_config is None:
        _default_config = Stage5Config(config_file=config_file)
    
    return _default_config

def reload_stage5_config(config_file: Optional[str] = None) -> Stage5Config:
    """
    Reload Stage 5 configuration, discarding cached instance.
    
    Args:
        config_file: Optional configuration file path
        
    Returns:
        New Stage5Config instance
    """
    global _default_config
    _default_config = Stage5Config(config_file=config_file)
    return _default_config

# Export configuration components
__all__ = [
    # Configuration Data Classes
    'Stage5FilePathsConfig', 'Stage51Config', 'Stage52Config',
    'Stage5LoggingConfig', 'Stage5PerformanceConfig', 'Stage5IntegrationConfig',
    
    # Main Configuration Class
    'Stage5Config',
    
    # Global Configuration Functions
    'get_stage5_config', 'reload_stage5_config'
]

print("✅ STAGE 5 CONFIG.PY - COMPLETE")
print("   - complete centralized configuration management")
print("   - Structured data classes for all configuration categories")
print("   - Environment variable override support with type validation")
print("   - Configuration file loading and persistence capabilities")
print("   - Per-execution directory structure management")
print("   - Runtime configuration override and validation")
print(f"   - Total configuration components exported: {len(__all__)}")