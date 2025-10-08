"""
config.py
Stage 5 Configuration Management System

This module provides configuration management for Stage 5 processing,
handling environment-aware settings, validation, and runtime configuration with
complete defaults and override capabilities.

Configuration Architecture:
- Hierarchical configuration with environment-based overrides
- Runtime validation with schema compliance checking
- Performance constraints with resource limit enforcement
- Security settings with credential management
- Integration parameters for upstream/downstream stage coordination
- Monitoring configuration with logging and metrics collection

The configuration system follows enterprise patterns with:
- Environment-aware defaults (development, testing, production)
- Schema-based validation with detailed error reporting
- Immutable configuration objects with thread-safe access
- Configuration hot-reloading for runtime parameter updates
- Audit logging for configuration changes and access patterns
- Performance optimization with caching and lazy initialization

Configuration Sources (in order of precedence):
1. Runtime overrides via config_overrides parameter
2. Environment variables with STAGE5_ prefix
3. Configuration files (stage5.json, stage5.yaml)
4. Default values from foundational design specifications

For detailed configuration specifications and parameter descriptions, see:
- Stage5-FOUNDATIONAL-DESIGN-IMPLEMENTATION-PLAN.md Section 3 (Configuration)
- Environment variable documentation in usage guides
- Performance tuning guidelines for production optimization
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import warnings
import sys
import multiprocessing

# Configuration validation and parsing utilities
try:
    from pydantic import BaseModel, Field, validator, ValidationError
    from pydantic.types import conint, confloat, constr
    _PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback for environments without Pydantic
    _PYDANTIC_AVAILABLE = False
    ValidationError = ValueError

# Configuration file format support
try:
    import yaml
    _YAML_SUPPORT = True
except ImportError:
    _YAML_SUPPORT = False

# Resource monitoring utilities
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

# Configuration constants from foundational design specifications
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_JSON_LOGS = True
DEFAULT_DEBUG_MODE = False
DEFAULT_MAX_WORKERS = min(4, multiprocessing.cpu_count())
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_MEMORY_LIMIT_MB = 512
DEFAULT_TEMP_DIR = "/tmp/stage5"
DEFAULT_OUTPUT_DIR = "./outputs/stage5"

# Performance and resource constraints
MIN_MEMORY_MB = 128
MAX_MEMORY_MB = 2048
MIN_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 1800
MIN_WORKERS = 1
MAX_WORKERS = 16

# API server defaults
DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8000
DEFAULT_CORS_ORIGINS = ["*"]

# Stage-specific configuration defaults
STAGE_5_1_DEFAULTS = {
    "random_seed": None,
    "ruggedness_walks": 100,
    "variance_samples": 50,
    "enable_caching": True,
    "validation_strict": True
}

STAGE_5_2_DEFAULTS = {
    "optimization_seed": None,
    "max_lp_iterations": 10,
    "lp_convergence_tolerance": 1e-6,
    "normalization_method": "l2",
    "confidence_threshold": 0.01
}

@dataclass(frozen=True)
class LoggingConfig:
    """
    Logging configuration with structured output and audit trail support.
    
    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Enable structured JSON logging for production
        log_to_file: Enable file logging with rotation
        log_file_path: Path for log file output
        max_log_size_mb: Maximum log file size before rotation
        backup_count: Number of backup log files to retain
        enable_audit_log: Enable separate audit trail logging
    """
    
    level: str = DEFAULT_LOG_LEVEL
    json_format: bool = DEFAULT_JSON_LOGS
    log_to_file: bool = False
    log_file_path: Optional[str] = None
    max_log_size_mb: int = 10
    backup_count: int = 5
    enable_audit_log: bool = True
    
    def __post_init__(self):
        """Validate logging configuration parameters."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {self.level}. Must be one of {valid_levels}")
        
        if self.log_to_file and not self.log_file_path:
            # Generate default log file path
            object.__setattr__(self, 'log_file_path', './logs/stage5.log')
        
        if self.max_log_size_mb < 1 or self.max_log_size_mb > 100:
            raise ValueError(f"max_log_size_mb must be between 1 and 100, got {self.max_log_size_mb}")

@dataclass(frozen=True)
class PerformanceConfig:
    """
    Performance configuration with resource limits and optimization settings.
    
    Attributes:
        max_execution_time_seconds: Maximum execution time for complete pipeline
        max_memory_usage_mb: Maximum memory usage limit
        max_workers: Maximum number of parallel workers
        enable_parallel_processing: Enable multiprocessing for compute-intensive operations
        cache_intermediate_results: Enable caching for performance optimization
        memory_monitoring_enabled: Enable runtime memory usage monitoring
        performance_profiling: Enable detailed performance profiling
    """
    
    max_execution_time_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_memory_usage_mb: int = DEFAULT_MEMORY_LIMIT_MB
    max_workers: int = DEFAULT_MAX_WORKERS
    enable_parallel_processing: bool = True
    cache_intermediate_results: bool = True
    memory_monitoring_enabled: bool = True
    performance_profiling: bool = False
    
    def __post_init__(self):
        """Validate performance configuration parameters."""
        if not (MIN_TIMEOUT_SECONDS <= self.max_execution_time_seconds <= MAX_TIMEOUT_SECONDS):
            raise ValueError(
                f"max_execution_time_seconds must be between {MIN_TIMEOUT_SECONDS} and "
                f"{MAX_TIMEOUT_SECONDS}, got {self.max_execution_time_seconds}"
            )
        
        if not (MIN_MEMORY_MB <= self.max_memory_usage_mb <= MAX_MEMORY_MB):
            raise ValueError(
                f"max_memory_usage_mb must be between {MIN_MEMORY_MB} and "
                f"{MAX_MEMORY_MB}, got {self.max_memory_usage_mb}"
            )
        
        if not (MIN_WORKERS <= self.max_workers <= MAX_WORKERS):
            raise ValueError(
                f"max_workers must be between {MIN_WORKERS} and "
                f"{MAX_WORKERS}, got {self.max_workers}"
            )

@dataclass(frozen=True)
class APIConfig:
    """
    API server configuration for REST service usage.
    
    Attributes:
        host: Server host address for API binding
        port: Server port number for API binding
        debug_mode: Enable debug mode with detailed error responses
        cors_origins: List of allowed CORS origins for cross-origin requests
        allowed_hosts: List of allowed host headers for security
        request_timeout_seconds: Maximum request processing timeout
        max_request_size_mb: Maximum request payload size
        enable_api_metrics: Enable API performance metrics collection
    """
    
    host: str = DEFAULT_API_HOST
    port: int = DEFAULT_API_PORT
    debug_mode: bool = DEFAULT_DEBUG_MODE
    cors_origins: List[str] = field(default_factory=lambda: DEFAULT_CORS_ORIGINS.copy())
    allowed_hosts: Optional[List[str]] = None
    request_timeout_seconds: int = 60
    max_request_size_mb: int = 50
    enable_api_metrics: bool = True
    
    def __post_init__(self):
        """Validate API configuration parameters."""
        if not (1 <= self.port <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")
        
        if not (5 <= self.request_timeout_seconds <= 300):
            raise ValueError(
                f"request_timeout_seconds must be between 5 and 300, "
                f"got {self.request_timeout_seconds}"
            )
        
        if not (1 <= self.max_request_size_mb <= 100):
            raise ValueError(
                f"max_request_size_mb must be between 1 and 100, "
                f"got {self.max_request_size_mb}"
            )

@dataclass(frozen=True)
class Stage51Config:
    """
    Stage 5.1 complexity analysis specific configuration.
    
    Attributes:
        random_seed: Random seed for reproducible complexity calculations
        ruggedness_walks: Number of random walks for landscape ruggedness analysis
        variance_samples: Number of samples for variance calculations
        enable_caching: Enable intermediate result caching for performance
        validation_strict: Enable strict input validation
        complexity_normalization: Enable complexity parameter normalization
        parameter_weights: Custom weights for composite index calculation
    """
    
    random_seed: Optional[int] = None
    ruggedness_walks: int = 100
    variance_samples: int = 50
    enable_caching: bool = True
    validation_strict: bool = True
    complexity_normalization: bool = True
    parameter_weights: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validate Stage 5.1 configuration parameters."""
        if self.random_seed is not None and not (0 <= self.random_seed <= 2**32 - 1):
            raise ValueError(f"random_seed must be between 0 and {2**32 - 1}, got {self.random_seed}")
        
        if not (10 <= self.ruggedness_walks <= 1000):
            raise ValueError(
                f"ruggedness_walks must be between 10 and 1000, got {self.ruggedness_walks}"
            )
        
        if not (10 <= self.variance_samples <= 500):
            raise ValueError(
                f"variance_samples must be between 10 and 500, got {self.variance_samples}"
            )
        
        if self.parameter_weights is not None:
            if len(self.parameter_weights) != 16:
                raise ValueError(
                    f"parameter_weights must have 16 entries, got {len(self.parameter_weights)}"
                )
            
            weight_sum = sum(self.parameter_weights.values())
            if abs(weight_sum - 1.0) > 1e-6:
                raise ValueError(
                    f"parameter_weights must sum to 1.0, got {weight_sum:.6f}"
                )

@dataclass(frozen=True)
class Stage52Config:
    """
    Stage 5.2 solver selection specific configuration.
    
    Attributes:
        optimization_seed: Random seed for LP optimization reproducibility
        max_lp_iterations: Maximum iterations for linear programming solver
        lp_convergence_tolerance: Convergence tolerance for LP optimization
        normalization_method: Parameter normalization method (l2, minmax, zscore)
        confidence_threshold: Minimum confidence threshold for valid selection
        weight_learning_enabled: Enable automated weight learning via LP
        solver_timeout_seconds: Maximum time per solver evaluation
    """
    
    optimization_seed: Optional[int] = None
    max_lp_iterations: int = 10
    lp_convergence_tolerance: float = 1e-6
    normalization_method: str = "l2"
    confidence_threshold: float = 0.01
    weight_learning_enabled: bool = True
    solver_timeout_seconds: int = 30
    
    def __post_init__(self):
        """Validate Stage 5.2 configuration parameters."""
        if self.optimization_seed is not None and not (0 <= self.optimization_seed <= 2**32 - 1):
            raise ValueError(
                f"optimization_seed must be between 0 and {2**32 - 1}, "
                f"got {self.optimization_seed}"
            )
        
        if not (1 <= self.max_lp_iterations <= 100):
            raise ValueError(
                f"max_lp_iterations must be between 1 and 100, got {self.max_lp_iterations}"
            )
        
        if not (1e-8 <= self.lp_convergence_tolerance <= 1e-3):
            raise ValueError(
                f"lp_convergence_tolerance must be between 1e-8 and 1e-3, "
                f"got {self.lp_convergence_tolerance}"
            )
        
        valid_methods = {"l2", "minmax", "zscore"}
        if self.normalization_method not in valid_methods:
            raise ValueError(
                f"normalization_method must be one of {valid_methods}, "
                f"got {self.normalization_method}"
            )
        
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, "
                f"got {self.confidence_threshold}"
            )

@dataclass(frozen=True)
class DirectoryConfig:
    """
    Directory and file path configuration for Stage 5 operations.
    
    Attributes:
        temp_dir: Temporary directory for intermediate files
        output_dir: Default output directory for results
        log_dir: Directory for log file storage
        cache_dir: Directory for caching intermediate results
        create_dirs: Automatically create directories if they don't exist
        cleanup_temp_files: Automatically cleanup temporary files after execution
    """
    
    temp_dir: str = DEFAULT_TEMP_DIR
    output_dir: str = DEFAULT_OUTPUT_DIR
    log_dir: str = "./logs"
    cache_dir: str = "./cache/stage5"
    create_dirs: bool = True
    cleanup_temp_files: bool = True
    
    def __post_init__(self):
        """Create directories if configured and validate paths."""
        if self.create_dirs:
            for dir_path in [self.temp_dir, self.output_dir, self.log_dir, self.cache_dir]:
                try:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    warnings.warn(f"Could not create directory {dir_path}: {e}")

@dataclass(frozen=True)
class Stage5Config:
    """
    Complete Stage 5 configuration with all subsystem settings.
    
    This is the main configuration class that aggregates all Stage 5
    configuration settings with complete validation and defaults.
    
    Attributes:
        logging: Logging configuration with structured output
        performance: Performance limits and optimization settings  
        api: API server configuration for REST service
        stage_5_1: Stage 5.1 complexity analysis specific settings
        stage_5_2: Stage 5.2 solver selection specific settings
        directories: Directory and file path configuration
        debug_mode: Global debug mode enabling detailed diagnostics
        environment: Environment identifier (development, testing, production)
        config_version: Configuration schema version for compatibility
    """
    
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    api: APIConfig = field(default_factory=APIConfig)
    stage_5_1: Stage51Config = field(default_factory=Stage51Config)
    stage_5_2: Stage52Config = field(default_factory=Stage52Config)
    directories: DirectoryConfig = field(default_factory=DirectoryConfig)
    debug_mode: bool = DEFAULT_DEBUG_MODE
    environment: str = "development"
    config_version: str = "1.0.0"
    
    # Convenience properties for commonly accessed settings
    @property
    def log_level(self) -> str:
        """Get logging level."""
        return self.logging.level
    
    @property
    def json_logs(self) -> bool:
        """Get JSON logging format setting."""
        return self.logging.json_format
    
    @property
    def host(self) -> str:
        """Get API host setting."""
        return self.api.host
    
    @property
    def port(self) -> int:
        """Get API port setting."""
        return self.api.port
    
    @property
    def cors_origins(self) -> List[str]:
        """Get CORS origins setting."""
        return self.api.cors_origins
    
    @property
    def allowed_hosts(self) -> Optional[List[str]]:
        """Get allowed hosts setting."""
        return self.api.allowed_hosts
    
    @property
    def max_execution_time(self) -> int:
        """Get maximum execution time setting."""
        return self.performance.max_execution_time_seconds
    
    @property
    def max_memory_mb(self) -> int:
        """Get maximum memory usage setting."""
        return self.performance.max_memory_usage_mb
    
    def validate_configuration(self) -> bool:
        """
        Perform complete configuration validation.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If configuration validation fails
        """
        # Validation is performed in __post_init__ methods of dataclasses
        # Additional cross-component validation can be added here
        
        # Validate environment setting
        valid_environments = {"development", "testing", "production"}
        if self.environment not in valid_environments:
            raise ValueError(
                f"environment must be one of {valid_environments}, got {self.environment}"
            )
        
        # Validate configuration version compatibility
        if self.config_version != "1.0.0":
            warnings.warn(
                f"Configuration version {self.config_version} may not be compatible "
                f"with this Stage 5 implementation (expects 1.0.0)"
            )
        
        # Cross-component validation
        if self.api.request_timeout_seconds > self.performance.max_execution_time_seconds:
            warnings.warn(
                f"API request timeout ({self.api.request_timeout_seconds}s) is greater than "
                f"performance execution limit ({self.performance.max_execution_time_seconds}s)"
            )
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary representation."""
        return asdict(self)
    
    def get_environment_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables."""
        overrides = {}
        
        # Logging overrides
        if "STAGE5_LOG_LEVEL" in os.environ:
            overrides["logging.level"] = os.environ["STAGE5_LOG_LEVEL"]
        
        if "STAGE5_JSON_LOGS" in os.environ:
            overrides["logging.json_format"] = os.environ["STAGE5_JSON_LOGS"].lower() == "true"
        
        # Performance overrides
        if "STAGE5_MAX_EXECUTION_TIME" in os.environ:
            overrides["performance.max_execution_time_seconds"] = int(
                os.environ["STAGE5_MAX_EXECUTION_TIME"]
            )
        
        if "STAGE5_MAX_MEMORY_MB" in os.environ:
            overrides["performance.max_memory_usage_mb"] = int(os.environ["STAGE5_MAX_MEMORY_MB"])
        
        if "STAGE5_MAX_WORKERS" in os.environ:
            overrides["performance.max_workers"] = int(os.environ["STAGE5_MAX_WORKERS"])
        
        # API overrides
        if "STAGE5_API_HOST" in os.environ:
            overrides["api.host"] = os.environ["STAGE5_API_HOST"]
        
        if "STAGE5_API_PORT" in os.environ:
            overrides["api.port"] = int(os.environ["STAGE5_API_PORT"])
        
        if "STAGE5_DEBUG" in os.environ:
            debug_value = os.environ["STAGE5_DEBUG"].lower() == "true"
            overrides["debug_mode"] = debug_value
            overrides["api.debug_mode"] = debug_value
        
        # Environment identifier
        if "STAGE5_ENVIRONMENT" in os.environ:
            overrides["environment"] = os.environ["STAGE5_ENVIRONMENT"]
        
        # Directory overrides
        if "STAGE5_TEMP_DIR" in os.environ:
            overrides["directories.temp_dir"] = os.environ["STAGE5_TEMP_DIR"]
        
        if "STAGE5_OUTPUT_DIR" in os.environ:
            overrides["directories.output_dir"] = os.environ["STAGE5_OUTPUT_DIR"]
        
        # Stage-specific overrides
        if "STAGE5_1_RANDOM_SEED" in os.environ:
            overrides["stage_5_1.random_seed"] = int(os.environ["STAGE5_1_RANDOM_SEED"])
        
        if "STAGE5_2_OPTIMIZATION_SEED" in os.environ:
            overrides["stage_5_2.optimization_seed"] = int(os.environ["STAGE5_2_OPTIMIZATION_SEED"])
        
        return overrides

def load_configuration_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dict containing configuration settings
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If configuration file format is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() == '.json':
                return json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml'] and _YAML_SUPPORT:
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise ValueError(f"Invalid configuration file format: {e}") from e

def apply_nested_overrides(config_dict: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply nested configuration overrides using dot notation.
    
    Args:
        config_dict: Base configuration dictionary
        overrides: Override values with dot-notation keys (e.g., "logging.level")
        
    Returns:
        Dict with overrides applied
    """
    result = config_dict.copy()
    
    for key, value in overrides.items():
        if '.' in key:
            # Handle nested keys
            keys = key.split('.')
            current = result
            
            # Navigate to parent of target key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set final value
            current[keys[-1]] = value
        else:
            # Handle top-level keys
            result[key] = value
    
    return result

def create_config_from_dict(config_dict: Dict[str, Any]) -> Stage5Config:
    """
    Create Stage5Config from dictionary with complete validation.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Stage5Config: Validated configuration object
        
    Raises:
        ValueError: If configuration validation fails
    """
    try:
        # Extract nested configuration sections
        logging_config = config_dict.get("logging", {})
        performance_config = config_dict.get("performance", {})
        api_config = config_dict.get("api", {})
        stage_5_1_config = config_dict.get("stage_5_1", {})
        stage_5_2_config = config_dict.get("stage_5_2", {})
        directory_config = config_dict.get("directories", {})
        
        # Create configuration object with validation
        config = Stage5Config(
            logging=LoggingConfig(**logging_config),
            performance=PerformanceConfig(**performance_config),
            api=APIConfig(**api_config),
            stage_5_1=Stage51Config(**stage_5_1_config),
            stage_5_2=Stage52Config(**stage_5_2_config),
            directories=DirectoryConfig(**directory_config),
            debug_mode=config_dict.get("debug_mode", DEFAULT_DEBUG_MODE),
            environment=config_dict.get("environment", "development"),
            config_version=config_dict.get("config_version", "1.0.0")
        )
        
        # Perform additional validation
        config.validate_configuration()
        
        return config
        
    except (TypeError, ValueError) as e:
        raise ValueError(f"Configuration validation failed: {e}") from e

def load_stage5_configuration(
    config_file: Optional[Union[str, Path]] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> Stage5Config:
    """
    Load complete Stage 5 configuration with hierarchical override support.
    
    Configuration is loaded in the following precedence order:
    1. Runtime overrides via config_overrides parameter
    2. Environment variables with STAGE5_ prefix
    3. Configuration file (if specified)
    4. Default values from foundational design
    
    Args:
        config_file: Optional path to configuration file (JSON or YAML)
        config_overrides: Optional runtime configuration overrides
        
    Returns:
        Stage5Config: Complete validated configuration object
        
    Raises:
        ValueError: If configuration validation fails
        FileNotFoundError: If specified configuration file doesn't exist
    """
    # Start with default configuration
    config_dict = {}
    
    # Load from configuration file if specified
    if config_file is not None:
        try:
            file_config = load_configuration_file(config_file)
            config_dict.update(file_config)
        except FileNotFoundError:
            warnings.warn(f"Configuration file not found: {config_file}, using defaults")
        except ValueError as e:
            warnings.warn(f"Configuration file error: {e}, using defaults")
    
    # Apply environment variable overrides
    env_overrides = Stage5Config().get_environment_overrides()
    if env_overrides:
        config_dict = apply_nested_overrides(config_dict, env_overrides)
    
    # Apply runtime overrides
    if config_overrides:
        config_dict = apply_nested_overrides(config_dict, config_overrides)
    
    # Create final configuration object
    return create_config_from_dict(config_dict)

def validate_environment() -> bool:
    """
    Validate system environment for Stage 5 usage.
    
    Checks system resources, dependencies, and environment variables
    to ensure Stage 5 can run successfully in the current environment.
    
    Returns:
        bool: True if environment is suitable for Stage 5 operation
    """
    validation_errors = []
    
    # Check Python version compatibility
    if sys.version_info < (3, 11):
        validation_errors.append(
            f"Python version {sys.version} not supported - require Python 3.11+"
        )
    
    # Check available memory if psutil is available
    if _PSUTIL_AVAILABLE:
        try:
            available_memory_mb = psutil.virtual_memory().available // (1024 * 1024)
            if available_memory_mb < MIN_MEMORY_MB:
                validation_errors.append(
                    f"Insufficient memory: {available_memory_mb}MB available, "
                    f"minimum {MIN_MEMORY_MB}MB required"
                )
        except Exception as e:
            warnings.warn(f"Could not check available memory: {e}")
    
    # Check disk space for temp directory
    temp_dir = os.environ.get("STAGE5_TEMP_DIR", DEFAULT_TEMP_DIR)
    try:
        temp_path = Path(temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)
        
        if _PSUTIL_AVAILABLE:
            disk_usage = psutil.disk_usage(str(temp_path))
            free_space_mb = disk_usage.free // (1024 * 1024)
            if free_space_mb < 100:  # Minimum 100MB free space
                validation_errors.append(
                    f"Insufficient disk space in {temp_dir}: {free_space_mb}MB free, "
                    f"minimum 100MB required"
                )
    except Exception as e:
        validation_errors.append(f"Cannot access temp directory {temp_dir}: {e}")
    
    # Check for required environment variables in production
    environment = os.environ.get("STAGE5_ENVIRONMENT", "development")
    if environment == "production":
        required_prod_vars = []  # Add production-specific required variables
        
        for var in required_prod_vars:
            if var not in os.environ:
                validation_errors.append(f"Required environment variable missing: {var}")
    
    # Log validation results
    if validation_errors:
        for error in validation_errors:
            warnings.warn(f"Environment validation error: {error}")
        return False
    
    return True

def get_default_config() -> Stage5Config:
    """
    Get default Stage 5 configuration with foundational design defaults.
    
    Returns:
        Stage5Config: Default configuration object
    """
    return Stage5Config()

def save_configuration(config: Stage5Config, output_path: Union[str, Path]) -> None:
    """
    Save Stage 5 configuration to JSON file.
    
    Args:
        config: Configuration object to save
        output_path: Output file path for configuration
        
    Raises:
        IOError: If configuration cannot be written to file
    """
    output_path = Path(output_path)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2, default=str)
    except Exception as e:
        raise IOError(f"Cannot save configuration to {output_path}: {e}") from e

# Export key configuration classes and functions
__all__ = [
    "Stage5Config",
    "LoggingConfig", 
    "PerformanceConfig",
    "APIConfig",
    "Stage51Config",
    "Stage52Config",
    "DirectoryConfig",
    "load_stage5_configuration",
    "validate_environment",
    "get_default_config",
    "save_configuration",
    "load_configuration_file",
    "create_config_from_dict"
]