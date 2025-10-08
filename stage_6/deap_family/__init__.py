# -*- coding: utf-8 -*-
"""
Stage 6.3 DEAP Solver Family - Root Package Initialization

This module serves as the master initialization and orchestration point for the complete
DEAP (Distributed Evolutionary Algorithms in Python) Solver Family within Stage 6 of 
the Advanced Scheduling Engine. It provides a unified interface for evolutionary 
optimization algorithms while maintaining strict adherence to theoretical foundations,
mathematical frameworks, and formal models.

THEORETICAL FOUNDATIONS & MATHEMATICAL COMPLIANCE:
- Stage 6.3 DEAP Foundational Framework: Complete evolutionary algorithm suite
- Algorithm 11.2: Integrated Evolutionary Process with multi-objective optimization
- Definition 2.1: Evolutionary Algorithm Framework EA = (λ, μ, σ, τ, χ, Ψ)
- Definition 2.2: Schedule Genotype Encoding g: course → (faculty, room, timeslot, batch)
- Definition 2.4: Multi-Objective Fitness Model f(g) = (f₁, f₂, f₃, f₄, f₅)
- Theorem 3.2: GA Schema Theorem for scheduling pattern preservation
- Theorem 8.4: NSGA-II Convergence Properties with Pareto front maintenance
- Stage 3 Data Compilation: Bijective mapping and constraint rule integration
- Dynamic Parametric System: EAV parameter integration with real-time adaptation

EVOLUTIONARY ALGORITHM SUITE:
1. Genetic Algorithm (GA): Tournament selection with order crossover and swap mutation
2. Genetic Programming (GP): Tree evolution with bloat control and heuristic optimization
3. Evolution Strategies (ES): CMA adaptation with self-adaptive parameter control
4. Differential Evolution (DE): Adaptive parameter control with global optimization
5. Particle Swarm Optimization (PSO): Inertia weight adaptation with velocity control
6. NSGA-II: Multi-objective Pareto dominance with crowding distance preservation

ARCHITECTURAL DESIGN PRINCIPLES:
- Course-centric genotype representation with bijective phenotype mapping
- Three-layer pipeline: Input Modeling (≤200MB) → Processing (≤250MB) → Output Modeling (≤100MB)
- Single-threaded execution with deterministic resource usage patterns
- Fail-fast validation with complete error propagation and audit trails
- Memory-bounded processing with real-time constraint monitoring and enforcement
- Zero I/O complexity with in-memory data structures and layer-by-layer processing

SYSTEM INTEGRATION ARCHITECTURE:
- Stage 3 Data Compilation integration for constraint rules and eligibility mapping
- Dynamic Parametric System integration for EAV parameter real-time adaptation
- Stage 7 Output Validation preparation with twelve-threshold compliance checking
- FastAPI REST interface for backend integration and usage readiness
- complete audit logging and performance profiling for SIH evaluation

Author: Student Team
Date: October 2025
Version: 1.0.0

CURSOR IDE & JETBRAINS IDE OPTIMIZATION:
- Complete type hints with Pydantic model validation for enhanced IntelliSense support
- Cross-module reference tracking with explicit import paths for dependency analysis
- complete docstring coverage for automatic documentation generation and API discovery
- Professional exception hierarchy with detailed error context for advanced debugging
- Real-time performance monitoring integration for IDE resource tracking and optimization
- Mathematical formula references in comments for theoretical validation and compliance checking
"""

import sys
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union, Type
from typing import get_type_hints, get_origin, get_args
import importlib.metadata
from pathlib import Path
import os
from datetime import datetime
import uuid
import gc
import psutil
import traceback

# Core Python Libraries for Scientific Computing and Data Processing
try:
    import numpy as np
    import pandas as pd
    import scipy
    import networkx as nx
    from pydantic import BaseModel, Field, __version__ as pydantic_version
except ImportError as e:
    critical_error = f"Critical scientific computing dependencies missing: {str(e)}"
    print(f"CRITICAL ERROR: {critical_error}", file=sys.stderr)
    raise ImportError(critical_error) from e

# DEAP Evolutionary Computing Library - Essential for Algorithm Implementation
try:
    import deap
    from deap import __version__ as deap_version
    from deap import base, creator, tools, algorithms
except ImportError as e:
    critical_error = f"DEAP evolutionary computing library not available: {str(e)}"
    print(f"CRITICAL ERROR: {critical_error}", file=sys.stderr)
    raise ImportError(critical_error) from e

# FastAPI and REST Interface Dependencies
try:
    import fastapi
    from fastapi import __version__ as fastapi_version
    import uvicorn
    import aiofiles
except ImportError as e:
    # FastAPI is optional for core functionality but required for usage
    fastapi_available = False
    fastapi_version = "NOT AVAILABLE"
    warnings.warn(f"FastAPI components not available: {str(e)}", ImportWarning)
else:
    fastapi_available = True

# Suppress non-critical warnings during initialization
warnings.filterwarnings('ignore', category=UserWarning, module='deap')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='numpy')

# Package Metadata and Version Information
__package_name__ = "deap_family"
__version__ = "1.0.0"
__build__ = "20251008-production"
__author__ = "Student Team"
__description__ = "Stage 6.3 DEAP Solver Family - Complete Evolutionary Optimization Suite"
__python_requires__ = ">=3.9"

# Dependency Version Requirements and Compatibility Validation
REQUIRED_VERSIONS = {
    'numpy': '>=1.24.0',
    'pandas': '>=2.0.0',
    'scipy': '>=1.11.0',
    'networkx': '>=3.2.0',
    'pydantic': '>=2.0.0',
    'deap': '>=1.4.0'
}

# Stage 6.3 DEAP Family Component Imports with Error Handling
# Maintaining strict import order to prevent circular dependencies
try:
    # Core configuration and pipeline orchestration
    from .deap_family_config import (
        DEAPFamilyConfig,
        SolverID,
        PopulationConfig,
        OperatorConfig,
        FitnessWeights,
        PathConfig,
        MemoryConstraints,
        validate_deap_family_config
    )
    
    from .deap_family_main import (
        DEAPFamilyPipeline,
        PipelineContext,
        MemoryMonitor,
        ExecutionTimer,
        AuditLogger,
        run_deap_family_optimization
    )
    
except ImportError as e:
    error_msg = f"Failed to import core DEAP family components: {str(e)}"
    logging.error(error_msg)
    raise ImportError(f"Core component initialization failed: {error_msg}") from e

try:
    # Input modeling layer components
    from .input_model import (
        build_input_context,
        DEAPInputModelLoader,
        DEAPInputModelValidator,
        DEAPInputMetadataGenerator,
        InputModelContext,
        CourseEligibilityMap,
        ConstraintRulesMap,
        BijectionMappingData
    )
    
except ImportError as e:
    error_msg = f"Failed to import input modeling components: {str(e)}"
    logging.error(error_msg)
    raise ImportError(f"Input modeling initialization failed: {error_msg}") from e

try:
    # Processing layer components - Evolutionary computing infrastructure
    from .processing import (
        ProcessingOrchestrator,
        ProcessingResult,
        run_evolutionary_optimization,
        ProcessingLayerError,
        ProcessingConfigurationError,
        ProcessingMemoryError,
        ProcessingValidationError
    )
    
    # Individual processing components for advanced usage
    from .processing.population import (
        PopulationManager,
        IndividualType,
        PopulationType,
        FitnessType,
        PopulationStatistics
    )
    
    from .processing.operators import (
        OperatorManager,
        CrossoverOperators,
        MutationOperators,
        SelectionOperators
    )
    
    from .processing.evaluator import (
        DEAPMultiObjectiveFitnessEvaluator,
        ObjectiveMetrics,
        EvaluationStatistics
    )
    
    from .processing.engine import (
        EvolutionaryAlgorithmFactory,
        EvolutionaryResult,
        EvolutionaryRunStatistics
    )
    
    from .processing.logging import (
        EvolutionaryLogger,
        GenerationMetrics,
        ConvergenceAnalyzer
    )
    
except ImportError as e:
    error_msg = f"Failed to import processing layer components: {str(e)}"
    logging.error(error_msg)
    raise ImportError(f"Processing layer initialization failed: {error_msg}") from e

try:
    # Output modeling layer components
    from .output_model import (
        generate_complete_output,
        SolutionDecoder,
        ScheduleWriter,
        OutputMetadataGenerator,
        DecodedSchedule,
        ScheduleValidationResult,
        OutputMetadata
    )
    
except ImportError as e:
    error_msg = f"Failed to import output modeling components: {str(e)}"
    logging.error(error_msg)
    raise ImportError(f"Output modeling initialization failed: {error_msg}") from e

# Optional API layer components - Required for usage but not core functionality
try:
    if fastapi_available:
        from .api import (
            create_deap_family_app,
            DEAPOptimizationRequest,
            DEAPOptimizationResponse,
            DEAPHealthResponse,
            DEAPAlgorithmInfo
        )
        api_components_available = True
    else:
        api_components_available = False
except ImportError as e:
    api_components_available = False
    warnings.warn(f"API components not available: {str(e)}", ImportWarning)

class DEAPFamilyError(Exception):
    """
    Base exception class for all DEAP Solver Family errors.
    
    Provides structured error reporting with complete context preservation
    for audit trails, debugging support, and production error handling.
    """
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 component: Optional[str] = None, phase: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.component = component
        self.phase = phase
        self.timestamp = datetime.now()
        self.execution_id = str(uuid.uuid4())
        self.python_version = sys.version
        self.package_version = __version__

class DEAPFamilyInitializationError(DEAPFamilyError):
    """Raised when DEAP family initialization fails due to configuration or dependency issues."""
    pass

class DEAPFamilyConfigurationError(DEAPFamilyError):
    """Raised when DEAP family configuration is invalid or incomplete."""
    pass

class DEAPFamilySystemInfo:
    """
    System information and capability detection for DEAP Solver Family.
    
    Provides complete system analysis, dependency validation, and
    performance characteristics assessment for usage planning.
    """
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Collect complete system information for usage and debugging.
        
        Returns:
            Dict containing complete system and dependency information
        """
        try:
            # System resources
            memory_info = psutil.virtual_memory()
            cpu_info = {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
            }
            
            # Python environment
            python_info = {
                'version': sys.version,
                'executable': sys.executable,
                'platform': sys.platform,
                'path': sys.path[:3]  # First 3 entries only for brevity
            }
            
            # Package versions
            package_versions = {
                'deap_family': __version__,
                'numpy': np.__version__,
                'pandas': pd.__version__,
                'scipy': scipy.__version__,
                'networkx': nx.__version__,
                'pydantic': pydantic_version,
                'deap': deap_version,
                'fastapi': fastapi_version
            }
            
            # Memory constraints validation
            memory_constraints = {
                'total_mb': round(memory_info.total / (1024 * 1024)),
                'available_mb': round(memory_info.available / (1024 * 1024)),
                'can_support_512mb_constraint': memory_info.available > (512 * 1024 * 1024),
                'recommended_max_problem_size': min(2000, memory_info.available // (1024 * 1024) // 2)
            }
            
            return {
                'system': {
                    'memory': memory_constraints,
                    'cpu': cpu_info,
                    'python': python_info
                },
                'packages': package_versions,
                'capabilities': {
                    'api_available': api_components_available,
                    'fastapi_version': fastapi_version,
                    'can_run_optimization': True,
                    'max_recommended_students': memory_constraints['recommended_max_problem_size']
                },
                'timestamp': datetime.now().isoformat(),
                'package_info': {
                    'name': __package_name__,
                    'version': __version__,
                    'build': __build__,
                    'author': __author__
                }
            }
            
        except Exception as e:
            return {
                'error': f"Failed to collect system information: {str(e)}",
                'timestamp': datetime.now().isoformat(),
                'package_version': __version__
            }
    
    @staticmethod
    def validate_dependencies() -> Tuple[bool, List[str]]:
        """
        Validate all required dependencies and their versions.
        
        Returns:
            Tuple of (all_valid, error_messages)
        """
        errors = []
        
        try:
            # Check numpy version
            if not np.__version__ >= REQUIRED_VERSIONS['numpy'].replace('>=', ''):
                errors.append(f"NumPy version {np.__version__} < required {REQUIRED_VERSIONS['numpy']}")
            
            # Check pandas version
            if not pd.__version__ >= REQUIRED_VERSIONS['pandas'].replace('>=', ''):
                errors.append(f"Pandas version {pd.__version__} < required {REQUIRED_VERSIONS['pandas']}")
            
            # Check scipy version
            if not scipy.__version__ >= REQUIRED_VERSIONS['scipy'].replace('>=', ''):
                errors.append(f"SciPy version {scipy.__version__} < required {REQUIRED_VERSIONS['scipy']}")
            
            # Check DEAP version
            if not deap_version >= REQUIRED_VERSIONS['deap'].replace('>=', ''):
                errors.append(f"DEAP version {deap_version} < required {REQUIRED_VERSIONS['deap']}")
            
            # Check memory availability
            memory_info = psutil.virtual_memory()
            if memory_info.available < (1024 * 1024 * 1024):  # Less than 1GB available
                errors.append("Insufficient memory available (< 1GB) for optimization processing")
            
        except Exception as e:
            errors.append(f"Dependency validation failed: {str(e)}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def estimate_problem_capacity() -> Dict[str, int]:
        """
        Estimate maximum problem sizes based on available system resources.
        
        Returns:
            Dict with estimated capacity limits for different problem parameters
        """
        try:
            memory_info = psutil.virtual_memory()
            available_mb = memory_info.available // (1024 * 1024)
            
            # Conservative estimates based on memory usage patterns observed during development
            # Input layer: ~140KB per course, Processing: ~7KB per individual per course
            # Output layer: ~2KB per course for final output
            
            # Estimate maximum courses (primary constraint)
            max_courses = min(500, available_mb // 2)  # Very conservative estimate
            
            # Estimate maximum students (typically 3-4 courses per student)
            max_students = max_courses * 3  # Conservative ratio
            
            # Estimate population size based on available memory
            max_population_size = min(500, (available_mb * 1024 * 1024) // (max_courses * 50))
            
            return {
                'max_students': max_students,
                'max_courses': max_courses,
                'max_faculty': max_courses // 2,  # Typical ratio
                'max_rooms': max_courses // 3,    # Typical ratio
                'max_timeslots': 40,              # Standard academic schedule
                'max_population_size': max_population_size,
                'max_generations': 1000,          # Algorithm dependent
                'available_memory_mb': available_mb
            }
            
        except Exception as e:
            # Return minimal safe defaults if estimation fails
            return {
                'max_students': 500,
                'max_courses': 200,
                'max_faculty': 50,
                'max_rooms': 30,
                'max_timeslots': 40,
                'max_population_size': 100,
                'max_generations': 500,
                'available_memory_mb': 0,
                'estimation_error': str(e)
            }

def initialize_deap_family(
    enable_logging: bool = True,
    log_level: str = "INFO",
    validate_system: bool = True
) -> Dict[str, Any]:
    """
    Initialize the complete DEAP Solver Family with system validation and configuration.
    
    This function performs complete system initialization including dependency
    validation, logging configuration, memory constraint verification, and
    component readiness assessment.
    
    Args:
        enable_logging: Whether to configure logging for the DEAP family
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        validate_system: Whether to perform system validation and capacity estimation
        
    Returns:
        Dict containing initialization results, system information, and status
        
    Raises:
        DEAPFamilyInitializationError: If critical initialization steps fail
        DEAPFamilyConfigurationError: If system configuration is inadequate
    """
    try:
        initialization_start = datetime.now()
        initialization_id = str(uuid.uuid4())
        
        # Configure logging if requested
        if enable_logging:
            logging.basicConfig(
                level=getattr(logging, log_level.upper()),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Initializing DEAP Solver Family v{__version__}")
        
        # System validation if requested
        system_info = DEAPFamilySystemInfo.get_system_info()
        capacity_estimates = DEAPFamilySystemInfo.estimate_problem_capacity()
        
        if validate_system:
            deps_valid, dep_errors = DEAPFamilySystemInfo.validate_dependencies()
            if not deps_valid:
                error_details = {
                    'dependency_errors': dep_errors,
                    'system_info': system_info
                }
                raise DEAPFamilyInitializationError(
                    "System validation failed - critical dependencies missing or incompatible",
                    context=error_details,
                    component="initialization",
                    phase="system_validation"
                )
        
        # Memory constraint validation
        available_memory_mb = system_info.get('system', {}).get('memory', {}).get('available_mb', 0)
        if available_memory_mb < 1024:  # Require at least 1GB available
            raise DEAPFamilyConfigurationError(
                f"Insufficient memory for optimization: {available_memory_mb}MB < 1024MB required",
                context={'available_memory_mb': available_memory_mb},
                component="initialization",
                phase="memory_validation"
            )
        
        # Component availability check
        component_status = {
            'core_components': True,  # Already imported successfully
            'input_modeling': True,   # Already imported successfully
            'processing_layer': True, # Already imported successfully
            'output_modeling': True,  # Already imported successfully
            'api_layer': api_components_available,
            'fastapi_available': fastapi_available
        }
        
        # Calculate initialization duration
        initialization_duration = (datetime.now() - initialization_start).total_seconds()
        
        # Prepare complete initialization result
        initialization_result = {
            'status': 'SUCCESS',
            'initialization_id': initialization_id,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': initialization_duration,
            'package_info': {
                'name': __package_name__,
                'version': __version__,
                'build': __build__,
                'author': __author__,
                'description': __description__
            },
            'system_info': system_info,
            'capacity_estimates': capacity_estimates,
            'component_status': component_status,
            'supported_algorithms': [solver.value for solver in SolverID],
            'ready_for_optimization': all(component_status.values()) or not api_components_available
        }
        
        logger.info(f"DEAP Solver Family initialization completed successfully in {initialization_duration:.3f}s")
        logger.info(f"System capacity: up to {capacity_estimates['max_students']} students, {capacity_estimates['max_courses']} courses")
        logger.info(f"Supported algorithms: {', '.join(initialization_result['supported_algorithms'])}")
        
        if not api_components_available:
            logger.warning("API components not available - core optimization functionality ready")
        
        return initialization_result
        
    except Exception as e:
        error_msg = f"DEAP Solver Family initialization failed: {str(e)}"
        
        if enable_logging:
            logger = logging.getLogger(__name__)
            logger.error(error_msg)
            logger.debug(f"Initialization error traceback: {traceback.format_exc()}")
        
        # Re-raise with appropriate error type
        if isinstance(e, (DEAPFamilyInitializationError, DEAPFamilyConfigurationError)):
            raise
        else:
            raise DEAPFamilyInitializationError(
                error_msg,
                context={
                    'original_error': str(e),
                    'error_type': type(e).__name__,
                    'traceback': traceback.format_exc()
                },
                component="initialization",
                phase="setup"
            ) from e

def create_deap_optimization_pipeline(
    input_paths: Dict[str, str],
    output_path: str,
    solver_id: Union[str, SolverID],
    **kwargs
) -> DEAPFamilyPipeline:
    """
    Create a configured DEAP optimization pipeline ready for execution.
    
    This function provides a simplified interface for creating optimization pipelines
    with automatic configuration validation and component initialization.
    
    Args:
        input_paths: Dictionary with paths to Stage 3 output files (L_raw, L_rel, L_idx)
        output_path: Path for optimization output files and reports
        solver_id: Evolutionary algorithm to use (GA, GP, ES, DE, PSO, NSGA2)
        **kwargs: Additional configuration parameters for fine-tuning
        
    Returns:
        Configured DEAPFamilyPipeline ready for optimization execution
        
    Raises:
        DEAPFamilyConfigurationError: If configuration parameters are invalid
        DEAPFamilyInitializationError: If pipeline creation fails
        
    Example:
        ```python
        pipeline = create_deap_optimization_pipeline(
            input_paths={
                'L_raw': 'stage3/L_raw.parquet',
                'L_rel': 'stage3/L_rel.graphml',
                'L_idx': 'stage3/L_idx.feather'
            },
            output_path='stage6_results/',
            solver_id='NSGA2',
            population_size=200,
            generations=500
        )
        result = pipeline.run()
        ```
    """
    try:
        # Convert solver_id to enum if necessary
        if isinstance(solver_id, str):
            try:
                solver_enum = SolverID(solver_id.upper())
            except ValueError:
                raise DEAPFamilyConfigurationError(
                    f"Unsupported solver algorithm: {solver_id}",
                    context={'available_solvers': [s.value for s in SolverID]},
                    component="pipeline_creation",
                    phase="solver_validation"
                )
        else:
            solver_enum = solver_id
        
        # Create configuration with provided parameters
        config = DEAPFamilyConfig(
            solver_id=solver_enum,
            paths=PathConfig(
                input_base_path=str(Path(input_paths.get('L_raw', '')).parent),
                output_base_path=output_path,
                **input_paths
            ),
            population_config=PopulationConfig(
                size=kwargs.get('population_size', 200),
                generations=kwargs.get('generations', 500)
            ),
            operator_config=OperatorConfig(
                crossover_probability=kwargs.get('crossover_prob', 0.8),
                mutation_probability=kwargs.get('mutation_prob', 0.2)
            ),
            fitness_weights=FitnessWeights(
                constraint_violation=kwargs.get('constraint_weight', 0.4),
                resource_utilization=kwargs.get('resource_weight', 0.2),
                preference_satisfaction=kwargs.get('preference_weight', 0.15),
                workload_balance=kwargs.get('workload_weight', 0.15),
                schedule_compactness=kwargs.get('compactness_weight', 0.1)
            )
        )
        
        # Validate configuration
        validation_result = validate_deap_family_config(config)
        if not validation_result.is_valid:
            raise DEAPFamilyConfigurationError(
                "Pipeline configuration validation failed",
                context={
                    'validation_errors': validation_result.errors,
                    'provided_config': config.dict()
                },
                component="pipeline_creation",
                phase="config_validation"
            )
        
        # Create pipeline instance
        pipeline = DEAPFamilyPipeline(config)
        
        logger = logging.getLogger(__name__)
        logger.info(f"DEAP optimization pipeline created successfully for {solver_enum.value}")
        
        return pipeline
        
    except Exception as e:
        error_msg = f"Failed to create DEAP optimization pipeline: {str(e)}"
        logger = logging.getLogger(__name__)
        logger.error(error_msg)
        
        if isinstance(e, (DEAPFamilyConfigurationError, DEAPFamilyInitializationError)):
            raise
        else:
            raise DEAPFamilyInitializationError(
                error_msg,
                context={
                    'input_paths': input_paths,
                    'output_path': output_path,
                    'solver_id': str(solver_id),
                    'kwargs': kwargs,
                    'error_type': type(e).__name__
                },
                component="pipeline_creation"
            ) from e

# Public API Interface - Core Functions and Classes
__all__ = [
    # Package information
    '__version__',
    '__author__',
    '__description__',
    
    # Core configuration and pipeline
    'DEAPFamilyConfig',
    'DEAPFamilyPipeline',
    'SolverID',
    'create_deap_optimization_pipeline',
    'run_deap_family_optimization',
    
    # Initialization and system utilities
    'initialize_deap_family',
    'DEAPFamilySystemInfo',
    
    # Input modeling components
    'build_input_context',
    'InputModelContext',
    'DEAPInputModelLoader',
    'DEAPInputModelValidator',
    
    # Processing layer components
    'ProcessingOrchestrator',
    'ProcessingResult',
    'run_evolutionary_optimization',
    'PopulationManager',
    'OperatorManager',
    'DEAPMultiObjectiveFitnessEvaluator',
    'EvolutionaryAlgorithmFactory',
    
    # Output modeling components
    'generate_complete_output',
    'SolutionDecoder',
    'ScheduleWriter',
    'OutputMetadataGenerator',
    
    # Exception hierarchy
    'DEAPFamilyError',
    'DEAPFamilyInitializationError',
    'DEAPFamilyConfigurationError',
    'ProcessingLayerError',
    'ProcessingConfigurationError',
    'ProcessingMemoryError',
    'ProcessingValidationError',
    
    # Type definitions
    'IndividualType',
    'PopulationType',
    'FitnessType',
    
    # Data models
    'PopulationConfig',
    'OperatorConfig',
    'FitnessWeights',
    'PathConfig',
    'MemoryConstraints',
    'ObjectiveMetrics',
    'EvolutionaryRunStatistics'
]

# Add API components to public interface if available
if api_components_available:
    __all__.extend([
        'create_deap_family_app',
        'DEAPOptimizationRequest',
        'DEAPOptimizationResponse',
        'DEAPHealthResponse',
        'DEAPAlgorithmInfo'
    ])

# Module initialization with system validation
try:
    _initialization_result = initialize_deap_family(
        enable_logging=True,
        log_level="INFO",
        validate_system=True
    )
    
    # Store initialization result for external access
    INITIALIZATION_RESULT = _initialization_result
    SYSTEM_INFO = _initialization_result.get('system_info', {})
    CAPACITY_ESTIMATES = _initialization_result.get('capacity_estimates', {})
    
except Exception as init_error:
    # Critical initialization failure - log and re-raise
    print(f"CRITICAL: DEAP Solver Family initialization failed: {str(init_error)}", file=sys.stderr)
    print(f"Error details: {traceback.format_exc()}", file=sys.stderr)
    raise init_error

# Success message
logger = logging.getLogger(__name__)
logger.info("="*80)
logger.info("DEAP SOLVER FAMILY - STAGE 6.3 INITIALIZATION COMPLETE")
logger.info(f"Version: {__version__} (Build: {__build__})")
logger.info(f"Supported Algorithms: {', '.join([s.value for s in SolverID])}")
logger.info(f"System Capacity: {CAPACITY_ESTIMATES.get('max_students', 'Unknown')} students")
logger.info(f"API Layer: {'Available' if api_components_available else 'Not Available'}")
logger.info("Ready for evolutionary scheduling optimization")
logger.info("="*80)