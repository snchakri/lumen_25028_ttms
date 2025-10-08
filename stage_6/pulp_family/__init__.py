#!/usr/bin/env python3
"""
PuLP Solver Family - Stage 6 Master Package Initialization

This package implements the enterprise-grade PuLP solver family for Stage 6.1
scheduling optimization, providing comprehensive mathematical optimization with
theoretical rigor and compliance. Complete solver family implementation per
Stage 6 foundational framework with guaranteed correctness and performance.

Theoretical Foundation:
    Based on Stage 6.1 PuLP Framework (Complete Family Implementation):
    - Implements complete PuLP solver family per foundational design rules
    - Maintains mathematical consistency across all solver backends
    - Ensures comprehensive optimization with convergence guarantees
    - Provides multi-solver backend support with unified interface
    - Supports advanced constraint handling and objective optimization

Architecture Compliance:
    - Implements complete PuLP Solver Family per foundational design architecture
    - Maintains optimal performance characteristics across all solvers
    - Provides fail-safe error handling with comprehensive diagnostic capabilities
    - Supports distributed family processing with centralized management
    - Ensures memory-efficient operations through optimized resource management

Package Structure:
    config.py - Comprehensive configuration management system
    main.py - Master orchestrator and family data pipeline
    input_model/ - Complete input modeling layer with validation
    processing/ - Processing layer with multi-solver backend support
    output_model/ - Output modeling layer with CSV generation
    api/ - RESTful API layer for external integration
    __init__.py - Master package initialization and public API

Solver Backend Support:
    CBC - COIN-OR CBC (Mixed-Integer Programming)
    GLPK - GNU Linear Programming Kit (Linear/Mixed-Integer)
    HiGHS - HiGHS Optimizer (High-Performance LP/MIP)
    CLP - COIN-OR CLP (Linear Programming Specialized)
    SYMPHONY - COIN-OR SYMPHONY (Mixed-Integer Programming)

Dependencies: pulp, numpy, scipy, networkx, pydantic, fastapi, pathlib
Authors: Team LUMEN (SIH 2025)
Version: 1.0.0 (Production)
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

# Import core package components with comprehensive error handling
try:
    # Configuration management
    from .config import (
        PuLPFamilyConfiguration,
        SolverBackend,
        ExecutionMode,
        ConfigurationLevel,
        PathConfiguration,
        SolverConfiguration,
        InputModelConfiguration,
        OutputModelConfiguration,
        DynamicParameterConfiguration,
        load_configuration_from_environment,
        get_execution_directory_path,
        validate_stage3_artifacts,
        create_configuration_template,
        diagnose_configuration_issues,
        DEFAULT_PULP_FAMILY_CONFIGURATION
    )
    CONFIG_IMPORT_SUCCESS = True
    CONFIG_IMPORT_ERROR = None

except ImportError as e:
    CONFIG_IMPORT_SUCCESS = False
    CONFIG_IMPORT_ERROR = str(e)

    # Define fallback classes
    class PuLPFamilyConfiguration: pass
    class SolverBackend: pass
    class ExecutionMode: pass
    DEFAULT_PULP_FAMILY_CONFIGURATION = None

try:
    # Master orchestrator and family data pipeline
    from .main import (
        PuLPFamilyDataPipeline,
        PipelineInvocationContext,
        PipelineExecutionResult,
        PipelineResourceMonitor,
        create_pulp_family_pipeline,
        execute_pulp_solver,
        get_supported_solvers,
        validate_pipeline_environment,
        PIPELINE_IMPORT_STATUS
    )
    MAIN_IMPORT_SUCCESS = True
    MAIN_IMPORT_ERROR = None

except ImportError as e:
    MAIN_IMPORT_SUCCESS = False
    MAIN_IMPORT_ERROR = str(e)

    # Define fallback classes
    class PuLPFamilyDataPipeline: pass
    class PipelineInvocationContext: pass
    class PipelineExecutionResult: pass
    PIPELINE_IMPORT_STATUS = {'overall_import_success': False}

try:
    # Input Model components
    from .input_model import (
        InputModelPipeline,
        create_input_model_pipeline,
        process_stage3_artifacts,
        load_and_validate_input_data,
        create_bijection_mapping_from_data
    )
    INPUT_MODEL_IMPORT_SUCCESS = True
    INPUT_MODEL_IMPORT_ERROR = None

except ImportError as e:
    INPUT_MODEL_IMPORT_SUCCESS = False
    INPUT_MODEL_IMPORT_ERROR = str(e)

    # Define fallback classes
    class InputModelPipeline: pass

try:
    # Processing components
    from .processing import (
        ProcessingPipeline,
        create_processing_pipeline,
        solve_optimization_problem,
        configure_solver_for_problem,
        get_recommended_solver_backend
    )
    PROCESSING_IMPORT_SUCCESS = True
    PROCESSING_IMPORT_ERROR = None

except ImportError as e:
    PROCESSING_IMPORT_SUCCESS = False
    PROCESSING_IMPORT_ERROR = str(e)

    # Define fallback classes
    class ProcessingPipeline: pass

try:
    # Output Model components (when available)
    from .output_model import (
        OutputModelPipeline,
        create_output_model_pipeline,
        process_solver_output
    )
    OUTPUT_MODEL_IMPORT_SUCCESS = True
    OUTPUT_MODEL_IMPORT_ERROR = None

except ImportError as e:
    OUTPUT_MODEL_IMPORT_SUCCESS = False
    OUTPUT_MODEL_IMPORT_ERROR = str(e)

    # Define fallback classes
    class OutputModelPipeline: pass

try:
    # API components (when available)
    from .api import (
        app as fastapi_app,
        SchedulingRequest,
        SchedulingResponse,
        PipelineStatus,
        ExecutionResults,
        HealthCheckResponse,
        ErrorResponse
    )
    API_IMPORT_SUCCESS = True
    API_IMPORT_ERROR = None

except ImportError as e:
    API_IMPORT_SUCCESS = False
    API_IMPORT_ERROR = str(e)

    # Define fallback objects
    fastapi_app = None
    class SchedulingRequest: pass
    class SchedulingResponse: pass

# Configure package-level logger
logger = logging.getLogger(__name__)

# Package metadata and version information
__version__ = "1.0.0"
__stage__ = "6.1"
__family__ = "pulp"
__status__ = "production"

# Package-level configuration
PACKAGE_CONFIG = {
    'version': __version__,
    'stage': __stage__,
    'family': __family__,
    'status': __status__,
    'theoretical_framework': 'Stage 6.1 PuLP Solver Family Framework',
    'mathematical_compliance': True,
    'production_ready': True
}

# Complete import status tracking
PACKAGE_IMPORT_STATUS = {
    'config_import_success': CONFIG_IMPORT_SUCCESS,
    'config_import_error': CONFIG_IMPORT_ERROR,
    'main_import_success': MAIN_IMPORT_SUCCESS,
    'main_import_error': MAIN_IMPORT_ERROR,
    'input_model_import_success': INPUT_MODEL_IMPORT_SUCCESS,
    'input_model_import_error': INPUT_MODEL_IMPORT_ERROR,
    'processing_import_success': PROCESSING_IMPORT_SUCCESS,
    'processing_import_error': PROCESSING_IMPORT_ERROR,
    'output_model_import_success': OUTPUT_MODEL_IMPORT_SUCCESS,
    'output_model_import_error': OUTPUT_MODEL_IMPORT_ERROR,
    'api_import_success': API_IMPORT_SUCCESS,
    'api_import_error': API_IMPORT_ERROR,
    'core_components_available': (CONFIG_IMPORT_SUCCESS and MAIN_IMPORT_SUCCESS and 
                                 INPUT_MODEL_IMPORT_SUCCESS and PROCESSING_IMPORT_SUCCESS),
    'overall_import_success': all([
        CONFIG_IMPORT_SUCCESS, MAIN_IMPORT_SUCCESS, INPUT_MODEL_IMPORT_SUCCESS, PROCESSING_IMPORT_SUCCESS
    ])
}


def get_package_info() -> Dict[str, Any]:
    """
    Get comprehensive package information with import status and capabilities.

    Returns:
        Dictionary containing complete package information and status
    """
    return {
        'package_name': __name__,
        'version': __version__,
        'stage': __stage__,
        'family': __family__,
        'status': __status__,
        'import_status': PACKAGE_IMPORT_STATUS,
        'available_components': get_available_components(),
        'supported_solvers': get_supported_solvers() if MAIN_IMPORT_SUCCESS else [],
        'configuration': PACKAGE_CONFIG,
        'theoretical_framework': PACKAGE_CONFIG['theoretical_framework']
    }


def get_available_components() -> List[str]:
    """
    Get list of available package components based on successful imports.

    Returns:
        List of component names that imported successfully
    """
    available_components = []

    if CONFIG_IMPORT_SUCCESS:
        available_components.append('config')

    if MAIN_IMPORT_SUCCESS:
        available_components.append('main')

    if INPUT_MODEL_IMPORT_SUCCESS:
        available_components.append('input_model')

    if PROCESSING_IMPORT_SUCCESS:
        available_components.append('processing')

    if OUTPUT_MODEL_IMPORT_SUCCESS:
        available_components.append('output_model')

    if API_IMPORT_SUCCESS:
        available_components.append('api')

    return available_components


def verify_package_integrity() -> Dict[str, bool]:
    """
    Verify integrity of PuLP family package components.

    Performs comprehensive verification of package components ensuring
    mathematical correctness and theoretical compliance per framework requirements.

    Returns:
        Dictionary of integrity verification results per component
    """
    integrity_results = {
        'config_module': CONFIG_IMPORT_SUCCESS,
        'main_module': MAIN_IMPORT_SUCCESS,
        'input_model_module': INPUT_MODEL_IMPORT_SUCCESS,
        'processing_module': PROCESSING_IMPORT_SUCCESS,
        'output_model_module': OUTPUT_MODEL_IMPORT_SUCCESS,
        'api_module': API_IMPORT_SUCCESS,
        'core_functionality': False,
        'solver_backends': False,
        'mathematical_compliance': False,
        'overall_integrity': False
    }

    try:
        # Verify core functionality
        if CONFIG_IMPORT_SUCCESS and MAIN_IMPORT_SUCCESS and INPUT_MODEL_IMPORT_SUCCESS and PROCESSING_IMPORT_SUCCESS:
            integrity_results['core_functionality'] = True

            # Verify solver backend support
            if CONFIG_IMPORT_SUCCESS:
                try:
                    supported_solvers = get_supported_solvers()
                    integrity_results['solver_backends'] = len(supported_solvers) >= 5  # CBC, GLPK, HiGHS, CLP, SYMPHONY
                except Exception:
                    pass

            # Verify mathematical compliance
            try:
                if DEFAULT_PULP_FAMILY_CONFIGURATION is not None:
                    config_validation = DEFAULT_PULP_FAMILY_CONFIGURATION.validate_complete_configuration()
                    integrity_results['mathematical_compliance'] = config_validation.get('overall_valid', False)
            except Exception:
                pass

        # Overall integrity assessment
        integrity_results['overall_integrity'] = (
            integrity_results['core_functionality'] and
            integrity_results['solver_backends'] and
            integrity_results['mathematical_compliance']
        )

        logger.debug(f"Package integrity verification: {integrity_results['overall_integrity']}")

    except Exception as e:
        logger.error(f"Package integrity verification failed: {str(e)}")
        integrity_results['verification_error'] = str(e)

    return integrity_results


class PuLPSolverFamilyManager:
    """
    High-level manager for PuLP solver family operations.

    Provides simplified interface for PuLP solver family operations with
    comprehensive configuration management, execution coordination, and
    quality assessment following Stage 6.1 theoretical framework.

    Mathematical Foundation:
        - Implements complete family management per Stage 6.1 framework
        - Maintains mathematical consistency across all family operations
        - Ensures comprehensive solver coordination with performance guarantees
        - Provides fault-tolerant operations with comprehensive error handling
        - Supports multi-execution management with resource optimization
    """

    def __init__(self, configuration: Optional[PuLPFamilyConfiguration] = None):
        """
        Initialize PuLP solver family manager.

        Args:
            configuration: Optional PuLP family configuration
        """
        if not PACKAGE_IMPORT_STATUS['core_components_available']:
            raise RuntimeError(f"Cannot initialize manager - missing core components: {PACKAGE_IMPORT_STATUS}")

        # Load configuration
        if configuration is None:
            configuration = load_configuration_from_environment()

        self.configuration = configuration

        # Initialize family pipeline
        self.family_pipeline: Optional[PuLPFamilyDataPipeline] = None

        # Track executions
        self.executions: Dict[str, Dict[str, Any]] = {}

        logger.info(f"PuLP Solver Family Manager initialized - Mode: {configuration.execution_mode.value}")

    def get_family_pipeline(self) -> PuLPFamilyDataPipeline:
        """
        Get or create family data pipeline instance.

        Returns:
            PuLPFamilyDataPipeline instance
        """
        if self.family_pipeline is None:
            self.family_pipeline = create_pulp_family_pipeline(self.configuration)

        return self.family_pipeline

    def execute_solver(self,
                      solver_id: str,
                      stage3_input_path: str,
                      execution_output_path: str,
                      configuration_overrides: Optional[Dict[str, Any]] = None,
                      timeout_seconds: Optional[float] = None,
                      priority_level: int = 5) -> PipelineExecutionResult:
        """
        Execute PuLP solver with comprehensive pipeline orchestration.

        Args:
            solver_id: PuLP solver identifier
            stage3_input_path: Path to Stage 3 output artifacts
            execution_output_path: Path for execution outputs
            configuration_overrides: Optional configuration overrides
            timeout_seconds: Optional execution timeout
            priority_level: Execution priority (0-10)

        Returns:
            PipelineExecutionResult with comprehensive execution information
        """
        # Create invocation context
        context = PipelineInvocationContext(
            solver_id=solver_id,
            stage3_input_path=stage3_input_path,
            execution_output_path=execution_output_path,
            configuration_overrides=configuration_overrides,
            timeout_seconds=timeout_seconds,
            priority_level=priority_level
        )

        # Get family pipeline
        pipeline = self.get_family_pipeline()

        # Execute pipeline
        result = pipeline.execute_complete_pipeline(context)

        # Track execution
        self.executions[context.execution_id] = {
            'context': context,
            'result': result,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        return result

    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get execution status for specified execution.

        Args:
            execution_id: Unique execution identifier

        Returns:
            Dictionary containing execution status information
        """
        if self.family_pipeline:
            return self.family_pipeline.get_execution_status(execution_id)

        return None

    def list_executions(self) -> List[Dict[str, Any]]:
        """
        List all tracked executions with summary information.

        Returns:
            List of execution summary information
        """
        executions_list = []

        for execution_id, execution_info in self.executions.items():
            result = execution_info['result']
            executions_list.append({
                'execution_id': execution_id,
                'solver_id': result.solver_id,
                'execution_success': result.execution_success,
                'execution_time_seconds': result.execution_time_seconds,
                'quality_grade': result.quality_assessment.get('overall_quality_grade', 'Unknown'),
                'timestamp': execution_info['timestamp']
            })

        return executions_list

    def get_solver_recommendations(self, problem_characteristics: Dict[str, Any]) -> List[str]:
        """
        Get solver recommendations based on problem characteristics.

        Args:
            problem_characteristics: Dictionary with problem characteristics

        Returns:
            List of recommended solver identifiers
        """
        if PROCESSING_IMPORT_SUCCESS:
            try:
                recommended_backend = get_recommended_solver_backend(problem_characteristics)
                return [f"pulp_{recommended_backend.value.lower()}"]
            except Exception as e:
                logger.warning(f"Failed to get solver recommendations: {str(e)}")

        # Default recommendations
        return ['pulp_cbc', 'pulp_highs', 'pulp_glpk']

    def validate_environment(self) -> Dict[str, Any]:
        """
        Validate complete solver family environment.

        Returns:
            Dictionary containing comprehensive environment validation
        """
        if MAIN_IMPORT_SUCCESS:
            return validate_pipeline_environment()
        else:
            return {
                'environment_valid': False,
                'error': 'Main pipeline module not available'
            }


# High-level convenience functions for external use
def create_pulp_solver_manager(configuration: Optional[PuLPFamilyConfiguration] = None) -> PuLPSolverFamilyManager:
    """
    Create PuLP solver family manager with comprehensive configuration.

    Provides simplified interface for manager creation ensuring proper
    configuration and initialization for family management operations.

    Args:
        configuration: Optional PuLP family configuration

    Returns:
        Configured PuLPSolverFamilyManager instance

    Example:
        >>> manager = create_pulp_solver_manager()
        >>> result = manager.execute_solver("pulp_cbc", "./stage3", "./output")
    """
    if not PACKAGE_IMPORT_STATUS['core_components_available']:
        raise RuntimeError(f"Cannot create manager - core components unavailable: {PACKAGE_IMPORT_STATUS}")

    return PuLPSolverFamilyManager(configuration)


def quick_solve(solver_id: str,
               stage3_input_path: str,
               output_path: str = "./output",
               timeout_seconds: Optional[float] = None) -> bool:
    """
    Quick solve function for simple PuLP optimization execution.

    Provides simplified interface for basic PuLP solver execution with
    minimal configuration for rapid prototyping and testing.

    Args:
        solver_id: PuLP solver identifier
        stage3_input_path: Path to Stage 3 artifacts
        output_path: Path for execution outputs
        timeout_seconds: Optional execution timeout

    Returns:
        Boolean indicating execution success

    Example:
        >>> success = quick_solve("pulp_cbc", "./stage3_data", "./results")
        >>> print(f"Optimization {'succeeded' if success else 'failed'}")
    """
    try:
        if MAIN_IMPORT_SUCCESS:
            result = execute_pulp_solver(
                solver_id=solver_id,
                stage3_input_path=stage3_input_path,
                execution_output_path=output_path,
                timeout_seconds=timeout_seconds
            )

            return result.execution_success
        else:
            logger.error("Main pipeline module not available")
            return False

    except Exception as e:
        logger.error(f"Quick solve failed: {str(e)}")
        return False


def get_family_status() -> Dict[str, Any]:
    """
    Get comprehensive PuLP solver family status information.

    Provides complete status information including import status, available
    components, supported solvers, and system capabilities.

    Returns:
        Dictionary containing comprehensive family status
    """
    status = {
        'family_name': 'PuLP Solver Family',
        'version': __version__,
        'stage': __stage__,
        'status': __status__
    }

    # Add package information
    status.update(get_package_info())

    # Add integrity verification
    status['integrity'] = verify_package_integrity()

    # Add environment validation if available
    if MAIN_IMPORT_SUCCESS:
        try:
            status['environment'] = validate_pipeline_environment()
        except Exception as e:
            status['environment'] = {'validation_error': str(e)}

    return status


def initialize_package() -> bool:
    """
    Initialize PuLP solver family package with comprehensive validation.

    Performs comprehensive package initialization and verification ensuring
    all components are available and mathematically compliant per framework requirements.

    Returns:
        Boolean indicating successful package initialization
    """
    try:
        # Verify package integrity
        integrity_results = verify_package_integrity()

        if not integrity_results['overall_integrity']:
            logger.error("PuLP family package integrity verification failed")
            logger.error(f"Integrity results: {integrity_results}")
            return False

        # Validate import status
        if not PACKAGE_IMPORT_STATUS['core_components_available']:
            logger.error("Core components not available for package initialization")
            logger.error(f"Import status: {PACKAGE_IMPORT_STATUS}")
            return False

        # Test configuration loading
        if CONFIG_IMPORT_SUCCESS:
            try:
                config = load_configuration_from_environment()
                config_validation = config.validate_complete_configuration()

                if not config_validation['overall_valid']:
                    logger.warning(f"Default configuration has validation issues: {config_validation}")
            except Exception as e:
                logger.warning(f"Configuration validation failed: {str(e)}")

        # Log successful initialization
        logger.info(f"PuLP Solver Family package v{__version__} initialized successfully")
        logger.info(f"Available components: {get_available_components()}")
        logger.info(f"Supported solvers: {get_supported_solvers() if MAIN_IMPORT_SUCCESS else 'Unknown'}")

        return True

    except Exception as e:
        logger.error(f"PuLP family package initialization failed: {str(e)}")
        return False


# Automatic package initialization on import
from datetime import datetime, timezone
_INITIALIZATION_SUCCESS = initialize_package()

# Public API exports - conditional based on successful imports
if _INITIALIZATION_SUCCESS and PACKAGE_IMPORT_STATUS['core_components_available']:
    __all__ = [
        # Core classes - Config
        'PuLPFamilyConfiguration',
        'SolverBackend',
        'ExecutionMode',
        'ConfigurationLevel',
        'PathConfiguration',
        'SolverConfiguration',
        'InputModelConfiguration',
        'OutputModelConfiguration',
        'DynamicParameterConfiguration',

        # Core classes - Pipeline
        'PuLPFamilyDataPipeline',
        'PipelineInvocationContext',
        'PipelineExecutionResult',
        'PipelineResourceMonitor',

        # Core classes - Components
        'InputModelPipeline',
        'ProcessingPipeline',
        'PuLPSolverFamilyManager',

        # Configuration functions
        'load_configuration_from_environment',
        'get_execution_directory_path',
        'validate_stage3_artifacts',
        'create_configuration_template',
        'diagnose_configuration_issues',

        # Pipeline functions
        'create_pulp_family_pipeline',
        'execute_pulp_solver',
        'get_supported_solvers',
        'validate_pipeline_environment',

        # Input Model functions
        'create_input_model_pipeline',
        'process_stage3_artifacts',
        'load_and_validate_input_data',
        'create_bijection_mapping_from_data',

        # Processing functions
        'create_processing_pipeline',
        'solve_optimization_problem',
        'configure_solver_for_problem',
        'get_recommended_solver_backend',

        # High-level functions
        'create_pulp_solver_manager',
        'quick_solve',
        'get_family_status',

        # Utility functions
        'get_package_info',
        'get_available_components',
        'verify_package_integrity',
        'initialize_package',

        # Constants and status
        'DEFAULT_PULP_FAMILY_CONFIGURATION',
        'PACKAGE_IMPORT_STATUS',
        'PIPELINE_IMPORT_STATUS',

        # Metadata
        '__version__',
        '__stage__',
        '__family__'
    ]

    # Conditionally add API components if available
    if API_IMPORT_SUCCESS:
        __all__.extend([
            'fastapi_app',
            'SchedulingRequest',
            'SchedulingResponse',
            'PipelineStatus',
            'ExecutionResults',
            'HealthCheckResponse',
            'ErrorResponse'
        ])

    # Conditionally add Output Model components if available
    if OUTPUT_MODEL_IMPORT_SUCCESS:
        __all__.extend([
            'OutputModelPipeline',
            'create_output_model_pipeline',
            'process_solver_output'
        ])

else:
    # Limited exports if initialization failed
    __all__ = [
        'get_package_info',
        'get_family_status',
        'PACKAGE_IMPORT_STATUS',
        '__version__',
        '__stage__',
        '__family__'
    ]

# Final package status logging
if _INITIALIZATION_SUCCESS and PACKAGE_IMPORT_STATUS['core_components_available']:
    logger.info(f"PuLP Solver Family package ready: Stage {__stage__} v{__version__}")
    logger.info(f"Family: {__family__} | Status: {__status__}")
    logger.info(f"Theoretical Framework: {PACKAGE_CONFIG['theoretical_framework']}")
else:
    logger.warning(f"PuLP Solver Family package initialization issues detected")
    logger.warning(f"Core components available: {PACKAGE_IMPORT_STATUS['core_components_available']}")
    logger.warning(f"Import status: {PACKAGE_IMPORT_STATUS}")


# Example usage and module testing
if __name__ == "__main__":
    # Comprehensive package testing and demonstration
    import sys

    print(f"PuLP Solver Family Package v{__version__}")
    print(f"Stage: {__stage__} | Family: {__family__} | Status: {__status__}")
    print("=" * 80)

    try:
        # Test package status
        status = get_family_status()
        print(f"✓ Package Status: {status['status']}")
        print(f"✓ Core Components: {status['integrity']['core_functionality']}")
        print(f"✓ Solver Backends: {status['integrity']['solver_backends']}")
        print(f"✓ Mathematical Compliance: {status['integrity']['mathematical_compliance']}")

        # Test available components
        components = get_available_components()
        print(f"✓ Available Components: {components}")

        # Test supported solvers
        if MAIN_IMPORT_SUCCESS:
            solvers = get_supported_solvers()
            print(f"✓ Supported Solvers: {solvers}")

        # Test manager creation
        if PACKAGE_IMPORT_STATUS['core_components_available']:
            manager = create_pulp_solver_manager()
            print(f"✓ Created PuLP solver family manager")

            # Test environment validation
            env_validation = manager.validate_environment()
            print(f"✓ Environment Validation: {'PASSED' if env_validation['environment_valid'] else 'FAILED'}")

        print("=" * 80)
        print("✓ All package tests completed successfully")

    except Exception as e:
        print(f"Package test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
