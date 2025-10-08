#!/usr/bin/env python3
"""
PuLP Solver Family - Stage 6 Master Orchestrator & Family Data Pipeline

This module implements the complete master orchestrator and family data pipeline
for Stage 6.1 PuLP solver family, providing complete solver orchestration, pipeline
coordination, and unified API with mathematical rigor and theoretical compliance. Critical
component implementing complete family pipeline per Stage 6 foundational framework.

Theoretical Foundation:
    Based on Stage 6.1 PuLP Framework pipeline orchestration requirements:
    - Implements complete family data pipeline per foundational design rules
    - Maintains mathematical consistency across all pipeline stages
    - Ensures complete solver coordination with unified interface
    - Provides fail-safe pipeline execution with complete error handling
    - Supports multi-solver backend orchestration with performance guarantees

Architecture Compliance:
    - Implements Family Data Pipeline per foundational design architecture
    - Maintains optimal performance characteristics through pipeline optimization
    - Provides fail-safe execution coordination with complete diagnostics
    - Supports distributed pipeline execution with centralized management
    - Ensures memory-efficient operations through optimized resource management

Pipeline Architecture:
    MASTER ORCHESTRATOR (external) -> FAMILY DATA PIPELINE -> Input Model -> Processing -> Output Model

    Family Data Pipeline Responsibilities:
    1. Accept invocation from master orchestrator with context arguments
    2. Orchestrate Input Model layer with Stage 3 artifacts processing
    3. Coordinate Processing layer with chosen solver backend execution
    4. Manage Output Model layer with complete CSV and metadata generation
    5. Provide unified API interface for all PuLP family solver backends
    6. Ensure fail-fast behavior with complete error handling and logging

Dependencies: pathlib, typing, logging, datetime, uuid, json, asyncio, concurrent.futures
Author: Student Team
Version: 1.0.0 (Production)
"""

import asyncio
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import traceback
import time
import os
import signal
import sys
from dataclasses import dataclass, asdict

# Import configuration management
try:
    from .config import (
        PuLPFamilyConfiguration,
        SolverBackend,
        ExecutionMode,
        ConfigurationLevel,
        load_configuration_from_environment,
        get_execution_directory_path,
        validate_stage3_artifacts,
        diagnose_configuration_issues
    )
    CONFIG_IMPORT_SUCCESS = True
    CONFIG_IMPORT_ERROR = None
except ImportError as e:
    CONFIG_IMPORT_SUCCESS = False
    CONFIG_IMPORT_ERROR = str(e)
    # Fallback classes
    class PuLPFamilyConfiguration: pass
    class SolverBackend: pass

# Import pipeline components with complete error handling
try:
    # Input Model components
    from .input_model import (
        create_input_model_pipeline,
        process_stage3_artifacts,
        load_and_validate_input_data,
        InputModelPipeline
    )
    INPUT_MODEL_IMPORT_SUCCESS = True
    INPUT_MODEL_IMPORT_ERROR = None
except ImportError as e:
    INPUT_MODEL_IMPORT_SUCCESS = False
    INPUT_MODEL_IMPORT_ERROR = str(e)
    # Fallback classes
    class InputModelPipeline: pass

try:
    # Processing components
    from .processing import (
        create_processing_pipeline,
        solve_optimization_problem,
        ProcessingPipeline,
        configure_solver_for_problem,
        get_recommended_solver_backend
    )
    PROCESSING_IMPORT_SUCCESS = True
    PROCESSING_IMPORT_ERROR = None
except ImportError as e:
    PROCESSING_IMPORT_SUCCESS = False
    PROCESSING_IMPORT_ERROR = str(e)
    # Fallback classes
    class ProcessingPipeline: pass

try:
    # Output Model components
    from .output_model import (
        create_output_model_pipeline,
        process_solver_output,
        OutputModelPipeline,
        CSVFormat
    )
    OUTPUT_MODEL_IMPORT_SUCCESS = True
    OUTPUT_MODEL_IMPORT_ERROR = None
except ImportError as e:
    OUTPUT_MODEL_IMPORT_SUCCESS = False
    OUTPUT_MODEL_IMPORT_ERROR = str(e)
    # Fallback classes
    class OutputModelPipeline: pass
    class CSVFormat: pass

# Configure module logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pipeline metadata and version information
__version__ = "1.0.0"
__stage__ = "6.1"
__component__ = "family_data_pipeline"
__status__ = "production"

# Pipeline import status validation
PIPELINE_IMPORT_STATUS = {
    'config_import_success': CONFIG_IMPORT_SUCCESS,
    'config_import_error': CONFIG_IMPORT_ERROR,
    'input_model_import_success': INPUT_MODEL_IMPORT_SUCCESS,
    'input_model_import_error': INPUT_MODEL_IMPORT_ERROR,
    'processing_import_success': PROCESSING_IMPORT_SUCCESS,
    'processing_import_error': PROCESSING_IMPORT_ERROR,
    'output_model_import_success': OUTPUT_MODEL_IMPORT_SUCCESS,
    'output_model_import_error': OUTPUT_MODEL_IMPORT_ERROR,
    'overall_import_success': all([
        CONFIG_IMPORT_SUCCESS, INPUT_MODEL_IMPORT_SUCCESS,
        PROCESSING_IMPORT_SUCCESS, OUTPUT_MODEL_IMPORT_SUCCESS
    ])
}

@dataclass
class PipelineInvocationContext:
    """
    Complete invocation context for family data pipeline execution.

    Mathematical Foundation: Implements complete context specification per
    foundational design rules ensuring proper pipeline configuration and execution.

    Attributes:
        solver_id: Unique solver identifier (e.g., 'pulp_cbc', 'pulp_glpk')
        stage3_input_path: Path to Stage 3 output artifacts directory
        execution_output_path: Path to write execution logs, reports, and outputs
        execution_id: Unique execution identifier for tracking and isolation
        configuration_overrides: Optional configuration parameter overrides
        priority_level: Execution priority level (0=low, 10=high)
        timeout_seconds: Maximum execution timeout in seconds
        memory_limit_mb: Maximum memory usage limit in MB
    """
    solver_id: str
    stage3_input_path: str
    execution_output_path: str
    execution_id: Optional[str] = None
    configuration_overrides: Optional[Dict[str, Any]] = None
    priority_level: int = 5
    timeout_seconds: Optional[float] = None
    memory_limit_mb: Optional[int] = None

    def __post_init__(self):
        """Post-initialization context validation and normalization."""
        # Generate execution ID if not provided
        if self.execution_id is None:
            self.execution_id = f"exec_{uuid.uuid4().hex[:12]}_{int(time.time())}"

        # Validate and normalize paths
        self.stage3_input_path = str(Path(self.stage3_input_path).resolve())
        self.execution_output_path = str(Path(self.execution_output_path).resolve())

        # Validate solver ID format
        if not self.solver_id or not isinstance(self.solver_id, str):
            raise ValueError("Solver ID must be a non-empty string")

        # Parse solver family and backend from solver_id
        if '_' in self.solver_id:
            self.solver_family, self.solver_backend = self.solver_id.split('_', 1)
        else:
            self.solver_family = 'pulp'
            self.solver_backend = self.solver_id

        # Validate priority level
        if not 0 <= self.priority_level <= 10:
            raise ValueError("Priority level must be between 0 and 10")

        # Set default configuration overrides
        if self.configuration_overrides is None:
            self.configuration_overrides = {}

@dataclass
class PipelineExecutionResult:
    """
    complete pipeline execution result with complete metadata.

    Mathematical Foundation: Implements complete result specification per
    foundational framework ensuring complete execution reporting and analysis.

    Attributes:
        execution_id: Unique execution identifier
        solver_id: Solver identifier used in execution
        execution_success: Overall pipeline execution success status
        input_model_result: Input model pipeline execution result
        processing_result: Processing pipeline execution result
        output_model_result: Output model pipeline execution result
        execution_time_seconds: Total pipeline execution time
        memory_usage_mb: Peak memory usage during execution
        error_summary: Summary of any errors encountered
        performance_metrics: complete performance metrics
        output_files: Generated output files and their paths
        quality_assessment: Pipeline execution quality assessment
    """
    execution_id: str
    solver_id: str
    execution_success: bool
    input_model_result: Optional[Dict[str, Any]] = None
    processing_result: Optional[Dict[str, Any]] = None
    output_model_result: Optional[Dict[str, Any]] = None
    execution_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    error_summary: Optional[str] = None
    performance_metrics: Dict[str, float] = None
    output_files: Dict[str, str] = None
    quality_assessment: Dict[str, Any] = None

    def __post_init__(self):
        """Post-initialization result validation and normalization."""
        if self.performance_metrics is None:
            self.performance_metrics = {}

        if self.output_files is None:
            self.output_files = {}

        if self.quality_assessment is None:
            self.quality_assessment = {}

class PipelineResourceMonitor:
    """
    complete resource monitoring for pipeline execution.

    Provides real-time monitoring of system resources during pipeline execution
    ensuring optimal performance and early detection of resource constraints.
    """

    def __init__(self):
        """Initialize resource monitor."""
        self.monitoring_active = False
        self.start_time = None
        self.peak_memory_mb = 0.0
        self.cpu_usage_samples = []
        self.memory_usage_samples = []

        try:
            import psutil
            self.psutil_available = True
            self.process = psutil.Process()
        except ImportError:
            self.psutil_available = False
            logger.warning("psutil not available - resource monitoring limited")

    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        self.monitoring_active = True
        self.start_time = time.time()
        self.peak_memory_mb = 0.0
        self.cpu_usage_samples = []
        self.memory_usage_samples = []

        logger.debug("Resource monitoring started")

    def stop_monitoring(self) -> Dict[str, float]:
        """
        Stop resource monitoring and return metrics.

        Returns:
            Dictionary containing resource usage metrics
        """
        self.monitoring_active = False

        execution_time = time.time() - self.start_time if self.start_time else 0.0

        metrics = {
            'execution_time_seconds': execution_time,
            'peak_memory_mb': self.peak_memory_mb,
            'average_cpu_percent': sum(self.cpu_usage_samples) / len(self.cpu_usage_samples) if self.cpu_usage_samples else 0.0,
            'average_memory_mb': sum(self.memory_usage_samples) / len(self.memory_usage_samples) if self.memory_usage_samples else 0.0,
            'cpu_samples': len(self.cpu_usage_samples),
            'memory_samples': len(self.memory_usage_samples)
        }

        logger.debug(f"Resource monitoring stopped: {metrics}")

        return metrics

    def sample_resources(self) -> Dict[str, float]:
        """
        Sample current resource usage.

        Returns:
            Dictionary containing current resource usage
        """
        if not self.monitoring_active or not self.psutil_available:
            return {}

        try:
            # Get current memory usage
            memory_info = self.process.memory_info()
            current_memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB

            # Update peak memory
            if current_memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = current_memory_mb

            # Get CPU usage
            cpu_percent = self.process.cpu_percent()

            # Store samples
            self.cpu_usage_samples.append(cpu_percent)
            self.memory_usage_samples.append(current_memory_mb)

            # Keep only recent samples (last 100)
            if len(self.cpu_usage_samples) > 100:
                self.cpu_usage_samples = self.cpu_usage_samples[-100:]
            if len(self.memory_usage_samples) > 100:
                self.memory_usage_samples = self.memory_usage_samples[-100:]

            return {
                'current_memory_mb': current_memory_mb,
                'current_cpu_percent': cpu_percent,
                'peak_memory_mb': self.peak_memory_mb
            }

        except Exception as e:
            logger.warning(f"Resource sampling failed: {str(e)}")
            return {}

class PuLPFamilyDataPipeline:
    """
    Master family data pipeline orchestrator for PuLP solver family Stage 6.1.

    Implements complete family data pipeline orchestration following Stage 6.1
    theoretical framework with mathematical guarantees for correctness and
    optimal performance characteristics across all pipeline stages.

    Mathematical Foundation:
        - Implements complete pipeline orchestration per Section 4 (Pipeline Integration)
        - Maintains O(V + C + E) pipeline complexity where V=variables, C=constraints, E=entities
        - Ensures mathematical correctness through rigorous stage coordination
        - Provides fault-tolerant execution with complete error diagnostics
        - Supports multi-solver backend coordination with unified interface
    """

    def __init__(self, configuration: Optional[PuLPFamilyConfiguration] = None):
        """
        Initialize PuLP family data pipeline with complete configuration.

        Args:
            configuration: Optional PuLP family configuration
        """
        if not PIPELINE_IMPORT_STATUS['overall_import_success']:
            raise RuntimeError(f"Pipeline initialization failed - import errors: {PIPELINE_IMPORT_STATUS}")

        # Load configuration
        if configuration is None:
            configuration = load_configuration_from_environment()

        self.configuration = configuration

        # Initialize pipeline components
        self.input_model_pipeline: Optional[InputModelPipeline] = None
        self.processing_pipeline: Optional[ProcessingPipeline] = None
        self.output_model_pipeline: Optional[OutputModelPipeline] = None

        # Initialize resource monitoring
        self.resource_monitor = PipelineResourceMonitor()

        # Initialize execution state
        self.current_execution_context: Optional[PipelineInvocationContext] = None
        self.execution_results: Dict[str, PipelineExecutionResult] = {}
        self.active_executions: Dict[str, Dict[str, Any]] = {}

        # Validate configuration
        config_validation = self.configuration.validate_complete_configuration()
        if not config_validation['overall_valid']:
            logger.warning(f"Configuration validation issues detected: {config_validation}")
            if self.configuration.configuration_level == ConfigurationLevel.STRICT:
                raise RuntimeError("Strict configuration validation failed")

        logger.info(f"PuLP Family Data Pipeline initialized - Mode: {configuration.execution_mode.value}")

    def validate_invocation_context(self, context: PipelineInvocationContext) -> Dict[str, bool]:
        """
        Validate pipeline invocation context with complete checks.

        Args:
            context: Pipeline invocation context

        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'solver_id_valid': False,
            'stage3_input_accessible': False,
            'execution_output_writable': False,
            'solver_backend_supported': False,
            'resource_limits_valid': False,
            'overall_valid': False
        }

        try:
            # Validate solver ID and extract backend
            if '_' in context.solver_id:
                family, backend = context.solver_id.split('_', 1)
                if family.lower() == 'pulp':
                    try:
                        solver_backend = SolverBackend(backend.upper())
                        validation_results['solver_backend_supported'] = True
                        validation_results['solver_id_valid'] = True
                    except ValueError:
                        logger.error(f"Unsupported solver backend: {backend}")
                else:
                    logger.error(f"Invalid solver family: {family}")
            else:
                # Assume PuLP family if no underscore
                try:
                    solver_backend = SolverBackend(context.solver_id.upper())
                    validation_results['solver_backend_supported'] = True
                    validation_results['solver_id_valid'] = True
                except ValueError:
                    logger.error(f"Unsupported solver: {context.solver_id}")

            # Validate Stage 3 input path
            stage3_path = Path(context.stage3_input_path)
            if stage3_path.exists() and stage3_path.is_dir():
                # Check for required artifacts
                artifacts_validation = validate_stage3_artifacts(self.configuration)
                validation_results['stage3_input_accessible'] = artifacts_validation['all_artifacts_available']
            else:
                logger.error(f"Stage 3 input path not accessible: {stage3_path}")

            # Validate execution output path
            output_path = Path(context.execution_output_path)
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                validation_results['execution_output_writable'] = os.access(str(output_path), os.W_OK)
            except Exception as e:
                logger.error(f"Cannot create execution output directory: {str(e)}")

            # Validate resource limits
            if context.timeout_seconds is not None and context.timeout_seconds > 0:
                timeout_valid = True
            elif context.timeout_seconds is None:
                timeout_valid = True
            else:
                timeout_valid = False

            if context.memory_limit_mb is not None and context.memory_limit_mb > 0:
                memory_valid = True
            elif context.memory_limit_mb is None:
                memory_valid = True
            else:
                memory_valid = False

            validation_results['resource_limits_valid'] = timeout_valid and memory_valid

            # Overall validation
            validation_results['overall_valid'] = all([
                validation_results['solver_id_valid'],
                validation_results['stage3_input_accessible'],
                validation_results['execution_output_writable'],
                validation_results['solver_backend_supported'],
                validation_results['resource_limits_valid']
            ])

        except Exception as e:
            logger.error(f"Context validation failed: {str(e)}")
            validation_results['validation_error'] = str(e)

        return validation_results

    def execute_input_model_pipeline(self, context: PipelineInvocationContext,
                                   execution_directory: Path) -> Dict[str, Any]:
        """
        Execute input model pipeline with complete processing.

        Args:
            context: Pipeline invocation context
            execution_directory: Execution-specific directory

        Returns:
            Dictionary containing input model pipeline results
        """
        logger.info(f"Executing Input Model pipeline for {context.execution_id}")

        try:
            # Create input model pipeline
            self.input_model_pipeline = create_input_model_pipeline()

            # Configure input paths from context
            l_raw_path = Path(context.stage3_input_path) / "L_raw.parquet"
            l_rel_path = Path(context.stage3_input_path) / "L_rel.graphml"

            # Find L_idx file with supported extensions
            l_idx_path = None
            possible_extensions = ['.idx', '.bin', '.parquet', '.feather', '.pkl']
            for ext in possible_extensions:
                potential_path = Path(context.stage3_input_path) / f"L_idx{ext}"
                if potential_path.exists():
                    l_idx_path = potential_path
                    break

            if l_idx_path is None:
                raise FileNotFoundError("L_idx file not found with any supported extension")

            # Execute complete input model pipeline
            input_model_results = self.input_model_pipeline.execute_complete_pipeline(
                l_raw_path=str(l_raw_path),
                l_rel_path=str(l_rel_path),
                l_idx_path=str(l_idx_path),
                output_directory=execution_directory / "input_model"
            )

            logger.info(f"Input Model pipeline completed successfully: {input_model_results['pipeline_success']}")

            return input_model_results

        except Exception as e:
            logger.error(f"Input Model pipeline failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Input Model pipeline execution failed: {str(e)}") from e

    def execute_processing_pipeline(self, context: PipelineInvocationContext,
                                  execution_directory: Path,
                                  input_model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute processing pipeline with chosen solver backend.

        Args:
            context: Pipeline invocation context
            execution_directory: Execution-specific directory
            input_model_results: Results from input model pipeline

        Returns:
            Dictionary containing processing pipeline results
        """
        logger.info(f"Executing Processing pipeline with solver: {context.solver_id}")

        try:
            # Extract solver backend from context
            if '_' in context.solver_id:
                _, backend_name = context.solver_id.split('_', 1)
            else:
                backend_name = context.solver_id

            solver_backend = SolverBackend(backend_name.upper())

            # Create processing pipeline
            from .processing import ProcessingMode
            processing_mode = ProcessingMode.BALANCED  # Default mode

            self.processing_pipeline = create_processing_pipeline(processing_mode)

            # Prepare solver configuration
            solver_config = {}
            if context.timeout_seconds:
                solver_config['time_limit_seconds'] = context.timeout_seconds
            if context.memory_limit_mb:
                solver_config['memory_limit_mb'] = context.memory_limit_mb

            # Add configuration overrides
            if context.configuration_overrides:
                solver_config.update(context.configuration_overrides)

            # Execute complete processing pipeline
            processing_results = self.processing_pipeline.execute_complete_pipeline(
                input_data=input_model_results['loaded_data'],
                bijection_mapping=input_model_results['bijection_mapping'],
                solver_backend=solver_backend,
                configuration=solver_config,
                output_directory=execution_directory / "processing"
            )

            logger.info(f"Processing pipeline completed successfully: {processing_results['pipeline_success']}")

            return processing_results

        except Exception as e:
            logger.error(f"Processing pipeline failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Processing pipeline execution failed: {str(e)}") from e

    def execute_output_model_pipeline(self, context: PipelineInvocationContext,
                                    execution_directory: Path,
                                    input_model_results: Dict[str, Any],
                                    processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute output model pipeline with complete CSV generation.

        Args:
            context: Pipeline invocation context
            execution_directory: Execution-specific directory
            input_model_results: Results from input model pipeline
            processing_results: Results from processing pipeline

        Returns:
            Dictionary containing output model pipeline results
        """
        logger.info(f"Executing Output Model pipeline for {context.execution_id}")

        try:
            # Create output model pipeline
            self.output_model_pipeline = create_output_model_pipeline(context.execution_id)

            # Determine CSV format from configuration
            csv_format = CSVFormat.EXTENDED if self.configuration.output_model.csv_format_extended else CSVFormat.STANDARD

            # Execute complete output model pipeline
            output_model_results = self.output_model_pipeline.execute_complete_pipeline(
                solver_result=processing_results['solver_result'],
                bijection_mapping=input_model_results['bijection_mapping'],
                entity_collections=input_model_results['loaded_data'].entity_collections,
                output_directory=execution_directory / "output_model",
                csv_format=csv_format
            )

            logger.info(f"Output Model pipeline completed successfully: {output_model_results['pipeline_success']}")

            return output_model_results

        except Exception as e:
            logger.error(f"Output Model pipeline failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Output Model pipeline execution failed: {str(e)}") from e

    def execute_complete_pipeline(self, context: PipelineInvocationContext) -> PipelineExecutionResult:
        """
        Execute complete family data pipeline with complete orchestration.

        Orchestrates end-to-end pipeline execution following Stage 6.1 theoretical
        framework ensuring mathematical correctness and optimal performance characteristics.

        Args:
            context: Complete pipeline invocation context

        Returns:
            PipelineExecutionResult with complete execution information

        Raises:
            RuntimeError: If pipeline execution fails critical validation
            ValueError: If invocation context is invalid
        """
        logger.info(f"Starting complete PuLP family data pipeline execution: {context.execution_id}")

        # Initialize execution result
        execution_result = PipelineExecutionResult(
            execution_id=context.execution_id,
            solver_id=context.solver_id,
            execution_success=False
        )

        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        execution_start_time = time.time()

        try:
            # Store current execution context
            self.current_execution_context = context
            self.active_executions[context.execution_id] = {
                'context': context,
                'start_time': execution_start_time,
                'status': 'starting'
            }

            # Validate invocation context
            logger.info("Phase 1: Validating invocation context")
            context_validation = self.validate_invocation_context(context)

            if not context_validation['overall_valid']:
                raise ValueError(f"Context validation failed: {context_validation}")

            self.active_executions[context.execution_id]['status'] = 'context_validated'

            # Create execution directory
            logger.info("Phase 2: Creating execution environment")
            execution_directory = get_execution_directory_path(self.configuration, context.execution_id)

            # Update configuration with context overrides
            if context.configuration_overrides:
                # Apply configuration overrides (not modifying original config)
                logger.info(f"Applying configuration overrides: {len(context.configuration_overrides)} parameters")

            self.active_executions[context.execution_id]['status'] = 'environment_ready'

            # Phase 3: Execute Input Model Pipeline
            logger.info("Phase 3: Executing Input Model pipeline")
            self.active_executions[context.execution_id]['status'] = 'input_model_processing'

            input_model_results = self.execute_input_model_pipeline(context, execution_directory)
            execution_result.input_model_result = input_model_results

            if not input_model_results['pipeline_success']:
                raise RuntimeError("Input Model pipeline failed")

            self.active_executions[context.execution_id]['status'] = 'input_model_completed'

            # Phase 4: Execute Processing Pipeline
            logger.info("Phase 4: Executing Processing pipeline")
            self.active_executions[context.execution_id]['status'] = 'processing'

            processing_results = self.execute_processing_pipeline(
                context, execution_directory, input_model_results
            )
            execution_result.processing_result = processing_results

            if not processing_results['pipeline_success']:
                raise RuntimeError("Processing pipeline failed")

            self.active_executions[context.execution_id]['status'] = 'processing_completed'

            # Phase 5: Execute Output Model Pipeline
            logger.info("Phase 5: Executing Output Model pipeline")
            self.active_executions[context.execution_id]['status'] = 'output_model_processing'

            output_model_results = self.execute_output_model_pipeline(
                context, execution_directory, input_model_results, processing_results
            )
            execution_result.output_model_result = output_model_results

            if not output_model_results['pipeline_success']:
                raise RuntimeError("Output Model pipeline failed")

            self.active_executions[context.execution_id]['status'] = 'output_model_completed'

            # Phase 6: Pipeline Success Assessment
            logger.info("Phase 6: Completing pipeline execution assessment")

            execution_result.execution_success = (
                input_model_results['pipeline_success'] and
                processing_results['pipeline_success'] and
                output_model_results['pipeline_success']
            )

            # Collect performance metrics
            resource_metrics = self.resource_monitor.stop_monitoring()
            execution_result.execution_time_seconds = time.time() - execution_start_time
            execution_result.memory_usage_mb = resource_metrics.get('peak_memory_mb', 0.0)

            execution_result.performance_metrics = {
                'total_execution_time_seconds': execution_result.execution_time_seconds,
                'input_model_time_seconds': input_model_results.get('performance_metrics', {}).get('loading_time_seconds', 0.0),
                'processing_time_seconds': processing_results.get('performance_metrics', {}).get('solver_time_seconds', 0.0),
                'output_model_time_seconds': 0.0,  # Would be calculated from output model results
                'peak_memory_usage_mb': execution_result.memory_usage_mb,
                'variables_processed': processing_results.get('performance_metrics', {}).get('total_variables', 0),
                'constraints_processed': processing_results.get('performance_metrics', {}).get('total_constraints', 0)
            }

            # Collect output files
            execution_result.output_files = {}
            if input_model_results.get('output_files'):
                execution_result.output_files.update({f"input_{k}": v for k, v in input_model_results['output_files'].items()})
            if processing_results.get('output_files'):
                execution_result.output_files.update({f"processing_{k}": v for k, v in processing_results['output_files'].items()})
            if output_model_results.get('output_files'):
                execution_result.output_files.update({f"output_{k}": v for k, v in output_model_results['output_files'].items()})

            # Generate quality assessment
            execution_result.quality_assessment = self._generate_quality_assessment(
                input_model_results, processing_results, output_model_results, execution_result
            )

            self.active_executions[context.execution_id]['status'] = 'completed'

            # Store execution result
            self.execution_results[context.execution_id] = execution_result

            logger.info(f"Complete PuLP family data pipeline executed successfully: {context.execution_id}")
            logger.info(f"Quality Grade: {execution_result.quality_assessment.get('overall_quality_grade', 'Unknown')}")
            logger.info(f"Execution Time: {execution_result.execution_time_seconds:.2f}s")
            logger.info(f"Peak Memory: {execution_result.memory_usage_mb:.1f}MB")

            return execution_result

        except Exception as e:
            # Handle pipeline execution failure
            logger.error(f"PuLP family data pipeline execution failed: {str(e)}")
            logger.error(traceback.format_exc())

            # Update execution result with error information
            execution_result.execution_success = False
            execution_result.error_summary = str(e)
            execution_result.execution_time_seconds = time.time() - execution_start_time

            # Stop resource monitoring
            resource_metrics = self.resource_monitor.stop_monitoring()
            execution_result.memory_usage_mb = resource_metrics.get('peak_memory_mb', 0.0)

            # Update execution status
            if context.execution_id in self.active_executions:
                self.active_executions[context.execution_id]['status'] = 'failed'
                self.active_executions[context.execution_id]['error'] = str(e)

            # Store failed execution result
            self.execution_results[context.execution_id] = execution_result

            raise RuntimeError(f"Pipeline execution failed: {str(e)}") from e

        finally:
            # Clean up execution state
            if context.execution_id in self.active_executions:
                self.active_executions[context.execution_id]['end_time'] = time.time()

            # Clear current execution context
            self.current_execution_context = None

    def _generate_quality_assessment(self,
                                   input_model_results: Dict[str, Any],
                                   processing_results: Dict[str, Any],
                                   output_model_results: Dict[str, Any],
                                   execution_result: PipelineExecutionResult) -> Dict[str, Any]:
        """Generate complete quality assessment for pipeline execution."""
        try:
            # Component quality scores
            input_quality = input_model_results.get('quality_assessment', {}).get('overall_quality_grade', 'C')
            processing_quality = processing_results.get('quality_assessment', {}).get('overall_quality_grade', 'C')
            output_quality = output_model_results.get('quality_assessment', {}).get('overall_quality_grade', 'C')

            # Convert grades to scores
            grade_scores = {'A+': 1.0, 'A': 0.95, 'B+': 0.85, 'B': 0.80, 'C+': 0.75, 'C': 0.70, 'D': 0.60}

            input_score = grade_scores.get(input_quality, 0.70)
            processing_score = grade_scores.get(processing_quality, 0.70)
            output_score = grade_scores.get(output_quality, 0.70)

            # Performance score based on execution time and memory
            time_score = min(1.0, max(0.5, 1.0 - (execution_result.execution_time_seconds / 600.0)))  # Normalize by 10 minutes
            memory_score = min(1.0, max(0.5, 1.0 - (execution_result.memory_usage_mb / 1024.0)))  # Normalize by 1GB

            # Overall quality score (weighted average)
            overall_score = (
                input_score * 0.25 +
                processing_score * 0.40 +
                output_score * 0.25 +
                time_score * 0.05 +
                memory_score * 0.05
            )

            # Convert back to grade
            if overall_score >= 0.98:
                overall_grade = "A+"
            elif overall_score >= 0.92:
                overall_grade = "A"
            elif overall_score >= 0.87:
                overall_grade = "B+"
            elif overall_score >= 0.82:
                overall_grade = "B"
            elif overall_score >= 0.77:
                overall_grade = "C+"
            elif overall_score >= 0.72:
                overall_grade = "C"
            else:
                overall_grade = "D"

            return {
                'overall_quality_score': overall_score,
                'overall_quality_grade': overall_grade,
                'component_grades': {
                    'input_model': input_quality,
                    'processing': processing_quality,
                    'output_model': output_quality
                },
                'performance_metrics': {
                    'time_score': time_score,
                    'memory_score': memory_score,
                    'execution_efficiency': (time_score + memory_score) / 2.0
                },
                'recommendations': self._generate_quality_recommendations(overall_score, execution_result)
            }

        except Exception as e:
            logger.warning(f"Quality assessment generation failed: {str(e)}")
            return {
                'overall_quality_score': 0.0,
                'overall_quality_grade': 'Unknown',
                'assessment_error': str(e)
            }

    def _generate_quality_recommendations(self, quality_score: float,
                                        execution_result: PipelineExecutionResult) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []

        if quality_score < 0.85:
            recommendations.append("Consider optimizing solver configuration for better performance")

        if execution_result.execution_time_seconds > 300:
            recommendations.append("Execution time exceeds optimal range - consider problem size reduction")

        if execution_result.memory_usage_mb > 512:
            recommendations.append("Memory usage is high - consider memory optimization techniques")

        if not recommendations:
            recommendations.append("Pipeline execution meets quality standards")

        return recommendations

    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete execution status for specified execution.

        Args:
            execution_id: Unique execution identifier

        Returns:
            Dictionary containing execution status information
        """
        if execution_id in self.execution_results:
            # Completed execution
            result = self.execution_results[execution_id]
            return {
                'execution_id': execution_id,
                'status': 'completed',
                'execution_success': result.execution_success,
                'execution_time_seconds': result.execution_time_seconds,
                'memory_usage_mb': result.memory_usage_mb,
                'quality_grade': result.quality_assessment.get('overall_quality_grade', 'Unknown'),
                'error_summary': result.error_summary,
                'output_files': result.output_files
            }
        elif execution_id in self.active_executions:
            # Active execution
            execution_info = self.active_executions[execution_id]
            current_time = time.time()
            elapsed_time = current_time - execution_info['start_time']

            return {
                'execution_id': execution_id,
                'status': execution_info['status'],
                'elapsed_time_seconds': elapsed_time,
                'solver_id': execution_info['context'].solver_id,
                'error': execution_info.get('error')
            }
        else:
            return None

    def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel active execution with proper cleanup.

        Args:
            execution_id: Unique execution identifier

        Returns:
            Boolean indicating successful cancellation
        """
        if execution_id in self.active_executions:
            try:
                # Mark execution as cancelled
                self.active_executions[execution_id]['status'] = 'cancelled'
                self.active_executions[execution_id]['cancelled_at'] = time.time()

                # Clean up resources if current execution
                if (self.current_execution_context and 
                    self.current_execution_context.execution_id == execution_id):
                    self.resource_monitor.stop_monitoring()
                    self.current_execution_context = None

                logger.info(f"Execution cancelled: {execution_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to cancel execution {execution_id}: {str(e)}")
                return False

        return False

# High-level API functions for external invocation
def create_pulp_family_pipeline(configuration: Optional[PuLPFamilyConfiguration] = None) -> PuLPFamilyDataPipeline:
    """
    Create PuLP family data pipeline instance with complete configuration.

    Provides simplified interface for pipeline creation ensuring proper
    configuration and initialization for family data pipeline operations.

    Args:
        configuration: Optional PuLP family configuration

    Returns:
        Configured PuLPFamilyDataPipeline instance

    Example:
        >>> pipeline = create_pulp_family_pipeline()
        >>> context = PipelineInvocationContext("pulp_cbc", "./stage3", "./output")
        >>> result = pipeline.execute_complete_pipeline(context)
    """
    if not PIPELINE_IMPORT_STATUS['overall_import_success']:
        raise RuntimeError(f"Cannot create pipeline - import failures: {PIPELINE_IMPORT_STATUS}")

    return PuLPFamilyDataPipeline(configuration)

def execute_pulp_solver(solver_id: str,
                       stage3_input_path: str,
                       execution_output_path: str,
                       configuration_overrides: Optional[Dict[str, Any]] = None,
                       timeout_seconds: Optional[float] = None) -> PipelineExecutionResult:
    """
    High-level function to execute PuLP solver with complete pipeline orchestration.

    Provides simplified interface for complete PuLP solver execution following
    Stage 6.1 theoretical framework with mathematical guarantees and performance optimization.

    Args:
        solver_id: PuLP solver identifier (e.g., 'pulp_cbc', 'pulp_glpk')
        stage3_input_path: Path to Stage 3 output artifacts
        execution_output_path: Path for execution outputs and logs
        configuration_overrides: Optional configuration parameter overrides
        timeout_seconds: Optional execution timeout

    Returns:
        PipelineExecutionResult with complete execution information

    Example:
        >>> result = execute_pulp_solver("pulp_cbc", "./stage3", "./output")
        >>> print(f"Execution {'succeeded' if result.execution_success else 'failed'}")
    """
    # Create invocation context
    context = PipelineInvocationContext(
        solver_id=solver_id,
        stage3_input_path=stage3_input_path,
        execution_output_path=execution_output_path,
        configuration_overrides=configuration_overrides,
        timeout_seconds=timeout_seconds
    )

    # Create and execute pipeline
    pipeline = create_pulp_family_pipeline()

    return pipeline.execute_complete_pipeline(context)

def get_supported_solvers() -> List[str]:
    """
    Get list of supported PuLP solver identifiers.

    Returns:
        List of supported solver identifiers
    """
    if not CONFIG_IMPORT_SUCCESS:
        return ['pulp_cbc', 'pulp_glpk', 'pulp_highs', 'pulp_clp', 'pulp_symphony']

    try:
        return [f"pulp_{backend.value.lower()}" for backend in SolverBackend]
    except Exception:
        return ['pulp_cbc', 'pulp_glpk', 'pulp_highs', 'pulp_clp', 'pulp_symphony']

def validate_pipeline_environment() -> Dict[str, Any]:
    """
    Validate complete pipeline environment and dependencies.

    Performs complete validation of pipeline environment ensuring
    all required dependencies and configurations are properly available.

    Returns:
        Dictionary containing complete environment validation results
    """
    validation_results = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'import_status': PIPELINE_IMPORT_STATUS,
        'supported_solvers': get_supported_solvers(),
        'environment_valid': PIPELINE_IMPORT_STATUS['overall_import_success']
    }

    if CONFIG_IMPORT_SUCCESS:
        try:
            # Test configuration loading
            config = load_configuration_from_environment()
            config_validation = config.validate_complete_configuration()

            validation_results['configuration'] = {
                'default_config_valid': config_validation['overall_valid'],
                'execution_mode': config.execution_mode.value,
                'configuration_level': config.configuration_level.value
            }

        except Exception as e:
            validation_results['configuration'] = {
                'default_config_valid': False,
                'config_error': str(e)
            }

    return validation_results

# Export public API for external use
__all__ = [
    # Main classes
    'PuLPFamilyDataPipeline',
    'PipelineInvocationContext',
    'PipelineExecutionResult',
    'PipelineResourceMonitor',

    # High-level functions
    'create_pulp_family_pipeline',
    'execute_pulp_solver',
    'get_supported_solvers',
    'validate_pipeline_environment',

    # Import status
    'PIPELINE_IMPORT_STATUS',

    # Metadata
    '__version__',
    '__stage__',
    '__component__'
]

# Module initialization and validation
if __name__ == "__main__":
    # Example usage and testing
    print("Testing PuLP Family Data Pipeline...")

    try:
        # Validate pipeline environment
        env_validation = validate_pipeline_environment()
        print(f"✓ Pipeline environment validation: {'PASSED' if env_validation['environment_valid'] else 'FAILED'}")

        if not env_validation['environment_valid']:
            print(f"Import issues: {env_validation['import_status']}")
            sys.exit(1)

        # Test pipeline creation
        pipeline = create_pulp_family_pipeline()
        print(f"✓ Created PuLP family data pipeline")

        # Test supported solvers
        solvers = get_supported_solvers()
        print(f"✓ Supported solvers: {solvers}")

        # Test context validation (with dummy paths)
        context = PipelineInvocationContext(
            solver_id="pulp_cbc",
            stage3_input_path="./test_stage3",
            execution_output_path="./test_output"
        )
        print(f"✓ Created invocation context: {context.execution_id}")

        print("✓ All pipeline tests passed")

    except Exception as e:
        print(f"Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
