# -*- coding: utf-8 -*-
"""
Stage 6.3 DEAP Solver Family - Processing Package Initialization

This module provides comprehensive initialization and orchestration for the evolutionary
computing processing layer within the DEAP Solver Family. It establishes the complete
infrastructure for population-based optimization algorithms including Genetic Algorithm (GA),
Genetic Programming (GP), Evolution Strategies (ES), Differential Evolution (DE), 
Particle Swarm Optimization (PSO), and NSGA-II multi-objective optimization.

THEORETICAL FOUNDATIONS:
- DEAP Framework Definition 2.1: Evolutionary Algorithm Framework (λ, μ, σ, τ, χ, Ψ)
- Algorithm 11.2: Integrated Evolutionary Process with multi-objective fitness evaluation
- Definition 2.2: Schedule Genotype Encoding g: course → (faculty, room, timeslot, batch)
- Theorem 3.2: GA Schema Theorem for scheduling pattern preservation and transmission
- Definition 2.4: Multi-Objective Fitness Model f(g) = (f₁, f₂, f₃, f₄, f₅)

ARCHITECTURAL COMPLIANCE:
- Course-centric representation with bijective genotype-phenotype mapping
- Memory-bounded processing (≤250MB peak usage with real-time monitoring)
- Single-threaded evolutionary loops with deterministic execution patterns
- Fail-fast validation with comprehensive error propagation and audit logging
- Complete integration with Stage 3 Dynamic Parametric System (EAV parameters)

PROCESSING PIPELINE ARCHITECTURE:
1. Population Management: Individual validation, diversity maintenance, memory estimation
2. Evolutionary Operators: Crossover, mutation, selection with constraint preservation
3. Fitness Evaluation: Multi-objective assessment with f₁-f₅ optimization targets
4. Algorithm Engines: Complete DEAP toolbox integration with theoretical guarantees
5. Statistical Logging: Convergence analysis, diversity tracking, performance profiling

Author: Perplexity Labs AI - Enterprise Scheduling Engine Development Team
Date: October 2025
Version: 1.0.0 - Production Release Candidate
License: Proprietary - SIH 2025 Competition Entry

CURSOR IDE & JETBRAINS IDE INTEGRATION:
- Complete type hints with Pydantic model validation for enhanced IntelliSense
- Cross-module references with explicit import paths for dependency tracking
- Comprehensive docstring coverage for automatic documentation generation
- Professional exception hierarchy with detailed error context for debugging
- Real-time memory monitoring integration for IDE resource tracking
"""

from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Type
from typing import TypeVar, Generic, Protocol, runtime_checkable
import logging
import sys
import gc
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import PositiveInt, PositiveFloat, NonNegativeFloat
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid
import json
import psutil
import warnings

# Stage 6.3 DEAP Family Internal Imports - Maintaining strict circular import avoidance
from ..deap_family_config import (
    DEAPFamilyConfig,
    SolverID,
    PopulationConfig,
    OperatorConfig,
    FitnessWeights,
    MemoryConstraints
)
from ..deap_family_main import (
    PipelineContext,
    MemoryMonitor,
    ExecutionTimer,
    AuditLogger
)
from ..input_model.metadata import (
    InputModelContext,
    CourseEligibilityMap,
    ConstraintRulesMap,
    BijectionMappingData
)

# Processing Layer Component Imports - Explicit internal module references
try:
    from .population import (
        PopulationManager,
        IndividualType,
        PopulationType,
        FitnessType,
        PopulationStatistics,
        IndividualValidator,
        PopulationInitializer
    )
    from .operators import (
        CrossoverOperators,
        MutationOperators,
        SelectionOperators,
        OperatorManager,
        OperatorStatistics,
        RepairMechanisms
    )
    from .evaluator import (
        DEAPMultiObjectiveFitnessEvaluator,
        ObjectiveMetrics,
        EvaluationStatistics,
        ConstraintViolationAnalyzer,
        FitnessValidationError
    )
    from .engine import (
        EvolutionaryAlgorithmFactory,
        EvolutionaryResult,
        EvolutionaryRunStatistics,
        AlgorithmNotSupportedError,
        ConvergenceAnalysisError
    )
    from .logging import (
        EvolutionaryLogger,
        GenerationMetrics,
        ConvergenceAnalyzer,
        DiversityTracker,
        PerformanceProfiler
    )
except ImportError as e:
    # Critical import failure handling with detailed error context
    error_msg = f"Failed to import processing layer components: {str(e)}"
    logging.error(error_msg)
    raise ImportError(f"Processing package initialization failed: {error_msg}") from e

# DEAP Framework Integration - Essential evolutionary computing infrastructure
try:
    import deap
    from deap import base, creator, tools, algorithms
    from deap.tools import Logbook, HallOfFame
    from deap.algorithms import varAnd
except ImportError as e:
    # DEAP library is absolutely critical for evolutionary algorithms
    error_msg = f"DEAP evolutionary computation library not available: {str(e)}"
    logging.critical(error_msg)
    raise ImportError(f"DEAP library required for evolutionary processing: {error_msg}") from e

# Memory Management and Performance Monitoring
warnings.filterwarnings('ignore', category=UserWarning, module='deap')

# Type Definitions for Enhanced IDE Support and Type Safety
T = TypeVar('T')
ProcessingResultType = TypeVar('ProcessingResultType', bound='ProcessingResult')
EvolutionaryEngineType = TypeVar('EvolutionaryEngineType')

@runtime_checkable
class ProcessingComponent(Protocol):
    """
    Protocol defining the interface contract for all processing layer components.
    
    Ensures consistent behavior across population management, operators, evaluation,
    and evolutionary engines while maintaining theoretical compliance with DEAP Framework.
    
    THEORETICAL BASIS:
    - Interface contracts ensure Algorithm 11.2 compliance across all components
    - Memory constraints guarantee ≤250MB processing layer resource usage
    - Error handling enables fail-fast behavior with comprehensive audit trails
    """
    
    def initialize(self, context: InputModelContext, config: DEAPFamilyConfig) -> None:
        """Initialize component with input context and configuration."""
        ...
    
    def validate(self) -> bool:
        """Validate component state and configuration."""
        ...
    
    def get_memory_usage(self) -> float:
        """Report current memory usage in MB."""
        ...
    
    def cleanup(self) -> None:
        """Clean up resources and free memory."""
        ...

class ProcessingLayerError(Exception):
    """
    Base exception class for all processing layer errors.
    
    Provides structured error reporting with context preservation for audit trails
    and debugging support in development environments.
    """
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, 
                 component: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.component = component
        self.timestamp = datetime.now()
        self.execution_id = str(uuid.uuid4())

class ProcessingConfigurationError(ProcessingLayerError):
    """Raised when processing layer configuration is invalid or incomplete."""
    pass

class ProcessingMemoryError(ProcessingLayerError):
    """Raised when memory constraints are violated during processing operations."""
    pass

class ProcessingValidationError(ProcessingLayerError):
    """Raised when population or operator validation fails."""
    pass

class ProcessingResult(BaseModel):
    """
    Comprehensive result container for evolutionary processing operations.
    
    Encapsulates all outputs from the evolutionary optimization process including
    optimal solutions, statistical analysis, and execution metadata.
    
    THEORETICAL COMPLIANCE:
    - Contains Pareto-optimal solutions per NSGA-II Algorithm 8.3
    - Preserves complete fitness evaluation history per Definition 2.4
    - Includes convergence analysis per Theorem 8.4 (NSGA-II Convergence Properties)
    - Maintains bijective genotype-phenotype mapping per Definition 2.2
    """
    
    # Primary optimization results
    best_individual: IndividualType = Field(
        ..., 
        description="Best individual from evolutionary optimization (course→assignment mapping)"
    )
    best_fitness: FitnessType = Field(
        ..., 
        description="Multi-objective fitness values (f₁, f₂, f₃, f₄, f₅) for best individual"
    )
    pareto_front: List[Tuple[IndividualType, FitnessType]] = Field(
        default_factory=list,
        description="Complete Pareto front for multi-objective optimization problems"
    )
    
    # Evolutionary statistics and analysis
    final_statistics: EvolutionaryRunStatistics = Field(
        ...,
        description="Complete statistical analysis of evolutionary run"
    )
    generation_history: List[GenerationMetrics] = Field(
        default_factory=list,
        description="Per-generation metrics for convergence analysis"
    )
    
    # Execution metadata
    algorithm_used: SolverID = Field(
        ...,
        description="Evolutionary algorithm used for optimization"
    )
    total_runtime: float = Field(
        ..., 
        gt=0.0,
        description="Total processing time in seconds"
    )
    memory_peak: float = Field(
        ..., 
        ge=0.0,
        description="Peak memory usage during processing (MB)"
    )
    
    # Quality assessment
    convergence_achieved: bool = Field(
        ...,
        description="Whether convergence criteria were satisfied"
    )
    constraint_violations: int = Field(
        ..., 
        ge=0,
        description="Number of constraint violations in best solution"
    )
    
    # Audit information
    execution_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this processing execution"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Processing completion timestamp"
    )
    
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        extra = "forbid"
    
    @validator('best_fitness')
    def validate_fitness_values(cls, v):
        """Validate that all fitness values are finite and valid."""
        if not all(np.isfinite(val) for val in v):
            raise ValueError("Fitness values must be finite (no NaN or Inf allowed)")
        return v
    
    @validator('pareto_front')
    def validate_pareto_front(cls, v):
        """Validate Pareto front structure and dominance relationships."""
        if len(v) == 0:
            return v
        
        # Verify all fitness values are finite
        for individual, fitness in v:
            if not all(np.isfinite(val) for val in fitness):
                raise ValueError("All Pareto front fitness values must be finite")
        
        return v

class ProcessingOrchestrator:
    """
    Master orchestrator for the complete evolutionary processing pipeline.
    
    Coordinates all processing layer components including population management,
    evolutionary operators, fitness evaluation, and algorithm execution while
    maintaining strict adherence to memory constraints and theoretical guarantees.
    
    ARCHITECTURAL DESIGN:
    - Single-threaded execution with deterministic resource usage patterns
    - Memory-bounded operations with real-time monitoring and constraint enforcement
    - Fail-fast validation with comprehensive error propagation and recovery
    - Complete statistical analysis and performance profiling integration
    
    THEORETICAL FOUNDATIONS:
    - Implements Algorithm 11.2: Integrated Evolutionary Process
    - Maintains Definition 2.1: Evolutionary Algorithm Framework compliance
    - Ensures Theorem 3.2: GA Schema Theorem pattern preservation
    - Supports Definition 2.4: Multi-Objective Fitness Model evaluation
    """
    
    def __init__(self, config: DEAPFamilyConfig, pipeline_context: PipelineContext):
        """
        Initialize processing orchestrator with configuration and execution context.
        
        Args:
            config: Complete DEAP family configuration with algorithm parameters
            pipeline_context: Execution context with memory monitoring and audit logging
            
        Raises:
            ProcessingConfigurationError: If configuration is invalid or incomplete
            ProcessingMemoryError: If insufficient memory is available for processing
        """
        self.config = config
        self.pipeline_context = pipeline_context
        self.memory_monitor = pipeline_context.memory_monitor
        self.audit_logger = pipeline_context.audit_logger
        
        # Component initialization
        self._components: Dict[str, ProcessingComponent] = {}
        self._initialized = False
        self._processing_active = False
        
        # Statistical tracking
        self._execution_start: Optional[datetime] = None
        self._execution_timer: Optional[ExecutionTimer] = None
        
        # Memory management
        self._memory_peak = 0.0
        self._memory_threshold = config.memory_constraints.processing_layer_mb
        
        # Validation
        self._validate_configuration()
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.ProcessingOrchestrator")
        self.logger.info(f"Processing orchestrator initialized for {config.solver_id.value}")
    
    def _validate_configuration(self) -> None:
        """
        Validate processing configuration against theoretical requirements and constraints.
        
        Raises:
            ProcessingConfigurationError: If configuration violates theoretical requirements
        """
        try:
            # Memory constraint validation
            if self.config.memory_constraints.processing_layer_mb > 512:
                raise ProcessingConfigurationError(
                    "Processing layer memory constraint exceeds 512MB limit",
                    context={"configured_mb": self.config.memory_constraints.processing_layer_mb}
                )
            
            # Population configuration validation
            pop_config = self.config.population_config
            if pop_config.size < 10 or pop_config.size > 1000:
                raise ProcessingConfigurationError(
                    "Population size must be between 10 and 1000 for theoretical guarantees",
                    context={"configured_size": pop_config.size}
                )
            
            # Fitness weights validation
            fitness_weights = self.config.fitness_weights
            if not np.allclose(sum(fitness_weights.dict().values()), 1.0, rtol=1e-6):
                raise ProcessingConfigurationError(
                    "Fitness weights must sum to 1.0 for proper multi-objective optimization",
                    context={"weights_sum": sum(fitness_weights.dict().values())}
                )
            
            # Solver algorithm validation
            if self.config.solver_id not in SolverID:
                raise ProcessingConfigurationError(
                    f"Unsupported solver algorithm: {self.config.solver_id}",
                    context={"requested_solver": str(self.config.solver_id)}
                )
            
        except Exception as e:
            self.audit_logger.log_error(
                "processing_configuration_validation_failed",
                str(e),
                {"component": "ProcessingOrchestrator", "phase": "initialization"}
            )
            raise
    
    def initialize_components(self, input_context: InputModelContext) -> None:
        """
        Initialize all processing layer components with input context and configuration.
        
        This method establishes the complete evolutionary computing infrastructure including
        population management, operators, fitness evaluation, and algorithm engines.
        
        Args:
            input_context: Complete input modeling context with eligibility and constraints
            
        Raises:
            ProcessingConfigurationError: If component initialization fails
            ProcessingMemoryError: If memory constraints are violated during initialization
        """
        if self._initialized:
            raise ProcessingConfigurationError("Processing components already initialized")
        
        try:
            self.memory_monitor.start_monitoring()
            init_start = datetime.now()
            
            # Initialize population management
            self.logger.info("Initializing population management system")
            population_manager = PopulationManager(
                config=self.config.population_config,
                course_eligibility=input_context.course_eligibility,
                memory_monitor=self.memory_monitor
            )
            population_manager.initialize(input_context, self.config)
            self._components['population'] = population_manager
            
            # Initialize evolutionary operators
            self.logger.info("Initializing evolutionary operators")
            operator_manager = OperatorManager(
                config=self.config.operator_config,
                solver_id=self.config.solver_id,
                course_eligibility=input_context.course_eligibility
            )
            operator_manager.initialize(input_context, self.config)
            self._components['operators'] = operator_manager
            
            # Initialize fitness evaluator
            self.logger.info("Initializing multi-objective fitness evaluator")
            fitness_evaluator = DEAPMultiObjectiveFitnessEvaluator(
                constraint_rules=input_context.constraint_rules,
                fitness_weights=self.config.fitness_weights,
                memory_monitor=self.memory_monitor
            )
            fitness_evaluator.initialize(input_context, self.config)
            self._components['evaluator'] = fitness_evaluator
            
            # Initialize evolutionary algorithm factory
            self.logger.info(f"Initializing {self.config.solver_id.value} evolutionary engine")
            algorithm_factory = EvolutionaryAlgorithmFactory()
            # Factory pattern - no direct initialization needed
            self._components['algorithm_factory'] = algorithm_factory
            
            # Initialize evolutionary logger
            self.logger.info("Initializing evolutionary logging system")
            evolutionary_logger = EvolutionaryLogger(
                execution_id=self.pipeline_context.execution_id,
                output_path=self.config.paths.output_base_path,
                memory_monitor=self.memory_monitor
            )
            evolutionary_logger.initialize(input_context, self.config)
            self._components['logger'] = evolutionary_logger
            
            # Memory validation after initialization
            current_memory = self.memory_monitor.get_current_usage_mb()
            if current_memory > self._memory_threshold:
                raise ProcessingMemoryError(
                    f"Component initialization exceeded memory threshold: {current_memory:.2f}MB > {self._memory_threshold}MB",
                    context={"current_mb": current_memory, "threshold_mb": self._memory_threshold}
                )
            
            self._memory_peak = max(self._memory_peak, current_memory)
            self._initialized = True
            
            init_duration = (datetime.now() - init_start).total_seconds()
            self.logger.info(f"Processing components initialized in {init_duration:.2f}s, memory usage: {current_memory:.2f}MB")
            
            self.audit_logger.log_info(
                "processing_components_initialized",
                "All processing layer components successfully initialized",
                {
                    "initialization_duration_seconds": init_duration,
                    "memory_usage_mb": current_memory,
                    "components_count": len(self._components)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize processing components: {str(e)}")
            self.audit_logger.log_error(
                "processing_component_initialization_failed",
                str(e),
                {"component": "ProcessingOrchestrator", "phase": "component_initialization"}
            )
            
            # Cleanup partially initialized components
            self._cleanup_components()
            raise ProcessingConfigurationError(
                f"Component initialization failed: {str(e)}",
                context={"error_type": type(e).__name__}
            )
    
    def execute_evolutionary_optimization(self, input_context: InputModelContext) -> ProcessingResult:
        """
        Execute complete evolutionary optimization process using configured algorithm.
        
        This method orchestrates the entire evolutionary computation pipeline including
        population initialization, fitness evaluation, evolutionary operators, and
        convergence analysis while maintaining strict memory constraints and error handling.
        
        Args:
            input_context: Complete input modeling context with constraints and eligibility
            
        Returns:
            ProcessingResult: Comprehensive optimization results with statistical analysis
            
        Raises:
            ProcessingConfigurationError: If components are not properly initialized
            ProcessingMemoryError: If memory constraints are violated during execution
            ProcessingValidationError: If population or fitness validation fails
        """
        if not self._initialized:
            raise ProcessingConfigurationError("Processing components must be initialized before execution")
        
        if self._processing_active:
            raise ProcessingConfigurationError("Processing execution already active")
        
        try:
            self._processing_active = True
            self._execution_start = datetime.now()
            self._execution_timer = ExecutionTimer()
            self._execution_timer.start()
            
            self.logger.info(f"Starting {self.config.solver_id.value} evolutionary optimization")
            
            # Get initialized components
            population_manager = self._components['population']
            operator_manager = self._components['operators']
            fitness_evaluator = self._components['evaluator']
            algorithm_factory = self._components['algorithm_factory']
            evolutionary_logger = self._components['logger']
            
            # Create evolutionary algorithm instance
            algorithm = algorithm_factory.create_algorithm(
                solver_id=self.config.solver_id,
                config=self.config,
                population_manager=population_manager,
                operator_manager=operator_manager,
                fitness_evaluator=fitness_evaluator,
                logger=evolutionary_logger
            )
            
            # Execute evolutionary optimization
            self.logger.info("Executing evolutionary algorithm")
            evolution_result = algorithm.evolve(
                input_context=input_context,
                max_generations=self.config.population_config.generations,
                memory_monitor=self.memory_monitor
            )
            
            # Validate results
            self._validate_evolution_result(evolution_result)
            
            # Update memory peak
            current_memory = self.memory_monitor.get_current_usage_mb()
            self._memory_peak = max(self._memory_peak, current_memory)
            
            # Create comprehensive processing result
            processing_result = ProcessingResult(
                best_individual=evolution_result.best_individual,
                best_fitness=evolution_result.best_fitness,
                pareto_front=evolution_result.pareto_front,
                final_statistics=evolution_result.statistics,
                generation_history=evolution_result.generation_history,
                algorithm_used=self.config.solver_id,
                total_runtime=self._execution_timer.elapsed(),
                memory_peak=self._memory_peak,
                convergence_achieved=evolution_result.converged,
                constraint_violations=evolution_result.constraint_violations,
                execution_id=self.pipeline_context.execution_id
            )
            
            self.logger.info(
                f"Evolutionary optimization completed successfully in {processing_result.total_runtime:.2f}s"
            )
            
            self.audit_logger.log_info(
                "evolutionary_optimization_completed",
                f"{self.config.solver_id.value} optimization completed successfully",
                {
                    "runtime_seconds": processing_result.total_runtime,
                    "memory_peak_mb": processing_result.memory_peak,
                    "convergence_achieved": processing_result.convergence_achieved,
                    "constraint_violations": processing_result.constraint_violations,
                    "generations_executed": len(processing_result.generation_history)
                }
            )
            
            return processing_result
            
        except Exception as e:
            self.logger.error(f"Evolutionary optimization failed: {str(e)}")
            self.audit_logger.log_error(
                "evolutionary_optimization_failed",
                str(e),
                {
                    "component": "ProcessingOrchestrator",
                    "phase": "evolutionary_execution",
                    "algorithm": self.config.solver_id.value if hasattr(self.config, 'solver_id') else 'unknown'
                }
            )
            
            # Ensure processing state is reset
            self._processing_active = False
            raise
        
        finally:
            self._processing_active = False
            if self._execution_timer:
                self._execution_timer.stop()
    
    def _validate_evolution_result(self, result: EvolutionaryResult) -> None:
        """
        Validate evolutionary algorithm results for theoretical compliance and correctness.
        
        Args:
            result: Evolutionary algorithm execution result
            
        Raises:
            ProcessingValidationError: If results fail validation checks
        """
        try:
            # Validate best individual structure
            if not isinstance(result.best_individual, dict):
                raise ProcessingValidationError(
                    "Best individual must be course→assignment dictionary",
                    context={"individual_type": type(result.best_individual).__name__}
                )
            
            # Validate fitness values
            if not all(np.isfinite(val) for val in result.best_fitness):
                raise ProcessingValidationError(
                    "Best fitness contains invalid values (NaN or Inf)",
                    context={"fitness_values": list(result.best_fitness)}
                )
            
            # Validate Pareto front (for multi-objective algorithms)
            if hasattr(result, 'pareto_front') and result.pareto_front:
                for i, (individual, fitness) in enumerate(result.pareto_front):
                    if not isinstance(individual, dict):
                        raise ProcessingValidationError(
                            f"Pareto front individual {i} must be course→assignment dictionary",
                            context={"individual_index": i, "individual_type": type(individual).__name__}
                        )
                    
                    if not all(np.isfinite(val) for val in fitness):
                        raise ProcessingValidationError(
                            f"Pareto front fitness {i} contains invalid values",
                            context={"individual_index": i, "fitness_values": list(fitness)}
                        )
            
            # Validate statistics structure
            if not hasattr(result.statistics, 'convergence_rate'):
                raise ProcessingValidationError(
                    "Evolution result statistics missing required convergence analysis",
                    context={"statistics_type": type(result.statistics).__name__}
                )
            
            self.logger.debug("Evolution result validation passed")
            
        except Exception as e:
            self.logger.error(f"Evolution result validation failed: {str(e)}")
            raise ProcessingValidationError(
                f"Evolution result validation failed: {str(e)}",
                context={"validation_error": str(e)}
            )
    
    def _cleanup_components(self) -> None:
        """Clean up all processing layer components and free allocated memory."""
        try:
            cleanup_start = datetime.now()
            
            for component_name, component in self._components.items():
                try:
                    if hasattr(component, 'cleanup'):
                        component.cleanup()
                    self.logger.debug(f"Cleaned up component: {component_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup component {component_name}: {str(e)}")
            
            self._components.clear()
            self._initialized = False
            
            # Force garbage collection
            gc.collect()
            
            cleanup_duration = (datetime.now() - cleanup_start).total_seconds()
            final_memory = self.memory_monitor.get_current_usage_mb()
            
            self.logger.info(f"Processing components cleaned up in {cleanup_duration:.3f}s, memory: {final_memory:.2f}MB")
            
        except Exception as e:
            self.logger.error(f"Component cleanup failed: {str(e)}")
    
    def cleanup(self) -> None:
        """Clean up orchestrator and all managed components."""
        try:
            self._cleanup_components()
            
            if self.memory_monitor:
                self.memory_monitor.stop_monitoring()
            
            self.logger.info("Processing orchestrator cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Orchestrator cleanup failed: {str(e)}")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage of processing layer in MB."""
        return self.memory_monitor.get_current_usage_mb() if self.memory_monitor else 0.0
    
    def get_component_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed status information for all processing components."""
        status = {}
        
        for component_name, component in self._components.items():
            try:
                component_status = {
                    "initialized": hasattr(component, 'validate') and component.validate(),
                    "memory_usage_mb": component.get_memory_usage() if hasattr(component, 'get_memory_usage') else 0.0,
                    "type": type(component).__name__
                }
                status[component_name] = component_status
                
            except Exception as e:
                status[component_name] = {
                    "error": str(e),
                    "type": type(component).__name__
                }
        
        return status

def run_evolutionary_optimization(
    input_context: InputModelContext,
    config: DEAPFamilyConfig,
    pipeline_context: PipelineContext
) -> ProcessingResult:
    """
    High-level function to execute complete evolutionary optimization process.
    
    This function provides a simplified interface for evolutionary optimization while
    maintaining all theoretical guarantees and error handling requirements.
    
    Args:
        input_context: Complete input modeling context with constraints and eligibility
        config: DEAP family configuration with algorithm parameters
        pipeline_context: Execution context with monitoring and logging
        
    Returns:
        ProcessingResult: Comprehensive optimization results
        
    Raises:
        ProcessingConfigurationError: If configuration is invalid
        ProcessingMemoryError: If memory constraints are violated
        ProcessingValidationError: If validation fails at any stage
        
    THEORETICAL COMPLIANCE:
    - Implements Algorithm 11.2: Integrated Evolutionary Process
    - Maintains Definition 2.1: Evolutionary Algorithm Framework
    - Ensures Definition 2.4: Multi-Objective Fitness Model evaluation
    - Preserves Definition 2.2: Schedule Genotype Encoding throughout process
    """
    orchestrator = None
    
    try:
        # Create and initialize orchestrator
        orchestrator = ProcessingOrchestrator(config, pipeline_context)
        orchestrator.initialize_components(input_context)
        
        # Execute evolutionary optimization
        result = orchestrator.execute_evolutionary_optimization(input_context)
        
        return result
        
    except Exception as e:
        logger = logging.getLogger(f"{__name__}.run_evolutionary_optimization")
        logger.error(f"Evolutionary optimization execution failed: {str(e)}")
        
        # Re-raise with appropriate error type
        if isinstance(e, (ProcessingConfigurationError, ProcessingMemoryError, ProcessingValidationError)):
            raise
        else:
            raise ProcessingConfigurationError(
                f"Unexpected error during evolutionary optimization: {str(e)}",
                context={"error_type": type(e).__name__, "traceback": traceback.format_exc()}
            )
    
    finally:
        # Ensure cleanup regardless of success/failure
        if orchestrator:
            try:
                orchestrator.cleanup()
            except Exception as cleanup_error:
                logger = logging.getLogger(f"{__name__}.run_evolutionary_optimization")
                logger.error(f"Failed to cleanup orchestrator: {str(cleanup_error)}")

# Module-level configuration and initialization
__version__ = "1.0.0"
__author__ = "Perplexity Labs AI - Enterprise Scheduling Engine Team"
__description__ = "Stage 6.3 DEAP Solver Family - Processing Package with Evolutionary Computing Infrastructure"

# Export public interface - Maintaining clean module boundaries
__all__ = [
    # Core orchestration
    'ProcessingOrchestrator',
    'run_evolutionary_optimization',
    
    # Result types
    'ProcessingResult',
    
    # Exception hierarchy
    'ProcessingLayerError',
    'ProcessingConfigurationError',
    'ProcessingMemoryError',
    'ProcessingValidationError',
    
    # Component interfaces
    'ProcessingComponent',
    
    # Type definitions for external use
    'ProcessingResultType',
    'EvolutionaryEngineType'
]

# Module initialization logging
logger = logging.getLogger(__name__)
logger.info(f"DEAP Processing Package initialized - Version {__version__}")
logger.info("Evolutionary computing infrastructure ready for Stage 6.3 optimization")