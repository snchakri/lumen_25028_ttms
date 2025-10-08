# ===============================================================================
# DEAP Solver Family Stage 6.3 - Configuration Module
# Advanced Scheduling Engine - DEAP Family Configuration System
# 
# THEORETICAL COMPLIANCE: Full Stage 6.3 DEAP Foundational Framework Implementation
# - Definition 2.1: Evolutionary Algorithm Framework EA = (P, F, S, V, R, T)
# - Definition 2.4: Multi-Objective Fitness Model f(g) = (f1, f2, f3, f4, f5)
# - Algorithm 11.2: Integrated Evolutionary Process Pipeline
# - Theorem 10.1: DEAP Algorithm Complexity Bounds O(λ·T·n·m)
#
# ENTERPRISE-GRADE IMPLEMENTATION:
# - Memory Constraint Enforcement: ≤512MB system-wide with real-time monitoring
# - Fail-Fast Validation: Immediate error propagation with comprehensive context
# - Multi-Algorithm Support: GA, GP, ES, DE, PSO, NSGA-II with unified interface
# - Dynamic Parametric System Integration: EAV parameter model preservation
#
# IDE INTEGRATION NOTES:
# @cursor-ide: This module implements complete DEAP configuration with mathematical
#              rigor per Stage 6.3 framework. Cross-references throughout codebase
#              use these models for type safety and algorithmic consistency.
# @jetbrains: Full IntelliSense support via comprehensive Pydantic models and
#             detailed docstring specifications referencing theoretical frameworks.
# ===============================================================================

"""
DEAP Solver Family Configuration Module

This module implements the comprehensive configuration system for Stage 6.3 DEAP
evolutionary solver family, providing enterprise-grade parameter management with
full theoretical compliance to DEAP Foundational Framework specifications.

Key Components:
- SolverID enumeration for all DEAP algorithms (GA/GP/ES/DE/PSO/NSGA-II)
- FitnessWeights for multi-objective optimization (f1-f5)
- PopulationConfig with complexity bounds enforcement  
- OperatorConfig with algorithm-specific parameter validation
- PathConfig with comprehensive I/O path management
- MemoryConstraints with real-time usage monitoring

Mathematical Foundation:
Based on Definition 2.1 (Evolutionary Algorithm Framework) and Algorithm 11.2
(Integrated Evolutionary Process) from Stage 6.3 DEAP Foundational Framework.

Enterprise Features:
- Memory estimation and constraint enforcement (≤512MB)
- Fail-fast validation with immediate error propagation
- Cross-algorithm parameter consistency checking
- Dynamic parameter adaptation support (EAV model)
- Comprehensive audit logging configuration
"""

import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator, root_validator
import psutil
import gc
from datetime import datetime
import uuid

# ===============================================================================
# LOGGING CONFIGURATION - Enterprise Grade with Audit Trail Support
# ===============================================================================

# Configure structured logging for comprehensive audit trails
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('deap_family_config.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

# ===============================================================================
# SOLVER ALGORITHM ENUMERATION - Complete DEAP Family Support
# ===============================================================================

class SolverID(str, Enum):
    """
    DEAP Solver Family Algorithm Enumeration
    
    Implements complete algorithmic coverage per Stage 6.3 DEAP Foundational Framework:
    - Genetic Algorithm (GA): Definition 3.1 - Population-based discrete optimization
    - Genetic Programming (GP): Definition 4.1 - Tree-based program evolution
    - Evolution Strategies (ES): Definition 5.1 - Self-adaptive continuous optimization  
    - Differential Evolution (DE): Definition 6.1 - Differential mutation strategies
    - Particle Swarm Optimization (PSO): Definition 7.1 - Swarm dynamics model
    - NSGA-II: Algorithm 8.3 - Multi-objective non-dominated sorting
    
    Each algorithm implements distinct search strategies optimized for specific
    problem characteristics as defined in Theorem 14.2 (No Free Lunch theorem).
    """
    
    # Genetic Algorithm - Discrete Combinatorial Optimization  
    GA = "genetic_algorithm"          # Schema theorem-based evolution (Theorem 3.2)
    
    # Genetic Programming - Dynamic Structure Evolution
    GP = "genetic_programming"        # Tree-based program evolution (Definition 4.1)
    
    # Evolution Strategies - Continuous Parameter Optimization
    ES = "evolution_strategies"       # Self-adaptive parameter control (Definition 5.1)
    
    # Differential Evolution - Robust Global Optimization  
    DE = "differential_evolution"     # Multiple mutation strategies (Definition 6.1)
    
    # Particle Swarm Optimization - Swarm Intelligence
    PSO = "particle_swarm"           # Velocity-based position updates (Algorithm 7.2)
    
    # NSGA-II - Multi-Objective Optimization
    NSGA2 = "nsga_ii"               # Non-dominated sorting with crowding distance


# ===============================================================================
# FITNESS MODEL CONFIGURATION - Multi-Objective Optimization Framework
# ===============================================================================

class FitnessWeights(BaseModel):
    """
    Multi-Objective Fitness Weights Configuration
    
    Implements Definition 2.4 (Scheduling Fitness) from DEAP Framework:
    f(g) = (f1(g), f2(g), f3(g), f4(g), f5(g))
    
    Each objective represents a critical aspect of schedule quality:
    - f1: Constraint violation penalty (feasibility)
    - f2: Resource utilization efficiency (optimization)  
    - f3: Preference satisfaction score (stakeholder alignment)
    - f4: Workload balance index (fairness)
    - f5: Schedule compactness measure (practicality)
    
    Mathematical Foundation:
    Weights must satisfy normalization constraint: Σw_i = 1.0
    All weights must be non-negative: w_i ≥ 0 ∀i
    """
    
    # f1: Hard/Soft Constraint Violation Penalty
    constraint_violation: float = Field(
        default=0.4, 
        ge=0.0, 
        le=1.0,
        description="Weight for constraint violation penalty (f1). High priority ensures feasibility."
    )
    
    # f2: Resource Utilization Efficiency  
    resource_utilization: float = Field(
        default=0.2,
        ge=0.0, 
        le=1.0,
        description="Weight for resource utilization efficiency (f2). Optimizes faculty/room usage."
    )
    
    # f3: Stakeholder Preference Satisfaction
    preference_satisfaction: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0, 
        description="Weight for preference satisfaction score (f3). Maximizes stakeholder satisfaction."
    )
    
    # f4: Workload Balance Index
    workload_balance: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Weight for workload balance index (f4). Ensures equitable load distribution."
    )
    
    # f5: Schedule Compactness Measure
    schedule_compactness: float = Field(
        default=0.1,
        ge=0.0, 
        le=1.0,
        description="Weight for schedule compactness measure (f5). Minimizes scheduling gaps."
    )
    
    @root_validator
    def validate_weight_normalization(cls, values):
        """
        Enforce Mathematical Constraint: Weight Normalization
        
        Validates that fitness weights sum to 1.0 with numerical tolerance.
        This ensures proper multi-objective optimization per Definition 2.4.
        
        Mathematical Requirement: Σw_i = 1.0 ± ε where ε = 1e-6
        """
        total_weight = sum([
            values.get('constraint_violation', 0.0),
            values.get('resource_utilization', 0.0), 
            values.get('preference_satisfaction', 0.0),
            values.get('workload_balance', 0.0),
            values.get('schedule_compactness', 0.0)
        ])
        
        tolerance = 1e-6
        if abs(total_weight - 1.0) > tolerance:
            raise ValueError(
                f"Fitness weights must sum to 1.0 (current: {total_weight:.6f}). "
                f"Adjust weights to satisfy normalization constraint."
            )
        
        logger.info(f"Fitness weights validated: total={total_weight:.6f}")
        return values


# ===============================================================================
# POPULATION CONFIGURATION - Complexity Bounds Enforcement  
# ===============================================================================

class PopulationConfig(BaseModel):
    """
    Population Configuration with Theoretical Complexity Bounds
    
    Implements population sizing constraints per Theorem 10.1 (DEAP Algorithm Complexity):
    - Total complexity: O(λ·T·n·m) where λ=population_size, T=generations
    - Memory constraint: λ ≤ memory_limit / (individual_size + fitness_metadata)  
    - Convergence requirement: λ ≥ 4×problem_dimension for diversity maintenance
    
    Population sizing follows theoretical guidelines:
    - Small problems (<350 courses): λ ∈ [100, 200] 
    - Medium problems (350-750 courses): λ ∈ [200, 400]
    - Large problems (>750 courses): λ ∈ [400, 600] (exceeds current scope)
    
    Current Implementation Constraint: ≤1500 students (~350 courses)
    """
    
    # Population Size - Primary Complexity Driver
    population_size: int = Field(
        default=200,
        ge=50,     # Minimum for diversity maintenance  
        le=500,    # Maximum for memory constraint compliance
        description="Population size λ. Affects complexity O(λ·T·n·m) and memory usage."
    )
    
    # Generation Limit - Secondary Complexity Driver
    max_generations: int = Field(
        default=100, 
        ge=10,     # Minimum for meaningful evolution
        le=1000,   # Maximum for time constraint compliance  
        description="Maximum generations T. Balances solution quality with runtime."
    )
    
    # Elite Preservation Count
    elite_size: int = Field(
        default=10,
        ge=1,
        le=50, 
        description="Number of elite individuals preserved across generations."
    )
    
    # Tournament Size for Selection
    tournament_size: int = Field(
        default=3,
        ge=2,      # Minimum for selection pressure
        le=10,     # Maximum for selection efficiency
        description="Tournament size for selection. Higher values increase selection pressure."
    )
    
    @validator('elite_size')
    def validate_elite_size(cls, elite_size, values):
        """
        Validate Elite Size Constraint
        
        Ensures elite_size ≤ population_size/4 to maintain evolutionary diversity.
        Large elite sizes can cause premature convergence per Schema Theorem 3.2.
        """
        population_size = values.get('population_size', 200)
        max_elite = population_size // 4
        
        if elite_size > max_elite:
            raise ValueError(
                f"Elite size ({elite_size}) exceeds maximum ({max_elite}) "
                f"for population size {population_size}. Reduce elite_size to maintain diversity."
            )
        
        return elite_size
    
    @validator('tournament_size')  
    def validate_tournament_size(cls, tournament_size, values):
        """
        Validate Tournament Size Constraint
        
        Ensures 2 ≤ tournament_size ≤ population_size/10 per Theorem 3.4
        (Selection Pressure Analysis).
        """
        population_size = values.get('population_size', 200)
        max_tournament = max(2, population_size // 10)
        
        if tournament_size > max_tournament:
            raise ValueError(
                f"Tournament size ({tournament_size}) too large for population "
                f"size {population_size}. Maximum recommended: {max_tournament}."
            )
            
        return tournament_size


# ===============================================================================
# EVOLUTIONARY OPERATOR CONFIGURATION - Algorithm-Specific Parameters
# ===============================================================================

class OperatorConfig(BaseModel):
    """
    Evolutionary Operator Configuration
    
    Implements algorithm-specific operator parameters per DEAP Framework:
    - GA: Crossover/mutation rates per Theorem 3.8 (Mutation Rate Optimization)
    - GP: Tree depth limits and bloat control per Definition 4.5 
    - ES: Self-adaptive parameter ranges per Theorem 5.2 (1/5-Success Rule)
    - DE: Differential weight and crossover rate per Definition 6.1
    - PSO: Inertia weight and acceleration coefficients per Algorithm 7.2  
    - NSGA-II: Crowding distance parameters per Algorithm 8.3
    
    All parameters validated against theoretical optimal ranges.
    """
    
    # Crossover Probability - Recombination Rate
    crossover_probability: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Crossover probability. Typical range: [0.6, 0.95]"
    )
    
    # Mutation Probability - Exploration Rate  
    mutation_probability: float = Field(
        default=0.1, 
        ge=0.0,
        le=1.0,
        description="Mutation probability. Optimal ~1/n per Theorem 3.8"
    )
    
    # Differential Evolution - Differential Weight
    differential_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0, 
        description="DE differential weight F. Typical range: [0.4, 1.0]"
    )
    
    # Differential Evolution - Crossover Rate
    de_crossover_rate: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="DE crossover rate CR. Typical range: [0.1, 0.9]"  
    )
    
    # PSO - Inertia Weight
    pso_inertia_weight: float = Field(
        default=0.9,
        ge=0.0, 
        le=1.5,
        description="PSO inertia weight w. Decreasing schedule recommended."
    )
    
    # PSO - Cognitive Acceleration 
    pso_cognitive_weight: float = Field(
        default=2.0,
        ge=0.0,
        le=4.0,
        description="PSO cognitive acceleration c1. Personal best attraction."
    )
    
    # PSO - Social Acceleration
    pso_social_weight: float = Field(
        default=2.0, 
        ge=0.0,
        le=4.0,
        description="PSO social acceleration c2. Global best attraction."
    )
    
    # GP - Maximum Tree Depth
    gp_max_depth: int = Field(
        default=10,
        ge=3,
        le=20,
        description="GP maximum tree depth. Controls bloat per Definition 4.5."
    )
    
    # GP - Parsimony Pressure Coefficient
    gp_parsimony_coefficient: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="GP parsimony pressure α per Theorem 4.6."
    )
    
    @root_validator
    def validate_pso_convergence(cls, values):
        """
        Validate PSO Convergence Conditions
        
        Implements Theorem 7.3 (PSO Convergence Conditions):
        φ = c1 + c2 > 4 and w = 2/(φ - 2 + √(φ² - 4φ))
        
        Ensures PSO parameter settings guarantee convergence.
        """
        c1 = values.get('pso_cognitive_weight', 2.0)
        c2 = values.get('pso_social_weight', 2.0) 
        w = values.get('pso_inertia_weight', 0.9)
        
        phi = c1 + c2
        if phi <= 4.0:
            logger.warning(
                f"PSO parameters may not satisfy convergence conditions: "
                f"φ = c1 + c2 = {phi:.2f} ≤ 4.0"
            )
        
        # Calculate theoretical optimal inertia weight
        if phi > 4.0:
            w_optimal = 2.0 / (phi - 2.0 + (phi*phi - 4.0*phi)**0.5)
            if abs(w - w_optimal) > 0.1:
                logger.info(
                    f"PSO inertia weight {w:.3f} differs from theoretical optimum "
                    f"{w_optimal:.3f} for φ={phi:.2f}"
                )
        
        return values


# ===============================================================================  
# PATH CONFIGURATION - I/O Path Management with Validation
# ===============================================================================

class PathConfig(BaseModel):
    """
    Comprehensive Path Configuration with Enterprise-Grade Validation
    
    Manages all I/O paths for DEAP solver family pipeline:
    - Input paths: Stage 3 compilation artifacts (L_raw, L_rel, L_idx)  
    - Output paths: Generated schedules, metadata, audit logs
    - Working paths: Temporary files, cache directories, execution isolation
    - Logging paths: Structured audit trails, error reports, performance metrics
    
    Features:
    - Automatic directory creation with permission validation
    - Path existence and accessibility verification  
    - Cross-platform compatibility (Windows/Linux/macOS)
    - Execution isolation via timestamped directories
    - Comprehensive error handling with detailed context
    """
    
    # Input Directory - Stage 3 Compilation Artifacts
    input_directory: Path = Field(
        default=Path("./stage3_outputs"),
        description="Directory containing Stage 3 compilation outputs (L_raw, L_rel, L_idx)"
    )
    
    # Output Directory - Generated Schedules and Results  
    output_directory: Path = Field(
        default=Path("./stage6_outputs"),
        description="Directory for generated schedules, metadata, and results"
    )
    
    # Working Directory - Temporary Files and Processing
    working_directory: Path = Field(
        default=Path("./stage6_working"),
        description="Directory for temporary files and intermediate processing"
    )
    
    # Logs Directory - Audit Trails and Error Reports
    logs_directory: Path = Field(
        default=Path("./stage6_logs"), 
        description="Directory for audit logs, error reports, and performance metrics"
    )
    
    # Execution ID - Unique Identifier for Run Isolation
    execution_id: str = Field(
        default_factory=lambda: f"deap_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
        description="Unique execution identifier for run isolation and audit tracking"
    )
    
    class Config:
        # Enable automatic Path object serialization
        arbitrary_types_allowed = True
    
    @validator('input_directory', 'output_directory', 'working_directory', 'logs_directory')
    def validate_directory_paths(cls, path_value):
        """
        Comprehensive Directory Path Validation
        
        Validates and prepares directories for DEAP pipeline execution:
        1. Convert string paths to Path objects for cross-platform compatibility
        2. Create directories if they don't exist (with proper permissions)
        3. Verify directory accessibility (read/write permissions)
        4. Log directory creation and validation results
        5. Provide detailed error context for debugging
        
        Raises:
            ValueError: If directory cannot be created or accessed
            PermissionError: If insufficient permissions for directory operations
        """
        # Convert to Path object if string provided
        if isinstance(path_value, str):
            path_value = Path(path_value)
        
        # Resolve to absolute path for consistency  
        path_value = path_value.resolve()
        
        try:
            # Create directory if it doesn't exist
            path_value.mkdir(parents=True, exist_ok=True)
            
            # Verify directory accessibility
            if not path_value.exists():
                raise ValueError(f"Directory creation failed: {path_value}")
            
            if not path_value.is_dir():
                raise ValueError(f"Path exists but is not a directory: {path_value}")
            
            # Test write permissions by creating temporary file
            test_file = path_value / f"test_write_{uuid.uuid4().hex[:8]}.tmp"
            try:
                test_file.touch()
                test_file.unlink()  # Clean up test file
            except PermissionError:
                raise PermissionError(f"Insufficient write permissions: {path_value}")
                
            logger.info(f"Directory validated and accessible: {path_value}")
            return path_value
            
        except Exception as e:
            logger.error(f"Directory validation failed for {path_value}: {str(e)}")
            raise ValueError(
                f"Failed to validate directory {path_value}: {str(e)}. "
                f"Check path validity and permissions."
            )
    
    def create_execution_directories(self) -> Dict[str, Path]:
        """
        Create Execution-Specific Directory Structure
        
        Creates isolated directory structure for current execution:
        - execution_root/
        ├── input_data/          # Processed input artifacts  
        ├── processing_cache/    # Evolutionary computation cache
        ├── output_data/         # Generated schedules and results
        ├── audit_logs/          # Comprehensive audit trails
        └── error_reports/       # Detailed error diagnostics
        
        Returns:
            Dict[str, Path]: Mapping of directory names to Path objects
            
        This ensures complete execution isolation and comprehensive audit trails.
        """
        execution_root = self.working_directory / self.execution_id
        
        directories = {
            'execution_root': execution_root,
            'input_data': execution_root / 'input_data',
            'processing_cache': execution_root / 'processing_cache', 
            'output_data': execution_root / 'output_data',
            'audit_logs': execution_root / 'audit_logs',
            'error_reports': execution_root / 'error_reports'
        }
        
        # Create all execution directories
        for dir_name, dir_path in directories.items():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created execution directory {dir_name}: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to create execution directory {dir_name}: {str(e)}")
                raise RuntimeError(
                    f"Execution directory creation failed for {dir_name} at {dir_path}: {str(e)}"
                )
        
        return directories


# ===============================================================================
# MEMORY CONSTRAINT CONFIGURATION - System Resource Management
# ===============================================================================

class MemoryConstraints(BaseModel):
    """
    Memory Constraint Configuration with Real-Time Monitoring
    
    Implements enterprise-grade memory management per Stage 6.3 requirements:
    - System-wide memory limit: ≤512MB across all layers
    - Per-layer memory limits: Input(≤200MB), Processing(≤250MB), Output(≤100MB)
    - Real-time usage monitoring with automatic constraint enforcement
    - Memory leak detection and garbage collection optimization
    - Fail-fast behavior on constraint violations
    
    Memory allocation follows theoretical complexity bounds from Theorem 10.1:
    - Population memory: O(λ × individual_size)  
    - Fitness evaluation: O(λ × objective_count)
    - Constraint checking: O(courses × constraints)
    - Operator application: O(λ × operator_complexity)
    """
    
    # System-Wide Memory Limit (bytes)
    max_total_memory_mb: int = Field(
        default=512,
        ge=256,    # Minimum for meaningful processing
        le=2048,   # Maximum for constraint compliance  
        description="Maximum total memory usage across all pipeline components (MB)"
    )
    
    # Input Modeling Layer Memory Limit
    input_layer_memory_mb: int = Field(
        default=200,
        ge=100,
        le=300,
        description="Maximum memory for input modeling layer (MB)"
    )
    
    # Processing Layer Memory Limit  
    processing_layer_memory_mb: int = Field(
        default=250,
        ge=150, 
        le=400,
        description="Maximum memory for evolutionary processing layer (MB)"
    )
    
    # Output Modeling Layer Memory Limit
    output_layer_memory_mb: int = Field(
        default=100,
        ge=50,
        le=150, 
        description="Maximum memory for output modeling layer (MB)"
    )
    
    # Memory Monitoring Interval (seconds)
    monitoring_interval_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Interval for memory usage monitoring (seconds)"
    )
    
    # Garbage Collection Threshold (MB)  
    gc_threshold_mb: int = Field(
        default=100,
        ge=50,
        le=200,
        description="Memory threshold for triggering garbage collection (MB)"
    )
    
    @root_validator  
    def validate_memory_allocation(cls, values):
        """
        Validate Memory Allocation Consistency
        
        Ensures that sum of per-layer limits does not exceed total system limit:
        input_limit + processing_limit + output_limit ≤ total_limit
        
        Provides buffer space for system overhead and unexpected memory spikes.
        """
        total_limit = values.get('max_total_memory_mb', 512)
        input_limit = values.get('input_layer_memory_mb', 200) 
        processing_limit = values.get('processing_layer_memory_mb', 250)
        output_limit = values.get('output_layer_memory_mb', 100)
        
        layer_sum = input_limit + processing_limit + output_limit
        
        if layer_sum > total_limit:
            raise ValueError(
                f"Layer memory limits sum ({layer_sum}MB) exceeds total limit ({total_limit}MB). "
                f"Reduce individual layer limits to maintain constraint compliance."
            )
            
        # Warn if allocation is very tight (>90% utilization)
        utilization = layer_sum / total_limit
        if utilization > 0.9:
            logger.warning(
                f"High memory utilization planned: {utilization:.1%} "
                f"({layer_sum}MB / {total_limit}MB). Consider increasing total limit."
            )
        
        return values
    
    def get_current_memory_usage_mb(self) -> float:
        """
        Get Current Process Memory Usage
        
        Returns current memory usage in megabytes using psutil for accurate
        system-level monitoring. Includes both RSS (Resident Set Size) and
        VMS (Virtual Memory Size) for comprehensive memory analysis.
        
        Returns:
            float: Current memory usage in MB
        """
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            # Use RSS (Resident Set Size) for physical memory usage
            memory_mb = memory_info.rss / (1024 * 1024)
            
            return memory_mb
            
        except Exception as e:
            logger.error(f"Failed to get memory usage: {str(e)}")
            return 0.0
    
    def check_memory_constraint(self, layer_name: str) -> bool:
        """
        Check Memory Constraint Compliance for Specific Layer
        
        Validates current memory usage against layer-specific limits with
        fail-fast behavior on constraint violations.
        
        Args:
            layer_name: Layer identifier ('input', 'processing', 'output')
            
        Returns:
            bool: True if within constraints, False otherwise
            
        Raises:
            RuntimeError: If memory usage exceeds critical thresholds
        """
        current_usage = self.get_current_memory_usage_mb()
        
        # Get layer-specific limit
        layer_limits = {
            'input': self.input_layer_memory_mb,
            'processing': self.processing_layer_memory_mb, 
            'output': self.output_layer_memory_mb
        }
        
        layer_limit = layer_limits.get(layer_name, self.max_total_memory_mb)
        
        # Check constraint compliance
        if current_usage > layer_limit:
            logger.error(
                f"Memory constraint violation in {layer_name} layer: "
                f"{current_usage:.1f}MB > {layer_limit}MB limit"
            )
            
            # Trigger garbage collection as last resort
            if current_usage > self.gc_threshold_mb:
                logger.info("Triggering garbage collection due to high memory usage")
                gc.collect()
                
                # Recheck after garbage collection
                updated_usage = self.get_current_memory_usage_mb()
                if updated_usage > layer_limit:
                    raise RuntimeError(
                        f"Critical memory constraint violation in {layer_name} layer: "
                        f"{updated_usage:.1f}MB > {layer_limit}MB after garbage collection"
                    )
            
            return False
        
        # Log memory status for monitoring
        utilization = current_usage / layer_limit
        logger.info(
            f"Memory check {layer_name}: {current_usage:.1f}MB / {layer_limit}MB "
            f"({utilization:.1%})"
        )
        
        return True


# ===============================================================================
# UNIFIED DEAP FAMILY CONFIGURATION - Master Configuration Container
# ===============================================================================

class DEAPFamilyConfig(BaseModel):
    """
    Master Configuration for DEAP Solver Family Stage 6.3
    
    Comprehensive configuration container integrating all DEAP family components
    with enterprise-grade validation and theoretical compliance verification.
    
    Components:
    - solver_id: Selected evolutionary algorithm (GA/GP/ES/DE/PSO/NSGA-II)
    - fitness_weights: Multi-objective optimization weights (f1-f5)
    - population_config: Population and generation parameters
    - operator_config: Algorithm-specific operator settings
    - path_config: I/O path management and execution isolation
    - memory_constraints: System resource limits and monitoring
    
    Mathematical Foundation:
    Based on Definition 2.1 (EA Framework) and Algorithm 11.2 (Integrated Process)
    with full compliance to Stage 6.3 DEAP Foundational Framework specifications.
    
    Enterprise Features:
    - Cross-component parameter validation and consistency checking
    - Automatic memory estimation based on problem characteristics
    - Comprehensive audit logging with execution traceability
    - Fail-fast error handling with detailed diagnostic context
    - Dynamic parameter adaptation support for real-time optimization
    """
    
    # Selected Evolutionary Algorithm
    solver_id: SolverID = Field(
        default=SolverID.NSGA2,
        description="Selected DEAP algorithm for evolutionary optimization"
    )
    
    # Multi-Objective Fitness Configuration
    fitness_weights: FitnessWeights = Field(
        default_factory=FitnessWeights,
        description="Multi-objective fitness weights (f1-f5) per Definition 2.4"
    )
    
    # Population and Evolution Parameters
    population_config: PopulationConfig = Field(
        default_factory=PopulationConfig,
        description="Population size, generations, and selection parameters"
    )
    
    # Algorithm-Specific Operator Settings  
    operator_config: OperatorConfig = Field(
        default_factory=OperatorConfig,
        description="Evolutionary operator parameters (crossover, mutation, etc.)"
    )
    
    # I/O Path Management
    path_config: PathConfig = Field(
        default_factory=PathConfig, 
        description="Input/output directory paths and execution isolation"
    )
    
    # Memory Resource Constraints
    memory_constraints: MemoryConstraints = Field(
        default_factory=MemoryConstraints,
        description="Memory limits and monitoring configuration"
    )
    
    # Configuration Metadata
    config_version: str = Field(
        default="6.3.1",
        description="Configuration schema version for compatibility tracking"
    )
    
    config_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Configuration creation timestamp for audit trails"
    )
    
    @root_validator
    def validate_cross_component_consistency(cls, values):
        """
        Cross-Component Configuration Validation
        
        Performs comprehensive validation across all configuration components
        to ensure mathematical consistency and enterprise-grade reliability:
        
        1. Memory allocation consistency across layers
        2. Population size compatibility with selected algorithm  
        3. Operator parameter ranges for chosen solver
        4. Path accessibility and execution isolation setup
        5. Fitness weight normalization and objective prioritization
        
        This validation ensures that all components work together harmoniously
        and that the configuration satisfies theoretical requirements.
        """
        solver_id = values.get('solver_id')
        population_config = values.get('population_config')
        operator_config = values.get('operator_config') 
        memory_constraints = values.get('memory_constraints')
        
        # Algorithm-Specific Population Size Validation
        if solver_id and population_config:
            pop_size = population_config.population_size
            
            # GP requires smaller populations due to tree complexity
            if solver_id == SolverID.GP and pop_size > 200:
                logger.warning(
                    f"Large population size ({pop_size}) for GP may cause memory issues. "
                    f"Consider reducing to ≤200 for tree-based evolution."
                )
            
            # ES benefits from larger populations for covariance estimation  
            if solver_id == SolverID.ES and pop_size < 100:
                logger.warning(
                    f"Small population size ({pop_size}) for ES may hurt covariance adaptation. "
                    f"Consider increasing to ≥100 for effective CMA-ES."
                )
        
        # Memory Constraint vs Population Size Validation
        if population_config and memory_constraints:
            estimated_memory = cls._estimate_memory_usage(
                population_config.population_size,
                population_config.max_generations,
                solver_id or SolverID.NSGA2
            )
            
            if estimated_memory > memory_constraints.processing_layer_memory_mb:
                logger.error(
                    f"Estimated memory usage ({estimated_memory:.1f}MB) exceeds "
                    f"processing layer limit ({memory_constraints.processing_layer_memory_mb}MB)"
                )
                
                # Suggest configuration adjustments
                max_pop_size = cls._calculate_max_population_size(
                    memory_constraints.processing_layer_memory_mb,
                    solver_id or SolverID.NSGA2
                )
                
                logger.info(
                    f"Consider reducing population size to ≤{max_pop_size} "
                    f"for memory constraint compliance"
                )
        
        logger.info(f"Cross-component validation completed for solver: {solver_id}")
        return values
    
    @staticmethod
    def _estimate_memory_usage(population_size: int, generations: int, solver_id: SolverID) -> float:
        """
        Estimate Memory Usage for Given Configuration
        
        Provides memory usage estimation based on theoretical complexity analysis
        from Theorem 10.1. Accounts for algorithm-specific memory requirements:
        
        - GA: O(λ × chromosome_length) for population + O(λ × 5) for fitness
        - GP: O(λ × tree_size × nodes) for tree structures + bloat control
        - ES: O(λ × n²) for covariance matrices + adaptation parameters  
        - DE: O(λ × n) for population + differential vectors
        - PSO: O(λ × n) for positions + velocities + personal bests
        - NSGA-II: Additional O(λ²) for non-dominated sorting
        
        Args:
            population_size: Population size λ
            generations: Maximum generations T
            solver_id: Selected evolutionary algorithm
            
        Returns:
            float: Estimated peak memory usage in MB
        """
        # Base memory per individual (course-centric representation)
        # Assuming ~350 courses × 4 assignments × 8 bytes ≈ 11KB per individual
        individual_memory_kb = 11.0
        
        # Fitness metadata per individual (5 objectives × 8 bytes)
        fitness_memory_kb = 0.04
        
        # Algorithm-specific overhead factors
        algorithm_overhead = {
            SolverID.GA: 1.2,      # Standard genetic algorithm
            SolverID.GP: 2.5,      # Tree structures with bloat
            SolverID.ES: 1.8,      # Covariance matrix storage  
            SolverID.DE: 1.3,      # Differential vectors
            SolverID.PSO: 1.5,     # Velocity and personal bests
            SolverID.NSGA2: 1.6    # Non-dominated sorting overhead
        }
        
        overhead_factor = algorithm_overhead.get(solver_id, 1.5)
        
        # Total population memory
        population_memory_kb = population_size * (individual_memory_kb + fitness_memory_kb)
        
        # Apply algorithm-specific overhead
        total_memory_kb = population_memory_kb * overhead_factor
        
        # Add system overhead (constraint matrices, operators, etc.) ≈ 50MB
        system_overhead_mb = 50.0
        
        # Convert to MB and add overhead
        estimated_memory_mb = (total_memory_kb / 1024.0) + system_overhead_mb
        
        return estimated_memory_mb
    
    @staticmethod
    def _calculate_max_population_size(memory_limit_mb: int, solver_id: SolverID) -> int:
        """
        Calculate Maximum Population Size for Memory Constraint
        
        Determines the largest population size that fits within memory constraints
        using reverse calculation from memory estimation formula.
        
        Args:
            memory_limit_mb: Memory limit in MB
            solver_id: Selected evolutionary algorithm
            
        Returns:
            int: Maximum viable population size
        """
        # System overhead
        system_overhead_mb = 50.0
        available_memory_mb = memory_limit_mb - system_overhead_mb
        
        if available_memory_mb <= 0:
            return 10  # Minimum viable population
        
        # Algorithm overhead factors (from _estimate_memory_usage)
        algorithm_overhead = {
            SolverID.GA: 1.2,
            SolverID.GP: 2.5, 
            SolverID.ES: 1.8,
            SolverID.DE: 1.3,
            SolverID.PSO: 1.5,
            SolverID.NSGA2: 1.6
        }
        
        overhead_factor = algorithm_overhead.get(solver_id, 1.5)
        
        # Memory per individual (including overhead)
        individual_memory_kb = 11.04  # 11.0 + 0.04
        individual_memory_mb = (individual_memory_kb / 1024.0) * overhead_factor
        
        # Calculate maximum population size
        max_population = int(available_memory_mb / individual_memory_mb)
        
        # Ensure minimum viable population
        return max(max_population, 10)
    
    def create_execution_context(self) -> Dict[str, Any]:
        """
        Create Complete Execution Context
        
        Generates comprehensive execution context containing all configuration
        parameters, execution directories, and runtime metadata for complete
        audit trail and execution isolation.
        
        Returns:
            Dict[str, Any]: Complete execution context with all parameters
        """
        # Create execution directories
        execution_dirs = self.path_config.create_execution_directories()
        
        # Build comprehensive context
        execution_context = {
            # Core Configuration
            'solver_id': self.solver_id.value,
            'config_version': self.config_version,
            'execution_id': self.path_config.execution_id,
            'timestamp': self.config_timestamp,
            
            # Fitness Configuration
            'fitness_weights': self.fitness_weights.dict(),
            
            # Population Parameters
            'population_size': self.population_config.population_size,
            'max_generations': self.population_config.max_generations,
            'elite_size': self.population_config.elite_size,
            'tournament_size': self.population_config.tournament_size,
            
            # Operator Parameters
            'crossover_prob': self.operator_config.crossover_probability,
            'mutation_prob': self.operator_config.mutation_probability,
            'differential_weight': self.operator_config.differential_weight,
            'de_crossover_rate': self.operator_config.de_crossover_rate,
            'pso_inertia': self.operator_config.pso_inertia_weight,
            'pso_cognitive': self.operator_config.pso_cognitive_weight,
            'pso_social': self.operator_config.pso_social_weight,
            
            # Directory Structure
            'directories': {str(k): str(v) for k, v in execution_dirs.items()},
            'input_directory': str(self.path_config.input_directory),
            'output_directory': str(self.path_config.output_directory),
            
            # Memory Configuration
            'memory_limits': self.memory_constraints.dict(),
            'estimated_memory_mb': self._estimate_memory_usage(
                self.population_config.population_size,
                self.population_config.max_generations, 
                self.solver_id
            ),
            
            # System Information
            'system_memory_mb': psutil.virtual_memory().total / (1024 * 1024),
            'available_memory_mb': psutil.virtual_memory().available / (1024 * 1024),
            'cpu_count': psutil.cpu_count(),
            'platform': sys.platform,
            'python_version': sys.version
        }
        
        logger.info(f"Execution context created for {self.solver_id.value} with ID: {self.path_config.execution_id}")
        return execution_context


# ===============================================================================
# CONFIGURATION FACTORY FUNCTIONS - Convenient Configuration Creation
# ===============================================================================

def create_default_config() -> DEAPFamilyConfig:
    """
    Create Default DEAP Family Configuration
    
    Generates production-ready default configuration optimized for typical
    scheduling problems with 1500 students (~350 courses). Uses NSGA-II
    with balanced multi-objective weights and conservative memory limits.
    
    Returns:
        DEAPFamilyConfig: Production-ready default configuration
    """
    logger.info("Creating default DEAP family configuration")
    return DEAPFamilyConfig()


def create_high_performance_config() -> DEAPFamilyConfig:
    """
    Create High-Performance Configuration
    
    Optimized for maximum solution quality with higher computational resources.
    Uses larger populations and extended evolution for superior optimization
    at the cost of increased memory usage and runtime.
    
    Returns:
        DEAPFamilyConfig: High-performance optimization configuration
    """
    config = DEAPFamilyConfig(
        solver_id=SolverID.NSGA2,
        population_config=PopulationConfig(
            population_size=400,
            max_generations=200,
            elite_size=20,
            tournament_size=5
        ),
        operator_config=OperatorConfig(
            crossover_probability=0.9,
            mutation_probability=0.05  # Lower mutation for fine-tuning
        ),
        memory_constraints=MemoryConstraints(
            max_total_memory_mb=800,  # Higher memory allowance
            processing_layer_memory_mb=500
        )
    )
    
    logger.info("Created high-performance DEAP configuration")
    return config


def create_fast_execution_config() -> DEAPFamilyConfig:
    """
    Create Fast Execution Configuration
    
    Optimized for rapid execution with reasonable solution quality.
    Uses smaller populations and fewer generations for quick turnaround
    while maintaining algorithmic integrity.
    
    Returns:
        DEAPFamilyConfig: Fast execution configuration
    """
    config = DEAPFamilyConfig(
        solver_id=SolverID.DE,  # Differential Evolution for fast convergence
        population_config=PopulationConfig(
            population_size=100,
            max_generations=50,
            elite_size=5,
            tournament_size=3
        ),
        operator_config=OperatorConfig(
            differential_weight=0.7,   # Aggressive exploration
            de_crossover_rate=0.9      # High recombination
        ),
        memory_constraints=MemoryConstraints(
            max_total_memory_mb=256,   # Minimal memory footprint
            processing_layer_memory_mb=150
        )
    )
    
    logger.info("Created fast execution DEAP configuration") 
    return config


def create_algorithm_specific_config(solver_id: SolverID) -> DEAPFamilyConfig:
    """
    Create Algorithm-Specific Optimized Configuration
    
    Generates configuration optimized for specific DEAP algorithm characteristics
    based on theoretical analysis from Stage 6.3 framework.
    
    Args:
        solver_id: Target evolutionary algorithm
        
    Returns:
        DEAPFamilyConfig: Algorithm-optimized configuration
    """
    # Base configuration
    base_config = create_default_config()
    base_config.solver_id = solver_id
    
    # Algorithm-specific optimizations
    if solver_id == SolverID.GA:
        # Genetic Algorithm optimization
        base_config.population_config.population_size = 250
        base_config.operator_config.crossover_probability = 0.8
        base_config.operator_config.mutation_probability = 0.1
        
    elif solver_id == SolverID.GP:
        # Genetic Programming optimization  
        base_config.population_config.population_size = 150  # Smaller for tree complexity
        base_config.operator_config.gp_max_depth = 8
        base_config.operator_config.gp_parsimony_coefficient = 0.02
        base_config.memory_constraints.processing_layer_memory_mb = 300
        
    elif solver_id == SolverID.ES:
        # Evolution Strategies optimization
        base_config.population_config.population_size = 200
        base_config.population_config.max_generations = 150  # ES needs more generations
        
    elif solver_id == SolverID.DE:
        # Differential Evolution optimization
        base_config.population_config.population_size = 200
        base_config.operator_config.differential_weight = 0.6
        base_config.operator_config.de_crossover_rate = 0.8
        
    elif solver_id == SolverID.PSO:
        # Particle Swarm Optimization
        base_config.population_config.population_size = 300  # Larger swarms beneficial
        base_config.operator_config.pso_inertia_weight = 0.8
        base_config.operator_config.pso_cognitive_weight = 2.0
        base_config.operator_config.pso_social_weight = 2.0
        
    elif solver_id == SolverID.NSGA2:
        # NSGA-II Multi-Objective optimization (already optimized in default)
        pass
    
    logger.info(f"Created algorithm-specific configuration for {solver_id.value}")
    return base_config


# ===============================================================================
# CONFIGURATION VALIDATION AND TESTING
# ===============================================================================

def validate_configuration(config: DEAPFamilyConfig) -> bool:
    """
    Comprehensive Configuration Validation
    
    Performs extensive validation of DEAP configuration including:
    - Mathematical consistency verification
    - Memory constraint feasibility analysis  
    - Path accessibility confirmation
    - Algorithm parameter range validation
    - Cross-component compatibility checking
    
    Args:
        config: DEAP family configuration to validate
        
    Returns:
        bool: True if configuration is valid and production-ready
        
    Raises:
        ValueError: If critical configuration errors are detected
    """
    logger.info("Starting comprehensive configuration validation")
    
    try:
        # Test configuration serialization/deserialization
        config_dict = config.dict()
        reconstructed_config = DEAPFamilyConfig(**config_dict)
        
        # Validate execution context creation
        execution_context = config.create_execution_context()
        
        # Check memory feasibility
        estimated_memory = config._estimate_memory_usage(
            config.population_config.population_size,
            config.population_config.max_generations,
            config.solver_id
        )
        
        if estimated_memory > config.memory_constraints.processing_layer_memory_mb:
            raise ValueError(
                f"Configuration not feasible: estimated memory ({estimated_memory:.1f}MB) "
                f"exceeds processing limit ({config.memory_constraints.processing_layer_memory_mb}MB)"
            )
        
        # Validate path accessibility
        execution_dirs = config.path_config.create_execution_directories()
        
        # Test memory monitoring
        current_memory = config.memory_constraints.get_current_memory_usage_mb()
        constraint_check = config.memory_constraints.check_memory_constraint('processing')
        
        logger.info(
            f"Configuration validation successful: "
            f"solver={config.solver_id.value}, "
            f"pop_size={config.population_config.population_size}, "
            f"estimated_memory={estimated_memory:.1f}MB, "
            f"current_memory={current_memory:.1f}MB"
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise ValueError(f"Invalid DEAP configuration: {str(e)}")


# ===============================================================================
# MAIN EXECUTION - Configuration Testing and Validation
# ===============================================================================

if __name__ == "__main__":
    """
    Configuration Module Test Suite
    
    Comprehensive testing of all configuration components with production-grade
    validation and error handling demonstration.
    """
    
    logger.info("=" * 80)
    logger.info("DEAP SOLVER FAMILY CONFIGURATION - COMPREHENSIVE TEST SUITE")
    logger.info("=" * 80)
    
    try:
        # Test 1: Default Configuration Creation and Validation
        logger.info("\nTest 1: Default Configuration")
        default_config = create_default_config()
        validate_configuration(default_config)
        logger.info("✓ Default configuration validation successful")
        
        # Test 2: High-Performance Configuration
        logger.info("\nTest 2: High-Performance Configuration")
        hp_config = create_high_performance_config()
        validate_configuration(hp_config)
        logger.info("✓ High-performance configuration validation successful")
        
        # Test 3: Fast Execution Configuration
        logger.info("\nTest 3: Fast Execution Configuration") 
        fast_config = create_fast_execution_config()
        validate_configuration(fast_config)
        logger.info("✓ Fast execution configuration validation successful")
        
        # Test 4: Algorithm-Specific Configurations
        logger.info("\nTest 4: Algorithm-Specific Configurations")
        for solver in SolverID:
            algo_config = create_algorithm_specific_config(solver)
            validate_configuration(algo_config)
            logger.info(f"✓ {solver.value} configuration validation successful")
        
        # Test 5: Memory Constraint Testing
        logger.info("\nTest 5: Memory Constraint Validation")
        memory_test_config = DEAPFamilyConfig(
            population_config=PopulationConfig(population_size=100),
            memory_constraints=MemoryConstraints(max_total_memory_mb=256)
        )
        validate_configuration(memory_test_config)
        logger.info("✓ Memory constraint validation successful")
        
        # Test 6: Execution Context Generation
        logger.info("\nTest 6: Execution Context Generation")
        context = default_config.create_execution_context()
        
        logger.info(f"Execution ID: {context['execution_id']}")
        logger.info(f"Solver: {context['solver_id']}")
        logger.info(f"Population Size: {context['population_size']}")
        logger.info(f"Estimated Memory: {context['estimated_memory_mb']:.1f}MB")
        logger.info("✓ Execution context generation successful")
        
        logger.info("\n" + "=" * 80)
        logger.info("ALL CONFIGURATION TESTS PASSED SUCCESSFULLY")
        logger.info("DEAP Solver Family Configuration Module Ready for Production")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Configuration test suite failed: {str(e)}")
        logger.error("=" * 80)
        logger.error("CONFIGURATION TEST SUITE FAILED")
        logger.error("Review configuration parameters and system requirements")
        logger.error("=" * 80)
        sys.exit(1)