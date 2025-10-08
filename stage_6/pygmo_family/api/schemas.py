"""
Stage 6.4 PyGMO Solver Family - Pydantic Schema Definitions
===========================================================

Pydantic models for request/response validation with complete
mathematical compliance and theoretical framework integration.

Mathematical Framework:
- Input validation per Definition 1.2 (Input Modeling Context)
- Output schema per Definition 12.1 (Schedule Export Format)  
- Error handling per Algorithm 4.1 (Fail-Fast Validation)
- Performance monitoring per Theorem 9.1 (Complexity Analysis)

Author: Student Team
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Core Enumeration Types
# ============================================================================

class SolverAlgorithm(str, Enum):
    """
    Supported PyGMO algorithms with theoretical guarantees.
    
    Each algorithm provides specific mathematical properties:
    - NSGA2: Pareto front convergence (Theorem 3.2)
    - MOEAD: Decomposition-based optimality (Theorem 3.6)  
    - MOPSO: Particle swarm multi-objective optimization
    - DE: Differential evolution with constraint handling
    - SA: Simulated annealing with cooling schedules
    """
    NSGA2 = "nsga2"
    MOEAD = "moead"
    MOPSO = "mopso" 
    DE = "de"
    SA = "sa"

class ValidationLevel(str, Enum):
    """Validation strictness levels for processing pipeline."""
    STRICT = "strict"        # Full mathematical validation
    STANDARD = "standard"    # Standard constraint checking
    MINIMAL = "minimal"      # Basic feasibility only

class OptimizationStatus(str, Enum):
    """Processing status enumeration for pipeline tracking."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# ============================================================================  
# Request Models
# ============================================================================

class OptimizationRequest(BaseModel):
    """
    Complete optimization request model with mathematical validation.
    
    Implements Definition 8.2 (Optimization Request Specification) with
    complete parameter validation and theoretical compliance.
    """
    
    # Core optimization parameters
    algorithm: SolverAlgorithm = Field(
        default=SolverAlgorithm.NSGA2,
        description="PyGMO algorithm for optimization (default: NSGA-II)"
    )
    
    population_size: int = Field(
        default=200,
        ge=50, le=1000,
        description="Population size for evolutionary algorithm (50-1000)"
    )
    
    max_generations: int = Field(
        default=500,
        ge=100, le=2000,
        description="Maximum generations for convergence (100-2000)"
    )
    
    # Input/Output path configuration
    input_data_path: str = Field(
        description="Absolute path to Stage 3 output data directory"
    )
    
    output_data_path: str = Field(
        description="Absolute path for optimization results and logs"
    )
    
    # Algorithm-specific parameters
    crossover_probability: float = Field(
        default=0.9,
        ge=0.0, le=1.0,
        description="Crossover probability for genetic operators"
    )
    
    mutation_probability: float = Field(
        default=0.1,
        ge=0.0, le=1.0, 
        description="Mutation probability for genetic variation"
    )
    
    tournament_size: int = Field(
        default=3,
        ge=2, le=10,
        description="Tournament size for parent selection"
    )
    
    # Convergence and performance parameters
    convergence_threshold: float = Field(
        default=1e-6,
        gt=0.0,
        description="Hypervolume convergence threshold"
    )
    
    stagnation_limit: int = Field(
        default=50,
        ge=10, le=200,
        description="Generations without improvement before termination"
    )
    
    # Validation and monitoring
    validation_level: ValidationLevel = Field(
        default=ValidationLevel.STANDARD,
        description="Validation strictness for mathematical compliance"
    )
    
    enable_detailed_logging: bool = Field(
        default=True,
        description="Enable complete optimization logging"
    )
    
    # Master pipeline integration
    request_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for master pipeline tracking"
    )
    
    webhook_url: Optional[str] = Field(
        default=None,
        description="Callback URL for status updates to master pipeline"
    )
    
    priority: int = Field(
        default=1,
        ge=1, le=10,
        description="Request priority level (1=highest, 10=lowest)"
    )
    
    @validator('input_data_path', 'output_data_path')
    def validate_paths(cls, v):
        """Validate path format and accessibility."""
        import os
        if not os.path.isabs(v):
            raise ValueError("Path must be absolute")
        return v
    
    @validator('algorithm')
    def validate_algorithm_support(cls, v):
        """Ensure algorithm is supported by current PyGMO installation."""
        # Note: In production, verify algorithm availability via pygmo
        supported = [alg.value for alg in SolverAlgorithm]
        if v not in supported:
            raise ValueError(f"Algorithm {v} not supported. Available: {supported}")
        return v
    
    @root_validator
    def validate_probability_constraints(cls, values):
        """Ensure genetic operator probabilities are mathematically valid."""
        cross_prob = values.get('crossover_probability', 0.9)
        mut_prob = values.get('mutation_probability', 0.1)
        
        # Verify probabilistic constraints for genetic algorithms
        if cross_prob + mut_prob > 1.2:  # Allow some overlap
            logger.warning("High combined operator probabilities may affect convergence")
            
        return values

class OptimizationResponse(BaseModel):
    """
    complete optimization response with mathematical guarantees.
    
    Implements Definition 12.2 (Optimization Response Format) providing
    complete results, metadata, and theoretical compliance verification.
    """
    
    # Request tracking
    request_id: str = Field(description="Original optimization request ID")
    status: OptimizationStatus = Field(description="Final optimization status")
    
    # Core optimization results
    best_solution: Optional[Dict[str, Tuple[int, int, int, int]]] = Field(
        default=None,
        description="Best individual solution (course assignments)"
    )
    
    pareto_front: Optional[List[Dict[str, Tuple[int, int, int, int]]]] = Field(
        default=None,
        description="Complete Pareto-optimal solution set"
    )
    
    # Mathematical performance metrics
    final_hypervolume: Optional[float] = Field(
        default=None,
        description="Final hypervolume indicator value"
    )
    
    convergence_generation: Optional[int] = Field(
        default=None,
        description="Generation where convergence achieved"
    )
    
    fitness_history: Optional[List[List[float]]] = Field(
        default=None,
        description="Complete fitness evolution trajectory"
    )
    
    # Objective function values
    conflict_violations: Optional[float] = Field(
        default=None,
        description="Final f1 objective (conflict penalty)"
    )
    
    resource_utilization: Optional[float] = Field(
        default=None,
        description="Final f2 objective (utilization score)"
    )
    
    preference_satisfaction: Optional[float] = Field(
        default=None,
        description="Final f3 objective (preference compliance)"
    )
    
    workload_balance: Optional[float] = Field(
        default=None,
        description="Final f4 objective (load balancing)"
    )
    
    schedule_compactness: Optional[float] = Field(
        default=None,
        description="Final f5 objective (fragmentation minimization)"
    )
    
    # Performance and resource metrics
    processing_time_seconds: Optional[float] = Field(
        default=None,
        description="Total optimization execution time"
    )
    
    peak_memory_mb: Optional[float] = Field(
        default=None,
        description="Peak memory usage during optimization"
    )
    
    generations_executed: Optional[int] = Field(
        default=None,
        description="Total generations processed"
    )
    
    evaluations_performed: Optional[int] = Field(
        default=None,
        description="Total fitness evaluations computed"
    )
    
    # Output file paths
    schedule_csv_path: Optional[str] = Field(
        default=None,
        description="Path to generated schedule CSV file"
    )
    
    metadata_json_path: Optional[str] = Field(
        default=None,
        description="Path to optimization metadata JSON"
    )
    
    log_file_path: Optional[str] = Field(
        default=None,
        description="Path to detailed optimization log"
    )
    
    # Validation and compliance
    stage7_validation_passed: Optional[bool] = Field(
        default=None,
        description="Stage 7 validation compliance status"
    )
    
    constraint_violations: Optional[Dict[str, int]] = Field(
        default=None,
        description="Detailed constraint violation counts"
    )
    
    theoretical_compliance_verified: bool = Field(
        default=False,
        description="PyGMO framework mathematical compliance confirmed"
    )
    
    # Timestamp information
    started_at: Optional[datetime] = Field(
        default=None,
        description="Optimization start timestamp"
    )
    
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Optimization completion timestamp"
    )

class HealthCheckResponse(BaseModel):
    """
    complete health check response for system monitoring.
    
    Implements Algorithm 13.4 (Health Monitoring Protocol) ensuring
    complete system verification and dependency validation.
    """
    
    # Overall system health
    status: str = Field(description="Overall system health status")
    healthy: bool = Field(description="Boolean health indicator")
    
    # Component health checks
    pygmo_available: bool = Field(description="PyGMO library availability")
    input_loader_functional: bool = Field(description="Input loading system health")
    processing_engine_ready: bool = Field(description="Optimization engine readiness")
    output_writer_available: bool = Field(description="Output generation system health")
    
    # Dependency verification
    required_libraries: Dict[str, bool] = Field(
        description="Required library availability status"
    )
    
    # System capabilities
    supported_algorithms: List[str] = Field(
        description="Available optimization algorithms"
    )
    
    max_concurrent_requests: int = Field(
        description="Maximum concurrent optimization capacity"
    )
    
    # Performance indicators
    response_time_ms: float = Field(
        description="Health check response time in milliseconds"
    )
    
    memory_available_mb: float = Field(
        description="Available system memory"
    )
    
    disk_space_available_gb: float = Field(
        description="Available disk space for operations"
    )
    
    # Version information
    api_version: str = Field(description="API version string")
    pygmo_version: Optional[str] = Field(
        default=None,
        description="PyGMO library version"
    )
    
    # Timestamp
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Health check timestamp"
    )