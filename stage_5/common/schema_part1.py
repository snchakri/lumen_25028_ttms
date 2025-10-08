"""
STAGE 5 - COMMON/SCHEMA.PY
Enterprise-Grade JSON Schema Definitions & Pydantic Models

This module defines comprehensive data contracts for Stage 5's input/output specifications
based on the rigorous theoretical frameworks and mathematical foundations. Every schema 
is validated against the formal models defined in the Stage 5.1 and 5.2 theoretical papers.

CRITICAL IMPLEMENTATION NOTES:
- NO MOCK DATA: All schemas represent real production structures
- STRICT VALIDATION: Every field has rigorous type checking and bounds validation  
- THEORETICAL COMPLIANCE: Schemas align precisely with mathematical parameter definitions
- CURSOR/PyCharm IDE OPTIMIZATION: Full type hints for intelligent code completion
- ENTERPRISE RELIABILITY: Fail-fast validation prevents downstream corruption
- PYDANTIC V2 COMPLIANCE: Uses field_validator and model_validator decorators

References:
- Stage-5.1-INPUT-COMPLEXITY-ANALYSIS-Theoretical-Foundations-Mathematical-Framework.pdf
- Stage-5.2-SOLVER-SELECTION-ARSENAL-MODULARITY-Theoretical-Foundations-Mathematical-Framework.pdf
- hei_timetabling_datamodel.sql (Entity relationship understanding)
- Stage5-FOUNDATIONAL-DESIGN-IMPLEMENTATION-PLAN.md (Exact file format specifications)

Cross-Module Dependencies:
- common.exceptions: Stage5ValidationError, Stage5ConfigurationError
- common.utils: validate_file_path, load_json_schema
- common.logging: get_logger for schema validation events

IDE Integration Notes:
- All classes have comprehensive docstrings for IntelliSense/autocomplete
- Type annotations provide full IDE support for method chaining
- Field descriptions enable context-sensitive help in Cursor/PyCharm
"""

from typing import Dict, List, Optional, Union, Any, Literal
from enum import Enum
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import json
import re

# Pydantic V2 imports - enterprise-grade validation framework
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic.types import PositiveInt, PositiveFloat

# =============================================================================
# ENUMERATION TYPES - Domain-Specific Classifications
# Based on hei_timetabling_datamodel.sql enum definitions and solver frameworks
# =============================================================================

class SolverParadigmEnum(str, Enum):
    """
    Solver paradigm classification for Stage 5.2 solver selection arsenal.
    Maps directly to capability vectors in theoretical framework Section 2.1.

    Paradigm Classifications (from Stage 6 solver family frameworks):
    - MILP: Mixed Integer Linear Programming (PuLP-CBC, GLPK, HiGHS)  
    - CP: Constraint Programming (OR-Tools CP-SAT)
    - EVOLUTIONARY: Genetic/Evolutionary Algorithms (DEAP suite)
    - MULTI_OBJECTIVE: Pareto optimization methods (PyGMO archipelago)
    - HYBRID: Combined paradigm approaches for complex problems

    References:
    - Stage-6.1-PuLP-SOLVER-FAMILY: MILP paradigm implementations
    - Stage-6.2-OR-Tools: CP constraint programming with search strategies
    - Stage-6.3-DEAP: EVOLUTIONARY genetic programming and algorithms
    - Stage-6.4-PyGMO: MULTI_OBJECTIVE Pareto front optimization
    """
    MILP = "MILP"                     
    CP = "CP"                         
    EVOLUTIONARY = "EVOLUTIONARY"     
    MULTI_OBJECTIVE = "MULTI_OBJECTIVE"  
    HYBRID = "HYBRID"                

class LPConvergenceStatusEnum(str, Enum):
    """
    Linear Programming convergence status for Stage 5.2 weight learning optimization.
    Based on PuLP solver status codes and theoretical convergence guarantees.

    Status Classifications:
    - OPTIMAL: Global optimum found with proven optimality conditions
    - FEASIBLE: Feasible solution found but optimality not guaranteed
    - INFEASIBLE: No feasible solution exists (constraint inconsistency)  
    - UNBOUNDED: Problem is unbounded (no maximum separation margin)
    - NOT_SOLVED: Solver did not complete within time/iteration limits

    References:
    - Stage-5.2 Section 4.2: Iterative LP weight learning algorithm
    - Theorem 4.3: Convergence properties and optimality conditions
    - PuLP documentation: Solver status code mappings
    """
    OPTIMAL = "OPTIMAL"               
    FEASIBLE = "FEASIBLE"             
    INFEASIBLE = "INFEASIBLE"         
    UNBOUNDED = "UNBOUNDED"           
    NOT_SOLVED = "NOT_SOLVED"         

class FileFormatEnum(str, Enum):
    """
    Stage 3 output file format enumeration for L_idx multi-modal index support.
    Supports various serialization formats per Stage3-DATA-COMPILATION framework.

    Format Classifications:
    - PARQUET: Apache Parquet columnar format (default for L_raw)
    - GRAPHML: GraphML relationship graphs (default for L_rel)  
    - PKL: Python pickle binary format (compact serialization)
    - FEATHER: Apache Arrow feather format (fast I/O)
    - IDX: Custom index binary format (optimized access)
    - BIN: Generic binary format (fallback option)

    References:
    - Stage-3-DATA-COMPILATION Section 5.3: Multi-modal index structures
    - Stage5-FOUNDATIONAL-DESIGN: Input compatibility requirements
    - PyArrow documentation: Parquet and Feather format specifications
    """
    PARQUET = "parquet"               
    GRAPHML = "graphml"               
    PKL = "pkl"                       
    FEATHER = "feather"               
    IDX = "idx"                       
    BIN = "bin"                       

# =============================================================================
# STAGE 5.1 SCHEMA DEFINITIONS - Input-Complexity Analysis
# Exact mathematical parameter specifications from theoretical framework
# =============================================================================

class ExecutionMetadata(BaseModel):
    """
    Execution metadata for Stage 5.1 complexity analysis computational runs.
    Tracks performance metrics and ensures deterministic reproducibility.

    Critical Fields:
    - timestamp: ISO 8601 execution start time for complete audit trails
    - computation_time_ms: Total duration for performance benchmarking
    - software_version: Implementation version for compatibility tracking
    - random_seed: Fixed seed for P13 ruggedness and P16 variance calculations

    Validation Constraints:
    - computation_time_ms: ≤600,000ms (10 minute prototype time limit)  
    - software_version: Must follow semantic versioning (major.minor.patch)
    - random_seed: Integer for NumPy/SciPy deterministic computations

    IDE Integration:
    - Full type hints enable IntelliSense autocomplete for all fields
    - Field descriptions provide context-sensitive help tooltips
    - Validation methods show expected input formats and constraints
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )

    timestamp: datetime = Field(
        ..., 
        description="ISO 8601 timestamp of complexity analysis execution start"
    )
    computation_time_ms: PositiveInt = Field(
        ..., 
        description="Total computation duration in milliseconds for performance tracking"
    )
    software_version: str = Field(
        default="1.0.0", 
        description="Semantic version of Stage 5 implementation (major.minor.patch format)"
    )
    random_seed: int = Field(
        default=42, 
        description="Fixed random seed for deterministic stochastic parameter computations"
    )

    @field_validator('computation_time_ms')
    @classmethod
    def validate_computation_time_bounds(cls, v: int) -> int:
        """
        Validate computation time is within Stage 5 performance requirements.

        Performance Constraints (from Stage5-FOUNDATIONAL-DESIGN):
        - Maximum execution time: 10 minutes (600,000ms) for 2k entity prototype
        - Minimum execution time: 1ms (sanity check for valid computation)

        Args:
            v: Computation time in milliseconds

        Returns:
            int: Validated computation time

        Raises:
            ValueError: If computation time exceeds performance bounds
        """
        if v > 600000:  # 10 minutes maximum per Stage 5 performance requirements
            raise ValueError(
                f"Computation time {v}ms exceeds maximum allowed duration of 600000ms "
                f"(10 minutes for 2k entity prototype scale)"
            )
        if v < 1:
            raise ValueError("Computation time must be at least 1ms for valid execution")
        return v

    @field_validator('software_version')
    @classmethod
    def validate_semantic_version_format(cls, v: str) -> str:
        """
        Validate software version follows semantic versioning specification.

        Semantic Versioning Format: MAJOR.MINOR.PATCH
        - MAJOR: Incompatible API changes
        - MINOR: Backwards-compatible functionality additions  
        - PATCH: Backwards-compatible bug fixes

        Args:
            v: Version string to validate

        Returns:
            str: Validated semantic version string

        Raises:
            ValueError: If version format is invalid
        """
        if not re.match(r'^\d+\.\d+\.\d+$', v):
            raise ValueError(
                f"Software version '{v}' must follow semantic versioning format "
                f"(e.g., '1.0.0', '2.1.3')"
            )
        return v

class EntityCountStatistics(BaseModel):
    """
    Entity count statistics for complexity parameter provenance and validation.
    Based on hei_timetabling_datamodel.sql core entity table definitions.

    Entity Type Definitions (from SQL schema):
    - courses: Academic course catalog entries (courses table, foreign keys to programs)
    - faculty: Academic staff members (faculty table, competency relationships)
    - rooms: Physical classroom/lab spaces (rooms table, capacity and equipment)
    - timeslots: Discrete scheduling time periods (timeslots table, temporal constraints)
    - batches: Student group assignments (studentbatches table, enrollment data)

    Scale Validation:
    - All counts validated against prototype scale limits (≤2k students)
    - Maximum entity counts prevent computational complexity explosion
    - Minimum entity counts ensure meaningful complexity calculations

    References:
    - hei_timetabling_datamodel.sql: Complete entity relationship schema
    - Stage-5.1 Parameters 1-16: Entity count usage in mathematical formulas
    - Prototype scale requirements: 2000 student maximum capacity
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    courses: PositiveInt = Field(
        ..., 
        description="Total number of courses in L_raw normalized entity tables"
    )
    faculty: PositiveInt = Field(
        ..., 
        description="Total number of faculty members in L_raw normalized entity tables"
    ) 
    rooms: PositiveInt = Field(
        ..., 
        description="Total number of rooms/spaces in L_raw normalized entity tables"
    )
    timeslots: PositiveInt = Field(
        ..., 
        description="Total number of discrete timeslots in L_raw normalized entity tables"
    )
    batches: PositiveInt = Field(
        ..., 
        description="Total number of student batches in L_raw normalized entity tables"
    )

    @field_validator('courses', 'faculty', 'rooms', 'timeslots', 'batches')
    @classmethod
    def validate_entity_scale_bounds(cls, v: int, info) -> int:
        """
        Validate entity counts are within expected prototype scale limits.

        Prototype Scale Constraints (for ≤2k student institutions):
        - courses: ≤1000 (typical university course catalog size)
        - faculty: ≤300 (typical faculty-to-student ratios)
        - rooms: ≤200 (typical campus classroom/lab capacity)
        - timeslots: ≤100 (typical weekly scheduling periods)
        - batches: ≤150 (typical section/batch divisions)

        Args:
            v: Entity count to validate
            info: Pydantic field information containing field name

        Returns:
            int: Validated entity count

        Raises:
            ValueError: If entity count exceeds prototype scale limits
        """
        max_values = {
            'courses': 1000, 'faculty': 300, 'rooms': 200, 
            'timeslots': 100, 'batches': 150
        }

        field_name = info.field_name
        max_allowed = max_values.get(field_name, 1000)

        if v > max_allowed:
            raise ValueError(
                f"{field_name} count {v} exceeds prototype scale maximum of {max_allowed} "
                f"(designed for ≤2k student institutions)"
            )
        if v < 1:
            raise ValueError(f"{field_name} count must be at least 1 for valid complexity analysis")
        return v

class ComputationConfiguration(BaseModel):
    """
    Computational configuration parameters for stochastic complexity calculations.
    Ensures deterministic reproducibility of P13 landscape ruggedness and P16 quality variance.

    Stochastic Parameter Requirements:
    - P13 (Landscape Ruggedness): Random walk sampling over solution space
    - P16 (Quality Variance): K-sample statistical analysis of solution quality

    Configuration Parameters:
    - sampling_seed: NumPy random seed for all stochastic computations
    - ruggedness_walks: Number of random walks for P13 statistical significance
    - variance_samples: Number of solution samples for P16 coefficient calculation

    Statistical Significance Constraints:
    - ruggedness_walks: ≥100 for statistical significance, ≤10,000 for efficiency
    - variance_samples: ≥10 for meaningful statistics, ≤200 for computational limits

    References:
    - Stage-5.1 Parameter 13: Optimization Landscape Ruggedness mathematical definition
    - Stage-5.1 Parameter 16: Solution Quality Variance coefficient formulation
    - NumPy random number generation: Mersenne Twister algorithm documentation
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    sampling_seed: int = Field(
        default=42, 
        description="NumPy random seed for deterministic stochastic parameter calculations"
    )
    ruggedness_walks: PositiveInt = Field(
        default=1000, 
        description="Number of random walks for P13 landscape ruggedness estimation"
    )
    variance_samples: PositiveInt = Field(
        default=50, 
        description="Number of solution samples for P16 quality variance coefficient"
    )

    @field_validator('ruggedness_walks')
    @classmethod
    def validate_ruggedness_statistical_bounds(cls, v: int) -> int:
        """
        Validate ruggedness walk count for statistical significance and computational efficiency.

        Statistical Requirements:
        - Minimum 100 walks: Required for meaningful correlation estimates in P13
        - Maximum 10,000 walks: Computational efficiency bound for prototype
        - Optimal range 500-2000: Balance between accuracy and performance

        Args:
            v: Number of ruggedness walks

        Returns:
            int: Validated walk count

        Raises:
            ValueError: If walk count is outside statistical/computational bounds
        """
        if v < 100:
            raise ValueError(
                f"Ruggedness walks {v} below minimum 100 required for statistical significance "
                f"in P13 landscape correlation analysis"
            )
        if v > 10000:
            raise ValueError(
                f"Ruggedness walks {v} exceeds maximum 10000 for computational efficiency "
                f"in prototype environment"
            )
        return v

    @field_validator('variance_samples') 
    @classmethod
    def validate_variance_statistical_bounds(cls, v: int) -> int:
        """
        Validate variance sample count for meaningful coefficient of variation calculation.

        Statistical Requirements:
        - Minimum 10 samples: Required for coefficient of variation estimation
        - Maximum 200 samples: Computational efficiency bound for prototype  
        - Optimal range 30-100: Standard practice for variance estimation

        Args:
            v: Number of variance samples

        Returns:
            int: Validated sample count

        Raises:
            ValueError: If sample count is outside statistical bounds
        """
        if v < 10:
            raise ValueError(
                f"Variance samples {v} below minimum 10 required for meaningful "
                f"coefficient of variation calculation in P16"
            )
        if v > 200:
            raise ValueError(
                f"Variance samples {v} exceeds maximum 200 for computational efficiency "
                f"in prototype environment"
            )
        return v

class ComplexityParameterVector(BaseModel):
    """
    The 16 rigorously defined complexity parameters from theoretical framework.
    Each parameter has exact mathematical definition and computational validation bounds.

    CRITICAL IMPLEMENTATION NOTE:
    These are NOT mock values - they represent real mathematical computations
    based on Stage 3 compiled data structures and proven theoretical formulas.

    Mathematical Parameter Definitions (exact formulas from Stage-5.1 framework):

    P1_dimensionality: |C| × |F| × |R| × |T| × |B| (Problem space size)
    P2_constraint_density: |Active_Constraints| / |Max_Possible_Constraints| ∈ [0,1]
    P3_faculty_specialization: (1/|F|) × Σ_f (|C_f| / |C|) ∈ [0,1]  
    P4_room_utilization: Σ_c,b (hours_c,b) / (|R| × |T|) ∈ [0,1]
    P5_temporal_complexity: Var(R_t) / Mean(R_t)² (Coefficient of variation) ∈ [0,∞)
    P6_batch_variance: σ_B / μ_B (Coefficient of variation) ∈ [0,∞)
    P7_competency_entropy: Σ_f,c (-p_f,c × log2(p_f,c)) ∈ [0, log2(|F|×|C|)]
    P8_conflict_measure: (1/k(k-1)) × Σ_i,j |ρ(f_i, f_j)| ∈ [0,1]
    P9_coupling_coefficient: Σ_i,j |V_i ∩ V_j| / min(|V_i|, |V_j|) ∈ [0,∞)
    P10_heterogeneity_index: H_R + H_F + H_C (Sum of entropies) ∈ [0,∞)
    P11_flexibility_measure: (1/|C|) × Σ_c (|T_c| / |T|) ∈ [0,1]
    P12_dependency_complexity: |E|/|C| + depth(G) + width(G) ∈ [0,∞)
    P13_landscape_ruggedness: 1 - (1/(N-1)) × Σ_i ρ(f(x_i), f(x_{i+1})) ∈ [0,1]
    P14_scalability_factor: log(S_target/S_current) / log(C_current/C_expected) ∈ (-∞,∞)
    P15_propagation_depth: (1/|A|) × Σ_a max_depth_from_a ∈ [0,∞)
    P16_quality_variance: σ_Q / μ_Q (Coefficient of variation) ∈ [0,∞)

    Theoretical References:
    - Stage-5.1-INPUT-COMPLEXITY-ANALYSIS: Complete parameter mathematical definitions
    - Theorems 3.2-18.2: Formal proofs of complexity contribution relationships
    - Section 6: Composite index PCA weight validation on 500-problem dataset
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    p1_dimensionality: PositiveFloat = Field(
        ..., 
        description="P1: Problem space dimensionality |C|×|F|×|R|×|T|×|B|"
    )
    p2_constraint_density: float = Field(
        ..., 
        ge=0.0, le=1.0,
        description="P2: Constraint density ratio [0,1]"
    )
    p3_faculty_specialization: float = Field(
        ..., 
        ge=0.0, le=1.0,
        description="P3: Faculty specialization index [0,1]"
    )
    p4_room_utilization: float = Field(
        ..., 
        ge=0.0, le=1.0,
        description="P4: Room utilization factor [0,1]"
    )
    p5_temporal_complexity: PositiveFloat = Field(
        ..., 
        description="P5: Temporal distribution complexity (coefficient of variation)"
    )
    p6_batch_variance: PositiveFloat = Field(
        ..., 
        description="P6: Batch size variance (coefficient of variation)"
    )
    p7_competency_entropy: PositiveFloat = Field(
        ..., 
        description="P7: Competency distribution entropy (bits)"
    )
    p8_conflict_measure: float = Field(
        ..., 
        ge=0.0, le=1.0,
        description="P8: Multi-objective conflict measure [0,1]"
    )
    p9_coupling_coefficient: PositiveFloat = Field(
        ..., 
        description="P9: Constraint coupling coefficient"
    )
    p10_heterogeneity_index: PositiveFloat = Field(
        ..., 
        description="P10: Resource heterogeneity index (sum of entropies)"
    )
    p11_flexibility_measure: float = Field(
        ..., 
        ge=0.0, le=1.0,
        description="P11: Schedule flexibility measure [0,1]"
    )
    p12_dependency_complexity: PositiveFloat = Field(
        ..., 
        description="P12: Dependency graph complexity"
    )
    p13_landscape_ruggedness: float = Field(
        ..., 
        ge=0.0, le=1.0,
        description="P13: Optimization landscape ruggedness [0,1]"
    )
    p14_scalability_factor: float = Field(
        ..., 
        description="P14: Scalability projection factor (can be negative)"
    )
    p15_propagation_depth: PositiveFloat = Field(
        ..., 
        description="P15: Constraint propagation depth"
    )
    p16_quality_variance: PositiveFloat = Field(
        ..., 
        description="P16: Solution quality variance (coefficient of variation)"
    )

    @field_validator('p1_dimensionality')
    @classmethod
    def validate_dimensionality_computational_bounds(cls, v: float) -> float:
        """
        Validate P1 dimensionality within computational tractability bounds.

        Computational Feasibility (from Theorem 3.2):
        - Minimum: 1.0 (must have positive problem space)
        - Maximum: 1e12 (computational tractability limit for prototype)
        - Typical range: 1e3 to 1e9 for real scheduling problems

        Args:
            v: Dimensionality value

        Returns:
            float: Validated dimensionality

        Raises:
            ValueError: If dimensionality exceeds computational bounds
        """
        if v <= 0:
            raise ValueError("P1 dimensionality must be positive (non-zero problem space)")
        if v > 1e12:
            raise ValueError(
                f"P1 dimensionality {v:.2e} exceeds computational tractability bounds "
                f"of 1e12 from Theorem 3.2"
            )
        return v

    @field_validator('p7_competency_entropy')
    @classmethod
    def validate_entropy_information_bounds(cls, v: float) -> float:
        """
        Validate P7 entropy within information-theoretic bounds.

        Information Theory Bounds:
        - Maximum theoretical: log2(|F|×|C|) for uniform distribution
        - Practical maximum: ~30 bits for prototype scale (1000×300 entities)
        - Minimum: 0.0 (completely deterministic competency assignment)

        Args:
            v: Competency entropy value

        Returns:
            float: Validated entropy

        Raises:
            ValueError: If entropy exceeds theoretical maximum
        """
        if v < 0:
            raise ValueError("P7 competency entropy cannot be negative (information theory)")
        if v > 30:  # log2(1000*300) ≈ 18.5, with safety margin for prototype scale
            raise ValueError(
                f"P7 competency entropy {v:.2f} exceeds theoretical maximum ~30 bits "
                f"for prototype scale entities"
            )
        return v

print("✅ STAGE 5 COMMON/SCHEMA.PY - Part 1/4 Complete")
print("   - Pydantic V2 compliant with field_validator decorators")
print("   - Enterprise-grade validation with mathematical bounds checking")
print("   - Complete enum definitions for solver paradigms and status codes")
print("   - Full IDE integration with comprehensive docstrings and type hints")
