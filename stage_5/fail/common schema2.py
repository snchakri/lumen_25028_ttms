# STAGE 5 - COMMON/SCHEMA.PY (CONTINUED - Part 2/3)
# Complex Parameters and Stage 5.2 Schema Definitions

class ComplexityParameters(BaseModel):
    """
    The 16 rigorously defined complexity parameters from theoretical framework.
    Each parameter has exact mathematical definition and computational bounds validation.
    
    CRITICAL: These are NOT mock values - they represent real mathematical computations
    based on Stage 3 compiled data structures and theoretical formulas from papers.
    
    Mathematical Parameter Definitions (exact formulas from theoretical framework):
    
    P1_dimensionality: |C| × |F| × |R| × |T| × |B| (Problem space size)
        Range: [1, 1e12] (computational tractability limit from Theorem 3.2)
    
    P2_constraint_density: |Active_Constraints| / |Max_Possible_Constraints| 
        Range: [0,1] (normalized ratio)
    
    P3_faculty_specialization: (1/|F|) × Σ_f (|C_f| / |C|)
        Range: [0,1] (normalized specialization index)
    
    P4_room_utilization: Σ_c,b (hours_c,b) / (|R| × |T|)
        Range: [0,1] (normalized utilization factor)
    
    P5_temporal_complexity: Var(R_t) / Mean(R_t)² (Coefficient of variation)
        Range: [0,∞) (variance normalized by squared mean)
    
    P6_batch_variance: σ_B / μ_B (Coefficient of variation)
        Range: [0,∞) (standard deviation over mean ratio)
    
    P7_competency_entropy: Σ_f,c (-p_f,c × log2(p_f,c))
        Range: [0, log2(|F|×|C|)] ≈ [0,30] for prototype scale
    
    P8_conflict_measure: (1/k(k-1)) × Σ_i,j |ρ(f_i, f_j)|
        Range: [0,1] (normalized multi-objective conflict)
    
    P9_coupling_coefficient: Σ_i,j |V_i ∩ V_j| / min(|V_i|, |V_j|)
        Range: [0,∞) (constraint coupling strength)
    
    P10_heterogeneity_index: H_R + H_F + H_C (Sum of entropies)
        Range: [0,∞) (additive entropy across resource types)
    
    P11_flexibility_measure: (1/|C|) × Σ_c (|T_c| / |T|)
        Range: [0,1] (normalized scheduling flexibility)
    
    P12_dependency_complexity: |E|/|C| + depth(G) + width(G)
        Range: [0,∞) (dependency graph structural complexity)
    
    P13_landscape_ruggedness: 1 - (1/(N-1)) × Σ_i ρ(f(x_i), f(x_{i+1}))
        Range: [0,1] (landscape smoothness measure)
    
    P14_scalability_factor: log(S_target/S_current) / log(C_current/C_expected)
        Range: (-∞,∞) (logarithmic scalability projection)
    
    P15_propagation_depth: (1/|A|) × Σ_a max_depth_from_a
        Range: [0,∞) (average constraint propagation depth)
    
    P16_quality_variance: σ_Q / μ_Q (Coefficient of variation)
        Range: [0,∞) (solution quality variance measure)
    
    References:
    - Stage-5.1-INPUT-COMPLEXITY-ANALYSIS: Parameters 1-16 mathematical definitions
    - Theorems 3.2-18.2: Formal proofs of complexity relationships and bounds
    """
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
    def validate_dimensionality_bounds(cls, v: float) -> float:
        """Validate P1 dimensionality within computational feasibility bounds."""
        if v < 1:
            raise ValueError("P1 dimensionality must be positive")
        if v > 1e12:  # Computational tractability limit from Theorem 3.2
            raise ValueError("P1 dimensionality exceeds computational tractability bounds")
        return v

    @field_validator('p7_competency_entropy')
    @classmethod
    def validate_entropy_bounds(cls, v: float) -> float:
        """Validate P7 entropy is within information-theoretic bounds."""
        if v > 30:  # log2(1000*300) ≈ 18.5 for max prototype scale
            raise ValueError("P7 competency entropy exceeds theoretical maximum")
        return v

class ComplexityMetricsSchema(BaseModel):
    """
    Complete Stage 5.1 output schema for complexity analysis results.
    This is the EXACT format that Stage 5.1 must produce and Stage 5.2 must consume.
    
    Schema Version: 1.0.0 (semantic versioning for compatibility)
    
    Critical Implementation Notes:
    - This schema MUST match the JSON examples in Stage5-FOUNDATIONAL-DESIGN-IMPLEMENTATION-PLAN.md
    - Every field is mandatory and validated according to theoretical constraints
    - The composite_index uses PCA-validated weights from empirical analysis
    - All metadata enables full execution auditability and reproducibility
    - parameter_statistics contains entity_counts and computation_notes objects
    """
    schema_version: str = Field(
        default="1.0.0", 
        description="Schema version for compatibility checking"
    )
    execution_metadata: ExecutionMetadata = Field(
        ..., 
        description="Execution tracking and performance metadata"
    )
    complexity_parameters: ComplexityParameters = Field(
        ..., 
        description="The 16 rigorously computed complexity parameters"
    )
    composite_index: PositiveFloat = Field(
        ..., 
        description="PCA-weighted composite complexity index"
    )
    parameter_statistics: Dict[str, Any] = Field(
        ..., 
        description="Statistical metadata and provenance information"
    )

    @field_validator('schema_version')
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        """Validate schema version follows semantic versioning."""
        if not re.match(r'^\d+\.\d+\.\d+$', v):
            raise ValueError("Schema version must follow semantic versioning format")
        return v

    @field_validator('parameter_statistics')
    @classmethod
    def validate_statistics_structure(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure parameter_statistics contains required provenance fields."""
        required_keys = {'entity_counts', 'computation_notes'}
        if not all(key in v for key in required_keys):
            raise ValueError(f"parameter_statistics must contain keys: {required_keys}")
        return v

    @field_validator('composite_index')
    @classmethod
    def validate_composite_bounds(cls, v: float) -> float:
        """Validate composite index is within expected empirical bounds."""
        if v < 0:
            raise ValueError("Composite index must be non-negative")
        if v > 50:  # Empirical maximum from 500-problem validation dataset
            raise ValueError("Composite index exceeds empirical maximum bounds")
        return v

# =============================================================================
# STAGE 5.2 SCHEMAS - SOLVER SELECTION & ARSENAL MODULARITY
# L2 normalization and LP-based weight learning validation models  
# =============================================================================

class SolverLimits(BaseModel):
    """
    Solver capacity and performance limits for feasibility checking.
    Based on empirical characterization of solver families in Stage 6 frameworks.
    
    Limit Specifications:
    - max_variables: Maximum decision variables the solver can handle
    - max_constraints: Maximum constraints for MILP/CP solvers (optional for evolutionary)
    - time_limit_default: Default time limit in seconds for solver execution
    - population_size: Population size for evolutionary algorithms (DEAP/PyGMO only)
    - generations: Maximum generations for evolutionary methods (DEAP/PyGMO only)
    - archipelago_size: Archipelago size for PyGMO island model (PyGMO only)
    - migration_rate: Migration rate for island models (PyGMO only, range [0,1])
    
    References:
    - Stage-6.1-PuLP: CBC solver limits and performance characteristics
    - Stage-6.2-OR-Tools: CP-SAT scaling properties and variable limits
    - Stage-6.3-DEAP: Population size and generation constraints
    - Stage-6.4-PyGMO: Archipelago architecture and migration parameters
    """
    max_variables: PositiveInt = Field(
        ..., 
        description="Maximum decision variables the solver can handle"
    )
    max_constraints: Optional[PositiveInt] = Field(
        None, 
        description="Maximum constraints for MILP/CP solvers"
    )
    time_limit_default: PositiveInt = Field(
        default=300, 
        description="Default time limit in seconds"
    )
    population_size: Optional[PositiveInt] = Field(
        None, 
        description="Population size for evolutionary algorithms"
    )
    generations: Optional[PositiveInt] = Field(
        None, 
        description="Maximum generations for evolutionary methods"
    )
    archipelago_size: Optional[PositiveInt] = Field(
        None, 
        description="Archipelago size for PyGMO solvers"
    )
    migration_rate: Optional[float] = Field(
        None, 
        ge=0.0, le=1.0,
        description="Migration rate for island models"
    )

class SolverCapability(BaseModel):
    """
    Individual solver capability specification for Stage 5.2 arsenal.
    Each solver is characterized by a 16-dimensional capability vector corresponding
    to the 16 complexity parameters from Stage 5.1.
    
    CRITICAL: The capability_vector MUST have exactly 16 elements matching P1-P16.
    Values represent solver effectiveness for each complexity dimension (0-10 scale).
    
    Capability Vector Mapping:
    - capability_vector[0] → P1 (problem space dimensionality handling)
    - capability_vector[1] → P2 (constraint density management)  
    - capability_vector[2] → P3 (faculty specialization optimization)
    - ... (continues for all 16 parameters)
    - capability_vector[15] → P16 (quality variance handling)
    
    Validation Rules:
    - solver_id: Must follow naming convention for cross-engine compatibility
    - capability_vector: Exactly 16 elements, each in [0,10] effectiveness scale
    - paradigm: Must be one of the defined solver paradigm enumerations
    - limits: Must contain valid solver capacity constraints
    
    References:
    - Stage-5.2 Section 2.2: Solver capability matrix formulation
    - Stage-5.2 Section 3.1: L2 normalization mathematical framework  
    - Empirical characterization studies for PuLP, OR-Tools, DEAP, PyGMO families
    """
    solver_id: str = Field(
        ..., 
        description="Universal solver identifier across engine"
    )
    display_name: str = Field(
        ..., 
        min_length=3, max_length=100,
        description="Human-readable solver name"
    )
    paradigm: SolverParadigmEnum = Field(
        ..., 
        description="Solver paradigm classification"
    )
    capability_vector: List[float] = Field(
        ..., 
        description="16-dimensional capability vector [P1-P16]"
    )
    limits: SolverLimits = Field(
        ..., 
        description="Solver capacity and performance constraints"
    )

    @field_validator('capability_vector')
    @classmethod
    def validate_capability_dimension_and_values(cls, v: List[float]) -> List[float]:
        """Ensure capability vector has exactly 16 dimensions with valid values for P1-P16."""
        if len(v) != 16:
            raise ValueError(f"Capability vector must have exactly 16 elements, got {len(v)}")
        
        # Validate each capability value is in [0,10] effectiveness scale
        for i, val in enumerate(v):
            if not (0.0 <= val <= 10.0):
                raise ValueError(f"Capability vector element {i} must be in [0,10], got {val}")
        return v

    @field_validator('solver_id')
    @classmethod
    def validate_solver_id_format(cls, v: str) -> str:
        """Ensure solver_id follows naming convention for cross-engine compatibility."""
        if not v:
            raise ValueError("solver_id cannot be empty")
        if len(v) > 50:
            raise ValueError("solver_id must be ≤50 characters")
        # Validate alphanumeric with underscores, starting with letter
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', v):
            raise ValueError("solver_id must start with letter and contain only alphanumeric/underscore")
        return v.lower()  # Normalize to lowercase for consistency

print("✅ STAGE 5 COMMON/SCHEMA.PY - Part 2/3 Complete")
print("   - ComplexityParameters with all 16 mathematical parameter validations")
print("   - ComplexityMetricsSchema with complete Stage 5.1 output specification")
print("   - SolverCapability and SolverLimits for Stage 5.2 arsenal management")