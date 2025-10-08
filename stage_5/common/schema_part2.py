
class ComplexityMetricsSchema(BaseModel):
    """
    Complete Stage 5.1 output schema for complexity analysis results.
    This is the EXACT format that Stage 5.1 must produce and Stage 5.2 must consume.

    Schema Structure:
    - schema_version: Semantic versioning for compatibility tracking
    - execution_metadata: Performance and timing information
    - complexity_parameters: The 16 rigorously computed complexity parameters
    - composite_index: PCA-weighted composite complexity index
    - parameter_statistics: Statistical metadata and provenance information

    Critical Implementation Notes:
    - This schema MUST match JSON examples in Stage5-FOUNDATIONAL-DESIGN-IMPLEMENTATION-PLAN.md
    - Every field is mandatory and validated according to theoretical constraints
    - The composite_index uses PCA-validated weights from empirical 500-problem dataset
    - All metadata enables full execution auditability and reproducibility
    - Schema serves as data contract between Stage 5.1 and Stage 5.2

    Cross-Stage Integration:
    - Input to Stage 5.2 solver selection algorithm
    - References solver_capabilities.json for capability vector matching
    - Enables L2 normalization and LP weight learning optimization
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    schema_version: str = Field(
        default="1.0.0", 
        description="Schema version for compatibility checking across Stage 5 versions"
    )
    execution_metadata: ExecutionMetadata = Field(
        ..., 
        description="Execution tracking and performance metadata"
    )
    complexity_parameters: ComplexityParameterVector = Field(
        ..., 
        description="The 16 rigorously computed complexity parameters P1-P16"
    )
    composite_index: PositiveFloat = Field(
        ..., 
        description="PCA-weighted composite complexity index from empirical validation"
    )
    parameter_statistics: Dict[str, Any] = Field(
        ..., 
        description="Statistical metadata and computation provenance information"
    )

    @field_validator('schema_version')
    @classmethod
    def validate_schema_semantic_versioning(cls, v: str) -> str:
        """
        Validate schema version follows semantic versioning specification.

        Ensures compatibility tracking across Stage 5 implementation versions.
        Format: MAJOR.MINOR.PATCH where changes in MAJOR indicate breaking changes.

        Args:
            v: Schema version string

        Returns:
            str: Validated semantic version

        Raises:
            ValueError: If version format is invalid
        """
        if not re.match(r'^\d+\.\d+\.\d+$', v):
            raise ValueError(
                f"Schema version '{v}' must follow semantic versioning format (e.g., '1.0.0')"
            )
        return v

    @field_validator('parameter_statistics')
    @classmethod
    def validate_statistics_required_structure(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameter_statistics contains all required provenance fields.

        Required Structure:
        - entity_counts: EntityCountStatistics for parameter provenance
        - computation_notes: ComputationConfiguration for reproducibility

        Args:
            v: Statistics dictionary to validate

        Returns:
            Dict[str, Any]: Validated statistics dictionary

        Raises:
            ValueError: If required structure fields are missing
        """
        required_keys = {'entity_counts', 'computation_notes'}
        missing_keys = required_keys - set(v.keys())
        if missing_keys:
            raise ValueError(
                f"parameter_statistics missing required keys: {missing_keys}. "
                f"Must contain 'entity_counts' and 'computation_notes' for provenance tracking."
            )
        return v

    @field_validator('composite_index')
    @classmethod
    def validate_composite_empirical_bounds(cls, v: float) -> float:
        """
        Validate composite index is within empirically validated bounds.

        Empirical Bounds (from 500-problem validation dataset):
        - Minimum: 0.0 (trivial scheduling problem)
        - Maximum: 50.0 (most complex validated scheduling problem)
        - Typical range: 2.0 to 25.0 for real educational institutions

        Args:
            v: Composite index value

        Returns:
            float: Validated composite index

        Raises:
            ValueError: If index exceeds empirical bounds
        """
        if v < 0:
            raise ValueError("Composite index must be non-negative (complexity cannot be negative)")
        if v > 50:
            raise ValueError(
                f"Composite index {v:.2f} exceeds empirical maximum 50.0 from "
                f"500-problem validation dataset"
            )
        return v

# =============================================================================
# STAGE 5.2 SCHEMA DEFINITIONS - Solver Selection & Arsenal Modularity  
# L2 normalization and LP-based weight learning schemas
# =============================================================================

class SolverCapabilityLimits(BaseModel):
    """
    Solver capacity and performance limits for feasibility checking.
    Based on empirical characterization of solver families in Stage 6 frameworks.

    Limit Categories:
    - Computational: max_variables, max_constraints for problem scale
    - Temporal: time_limit_default for execution bounds
    - Algorithmic: population_size, generations for evolutionary methods
    - Architectural: archipelago_size, migration_rate for distributed optimization

    References:
    - Stage-6.1-PuLP: CBC solver limits and performance characteristics
    - Stage-6.2-OR-Tools: CP-SAT scaling properties and variable limits
    - Stage-6.3-DEAP: Population size and generation constraints for genetic algorithms
    - Stage-6.4-PyGMO: Archipelago architecture and migration parameters for global optimization
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    max_variables: PositiveInt = Field(
        ..., 
        description="Maximum decision variables the solver can handle efficiently"
    )
    max_constraints: Optional[PositiveInt] = Field(
        None, 
        description="Maximum constraints for MILP/CP solvers (None for evolutionary methods)"
    )
    time_limit_default: PositiveInt = Field(
        default=300, 
        description="Default time limit in seconds for solver execution"
    )
    population_size: Optional[PositiveInt] = Field(
        None, 
        description="Population size for evolutionary algorithms (None for exact methods)"
    )
    generations: Optional[PositiveInt] = Field(
        None, 
        description="Maximum generations for evolutionary methods (None for exact methods)"
    )
    archipelago_size: Optional[PositiveInt] = Field(
        None, 
        description="Archipelago size for PyGMO distributed optimization (None for single-island)"
    )
    migration_rate: Optional[float] = Field(
        None, 
        ge=0.0, le=1.0,
        description="Migration rate for island models [0,1] (None for non-distributed methods)"
    )

class SolverCapabilityProfile(BaseModel):
    """
    Individual solver capability specification for Stage 5.2 arsenal.
    Each solver characterized by 16-dimensional capability vector corresponding
    to the 16 complexity parameters from Stage 5.1.

    Capability Vector Structure:
    - 16 elements exactly matching P1-P16 complexity parameters
    - Values 0-10 representing solver effectiveness for each complexity dimension
    - Empirically determined through performance benchmarking on test problems

    CRITICAL REQUIREMENT:
    The capability_vector MUST have exactly 16 elements matching P1-P16 order.
    Values represent solver effectiveness (0=poor, 10=excellent) for each complexity type.

    Solver Identification:
    - solver_id: Universal identifier across entire scheduling engine
    - display_name: Human-readable name for user interfaces
    - paradigm: Classification for algorithmic approach categorization

    References:
    - Stage-5.2 Section 2.2: Solver capability matrix mathematical formulation
    - Stage-5.2 Section 3.1: L2 normalization mathematical framework  
    - Empirical characterization studies for PuLP, OR-Tools, DEAP, PyGMO families
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    solver_id: str = Field(
        ..., 
        min_length=1, max_length=50,
        description="Universal solver identifier across scheduling engine"
    )
    display_name: str = Field(
        ..., 
        min_length=3, max_length=100,
        description="Human-readable solver name for interfaces"
    )
    paradigm: SolverParadigmEnum = Field(
        ..., 
        description="Solver paradigm classification (MILP, CP, EVOLUTIONARY, etc.)"
    )
    capability_vector: List[float] = Field(
        ..., 
        description="16-dimensional capability vector matching P1-P16 complexity parameters"
    )
    limits: SolverCapabilityLimits = Field(
        ..., 
        description="Solver capacity and performance constraints"
    )

    @field_validator('capability_vector')
    @classmethod
    def validate_capability_dimensionality_and_bounds(cls, v: List[float]) -> List[float]:
        """
        Validate capability vector has exactly 16 dimensions with proper value bounds.

        Validation Requirements:
        - Exactly 16 elements: Must match P1-P16 complexity parameters
        - Value bounds: 0.0 ≤ value ≤ 10.0 for each capability dimension
        - No missing values: All 16 dimensions must be specified

        Args:
            v: Capability vector to validate

        Returns:
            List[float]: Validated 16-dimensional capability vector

        Raises:
            ValueError: If dimensionality or value bounds are incorrect
        """
        if len(v) != 16:
            raise ValueError(
                f"Capability vector must have exactly 16 elements for P1-P16 parameters, "
                f"got {len(v)} elements"
            )

        for i, capability in enumerate(v, 1):
            if not (0.0 <= capability <= 10.0):
                raise ValueError(
                    f"Capability vector element P{i} value {capability} must be in range [0.0, 10.0]"
                )

        return v

    @field_validator('solver_id')
    @classmethod
    def validate_solver_id_format(cls, v: str) -> str:
        """
        Validate solver_id follows naming convention for cross-engine compatibility.

        Naming Convention:
        - Must start with alphanumeric character
        - Can contain alphanumeric characters, underscores, hyphens
        - Must be unique within solver arsenal
        - Recommended format: {family}_{algorithm} (e.g., 'pulp_cbc', 'ortools_cpsat')

        Args:
            v: Solver ID to validate

        Returns:
            str: Validated and normalized solver ID (lowercase)

        Raises:
            ValueError: If solver ID format is invalid
        """
        if not v:
            raise ValueError("solver_id cannot be empty")

        # Normalize to lowercase for consistency
        v_normalized = v.lower()

        # Validate format: alphanumeric start, then alphanumeric/underscore/hyphen
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$', v):
            raise ValueError(
                f"solver_id '{v}' must start with alphanumeric character and contain only "
                f"alphanumeric characters, underscores, and hyphens"
            )

        return v_normalized

class SolverArsenalConfiguration(BaseModel):
    """
    Complete solver arsenal specification for Stage 5.2 input.
    Contains all available solvers with their capability profiles for selection optimization.

    Arsenal Structure:
    - schema_version: Version compatibility tracking for arsenal updates
    - last_updated: Currency timestamp for cache invalidation and freshness checking
    - solver_arsenal: Complete list of available solver capability profiles

    Arsenal Requirements:
    - Minimum 1 solver: Must have at least one available solver
    - Unique solver IDs: No duplicate solver identifiers within arsenal
    - Paradigm diversity: Should cover multiple solver paradigms for reliableness

    CRITICAL USAGE NOTE:
    This file is pre-loaded and shared across all Stage 5.2 executions.
    Updates require careful validation and compatibility testing with existing systems.

    References:
    - Stage-5.2 Section 2: Arsenal modularity mathematical framework
    - Stage-6.* frameworks: Individual solver family capability characterizations
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    schema_version: str = Field(
        default="1.0.0", 
        description="Arsenal schema version for compatibility tracking"
    )
    last_updated: datetime = Field(
        ..., 
        description="Last modification timestamp of solver arsenal configuration"
    )
    solver_arsenal: List[SolverCapabilityProfile] = Field(
        ..., 
        min_length=1,
        description="Complete list of available solver capability profiles"
    )

    @field_validator('solver_arsenal')
    @classmethod
    def validate_unique_solver_identifiers(cls, v: List[SolverCapabilityProfile]) -> List[SolverCapabilityProfile]:
        """
        Validate all solver IDs are unique within the arsenal.

        Uniqueness Requirements:
        - No duplicate solver_id values across arsenal
        - Case-insensitive uniqueness (normalized to lowercase)
        - Clear error messages for duplicate detection

        Args:
            v: List of solver capability profiles

        Returns:
            List[SolverCapabilityProfile]: Validated solver arsenal

        Raises:
            ValueError: If duplicate solver IDs are found
        """
        solver_ids = [solver.solver_id.lower() for solver in v]
        unique_ids = set(solver_ids)

        if len(solver_ids) != len(unique_ids):
            # Find duplicates for detailed error message
            duplicates = [sid for sid in unique_ids if solver_ids.count(sid) > 1]
            raise ValueError(
                f"Duplicate solver IDs found in arsenal: {duplicates}. "
                f"All solver IDs must be unique within the arsenal."
            )

        return v

    @field_validator('solver_arsenal')  
    @classmethod
    def validate_paradigm_diversity(cls, v: List[SolverCapabilityProfile]) -> List[SolverCapabilityProfile]:
        """
        Validate arsenal contains diverse solver paradigms for reliableness.

        Diversity Requirements:
        - At least 2 different paradigms: Ensures algorithm diversity
        - Recommended: Cover MILP, CP, EVOLUTIONARY, MULTI_OBJECTIVE paradigms
        - Prevents over-reliance on single algorithmic approach

        Args:
            v: List of solver capability profiles

        Returns:
            List[SolverCapabilityProfile]: Validated solver arsenal

        Raises:
            ValueError: If insufficient paradigm diversity
        """
        paradigms = {solver.paradigm for solver in v}

        if len(paradigms) < 2:
            available_paradigms = [solver.paradigm.value for solver in v]
            raise ValueError(
                f"Arsenal must contain at least 2 different solver paradigms for reliableness, "
                f"found paradigms: {available_paradigms}. "
                f"Consider adding solvers from different paradigm families."
            )

        return v

class SolverSelectionChoice(BaseModel):
    """
    Selected solver specification from Stage 5.2 optimization process.
    Contains the winning solver with confidence metrics and selection rationale.

    Selection Components:
    - solver_id: Identifier of chosen solver from arsenal
    - confidence: Selection confidence based on separation margin from LP optimization
    - match_score: Normalized match score from L2 optimization process
    - rationale: Human-readable explanation for selection decision

    Confidence Calculation:
    - Based on separation margin between top-ranked and second-best solvers
    - Higher separation margins indicate more confident selection decisions
    - Values near 1.0 indicate clear winner, values near 0.0 indicate close competition
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    solver_id: str = Field(
        ..., 
        description="Selected solver identifier from arsenal"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, le=1.0,
        description="Selection confidence based on LP optimization separation margin"
    )
    match_score: PositiveFloat = Field(
        ..., 
        description="Normalized match score from L2 optimization process"
    )
    rationale: Optional[str] = Field(
        None, 
        description="Human-readable selection rationale and reasoning"
    )

class SolverRankingEntry(BaseModel):
    """
    Individual solver ranking entry with scores and separation margins.
    Used for complete solver ranking output from Stage 5.2 optimization.

    Ranking Components:
    - solver_id: Solver identifier for ranking entry
    - score: Normalized match score from optimization process
    - margin: Score separation margin from top-ranked solver

    Margin Calculation:
    - margin = score_of_top_solver - score_of_current_solver
    - Top-ranked solver has margin = 0.0 by definition
    - Larger margins indicate greater performance differences
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    solver_id: str = Field(
        ..., 
        description="Solver identifier for ranking entry"
    )
    score: PositiveFloat = Field(
        ..., 
        description="Normalized match score from optimization process"
    )
    margin: float = Field(
        ..., 
        ge=0.0,
        description="Score separation margin from top-ranked solver"
    )

class LPConvergenceInformation(BaseModel):
    """
    Linear programming convergence tracking for Stage 5.2 weight learning.
    Monitors the iterative LP optimization process and solution quality.

    Convergence Metrics:
    - iterations: Number of LP iterations until convergence
    - status: Final LP solver status (OPTIMAL, FEASIBLE, etc.)
    - objective_value: Final LP objective value (separation margin)
    - tolerance: Convergence tolerance threshold used

    References:
    - Stage-5.2 Section 4.2: Iterative algorithm convergence properties
    - Theorem 4.3: LP optimality conditions and convergence guarantees
    - PuLP documentation: Solver status codes and convergence criteria
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    iterations: PositiveInt = Field(
        ..., 
        description="Number of LP iterations until convergence"
    )
    status: LPConvergenceStatusEnum = Field(
        ..., 
        description="Final LP solver convergence status"
    )
    objective_value: float = Field(
        ..., 
        description="Final LP objective value (separation margin)"
    )
    tolerance: PositiveFloat = Field(
        default=1e-6, 
        description="Convergence tolerance threshold"
    )

    @field_validator('iterations')
    @classmethod
    def validate_convergence_iteration_bounds(cls, v: int) -> int:
        """
        Validate LP convergence within expected iteration bounds.

        Iteration Bounds (from Theorem 4.3):
        - Maximum expected: 20 iterations for 16-parameter optimization
        - Typical range: 3-8 iterations for well-conditioned problems
        - Warning threshold: >10 iterations may indicate numerical issues

        Args:
            v: Number of iterations until convergence

        Returns:
            int: Validated iteration count

        Raises:
            ValueError: If iteration count exceeds theoretical bounds
        """
        if v > 20:
            raise ValueError(
                f"LP convergence required {v} iterations, exceeds theoretical maximum "
                f"of 20 iterations from Theorem 4.3. This may indicate numerical "
                f"conditioning issues or infeasible constraints."
            )
        return v

print("✅ STAGE 5 COMMON/SCHEMA.PY - Part 2/4 Complete")
print("   - Complete Stage 5.1 ComplexityMetricsSchema with full validation")
print("   - complete Stage 5.2 solver selection schemas")
print("   - LP convergence tracking with mathematical bounds checking")
