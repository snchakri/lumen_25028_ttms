# STAGE 5 - COMMON/SCHEMA.PY (FINAL - Part 3/3)
# Complete Stage 5.2 output schemas and shared validation models

class SolverArsenalSchema(BaseModel):
    """
    Complete solver arsenal specification for Stage 5.2 input.
    Contains all available solvers with their capability profiles for selection.
    
    Schema Structure:
    - schema_version: Compatibility tracking for arsenal format changes
    - last_updated: Arsenal currency timestamp for cache invalidation
    - solver_arsenal: Complete list of available solver specifications
    
    CRITICAL: This file is pre-loaded and shared across all Stage 5.2 executions.
    Updates require careful validation and compatibility testing across the engine.
    
    Validation Rules:
    - solver_arsenal: Must contain at least 1 solver, all IDs must be unique
    - paradigm_coverage: Arsenal must contain at least 2 different solver paradigms
    - capability_consistency: All solvers must have valid 16-D capability vectors
    """
    schema_version: str = Field(
        default="1.0.0", 
        description="Arsenal schema version"
    )
    last_updated: datetime = Field(
        ..., 
        description="Last modification timestamp of solver arsenal"
    )
    solver_arsenal: List[SolverCapability] = Field(
        ..., 
        min_length=1,
        description="Complete list of available solvers"
    )

    @field_validator('solver_arsenal')
    @classmethod
    def validate_unique_solver_ids(cls, v: List[SolverCapability]) -> List[SolverCapability]:
        """Ensure all solver IDs are unique within the arsenal."""
        solver_ids = [solver.solver_id for solver in v]
        if len(solver_ids) != len(set(solver_ids)):
            raise ValueError("Solver IDs must be unique within arsenal")
        return v

    @field_validator('solver_arsenal')
    @classmethod
    def validate_paradigm_coverage(cls, v: List[SolverCapability]) -> List[SolverCapability]:
        """Ensure arsenal covers major solver paradigms for robustness."""
        paradigms = {solver.paradigm for solver in v}
        if len(paradigms) < 2:
            raise ValueError("Arsenal must contain at least 2 different solver paradigms")
        return v

class SolverChoice(BaseModel):
    """
    Selected solver specification from Stage 5.2 optimization.
    Contains the winning solver with confidence metrics and rationale.
    
    Selection Fields:
    - solver_id: Selected solver identifier (matches arsenal solver_id)
    - confidence: Selection confidence based on separation margin [0,1]
    - match_score: Normalized match score from L2 optimization
    - rationale: Optional human-readable selection explanation
    """
    solver_id: str = Field(..., description="Selected solver identifier")
    confidence: float = Field(
        ..., 
        ge=0.0, le=1.0,
        description="Selection confidence based on separation margin"
    )
    match_score: PositiveFloat = Field(
        ..., 
        description="Normalized match score from L2 optimization"
    )
    rationale: Optional[str] = Field(
        None, 
        description="Human-readable selection rationale"
    )

class SolverRanking(BaseModel):
    """
    Individual solver ranking entry with scores and margins.
    Used for complete solver ranking output from Stage 5.2 selection process.
    
    Ranking Fields:
    - solver_id: Solver identifier (matches arsenal entries)
    - score: Normalized match score from capability-complexity correlation
    - margin: Score separation margin from top solver (always ≥0)
    """
    solver_id: str = Field(..., description="Solver identifier")
    score: PositiveFloat = Field(..., description="Normalized match score")
    margin: float = Field(
        ..., 
        ge=0.0,
        description="Score separation margin from top solver"
    )

class LPConvergenceInfo(BaseModel):
    """
    Linear programming convergence information for Stage 5.2 weight learning.
    Tracks the iterative LP optimization process and solution quality metrics.
    
    Convergence Fields:
    - iterations: Number of LP iterations until convergence (≤20 per theory)
    - status: Final LP solver status from convergence enumeration
    - objective_value: Final LP objective value (separation margin achieved)
    - tolerance: Convergence tolerance threshold for iteration termination
    
    Validation Rules:
    - iterations: Must be ≤20 per theoretical bound from Section 4.2
    - tolerance: Must be positive for numerical convergence detection
    - objective_value: Final separation margin (can be 0 if all solvers equivalent)
    
    References:
    - Stage-5.2 Section 4.2: Iterative algorithm convergence properties
    - Theorem 4.3: LP optimality conditions and convergence guarantees
    """
    iterations: PositiveInt = Field(
        ..., 
        description="Number of LP iterations until convergence"
    )
    status: LPConvergenceStatus = Field(
        ..., 
        description="Final LP solver status"
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
    def validate_convergence_iterations(cls, v: int) -> int:
        """Ensure convergence within expected iteration bounds."""
        if v > 20:  # Theoretical bound from Section 4.2
            raise ValueError(f"LP convergence required {v} iterations, exceeds expected maximum of 20")
        return v

class OptimizationDetails(BaseModel):
    """
    Detailed optimization process information for Stage 5.2 weight learning.
    Contains learned weights, normalization factors, and LP convergence data.
    
    Mathematical Components:
    - learned_weights: 16-D weight vector from LP optimization (Σw_j = 1, w_j ≥ 0)
    - normalization_factors: L2 normalization denominators for each parameter
    - separation_margin: Final LP objective value (d in maximize d formulation)
    - lp_convergence: Detailed convergence tracking information
    
    Validation Rules:
    - learned_weights: Exactly 16 elements, sum = 1.0 (simplex constraint)
    - normalization_factors: Exactly 16 positive elements for L2 scaling
    - separation_margin: Non-negative final optimization objective value
    """
    learned_weights: List[float] = Field(
        ..., 
        description="Learned parameter weights from LP"
    )
    normalization_factors: List[PositiveFloat] = Field(
        ..., 
        description="L2 normalization factors for each parameter"
    )
    separation_margin: float = Field(
        ..., 
        ge=0.0,
        description="Final separation margin from LP optimization"
    )
    lp_convergence: LPConvergenceInfo = Field(
        ..., 
        description="LP convergence tracking information"
    )

    @field_validator('learned_weights')
    @classmethod
    def validate_weight_dimension_and_simplex(cls, v: List[float]) -> List[float]:
        """Validate weight vector dimension and simplex constraint."""
        if len(v) != 16:
            raise ValueError(f"Learned weights must have exactly 16 elements, got {len(v)}")
        
        # Validate all weights are non-negative
        for i, weight in enumerate(v):
            if weight < 0:
                raise ValueError(f"Weight {i} must be non-negative, got {weight}")
        
        # Validate simplex constraint: sum = 1.0 (within numerical tolerance)
        weight_sum = sum(v)
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Learned weights must sum to 1.0, got {weight_sum}")
        return v

    @field_validator('normalization_factors')
    @classmethod
    def validate_normalization_dimension(cls, v: List[PositiveFloat]) -> List[PositiveFloat]:
        """Ensure normalization factors match 16-parameter structure."""
        if len(v) != 16:
            raise ValueError(f"Normalization factors must have exactly 16 elements, got {len(v)}")
        return v

class SelectionResult(BaseModel):
    """
    Complete solver selection result from Stage 5.2 optimization.
    Contains chosen solver and complete ranking with mathematical validation.
    
    Result Components:
    - chosen_solver: Selected optimal solver with confidence metrics
    - ranking: Complete solver ranking by match score (descending order)
    
    Validation Rules:
    - ranking: Must be non-empty, sorted by descending score
    - consistency: Top-ranked solver must match chosen solver
    - margin_calculation: Verify margin calculations are correct
    """
    chosen_solver: SolverChoice = Field(
        ..., 
        description="Selected optimal solver"
    )
    ranking: List[SolverRanking] = Field(
        ..., 
        min_length=1,
        description="Complete solver ranking by match score"
    )

    @model_validator(mode='after')
    def validate_ranking_consistency(self):
        """Ensure ranking is consistent with chosen solver and mathematically correct."""
        if not self.ranking:
            return self
            
        # Top-ranked solver should match chosen solver
        top_solver = self.ranking[0]
        if top_solver.solver_id != self.chosen_solver.solver_id:
            raise ValueError("Top-ranked solver must match chosen solver")
        
        # Verify ranking is sorted by descending score
        scores = [entry.score for entry in self.ranking]
        if scores != sorted(scores, reverse=True):
            raise ValueError("Ranking must be sorted by descending match score")
            
        # Verify margin calculations are correct (margin = top_score - current_score)
        top_score = scores[0] if scores else 0
        for entry in self.ranking:
            expected_margin = top_score - entry.score
            if abs(entry.margin - expected_margin) > 1e-6:  # Numerical tolerance
                raise ValueError(f"Incorrect margin calculation for {entry.solver_id}")
                
        return self

class SolverSelectionSchema(BaseModel):
    """
    Complete Stage 5.2 output schema for solver selection results.
    This is the EXACT format that Stage 5.2 must produce for Stage 6 consumption.
    
    Schema Components:
    - schema_version: Version compatibility tracking for downstream integration
    - execution_metadata: Performance timing and execution context information
    - selection_result: Chosen solver and complete ranking with validation
    - optimization_details: Mathematical optimization process transparency
    
    CRITICAL: This schema represents the final handoff to Stage 6 solver execution.
    All downstream stages depend on this exact format specification for integration.
    """
    schema_version: str = Field(
        default="1.0.0", 
        description="Schema version for compatibility"
    )
    execution_metadata: ExecutionMetadata = Field(
        ..., 
        description="Execution timing and metadata"
    )
    selection_result: SelectionResult = Field(
        ..., 
        description="Solver selection results and ranking"
    )
    optimization_details: OptimizationDetails = Field(
        ..., 
        description="L2 normalization and LP optimization details"
    )

# =============================================================================
# SHARED INPUT/OUTPUT VALIDATION SCHEMAS
# Common structures for file paths, contexts, and error handling
# =============================================================================

class Stage3OutputPaths(BaseModel):
    """
    Stage 3 output file path specifications for Stage 5.1 input validation.
    Supports multiple index formats per Stage 3 compilation framework flexibility.
    
    Path Structure Requirements:
    - l_raw_path: Normalized entity tables in Parquet format (.parquet)
    - l_rel_path: Relationship graphs in GraphML format (.graphml)
    - l_idx_path: Multi-modal indices (supports .pkl/.parquet/.feather/.idx/.bin)
    
    Validation Rules:
    - File extensions must match expected formats for each component
    - Paths must be non-empty strings representing valid file locations
    - L_idx accepts multiple formats for compilation framework compatibility
    
    References:
    - Stage-3-DATA-COMPILATION: Output artifact specifications
    - Stage5-FOUNDATIONAL-DESIGN: Input compatibility matrix requirements
    """
    l_raw_path: str = Field(
        ..., 
        description="Path to L_raw normalized entities (Parquet format)"
    )
    l_rel_path: str = Field(
        ..., 
        description="Path to L_rel relationship graphs (GraphML format)"
    )
    l_idx_path: str = Field(
        ..., 
        description="Path to L_idx indices (multiple formats supported)"
    )

    @field_validator('l_raw_path')
    @classmethod
    def validate_l_raw_extension(cls, v: str) -> str:
        """Ensure L_raw file has correct Parquet extension."""
        if not v.lower().endswith('.parquet'):
            raise ValueError("L_raw file must have .parquet extension")
        return v

    @field_validator('l_rel_path')
    @classmethod
    def validate_l_rel_extension(cls, v: str) -> str:
        """Ensure L_rel file has correct GraphML extension."""
        if not v.lower().endswith('.graphml'):
            raise ValueError("L_rel file must have .graphml extension")
        return v

    @field_validator('l_idx_path')
    @classmethod
    def validate_l_idx_format(cls, v: str) -> str:
        """Validate L_idx file format is supported by Stage 5.1."""
        supported_extensions = ['.pkl', '.parquet', '.feather', '.idx', '.bin']
        if not any(v.lower().endswith(ext) for ext in supported_extensions):
            raise ValueError(f"L_idx file must have one of extensions: {supported_extensions}")
        return v

class ExecutionContext(BaseModel):
    """
    Execution context for Stage 5.1 and 5.2 runner configurations.
    Provides path management and configuration overrides for different environments.
    
    Context Fields:
    - input_paths: Stage 3 output file locations (optional for Stage 5.2)
    - output_directory: Directory for Stage 5 output files and logs
    - config_overrides: Runtime configuration parameter overrides
    - execution_id: Unique identifier for this execution run (audit tracking)
    
    Usage Patterns:
    - Stage 5.1: Requires input_paths for Stage 3 data consumption
    - Stage 5.2: input_paths optional (uses Stage 5.1 output + capabilities file)
    - Both stages: Require output_directory and execution_id for results
    """
    input_paths: Optional[Stage3OutputPaths] = Field(
        None, 
        description="Stage 3 output file paths"
    )
    output_directory: str = Field(
        ..., 
        description="Output directory for Stage 5 results"
    )
    config_overrides: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Configuration parameter overrides"
    )
    execution_id: str = Field(
        ..., 
        description="Unique execution identifier for audit tracking"
    )

    @field_validator('execution_id')
    @classmethod
    def validate_execution_id_format(cls, v: str) -> str:
        """Ensure execution ID follows reasonable format for audit tracking."""
        if not v or len(v) < 8:
            raise ValueError("Execution ID must be at least 8 characters")
        if len(v) > 100:  # Reasonable upper bound for ID length
            raise ValueError("Execution ID must be ≤100 characters")
        return v

# Export all schema classes for module imports
__all__ = [
    # Enumerations
    'SolverParadigmEnum', 'LPConvergenceStatus', 'FileFormatEnum',
    
    # Stage 5.1 Schemas
    'ExecutionMetadata', 'EntityCounts', 'ComputationNotes', 
    'ComplexityParameters', 'ComplexityMetricsSchema',
    
    # Stage 5.2 Schemas
    'SolverLimits', 'SolverCapability', 'SolverArsenalSchema',
    'SolverChoice', 'SolverRanking', 'LPConvergenceInfo',
    'OptimizationDetails', 'SelectionResult', 'SolverSelectionSchema',
    
    # Shared Schemas
    'Stage3OutputPaths', 'ExecutionContext'
]

print("✅ STAGE 5 COMMON/SCHEMA.PY - COMPLETE (Part 3/3)")
print("   - SolverSelectionSchema with complete Stage 5.2 output specification")
print("   - OptimizationDetails with L2 normalization and LP convergence tracking")
print("   - Stage3OutputPaths and ExecutionContext for input/output management")
print("   - All schemas validated with enterprise-grade mathematical bounds checking")
print("   - 100% Pydantic V2 compliance with field_validator and model_validator decorators")
print(f"   - Total schema classes exported: {len(__all__)}")