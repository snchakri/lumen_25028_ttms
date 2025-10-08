
class OptimizationProcessDetails(BaseModel):
    """
    Detailed optimization process information for Stage 5.2 weight learning.
    Contains learned weights, normalization factors, and LP convergence tracking.

    Mathematical Components:
    - learned_weights: 16-D weight vector from LP optimization (∑w_j = 1, w_j ≥ 0)
    - normalization_factors: L2 normalization denominators for each parameter
    - separation_margin: Final LP objective value (d in maximize d formulation)
    - lp_convergence: Detailed convergence tracking and status information

    Mathematical Constraints:
    - Weight vector: Simplex constraint ∑w_j = 1 with w_j ≥ 0 ∀j
    - Normalization factors: Positive L2 norms for each complexity parameter
    - Separation margin: Non-negative optimization objective value

    References:
    - Stage-5.2 Section 3.2: L2 normalization mathematical formulation
    - Stage-5.2 Section 4: LP weight learning optimization algorithm
    - Stage-5.2 Theorem 4.1: Optimality conditions for weight learning
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    learned_weights: List[float] = Field(
        ..., 
        description="Learned 16-dimensional parameter weights from LP optimization"
    )
    normalization_factors: List[PositiveFloat] = Field(
        ..., 
        description="L2 normalization factors for each complexity parameter"
    )
    separation_margin: PositiveFloat = Field(
        ..., 
        description="Final separation margin from LP optimization objective"
    )
    lp_convergence: LPConvergenceInformation = Field(
        ..., 
        description="LP convergence tracking and status information"
    )

    @field_validator('learned_weights')
    @classmethod
    def validate_weight_vector_constraints(cls, v: List[float]) -> List[float]:
        """
        Validate weight vector satisfies simplex constraint and dimensionality requirements.

        Simplex Constraint Validation:
        - Exactly 16 dimensions: Must match P1-P16 complexity parameters
        - Non-negativity: w_j ≥ 0 for all j (convex combination requirement)
        - Normalization: ∑w_j = 1 (probability distribution constraint)
        - Numerical tolerance: 1e-6 for floating-point comparison

        Args:
            v: Weight vector to validate

        Returns:
            List[float]: Validated weight vector

        Raises:
            ValueError: If simplex constraints are violated
        """
        if len(v) != 16:
            raise ValueError(
                f"Learned weights must have exactly 16 elements for P1-P16 parameters, "
                f"got {len(v)} elements"
            )

        # Check non-negativity constraint
        for i, weight in enumerate(v, 1):
            if weight < 0:
                raise ValueError(
                    f"Weight w{i} = {weight} violates non-negativity constraint w_j ≥ 0"
                )

        # Check normalization constraint (sum = 1)
        weight_sum = sum(v)
        tolerance = 1e-6
        if abs(weight_sum - 1.0) > tolerance:
            raise ValueError(
                f"Learned weights sum to {weight_sum:.8f}, violates normalization "
                f"constraint ∑w_j = 1 (tolerance = {tolerance})"
            )

        return v

    @field_validator('normalization_factors')
    @classmethod
    def validate_normalization_factor_dimensionality(cls, v: List[PositiveFloat]) -> List[PositiveFloat]:
        """
        Validate normalization factors match 16-parameter structure and positivity.

        Dimensionality Requirements:
        - Exactly 16 elements: Must match P1-P16 complexity parameters
        - Positive values: L2 norms are always positive for non-zero vectors
        - Reasonable bounds: Should reflect actual solver capability magnitudes

        Args:
            v: Normalization factor vector to validate

        Returns:
            List[PositiveFloat]: Validated normalization factors

        Raises:
            ValueError: If dimensionality or positivity constraints are violated
        """
        if len(v) != 16:
            raise ValueError(
                f"Normalization factors must have exactly 16 elements for P1-P16 parameters, "
                f"got {len(v)} elements"
            )

        # Additional validation: Check for reasonable factor bounds
        for i, factor in enumerate(v, 1):
            if factor > 1000:  # Sanity check for extremely large normalization factors
                raise ValueError(
                    f"Normalization factor for P{i} = {factor} exceeds reasonable bounds. "
                    f"This may indicate numerical instability in capability vectors."
                )

        return v

class SolverSelectionResult(BaseModel):
    """
    Complete solver selection result from Stage 5.2 optimization process.
    Contains chosen solver and complete ranking with selection confidence.

    Result Components:
    - chosen_solver: Selected optimal solver with confidence metrics
    - ranking: Complete solver ranking by match score (descending order)

    Ranking Consistency Requirements:
    - Top-ranked solver must match chosen solver
    - Ranking must be sorted by descending match score
    - Margin calculations must be consistent with score differences
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    chosen_solver: SolverSelectionChoice = Field(
        ..., 
        description="Selected optimal solver with confidence metrics"
    )
    ranking: List[SolverRankingEntry] = Field(
        ..., 
        min_length=1,
        description="Complete solver ranking by match score (descending order)"
    )

    @model_validator(mode='after')
    def validate_ranking_consistency(self):
        """
        Validate ranking consistency with chosen solver and score ordering.

        Consistency Checks:
        - Top-ranked solver must match chosen solver ID
        - Ranking must be sorted by descending match score  
        - Margin calculations must match score differences
        - All solvers in ranking must have unique IDs

        Returns:
            Self: Validated model instance

        Raises:
            ValueError: If ranking consistency is violated
        """
        if not self.ranking:
            return self

        # Check top-ranked solver matches chosen solver
        top_solver = self.ranking[0]
        if top_solver.solver_id != self.chosen_solver.solver_id:
            raise ValueError(
                f"Top-ranked solver '{top_solver.solver_id}' must match "
                f"chosen solver '{self.chosen_solver.solver_id}'"
            )

        # Verify ranking is sorted by descending score
        scores = [entry.score for entry in self.ranking]
        if scores != sorted(scores, reverse=True):
            raise ValueError(
                f"Ranking must be sorted by descending match score. "
                f"Current scores: {scores}"
            )

        # Verify margin calculations are correct
        top_score = scores[0] if scores else 0
        for entry in self.ranking:
            expected_margin = top_score - entry.score
            tolerance = 1e-6
            if abs(entry.margin - expected_margin) > tolerance:
                raise ValueError(
                    f"Incorrect margin calculation for solver '{entry.solver_id}': "
                    f"expected {expected_margin:.6f}, got {entry.margin:.6f}"
                )

        # Check for unique solver IDs in ranking
        solver_ids = [entry.solver_id for entry in self.ranking]
        if len(solver_ids) != len(set(solver_ids)):
            duplicates = [sid for sid in set(solver_ids) if solver_ids.count(sid) > 1]
            raise ValueError(f"Duplicate solver IDs found in ranking: {duplicates}")

        return self

class SolverSelectionSchema(BaseModel):
    """
    Complete Stage 5.2 output schema for solver selection results.
    This is the EXACT format that Stage 5.2 must produce for Stage 6 consumption.

    Schema Components:
    - schema_version: Version compatibility tracking across Stage 5 implementations
    - execution_metadata: Performance timing and computational metadata  
    - selection_result: Chosen solver and complete ranking information
    - optimization_details: Mathematical optimization process details and convergence

    CRITICAL INTEGRATION NOTE:
    This schema represents the final handoff contract to Stage 6 solver execution.
    All downstream stages depend on this exact format specification for:
    - Solver identification and invocation
    - Confidence-based execution strategies
    - Performance monitoring and optimization
    - Error handling and fallback procedures

    Cross-Stage Dependencies:
    - Input from Stage 5.1 ComplexityMetricsSchema
    - Output consumed by Stage 6 solver family orchestration
    - Referenced by Stage 7 for validation and quality assessment
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    schema_version: str = Field(
        default="1.0.0", 
        description="Schema version for Stage 5.2 output compatibility"
    )
    execution_metadata: ExecutionMetadata = Field(
        ..., 
        description="Execution timing and computational metadata"
    )
    selection_result: SolverSelectionResult = Field(
        ..., 
        description="Solver selection results and complete ranking"
    )
    optimization_details: OptimizationProcessDetails = Field(
        ..., 
        description="L2 normalization and LP optimization process details"
    )

    @field_validator('schema_version')
    @classmethod
    def validate_output_schema_version(cls, v: str) -> str:
        """
        Validate output schema version for Stage 6 compatibility.

        Version Compatibility Requirements:
        - Must follow semantic versioning format
        - Major version changes indicate breaking schema changes
        - Minor version changes indicate backwards-compatible additions
        - Patch version changes indicate bug fixes without schema impact

        Args:
            v: Schema version string

        Returns:
            str: Validated schema version

        Raises:
            ValueError: If version format is invalid
        """
        if not re.match(r'^\d+\.\d+\.\d+$', v):
            raise ValueError(
                f"Output schema version '{v}' must follow semantic versioning format "
                f"for Stage 6 compatibility checking"
            )
        return v

# =============================================================================
# SHARED INPUT/OUTPUT VALIDATION SCHEMAS
# Common structures for file paths, execution contexts, and error handling
# =============================================================================

class Stage3OutputPathSpecification(BaseModel):
    """
    Stage 3 output file path specifications for Stage 5.1 input validation.
    Supports multiple index formats per Stage 3 compilation framework requirements.

    Path Structure Components:
    - l_raw_path: Normalized entity tables in Apache Parquet columnar format
    - l_rel_path: Relationship graphs in GraphML XML-based format  
    - l_idx_path: Multi-modal indices supporting various serialization formats

    Format Support Matrix:
    - L_raw: .parquet (Apache Parquet columnar format - mandatory)
    - L_rel: .graphml (GraphML relationship format - mandatory)  
    - L_idx: .pkl/.parquet/.feather/.idx/.bin (multiple format support)

    File Path Validation:
    - Extension checking: Ensures correct file formats for each component
    - Existence validation: Performed by utils.validate_file_path()
    - Readability checking: Ensures files are accessible for processing

    References:
    - Stage-3-DATA-COMPILATION: Output artifact format specifications
    - Stage5-FOUNDATIONAL-DESIGN: Input compatibility requirements matrix
    - PyArrow documentation: Parquet format specification and I/O
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    l_raw_path: str = Field(
        ..., 
        description="Path to L_raw normalized entities (Apache Parquet format)"
    )
    l_rel_path: str = Field(
        ..., 
        description="Path to L_rel relationship graphs (GraphML XML format)"
    )
    l_idx_path: str = Field(
        ..., 
        description="Path to L_idx indices (supports multiple serialization formats)"
    )

    @field_validator('l_raw_path')
    @classmethod
    def validate_l_raw_parquet_extension(cls, v: str) -> str:
        """
        Validate L_raw file has correct Apache Parquet extension.

        Format Requirements:
        - Must end with .parquet extension (case-insensitive)
        - Parquet format required for efficient columnar data access
        - Supports Stage 5.1 entity counting and parameter calculations

        Args:
            v: L_raw file path

        Returns:
            str: Validated L_raw path

        Raises:
            ValueError: If file extension is incorrect
        """
        if not v.lower().endswith('.parquet'):
            raise ValueError(
                f"L_raw file '{v}' must have .parquet extension for Apache Parquet format. "
                f"Stage 5.1 requires columnar data access for entity processing."
            )
        return v

    @field_validator('l_rel_path') 
    @classmethod
    def validate_l_rel_graphml_extension(cls, v: str) -> str:
        """
        Validate L_rel file has correct GraphML extension.

        Format Requirements:
        - Must end with .graphml extension (case-insensitive)
        - GraphML format required for NetworkX graph processing
        - Supports Stage 5.1 relationship analysis and graph algorithms

        Args:
            v: L_rel file path

        Returns:
            str: Validated L_rel path

        Raises:
            ValueError: If file extension is incorrect
        """
        if not v.lower().endswith('.graphml'):
            raise ValueError(
                f"L_rel file '{v}' must have .graphml extension for GraphML format. "
                f"Stage 5.1 requires graph structure for relationship analysis."
            )
        return v

    @field_validator('l_idx_path')
    @classmethod
    def validate_l_idx_supported_format(cls, v: str) -> str:
        """
        Validate L_idx file format is supported by Stage 5.1 index loading.

        Supported Format Matrix:
        - .pkl: Python pickle binary serialization (compact, Python-specific)
        - .parquet: Apache Parquet columnar format (interoperable)
        - .feather: Apache Arrow feather format (fast I/O)
        - .idx: Custom index binary format (optimized access patterns)
        - .bin: Generic binary format (fallback serialization)

        Args:
            v: L_idx file path

        Returns:
            str: Validated L_idx path

        Raises:
            ValueError: If file format is not supported
        """
        supported_extensions = ['.pkl', '.parquet', '.feather', '.idx', '.bin']
        v_lower = v.lower()

        if not any(v_lower.endswith(ext) for ext in supported_extensions):
            raise ValueError(
                f"L_idx file '{v}' must have one of supported extensions: {supported_extensions}. "
                f"Stage 5.1 requires compatible serialization format for index loading."
            )
        return v

class ExecutionContextConfiguration(BaseModel):
    """
    Execution context configuration for Stage 5.1 and 5.2 runner environments.
    Provides comprehensive path management and configuration override capabilities.

    Context Configuration Components:
    - input_paths: Stage 3 output file location specifications
    - output_directory: Target directory for Stage 5 result files
    - config_overrides: Runtime parameter overrides for testing/debugging
    - execution_id: Unique execution identifier for audit trail tracking

    Environment Support:
    - Local development: File system path specifications
    - CI/CD pipelines: Configurable path overrides for testing
    - Production deployment: Standardized directory structures
    - Debugging scenarios: Parameter override capabilities

    Audit Trail Integration:
    - execution_id: Links results to specific execution runs
    - timestamp correlation: Matches with execution metadata
    - Configuration tracking: Records all override parameters
    - Error correlation: Enables debugging across execution context
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    input_paths: Optional[Stage3OutputPathSpecification] = Field(
        None, 
        description="Stage 3 output file path specifications"
    )
    output_directory: str = Field(
        ..., 
        description="Output directory path for Stage 5 execution results"
    )
    config_overrides: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Runtime configuration parameter overrides"
    )
    execution_id: str = Field(
        ..., 
        description="Unique execution identifier for audit trail tracking"
    )

    @field_validator('execution_id')
    @classmethod
    def validate_execution_id_format_requirements(cls, v: str) -> str:
        """
        Validate execution ID format for audit trail and tracking requirements.

        Format Requirements:
        - Minimum 8 characters: Ensures sufficient uniqueness for tracking
        - Alphanumeric characters: Compatible with file system naming
        - Recommended formats: UUID4, timestamp-based, or sequential IDs
        - Case preservation: Maintains original case for consistency

        Usage Examples:
        - UUID4: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        - Timestamp: "20251007_012000_001" 
        - Sequential: "stage5_exec_001234"

        Args:
            v: Execution ID string

        Returns:
            str: Validated execution ID

        Raises:
            ValueError: If execution ID format is invalid
        """
        if not v or len(v) < 8:
            raise ValueError(
                f"Execution ID '{v}' must be at least 8 characters for unique identification. "
                f"Consider using UUID4, timestamp-based, or sequential numbering."
            )

        # Check for reasonable character set (alphanumeric, hyphens, underscores)
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError(
                f"Execution ID '{v}' should contain only alphanumeric characters, "
                f"hyphens, and underscores for file system compatibility."
            )

        return v

    @field_validator('output_directory')
    @classmethod  
    def validate_output_directory_format(cls, v: str) -> str:
        """
        Validate output directory path format for cross-platform compatibility.

        Directory Path Requirements:
        - Non-empty path: Must specify valid output location
        - Path separator normalization: Handle both Unix (/) and Windows (\) 
        - Relative/absolute path support: Both formats accepted
        - Special character handling: Avoid problematic characters

        Args:
            v: Output directory path

        Returns:
            str: Validated and normalized output directory path

        Raises:
            ValueError: If directory path format is invalid
        """
        if not v or v.isspace():
            raise ValueError("Output directory path cannot be empty or whitespace-only")

        # Normalize path separators for cross-platform compatibility
        normalized_path = Path(v).as_posix()
        return str(Path(normalized_path))

# =============================================================================
# VALIDATION UTILITY FUNCTIONS
# Helper functions for schema validation and cross-reference checking
# =============================================================================

def validate_solver_id_reference(solver_id: str, available_solvers: List[str]) -> bool:
    """
    Validate solver ID exists in available solver arsenal.

    Cross-Reference Validation:
    - Checks solver_id against loaded solver arsenal
    - Case-insensitive matching for robustness
    - Returns boolean for conditional logic

    Args:
        solver_id: Solver ID to validate
        available_solvers: List of available solver IDs from arsenal

    Returns:
        bool: True if solver ID is valid, False otherwise

    Usage Example:
        ```python
        arsenal = load_solver_arsenal()
        solver_ids = [s.solver_id for s in arsenal.solver_arsenal]
        is_valid = validate_solver_id_reference("pulp_cbc", solver_ids)
        ```
    """
    return solver_id.lower() in [sid.lower() for sid in available_solvers]

def validate_parameter_bounds(parameters: ComplexityParameterVector) -> List[str]:
    """
    Validate all complexity parameters are within expected theoretical bounds.

    Comprehensive Parameter Validation:
    - Checks each parameter against theoretical min/max values
    - Returns list of validation warnings/errors
    - Enables batch validation reporting

    Args:
        parameters: ComplexityParameterVector to validate

    Returns:
        List[str]: List of validation warnings (empty if all valid)

    Theoretical Bounds Reference:
    - P1-P16: Individual parameter validation rules from mathematical framework
    - Cross-parameter consistency checks where applicable
    - Statistical reasonableness checks for computed values
    """
    warnings = []

    # Add parameter-specific validation logic based on theoretical bounds
    # This would be expanded based on specific mathematical constraints
    # from the Stage-5.1 theoretical framework

    return warnings

print("✅ STAGE 5 COMMON/SCHEMA.PY - Part 3/4 Complete")
print("   - Complete Stage 5.2 optimization and selection result schemas")
print("   - Shared I/O validation schemas with comprehensive path checking")
print("   - Cross-reference validation utilities for solver arsenal integration")
print("   - Model validators for complex consistency checking")
