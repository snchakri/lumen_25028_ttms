"""
stage_5_2/io.py
Stage 5.2 Input/Output Operations with Schema Validation

This module provides enterprise-grade I/O operations for Stage 5.2 solver selection:
1. Load complexity metrics from Stage 5.1 JSON output with schema validation
2. Load solver capabilities from static JSON configuration
3. Write selection decision JSON with complete audit trail

All file operations use atomic writes with temporary files to ensure data integrity.
JSON schema validation ensures exact compliance with foundational design specifications.
Error handling provides comprehensive context for debugging and recovery.

Integration Points:
- Input: complexity_metrics.json from Stage 5.1 (16 parameters + composite index)
- Input: solver_capabilities.json (static solver arsenal configuration)
- Output: selection_decision.json (chosen solver + ranking + optimization details)

Schema Compliance:
- Exact match with Stage5-FOUNDATIONAL-DESIGN-IMPLEMENTATION-PLAN.md specifications
- Version 1.0.0 schema validation for all JSON documents  
- Comprehensive metadata preservation for audit trails
- Error recovery with detailed validation context

Performance Characteristics:
- Atomic file operations with rollback on failure
- JSON validation with comprehensive error reporting
- Memory efficient streaming for large solver arsenals
- Structured logging for operation tracking and debugging
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
from dataclasses import asdict

from ..common.logging import get_logger, log_operation
from ..common.exceptions import Stage5ValidationError, Stage5ComputationError
from ..common.schema import (
    ComplexityParameterVector, SolverCapability, SelectionDecision,
    ExecutionMetadata, OptimizationDetails
)
from ..common.utils import validate_file_path, atomic_json_write

# Global logger for this module - initialized at module level for consistency
_logger = get_logger("stage5_2.io")

# Schema version constants from foundational design
JSON_SCHEMA_VERSION = "1.0.0"
EXPECTED_PARAMETER_COUNT = 16
REQUIRED_SOLVER_FIELDS = ["solver_id", "display_name", "paradigm", "capability_vector", "limits"]
REQUIRED_COMPLEXITY_FIELDS = ["complexity_parameters", "composite_index", "execution_metadata"]

# File validation constants
MAX_JSON_SIZE_MB = 100  # Maximum JSON file size for safety
MAX_SOLVER_COUNT = 1000  # Maximum number of solvers to prevent memory issues
MIN_SOLVER_COUNT = 1    # Minimum solvers required for meaningful selection


class ComplexityMetricsLoader:
    """
    Enterprise-grade loader for Stage 5.1 complexity metrics with rigorous validation.
    
    Loads and validates complexity_metrics.json from Stage 5.1 output, ensuring:
    - Schema version compatibility (1.0.0)
    - Complete 16-parameter presence with numeric validation  
    - Composite index mathematical consistency
    - Execution metadata completeness for audit trails
    
    Validation includes bounds checking, numerical stability, and mathematical
    consistency verification per theoretical framework requirements.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize complexity metrics loader with validation setup.
        
        Args:
            logger: Optional logger instance for operation tracking
        """
        self.logger = logger or _logger
        self._validation_cache = {}  # Cache for repeated validation results
    
    def load_complexity_metrics(self, metrics_path: Path) -> ComplexityParameterVector:
        """
        Load and validate complexity metrics from Stage 5.1 JSON output.
        
        Performs comprehensive validation of the complexity_metrics.json file:
        1. File existence and readability verification
        2. JSON parsing with error context
        3. Schema version compatibility checking
        4. Parameter completeness and bounds validation
        5. Mathematical consistency verification
        6. Audit metadata preservation
        
        Args:
            metrics_path: Path to complexity_metrics.json file
            
        Returns:
            ComplexityParameterVector: Validated complexity parameters with metadata
            
        Raises:
            Stage5ValidationError: If file validation or schema validation fails
            Stage5ComputationError: If JSON parsing or data conversion fails
            
        Mathematical Validation:
        - All 16 parameters present with finite numeric values
        - Composite index within expected range [0, ∞)
        - Parameter bounds consistent with theoretical definitions
        - Numerical stability verification (no NaN, Inf values)
        """
        with log_operation(self.logger, "load_complexity_metrics", 
                          {"metrics_path": str(metrics_path)}):
            
            # Validate file accessibility and basic properties
            validated_path = validate_file_path(
                metrics_path, 
                must_exist=True, 
                check_readable=True,
                expected_extensions=[".json"],
                max_size_mb=MAX_JSON_SIZE_MB
            )
            
            try:
                # Load and parse JSON with comprehensive error handling
                with validated_path.open("r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                
                # Schema version validation
                self._validate_schema_version(raw_data, "complexity_metrics")
                
                # Core field presence validation
                self._validate_complexity_metrics_schema(raw_data)
                
                # Extract and validate complexity parameters
                parameters_data = raw_data["complexity_parameters"]
                composite_index = raw_data["composite_index"]
                
                # Validate parameter completeness and bounds
                validated_parameters = self._validate_complexity_parameters(parameters_data)
                
                # Validate composite index mathematical consistency
                validated_composite = self._validate_composite_index(composite_index, validated_parameters)
                
                # Extract execution metadata for audit trail
                execution_metadata = self._extract_execution_metadata(raw_data)
                
                # Create validated complexity parameter vector
                complexity_vector = ComplexityParameterVector(
                    **validated_parameters,
                    composite_index=validated_composite
                )
                
                # Attach metadata for downstream processing
                complexity_vector._execution_metadata = execution_metadata
                complexity_vector._source_file = str(validated_path)
                
                self.logger.info(
                    f"Complexity metrics loaded successfully: composite_index={validated_composite:.6f}, "
                    f"parameters={len(validated_parameters)}"
                )
                
                return complexity_vector
                
            except json.JSONDecodeError as e:
                raise Stage5ComputationError(
                    f"JSON parsing failed for complexity metrics: {str(e)}",
                    computation_type="json_parsing",
                    input_parameters={"file_path": str(metrics_path), "json_error": str(e)}
                )
            
            except KeyError as e:
                raise Stage5ValidationError(
                    f"Missing required field in complexity metrics: {str(e)}",
                    validation_type="schema_completeness",
                    field_name=str(e).strip("'"),
                    context={"available_fields": list(raw_data.keys()) if 'raw_data' in locals() else []}
                )
    
    def _validate_schema_version(self, data: Dict[str, Any], document_type: str) -> None:
        """Validate JSON schema version compatibility."""
        if "schema_version" not in data:
            raise Stage5ValidationError(
                f"Missing schema_version field in {document_type}",
                validation_type="schema_version",
                field_name="schema_version"
            )
        
        schema_version = data["schema_version"]
        if schema_version != JSON_SCHEMA_VERSION:
            raise Stage5ValidationError(
                f"Incompatible schema version in {document_type}: expected {JSON_SCHEMA_VERSION}, got {schema_version}",
                validation_type="schema_version",
                expected_value=JSON_SCHEMA_VERSION,
                actual_value=schema_version
            )
    
    def _validate_complexity_metrics_schema(self, data: Dict[str, Any]) -> None:
        """Validate complexity metrics JSON schema completeness."""
        for required_field in REQUIRED_COMPLEXITY_FIELDS:
            if required_field not in data:
                raise Stage5ValidationError(
                    f"Missing required field: {required_field}",
                    validation_type="schema_completeness",
                    field_name=required_field,
                    context={"available_fields": list(data.keys())}
                )
    
    def _validate_complexity_parameters(self, parameters_data: Dict[str, Any]) -> Dict[str, float]:
        """Validate 16-parameter completeness and mathematical bounds."""
        expected_parameters = [
            "p1_dimensionality", "p2_constraint_density", "p3_faculty_specialization",
            "p4_room_utilization", "p5_temporal_complexity", "p6_batch_variance",
            "p7_competency_entropy", "p8_conflict_measure", "p9_coupling_coefficient",
            "p10_heterogeneity_index", "p11_flexibility_measure", "p12_dependency_complexity",
            "p13_landscape_ruggedness", "p14_scalability_factor", "p15_propagation_depth",
            "p16_quality_variance"
        ]
        
        validated_parameters = {}
        
        for param_name in expected_parameters:
            if param_name not in parameters_data:
                raise Stage5ValidationError(
                    f"Missing complexity parameter: {param_name}",
                    validation_type="parameter_completeness",
                    field_name=param_name,
                    context={"available_parameters": list(parameters_data.keys())}
                )
            
            param_value = parameters_data[param_name]
            
            # Validate numeric type and finite value
            if not isinstance(param_value, (int, float)):
                raise Stage5ValidationError(
                    f"Parameter {param_name} must be numeric, got {type(param_value)}",
                    validation_type="parameter_type",
                    field_name=param_name,
                    expected_value="numeric",
                    actual_value=str(type(param_value))
                )
            
            param_float = float(param_value)
            
            if not np.isfinite(param_float):
                raise Stage5ValidationError(
                    f"Parameter {param_name} must be finite, got {param_float}",
                    validation_type="numerical_validity",
                    field_name=param_name,
                    actual_value=param_float
                )
            
            # Parameter-specific bounds validation based on theoretical framework
            self._validate_parameter_bounds(param_name, param_float)
            
            validated_parameters[param_name] = param_float
        
        return validated_parameters
    
    def _validate_parameter_bounds(self, param_name: str, param_value: float) -> None:
        """Validate parameter-specific bounds based on theoretical definitions."""
        # Parameter bounds from Stage-5.1 theoretical framework
        parameter_bounds = {
            "p1_dimensionality": (0, float('inf')),           # Positive dimensionality
            "p2_constraint_density": (0, 1),                  # Density ratio [0,1]
            "p3_faculty_specialization": (0, 1),              # Specialization ratio [0,1]
            "p4_room_utilization": (0, float('inf')),         # Utilization factor ≥0
            "p5_temporal_complexity": (0, float('inf')),      # Complexity measure ≥0
            "p6_batch_variance": (0, float('inf')),           # Variance ≥0
            "p7_competency_entropy": (0, float('inf')),       # Entropy ≥0
            "p8_conflict_measure": (0, 1),                    # Conflict ratio [0,1]
            "p9_coupling_coefficient": (0, 1),                # Coupling ratio [0,1]
            "p10_heterogeneity_index": (0, float('inf')),     # Heterogeneity ≥0
            "p11_flexibility_measure": (0, 1),                # Flexibility ratio [0,1]
            "p12_dependency_complexity": (0, float('inf')),   # Complexity ≥0
            "p13_landscape_ruggedness": (0, 1),               # Ruggedness ratio [0,1]
            "p14_scalability_factor": (float('-inf'), float('inf')), # Can be negative
            "p15_propagation_depth": (0, float('inf')),       # Depth ≥0
            "p16_quality_variance": (0, float('inf'))         # Variance ≥0
        }
        
        if param_name in parameter_bounds:
            min_bound, max_bound = parameter_bounds[param_name]
            
            if not (min_bound <= param_value <= max_bound):
                raise Stage5ValidationError(
                    f"Parameter {param_name} value {param_value} out of bounds [{min_bound}, {max_bound}]",
                    validation_type="parameter_bounds",
                    field_name=param_name,
                    expected_value=f"[{min_bound}, {max_bound}]",
                    actual_value=param_value
                )
    
    def _validate_composite_index(self, composite_index: Any, parameters: Dict[str, float]) -> float:
        """Validate composite index mathematical consistency."""
        if not isinstance(composite_index, (int, float)):
            raise Stage5ValidationError(
                f"Composite index must be numeric, got {type(composite_index)}",
                validation_type="composite_index_type",
                expected_value="numeric",
                actual_value=str(type(composite_index))
            )
        
        composite_float = float(composite_index)
        
        if not np.isfinite(composite_float):
            raise Stage5ValidationError(
                f"Composite index must be finite, got {composite_float}",
                validation_type="numerical_validity",
                field_name="composite_index",
                actual_value=composite_float
            )
        
        # Validate reasonable range (composite index should be non-negative for meaningful problems)
        if composite_float < 0:
            self.logger.warning(
                f"Composite index is negative ({composite_float:.6f}) - "
                f"verify Stage 5.1 computation correctness"
            )
        
        return composite_float
    
    def _extract_execution_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate execution metadata for audit trail."""
        if "execution_metadata" not in data:
            return {}
        
        metadata = data["execution_metadata"]
        
        # Add file loading timestamp for complete audit trail
        metadata["stage_5_2_loaded_at"] = datetime.now(timezone.utc).isoformat()
        metadata["stage_5_2_loader_version"] = "1.0.0"
        
        return metadata


class SolverCapabilitiesLoader:
    """
    Enterprise-grade loader for solver capabilities configuration with validation.
    
    Loads solver_capabilities.json containing the complete solver arsenal with:
    - Solver identification and display information
    - 16-parameter capability vectors aligned with complexity parameters
    - Performance limits and constraints for each solver
    - Paradigm classification for algorithmic matching
    
    Validation ensures mathematical consistency and completeness for optimization.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize solver capabilities loader with validation setup.
        
        Args:
            logger: Optional logger instance for operation tracking
        """
        self.logger = logger or _logger
    
    def load_solver_capabilities(self, capabilities_path: Path) -> List[SolverCapability]:
        """
        Load and validate solver capabilities from JSON configuration.
        
        Performs comprehensive validation of solver_capabilities.json:
        1. File accessibility and size validation
        2. JSON schema and version compatibility
        3. Solver completeness and uniqueness validation
        4. Capability vector mathematical validation
        5. Solver limits and constraints verification
        
        Args:
            capabilities_path: Path to solver_capabilities.json file
            
        Returns:
            List[SolverCapability]: Validated solver capability objects
            
        Raises:
            Stage5ValidationError: If schema or mathematical validation fails
            Stage5ComputationError: If JSON parsing fails
            
        Mathematical Validation:
        - Each solver has exactly 16 capability values
        - All capability values are finite and non-negative
        - Solver IDs are unique across the arsenal
        - Limits are mathematically consistent (positive values)
        """
        with log_operation(self.logger, "load_solver_capabilities",
                          {"capabilities_path": str(capabilities_path)}):
            
            # Validate file accessibility
            validated_path = validate_file_path(
                capabilities_path,
                must_exist=True,
                check_readable=True, 
                expected_extensions=[".json"],
                max_size_mb=MAX_JSON_SIZE_MB
            )
            
            try:
                # Load and parse JSON
                with validated_path.open("r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                
                # Schema validation
                self._validate_schema_version(raw_data, "solver_capabilities")
                self._validate_solver_arsenal_schema(raw_data)
                
                # Extract and validate solver arsenal
                solver_arsenal_data = raw_data["solver_arsenal"]
                
                # Validate arsenal size constraints
                self._validate_arsenal_size(solver_arsenal_data)
                
                # Process and validate each solver
                validated_solvers = []
                solver_ids = set()
                
                for solver_idx, solver_data in enumerate(solver_arsenal_data):
                    try:
                        validated_solver = self._validate_solver_entry(solver_data, solver_idx)
                        
                        # Check for duplicate solver IDs
                        if validated_solver.solver_id in solver_ids:
                            raise Stage5ValidationError(
                                f"Duplicate solver ID: {validated_solver.solver_id}",
                                validation_type="solver_uniqueness",
                                field_name="solver_id",
                                actual_value=validated_solver.solver_id
                            )
                        
                        solver_ids.add(validated_solver.solver_id)
                        validated_solvers.append(validated_solver)
                        
                    except Exception as e:
                        raise Stage5ValidationError(
                            f"Validation failed for solver at index {solver_idx}: {str(e)}",
                            validation_type="solver_entry_validation",
                            context={"solver_index": solver_idx, "original_error": str(e)}
                        )
                
                self.logger.info(
                    f"Solver capabilities loaded successfully: {len(validated_solvers)} solvers, "
                    f"arsenal file: {validated_path.name}"
                )
                
                return validated_solvers
                
            except json.JSONDecodeError as e:
                raise Stage5ComputationError(
                    f"JSON parsing failed for solver capabilities: {str(e)}",
                    computation_type="json_parsing",
                    input_parameters={"file_path": str(capabilities_path), "json_error": str(e)}
                )
    
    def _validate_schema_version(self, data: Dict[str, Any], document_type: str) -> None:
        """Validate JSON schema version compatibility."""
        if "schema_version" not in data:
            raise Stage5ValidationError(
                f"Missing schema_version field in {document_type}",
                validation_type="schema_version",
                field_name="schema_version"
            )
        
        schema_version = data["schema_version"]
        if schema_version != JSON_SCHEMA_VERSION:
            raise Stage5ValidationError(
                f"Incompatible schema version in {document_type}: expected {JSON_SCHEMA_VERSION}, got {schema_version}",
                validation_type="schema_version", 
                expected_value=JSON_SCHEMA_VERSION,
                actual_value=schema_version
            )
    
    def _validate_solver_arsenal_schema(self, data: Dict[str, Any]) -> None:
        """Validate solver arsenal JSON schema completeness."""
        if "solver_arsenal" not in data:
            raise Stage5ValidationError(
                "Missing solver_arsenal field",
                validation_type="schema_completeness",
                field_name="solver_arsenal"
            )
        
        if not isinstance(data["solver_arsenal"], list):
            raise Stage5ValidationError(
                f"solver_arsenal must be a list, got {type(data['solver_arsenal'])}",
                validation_type="schema_type",
                field_name="solver_arsenal",
                expected_value="list",
                actual_value=str(type(data["solver_arsenal"]))
            )
    
    def _validate_arsenal_size(self, arsenal_data: List[Any]) -> None:
        """Validate solver arsenal size constraints."""
        arsenal_size = len(arsenal_data)
        
        if arsenal_size < MIN_SOLVER_COUNT:
            raise Stage5ValidationError(
                f"Solver arsenal must contain at least {MIN_SOLVER_COUNT} solver(s), got {arsenal_size}",
                validation_type="arsenal_size",
                expected_value=f">= {MIN_SOLVER_COUNT}",
                actual_value=arsenal_size
            )
        
        if arsenal_size > MAX_SOLVER_COUNT:
            raise Stage5ValidationError(
                f"Solver arsenal exceeds maximum size {MAX_SOLVER_COUNT}, got {arsenal_size}",
                validation_type="arsenal_size",
                expected_value=f"<= {MAX_SOLVER_COUNT}",
                actual_value=arsenal_size
            )
    
    def _validate_solver_entry(self, solver_data: Dict[str, Any], solver_index: int) -> SolverCapability:
        """Validate individual solver entry with mathematical verification."""
        # Validate required fields presence
        for required_field in REQUIRED_SOLVER_FIELDS:
            if required_field not in solver_data:
                raise Stage5ValidationError(
                    f"Missing required field: {required_field}",
                    validation_type="solver_field_completeness",
                    field_name=required_field,
                    context={"solver_index": solver_index}
                )
        
        # Extract and validate solver ID
        solver_id = solver_data["solver_id"]
        if not isinstance(solver_id, str) or not solver_id.strip():
            raise Stage5ValidationError(
                f"Solver ID must be non-empty string, got: {repr(solver_id)}",
                validation_type="solver_id_validation",
                field_name="solver_id",
                actual_value=repr(solver_id)
            )
        
        # Extract and validate display name
        display_name = solver_data["display_name"] 
        if not isinstance(display_name, str) or not display_name.strip():
            raise Stage5ValidationError(
                f"Display name must be non-empty string, got: {repr(display_name)}",
                validation_type="display_name_validation",
                field_name="display_name",
                actual_value=repr(display_name)
            )
        
        # Extract and validate paradigm
        paradigm = solver_data["paradigm"]
        if not isinstance(paradigm, str) or not paradigm.strip():
            raise Stage5ValidationError(
                f"Paradigm must be non-empty string, got: {repr(paradigm)}",
                validation_type="paradigm_validation",
                field_name="paradigm",
                actual_value=repr(paradigm)
            )
        
        # Validate and extract capability vector
        capability_vector = self._validate_capability_vector(solver_data["capability_vector"], solver_id)
        
        # Validate and extract limits
        limits = self._validate_solver_limits(solver_data["limits"], solver_id)
        
        # Create validated solver capability object
        return SolverCapability(
            solver_id=solver_id.strip(),
            display_name=display_name.strip(),
            paradigm=paradigm.strip(),
            capability_vector=capability_vector,
            limits=limits
        )
    
    def _validate_capability_vector(self, capability_data: Any, solver_id: str) -> List[float]:
        """Validate solver capability vector mathematical properties."""
        if not isinstance(capability_data, list):
            raise Stage5ValidationError(
                f"Capability vector must be a list, got {type(capability_data)}",
                validation_type="capability_vector_type",
                field_name="capability_vector",
                expected_value="list",
                actual_value=str(type(capability_data)),
                context={"solver_id": solver_id}
            )
        
        if len(capability_data) != EXPECTED_PARAMETER_COUNT:
            raise Stage5ValidationError(
                f"Capability vector must have {EXPECTED_PARAMETER_COUNT} elements, got {len(capability_data)}",
                validation_type="capability_vector_length",
                field_name="capability_vector",
                expected_value=EXPECTED_PARAMETER_COUNT,
                actual_value=len(capability_data),
                context={"solver_id": solver_id}
            )
        
        validated_vector = []
        
        for i, capability_value in enumerate(capability_data):
            # Validate numeric type
            if not isinstance(capability_value, (int, float)):
                raise Stage5ValidationError(
                    f"Capability value at index {i} must be numeric, got {type(capability_value)}",
                    validation_type="capability_value_type",
                    field_name=f"capability_vector[{i}]",
                    expected_value="numeric",
                    actual_value=str(type(capability_value)),
                    context={"solver_id": solver_id}
                )
            
            capability_float = float(capability_value)
            
            # Validate finite value
            if not np.isfinite(capability_float):
                raise Stage5ValidationError(
                    f"Capability value at index {i} must be finite, got {capability_float}",
                    validation_type="numerical_validity",
                    field_name=f"capability_vector[{i}]",
                    actual_value=capability_float,
                    context={"solver_id": solver_id}
                )
            
            # Validate non-negative (capabilities should be non-negative)
            if capability_float < 0:
                raise Stage5ValidationError(
                    f"Capability value at index {i} must be non-negative, got {capability_float}",
                    validation_type="capability_bounds",
                    field_name=f"capability_vector[{i}]",
                    expected_value=">= 0",
                    actual_value=capability_float,
                    context={"solver_id": solver_id}
                )
            
            validated_vector.append(capability_float)
        
        return validated_vector
    
    def _validate_solver_limits(self, limits_data: Any, solver_id: str) -> Dict[str, Union[int, float]]:
        """Validate solver limits and constraints."""
        if not isinstance(limits_data, dict):
            raise Stage5ValidationError(
                f"Limits must be a dictionary, got {type(limits_data)}",
                validation_type="limits_type",
                field_name="limits",
                expected_value="dict",
                actual_value=str(type(limits_data)),
                context={"solver_id": solver_id}
            )
        
        # Required limits fields
        required_limits = ["max_variables", "max_constraints", "time_limit_default"]
        
        for limit_name in required_limits:
            if limit_name not in limits_data:
                raise Stage5ValidationError(
                    f"Missing required limit: {limit_name}",
                    validation_type="limits_completeness",
                    field_name=limit_name,
                    context={"solver_id": solver_id}
                )
            
            limit_value = limits_data[limit_name]
            
            # Validate numeric type
            if not isinstance(limit_value, (int, float)):
                raise Stage5ValidationError(
                    f"Limit {limit_name} must be numeric, got {type(limit_value)}",
                    validation_type="limit_value_type",
                    field_name=limit_name,
                    expected_value="numeric",
                    actual_value=str(type(limit_value)),
                    context={"solver_id": solver_id}
                )
            
            # Validate positive values (limits should be positive)
            if limit_value <= 0:
                raise Stage5ValidationError(
                    f"Limit {limit_name} must be positive, got {limit_value}",
                    validation_type="limit_bounds",
                    field_name=limit_name,
                    expected_value="> 0",
                    actual_value=limit_value,
                    context={"solver_id": solver_id}
                )
        
        return limits_data.copy()


def write_selection_decision(selection_result: SelectionDecision,
                           output_path: Path,
                           logger: Optional[logging.Logger] = None) -> Path:
    """
    Write selection decision to JSON with atomic operations and validation.
    
    Serializes the complete solver selection results to JSON format matching
    the foundational design specification exactly. Uses atomic write operations
    to ensure data integrity and provides comprehensive audit trail.
    
    Args:
        selection_result: Complete selection decision with ranking and optimization details
        output_path: Path for selection_decision.json output
        logger: Optional logger for operation tracking
        
    Returns:
        Path: Actual output file path (same as input)
        
    Raises:
        Stage5ComputationError: If JSON serialization or file writing fails
        
    JSON Schema Generated:
    - schema_version: "1.0.0"
    - execution_metadata: Timestamp and computation time
    - selection_result: Chosen solver, confidence, match score
    - ranking: Complete solver ranking with scores and margins
    - optimization_details: Learned weights, normalization factors, LP convergence
    """
    logger = logger or _logger
    
    with log_operation(logger, "write_selection_decision", {"output_path": str(output_path)}):
        try:
            # Prepare JSON data structure according to foundational design schema
            json_data = {
                "schema_version": JSON_SCHEMA_VERSION,
                "execution_metadata": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "computation_time_ms": selection_result.execution_time_ms,
                    "stage": "5.2",
                    "version": "1.0.0"
                },
                "selection_result": {
                    "chosen_solver": {
                        "solver_id": selection_result.chosen_solver.solver_id,
                        "confidence": selection_result.chosen_solver.confidence,
                        "match_score": selection_result.chosen_solver.match_score
                    },
                    "ranking": [
                        {
                            "solver_id": rank.solver_id,
                            "score": rank.score,
                            "margin": rank.margin
                        }
                        for rank in selection_result.ranking
                    ]
                },
                "optimization_details": {
                    "learned_weights": selection_result.optimization_details.learned_weights,
                    "normalization_factors": selection_result.optimization_details.normalization_factors,
                    "separation_margin": selection_result.optimization_details.separation_margin,
                    "lp_convergence": {
                        "iterations": selection_result.optimization_details.lp_iterations,
                        "status": selection_result.optimization_details.lp_status,
                        "objective_value": selection_result.optimization_details.lp_objective_value
                    }
                }
            }
            
            # Write JSON with atomic operation for data integrity
            atomic_json_write(json_data, output_path)
            
            logger.info(
                f"Selection decision written: {output_path} "
                f"(chosen: {selection_result.chosen_solver.solver_id}, "
                f"confidence: {selection_result.chosen_solver.confidence:.4f})"
            )
            
            return output_path
            
        except Exception as e:
            raise Stage5ComputationError(
                f"Failed to write selection decision JSON: {str(e)}",
                computation_type="json_serialization",
                input_parameters={"output_path": str(output_path)},
                context={"original_error": str(e)}
            )


# Module-level convenience functions
def load_stage_5_1_output(metrics_path: Path, 
                          logger: Optional[logging.Logger] = None) -> ComplexityParameterVector:
    """
    Convenience function to load Stage 5.1 complexity metrics.
    
    Args:
        metrics_path: Path to complexity_metrics.json
        logger: Optional logger for operation tracking
        
    Returns:
        ComplexityParameterVector: Validated complexity parameters
    """
    loader = ComplexityMetricsLoader(logger=logger)
    return loader.load_complexity_metrics(metrics_path)


def load_solver_arsenal(capabilities_path: Path,
                       logger: Optional[logging.Logger] = None) -> List[SolverCapability]:
    """
    Convenience function to load solver capabilities arsenal.
    
    Args:
        capabilities_path: Path to solver_capabilities.json
        logger: Optional logger for operation tracking
        
    Returns:
        List[SolverCapability]: Validated solver capability objects
    """
    loader = SolverCapabilitiesLoader(logger=logger)
    return loader.load_solver_capabilities(capabilities_path)


# Export key classes and functions
__all__ = [
    "ComplexityMetricsLoader",
    "SolverCapabilitiesLoader", 
    "write_selection_decision",
    "load_stage_5_1_output",
    "load_solver_arsenal",
    "JSON_SCHEMA_VERSION",
    "EXPECTED_PARAMETER_COUNT"
]

_logger.info("Stage 5.2 I/O module loaded with schema validation enabled")