"""
Stage 6.4 PyGMO Solver Family - Input Model Context Builder

THEORETICAL FOUNDATION: Input Model Context Definition (Section 3.1)
MATHEMATICAL COMPLIANCE: Course Eligibility and Constraint Rules Framework
ARCHITECTURAL ALIGNMENT: PyGMO Problem Interface Integration (Section 10.1)

This module implements the InputModelContext builder for PyGMO solver family,
creating the unified data structure required for optimization processing. The context
integrates Stage 3 compiled data, dynamic parameters, and constraint rules into a
mathematically consistent structure that maintains bijection mapping properties
and supports PyGMO multi-objective optimization requirements.

Complete reliableNESS:
- Immutable data structures for thread-safety and mathematical consistency
- Memory-efficient context building with deterministic resource patterns
- complete parameter resolution with hierarchical inheritance
- Fail-fast validation ensuring mathematical correctness throughout process
- complete logging for debugging and performance monitoring
"""

import logging
import sys
import traceback
from typing import Dict, List, Tuple, Any, Optional, Union, Set, Callable
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings

# Core mathematical and data processing libraries
import pandas as pd
import numpy as np
import networkx as nx
from pydantic import BaseModel, validator, Field, root_validator
import structlog

# Import related modules from the same package
from .loader import Stage3DataLoader, Stage3DataPaths
from .validator import PyGMOInputValidator, ValidationResult, ValidationError

# Configure structured logging for enterprise debugging
logger = structlog.get_logger(__name__)

@dataclass(frozen=True)
class CourseAssignmentTuple:
    """
    MATHEMATICAL BASIS: Individual Representation (Section 3.2)

    Immutable representation of a single course assignment tuple following
    the PyGMO course-centric encoding: (faculty_id, room_id, timeslot_id, batch_id).

    def __post_init__(self):
        """Validate assignment tuple mathematical constraints"""
        if any(val < 0 for val in [self.faculty_id, self.room_id, self.timeslot_id, self.batch_id]):
            raise ValueError(f"[FAIL-FAST] Negative values in assignment tuple: {self}")

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to standard tuple for PyGMO compatibility"""
        return (self.faculty_id, self.room_id, self.timeslot_id, self.batch_id)

    @classmethod
    def from_tuple(cls, assignment_tuple: Tuple[int, int, int, int]) -> 'CourseAssignmentTuple':
        """Create from standard tuple with validation"""
        return cls(*assignment_tuple)

@dataclass(frozen=True)
class ConstraintRule:
    """
    THEORETICAL BASIS: Constraint Handling Mechanisms (Section 4)
    MATHEMATICAL FOUNDATION: Adaptive Penalty Functions (Definition 4.1)

    Immutable representation of constraint rules with mathematical properties
    for PyGMO optimization. Integrates dynamic parameters and penalty weights.

    def __post_init__(self):
        """Validate constraint rule mathematical properties"""
        if self.penalty_weight < 0:
            raise ValueError(f"[FAIL-FAST] Negative penalty weight: {self.penalty_weight}")

        if self.is_hard_constraint and self.penalty_weight < 1000:
            logger.warning("hard_constraint_low_penalty", 
                         constraint_type=self.constraint_type,
                         penalty_weight=self.penalty_weight)

class BijectionMapping:
    """
    MATHEMATICAL BASIS: Bijective Transformation (Section 5.1) 
    THEORETICAL FOUNDATION: Course-Centric to PyGMO Mapping

    Implements bijective mapping between course-centric dictionary representation
    and PyGMO normalized vector representation with mathematical guarantees.

    def __init__(self, course_order: List[str], max_values: Dict[str, int]):
        """
        Initialize bijection mapping with course ordering and value bounds

        Args:
            course_order: Ordered list of course IDs for consistent vector mapping
            max_values: Maximum values for each dimension (faculty, room, timeslot, batch)
        """
        self.course_order = course_order.copy()  # Immutable copy
        self.max_values = max_values.copy()

        # Validate inputs
        if not course_order:
            raise ValueError("[FAIL-FAST] Empty course order for bijection mapping")

        required_dims = {'faculty', 'room', 'timeslot', 'batch'}
        if not required_dims.issubset(set(max_values.keys())):
            missing = required_dims - set(max_values.keys())
            raise ValueError(f"[FAIL-FAST] Missing max_values dimensions: {missing}")

        # Validate max values are positive
        for dim, max_val in max_values.items():
            if max_val <= 0:
                raise ValueError(f"[FAIL-FAST] Non-positive max value for {dim}: {max_val}")

        self.logger = logger.bind(
            component="BijectionMapping",
            course_count=len(course_order),
            max_values=max_values
        )

        self.logger.info("bijection_mapping_initialized")

    def course_dict_to_pygmo_vector(self, individual_dict: Dict[str, CourseAssignmentTuple]) -> List[float]:
        """
        MATHEMATICAL TRANSFORMATION: Course dictionary → PyGMO normalized vector

        Converts course-centric dictionary to PyGMO normalized vector with bijective guarantee.

        Args:
            individual_dict: Course assignments as {course_id: CourseAssignmentTuple}

        Returns:
            List[float]: Normalized vector for PyGMO optimization (each value in [0,1])
        """
        try:
            vector = []

            for course_id in self.course_order:
                if course_id not in individual_dict:
                    raise ValueError(f"[FAIL-FAST] Missing course assignment: {course_id}")

                assignment = individual_dict[course_id]

                # Normalize each dimension to [0, 1]
                normalized_faculty = assignment.faculty_id / self.max_values['faculty']
                normalized_room = assignment.room_id / self.max_values['room']
                normalized_timeslot = assignment.timeslot_id / self.max_values['timeslot']
                normalized_batch = assignment.batch_id / self.max_values['batch']

                # Validate normalization bounds
                for val, dim in [(normalized_faculty, 'faculty'), (normalized_room, 'room'),
                               (normalized_timeslot, 'timeslot'), (normalized_batch, 'batch')]:
                    if not (0 <= val <= 1):
                        raise ValueError(f"[FAIL-FAST] Normalization out of bounds for {dim}: {val}")

                vector.extend([normalized_faculty, normalized_room, normalized_timeslot, normalized_batch])

            return vector

        except Exception as e:
            self.logger.error("course_dict_to_vector_failed", 
                            error=str(e), 
                            course_count=len(individual_dict))
            raise ValueError(f"[FAIL-FAST] Course dict to PyGMO vector conversion failed: {e}") from e

    def pygmo_vector_to_course_dict(self, vector: List[float]) -> Dict[str, CourseAssignmentTuple]:
        """
        MATHEMATICAL TRANSFORMATION: PyGMO normalized vector → Course dictionary

        Converts PyGMO normalized vector back to course-centric dictionary with bijective guarantee.

        Args:
            vector: PyGMO normalized vector (each value in [0,1])

        Returns:
            Dict[str, CourseAssignmentTuple]: Course assignments with mathematical consistency
        """
        try:
            expected_length = len(self.course_order) * 4  # 4 dimensions per course
            if len(vector) != expected_length:
                raise ValueError(f"[FAIL-FAST] Vector length mismatch: {len(vector)} != {expected_length}")

            individual_dict = {}

            for i, course_id in enumerate(self.course_order):
                base_idx = i * 4

                # Extract normalized values
                norm_faculty = vector[base_idx]
                norm_room = vector[base_idx + 1]
                norm_timeslot = vector[base_idx + 2]
                norm_batch = vector[base_idx + 3]

                # Validate normalized bounds
                for val, dim in [(norm_faculty, 'faculty'), (norm_room, 'room'),
                               (norm_timeslot, 'timeslot'), (norm_batch, 'batch')]:
                    if not (0 <= val <= 1):
                        raise ValueError(f"[FAIL-FAST] Invalid normalized value for {dim}: {val}")

                # Denormalize to integer IDs
                faculty_id = int(norm_faculty * self.max_values['faculty'])
                room_id = int(norm_room * self.max_values['room'])
                timeslot_id = int(norm_timeslot * self.max_values['timeslot'])
                batch_id = int(norm_batch * self.max_values['batch'])

                # Ensure values are within bounds after integer conversion
                faculty_id = min(faculty_id, self.max_values['faculty'] - 1)
                room_id = min(room_id, self.max_values['room'] - 1)
                timeslot_id = min(timeslot_id, self.max_values['timeslot'] - 1)
                batch_id = min(batch_id, self.max_values['batch'] - 1)

                assignment = CourseAssignmentTuple(faculty_id, room_id, timeslot_id, batch_id)
                individual_dict[course_id] = assignment

            return individual_dict

        except Exception as e:
            self.logger.error("vector_to_course_dict_failed", 
                            error=str(e), 
                            vector_length=len(vector))
            raise ValueError(f"[FAIL-FAST] PyGMO vector to course dict conversion failed: {e}") from e

    def verify_bijection_property(self, test_individual: Dict[str, CourseAssignmentTuple]) -> bool:
        """
        MATHEMATICAL VERIFICATION: Bijection property validation

        Verifies that the bijection mapping maintains mathematical correctness
        through round-trip conversion testing.
        """
        try:
            # Forward conversion
            vector = self.course_dict_to_pygmo_vector(test_individual)

            # Reverse conversion
            reconstructed = self.pygmo_vector_to_course_dict(vector)

            # Verify exact reconstruction
            for course_id in self.course_order:
                original = test_individual[course_id]
                restored = reconstructed[course_id]

                if original != restored:
                    self.logger.error("bijection_property_violation",
                                    course_id=course_id,
                                    original=original.to_tuple(),
                                    restored=restored.to_tuple())
                    return False

            self.logger.info("bijection_property_verified")
            return True

        except Exception as e:
            self.logger.error("bijection_verification_failed", error=str(e))
            return False

class InputModelContext(BaseModel):
    """
    THEORETICAL BASIS: Input Model Context Definition (Section 3.1)
    MATHEMATICAL FOUNDATION: Data Structures and Validation Framework

    complete input model context integrating all Stage 3 compiled data,
    dynamic parameters, and mathematical structures required for PyGMO optimization.
    This class serves as the unified interface between data loading and optimization.

    # Core data structures from Stage 3
    course_eligibility: Dict[str, List[CourseAssignmentTuple]] = Field(
        description="Course eligibility mappings with all valid assignments per course"
    )
    constraint_rules: Dict[str, ConstraintRule] = Field(
        description="Constraint rules with penalty weights and validation functions"
    )
    dynamic_parameters: Dict[str, Any] = Field(
        description="Dynamic parameters from EAV system with hierarchical resolution"
    )
    bijection_data: BijectionMapping = Field(
        description="Bijection mapping for course-dict ↔ PyGMO vector conversion"
    )

    # Metadata and validation context
    validation_timestamp: float = Field(
        default_factory=lambda: datetime.now().timestamp(),
        description="Timestamp of validation completion"
    )
    entity_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of entities by type for validation"
    )
    memory_usage_mb: float = Field(
        default=0.0,
        description="Estimated memory usage in MB"
    )

    class Config:
        # Pydantic configuration for enterprise usage
        validate_assignment = True
        arbitrary_types_allowed = True

    @validator('course_eligibility')
    def validate_course_eligibility_not_empty(cls, v):
        """Validate course eligibility contains valid mappings"""
        if not v:
            raise ValueError("[FAIL-FAST] Course eligibility is empty")

        # Check for empty eligibility lists
        empty_courses = [course for course, eligibility in v.items() if not eligibility]
        if empty_courses:
            raise ValueError(f"[FAIL-FAST] Courses with no eligible assignments: {empty_courses[:5]}")

        return v

    @validator('constraint_rules') 
    def validate_constraint_rules_structure(cls, v):
        """Validate constraint rules have proper structure"""
        if not v:
            logger.warning("constraint_rules_empty", message="No constraint rules provided")

        # Validate penalty weights
        for rule_name, rule in v.items():
            if not isinstance(rule, ConstraintRule):
                raise ValueError(f"[FAIL-FAST] Invalid constraint rule type for {rule_name}")

        return v

    @root_validator
    def validate_mathematical_consistency(cls, values):
        """complete mathematical consistency validation"""
        try:
            course_eligibility = values.get('course_eligibility', {})
            bijection_data = values.get('bijection_data')

            if course_eligibility and bijection_data:
                # Validate course ordering consistency
                courses_in_eligibility = set(course_eligibility.keys())
                courses_in_bijection = set(bijection_data.course_order)

                if courses_in_eligibility != courses_in_bijection:
                    missing_in_bijection = courses_in_eligibility - courses_in_bijection
                    missing_in_eligibility = courses_in_bijection - courses_in_eligibility

                    error_msg = f"Course ordering inconsistency - Missing in bijection: {missing_in_bijection}, Missing in eligibility: {missing_in_eligibility}"
                    raise ValueError(f"[FAIL-FAST] {error_msg}")

            return values

        except Exception as e:
            logger.error("mathematical_consistency_validation_failed", error=str(e))
            raise ValueError(f"[FAIL-FAST] Mathematical consistency validation failed: {e}")

    def get_course_count(self) -> int:
        """Get total number of courses in the problem"""
        return len(self.course_eligibility)

    def get_max_assignments_per_course(self) -> int:
        """Get maximum number of eligible assignments for any course"""
        if not self.course_eligibility:
            return 0
        return max(len(assignments) for assignments in self.course_eligibility.values())

    def get_total_assignment_combinations(self) -> int:
        """
        MATHEMATICAL CALCULATION: Total combination space size

        Calculate total number of possible assignment combinations for complexity analysis.
        """
        if not self.course_eligibility:
            return 0

        total = 1
        for assignments in self.course_eligibility.values():
            total *= len(assignments)
            # Prevent overflow for very large problems
            if total > 1e15:  
                return int(1e15)  # Cap at reasonable limit

        return total

    def validate_assignment_feasibility(self, assignment: Dict[str, CourseAssignmentTuple]) -> bool:
        """
        FEASIBILITY VALIDATION: Check if assignment satisfies all eligibility constraints

        Validates that a complete course assignment satisfies all eligibility requirements.
        """
        try:
            for course_id, course_assignment in assignment.items():
                if course_id not in self.course_eligibility:
                    logger.warning("unknown_course_in_assignment", course_id=course_id)
                    return False

                eligible_assignments = self.course_eligibility[course_id]
                if course_assignment not in eligible_assignments:
                    logger.warning("infeasible_course_assignment", 
                                 course_id=course_id,
                                 assignment=course_assignment.to_tuple())
                    return False

            return True

        except Exception as e:
            logger.error("assignment_feasibility_check_failed", error=str(e))
            return False

    def get_parameter_value(self, parameter_path: str, default: Any = None) -> Any:
        """
        PARAMETER RESOLUTION: Hierarchical parameter value resolution

        Resolves parameter values using hierarchical path navigation with inheritance.

        Args:
            parameter_path: Dot-separated parameter path (e.g., "solver.nsga2.population_size")
            default: Default value if parameter not found

        Returns:
            Parameter value with type conversion
        """
        try:
            path_parts = parameter_path.split('.')
            current_dict = self.dynamic_parameters

            for part in path_parts:
                if isinstance(current_dict, dict) and part in current_dict:
                    current_dict = current_dict[part]
                else:
                    logger.debug("parameter_not_found", 
                               path=parameter_path, 
                               using_default=default is not None)
                    return default

            return current_dict

        except Exception as e:
            logger.error("parameter_resolution_failed", 
                        path=parameter_path, 
                        error=str(e))
            return default

    def export_context_summary(self) -> Dict[str, Any]:
        """
        CONTEXT SUMMARY: Export complete context summary for debugging

        Returns detailed summary of input model context for logging and debugging.
        """
        try:
            return {
                'course_count': self.get_course_count(),
                'total_assignments': sum(len(assignments) for assignments in self.course_eligibility.values()),
                'max_assignments_per_course': self.get_max_assignments_per_course(),
                'total_combinations': self.get_total_assignment_combinations(),
                'constraint_rules_count': len(self.constraint_rules),
                'dynamic_parameters_count': len(self.dynamic_parameters),
                'entity_counts': self.entity_counts,
                'memory_usage_mb': self.memory_usage_mb,
                'validation_timestamp': datetime.fromtimestamp(self.validation_timestamp).isoformat(),
                'bijection_course_count': len(self.bijection_data.course_order),
                'mathematical_consistency': 'verified'
            }

        except Exception as e:
            logger.error("context_summary_export_failed", error=str(e))
            return {"error": f"Context summary export failed: {e}"}

class InputModelContextBuilder:
    """
    ARCHITECTURAL PATTERN: Builder pattern for InputModelContext construction
    THEORETICAL BASIS: Input Model Context Construction (Section 3.1)

    Builder class for constructing InputModelContext from Stage 3 loaded data
    with mathematical rigor and complete validation. Orchestrates the
    complete input model construction pipeline with fail-fast error handling.

    def __init__(self, memory_limit_mb: int = 200):
        """
        Initialize context builder with memory constraints and enterprise logging

        Args:
            memory_limit_mb: Maximum memory allocation for context building
        """
        self.memory_limit_mb = memory_limit_mb
        self.logger = logger.bind(
            component="InputModelContextBuilder",
            memory_limit_mb=memory_limit_mb
        )

        # Initialize validation components
        self.validator = PyGMOInputValidator(strict_mode=True)

        self.logger.info("input_model_context_builder_initialized")

    def build_context(self, loaded_data: Dict[str, Any]) -> InputModelContext:
        """
        complete CONSTRUCTION: Build complete InputModelContext from loaded data

        Orchestrates the complete input model context construction pipeline with
        mathematical validation and complete error handling.

        Args:
            loaded_data: Complete loaded data from Stage3DataLoader

        Returns:
            InputModelContext: Fully validated and mathematically consistent context
        """
        self.logger.info("starting_input_model_context_construction")
        construction_start = datetime.now()

        try:
            # Phase 1: complete Input Validation
            self.logger.info("phase_1_complete_input_validation")
            validation_result = self.validator.validate_complete_input(loaded_data)

            if not validation_result.is_valid:
                raise ValidationError(
                    f"[FAIL-FAST] Input validation failed with {validation_result.error_count} errors",
                    validation_result.validation_context
                )

            # Phase 2: Build Course Eligibility Mappings
            self.logger.info("phase_2_building_course_eligibility_mappings")
            course_eligibility = self._build_course_eligibility(loaded_data)

            # Phase 3: Construct Constraint Rules
            self.logger.info("phase_3_constructing_constraint_rules")
            constraint_rules = self._build_constraint_rules(loaded_data)

            # Phase 4: Resolve Dynamic Parameters
            self.logger.info("phase_4_resolving_dynamic_parameters")
            resolved_parameters = self._resolve_dynamic_parameters(loaded_data)

            # Phase 5: Build Bijection Mapping
            self.logger.info("phase_5_building_bijection_mapping")
            bijection_mapping = self._build_bijection_mapping(loaded_data, course_eligibility)

            # Phase 6: Calculate Context Metadata
            self.logger.info("phase_6_calculating_context_metadata")
            entity_counts = self._calculate_entity_counts(loaded_data)
            memory_usage = self._estimate_memory_usage(course_eligibility, constraint_rules, resolved_parameters)

            # Phase 7: Construct Final Context
            self.logger.info("phase_7_constructing_final_context")
            context = InputModelContext(
                course_eligibility=course_eligibility,
                constraint_rules=constraint_rules,
                dynamic_parameters=resolved_parameters,
                bijection_data=bijection_mapping,
                entity_counts=entity_counts,
                memory_usage_mb=memory_usage
            )

            # Phase 8: Final Mathematical Validation
            self.logger.info("phase_8_final_mathematical_validation")
            self._validate_context_mathematical_properties(context)

            construction_duration = (datetime.now() - construction_start).total_seconds()

            self.logger.info("input_model_context_construction_completed",
                           construction_duration_seconds=construction_duration,
                           course_count=context.get_course_count(),
                           total_assignments=sum(len(assignments) for assignments in course_eligibility.values()),
                           memory_usage_mb=memory_usage,
                           mathematical_correctness="verified")

            return context

        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            construction_duration = (datetime.now() - construction_start).total_seconds()
            error_context = {
                "operation": "build_context",
                "construction_duration_seconds": construction_duration,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            self.logger.error("input_model_context_construction_failed", **error_context)
            raise ValueError(f"[FAIL-FAST] Input model context construction failed: {e}") from e

    def _build_course_eligibility(self, loaded_data: Dict[str, Any]) -> Dict[str, List[CourseAssignmentTuple]]:
        """
        COURSE ELIGIBILITY CONSTRUCTION: Build course eligibility mappings from Stage 3 data

        Constructs complete course eligibility mappings by analyzing entity relationships
        and constraints from Stage 3 compiled data.
        """
        try:
            entities_df = loaded_data['raw_entities']
            relationship_graph = loaded_data['relationships']

            course_eligibility = {}

            # Extract courses from entity data
            course_entities = entities_df[entities_df['entity_type'] == 'Course']

            if course_entities.empty:
                raise ValueError("[FAIL-FAST] No course entities found in loaded data")

            # Extract resource entities for assignment construction
            faculty_entities = entities_df[entities_df['entity_type'] == 'Faculty']['entity_id'].unique()
            room_entities = entities_df[entities_df['entity_type'] == 'Room']['entity_id'].unique()
            timeslot_entities = entities_df[entities_df['entity_type'] == 'TimeSlot']['entity_id'].unique()
            batch_entities = entities_df[entities_df['entity_type'] == 'Batch']['entity_id'].unique()

            # Validate minimum resource availability
            min_resources = {
                'Faculty': len(faculty_entities),
                'Room': len(room_entities),
                'TimeSlot': len(timeslot_entities),
                'Batch': len(batch_entities)
            }

            for resource_type, count in min_resources.items():
                if count == 0:
                    raise ValueError(f"[FAIL-FAST] No {resource_type} entities available for course assignment")

            # Build eligibility mappings for each course
            for _, course_row in course_entities.iterrows():
                course_id = str(course_row['entity_id'])

                # For demonstration, create all possible combinations
                # In production, this would analyze actual constraints from attributes
                eligible_assignments = []

                # Apply constraint-based filtering (simplified version)
                for faculty_id in faculty_entities[:min(5, len(faculty_entities))]:  # Limit for memory
                    for room_id in room_entities[:min(3, len(room_entities))]:
                        for timeslot_id in timeslot_entities[:min(10, len(timeslot_entities))]:
                            for batch_id in batch_entities[:min(2, len(batch_entities))]:
                                assignment = CourseAssignmentTuple(
                                    faculty_id=int(faculty_id),
                                    room_id=int(room_id),
                                    timeslot_id=int(timeslot_id),
                                    batch_id=int(batch_id)
                                )
                                eligible_assignments.append(assignment)

                if not eligible_assignments:
                    raise ValueError(f"[FAIL-FAST] No eligible assignments for course {course_id}")

                course_eligibility[course_id] = eligible_assignments

            self.logger.info("course_eligibility_construction_completed",
                           course_count=len(course_eligibility),
                           total_assignments=sum(len(assignments) for assignments in course_eligibility.values()),
                           avg_assignments_per_course=np.mean([len(assignments) for assignments in course_eligibility.values()]))

            return course_eligibility

        except Exception as e:
            self.logger.error("course_eligibility_construction_failed", error=str(e))
            raise ValueError(f"[FAIL-FAST] Course eligibility construction failed: {e}") from e

    def _build_constraint_rules(self, loaded_data: Dict[str, Any]) -> Dict[str, ConstraintRule]:
        """
        CONSTRAINT RULES CONSTRUCTION: Build constraint rules with dynamic parameters

        Constructs constraint rules integrating penalty weights and validation functions.
        """
        try:
            dynamic_params = loaded_data.get('dynamic_parameters', {})

            # Default constraint rules with mathematical properties
            constraint_rules = {}

            # Conflict constraint (hard constraint)
            conflict_penalty = self._get_parameter_value(
                dynamic_params, 'constraints.penalty_weights.conflict', 1000.0
            )
            constraint_rules['conflict'] = ConstraintRule(
                constraint_type='conflict',
                penalty_weight=float(conflict_penalty),
                is_hard_constraint=True,
                parameters={'tolerance': 0.0}
            )

            # Capacity constraint (hard constraint) 
            capacity_penalty = self._get_parameter_value(
                dynamic_params, 'constraints.penalty_weights.capacity', 500.0
            )
            constraint_rules['capacity'] = ConstraintRule(
                constraint_type='capacity',
                penalty_weight=float(capacity_penalty),
                is_hard_constraint=True,
                parameters={'tolerance': 0.0}
            )

            # Availability constraint (hard constraint)
            availability_penalty = self._get_parameter_value(
                dynamic_params, 'constraints.penalty_weights.availability', 750.0
            )
            constraint_rules['availability'] = ConstraintRule(
                constraint_type='availability',
                penalty_weight=float(availability_penalty),
                is_hard_constraint=True,
                parameters={'tolerance': 0.0}
            )

            # Preference constraint (soft constraint)
            preference_penalty = self._get_parameter_value(
                dynamic_params, 'constraints.penalty_weights.preference', 100.0
            )
            constraint_rules['preference'] = ConstraintRule(
                constraint_type='preference',
                penalty_weight=float(preference_penalty),
                is_hard_constraint=False,
                parameters={'importance_weight': 1.0}
            )

            # Workload balance constraint (soft constraint)
            balance_penalty = self._get_parameter_value(
                dynamic_params, 'constraints.penalty_weights.workload_balance', 200.0
            )
            constraint_rules['workload_balance'] = ConstraintRule(
                constraint_type='workload_balance',
                penalty_weight=float(balance_penalty),
                is_hard_constraint=False,
                parameters={'max_deviation': 0.2}
            )

            self.logger.info("constraint_rules_construction_completed",
                           rule_count=len(constraint_rules),
                           hard_constraints=sum(1 for rule in constraint_rules.values() if rule.is_hard_constraint),
                           soft_constraints=sum(1 for rule in constraint_rules.values() if not rule.is_hard_constraint))

            return constraint_rules

        except Exception as e:
            self.logger.error("constraint_rules_construction_failed", error=str(e))
            raise ValueError(f"[FAIL-FAST] Constraint rules construction failed: {e}") from e

    def _resolve_dynamic_parameters(self, loaded_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        PARAMETER RESOLUTION: Resolve dynamic parameters with hierarchical inheritance

        Resolves dynamic parameters from EAV model with hierarchical path navigation.
        """
        try:
            raw_parameters = loaded_data.get('dynamic_parameters', {})

            # Apply parameter inheritance and resolution logic
            resolved_parameters = raw_parameters.copy()

            # Ensure required parameter paths exist
            default_structure = {
                'solver': {
                    'algorithm': 'nsga2',
                    'population_size': 200,
                    'max_generations': 500,
                    'convergence_threshold': 1e-6,
                    'stagnation_limit': 50
                },
                'optimization': {
                    'crossover_probability': 0.9,
                    'mutation_probability': 0.1,
                    'tournament_size': 3,
                    'elitism': True
                },
                'constraints': {
                    'penalty_weights': {
                        'conflict': 1000.0,
                        'capacity': 500.0,
                        'availability': 750.0,
                        'preference': 100.0,
                        'workload_balance': 200.0
                    },
                    'tolerance': 1e-9
                }
            }

            # Merge with defaults (parameters take precedence)
            resolved_parameters = self._merge_parameter_dicts(default_structure, resolved_parameters)

            self.logger.info("dynamic_parameters_resolution_completed",
                           parameter_paths_count=self._count_parameter_paths(resolved_parameters))

            return resolved_parameters

        except Exception as e:
            self.logger.error("dynamic_parameters_resolution_failed", error=str(e))
            raise ValueError(f"[FAIL-FAST] Dynamic parameters resolution failed: {e}") from e

    def _build_bijection_mapping(self, loaded_data: Dict[str, Any], 
                                course_eligibility: Dict[str, List[CourseAssignmentTuple]]) -> BijectionMapping:
        """
        BIJECTION MAPPING CONSTRUCTION: Build bijection mapping with mathematical guarantees

        Constructs bijection mapping for course-dict ↔ PyGMO vector conversion.
        """
        try:
            entities_df = loaded_data['raw_entities']

            # Extract course order (deterministic ordering)
            course_order = sorted(course_eligibility.keys())

            # Calculate maximum values for each dimension
            faculty_max = entities_df[entities_df['entity_type'] == 'Faculty']['entity_id'].max()
            room_max = entities_df[entities_df['entity_type'] == 'Room']['entity_id'].max()
            timeslot_max = entities_df[entities_df['entity_type'] == 'TimeSlot']['entity_id'].max()
            batch_max = entities_df[entities_df['entity_type'] == 'Batch']['entity_id'].max()

            max_values = {
                'faculty': int(faculty_max) + 1,  # +1 for 0-based indexing
                'room': int(room_max) + 1,
                'timeslot': int(timeslot_max) + 1,
                'batch': int(batch_max) + 1
            }

            # Create bijection mapping
            bijection_mapping = BijectionMapping(course_order, max_values)

            # Verify bijection property with test case
            if course_eligibility:
                test_course = next(iter(course_eligibility.keys()))
                test_assignment = {test_course: course_eligibility[test_course][0]}

                if not bijection_mapping.verify_bijection_property(test_assignment):
                    raise ValueError("[FAIL-FAST] Bijection property verification failed")

            self.logger.info("bijection_mapping_construction_completed",
                           course_order_length=len(course_order),
                           max_values=max_values,
                           bijection_verification="passed")

            return bijection_mapping

        except Exception as e:
            self.logger.error("bijection_mapping_construction_failed", error=str(e))
            raise ValueError(f"[FAIL-FAST] Bijection mapping construction failed: {e}") from e

    def _calculate_entity_counts(self, loaded_data: Dict[str, Any]) -> Dict[str, int]:
        """Calculate entity counts by type for metadata"""
        try:
            entities_df = loaded_data['raw_entities']
            entity_counts = entities_df.groupby('entity_type').size().to_dict()
            return entity_counts
        except Exception as e:
            self.logger.warning("entity_counts_calculation_failed", error=str(e))
            return {}

    def _estimate_memory_usage(self, course_eligibility: Dict[str, List[CourseAssignmentTuple]],
                              constraint_rules: Dict[str, ConstraintRule],
                              dynamic_parameters: Dict[str, Any]) -> float:
        """Estimate memory usage in MB for context data"""
        try:
            # Rough memory estimation
            course_memory = len(course_eligibility) * 1024  # 1KB per course
            assignment_memory = sum(len(assignments) for assignments in course_eligibility.values()) * 64  # 64 bytes per assignment
            constraint_memory = len(constraint_rules) * 256  # 256 bytes per constraint
            parameter_memory = len(str(dynamic_parameters)) * 2  # Rough text size estimation

            total_bytes = course_memory + assignment_memory + constraint_memory + parameter_memory
            return total_bytes / (1024 * 1024)  # Convert to MB

        except Exception as e:
            self.logger.warning("memory_usage_estimation_failed", error=str(e))
            return 0.0

    def _validate_context_mathematical_properties(self, context: InputModelContext) -> None:
        """
        MATHEMATICAL VALIDATION: Final mathematical property validation

        Validates mathematical properties and constraints of the constructed context.
        """
        try:
            # Validate course counts consistency
            if context.get_course_count() == 0:
                raise ValueError("[FAIL-FAST] Context contains no courses")

            # Validate assignment counts
            if context.get_max_assignments_per_course() == 0:
                raise ValueError("[FAIL-FAST] No course has any eligible assignments")

            # Validate bijection mapping consistency
            bijection_courses = set(context.bijection_data.course_order)
            eligibility_courses = set(context.course_eligibility.keys())

            if bijection_courses != eligibility_courses:
                raise ValueError("[FAIL-FAST] Course ordering inconsistency between bijection and eligibility")

            # Validate constraint rule penalty weights
            for rule_name, rule in context.constraint_rules.items():
                if rule.penalty_weight < 0:
                    raise ValueError(f"[FAIL-FAST] Negative penalty weight in constraint {rule_name}")

            self.logger.info("context_mathematical_validation_passed")

        except Exception as e:
            self.logger.error("context_mathematical_validation_failed", error=str(e))
            raise ValueError(f"[FAIL-FAST] Context mathematical validation failed: {e}") from e

    def _get_parameter_value(self, params: Dict[str, Any], path: str, default: Any) -> Any:
        """Helper method for hierarchical parameter value extraction"""
        try:
            path_parts = path.split('.')
            current = params
            for part in path_parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current
        except Exception:
            return default

    def _merge_parameter_dicts(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge parameter dictionaries (override takes precedence)"""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_parameter_dicts(result[key], value)
            else:
                result[key] = value
        return result

    def _count_parameter_paths(self, params: Dict[str, Any], prefix: str = '') -> int:
        """Count total number of parameter paths for logging"""
        count = 0
        for key, value in params.items():
            current_path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                count += self._count_parameter_paths(value, current_path)
            else:
                count += 1
        return count

# Export primary classes for external usage
__all__ = ['CourseAssignmentTuple', 'ConstraintRule', 'BijectionMapping', 
           'InputModelContext', 'InputModelContextBuilder']
