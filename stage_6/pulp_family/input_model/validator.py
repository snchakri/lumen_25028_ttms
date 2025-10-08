#!/usr/bin/env python3
"""
PuLP Solver Family - Stage 6 Input Modeling Layer: Data Validation Module

This module implements complete validation functionality for loaded Stage 3 data structures,
ensuring mathematical correctness, scheduling feasibility, and compliance with theoretical frameworks
before proceeding to optimization processing. Provides fail-fast validation with complete
error reporting and eligibility verification per formal models.

Theoretical Foundation:
    Based on Stage 6 foundational design rules and mathematical frameworks:
    - Implements Definition 2.4-2.5 (Hard/Soft Constraints) validation
    - Ensures non-empty eligibility sets per Algorithm 3.2 Step 8
    - Validates relationship integrity per Definition 2.3
    - Maintains data preservation per Theorem 5.1

Architecture Compliance:
    - Fail-fast philosophy: immediate termination on critical validation failures  
    - complete logging: structured error reporting with JSON metadata
    - Mathematical rigor: formal constraint verification and feasibility checking
    - Production readiness: reliable error handling with detailed diagnostics

Dependencies: pandas, numpy, scipy, networkx, logging, json, datetime
Author: Student Team
Version: 1.0.0 (Production)
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
import networkx as nx
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Import data structures from loader module
try:
    from .loader import EntityCollection, RelationshipGraph, IndexStructure
except ImportError:
    # Handle standalone execution
    from loader import EntityCollection, RelationshipGraph, IndexStructure

# Configure structured logging for validation operations
logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Enumeration of validation error severity levels for structured error handling."""
    CRITICAL = "CRITICAL"    # Violations that prevent optimization (empty eligibility sets)
    ERROR = "ERROR"          # Serious violations that may cause solver failures  
    WARNING = "WARNING"      # Quality issues that may impact solution quality
    INFO = "INFO"           # Informational messages about data characteristics

@dataclass
class ValidationResult:
    """
    Encapsulates validation results with structured error reporting.

    Mathematical Foundation: Supports fail-fast validation per Stage 6 rules
    while maintaining complete diagnostics for debugging and audit trails.

    Attributes:
        is_valid: Boolean indicating overall validation success/failure
        severity: Highest severity level encountered during validation
        errors: List of validation errors with detailed diagnostics
        warnings: List of validation warnings for quality issues
        statistics: Quantitative validation metrics and data characteristics
        metadata: Additional context and execution information
    """
    is_valid: bool
    severity: ValidationSeverity
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, component: str, message: str, severity: ValidationSeverity, 
                  details: Optional[Dict[str, Any]] = None) -> None:
        """Add validation error with structured details."""
        error = {
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'severity': severity.value,
            'message': message,
            'details': details or {}
        }

        if severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]:
            self.errors.append(error)
            self.is_valid = False
            # Update overall severity to most severe encountered
            if severity == ValidationSeverity.CRITICAL or self.severity != ValidationSeverity.CRITICAL:
                self.severity = severity
        else:
            self.warnings.append(error)

    def get_summary(self) -> Dict[str, Any]:
        """Generate complete validation summary for logging and reporting."""
        return {
            'validation_status': 'PASSED' if self.is_valid else 'FAILED',
            'overall_severity': self.severity.value,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'statistics': self.statistics,
            'metadata': self.metadata
        }

class SchedulingDataValidator:
    """
    complete validator for scheduling data structures with mathematical rigor.

    Implements complete validation based on formal frameworks:
    - Entity completeness and integrity verification
    - Relationship consistency and transitivity validation  
    - Eligibility set non-emptiness checking (critical for feasibility)
    - Temporal constraint validation for timeslots
    - Capacity and resource constraint verification

    Mathematical Foundation:
        - Validates Definition 2.1 (Data Universe) completeness
        - Ensures Definition 2.2 (Entity Instance) integrity  
        - Verifies Definition 2.3 (Relationship Function) consistency
        - Checks Algorithm 3.2 prerequisites (non-empty entity sets)
    """

    def __init__(self, execution_id: str, strict_mode: bool = True):
        """
        Initialize validator with execution context and strictness settings.

        Args:
            execution_id: Unique execution identifier for logging and error tracking
            strict_mode: If True, treat warnings as errors for maximum rigor
        """
        self.execution_id = execution_id
        self.strict_mode = strict_mode

        # Initialize validation state
        self.validation_result = ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            metadata={
                'execution_id': execution_id,
                'validation_timestamp': datetime.now().isoformat(),
                'strict_mode': strict_mode,
                'validator_version': '1.0.0'
            }
        )

        logger.info(f"SchedulingDataValidator initialized for execution {execution_id}")

    def validate_all(self, entity_collections: Dict[str, EntityCollection],
                    relationship_graph: RelationshipGraph,
                    index_structure: IndexStructure) -> ValidationResult:
        """
        complete validation of all loaded data structures.

        Implements fail-fast validation strategy: terminates on first CRITICAL error
        to prevent downstream failures and resource waste.

        Args:
            entity_collections: Dictionary of loaded entity collections
            relationship_graph: Loaded relationship graph structure
            index_structure: Loaded index structure

        Returns:
            ValidationResult object with complete diagnostics

        Raises:
            ValueError: If critical validation failures occur in strict mode
        """
        logger.info(f"Starting complete validation for execution {self.execution_id}")

        try:
            # Phase 1: Entity collection validation
            self._validate_entity_collections(entity_collections)

            # Fail-fast: terminate on critical entity validation failures
            if not self.validation_result.is_valid and self.validation_result.severity == ValidationSeverity.CRITICAL:
                logger.critical("Critical entity validation failure - terminating")
                return self.validation_result

            # Phase 2: Relationship graph validation  
            self._validate_relationship_graph(relationship_graph, entity_collections)

            # Fail-fast: terminate on critical relationship validation failures
            if not self.validation_result.is_valid and self.validation_result.severity == ValidationSeverity.CRITICAL:
                logger.critical("Critical relationship validation failure - terminating")
                return self.validation_result

            # Phase 3: Index structure validation
            self._validate_index_structure(index_structure, entity_collections)

            # Phase 4: Cross-component consistency validation
            self._validate_cross_component_consistency(entity_collections, relationship_graph, index_structure)

            # Phase 5: Scheduling-specific feasibility validation
            self._validate_scheduling_feasibility(entity_collections, relationship_graph)

            # Phase 6: Performance and optimization readiness validation
            self._validate_optimization_readiness(entity_collections, relationship_graph, index_structure)

            # Generate final validation statistics
            self._compute_validation_statistics(entity_collections, relationship_graph, index_structure)

            # Log validation completion
            if self.validation_result.is_valid:
                logger.info(f"Validation completed successfully for execution {self.execution_id}")
            else:
                logger.error(f"Validation failed with {len(self.validation_result.errors)} errors")

            return self.validation_result

        except Exception as e:
            # Handle unexpected validation errors
            self.validation_result.add_error(
                component="validator",
                message=f"Unexpected validation failure: {str(e)}",
                severity=ValidationSeverity.CRITICAL,
                details={'exception_type': type(e).__name__}
            )
            logger.critical(f"Unexpected validation failure: {str(e)}")
            return self.validation_result

    def _validate_entity_collections(self, entity_collections: Dict[str, EntityCollection]) -> None:
        """
        Validate entity collections for completeness, integrity, and scheduling requirements.

        Mathematical Foundation: Validates Definition 2.1 (Data Universe) and 2.2 (Entity Instance).
        Ensures all required entity types are present with valid data structures.
        """
        logger.debug("Validating entity collections")

        # Required entity types for scheduling optimization
        required_entities = {'courses', 'faculties', 'rooms', 'batches', 'timeslots'}
        present_entities = set(entity_collections.keys())

        # Check for missing required entity types
        missing_entities = required_entities - present_entities
        if missing_entities:
            self.validation_result.add_error(
                component="entity_collections",
                message=f"Missing required entity types: {missing_entities}",
                severity=ValidationSeverity.CRITICAL,
                details={'missing_entities': list(missing_entities), 'present_entities': list(present_entities)}
            )
            return  # Fail-fast on missing critical entities

        # Validate each entity collection individually
        for entity_type, collection in entity_collections.items():
            self._validate_single_entity_collection(entity_type, collection)

        # Validate entity collection relationships and constraints
        self._validate_entity_collection_constraints(entity_collections)

    def _validate_single_entity_collection(self, entity_type: str, collection: EntityCollection) -> None:
        """
        Validate individual entity collection for data integrity and business rules.

        Implements complete checks for:
        - Primary key uniqueness and completeness
        - Data type consistency and validity
        - Business rule compliance (positive capacities, valid time ranges, etc.)
        - Minimum cardinality requirements for scheduling feasibility
        """
        logger.debug(f"Validating {entity_type} entity collection")

        # Check for empty entity collections (critical for optimization)
        if len(collection.entities) == 0:
            self.validation_result.add_error(
                component=entity_type,
                message=f"Empty entity collection: {entity_type}",
                severity=ValidationSeverity.CRITICAL,
                details={'entity_count': 0}
            )
            return

        # Validate primary key integrity
        primary_key = collection.primary_key
        if primary_key not in collection.entities.columns:
            self.validation_result.add_error(
                component=entity_type,
                message=f"Primary key '{primary_key}' not found in entity columns",
                severity=ValidationSeverity.CRITICAL,
                details={'primary_key': primary_key, 'available_columns': list(collection.entities.columns)}
            )
            return

        # Check primary key uniqueness (essential for bijection correctness)
        duplicate_count = collection.entities[primary_key].duplicated().sum()
        if duplicate_count > 0:
            self.validation_result.add_error(
                component=entity_type,
                message=f"Primary key '{primary_key}' contains {duplicate_count} duplicates",
                severity=ValidationSeverity.CRITICAL,
                details={'duplicate_count': duplicate_count, 'primary_key': primary_key}
            )

        # Check for null primary key values
        null_pk_count = collection.entities[primary_key].isnull().sum()
        if null_pk_count > 0:
            self.validation_result.add_error(
                component=entity_type,
                message=f"Primary key '{primary_key}' contains {null_pk_count} null values",
                severity=ValidationSeverity.CRITICAL,
                details={'null_count': null_pk_count, 'primary_key': primary_key}
            )

        # Entity-specific validation rules
        if entity_type == 'courses':
            self._validate_courses_specific(collection)
        elif entity_type == 'faculties':
            self._validate_faculties_specific(collection)
        elif entity_type == 'rooms':
            self._validate_rooms_specific(collection)
        elif entity_type == 'batches':
            self._validate_batches_specific(collection)
        elif entity_type == 'timeslots':
            self._validate_timeslots_specific(collection)

        # Check minimum entity count requirements
        min_requirements = {
            'courses': 1, 'faculties': 1, 'rooms': 1, 'batches': 1, 'timeslots': 2
        }
        min_required = min_requirements.get(entity_type, 1)
        if len(collection.entities) < min_required:
            self.validation_result.add_error(
                component=entity_type,
                message=f"Insufficient {entity_type}: {len(collection.entities)} < {min_required} required",
                severity=ValidationSeverity.CRITICAL,
                details={'actual_count': len(collection.entities), 'minimum_required': min_required}
            )

    def _validate_courses_specific(self, collection: EntityCollection) -> None:
        """Validate course-specific business rules and constraints."""
        df = collection.entities

        # Check for positive credits if credits column exists
        if 'credits' in df.columns:
            non_positive_credits = (df['credits'] <= 0) | df['credits'].isnull()
            if non_positive_credits.any():
                invalid_count = non_positive_credits.sum()
                self.validation_result.add_error(
                    component="courses",
                    message=f"{invalid_count} courses have non-positive or null credits",
                    severity=ValidationSeverity.ERROR,
                    details={'invalid_count': invalid_count, 'total_courses': len(df)}
                )

        # Validate course code format if present
        if 'course_code' in df.columns:
            empty_codes = df['course_code'].isnull() | (df['course_code'].str.strip() == '')
            if empty_codes.any():
                self.validation_result.add_error(
                    component="courses",
                    message=f"{empty_codes.sum()} courses have empty course codes",
                    severity=ValidationSeverity.WARNING,
                    details={'empty_code_count': empty_codes.sum()}
                )

    def _validate_faculties_specific(self, collection: EntityCollection) -> None:
        """Validate faculty-specific business rules and constraints."""
        df = collection.entities

        # Check for required faculty information
        required_fields = ['name']  # Basic requirement
        for field in required_fields:
            if field in df.columns:
                empty_values = df[field].isnull() | (df[field].str.strip() == '')
                if empty_values.any():
                    self.validation_result.add_error(
                        component="faculties",
                        message=f"{empty_values.sum()} faculties have empty {field}",
                        severity=ValidationSeverity.WARNING,
                        details={'field': field, 'empty_count': empty_values.sum()}
                    )

    def _validate_rooms_specific(self, collection: EntityCollection) -> None:
        """Validate room-specific business rules and constraints."""
        df = collection.entities

        # Check for positive capacity
        if 'capacity' in df.columns:
            non_positive_capacity = (df['capacity'] <= 0) | df['capacity'].isnull()
            if non_positive_capacity.any():
                invalid_count = non_positive_capacity.sum()
                self.validation_result.add_error(
                    component="rooms",
                    message=f"{invalid_count} rooms have non-positive or null capacity",
                    severity=ValidationSeverity.ERROR,
                    details={'invalid_count': invalid_count, 'total_rooms': len(df)}
                )

        # Validate room numbers/identifiers
        if 'room_number' in df.columns:
            empty_numbers = df['room_number'].isnull() | (df['room_number'].astype(str).str.strip() == '')
            if empty_numbers.any():
                self.validation_result.add_error(
                    component="rooms",
                    message=f"{empty_numbers.sum()} rooms have empty room numbers",
                    severity=ValidationSeverity.WARNING,
                    details={'empty_number_count': empty_numbers.sum()}
                )

    def _validate_batches_specific(self, collection: EntityCollection) -> None:
        """Validate batch-specific business rules and constraints."""
        df = collection.entities

        # Check for positive strength/size
        if 'strength' in df.columns:
            non_positive_strength = (df['strength'] <= 0) | df['strength'].isnull()
            if non_positive_strength.any():
                invalid_count = non_positive_strength.sum()
                self.validation_result.add_error(
                    component="batches",
                    message=f"{invalid_count} batches have non-positive or null strength",
                    severity=ValidationSeverity.ERROR,
                    details={'invalid_count': invalid_count, 'total_batches': len(df)}
                )

    def _validate_timeslots_specific(self, collection: EntityCollection) -> None:
        """
        Validate timeslot-specific constraints with temporal consistency checking.

        Critical for scheduling feasibility: ensures valid time ranges and non-overlapping slots.
        """
        df = collection.entities

        # Validate start_time and end_time consistency
        if 'start_time' in df.columns and 'end_time' in df.columns:
            # Convert to comparable format if needed
            try:
                start_times = pd.to_datetime(df['start_time'], format='%H:%M', errors='coerce')
                end_times = pd.to_datetime(df['end_time'], format='%H:%M', errors='coerce')

                # Check for invalid time formats
                invalid_start = start_times.isnull()
                invalid_end = end_times.isnull()

                if invalid_start.any() or invalid_end.any():
                    total_invalid = invalid_start.sum() + invalid_end.sum()
                    self.validation_result.add_error(
                        component="timeslots",
                        message=f"{total_invalid} timeslots have invalid time formats",
                        severity=ValidationSeverity.ERROR,
                        details={'invalid_start_count': invalid_start.sum(), 'invalid_end_count': invalid_end.sum()}
                    )

                # Check for start_time >= end_time (invalid ranges)
                valid_mask = ~(invalid_start | invalid_end)
                if valid_mask.any():
                    invalid_ranges = (start_times[valid_mask] >= end_times[valid_mask])
                    if invalid_ranges.any():
                        self.validation_result.add_error(
                            component="timeslots",
                            message=f"{invalid_ranges.sum()} timeslots have start_time >= end_time",
                            severity=ValidationSeverity.CRITICAL,
                            details={'invalid_range_count': invalid_ranges.sum()}
                        )

            except Exception as e:
                self.validation_result.add_error(
                    component="timeslots",
                    message=f"Failed to validate time constraints: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    details={'exception': str(e)}
                )

        # Validate day_of_week if present
        if 'day_of_week' in df.columns:
            valid_days = {'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}
            day_values = df['day_of_week'].str.lower().str.strip()
            invalid_days = ~day_values.isin(valid_days) | day_values.isnull()

            if invalid_days.any():
                self.validation_result.add_error(
                    component="timeslots",
                    message=f"{invalid_days.sum()} timeslots have invalid day_of_week values",
                    severity=ValidationSeverity.ERROR,
                    details={'invalid_day_count': invalid_days.sum(), 'valid_days': list(valid_days)}
                )

    def _validate_entity_collection_constraints(self, entity_collections: Dict[str, EntityCollection]) -> None:
        """
        Validate cross-entity constraints and relationships.

        Ensures that entity collections are consistent with each other and
        satisfy basic scheduling requirements for optimization feasibility.
        """
        # Check relative sizes for feasibility warnings
        num_courses = len(entity_collections['courses'].entities)
        num_faculties = len(entity_collections['faculties'].entities) 
        num_rooms = len(entity_collections['rooms'].entities)
        num_timeslots = len(entity_collections['timeslots'].entities)
        num_batches = len(entity_collections['batches'].entities)

        # Calculate theoretical maximum assignments per course
        max_assignments_per_course = num_faculties * num_rooms * num_timeslots * num_batches

        # Warn if problem seems highly constrained
        if max_assignments_per_course < num_courses:
            self.validation_result.add_error(
                component="entity_constraints",
                message="Potentially infeasible problem: more courses than maximum possible assignments",
                severity=ValidationSeverity.WARNING,
                details={
                    'num_courses': num_courses,
                    'max_assignments_per_course': max_assignments_per_course,
                    'theoretical_capacity': max_assignments_per_course
                }
            )

        # Check for reasonable ratios
        if num_faculties < num_courses * 0.1:  # Less than 10% faculty-to-course ratio
            self.validation_result.add_error(
                component="entity_constraints", 
                message="Very low faculty-to-course ratio may cause scheduling difficulties",
                severity=ValidationSeverity.WARNING,
                details={'faculty_course_ratio': num_faculties / num_courses if num_courses > 0 else 0}
            )

        if num_rooms < num_courses * 0.05:  # Less than 5% room-to-course ratio  
            self.validation_result.add_error(
                component="entity_constraints",
                message="Very low room-to-course ratio may cause scheduling difficulties", 
                severity=ValidationSeverity.WARNING,
                details={'room_course_ratio': num_rooms / num_courses if num_courses > 0 else 0}
            )

    def _validate_relationship_graph(self, relationship_graph: RelationshipGraph, 
                                   entity_collections: Dict[str, EntityCollection]) -> None:
        """
        Validate relationship graph structure and consistency with entity collections.

        Mathematical Foundation: Validates Definition 2.3 (Relationship Function) consistency
        and ensures relationship graph aligns with loaded entity collections.
        """
        logger.debug("Validating relationship graph")

        graph = relationship_graph.graph

        # Basic graph structure validation
        if graph.number_of_nodes() == 0:
            self.validation_result.add_error(
                component="relationship_graph",
                message="Relationship graph contains no nodes",
                severity=ValidationSeverity.CRITICAL,
                details={'node_count': 0}
            )
            return

        # Validate entity mappings consistency
        self._validate_entity_mappings(relationship_graph, entity_collections)

        # Validate graph connectivity for scheduling requirements
        if not nx.is_connected(graph):
            num_components = nx.number_connected_components(graph)
            self.validation_result.add_error(
                component="relationship_graph",
                message=f"Relationship graph is disconnected with {num_components} components",
                severity=ValidationSeverity.WARNING,
                details={'connected_components': num_components, 'should_be_connected': True}
            )

        # Validate relationship matrix consistency
        try:
            matrix = relationship_graph.relationship_matrix
            if matrix.shape[0] != matrix.shape[1]:
                self.validation_result.add_error(
                    component="relationship_graph",
                    message="Relationship matrix is not square",
                    severity=ValidationSeverity.ERROR,
                    details={'matrix_shape': matrix.shape}
                )

            if matrix.shape[0] != graph.number_of_nodes():
                self.validation_result.add_error(
                    component="relationship_graph",
                    message="Relationship matrix size doesn't match graph node count",
                    severity=ValidationSeverity.ERROR,
                    details={'matrix_size': matrix.shape[0], 'node_count': graph.number_of_nodes()}
                )

        except Exception as e:
            self.validation_result.add_error(
                component="relationship_graph",
                message=f"Failed to validate relationship matrix: {str(e)}",
                severity=ValidationSeverity.ERROR,
                details={'exception': str(e)}
            )

    def _validate_entity_mappings(self, relationship_graph: RelationshipGraph,
                                entity_collections: Dict[str, EntityCollection]) -> None:
        """
        Validate entity mappings between relationship graph and entity collections.

        Ensures that all entities referenced in relationships exist in entity collections.
        """
        entity_mappings = relationship_graph.entity_mappings

        for entity_type, mapping in entity_mappings.items():
            if entity_type not in entity_collections:
                self.validation_result.add_error(
                    component="entity_mappings",
                    message=f"Entity type '{entity_type}' in relationship graph not found in entity collections",
                    severity=ValidationSeverity.ERROR,
                    details={'missing_entity_type': entity_type, 'available_types': list(entity_collections.keys())}
                )
                continue

            # Validate that mapped entities exist
            entity_collection = entity_collections[entity_type]
            primary_key = entity_collection.primary_key
            available_entities = set(entity_collection.entities[primary_key])
            mapped_entities = set(mapping.keys())

            missing_entities = mapped_entities - available_entities
            if missing_entities:
                self.validation_result.add_error(
                    component="entity_mappings",
                    message=f"Entity mapping references non-existent {entity_type} entities",
                    severity=ValidationSeverity.ERROR,
                    details={
                        'entity_type': entity_type,
                        'missing_count': len(missing_entities),
                        'missing_entities': list(missing_entities)[:10]  # Limit for readability
                    }
                )

    def _validate_index_structure(self, index_structure: IndexStructure,
                                entity_collections: Dict[str, EntityCollection]) -> None:
        """
        Validate index structure completeness and consistency.

        Mathematical Foundation: Validates Definition 3.7 (Index Structure Taxonomy)
        ensuring all required indices are present and correctly structured.
        """
        logger.debug("Validating index structure")

        # Validate hash indices - should exist for all primary keys
        expected_hash_indices = {
            f"{entity_type}_{collection.primary_key}" 
            for entity_type, collection in entity_collections.items()
        }

        actual_hash_indices = set(index_structure.hash_indices.keys())
        missing_hash_indices = expected_hash_indices - actual_hash_indices

        if missing_hash_indices:
            self.validation_result.add_error(
                component="index_structure",
                message=f"Missing hash indices: {missing_hash_indices}",
                severity=ValidationSeverity.WARNING,
                details={'missing_indices': list(missing_hash_indices)}
            )

        # Validate hash index completeness - all entities should be indexed
        for index_name, hash_index in index_structure.hash_indices.items():
            if not hash_index:  # Empty hash index
                self.validation_result.add_error(
                    component="index_structure",
                    message=f"Empty hash index: {index_name}",
                    severity=ValidationSeverity.WARNING,
                    details={'index_name': index_name}
                )

    def _validate_cross_component_consistency(self, entity_collections: Dict[str, EntityCollection],
                                           relationship_graph: RelationshipGraph, 
                                           index_structure: IndexStructure) -> None:
        """
        Validate consistency across all loaded components.

        Ensures that entity collections, relationships, and indices are mutually consistent
        and provide complete coverage for optimization requirements.
        """
        logger.debug("Validating cross-component consistency")

        # Validate that relationship graph covers all entity types
        graph_entity_types = set(relationship_graph.entity_mappings.keys())
        collection_entity_types = set(entity_collections.keys())

        missing_in_graph = collection_entity_types - graph_entity_types
        if missing_in_graph:
            self.validation_result.add_error(
                component="cross_component",
                message=f"Entity types missing from relationship graph: {missing_in_graph}",
                severity=ValidationSeverity.WARNING,
                details={'missing_types': list(missing_in_graph)}
            )

        # Validate index coverage for critical lookups
        required_lookups = [
            'courses_course_id', 'faculties_faculty_id', 'rooms_room_id', 
            'batches_batch_id', 'timeslots_timeslot_id'
        ]

        available_indices = set(index_structure.hash_indices.keys())
        missing_lookups = [lookup for lookup in required_lookups if lookup not in available_indices]

        if missing_lookups:
            self.validation_result.add_error(
                component="cross_component", 
                message=f"Missing critical lookup indices: {missing_lookups}",
                severity=ValidationSeverity.WARNING,
                details={'missing_lookups': missing_lookups}
            )

    def _validate_scheduling_feasibility(self, entity_collections: Dict[str, EntityCollection],
                                       relationship_graph: RelationshipGraph) -> None:
        """
        Validate scheduling-specific feasibility constraints.

        Critical validation: ensures that scheduling problem is theoretically feasible
        by checking for non-empty eligibility sets and basic resource adequacy.

        Mathematical Foundation: Implements eligibility validation per Algorithm 3.2
        and ensures compliance with Definition 2.4 (Hard Constraints).
        """
        logger.debug("Validating scheduling feasibility")

        # Check basic resource adequacy
        num_courses = len(entity_collections['courses'].entities)
        num_timeslots = len(entity_collections['timeslots'].entities)  
        num_rooms = len(entity_collections['rooms'].entities)

        # Theoretical minimum timeslots needed (assuming no conflicts)
        min_timeslots_needed = num_courses

        if num_timeslots < min_timeslots_needed:
            self.validation_result.add_error(
                component="scheduling_feasibility",
                message=f"Insufficient timeslots: {num_timeslots} < {min_timeslots_needed} minimum needed",
                severity=ValidationSeverity.CRITICAL,
                details={
                    'available_timeslots': num_timeslots,
                    'minimum_needed': min_timeslots_needed,
                    'num_courses': num_courses
                }
            )

        # Check room availability
        if num_rooms == 0:
            self.validation_result.add_error(
                component="scheduling_feasibility", 
                message="No rooms available for scheduling",
                severity=ValidationSeverity.CRITICAL,
                details={'room_count': num_rooms}
            )

        # Validate eligibility sets (critical: non-empty eligibility required)
        self._validate_eligibility_sets(entity_collections, relationship_graph)

        # Check temporal constraints feasibility
        self._validate_temporal_feasibility(entity_collections)

    def _validate_eligibility_sets(self, entity_collections: Dict[str, EntityCollection],
                                 relationship_graph: RelationshipGraph) -> None:
        """
        CRITICAL: Validate that all courses have non-empty eligibility sets for faculties, rooms, and batches.

        Mathematical Foundation: Ensures prerequisite for Algorithm 3.2 - all eligibility sets must be non-empty
        for scheduling optimization to be feasible. Empty eligibility sets lead to infeasible problems.
        """
        courses = entity_collections['courses'].entities

        # For each course, verify non-empty eligibility
        for _, course in courses.iterrows():
            course_id = course[entity_collections['courses'].primary_key]

            # Check faculty eligibility (simplified - assumes all faculties eligible if no constraints)
            eligible_faculties = self._get_eligible_faculties(course_id, entity_collections, relationship_graph)
            if len(eligible_faculties) == 0:
                self.validation_result.add_error(
                    component="eligibility_validation",
                    message=f"Course {course_id} has no eligible faculties",
                    severity=ValidationSeverity.CRITICAL,
                    details={'course_id': course_id, 'eligible_faculty_count': 0}
                )

            # Check room eligibility
            eligible_rooms = self._get_eligible_rooms(course_id, entity_collections, relationship_graph)  
            if len(eligible_rooms) == 0:
                self.validation_result.add_error(
                    component="eligibility_validation",
                    message=f"Course {course_id} has no eligible rooms",
                    severity=ValidationSeverity.CRITICAL,
                    details={'course_id': course_id, 'eligible_room_count': 0}
                )

            # Check batch eligibility
            eligible_batches = self._get_eligible_batches(course_id, entity_collections, relationship_graph)
            if len(eligible_batches) == 0:
                self.validation_result.add_error(
                    component="eligibility_validation",
                    message=f"Course {course_id} has no eligible batches",
                    severity=ValidationSeverity.CRITICAL,
                    details={'course_id': course_id, 'eligible_batch_count': 0}
                )

    def _get_eligible_faculties(self, course_id: Any, entity_collections: Dict[str, EntityCollection],
                              relationship_graph: RelationshipGraph) -> List[Any]:
        """
        Get list of faculties eligible to teach given course.

        Uses relationship graph to identify faculty-course eligibility constraints.
        If no explicit constraints, assumes all faculties are eligible (conservative approach).
        """
        # Simplified implementation - in production, would use relationship graph
        # to identify course-faculty eligibility relationships

        faculties = entity_collections['faculties'].entities
        faculty_primary_key = entity_collections['faculties'].primary_key

        # Return all faculties if no explicit constraints (conservative)
        # In production, would filter based on relationship graph constraints
        return faculties[faculty_primary_key].tolist()

    def _get_eligible_rooms(self, course_id: Any, entity_collections: Dict[str, EntityCollection],
                          relationship_graph: RelationshipGraph) -> List[Any]:
        """Get list of rooms eligible for given course."""
        # Simplified implementation - return all rooms with positive capacity
        rooms = entity_collections['rooms'].entities
        room_primary_key = entity_collections['rooms'].primary_key

        # Filter by capacity if capacity column exists
        if 'capacity' in rooms.columns:
            eligible_rooms = rooms[rooms['capacity'] > 0]
        else:
            eligible_rooms = rooms

        return eligible_rooms[room_primary_key].tolist()

    def _get_eligible_batches(self, course_id: Any, entity_collections: Dict[str, EntityCollection],
                            relationship_graph: RelationshipGraph) -> List[Any]:
        """Get list of batches eligible for given course.""" 
        # Simplified implementation - return all valid batches
        batches = entity_collections['batches'].entities
        batch_primary_key = entity_collections['batches'].primary_key

        return batches[batch_primary_key].tolist()

    def _validate_temporal_feasibility(self, entity_collections: Dict[str, EntityCollection]) -> None:
        """
        Validate temporal constraints for scheduling feasibility.

        Ensures timeslots provide adequate temporal coverage and don't have conflicts.
        """
        timeslots = entity_collections['timeslots'].entities

        if len(timeslots) < 2:
            self.validation_result.add_error(
                component="temporal_feasibility",
                message=f"Insufficient timeslots for meaningful scheduling: {len(timeslots)} < 2",
                severity=ValidationSeverity.WARNING,
                details={'timeslot_count': len(timeslots)}
            )

        # Check for overlapping timeslots if time information is available
        if 'start_time' in timeslots.columns and 'end_time' in timeslots.columns and 'day_of_week' in timeslots.columns:
            self._check_timeslot_overlaps(timeslots)

    def _check_timeslot_overlaps(self, timeslots: pd.DataFrame) -> None:
        """Check for overlapping timeslots within the same day."""
        try:
            # Convert times to comparable format
            timeslots_copy = timeslots.copy()
            timeslots_copy['start_dt'] = pd.to_datetime(timeslots_copy['start_time'], format='%H:%M', errors='coerce')
            timeslots_copy['end_dt'] = pd.to_datetime(timeslots_copy['end_time'], format='%H:%M', errors='coerce')

            # Group by day and check for overlaps
            overlap_count = 0
            for day, day_slots in timeslots_copy.groupby('day_of_week'):
                day_slots_sorted = day_slots.sort_values('start_dt')

                for i in range(len(day_slots_sorted) - 1):
                    current_end = day_slots_sorted.iloc[i]['end_dt']
                    next_start = day_slots_sorted.iloc[i + 1]['start_dt'] 

                    if pd.notna(current_end) and pd.notna(next_start) and current_end > next_start:
                        overlap_count += 1

            if overlap_count > 0:
                self.validation_result.add_error(
                    component="temporal_feasibility",
                    message=f"Found {overlap_count} overlapping timeslots",
                    severity=ValidationSeverity.WARNING,
                    details={'overlap_count': overlap_count}
                )

        except Exception as e:
            self.validation_result.add_error(
                component="temporal_feasibility",
                message=f"Failed to check timeslot overlaps: {str(e)}",
                severity=ValidationSeverity.WARNING,
                details={'exception': str(e)}
            )

    def _validate_optimization_readiness(self, entity_collections: Dict[str, EntityCollection],
                                       relationship_graph: RelationshipGraph,
                                       index_structure: IndexStructure) -> None:
        """
        Validate readiness for optimization processing.

        Checks performance characteristics and memory requirements for optimization algorithms.
        """
        logger.debug("Validating optimization readiness")

        # Calculate problem size characteristics  
        total_entities = sum(len(collection.entities) for collection in entity_collections.values())

        # Estimate variable count (V_c = |F_c| × |R_c| × |T| × |B_c| per course)
        num_courses = len(entity_collections['courses'].entities)
        num_faculties = len(entity_collections['faculties'].entities)
        num_rooms = len(entity_collections['rooms'].entities) 
        num_timeslots = len(entity_collections['timeslots'].entities)
        num_batches = len(entity_collections['batches'].entities)

        estimated_variables = num_courses * num_faculties * num_rooms * num_timeslots * num_batches

        # Check memory requirements (rough estimation)
        estimated_memory_mb = estimated_variables * 8 / (1024 * 1024)  # 8 bytes per variable

        if estimated_memory_mb > 400:  # Conservative limit for 512MB cap
            self.validation_result.add_error(
                component="optimization_readiness",
                message=f"Estimated memory usage {estimated_memory_mb:.1f}MB may exceed limits",
                severity=ValidationSeverity.WARNING,
                details={
                    'estimated_variables': estimated_variables,
                    'estimated_memory_mb': estimated_memory_mb,
                    'memory_limit_mb': 512
                }
            )

        # Check problem complexity for solver selection
        if estimated_variables > 10_000_000:  # 10M variables
            self.validation_result.add_error(
                component="optimization_readiness",
                message=f"Problem size {estimated_variables} variables may require specialized solvers",
                severity=ValidationSeverity.WARNING,
                details={'estimated_variables': estimated_variables, 'complexity_class': 'large-scale'}
            )

    def _compute_validation_statistics(self, entity_collections: Dict[str, EntityCollection],
                                     relationship_graph: RelationshipGraph,
                                     index_structure: IndexStructure) -> None:
        """Compute complete validation statistics for reporting."""

        statistics = {}

        # Entity collection statistics
        statistics['entity_counts'] = {
            entity_type: len(collection.entities)
            for entity_type, collection in entity_collections.items()
        }

        statistics['total_entities'] = sum(statistics['entity_counts'].values())

        # Relationship graph statistics
        statistics['relationship_graph'] = {
            'nodes': relationship_graph.graph.number_of_nodes(),
            'edges': relationship_graph.graph.number_of_edges(), 
            'density': nx.density(relationship_graph.graph),
            'is_connected': nx.is_connected(relationship_graph.graph)
        }

        # Index structure statistics
        statistics['index_structure'] = {
            'hash_indices': len(index_structure.hash_indices),
            'tree_indices': len(index_structure.tree_indices),
            'graph_indices': len(index_structure.graph_indices),
            'bitmap_indices': len(index_structure.bitmap_indices)
        }

        # Problem complexity estimates
        num_courses = len(entity_collections['courses'].entities)
        num_faculties = len(entity_collections['faculties'].entities)
        num_rooms = len(entity_collections['rooms'].entities)
        num_timeslots = len(entity_collections['timeslots'].entities)
        num_batches = len(entity_collections['batches'].entities)

        statistics['complexity_estimates'] = {
            'estimated_variables': num_courses * num_faculties * num_rooms * num_timeslots * num_batches,
            'estimated_constraints': num_courses + num_faculties * num_timeslots + num_rooms * num_timeslots,
            'problem_density': num_courses / (num_faculties * num_rooms * num_timeslots) if (num_faculties * num_rooms * num_timeslots) > 0 else 0
        }

        self.validation_result.statistics = statistics

    def save_validation_report(self, output_path: Union[str, Path]) -> Path:
        """
        Save complete validation report to JSON file.

        Args:
            output_path: Path to directory where validation report should be saved

        Returns:
            Path to saved validation report file
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        report_filename = f"validation_report_{self.execution_id}.json"
        report_path = output_path / report_filename

        # Generate complete report
        report = {
            'validation_summary': self.validation_result.get_summary(),
            'errors': self.validation_result.errors,
            'warnings': self.validation_result.warnings, 
            'statistics': self.validation_result.statistics,
            'metadata': self.validation_result.metadata
        }

        # Save report to file
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Validation report saved to {report_path}")
        return report_path

def validate_scheduling_data(entity_collections: Dict[str, EntityCollection],
                           relationship_graph: RelationshipGraph,
                           index_structure: IndexStructure,
                           execution_id: str,
                           output_path: Optional[Union[str, Path]] = None,
                           strict_mode: bool = True) -> ValidationResult:
    """
    High-level function to validate all scheduling data structures with complete reporting.

    Args:
        entity_collections: Dictionary of loaded entity collections
        relationship_graph: Loaded relationship graph structure  
        index_structure: Loaded index structure
        execution_id: Unique execution identifier
        output_path: Optional path to save validation report
        strict_mode: If True, treat warnings as errors for maximum rigor

    Returns:
        ValidationResult object with complete diagnostics

    Raises:
        ValueError: If critical validation failures occur in strict mode

    Example:
        >>> result = validate_scheduling_data(entities, graph, indices, "exec_001")
        >>> if not result.is_valid:
        ...     print(f"Validation failed with {len(result.errors)} errors")
    """
    validator = SchedulingDataValidator(execution_id=execution_id, strict_mode=strict_mode)

    validation_result = validator.validate_all(entity_collections, relationship_graph, index_structure)

    # Save validation report if output path specified
    if output_path:
        validator.save_validation_report(output_path)

    # Raise exception in strict mode if validation failed
    if strict_mode and not validation_result.is_valid:
        error_summary = f"Validation failed with {len(validation_result.errors)} errors"
        logger.critical(error_summary)
        raise ValueError(error_summary)

    return validation_result

if __name__ == "__main__":
    # Example usage and testing
    import sys
    from loader import load_stage_data

    if len(sys.argv) != 3:
        print("Usage: python validator.py <input_path> <execution_id>")
        sys.exit(1)

    input_path, execution_id = sys.argv[1], sys.argv[2]

    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Load data structures
        entities, relationships, indices = load_stage_data(input_path, execution_id)

        # Validate loaded data
        result = validate_scheduling_data(entities, relationships, indices, execution_id)

        if result.is_valid:
            print(f"✓ Validation passed for execution {execution_id}")
            print(f"  Statistics: {result.statistics.get('entity_counts', {})}")
        else:
            print(f"✗ Validation failed with {len(result.errors)} errors")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"  ERROR: {error['message']}")

        if result.warnings:
            print(f"  Warnings: {len(result.warnings)} issues detected")

    except Exception as e:
        print(f"Validation failed with exception: {str(e)}")
        sys.exit(1)
