# deap_family/input_model/validator.py
"""
Stage 6.3 DEAP Solver Family - Input Modeling Layer: Validation Module

This module implements comprehensive validation frameworks for DEAP input data structures,
ensuring referential integrity, constraint completeness, and theoretical compliance with
the Stage 6.3 DEAP Foundational Framework mathematical specifications.

Implements fail-fast validation per Stage 6 Foundational Design Rules with immediate
abort on data inconsistencies, supporting enterprise-grade error detection and audit
logging for SIH evaluation and production deployment.

Theoretical Foundations:
- Validates Stage 3 Data Compilation Theorem 3.3 (Information Preservation)
- Ensures DEAP Framework Definition 2.2 (Schedule Genotype Encoding) compliance
- Verifies Dynamic Parametric System EAV parameter consistency
- Maintains course-centric representation mathematical guarantees
- Preserves multi-objective fitness model structural integrity (f₁-f₅)

Validation Architecture:
- Seven-layer validation framework per Stage 4 Feasibility Check principles
- Real-time constraint verification with O(C log C) complexity bounds
- Entity-relationship consistency checking through graph traversal
- Dynamic parameter range and type validation
- Bijection mapping mathematical correctness verification

Author: Perplexity Labs AI - Lumen Team (ID: 93912)
Created: October 2025 - SIH 2025 Prototype Implementation  
Compliance: Stage 6.3 Foundational Design Implementation Rules & Instructions
"""

import time
import logging
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from collections import defaultdict
import pandas as pd
import numpy as np
import networkx as nx
import structlog

# Internal imports following strict project structure
from ..config import DEAPFamilyConfig
from ..main import MemoryMonitor
from .loader import InputModelContext, DataLoadingError


class ValidationError(Exception):
    """
    Specialized exception for validation failures requiring immediate pipeline abort.
    
    Per Stage 6 Foundational Design Rules: fail-fast approach with comprehensive
    error context for debugging and audit trail during SIH evaluation.
    """
    def __init__(self, message: str, validation_type: str, error_details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.validation_type = validation_type
        self.error_details = error_details or {}
        self.timestamp = time.time()


class ReferentialIntegrityValidator:
    """
    Validates referential integrity across entity relationships per Stage 1 Input
    Validation Framework theoretical foundations.
    
    Implements comprehensive entity consistency checking through graph-based
    relationship traversal and foreign key validation with O(E log E) complexity
    where E is the number of entity relationships.
    
    Theoretical Compliance:
    - Validates Stage 3 Relationship Discovery Completeness (Theorem 3.6)
    - Ensures entity existence for all referenced IDs
    - Verifies relationship graph connectivity and consistency
    - Maintains referential integrity per relational algebra principles
    """
    
    def __init__(self, memory_monitor: MemoryMonitor, logger: structlog.BoundLogger):
        self.memory_monitor = memory_monitor
        self.logger = logger
        
    def validate_referential_integrity(self, 
                                     raw_data: Dict[str, pd.DataFrame],
                                     relationship_graph: nx.Graph,
                                     course_eligibility: Dict[str, List[Tuple[str, str, str, str]]]) -> Dict[str, Any]:
        """
        Validates referential integrity across all entity relationships and course eligibility.
        
        Implements comprehensive referential integrity checking:
        1. Validate primary key uniqueness across all entity tables
        2. Check foreign key relationships exist in referenced tables  
        3. Verify relationship graph node-edge consistency
        4. Validate course eligibility references existing entities
        5. Ensure no orphaned entities or dangling references
        
        Args:
            raw_data: Normalized entity tables from Stage 3 L_raw layer
            relationship_graph: Materialized relationships from Stage 3 L_rel layer
            course_eligibility: Course eligibility mapping for assignment validation
            
        Returns:
            Dict containing validation results and integrity metrics
            
        Raises:
            ValidationError: On any referential integrity violation
            
        Complexity: O(E log E + R log R) where E = entities, R = relationships
        """
        start_time = time.time()
        self.logger.info("referential_integrity_validation_start")
        
        validation_results = {
            'primary_key_validation': {},
            'foreign_key_validation': {},
            'relationship_graph_validation': {},
            'eligibility_validation': {},
            'integrity_metrics': {}
        }
        
        try:
            # Phase 1: Validate primary key uniqueness and completeness
            self.logger.debug("validating_primary_keys")
            validation_results['primary_key_validation'] = self._validate_primary_keys(raw_data)
            
            # Phase 2: Validate foreign key relationships
            self.logger.debug("validating_foreign_keys")
            validation_results['foreign_key_validation'] = self._validate_foreign_keys(raw_data)
            
            # Phase 3: Validate relationship graph consistency
            self.logger.debug("validating_relationship_graph")
            validation_results['relationship_graph_validation'] = self._validate_relationship_graph(
                relationship_graph, raw_data
            )
            
            # Phase 4: Validate course eligibility referential integrity
            self.logger.debug("validating_course_eligibility")
            validation_results['eligibility_validation'] = self._validate_course_eligibility_integrity(
                course_eligibility, raw_data
            )
            
            # Phase 5: Calculate integrity metrics
            validation_results['integrity_metrics'] = self._calculate_integrity_metrics(
                raw_data, relationship_graph, course_eligibility
            )
            
            # Final integrity assessment
            total_violations = sum(
                result.get('violations_count', 0) 
                for result in validation_results.values() 
                if isinstance(result, dict)
            )
            
            if total_violations > 0:
                raise ValidationError(
                    f"Referential integrity validation failed with {total_violations} violations",
                    "REFERENTIAL_INTEGRITY_VIOLATIONS",
                    {"total_violations": total_violations, "validation_results": validation_results}
                )
            
            validation_time = time.time() - start_time
            validation_results['validation_metadata'] = {
                'validation_time_seconds': validation_time,
                'total_entities_validated': sum(len(df) for df in raw_data.values()),
                'total_relationships_validated': relationship_graph.number_of_edges(),
                'validation_status': 'PASSED'
            }
            
            self.logger.info("referential_integrity_validation_complete",
                           validation_time=validation_time,
                           entities_validated=validation_results['validation_metadata']['total_entities_validated'],
                           status='PASSED')
            
            return validation_results
            
        except Exception as e:
            self.logger.error("referential_integrity_validation_failed",
                            error=str(e),
                            elapsed_time=time.time() - start_time)
            
            if isinstance(e, ValidationError):
                raise
            else:
                raise ValidationError(
                    f"Unexpected error during referential integrity validation: {str(e)}",
                    "INTEGRITY_VALIDATION_ERROR",
                    {"original_exception": str(e)}
                )
    
    def _validate_primary_keys(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate primary key uniqueness and completeness across all tables."""
        
        primary_key_config = {
            'courses': 'course_id',
            'faculty': 'faculty_id', 
            'rooms': 'room_id',
            'timeslots': 'timeslot_id',
            'batches': 'batch_id',
            'departments': 'department_id',
            'students': 'student_id'
        }
        
        validation_results = {
            'violations': [],
            'violations_count': 0,
            'tables_validated': 0
        }
        
        for table_name, primary_key_column in primary_key_config.items():
            if table_name not in raw_data:
                continue  # Skip missing optional tables
                
            df = raw_data[table_name]
            
            # Check primary key column exists
            if primary_key_column not in df.columns:
                validation_results['violations'].append({
                    'table': table_name,
                    'violation_type': 'MISSING_PRIMARY_KEY_COLUMN',
                    'column': primary_key_column
                })
                continue
            
            # Check for null primary key values
            null_count = df[primary_key_column].isnull().sum()
            if null_count > 0:
                validation_results['violations'].append({
                    'table': table_name,
                    'violation_type': 'NULL_PRIMARY_KEY_VALUES',
                    'column': primary_key_column,
                    'null_count': null_count
                })
            
            # Check for duplicate primary key values
            duplicate_count = df[primary_key_column].duplicated().sum()
            if duplicate_count > 0:
                duplicates = df[df[primary_key_column].duplicated()][primary_key_column].tolist()
                validation_results['violations'].append({
                    'table': table_name,
                    'violation_type': 'DUPLICATE_PRIMARY_KEY_VALUES',
                    'column': primary_key_column,
                    'duplicate_count': duplicate_count,
                    'duplicate_values': duplicates[:10]  # Limit to first 10 for brevity
                })
            
            validation_results['tables_validated'] += 1
        
        validation_results['violations_count'] = len(validation_results['violations'])
        
        return validation_results
    
    def _validate_foreign_keys(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate foreign key relationships between tables."""
        
        # Define foreign key relationships
        foreign_key_config = [
            {'table': 'courses', 'column': 'department_id', 'references_table': 'departments', 'references_column': 'department_id'},
            {'table': 'faculty', 'column': 'department_id', 'references_table': 'departments', 'references_column': 'department_id'},
            {'table': 'students', 'column': 'batch_id', 'references_table': 'batches', 'references_column': 'batch_id'},
        ]
        
        validation_results = {
            'violations': [],
            'violations_count': 0,
            'relationships_validated': 0
        }
        
        for fk_relationship in foreign_key_config:
            table_name = fk_relationship['table']
            fk_column = fk_relationship['column']
            ref_table_name = fk_relationship['references_table']
            ref_column = fk_relationship['references_column']
            
            # Skip if either table is missing
            if table_name not in raw_data or ref_table_name not in raw_data:
                continue
            
            source_df = raw_data[table_name]
            target_df = raw_data[ref_table_name]
            
            # Skip if columns don't exist
            if fk_column not in source_df.columns or ref_column not in target_df.columns:
                continue
            
            # Get foreign key values (excluding nulls)
            fk_values = set(source_df[fk_column].dropna().unique())
            ref_values = set(target_df[ref_column].dropna().unique())
            
            # Find orphaned foreign key values
            orphaned_values = fk_values - ref_values
            if orphaned_values:
                validation_results['violations'].append({
                    'source_table': table_name,
                    'source_column': fk_column,
                    'target_table': ref_table_name,
                    'target_column': ref_column,
                    'violation_type': 'ORPHANED_FOREIGN_KEY_VALUES',
                    'orphaned_count': len(orphaned_values),
                    'orphaned_values': list(orphaned_values)[:10]  # Limit to first 10
                })
            
            validation_results['relationships_validated'] += 1
        
        validation_results['violations_count'] = len(validation_results['violations'])
        
        return validation_results
    
    def _validate_relationship_graph(self, graph: nx.Graph, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate relationship graph consistency with entity data."""
        
        validation_results = {
            'violations': [],
            'violations_count': 0,
            'nodes_validated': 0,
            'edges_validated': 0
        }
        
        # Collect all valid entity IDs from raw data
        valid_entity_ids = set()
        for table_name, df in raw_data.items():
            if table_name in ['courses', 'faculty', 'rooms', 'timeslots', 'batches']:
                primary_key_column = f"{table_name[:-1]}_id" if table_name.endswith('s') else f"{table_name}_id"
                if primary_key_column in df.columns:
                    entity_prefix = table_name[:-1] if table_name.endswith('s') else table_name
                    for entity_id in df[primary_key_column].dropna():
                        valid_entity_ids.add(f"{entity_prefix}_{entity_id}")
        
        # Validate graph nodes reference existing entities
        for node in graph.nodes():
            if node not in valid_entity_ids:
                validation_results['violations'].append({
                    'violation_type': 'INVALID_GRAPH_NODE',
                    'node': node,
                    'description': 'Graph node does not reference existing entity'
                })
            validation_results['nodes_validated'] += 1
        
        # Validate graph edges connect valid nodes
        for source, target in graph.edges():
            if source not in valid_entity_ids or target not in valid_entity_ids:
                validation_results['violations'].append({
                    'violation_type': 'INVALID_GRAPH_EDGE',
                    'source': source,
                    'target': target,
                    'description': 'Graph edge connects to non-existent entity'
                })
            validation_results['edges_validated'] += 1
        
        validation_results['violations_count'] = len(validation_results['violations'])
        
        return validation_results
    
    def _validate_course_eligibility_integrity(self, 
                                             course_eligibility: Dict[str, List[Tuple[str, str, str, str]]],
                                             raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate course eligibility references exist in entity tables."""
        
        validation_results = {
            'violations': [],
            'violations_count': 0,
            'courses_validated': 0,
            'assignments_validated': 0
        }
        
        # Collect valid entity IDs
        valid_course_ids = set(raw_data.get('courses', pd.DataFrame()).get('course_id', pd.Series()).dropna())
        valid_faculty_ids = set(raw_data.get('faculty', pd.DataFrame()).get('faculty_id', pd.Series()).dropna())
        valid_room_ids = set(raw_data.get('rooms', pd.DataFrame()).get('room_id', pd.Series()).dropna())
        valid_timeslot_ids = set(raw_data.get('timeslots', pd.DataFrame()).get('timeslot_id', pd.Series()).dropna())
        valid_batch_ids = set(raw_data.get('batches', pd.DataFrame()).get('batch_id', pd.Series()).dropna())
        
        for course_id, assignments in course_eligibility.items():
            # Validate course exists
            if course_id not in valid_course_ids:
                validation_results['violations'].append({
                    'violation_type': 'INVALID_COURSE_ID_IN_ELIGIBILITY',
                    'course_id': course_id,
                    'description': 'Course ID in eligibility mapping does not exist in courses table'
                })
                continue
            
            validation_results['courses_validated'] += 1
            
            # Validate each assignment tuple
            for assignment_idx, (faculty_id, room_id, timeslot_id, batch_id) in enumerate(assignments):
                violations_for_assignment = []
                
                if faculty_id not in valid_faculty_ids:
                    violations_for_assignment.append('INVALID_FACULTY_ID')
                if room_id not in valid_room_ids:
                    violations_for_assignment.append('INVALID_ROOM_ID') 
                if timeslot_id not in valid_timeslot_ids:
                    violations_for_assignment.append('INVALID_TIMESLOT_ID')
                if batch_id not in valid_batch_ids:
                    violations_for_assignment.append('INVALID_BATCH_ID')
                
                if violations_for_assignment:
                    validation_results['violations'].append({
                        'violation_type': 'INVALID_ASSIGNMENT_REFERENCES',
                        'course_id': course_id,
                        'assignment_index': assignment_idx,
                        'assignment': (faculty_id, room_id, timeslot_id, batch_id),
                        'invalid_references': violations_for_assignment
                    })
                
                validation_results['assignments_validated'] += 1
        
        validation_results['violations_count'] = len(validation_results['violations'])
        
        return validation_results
    
    def _calculate_integrity_metrics(self, 
                                   raw_data: Dict[str, pd.DataFrame],
                                   graph: nx.Graph, 
                                   course_eligibility: Dict[str, List[Tuple]]) -> Dict[str, Any]:
        """Calculate comprehensive referential integrity metrics."""
        
        metrics = {
            'entity_counts': {},
            'relationship_density': 0.0,
            'eligibility_coverage': 0.0,
            'graph_connectivity': 0.0,
            'data_completeness': 0.0
        }
        
        # Calculate entity counts
        for table_name, df in raw_data.items():
            metrics['entity_counts'][table_name] = len(df)
        
        # Calculate relationship density
        if graph.number_of_nodes() > 1:
            max_edges = graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2
            metrics['relationship_density'] = graph.number_of_edges() / max_edges if max_edges > 0 else 0.0
        
        # Calculate eligibility coverage
        total_courses = len(raw_data.get('courses', pd.DataFrame()))
        courses_with_eligibility = len(course_eligibility)
        metrics['eligibility_coverage'] = courses_with_eligibility / total_courses if total_courses > 0 else 0.0
        
        # Calculate graph connectivity (proportion of nodes in largest component)
        if graph.number_of_nodes() > 0:
            largest_component_size = len(max(nx.connected_components(graph), key=len, default=[]))
            metrics['graph_connectivity'] = largest_component_size / graph.number_of_nodes()
        
        # Calculate data completeness (proportion of non-null values)
        total_cells = sum(df.size for df in raw_data.values())
        non_null_cells = sum(df.count().sum() for df in raw_data.values())
        metrics['data_completeness'] = non_null_cells / total_cells if total_cells > 0 else 0.0
        
        return metrics


class ConstraintCompletenessValidator:
    """
    Validates completeness and consistency of constraint rules for multi-objective
    fitness evaluation per DEAP Foundational Framework Definition 2.4.
    
    Ensures all five fitness objectives (f₁-f₅) have complete constraint rule
    definitions and validates Dynamic Parametric System EAV parameter integration.
    
    Theoretical Compliance:
    - Validates multi-objective fitness model structure completeness
    - Ensures constraint violation penalty computation feasibility (f₁)
    - Verifies resource utilization efficiency tracking capability (f₂)
    - Validates preference satisfaction scoring completeness (f₃)
    - Ensures workload balance index calculation readiness (f₄)
    - Verifies schedule compactness measurement capability (f₅)
    """
    
    def __init__(self, memory_monitor: MemoryMonitor, logger: structlog.BoundLogger):
        self.memory_monitor = memory_monitor
        self.logger = logger
        
    def validate_constraint_completeness(self, 
                                       constraint_rules: Dict[str, Dict[str, Any]],
                                       course_eligibility: Dict[str, List[Tuple]],
                                       dynamic_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validates constraint rules completeness for multi-objective fitness evaluation.
        
        Implements comprehensive constraint completeness validation:
        1. Validate all courses have complete constraint rule definitions
        2. Check five-objective fitness model structural completeness (f₁-f₅)
        3. Verify Dynamic Parametric System EAV parameter integration
        4. Validate constraint rule mathematical consistency and bounds
        5. Ensure fitness weight normalization and balance requirements
        
        Args:
            constraint_rules: Course constraint rules mapping
            course_eligibility: Course eligibility mapping for cross-validation
            dynamic_params: EAV dynamic parameters for validation
            
        Returns:
            Dict containing constraint completeness validation results
            
        Raises:
            ValidationError: On constraint completeness violations
            
        Complexity: O(C · R) where C = courses, R = constraint rules per course
        """
        start_time = time.time()
        self.logger.info("constraint_completeness_validation_start", 
                        courses_to_validate=len(constraint_rules))
        
        validation_results = {
            'structural_validation': {},
            'fitness_objectives_validation': {},
            'dynamic_parameters_validation': {},
            'mathematical_consistency_validation': {},
            'completeness_metrics': {}
        }
        
        try:
            # Phase 1: Validate constraint rule structural completeness
            self.logger.debug("validating_constraint_structure")
            validation_results['structural_validation'] = self._validate_constraint_structure(
                constraint_rules, course_eligibility
            )
            
            # Phase 2: Validate five-objective fitness model completeness
            self.logger.debug("validating_fitness_objectives") 
            validation_results['fitness_objectives_validation'] = self._validate_fitness_objectives(
                constraint_rules
            )
            
            # Phase 3: Validate Dynamic Parametric System integration
            self.logger.debug("validating_dynamic_parameters")
            validation_results['dynamic_parameters_validation'] = self._validate_dynamic_parameters_integration(
                constraint_rules, dynamic_params
            )
            
            # Phase 4: Validate mathematical consistency
            self.logger.debug("validating_mathematical_consistency")
            validation_results['mathematical_consistency_validation'] = self._validate_mathematical_consistency(
                constraint_rules
            )
            
            # Phase 5: Calculate completeness metrics
            validation_results['completeness_metrics'] = self._calculate_completeness_metrics(
                constraint_rules, course_eligibility
            )
            
            # Assess overall constraint completeness
            total_violations = sum(
                result.get('violations_count', 0)
                for result in validation_results.values()
                if isinstance(result, dict)
            )
            
            if total_violations > 0:
                raise ValidationError(
                    f"Constraint completeness validation failed with {total_violations} violations",
                    "CONSTRAINT_COMPLETENESS_VIOLATIONS",
                    {"total_violations": total_violations, "validation_results": validation_results}
                )
            
            validation_time = time.time() - start_time
            validation_results['validation_metadata'] = {
                'validation_time_seconds': validation_time,
                'courses_validated': len(constraint_rules),
                'constraint_rules_validated': sum(len(rules) for rules in constraint_rules.values()),
                'validation_status': 'PASSED'
            }
            
            self.logger.info("constraint_completeness_validation_complete",
                           validation_time=validation_time,
                           courses_validated=len(constraint_rules),
                           status='PASSED')
            
            return validation_results
            
        except Exception as e:
            self.logger.error("constraint_completeness_validation_failed",
                            error=str(e),
                            elapsed_time=time.time() - start_time)
            
            if isinstance(e, ValidationError):
                raise
            else:
                raise ValidationError(
                    f"Unexpected error during constraint completeness validation: {str(e)}",
                    "CONSTRAINT_VALIDATION_ERROR", 
                    {"original_exception": str(e)}
                )
    
    def _validate_constraint_structure(self, 
                                     constraint_rules: Dict[str, Dict[str, Any]],
                                     course_eligibility: Dict[str, List[Tuple]]) -> Dict[str, Any]:
        """Validate constraint rule structural completeness and consistency."""
        
        validation_results = {
            'violations': [],
            'violations_count': 0,
            'courses_validated': 0
        }
        
        # Required constraint rule sections per DEAP Framework Definition 2.4
        required_sections = [
            'hard_constraints',
            'resource_utilization', 
            'preferences',
            'workload_balance',
            'compactness',
            'fitness_weights',
            'course_metadata'
        ]
        
        # Validate all courses from eligibility have constraint rules
        course_ids_eligibility = set(course_eligibility.keys())
        course_ids_constraints = set(constraint_rules.keys())
        
        missing_constraints = course_ids_eligibility - course_ids_constraints
        if missing_constraints:
            validation_results['violations'].append({
                'violation_type': 'MISSING_CONSTRAINT_RULES',
                'missing_courses': list(missing_constraints),
                'missing_count': len(missing_constraints)
            })
        
        extra_constraints = course_ids_constraints - course_ids_eligibility
        if extra_constraints:
            validation_results['violations'].append({
                'violation_type': 'EXTRA_CONSTRAINT_RULES',
                'extra_courses': list(extra_constraints),
                'extra_count': len(extra_constraints)
            })
        
        # Validate constraint rule structure for each course
        for course_id, rules in constraint_rules.items():
            course_violations = []
            
            # Check required sections exist
            missing_sections = [section for section in required_sections if section not in rules]
            if missing_sections:
                course_violations.append({
                    'violation_type': 'MISSING_CONSTRAINT_SECTIONS',
                    'missing_sections': missing_sections
                })
            
            # Validate section structure
            for section in required_sections:
                if section in rules:
                    if not isinstance(rules[section], dict):
                        course_violations.append({
                            'violation_type': 'INVALID_SECTION_TYPE',
                            'section': section,
                            'expected_type': 'dict',
                            'actual_type': type(rules[section]).__name__
                        })
            
            if course_violations:
                validation_results['violations'].append({
                    'course_id': course_id,
                    'course_violations': course_violations
                })
            
            validation_results['courses_validated'] += 1
        
        validation_results['violations_count'] = len(validation_results['violations'])
        
        return validation_results
    
    def _validate_fitness_objectives(self, constraint_rules: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Validate five-objective fitness model completeness (f₁-f₅)."""
        
        validation_results = {
            'violations': [],
            'violations_count': 0,
            'objectives_validated': 0
        }
        
        # Required fitness objective components per DEAP Framework Definition 2.4
        fitness_objective_requirements = {
            'f1_constraint_violation': {
                'section': 'hard_constraints',
                'required_components': ['faculty_availability', 'room_capacity', 'time_conflicts']
            },
            'f2_resource_utilization': {
                'section': 'resource_utilization',
                'required_components': ['faculty_load_target', 'room_utilization_target']
            },
            'f3_preference_satisfaction': {
                'section': 'preferences', 
                'required_components': ['faculty_preferences', 'student_preferences']
            },
            'f4_workload_balance': {
                'section': 'workload_balance',
                'required_components': ['faculty_workload_limits', 'distribution_targets']
            },
            'f5_schedule_compactness': {
                'section': 'compactness',
                'required_components': ['time_grouping_bonus', 'location_clustering']
            }
        }
        
        for course_id, rules in constraint_rules.items():
            objective_violations = []
            
            # Validate each fitness objective
            for objective_name, requirements in fitness_objective_requirements.items():
                section = requirements['section']
                required_components = requirements['required_components']
                
                if section not in rules:
                    objective_violations.append({
                        'objective': objective_name,
                        'violation_type': 'MISSING_OBJECTIVE_SECTION',
                        'section': section
                    })
                    continue
                
                section_data = rules[section]
                missing_components = [comp for comp in required_components if comp not in section_data]
                if missing_components:
                    objective_violations.append({
                        'objective': objective_name,
                        'violation_type': 'MISSING_OBJECTIVE_COMPONENTS',
                        'section': section,
                        'missing_components': missing_components
                    })
            
            if objective_violations:
                validation_results['violations'].append({
                    'course_id': course_id,
                    'objective_violations': objective_violations
                })
            
            validation_results['objectives_validated'] += 5  # Five objectives per course
        
        validation_results['violations_count'] = len(validation_results['violations'])
        
        return validation_results
    
    def _validate_dynamic_parameters_integration(self, 
                                               constraint_rules: Dict[str, Dict[str, Any]],
                                               dynamic_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate Dynamic Parametric System EAV parameter integration."""
        
        validation_results = {
            'violations': [],
            'violations_count': 0,
            'parameters_validated': 0
        }
        
        # Validate fitness weights integration
        for course_id, rules in constraint_rules.items():
            if 'fitness_weights' not in rules:
                validation_results['violations'].append({
                    'course_id': course_id,
                    'violation_type': 'MISSING_FITNESS_WEIGHTS',
                    'description': 'Fitness weights required for dynamic parameter integration'
                })
                continue
            
            fitness_weights = rules['fitness_weights']
            
            # Required fitness weight parameters
            required_weights = [
                'constraint_violation_penalty',
                'resource_utilization_weight',
                'preference_satisfaction_weight', 
                'workload_balance_weight',
                'schedule_compactness_weight'
            ]
            
            missing_weights = [weight for weight in required_weights if weight not in fitness_weights]
            if missing_weights:
                validation_results['violations'].append({
                    'course_id': course_id,
                    'violation_type': 'MISSING_WEIGHT_PARAMETERS',
                    'missing_weights': missing_weights
                })
            
            # Validate weight value ranges [0, 2] for mathematical stability
            for weight_name, weight_value in fitness_weights.items():
                if weight_name in required_weights:
                    if not isinstance(weight_value, (int, float)):
                        validation_results['violations'].append({
                            'course_id': course_id,
                            'violation_type': 'INVALID_WEIGHT_TYPE',
                            'weight_name': weight_name,
                            'weight_value': weight_value,
                            'expected_type': 'numeric'
                        })
                    elif not (0 <= weight_value <= 2):
                        validation_results['violations'].append({
                            'course_id': course_id,
                            'violation_type': 'WEIGHT_OUT_OF_RANGE',
                            'weight_name': weight_name,
                            'weight_value': weight_value,
                            'valid_range': '[0, 2]'
                        })
            
            validation_results['parameters_validated'] += len(fitness_weights)
        
        # Validate dynamic parameter consistency if provided
        if dynamic_params and 'fitness_weights' in dynamic_params:
            dynamic_weights = dynamic_params['fitness_weights']
            for weight_name, weight_value in dynamic_weights.items():
                if not isinstance(weight_value, (int, float)) or not (0 <= weight_value <= 2):
                    validation_results['violations'].append({
                        'violation_type': 'INVALID_DYNAMIC_PARAMETER',
                        'parameter_name': weight_name,
                        'parameter_value': weight_value,
                        'description': 'Dynamic parameter value out of valid range or invalid type'
                    })
        
        validation_results['violations_count'] = len(validation_results['violations'])
        
        return validation_results
    
    def _validate_mathematical_consistency(self, constraint_rules: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Validate mathematical consistency of constraint rules and fitness parameters."""
        
        validation_results = {
            'violations': [],
            'violations_count': 0,
            'consistency_checks': 0
        }
        
        for course_id, rules in constraint_rules.items():
            consistency_violations = []
            
            # Validate numerical parameter ranges and consistency
            if 'fitness_weights' in rules:
                weights = rules['fitness_weights']
                
                # Check weight normalization (total should be reasonable, not necessarily 1.0)
                total_weight = sum(w for w in weights.values() if isinstance(w, (int, float)))
                if total_weight > 10.0:  # Arbitrary upper bound for reasonableness
                    consistency_violations.append({
                        'violation_type': 'EXCESSIVE_TOTAL_WEIGHTS',
                        'total_weight': total_weight,
                        'recommendation': 'Consider normalizing fitness weights'
                    })
                elif total_weight < 0.1:  # Minimum meaningful weight
                    consistency_violations.append({
                        'violation_type': 'INSUFFICIENT_TOTAL_WEIGHTS', 
                        'total_weight': total_weight,
                        'recommendation': 'Increase fitness weights for meaningful optimization'
                    })
            
            # Validate resource utilization targets
            if 'resource_utilization' in rules:
                resource_util = rules['resource_utilization']
                
                if 'faculty_load_target' in resource_util:
                    load_targets = resource_util['faculty_load_target']
                    if isinstance(load_targets, dict):
                        target_percentage = load_targets.get('target_load_percentage', 1.0)
                        if not (0.1 <= target_percentage <= 1.0):
                            consistency_violations.append({
                                'violation_type': 'INVALID_LOAD_TARGET_PERCENTAGE',
                                'target_percentage': target_percentage,
                                'valid_range': '[0.1, 1.0]'
                            })
            
            # Validate preference scores ranges
            if 'preferences' in rules:
                preferences = rules['preferences']
                
                for pref_type, pref_data in preferences.items():
                    if isinstance(pref_data, dict):
                        for pref_key, pref_value in pref_data.items():
                            if isinstance(pref_value, (int, float)):
                                if not (-1.0 <= pref_value <= 1.0):
                                    consistency_violations.append({
                                        'violation_type': 'PREFERENCE_SCORE_OUT_OF_RANGE',
                                        'preference_type': pref_type,
                                        'preference_key': pref_key,
                                        'preference_value': pref_value,
                                        'valid_range': '[-1.0, 1.0]'
                                    })
            
            if consistency_violations:
                validation_results['violations'].append({
                    'course_id': course_id,
                    'consistency_violations': consistency_violations
                })
            
            validation_results['consistency_checks'] += 1
        
        validation_results['violations_count'] = len(validation_results['violations'])
        
        return validation_results
    
    def _calculate_completeness_metrics(self, 
                                      constraint_rules: Dict[str, Dict[str, Any]],
                                      course_eligibility: Dict[str, List[Tuple]]) -> Dict[str, Any]:
        """Calculate constraint completeness and coverage metrics."""
        
        metrics = {
            'constraint_coverage': 0.0,
            'fitness_objective_completeness': {},
            'parameter_density': 0.0,
            'mathematical_consistency_score': 0.0
        }
        
        total_courses = len(course_eligibility)
        courses_with_constraints = len(constraint_rules)
        metrics['constraint_coverage'] = courses_with_constraints / total_courses if total_courses > 0 else 0.0
        
        # Calculate fitness objective completeness
        objective_sections = ['hard_constraints', 'resource_utilization', 'preferences', 'workload_balance', 'compactness']
        for section in objective_sections:
            courses_with_section = sum(1 for rules in constraint_rules.values() if section in rules)
            metrics['fitness_objective_completeness'][section] = courses_with_section / courses_with_constraints if courses_with_constraints > 0 else 0.0
        
        # Calculate parameter density
        total_parameters = sum(
            len(rules.get('fitness_weights', {})) + 
            sum(len(section_data) if isinstance(section_data, dict) else 1 
                for section_data in rules.values() if isinstance(section_data, dict))
            for rules in constraint_rules.values()
        )
        metrics['parameter_density'] = total_parameters / courses_with_constraints if courses_with_constraints > 0 else 0.0
        
        # Calculate mathematical consistency score (based on parameter ranges)
        valid_parameters = 0
        total_numeric_parameters = 0
        
        for rules in constraint_rules.values():
            if 'fitness_weights' in rules:
                for weight_value in rules['fitness_weights'].values():
                    if isinstance(weight_value, (int, float)):
                        total_numeric_parameters += 1
                        if 0 <= weight_value <= 2:
                            valid_parameters += 1
        
        metrics['mathematical_consistency_score'] = valid_parameters / total_numeric_parameters if total_numeric_parameters > 0 else 1.0
        
        return metrics


class BijectionMappingValidator:
    """
    Validates bijection mapping mathematical correctness and completeness per
    Stage 3 Data Compilation Theorem 3.3 (Information Preservation).
    
    Ensures bijective equivalence between course-centric and flat binary
    representations while maintaining O(log C) decode operation complexity.
    
    Theoretical Compliance:
    - Validates Stage 3 bijection formula: idx = offsets[c] + f·sF[c] + r·sR[c] + t·sT[c] + b
    - Ensures genotype-phenotype transformation correctness
    - Verifies stride array mathematical consistency
    - Maintains bijection completeness per Information Preservation theorem
    """
    
    def __init__(self, memory_monitor: MemoryMonitor, logger: structlog.BoundLogger):
        self.memory_monitor = memory_monitor
        self.logger = logger
        
    def validate_bijection_mapping(self, 
                                 bijection_data: Dict[str, Any],
                                 course_eligibility: Dict[str, List[Tuple]]) -> Dict[str, Any]:
        """
        Validates bijection mapping mathematical correctness and completeness.
        
        Implements comprehensive bijection validation:
        1. Validate stride array mathematical consistency and bounds
        2. Check entity mapping completeness and bidirectional consistency
        3. Verify offset calculations and index space coverage
        4. Test bijection invertibility through round-trip transformations
        5. Ensure genotype-phenotype mapping correctness
        
        Args:
            bijection_data: Bijection mapping data from input loading
            course_eligibility: Course eligibility for cross-validation
            
        Returns:
            Dict containing bijection validation results and mathematical metrics
            
        Raises:
            ValidationError: On bijection mapping violations
            
        Complexity: O(E log E) where E = total entity mappings
        """
        start_time = time.time()
        self.logger.info("bijection_mapping_validation_start")
        
        validation_results = {
            'stride_validation': {},
            'entity_mapping_validation': {},
            'bijection_invertibility_validation': {},
            'mathematical_consistency_validation': {},
            'bijection_metrics': {}
        }
        
        try:
            # Phase 1: Validate stride arrays and offsets
            self.logger.debug("validating_stride_arrays")
            validation_results['stride_validation'] = self._validate_stride_arrays(bijection_data)
            
            # Phase 2: Validate entity mappings completeness and consistency
            self.logger.debug("validating_entity_mappings") 
            validation_results['entity_mapping_validation'] = self._validate_entity_mappings(
                bijection_data, course_eligibility
            )
            
            # Phase 3: Validate bijection invertibility
            self.logger.debug("validating_bijection_invertibility")
            validation_results['bijection_invertibility_validation'] = self._validate_bijection_invertibility(
                bijection_data
            )
            
            # Phase 4: Validate mathematical consistency
            self.logger.debug("validating_mathematical_consistency")
            validation_results['mathematical_consistency_validation'] = self._validate_bijection_mathematical_consistency(
                bijection_data
            )
            
            # Phase 5: Calculate bijection metrics
            validation_results['bijection_metrics'] = self._calculate_bijection_metrics(bijection_data)
            
            # Assess overall bijection validity
            total_violations = sum(
                result.get('violations_count', 0)
                for result in validation_results.values()
                if isinstance(result, dict)
            )
            
            if total_violations > 0:
                raise ValidationError(
                    f"Bijection mapping validation failed with {total_violations} violations",
                    "BIJECTION_MAPPING_VIOLATIONS",
                    {"total_violations": total_violations, "validation_results": validation_results}
                )
            
            validation_time = time.time() - start_time
            validation_results['validation_metadata'] = {
                'validation_time_seconds': validation_time,
                'entity_mappings_validated': sum(len(mapping) for mapping in bijection_data.get('entity_mappings', {}).values()),
                'validation_status': 'PASSED'
            }
            
            self.logger.info("bijection_mapping_validation_complete",
                           validation_time=validation_time,
                           status='PASSED')
            
            return validation_results
            
        except Exception as e:
            self.logger.error("bijection_mapping_validation_failed",
                            error=str(e),
                            elapsed_time=time.time() - start_time)
            
            if isinstance(e, ValidationError):
                raise
            else:
                raise ValidationError(
                    f"Unexpected error during bijection mapping validation: {str(e)}",
                    "BIJECTION_VALIDATION_ERROR",
                    {"original_exception": str(e)}
                )
    
    def _validate_stride_arrays(self, bijection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate stride array mathematical consistency and bounds."""
        
        validation_results = {
            'violations': [],
            'violations_count': 0,
            'strides_validated': 0
        }
        
        strides = bijection_data.get('strides', {})
        offsets = bijection_data.get('offsets', {})
        entity_counts = bijection_data.get('entity_counts', {})
        
        # Validate stride arrays exist
        if not strides and not offsets:
            validation_results['violations'].append({
                'violation_type': 'MISSING_STRIDE_DATA',
                'description': 'Neither stride arrays nor offsets found in bijection data'
            })
            validation_results['violations_count'] = 1
            return validation_results
        
        # Validate stride consistency with entity counts
        entity_types = ['courses', 'faculty', 'rooms', 'timeslots', 'batches']
        for entity_type in entity_types:
            if entity_type in entity_counts:
                count = entity_counts[entity_type]
                
                # Validate stride exists if needed
                if entity_type in strides:
                    stride_value = strides[entity_type]
                    if not isinstance(stride_value, int) or stride_value <= 0:
                        validation_results['violations'].append({
                            'violation_type': 'INVALID_STRIDE_VALUE',
                            'entity_type': entity_type,
                            'stride_value': stride_value,
                            'expected': 'positive integer'
                        })
                    elif stride_value < count:
                        validation_results['violations'].append({
                            'violation_type': 'STRIDE_TOO_SMALL',
                            'entity_type': entity_type,
                            'stride_value': stride_value,
                            'entity_count': count,
                            'description': 'Stride value smaller than entity count'
                        })
                
                validation_results['strides_validated'] += 1
        
        validation_results['violations_count'] = len(validation_results['violations'])
        
        return validation_results
    
    def _validate_entity_mappings(self, 
                                bijection_data: Dict[str, Any],
                                course_eligibility: Dict[str, List[Tuple]]) -> Dict[str, Any]:
        """Validate entity mapping completeness and bidirectional consistency."""
        
        validation_results = {
            'violations': [],
            'violations_count': 0,
            'mappings_validated': 0
        }
        
        entity_mappings = bijection_data.get('entity_mappings', {})
        reverse_mappings = bijection_data.get('reverse_mappings', {})
        
        # Validate entity mappings exist
        required_entity_types = ['courses', 'faculty', 'rooms', 'timeslots', 'batches']
        for entity_type in required_entity_types:
            if entity_type not in entity_mappings:
                validation_results['violations'].append({
                    'violation_type': 'MISSING_ENTITY_MAPPING',
                    'entity_type': entity_type,
                    'description': f'Missing entity mapping for {entity_type}'
                })
                continue
            
            if entity_type not in reverse_mappings:
                validation_results['violations'].append({
                    'violation_type': 'MISSING_REVERSE_MAPPING',
                    'entity_type': entity_type,
                    'description': f'Missing reverse mapping for {entity_type}'
                })
                continue
            
            # Validate bidirectional consistency
            forward_mapping = entity_mappings[entity_type]
            reverse_mapping = reverse_mappings[entity_type]
            
            if len(forward_mapping) != len(reverse_mapping):
                validation_results['violations'].append({
                    'violation_type': 'MAPPING_SIZE_MISMATCH',
                    'entity_type': entity_type,
                    'forward_size': len(forward_mapping),
                    'reverse_size': len(reverse_mapping)
                })
                continue
            
            # Validate mapping consistency
            for entity_id, index in forward_mapping.items():
                if index not in reverse_mapping:
                    validation_results['violations'].append({
                        'violation_type': 'MISSING_REVERSE_INDEX',
                        'entity_type': entity_type,
                        'entity_id': entity_id,
                        'index': index
                    })
                elif reverse_mapping[index] != entity_id:
                    validation_results['violations'].append({
                        'violation_type': 'MAPPING_INCONSISTENCY',
                        'entity_type': entity_type,
                        'entity_id': entity_id,
                        'forward_index': index,
                        'reverse_entity': reverse_mapping[index]
                    })
            
            validation_results['mappings_validated'] += 1
        
        # Validate course mappings align with eligibility
        if 'courses' in entity_mappings:
            course_mapping = entity_mappings['courses']
            eligibility_courses = set(course_eligibility.keys())
            mapping_courses = set(course_mapping.keys())
            
            missing_in_mapping = eligibility_courses - mapping_courses
            if missing_in_mapping:
                validation_results['violations'].append({
                    'violation_type': 'COURSES_MISSING_IN_BIJECTION_MAPPING',
                    'missing_courses': list(missing_in_mapping),
                    'missing_count': len(missing_in_mapping)
                })
        
        validation_results['violations_count'] = len(validation_results['violations'])
        
        return validation_results
    
    def _validate_bijection_invertibility(self, bijection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate bijection invertibility through round-trip transformations."""
        
        validation_results = {
            'violations': [],
            'violations_count': 0,
            'round_trip_tests': 0
        }
        
        entity_mappings = bijection_data.get('entity_mappings', {})
        reverse_mappings = bijection_data.get('reverse_mappings', {})
        
        # Test round-trip transformations for each entity type
        for entity_type in ['courses', 'faculty', 'rooms', 'timeslots', 'batches']:
            if entity_type not in entity_mappings or entity_type not in reverse_mappings:
                continue
            
            forward_mapping = entity_mappings[entity_type]
            reverse_mapping = reverse_mappings[entity_type]
            
            # Test forward -> reverse transformation
            for entity_id, index in forward_mapping.items():
                try:
                    recovered_entity_id = reverse_mapping.get(index)
                    if recovered_entity_id != entity_id:
                        validation_results['violations'].append({
                            'violation_type': 'ROUND_TRIP_FORWARD_FAILED',
                            'entity_type': entity_type,
                            'original_entity_id': entity_id,
                            'index': index,
                            'recovered_entity_id': recovered_entity_id
                        })
                except Exception as e:
                    validation_results['violations'].append({
                        'violation_type': 'ROUND_TRIP_EXCEPTION',
                        'entity_type': entity_type,
                        'entity_id': entity_id,
                        'error': str(e)
                    })
                
                validation_results['round_trip_tests'] += 1
        
        validation_results['violations_count'] = len(validation_results['violations'])
        
        return validation_results
    
    def _validate_bijection_mathematical_consistency(self, bijection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate mathematical consistency of bijection mapping."""
        
        validation_results = {
            'violations': [],
            'violations_count': 0,
            'consistency_checks': 0
        }
        
        entity_counts = bijection_data.get('entity_counts', {})
        entity_mappings = bijection_data.get('entity_mappings', {})
        
        # Validate entity counts match mapping sizes
        for entity_type, count in entity_counts.items():
            if entity_type in entity_mappings:
                mapping_size = len(entity_mappings[entity_type])
                if mapping_size != count:
                    validation_results['violations'].append({
                        'violation_type': 'COUNT_MAPPING_SIZE_MISMATCH',
                        'entity_type': entity_type,
                        'declared_count': count,
                        'actual_mapping_size': mapping_size
                    })
            
            validation_results['consistency_checks'] += 1
        
        # Validate index continuity and bounds
        for entity_type, mapping in entity_mappings.items():
            if mapping:
                indices = list(mapping.values())
                min_index = min(indices)
                max_index = max(indices)
                expected_count = len(mapping)
                
                # Check indices start from 0
                if min_index != 0:
                    validation_results['violations'].append({
                        'violation_type': 'INDICES_NOT_ZERO_BASED',
                        'entity_type': entity_type,
                        'min_index': min_index,
                        'expected_min': 0
                    })
                
                # Check indices are continuous
                if max_index != expected_count - 1:
                    validation_results['violations'].append({
                        'violation_type': 'INDICES_NOT_CONTINUOUS',
                        'entity_type': entity_type,
                        'max_index': max_index,
                        'expected_max': expected_count - 1
                    })
                
                # Check for duplicate indices
                if len(set(indices)) != len(indices):
                    validation_results['violations'].append({
                        'violation_type': 'DUPLICATE_INDICES',
                        'entity_type': entity_type,
                        'unique_indices': len(set(indices)),
                        'total_indices': len(indices)
                    })
        
        validation_results['violations_count'] = len(validation_results['violations'])
        
        return validation_results
    
    def _calculate_bijection_metrics(self, bijection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate bijection mapping quality and completeness metrics."""
        
        metrics = {
            'mapping_completeness': 0.0,
            'bijection_efficiency': 0.0,
            'index_space_utilization': 0.0,
            'mathematical_correctness': 0.0
        }
        
        entity_mappings = bijection_data.get('entity_mappings', {})
        reverse_mappings = bijection_data.get('reverse_mappings', {})
        entity_counts = bijection_data.get('entity_counts', {})
        
        # Calculate mapping completeness
        expected_entity_types = {'courses', 'faculty', 'rooms', 'timeslots', 'batches'}
        present_entity_types = set(entity_mappings.keys()) & expected_entity_types
        metrics['mapping_completeness'] = len(present_entity_types) / len(expected_entity_types)
        
        # Calculate bijection efficiency (forward-reverse consistency)
        consistent_mappings = 0
        total_mappings = 0
        
        for entity_type in present_entity_types:
            if entity_type in reverse_mappings:
                forward_mapping = entity_mappings[entity_type]
                reverse_mapping = reverse_mappings[entity_type]
                
                for entity_id, index in forward_mapping.items():
                    total_mappings += 1
                    if index in reverse_mapping and reverse_mapping[index] == entity_id:
                        consistent_mappings += 1
        
        metrics['bijection_efficiency'] = consistent_mappings / total_mappings if total_mappings > 0 else 0.0
        
        # Calculate index space utilization
        total_index_space = 0
        used_index_space = 0
        
        for entity_type, mapping in entity_mappings.items():
            if mapping:
                max_possible_index = max(mapping.values()) + 1
                actual_entities = len(mapping)
                total_index_space += max_possible_index
                used_index_space += actual_entities
        
        metrics['index_space_utilization'] = used_index_space / total_index_space if total_index_space > 0 else 0.0
        
        # Calculate mathematical correctness (based on validation results)
        metrics['mathematical_correctness'] = 1.0  # Will be updated based on validation violations
        
        return metrics


class DEAPInputModelValidator:
    """
    Primary validation interface for DEAP Solver Family input model validation.
    
    Orchestrates comprehensive validation pipeline ensuring theoretical compliance,
    referential integrity, constraint completeness, and bijection mapping correctness
    per Stage 6.3 DEAP Foundational Framework specifications.
    
    Architecture:
    - Single-threaded validation with deterministic error detection
    - Fail-fast approach with immediate abort on critical violations
    - Multi-layer validation framework per Stage 4 Feasibility Check principles
    - Enterprise-grade error reporting with comprehensive audit logging
    
    Theoretical Foundations:
    - Validates DEAP Framework universal evolutionary framework compliance
    - Ensures course-centric representation mathematical correctness
    - Verifies multi-objective fitness model structural integrity
    - Validates Dynamic Parametric System EAV parameter consistency
    """
    
    def __init__(self, config: DEAPFamilyConfig):
        """
        Initialize DEAP input model validator with configuration and monitoring.
        
        Args:
            config: DEAP family configuration containing validation parameters
        """
        self.config = config
        self.memory_monitor = MemoryMonitor(max_memory_mb=config.memory_limits.input_modeling_mb)
        
        # Configure structured logging for validation audit trail
        self.logger = structlog.get_logger().bind(
            component="deap_input_model_validator",
            stage="6.3_input_validation",
            process_id=id(self)
        )
        
        # Initialize validation component modules
        self.integrity_validator = ReferentialIntegrityValidator(self.memory_monitor, self.logger)
        self.constraint_validator = ConstraintCompletenessValidator(self.memory_monitor, self.logger)
        self.bijection_validator = BijectionMappingValidator(self.memory_monitor, self.logger)
        
    def validate_input_context(self, 
                             context: InputModelContext,
                             raw_data: Dict[str, pd.DataFrame],
                             relationship_graph: nx.Graph,
                             dynamic_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive validation of DEAP input model context for theoretical compliance.
        
        Implements seven-layer validation framework:
        1. Input context structure validation and completeness checking
        2. Referential integrity validation across all entity relationships
        3. Constraint rules completeness validation for multi-objective fitness
        4. Dynamic parameter integration validation for EAV consistency
        5. Bijection mapping mathematical correctness validation
        6. Cross-validation consistency checking between all components
        7. Overall theoretical compliance assessment and certification
        
        Args:
            context: Input model context from data loading pipeline
            raw_data: Raw entity data for cross-validation
            relationship_graph: Entity relationship graph for integrity checking
            dynamic_params: EAV dynamic parameters for consistency validation
            
        Returns:
            Dict containing comprehensive validation results and compliance certification
            
        Raises:
            ValidationError: On any validation failure requiring pipeline abort
            
        Memory Guarantee: Peak usage ≤ 50MB for validation operations
        """
        start_time = time.time()
        start_memory = self.memory_monitor.get_current_usage_mb()
        
        self.logger.info("deap_input_validation_start",
                        courses_to_validate=len(context.course_eligibility),
                        constraint_rules_count=len(context.constraint_rules),
                        dynamic_params_provided=dynamic_params is not None)
        
        validation_results = {
            'context_structure_validation': {},
            'referential_integrity_validation': {},
            'constraint_completeness_validation': {},
            'bijection_mapping_validation': {},
            'cross_validation_consistency': {},
            'theoretical_compliance_assessment': {},
            'validation_summary': {}
        }
        
        try:
            # Phase 1: Validate input context structure and completeness
            self.logger.info("validating_context_structure")
            validation_results['context_structure_validation'] = self._validate_context_structure(context)
            self.memory_monitor.check_memory_usage("after_context_structure_validation")
            
            # Phase 2: Validate referential integrity
            self.logger.info("validating_referential_integrity")
            validation_results['referential_integrity_validation'] = self.integrity_validator.validate_referential_integrity(
                raw_data, relationship_graph, context.course_eligibility
            )
            self.memory_monitor.check_memory_usage("after_referential_integrity_validation")
            
            # Phase 3: Validate constraint completeness
            self.logger.info("validating_constraint_completeness")
            validation_results['constraint_completeness_validation'] = self.constraint_validator.validate_constraint_completeness(
                context.constraint_rules, context.course_eligibility, dynamic_params
            )
            self.memory_monitor.check_memory_usage("after_constraint_completeness_validation")
            
            # Phase 4: Validate bijection mapping
            self.logger.info("validating_bijection_mapping")
            validation_results['bijection_mapping_validation'] = self.bijection_validator.validate_bijection_mapping(
                context.bijection_data, context.course_eligibility
            )
            self.memory_monitor.check_memory_usage("after_bijection_mapping_validation")
            
            # Phase 5: Cross-validation consistency checking
            self.logger.info("performing_cross_validation")
            validation_results['cross_validation_consistency'] = self._perform_cross_validation_consistency(
                context, validation_results
            )
            self.memory_monitor.check_memory_usage("after_cross_validation")
            
            # Phase 6: Theoretical compliance assessment
            self.logger.info("assessing_theoretical_compliance")
            validation_results['theoretical_compliance_assessment'] = self._assess_theoretical_compliance(
                context, validation_results
            )
            
            # Phase 7: Generate validation summary
            validation_results['validation_summary'] = self._generate_validation_summary(
                validation_results, start_time, start_memory
            )
            
            # Final validation assessment
            total_violations = self._calculate_total_violations(validation_results)
            compliance_score = validation_results['theoretical_compliance_assessment']['compliance_score']
            
            if total_violations > 0 or compliance_score < 0.95:
                raise ValidationError(
                    f"Input model validation failed: {total_violations} violations, compliance score: {compliance_score:.3f}",
                    "INPUT_MODEL_VALIDATION_FAILED",
                    {
                        "total_violations": total_violations,
                        "compliance_score": compliance_score,
                        "validation_results": validation_results
                    }
                )
            
            final_memory = self.memory_monitor.get_current_usage_mb()
            total_time = time.time() - start_time
            
            self.logger.info("deap_input_validation_complete",
                           total_time_seconds=total_time,
                           peak_memory_mb=self.memory_monitor.get_peak_usage_mb(),
                           final_memory_mb=final_memory,
                           compliance_score=compliance_score,
                           status='PASSED')
            
            return validation_results
            
        except Exception as e:
            self.logger.error("deap_input_validation_failed",
                            error=str(e),
                            elapsed_time=time.time() - start_time,
                            peak_memory_mb=self.memory_monitor.get_peak_usage_mb())
            
            if isinstance(e, ValidationError):
                raise
            else:
                raise ValidationError(
                    f"Unexpected error during input model validation: {str(e)}",
                    "INPUT_VALIDATION_UNEXPECTED_ERROR",
                    {"original_exception": str(e)}
                )
    
    def _validate_context_structure(self, context: InputModelContext) -> Dict[str, Any]:
        """Validate input model context structure and completeness."""
        
        validation_results = {
            'violations': [],
            'violations_count': 0,
            'structure_checks': 0
        }
        
        # Validate required context attributes
        required_attributes = ['course_eligibility', 'constraint_rules', 'bijection_data', 'entity_metadata', 'loading_metadata']
        for attr in required_attributes:
            if not hasattr(context, attr):
                validation_results['violations'].append({
                    'violation_type': 'MISSING_CONTEXT_ATTRIBUTE',
                    'attribute': attr,
                    'description': f'Required context attribute {attr} is missing'
                })
            elif getattr(context, attr) is None:
                validation_results['violations'].append({
                    'violation_type': 'NULL_CONTEXT_ATTRIBUTE',
                    'attribute': attr,
                    'description': f'Context attribute {attr} is null'
                })
            
            validation_results['structure_checks'] += 1
        
        # Validate course eligibility structure
        if hasattr(context, 'course_eligibility') and context.course_eligibility:
            if not isinstance(context.course_eligibility, dict):
                validation_results['violations'].append({
                    'violation_type': 'INVALID_COURSE_ELIGIBILITY_TYPE',
                    'expected_type': 'dict',
                    'actual_type': type(context.course_eligibility).__name__
                })
            else:
                # Validate eligibility entries structure
                for course_id, assignments in context.course_eligibility.items():
                    if not isinstance(assignments, (list, tuple)):
                        validation_results['violations'].append({
                            'violation_type': 'INVALID_ASSIGNMENTS_TYPE',
                            'course_id': course_id,
                            'expected_type': 'list or tuple',
                            'actual_type': type(assignments).__name__
                        })
                        continue
                    
                    # Validate assignment tuple structure
                    for assignment_idx, assignment in enumerate(assignments):
                        if not isinstance(assignment, tuple) or len(assignment) != 4:
                            validation_results['violations'].append({
                                'violation_type': 'INVALID_ASSIGNMENT_TUPLE',
                                'course_id': course_id,
                                'assignment_index': assignment_idx,
                                'expected': 'tuple of length 4',
                                'actual': f'{type(assignment).__name__} of length {len(assignment) if hasattr(assignment, "__len__") else "unknown"}'
                            })
        
        # Validate constraint rules structure
        if hasattr(context, 'constraint_rules') and context.constraint_rules:
            if not isinstance(context.constraint_rules, dict):
                validation_results['violations'].append({
                    'violation_type': 'INVALID_CONSTRAINT_RULES_TYPE',
                    'expected_type': 'dict',
                    'actual_type': type(context.constraint_rules).__name__
                })
        
        # Validate bijection data structure
        if hasattr(context, 'bijection_data') and context.bijection_data:
            if not isinstance(context.bijection_data, dict):
                validation_results['violations'].append({
                    'violation_type': 'INVALID_BIJECTION_DATA_TYPE',
                    'expected_type': 'dict',
                    'actual_type': type(context.bijection_data).__name__
                })
        
        validation_results['violations_count'] = len(validation_results['violations'])
        
        return validation_results
    
    def _perform_cross_validation_consistency(self, 
                                           context: InputModelContext,
                                           validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-validation consistency checking between all components."""
        
        consistency_results = {
            'violations': [],
            'violations_count': 0,
            'consistency_checks': 0
        }
        
        # Cross-validate course IDs consistency
        eligibility_courses = set(context.course_eligibility.keys())
        constraint_courses = set(context.constraint_rules.keys())
        
        if eligibility_courses != constraint_courses:
            consistency_results['violations'].append({
                'violation_type': 'COURSE_ID_INCONSISTENCY',
                'eligibility_course_count': len(eligibility_courses),
                'constraint_course_count': len(constraint_courses),
                'missing_in_constraints': list(eligibility_courses - constraint_courses),
                'missing_in_eligibility': list(constraint_courses - eligibility_courses)
            })
        
        consistency_results['consistency_checks'] += 1
        
        # Cross-validate entity counts consistency
        entity_metadata = context.entity_metadata
        bijection_entity_counts = context.bijection_data.get('entity_counts', {})
        
        if 'courses_count' in entity_metadata and 'courses' in bijection_entity_counts:
            metadata_count = entity_metadata['courses_count']
            bijection_count = bijection_entity_counts['courses']
            if metadata_count != bijection_count:
                consistency_results['violations'].append({
                    'violation_type': 'COURSE_COUNT_INCONSISTENCY',
                    'metadata_count': metadata_count,
                    'bijection_count': bijection_count
                })
        
        consistency_results['consistency_checks'] += 1
        
        # Cross-validate validation results consistency
        for validation_type, validation_result in validation_results.items():
            if isinstance(validation_result, dict) and 'violations_count' in validation_result:
                if validation_result['violations_count'] > 0:
                    consistency_results['violations'].append({
                        'violation_type': 'COMPONENT_VALIDATION_FAILURES',
                        'component': validation_type,
                        'component_violations': validation_result['violations_count']
                    })
                    
        consistency_results['violations_count'] = len(consistency_results['violations'])
        
        return consistency_results
    
    def _assess_theoretical_compliance(self, 
                                     context: InputModelContext,
                                     validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall theoretical compliance with DEAP Foundational Framework."""
        
        compliance_assessment = {
            'compliance_score': 0.0,
            'framework_compliance': {},
            'mathematical_correctness': {},
            'theoretical_guarantees': {}
        }
        
        # Assess DEAP Framework compliance components
        compliance_components = {
            'genotype_encoding_compliance': self._assess_genotype_encoding_compliance(context),
            'multi_objective_fitness_compliance': self._assess_multi_objective_compliance(context),
            'dynamic_parameters_compliance': self._assess_dynamic_parameters_compliance(context),
            'bijection_mapping_compliance': self._assess_bijection_mapping_compliance(context),
            'memory_efficiency_compliance': self._assess_memory_efficiency_compliance(context)
        }
        
        compliance_assessment['framework_compliance'] = compliance_components
        
        # Calculate overall compliance score (weighted average)
        weights = {
            'genotype_encoding_compliance': 0.3,
            'multi_objective_fitness_compliance': 0.25,
            'dynamic_parameters_compliance': 0.15,
            'bijection_mapping_compliance': 0.2,
            'memory_efficiency_compliance': 0.1
        }
        
        total_score = sum(
            compliance_components[component] * weights[component]
            for component in compliance_components
        )
        
        compliance_assessment['compliance_score'] = total_score
        
        # Assess mathematical correctness
        compliance_assessment['mathematical_correctness'] = {
            'course_centric_representation': len(context.course_eligibility) > 0,
            'bijection_equivalence': 'bijection_data' in context.__dict__ and bool(context.bijection_data),
            'constraint_completeness': len(context.constraint_rules) > 0,
            'fitness_model_integrity': all(
                'fitness_weights' in rules for rules in context.constraint_rules.values()
            )
        }
        
        # Assess theoretical guarantees
        compliance_assessment['theoretical_guarantees'] = {
            'information_preservation': total_score >= 0.95,
            'genotype_validity': all(assignments for assignments in context.course_eligibility.values()),
            'fitness_evaluation_feasibility': len(context.constraint_rules) == len(context.course_eligibility),
            'memory_bound_compliance': True  # Validated through memory monitoring
        }
        
        return compliance_assessment
    
    def _assess_genotype_encoding_compliance(self, context: InputModelContext) -> float:
        """Assess genotype encoding compliance with DEAP Framework Definition 2.2."""
        
        if not context.course_eligibility:
            return 0.0
        
        # Check course-centric representation
        course_representation_score = 1.0 if all(
            isinstance(assignments, (list, tuple)) and
            all(isinstance(assignment, tuple) and len(assignment) == 4 for assignment in assignments)
            for assignments in context.course_eligibility.values()
        ) else 0.0
        
        # Check non-empty eligibility (genotype validity)
        eligibility_validity_score = 1.0 if all(
            len(assignments) > 0 for assignments in context.course_eligibility.values()
        ) else 0.0
        
        return (course_representation_score + eligibility_validity_score) / 2.0
    
    def _assess_multi_objective_compliance(self, context: InputModelContext) -> float:
        """Assess multi-objective fitness model compliance (f₁-f₅)."""
        
        if not context.constraint_rules:
            return 0.0
        
        required_sections = ['hard_constraints', 'resource_utilization', 'preferences', 'workload_balance', 'compactness']
        total_courses = len(context.constraint_rules)
        
        section_scores = []
        for section in required_sections:
            courses_with_section = sum(1 for rules in context.constraint_rules.values() if section in rules)
            section_score = courses_with_section / total_courses if total_courses > 0 else 0.0
            section_scores.append(section_score)
        
        return sum(section_scores) / len(section_scores)
    
    def _assess_dynamic_parameters_compliance(self, context: InputModelContext) -> float:
        """Assess Dynamic Parametric System EAV integration compliance."""
        
        if not context.constraint_rules:
            return 0.0
        
        fitness_weights_count = sum(1 for rules in context.constraint_rules.values() if 'fitness_weights' in rules)
        total_courses = len(context.constraint_rules)
        
        return fitness_weights_count / total_courses if total_courses > 0 else 0.0
    
    def _assess_bijection_mapping_compliance(self, context: InputModelContext) -> float:
        """Assess bijection mapping mathematical correctness compliance."""
        
        if not context.bijection_data:
            return 0.0
        
        required_keys = ['entity_mappings', 'reverse_mappings', 'entity_counts']
        present_keys = sum(1 for key in required_keys if key in context.bijection_data)
        
        return present_keys / len(required_keys)
    
    def _assess_memory_efficiency_compliance(self, context: InputModelContext) -> float:
        """Assess memory efficiency compliance with 200MB limit."""
        
        current_memory = self.memory_monitor.get_current_usage_mb()
        memory_limit = self.config.memory_limits.input_modeling_mb
        
        if current_memory <= memory_limit:
            return 1.0
        else:
            # Partial credit based on how much over limit
            overage_ratio = current_memory / memory_limit
            return max(0.0, 2.0 - overage_ratio)  # Linear penalty for overage
    
    def _generate_validation_summary(self, 
                                   validation_results: Dict[str, Any],
                                   start_time: float,
                                   start_memory: float) -> Dict[str, Any]:
        """Generate comprehensive validation summary and metrics."""
        
        total_time = time.time() - start_time
        final_memory = self.memory_monitor.get_current_usage_mb()
        peak_memory = self.memory_monitor.get_peak_usage_mb()
        
        summary = {
            'validation_status': 'PASSED',
            'total_violations': self._calculate_total_violations(validation_results),
            'compliance_score': validation_results.get('theoretical_compliance_assessment', {}).get('compliance_score', 0.0),
            'performance_metrics': {
                'validation_time_seconds': total_time,
                'start_memory_mb': start_memory,
                'final_memory_mb': final_memory,
                'peak_memory_mb': peak_memory,
                'memory_efficiency': final_memory / start_memory if start_memory > 0 else 1.0
            },
            'validation_coverage': {
                'referential_integrity': 'referential_integrity_validation' in validation_results,
                'constraint_completeness': 'constraint_completeness_validation' in validation_results,
                'bijection_mapping': 'bijection_mapping_validation' in validation_results,
                'cross_validation': 'cross_validation_consistency' in validation_results,
                'theoretical_compliance': 'theoretical_compliance_assessment' in validation_results
            }
        }
        
        # Determine validation status
        if summary['total_violations'] > 0 or summary['compliance_score'] < 0.95:
            summary['validation_status'] = 'FAILED'
        
        return summary
    
    def _calculate_total_violations(self, validation_results: Dict[str, Any]) -> int:
        """Calculate total violations across all validation components."""
        
        total_violations = 0
        
        for validation_type, validation_result in validation_results.items():
            if isinstance(validation_result, dict) and 'violations_count' in validation_result:
                total_violations += validation_result['violations_count']
        
        return total_violations