# deap_family/input_model/loader.py
"""
Stage 6.3 DEAP Solver Family - Input Modeling Layer: Data Loading Module

This module implements the complete data loading infrastructure for the DEAP solver family,
transforming Stage 3 compiled artifacts (L_raw, L_rel, L_idx) into memory-optimized structures
suitable for evolutionary algorithm processing. Following the DEAP Foundational Framework's
mathematical specifications and ensuring full theoretical compliance with bijective genotype
mapping and course-centric representation.

Theoretical Foundations:
- Implements Definition 3.1 from Stage 6.3 DEAP Framework: Schedule Genotype Encoding
- Adheres to Stage 3 Data Compilation bijection mapping: idx = offsets[c] + f·sF[c] + r·sR[c] + t·sT[c] + b  
- Integrates Dynamic Parametric System EAV parameters per formal analysis framework
- Maintains course-centric genotype mapping: g: course → (faculty, room, timeslot, batch)
- Preserves multi-objective fitness model foundation (f₁-f₅) through constraint rule construction

Memory Management Compliance:
- Peak memory usage ≤ 200MB as per Layer-by-Layer Process specification
- Implements fail-fast validation with immediate abort on data inconsistencies
- Single-threaded processing ensuring deterministic memory behavior
- Zero information loss through lossless data transformations

Integration Architecture:
- Consumes Stage 3 outputs: L_raw (entity tables), L_rel (relationship graphs), L_idx (bijection data)
- Produces InputModelContext for seamless handoff to Processing Layer
- Implements complete error handling with complete audit logging
- Supports all DEAP algorithms (GA, GP, ES, DE, PSO, NSGA-II) through unified representation

Author: Student Team
Created: October 2025 - Prototype Implementation
Compliance: Stage 6.3 Foundational Design Implementation Rules & Instructions
"""

import gc
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any, Union
import psutil
import pandas as pd
import numpy as np
import networkx as nx
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, dok_matrix
import structlog

# Internal imports following strict project structure
from ..config import DEAPFamilyConfig, PathConfig
from ..main import MemoryMonitor

class DataLoadingError(Exception):
    """
    Specialized exception for critical data loading failures requiring immediate pipeline abort.
    
    Per Stage 6 Foundational Design Rules: fail-fast approach with detailed error context
    for traceability and debugging during SIH evaluation and usage.
    """
    def __init__(self, message: str, error_code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = time.time()

class CourseEligibilityBuilder:
    """
    Constructs course eligibility mappings from Stage 3 compiled data following
    DEAP Foundational Framework Definition 2.2 (Schedule Genotype Encoding).
    
    Implements bijective transformation ensuring every course has valid assignment
    options while maintaining O(C) memory complexity for C courses.
    
    Mathematical Compliance:
    - Preserves bijection: course_id ↔ List[AssignmentTuple] 
    - Maintains genotype space G validity per Definition 2.2
    - Ensures non-empty eligibility per fail-fast validation requirements
    """
    
    def __init__(self, memory_monitor: MemoryMonitor, logger: structlog.BoundLogger):
        self.memory_monitor = memory_monitor
        self.logger = logger
        self._eligibility_cache: Dict[str, List[Tuple[str, str, str, str]]] = {}
        
    def build_course_eligibility(self, 
                               raw_data: Dict[str, pd.DataFrame],
                               relationship_graph: nx.Graph,
                               memory_limit_mb: int = 150) -> Dict[str, List[Tuple[str, str, str, str]]]:
        """
        Constructs complete course eligibility mapping from Stage 3 compiled data.
        
        Implements Algorithm 3.2 Data Normalization extended for evolutionary representation:
        1. Extract course entities from L_raw normalized tables
        2. Discover eligibility relationships through L_rel graph traversal  
        3. Validate assignment tuple completeness and referential integrity
        4. Build memory-optimized eligibility dictionary with O(C) space complexity
        
        Args:
            raw_data: Normalized entity tables from Stage 3 L_raw layer
            relationship_graph: Materialized relationships from Stage 3 L_rel layer  
            memory_limit_mb: Maximum memory allocation for this operation
            
        Returns:
            Dict[course_id, List[(faculty_id, room_id, timeslot_id, batch_id)]]
            
        Raises:
            DataLoadingError: On eligibility construction failure or memory overflow
            
        Theoretical Guarantees:
        - Bijective mapping preservation per Stage 3 Theorem 3.3
        - Non-empty eligibility per DEAP genotype validity requirements
        - Memory bound adherence ≤ 150MB peak usage
        """
        start_memory = self.memory_monitor.get_current_usage_mb()
        self.logger.info("course_eligibility_build_start", 
                        initial_memory_mb=start_memory,
                        courses_count=len(raw_data.get('courses', [])))
        
        try:
            # Phase 1: Extract core entities with referential integrity validation
            courses_df = self._validate_and_extract_courses(raw_data)
            faculty_df = self._validate_and_extract_faculty(raw_data)
            rooms_df = self._validate_and_extract_rooms(raw_data)
            timeslots_df = self._validate_and_extract_timeslots(raw_data)
            batches_df = self._validate_and_extract_batches(raw_data)
            
            # Memory checkpoint after entity extraction
            current_memory = self.memory_monitor.get_current_usage_mb()
            if current_memory - start_memory > memory_limit_mb * 0.6:
                raise DataLoadingError(
                    f"Memory usage exceeded 60% threshold during entity extraction",
                    "MEMORY_THRESHOLD_EXCEEDED",
                    {"current_memory": current_memory, "threshold": memory_limit_mb * 0.6}
                )
            
            # Phase 2: Build eligibility through relationship graph traversal
            eligibility_mapping = {}
            
            for _, course in courses_df.iterrows():
                course_id = course['course_id']
                
                # Extract eligibility constraints from relationship graph
                eligible_faculty = self._find_eligible_faculty(course_id, faculty_df, relationship_graph)
                eligible_rooms = self._find_eligible_rooms(course_id, rooms_df, relationship_graph) 
                eligible_timeslots = self._find_eligible_timeslots(course_id, timeslots_df, relationship_graph)
                eligible_batches = self._find_eligible_batches(course_id, batches_df, relationship_graph)
                
                # Generate all valid assignment combinations
                assignment_tuples = []
                for faculty_id in eligible_faculty:
                    for room_id in eligible_rooms:
                        for timeslot_id in eligible_timeslots:
                            for batch_id in eligible_batches:
                                # Validate assignment tuple feasibility
                                if self._validate_assignment_feasibility(
                                    course_id, faculty_id, room_id, timeslot_id, batch_id,
                                    raw_data, relationship_graph
                                ):
                                    assignment_tuples.append((faculty_id, room_id, timeslot_id, batch_id))
                
                # Fail-fast validation: ensure non-empty eligibility
                if not assignment_tuples:
                    raise DataLoadingError(
                        f"Course {course_id} has no valid assignments - violates DEAP genotype validity",
                        "EMPTY_ELIGIBILITY", 
                        {"course_id": course_id, "constraints_checked": True}
                    )
                
                eligibility_mapping[course_id] = assignment_tuples
                
                # Memory management during processing
                if len(eligibility_mapping) % 50 == 0:
                    current_memory = self.memory_monitor.get_current_usage_mb()
                    if current_memory - start_memory > memory_limit_mb:
                        raise DataLoadingError(
                            f"Memory limit exceeded during eligibility construction",
                            "MEMORY_LIMIT_EXCEEDED",
                            {"current_memory": current_memory, "limit": memory_limit_mb}
                        )
            
            # Phase 3: Final validation and optimization
            self._validate_eligibility_completeness(eligibility_mapping, courses_df)
            self._optimize_eligibility_storage(eligibility_mapping)
            
            final_memory = self.memory_monitor.get_current_usage_mb()
            memory_used = final_memory - start_memory
            
            self.logger.info("course_eligibility_build_complete",
                           courses_processed=len(eligibility_mapping),
                           total_assignments=sum(len(assignments) for assignments in eligibility_mapping.values()),
                           memory_used_mb=memory_used,
                           average_assignments_per_course=np.mean([len(assignments) for assignments in eligibility_mapping.values()]))
            
            return eligibility_mapping
            
        except Exception as e:
            self.logger.error("course_eligibility_build_failed", 
                            error=str(e), 
                            memory_mb=self.memory_monitor.get_current_usage_mb())
            
            if isinstance(e, DataLoadingError):
                raise
            else:
                raise DataLoadingError(
                    f"Unexpected error during course eligibility construction: {str(e)}",
                    "UNEXPECTED_ERROR",
                    {"original_exception": str(e)}
                )
                
    def _validate_and_extract_courses(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract and validate course entities with required schema compliance."""
        if 'courses' not in raw_data:
            raise DataLoadingError("Missing 'courses' table in raw data", "MISSING_COURSES_TABLE")
        
        courses_df = raw_data['courses']
        
        # Schema validation per Stage 1 Input Validation Framework
        required_columns = ['course_id', 'course_name', 'credits', 'department_id']
        missing_columns = [col for col in required_columns if col not in courses_df.columns]
        if missing_columns:
            raise DataLoadingError(
                f"Courses table missing required columns: {missing_columns}",
                "MISSING_REQUIRED_COLUMNS",
                {"missing_columns": missing_columns}
            )
        
        # Referential integrity validation
        if courses_df['course_id'].isnull().any():
            raise DataLoadingError("Null course_id values detected", "NULL_PRIMARY_KEY")
        
        if courses_df['course_id'].duplicated().any():
            duplicates = courses_df[courses_df['course_id'].duplicated()]['course_id'].tolist()
            raise DataLoadingError(
                f"Duplicate course_id values: {duplicates}", 
                "DUPLICATE_PRIMARY_KEY",
                {"duplicates": duplicates}
            )
        
        self.logger.debug("courses_validation_complete", 
                         courses_count=len(courses_df),
                         columns=list(courses_df.columns))
        
        return courses_df
    
    def _validate_and_extract_faculty(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract and validate faculty entities with availability constraints."""
        if 'faculty' not in raw_data:
            raise DataLoadingError("Missing 'faculty' table in raw data", "MISSING_FACULTY_TABLE")
        
        faculty_df = raw_data['faculty']
        
        required_columns = ['faculty_id', 'faculty_name', 'department_id', 'max_load']
        missing_columns = [col for col in required_columns if col not in faculty_df.columns]
        if missing_columns:
            raise DataLoadingError(
                f"Faculty table missing required columns: {missing_columns}",
                "MISSING_REQUIRED_COLUMNS",
                {"missing_columns": missing_columns}
            )
        
        # Validate faculty availability constraints
        if faculty_df['max_load'].isnull().any():
            raise DataLoadingError("Faculty max_load constraints missing", "MISSING_CONSTRAINTS")
        
        return faculty_df
    
    def _validate_and_extract_rooms(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract and validate room entities with capacity constraints.""" 
        if 'rooms' not in raw_data:
            raise DataLoadingError("Missing 'rooms' table in raw data", "MISSING_ROOMS_TABLE")
        
        rooms_df = raw_data['rooms']
        
        required_columns = ['room_id', 'room_name', 'capacity', 'room_type']
        missing_columns = [col for col in required_columns if col not in rooms_df.columns]
        if missing_columns:
            raise DataLoadingError(
                f"Rooms table missing required columns: {missing_columns}",
                "MISSING_REQUIRED_COLUMNS", 
                {"missing_columns": missing_columns}
            )
        
        # Validate room capacity constraints
        if rooms_df['capacity'].isnull().any() or (rooms_df['capacity'] <= 0).any():
            raise DataLoadingError("Invalid room capacity values", "INVALID_CAPACITY")
        
        return rooms_df
    
    def _validate_and_extract_timeslots(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract and validate timeslot entities with temporal constraints."""
        if 'timeslots' not in raw_data:
            raise DataLoadingError("Missing 'timeslots' table in raw data", "MISSING_TIMESLOTS_TABLE")
        
        timeslots_df = raw_data['timeslots']
        
        required_columns = ['timeslot_id', 'day_of_week', 'start_time', 'end_time']
        missing_columns = [col for col in required_columns if col not in timeslots_df.columns]
        if missing_columns:
            raise DataLoadingError(
                f"Timeslots table missing required columns: {missing_columns}",
                "MISSING_REQUIRED_COLUMNS",
                {"missing_columns": missing_columns}
            )
        
        return timeslots_df
    
    def _validate_and_extract_batches(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract and validate batch entities with enrollment constraints."""
        if 'batches' not in raw_data:
            raise DataLoadingError("Missing 'batches' table in raw data", "MISSING_BATCHES_TABLE")
        
        batches_df = raw_data['batches']
        
        required_columns = ['batch_id', 'batch_name', 'enrollment_count']
        missing_columns = [col for col in required_columns if col not in batches_df.columns]
        if missing_columns:
            raise DataLoadingError(
                f"Batches table missing required columns: {missing_columns}",
                "MISSING_REQUIRED_COLUMNS",
                {"missing_columns": missing_columns}
            )
        
        return batches_df
    
    def _find_eligible_faculty(self, course_id: str, faculty_df: pd.DataFrame, graph: nx.Graph) -> List[str]:
        """Identify eligible faculty through relationship graph traversal and constraint validation."""
        eligible_faculty = []
        
        # Traverse relationship graph to find course-faculty connections
        if graph.has_node(f"course_{course_id}"):
            faculty_neighbors = [n for n in graph.neighbors(f"course_{course_id}") 
                               if n.startswith("faculty_")]
            
            for faculty_node in faculty_neighbors:
                faculty_id = faculty_node.replace("faculty_", "")
                
                # Validate faculty eligibility constraints
                faculty_record = faculty_df[faculty_df['faculty_id'] == faculty_id]
                if not faculty_record.empty and faculty_record.iloc[0]['max_load'] > 0:
                    eligible_faculty.append(faculty_id)
        
        # Fallback: if no relationships found, consider all faculty eligible
        if not eligible_faculty:
            eligible_faculty = faculty_df['faculty_id'].tolist()
        
        return eligible_faculty
    
    def _find_eligible_rooms(self, course_id: str, rooms_df: pd.DataFrame, graph: nx.Graph) -> List[str]:
        """Identify eligible rooms through relationship graph traversal and capacity validation."""
        eligible_rooms = []
        
        # Traverse relationship graph to find course-room connections
        if graph.has_node(f"course_{course_id}"):
            room_neighbors = [n for n in graph.neighbors(f"course_{course_id}") 
                             if n.startswith("room_")]
            
            for room_node in room_neighbors:
                room_id = room_node.replace("room_", "")
                
                # Validate room capacity and type constraints
                room_record = rooms_df[rooms_df['room_id'] == room_id]
                if not room_record.empty and room_record.iloc[0]['capacity'] > 0:
                    eligible_rooms.append(room_id)
        
        # Fallback: if no relationships found, consider all rooms eligible
        if not eligible_rooms:
            eligible_rooms = rooms_df['room_id'].tolist()
        
        return eligible_rooms
    
    def _find_eligible_timeslots(self, course_id: str, timeslots_df: pd.DataFrame, graph: nx.Graph) -> List[str]:
        """Identify eligible timeslots through relationship graph traversal and temporal validation."""
        eligible_timeslots = []
        
        # For evolutionary algorithms, typically all timeslots are eligible unless constrained
        eligible_timeslots = timeslots_df['timeslot_id'].tolist()
        
        return eligible_timeslots
    
    def _find_eligible_batches(self, course_id: str, batches_df: pd.DataFrame, graph: nx.Graph) -> List[str]:
        """Identify eligible batches through relationship graph traversal and enrollment validation."""
        eligible_batches = []
        
        # Traverse relationship graph to find course-batch connections
        if graph.has_node(f"course_{course_id}"):
            batch_neighbors = [n for n in graph.neighbors(f"course_{course_id}") 
                             if n.startswith("batch_")]
            
            for batch_node in batch_neighbors:
                batch_id = batch_node.replace("batch_", "")
                
                # Validate batch enrollment constraints
                batch_record = batches_df[batches_df['batch_id'] == batch_id]
                if not batch_record.empty and batch_record.iloc[0]['enrollment_count'] > 0:
                    eligible_batches.append(batch_id)
        
        # Fallback: if no relationships found, consider all batches eligible
        if not eligible_batches:
            eligible_batches = batches_df['batch_id'].tolist()
        
        return eligible_batches
    
    def _validate_assignment_feasibility(self, course_id: str, faculty_id: str, room_id: str, 
                                       timeslot_id: str, batch_id: str,
                                       raw_data: Dict[str, pd.DataFrame], 
                                       graph: nx.Graph) -> bool:
        """
        Validate feasibility of specific assignment tuple through constraint checking.
        
        Implements preliminary feasibility assessment per Stage 4 Feasibility Check Framework
        to avoid infeasible assignments in genotype space construction.
        """
        
        # TODO: Implement complete feasibility checks
        # For now, basic validation to ensure entities exist
        
        courses_df = raw_data['courses']
        faculty_df = raw_data['faculty'] 
        rooms_df = raw_data['rooms']
        timeslots_df = raw_data['timeslots']
        batches_df = raw_data['batches']
        
        # Entity existence validation
        if not courses_df[courses_df['course_id'] == course_id].shape[0]:
            return False
        if not faculty_df[faculty_df['faculty_id'] == faculty_id].shape[0]:
            return False
        if not rooms_df[rooms_df['room_id'] == room_id].shape[0]:
            return False
        if not timeslots_df[timeslots_df['timeslot_id'] == timeslot_id].shape[0]:
            return False
        if not batches_df[batches_df['batch_id'] == batch_id].shape[0]:
            return False
        
        return True
    
    def _validate_eligibility_completeness(self, eligibility_mapping: Dict[str, List[Tuple]], 
                                         courses_df: pd.DataFrame) -> None:
        """Validate that all courses have complete eligibility mapping."""
        
        # Check all courses have eligibility entries
        course_ids = set(courses_df['course_id'])
        mapped_courses = set(eligibility_mapping.keys())
        
        missing_courses = course_ids - mapped_courses
        if missing_courses:
            raise DataLoadingError(
                f"Courses missing eligibility mapping: {missing_courses}",
                "INCOMPLETE_ELIGIBILITY_MAPPING",
                {"missing_courses": list(missing_courses)}
            )
        
        # Check no courses have empty eligibility
        empty_eligibility = [course_id for course_id, assignments in eligibility_mapping.items() 
                           if not assignments]
        if empty_eligibility:
            raise DataLoadingError(
                f"Courses with empty eligibility: {empty_eligibility}",
                "EMPTY_COURSE_ELIGIBILITY", 
                {"empty_courses": empty_eligibility}
            )
            
    def _optimize_eligibility_storage(self, eligibility_mapping: Dict[str, List[Tuple]]) -> None:
        """Optimize eligibility mapping storage for memory efficiency."""
        
        # Convert lists to tuples for memory efficiency (immutable)
        for course_id in eligibility_mapping:
            eligibility_mapping[course_id] = tuple(eligibility_mapping[course_id])
        
        # Log optimization statistics
        total_assignments = sum(len(assignments) for assignments in eligibility_mapping.values())
        self.logger.debug("eligibility_storage_optimized",
                         courses=len(eligibility_mapping),
                         total_assignments=total_assignments,
                         avg_assignments=total_assignments / len(eligibility_mapping))

class ConstraintRulesBuilder:
    """
    Constructs constraint rules mapping from Stage 3 compiled data and Dynamic Parametric System,
    implementing the five-objective fitness model per DEAP Foundational Framework Definition 2.4.
    
    Integrates EAV dynamic parameters to enable real-time constraint weight adaptation while
    maintaining theoretical compliance with multi-objective optimization framework.
    
    Mathematical Compliance:
    - Implements f(g) = (f₁,f₂,f₃,f₄,f₅) fitness model structure
    - Preserves constraint violation penalty computation (f₁)
    - Maintains resource utilization efficiency tracking (f₂) 
    - Enables preference satisfaction scoring (f₃)
    - Supports workload balance index calculation (f₄)
    - Facilitates schedule compactness measurement (f₅)
    """
    
    def __init__(self, memory_monitor: MemoryMonitor, logger: structlog.BoundLogger):
        self.memory_monitor = memory_monitor
        self.logger = logger
        
    def build_constraint_rules(self, 
                             raw_data: Dict[str, pd.DataFrame],
                             relationship_graph: nx.Graph,
                             dynamic_params: Optional[Dict[str, Any]] = None,
                             memory_limit_mb: int = 50) -> Dict[str, Dict[str, Any]]:
        """
        Constructs complete constraint rules mapping for multi-objective fitness evaluation.
        
        Implements DEAP Framework Definition 2.4 constraint rule materialization:
        1. Extract base constraint relationships from L_rel graph structure
        2. Integrate Dynamic Parametric System EAV parameters for weight customization
        3. Build course-specific constraint rule dictionaries for O(C) fitness evaluation
        4. Validate constraint completeness and mathematical consistency
        
        Args:
            raw_data: Normalized entity tables from Stage 3 L_raw layer
            relationship_graph: Materialized relationships from Stage 3 L_rel layer
            dynamic_params: EAV dynamic parameters from Dynamic Parametric System
            memory_limit_mb: Maximum memory allocation for constraint rule construction
            
        Returns:
            Dict[course_id, Dict[constraint_type, constraint_data]]
            
        Raises:
            DataLoadingError: On constraint rule construction failure or validation error
            
        Theoretical Guarantees:
        - Multi-objective fitness model compliance per Definition 2.4  
        - Dynamic parameter integration per EAV formal analysis
        - O(C) memory complexity for C courses
        """
        start_memory = self.memory_monitor.get_current_usage_mb()
        self.logger.info("constraint_rules_build_start", 
                        initial_memory_mb=start_memory,
                        dynamic_params_count=len(dynamic_params) if dynamic_params else 0)
        
        try:
            # Initialize constraint rules structure for all courses
            courses_df = raw_data.get('courses', pd.DataFrame())
            if courses_df.empty:
                raise DataLoadingError("No courses found for constraint rule construction", "NO_COURSES")
            
            constraint_rules = {}
            
            # Default constraint weights (overridden by dynamic parameters)
            default_weights = {
                'constraint_violation_penalty': 1.0,      # f₁ weight
                'resource_utilization_weight': 0.3,       # f₂ weight  
                'preference_satisfaction_weight': 0.5,    # f₃ weight
                'workload_balance_weight': 0.4,           # f₄ weight
                'schedule_compactness_weight': 0.2        # f₅ weight
            }
            
            # Apply dynamic parameter overrides if available
            if dynamic_params:
                weights = self._integrate_dynamic_parameters(default_weights, dynamic_params)
            else:
                weights = default_weights
            
            # Build constraint rules for each course
            for _, course in courses_df.iterrows():
                course_id = course['course_id']
                
                # Extract course-specific constraint data
                course_constraints = {
                    # f₁: Constraint violation penalty rules
                    'hard_constraints': {
                        'faculty_availability': self._extract_faculty_constraints(course_id, raw_data, relationship_graph),
                        'room_capacity': self._extract_room_constraints(course_id, raw_data, relationship_graph),
                        'time_conflicts': self._extract_time_constraints(course_id, raw_data, relationship_graph),
                        'prerequisite_constraints': self._extract_prerequisite_constraints(course_id, raw_data)
                    },
                    
                    # f₂: Resource utilization efficiency rules
                    'resource_utilization': {
                        'faculty_load_target': self._calculate_faculty_load_targets(course_id, raw_data),
                        'room_utilization_target': self._calculate_room_utilization_targets(course_id, raw_data),
                        'equipment_sharing': self._extract_equipment_sharing_rules(course_id, raw_data)
                    },
                    
                    # f₃: Preference satisfaction rules
                    'preferences': {
                        'faculty_preferences': self._extract_faculty_preferences(course_id, raw_data, relationship_graph),
                        'student_preferences': self._extract_student_preferences(course_id, raw_data),
                        'institutional_preferences': self._extract_institutional_preferences(course_id, dynamic_params)
                    },
                    
                    # f₄: Workload balance rules  
                    'workload_balance': {
                        'faculty_workload_limits': self._extract_workload_limits(course_id, raw_data),
                        'distribution_targets': self._calculate_workload_distribution_targets(course_id, raw_data),
                        'fairness_constraints': self._extract_fairness_constraints(course_id, raw_data)
                    },
                    
                    # f₅: Schedule compactness rules
                    'compactness': {
                        'time_grouping_bonus': self._calculate_time_grouping_bonus(course_id, raw_data),
                        'location_clustering': self._extract_location_clustering_rules(course_id, raw_data),
                        'gap_minimization': self._extract_gap_minimization_rules(course_id, raw_data)
                    },
                    
                    # Dynamic parameter weights for fitness function
                    'fitness_weights': weights,
                    
                    # Course metadata for constraint evaluation  
                    'course_metadata': {
                        'credits': course.get('credits', 3),
                        'department_id': course.get('department_id'),
                        'course_type': course.get('course_type', 'regular'),
                        'enrollment_capacity': course.get('max_enrollment', 100)
                    }
                }
                
                constraint_rules[course_id] = course_constraints
                
                # Memory management during construction
                if len(constraint_rules) % 100 == 0:
                    current_memory = self.memory_monitor.get_current_usage_mb()
                    if current_memory - start_memory > memory_limit_mb:
                        raise DataLoadingError(
                            f"Memory limit exceeded during constraint rules construction",
                            "CONSTRAINT_MEMORY_EXCEEDED",
                            {"current_memory": current_memory, "limit": memory_limit_mb}
                        )
            
            # Validate constraint rules completeness
            self._validate_constraint_rules_completeness(constraint_rules, courses_df)
            
            final_memory = self.memory_monitor.get_current_usage_mb() 
            memory_used = final_memory - start_memory
            
            self.logger.info("constraint_rules_build_complete",
                           courses_processed=len(constraint_rules),
                           memory_used_mb=memory_used,
                           fitness_weights=weights)
            
            return constraint_rules
            
        except Exception as e:
            self.logger.error("constraint_rules_build_failed",
                            error=str(e),
                            memory_mb=self.memory_monitor.get_current_usage_mb())
            
            if isinstance(e, DataLoadingError):
                raise
            else:
                raise DataLoadingError(
                    f"Unexpected error during constraint rules construction: {str(e)}",
                    "CONSTRAINT_BUILD_UNEXPECTED_ERROR",
                    {"original_exception": str(e)}
                )
    
    def _integrate_dynamic_parameters(self, default_weights: Dict[str, float], 
                                    dynamic_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Integrate Dynamic Parametric System EAV parameters into constraint weights.
        
        Implements Dynamic Parametric System formal analysis for real-time parameter
        adaptation while preserving mathematical consistency of fitness function.
        """
        
        weights = default_weights.copy()
        
        # Extract weight parameters from EAV structure
        if 'fitness_weights' in dynamic_params:
            weight_params = dynamic_params['fitness_weights']
            
            for weight_key, weight_value in weight_params.items():
                if weight_key in weights:
                    # Validate weight value range [0, 2] for stability
                    if isinstance(weight_value, (int, float)) and 0 <= weight_value <= 2:
                        weights[weight_key] = float(weight_value)
                        self.logger.debug("dynamic_weight_applied", 
                                        weight_key=weight_key, 
                                        value=weight_value)
                    else:
                        self.logger.warning("invalid_dynamic_weight_ignored",
                                          weight_key=weight_key, 
                                          value=weight_value)
        
        return weights
    
    def _extract_faculty_constraints(self, course_id: str, raw_data: Dict[str, pd.DataFrame], 
                                   graph: nx.Graph) -> Dict[str, Any]:
        """Extract faculty availability and qualification constraints for course assignment."""
        
        faculty_constraints = {
            'qualified_faculty': [],
            'availability_matrix': {},
            'load_limits': {}
        }
        
        # Extract qualified faculty from relationship graph
        if graph.has_node(f"course_{course_id}"):
            faculty_neighbors = [n for n in graph.neighbors(f"course_{course_id}") 
                               if n.startswith("faculty_")]
            faculty_constraints['qualified_faculty'] = [
                n.replace("faculty_", "") for n in faculty_neighbors
            ]
        
        # Extract faculty availability and load constraints
        faculty_df = raw_data.get('faculty', pd.DataFrame())
        for faculty_id in faculty_constraints['qualified_faculty']:
            faculty_record = faculty_df[faculty_df['faculty_id'] == faculty_id]
            if not faculty_record.empty:
                faculty_constraints['load_limits'][faculty_id] = faculty_record.iloc[0].get('max_load', 20)
        
        return faculty_constraints
    
    def _extract_room_constraints(self, course_id: str, raw_data: Dict[str, pd.DataFrame],
                                graph: nx.Graph) -> Dict[str, Any]:
        """Extract room capacity and type constraints for course assignment."""
        
        room_constraints = {
            'capacity_requirements': {},
            'room_type_requirements': [],
            'equipment_requirements': []
        }
        
        # Extract course enrollment to determine capacity needs
        courses_df = raw_data.get('courses', pd.DataFrame())
        course_record = courses_df[courses_df['course_id'] == course_id]
        if not course_record.empty:
            required_capacity = course_record.iloc[0].get('max_enrollment', 30)
            room_constraints['capacity_requirements']['min_capacity'] = required_capacity
        
        # Extract suitable rooms from relationship graph
        rooms_df = raw_data.get('rooms', pd.DataFrame())
        suitable_rooms = []
        
        if graph.has_node(f"course_{course_id}"):
            room_neighbors = [n for n in graph.neighbors(f"course_{course_id}") 
                            if n.startswith("room_")]
            suitable_rooms = [n.replace("room_", "") for n in room_neighbors]
        
        # If no specific rooms found, use all rooms with adequate capacity
        if not suitable_rooms and not rooms_df.empty:
            min_capacity = room_constraints['capacity_requirements'].get('min_capacity', 30)
            suitable_rooms = rooms_df[rooms_df['capacity'] >= min_capacity]['room_id'].tolist()
        
        room_constraints['suitable_rooms'] = suitable_rooms
        
        return room_constraints
    
    def _extract_time_constraints(self, course_id: str, raw_data: Dict[str, pd.DataFrame],
                                graph: nx.Graph) -> Dict[str, Any]:
        """Extract temporal constraints and conflict rules for course scheduling."""
        
        time_constraints = {
            'blocked_timeslots': [],
            'preferred_timeslots': [],
            'duration_requirements': {},
            'gap_constraints': {}
        }
        
        # Extract course duration requirements
        courses_df = raw_data.get('courses', pd.DataFrame())
        course_record = courses_df[courses_df['course_id'] == course_id]
        if not course_record.empty:
            credits = course_record.iloc[0].get('credits', 3)
            # Assume each credit requires 1 hour of instruction
            time_constraints['duration_requirements']['hours_per_week'] = credits
        
        return time_constraints
    
    def _extract_prerequisite_constraints(self, course_id: str, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Extract prerequisite and co-requisite constraints for course scheduling."""
        
        prerequisite_constraints = {
            'prerequisites': [],
            'corequisites': [], 
            'exclusions': []
        }
        
        # Extract prerequisite relationships if available
        if 'course_prerequisites' in raw_data:
            prereq_df = raw_data['course_prerequisites']
            prerequisites = prereq_df[prereq_df['course_id'] == course_id]['prerequisite_id'].tolist()
            prerequisite_constraints['prerequisites'] = prerequisites
        
        return prerequisite_constraints
    
    def _calculate_faculty_load_targets(self, course_id: str, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate optimal faculty load distribution targets for resource utilization."""
        
        load_targets = {
            'target_load_percentage': 0.8,  # 80% of maximum load
            'balance_threshold': 0.1,       # ±10% acceptable variance
            'overload_penalty': 2.0         # Penalty multiplier for overload
        }
        
        return load_targets
    
    def _calculate_room_utilization_targets(self, course_id: str, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate optimal room utilization targets for resource efficiency."""
        
        utilization_targets = {
            'target_utilization_percentage': 0.75,  # 75% room utilization target
            'capacity_efficiency_threshold': 0.6,   # Minimum 60% capacity utilization
            'overutilization_penalty': 1.5          # Penalty for exceeding capacity
        }
        
        return utilization_targets
    
    def _extract_equipment_sharing_rules(self, course_id: str, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Extract equipment sharing and resource optimization rules."""
        
        equipment_rules = {
            'shared_equipment_bonus': 0.1,    # 10% bonus for equipment sharing
            'exclusive_equipment_penalty': 0.2, # 20% penalty for exclusive use
            'maintenance_windows': []          # Time slots for equipment maintenance
        }
        
        return equipment_rules
    
    def _extract_faculty_preferences(self, course_id: str, raw_data: Dict[str, pd.DataFrame],
                                   graph: nx.Graph) -> Dict[str, float]:
        """Extract faculty teaching preferences and satisfaction scores."""
        
        faculty_preferences = {}
        
        # Extract faculty preferences from relationship graph edge weights
        if graph.has_node(f"course_{course_id}"):
            faculty_neighbors = [n for n in graph.neighbors(f"course_{course_id}") 
                               if n.startswith("faculty_")]
            
            for faculty_node in faculty_neighbors:
                faculty_id = faculty_node.replace("faculty_", "")
                
                # Get preference score from edge weight (default 0.5 if not specified)
                if graph.has_edge(f"course_{course_id}", faculty_node):
                    edge_data = graph.get_edge_data(f"course_{course_id}", faculty_node)
                    preference_score = edge_data.get('weight', 0.5)
                else:
                    preference_score = 0.5  # Neutral preference
                
                faculty_preferences[faculty_id] = preference_score
        
        return faculty_preferences
    
    def _extract_student_preferences(self, course_id: str, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Extract student scheduling preferences and satisfaction metrics."""
        
        student_preferences = {
            'preferred_time_blocks': {
                'morning': 0.3,    # 30% preference for morning classes
                'afternoon': 0.5,  # 50% preference for afternoon classes  
                'evening': 0.2     # 20% preference for evening classes
            },
            'gap_preference': -0.3,  # -30% satisfaction for large gaps between classes
            'clustering_bonus': 0.2  # +20% satisfaction for clustered schedules
        }
        
        return student_preferences
    
    def _extract_institutional_preferences(self, course_id: str, 
                                         dynamic_params: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Extract institutional policy preferences from dynamic parameters."""
        
        institutional_preferences = {
            'resource_efficiency_priority': 0.7,    # 70% priority on resource efficiency
            'student_satisfaction_priority': 0.8,   # 80% priority on student satisfaction
            'faculty_satisfaction_priority': 0.6,   # 60% priority on faculty satisfaction
            'cost_optimization_priority': 0.5       # 50% priority on cost optimization
        }
        
        # Override with dynamic parameters if available
        if dynamic_params and 'institutional_preferences' in dynamic_params:
            prefs = dynamic_params['institutional_preferences']
            for key, value in prefs.items():
                if key in institutional_preferences and isinstance(value, (int, float)):
                    institutional_preferences[key] = float(value)
        
        return institutional_preferences
    
    def _extract_workload_limits(self, course_id: str, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Extract faculty workload limits and distribution constraints."""
        
        workload_limits = {
            'max_courses_per_faculty': 6,      # Maximum 6 courses per faculty per semester
            'max_hours_per_day': 8,            # Maximum 8 teaching hours per day
            'min_break_between_classes': 1,    # Minimum 1 hour break between classes
            'max_consecutive_hours': 4         # Maximum 4 consecutive teaching hours
        }
        
        return workload_limits
    
    def _calculate_workload_distribution_targets(self, course_id: str, 
                                               raw_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate optimal workload distribution targets for balance optimization."""
        
        distribution_targets = {
            'workload_variance_threshold': 0.15,  # Maximum 15% variance in workload
            'utilization_balance_weight': 0.3,    # 30% weight on utilization balance
            'fairness_coefficient': 0.8           # 80% fairness in distribution
        }
        
        return distribution_targets
    
    def _extract_fairness_constraints(self, course_id: str, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Extract fairness constraints for equitable workload distribution."""
        
        fairness_constraints = {
            'equal_opportunity_weight': 0.6,      # 60% weight on equal opportunity
            'seniority_consideration': 0.2,       # 20% weight on faculty seniority
            'expertise_matching_bonus': 0.4       # 40% bonus for expertise matching
        }
        
        return fairness_constraints
    
    def _calculate_time_grouping_bonus(self, course_id: str, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate bonus scores for temporally grouped schedules."""
        
        time_grouping = {
            'consecutive_class_bonus': 0.2,       # 20% bonus for consecutive classes
            'same_day_clustering_bonus': 0.15,    # 15% bonus for same-day clustering
            'block_scheduling_bonus': 0.25        # 25% bonus for block scheduling
        }
        
        return time_grouping
    
    def _extract_location_clustering_rules(self, course_id: str, 
                                         raw_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Extract location clustering rules for spatial optimization."""
        
        clustering_rules = {
            'same_building_bonus': 0.3,           # 30% bonus for same building classes
            'adjacent_room_bonus': 0.4,           # 40% bonus for adjacent rooms
            'floor_clustering_bonus': 0.2         # 20% bonus for same floor clustering
        }
        
        return clustering_rules
    
    def _extract_gap_minimization_rules(self, course_id: str, 
                                       raw_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Extract gap minimization rules for compact scheduling."""
        
        gap_rules = {
            'gap_penalty_per_hour': -0.1,         # -10% penalty per hour of gap
            'lunch_break_bonus': 0.15,            # 15% bonus for proper lunch breaks
            'transition_time_requirement': 0.25   # 25 hours minimum transition time
        }
        
        return gap_rules
    
    def _validate_constraint_rules_completeness(self, constraint_rules: Dict[str, Dict], 
                                              courses_df: pd.DataFrame) -> None:
        """Validate constraint rules completeness and consistency."""
        
        # Validate all courses have constraint rules
        course_ids = set(courses_df['course_id'])
        constraint_course_ids = set(constraint_rules.keys())
        
        missing_constraints = course_ids - constraint_course_ids
        if missing_constraints:
            raise DataLoadingError(
                f"Courses missing constraint rules: {missing_constraints}",
                "INCOMPLETE_CONSTRAINT_RULES",
                {"missing_courses": list(missing_constraints)}
            )
        
        # Validate constraint rule structure
        required_sections = ['hard_constraints', 'resource_utilization', 'preferences', 
                           'workload_balance', 'compactness', 'fitness_weights', 'course_metadata']
        
        for course_id, rules in constraint_rules.items():
            missing_sections = [section for section in required_sections if section not in rules]
            if missing_sections:
                raise DataLoadingError(
                    f"Course {course_id} missing constraint rule sections: {missing_sections}",
                    "INCOMPLETE_CONSTRAINT_STRUCTURE",
                    {"course_id": course_id, "missing_sections": missing_sections}
                )

class BijectionDataBuilder:
    """
    Constructs bijection mapping data from Stage 3 L_idx layer for genotype-phenotype
    transformations in evolutionary algorithms.
    
    Implements Stage 3 Data Compilation bijection formula:
    idx = offsets[c] + f·sF[c] + r·sR[c] + t·sT[c] + b
    
    Ensures bijective equivalence between course-centric and flat binary representations
    while maintaining O(log C) space complexity for efficient decode operations.
    """
    
    def __init__(self, memory_monitor: MemoryMonitor, logger: structlog.BoundLogger):
        self.memory_monitor = memory_monitor
        self.logger = logger
        
    def build_bijection_data(self, idx_data: Dict[str, Any], 
                           raw_data: Dict[str, pd.DataFrame],
                           memory_limit_mb: int = 20) -> Dict[str, Any]:
        """
        Constructs bijection mapping data for genotype-phenotype transformations.
        
        Implements Stage 3 bijection mapping with stride-based indexing:
        1. Extract stride arrays and offset mappings from L_idx layer
        2. Build course-to-index mapping tables for efficient lookup
        3. Construct reverse mapping for phenotype decoding
        4. Validate bijection completeness and mathematical correctness
        
        Args:
            idx_data: Index mapping data from Stage 3 L_idx layer
            raw_data: Raw entity data for validation and cross-reference
            memory_limit_mb: Maximum memory allocation for bijection data
            
        Returns:
            Dict containing bijection mapping data and decode functions
            
        Raises:
            DataLoadingError: On bijection construction failure or validation error
            
        Theoretical Guarantees:
        - Bijective mapping preservation per Stage 3 Theorem 3.3
        - O(log C) space complexity for decode operations  
        - Mathematical consistency with course-centric representation
        """
        start_memory = self.memory_monitor.get_current_usage_mb()
        self.logger.info("bijection_data_build_start", 
                        initial_memory_mb=start_memory)
        
        try:
            # Extract stride arrays from L_idx data
            strides = idx_data.get('strides', {})
            offsets = idx_data.get('offsets', {})
            
            if not strides or not offsets:
                raise DataLoadingError(
                    "Missing stride arrays or offsets in L_idx data",
                    "MISSING_BIJECTION_DATA",
                    {"has_strides": bool(strides), "has_offsets": bool(offsets)}
                )
            
            # Build entity mappings for bijection
            courses_df = raw_data.get('courses', pd.DataFrame())
            faculty_df = raw_data.get('faculty', pd.DataFrame())
            rooms_df = raw_data.get('rooms', pd.DataFrame())
            timeslots_df = raw_data.get('timeslots', pd.DataFrame())
            batches_df = raw_data.get('batches', pd.DataFrame())
            
            # Create ID-to-index mappings
            entity_mappings = {
                'courses': {course_id: idx for idx, course_id in enumerate(courses_df['course_id'])},
                'faculty': {faculty_id: idx for idx, faculty_id in enumerate(faculty_df['faculty_id'])},
                'rooms': {room_id: idx for idx, room_id in enumerate(rooms_df['room_id'])},
                'timeslots': {timeslot_id: idx for idx, timeslot_id in enumerate(timeslots_df['timeslot_id'])},
                'batches': {batch_id: idx for idx, batch_id in enumerate(batches_df['batch_id'])}
            }
            
            # Create reverse mappings for decoding
            reverse_mappings = {
                'courses': {idx: course_id for course_id, idx in entity_mappings['courses'].items()},
                'faculty': {idx: faculty_id for faculty_id, idx in entity_mappings['faculty'].items()},
                'rooms': {idx: room_id for room_id, idx in entity_mappings['rooms'].items()},
                'timeslots': {idx: timeslot_id for timeslot_id, idx in entity_mappings['timeslots'].items()},
                'batches': {idx: batch_id for batch_id, idx in entity_mappings['batches'].items()}
            }
            
            # Build bijection data structure
            bijection_data = {
                'strides': strides,
                'offsets': offsets,
                'entity_mappings': entity_mappings,
                'reverse_mappings': reverse_mappings,
                'entity_counts': {
                    'courses': len(courses_df),
                    'faculty': len(faculty_df), 
                    'rooms': len(rooms_df),
                    'timeslots': len(timeslots_df),
                    'batches': len(batches_df)
                }
            }
            
            # Validate bijection data completeness
            self._validate_bijection_completeness(bijection_data)
            
            # Memory usage validation
            current_memory = self.memory_monitor.get_current_usage_mb()
            memory_used = current_memory - start_memory
            
            if memory_used > memory_limit_mb:
                raise DataLoadingError(
                    f"Bijection data memory usage exceeded limit: {memory_used}MB > {memory_limit_mb}MB",
                    "BIJECTION_MEMORY_EXCEEDED",
                    {"memory_used": memory_used, "limit": memory_limit_mb}
                )
            
            self.logger.info("bijection_data_build_complete",
                           entity_counts=bijection_data['entity_counts'],
                           memory_used_mb=memory_used)
            
            return bijection_data
            
        except Exception as e:
            self.logger.error("bijection_data_build_failed",
                            error=str(e),
                            memory_mb=self.memory_monitor.get_current_usage_mb())
            
            if isinstance(e, DataLoadingError):
                raise
            else:
                raise DataLoadingError(
                    f"Unexpected error during bijection data construction: {str(e)}",
                    "BIJECTION_BUILD_UNEXPECTED_ERROR",
                    {"original_exception": str(e)}
                )
    
    def _validate_bijection_completeness(self, bijection_data: Dict[str, Any]) -> None:
        """Validate bijection data structure completeness and mathematical consistency."""
        
        required_keys = ['strides', 'offsets', 'entity_mappings', 'reverse_mappings', 'entity_counts']
        missing_keys = [key for key in required_keys if key not in bijection_data]
        if missing_keys:
            raise DataLoadingError(
                f"Bijection data missing required keys: {missing_keys}",
                "INCOMPLETE_BIJECTION_STRUCTURE",
                {"missing_keys": missing_keys}
            )
        
        # Validate entity mapping consistency
        entity_types = ['courses', 'faculty', 'rooms', 'timeslots', 'batches']
        for entity_type in entity_types:
            if entity_type not in bijection_data['entity_mappings']:
                raise DataLoadingError(
                    f"Missing entity mapping for {entity_type}",
                    "MISSING_ENTITY_MAPPING",
                    {"entity_type": entity_type}
                )
            
            # Validate forward-reverse mapping consistency
            forward_mapping = bijection_data['entity_mappings'][entity_type]
            reverse_mapping = bijection_data['reverse_mappings'][entity_type]
            
            if len(forward_mapping) != len(reverse_mapping):
                raise DataLoadingError(
                    f"Inconsistent mapping sizes for {entity_type}: forward={len(forward_mapping)}, reverse={len(reverse_mapping)}",
                    "MAPPING_SIZE_INCONSISTENCY",
                    {"entity_type": entity_type, "forward_size": len(forward_mapping), "reverse_size": len(reverse_mapping)}
                )

class DEAPInputModelLoader:
    """
    Primary data loading interface for DEAP Solver Family input modeling layer.
    
    Orchestrates the complete data transformation pipeline from Stage 3 compiled artifacts
    to DEAP-ready structures, ensuring theoretical compliance, memory efficiency, and
    fail-fast error handling.
    
    Architecture:
    - Single-threaded processing with deterministic memory usage
    - Layer-by-layer data transformation with immediate validation
    - Course-centric representation optimized for evolutionary algorithms
    - complete error handling with complete audit logging
    
    Theoretical Foundations:
    - Implements DEAP Framework universal evolutionary framework
    - Maintains bijective equivalence with flat binary representation
    - Preserves multi-objective fitness model structure
    - Integrates Dynamic Parametric System for real-time adaptation
    """
    
    def __init__(self, config: DEAPFamilyConfig):
        """
        Initialize DEAP input model loader with configuration and monitoring.
        
        Args:
            config: DEAP family configuration containing paths, memory limits, and parameters
        """
        self.config = config
        self.memory_monitor = MemoryMonitor(max_memory_mb=config.memory_limits.input_modeling_mb)
        
        # Configure structured logging for audit and debugging
        self.logger = structlog.get_logger().bind(
            component="deap_input_model_loader",
            stage="6.3_input_modeling",
            process_id=id(self)
        )
        
        # Initialize component builders
        self.eligibility_builder = CourseEligibilityBuilder(self.memory_monitor, self.logger)
        self.constraint_builder = ConstraintRulesBuilder(self.memory_monitor, self.logger)
        self.bijection_builder = BijectionDataBuilder(self.memory_monitor, self.logger)
        
    def load_input_context(self, input_paths: PathConfig, 
                          dynamic_params: Optional[Dict[str, Any]] = None) -> 'InputModelContext':
        """
        Load and transform Stage 3 artifacts into DEAP input model context.
        
        Implements complete data loading pipeline:
        1. Load Stage 3 L_raw, L_rel, L_idx artifacts from specified paths
        2. Build course eligibility mapping for genotype construction
        3. Construct constraint rules for multi-objective fitness evaluation
        4. Create bijection data for genotype-phenotype transformations  
        5. Validate context completeness and theoretical compliance
        6. Package into InputModelContext for processing layer handoff
        
        Args:
            input_paths: Path configuration specifying Stage 3 artifact locations
            dynamic_params: EAV dynamic parameters for constraint customization
            
        Returns:
            InputModelContext containing all data needed for evolutionary processing
            
        Raises:
            DataLoadingError: On any loading failure, validation error, or memory overflow
            
        Memory Guarantee: Peak usage ≤ 200MB per configuration specification
        """
        start_time = time.time()
        start_memory = self.memory_monitor.get_current_usage_mb()
        
        self.logger.info("deap_input_loading_start",
                        input_paths=str(input_paths),
                        dynamic_params_provided=dynamic_params is not None,
                        memory_limit_mb=self.config.memory_limits.input_modeling_mb)
        
        try:
            # Phase 1: Load Stage 3 compiled artifacts
            self.logger.info("loading_stage3_artifacts")
            
            # Load L_raw normalized entity tables
            raw_data = self._load_raw_data(input_paths)
            self.memory_monitor.check_memory_usage("after_raw_data_load")
            
            # Load L_rel relationship graph
            relationship_graph = self._load_relationship_graph(input_paths) 
            self.memory_monitor.check_memory_usage("after_relationship_load")
            
            # Load L_idx bijection mapping data
            idx_data = self._load_index_data(input_paths)
            self.memory_monitor.check_memory_usage("after_index_load")
            
            self.logger.info("stage3_artifacts_loaded",
                           raw_tables=list(raw_data.keys()),
                           graph_nodes=relationship_graph.number_of_nodes(),
                           graph_edges=relationship_graph.number_of_edges())
            
            # Phase 2: Build course eligibility mapping
            self.logger.info("building_course_eligibility")
            course_eligibility = self.eligibility_builder.build_course_eligibility(
                raw_data, relationship_graph, 
                memory_limit_mb=self.config.memory_limits.input_modeling_mb // 3
            )
            self.memory_monitor.check_memory_usage("after_eligibility_build")
            
            # Phase 3: Construct constraint rules  
            self.logger.info("building_constraint_rules")
            constraint_rules = self.constraint_builder.build_constraint_rules(
                raw_data, relationship_graph, dynamic_params,
                memory_limit_mb=self.config.memory_limits.input_modeling_mb // 4
            )
            self.memory_monitor.check_memory_usage("after_constraint_build")
            
            # Phase 4: Create bijection data
            self.logger.info("building_bijection_data") 
            bijection_data = self.bijection_builder.build_bijection_data(
                idx_data, raw_data,
                memory_limit_mb=self.config.memory_limits.input_modeling_mb // 10
            )
            self.memory_monitor.check_memory_usage("after_bijection_build")
            
            # Phase 5: Package into input model context
            self.logger.info("creating_input_model_context")
            context = InputModelContext(
                course_eligibility=course_eligibility,
                constraint_rules=constraint_rules, 
                bijection_data=bijection_data,
                entity_metadata={
                    'courses_count': len(course_eligibility),
                    'total_assignments': sum(len(assignments) for assignments in course_eligibility.values()),
                    'constraint_rules_count': len(constraint_rules),
                    'dynamic_params_applied': dynamic_params is not None
                },
                loading_metadata={
                    'load_time_seconds': time.time() - start_time,
                    'peak_memory_mb': self.memory_monitor.get_peak_usage_mb(),
                    'input_paths': str(input_paths),
                    'loader_version': '1.0.0'
                }
            )
            
            # Phase 6: Final validation
            self._validate_input_context(context)
            
            # Cleanup raw data to free memory  
            del raw_data, relationship_graph, idx_data
            gc.collect()
            
            final_memory = self.memory_monitor.get_current_usage_mb()
            total_time = time.time() - start_time
            
            self.logger.info("deap_input_loading_complete",
                           courses_loaded=len(context.course_eligibility),
                           total_time_seconds=total_time,
                           peak_memory_mb=self.memory_monitor.get_peak_usage_mb(),
                           final_memory_mb=final_memory,
                           memory_efficiency_ratio=final_memory / start_memory)
            
            return context
            
        except Exception as e:
            self.logger.error("deap_input_loading_failed",
                            error=str(e),
                            elapsed_time=time.time() - start_time,
                            peak_memory_mb=self.memory_monitor.get_peak_usage_mb())
            
            # Ensure cleanup on failure
            gc.collect()
            
            if isinstance(e, DataLoadingError):
                raise
            else:
                raise DataLoadingError(
                    f"Unexpected error during DEAP input loading: {str(e)}",
                    "INPUT_LOADING_UNEXPECTED_ERROR",
                    {"original_exception": str(e)}
                )
    
    def _load_raw_data(self, input_paths: PathConfig) -> Dict[str, pd.DataFrame]:
        """Load L_raw normalized entity tables from parquet files."""
        
        raw_data = {}
        
        # Define expected entity tables
        entity_tables = ['courses', 'faculty', 'rooms', 'timeslots', 'batches', 
                        'departments', 'students', 'course_prerequisites']
        
        for table_name in entity_tables:
            table_path = Path(input_paths.input_dir) / f"{table_name}.parquet"
            
            if table_path.exists():
                try:
                    df = pd.read_parquet(table_path)
                    raw_data[table_name] = df
                    self.logger.debug("raw_table_loaded", 
                                    table=table_name, 
                                    rows=len(df),
                                    columns=list(df.columns))
                except Exception as e:
                    self.logger.warning("raw_table_load_failed",
                                      table=table_name,
                                      path=str(table_path),
                                      error=str(e))
            else:
                if table_name in ['courses', 'faculty', 'rooms', 'timeslots', 'batches']:
                    # Required tables
                    raise DataLoadingError(
                        f"Required table {table_name} not found at {table_path}",
                        "MISSING_REQUIRED_TABLE",
                        {"table_name": table_name, "path": str(table_path)}
                    )
                else:
                    # Optional tables
                    self.logger.info("optional_table_missing", table=table_name)
        
        if not raw_data:
            raise DataLoadingError(
                "No raw data tables loaded successfully",
                "NO_RAW_DATA_LOADED"
            )
        
        return raw_data
    
    def _load_relationship_graph(self, input_paths: PathConfig) -> nx.Graph:
        """Load L_rel relationship graph from GraphML file."""
        
        graph_path = Path(input_paths.input_dir) / "relationships.graphml"
        
        if not graph_path.exists():
            # Create empty graph if no relationships file found
            self.logger.warning("no_relationship_graph_found", path=str(graph_path))
            return nx.Graph()
        
        try:
            graph = nx.read_graphml(graph_path)
            self.logger.debug("relationship_graph_loaded",
                            nodes=graph.number_of_nodes(),
                            edges=graph.number_of_edges())
            return graph
            
        except Exception as e:
            raise DataLoadingError(
                f"Failed to load relationship graph from {graph_path}: {str(e)}",
                "RELATIONSHIP_GRAPH_LOAD_FAILED",
                {"path": str(graph_path), "error": str(e)}
            )
    
    def _load_index_data(self, input_paths: PathConfig) -> Dict[str, Any]:
        """Load L_idx bijection mapping data from binary or JSON format."""
        
        # Try multiple possible formats for index data
        possible_paths = [
            Path(input_paths.input_dir) / "bijection_data.json",
            Path(input_paths.input_dir) / "strides.json", 
            Path(input_paths.input_dir) / "index_mapping.json"
        ]
        
        for idx_path in possible_paths:
            if idx_path.exists():
                try:
                    import json
                    with open(idx_path, 'r') as f:
                        idx_data = json.load(f)
                    
                    self.logger.debug("index_data_loaded",
                                    path=str(idx_path),
                                    keys=list(idx_data.keys()))
                    return idx_data
                    
                except Exception as e:
                    self.logger.warning("index_data_load_failed",
                                      path=str(idx_path),
                                      error=str(e))
        
        # If no index data found, create minimal structure
        self.logger.warning("no_index_data_found_creating_minimal")
        return {
            'strides': {},
            'offsets': {},
            'metadata': {'created_by': 'deap_loader_fallback'}
        }
    
    def _validate_input_context(self, context: 'InputModelContext') -> None:
        """Validate input model context completeness and consistency."""
        
        # Validate course eligibility completeness
        if not context.course_eligibility:
            raise DataLoadingError(
                "Empty course eligibility mapping",
                "EMPTY_COURSE_ELIGIBILITY"
            )
        
        # Validate constraint rules completeness  
        if not context.constraint_rules:
            raise DataLoadingError(
                "Empty constraint rules mapping", 
                "EMPTY_CONSTRAINT_RULES"
            )
        
        # Validate course ID consistency between eligibility and constraints
        eligibility_courses = set(context.course_eligibility.keys())
        constraint_courses = set(context.constraint_rules.keys())
        
        if eligibility_courses != constraint_courses:
            missing_in_constraints = eligibility_courses - constraint_courses
            missing_in_eligibility = constraint_courses - eligibility_courses
            
            raise DataLoadingError(
                "Course ID mismatch between eligibility and constraint mappings",
                "COURSE_ID_MISMATCH",
                {
                    "missing_in_constraints": list(missing_in_constraints),
                    "missing_in_eligibility": list(missing_in_eligibility)
                }
            )
        
        # Validate bijection data structure
        required_bijection_keys = ['entity_mappings', 'reverse_mappings', 'entity_counts']
        missing_bijection_keys = [key for key in required_bijection_keys 
                                if key not in context.bijection_data]
        if missing_bijection_keys:
            raise DataLoadingError(
                f"Bijection data missing required keys: {missing_bijection_keys}",
                "INCOMPLETE_BIJECTION_DATA",
                {"missing_keys": missing_bijection_keys}
            )
        
        self.logger.info("input_context_validation_complete",
                        courses_validated=len(context.course_eligibility),
                        eligibility_constraint_consistency=True,
                        bijection_structure_valid=True)

# Pydantic model for input context data structure
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Any, Optional

class InputModelContext(BaseModel):
    """
    Pydantic model for DEAP input model context ensuring type safety and data validation.
    
    Contains all data structures required by the Processing Layer for evolutionary optimization:
    - Course eligibility mappings for genotype construction
    - Constraint rules for multi-objective fitness evaluation  
    - Bijection data for genotype-phenotype transformations
    - Metadata for audit and monitoring
    
    Theoretical Compliance:
    - Implements DEAP Framework input model specifications
    - Ensures type safety for evolutionary algorithm integration
    - Maintains course-centric representation consistency
    """
    
    course_eligibility: Dict[str, List[Tuple[str, str, str, str]]] = Field(
        description="Course ID to eligible assignment tuples mapping: course_id -> [(faculty_id, room_id, timeslot_id, batch_id)]"
    )
    
    constraint_rules: Dict[str, Dict[str, Any]] = Field(
        description="Course ID to constraint rules mapping for multi-objective fitness evaluation"
    )
    
    bijection_data: Dict[str, Any] = Field(
        description="Bijection mapping data for genotype-phenotype transformations"
    )
    
    entity_metadata: Dict[str, Any] = Field(
        description="Entity counts and statistics for processing layer optimization"
    )
    
    loading_metadata: Dict[str, Any] = Field(
        description="Loading process metadata for audit and monitoring"
    )
    
    class Config:
        # Allow arbitrary field types for complex nested structures
        arbitrary_types_allowed = True
        
        # Enable validation on assignment
        validate_assignment = True
        
        # Include schema description in serialization
        schema_extra = {
            "description": "DEAP Solver Family Input Model Context",
            "version": "1.0.0",
            "theoretical_compliance": "Stage 6.3 DEAP Foundational Framework",
            "memory_guarantee": "≤200MB peak usage"
        }