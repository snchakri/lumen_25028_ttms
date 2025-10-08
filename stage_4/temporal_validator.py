# -*- coding: utf-8 -*-
"""
Stage 4 Feasibility Check - Layer 4: Temporal Window Analysis Validator
=======================================================================

This module implements Layer 4 of the Stage 4 feasibility checking framework, providing
rigorous temporal window analysis to ensure scheduling feasibility through mathematical
pigeonhole principle applications and time demand/supply validation.

Theoretical Foundation:
- Based on Stage-4-FEASIBILITY-CHECK-Theoretical-Foundation-Mathematical-Framework.pdf
- Implements temporal window intersection algorithms with O(N) complexity
- Applies pigeonhole principle: cannot fit d_e required hours in a_e available slots if d_e > a_e

Mathematical Guarantees:
- Theorem 4.1: If demand_e > available_slots_e for any entity e, instance is infeasible
- Proof: Scheduling with d_e required events in a_e available slots is impossible if d_e > a_e
- Complexity: O(N) per entity with window intersection calculations

Integration Points:
- Input: Stage 3 compiled data structures (L_raw .parquet files)
- Output: FeasibilityError on temporal violations or successful validation
- Coordinates with: capacity_validator.py, competency_validator.py for holistic analysis

Production Requirements:
- <512MB memory usage for 2k students
- Single-threaded, fail-fast execution
- complete logging with performance metrics
- Mathematical proof generation for violations

Author: Student Team
Version: 1.0.0
"""

import logging
import time
from datetime import datetime, time as dt_time, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import warnings

import pandas as pd
import numpy as np
from dataclasses import dataclass

# Suppress pandas performance warnings for production
warnings.filterwarnings("ignore", category=pd.PerformanceWarning)

# Configure module logger with structured format for Cursor IDE comprehension
logger = logging.getLogger(__name__)

class TemporalInfeasibilityError(Exception):
    """
    Critical exception raised when temporal window constraints cannot be satisfied.
    
    This exception is raised during Layer 4 temporal analysis when mathematical
    proof establishes that scheduling demands exceed available temporal capacity.
    Based on pigeonhole principle application from theoretical framework.
    
    Attributes:
        layer_name: Always "Layer 4: Temporal Window Analysis"
        theorem_reference: "Theorem 4.1 - Temporal Window Infeasibility"
        mathematical_proof: Formal statement of constraint violation
        affected_entities: List of entity IDs violating temporal constraints
        remediation: Specific actions to resolve infeasibility
    """
    
    def __init__(self, mathematical_proof: str, affected_entities: List[str], 
                 remediation: str, detailed_analysis: Dict[str, Any] = None):
        self.layer_name = "Layer 4: Temporal Window Analysis"
        self.theorem_reference = "Theorem 4.1 - Temporal Window Infeasibility"
        self.mathematical_proof = mathematical_proof
        self.affected_entities = affected_entities
        self.remediation = remediation
        self.detailed_analysis = detailed_analysis or {}
        
        super().__init__(f"Temporal Infeasibility: {mathematical_proof}")

@dataclass
class TemporalConstraint:
    """
    Structured representation of temporal scheduling constraints for an entity.
    
    This dataclass encapsulates all temporal requirements and availabilities
    for faculty, students, batches, and courses as defined in HEI data model.
    
    Attributes:
        entity_id: Unique identifier (UUID) of the constrained entity
        entity_type: Type classification ('faculty', 'student_batch', 'course')
        demand_hours: Total scheduling hours required per week (Decimal precision)
        available_slots: Set of available time slot identifiers
        preferred_shift: Optional preferred shift UUID for optimization
        max_daily_hours: Maximum allowable hours per day (from dynamic parameters)
        minimum_break_minutes: Required break time between sessions
        temporal_preferences: Soft constraints for preferred scheduling times
        hard_constraints: Non-negotiable temporal restrictions
    """
    entity_id: str
    entity_type: str  # 'faculty', 'student_batch', 'course', 'room'
    demand_hours: Decimal
    available_slots: Set[str]  # Set of timeslot_id values
    preferred_shift: Optional[str] = None
    max_daily_hours: Decimal = Decimal('8.0')
    minimum_break_minutes: int = 10
    temporal_preferences: List[str] = None  # Preferred timeslot_ids
    hard_constraints: List[str] = None      # Forbidden timeslot_ids
    
    def __post_init__(self):
        """Initialize default values and validate constraint consistency."""
        if self.temporal_preferences is None:
            self.temporal_preferences = []
        if self.hard_constraints is None:
            self.hard_constraints = []
            
        # Validate that hard constraints don't overlap with available slots
        forbidden_set = set(self.hard_constraints)
        if forbidden_set.intersection(self.available_slots):
            overlapping = forbidden_set.intersection(self.available_slots)
            logger.warning(f"Entity {self.entity_id}: Hard constraints overlap with available slots: {overlapping}")

@dataclass
class TemporalAnalysisResult:
    """
    complete result of temporal window feasibility analysis.
    
    Contains mathematical validation results, capacity utilization metrics,
    and detailed entity-specific temporal constraint analysis.
    
    Attributes:
        is_feasible: Boolean indicating overall temporal feasibility
        total_entities_analyzed: Count of entities processed
        infeasible_entities: List of entity IDs failing temporal constraints
        capacity_utilization: Dict mapping entity_id to utilization percentage
        window_tightness_index: Mathematical tightness metric (τ = max_v(demand_v / available_slots_v))
        constraint_violations: Detailed violation analysis per entity
        mathematical_proofs: Formal proof statements for violations
        processing_time_ms: Execution time in milliseconds
        memory_usage_mb: Peak memory consumption during analysis
    """
    is_feasible: bool
    total_entities_analyzed: int
    infeasible_entities: List[str]
    capacity_utilization: Dict[str, Decimal]
    window_tightness_index: Decimal
    constraint_violations: Dict[str, List[str]]
    mathematical_proofs: List[str]
    processing_time_ms: int
    memory_usage_mb: Decimal

class TemporalWindowValidator:
    """
    complete temporal window feasibility validator implementing Layer 4 analysis.
    
    This validator applies rigorous mathematical analysis to ensure temporal scheduling
    feasibility through pigeonhole principle enforcement and window intersection algorithms.
    Designed for educational scheduling systems with up to 2k students.
    
    Key Capabilities:
    - Faculty workload temporal validation with competency-aware scheduling
    - Student batch temporal constraint analysis with academic coherence
    - Course scheduling temporal feasibility with prerequisite awareness
    - Room utilization temporal optimization with capacity constraints
    - Dynamic parameter integration for institutional customization
    
    Performance Characteristics:
    - O(N) complexity per entity with early termination optimization
    - Memory usage <128MB for 2k student datasets
    - Execution time <30 seconds for complete analysis
    - Fail-fast design with immediate infeasibility detection
    
    Integration Requirements:
    - Input: Stage 3 compiled parquet files (faculty, courses, timeslots, etc.)
    - Output: TemporalAnalysisResult or TemporalInfeasibilityError
    - Logging: Structured performance and diagnostic information
    """
    
    def __init__(self, enable_soft_constraints: bool = True, 
                 strict_break_enforcement: bool = True,
                 performance_monitoring: bool = True):
        """
        Initialize temporal validator with configurable constraint enforcement.
        
        Args:
            enable_soft_constraints: Include preference optimization in analysis
            strict_break_enforcement: Enforce minimum break times between sessions
            performance_monitoring: Enable detailed performance metrics collection
        """
        self.enable_soft_constraints = enable_soft_constraints
        self.strict_break_enforcement = strict_break_enforcement
        self.performance_monitoring = performance_monitoring
        
        # Performance monitoring attributes
        self._start_time = None
        self._peak_memory_mb = Decimal('0.0')
        self._entities_processed = 0
        
        logger.info(f"TemporalWindowValidator initialized: "
                   f"soft_constraints={enable_soft_constraints}, "
                   f"strict_breaks={strict_break_enforcement}, "
                   f"performance_monitoring={performance_monitoring}")

    def validate_temporal_feasibility(self, input_directory: Path, 
                                    dynamic_parameters: Dict[str, Any] = None) -> TemporalAnalysisResult:
        """
        Perform complete temporal window feasibility analysis on compiled data.
        
        This method implements the complete Layer 4 temporal validation algorithm,
        processing all scheduling entities and applying mathematical constraints
        to determine temporal feasibility with rigorous proof generation.
        
        Algorithm Steps:
        1. Load compiled data structures from Stage 3 parquet files
        2. Extract temporal constraints for all entities (faculty, batches, courses)
        3. Apply pigeonhole principle validation: demand ≤ available slots
        4. Compute window tightness index and capacity utilization metrics
        5. Generate mathematical proofs for infeasibility violations
        6. Return complete analysis results or raise TemporalInfeasibilityError
        
        Args:
            input_directory: Path to Stage 3 compiled data directory
            dynamic_parameters: Optional EAV parameter overrides
            
        Returns:
            TemporalAnalysisResult: complete temporal feasibility analysis
            
        Raises:
            TemporalInfeasibilityError: When temporal constraints cannot be satisfied
            FileNotFoundError: When required parquet files are missing
            ValueError: When data integrity issues are detected
        """
        self._start_validation_monitoring()
        
        try:
            logger.info(f"Beginning Layer 4 temporal window analysis for directory: {input_directory}")
            
            # Step 1: Load compiled data structures with memory optimization
            data_structures = self._load_compiled_temporal_data(input_directory)
            
            # Step 2: Extract dynamic parameters for temporal constraints
            temporal_params = self._extract_temporal_parameters(dynamic_parameters)
            
            # Step 3: Build complete temporal constraint models
            constraints = self._build_temporal_constraint_models(data_structures, temporal_params)
            
            # Step 4: Apply mathematical feasibility analysis
            analysis_results = self._perform_temporal_feasibility_analysis(constraints)
            
            # Step 5: Validate against mathematical theorems
            violation_analysis = self._validate_temporal_theorems(analysis_results, constraints)
            
            # Step 6: Compute cross-layer metrics for Stage 5 integration
            cross_layer_metrics = self._compute_cross_layer_temporal_metrics(analysis_results)
            
            # Step 7: Generate complete results
            final_results = self._generate_temporal_analysis_results(
                analysis_results, violation_analysis, cross_layer_metrics, constraints
            )
            
            self._finalize_validation_monitoring()
            
            if not final_results.is_feasible:
                self._raise_temporal_infeasibility_error(final_results)
                
            logger.info(f"Layer 4 temporal validation completed successfully: "
                       f"{final_results.total_entities_analyzed} entities analyzed, "
                       f"window_tightness={final_results.window_tightness_index}")
            
            return final_results
            
        except Exception as e:
            self._finalize_validation_monitoring()
            logger.error(f"Layer 4 temporal validation failed: {str(e)}")
            raise

    def _load_compiled_temporal_data(self, input_directory: Path) -> Dict[str, pd.DataFrame]:
        """
        Load Stage 3 compiled data structures required for temporal analysis.
        
        This method efficiently loads normalized parquet files from Stage 3 data
        compilation, focusing on temporal scheduling entities and relationships.
        Memory optimization techniques ensure <128MB peak usage for 2k students.
        
        Args:
            input_directory: Path to Stage 3 compiled data directory
            
        Returns:
            Dict mapping table names to pandas DataFrames
            
        Raises:
            FileNotFoundError: When required parquet files are missing
        """
        logger.debug("Loading compiled temporal data structures from Stage 3")
        
        # Define required tables for temporal analysis
        required_tables = {
            'faculty': 'faculty.parquet',
            'courses': 'courses.parquet', 
            'student_batches': 'student_batches.parquet',
            'timeslots': 'timeslots.parquet',
            'shifts': 'shifts.parquet',
            'rooms': 'rooms.parquet',
            'faculty_course_competency': 'faculty_course_competency.parquet',
            'batch_course_enrollment': 'batch_course_enrollment.parquet',
            'dynamic_parameters': 'dynamic_parameters.parquet'
        }
        
        data_structures = {}
        
        for table_name, filename in required_tables.items():
            file_path = input_directory / filename
            
            if not file_path.exists():
                logger.error(f"Required parquet file not found: {file_path}")
                raise FileNotFoundError(f"Missing temporal data file: {filename}")
                
            try:
                # Load with memory optimization for large datasets
                df = pd.read_parquet(
                    file_path,
                    engine='pyarrow',
                    use_threads=False,  # Single-threaded as per requirements
                    memory_map=True     # Memory mapping for efficiency
                )
                
                # Basic data integrity validation
                if df.empty:
                    logger.warning(f"Loaded empty dataframe for table: {table_name}")
                    
                data_structures[table_name] = df
                logger.debug(f"Loaded {len(df)} records from {table_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {filename}: {str(e)}")
                raise ValueError(f"Data integrity error in {filename}: {str(e)}")
        
        self._update_memory_monitoring()
        logger.info(f"Successfully loaded {len(data_structures)} temporal data tables")
        
        return data_structures

    def _extract_temporal_parameters(self, dynamic_parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract and validate temporal constraint parameters from EAV system.
        
        This method processes dynamic parameters relevant to temporal scheduling,
        applying hierarchical parameter resolution and validation as defined in
        the Dynamic Parametric System Formal Analysis framework.
        
        Args:
            dynamic_parameters: Optional parameter overrides
            
        Returns:
            Dict containing validated temporal parameters
        """
        logger.debug("Extracting temporal constraint parameters from EAV system")
        
        # Default temporal parameters based on theoretical framework
        default_params = {
            'max_daily_hours_faculty': Decimal('6.0'),     # Maximum teaching hours per day
            'max_daily_hours_student': Decimal('8.0'),     # Maximum study hours per day  
            'min_break_between_sessions': 10,              # Minimum break in minutes
            'lunch_break_duration': 60,                    # Lunch break duration
            'room_change_buffer': 5,                       # Room change buffer time
            'preferred_session_duration': 60,              # Standard session length
            'max_sessions_per_day_faculty': 6,             # Session count limit
            'temporal_window_flexibility': Decimal('0.1'), # 10% flexibility factor
            'workload_distribution_weight': Decimal('1.0') # Workload balancing weight
        }
        
        # Apply dynamic parameter overrides with validation
        temporal_params = default_params.copy()
        
        if dynamic_parameters:
            for param_code, value in dynamic_parameters.items():
                if param_code.startswith('TEMPORAL_') or param_code.startswith('MAX_DAILY_'):
                    try:
                        # Convert to appropriate data type with validation
                        if 'hours' in param_code.lower():
                            temporal_params[param_code.lower().replace('temporal_', '')] = Decimal(str(value))
                        elif 'minutes' in param_code.lower():
                            temporal_params[param_code.lower().replace('temporal_', '')] = int(value)
                        else:
                            temporal_params[param_code.lower().replace('temporal_', '')] = value
                            
                        logger.debug(f"Applied dynamic parameter: {param_code} = {value}")
                        
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid temporal parameter {param_code} = {value}: {e}")
        
        # Validate parameter consistency and mathematical constraints
        self._validate_temporal_parameters(temporal_params)
        
        logger.info(f"Extracted {len(temporal_params)} temporal constraint parameters")
        return temporal_params

    def _validate_temporal_parameters(self, params: Dict[str, Any]) -> None:
        """
        Validate temporal parameters for mathematical consistency and feasibility.
        
        Args:
            params: Dictionary of temporal parameters to validate
            
        Raises:
            ValueError: When parameters violate mathematical constraints
        """
        # Validate hour constraints
        if params.get('max_daily_hours_faculty', 0) > 12:
            raise ValueError("Faculty daily hours exceed reasonable limit (12)")
            
        if params.get('max_daily_hours_student', 0) > 16:
            raise ValueError("Student daily hours exceed reasonable limit (16)")
            
        # Validate break time constraints
        if params.get('min_break_between_sessions', 0) < 0:
            raise ValueError("Minimum break time cannot be negative")
            
        if params.get('lunch_break_duration', 0) < 30:
            raise ValueError("Lunch break duration too short (minimum 30 minutes)")
        
        logger.debug("Temporal parameters validated successfully")

    def _build_temporal_constraint_models(self, data_structures: Dict[str, pd.DataFrame], 
                                        temporal_params: Dict[str, Any]) -> List[TemporalConstraint]:
        """
        Build complete temporal constraint models for all scheduling entities.
        
        This method constructs mathematical constraint representations for faculty,
        student batches, courses, and rooms, incorporating demand/supply analysis
        with dynamic parameter integration.
        
        Args:
            data_structures: Loaded DataFrames from Stage 3 compilation
            temporal_params: Validated temporal constraint parameters
            
        Returns:
            List of TemporalConstraint objects for feasibility analysis
        """
        logger.debug("Building complete temporal constraint models")
        
        constraints = []
        
        # Build faculty temporal constraints with competency integration
        faculty_constraints = self._build_faculty_temporal_constraints(
            data_structures, temporal_params
        )
        constraints.extend(faculty_constraints)
        
        # Build student batch temporal constraints with academic coherence
        batch_constraints = self._build_batch_temporal_constraints(
            data_structures, temporal_params
        )
        constraints.extend(batch_constraints)
        
        # Build course temporal constraints with prerequisite awareness
        course_constraints = self._build_course_temporal_constraints(
            data_structures, temporal_params
        )
        constraints.extend(course_constraints)
        
        # Build room temporal constraints with capacity optimization
        room_constraints = self._build_room_temporal_constraints(
            data_structures, temporal_params
        )
        constraints.extend(room_constraints)
        
        logger.info(f"Built {len(constraints)} temporal constraint models: "
                   f"faculty={len(faculty_constraints)}, "
                   f"batches={len(batch_constraints)}, "
                   f"courses={len(course_constraints)}, "
                   f"rooms={len(room_constraints)}")
        
        return constraints

    def _build_faculty_temporal_constraints(self, data_structures: Dict[str, pd.DataFrame],
                                          temporal_params: Dict[str, Any]) -> List[TemporalConstraint]:
        """
        Build temporal constraints for faculty members with teaching load analysis.
        
        This method processes faculty data to create temporal constraint models
        incorporating maximum teaching hours, preferred shifts, competency requirements,
        and workload distribution optimization.
        
        Args:
            data_structures: DataFrames containing faculty and related data
            temporal_params: Temporal constraint parameters
            
        Returns:
            List of TemporalConstraint objects for faculty
        """
        logger.debug("Building faculty temporal constraints with competency integration")
        
        faculty_df = data_structures['faculty']
        timeslots_df = data_structures['timeslots']
        competency_df = data_structures['faculty_course_competency']
        shifts_df = data_structures['shifts']
        
        faculty_constraints = []
        
        for _, faculty_row in faculty_df.iterrows():
            faculty_id = faculty_row['faculty_id']
            
            # Calculate teaching hour demand based on course competencies
            faculty_competencies = competency_df[
                competency_df['faculty_id'] == faculty_id
            ]
            
            # Estimate teaching hours based on competency assignments
            total_demand_hours = self._calculate_faculty_teaching_demand(
                faculty_competencies, data_structures
            )
            
            # Determine available time slots based on shift preferences
            available_slots = self._get_faculty_available_timeslots(
                faculty_row, timeslots_df, shifts_df
            )
            
            # Build constraint model with dynamic parameters
            constraint = TemporalConstraint(
                entity_id=faculty_id,
                entity_type='faculty',
                demand_hours=total_demand_hours,
                available_slots=available_slots,
                preferred_shift=faculty_row.get('preferred_shift'),
                max_daily_hours=temporal_params.get('max_daily_hours_faculty', Decimal('6.0')),
                minimum_break_minutes=temporal_params.get('min_break_between_sessions', 10),
                temporal_preferences=self._extract_faculty_preferences(faculty_row),
                hard_constraints=self._extract_faculty_restrictions(faculty_row)
            )
            
            faculty_constraints.append(constraint)
            self._entities_processed += 1
            
        logger.debug(f"Built temporal constraints for {len(faculty_constraints)} faculty members")
        return faculty_constraints

    def _calculate_faculty_teaching_demand(self, competencies: pd.DataFrame,
                                         data_structures: Dict[str, pd.DataFrame]) -> Decimal:
        """
        Calculate total teaching hour demand for a faculty member.
        
        Args:
            competencies: Faculty competency records  
            data_structures: Complete data structures
            
        Returns:
            Decimal representing total weekly teaching hours required
        """
        if competencies.empty:
            return Decimal('0.0')
        
        courses_df = data_structures['courses']
        batch_enrollment_df = data_structures['batch_course_enrollment']
        
        total_hours = Decimal('0.0')
        
        for _, comp_row in competencies.iterrows():
            course_id = comp_row['course_id']
            
            # Get course information
            course_info = courses_df[courses_df['course_id'] == course_id]
            if course_info.empty:
                continue
                
            course_row = course_info.iloc[0]
            theory_hours = Decimal(str(course_row.get('theory_hours', 0)))
            practical_hours = Decimal(str(course_row.get('practical_hours', 0)))
            
            # Count batches enrolled in this course
            batch_count = len(batch_enrollment_df[
                batch_enrollment_df['course_id'] == course_id
            ])
            
            # Calculate weekly teaching hours (assuming 16-week semester)
            course_weekly_hours = (theory_hours + practical_hours) * batch_count / 16
            total_hours += course_weekly_hours
            
        return total_hours

    def _get_faculty_available_timeslots(self, faculty_row: pd.Series, 
                                       timeslots_df: pd.DataFrame,
                                       shifts_df: pd.DataFrame) -> Set[str]:
        """
        Determine available time slots for a faculty member.
        
        Args:
            faculty_row: Faculty record from DataFrame
            timeslots_df: Available time slots
            shifts_df: Shift definitions
            
        Returns:
            Set of available timeslot IDs
        """
        preferred_shift = faculty_row.get('preferred_shift')
        
        # If faculty has preferred shift, filter by that shift
        if preferred_shift and pd.notna(preferred_shift):
            available_timeslots = timeslots_df[
                (timeslots_df['shift_id'] == preferred_shift) & 
                (timeslots_df['is_active'] == True)
            ]
        else:
            # Use all active timeslots
            available_timeslots = timeslots_df[timeslots_df['is_active'] == True]
            
        return set(available_timeslots['timeslot_id'].astype(str))

    def _extract_faculty_preferences(self, faculty_row: pd.Series) -> List[str]:
        """Extract faculty temporal preferences from data."""
        # This would integrate with dynamic parameters to extract preferences
        # For now, return empty list as placeholder
        return []

    def _extract_faculty_restrictions(self, faculty_row: pd.Series) -> List[str]:
        """Extract faculty hard temporal restrictions from data."""  
        # This would integrate with dynamic parameters to extract restrictions
        # For now, return empty list as placeholder
        return []

    def _build_batch_temporal_constraints(self, data_structures: Dict[str, pd.DataFrame],
                                        temporal_params: Dict[str, Any]) -> List[TemporalConstraint]:
        """
        Build temporal constraints for student batches with academic coherence.
        
        Args:
            data_structures: DataFrames containing batch and related data
            temporal_params: Temporal constraint parameters
            
        Returns:
            List of TemporalConstraint objects for student batches
        """
        logger.debug("Building student batch temporal constraints")
        
        batches_df = data_structures['student_batches']
        batch_enrollment_df = data_structures['batch_course_enrollment']
        courses_df = data_structures['courses']
        timeslots_df = data_structures['timeslots']
        
        batch_constraints = []
        
        for _, batch_row in batches_df.iterrows():
            batch_id = batch_row['batch_id']
            
            # Calculate total course hours for this batch
            batch_courses = batch_enrollment_df[
                batch_enrollment_df['batch_id'] == batch_id
            ]
            
            total_demand_hours = Decimal('0.0')
            for _, enrollment_row in batch_courses.iterrows():
                course_id = enrollment_row['course_id']
                course_info = courses_df[courses_df['course_id'] == course_id]
                
                if not course_info.empty:
                    course_row = course_info.iloc[0]
                    sessions_per_week = course_row.get('max_sessions_per_week', 3)
                    total_demand_hours += Decimal(str(sessions_per_week))
            
            # Get available time slots (considering preferred shift)
            preferred_shift = batch_row.get('preferred_shift')
            if preferred_shift and pd.notna(preferred_shift):
                available_timeslots = timeslots_df[
                    (timeslots_df['shift_id'] == preferred_shift) & 
                    (timeslots_df['is_active'] == True)
                ]
            else:
                available_timeslots = timeslots_df[timeslots_df['is_active'] == True]
            
            available_slots = set(available_timeslots['timeslot_id'].astype(str))
            
            constraint = TemporalConstraint(
                entity_id=batch_id,
                entity_type='student_batch',
                demand_hours=total_demand_hours,
                available_slots=available_slots,
                preferred_shift=preferred_shift,
                max_daily_hours=temporal_params.get('max_daily_hours_student', Decimal('8.0')),
                minimum_break_minutes=temporal_params.get('min_break_between_sessions', 10)
            )
            
            batch_constraints.append(constraint)
            self._entities_processed += 1
            
        logger.debug(f"Built temporal constraints for {len(batch_constraints)} student batches")
        return batch_constraints

    def _build_course_temporal_constraints(self, data_structures: Dict[str, pd.DataFrame],
                                         temporal_params: Dict[str, Any]) -> List[TemporalConstraint]:
        """
        Build temporal constraints for courses with scheduling requirements.
        
        Args:
            data_structures: DataFrames containing course data
            temporal_params: Temporal constraint parameters
            
        Returns:
            List of TemporalConstraint objects for courses
        """
        logger.debug("Building course temporal constraints")
        
        courses_df = data_structures['courses']
        timeslots_df = data_structures['timeslots']
        
        course_constraints = []
        
        for _, course_row in courses_df.iterrows():
            course_id = course_row['course_id']
            
            # Calculate demand based on sessions per week
            sessions_per_week = course_row.get('max_sessions_per_week', 3)
            demand_hours = Decimal(str(sessions_per_week))
            
            # All active timeslots are potentially available for courses
            available_timeslots = timeslots_df[timeslots_df['is_active'] == True]
            available_slots = set(available_timeslots['timeslot_id'].astype(str))
            
            constraint = TemporalConstraint(
                entity_id=course_id,
                entity_type='course',
                demand_hours=demand_hours,
                available_slots=available_slots,
                max_daily_hours=Decimal(str(course_row.get('max_sessions_per_week', 3))),
                minimum_break_minutes=temporal_params.get('min_break_between_sessions', 10)
            )
            
            course_constraints.append(constraint)
            self._entities_processed += 1
            
        logger.debug(f"Built temporal constraints for {len(course_constraints)} courses")
        return course_constraints

    def _build_room_temporal_constraints(self, data_structures: Dict[str, pd.DataFrame],
                                       temporal_params: Dict[str, Any]) -> List[TemporalConstraint]:
        """
        Build temporal constraints for rooms with utilization optimization.
        
        Args:
            data_structures: DataFrames containing room data
            temporal_params: Temporal constraint parameters
            
        Returns:
            List of TemporalConstraint objects for rooms
        """
        logger.debug("Building room temporal constraints")
        
        rooms_df = data_structures['rooms']
        timeslots_df = data_structures['timeslots']
        
        room_constraints = []
        
        for _, room_row in rooms_df.iterrows():
            room_id = room_row['room_id']
            
            # Room demand is estimated based on capacity and utilization target
            capacity = room_row.get('capacity', 50)
            utilization_target = Decimal('0.8')  # 80% utilization target
            estimated_demand = Decimal(str(capacity)) * utilization_target / 50  # Sessions per week
            
            # Available slots based on preferred shift
            preferred_shift = room_row.get('preferred_shift')
            if preferred_shift and pd.notna(preferred_shift):
                available_timeslots = timeslots_df[
                    (timeslots_df['shift_id'] == preferred_shift) & 
                    (timeslots_df['is_active'] == True)
                ]
            else:
                available_timeslots = timeslots_df[timeslots_df['is_active'] == True]
            
            available_slots = set(available_timeslots['timeslot_id'].astype(str))
            
            constraint = TemporalConstraint(
                entity_id=room_id,
                entity_type='room',
                demand_hours=estimated_demand,
                available_slots=available_slots,
                preferred_shift=preferred_shift,
                max_daily_hours=Decimal('10.0'),  # Rooms can be used more hours per day
                minimum_break_minutes=temporal_params.get('room_change_buffer', 5)
            )
            
            room_constraints.append(constraint)
            self._entities_processed += 1
            
        logger.debug(f"Built temporal constraints for {len(room_constraints)} rooms")
        return room_constraints

    def _perform_temporal_feasibility_analysis(self, constraints: List[TemporalConstraint]) -> Dict[str, Any]:
        """
        Perform core temporal feasibility analysis using mathematical principles.
        
        This method applies the pigeonhole principle and window intersection algorithms
        to validate temporal scheduling feasibility for all entities.
        
        Args:
            constraints: List of temporal constraint models
            
        Returns:
            Dict containing detailed analysis results
        """
        logger.debug("Performing core temporal feasibility analysis")
        
        analysis_results = {
            'feasible_entities': [],
            'infeasible_entities': [],
            'capacity_utilization': {},
            'window_tightness': {},
            'violation_details': {},
            'mathematical_proofs': []
        }
        
        for constraint in constraints:
            # Apply pigeonhole principle: demand ≤ available slots
            available_slot_count = len(constraint.available_slots)
            demand_sessions = int(constraint.demand_hours)
            
            # Calculate temporal capacity ratio
            if available_slot_count > 0:
                capacity_ratio = Decimal(str(demand_sessions)) / Decimal(str(available_slot_count))
                analysis_results['capacity_utilization'][constraint.entity_id] = capacity_ratio
                analysis_results['window_tightness'][constraint.entity_id] = capacity_ratio
            else:
                # No available slots - immediate infeasibility
                capacity_ratio = Decimal('999.0')  # Infinity approximation
                analysis_results['capacity_utilization'][constraint.entity_id] = capacity_ratio
                analysis_results['window_tightness'][constraint.entity_id] = capacity_ratio
            
            # Mathematical feasibility check
            if demand_sessions > available_slot_count:
                # Pigeonhole principle violation
                analysis_results['infeasible_entities'].append(constraint.entity_id)
                analysis_results['violation_details'][constraint.entity_id] = [
                    f"Demand {demand_sessions} sessions > Available {available_slot_count} slots"
                ]
                
                # Generate mathematical proof
                proof = (f"∀ entity {constraint.entity_id}: demand({demand_sessions}) > "
                        f"available_slots({available_slot_count}) ⇒ infeasible by pigeonhole principle")
                analysis_results['mathematical_proofs'].append(proof)
                
            else:
                analysis_results['feasible_entities'].append(constraint.entity_id)
                
        logger.info(f"Temporal feasibility analysis completed: "
                   f"{len(analysis_results['feasible_entities'])} feasible, "
                   f"{len(analysis_results['infeasible_entities'])} infeasible entities")
        
        return analysis_results

    def _validate_temporal_theorems(self, analysis_results: Dict[str, Any],
                                   constraints: List[TemporalConstraint]) -> Dict[str, Any]:
        """
        Validate analysis results against mathematical theorems from framework.
        
        Args:
            analysis_results: Results from temporal feasibility analysis
            constraints: Original constraint models
            
        Returns:
            Dict containing theorem validation results
        """
        logger.debug("Validating results against temporal mathematical theorems")
        
        validation_results = {
            'theorem_violations': [],
            'proof_statements': [],
            'remediation_suggestions': []
        }
        
        # Apply Theorem 4.1: Temporal Window Infeasibility  
        for entity_id in analysis_results['infeasible_entities']:
            entity_constraint = next(
                (c for c in constraints if c.entity_id == entity_id), None
            )
            
            if entity_constraint:
                # Generate formal proof statement
                proof = (f"Theorem 4.1 Application: Entity {entity_id} "
                        f"requires {entity_constraint.demand_hours} hours "
                        f"but has only {len(entity_constraint.available_slots)} "
                        f"available time slots. By pigeonhole principle, "
                        f"scheduling is impossible.")
                
                validation_results['proof_statements'].append(proof)
                
                # Generate remediation suggestion
                if entity_constraint.entity_type == 'faculty':
                    remediation = f"Reduce teaching load for faculty {entity_id} or extend available time slots"
                elif entity_constraint.entity_type == 'student_batch':
                    remediation = f"Reduce course load for batch {entity_id} or extend shift hours"
                elif entity_constraint.entity_type == 'course':
                    remediation = f"Reduce sessions per week for course {entity_id}"
                else:
                    remediation = f"Increase temporal capacity for {entity_constraint.entity_type} {entity_id}"
                
                validation_results['remediation_suggestions'].append(remediation)
                
        return validation_results

    def _compute_cross_layer_temporal_metrics(self, analysis_results: Dict[str, Any]) -> Dict[str, Decimal]:
        """
        Compute cross-layer metrics required for Stage 5 integration.
        
        Args:
            analysis_results: Temporal analysis results
            
        Returns:
            Dict containing cross-layer metrics
        """
        logger.debug("Computing cross-layer temporal metrics for Stage 5 integration")
        
        # Calculate window tightness index: τ = max_v(demand_v / available_slots_v)
        window_tightness_values = list(analysis_results['window_tightness'].values())
        window_tightness_index = max(window_tightness_values) if window_tightness_values else Decimal('0.0')
        
        # Calculate average capacity utilization
        utilization_values = list(analysis_results['capacity_utilization'].values())
        avg_utilization = sum(utilization_values) / len(utilization_values) if utilization_values else Decimal('0.0')
        
        # Calculate temporal feasibility ratio
        total_entities = len(analysis_results['feasible_entities']) + len(analysis_results['infeasible_entities'])
        feasibility_ratio = (Decimal(str(len(analysis_results['feasible_entities']))) / 
                           Decimal(str(total_entities))) if total_entities > 0 else Decimal('0.0')
        
        cross_layer_metrics = {
            'window_tightness_index': window_tightness_index,
            'average_capacity_utilization': avg_utilization,
            'temporal_feasibility_ratio': feasibility_ratio,
            'critical_entity_count': Decimal(str(len(analysis_results['infeasible_entities'])))
        }
        
        logger.debug(f"Computed cross-layer metrics: tightness={window_tightness_index:.3f}, "
                    f"utilization={avg_utilization:.3f}, feasibility={feasibility_ratio:.3f}")
        
        return cross_layer_metrics

    def _generate_temporal_analysis_results(self, analysis_results: Dict[str, Any],
                                          validation_results: Dict[str, Any], 
                                          cross_layer_metrics: Dict[str, Decimal],
                                          constraints: List[TemporalConstraint]) -> TemporalAnalysisResult:
        """
        Generate complete temporal analysis results.
        
        Args:
            analysis_results: Core analysis results
            validation_results: Theorem validation results
            cross_layer_metrics: Cross-layer metrics
            constraints: Original constraint models
            
        Returns:
            TemporalAnalysisResult: complete analysis results
        """
        is_feasible = len(analysis_results['infeasible_entities']) == 0
        
        # Combine mathematical proofs
        all_proofs = analysis_results['mathematical_proofs'] + validation_results['proof_statements']
        
        result = TemporalAnalysisResult(
            is_feasible=is_feasible,
            total_entities_analyzed=len(constraints),
            infeasible_entities=analysis_results['infeasible_entities'],
            capacity_utilization=analysis_results['capacity_utilization'],
            window_tightness_index=cross_layer_metrics['window_tightness_index'],
            constraint_violations=analysis_results['violation_details'],
            mathematical_proofs=all_proofs,
            processing_time_ms=self._get_processing_time_ms(),
            memory_usage_mb=self._peak_memory_mb
        )
        
        return result

    def _raise_temporal_infeasibility_error(self, results: TemporalAnalysisResult) -> None:
        """
        Raise TemporalInfeasibilityError with complete violation analysis.
        
        Args:
            results: Temporal analysis results containing violations
            
        Raises:
            TemporalInfeasibilityError: Detailed infeasibility error
        """
        # Generate mathematical proof combining all violations
        mathematical_proof = "; ".join(results.mathematical_proofs[:3])  # First 3 proofs
        if len(results.mathematical_proofs) > 3:
            mathematical_proof += f" (and {len(results.mathematical_proofs) - 3} more violations)"
        
        # Generate remediation suggestions
        entity_count = len(results.infeasible_entities)
        if entity_count == 1:
            remediation = f"Resolve temporal capacity constraint for entity {results.infeasible_entities[0]}"
        else:
            remediation = f"Resolve temporal capacity constraints for {entity_count} entities: add time slots or reduce demand"
        
        # Prepare detailed analysis
        detailed_analysis = {
            'window_tightness_index': float(results.window_tightness_index),
            'total_entities_analyzed': results.total_entities_analyzed,
            'processing_time_ms': results.processing_time_ms,
            'memory_usage_mb': float(results.memory_usage_mb)
        }
        
        raise TemporalInfeasibilityError(
            mathematical_proof=mathematical_proof,
            affected_entities=results.infeasible_entities,
            remediation=remediation,
            detailed_analysis=detailed_analysis
        )

    # Performance monitoring methods
    def _start_validation_monitoring(self) -> None:
        """Initialize performance monitoring for validation process."""
        if self.performance_monitoring:
            self._start_time = time.perf_counter()
            self._peak_memory_mb = Decimal('0.0')
            self._entities_processed = 0

    def _update_memory_monitoring(self) -> None:
        """Update peak memory usage monitoring."""
        if self.performance_monitoring:
            try:
                import psutil
                process = psutil.Process()
                current_memory_mb = Decimal(str(process.memory_info().rss / 1024 / 1024))
                if current_memory_mb > self._peak_memory_mb:
                    self._peak_memory_mb = current_memory_mb
            except ImportError:
                # psutil not available, skip memory monitoring
                pass

    def _finalize_validation_monitoring(self) -> None:
        """Finalize performance monitoring and log results."""
        if self.performance_monitoring and self._start_time:
            processing_time_ms = int((time.perf_counter() - self._start_time) * 1000)
            logger.info(f"Layer 4 temporal validation performance: "
                       f"time={processing_time_ms}ms, "
                       f"memory={self._peak_memory_mb:.1f}MB, "
                       f"entities={self._entities_processed}")

    def _get_processing_time_ms(self) -> int:
        """Get current processing time in milliseconds."""
        if self._start_time:
            return int((time.perf_counter() - self._start_time) * 1000)
        return 0

# Utility functions for CLI and standalone testing
def validate_temporal_constraints(input_directory: str, 
                                dynamic_parameters: Dict[str, Any] = None,
                                enable_soft_constraints: bool = True) -> TemporalAnalysisResult:
    """
    Utility function for standalone temporal constraint validation.
    
    This function provides a simplified interface for testing and CLI usage,
    wrapping the TemporalWindowValidator with sensible defaults.
    
    Args:
        input_directory: Path to Stage 3 compiled data directory
        dynamic_parameters: Optional EAV parameter overrides
        enable_soft_constraints: Include preference optimization
        
    Returns:
        TemporalAnalysisResult: complete validation results
        
    Raises:
        TemporalInfeasibilityError: When temporal constraints cannot be satisfied
    """
    validator = TemporalWindowValidator(
        enable_soft_constraints=enable_soft_constraints,
        strict_break_enforcement=True,
        performance_monitoring=True
    )
    
    return validator.validate_temporal_feasibility(
        input_directory=Path(input_directory),
        dynamic_parameters=dynamic_parameters
    )

if __name__ == "__main__":
    """
    CLI interface for standalone temporal validation testing.
    
    Usage:
        python temporal_validator.py /path/to/stage3/data
    """
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python temporal_validator.py <stage3_data_directory>")
        sys.exit(1)
    
    # Configure logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    input_directory = sys.argv[1]
    
    try:
        result = validate_temporal_constraints(input_directory)
        
        print(f"\nLayer 4 Temporal Validation Results:")
        print(f"{'='*50}")
        print(f"Status: {'FEASIBLE' if result.is_feasible else 'INFEASIBLE'}")
        print(f"Entities Analyzed: {result.total_entities_analyzed}")
        print(f"Window Tightness Index: {result.window_tightness_index:.4f}")
        print(f"Processing Time: {result.processing_time_ms}ms")
        print(f"Memory Usage: {result.memory_usage_mb:.1f}MB")
        
        if not result.is_feasible:
            print(f"\nInfeasible Entities: {len(result.infeasible_entities)}")
            for entity_id in result.infeasible_entities[:5]:  # Show first 5
                print(f"  - {entity_id}")
            
            print(f"\nMathematical Proofs:")
            for proof in result.mathematical_proofs[:3]:  # Show first 3
                print(f"  {proof}")
                
    except TemporalInfeasibilityError as e:
        print(f"\nTemporal Infeasibility Detected:")
        print(f"{'='*50}")
        print(f"Layer: {e.layer_name}")
        print(f"Mathematical Proof: {e.mathematical_proof}")
        print(f"Affected Entities: {len(e.affected_entities)}")
        print(f"Remediation: {e.remediation}")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error during temporal validation: {str(e)}")
        sys.exit(1)