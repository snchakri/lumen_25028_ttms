"""
Variable Creation Module

Implements Definition 2.3: Variable Domain Specification
from Stage-6.2 OR-Tools Foundational Framework.

Creates all decision variables for the scheduling CSP:
1. Binary Assignment Variables: X_assignment(c, f, r, t, b) ∈ {0, 1}
2. Time Selection Variables: X_time(c, b) ∈ {1, 2, ..., T}
3. Preference Variables: X_preference(f, c) ∈ [0, 10] ∩ ℤ

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from input_model.loader import CompiledData
from input_model.bijection import BijectiveMapper
from config import SolverParameters, SolverType


@dataclass
class VariableSet:
    """
    Complete set of decision variables for scheduling CSP.
    
    Per Definition 2.3:
    - X_assignment: Binary assignment variables
    - X_time: Time selection variables
    - X_preference: Preference satisfaction variables
    """
    # Binary assignment variables: X_assignment(c, f, r, t, b) ∈ {0, 1}
    assignment_vars: Dict[Tuple[str, str, str, str, str], Any] = field(default_factory=dict)
    
    # Time selection variables: X_time(c, b) ∈ {1, 2, ..., T}
    time_vars: Dict[Tuple[str, str], Any] = field(default_factory=dict)
    
    # Preference variables: X_preference(f, c) ∈ [0, 10] ∩ ℤ
    preference_vars: Dict[Tuple[str, str], Any] = field(default_factory=dict)
    
    # Auxiliary variables for complex constraints
    auxiliary_vars: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    n_assignment_vars: int = 0
    n_time_vars: int = 0
    n_preference_vars: int = 0
    n_auxiliary_vars: int = 0


class VariableCreator:
    """
    Create variables per Definition 2.3.
    
    Variable Types:
    1. Binary Assignment Variables: X_assignment(c, f, r, t, b) ∈ {0, 1}
    2. Time Selection Variables: X_time(c, b) ∈ {1, 2, ..., T}
    3. Preference Variables: X_preference(f, c) ∈ [0, 10] ∩ ℤ
    """
    
    def __init__(self, config: SolverParameters, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def create_variables_cpsat(
        self,
        model: cp_model.CpModel,
        compiled_data: CompiledData,
        bijective_mapper: BijectiveMapper
    ) -> VariableSet:
        """
        Create all variables for CP-SAT solver.
        
        Args:
            model: CP-SAT model
            compiled_data: Stage 3 compiled data
            bijective_mapper: Bijective mappings
            
        Returns:
            VariableSet with all decision variables
        """
        self.logger.info("Creating CP-SAT variables")
        
        variable_set = VariableSet()
        
        # 1. Create Binary Assignment Variables: X_assignment(c, f, r, t, b) ∈ {0, 1}
        self.logger.info("Creating binary assignment variables")
        variable_set.assignment_vars = self._create_assignment_vars_cpsat(
            model, compiled_data, bijective_mapper
        )
        variable_set.n_assignment_vars = len(variable_set.assignment_vars)
        self.logger.info(f"Created {variable_set.n_assignment_vars} assignment variables")
        
        # 2. Create Time Selection Variables: X_time(c, b) ∈ {1, 2, ..., T}
        self.logger.info("Creating time selection variables")
        variable_set.time_vars = self._create_time_vars_cpsat(
            model, compiled_data, bijective_mapper
        )
        variable_set.n_time_vars = len(variable_set.time_vars)
        self.logger.info(f"Created {variable_set.n_time_vars} time selection variables")
        
        # 3. Create Preference Variables: X_preference(f, c) ∈ [0, 10] ∩ ℤ
        self.logger.info("Creating preference variables")
        variable_set.preference_vars = self._create_preference_vars_cpsat(
            model, compiled_data, bijective_mapper
        )
        variable_set.n_preference_vars = len(variable_set.preference_vars)
        self.logger.info(f"Created {variable_set.n_preference_vars} preference variables")
        
        # 4. Create Auxiliary Variables for complex constraints
        self.logger.info("Creating auxiliary variables")
        variable_set.auxiliary_vars = self._create_auxiliary_vars_cpsat(
            model, compiled_data, bijective_mapper
        )
        variable_set.n_auxiliary_vars = len(variable_set.auxiliary_vars)
        self.logger.info(f"Created {variable_set.n_auxiliary_vars} auxiliary variables")
        
        total_vars = (
            variable_set.n_assignment_vars +
            variable_set.n_time_vars +
            variable_set.n_preference_vars +
            variable_set.n_auxiliary_vars
        )
        self.logger.info(f"Total variables created: {total_vars}")
        
        return variable_set
    
    def _create_assignment_vars_cpsat(
        self,
        model: cp_model.CpModel,
        compiled_data: CompiledData,
        bijective_mapper: BijectiveMapper
    ) -> Dict[Tuple[str, str, str, str, str], Any]:
        """
        Create binary assignment variables: X_assignment(c, f, r, t, b) ∈ {0, 1}
        
        Each variable represents whether:
        - Course c is assigned to
        - Faculty f in
        - Room r at
        - Timeslot t for
        - Batch b
        """
        assignment_vars = {}
        
        # Get entity lists
        courses = compiled_data.L_raw.get('courses', None)
        faculty = compiled_data.L_raw.get('faculty', None)
        rooms = compiled_data.L_raw.get('rooms', None)
        timeslots = compiled_data.L_raw.get('timeslots', None)
        batches = compiled_data.L_raw.get('student_batches', None)
        
        if courses is None or faculty is None or rooms is None or timeslots is None or batches is None:
            self.logger.warning("Missing entity data for assignment variables")
            return assignment_vars
        
        # Create variables for all valid combinations
        # NOTE: In full implementation, we should filter by feasibility (e.g., faculty competency)
        # For now, create all combinations
        
        course_ids = courses['course_id'].unique() if 'course_id' in courses.columns else []
        faculty_ids = faculty['faculty_id'].unique() if 'faculty_id' in faculty.columns else []
        room_ids = rooms['room_id'].unique() if 'room_id' in rooms.columns else []
        timeslot_ids = timeslots['timeslot_id'].unique() if 'timeslot_id' in timeslots.columns else []
        batch_ids = batches['batch_id'].unique() if 'batch_id' in batches.columns else []
        
        self.logger.debug(f"Creating assignment variables for: {len(course_ids)} courses, "
                         f"{len(faculty_ids)} faculty, {len(room_ids)} rooms, "
                         f"{len(timeslot_ids)} timeslots, {len(batch_ids)} batches")
        
        # Create variables (with feasibility filtering in full implementation)
        for course_id in course_ids:
            for faculty_id in faculty_ids:
                for room_id in room_ids:
                    for timeslot_id in timeslots:
                        for batch_id in batch_ids:
                            var_name = f"x_assign_{course_id}_{faculty_id}_{room_id}_{timeslot_id}_{batch_id}"
                            var = model.NewBoolVar(var_name)
                            assignment_vars[(str(course_id), str(faculty_id), str(room_id), 
                                           str(timeslot_id), str(batch_id))] = var
        
        return assignment_vars
    
    def _create_time_vars_cpsat(
        self,
        model: cp_model.CpModel,
        compiled_data: CompiledData,
        bijective_mapper: BijectiveMapper
    ) -> Dict[Tuple[str, str], Any]:
        """
        Create time selection variables: X_time(c, b) ∈ {1, 2, ..., T}
        
        Each variable represents the timeslot assigned to course c for batch b.
        """
        time_vars = {}
        
        courses = compiled_data.L_raw.get('courses', None)
        batches = compiled_data.L_raw.get('student_batches', None)
        timeslots = compiled_data.L_raw.get('timeslots', None)
        
        if courses is None or batches is None or timeslots is None:
            return time_vars
        
        course_ids = courses['course_id'].unique() if 'course_id' in courses.columns else []
        batch_ids = batches['batch_id'].unique() if 'batch_id' in batches.columns else []
        n_timeslots = len(timeslots)
        
        for course_id in course_ids:
            for batch_id in batch_ids:
                var_name = f"x_time_{course_id}_{batch_id}"
                # Domain: [1, n_timeslots]
                var = model.NewIntVar(1, n_timeslots, var_name)
                time_vars[(str(course_id), str(batch_id))] = var
        
        return time_vars
    
    def _create_preference_vars_cpsat(
        self,
        model: cp_model.CpModel,
        compiled_data: CompiledData,
        bijective_mapper: BijectiveMapper
    ) -> Dict[Tuple[str, str], Any]:
        """
        Create preference variables: X_preference(f, c) ∈ [0, 10] ∩ ℤ
        
        Each variable represents the preference satisfaction level for faculty f teaching course c.
        """
        preference_vars = {}
        
        faculty = compiled_data.L_raw.get('faculty', None)
        courses = compiled_data.L_raw.get('courses', None)
        
        if faculty is None or courses is None:
            return preference_vars
        
        faculty_ids = faculty['faculty_id'].unique() if 'faculty_id' in faculty.columns else []
        course_ids = courses['course_id'].unique() if 'course_id' in courses.columns else []
        
        for faculty_id in faculty_ids:
            for course_id in course_ids:
                var_name = f"x_pref_{faculty_id}_{course_id}"
                # Domain: [0, 10] ∩ ℤ
                var = model.NewIntVar(0, 10, var_name)
                preference_vars[(str(faculty_id), str(course_id))] = var
        
        return preference_vars
    
    def _create_auxiliary_vars_cpsat(
        self,
        model: cp_model.CpModel,
        compiled_data: CompiledData,
        bijective_mapper: BijectiveMapper
    ) -> Dict[str, Any]:
        """
        Create auxiliary variables for complex constraints.
        
        Examples:
        - Faculty workload counters
        - Room utilization indicators
        - Batch schedule density metrics
        """
        auxiliary_vars = {}
        
        faculty = compiled_data.L_raw.get('faculty', None)
        rooms = compiled_data.L_raw.get('rooms', None)
        batches = compiled_data.L_raw.get('student_batches', None)
        
        if faculty is not None:
            faculty_ids = faculty['faculty_id'].unique() if 'faculty_id' in faculty.columns else []
            for faculty_id in faculty_ids:
                # Faculty workload counter
                var_name = f"aux_workload_{faculty_id}"
                var = model.NewIntVar(0, 100, var_name)  # Max 100 courses
                auxiliary_vars[var_name] = var
        
        if rooms is not None:
            room_ids = rooms['room_id'].unique() if 'room_id' in rooms.columns else []
            for room_id in room_ids:
                # Room utilization counter
                var_name = f"aux_room_util_{room_id}"
                var = model.NewIntVar(0, 100, var_name)  # Max 100 timeslots
                auxiliary_vars[var_name] = var
        
        if batches is not None:
            batch_ids = batches['batch_id'].unique() if 'batch_id' in batches.columns else []
            for batch_id in batch_ids:
                # Batch schedule density
                var_name = f"aux_batch_density_{batch_id}"
                var = model.NewIntVar(0, 100, var_name)  # Max 100 courses
                auxiliary_vars[var_name] = var
        
        return auxiliary_vars
    
    def create_variables_linear(
        self,
        solver: pywraplp.Solver,
        compiled_data: CompiledData,
        bijective_mapper: BijectiveMapper
    ) -> VariableSet:
        """
        Create all variables for Linear Solver.
        
        Args:
            solver: Linear solver instance
            compiled_data: Stage 3 compiled data
            bijective_mapper: Bijective mappings
            
        Returns:
            VariableSet with all decision variables
        """
        self.logger.info("Creating Linear Solver variables")
        
        variable_set = VariableSet()
        
        # 1. Create Binary Assignment Variables
        self.logger.info("Creating binary assignment variables")
        variable_set.assignment_vars = self._create_assignment_vars_linear(
            solver, compiled_data, bijective_mapper
        )
        variable_set.n_assignment_vars = len(variable_set.assignment_vars)
        self.logger.info(f"Created {variable_set.n_assignment_vars} assignment variables")
        
        # 2. Create Time Selection Variables (as integer variables)
        self.logger.info("Creating time selection variables")
        variable_set.time_vars = self._create_time_vars_linear(
            solver, compiled_data, bijective_mapper
        )
        variable_set.n_time_vars = len(variable_set.time_vars)
        self.logger.info(f"Created {variable_set.n_time_vars} time selection variables")
        
        # 3. Create Preference Variables
        self.logger.info("Creating preference variables")
        variable_set.preference_vars = self._create_preference_vars_linear(
            solver, compiled_data, bijective_mapper
        )
        variable_set.n_preference_vars = len(variable_set.preference_vars)
        self.logger.info(f"Created {variable_set.n_preference_vars} preference variables")
        
        total_vars = (
            variable_set.n_assignment_vars +
            variable_set.n_time_vars +
            variable_set.n_preference_vars
        )
        self.logger.info(f"Total variables created: {total_vars}")
        
        return variable_set
    
    def _create_assignment_vars_linear(
        self,
        solver: pywraplp.Solver,
        compiled_data: CompiledData,
        bijective_mapper: BijectiveMapper
    ) -> Dict[Tuple[str, str, str, str, str], Any]:
        """Create binary assignment variables for Linear Solver."""
        assignment_vars = {}
        
        courses = compiled_data.L_raw.get('courses', None)
        faculty = compiled_data.L_raw.get('faculty', None)
        rooms = compiled_data.L_raw.get('rooms', None)
        timeslots = compiled_data.L_raw.get('timeslots', None)
        batches = compiled_data.L_raw.get('student_batches', None)
        
        if courses is None or faculty is None or rooms is None or timeslots is None or batches is None:
            return assignment_vars
        
        course_ids = courses['course_id'].unique() if 'course_id' in courses.columns else []
        faculty_ids = faculty['faculty_id'].unique() if 'faculty_id' in faculty.columns else []
        room_ids = rooms['room_id'].unique() if 'room_id' in rooms.columns else []
        timeslot_ids = timeslots['timeslot_id'].unique() if 'timeslot_id' in timeslots.columns else []
        batch_ids = batches['batch_id'].unique() if 'batch_id' in batches.columns else []
        
        for course_id in course_ids:
            for faculty_id in faculty_ids:
                for room_id in room_ids:
                    for timeslot_id in timeslot_ids:
                        for batch_id in batch_ids:
                            var_name = f"x_assign_{course_id}_{faculty_id}_{room_id}_{timeslot_id}_{batch_id}"
                            var = solver.BoolVar(var_name)
                            assignment_vars[(str(course_id), str(faculty_id), str(room_id), 
                                           str(timeslot_id), str(batch_id))] = var
        
        return assignment_vars
    
    def _create_time_vars_linear(
        self,
        solver: pywraplp.Solver,
        compiled_data: CompiledData,
        bijective_mapper: BijectiveMapper
    ) -> Dict[Tuple[str, str], Any]:
        """Create time selection variables for Linear Solver."""
        time_vars = {}
        
        courses = compiled_data.L_raw.get('courses', None)
        batches = compiled_data.L_raw.get('student_batches', None)
        timeslots = compiled_data.L_raw.get('timeslots', None)
        
        if courses is None or batches is None or timeslots is None:
            return time_vars
        
        course_ids = courses['course_id'].unique() if 'course_id' in courses.columns else []
        batch_ids = batches['batch_id'].unique() if 'batch_id' in batches.columns else []
        n_timeslots = len(timeslots)
        
        for course_id in course_ids:
            for batch_id in batch_ids:
                var_name = f"x_time_{course_id}_{batch_id}"
                var = solver.IntVar(1, n_timeslots, var_name)
                time_vars[(str(course_id), str(batch_id))] = var
        
        return time_vars
    
    def _create_preference_vars_linear(
        self,
        solver: pywraplp.Solver,
        compiled_data: CompiledData,
        bijective_mapper: BijectiveMapper
    ) -> Dict[Tuple[str, str], Any]:
        """Create preference variables for Linear Solver."""
        preference_vars = {}
        
        faculty = compiled_data.L_raw.get('faculty', None)
        courses = compiled_data.L_raw.get('courses', None)
        
        if faculty is None or courses is None:
            return preference_vars
        
        faculty_ids = faculty['faculty_id'].unique() if 'faculty_id' in faculty.columns else []
        course_ids = courses['course_id'].unique() if 'course_id' in courses.columns else []
        
        for faculty_id in faculty_ids:
            for course_id in course_ids:
                var_name = f"x_pref_{faculty_id}_{course_id}"
                var = solver.IntVar(0, 10, var_name)
                preference_vars[(str(faculty_id), str(course_id))] = var
        
        return preference_vars

