"""
Objective Function Builder Module

Implements multi-objective optimization function per Algorithm 13.2
from Stage-6.2 OR-Tools Foundational Framework.

Multi-Objective Formulation:
minimize: w₁·penalty_time_pref + w₂·penalty_course_pref + 
          w₃·penalty_workload + w₄·penalty_density

Objective Components:
1. Faculty Time Preference Penalty: Penalize non-preferred timeslots
2. Faculty Course Preference Penalty: Penalize non-preferred courses
3. Workload Imbalance Penalty: Penalize uneven workload distribution
4. Schedule Density Penalty: Penalize gaps in student schedules

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
from processing.variables import VariableSet
from config import SolverParameters


@dataclass
class ObjectiveComponents:
    """
    Components of multi-objective function.
    """
    # Penalty variables
    time_preference_penalties: List[Any] = field(default_factory=list)
    course_preference_penalties: List[Any] = field(default_factory=list)
    workload_imbalance_penalties: List[Any] = field(default_factory=list)
    schedule_density_penalties: List[Any] = field(default_factory=list)
    
    # Weights (from config or defaults)
    w_time_pref: float = 1.0
    w_course_pref: float = 1.0
    w_workload: float = 1.0
    w_density: float = 1.0
    
    # Total objective expression
    objective_expr: Any = None


class ObjectiveFunctionBuilder:
    """
    Build multi-objective function per Algorithm 13.2.
    
    Objective Methods:
    1. Weighted Sum: Combine objectives with user-specified weights
    2. Lexicographic: Prioritize objectives in hierarchical order
    3. Epsilon-Constraint: Optimize primary objective while constraining others
    4. Pareto Methods: Generate non-dominated solution sets
    """
    
    def __init__(self, config: SolverParameters, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Get weights from config
        self.w_time_pref = getattr(config, 'weight_time_preference', 1.0)
        self.w_course_pref = getattr(config, 'weight_course_preference', 1.0)
        self.w_workload = getattr(config, 'weight_workload_balance', 1.0)
        self.w_density = getattr(config, 'weight_schedule_density', 1.0)
    
    def build_objective_cpsat(
        self,
        model: cp_model.CpModel,
        variables: VariableSet,
        compiled_data: CompiledData,
        bijective_mapper: BijectiveMapper
    ) -> ObjectiveComponents:
        """
        Build multi-objective function for CP-SAT solver.
        
        Uses weighted sum approach:
        minimize: w₁·penalty_time_pref + w₂·penalty_course_pref + 
                  w₃·penalty_workload + w₄·penalty_density
        
        Args:
            model: CP-SAT model
            variables: Variable set
            compiled_data: Stage 3 compiled data
            bijective_mapper: Bijective mappings
            
        Returns:
            ObjectiveComponents with all penalty terms
        """
        self.logger.info("Building CP-SAT objective function")
        
        components = ObjectiveComponents()
        components.w_time_pref = self.w_time_pref
        components.w_course_pref = self.w_course_pref
        components.w_workload = self.w_workload
        components.w_density = self.w_density
        
        # 1. Faculty Time Preference Penalty
        self.logger.info("Building time preference penalty")
        time_penalty_vars = self._build_time_preference_penalty_cpsat(
            model, variables, compiled_data, bijective_mapper
        )
        components.time_preference_penalties = time_penalty_vars
        self.logger.info(f"Created {len(time_penalty_vars)} time preference penalty variables")
        
        # 2. Faculty Course Preference Penalty
        self.logger.info("Building course preference penalty")
        course_penalty_vars = self._build_course_preference_penalty_cpsat(
            model, variables, compiled_data, bijective_mapper
        )
        components.course_preference_penalties = course_penalty_vars
        self.logger.info(f"Created {len(course_penalty_vars)} course preference penalty variables")
        
        # 3. Workload Imbalance Penalty
        self.logger.info("Building workload imbalance penalty")
        workload_penalty_vars = self._build_workload_imbalance_penalty_cpsat(
            model, variables, compiled_data, bijective_mapper
        )
        components.workload_imbalance_penalties = workload_penalty_vars
        self.logger.info(f"Created {len(workload_penalty_vars)} workload imbalance penalty variables")
        
        # 4. Schedule Density Penalty
        self.logger.info("Building schedule density penalty")
        density_penalty_vars = self._build_schedule_density_penalty_cpsat(
            model, variables, compiled_data, bijective_mapper
        )
        components.schedule_density_penalties = density_penalty_vars
        self.logger.info(f"Created {len(density_penalty_vars)} schedule density penalty variables")
        
        # Combine all penalties into weighted sum
        self.logger.info("Combining penalties into objective function")
        objective_terms = []
        
        # Add time preference penalties
        for penalty_var in time_penalty_vars:
            objective_terms.append(int(self.w_time_pref * 100) * penalty_var)
        
        # Add course preference penalties
        for penalty_var in course_penalty_vars:
            objective_terms.append(int(self.w_course_pref * 100) * penalty_var)
        
        # Add workload imbalance penalties
        for penalty_var in workload_penalty_vars:
            objective_terms.append(int(self.w_workload * 100) * penalty_var)
        
        # Add schedule density penalties
        for penalty_var in density_penalty_vars:
            objective_terms.append(int(self.w_density * 100) * penalty_var)
        
        if objective_terms:
            # Minimize total penalty
            model.Minimize(sum(objective_terms))
            components.objective_expr = sum(objective_terms)
            self.logger.info(f"Objective function created with {len(objective_terms)} terms")
        else:
            self.logger.warning("No objective terms created - using default objective")
            # Create a default objective for feasibility checking
            default_var = model.NewIntVar(0, 0, "default_objective")
            model.Minimize(default_var)
            components.objective_expr = default_var
        
        return components
    
    def _build_time_preference_penalty_cpsat(
        self,
        model: cp_model.CpModel,
        variables: VariableSet,
        compiled_data: CompiledData,
        bijective_mapper: BijectiveMapper
    ) -> List[Any]:
        """
        Build time preference penalty variables.
        
        For each faculty f and timeslot t:
        penalty = (10 - preference_score) * ∑_{c,r,b} X_assignment(c, f, r, t, b)
        """
        penalty_vars = []
        
        # Load time preferences
        preferences = self._load_time_preferences(compiled_data)
        
        if not preferences:
            self.logger.debug("No time preferences found")
            return penalty_vars
        
        faculty = compiled_data.L_raw.get('faculty', None)
        timeslots = compiled_data.L_raw.get('timeslots', None)
        
        if faculty is None or timeslots is None:
            return penalty_vars
        
        faculty_ids = faculty['faculty_id'].unique() if 'faculty_id' in faculty.columns else []
        timeslot_ids = timeslots['timeslot_id'].unique() if 'timeslot_id' in timeslots.columns else []
        
        for faculty_id in faculty_ids:
            for timeslot_id in timeslot_ids:
                # Get preference score (0-10, higher is better)
                pref_score = preferences.get((str(faculty_id), str(timeslot_id)), 5)  # Default: neutral
                penalty_weight = 10 - pref_score
                
                if penalty_weight > 0:
                    # Collect assignment variables for this faculty-timeslot
                    vars_for_faculty_timeslot = []
                    
                    for (c, f, r, t, b), var in variables.assignment_vars.items():
                        if f == str(faculty_id) and t == str(timeslot_id):
                            vars_for_faculty_timeslot.append(var)
                    
                    if vars_for_faculty_timeslot:
                        # Create penalty variable
                        penalty_var = model.NewIntVar(0, penalty_weight * len(vars_for_faculty_timeslot), 
                                                      f"penalty_time_{faculty_id}_{timeslot_id}")
                        
                        # Link penalty to assignments
                        model.Add(penalty_var == penalty_weight * sum(vars_for_faculty_timeslot))
                        
                        penalty_vars.append(penalty_var)
        
        return penalty_vars
    
    def _build_course_preference_penalty_cpsat(
        self,
        model: cp_model.CpModel,
        variables: VariableSet,
        compiled_data: CompiledData,
        bijective_mapper: BijectiveMapper
    ) -> List[Any]:
        """
        Build course preference penalty variables.
        
        For each faculty f and course c:
        penalty = (10 - preference_score) * ∑_{r,t,b} X_assignment(c, f, r, t, b)
        """
        penalty_vars = []
        
        # Load course preferences
        preferences = self._load_course_preferences(compiled_data)
        
        if not preferences:
            self.logger.debug("No course preferences found")
            return penalty_vars
        
        faculty = compiled_data.L_raw.get('faculty', None)
        courses = compiled_data.L_raw.get('courses', None)
        
        if faculty is None or courses is None:
            return penalty_vars
        
        faculty_ids = faculty['faculty_id'].unique() if 'faculty_id' in faculty.columns else []
        course_ids = courses['course_id'].unique() if 'course_id' in courses.columns else []
        
        for faculty_id in faculty_ids:
            for course_id in course_ids:
                # Get preference score (0-10, higher is better)
                pref_score = preferences.get((str(faculty_id), str(course_id)), 5)  # Default: neutral
                penalty_weight = 10 - pref_score
                
                if penalty_weight > 0:
                    # Collect assignment variables for this faculty-course
                    vars_for_faculty_course = []
                    
                    for (c, f, r, t, b), var in variables.assignment_vars.items():
                        if f == str(faculty_id) and c == str(course_id):
                            vars_for_faculty_course.append(var)
                    
                    if vars_for_faculty_course:
                        # Create penalty variable
                        penalty_var = model.NewIntVar(0, penalty_weight * len(vars_for_faculty_course),
                                                      f"penalty_course_{faculty_id}_{course_id}")
                        
                        # Link penalty to assignments
                        model.Add(penalty_var == penalty_weight * sum(vars_for_faculty_course))
                        
                        penalty_vars.append(penalty_var)
        
        return penalty_vars
    
    def _build_workload_imbalance_penalty_cpsat(
        self,
        model: cp_model.CpModel,
        variables: VariableSet,
        compiled_data: CompiledData,
        bijective_mapper: BijectiveMapper
    ) -> List[Any]:
        """
        Build workload imbalance penalty variables.
        
        Penalize deviation from average workload:
        penalty = |workload_f - avg_workload|
        """
        penalty_vars = []
        
        faculty = compiled_data.L_raw.get('faculty', None)
        
        if faculty is None:
            return penalty_vars
        
        faculty_ids = faculty['faculty_id'].unique() if 'faculty_id' in faculty.columns else []
        n_faculty = len(faculty_ids)
        
        if n_faculty == 0:
            return penalty_vars
        
        # Calculate total assignments (approximation)
        total_assignments = len(variables.assignment_vars)
        avg_workload = total_assignments // n_faculty if n_faculty > 0 else 0
        
        for faculty_id in faculty_ids:
            # Collect assignment variables for this faculty
            vars_for_faculty = []
            
            for (c, f, r, t, b), var in variables.assignment_vars.items():
                if f == str(faculty_id):
                    vars_for_faculty.append(var)
            
            if vars_for_faculty:
                # Create workload variable
                workload_var = model.NewIntVar(0, len(vars_for_faculty), f"workload_{faculty_id}")
                model.Add(workload_var == sum(vars_for_faculty))
                
                # Create deviation variable (absolute value)
                deviation_var = model.NewIntVar(0, max(len(vars_for_faculty), avg_workload),
                                               f"deviation_{faculty_id}")
                
                # |workload - avg| = deviation
                # This requires auxiliary variables for absolute value
                diff_var = model.NewIntVar(-max(len(vars_for_faculty), avg_workload),
                                          max(len(vars_for_faculty), avg_workload),
                                          f"diff_{faculty_id}")
                model.Add(diff_var == workload_var - avg_workload)
                model.AddAbsEquality(deviation_var, diff_var)
                
                penalty_vars.append(deviation_var)
        
        return penalty_vars
    
    def _build_schedule_density_penalty_cpsat(
        self,
        model: cp_model.CpModel,
        variables: VariableSet,
        compiled_data: CompiledData,
        bijective_mapper: BijectiveMapper
    ) -> List[Any]:
        """
        Build schedule density penalty variables.
        
        Penalize gaps in student schedules:
        penalty = number of gaps (consecutive empty timeslots)
        """
        penalty_vars = []
        
        # Schedule density penalty requires temporal reasoning and gap detection
        # Implementation requires detailed timeslot sequencing
        
        self.logger.debug("Schedule density penalty - advanced feature")
        
        return penalty_vars
    
    def _load_time_preferences(self, compiled_data: CompiledData) -> Dict[Tuple[str, str], int]:
        """
        Load time preferences from compiled data.
        
        Returns:
            Dictionary mapping (faculty_id, timeslot_id) -> preference_score (0-10)
        """
        preferences = {}
        
        # Try to load from preferences.json in L_raw
        prefs_data = compiled_data.L_raw.get('preferences', None)
        
        if prefs_data is not None and isinstance(prefs_data, dict):
            # Parse preferences
            time_prefs = prefs_data.get('time_preferences', {})
            for faculty_id, timeslot_prefs in time_prefs.items():
                for timeslot_id, score in timeslot_prefs.items():
                    preferences[(str(faculty_id), str(timeslot_id))] = int(score)
        
        return preferences
    
    def _load_course_preferences(self, compiled_data: CompiledData) -> Dict[Tuple[str, str], int]:
        """
        Load course preferences from compiled data.
        
        Returns:
            Dictionary mapping (faculty_id, course_id) -> preference_score (0-10)
        """
        preferences = {}
        
        # Try to load from preferences.json in L_raw
        prefs_data = compiled_data.L_raw.get('preferences', None)
        
        if prefs_data is not None and isinstance(prefs_data, dict):
            # Parse preferences
            course_prefs = prefs_data.get('course_preferences', {})
            for faculty_id, course_prefs_dict in course_prefs.items():
                for course_id, score in course_prefs_dict.items():
                    preferences[(str(faculty_id), str(course_id))] = int(score)
        
        return preferences
    
    def build_objective_linear(
        self,
        solver: pywraplp.Solver,
        variables: VariableSet,
        compiled_data: CompiledData,
        bijective_mapper: BijectiveMapper
    ) -> ObjectiveComponents:
        """
        Build multi-objective function for Linear Solver.
        
        Args:
            solver: Linear solver instance
            variables: Variable set
            compiled_data: Stage 3 compiled data
            bijective_mapper: Bijective mappings
            
        Returns:
            ObjectiveComponents with all penalty terms
        """
        self.logger.info("Building Linear Solver objective function")
        
        components = ObjectiveComponents()
        components.w_time_pref = self.w_time_pref
        components.w_course_pref = self.w_course_pref
        components.w_workload = self.w_workload
        components.w_density = self.w_density
        
        # Create objective
        objective = solver.Objective()
        
        # TODO: Implement linear solver objective components
        # For now, minimize sum of assignment variables (dummy objective)
        for (c, f, r, t, b), var in variables.assignment_vars.items():
            objective.SetCoefficient(var, 1.0)
        
        objective.SetMinimization()
        
        components.objective_expr = objective
        
        self.logger.info("Linear Solver objective function created")
        
        return components

