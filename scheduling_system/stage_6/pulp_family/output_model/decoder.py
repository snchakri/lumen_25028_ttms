"""
Solution Decoder - Algorithm 4.3: Schedule Construction

Implements solution decoding per Algorithm 4.3 with rigorous validation.

Compliance:
- Algorithm 4.3: Schedule Construction Process
- Definition 4.2: Schedule Construction Function φ : S* → T_schedule

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from pulp import LpProblem, LpVariable
import pandas as pd
import numpy as np


@dataclass
class Assignment:
    """Single schedule assignment."""
    
    assignment_id: str
    course_id: str
    faculty_id: str
    room_id: str
    timeslot_id: str
    batch_id: str
    day: Optional[str] = None
    time: Optional[str] = None
    duration: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'assignment_id': self.assignment_id,
            'course_id': self.course_id,
            'faculty_id': self.faculty_id,
            'room_id': self.room_id,
            'timeslot_id': self.timeslot_id,
            'batch_id': self.batch_id,
            'day': self.day,
            'time': self.time,
            'duration': self.duration
        }


@dataclass
class Schedule:
    """Complete schedule with all assignments."""
    
    assignments: List[Assignment] = field(default_factory=list)
    objective_value: Optional[float] = None
    solver_used: str = ""
    solve_time: float = 0.0
    n_conflicts: int = 0
    n_hard_violations: int = 0
    n_soft_violations: int = 0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        if not self.assignments:
            return pd.DataFrame()
        
        data = [assignment.to_dict() for assignment in self.assignments]
        return pd.DataFrame(data)
    
    def get_assignment_count(self) -> int:
        """Get total number of assignments."""
        return len(self.assignments)


class SolutionDecoder:
    """
    Decodes PuLP solution to schedule per Algorithm 4.3.
    
    Compliance: Algorithm 4.3, Definition 4.2
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize solution decoder."""
        self.logger = logger or logging.getLogger(__name__)
    
    def decode_solution(
        self,
        problem: LpProblem,
        variable_set,
        bijective_mapping,
        solver_result,
        l_raw: Dict[str, pd.DataFrame]
    ) -> Schedule:
        """
        Decode solution per Algorithm 4.3.
        
        Compliance: Algorithm 4.3
        
        Args:
            problem: Solved PuLP problem
            variable_set: VariableSet
            bijective_mapping: BijectiveMapper
            solver_result: SolverResult
            l_raw: L_raw layer for entity data
        
        Returns:
            Schedule with all assignments
        """
        self.logger.info("Decoding solution per Algorithm 4.3...")
        
        schedule = Schedule(
            objective_value=solver_result.objective_value,
            solver_used=solver_result.solver_type.value,
            solve_time=solver_result.execution_time
        )
        
        # Step 1: Assignment Extraction
        self.logger.info("Step 1: Extracting assignments...")
        assignments = self._extract_assignments(
            problem,
            variable_set,
            bijective_mapping,
            l_raw
        )
        
        schedule.assignments = assignments
        self.logger.info(f"Extracted {len(assignments)} assignments")
        
        # Step 2: Conflict Resolution
        self.logger.info("Step 2: Verifying no constraint violations...")
        conflicts = self._verify_no_conflicts(assignments)
        schedule.n_conflicts = len(conflicts)
        
        if conflicts:
            self.logger.warning(f"Found {len(conflicts)} conflicts in solution")
            for conflict in conflicts[:5]:  # Log first 5
                self.logger.warning(f"  Conflict: {conflict}")
        else:
            self.logger.info("No conflicts found - solution is valid")
        
        # Step 3: Quality Assessment
        self.logger.info("Step 3: Assessing solution quality...")
        quality_metrics = self._assess_solution_quality(
            schedule,
            variable_set,
            l_raw
        )
        
        self.logger.info(f"Solution quality metrics:")
        for metric, value in quality_metrics.items():
            self.logger.info(f"  - {metric}: {value}")
        
        # Step 4: Format Generation (done by writers)
        self.logger.info("Step 4: Format generation delegated to output writers")
        
        # Step 5: Validation (done by Stage 7)
        self.logger.info("Step 5: Validation delegated to Stage 7")
        
        self.logger.info(f"Solution decoding complete: {len(schedule.assignments)} assignments")
        
        return schedule
    
    def _extract_assignments(
        self,
        problem: LpProblem,
        variable_set,
        bijective_mapping,
        l_raw: Dict[str, pd.DataFrame]
    ) -> List[Assignment]:
        """
        Extract assignments where x_{c,f,r,t,b} = 1.
        
        Compliance: Algorithm 4.3 Step 1
        """
        assignments = []
        assignment_id = 0
        
        # Iterate through all assignment variables
        for var_name, var in variable_set.assignment_variables.items():
            # Check if variable is set to 1 (assigned)
            if var.varValue == 1.0:
                # Parse variable name to get indices
                c_idx, f_idx, r_idx, t_idx, b_idx = bijective_mapping.parse_variable_name(var_name)
                
                # Get original entity IDs
                course_id = bijective_mapping.course_mapping.get_id(c_idx)
                faculty_id = bijective_mapping.faculty_mapping.get_id(f_idx)
                room_id = bijective_mapping.room_mapping.get_id(r_idx)
                timeslot_id = bijective_mapping.timeslot_mapping.get_id(t_idx)
                batch_id = bijective_mapping.batch_mapping.get_id(b_idx)
                
                # Get additional information from L_raw
                day = None
                time = None
                duration = None
                
                if 'time_slots.csv' in l_raw:
                    timeslots_df = l_raw['time_slots.csv']
                    timeslot_row = timeslots_df[timeslots_df['slot_id'] == timeslot_id]
                    if not timeslot_row.empty:
                        day = timeslot_row.iloc[0].get('day_name')
                        time = timeslot_row.iloc[0].get('start_time')
                
                if 'courses.csv' in l_raw:
                    courses_df = l_raw['courses.csv']
                    course_row = courses_df[courses_df['course_id'] == course_id]
                    if not course_row.empty:
                        duration = int(course_row.iloc[0].get('theory_hours', 1))
                
                # Create assignment
                assignment = Assignment(
                    assignment_id=f"ASSIGN_{assignment_id:06d}",
                    course_id=course_id,
                    faculty_id=faculty_id,
                    room_id=room_id,
                    timeslot_id=timeslot_id,
                    batch_id=batch_id,
                    day=day,
                    time=time,
                    duration=duration
                )
                
                assignments.append(assignment)
                assignment_id += 1
        
        return assignments
    
    def _verify_no_conflicts(self, assignments: List[Assignment]) -> List[str]:
        """
        Verify no constraint violations in final schedule.
        
        Compliance: Algorithm 4.3 Step 2
        """
        conflicts = []
        
        # Check for faculty conflicts
        faculty_schedule = {}
        for assignment in assignments:
            key = (assignment.faculty_id, assignment.timeslot_id)
            if key in faculty_schedule:
                conflicts.append(
                    f"Faculty {assignment.faculty_id} conflict at timeslot {assignment.timeslot_id}"
                )
            else:
                faculty_schedule[key] = assignment
        
        # Check for room conflicts
        room_schedule = {}
        for assignment in assignments:
            key = (assignment.room_id, assignment.timeslot_id)
            if key in room_schedule:
                conflicts.append(
                    f"Room {assignment.room_id} conflict at timeslot {assignment.timeslot_id}"
                )
            else:
                room_schedule[key] = assignment
        
        # Check for batch conflicts
        batch_schedule = {}
        for assignment in assignments:
            key = (assignment.batch_id, assignment.timeslot_id)
            if key in batch_schedule:
                conflicts.append(
                    f"Batch {assignment.batch_id} conflict at timeslot {assignment.timeslot_id}"
                )
            else:
                batch_schedule[key] = assignment
        
        return conflicts
    
    def _assess_solution_quality(
        self,
        schedule: Schedule,
        variable_set,
        l_raw: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Calculate objective components and satisfaction metrics.
        
        Compliance: Algorithm 4.3 Step 3
        """
        metrics = {}
        
        # Course coverage
        if 'courses.csv' in l_raw:
            n_courses = len(l_raw['courses.csv'])
            n_scheduled = len(set(a.course_id for a in schedule.assignments))
            metrics['course_coverage'] = n_scheduled / n_courses if n_courses > 0 else 0.0
        
        # Faculty utilization
        if 'faculty.csv' in l_raw:
            n_faculty = len(l_raw['faculty.csv'])
            n_used_faculty = len(set(a.faculty_id for a in schedule.assignments))
            metrics['faculty_utilization'] = n_used_faculty / n_faculty if n_faculty > 0 else 0.0
        
        # Room utilization
        if 'rooms.csv' in l_raw:
            n_rooms = len(l_raw['rooms.csv'])
            n_used_rooms = len(set(a.room_id for a in schedule.assignments))
            metrics['room_utilization'] = n_used_rooms / n_rooms if n_rooms > 0 else 0.0
        
        # Timeslot utilization
        if 'time_slots.csv' in l_raw:
            n_timeslots = len(l_raw['time_slots.csv'])
            n_used_timeslots = len(set(a.timeslot_id for a in schedule.assignments))
            metrics['timeslot_utilization'] = n_used_timeslots / n_timeslots if n_timeslots > 0 else 0.0
        
        return metrics
    
    def validate_schedule(self, schedule: Schedule) -> bool:
        """
        Validate schedule for Stage 7 compliance.
        
        Returns:
            True if valid, False otherwise
        """
        # Check assignments exist
        if not schedule.assignments:
            self.logger.error("No assignments in schedule")
            return False
        
        # Check for conflicts
        conflicts = self._verify_no_conflicts(schedule.assignments)
        if conflicts:
            self.logger.error(f"Schedule has {len(conflicts)} conflicts")
            return False
        
        # Check objective value
        if schedule.objective_value is None:
            self.logger.error("No objective value")
            return False
        
        self.logger.info("Schedule validation passed")
        return True



