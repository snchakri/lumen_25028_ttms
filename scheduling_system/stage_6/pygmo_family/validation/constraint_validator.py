"""
Constraint Validator Module

Validates constraint satisfaction in solutions (INTERNAL ONLY, not Stage 7).

Validates:
- Hard constraints: Faculty/room conflicts, course assignments, competency
- Soft constraints: Preferences, capacity matching, time preferences
"""

from typing import Dict, Any, List, Tuple
from collections import defaultdict

from ..config import PyGMOConfig
from ..logging_system.logger import StructuredLogger


class ConstraintValidator:
    """
    Validates constraint satisfaction in solutions.
    """
    
    def __init__(self, config: PyGMOConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        
        self.logger.info("ConstraintValidator initialized successfully.")
    
    def validate_solution(self, assignments: List[Tuple], 
                         compiled_data: Any) -> Dict[str, Any]:
        """
        Validates all constraints for a given solution.
        
        Args:
            assignments: List of (course_id, faculty_id, room_id, timeslot_id, batch_id) tuples
            compiled_data: Compiled data from Stage 3
        
        Returns:
            Dictionary with validation results
        """
        self.logger.info(f"Validating constraints for {len(assignments)} assignments.")
        
        # Validate hard constraints
        hard_constraints = self._validate_hard_constraints(assignments, compiled_data)
        
        # Validate soft constraints
        soft_constraints = self._validate_soft_constraints(assignments, compiled_data)
        
        # Overall feasibility
        is_feasible = hard_constraints['is_satisfied']
        
        return {
            'is_feasible': is_feasible,
            'hard_constraints': hard_constraints,
            'soft_constraints': soft_constraints,
            'total_violations': hard_constraints['violation_count'] + soft_constraints['violation_count']
        }
    
    def _validate_hard_constraints(self, assignments: List[Tuple], 
                                  compiled_data: Any) -> Dict[str, Any]:
        """
        Validates hard constraints.
        """
        violations = {
            'faculty_conflicts': 0,
            'room_conflicts': 0,
            'unassigned_courses': 0,
            'incompetent_assignments': 0,
            'capacity_violations': 0
        }
        
        # Track assignments for conflict detection
        faculty_time_map = defaultdict(list)
        room_time_map = defaultdict(list)
        course_assigned = set()
        
        for course_id, faculty_id, room_id, timeslot_id, batch_id in assignments:
            # Track for conflict detection
            faculty_time_map[(faculty_id, timeslot_id)].append(course_id)
            room_time_map[(room_id, timeslot_id)].append(course_id)
            course_assigned.add(course_id)
            
            # Check competency
            competency = compiled_data.competency_matrix.get((faculty_id, course_id), 0.0)
            if competency < 0.5:
                violations['incompetent_assignments'] += 1
            
            # Check capacity
            enrollment = sum(count for (bid, cid), count in compiled_data.enrollment_matrix.items() 
                           if cid == course_id)
            room_capacity = compiled_data.rooms.get(room_id, {}).get('capacity', 0)
            if enrollment > room_capacity:
                violations['capacity_violations'] += 1
        
        # Check faculty conflicts
        for (faculty_id, timeslot_id), courses in faculty_time_map.items():
            if len(courses) > 1:
                violations['faculty_conflicts'] += len(courses) - 1
        
        # Check room conflicts
        for (room_id, timeslot_id), courses in room_time_map.items():
            if len(courses) > 1:
                violations['room_conflicts'] += len(courses) - 1
        
        # Check unassigned courses
        all_courses = set(compiled_data.courses.keys())
        unassigned = all_courses - course_assigned
        violations['unassigned_courses'] = len(unassigned)
        
        # Check if all hard constraints satisfied
        is_satisfied = all(count == 0 for count in violations.values())
        violation_count = sum(violations.values())
        
        return {
            'is_satisfied': is_satisfied,
            'violations': violations,
            'violation_count': violation_count
        }
    
    def _validate_soft_constraints(self, assignments: List[Tuple], 
                                  compiled_data: Any) -> Dict[str, Any]:
        """
        Validates soft constraints.
        """
        penalties = {
            'preference_penalty': 0.0,
            'capacity_mismatch': 0.0,
            'time_preference_penalty': 0.0,
            'workload_imbalance': 0.0
        }
        
        # Calculate preference penalties
        for course_id, faculty_id, room_id, timeslot_id, batch_id in assignments:
            # Faculty course preference (simplified)
            preference = compiled_data.competency_matrix.get((faculty_id, course_id), 0.5)
            penalties['preference_penalty'] += (1.0 - preference) ** 2
        
        # Calculate capacity mismatch
        for course_id, faculty_id, room_id, timeslot_id, batch_id in assignments:
            enrollment = sum(count for (bid, cid), count in compiled_data.enrollment_matrix.items() 
                           if cid == course_id)
            room_capacity = compiled_data.rooms.get(room_id, {}).get('capacity', 0)
            if room_capacity > 0:
                mismatch = abs(enrollment - room_capacity) / room_capacity
                penalties['capacity_mismatch'] += mismatch
        
        # Calculate workload imbalance
        faculty_workload = defaultdict(int)
        for course_id, faculty_id, room_id, timeslot_id, batch_id in assignments:
            faculty_workload[faculty_id] += 1
        
        if faculty_workload:
            workloads = list(faculty_workload.values())
            mean_workload = sum(workloads) / len(workloads)
            variance = sum((w - mean_workload) ** 2 for w in workloads) / len(workloads)
            penalties['workload_imbalance'] = variance
        
        # Calculate total penalty
        total_penalty = sum(penalties.values())
        violation_count = int(total_penalty)  # Simplified
        
        return {
            'penalties': penalties,
            'total_penalty': total_penalty,
            'violation_count': violation_count
        }
    
    def validate_feasibility(self, assignments: List[Tuple], 
                           compiled_data: Any) -> bool:
        """
        Quick feasibility check (hard constraints only).
        
        Args:
            assignments: List of assignments
            compiled_data: Compiled data
        
        Returns:
            True if feasible, False otherwise
        """
        hard_constraints = self._validate_hard_constraints(assignments, compiled_data)
        return hard_constraints['is_satisfied']


