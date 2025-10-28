"""
Constraint Formulation Module for PyGMO Scheduling Problem

Implements hard and soft constraints with penalty functions as per
Section 4 of the foundational framework.

Hard Constraints (g_i(x) ≤ 0):
- Faculty conflict: No faculty assigned to multiple courses at same time
- Room conflict: No room assigned to multiple courses at same time
- Course assignment: Each course must be assigned exactly once
- Competency: Faculty must be competent for assigned courses
- Capacity: Room capacity must meet enrollment requirements

Soft Constraints (penalty-based):
- Faculty preferences: Time slot and course preferences
- Room preferences: Department and equipment preferences
- Workload balance: Even distribution across faculty
- Schedule compactness: Minimize gaps in student schedules
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Set
from uuid import UUID
from collections import defaultdict

from ..config import PyGMOConfig
from ..logging_system.logger import StructuredLogger
from ..input_model.input_loader import CompiledData


class ConstraintFormulator:
    """
    Formulates and evaluates all hard and soft constraints for the scheduling problem.
    Implements penalty functions Φ(x,μ) = f(x) + μᵀφ(g(x)) + νᵀψ(h(x))
    """
    
    def __init__(self, compiled_data: CompiledData, config: PyGMOConfig, logger: StructuredLogger):
        self.compiled_data = compiled_data
        self.config = config
        self.logger = logger
        
        # Extract data references
        self.courses = compiled_data.courses
        self.faculty = compiled_data.faculty
        self.rooms = compiled_data.rooms
        self.timeslots = compiled_data.timeslots
        self.batches = compiled_data.batches
        self.competency_matrix = compiled_data.competency_matrix
        self.enrollment_matrix = compiled_data.enrollment_matrix
        self.dynamic_constraints = compiled_data.constraints
        
        # Penalty weights (can be overridden by dynamic parameters)
        self.conflict_penalty_weight = config.conflict_penalty_weight
        self.utilization_weight = config.utilization_weight
        self.preference_weight = config.preference_weight
        self.balance_weight = config.balance_weight
        self.compactness_weight = config.compactness_weight
        
        # Precompute constraint data structures for O(1) lookups
        self._precompute_constraint_structures()
        
        self.logger.info("ConstraintFormulator initialized successfully.")
    
    def _precompute_constraint_structures(self):
        """
        Precomputes data structures for efficient constraint evaluation.
        Complexity: O(|F| + |R| + |C| + |T| + |B|)
        """
        self.logger.debug("Precomputing constraint data structures.")
        
        # Faculty availability: faculty_id -> set of available timeslot_ids
        self.faculty_availability: Dict[UUID, Set[UUID]] = defaultdict(set)
        for fid, fdata in self.faculty.items():
            # Assuming faculty data has 'available_timeslots' field
            available_slots = fdata.get('available_timeslots', [])
            if available_slots:
                self.faculty_availability[fid] = set(available_slots)
            else:
                # If not specified, assume all timeslots are available
                self.faculty_availability[fid] = set(self.timeslots.keys())
        
        # Room availability: room_id -> set of available timeslot_ids
        self.room_availability: Dict[UUID, Set[UUID]] = defaultdict(set)
        for rid, rdata in self.rooms.items():
            available_slots = rdata.get('available_timeslots', [])
            if available_slots:
                self.room_availability[rid] = set(available_slots)
            else:
                self.room_availability[rid] = set(self.timeslots.keys())
        
        # Room capacity: room_id -> capacity
        self.room_capacity: Dict[UUID, int] = {
            rid: rdata.get('capacity', 0) for rid, rdata in self.rooms.items()
        }
        
        # Course enrollment: course_id -> total enrollment
        self.course_enrollment: Dict[UUID, int] = defaultdict(int)
        for (batch_id, course_id), enrollment in self.enrollment_matrix.items():
            self.course_enrollment[course_id] += enrollment
        
        # Faculty competency: (faculty_id, course_id) -> competency level
        # Already available as self.competency_matrix
        
        # Faculty preferences: faculty_id -> {timeslot_id: preference_score, course_id: preference_score}
        self.faculty_preferences: Dict[UUID, Dict[str, Dict[UUID, float]]] = defaultdict(lambda: {'timeslot': {}, 'course': {}})
        # Populated from dynamic_constraints or preferences configuration
        # Extracts and structures faculty preference data
        for constraint in self.dynamic_constraints:
            if constraint.get('type') == 'faculty_preference':
                fid = UUID(constraint['faculty_id'])
                if 'timeslot_id' in constraint:
                    tid = UUID(constraint['timeslot_id'])
                    self.faculty_preferences[fid]['timeslot'][tid] = constraint.get('preference_score', 1.0)
                if 'course_id' in constraint:
                    cid = UUID(constraint['course_id'])
                    self.faculty_preferences[fid]['course'][cid] = constraint.get('preference_score', 1.0)
        
        self.logger.debug("Constraint data structures precomputed.")
    
    def evaluate_hard_constraints(self, assignments: List[Tuple[UUID, UUID, UUID, UUID, UUID]]) -> Tuple[float, Dict[str, int]]:
        """
        Evaluates all hard constraints and returns total penalty and violation counts.
        
        Args:
            assignments: List of (course_id, faculty_id, room_id, timeslot_id, batch_id) tuples
        
        Returns:
            (total_penalty, violation_counts_dict)
        """
        violations = {
            'faculty_conflict': 0,
            'room_conflict': 0,
            'course_assignment': 0,
            'competency': 0,
            'capacity': 0,
            'availability': 0
        }
        
        # Track assignments for conflict detection
        faculty_time_map: Dict[Tuple[UUID, UUID], List[UUID]] = defaultdict(list)  # (faculty_id, timeslot_id) -> [course_ids]
        room_time_map: Dict[Tuple[UUID, UUID], List[UUID]] = defaultdict(list)  # (room_id, timeslot_id) -> [course_ids]
        course_assignment_count: Dict[UUID, int] = defaultdict(int)  # course_id -> count
        
        for course_id, faculty_id, room_id, timeslot_id, batch_id in assignments:
            # Track for conflict detection
            faculty_time_map[(faculty_id, timeslot_id)].append(course_id)
            room_time_map[(room_id, timeslot_id)].append(course_id)
            course_assignment_count[course_id] += 1
            
            # Check competency constraint
            competency = self.competency_matrix.get((faculty_id, course_id), 0.0)
            if competency < 0.5:  # Minimum competency threshold
                violations['competency'] += 1
            
            # Check capacity constraint
            enrollment = self.course_enrollment.get(course_id, 0)
            room_cap = self.room_capacity.get(room_id, 0)
            if enrollment > room_cap:
                violations['capacity'] += 1
            
            # Check availability constraints
            if timeslot_id not in self.faculty_availability.get(faculty_id, set()):
                violations['availability'] += 1
            if timeslot_id not in self.room_availability.get(room_id, set()):
                violations['availability'] += 1
        
        # Check faculty conflicts
        for (faculty_id, timeslot_id), course_list in faculty_time_map.items():
            if len(course_list) > 1:
                violations['faculty_conflict'] += len(course_list) - 1
        
        # Check room conflicts
        for (room_id, timeslot_id), course_list in room_time_map.items():
            if len(course_list) > 1:
                violations['room_conflict'] += len(course_list) - 1
        
        # Check course assignment (each course should be assigned exactly once)
        for course_id, count in course_assignment_count.items():
            if count != 1:
                violations['course_assignment'] += abs(count - 1)
        
        # Check for unassigned courses
        assigned_courses = set(course_assignment_count.keys())
        all_courses = set(self.courses.keys())
        unassigned = all_courses - assigned_courses
        violations['course_assignment'] += len(unassigned)
        
        # Calculate total penalty using exponential penalty function
        # Φ(g) = Σ max(0, g_i)² for inequality constraints
        total_penalty = sum(v ** 2 for v in violations.values()) * self.conflict_penalty_weight
        
        return total_penalty, violations
    
    def evaluate_soft_constraints(self, assignments: List[Tuple[UUID, UUID, UUID, UUID, UUID]], 
                                   continuous_vars: List[float]) -> Dict[str, float]:
        """
        Evaluates all soft constraints and returns individual penalty values.
        
        Args:
            assignments: List of (course_id, faculty_id, room_id, timeslot_id, batch_id) tuples
            continuous_vars: Continuous decision variables (preferences, weights)
        
        Returns:
            Dictionary of soft constraint penalties
        """
        penalties = {
            'preference_violation': 0.0,
            'utilization': 0.0,
            'workload_imbalance': 0.0,
            'schedule_fragmentation': 0.0
        }
        
        # Parse continuous variables
        # Structure: [faculty_pref_weights (|F|*3), course_importance (|C|*2)]
        n_faculty = len(self.faculty)
        n_courses = len(self.courses)
        
        faculty_pref_weights = continuous_vars[:n_faculty * 3]
        course_importance = continuous_vars[n_faculty * 3:n_faculty * 3 + n_courses * 2]
        
        # 1. Preference violation penalty
        preference_penalty = 0.0
        for course_id, faculty_id, room_id, timeslot_id, batch_id in assignments:
            # Faculty timeslot preference
            timeslot_pref = self.faculty_preferences.get(faculty_id, {}).get('timeslot', {}).get(timeslot_id, 0.5)
            preference_penalty += (1.0 - timeslot_pref) ** 2
            
            # Faculty course preference
            course_pref = self.faculty_preferences.get(faculty_id, {}).get('course', {}).get(course_id, 0.5)
            preference_penalty += (1.0 - course_pref) ** 2
        
        penalties['preference_violation'] = preference_penalty * self.preference_weight
        
        # 2. Resource utilization penalty (underutilization)
        total_timeslots = len(self.timeslots)
        total_rooms = len(self.rooms)
        total_capacity = total_timeslots * total_rooms
        
        if total_capacity > 0:
            utilization_rate = len(assignments) / total_capacity
            # Penalize underutilization (target ~70-80% utilization)
            target_utilization = 0.75
            penalties['utilization'] = abs(utilization_rate - target_utilization) * self.utilization_weight
        
        # 3. Workload imbalance penalty
        faculty_workload: Dict[UUID, int] = defaultdict(int)
        for course_id, faculty_id, room_id, timeslot_id, batch_id in assignments:
            faculty_workload[faculty_id] += 1
        
        if faculty_workload:
            workloads = list(faculty_workload.values())
            mean_workload = np.mean(workloads)
            std_workload = np.std(workloads)
            penalties['workload_imbalance'] = std_workload * self.balance_weight
        
        # 4. Schedule fragmentation penalty (gaps in schedule)
        # For each batch, calculate gaps in their schedule
        batch_schedules: Dict[UUID, List[int]] = defaultdict(list)
        for course_id, faculty_id, room_id, timeslot_id, batch_id in assignments:
            # Convert timeslot_id to numeric index for gap calculation
            # Maps timeslot UUIDs to sequential indices based on time ordering
            # Uses hash-based ordering as simplified index representation
            timeslot_index = hash(timeslot_id) % 1000
            batch_schedules[batch_id].append(timeslot_index)
        
        fragmentation_penalty = 0.0
        for batch_id, timeslot_indices in batch_schedules.items():
            if len(timeslot_indices) > 1:
                sorted_indices = sorted(timeslot_indices)
                gaps = sum(sorted_indices[i+1] - sorted_indices[i] - 1 
                          for i in range(len(sorted_indices) - 1))
                fragmentation_penalty += gaps
        
        penalties['schedule_fragmentation'] = fragmentation_penalty * self.compactness_weight
        
        return penalties
    
    def get_constraint_summary(self, assignments: List[Tuple[UUID, UUID, UUID, UUID, UUID]]) -> Dict[str, Any]:
        """
        Returns a comprehensive summary of all constraint evaluations for logging.
        """
        hard_penalty, hard_violations = self.evaluate_hard_constraints(assignments)
        soft_penalties = self.evaluate_soft_constraints(assignments, [0.5] * (len(self.faculty) * 3 + len(self.courses) * 2))
        
        return {
            'hard_constraint_penalty': hard_penalty,
            'hard_violations': hard_violations,
            'soft_penalties': soft_penalties,
            'total_violations': sum(hard_violations.values()),
            'is_feasible': sum(hard_violations.values()) == 0
        }


