"""
Fitness Evaluation Module for PyGMO Scheduling Problem

Implements the 5-objective fitness function as per Section 3 of the foundational framework:

f₁(x): Conflict penalty (hard constraints)
f₂(x): Resource underutilization
f₃(x): Preference violation
f₄(x): Workload imbalance
f₅(x): Schedule fragmentation

All objectives are to be MINIMIZED.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from uuid import UUID

from ..config import PyGMOConfig
from ..logging_system.logger import StructuredLogger
from ..input_model.input_loader import CompiledData
from .constraints import ConstraintFormulator


class FitnessEvaluator:
    """
    Evaluates the multi-objective fitness function for scheduling solutions.
    Implements the mathematical formulation from Section 3.2 of the foundations.
    """
    
    def __init__(self, compiled_data: CompiledData, config: PyGMOConfig, logger: StructuredLogger):
        self.compiled_data = compiled_data
        self.config = config
        self.logger = logger
        
        # Initialize constraint formulator
        self.constraint_formulator = ConstraintFormulator(compiled_data, config, logger)
        
        # Normalization factors for objectives (computed dynamically)
        self.normalization_factors = self._compute_normalization_factors()
        
        self.logger.info("FitnessEvaluator initialized successfully.")
    
    def _compute_normalization_factors(self) -> Dict[str, float]:
        """
        Computes normalization factors for each objective to ensure they are on comparable scales.
        This is critical for multi-objective optimization.
        
        Complexity: O(1) - based on problem dimensions
        """
        n_courses = len(self.compiled_data.courses)
        n_faculty = len(self.compiled_data.faculty)
        n_rooms = len(self.compiled_data.rooms)
        n_timeslots = len(self.compiled_data.timeslots)
        n_batches = len(self.compiled_data.batches)
        
        # Worst-case scenarios for normalization
        factors = {
            'conflict': float(n_courses ** 2),  # Maximum possible conflicts
            'utilization': 1.0,  # Already normalized to [0,1]
            'preference': float(n_courses * n_faculty),  # Maximum preference violations
            'balance': float(n_courses),  # Maximum workload std deviation
            'fragmentation': float(n_courses * n_timeslots)  # Maximum gaps
        }
        
        self.logger.debug(f"Normalization factors computed: {factors}")
        return factors
    
    def calculate_objectives(self, assignments: List[Tuple[UUID, UUID, UUID, UUID, UUID]], 
                            continuous_vars: List[float]) -> List[float]:
        """
        Calculates all 5 objectives for a given solution.
        
        Args:
            assignments: List of (course_id, faculty_id, room_id, timeslot_id, batch_id) tuples
            continuous_vars: Continuous decision variables
        
        Returns:
            List of 5 objective values [f1, f2, f3, f4, f5]
        """
        # Objective 1: Conflict penalty (hard constraints)
        f1 = self._objective_conflict_penalty(assignments)
        
        # Objective 2: Resource underutilization
        f2 = self._objective_resource_utilization(assignments)
        
        # Objective 3: Preference violation
        f3 = self._objective_preference_violation(assignments, continuous_vars)
        
        # Objective 4: Workload imbalance
        f4 = self._objective_workload_balance(assignments)
        
        # Objective 5: Schedule fragmentation
        f5 = self._objective_schedule_compactness(assignments)
        
        # Normalize objectives
        objectives = [
            f1 / self.normalization_factors['conflict'],
            f2 / self.normalization_factors['utilization'],
            f3 / self.normalization_factors['preference'],
            f4 / self.normalization_factors['balance'],
            f5 / self.normalization_factors['fragmentation']
        ]
        
        return objectives
    
    def _objective_conflict_penalty(self, assignments: List[Tuple[UUID, UUID, UUID, UUID, UUID]]) -> float:
        """
        f₁(x): Total penalty from hard constraint violations.
        
        Mathematical formulation (Section 3.2.1):
        f₁(x) = Σᵢ max(0, gᵢ(x))² + Σⱼ |hⱼ(x)|²
        
        where:
        - gᵢ(x) ≤ 0 are inequality constraints (capacity, availability)
        - hⱼ(x) = 0 are equality constraints (assignment requirements)
        """
        hard_penalty, violations = self.constraint_formulator.evaluate_hard_constraints(assignments)
        
        # Log severe violations
        if violations['faculty_conflict'] > 0 or violations['room_conflict'] > 0:
            self.logger.debug(f"Conflicts detected: {violations}")
        
        return hard_penalty
    
    def _objective_resource_utilization(self, assignments: List[Tuple[UUID, UUID, UUID, UUID, UUID]]) -> float:
        """
        f₂(x): Resource underutilization penalty.
        
        Mathematical formulation (Section 3.2.2):
        f₂(x) = Σᵣ (1 - uᵣ)² + Σₜ (1 - uₜ)²
        
        where:
        - uᵣ is utilization rate of room r
        - uₜ is utilization rate of timeslot t
        """
        n_rooms = len(self.compiled_data.rooms)
        n_timeslots = len(self.compiled_data.timeslots)
        
        # Calculate room utilization
        room_usage = {rid: 0 for rid in self.compiled_data.rooms.keys()}
        timeslot_usage = {tid: 0 for tid in self.compiled_data.timeslots.keys()}
        
        for course_id, faculty_id, room_id, timeslot_id, batch_id in assignments:
            room_usage[room_id] += 1
            timeslot_usage[timeslot_id] += 1
        
        # Normalize by total available slots
        room_underutilization = sum((1.0 - min(1.0, usage / n_timeslots)) ** 2 
                                    for usage in room_usage.values())
        timeslot_underutilization = sum((1.0 - min(1.0, usage / n_rooms)) ** 2 
                                        for usage in timeslot_usage.values())
        
        total_underutilization = room_underutilization + timeslot_underutilization
        
        return total_underutilization
    
    def _objective_preference_violation(self, assignments: List[Tuple[UUID, UUID, UUID, UUID, UUID]], 
                                       continuous_vars: List[float]) -> float:
        """
        f₃(x): Preference violation penalty.
        
        Mathematical formulation (Section 3.2.3):
        f₃(x) = Σₐ wₐ · (1 - pₐ)²
        
        where:
        - wₐ is the weight for assignment a (from continuous_vars)
        - pₐ is the preference score for assignment a
        """
        soft_penalties = self.constraint_formulator.evaluate_soft_constraints(assignments, continuous_vars)
        return soft_penalties['preference_violation']
    
    def _objective_workload_balance(self, assignments: List[Tuple[UUID, UUID, UUID, UUID, UUID]]) -> float:
        """
        f₄(x): Workload imbalance penalty.
        
        Mathematical formulation (Section 3.2.4):
        f₄(x) = σ(L) = √(Σᶠ (Lᶠ - L̄)² / |F|)
        
        where:
        - Lᶠ is the workload of faculty f
        - L̄ is the mean workload
        - σ(L) is the standard deviation of workload
        """
        from collections import defaultdict
        
        faculty_workload = defaultdict(int)
        for course_id, faculty_id, room_id, timeslot_id, batch_id in assignments:
            faculty_workload[faculty_id] += 1
        
        if not faculty_workload:
            return 0.0
        
        workloads = list(faculty_workload.values())
        
        # Add zero workload for unassigned faculty
        n_assigned = len(faculty_workload)
        n_total = len(self.compiled_data.faculty)
        workloads.extend([0] * (n_total - n_assigned))
        
        # Calculate standard deviation
        mean_workload = np.mean(workloads)
        std_workload = np.std(workloads)
        
        # Also penalize deviation from ideal workload
        ideal_workload = len(self.compiled_data.courses) / n_total if n_total > 0 else 0
        mean_deviation = abs(mean_workload - ideal_workload)
        
        return std_workload + mean_deviation
    
    def _objective_schedule_compactness(self, assignments: List[Tuple[UUID, UUID, UUID, UUID, UUID]]) -> float:
        """
        f₅(x): Schedule fragmentation penalty.
        
        Mathematical formulation (Section 3.2.5):
        f₅(x) = Σᵦ Σᵢ max(0, tᵢ₊₁ - tᵢ - 1)
        
        where:
        - b is a batch
        - tᵢ is the i-th timeslot in batch b's schedule (sorted)
        - Gaps are penalized
        """
        from collections import defaultdict
        
        # Build batch schedules with actual time ordering
        batch_schedules = defaultdict(list)
        
        # Create a mapping from timeslot_id to sequential index
        # This requires timeslot ordering information from compiled_data
        timeslot_order = {tid: idx for idx, tid in enumerate(sorted(self.compiled_data.timeslots.keys()))}
        
        for course_id, faculty_id, room_id, timeslot_id, batch_id in assignments:
            timeslot_index = timeslot_order.get(timeslot_id, 0)
            batch_schedules[batch_id].append(timeslot_index)
        
        # Calculate fragmentation for each batch
        total_fragmentation = 0.0
        for batch_id, timeslot_indices in batch_schedules.items():
            if len(timeslot_indices) <= 1:
                continue
            
            sorted_indices = sorted(timeslot_indices)
            
            # Count gaps
            gaps = 0
            for i in range(len(sorted_indices) - 1):
                gap = sorted_indices[i+1] - sorted_indices[i] - 1
                if gap > 0:
                    gaps += gap
            
            total_fragmentation += gaps
        
        return total_fragmentation
    
    def get_fitness_summary(self, assignments: List[Tuple[UUID, UUID, UUID, UUID, UUID]], 
                           continuous_vars: List[float]) -> Dict[str, Any]:
        """
        Returns a comprehensive summary of fitness evaluation for logging and analysis.
        """
        objectives = self.calculate_objectives(assignments, continuous_vars)
        constraint_summary = self.constraint_formulator.get_constraint_summary(assignments)
        
        return {
            'objectives': {
                'f1_conflict': objectives[0],
                'f2_utilization': objectives[1],
                'f3_preference': objectives[2],
                'f4_balance': objectives[3],
                'f5_compactness': objectives[4]
            },
            'objectives_raw': {
                'f1_conflict': objectives[0] * self.normalization_factors['conflict'],
                'f2_utilization': objectives[1] * self.normalization_factors['utilization'],
                'f3_preference': objectives[2] * self.normalization_factors['preference'],
                'f4_balance': objectives[3] * self.normalization_factors['balance'],
                'f5_compactness': objectives[4] * self.normalization_factors['fragmentation']
            },
            'constraint_summary': constraint_summary,
            'is_feasible': constraint_summary['is_feasible'],
            'total_penalty': sum(objectives)
        }


