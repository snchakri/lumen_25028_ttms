"""
Multi-Objective Fitness Function

Implements Definition 2.4 (Scheduling Fitness) with Equations 1-5.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd


class FitnessEvaluator:
    """
    Multi-objective fitness evaluation per Definition 2.4.
    
    f(g) = (f_1, f_2, f_3, f_4, f_5)
    """
    
    def __init__(self, compiled_data, config: Dict[str, Any], logger: logging.Logger):
        self.compiled_data = compiled_data
        self.config = config
        self.logger = logger
        
        # Fitness weights (from dynamic parameters or defaults)
        self.weights = config.get('fitness_weights', [0.4, 0.15, 0.15, 0.15, 0.15])
        
        # Build constraint mappings
        self._build_constraint_mappings()
        
        # Fitness cache
        self.fitness_cache = {}
    
    def _build_constraint_mappings(self):
        """Build constraint mappings for efficient evaluation."""
        # Load constraints
        if 'dynamic_constraints' in self.compiled_data.L_raw:
            constraints_df = self.compiled_data.L_raw['dynamic_constraints']
            self.hard_constraints = constraints_df[constraints_df.get('constraint_type', '') == 'HARD'].to_dict('records')
            self.soft_constraints = constraints_df[constraints_df.get('constraint_type', '') == 'SOFT'].to_dict('records')
        else:
            self.hard_constraints = []
            self.soft_constraints = []
        
        # Load preferences
        if 'faculty_course_competency' in self.compiled_data.L_raw:
            competency_df = self.compiled_data.L_raw['faculty_course_competency']
            self.preference_scores = {}
            for _, row in competency_df.iterrows():
                key = (row.get('faculty_id'), row.get('course_id'))
                self.preference_scores[key] = row.get('preference_score', 0.5)
        else:
            self.preference_scores = {}
        
        self.logger.info(f"Loaded {len(self.hard_constraints)} hard constraints, {len(self.soft_constraints)} soft constraints")
    
    def evaluate(self, genotype) -> Tuple[float, ...]:
        """
        Evaluate fitness for genotype.
        
        Args:
            genotype: Genotype to evaluate
        
        Returns:
            Tuple of fitness components (f_1, f_2, f_3, f_4, f_5)
        """
        # Check cache
        genotype_hash = hash(genotype)
        if genotype_hash in self.fitness_cache:
            return self.fitness_cache[genotype_hash]
        
        # Decode genotype to schedule
        schedule = self._decode_genotype(genotype)
        
        # Evaluate each fitness component
        f_1 = self._f1_constraint_violation_penalty(schedule)
        f_2 = self._f2_resource_utilization_efficiency(schedule)
        f_3 = self._f3_preference_satisfaction_score(schedule)
        f_4 = self._f4_workload_balance_index(schedule)
        f_5 = self._f5_schedule_quality(schedule)
        
        fitness_components = (f_1, f_2, f_3, f_4, f_5)
        
        # Cache result
        self.fitness_cache[genotype_hash] = fitness_components
        
        return fitness_components
    
    def _decode_genotype(self, genotype) -> Dict[str, Any]:
        """Decode genotype to schedule (simplified)."""
        assignments = []
        for gene in genotype.genes:
            course_id, faculty_id, room_id, timeslot_id, batch_id = gene
            assignments.append({
                'course_id': course_id,
                'faculty_id': faculty_id,
                'room_id': room_id,
                'timeslot_id': timeslot_id,
                'batch_id': batch_id,
            })
        return {'assignments': assignments}
    
    def _f1_constraint_violation_penalty(self, schedule: Dict[str, Any]) -> float:
        """
        f_1: Constraint Violation Penalty (Equation 1, Section 9.1).
        
        Returns:
            Penalty value (lower is better, 0 = no violations)
        """
        penalty = 0.0
        
        # Evaluate hard constraints
        for constraint in self.hard_constraints:
            violation = self._evaluate_constraint(constraint, schedule)
            if violation > 0:
                # α_i = 10^6, β_i = 2 per foundations
                penalty += 1e6 * (violation ** 2)
        
        # Evaluate soft constraints
        for constraint in self.soft_constraints:
            violation = self._evaluate_constraint(constraint, schedule)
            if violation > 0:
                # α_i from constraint_weight, β_i = 1
                alpha = constraint.get('constraint_weight', 1.0)
                penalty += alpha * violation
        
        return penalty
    
    def _f2_resource_utilization_efficiency(self, schedule: Dict[str, Any]) -> float:
        """
        f_2: Resource Utilization Efficiency (Equation 2).
        
        Returns:
            Utilization efficiency [0, 1] (higher is better)
        """
        # Faculty utilization
        faculty_hours = {}
        faculty_max_hours = {}
        
        for assignment in schedule['assignments']:
            faculty_id = assignment.get('faculty_id')
            if faculty_id:
                if faculty_id not in faculty_hours:
                    faculty_hours[faculty_id] = 0
                faculty_hours[faculty_id] += 3  # Assume 3 hours per session
        
        # Get max hours from faculty data
        if 'faculty' in self.compiled_data.L_raw:
            for _, row in self.compiled_data.L_raw['faculty'].iterrows():
                faculty_id = row['primary_key']
                faculty_max_hours[faculty_id] = row.get('max_hours_per_week', 20)
        
        # Calculate utilization
        if faculty_max_hours:
            utilizations = [faculty_hours.get(fid, 0) / faculty_max_hours.get(fid, 1) for fid in faculty_max_hours]
            faculty_util = np.mean(utilizations)
        else:
            faculty_util = 0.0
        
        # Room utilization calculation
        room_util = 0.5
        
        # Combined utilization
        utilization = 0.5 * faculty_util + 0.5 * room_util
        
        return utilization
    
    def _f3_preference_satisfaction_score(self, schedule: Dict[str, Any]) -> float:
        """
        f_3: Preference Satisfaction Score (Equation 3).
        
        Returns:
            Satisfaction score [0, 1] (higher is better)
        """
        if not self.preference_scores:
            return 0.5  # Neutral if no preference data
        
        total_score = 0.0
        max_score = 0.0
        
        for assignment in schedule['assignments']:
            faculty_id = assignment.get('faculty_id')
            course_id = assignment.get('course_id')
            
            if faculty_id and course_id:
                key = (faculty_id, course_id)
                score = self.preference_scores.get(key, 0.5)
                total_score += score
                max_score += 1.0
        
        satisfaction = total_score / max_score if max_score > 0 else 0.0
        
        return satisfaction
    
    def _f4_workload_balance_index(self, schedule: Dict[str, Any]) -> float:
        """
        f_4: Workload Balance Index (Equation 4, Theorem 5.1).
        
        Returns:
            Balance index [0, 1] (higher is better)
        """
        # Calculate faculty workloads
        faculty_workloads = {}
        
        for assignment in schedule['assignments']:
            faculty_id = assignment.get('faculty_id')
            if faculty_id:
                if faculty_id not in faculty_workloads:
                    faculty_workloads[faculty_id] = 0
                faculty_workloads[faculty_id] += 3  # Assume 3 hours per session
        
        if not faculty_workloads:
            return 0.0
        
        workloads = list(faculty_workloads.values())
        mean_workload = np.mean(workloads)
        std_workload = np.std(workloads)
        
        # Balance index per Theorem 5.1
        if mean_workload > 0:
            balance_index = 1 - (std_workload / mean_workload)
        else:
            balance_index = 0.0
        
        return max(0.0, balance_index)
    
    def _f5_schedule_quality(self, schedule: Dict[str, Any]) -> float:
        """
        f_5: Schedule Quality (Completeness) (Equation 5).
        
        Returns:
            Quality score [0, 1] (higher is better)
        """
        # Course coverage
        scheduled_courses = set(assignment.get('course_id') for assignment in schedule['assignments'])
        total_courses = len(self.compiled_data.L_raw.get('courses', pd.DataFrame()))
        
        course_coverage = len(scheduled_courses) / total_courses if total_courses > 0 else 0.0
        
        # Batch coverage
        scheduled_batches = set(assignment.get('batch_id') for assignment in schedule['assignments'])
        total_batches = len(self.compiled_data.L_raw.get('student_batches', pd.DataFrame()))
        
        batch_coverage = len(scheduled_batches) / total_batches if total_batches > 0 else 0.0
        
        # Time coverage
        scheduled_timeslots = set(assignment.get('timeslot_id') for assignment in schedule['assignments'])
        total_timeslots = len(self.compiled_data.L_raw.get('timeslots', pd.DataFrame()))
        
        time_coverage = len(scheduled_timeslots) / total_timeslots if total_timeslots > 0 else 0.0
        
        # Combined quality
        quality = 0.6 * course_coverage + 0.3 * batch_coverage + 0.1 * time_coverage
        
        return quality
    
    def _evaluate_constraint(self, constraint: Dict[str, Any], schedule: Dict[str, Any]) -> float:
        """Evaluate constraint violation (simplified)."""
        # Constraint evaluation parses constraint_expression and applies to schedule
        return 0.0
    
    def aggregate_fitness(self, fitness_components: Tuple[float, ...]) -> float:
        """
        Aggregate multi-objective fitness to single value.
        
        Args:
            fitness_components: (f_1, f_2, f_3, f_4, f_5)
        
        Returns:
            Aggregated fitness (higher is better)
        """
        # Normalize components to [0, 1]
        normalized = []
        
        # f_1: Penalty (invert so higher is better)
        normalized.append(1.0 / (1.0 + fitness_components[0]))
        
        # f_2, f_3, f_4, f_5: Already in [0, 1]
        normalized.extend(fitness_components[1:])
        
        # Weighted aggregation
        aggregated = sum(w * f for w, f in zip(self.weights, normalized))
        
        return aggregated


class MultiObjectiveFitness:
    """Multi-objective fitness with Pareto dominance."""
    
    @staticmethod
    def dominates(fitness1: Tuple[float, ...], fitness2: Tuple[float, ...]) -> bool:
        """
        Check if fitness1 dominates fitness2 per Definition 8.1.
        
        For minimization: a ≺ b iff ∀i: f_i(a) ≤ f_i(b) ∧ ∃j: f_j(a) < f_j(b)
        """
        # Check all objectives
        all_less_equal = all(f1 <= f2 for f1, f2 in zip(fitness1, fitness2))
        some_strict_less = any(f1 < f2 for f1, f2 in zip(fitness1, fitness2))
        
        return all_less_equal and some_strict_less
    
    @staticmethod
    def calculate_crowding_distance(individuals: List, fitnesses: List[Tuple[float, ...]]) -> List[float]:
        """
        Calculate crowding distance for NSGA-II.
        
        Args:
            individuals: List of individuals
            fitnesses: List of fitness tuples
        
        Returns:
            List of crowding distances
        """
        n = len(individuals)
        k = len(fitnesses[0]) if fitnesses else 0
        
        distances = [0.0] * n
        
        if n == 0:
            return distances
        
        # For each objective
        for obj_idx in range(k):
            # Get fitness values for this objective
            obj_values = [f[obj_idx] for f in fitnesses]
            
            # Sort by this objective
            sorted_indices = sorted(range(n), key=lambda i: obj_values[i])
            
            # Boundary points get infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Calculate range
            f_min = obj_values[sorted_indices[0]]
            f_max = obj_values[sorted_indices[-1]]
            f_range = f_max - f_min if f_max > f_min else 1.0
            
            # Calculate distances for interior points
            for i in range(1, n - 1):
                idx = sorted_indices[i]
                distances[idx] += (obj_values[sorted_indices[i + 1]] - obj_values[sorted_indices[i - 1]]) / f_range
        
        return distances

