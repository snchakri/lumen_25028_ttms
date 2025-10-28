"""
Metadata Extractor

Extracts problem metadata for algorithm parameter tuning and solver selection.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from typing import Dict, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .loader import CompiledData


@dataclass
class ProblemMetadata:
    """Problem metadata for solver selection."""
    
    # Problem size
    n_courses: int
    n_faculty: int
    n_rooms: int
    n_timeslots: int
    n_batches: int
    
    # Constraint characteristics
    n_hard_constraints: int
    n_soft_constraints: int
    constraint_density: float  # |constraints| / |possible_assignments|
    constraint_hardness: float  # |hard_constraints| / |total_constraints|
    
    # Faculty characteristics
    specialization_index: float  # Gini coefficient of faculty competencies
    
    # Resource characteristics
    avg_faculty_competency: float
    avg_room_utilization: float
    avg_batch_size: float
    
    # Optimization characteristics
    estimated_chromosome_length: int
    estimated_population_size: int
    
    # Problem complexity indicators (fields with defaults must come last)
    multi_objective: bool = True  # Always true for scheduling
    fitness_landscape_smooth: bool = False  # Estimate based on constraint density


class MetadataExtractor:
    """
    Extract problem metadata from compiled data.
    
    Used for intelligent solver selection per Definition 14.1.
    """
    
    def __init__(self, compiled_data: CompiledData, logger: logging.Logger):
        self.compiled_data = compiled_data
        self.logger = logger
    
    def extract_metadata(self) -> ProblemMetadata:
        """Extract complete problem metadata."""
        self.logger.info("Extracting problem metadata")
        
        # Extract problem size
        n_courses, n_faculty, n_rooms, n_timeslots, n_batches = self._extract_problem_size()
        
        # Extract constraint characteristics
        n_hard_constraints, n_soft_constraints, constraint_density, constraint_hardness = self._extract_constraint_characteristics()
        
        # Extract faculty characteristics
        specialization_index = self._extract_specialization_index()
        
        # Extract resource characteristics
        avg_faculty_competency, avg_room_utilization, avg_batch_size = self._extract_resource_characteristics()
        
        # Estimate optimization characteristics
        estimated_chromosome_length = self._estimate_chromosome_length()
        estimated_population_size = self._estimate_population_size()
        
        # Determine fitness landscape smoothness
        fitness_landscape_smooth = constraint_density < 0.5
        
        metadata = ProblemMetadata(
            n_courses=n_courses,
            n_faculty=n_faculty,
            n_rooms=n_rooms,
            n_timeslots=n_timeslots,
            n_batches=n_batches,
            n_hard_constraints=n_hard_constraints,
            n_soft_constraints=n_soft_constraints,
            constraint_density=constraint_density,
            constraint_hardness=constraint_hardness,
            specialization_index=specialization_index,
            fitness_landscape_smooth=fitness_landscape_smooth,
            avg_faculty_competency=avg_faculty_competency,
            avg_room_utilization=avg_room_utilization,
            avg_batch_size=avg_batch_size,
            estimated_chromosome_length=estimated_chromosome_length,
            estimated_population_size=estimated_population_size,
        )
        
        self.logger.info("Problem metadata extracted successfully")
        self._log_metadata(metadata)
        
        return metadata
    
    def _extract_problem_size(self) -> Tuple[int, int, int, int, int]:
        """Extract problem size dimensions."""
        n_courses = len(self.compiled_data.L_raw.get('courses', pd.DataFrame()))
        n_faculty = len(self.compiled_data.L_raw.get('faculty', pd.DataFrame()))
        n_rooms = len(self.compiled_data.L_raw.get('rooms', pd.DataFrame()))
        n_timeslots = len(self.compiled_data.L_raw.get('timeslots', pd.DataFrame()))
        n_batches = len(self.compiled_data.L_raw.get('student_batches', pd.DataFrame()))
        
        self.logger.info(f"Problem size: {n_courses} courses, {n_faculty} faculty, {n_rooms} rooms, {n_timeslots} timeslots, {n_batches} batches")
        
        return n_courses, n_faculty, n_rooms, n_timeslots, n_batches
    
    def _extract_constraint_characteristics(self) -> Tuple[int, int, float, float]:
        """Extract constraint characteristics."""
        if 'dynamic_constraints' not in self.compiled_data.L_raw:
            return 0, 0, 0.0, 0.0
        
        constraints_df = self.compiled_data.L_raw['dynamic_constraints']
        
        # Count hard and soft constraints
        n_hard_constraints = len(constraints_df[constraints_df.get('constraint_type', '') == 'HARD'])
        n_soft_constraints = len(constraints_df[constraints_df.get('constraint_type', '') == 'SOFT'])
        n_total_constraints = n_hard_constraints + n_soft_constraints
        
        # Calculate constraint density
        # |possible_assignments| ≈ n_courses × n_faculty × n_rooms × n_timeslots
        n_courses = len(self.compiled_data.L_raw.get('courses', pd.DataFrame()))
        n_faculty = len(self.compiled_data.L_raw.get('faculty', pd.DataFrame()))
        n_rooms = len(self.compiled_data.L_raw.get('rooms', pd.DataFrame()))
        n_timeslots = len(self.compiled_data.L_raw.get('timeslots', pd.DataFrame()))
        
        n_possible_assignments = n_courses * n_faculty * n_rooms * n_timeslots
        constraint_density = n_total_constraints / n_possible_assignments if n_possible_assignments > 0 else 0.0
        
        # Calculate constraint hardness
        constraint_hardness = n_hard_constraints / n_total_constraints if n_total_constraints > 0 else 0.0
        
        self.logger.info(f"Constraints: {n_hard_constraints} hard, {n_soft_constraints} soft, density={constraint_density:.4f}, hardness={constraint_hardness:.4f}")
        
        return n_hard_constraints, n_soft_constraints, constraint_density, constraint_hardness
    
    def _extract_specialization_index(self) -> float:
        """Extract faculty specialization index using Gini coefficient."""
        if 'faculty_course_competency' not in self.compiled_data.L_raw:
            return 0.0
        
        competency_df = self.compiled_data.L_raw['faculty_course_competency']
        
        # Count competencies per faculty
        faculty_competencies = competency_df.groupby('faculty_id').size()
        
        if len(faculty_competencies) == 0:
            return 0.0
        
        # Calculate Gini coefficient
        gini = self._calculate_gini_coefficient(faculty_competencies.values)
        
        self.logger.info(f"Faculty specialization index (Gini): {gini:.4f}")
        
        return gini
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient."""
        if len(values) == 0:
            return 0.0
        
        # Sort values
        sorted_values = np.sort(values)
        n = len(sorted_values)
        
        # Calculate Gini coefficient
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
        
        return gini
    
    def _extract_resource_characteristics(self) -> Tuple[float, float, float]:
        """Extract resource characteristics."""
        # Average faculty competency
        avg_faculty_competency = 0.0
        if 'faculty_course_competency' in self.compiled_data.L_raw:
            competency_df = self.compiled_data.L_raw['faculty_course_competency']
            if 'competency_level' in competency_df.columns:
                avg_faculty_competency = competency_df['competency_level'].mean()
        
        # Average room utilization (estimate based on capacity)
        avg_room_utilization = 0.0
        if 'rooms' in self.compiled_data.L_raw:
            rooms_df = self.compiled_data.L_raw['rooms']
            if 'capacity' in rooms_df.columns:
                avg_room_utilization = rooms_df['capacity'].mean()
        
        # Average batch size
        avg_batch_size = 0.0
        if 'student_batches' in self.compiled_data.L_raw:
            batches_df = self.compiled_data.L_raw['student_batches']
            if 'student_count' in batches_df.columns:
                avg_batch_size = batches_df['student_count'].mean()
        
        self.logger.info(f"Resource characteristics: avg_competency={avg_faculty_competency:.2f}, avg_room_size={avg_room_utilization:.2f}, avg_batch_size={avg_batch_size:.2f}")
        
        return avg_faculty_competency, avg_room_utilization, avg_batch_size
    
    def _estimate_chromosome_length(self) -> int:
        """Estimate chromosome length."""
        # Chromosome length = number of course-batch pairs requiring scheduling
        n_courses = len(self.compiled_data.L_raw.get('courses', pd.DataFrame()))
        n_batches = len(self.compiled_data.L_raw.get('student_batches', pd.DataFrame()))
        
        # Estimate: each course needs to be scheduled for each batch that takes it
        # Conservative estimate: assume each course is taken by 50% of batches on average
        estimated_length = int(n_courses * n_batches * 0.5)
        
        self.logger.info(f"Estimated chromosome length: {estimated_length}")
        
        return estimated_length
    
    def _estimate_population_size(self) -> int:
        """Estimate population size based on problem complexity."""
        # Per Theorem 10.1 and empirical recommendations
        n_courses = len(self.compiled_data.L_raw.get('courses', pd.DataFrame()))
        
        # Population size should scale with problem size
        # Base size: 50
        # Scaling factor: sqrt(n_courses)
        estimated_size = int(50 * np.sqrt(n_courses))
        
        # Clamp to reasonable range
        estimated_size = max(50, min(estimated_size, 500))
        
        self.logger.info(f"Estimated population size: {estimated_size}")
        
        return estimated_size
    
    def _log_metadata(self, metadata: ProblemMetadata):
        """Log extracted metadata."""
        self.logger.info("=" * 80)
        self.logger.info("EXTRACTED PROBLEM METADATA")
        self.logger.info("=" * 80)
        self.logger.info(f"Problem Size:")
        self.logger.info(f"  - Courses: {metadata.n_courses}")
        self.logger.info(f"  - Faculty: {metadata.n_faculty}")
        self.logger.info(f"  - Rooms: {metadata.n_rooms}")
        self.logger.info(f"  - Timeslots: {metadata.n_timeslots}")
        self.logger.info(f"  - Batches: {metadata.n_batches}")
        self.logger.info(f"Constraint Characteristics:")
        self.logger.info(f"  - Hard constraints: {metadata.n_hard_constraints}")
        self.logger.info(f"  - Soft constraints: {metadata.n_soft_constraints}")
        self.logger.info(f"  - Constraint density: {metadata.constraint_density:.4f}")
        self.logger.info(f"  - Constraint hardness: {metadata.constraint_hardness:.4f}")
        self.logger.info(f"Faculty Characteristics:")
        self.logger.info(f"  - Specialization index: {metadata.specialization_index:.4f}")
        self.logger.info(f"Optimization Characteristics:")
        self.logger.info(f"  - Multi-objective: {metadata.multi_objective}")
        self.logger.info(f"  - Fitness landscape smooth: {metadata.fitness_landscape_smooth}")
        self.logger.info(f"  - Estimated chromosome length: {metadata.estimated_chromosome_length}")
        self.logger.info(f"  - Estimated population size: {metadata.estimated_population_size}")
        self.logger.info("=" * 80)

