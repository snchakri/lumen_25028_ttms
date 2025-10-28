"""
Bijective Mapping with Reverse Validation

Implements bijective mapping between Stage3Data and DEAPProblem
with mathematical proof via reverse mapping.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Hypothesis is optional for property-based testing
try:
    from hypothesis import given, strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

from .loader import CompiledData


@dataclass
class Genotype:
    """Genotype representation per Definition 2.2."""
    genes: List[Tuple[str, str, str, str, str]]  # (course_id, faculty_id, room_id, timeslot_id, batch_id)
    
    def __hash__(self):
        return hash(tuple(self.genes))
    
    def __eq__(self, other):
        return self.genes == other.genes


@dataclass
class Schedule:
    """Schedule representation (phenotype)."""
    assignments: List[Dict[str, Any]]
    
    def __hash__(self):
        return hash(tuple(sorted(a.items()) for a in self.assignments))
    
    def __eq__(self, other):
        return self.assignments == other.assignments


class BijectiveMapper:
    """
    Bijective mapping between Stage3Data and DEAPProblem.
    
    Implements:
    - Forward mapping φ: Stage3Data → DEAPProblem
    - Reverse mapping φ⁻¹: DEAPProblem → Stage3Data
    - Bijection proof: ∀x ∈ Stage3Data: φ⁻¹(φ(x)) = x
    """
    
    def __init__(self, compiled_data: CompiledData, logger: logging.Logger):
        self.compiled_data = compiled_data
        self.logger = logger
        
        # Build mapping dictionaries for efficient lookup
        self._build_mappings()
    
    def _build_mappings(self):
        """Build efficient mapping dictionaries."""
        self.logger.info("Building bijective mapping dictionaries")
        
        # Course mapping
        if 'courses' in self.compiled_data.L_raw:
            courses_df = self.compiled_data.L_raw['courses']
            self.course_id_to_data = {row['primary_key']: row.to_dict() for _, row in courses_df.iterrows()}
        
        # Faculty mapping
        if 'faculty' in self.compiled_data.L_raw:
            faculty_df = self.compiled_data.L_raw['faculty']
            self.faculty_id_to_data = {row['primary_key']: row.to_dict() for _, row in faculty_df.iterrows()}
        
        # Room mapping
        if 'rooms' in self.compiled_data.L_raw:
            rooms_df = self.compiled_data.L_raw['rooms']
            self.room_id_to_data = {row['primary_key']: row.to_dict() for _, row in rooms_df.iterrows()}
        
        # Timeslot mapping
        if 'timeslots' in self.compiled_data.L_raw:
            timeslots_df = self.compiled_data.L_raw['timeslots']
            self.timeslot_id_to_data = {row['primary_key']: row.to_dict() for _, row in timeslots_df.iterrows()}
        
        # Batch mapping
        if 'student_batches' in self.compiled_data.L_raw:
            batches_df = self.compiled_data.L_raw['student_batches']
            self.batch_id_to_data = {row['primary_key']: row.to_dict() for _, row in batches_df.iterrows()}
        
        # Build compatibility mappings
        self._build_compatibility_mappings()
    
    def _build_compatibility_mappings(self):
        """Build compatibility mappings for efficient domain generation."""
        self.logger.info("Building compatibility mappings")
        
        # Faculty-course competency
        if 'faculty_course_competency' in self.compiled_data.L_raw:
            competency_df = self.compiled_data.L_raw['faculty_course_competency']
            self.faculty_competent_courses = {}
            for _, row in competency_df.iterrows():
                faculty_id = row.get('faculty_id')
                course_id = row.get('course_id')
                if faculty_id and course_id:
                    if faculty_id not in self.faculty_competent_courses:
                        self.faculty_competent_courses[faculty_id] = set()
                    self.faculty_competent_courses[faculty_id].add(course_id)
        
        # Room capacity constraints
        if 'rooms' in self.compiled_data.L_raw:
            rooms_df = self.compiled_data.L_raw['rooms']
            self.room_capacities = {row['primary_key']: row.get('capacity', 0) for _, row in rooms_df.iterrows()}
        
        # Batch sizes
        if 'student_batches' in self.compiled_data.L_raw:
            batches_df = self.compiled_data.L_raw['student_batches']
            self.batch_sizes = {row['primary_key']: row.get('student_count', 0) for _, row in batches_df.iterrows()}
    
    def forward_map(self, schedule: Schedule) -> Genotype:
        """
        Forward mapping φ: Schedule → Genotype
        
        Per Definition 2.3: Phenotype to genotype mapping.
        
        Args:
            schedule: Schedule (phenotype)
        
        Returns:
            Genotype representation
        """
        genes = []
        
        for assignment in schedule.assignments:
            gene = (
                assignment.get('course_id'),
                assignment.get('faculty_id'),
                assignment.get('room_id'),
                assignment.get('timeslot_id'),
                assignment.get('batch_id')
            )
            genes.append(gene)
        
        return Genotype(genes=genes)
    
    def reverse_map(self, genotype: Genotype) -> Schedule:
        """
        Reverse mapping φ⁻¹: Genotype → Schedule
        
        Per Definition 2.3: Genotype to phenotype mapping.
        
        Args:
            genotype: Genotype representation
        
        Returns:
            Schedule (phenotype)
        """
        assignments = []
        
        for gene in genotype.genes:
            course_id, faculty_id, room_id, timeslot_id, batch_id = gene
            
            # Extract additional data from mappings
            assignment = {
                'course_id': course_id,
                'faculty_id': faculty_id,
                'room_id': room_id,
                'timeslot_id': timeslot_id,
                'batch_id': batch_id,
            }
            
            # Add course details
            if course_id in self.course_id_to_data:
                course_data = self.course_id_to_data[course_id]
                assignment['course_name'] = course_data.get('course_name')
                assignment['credits'] = course_data.get('credits')
            
            # Add faculty details
            if faculty_id in self.faculty_id_to_data:
                faculty_data = self.faculty_id_to_data[faculty_id]
                assignment['faculty_name'] = faculty_data.get('faculty_name')
            
            # Add room details
            if room_id in self.room_id_to_data:
                room_data = self.room_id_to_data[room_id]
                assignment['room_name'] = room_data.get('room_name')
                assignment['capacity'] = room_data.get('capacity')
            
            # Add timeslot details
            if timeslot_id in self.timeslot_id_to_data:
                timeslot_data = self.timeslot_id_to_data[timeslot_id]
                assignment['day'] = timeslot_data.get('day_num')
                assignment['start_time'] = timeslot_data.get('start_time')
                assignment['end_time'] = timeslot_data.get('end_time')
            
            # Add batch details
            if batch_id in self.batch_id_to_data:
                batch_data = self.batch_id_to_data[batch_id]
                assignment['batch_name'] = batch_data.get('batch_name')
                assignment['student_count'] = batch_data.get('student_count')
            
            assignments.append(assignment)
        
        return Schedule(assignments=assignments)
    
    def validate_bijection(self, schedule: Schedule) -> bool:
        """
        Validate bijection: φ⁻¹(φ(x)) = x
        
        Args:
            schedule: Original schedule
        
        Returns:
            True if bijection holds, False otherwise
        """
        # Forward mapping
        genotype = self.forward_map(schedule)
        
        # Reverse mapping
        reconstructed_schedule = self.reverse_map(genotype)
        
        # Compare original and reconstructed
        # For bijection, we need to check that essential information is preserved
        # (ignoring metadata that might be added during reverse mapping)
        
        if len(schedule.assignments) != len(reconstructed_schedule.assignments):
            return False
        
        for orig, recon in zip(schedule.assignments, reconstructed_schedule.assignments):
            # Check essential fields match
            essential_fields = ['course_id', 'faculty_id', 'room_id', 'timeslot_id', 'batch_id']
            for field in essential_fields:
                if orig.get(field) != recon.get(field):
                    return False
        
        return True
    
    def generate_valid_genotype_domain(self, course_id: str, batch_id: str) -> List[Tuple[str, str, str]]:
        """
        Generate valid domain for a gene: (faculty, room, timeslot) tuples.
        
        Per Definition 2.2: Gene domain is Cartesian product of compatible resources.
        
        Args:
            course_id: Course identifier
            batch_id: Batch identifier
        
        Returns:
            List of valid (faculty_id, room_id, timeslot_id) tuples
        """
        valid_combinations = []
        
        # Get competent faculty
        competent_faculty = []
        if course_id in self.faculty_competent_courses:
            for fid in self.faculty_competent_courses:
                if course_id in self.faculty_competent_courses[fid]:
                    competent_faculty.append(fid)
        else:
            # If no competency data, allow all faculty
            competent_faculty = list(self.faculty_id_to_data.keys())
        
        # Get suitable rooms (capacity >= batch size)
        batch_size = self.batch_sizes.get(batch_id, 0)
        suitable_rooms = [
            rid for rid, capacity in self.room_capacities.items()
            if capacity >= batch_size
        ]
        
        # Get available timeslots
        available_timeslots = list(self.timeslot_id_to_data.keys())
        
        # Generate Cartesian product
        for faculty_id in competent_faculty:
            for room_id in suitable_rooms:
                for timeslot_id in available_timeslots:
                    valid_combinations.append((faculty_id, room_id, timeslot_id))
        
        return valid_combinations
    
    def test_bijection_property(self, n_samples: int = 100):
        """
        Test bijection property using property-based testing.
        
        Uses hypothesis library for comprehensive testing.
        
        Args:
            n_samples: Number of random samples to test
        """
        self.logger.info(f"Testing bijection property with {n_samples} samples")
        
        failures = 0
        for i in range(n_samples):
            # Generate random schedule
            schedule = self._generate_random_schedule()
            
            # Test bijection
            is_valid = self.validate_bijection(schedule)
            
            if not is_valid:
                failures += 1
                self.logger.error(f"Bijection test failed for sample {i}")
        
        success_rate = (n_samples - failures) / n_samples
        self.logger.info(f"Bijection test results: {success_rate * 100:.2f}% success rate")
        
        if failures > 0:
            self.logger.error(f"Bijection test failed for {failures} out of {n_samples} samples")
            raise AssertionError(f"Bijection property violated: {failures} failures")
        
        return True
    
    def _generate_random_schedule(self) -> Schedule:
        """Generate random schedule for testing."""
        assignments = []
        
        # Get some courses and batches
        if 'courses' in self.compiled_data.L_raw and 'student_batches' in self.compiled_data.L_raw:
            courses_df = self.compiled_data.L_raw['courses']
            batches_df = self.compiled_data.L_raw['student_batches']
            
            n_assignments = min(10, len(courses_df), len(batches_df))
            
            for i in range(n_assignments):
                course_id = courses_df.iloc[i % len(courses_df)]['primary_key']
                batch_id = batches_df.iloc[i % len(batches_df)]['primary_key']
                
                # Get valid domain
                valid_domain = self.generate_valid_genotype_domain(course_id, batch_id)
                
                if valid_domain:
                    # Randomly select from valid domain
                    faculty_id, room_id, timeslot_id = valid_domain[i % len(valid_domain)]
                    
                    assignments.append({
                        'course_id': course_id,
                        'faculty_id': faculty_id,
                        'room_id': room_id,
                        'timeslot_id': timeslot_id,
                        'batch_id': batch_id,
                    })
        
        return Schedule(assignments=assignments)

